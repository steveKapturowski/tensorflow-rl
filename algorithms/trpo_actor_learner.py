# -*- coding: utf-8 -*-
import time
import utils.ops
import utils.stats
import numpy as np
import utils.logger
import tensorflow as tf

from policy_based_actor_learner import BaseA3CLearner
from networks.policy_v_network import PolicyValueNetwork


logger = utils.logger.getLogger('trpo_actor_learner')


class TRPOLearner(BaseA3CLearner):
	'''
	Implementation of Trust Region Policy Optimization + Generalized Advantage Estimation 
	as described in https://arxiv.org/pdf/1506.02438.pdf

	∂'π = F^-1 ∂π where F is the Fischer Information Matrix
	We can't compute F^-1 directly except for very small networks
	so we'll use either conjugate gradient descent to approximate F^-1 ∂π
	'''

	def __init__(self, args):
		args.entropy_regularisation_strength = 0.0
		super(TRPOLearner, self).__init__(args)

		policy_conf = {'name': 'local_learning_{}'.format(self.actor_id),
					   'input_shape': self.input_shape,
					   'num_act': self.num_actions,
					   'args': args}

		#we use separate networks as in the paper since so we don't do damage to the trust region updates
		self.policy_network = PolicyValueNetwork(policy_conf, use_value_head=False)
		self.local_network = self.policy_network
		#self.value_network = PolicyValueNetwork(value_conf, use_policy_head=False)

		if self.actor_id == 0:
			var_list = self.policy_network.params #+self.value_network.params
			self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3,
                                        keep_checkpoint_every_n_hours=2)

		self.max_kl = .01
		self.cg_damping = 0.001
		self._build_ops()


	def _build_ops(self):
		eps = 1e-10
		self.action_probs = self.policy_network.output_layer_pi
		self.old_action_probs = tf.placeholder(tf.float32, shape=[None, self.num_actions], name="old_action_probs")

		action = tf.cast(tf.argmax(self.policy_network.selected_action_ph, axis=1), tf.int32)

		batch_idx = tf.range(0, tf.shape(action)[0])
		selected_prob = utils.ops.slice_2d(self.action_probs, batch_idx, action)
		old_selected_prob = utils.ops.slice_2d(self.old_action_probs, batch_idx, action)
		self.policy_loss = -tf.reduce_mean(tf.multiply(
			self.policy_network.adv_actor_ph,
			selected_prob / old_selected_prob
		))

		self.theta = utils.ops.flatten_vars(self.policy_network.params)
		self.kl = utils.stats.mean_kl_divergence_op(self.old_action_probs, self.action_probs)
		self.pg = utils.ops.flatten_vars(
			tf.gradients(self.policy_loss, self.policy_network.params))

		kl_firstfixed = tf.reduce_mean(tf.reduce_sum(tf.multiply(
			tf.stop_gradient(self.action_probs),
			tf.log(tf.stop_gradient(self.action_probs + eps) / (self.action_probs + eps))
		), axis=1))

		kl_grads = tf.gradients(kl_firstfixed, self.policy_network.params)
		flat_kl_grads = utils.ops.flatten_vars(kl_grads)


		self.fullstep, self.neggdotstepdir = self._conjugate_gradient_ops(-self.pg, flat_kl_grads)


	def _conjugate_gradient_ops(self, pg_grads, kl_grads, max_iterations=20, residual_tol=1e-10):
		'''
		Construct conjugate gradient descent algorithm inside computation graph for improved efficiency
		'''
		i0 = tf.constant(0, dtype=tf.int32)
		loop_condition = lambda i, r, p, x, rdotr: tf.logical_or(
			tf.less(rdotr, residual_tol), tf.less(i, max_iterations))


		def body(i, r, p, x, rdotr):
			fvp = utils.ops.flatten_vars(tf.gradients(
				tf.reduce_sum(tf.stop_gradient(p)*kl_grads),
				self.policy_network.params))

			z = fvp + self.cg_damping * p

			alpha = rdotr / (tf.reduce_sum(p*z) + 1e-8)
			x += alpha * p
			r -= alpha * z

			new_rdotr = tf.reduce_sum(r*r)
			beta = new_rdotr / (rdotr + 1e-8)
			p = r + beta * p

			return i+1, r, p, x, new_rdotr

		_, r, p, stepdir, rdotr = tf.while_loop(
			loop_condition,
			body,
			loop_vars=[i0,
					   pg_grads,
					   pg_grads,
					   tf.zeros_like(pg_grads),
					   tf.reduce_sum(pg_grads*pg_grads)])

		fvp = utils.ops.flatten_vars(tf.gradients(
			tf.reduce_sum(tf.stop_gradient(stepdir)*kl_grads),
			self.policy_network.params)) + self.cg_damping * stepdir

		shs = 0.5 * tf.reduce_sum(stepdir*fvp)
		fullstep = stepdir * tf.sqrt(2.0 * self.max_kl / shs)
		neggdotstepdir = -tf.reduce_sum(pg_grads*stepdir)

		return fullstep, -neggdotstepdir


	#TODO: move this into utils so it can be used with other algorithms
	def compute_gae(self, rewards, values, next_val):
		values = values + [next_val]

		adv_batch = list()
		for i in range(len(rewards)):
			gae = 0.0
			for j in range(i, len(rewards)):
				TD_i = rewards[j] + self.gamma*values[j+1] - values[j]
				gae += TD_i * (self.gamma*self.td_lambda)**(j - i)

			adv_batch.append(gae)

		return adv_batch


	def choose_next_action(self, state):
		action_probs = self.session.run(
			self.policy_network.output_layer_pi,
			feed_dict={self.policy_network.input_ph: [state]})
            
		action_probs = action_probs.reshape(-1)

		action_index = self.sample_policy_action(action_probs)
		new_action = np.zeros([self.num_actions])
		new_action[action_index] = 1

		return new_action, action_probs


	#There doesn't seem to be a clean way to build the line search into the computation graph
	#so we'll have to hop back and forth between cpu and gpu each iteration
	def linesearch(self, feed, x, fullstep, expected_improve_rate):
		accept_ratio = .05
		max_backtracks = 10

		fval = self.session.run(self.policy_loss, feed_dict=feed)

		for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
			xnew = x + stepfrac * fullstep
			self.assign_vars(self.policy_network, xnew)
			newfval, kl = self.session.run([self.policy_loss, self.kl], feed_dict=feed)

			improvement = fval - newfval
			logger.debug('Improvement {} / Mean KL {}'.format(improvement, kl))

			if improvement > 0 and kl < self.max_kl:
				return xnew

		logger.debug('No update')
		return x


	def update_grads(self, data):
		feed_dict={
			self.policy_network.input_ph:           data['state'],
			self.policy_network.selected_action_ph: data['action'],
			self.policy_network.adv_actor_ph:       data['reward'],
			self.old_action_probs:                  data['pi']
		}
		theta_prev, fullstep, neggdotstepdir = self.session.run(
			[self.theta, self.fullstep, self.neggdotstepdir], feed_dict=feed_dict)

		new_theta = self.linesearch(feed_dict, theta_prev, fullstep, neggdotstepdir)
		self.assign_vars(self.policy_network, new_theta)

		return self.session.run(self.kl, feed_dict)


	def _run(self):
		for epoch in range(10000):
			data = {
				'state':  list(),
				'pi':     list(),
				'action': list(),
				'reward': list(),
			}

			episode_rewards = list()
			for episode in range(50):
				s = self.emulator.get_initial_state()

				episode_over = False
				accumulated_rewards = list()
				while not episode_over and len(accumulated_rewards) < 5000:
					a, pi = self.choose_next_action(s)
					new_s, reward, episode_over = self.emulator.next(a)
					accumulated_rewards.append(self.rescale_reward(reward))

					data['state'].append(s)
					data['pi'].append(pi)
					data['action'].append(a)

					s = new_s

				mc_returns = np.array(accumulated_rewards)[::-1].cumsum()[::-1]
				episode_rewards.append(mc_returns[0])
				data['reward'].extend(mc_returns)
					
			kl = self.update_grads(data)
			mean_episode_reward = np.array(episode_rewards).mean()

			logger.info('Epoch {} / Mean KL Divergence {} / Mean Reward {}'.format(
				epoch+1, kl, mean_episode_reward))



