# -*- coding: utf-8 -*-
import time
import utils.stats
import numpy as np
import utils.logger
import tensorflow as tf

from actor_learner import ONE_LIFE_GAMES
from policy_based_actor_learner import BaseA3CLearner
from networks.policy_v_network import PolicyValueNetwork
from utils.replay_memory import ReplayMemory


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

		#we use separate networks as in the paper since so we don't do damage to the trust region updates
		self.policy_network = PolicyValueNetwork(args, use_value_head=False)
		# self.value_network = PolicyValueNetwork(args, use_policy_head=False)

		self.max_kl = .01
		self.cg_damping = 0.001
		self._build_ops()


	def flatten_vars(self, var_list):
		return tf.concat(0, [
			tf.reshape([np.prod(v.get_shape().as_list())])
			for v in var_list
		])


	def assign_vars(self, flat_params, var_list):
		'''
		restruture flat array into corrects shapes and assign new values
		'''


	def _build_ops(self):
		eps = 1e-10
		self.action_probs = self.policy_network.output_layer_pi
		self.old_action_probs = tf.placeholder(tf.float32, shape=[None, self.num_actions], name="old_action_probs")
		ratio_n = self.distribution.likelihood_ratio_sym(
			action, self.action_probs, self.old_action_probs)

		self.policy_loss = -tf.reduce_mean(ratio_n * advant)
        
		kl = utils.stats.kl_divergence(self.old_action_probs, self.action_probs)


		var_list = self.policy_network.params
		self.pg = tf.gradients(self.policy_loss, self.policy_network.params)
		N_obs = tf.cast(tf.shape(self.policy_network.input_ph)[0], tf.float32)
		# KL divergence where first arg is fixed
		# replace old->tf.stop_gradient from previous kl
		kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
			action_probs) * tf.log(tf.stop_gradient(self.action_probs + eps) / (self.action_probs + eps))) / N_obs
		grads = tf.gradients(kl_firstfixed, var_list)
		self.flat_tangent = tf.placeholder(dtype, shape=[None])
		shapes = map(var_shape, var_list)
		start = 0

		tangents = []
		for shape in shapes:
			size = np.prod(shape)
			param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
			tangents.append(param)
			start += size
		gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
		self.fvp = flatgrad(gvp, var_list)

		self.policy_assign_placeholders = [
			tf.placeholder(v.get_shape().as_list())
			for v in self.policy_network.params]
		self.policy_assign_ops = [tf.assign(v) for v in self.policy_network.params]
		self.fullstep, self.neggdotstepdir = self._conjugate_gradient_ops()


	def _conjugate_gradient_ops(self, grads, max_iterations=20, residual_tol=1e-10):
		'''
		Construct conjugate gradient descent algorithm inside computation graph for improved efficiency
		'''
		i0 = tf.constant(0, dtype=tf.int32)
		loop_condition = lambda i, r, p, x, rdotr: tf.logical_or(
			tf.less(rdotr, residual_tol), tf.less(i, max_iterations))


		def body(i, r, p, x, rdotr):
			z = self.fvp + self.cg_damping * p

			alpha = rdotr / (tf.reduce_sum(p*z) + 1e-8)
			x += alpha * p
			r -= alpha * z

			new_rdotr = r.dot(r)
			beta = new_rdotr / (rdotr + 1e-8)
			p = r + beta * p

			return i+1, r, p, x, new_rdotr

		_, r, p, stepdir, rdotr = tf.while_loop(
			loop_condition,
			body,
			loop_vars=[i0,
					   grads,
					   grads,
					   tf.zeros_like(grads),
					   tf.reduce_sum(grads*grads)])

		shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))
		fullstep = stepdir * np.sqrt(2.0 * self.max_kl / shs)
		neggdotstepdir = -grads.dot(stepdir)

		return fullstep, neggdotstepdir


	#There doesn't seem to be a clean way to build the line search into the computation graph
	#so we'll have to hop back and forth between cpu and gpu each iteration
	def linesearch(self, feed, fullstep, expected_improve_rate):
		accept_ratio = .1
		max_backtracks = 10

		fval = self.session.run(self.policy_loss, feed_dict=feed)

		for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
		    xnew = x + stepfrac * fullstep
		    self.assign_vars(xnew, self.policy_network.params)
		    newfval = self.session.run(self.policy_loss, feed_dict=feed)

		    actual_improve = fval - newfval
		    expected_improve = expected_improve_rate * stepfrac
		    ratio = actual_improve / expected_improve
		    if ratio > accept_ratio and actual_improve > 0:
		        return xnew
		return x


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


	def update_grads(self, data):
		prev_distributions = data['prev_distributions']

		feed_dict = {

		}
		fullstep, neggdotstepdir = self.session.run(
			[self.fullstep, self.neggdotstepdir], feed_dict=feed_dict)

		new_theta = linesearch(fullstep, neggdotstepdir)
		self.assign_vars(new_theta, self.policy_network.params)



		new_distributions = self.session.run(
			self.policy_network.output_layer_pi, feed_dict=feed_dict)

		kl_divergence = np.array([
			utils.stats.kl_divergence(p, q)
			for p, q in zip(prev_distributions, new_distributions)
		]).mean()

		return kl_divergence


	def choose_next_action(self, state):
		action_probs = self.session.run(
			self.policy_network.output_layer_pi,
			feed_dict={self.policy_network.input_ph: [state]})
            
		action_probs = action_probs.reshape(-1)

		action_index = self.sample_policy_action(action_probs)
		new_action = np.zeros([self.num_actions])
		new_action[action_index] = 1

		return new_action, action_probs


	def run(self):
		for epoch in range(100):
			episode_over = False
			data = {
				'state':  list(),
				'pi':     list(),
				'action': list(),
				'reward': list(),
			}

			for episdoe in range(10):
				s = self.emulator.get_initial_state()

				while not episode_over:
					a, pi = self.choose_next_action(s)
					new_s, reward, episode_over = self.emulator.next(a)
					reward = self.rescale_reward(reward)

					data['state'].append(s)
					data['pi'].append(pi)
					data['action'].append(a)
					data['reward'].append(reward)

					s = new_s
					
			self.update_grads(data)









                

