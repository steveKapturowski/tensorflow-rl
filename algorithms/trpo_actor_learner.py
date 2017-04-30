# -*- coding: utf-8 -*-
import time
import utils.ops
import utils.stats
import numpy as np
import utils.logger
import tensorflow as tf

from utils.distributions import DiagNormal
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

		self.batch_size = 512
		self.max_cg_iters = 20
		self.num_epochs = args.num_epochs
		self.cg_damping = args.cg_damping
		self.cg_subsample = args.cg_subsample
		self.max_kl = args.max_kl
		self.max_rollout = args.max_rollout
		self.episodes_per_batch = args.episodes_per_batch
		self.baseline_vars = args.baseline_vars
		self.experience_queue = args.experience_queue
		self.task_queue = args.task_queue
		self.append_timestep = args.arch == 'FC'


		policy_conf = {'name': 'policy_network_{}'.format(self.actor_id),
					   'input_shape': self.input_shape,
					   'num_act': self.num_actions,
					   'args': args}
		value_conf = policy_conf.copy()
		value_conf['name'] = 'value_network_{}'.format(self.actor_id)
		value_conf['input_shape'] = args.vf_input_shape

		self.device = '/gpu:0' if self.is_master() else '/cpu:0'
		with tf.device(self.device):
			#we use separate networks as in the paper since so we don't do damage to the trust region updates
			self.policy_network = args.network(policy_conf)
			self.value_network = PolicyValueNetwork(value_conf, use_policy_head=False)
			self.local_network = self.policy_network
			self._build_ops()

		if self.is_master():
			var_list = self.policy_network.params + self.value_network.params
			self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3,
                                        keep_checkpoint_every_n_hours=2)


	def _build_ops(self):
		eps = 1e-10
		self.dist_params = self.policy_network.dist.params()
		num_params = self.dist_params.get_shape().as_list()[1]
		self.old_params = tf.placeholder(tf.float32, shape=[None, num_params], name='old_params')

		selected_prob = tf.exp(self.policy_network.log_output_selected_action)
		old_dist = self.policy_network.dist.__class__(self.old_params)
		old_selected_prob = tf.exp(old_dist.log_likelihood(self.policy_network.selected_action_ph))

		self.theta = self.policy_network.flat_vars
		self.policy_loss = -tf.reduce_mean(tf.multiply(
			self.policy_network.adv_actor_ph,
			selected_prob / old_selected_prob
		))
		self.pg = utils.ops.flatten_vars(
			tf.gradients(self.policy_loss, self.policy_network.params))

		self.kl = tf.reduce_mean(self.policy_network.dist.kl_divergence(self.old_params))
		self.kl_firstfixed = tf.reduce_mean(self.policy_network.dist.kl_divergence(
			tf.stop_gradient(self.dist_params)))


		kl_grads = tf.gradients(self.kl_firstfixed, self.policy_network.params)
		flat_kl_grads = utils.ops.flatten_vars(kl_grads)

		self.pg_placeholder = tf.placeholder(tf.float32, shape=self.pg.get_shape().as_list(), name='pg_placeholder')
		self.fullstep, self.neggdotstepdir = self._conjugate_gradient_ops(
			-self.pg_placeholder, flat_kl_grads, max_iterations=self.max_cg_iters)


	def _conjugate_gradient_ops(self, pg_grads, kl_grads, max_iterations=20, residual_tol=1e-10):
		'''
		Construct conjugate gradient descent algorithm inside computation graph for improved efficiency
		'''
		i0 = tf.constant(0, dtype=tf.int32)
		loop_condition = lambda i, r, p, x, rdotr: tf.logical_and(
			tf.greater(rdotr, residual_tol), tf.less(i, max_iterations))


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

			new_rdotr = tf.Print(new_rdotr, [i, new_rdotr], 'Iteration / Residual: ')

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
			self.policy_network.params))

		shs = 0.5 * tf.reduce_sum(stepdir*fvp)
		lm = tf.sqrt((shs + 1e-8) / self.max_kl)
		fullstep = stepdir / lm
		neggdotstepdir = tf.reduce_sum(pg_grads*stepdir) / lm

		return fullstep, neggdotstepdir


	def choose_next_action(self, state):
		return self.policy_network.get_action(self.session, state)


	def run_minibatches(self, data, *ops):
		outputs = [np.zeros(op.get_shape().as_list(), dtype=np.float32) for op in ops]

		data_size = len(data['state'])
		for start in range(0, data_size, self.batch_size):
			end = start + np.minimum(self.batch_size, data_size-start)
			feed_dict={
				self.policy_network.input_ph:           data['state'][start:end],
				self.policy_network.selected_action_ph: data['action'][start:end],
				self.policy_network.adv_actor_ph:       data['reward'][start:end],
				self.old_params:                        data['pi'][start:end]
			}
			for i, output_i in enumerate(self.session.run(ops, feed_dict=feed_dict)):
				outputs[i] += output_i * (end-start)/float(data_size)

		return outputs


	def linesearch(self, data, x, fullstep, expected_improve_rate):
		accept_ratio = .1
		backtrack_ratio = .7
		max_backtracks = 15
    
		fval = self.run_minibatches(data, self.policy_loss)

		for (_n_backtracks, stepfrac) in enumerate(backtrack_ratio**np.arange(max_backtracks)):
		    xnew = x + stepfrac * fullstep
		    self.assign_vars(self.policy_network, xnew)
		    newfval, kl = self.run_minibatches(data, self.policy_loss, self.kl)

		    improvement = fval - newfval
		    logger.debug('Improvement {} / Mean KL {}'.format(improvement, kl))

		    expected_improve = expected_improve_rate * stepfrac
		    ratio = improvement / expected_improve
		    if ratio > accept_ratio and improvement > 0:
		    # if kl < self.max_kl and improvement > 0:
		        return xnew

		logger.debug('No update')
		return x


	def fit_baseline(self, data, mix_old=0):
		data_size = len(data['state'])
		proc_state = self.preprocess_value_state(data)
		print 'diffs', (data['mc_return'] - data['values']).mean()
		target = (1-mix_old)*data['mc_return'] + mix_old*data['values']
		grads = [np.zeros(g.get_shape().as_list(), dtype=np.float32) for g in self.value_network.get_gradients]

		#permute data in minibatches so we don't introduce bias
		perm = np.random.permutation(data_size)
		for start in range(0, data_size, self.batch_size):
			end = start + np.minimum(self.batch_size, data_size-start)
			batch_idx = perm[start:end]
			feed_dict={
				self.value_network.input_ph:         proc_state[batch_idx],
				self.value_network.critic_target_ph: target[batch_idx]
			}
			output_i = self.session.run(self.value_network.get_gradients, feed_dict=feed_dict)
			
			for i, g in enumerate(output_i):
				grads[i] += g * (end-start)/float(data_size)

			self._apply_gradients_to_shared_memory_vars(output_i, self.baseline_vars)
			self.sync_net_with_shared_memory(self.value_network, self.baseline_vars)


	def preprocess_value_state(self, data):
		if self.append_timestep: #this is particularly helpful on MuJoCo environments
			return np.hstack([data['state'], data['timestep'].reshape(-1, 1, 1)])
		else:
			return data['state']


	def predict_values(self, data):
		state = self.preprocess_value_state({
			'state': np.array(data['state']),
			'timestep': np.array(data['timestep'])})
		return self.session.run(
			self.value_network.output_layer_v,
			feed_dict={self.value_network.input_ph: state})[:, 0]


	def update_grads(self, data):
		#we need to compute the policy gradient in minibatches to avoid GPU OOM errors on Atari
		print 'fitting baseline...'
		self.fit_baseline(data)

		normalized_advantage = (data['advantage'] - data['advantage'].mean())/(data['advantage'].std() + 1e-8)
		data['reward'] = normalized_advantage

		print 'running policy gradient...'
		pg = self.run_minibatches(data, self.pg)[0]

		data_size = len(data['state'])
		subsample = np.random.choice(data_size, int(data_size*self.cg_subsample), replace=False)
		feed_dict={
			self.policy_network.input_ph:           data['state'][subsample],
			self.policy_network.selected_action_ph: data['action'][subsample],
			self.policy_network.adv_actor_ph:       data['reward'][subsample],
			self.old_params:                        data['pi'][subsample],
			self.pg_placeholder:                    pg
		}

		print 'running conjugate gradient descent...'
		theta_prev, fullstep, neggdotstepdir = self.session.run(
			[self.theta, self.fullstep, self.neggdotstepdir], feed_dict=feed_dict)

		print 'running linesearch...'
		new_theta = self.linesearch(data, theta_prev, fullstep, neggdotstepdir)
		self.assign_vars(self.policy_network, new_theta)

		return self.session.run(self.kl, feed_dict)


	def _run_worker(self):
		while True:
			signal = self.task_queue.get()
			if signal == 'EXIT':
				break

			self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
			self.sync_net_with_shared_memory(self.value_network, self.baseline_vars)
			s = self.emulator.get_initial_state()

			data = {
				'state':     list(),
				'pi':        list(),
				'action':    list(),
				'reward':    list(),
			}
			episode_over = False
			accumulated_rewards = list()
			while not episode_over and len(accumulated_rewards) < self.max_rollout:
				a, pi = self.choose_next_action(s)
				new_s, reward, episode_over = self.emulator.next(a)
				
				accumulated_rewards.append(reward)

				data['state'].append(s)
				data['pi'].append(pi)
				data['action'].append(a)
				data['reward'].append(reward)

				s = new_s

			mc_returns = list()
			running_total = 0.0
			for r in reversed(accumulated_rewards):
				running_total = r + self.gamma*running_total
				mc_returns.insert(0, running_total)

			timestep = np.arange(len(mc_returns), dtype=np.float32)/self.emulator.env.spec.timestep_limit
			data['timestep'] = timestep
			data['mc_return'] = mc_returns
			episode_reward = sum(accumulated_rewards)
			logger.debug('T{} / Episode Reward {}'.format(
				self.actor_id, episode_reward))

			self.experience_queue.put((data, episode_reward))
			

	def _run_master(self):
		for epoch in range(self.num_epochs):
			data = {
				'state':     list(),
				'pi':        list(),
				'action':    list(),
				'reward':    list(),
				'advantage': list(),
				'mc_return': list(),
				'timestep':  list(),
				'values':    list(),
			}
			#launch worker tasks
			for i in xrange(self.episodes_per_batch):
				self.task_queue.put(i)

			#collect worker experience
			episode_rewards = list()
			t0 = time.time()
			for _ in xrange(self.episodes_per_batch):
				worker_data, reward = self.experience_queue.get()
				episode_rewards.append(reward)

				values = self.predict_values(worker_data)
				advantages = self.compute_gae(worker_data['reward'], values.tolist(), 0)
				# advantages = worker_data['mc_return'] - values
				worker_data['values'] = values
				worker_data['advantage'] = advantages
				for key, value in worker_data.items():
					data[key].extend(value)
					
			t1 = time.time()
			kl = self.update_grads({
				k: np.array(v) for k, v in data.items()})
			self.update_shared_memory()
			t2 = time.time()

			mean_episode_reward = np.array(episode_rewards).mean()
			logger.info('Epoch {} / Mean KL Divergence {} / Mean Reward {} / Experience Time {:.2f}s / Training Time {:.2f}s'.format(
				epoch+1, kl, mean_episode_reward, t1-t0, t2-t1))


	def train(self):
		if self.is_master():
			self._run_master()
			for _ in xrange(self.num_actor_learners):
				self.task_queue.put('EXIT')
		else:
			self._run_worker()


