# -*- coding: utf-8 -*-
import time
import utils.stats
import numpy as np
import utils.logger
import tensorflow as tf

from actor_learner import ONE_LIFE_GAMES
from policy_based_actor_learner import BaseA3CLearner
from networks.policy_v_network import PolicyVNetwork
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
		super(TRPOLearner, self).__init__(args)

		#we use separate networks as in the paper since so we don't do damage to the trust region updates
		self.policy_network = PolicyVNetwork(args, use_value_head=False)
		# self.value_network = ValueNetwork(args)

		self.cg_damping = 0.001
		self._build_ops()


	def _build_ops(self):
        eps = 1e-10
        action_dist_n = self.policy_network.output_layer_pi
        N = tf.shape(obs)[0]
        ratio_n = self.distribution.likelihood_ratio_sym(action, action_dist_n, oldaction_dist)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        kl = self.distribution.kl_sym(oldaction_dist, action_dist_n)
        ent = self.distribution.entropy(action_dist_n)

        self.losses = [surr, kl, ent]

        var_list = tf.trainable_variables()
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            action_dist_n) * tf.log(tf.stop_gradient(action_dist_n + eps) / (action_dist_n + eps))) / Nf
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

		_, r, p, x, rdotr = tf.while_loop(
			loop_condition,
			body,
			loop_vars=[i0,
					   grads,
					   grads,
					   tf.zeros_like(grads),
					   tf.reduce_sum(grads*grads)])

		return x


	def conjugate_gradient(self, b, cg_iters=20, residual_tol=1e-10):
		'''Testing reference implementation from https://github.com/jjkke88/trpo'''
		def _fisher_vector_product(p):
			feed[self.flat_tangent] = p
			return self.session.run(self.fvp, feed) + config.cg_damping * p

		p = b.copy()
		r = b.copy()
		x = np.zeros_like(b)
		rdotr = r.dot(r)

		fmtstr = "%10i %10.3g %10.3g"
		titlestr = "%10s %10s %10s"
		if verbose: print titlestr % ("iter", "residual norm", "soln norm")

		for i in xrange(cg_iters):
			if callback is not None:
				callback(x)
			if verbose:
				print fmtstr % (i, rdotr, np.linalg.norm(x))
			z = _fisher_vector_product(p)
			v = rdotr / (p.dot(z) + 1e-8)
			x += v * p
			r -= v * z
			newrdotr = r.dot(r)
			mu = newrdotr / (rdotr + 1e-8)
			p = r + mu * p

			rdotr = newrdotr
			if rdotr < residual_tol:
				break

		print fmtstr % (i + 1, rdotr, np.linalg.norm(x))
		return x


	def linesearch(f, x, fullstep, expected_improve_rate):
		accept_ratio = .1
		max_backtracks = 10
		fval = f(x)
		for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
		    xnew = x + stepfrac * fullstep
		    newfval = f(xnew)
		    actual_improve = fval - newfval
		    expected_improve = expected_improve_rate * stepfrac
		    ratio = actual_improve / expected_improve
		    if ratio > accept_ratio and actual_improve > 0:
		        return xnew
		return x


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
		# grads = self.session.run(self.cg_ops, feed_dict=feed_dict)
		grads = self.session_run(self.local_network.get_gradients)



		# magic happens



		self.apply_gradients_to_shared_memory_vars(grads)

		new_distributions = self.session.run(
			self.local_network.output_layer_pi,
			feed_dict=feed_dict)

		kl_divergence = np.array([
			utils.stats.kl_divergence(p, q)
			for p, q in zip(prev_distributions, new_distributions)
		]).mean()

		return kl_divergence

                

