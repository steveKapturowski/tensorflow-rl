import numpy as np
import tensorflow as tf


class DiagNormal(object):
	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma
		self.dim = tf.shape(self.mu)[1]

	def log_likelihood(self, x):
		d = tf.cast(self.dim, tf.float32)
		return - 0.5 * tf.reduce_sum(tf.pow((x - self.mu) / self.sigma, 2), axis=1) \
			- 0.5 * tf.log(2.0 * np.pi) * d - tf.reduce_sum(tf.log(self.sigma), axis=1)

	def entropy(self):
		d = tf.cast(self.dim, tf.float32)
		return tf.reduce_sum(tf.log(self.sigma), axis=1) + .5 * np.log(2 * np.pi * np.e) * d

	def sample(self):
		return self.mu + self.sigma * tf.random_normal([self.dim])

