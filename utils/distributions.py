import numpy as np
import tensorflow as tf


class DiagNormal(object):
	'''
	Models Gaussian with Diagonal Covariance
	'''
	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma
		self.dim = tf.shape(self.mu)[1]

	def sample(self):
		return self.mu + self.sigma * tf.random_normal([self.dim])

	def log_likelihood(self, x):
		return -tf.reduce_sum(
			0.5 * tf.square((x - self.mu) / (self.sigma + 1e-8))
			+ tf.log(self.sigma + 1e-8) - 0.5 * tf.log(2.0 * np.pi), axis=1)

	def entropy(self):
		# return tf.reduce_sum(tf.log(self.sigma) + 0.5 * np.log(2 * np.pi * np.e), axis=1)
		return 0.5 * tf.reduce_sum(tf.log(2 * np.pi * self.sigma + 1e-8) + 1, axis=1)

	def kl_divergence(mu_2, sigma_2):
		return tf.reduce_sum(
			(tf.square(self.sigma) + tf.square(self.mu - mu_2)) / (2.0 * tf.square(sigma_2) + 1e-8) 
			+ tf.log(sigma_2/(self.sigma + 1e-8) + 1e-8) - 0.5, axis=1)

