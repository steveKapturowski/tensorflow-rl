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

	def params(self):
		return tf.concat(1, [self.mu, self.sigma])

	def sample(self):
		return self.mu + self.sigma * tf.random_normal([self.dim])

	def log_likelihood(self, x):
		return -tf.reduce_sum(
			0.5 * tf.square((x - self.mu) / (self.sigma + 1e-8))
			+ tf.log(self.sigma + 1e-8) - 0.5 * tf.log(2.0 * np.pi), axis=1)

	def entropy(self):
		return tf.reduce_sum(tf.log(self.sigma + 1e-8) + 0.5 * np.log(2 * np.pi * np.e), axis=1)

	def kl_divergence(self, params):
		mu_2, sigma_2 = tf.split(1, 2, params)
		return tf.reduce_sum(
			(tf.square(sigma_2) + tf.square(mu_2 - self.mu)) / (2.0 * tf.square(self.sigma) + 1e-8) 
			+ tf.log(self.sigma/(sigma_2 + 1e-8) + 1e-8) - 0.5, axis=1)


class Discrete(object):
	def __init__(self, logits):
		self.logits = logits
		self.probs = tf.nn.softmax(self.logits)
		self.log_probs = tf.nn.log_softmax(self.logits)
		self.dim = tf.shape(self.logits)[1]

	def params(self):
		return self.logits

	def sample(self):
		noisy_logits = self.logits - tf.log(-tf.log(tf.random_uniform([self.dim])))
		return tf.one_hot(tf.argmax(noisy_logits, self.dim))

	def log_likelihood(self, x):
		return self.log_probs

	def entropy(self):
		return tf.reduce_sum(self.probs * self.log_probs, axis=1)

	def kl_divergence(self, logits):
		eps = 1e-8
		probs = tf.nn.softmax(logits)
		return tf.reduce_sum(probs * tf.log((probs + eps) / (self.probs + eps)), axis=1)

