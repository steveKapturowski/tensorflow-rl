# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np


def mean_kl_divergence_op(P, Q, eps=1e-10):
	return tf.reduce_mean(tf.reduce_sum(P * tf.log((P + eps) / (Q + eps)), axis=1))


def kl_divergence(P, Q, eps = 1e-10):
	return (P * np.log((P + eps) / (Q + eps))).sum()


def jenson_shannon_divergence(P, Q, eps=1e-10):
	M = 0.5 * (P + Q)
	return 0.5 * (kl_divergence(P, M, eps=eps) + kl_divergence(Q, M, eps=eps))


def ar1_process(x_previous, mean, theta, sigma):
	'''Discrete Ornstein–Uhlenbeck / AR(1) process to produce temporally correlated noise'''
	return theta*(mean - x_previous) + sigma*np.random.normal()