import tensorflow as tf
import numpy as np


def slice_2d(x, inds0, inds1):
	#adapted from https://github.com/jjkke88/trpo/blob/master/utils.py
	inds0 = tf.cast(inds0, tf.int32)
	inds1 = tf.cast(inds1, tf.int32)
	shape = tf.cast(tf.shape(x), tf.int32)
	ncols = shape[1]
	x_flat = tf.reshape(x, [-1])
	return tf.gather(x_flat, inds0 * ncols + inds1)


def flatten_vars(var_list):
	return tf.concat([
		tf.reshape(v, [np.prod(v.get_shape().as_list())])
		for v in var_list
	], 0)