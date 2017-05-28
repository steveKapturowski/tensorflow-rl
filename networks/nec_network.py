# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from q_network import QNetwork
from utils.dnd import DND


class NECNetwork(QNetwork):
	def __init__(self, conf):
		self.delta = 1e-3
		self.capacity = 100000
		self.action_dnds = [DND(self.capacity) for _ in self.num_actions]

		super(NECNetwork, self).__init__(conf)


	def _build_q_head(self, input_state):
		self.q_values = tf.py_func(self.q_value_lookup, key, tf.float32)
		self.q_selected_action = tf.reduce_sum(self.q_values * self.selected_action_ph, axis=1)

		diff = tf.subtract(self.target_ph, self.q_selected_action)
		return self._huber_loss(diff)


	def q_value_lookup(self, key):
		values = np.zeros((self.num_actions,), np.float32)

		for i, dnd in enumerate(self.action_dnds):
			values, distances = dnd.query(input_state)
			kernel = 1 / (distances**2 + self.delta)
			values[i] = kernel * values

		return values
