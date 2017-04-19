# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np


class ReplayMemory(object):

	def __init__(self, maxlen, input_shape, action_size):
		self.maxlen = maxlen
		dirname = tempfile.mkdtemp()
		#use memory maps so we won't have to worry about eating up lots of RAM
		get_path = lambda name: os.path.join(dirname, name)
		self.screens = np.memmap(get_path('screens'), dtype=np.float32, mode='w+', shape=tuple([self.maxlen]+input_shape))
		self.actions = np.memmap(get_path('actions'), dtype=np.float32, mode='w+', shape=(self.maxlen, action_size))
		self.rewards = np.memmap(get_path('rewards'), dtype=np.float32, mode='w+', shape=(self.maxlen,))
		self.is_terminal = np.memmap(get_path('terminals'), dtype=np.bool, mode='w+', shape=(self.maxlen,))

		self.position = 0
		self.full = False

	# def _get_states(batch):
	# 	s = list()
	# 	for i in xrange(-3, 2):
	# 		s.append(self.screens[batch+i])
			
	# 	return np.vstack(s[:-1]), np.vstack(s[1:])

	def sample_batch(self, batch_size):
		batch = np.random.choice(len(self)-1, np.minimum(len(self), batch_size))
		
		# s_i, s_f = self._get_state(batch)
		s_i = self.screens[batch]
		s_f = self.screens[batch+1]
		a = self.actions[batch]
		r = self.rewards[batch]
		is_terminal = self.is_terminal[batch]

		return s_i, a, r, s_f, is_terminal

	def __len__(self):
		return self.maxlen if self.full else self.position

	def append(self, s_i, a, r, is_terminal):
		self.screens[self.position] = s_i
		self.actions[self.position] = a
		self.rewards[self.position] = r
		self.is_terminal[self.position] = is_terminal

		self.position = (self.position + 1) % self.maxlen




		
