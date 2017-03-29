# -*- coding: utf-8 -*-
import collections
import numpy as np


class ReplayMemory(collections.deque):

	def __init__(self, capacity):
		super(ReplayMemory, self).__init__(maxlen=capacity)


	# def sample_batch(self, batch_size):
	#     batch = np.random.choice(len(self), np.minimum(len(self), batch_size))
	#     data = [self[i] for i in batch]

	#     return [
	#         np.array([e[j] for e in data])
	#         for j in range(len(data[0]))]


	def sample_batch(self, batch_size):
		batch = np.random.choice(len(self)-1, np.minimum(len(self), batch_size))
		
		i_states  = np.array([self[i][0] for i in batch  ])
		actions   = np.array([self[i][1] for i in batch  ])
		rewards   = np.array([self[i][2] for i in batch  ])
		f_states  = np.array([self[i][0] for i in batch+1])
		terminals = np.array([self[i][3] for i in batch  ])

		return (
			i_states,
			actions,
			rewards,
			f_states,
			terminals,
		)


	def append(self, data):
		data = data[:3] + data[4:]
		super(ReplayMemory, self).append(data)




		
