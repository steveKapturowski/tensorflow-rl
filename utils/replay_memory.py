# -*- coding: utf-8 -*-
import collections
import numpy as np


class ReplayMemory(collections.deque):

    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(maxlen=capacity)


    def sample_batch(self, batch_size):
        batch = np.random.choice(len(self), np.minimum(len(self), batch_size))
        data = [self[i] for i in batch]

        return [
            np.array([e[j] for e in data])
            for j in range(len(data[0]))]

