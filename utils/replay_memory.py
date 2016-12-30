import collections
import numpy as np


class ReplayMemory(collections.deque):

    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(maxlen=capacity)


    def sample_batch(self, batch_size):
        batch = np.random.choice(len(self), np.minimum(len(self), batch_size))

        s_i = np.array([self[i][0] for i in batch])
        a_i = np.array([self[i][1] for i in batch])
        r_i = np.array([self[i][2] for i in batch])
        s_f = np.array([self[i][3] for i in batch])
        is_terminal = np.array([self[i][4] for i in batch])

        return s_i, a_i, r_i, s_f, is_terminal
