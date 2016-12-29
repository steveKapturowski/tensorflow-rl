import collections
import numpy as np


class ReplayMemory(collections.deque):

    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(maxlen=capacity)


    def sample_batch(self, batch_size):
        batch = np.random.choice(len(self), np.minimum(len(self), batch_size))

        s_i = np.array([self[i][0] for i in batch if not self[i][4]])
        a_i = np.array([self[i][1] for i in batch if not self[i][4]])
        r_i = np.array([self[i][2] for i in batch if not self[i][4]])
        s_f = np.array([self[i][3] for i in batch if not self[i][4]])
        is_terminal = np.array([self[i][4] for i in batch])

        return s_i, a_i, r_i, s_f, is_terminal
