import collections
import numpy as np


class ReplayMemory(collections.deque):

    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(maxlen=capacity)


    def sample_batch(self, batch_size):
        batch = np.random.sample(self, np.maximum(len(self), batch_size))

        s_i = np.array([e[0] for e in batch])
        a_i = np.array([e[1] for e in batch])
        r_i = np.array([e[2] for e in batch])
        s_i_plus_one = np.array([e[4] for e in batch])
        is_terminal = np.array([e[3] for e in batch])

        return s_i, a_i, r_i, s_i_plus_one, is_terminal
