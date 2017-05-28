import numpy as np
from annoy import AnnoyIndex
from collections import deque, OrderedDict


class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.keys = OrderedDict()

    def update(self, idx):
        if idx in self.keys:
            del self.keys[idx]
        self.keys[idx] = True

        if len(self.keys) > self.capacity-1:
            return self.keys.popitem(last=False)[0]
        else:
            return len(self.keys)


class DND(object):
    def __init__(self, capacity=100000, key_size=128, cache_size=32):
        self.capacity = capacity
        self.lru_cache = LRUCache(capacity)
        self.dup_cache = deque(maxlen=cache_size)
        self.index = AnnoyIndex(key_size, metric='euclidean')
        self.update_batch = np.zeros((1000, key_size))
        self.keys = np.zeros((capacity, key_size), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.insert_idx = 0
        self.insertions = 0

    def add(self, key, value):
        if not self.cache_lookup(key, value):
            self.keys[self.insert_idx] = key
            self.values[self.insert_idx] = value
            self.dup_cache.append(key)
            self.index.add_item(self.insert_idx, key)
            #advance insert position to least-recently-used key
            self.insert_idx = self.lru_cache.update(self.insert_idx)

        self.insertions += 1
        self.update_batch[self.insertions % 1000] = key
        #rebuilding the index is expensive so we don't want to do it too often
        if self.insertions % 1000 == 0:
            self.rebuild_index()


    def cache_lookup(self, key, value):
        for i, e in enumerate(self.dup_cache):
            if np.allclose(key, e):
                idx = self.size - len(self.dup_cache) + i
                self.values[idx] += self.alpha * value

                return True

    def rebuild_index(self):
        self.index.unbuild()
        self.index.build(50)


    def query(self, key, k_neighbors=40):
        indices, distances = self.index.get_nns_by_vector(
            key, k_neighbors, include_distances=True)

        for idx in indices:
            self.lru_cache.update(idx)

        return self.values[indices], distances


