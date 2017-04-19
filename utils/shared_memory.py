# -*- coding: utf-8 -*-
from multiprocessing import RawValue, RawArray, Semaphore, Lock
import ctypes
import numpy as np
import tensorflow as tf


class SharedCounter(object):
    def __init__(self, initval=0):
        self.val = RawValue('i', initval)
        self.last_step_update_target = RawValue('i', initval)
        self.lock = Lock()

    def increment(self, elapsed_steps=None):
        self.val.value += 1
        if ((elapsed_steps is not None) 
            and ((self.val.value - self.last_step_update_target.value) 
                >= elapsed_steps)):
            self.last_step_update_target.value = self.val.value
            return self.val.value, True
        else:
            return self.val.value, False

    def set_value(self, value):
        self.lock.acquire()
        self.val.value = value
        self.lock.release()

    def value(self):
        return self.val.value

class Barrier:
    def __init__(self, n):
        self.n = n
        self.counter = SharedCounter(0)
        self.barrier = Semaphore(0)

    def wait(self):
        with self.counter.lock:
            self.counter.val.value += 1
            if self.counter.val.value == self.n:
                self.barrier.release()

        self.barrier.acquire()
        self.barrier.release()

class SharedVars(object):
    def __init__(self, params, opt_type=None, lr=0, step=0):
        self.var_shapes = [
            var.get_shape().as_list()
            for var in params]
        self.size = sum([np.prod(shape) for shape in self.var_shapes])
        self.step = RawValue(ctypes.c_int, step)

        if opt_type == 'adam':
            self.ms = self.malloc_contiguous(self.size)
            self.vs = self.malloc_contiguous(self.size)
            self.lr = RawValue(ctypes.c_float, lr)
        elif opt_type == 'adamax':
            self.ms = self.malloc_contiguous(self.size)
            self.vs = self.malloc_contiguous(self.size)
            self.lr = RawValue(ctypes.c_float, lr)
        elif opt_type == 'rmsprop':
            self.vars = self.malloc_contiguous(self.size, np.ones(self.size, dtype=np.float))
        elif opt_type == 'momentum':
            self.vars = self.malloc_contiguous(self.size)
        else:
            self.vars = self.malloc_contiguous(self.size)

            
    def malloc_contiguous(self, size, initial_val=None):
        if initial_val is None:
            return RawArray(ctypes.c_float, size)
        else:
            return RawArray(ctypes.c_float, initial_val)


class SharedFlags(object):
    def __init__(self, num_actors):
        self.updated = RawArray(ctypes.c_int, num_actors)
            
