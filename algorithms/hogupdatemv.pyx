#from cython.parallel import prange
import cython
cimport cython
import numpy as np
cimport numpy as np
import ctypes
#from libc.stdlib cimport memcpy

cdef extern from "math.h" nogil:
    double sqrt(double m)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_copy(float[::1] dest, float[::1] src, unsigned int src_size) nogil:
    dest[:] = src
    #memcpy(&dest, &src, src_size * sizeof(float))
    
def copy(d, s):
    c_copy(d, s, s.size)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_apply_grads_mom_rmsprop(float[::1] m, 
                               float[::1] g, 
                               float[::1] p, 
                               unsigned int p_size, 
                               int _type, 
                               float lr,
                               float alpha, 
                               float e) nogil:
    
    cdef unsigned int i
    #for i in prange(p_size): #, schedule='static', chunksize=p_size/15):
    for i in range(p_size):
        m[i] = alpha * m[i] + (1 - alpha) * (g[i] ** 2)
        p[i] -= lr * g[i] / sqrt(m[i] + e)
        

def apply_grads_mom_rmsprop(_m, g, v, v_size, _type, lr, alpha, e):
    c_apply_grads_mom_rmsprop(_m, g, v, v_size, _type, lr, alpha, e)

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_apply_grads_adam(float[::1] m, 
                        float[::1] v, 
                        float[::1] g, 
                        float[::1] p,
                        unsigned int p_size, 
                        float lr, 
                        float b1, 
                        float b2, 
                        float e) nogil:
     
    for i in range(p_size):
        m[i] = b1 * m[i] + (1 - b1) * g[i]
        v[i] = b2 * v[i] + (1 - b2) * (g[i] ** 2)
        # OBS The learning rate lr we get should have already been updated outside this function according to the ADAM proposed formula 
        p[i] -= lr * m[i] / (sqrt(v[i]) + e)
        
def apply_grads_adam(m, v, g, p, p_size, lr, b1, b2, e):
    c_apply_grads_adam(m, v, g, p, p_size, lr, b1, b2, e)
    

    
    
