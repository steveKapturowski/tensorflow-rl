import cython
cimport cython
import numpy as np
cimport numpy as np
#from libc.stdlib cimport memcpy

#cdef extern float __sync_fetch_and_sub (float *var, float value) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void c_update(float[::1] p, unsigned int p_size, float[::1] g) nogil:
    cdef unsigned int i
    #for (i = 0; i < p_size; i++)
    for i in range(p_size):
        p[i] -= g[i]
        #__sync_fetch_and_sub(&p[i], g[i])


def update(p, s, g):
    c_update(p, s, g)


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
cdef c_apply_grads_mom_rmsprop(np.ndarray[float, ndim=1, mode="c"] m, 
                               np.ndarray[float, ndim=1, mode="c"] g, 
                               float[::1] p, 
                               unsigned int p_size, 
                               int _type, 
                               float lr,
                               float alpha, 
                               float e):
    
    if (_type == 0):  # momentum   
        m[:] = alpha * m + (1 - alpha) * g
        g[:] = lr * m
    
    elif (_type == 1):  # rmsprop
        m[:] = alpha * m + (1 - alpha) * (g ** 2)
        g[:] = lr * g / np.sqrt(m + e)
        
    # Apply grads
    c_update(p, p_size, g)

def apply_grads_mom_rmsprop(_m, g, v, v_size, _type, lr, alpha, e):
    c_apply_grads_mom_rmsprop(_m, g, v, v_size, _type, lr, alpha, e)

    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef c_apply_grads_adam(np.ndarray[float, ndim=1, mode="c"] m, 
                        np.ndarray[float, ndim=1, mode="c"] v, 
                        np.ndarray[float, ndim=1, mode="c"] g, 
                        float[::1] p,
                        unsigned int p_size, 
                        float lr, 
                        float b1, 
                        float b2, 
                        float e):
     
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * (g ** 2)
    # OBS The learning rate lr we get, should have already been updated outside this function according to the ADAM proposed formula 
    g[:] = lr * m / (np.sqrt(v) + e)
        
    # Apply grads
    c_update(p, p_size, g)

def apply_grads_adam(m, v, g, p, p_size, lr, b1, b2, e):
    c_apply_grads_adam(m, v, g, p, p_size, lr, b1, b2, e)
    

    
    
