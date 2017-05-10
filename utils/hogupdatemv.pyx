#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#from cython.parallel import prange
import cython
cimport cython
from libc.math cimport fabs, sqrt


cdef void c_copy(float[::1] dest, float[::1] src, unsigned int src_size) nogil:
    dest[:] = src
    #memcpy(&dest, &src, src_size * sizeof(float))
    
def copy(d, s):
    c_copy(d, s, s.size)


cdef void c_apply_grads_mom_rmsprop(float[::1] m, 
                               float[::1] g, 
                               float[::1] p, 
                               unsigned int p_size, 
                               int _type, 
                               float lr,
                               float alpha, 
                               float e) nogil:
    
    cdef unsigned int i
    if (_type == 0):  # momentum   
        for i in range(p_size):
            m[i] = alpha * m[i] + (1 - alpha) * g[i]
            p[i] -= lr * m[i]
    elif (_type == 1):  # rmsprop
        for i in range(p_size):
            m[i] = alpha * m[i] + (1 - alpha) * (g[i] ** 2)
            p[i] -= lr * g[i] / sqrt(m[i] + e)
        

def apply_grads_mom_rmsprop(_m, g, v, v_size, _type, lr, alpha, e):
    c_apply_grads_mom_rmsprop(_m, g, v, v_size, _type, lr, alpha, e)


cdef void c_apply_grads_adam(float[::1] m, 
                        float[::1] v, 
                        float[::1] g, 
                        float[::1] p,
                        unsigned int p_size, 
                        float lr, 
                        float b1, 
                        float b2, 
                        float e) nogil:
     
    cdef unsigned int i
    for i in range(p_size):
        m[i] = b1 * m[i] + (1 - b1) * g[i]
        v[i] = b2 * v[i] + (1 - b2) * (g[i] ** 2)
        # OBS The learning rate lr we get should have already been updated outside this function according to the ADAM proposed formula 
        p[i] -= lr * m[i] / (sqrt(v[i]) + e)
        
def apply_grads_adam(m, v, g, p, p_size, lr, b1, b2, e):
    c_apply_grads_adam(m, v, g, p, p_size, lr, b1, b2, e)


cdef void c_apply_grads_adamax(float[::1] m,
                        float[::1] u,
                        float[::1] g,
                        float[::1] p,
                        unsigned int p_size,
                        float lr,
                        float beta_1,
                        float beta_2,
                        unsigned int t) nogil:
    
    cdef float term1, term2
    cdef unsigned int i
    
    for i in range(p_size):
        m[i] = beta_1 * m[i] + (1 - beta_1) * g[i]
        term1 = beta_2 * u[i] + 1e-7
        term2 = fabs(g[i])

        u[i] = term1
        if term2 > term1:
            u[i] = term2

        p[i] -= (lr / (1 - beta_1**t)) * m[i] / u[i]
        
def apply_grads_adamax(m, u, g, p, p_size, lr, beta_1, beta_2, t):
    c_apply_grads_adamax(m, u, g, p, p_size, lr, beta_1, beta_2, t)


    
    
