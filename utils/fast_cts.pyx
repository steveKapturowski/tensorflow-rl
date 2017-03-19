cimport cython
import numpy as np
cimport numpy as np
# np.import_array()

from libc.math cimport log, exp


from skimage.transform import resize


# Parameters of the CTS model. For clarity, we take these as constants.
cdef double PRIOR_STAY_PROB = 0.5
cdef double PRIOR_SPLIT_PROB = 0.5
cdef double LOG_PRIOR_STAY_PROB = log(PRIOR_STAY_PROB)
cdef double LOG_PRIOR_SPLIT_PROB = log(1.0 - PRIOR_STAY_PROB)
# Sampling parameter. The maximum number of rejections before we give up and
# sample from the root estimator.
cdef int MAX_SAMPLE_REJECTIONS = 25

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double get_prior(str prior_name, int alphabet_size):
    if prior_name == 'perks':
        return 1.0 / np.double(alphabet_size)
    elif prior_name == 'jeffreys':
        return 0.5
    else: #use laplace prior
        return 1.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double log_add(double log_x, double log_y):
    """Given log x and log y, returns log(x + y)."""
    # Swap variables so log_y is larger.
    if log_x > log_y:
        log_x, log_y = log_y, log_x

    cdef delta = log_y - log_x
    return log(1 + exp(delta)) + log_x if delta <= 50.0 else log_y


class Error(Exception):
    """Base exception for the `cts` module."""
    pass


cdef class Estimator:
    """The estimator for a CTS node.
    This implements a Dirichlet-multinomial model with specified prior. This
    class does not perform alphabet checking, and will return invalid
    probabilities if it is ever fed more than `model.alphabet_size` distinct
    symbols.
    Args:
        model: Reference to CTS model. We expected model.symbol_prior to be
            a `float`.
    """
    cdef double count_total
    cdef CTS _model
    cdef dict counts
    
    def __init__(self, CTS model):
        self.counts = {}
        self.count_total = model.alphabet_size * model.symbol_prior
        self._model = model

    cdef double prob(self, int symbol):
        """Returns the probability assigned to this symbol."""
        cdef double count = self.counts.get(symbol, -1)
        # Allocate new symbols on the fly.
        if count == -1:
            count = self.counts[symbol] = self._model.symbol_prior

        return count / self.count_total

    cdef double update(self, int symbol):
        """Updates our count for the given symbol."""
        cdef double prob = self.prob(symbol)
        cdef double log_prob = log(prob)
        self.counts[symbol] = (
            self.counts.get(symbol, self._model.symbol_prior) + 1.0)
        self.count_total += 1.0
        return log_prob

    
cdef class CTSNode:
    """A node in the CTS tree.
    Each node contains a base Dirichlet estimator holding the statistics for
    this particular context, and pointers to its children.
    """
    cdef double _log_stay_prob
    cdef double _log_split_prob
    cdef CTS _model
    cdef Estimator estimator
    cdef dict _children

    def __init__(self, CTS model):
        self._children = {}

        self._log_stay_prob = LOG_PRIOR_STAY_PROB
        self._log_split_prob = LOG_PRIOR_SPLIT_PROB

        # Back pointer to the CTS model object.
        self._model = model
        self.estimator = Estimator(model)

    cdef double update(self, int[:] context, int symbol):
        """Updates this node and its children.
        Recursively updates estimators for all suffixes of context. Each
        estimator is updated with the given symbol. Also updates the mixing
        weights.
        """
        lp_estimator = self.estimator.update(symbol)

        # If not a leaf node, recurse, creating nodes as needed.
        cdef CTSNode child
        cdef double lp_child
        cdef double lp_node
        if len(context) > 0:
            child = self.get_child(context[-1])
            lp_child = child.update(context[:-1], symbol)
            lp_node = self.mix_prediction(lp_estimator, lp_child)

            self.update_switching_weights(lp_estimator, lp_child)

            return lp_node
        else:
            self._log_stay_prob = 0.0
            return lp_estimator

    cdef double log_prob(self, int[:] context, int symbol):
        cdef double lp_estimator = log(self.estimator.prob(symbol))

        if len(context) > 0:
            child = self.get_child(context[-1])

            lp_child = child.log_prob(context[:-1], symbol)

            return self.mix_prediction(lp_estimator, lp_child)
        else:
            return lp_estimator


    cdef CTSNode get_child(self, int symbol, bint allocate=True):
        cdef CTSNode node = self._children.get(symbol, None)

        if node is None and allocate:
            node = CTSNode(self._model)
            self._children[symbol] = node

        return node

    cdef double mix_prediction(self, double lp_estimator, double lp_child):
        cdef double numerator = log_add(lp_estimator + self._log_stay_prob,
                                     lp_child + self._log_split_prob)
        cdef double denominator = log_add(self._log_stay_prob,
                                       self._log_split_prob)

        return numerator - denominator

    cdef void update_switching_weights(self, double lp_estimator, double lp_child):
        cdef log_alpha = self._model.log_alpha
        cdef log_1_minus_alpha = self._model.log_1_minus_alpha

        # Avoid numerical issues with alpha = 1. This reverts to straight up
        # weighting.
        if log_1_minus_alpha == 0:
            self._log_stay_prob += lp_estimator
            self._log_split_prob += lp_child

        else:
            self._log_stay_prob = log_add(log_1_minus_alpha
                                                   + lp_estimator
                                                   + self._log_stay_prob,
                                                   log_alpha
                                                   + lp_child
                                                   + self._log_split_prob)

            self._log_split_prob = log_add(log_1_minus_alpha
                                                    + lp_child
                                                    + self._log_split_prob,
                                                    log_alpha
                                                    + lp_estimator
                                                    + self._log_stay_prob)
            

cdef class CTS:    
    cdef double _time
    cdef int context_length
    cdef int alphabet_size
    cdef double log_alpha
    cdef double log_1_minus_alpha
    cdef double symbol_prior
    cdef CTSNode _root
    cdef set alphabet
        
    def __init__(self, int context_length, set alphabet=None, int max_alphabet_size=256,
                 str symbol_prior='perks'):
        # Total number of symbols processed.
        self._time = 0.0
        self.context_length = context_length
        
        if alphabet is None:
            self.alphabet, self.alphabet_size = set(), max_alphabet_size
        else:
            self.alphabet, self.alphabet_size = alphabet, len(alphabet)

        # These are properly set when we call update().
        self.log_alpha, self.log_1_minus_alpha = 0.0, 0.0
        self.symbol_prior = get_prior(symbol_prior, self.alphabet_size) 


        # Create root. This must happen after setting alphabet & symbol prior.
        self._root = CTSNode(self)


    cpdef double update(self, int[:] context, int symbol):
        self._time += 1.0
        self.log_alpha = log(1.0 / (self._time + 1.0))
        self.log_1_minus_alpha = log(self._time / (self._time + 1.0))

        self.alphabet.add(symbol)

        cdef double log_prob = self._root.update(context, symbol)

        return log_prob

    cpdef double log_prob(self, int[:] context, int symbol):
        #context is assumed to have correct length
        return self._root.log_prob(context, symbol)


cdef class CTSDensityModel:
    cdef int height
    cdef int width
    cdef float beta
    cdef np.ndarray factors

    def __init__(self, int height=42, int width=42, float beta=0.05):
        self.height = height
        self.width = width
        self.beta = beta
        self.factors = np.array([[CTS(4) for _ in range(width)] for _ in range(height)])
        
    def get_height(self):
        return self.height
        
    def get_width(self):
        return self.width
        
    def update(self, obs):
        obs = resize(obs, (self.height, self.width))
        obs = (obs*16).astype(np.int32)
        log_prob, log_recoding_prob = self._update(obs)
        return self.exploration_bonus(log_prob, log_recoding_prob)
    
    cdef (double, double) _update(self, int[:, :] obs):
        cdef int[:] context = np.array([0, 0, 0, 0], np.int32)
        cdef double log_prob = 0.0
        cdef double log_recoding_prob = 0.0
        cdef int i
        cdef int j
        cdef np.ndarray[object, ndim=2] factors = self.factors

        for i in range(self.height):
            for j in range(self.width):
                context[0] = obs[i, j-1] if j > 0 else 0
                context[1] = obs[i-1, j] if i > 0 else 0
                context[2] = obs[i-1, j-1] if i > 0 and j > 0 else 0
                context[3] = obs[i-1, j+1] if i > 0 and j < factors.shape[1]-1 else 0

                log_prob += factors[i, j].update(context, obs[i, j])
                log_recoding_prob += factors[i, j].log_prob(context, obs[i, j])

        return log_prob, log_recoding_prob



    def exploration_bonus(self, log_prob, log_recoding_prob):
        prob = np.exp(log_prob)
        recoding_prob = np.exp(log_recoding_prob)

        pseudocount = prob * (1 - recoding_prob) / np.maximum(recoding_prob - prob, 1e-10)
        return self.beta / np.sqrt(pseudocount + .01)


__all__ = ["CTS", "CTSDensityModel"]