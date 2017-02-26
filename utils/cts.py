# -*- coding: utf-8 -*-
"""An implementation of the Context Tree Switching model.
The Context Tree Switching algorithm (Veness et al., 2012) is a variable
order Markov model with pleasant regret guarantees. It achieves excellent
performance over binary, and more generally small alphabet data.
This is a fairly vanilla implementation with readability favoured over
performance.
"""

import math
import random
import sys


# Parameters of the CTS model. For clarity, we take these as constants.
PRIOR_STAY_PROB = 0.5
PRIOR_SPLIT_PROB = 0.5
LOG_PRIOR_STAY_PROB = math.log(PRIOR_STAY_PROB)
LOG_PRIOR_SPLIT_PROB = math.log(1.0 - PRIOR_STAY_PROB)
# Sampling parameter. The maximum number of rejections before we give up and
# sample from the root estimator.
MAX_SAMPLE_REJECTIONS = 25

# These define the prior count assigned to each unseen symbol.
ESTIMATOR_PRIOR = {
    'laplace': (lambda unused_alphabet_size: 1.0),
    'jeffreys': (lambda unused_alphabet_size: 0.5),
    'perks': (lambda alphabet_size: 1.0 / alphabet_size),
}


def log_add(log_x, log_y):
    """Given log x and log y, returns log(x + y)."""
    # Swap variables so log_y is larger.
    if log_x > log_y:
        log_x, log_y = log_y, log_x

    # Use the log(1 + e^p) trick to compute this efficiently
    # If the difference is large enough, this is effectively log y.
    delta = log_y - log_x
    return math.log1p(math.exp(delta)) + log_x if delta <= 50.0 else log_y


class Error(Exception):
    """Base exception for the `cts` module."""
    pass

class Estimator(object):
    """The estimator for a CTS node.
    This implements a Dirichlet-multinomial model with specified prior. This
    class does not perform alphabet checking, and will return invalid
    probabilities if it is ever fed more than `model.alphabet_size` distinct
    symbols.
    Args:
        model: Reference to CTS model. We expected model.symbol_prior to be
            a `float`.
    """
    def __init__(self, model):
        self.counts = {}
        self.count_total = model.alphabet_size * model.symbol_prior
        self._model = model

    def prob(self, symbol):
        """Returns the probability assigned to this symbol."""
        count = self.counts.get(symbol, None)
        # Allocate new symbols on the fly.
        if count is None:
            count = self.counts[symbol] = self._model.symbol_prior

        return count / self.count_total

    def update(self, symbol):
        """Updates our count for the given symbol."""
        log_prob = math.log(self.prob(symbol))
        self.counts[symbol] = (
            self.counts.get(symbol, self._model.symbol_prior) + 1.0)
        self.count_total += 1.0
        return log_prob

    def sample(self, rejection_sampling):
        """Samples this estimator's PDF in linear time."""
        if rejection_sampling:
            # Automatically fail if this estimator is empty.
            if not self.counts:
                return None
            else:
                # TODO(mgbellemare): No need for rejection sampling here --
                # renormalize instead.
                symbol = None
                while symbol is None:
                    symbol = self._sample_once(use_prior_alphabet=False)

            return symbol
        else:
            if len(self._model.alphabet) < self._model.alphabet_size:
                raise Error(
                    'Cannot sample from prior without specifying alphabet')
            else:
                return self._sample_once(use_prior_alphabet=True)

    def _sample_once(self, use_prior_alphabet):
        """Samples once from the PDF.
        Args:
            use_prior_alphabet: If True, we will sample the alphabet given
                by the model to account for symbols not seen by this estimator.
                Otherwise, we return None.
        """
        random_index = random.uniform(0, self.count_total)

        for item, count in self.counts.items():
            if random_index < count:
                return item
            else:
                random_index -= count

        # We reach this point when we sampled a symbol which is not stored in
        # `self.counts`.
        if use_prior_alphabet:
            for symbol in self._model.alphabet:
                # Ignore symbols already accounted for.
                if symbol in self.counts:
                    continue
                elif random_index < self._model.symbol_prior:
                    return symbol
                else:
                    random_index -= self._model.symbol_prior

            # Account for numerical errors.
            if random_index < self._model.symbol_prior:
                sys.stderr.write('Warning: sampling issues, random_index={}'.
                                 format(random_index))
                # Return first item by default.
                return list(self._model.alphabet)[0]
            else:
                raise Error('Sampling failure, not enough symbols')
        else:
            return None

class CTSNode(object):
    """A node in the CTS tree.
    Each node contains a base Dirichlet estimator holding the statistics for
    this particular context, and pointers to its children.
    """

    def __init__(self, model):
        self._children = {}

        self._log_stay_prob = LOG_PRIOR_STAY_PROB
        self._log_split_prob = LOG_PRIOR_SPLIT_PROB

        # Back pointer to the CTS model object.
        self._model = model
        self.estimator = Estimator(model)

    def update(self, context, symbol):
        """Updates this node and its children.
        Recursively updates estimators for all suffixes of context. Each
        estimator is updated with the given symbol. Also updates the mixing
        weights.
        """
        lp_estimator = self.estimator.update(symbol)

        # If not a leaf node, recurse, creating nodes as needed.
        if len(context) > 0:
            # We recurse on the last element of the context vector.
            child = self.get_child(context[-1])
            lp_child = child.update(context[:-1], symbol)

            # This node predicts according to a mixture between its estimator
            # and its child.
            lp_node = self.mix_prediction(lp_estimator, lp_child)

            self.update_switching_weights(lp_estimator, lp_child)

            return lp_node
        else:
            # The log probability of staying at a leaf is log(1) = 0. This
            # isn't actually used in the code, tho.
            self._log_stay_prob = 0.0
            return lp_estimator

    def log_prob(self, context, symbol):
        """Computes the log probability of the symbol in this subtree."""
        lp_estimator = math.log(self.estimator.prob(symbol))

        if len(context) > 0:
            # See update() above. More efficient is to avoid creating the
            # nodes and use a default node, but we omit this for clarity.
            child = self.get_child(context[-1])

            lp_child = child.log_prob(context[:-1], symbol)

            return self.mix_prediction(lp_estimator, lp_child)
        else:
            return lp_estimator

    def sample(self, context, rejection_sampling):
        """Samples a symbol in the given context."""
        if len(context) > 0:
            # Determine whether we should use this estimator or our child's.
            log_prob_stay = (self._log_stay_prob
                             - log_add(self._log_stay_prob,
                                                self._log_split_prob))

            if random.random() < math.exp(log_prob_stay):
                return self.estimator.sample(rejection_sampling)
            else:
                # If using child, recurse.
                if rejection_sampling:
                    child = self.get_child(context[-1], allocate=False)
                    # We'll request another sample from the tree.
                    if child is None:
                        return None
                # TODO(mgbellemare): To avoid rampant memory allocation,
                # it's worthwhile to use a default estimator here rather than
                # recurse when the child is not allocated.
                else:
                    child = self.get_child(context[-1])

                symbol = child.sample(context[:-1], rejection_sampling)
                return symbol
        else:
            return self.estimator.sample(rejection_sampling)

    def get_child(self, symbol, allocate=True):
        """Returns the node corresponding to this symbol.
        New nodes are created as required, unless allocate is set to False.
        In this case, None is returned.
        """
        node = self._children.get(symbol, None)

        # If needed and requested, allocated a new node.
        if node is None and allocate:
            node = CTSNode(self._model)
            self._children[symbol] = node

        return node

    def mix_prediction(self, lp_estimator, lp_child):
        """Returns the mixture x = w * p + (1 - w) * q.
        Here, w is the posterior probability of using the estimator at this
        node, versus using recursively calling our child node.
        The mixture is computed in log space, which makes things slightly
        trickier.
        Let log_stay_prob_ = p' = log p, log_split_prob_ = q' = log q.
        The mixing coefficient w is
                w = e^p' / (e^p' + e^q'),
                v = e^q' / (e^p' + e^q').
        Then
                x = (e^{p' w'} + e^{q' v'}) / (e^w' + e^v').
        """
        numerator = log_add(lp_estimator + self._log_stay_prob,
                                     lp_child + self._log_split_prob)
        denominator = log_add(self._log_stay_prob,
                                       self._log_split_prob)
        return numerator - denominator

    def update_switching_weights(self, lp_estimator, lp_child):
        """Updates the switching weights according to Veness et al. (2012)."""
        log_alpha = self._model.log_alpha
        log_1_minus_alpha = self._model.log_1_minus_alpha

        # Avoid numerical issues with alpha = 1. This reverts to straight up
        # weighting.
        if log_1_minus_alpha == 0:
            self._log_stay_prob += lp_estimator
            self._log_split_prob += lp_child
        # Switching rule. It's possible to make this more concise, but we
        # leave it in full for clarity.
        else:
            # Mix in an alpha fraction of the other posterior:
            #   switchingStayPosterior = ((1 - alpha) * stayPosterior
            #                            + alpha * splitPosterior)
            # where here we store the unnormalized posterior.
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


class CTS(object):
    """A class implementing Context Tree Switching.
    This version works with large alphabets. By default it uses a Dirichlet
    estimator with a Perks prior (works reasonably well for medium-sized,
    sparse alphabets) at each node.
    Methods in this class assume a human-readable context ordering, where the
    last symbol in the context list is the most recent (in the case of
    sequential prediction). This is slightly unintuitive from a computer's
    perspective but makes the update more legible.
    There are also only weak constraints on the alphabet. Basically: don't use
    more than alphabet_size symbols unless you know what you're doing. These do
    symbols can be any integers and need not be contiguous.
    Alternatively, you may set the full alphabet before using the model.
    This will allow sampling from the model prior (which is otherwise not
    possible).
    """

    def __init__(self, context_length, alphabet=None, max_alphabet_size=256,
                 symbol_prior='perks'):
        """CTS constructor.
        Args:
            context_length: The number of variables which CTS conditions on.
                In general, increasing this term increases prediction accuracy
                and memory usage.
            alphabet: The alphabet over which we operate, as a `set`. Set to
                None to allow CTS to dynamically determine the alphabet.
            max_alphabet_size: The total number of symbols in the alphabet. For
                character-level prediction, leave it at 256 (or set alphabet).
                If alphabet is specified, this field is ignored.
            symbol_prior: (float or string) The prior used within each node's
                Dirichlet estimator. If a string is given, valid choices are
                'dirichlet', 'jeffreys', and 'perks'. This defaults to 'perks'.
        """
        # Total number of symbols processed.
        self._time = 0.0
        self.context_length = context_length
        # We store the observed alphabet in a set.
        if alphabet is None:
            self.alphabet, self.alphabet_size = set(), max_alphabet_size
        else:
            self.alphabet, self.alphabet_size = alphabet, len(alphabet)

        # These are properly set when we call update().
        self.log_alpha, self.log_1_minus_alpha = 0.0, 0.0

        # If we have an entry for it in our set of default priors, assume it's
        # one of our named priors.
        if symbol_prior in ESTIMATOR_PRIOR:
            self.symbol_prior = (
                float(ESTIMATOR_PRIOR[symbol_prior](self.alphabet_size)))
        else:
            self.symbol_prior = float(symbol_prior)  # Otherwise assume numeric.

        # Create root. This must happen after setting alphabet & symbol prior.
        self._root = CTSNode(self)

    def _check_context(self, context):
        """Verifies that the given context is of the expected length.
        Args:
            context: Context to be verified.
        """
        if self.context_length != len(context):
            raise Error('Invalid context length, {} != {}'
                        .format(self.context_length, len(context)))

    def update(self, context, symbol):
        """Updates the CTS model.
        Args:
            context: The context list, of size context_length, describing
                the variables on which CTS should condition. Context elements
                are assumed to be ranked in increasing order of importance.
                For example, in sequential prediction the most recent symbol
                should be context[-1].
            symbol: The symbol observed in this context.
        Returns:
            The log-probability assigned to the symbol before the update.
        Raises:
            Error: Provided context is of incorrect length.
        """
        # Set the switching parameters.
        self._time += 1.0
        self.log_alpha = math.log(1.0 / (self._time + 1.0))
        self.log_1_minus_alpha = math.log(self._time / (self._time + 1.0))

        # Nothing in the code actually prevents invalid contexts, but the
        # math won't work out.
        self._check_context(context)

        # Add symbol to seen alphabet.
        self.alphabet.add(symbol)
        if len(self.alphabet) > self.alphabet_size:
            raise Error('Too many distinct symbols')

        log_prob = self._root.update(context, symbol)

        return log_prob

    def log_prob(self, context, symbol):
        """Queries the CTS model.
        Args:
            context: As per ``update()``.
            symbol: As per ``update()``.
        Returns:
            The log-probability of the symbol in the context.
        Raises:
            Error: Provided context is of incorrect length.
        """
        self._check_context(context)
        return self._root.log_prob(context, symbol)

    def sample(self, context, rejection_sampling=True):
        """Samples a symbol from the model.
        Args:
            context: As per ``update()``.
            rejection_sampling: Whether to ignore samples from the prior.
        Returns:
            A symbol sampled according to the model. The default mode of
            operation is rejection sampling, which will ignore draws from the
            prior. This allows us to avoid providing an alphabet in full, and
            typically produces more pleasing samples (because they are never
            drawn from data for which we have no prior). If the full alphabet
            is provided (by setting self.alphabet) then `rejection_sampling`
            may be set to False. In this case, models may sample symbols in
            contexts they have never appeared in. This latter mode of operation
            is the mathematically correct one.
        """
        if self._time == 0 and rejection_sampling:
            raise Error('Cannot do rejection sampling on prior')

        self._check_context(context)
        symbol = self._root.sample(context, rejection_sampling)
        num_rejections = 0
        while rejection_sampling and symbol is None:
            num_rejections += 1
            if num_rejections >= MAX_SAMPLE_REJECTIONS:
                symbol = self._root.estimator.sample(rejection_sampling=True)
                # There should be *some* symbol in the root estimator.
                assert symbol is not None
            else:
                symbol = self._root.sample(context, rejection_sampling=True)

        return symbol

class ContextualSequenceModel(object):
    """A sequence model.
    This class maintains a context vector, i.e. a list of the most recent
    observations. It predicts by querying a contextual model (e.g. CTS) with
    this context vector.
    """

    def __init__(self, model=None, context_length=None, start_symbol=0):
        """Constructor.
        Args:
            model: The model to be used for prediction. If this is none but
                context_length is not, defaults to CTS(context_length).
            context_length: If model == None, the length of context for the
                underlying CTS model.
            start_symbol: The symbol with which to pad the first context
                vectors.
        """
        if model is None:
            if context_length is None:
                raise ValueError('Must specify model or model parameters')
            else:
                self.model = CTS(context_length)
        else:
            self.model = model

        self.context = [start_symbol] * self.model.context_length

    def observe(self, symbol):
        """Updates the current context without updating the model.
        The new context is generated by discarding the oldest symbol and
        inserting the new symbol in the rightmost position of the context
        vector.
        Args:
            symbol: Observed symbol.
        """
        self.context.append(symbol)
        self.context = self.context[1:]

    def update(self, symbol):
        """Updates the model with the new symbol.
        The current context is subsequently updated, as per ``observe()``.
        Args:
            symbol: Observed symbol.
        Returns:
            The log probability of the observed symbol.
        """
        log_prob = self.model.update(self.context, symbol)
        self.observe(symbol)
        return log_prob

    def log_prob(self, symbol):
        """Computes the log probability of the given symbol.
        Neither model nor context is subsequently updated.
        Args:
            symbol: Observed symbol.
        Returns:
            The log probability of the observed symbol.
        """
        return self.model.log_prob(self.context, symbol)

    def sample(self, rejection_sampling=True):
        """Samples a symbol according to the current context.
        Neither model nor context are updated.
        This may be used in combination with ``observe()`` to generate sample
        sequences without updating the model (though a die-hard Bayesian would
        use ``update()`` instead!).
        Args:
            rejection_sampling: If set to True, symbols are not drawn from
            the prior: only observed symbols are output. Setting to False
            requires specifying the model's alphabet (see ``CTS.__init__``
            above).
        Returns:
            The sampled symbol.
        """
        return self.model.sample(self.context, rejection_sampling)

__all__ = ["CTS", "ContextualSequenceModel"]