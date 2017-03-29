# -*- coding: utf-8 -*-
import logger


logger = logger.getLogger('intrinsic_motivation_actor_learner')

def only_on_train(return_val=None):
	def _only_on_train(func):
		def wrapper(*args, **kwargs):
			if args[0].is_train:
				return func(*args, **kwargs)
			else:
				return return_val

		return wrapper
	return _only_on_train


def Experimental(_cls):
	old_init = getattr(_cls, '__init__')
	def wrapped_init(self, *args, **kwargs):
		logger.warning('Using experimental class \'{0}\' -- See docstring for more details:\n{1}'.format(
			_cls.__name__, _cls.__doc__))
		old_init(self, *args, **kwargs)

	setattr(_cls, '__init__', wrapped_init)
	return _cls





