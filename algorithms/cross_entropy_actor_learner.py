# -*- coding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from actor_learner import ActorLearner, ONE_LIFE_GAMES
from utils.replay_memory import ReplayMemory


logger = utils.logger.getLogger('cross_entropy_actor_learner')

class CEMLearner(ActorLearner):
	def __init__(self, args):
		pass