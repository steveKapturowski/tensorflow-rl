# -*- coding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from actor_learner import ActorLearner, ONE_LIFE_GAMES
from networks.policy_v_network import PolicyVNetwork
from utils.replay_memory import ReplayMemory


logger = utils.logger.getLogger('trpo_actor_learner')

class TRPOLearner(ActorLearner):
	'''
	Implementation of Trust Region Policy Optimization + Generalized Advantage Estimation 
	as described in https://arxiv.org/pdf/1506.02438.pdf
	'''
	def __init__(self, args):
		pass

