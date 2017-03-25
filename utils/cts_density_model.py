# -*- encoding: utf-8 -*-
import numpy as np
from cts import CTS
from skimage.transform import resize


class CTSDensityModel(object):
	'''
	Implementation of CTS Density Model described in the paper
	"Unifying Count-Based Exploration and Intrinsic Motivation" (https//arxiv.org/abs/1606.01868)
	'''
	def __init__(self, height=21, width=21, beta=0.05):
		self.beta = beta
		self.factors = np.array([[CTS(4) for _ in range(width)] for _ in range(height)])


	def update(self, obs):
		obs = resize(obs, self.factors.shape)

		context = [0, 0, 0, 0]
		log_prob = 0.0
		log_recoding_prob = 0.0
		for i in range(self.factors.shape[0]):
			for j in range(self.factors.shape[1]):
				context[3] = obs[i, j-1] if j > 0 else 0
				context[2] = obs[i-1, j] if i > 0 else 0
				context[1] = obs[i-1, j-1] if i > 0 and j > 0 else 0
				context[0] = obs[i-1, j+1] if i > 0 and j < self.factors.shape[1]-1 else 0

				log_prob += self.factors[i, j].update(context, obs[i, j])
				log_recoding_prob += self.factors[i, j].log_prob(context, obs[i, j])

		return self.exploration_bonus(log_prob, log_recoding_prob)


	def exploration_bonus(self, log_prob, log_recoding_prob):
		prob = np.exp(log_prob)
		recoding_prob = np.exp(log_recoding_prob)

		pseudocount = prob * (1 - recoding_prob) / np.maximum(recoding_prob - prob, 1e-10)
		return self.beta / np.sqrt(pseudocount + .01)

