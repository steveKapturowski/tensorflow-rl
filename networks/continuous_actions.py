# -*- encoding: utf-8 -*-
import layers
import numpy as np
import tensorflow as tf
from q_network import QNetwork
from policy_v_network import PolicyValueNetwork


class ContinuousPolicyValueNetwork(PolicyValueNetwork):
    '''
    Shared policy-value network with polciy head parametrizing
    multivariate normal with diagonal covariance
    '''
    def __init__(self, conf, **kwargs):
        self.action_space = conf['args'].action_space
        super(ContinuousPolicyValueNetwork, self).__init__(conf, **kwargs)

    def _build_policy_head(self):
        self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')       
        self.w_mu, self.b_mu, self.mu = layers.fc(
            'mean', self.ox, self.num_actions)
        # self.w_sigma, self.b_sigma, self.sigma = layers.fc(
        #     'std', self.ox, self.num_actions, activation='softplus')
        self.sigma = self.mu

        self.N = tf.contrib.distributions.Normal(mu=self.mu, sigma=.2)
        # self.log_output_selected_action = tf.reduce_sum(self.N.log_pdf(self.selected_action_ph))
        self.log_output_selected_action = tf.pow(self.selected_action_ph - self.mu, 2)


        self.output_layer_entropy = .5*(tf.reduce_sum(2*self.N.entropy()-1, axis=1)+1)
        self.entropy = tf.reduce_mean(self.output_layer_entropy)

        self.actor_objective = -tf.reduce_mean(
            self.log_output_selected_action * self.adv_actor_ph
            # + self.beta * self.output_layer_entropy
        )
        self.sample_action = self.N.sample()

        return self.actor_objective

    def get_action(self, session, state):
        action = session.run([
            self.sample_action,
            self.mu,
            self.sigma
        ], feed_dict={self.input_ph: [state]})

        return action[0], (mu[0], sigma[0])

    def get_action_and_value(self, session, state):
        action, v, mu, sigma = session.run([
            self.sample_action,
            self.output_layer_v,
            self.mu,
            self.sigma
        ], feed_dict={self.input_ph: [state]})

        return action[0], v[0, 0], (mu[0], sigma[0])


class ContinuousPolicyNetwork(ContinuousPolicyValueNetwork):
    def __init__(self, conf):
        super(ContinuousPolicyNetwork, self).__init__(conf, use_value_head=False)


class NAFNetwork(QNetwork):
    '''
    Implements Normalized Advantage Functions from "Continuous Deep Q-Learning
    with Model-based Acceleration" (https://arxiv.org/pdf/1603.00748.pdf)
    '''
    def _build_q_head(self):
        self.w_value, self.b_value, self.value = layers.fc('fc_value', self.ox, 1, activation='linear')
        self.w_L, self.b_L, self.L_full = layers.fc('L_full', self.ox, self.num_actions, activation='linear')
        self.w_mu, self.b_mu, self.mu = layers.fc('mu', self.ox, self.num_actions, activation='linear')

        #elements above the main diagonal in L_full are unused
        D = tf.matrix_band_part(tf.exp(self.L_full) - L_full, 0, 0)
        L = tf.matrix_band_part(L_full, -1, 0) + D

        LT_u_minus_mu = tf.einsum('ikj,ik', L, self.selected_action_ph  - self.mu)
        self.advantage = tf.einsum('ijk,ikj->i', LT_u_minus_mu, LT_u_minus_mu)

        q_selected_action = self.value + self.advantage
        diff = tf.subtract(self.target_ph, q_selected_action)
        return self._huber_loss(diff)



