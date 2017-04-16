# -*- encoding: utf-8 -*-
import layers
import numpy as np
import tensorflow as tf
from q_network import QNetwork
from utils.distributions import DiagNormal
from policy_v_network import PolicyValueNetwork


class ContinuousPolicyValueNetwork(PolicyValueNetwork):
    '''
    Shared policy-value network with polciy head parametrizing
    multivariate normal with diagonal covariance
    '''
    def __init__(self, conf, **kwargs):
        self.action_space = conf['args'].action_space
        self.use_state_dependent_std = True
        super(ContinuousPolicyValueNetwork, self).__init__(conf, **kwargs)

    def _build_policy_head(self, input_state):
        self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')       
        self.w_mu, self.b_mu, self.mu = layers.fc(
            'mean', input_state, self.num_actions, activation='linear')
        self.sigma = self._build_sigma(input_state)

        self.N = DiagNormal(self.mu, self.sigma)
        self.log_output_selected_action = self.N.log_likelihood(self.selected_action_ph)
        self.log_output_selected_action = tf.expand_dims(self.log_output_selected_action, 1)
        
        self.output_layer_entropy = self.N.entropy()
        self.entropy = tf.reduce_sum(self.output_layer_entropy)

        self.actor_objective = -tf.reduce_sum(
            self.log_output_selected_action * self.adv_actor_ph
            + self.beta * self.output_layer_entropy
        )
        self.sample_action = self.N.sample()
        # self.sample_action = tf.Print(self.sample_action, [self.sample_action], 'Action: ')

        return self.actor_objective

    def _build_sigma(self, input_state):
        if self.use_state_dependent_std:
            self.w_sigma2, self.b_sigma2, self.sigma2 = layers.fc(
                'std2', input_state, self.num_actions, activation='softplus')
            return tf.sqrt(self.sigma2 + 1e-8)
        else:
            self.log_sigma = tf.get_variable('log_sigma', self.mu.get_shape().as_list()[1],
                dtype=tf.float32, initializer=tf.random_uniform_initializer(-4, -2))
            return tf.expand_dims(tf.exp(self.log_sigma), 0)

    def get_action(self, session, state, lstm_state=None):
        feed_dict = {self.input_ph: [state]}
        if lstm_state is not None:
            feed_dict[self.step_size] = [1]
            feed_dict[self.initial_lstm_state] = lstm_state

            action, lstm_state, mu, sigma = session.run([
                self.sample_action,
                self.lstm_state,
                self.mu,
                self.sigma], feed_dict=feed_dict)

            return action[0], (mu[0], sigma[0]), lstm_state
        else:
            action, mu, sigma = session.run([
                self.sample_action,
                self.mu,
                self.sigma], feed_dict=feed_dict)

            return action[0], (mu[0], sigma[0])

    def get_action_and_value(self, session, state, lstm_state=None):
        feed_dict = {self.input_ph: [state]}
        if lstm_state is not None:
            feed_dict[self.step_size] = [1]
            feed_dict[self.initial_lstm_state] = lstm_state

            action, v, lstm_state, mu, sigma = session.run([
                self.sample_action,
                self.output_layer_v,
                self.lstm_state,
                self.mu,
                self.sigma], feed_dict=feed_dict)

            return action[0], v[0, 0], (mu[0], sigma[0]), lstm_state
        else:
            action, v, mu, sigma = session.run([
                self.sample_action,
                self.output_layer_v,
                self.mu,
                self.sigma], feed_dict=feed_dict)

            return action[0], v[0, 0], (mu[0], sigma[0])


class ContinuousPolicyNetwork(ContinuousPolicyValueNetwork):
    def __init__(self, conf):
        super(ContinuousPolicyNetwork, self).__init__(conf, use_value_head=False)


class NAFNetwork(QNetwork):
    '''
    Implements Normalized Advantage Functions from "Continuous Deep Q-Learning
    with Model-based Acceleration" (https://arxiv.org/pdf/1603.00748.pdf)
    '''
    def _build_q_head(self, input_state):
        self.w_value, self.b_value, self.value = layers.fc('fc_value', input_state, 1, activation='linear')
        self.w_L, self.b_L, self.L_full = layers.fc('L_full', input_state, self.num_actions, activation='linear')
        self.w_mu, self.b_mu, self.mu = layers.fc('mu', input_state, self.num_actions, activation='linear')

        #elements above the main diagonal in L_full are unused
        D = tf.matrix_band_part(tf.exp(self.L_full) - L_full, 0, 0)
        L = tf.matrix_band_part(L_full, -1, 0) + D

        LT_u_minus_mu = tf.einsum('ikj,ik', L, self.selected_action_ph  - self.mu)
        self.advantage = tf.einsum('ijk,ikj->i', LT_u_minus_mu, LT_u_minus_mu)

        q_selected_action = self.value + self.advantage
        diff = tf.subtract(self.target_ph, q_selected_action)
        return self._huber_loss(diff)



