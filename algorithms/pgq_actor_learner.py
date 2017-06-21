# -*- coding: utf-8 -*-
import time
import ctypes
import numpy as np
import utils.logger
import tensorflow as tf
from utils.hogupdatemv import copy
from actor_learner import ONE_LIFE_GAMES
from utils.decorators import Experimental
from utils.replay_memory import ReplayMemory
from networks.policy_v_network import PolicyValueNetwork
from policy_based_actor_learner import BaseA3CLearner


logger = utils.logger.getLogger('pgq_actor_learner')


class BasePGQLearner(BaseA3CLearner):
    def __init__(self, args):

        super(BasePGQLearner, self).__init__(args)

        self.q_update_counter = 0
        self.replay_size = args.replay_size
        self.pgq_fraction = args.pgq_fraction
        self.batch_update_size = args.batch_update_size
        scope_name = 'local_learning_{}'.format(self.actor_id)
        conf_learning = {'name': scope_name,
                         'input_shape': self.input_shape,
                         'num_act': self.num_actions,
                         'args': args}

        with tf.device('/cpu:0'):
            self.local_network = PolicyValueNetwork(conf_learning)
        with tf.device('/gpu:0'), tf.variable_scope('', reuse=True):
            self.batch_network = PolicyValueNetwork(conf_learning)
            self._build_q_ops()

        self.reset_hidden_state()
        self.replay_memory = ReplayMemory(
            self.replay_size,
            self.local_network.get_input_shape(),
            self.num_actions)
            
        if self.is_master():
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)


    def _build_optimizer(self):
        super(BasePGQLearner, self)._build_optimizer()

        with tf.variable_scope('optimizer', reuse=True):
            q_gradients = [
                e[0] for e in self.optimizer.compute_gradients(
                self.local_network.loss, self.V_params)]
            q_gradients = self.batch_network._clip_grads(q_gradients)

            self.apply_q_gradients = self.optimizer.apply_gradients(
                zip(q_gradients, self.V_params), global_step=self.global_step)


    def _build_q_ops(self):
        # pgq specific initialization
        self.pgq_fraction = self.pgq_fraction
        self.batch_size = self.batch_update_size
        self.q_tilde = self.batch_network.beta * (
            self.batch_network.log_output_layer_pi
            + tf.expand_dims(self.batch_network.output_layer_entropy, 1)
        ) + self.batch_network.output_layer_v

        self.Qi, self.Qi_plus_1 = tf.split(axis=0, num_or_size_splits=2, value=self.q_tilde)
        self.V, _ = tf.split(axis=0, num_or_size_splits=2, value=self.batch_network.output_layer_v)
        self.log_pi, _ = tf.split(axis=0, num_or_size_splits=2, value=tf.expand_dims(self.batch_network.log_output_selected_action, 1))
        self.R = tf.placeholder('float32', [None], name='1-step_reward')

        self.terminal_indicator = tf.placeholder(tf.float32, [None], name='terminal_indicator')
        self.max_TQ = self.gamma*tf.reduce_max(self.Qi_plus_1, 1) * (1 - self.terminal_indicator)
        self.Q_a = tf.reduce_sum(self.Qi * tf.split(axis=0, num_or_size_splits=2, value=self.batch_network.selected_action_ph)[0], 1)

        self.q_objective = - self.pgq_fraction * tf.reduce_mean(tf.stop_gradient(self.R + self.max_TQ - self.Q_a) * (0.5 * self.V[:, 0] + self.log_pi[:, 0]))
        self.V_params = self.local_network.params


    def batch_q_update(self):
        if len(self.replay_memory) < self.replay_memory.maxlen//10:
            return

        s_i, a_i, r_i, s_f, is_terminal = self.replay_memory.sample_batch(self.batch_size)

        self.session.run(
            self.apply_q_gradients,
            feed_dict={
                self.R: r_i,
                self.batch_network.selected_action_ph: np.vstack([a_i, a_i]),
                self.batch_network.input_ph: np.vstack([s_i, s_f]),
                self.terminal_indicator: is_terminal.astype(np.int),
            }
        )


class PGQLearner(BasePGQLearner):
    def choose_next_action(self, state):
        network_output_v, network_output_pi = self.session.run(
                [self.local_network.output_layer_v,
                 self.local_network.output_layer_pi], 
                feed_dict={self.local_network.input_ph: [state]})

        network_output_pi = network_output_pi.reshape(-1)
        network_output_v = np.asscalar(network_output_v)

        action_index = self.sample_policy_action(network_output_pi)
        new_action = np.zeros([self.num_actions])
        new_action[action_index] = 1

        return new_action, network_output_v, network_output_pi


    def apply_update(self, states, actions, targets, advantages):
        feed_dict={
            self.local_network.input_ph: states,
            self.local_network.selected_action_ph: actions,
            self.local_network.critic_target_ph: targets,
            self.local_network.adv_actor_ph: advantages,
        }
        grads, entropy = self.session.run(
            [self.local_network.get_gradients, self.local_network.entropy],
            feed_dict=feed_dict)
        self.apply_gradients_to_shared_memory_vars(grads)

        self.q_update_counter += 1
        if self.q_update_counter % self.q_update_interval == 0:
            self.batch_q_update()

        return entropy

