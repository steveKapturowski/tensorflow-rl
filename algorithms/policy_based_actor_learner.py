# -*- encoding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from gym.spaces import Discrete
from utils import checkpoint_utils
from utils.decorators import only_on_train
from actor_learner import ActorLearner, ONE_LIFE_GAMES
from networks.policy_v_network import PolicyValueNetwork


logger = utils.logger.getLogger('policy_based_actor_learner')


class BaseA3CLearner(ActorLearner):
    def __init__(self, args):
        super(BaseA3CLearner, self).__init__(args)

        self.td_lambda = args.td_lambda
        self.action_space = args.action_space
        # self.learning_vars = args.learning_vars
        self.beta = args.entropy_regularisation_strength
        self.q_target_update_steps = args.q_target_update_steps


    def sample_policy_action(self, probs, temperature=0.5):
        probs = probs - np.finfo(np.float32).epsneg
    
        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index


    def compute_gae(self, rewards, values, next_val):
        values = values + [next_val]
        size = len(rewards)
        adv_batch = list()
        td_i = 0.0

        for i in range(size):
            j = size - 1 - i
            td_i = self.td_lambda*self.gamma*td_i + rewards[j] + self.gamma*values[j+1] - values[j]
            adv_batch.insert(0, td_i)

        return adv_batch


    def bootstrap_value(self, state, episode_over):
        if episode_over:
            R = 0
        else:
            R = self.session.run(
                self.local_network.output_layer_v,
                feed_dict={self.local_network.input_ph:[state]})[0][0]

        return R


    def compute_targets(self, rewards, values, state, episode_over):   
        R = self.bootstrap_value(state, episode_over)
        adv_batch = list()
        y_batch = list()
        for i in reversed(xrange(len(rewards))):
            idx = len(rewards)-i-1
            R = rewards[idx] + self.gamma * R
            y_batch.append(R)
            adv_batch.append(R - values[idx])

        y_batch.reverse()
        adv_batch.reverse()
        return y_batch, adv_batch


    def set_local_lstm_state(self):
        pass


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
        return entropy


    def train(self):
        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.task_index, 
            self.global_step.eval(self.session)))
        
        while not self.supervisor.should_stop():
        # while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            # self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            # self.save_vars()

            s = self.emulator.get_initial_state()
            self.reset_hidden_state()
            self.local_episode += 1
            episode_over = False
            total_episode_reward = 0.0
            episode_start_step = self.local_step
            
            while not episode_over:
                rewards = list()
                states  = list()
                actions = list()
                values  = list()
                local_step_start = self.local_step
                self.set_local_lstm_state()

                while self.local_step - local_step_start < self.max_local_steps and not episode_over:
                    # Choose next action and execute it
                    a, readout_v_t, readout_pi_t = self.choose_next_action(s)
                    if self.is_master() and (self.local_step % 100 == 0):
                        logger.debug("pi={}, V={}".format(readout_pi_t, readout_v_t))
                    
                    new_s, reward, episode_over = self.emulator.next(a)
                    total_episode_reward += reward
                    # Rescale or clip immediate reward
                    reward = self.rescale_reward(reward)
                
                    rewards.append(reward)
                    states.append(s)
                    actions.append(a)
                    values.append(readout_v_t)
                
                    s = new_s
                    self.local_step += 1
                    self.session.run(self.increment_step)
                
                targets, advantages = self.compute_targets(rewards, values, new_s, episode_over)
                entropy = self.apply_update(states, actions, targets, advantages)

            global_step = self.global_step.eval(self.session)
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_step / elapsed_time
            perf = "{:.0f}".format(steps_per_sec)
            logger.info("T{} / EPISODE {} / STEP {}k / REWARD {} / {} STEPS/s".format(
                self.task_index,
                self.local_episode,
                global_step/1000.,
                total_episode_reward,
                perf))

            self.log_summary(total_episode_reward, np.array(values).mean(), entropy)


class A3CLearner(BaseA3CLearner):
    def __init__(self, args):
        super(A3CLearner, self).__init__(args)

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'input_shape': self.input_shape,
                         'num_act': self.num_actions,
                         'args': args}

        self.local_network = args.network(conf_learning)
        self.reset_hidden_state()

        if self.is_master():
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3,
                                        keep_checkpoint_every_n_hours=2)


    def choose_next_action(self, state):
        return self.local_network.get_action_and_value(self.session, state)


class A3CLSTMLearner(BaseA3CLearner):
    def __init__(self, args):
        super(A3CLSTMLearner, self).__init__(args)

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'input_shape': self.input_shape,
                         'num_act': self.num_actions,
                         'args': args}

        self.local_network = args.network(conf_learning)
        self.reset_hidden_state()

        if self.is_master():
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3,
                                        keep_checkpoint_every_n_hours=2)


    def reset_hidden_state(self):
        self.lstm_state_out = np.zeros([1, 2*self.local_network.hidden_state_size])
        # self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
        #                                                     np.zeros([1, 256]))


    def set_local_lstm_state(self):
        self.local_lstm_state = np.copy(self.lstm_state_out)


    def choose_next_action(self, state):
        action, v, dist, self.lstm_state_out = self.local_network.get_action_and_value(
            self.session, state, lstm_state=self.lstm_state_out)
        return action, v, dist


    def bootstrap_value(self, state, episode_over):
        if episode_over:
            R = 0
        else:
            R = self.session.run(
                self.local_network.output_layer_v,
                feed_dict={
                    self.local_network.input_ph:[state],
                    self.local_network.step_size: [1],
                    self.local_network.initial_lstm_state: self.lstm_state_out,
                }
            )[0][0]

        return R


    def apply_update(self, states, actions, targets, advantages):
        feed_dict={
            self.local_network.input_ph: states,
            self.local_network.selected_action_ph: actions,
            self.local_network.critic_target_ph: targets,
            self.local_network.adv_actor_ph: advantages,
            self.local_network.step_size : [len(states)],
            self.local_network.initial_lstm_state: self.local_lstm_state,
        }
        grads, entropy = self.session.run(
            [self.local_network.get_gradients, self.local_network.entropy],
            feed_dict=feed_dict)

        self.apply_gradients_to_shared_memory_vars(grads)
        return entropy



