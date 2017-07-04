# -*- encoding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from collections import deque
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
        self.beta = args.entropy_regularisation_strength
        self.q_target_update_steps = args.q_target_update_steps


    def sample_policy_action(self, probs, temperature=0.5):
        probs = probs - np.finfo(np.float32).epsneg
    
        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index


    def bootstrap_value(self, state, episode_over):
        if episode_over:
            R = 0
        else:
            R = self.session.run(
                self.local_network.output_layer_v,
                feed_dict={self.local_network.input_ph:[state]})[0][0]

        return R


    def compute_gae(self, rewards, values, next_val):
        values = values + [next_val]
        size = len(rewards)
        adv_batch = list()
        td_i = 0.0

        for i in reversed(xrange(size)):
            td_i = rewards[i] + self.gamma*values[i+1] - values[i] + self.td_lambda*self.gamma*td_i 
            adv_batch.append(td_i)

        adv_batch.reverse()
        return adv_batch


    def set_local_lstm_state(self):
        pass


    def apply_update(self, states, actions, targets, advantages):
        feed_dict={
            self.local_network.input_ph: states,
            self.local_network.selected_action_ph: actions,
            self.local_network.critic_target_ph: targets,
            self.local_network.adv_actor_ph: advantages,
        }
        entropy, _ = self.session.run(
            [self.local_network.entropy, self.apply_gradients],
            feed_dict=feed_dict)

        return entropy


    def train(self):
        """ Main actor learner loop for advantage actor critic learning. """
        last_global_step = self.global_step.eval(self.session)
        logger.debug("Actor {} resuming at Step {}".format(self.task_index, last_global_step))
        
        episode_rewards = deque(maxlen=1000)
        while not self.supervisor.should_stop():
            s = self.emulator.get_initial_state()
            self.reset_hidden_state()
            self.local_episode += 1
            episode_over = False
            total_episode_reward = 0.0
            episode_start_time = time.time()
            episode_start_step = self.local_step
            
            while not episode_over:
                rewards = list()
                states  = list()
                actions = list()
                values  = list()
                local_step_start = self.local_step
                self.session.run(self.sync_local_network)
                self.set_local_lstm_state()

                while self.local_step - local_step_start < self.max_local_steps and not episode_over:
                    # Choose next action and execute it
                    a, readout_v_t, readout_pi_t = self.choose_next_action(s)
                    if self.is_master() and (self.local_step % 400 == 0):
                        logger.debug("π_a={:.4f} / V={:.4f}".format(readout_pi_t[a.argmax()], readout_v_t))
                    
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

                global_step = self.session.run(self.global_step_increment)
                
                next_val = self.bootstrap_value(new_s, episode_over)
                advantages = self.compute_gae(rewards, values, next_val)
                targets = self.compute_targets(rewards, next_val)
                # Compute gradients on the local policy/V network and apply them to shared memory 
                entropy = self.apply_update(states, actions, targets, advantages)

            episode_rewards.append(total_episode_reward)
            elapsed_time = time.time() - episode_start_time
            steps_per_sec = (self.local_step - episode_start_step) * self.num_actor_learners / elapsed_time
            perf = "{:.0f}".format(steps_per_sec)
            logger.info("T{} / EPISODE {} / STEP {}k / MEAN REWARD {:.1f} / {} STEPS/s".format(
                self.actor_id,
                self.local_episode,
                global_step/1000,
                np.array(episode_rewards).mean(),
                perf))

            self.log_summary(total_episode_reward, np.array(values).mean(), entropy)
            last_global_step = global_step


class A3CLearner(BaseA3CLearner):
    def __init__(self, args):
        super(A3CLearner, self).__init__(args)

        conf_local = {'name': 'local_network_{}'.format(self.actor_id),
                      'input_shape': self.input_shape,
                      'num_act': self.num_actions,
                      'args': args}
        conf_global = conf_local.copy()
        conf_global['name'] = 'global_network'
        self.local_network = args.network(conf_local)
        self.global_network = args.network(conf_global)

        self.sync_local_network = tf.group(*[
            l.assign(g) for l, g in zip(self.local_network.params, self.global_network.params)
        ])
        self.reset_hidden_state()

        if self.is_master():
            var_list = self.local_network.params + self.global_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3,
                                        keep_checkpoint_every_n_hours=2)


    def choose_next_action(self, state):
        return self.local_network.get_action_and_value(self.session, state)


class A3CLSTMLearner(BaseA3CLearner):
    def __init__(self, args):
        super(A3CLSTMLearner, self).__init__(args)

        conf_local = {'name': 'local_network_{}'.format(self.actor_id),
                      'input_shape': self.input_shape,
                      'num_act': self.num_actions,
                      'args': args}
        conf_global = conf_local.copy()
        conf_global['name'] = 'global_network'
        self.local_network = args.network(conf_local)
        self.global_network = args.network(conf_global)

        self.sync_local_network = tf.group(*[
            l.assign(g) for l, g in zip(self.local_network.params, self.global_network.params)
        ])
        self.reset_hidden_state()

        if self.is_master():
            var_list = self.local_network.params + self.global_network.params
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
        entropy, _ = self.session.run(
            [self.local_network.entropy, self.apply_gradients],
            feed_dict=feed_dict)

        return entropy



