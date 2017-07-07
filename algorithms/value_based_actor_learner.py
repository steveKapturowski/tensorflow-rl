# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import collections
import ctypes
import utils
import time
import sys

from actor_learner import ActorLearner
from networks.q_network import QNetwork
from networks.dueling_network import DuelingNetwork
from utils.decorators import only_on_train
from utils.hogupdatemv import copy


logger = utils.logger.getLogger('value_based_actor_learner')


class ValueBasedLearner(ActorLearner):

    def __init__(self, args, network_type=QNetwork):
        
        super(ValueBasedLearner, self).__init__(args)
        
        self.q_target_update_steps = args.q_target_update_steps
        self.scores = collections.deque(maxlen=1000)
        
        conf_local = {'name': "local_network_{}".format(self.actor_id),
                      'input_shape': self.input_shape,
                      'num_act': self.num_actions,
                      'args': args}
        conf_target = conf_local.copy()
        conf_target['name'] = 'target_network_{}'.format(self.actor_id)
        conf_global = conf_local.copy()
        conf_global['name'] = 'global_network'
        
        self.global_network = network_type(conf_global)
        self.local_network = network_type(conf_local)
        self.target_network = network_type(conf_target)

        if self.is_master():
            var_list = self.local_network.params + self.target_network.params            
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)

        # build assign ops for target network
        self.update_target_network = tf.group(
            *[t.assign(v) for t, v in zip(
                self.target_network.params,
                self.local_network.params)])
        self.sync_local_network = tf.group(
            *[l.assign(g) for l, g in zip(
                self.local_network.params,
                self.global_network.params)])

        # Exploration epsilons 
        self.initial_epsilon = 1.0
        self.final_epsilon = self.generate_final_epsilon()
        self.epsilon = self.initial_epsilon if self.is_train else args.final_epsilon
        self.epsilon_annealing_steps = args.epsilon_annealing_steps
        self.exploration_strategy = args.exploration_strategy
        self.bolzmann_temperature = args.bolzmann_temperature


    def generate_final_epsilon(self):
        values = [.01, .05, .1, .2]
        return values[self.actor_id % 4]


    def reduce_thread_epsilon(self):
        """ Linear annealing """
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_annealing_steps
        

    def _get_summary_vars(self):
        episode_reward = tf.Variable(0., name='episode_reward')
        s1 = tf.summary.scalar('Episode_Reward_{}'.format(self.actor_id), episode_reward)

        episode_avg_max_q = tf.Variable(0., name='episode_avg_max_q')
        s2 = tf.summary.scalar('Max_Q_Value_{}'.format(self.actor_id), episode_avg_max_q)
        
        logged_epsilon = tf.Variable(0., name='epsilon_'.format(self.actor_id))
        s3 = tf.summary.scalar('Epsilon_{}'.format(self.actor_id), logged_epsilon)
        
        return [episode_reward, episode_avg_max_q, logged_epsilon]


    def epsilon_greedy(self, q_values):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(q_values)


    def boltzmann_exploration(self, q_values):
        exp_minus_max = np.exp(q_values - q_values.max())
        probs = exp_minus_max / exp_minus_max.sum()
        
        return np.random.choice(self.num_actions, p=probs)


    def choose_next_action(self, state):
        """ Epsilon greedy """
        new_action = np.zeros([self.num_actions])
        
        q_values = self.session.run(
            self.local_network.output_layer,
            feed_dict={self.local_network.input_ph: [state]})[0]
            
        if self.exploration_strategy == 'epsilon-greedy':
            action_index = self.epsilon_greedy(q_values)
        else:
            action_index = self.boltzmann_exploration(q_values)
                
        new_action[action_index] = 1
        self.reduce_thread_epsilon()
        
        return new_action, q_values


    def bootstrap_value(self, state, episode_over):
        if episode_over:
            R = 0
        else:
            q_target_values = self.session.run(
                self.target_network.output_layer,
                feed_dict={self.target_network.input_ph: [state]})
            R = np.max(q_target_values)

        return R


    def apply_update(self, states, actions, targets):
        feed_dict={
            self.local_network.input_ph: states,
            self.local_network.target_ph: targets,
            self.local_network.selected_action_ph: actions,
        }
        grads = self.session.run(
            self.local_network.get_gradients,
            feed_dict=feed_dict)

        self.apply_gradients_to_shared_memory_vars(grads)


    def update_target(self):
        self.session.run(self.update_target_network)


    def prepare_state(self, state, total_episode_reward, steps_at_last_reward,
                      ep_t, episode_ave_max_q, episode_over):
        # Start a new game on reaching terminal state
        if episode_over:
            T = self.global_step.eval(self.session)
            t = self.local_step
            e_prog = float(t)/self.epsilon_annealing_steps
            episode_ave_max_q = episode_ave_max_q/float(ep_t)
            s1 = "Q_MAX {0:.4f}".format(episode_ave_max_q)
            s2 = "EPS {0:.4f}".format(self.epsilon)

            self.scores.append(total_episode_reward)
            logger.info('T{0} / STEP {1} / REWARD {2} / {3} / {4}'.format(
                self.actor_id, T, total_episode_reward, s1, s2))
            logger.info('ID: {0} -- RUNNING AVG: {1:.0f} ± {2:.0f} -- BEST: {3:.0f}'.format(
                self.actor_id,
                np.array(self.scores).mean(),
                2*np.array(self.scores).std(),
                max(self.scores),
            ))
            self.log_summary(
                total_episode_reward,
                episode_ave_max_q,
                self.epsilon)
                
            state = self.emulator.get_initial_state()
            total_episode_reward = 0
            episode_ave_max_q = 0
            episode_over = False
            ep_t = 0

        return state, total_episode_reward, steps_at_last_reward, ep_t, episode_ave_max_q, episode_over
        
        
class NStepQLearner(ValueBasedLearner):

    def train(self):
        """ Main actor learner loop for n-step Q learning. """
        logger.debug("Actor {} resuming at Step {}, {}".format(self.actor_id, 
            self.global_step.eval(self.session), time.ctime()))

        s = self.emulator.get_initial_state()        
        steps_at_last_reward = self.local_step
        exec_update_target = False
        total_episode_reward = 0
        episode_ave_max_q = 0
        episode_over = False
        qmax_down = 0
        qmax_up = 0
        prev_qmax = -10*6
        low_qmax = 0
        ep_t = 0
        
        while not self.supervisor.should_stop():
            # Sync local learning net with shared mem
            self.session.run(self.sync_local_network)

            rewards = list()
            states =  list()
            actions = list()
            local_step_start = self.local_step
            
            while not (episode_over 
                or (self.local_step - local_step_start == self.max_local_steps)):
                
                # Choose next action and execute it
                a, readout_t = self.choose_next_action(s)

                
                new_s, reward, episode_over = self.emulator.next(a)
                if reward != 0.0:
                    steps_at_last_reward = self.local_step

                total_episode_reward += reward
                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward)

                ep_t += 1
                
                rewards.append(reward)
                states.append(s)
                actions.append(a)
                
                s = new_s
                self.local_step += 1
                episode_ave_max_q += np.max(readout_t)
                
                global_step = self.session.run(self.global_step_increment)
                if global_step % self.q_target_update_steps == 0:
                    self.update_target()

            R = self.bootstrap_value(s, episode_over)
            targets = self.compute_targets(rewards, R)
            # Compute gradients on the local Q network 
            self.apply_update(states, actions, targets)

            s, total_episode_reward, steps_at_last_reward, ep_t, episode_ave_max_q, episode_over = \
                self.prepare_state(s, total_episode_reward, steps_at_last_reward, ep_t, episode_ave_max_q, episode_over)


class DuelingLearner(NStepQLearner):
    def __init__(self, args):
        super(DuelingLearner, self).__init__(args, network_type=DuelingNetwork)


class OneStepSARSALearner(ValueBasedLearner):

    def generate_final_epsilon(self):
        return 0.1

    def train(self):
        """ Main actor learner loop for 1-step SARSA learning. """
        logger.debug("Actor {} resuming at Step {}, {}".format(self.actor_id, 
            self.global_step.eval(self.session), time.ctime()))
        
        s = self.emulator.get_initial_state()

        states =  list()
        actions = list()
        targets = list()
        
        steps_at_last_reward = self.local_step
        exec_update_target = False
        total_episode_reward = 0
        episode_ave_max_q = 0
        episode_over = False
        qmax_down = 0
        qmax_up = 0
        prev_qmax = -10*6
        low_qmax = 0
        ep_t = 0

        # Choose initial action
        a, readout_t = self.choose_next_action(s)
        
        while not self.supervisor.should_stop():
            s_prime, reward, episode_over = self.emulator.next(a)
            if reward != 0.0:
                steps_at_last_reward = self.local_step
            
            ep_t += 1
            episode_ave_max_q += np.max(readout_t)

            actions.append(a)
            states.append(s)
            s = s_prime

            total_episode_reward += reward
            reward = self.rescale_reward(reward)
            if episode_over:
                y = reward
            else:
                # Choose action that we will execute in the next step 
                a_prime, readout_t = self.choose_next_action(s_prime)
                q_prime = readout_t[a_prime.argmax()]
                # Q_target in the new state for the next step action 
                q_prime = self.session.run(
                    self.target_network.output_layer, 
                    feed_dict={self.target_network.input_ph: [s_prime]}
                )[0][a_prime.argmax()]

                y = reward + self.gamma * q_prime
                a = a_prime

            targets.append(y)
            
            self.local_step += 1
            global_step = self.session.run(self.global_step_increment)
            
            # Compute grads and asynchronously apply them to shared memory
            if ((self.local_step % self.grads_update_steps == 0) 
                or episode_over):

                # Compute gradients on the local Q network     
                self.apply_update(states, actions, targets)
                self.session.run(self.sync_local_network)

                states =  list()
                actions = list()
                targets = list()

            # Copy shared learning network params to shared target network params 
            if global_step % self.q_target_update_steps == 0:
                self.update_target()

            s, total_episode_reward, steps_at_last_reward, ep_t, episode_ave_max_q, episode_over = \
                self.prepare_state(s, total_episode_reward, steps_at_last_reward, ep_t, episode_ave_max_q, episode_over)

