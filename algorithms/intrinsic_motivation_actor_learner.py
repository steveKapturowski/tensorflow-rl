# -*- encoding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from collections import deque
from utils import checkpoint_utils

from utils.fast_cts import CTSDensityModel
from policy_based_actor_learner import A3CLearner
from value_based_actor_learner import ValueBasedLearner


logger = utils.logger.getLogger('intrinsic_motivation_actor_learner')


class PseudoCountA3CLearner(A3CLearner):
    def __init__(self, args):
        super(PseudoCountA3CLearner, self).__init__(args)

        #more cython tuning could useful here
        self.density_model = CTSDensityModel(
            height=args.cts_rescale_dim,
            width=args.cts_rescale_dim,
            num_bins=args.cts_bins,
            beta=0.05)


    def _run(self):
        if not self.is_train:
            return self.test()

        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id, 
            self.global_step.value()))

        s = self.emulator.get_initial_state()
        total_episode_reward = 0.0
        mean_entropy = 0.0
        episode_start_step = 0
        
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            self.save_vars()

            local_step_start = self.local_step 
            
            reset_game = False
            episode_over = False

            bonuses   = deque(maxlen=100)
            rewards   = list()
            states    = list()
            actions   = list()
            values    = list()
            s_batch   = list()
            a_batch   = list()
            y_batch   = list()
            adv_batch = list()
            
            while not (episode_over 
                or (self.local_step - local_step_start 
                    == self.max_local_steps)):
                
                # Choose next action and execute it
                a, readout_v_t, readout_pi_t = self.choose_next_action(s)
                    
                new_s, reward, episode_over = self.emulator.next(a)
                total_episode_reward += reward
                
                current_frame = new_s[...,-1]
                bonus = self.density_model.update(current_frame)
                bonuses.append(bonus)

                if (self.actor_id == 0) and (self.local_step % 200 == 0):
                    bonus_array = np.array(bonuses)
                    logger.debug('π_a={:.4f} / V={:.4f} / Mean Bonus={:.4f} / Max Bonus={:.4f}'.format(
                        readout_pi_t[a.argmax()], readout_v_t, bonus_array.mean(), bonus_array.max()))

                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward + bonus)
                
                rewards.append(reward)
                states.append(s)
                actions.append(a)
                values.append(readout_v_t)
                
                s = new_s
                self.local_step += 1
                self.global_step.increment()
                
            
            # Calculate the value offered by critic in the new state.
            if episode_over:
                R = 0
            else:
                R = self.session.run(
                    self.local_network.output_layer_v,
                    feed_dict={self.local_network.input_ph:[new_s]})[0][0]
                            

            sel_actions = []
            for i in reversed(xrange(len(states))):
                R = rewards[i] + self.gamma * R

                y_batch.append(R)
                a_batch.append(actions[i])
                s_batch.append(states[i])
                adv_batch.append(R - values[i])
                
                sel_actions.append(np.argmax(actions[i]))


            # Compute gradients on the local policy/V network and apply them to shared memory  
            feed_dict={
                self.local_network.input_ph: s_batch, 
                self.local_network.critic_target_ph: y_batch,
                self.local_network.selected_action_ph: a_batch,
                self.local_network.adv_actor_ph: adv_batch,
            }
            grads, entropy = self.session.run(
                [self.local_network.get_gradients, self.local_network.entropy],
                feed_dict=feed_dict)

            self.apply_gradients_to_shared_memory_vars(grads)     

            delta_old = local_step_start - episode_start_step
            delta_new = self.local_step -  local_step_start
            mean_entropy = (mean_entropy*delta_old + entropy*delta_new) / (delta_old + delta_new)
            
            s, mean_entropy, episode_start_step, total_episode_reward, _ = self.prepare_state(
                s, mean_entropy, episode_start_step, total_episode_reward, self.local_step, sel_actions, episode_over)


class PseudoCountNStepQLearner(ValueBasedLearner):

    def __init__(self, args):
        super(PseudoCountNStepQLearner, self).__init__(args)

        #more cython tuning could useful here
        self.density_model = CTSDensityModel(
            height=args.cts_rescale_dim,
            width=args.cts_rescale_dim,
            num_bins=args.cts_bins,
            beta=0.05)


    def _run(self):
        """ Main actor learner loop for n-step Q learning. """
        if not self.is_train:
            return self.test()  

        logger.debug("Actor {} resuming at Step {}, {}".format(self.actor_id, 
            self.global_step.value(), time.ctime()))

        s = self.emulator.get_initial_state()
        
        s_batch = []
        a_batch = []
        y_batch = []
        bonuses = deque(maxlen=100)
        
        exec_update_target = False
        total_episode_reward = 0
        episode_ave_max_q = 0
        episode_over = False
        qmax_down = 0
        qmax_up = 0
        prev_qmax = -10*6
        low_qmax = 0
        ep_t = 0
        
        while (self.global_step.value() < self.max_global_steps):

            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            self.save_vars()

            rewards = []
            states = []
            actions = []
            local_step_start = self.local_step
            
            while not (episode_over 
                or (self.local_step - local_step_start == self.max_local_steps)):
                
                # Choose next action and execute it
                a, readout_t = self.choose_next_action(s)

                
                new_s, reward, episode_over = self.emulator.next(a)
                total_episode_reward += reward

                current_frame = new_s[...,-1]
                bonus = self.density_model.update(current_frame)
                bonuses.append(bonus)

                if (self.actor_id == 0) and (self.local_step % 200 == 0):
                    bonus_array = np.array(bonuses)
                    logger.debug('Mean Bonus={:.4f} / Max Bonus={:.4f}'.format(
                        bonus_array.mean(), bonus_array.max()))

                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward + bonus)

                ep_t += 1
                
                rewards.append(reward)
                states.append(s)
                actions.append(a)
                
                s = new_s
                self.local_step += 1
                episode_ave_max_q += np.max(readout_t)
                
                global_step, update_target = self.global_step.increment(
                    self.q_target_update_steps)

                if update_target:
                    update_target = False
                    exec_update_target = True
                
                self.local_network.global_step = global_step

            if episode_over:
                R = 0
            else:
                q_target_values_next_state = self.session.run(
                    self.target_network.output_layer, 
                    feed_dict={self.target_network.input_ph: 
                        [new_s]})
                R = np.max(q_target_values_next_state)
                   
            for i in reversed(xrange(len(states))):
                R = rewards[i] + self.gamma * R
                
                y_batch.append(R)
                a_batch.append(actions[i])
                s_batch.append(states[i])
                
            # Compute gradients on the local Q network     
            feed_dict={
                self.local_network.input_ph: s_batch,
                self.local_network.target_ph: y_batch,
                self.local_network.selected_action_ph: a_batch
            }

            grads = self.session.run(self.local_network.get_gradients,
                                          feed_dict=feed_dict)
            self.apply_gradients_to_shared_memory_vars(grads)
            
            s_batch = []
            a_batch = []
            y_batch = []
            
            if exec_update_target:
                self.update_target()
                exec_update_target = False

            # Sync local tensorflow target network params with shared target network params
            if self.target_update_flags.updated[self.actor_id] == 1:
                self.sync_net_with_shared_memory(self.target_network, self.target_vars)
                self.target_update_flags.updated[self.actor_id] = 0

            s, total_episode_reward, _, ep_t, episode_ave_max_q, episode_over = \
                self.prepare_state(s, total_episode_reward, self.local_step, ep_t, episode_ave_max_q, episode_over)

