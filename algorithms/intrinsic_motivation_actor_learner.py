# -*- encoding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from utils import checkpoint_utils

from utils.fast_cts import CTSDensityModel
from policy_based_actor_learner import A3CLearner


logger = utils.logger.getLogger('intrinsic_motivation_actor_learner')


class PseudoCountLearner(A3CLearner):
    def __init__(self, args):
        super(PseudoCountLearner, self).__init__(args)

        #more cython tuning could useful here
        self.density_model = CTSDensityModel(
            height=21, width=21, beta=0.05)


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

            bonuses   = list()
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
                
                if (self.actor_id == 0) and (self.local_step % 100 == 0):
                    logger.debug("pi={}, V={}".format(readout_pi_t, readout_v_t))
                    
                new_s, reward, episode_over = self.emulator.next(a)
                
                current_frame = new_s[...,-1]
                bonus = self.density_model.update(current_frame)
                bonuses.append(bonus)

                total_episode_reward += reward
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

            logger.debug('Bonuses: {}'.format(bonuses))
            
            s, mean_entropy, episode_start_step, total_episode_reward, _ = self.prepare_state(
                s, mean_entropy, episode_start_step, total_episode_reward, self.local_step, sel_actions, episode_over)


