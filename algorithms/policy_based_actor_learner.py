# -*- encoding: utf-8 -*-
import time
import numpy as np
import utils.logger
import checkpoint_utils
import tensorflow as tf
from actor_learner import ActorLearner, ONE_LIFE_GAMES
from networks.policy_v_network import PolicyVNetwork


logger = utils.logger.getLogger('policy_based_actor_learner')

class BaseA3CLearner(ActorLearner):
    def __init__(self, args):
        super(BaseA3CLearner, self).__init__(args)
        
        # Shared mem vars
        self.learning_vars = args.learning_vars
        self.q_target_update_steps = args.q_target_update_steps


    def sample_policy_action(self, probs, temperature=0.5):
        probs = probs - np.finfo(np.float32).epsneg
    
        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index


    def run(self):
        super(BaseA3CLearner, self).run()
        #cProfile.runctx('self._run()', globals(), locals(), 'profile-{}.out'.format(self.actor_id))
        self._run()


    @checkpoint_utils.only_on_train()
    def log_summary(self, total_episode_reward, mean_entropy):
        if (self.actor_id == 0):
            feed_dict = {self.summary_ph[0]: total_episode_reward, self.summary_ph[1]: mean_entropy}
            res = self.session.run(self.update_ops + [self.summary_op], feed_dict=feed_dict)
            self.summary_writer.add_summary(res[-1], self.global_step.value())


    def prepare_state(self, state, mean_entropy, episode_start_step, total_episode_reward, 
                      steps_at_last_reward, sel_actions, episode_over):
        # prevent the agent from getting stuck
        reset_game = False

        if (self.local_step - steps_at_last_reward > 5000
            or (self.emulator.get_lives() == 0
                and self.emulator.game not in ONE_LIFE_GAMES)):

            steps_at_last_reward = self.local_step
            episode_over = True
            reset_game = True

        # Start a new game on reaching terminal state
        if episode_over:
            elapsed_time = time.time() - self.start_time
            steps_per_sec = self.global_step.value() / elapsed_time
            perf = "{:.0f}".format(steps_per_sec)
            logger.info("T{} / EPISODE {} / STEP {}k / REWARD {} / {} STEPS/s, Actions {}".format(
                self.actor_id,
                self.local_episode,
                self.global_step.value()/1000,
                total_episode_reward,
                perf,
                sel_actions))
                
            self.log_summary(total_episode_reward, mean_entropy)

            self.reset_hidden_state()
            self.local_episode += 1
            episode_start_step = self.local_step
            steps_at_last_reward = self.local_step
            total_episode_reward = 0.0
            mean_entropy = 0.0

            if reset_game or self.emulator.game in ONE_LIFE_GAMES:
                state = self.emulator.get_initial_state()

        return state, mean_entropy, episode_start_step, total_episode_reward, steps_at_last_reward


class A3CLearner(BaseA3CLearner):
    def __init__(self, args):
        super(A3CLearner, self).__init__(args)

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'input_shape': self.input_shape,
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = PolicyVNetwork(conf_learning)
        self.reset_hidden_state()
            
        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)

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


    def _run(self):
        if not self.is_train:
            return self.test()

        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id, 
            self.global_step.value()))

        s = self.emulator.get_initial_state()
        steps_at_last_reward = self.local_step
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

                if reward != 0.0:
                    steps_at_last_reward = self.local_step


                total_episode_reward += reward
                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward)
                
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

            s, mean_entropy, episode_start_step, total_episode_reward, steps_at_last_reward = self.prepare_state(
                s, mean_entropy, episode_start_step, total_episode_reward, steps_at_last_reward, sel_actions, episode_over)


class A3CLSTMLearner(BaseA3CLearner):
    def __init__(self, args):
        super(A3CLSTMLearner, self).__init__(args)

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'input_shape': self.input_shape,
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = PolicyVNetwork(conf_learning)
        self.reset_hidden_state()
            
        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)


    def reset_hidden_state(self):
        self.lstm_state_out = np.zeros([1, 2*self.local_network.hidden_state_size])


    def choose_next_action(self, state):
        network_output_v, network_output_pi, self.lstm_state_out = self.session.run(
            [
                self.local_network.output_layer_v,
                self.local_network.output_layer_pi,
                self.local_network.lstm_state,
            ],
            feed_dict={
                self.local_network.input_ph: [state],
                self.local_network.step_size: [1],
                self.local_network.initial_lstm_state: self.lstm_state_out,
            })


        network_output_pi = network_output_pi.reshape(-1)
        network_output_v = np.asscalar(network_output_v)
            

        action_index = self.sample_policy_action(network_output_pi)
        new_action = np.zeros([self.num_actions])
        new_action[action_index] = 1

        return new_action, network_output_v, network_output_pi


    def _run(self):
        if not self.is_train:
            return self.test()


        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id, 
            self.global_step.value()))

        s = self.emulator.get_initial_state()
        steps_at_last_reward = self.local_step
        total_episode_reward = 0.0
        mean_entropy = 0.0
        episode_start_step = 0
        
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            self.save_vars()

            local_step_start = self.local_step
            local_lstm_state = np.copy(self.lstm_state_out)
            reset_game = False
            episode_over = False
            
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
                
                assert not np.allclose(local_lstm_state, self.lstm_state_out)


                if (self.actor_id == 0) and (self.local_step % 100 == 0):
                    logger.debug("pi={}, V={}".format(readout_pi_t, readout_v_t))
                    
                new_s, reward, episode_over = self.emulator.next(a)
                if reward != 0.0:
                    steps_at_last_reward = self.local_step


                total_episode_reward += reward
                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward)
                
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
                # compute with repsect to target network
                prev_lstm_state_out = self.lstm_state_out
                R = self.session.run(
                    self.local_network.output_layer_v,
                    feed_dict={
                        self.local_network.input_ph:[new_s],
                        self.local_network.step_size: [1],
                        self.local_network.initial_lstm_state: self.lstm_state_out,
                    }
                )[0][0]
                assert np.allclose(prev_lstm_state_out, self.lstm_state_out)
                            
             
            sel_actions = []
            for i in reversed(xrange(len(states))):
                R = rewards[i] + self.gamma * R

                y_batch.append(R)
                a_batch.append(actions[i])
                s_batch.append(states[i])
                adv_batch.append(R - values[i])
                
                sel_actions.append(np.argmax(actions[i]))
                
            # reverse everything so that the LSTM inputs are time-ordered
            y_batch.reverse()
            a_batch.reverse()
            s_batch.reverse()
            adv_batch.reverse()
            sel_actions.reverse()

            # Compute gradients on the local policy/V network and apply them to shared memory  
            feed_dict={
                self.local_network.input_ph: s_batch, 
                self.local_network.critic_target_ph: y_batch,
                self.local_network.selected_action_ph: a_batch,
                self.local_network.adv_actor_ph: adv_batch,
                self.local_network.step_size : [len(s_batch)],
                self.local_network.initial_lstm_state: local_lstm_state,
            }
            grads, entropy = self.session.run(
                [self.local_network.get_gradients, self.local_network.entropy],
                feed_dict=feed_dict)

            self.apply_gradients_to_shared_memory_vars(grads)

            delta_old = local_step_start - episode_start_step
            delta_new = self.local_step -  local_step_start
            mean_entropy = (mean_entropy*delta_old + entropy*delta_new) / (delta_old + delta_new)  
            
            s, mean_entropy, episode_start_step, total_episode_reward, steps_at_last_reward = self.prepare_state(
                s, mean_entropy, episode_start_step, total_episode_reward, steps_at_last_reward, sel_actions, episode_over)


