# -*- encoding: utf-8 -*-
from actor_learner import *
from policy_v_network import PolicyVNetwork, SequencePolicyVNetwork
import time
import utils
import numpy as np


class BaseA3CLearner(ActorLearner):
    def __init__(self, args):

        super(BaseA3CLearner, self).__init__(args)
        
        # Shared mem vars
        self.learning_vars = args.learning_vars
        self.q_target_update_steps = args.q_target_update_steps

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = PolicyVNetwork(conf_learning)
        self.reset_hidden_state()
            
        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)

    def sample_policy_action(self, probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg
    
        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index


    def run(self):
        super(BaseA3CLearner, self).run()
        #cProfile.runctx('self._run()', globals(), locals(), 'profile-{}.out'.format(self.actor_id))
        self._run()


    @utils.only_on_train()
    def log_summary(self, total_episode_reward):
        if (self.actor_id == 0):
            feed_dict = {self.summary_ph[0]: total_episode_reward}
            res = self.session.run(self.update_ops + [self.summary_op], feed_dict=feed_dict)
            self.summary_writer.add_summary(res[-1], self.global_step.value())


class A3CLearner(BaseA3CLearner):
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
        total_episode_reward = 0

        s_batch = []
        a_batch = []
        y_batch = []
        adv_batch = []
        
        reset_game = False
        episode_over = False
        start_time = time.time()
        steps_at_last_reward = self.local_step
        
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)

            local_step_start = self.local_step 
            
            rewards = []
            states = []
            actions = []
            values = []
            
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


            grads = self.session.run(
                                self.local_network.get_gradients,
                                feed_dict=feed_dict)

            self.apply_gradients_to_shared_memory_vars(grads)     
            
            s_batch = []
            a_batch = []
            y_batch = []          
            adv_batch = []
            
            # prevent the agent from getting stuck
            if (self.local_step - steps_at_last_reward > 5000
                or (self.emulator.env.ale.lives() == 0
                    and self.emulator.game not in ONE_LIFE_GAMES)):

                steps_at_last_reward = self.local_step
                episode_over = True
                reset_game = True


            # Start a new game on reaching terminal state
            if episode_over:
                elapsed_time = time.time() - start_time
                global_t = self.global_step.value()
                steps_per_sec = global_t / elapsed_time
                perf = "{:.0f}".format(steps_per_sec)
                logger.info("T{} / STEP {} / REWARD {} / {} STEPS/s, Actions {}".format(self.actor_id, global_t, total_episode_reward, perf, sel_actions))
                
                self.log_summary(total_episode_reward)

                episode_over = False
                total_episode_reward = 0
                steps_at_last_reward = self.local_step

                if reset_game or self.emulator.game in ONE_LIFE_GAMES:
                    s = self.emulator.get_initial_state()
                    reset_game = False




class A3CLSTMLearner(BaseA3CLearner):
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
        total_episode_reward = 0

        s_batch = []
        a_batch = []
        y_batch = []
        adv_batch = []
        
        reset_game = False
        episode_over = False
        start_time = time.time()
        steps_at_last_reward = self.local_step
        
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)

            # Sync target learning net with shared mem
            # if self.local_step % self.q_target_update_steps == 0: # try to stabilize training
            self.save_vars()

            local_step_start = self.local_step
            local_lstm_state = np.copy(self.lstm_state_out)
            
            rewards = []
            states = []
            actions = []
            values = []
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



            grads = self.session.run(
                                self.local_network.get_gradients,
                                feed_dict=feed_dict)

            self.apply_gradients_to_shared_memory_vars(grads)     
            
            s_batch = []
            a_batch = []
            y_batch = []          
            adv_batch = []

            # prevent the agent from getting stuck
            if (self.local_step - steps_at_last_reward > 5000
                or (self.emulator.env.ale.lives() == 0
                    and self.emulator.game not in ONE_LIFE_GAMES)):

                steps_at_last_reward = self.local_step
                episode_over = True
                reset_game = True


            # Start a new game on reaching terminal state
            if episode_over:
                elapsed_time = time.time() - start_time
                global_t = self.global_step.value()
                steps_per_sec = global_t / elapsed_time
                perf = "{:.0f}".format(steps_per_sec)
                logger.info("T{} / STEP {} / REWARD {} / {} STEPS/s, Actions {}".format(self.actor_id, global_t, total_episode_reward, perf, sel_actions))
                
                self.log_summary(total_episode_reward)
                
                if reset_game or self.emulator.game in ONE_LIFE_GAMES:
                    s = self.emulator.get_initial_state()

                self.reset_hidden_state()
                steps_at_last_reward = self.local_step
                total_episode_reward = 0
                episode_over = False
                reset_game = False





class ActionSequenceA3CLearner(BaseA3CLearner):
    def __init__(self, args):

        super(ActionSequenceA3CLearner, self).__init__(args)
        
        # Shared mem vars
        self.learning_vars = args.learning_vars
        self.q_target_update_steps = args.q_target_update_steps

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = SequencePolicyVNetwork(conf_learning)
        self.reset_hidden_state()
            
        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)


    def sample_action_sequence(self, state):
        value = self.session.run(
            self.local_network.output_layer_v,
            feed_dict={
                self.local_network.input_ph: [state],
            }
        )[0, 0]

        modify_state = False
        cell_state = np.zeros((1, 256*2))
        selected_action = np.hstack([np.zeros(self.num_actions), 1]) #`GO` token
        actions = list()

        while True:
            action_probs, cell_state = self.session.run(
                [
                    self.local_network.action_probs,
                    self.local_network.decoder_state,
                ],
                feed_dict={
                    self.local_network.modify_state:          modify_state,
                    self.local_network.input_ph:              [state],
                    self.local_network.decoder_initial_state: cell_state,
                    self.local_network.decoder_seq_lengths:   [1],
                    self.local_network.action_inputs:         [
                        [selected_action]*self.local_network.max_seq_length
                    ],
                }
            )

            selected_action = np.random.multinomial(1, action_probs[0, 0]-np.finfo(np.float32).epsneg)
            # print np.argmax(selected_action), action_probs[0, 0, np.argmax(selected_action)]
            actions.append(selected_action)
            modify_state = True

            if selected_action[self.num_actions] or len(actions) == self.local_network.max_seq_length:
                return actions, value


    def _run(self):
        if not self.is_train:
            return self.test()

        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id, 
            self.global_step.value()))

        s = self.emulator.get_initial_state()
        total_episode_reward = 0

        s_batch = []
        a_batch = []
        y_batch = []
        adv_batch = []
        
        reset_game = False
        episode_over = False
        start_time = time.time()
        steps_at_last_reward = self.local_step
        
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)

            local_step_start = self.local_step 
            
            rewards = []
            states = []
            actions = []
            values = []
            
            while not (episode_over 
                or (self.local_step - local_step_start 
                    == self.max_local_steps)):
                
                # Choose next action and execute it
                action_sequence, readout_v_t = self.sample_action_sequence(s)
                # if (self.actor_id == 0) and (self.local_step % 100 == 0):
                #     logger.debug("pi={}, V={}".format(readout_pi_t, readout_v_t))
                
                acc_reward = 0.0
                for a in action_sequence[:-1]:
                    new_s, reward, episode_over = self.emulator.next(a)
                    acc_reward += reward

                    if episode_over:
                        break

                reward = acc_reward
                if reward != 0.0:
                    steps_at_last_reward = self.local_step


                total_episode_reward += reward
                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward)
                
                rewards.append(reward)
                states.append(s)
                actions.append(action_sequence)
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
                


            seq_lengths = [len(seq) for seq in actions]
            padded_output_sequences = np.array([
                seq + [[0]*(self.num_actions+1)]*(self.local_network.max_seq_length-len(seq))
                for seq in a_batch
            ])

            go_input = np.zeros((len(s_batch), 1, self.num_actions+1))
            go_input[:,:,self.num_actions] = 1
            padded_input_sequences = np.hstack([go_input, padded_output_sequences[:,:-1,:]])

            print 'Sequence lengths:', seq_lengths
            print 'Actions:', [np.argmax(a) for a in a_batch[0]]


            feed_dict={
                self.local_network.input_ph:              s_batch, 
                self.local_network.critic_target_ph:      y_batch,
                self.local_network.adv_actor_ph:          adv_batch,
                self.local_network.decoder_initial_state: np.zeros((len(s_batch), 256*2)),
                self.local_network.action_inputs:         padded_input_sequences,
                self.local_network.action_outputs:        padded_output_sequences,
                self.local_network.modify_state:          False,
                self.local_network.decoder_seq_lengths:   seq_lengths,
            }
            entropy, advantage, grads = self.session.run(
                [
                    self.local_network.entropy,
                    self.local_network.actor_advantage_term,
                    self.local_network.get_gradients
                ],
                feed_dict=feed_dict)

            print 'Entropy:', entropy, 'Adv:', advantage

            self.apply_gradients_to_shared_memory_vars(grads)     
            
            s_batch = []
            a_batch = []
            y_batch = []          
            adv_batch = []
            
            # prevent the agent from getting stuck
            if (self.local_step - steps_at_last_reward > 5000
                or (self.emulator.env.ale.lives() == 0
                    and self.emulator.game not in ONE_LIFE_GAMES)):

                steps_at_last_reward = self.local_step
                episode_over = True
                reset_game = True


            # Start a new game on reaching terminal state
            if episode_over:
                elapsed_time = time.time() - start_time
                global_t = self.global_step.value()
                steps_per_sec = global_t / elapsed_time
                perf = "{:.0f}".format(steps_per_sec)
                logger.info("T{} / STEP {} / REWARD {} / {} STEPS/s, Actions {}".format(self.actor_id, global_t, total_episode_reward, perf, sel_actions))
                
                self.log_summary(total_episode_reward)

                episode_over = False
                total_episode_reward = 0
                steps_at_last_reward = self.local_step

                if reset_game or self.emulator.game in ONE_LIFE_GAMES:
                    s = self.emulator.get_initial_state()
                    reset_game = False


