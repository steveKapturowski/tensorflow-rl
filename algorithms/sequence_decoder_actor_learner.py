# -*- encoding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf

from utils.forked_debugger import ForkedPdb as Pdb
from actor_learner import ActorLearner, ONE_LIFE_GAMES
from networks.policy_v_network import SequencePolicyVNetwork, PolicyRepeatNetwork
from algorithms.policy_based_actor_learner import BaseA3CLearner


logger = utils.logger.getLogger('action_sequence_actor_learner')


class ActionSequenceA3CLearner(BaseA3CLearner):
    def __init__(self, args):

        super(ActionSequenceA3CLearner, self).__init__(args)
        
        # Shared mem vars
        self.learning_vars = args.learning_vars

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'input_shape': args.input_shape,
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = SequencePolicyVNetwork(conf_learning)
        self.reset_hidden_state()
            
        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)


    def sample_action_sequence(self, state):
        allowed_actions = np.ones((self.local_network.max_decoder_steps, self.local_network.num_actions+1))
        allowed_actions[0, -1] = 0

        action_inputs = np.zeros((1,self.local_network.max_decoder_steps,self.num_actions+1))
        action_inputs[0, 0, -1] = 1

        actions, value = self.session.run(
            [
                self.local_network.actions,
                self.local_network.output_layer_v,
            ],
            feed_dict={
                self.local_network.input_ph:              [state],
                self.local_network.decoder_seq_lengths:   [self.local_network.max_decoder_steps],
                self.local_network.allowed_actions:       [allowed_actions],
                self.local_network.use_fixed_action:      False,
                self.local_network.temperature:           1.0,
                self.local_network.action_outputs:        np.zeros((1,self.local_network.max_decoder_steps,self.num_actions+1)),
                self.local_network.action_inputs:         action_inputs,
                self.local_network.decoder_initial_state: np.zeros((1, self.local_network.decoder_hidden_state_size*2)),
            }
        )

        return actions[0], value[0, 0]


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
        seq_len_batch = []
        
        reset_game = False
        episode_over = False
        start_time = time.time()
        steps_at_last_reward = self.local_step
        
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            self.save_vars()

            local_step_start = self.local_step 
            
            rewards = []
            states = []
            actions = []
            values = []
            seq_lengths = []
            
            while not (episode_over 
                or (self.local_step - local_step_start 
                    == self.max_local_steps)):
                
                # Choose next action and execute it
                action_sequence, readout_v_t = self.sample_action_sequence(s)
                # if (self.actor_id == 0) and (self.local_step % 100 == 0):
                #     logger.debug("pi={}, V={}".format(readout_pi_t, readout_v_t))
                
                acc_reward = 0.0
                length = 0

                for action in action_sequence:
                    length += 1
                    a = np.argmax(action)
                    if a == self.num_actions or episode_over:
                        break

                    new_s, reward, episode_over = self.emulator.next(action[:self.num_actions])
                    acc_reward += reward

                reward = acc_reward
                if reward != 0.0:
                    steps_at_last_reward = self.local_step


                total_episode_reward += reward
                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward)
                
                rewards.append(reward)
                seq_lengths.append(length)
                states.append(s)
                actions.append(action_sequence)
                values.append(readout_v_t)
                

                s = new_s
                self.local_step += 1
                self.global_step.increment()

                if self.local_step % 1000 == 0:
                    pass
                    # Pdb().set_trace()
                
            
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
                seq_len_batch.append(seq_lengths[i])
                
                sel_actions.append(np.argmax(actions[i]))
            
            padded_output_sequences = np.array([
                np.vstack([seq[:length, :], np.zeros((max(seq_len_batch)-length, self.num_actions+1))])
                for length, seq in zip(seq_len_batch, a_batch)
            ])

            go_input = np.zeros((len(s_batch), 1, self.num_actions+1))
            go_input[:,:,self.num_actions] = 1
            padded_input_sequences = np.hstack([go_input, padded_output_sequences[:,:-1,:]])

            print 'Sequence lengths:', seq_lengths
            print 'Actions:', [np.argmax(a) for a in a_batch[0]]

            allowed_actions = np.ones((len(s_batch), max(seq_len_batch), self.num_actions+1))
            allowed_actions[:, 0, -1] = 0 #empty sequence is not a valid action

            feed_dict={
                self.local_network.input_ph:              s_batch, 
                self.local_network.critic_target_ph:      y_batch,
                self.local_network.adv_actor_ph:          adv_batch,
                self.local_network.decoder_initial_state: np.zeros((len(s_batch), self.local_network.decoder_hidden_state_size*2)),
                self.local_network.action_inputs:         padded_input_sequences,
                self.local_network.action_outputs:        padded_output_sequences,
                self.local_network.allowed_actions:       allowed_actions,
                self.local_network.use_fixed_action:      True,
                self.local_network.decoder_seq_lengths:   seq_lengths,
                self.local_network.temperature:           1.0,
            }
            entropy, grads = self.session.run(
                [
                    self.local_network.entropy,
                    # self.local_network.adv_critic,
                    # self.local_network.output_layer_v,
                    self.local_network.get_gradients
                ],
                feed_dict=feed_dict)

            print 'Entropy:', entropy #, 'Adv:', advantage #, 'Value:', value
            self.apply_gradients_to_shared_memory_vars(grads)     
            
            s_batch = []
            a_batch = []
            y_batch = []          
            adv_batch = []
            seq_len_batch = []

            
            # prevent the agent from getting stuck
            if (self.local_step - steps_at_last_reward > 5000
                or (self.emulator.get_lives() == 0
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
                
                self.log_summary(total_episode_reward, entropy)

                episode_over = False
                total_episode_reward = 0
                steps_at_last_reward = self.local_step

                if reset_game or self.emulator.game in ONE_LIFE_GAMES:
                    s = self.emulator.get_initial_state()
                    reset_game = False


class ARA3CLearner(BaseA3CLearner):
    def __init__(self, args):
        super(ARA3CLearner, self).__init__(args)

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'input_shape': self.input_shape,
                         'num_act': self.num_actions,
                         'args': args}

        self.local_network = PolicyRepeatNetwork(conf_learning)
        self.reset_hidden_state()

        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3,
                                        keep_checkpoint_every_n_hours=2)


    def choose_next_action(self, state):
        network_output_v, network_output_pi, action_repeat_probs = self.session.run(
            [
                self.local_network.output_layer_v,
                self.local_network.output_layer_pi,
                self.local_network.action_repeat_probs,
            ],
            feed_dict={
                self.local_network.input_ph: [state],
            })

        network_output_pi = network_output_pi.reshape(-1)
        network_output_v = np.asscalar(network_output_v)

        action_index = self.sample_policy_action(network_output_pi)
        new_action = np.zeros([self.num_actions])
        new_action[action_index] = 1

        action_repeat = 1 + np.random.choice(
            action_repeat_probs.shape[-1],
            p=action_repeat_probs[0, action_index, :])

        return new_action, network_output_v, network_output_pi, action_repeat


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

            rewards        = list()
            states         = list()
            actions        = list()
            values         = list()
            s_batch        = list()
            a_batch        = list()
            y_batch        = list()
            ar_batch       = list()
            adv_batch      = list()
            action_repeats = list()
            
            while not (episode_over 
                or (self.local_step - local_step_start 
                    == self.max_local_steps)):
                
                # Choose next action and execute it
                a, readout_v_t, readout_pi_t, action_repeat = self.choose_next_action(s)
                
                if (self.actor_id == 0) and (self.local_step % 100 == 0):
                    logger.debug("π_a={:.4f} / V={:.4f} repeat={}".format(
                        readout_pi_t[a.argmax()], readout_v_t, action_repeat))

                reward = 0.0
                for _ in range(action_repeat):
                    new_s, reward_i, episode_over = self.emulator.next(a)
                    reward += reward_i

                if reward != 0.0:
                    steps_at_last_reward = self.local_step


                total_episode_reward += reward
                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward)
                
                rewards.append(reward)
                states.append(s)
                actions.append(a)
                values.append(readout_v_t)
                action_repeats.append(action_repeat)
                
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
                ar_batch.append(action_repeats[i])
                
                sel_actions.append(np.argmax(actions[i]))
                

            # Compute gradients on the local policy/V network and apply them to shared memory  
            feed_dict={
                self.local_network.input_ph: s_batch, 
                self.local_network.critic_target_ph: y_batch,
                self.local_network.selected_action_ph: a_batch,
                self.local_network.adv_actor_ph: adv_batch,
                self.local_network.selected_repeat: ar_batch,
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


