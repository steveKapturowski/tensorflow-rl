# -*- coding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from actor_learner import ONE_LIFE_GAMES
from utils.replay_memory import ReplayMemory
from networks.policy_v_network import PolicyVNetwork
from policy_based_actor_learner import BaseA3CLearner


logger = utils.logger.getLogger('pgq_actor_learner')

class PGQLearner(BaseA3CLearner):
    def __init__(self, args):

        super(PGQLearner, self).__init__(args)

        conf_learning = {'name': 'local_learning_{}'.format(self.actor_id),
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = PolicyVNetwork(conf_learning)
        self.reset_hidden_state()
            
        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)

        # pgq specific initialization
        self.batch_size = 32
        self.replay_memory = ReplayMemory(args.replay_size)
        self.q_tilde = self.local_network.beta * (
            self.local_network.log_output_layer_pi
            + tf.expand_dims(self.local_network.output_layer_entropy, 1)
        ) + self.local_network.output_layer_v

        self.Qi, self.Qi_plus_1 = tf.split(axis=0, num_or_size_splits=2, value=self.q_tilde)
        self.V, _ = tf.split(axis=0, num_or_size_splits=2, value=self.local_network.output_layer_v)
        self.pi, _ = tf.split(axis=0, num_or_size_splits=2, value=tf.expand_dims(self.local_network.log_output_selected_action, 1))
        self.R = tf.placeholder('float32', [None], name='1-step_reward')

        self.terminal_indicator = tf.placeholder(tf.float32, [None], name='terminal_indicator')
        self.max_TQ = self.gamma*tf.reduce_max(self.Qi_plus_1, 1) * (1 - self.terminal_indicator)
        self.Q_a = tf.reduce_sum(self.Qi * tf.split(axis=0, num_or_size_splits=2, value=self.local_network.selected_action_ph)[0], 1)
        self.q_objective = -0.5 * tf.reduce_mean(tf.stop_gradient(self.R + self.max_TQ - self.Q_a) * (self.V + self.pi))


        self.V_params = self.local_network.params #[var for var in self.local_network.params if 'policy' not in var.name]
        self.q_gradients = tf.gradients(self.q_objective, self.V_params)

        # if self.clip_norm_type == 'global':
        #     # Clip network grads by network norm
        #     self.q_gradients = tf.clip_by_global_norm(
        #                 self.q_gradients, self.clip_norm)[0]
        # elif self.clip_norm_type == 'local':
        #     # Clip layer grads by layer norm
        #     self.q_gradients = [tf.clip_by_norm(
        #                 g, self.clip_norm) for g in self.q_gradients]


        if (self.optimizer_mode == "local"):
            if (self.optimizer_type == "rmsprop"):
                self.batch_opt_st = np.ones(size, dtype=ctypes.c_float)
            else:
                self.batch_opt_st = np.zeros(size, dtype=ctypes.c_float)
        elif (self.optimizer_mode == "shared"):
                self.batch_opt_st = args.batch_opt_state


    def apply_batch_q_update(self):
        s_i, a_i, r_i, s_f, is_terminal = self.replay_memory.sample_batch(self.batch_size)

        batch_grads = self.session.run(
            self.q_gradients,
            feed_dict={
                self.R: r_i,
                self.local_network.selected_action_ph: np.vstack([a_i, a_i]),
                self.local_network.input_ph: np.vstack([s_i, s_f]),
                self.terminal_indicator: is_terminal.astype(np.int),
            }
        )

        self._apply_gradients_to_shared_memory_vars(batch_grads, opt_st=self.batch_opt_st)


    def choose_next_action(self, state):
        network_output_v, network_output_pi, q_tilde = self.session.run(
                [self.local_network.output_layer_v,
                 self.local_network.output_layer_pi,
                 self.local_network.q_tilde], 
                feed_dict={self.local_network.input_ph: [state]})
            
        network_output_pi = network_output_pi.reshape(-1)
        network_output_v = np.asscalar(network_output_v)


        action_index = self.sample_policy_action(network_output_pi)
        new_action = np.zeros([self.num_actions])
        new_action[action_index] = 1


        return new_action, network_output_v, network_output_pi, q_tilde[action_index]


    def _run(self):
        if not self.is_train:
            return self.test()

        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id, 
            self.global_step.value()))

        s = self.emulator.get_initial_state()
        total_episode_reward = 0
        
        reset_game = False
        episode_over = False
        start_time = time.time()
        steps_at_last_reward = self.local_step
        
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            self.save_vars()

            local_step_start = self.local_step 
            
            states    = list()
            rewards   = list()
            actions   = list()
            values    = list()
            q_tildes  = list()
            s_batch   = list()
            a_batch   = list()
            y_batch   = list()
            adv_batch = list()
            
            while not (episode_over 
                or (self.local_step - local_step_start 
                    == self.max_local_steps)):
                
                # Choose next action and execute it
                a, readout_v_t, readout_pi_t, q_tilde = self.choose_next_action(s)
                
                if (self.actor_id == 0) and (self.local_step % 100 == 0):
                    logger.debug("pi={}, V={}".format(readout_pi_t, readout_v_t))
                    
                new_s, reward, episode_over = self.emulator.next(a)

                if reward != 0.0:
                    steps_at_last_reward = self.local_step


                total_episode_reward += reward
                # Rescale or clip immediate reward
                reward = self.rescale_reward(reward)
                self.replay_memory.append((s, a, reward, new_s, episode_over))
                
                rewards.append(reward)
                states.append(s)
                actions.append(a)
                values.append(readout_v_t)
                q_tildes.append(q_tilde)
                
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
                # adv_batch.append(R - values[i])
                adv_batch.append(R - q_tildes[i])
                
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

            # let the replay buffer fill up a bit before we start sampling
            if self.local_step > 1000:
                self.apply_batch_q_update()
            



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

                self.reset_hidden_state()
                steps_at_last_reward = self.local_step
                total_episode_reward = 0
                episode_over = False

                if reset_game or self.emulator.game in ONE_LIFE_GAMES:
                    s = self.emulator.get_initial_state()
                    reset_game = False

