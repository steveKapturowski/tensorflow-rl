import time
import numpy as np
import utils.logger
import tensorflow as tf
from actor_learner import ONE_LIFE_GAMES
from utils.replay_memory import ReplayMemory
from policy_based_actor_learner import BaseA3CLearner


logger = utils.logger.getLogger('pgq')

class PGQLearner(BaseA3CLearner):
    def __init__(self, args):

        super(PGQLearner, self).__init__(args)

        self.batch_size = 32
        self.replay_memory = ReplayMemory(args.replay_size)
        self.q_estimate = self.local_network.beta * (
            self.local_network.log_output_layer_pi
            + tf.expand_dims(self.local_network.output_layer_entropy, 1)
        ) + self.local_network.output_layer_v

        self.Q, self.TQ = tf.split(0, 2, self.q_estimate)
        self.V, _ = tf.split(0, 2, self.local_network.output_layer_v)
        self.pi, _ = tf.split(0, 2, tf.expand_dims(self.local_network.log_output_selected_action, 1))
        self.R = tf.placeholder('float32', [None], name='1-step_reward')

        self.terminal_indicator = tf.placeholder(tf.float32, [None], name='terminal_indicator')
        self.max_TQ = self.gamma*tf.reduce_max(self.TQ, 1) * (1 - self.terminal_indicator)
        self.Q_a = tf.reduce_sum(self.Q * tf.split(0, 2, self.local_network.selected_action_ph)[0], 1)
        self.q_objective = -0.5 * tf.reduce_mean(tf.stop_gradient(self.R + self.max_TQ - self.Q_a) * (self.V + self.pi))


        self.V_params = self.local_network.params #[var for var in self.local_network.params if 'policy' not in var.name]
        self.q_gradients = tf.gradients(self.q_objective, self.V_params)

        if (self.optimizer_mode == "local"):
            if (self.optimizer_type == "rmsprop"):
                self.batch_opt_st = np.ones(size, dtype=ctypes.c_float)
            else:
                self.batch_opt_st = np.zeros(size, dtype=ctypes.c_float)
        elif (self.optimizer_mode == "shared"):
                self.batch_opt_st = args.opt_state


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
                self.replay_memory.append((s, a, reward, new_s, episode_over))
                
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
            if self.local_step > 1000:
                self.apply_batch_q_update()
            
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

