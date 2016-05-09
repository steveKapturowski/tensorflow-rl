from actor_learner import *

class OneStepQLearner(ActorLearner):

    def __init__(self, session, optimizer_conf, emulator_conf, alg_conf, 
        summary_conf):
        
        super(OneStepQLearner, self).__init__(session, optimizer_conf, 
            emulator_conf, alg_conf, summary_conf)
        
    def run(self):
        """ Main actor learner loop for 1-step Q learning. """
        self.s = self.emulator.get_initial_state()
        total_episode_reward = 0

        logger.debug("Actor {} resuming at Step {}".format(self.getName(), 
            self.global_step))

        while (self.global_step < self.max_global_steps):
            
            # Obs! This step is not in Alg. 1 of 
            # http://arxiv.org/pdf/1602.01783.pdf, but it should be equivalent 
            # to the "synchronize thread-specific parameters" step in Alg. 2 
            # when we use local network replicas in Alg. 1
            if (self.local_network is not self.shared_network): 
                self.session.run(
                    self.local_network.sync_parameters_w_shared_network)

            # Choose next action and execute it
            a = self.choose_next_action(self.s, 'e-greedy')
            s_prime, reward, self.terminal_state = self.execute_action(a)

            total_episode_reward += reward

            # Rescale or clip immediate reward
            if self.rescale_rewards:
                reward = self.rescale_reward(reward)
            else:
                reward = self.clip_reward(reward)

            if self.terminal_state:
                y = reward
            else:
                # Q_target in the new state
                q_target_values_new_state = self.session.run(
                    self.target_network.output_layer, 
                    feed_dict={self.target_network.input_placeholder: s_prime})
                y = reward + self.gamma * np.max(q_target_values_new_state)
                   
            # Compute gradients on the local (or shared) Q network     
            feed_dict={self.local_network.input_placeholder: self.s, 
                       self.local_network.target_placeholder: 
                       np.reshape(y, (1)),
                       self.local_network.selected_action_placeholder: 
                       np.reshape(a, (1, self.num_actions))}
            result = self.session.run(self.local_network.get_gradients + 
                                      [self.local_network.loss, 
                                       self.local_network.loss_summary],
                                      feed_dict=feed_dict)
            grads = result[0:-2]
            loss_value = result[-2]
            summary_str1 =  result[-1]

            # Accumulate gradients
            for i in xrange(len(grads)):
                self.accumulated_grads[i] += grads[i]

            self.local_step += 1
              
            # Prepare next state for the actor
            if self.terminal_state:
                self.update_max_score(total_episode_reward)
                total_episode_reward = 0               
                self.s = self.emulator.get_initial_state()
            else:
                self.s = s_prime

            self.lock.acquire()
            self.global_step = self.session.run(self.increase_global_step_op)
            self.lock.release()
            
            if self.global_step % self.q_target_update_steps == 0:
                self.session.run(
                    self.target_network.sync_parameters_w_shared_network)
                logger.debug("Actor {} updating target at Step {}".format(
                    self.getName(), self.global_step))

            # Asynchronously apply accumulated gradients to the shared network
            if ((self.local_step % self.grads_update_steps == 0) 
                or self.terminal_state):                 
                self.apply_gradients_to_shared_network()

            for i in xrange(len(self.accumulated_grads)):
                self.accumulated_grads[i] = 0.0

            if self.local_step % 100 == 0:
                self.summary_writer.add_summary(summary_str1, self.global_step)
                self.summary_writer.flush()


class NStepQLearner(ActorLearner):

    def __init__(self, session, optimizer_conf, emulator_conf, alg_conf, 
        summary_conf):
        
        super(NStepQLearner, self).__init__(session, optimizer_conf, 
            emulator_conf, alg_conf, summary_conf)
        
        self.local_step_start = 1
        
        self.max_local_steps = alg_conf["max_local_steps"]
        
    def run(self):
        """ Main actor learner loop for n-step Q learning. """        
        self.s = self.emulator.get_initial_state()
        total_episode_reward = 0

        logger.debug("Actor {} resuming at Step {}".format(self.getName(), 
            self.global_step))
        
        self.terminal_state = False

        while (self.global_step < self.max_global_steps):

            for i in xrange(len(self.accumulated_grads)):
                self.accumulated_grads[i] = 0.0
            
            # Synchronize thread-specific parameters 
            self.session.run(
                self.local_network.sync_parameters_w_shared_network)

            self.local_step_start = self.local_step 
            
            rewards = []
            states = []
            actions = []
            while not (self.terminal_state 
                or (self.local_step - self.local_step_start 
                    == self.max_local_steps)):
                # Choose next action and execute it
                a = self.choose_next_action(self.s, 'e-greedy')
                new_s, reward, self.terminal_state = self.execute_action(a)
                
                total_episode_reward += reward
                
                # Rescale or clip immediate reward
                if self.rescale_rewards:
                    reward = self.rescale_reward(reward)
                else:
                    reward = self.clip_reward(reward)

                rewards.append(reward)
                states.append(self.s)
                actions.append(a)
                
                self.s = new_s
                self.local_step += 1

                self.lock.acquire()
                self.global_step = self.session.run(
                    self.increase_global_step_op)            
                self.lock.release()

            if self.terminal_state:
                R = 0
            else:
                # Q_target in the new state
                q_target_values_last_state = self.session.run(
                    self.target_network.output_layer, 
                    feed_dict={self.target_network.input_placeholder: 
                        states[-1]})
                R = np.max(q_target_values_last_state)
                   
            for i in reversed(xrange(len(states))):
                R = rewards[i] + self.gamma * R
                
                # Compute gradients on the local (or shared) network    
                feed_dict={self.local_network.input_placeholder: states[i], 
                           self.local_network.target_placeholder: 
                           np.reshape(R, (1)),
                           self.local_network.selected_action_placeholder: 
                           np.reshape(actions[i], (1, self.num_actions))}
                result = self.session.run(self.local_network.get_gradients + 
                                          [self.local_network.loss, 
                                           self.local_network.loss_summary],
                                          feed_dict=feed_dict)
                grads = result[0:-2]
                loss_value = result[-2]
                summary_str1 =  result[-1]
            
                # Accumulate gradients
                for j in xrange(len(grads)):
                    self.accumulated_grads[j] += grads[j]
                
            
            self.apply_gradients_to_shared_network()
            
            if self.global_step % self.q_target_update_steps == 0:
                self.session.run(
                    self.target_network.sync_parameters_w_shared_network)
                logger.debug("Actor {} updating target at Step {}".format(
                    self.getName(), self.global_step))

            if self.local_step % 100 == 0:
                self.summary_writer.add_summary(summary_str1, self.global_step)
                self.summary_writer.flush()

            # Start a new game on reaching terminal state
            if self.terminal_state:
                self.update_max_score(total_episode_reward)
                total_episode_reward = 0
                self.s = self.emulator.get_initial_state()
                self.terminal_state = False


class OneStepSARSALearner(ActorLearner):

    def __init__(self, session, optimizer_conf, emulator_conf, alg_conf, 
        summary_conf):
        
        super(OneStepSARSALearner, self).__init__(session, optimizer_conf, 
            emulator_conf, alg_conf, summary_conf)
        
    def run(self):
        """ Main actor learner loop for 1-step SARSA learning. """
        self.s = self.emulator.get_initial_state()
        total_episode_reward = 0
        # Choose next action
        a = self.choose_next_action(self.s, 'e-greedy')
        
        logger.debug("Actor {} resuming at Step {}".format(self.getName(), 
            self.global_step))

        while (self.global_step < self.max_global_steps):

            # Obs! This step is not in Alg. 1 of 
            # http://arxiv.org/pdf/1602.01783.pdf, but it should be equivalent 
            # to the "synchronize thread-specific parameters" step in Alg. 2 
            # when we use local network replicas in Alg. 1
            if (self.local_network is not self.shared_network):
                self.session.run(
                    self.local_network.sync_parameters_w_shared_network)

            s_prime, reward, self.terminal_state = self.execute_action(a)
            total_episode_reward += reward

            # Rescale or clip immediate reward
            if self.rescale_rewards:
                reward = self.rescale_reward(reward)
            else:
                reward = self.clip_reward(reward)

            if self.terminal_state:
                y = reward
            else:
                # Choose action that we will execute in the next step 
                a_prime = self.choose_next_action(s_prime, 'e-greedy')
                # Q_target in the new state for the next step action 
                q_target_values_new_state = self.session.run(
                    self.target_network.output_layer, 
                    feed_dict={self.target_network.input_placeholder: s_prime})[0]
                y = reward + self.gamma * \
                    q_target_values_new_state[np.argmax(a_prime)]
                a = a_prime
                   
            # Compute gradients on the local (or shared) network   
            feed_dict={self.local_network.input_placeholder: self.s, 
                       self.local_network.target_placeholder: 
                       np.reshape(y, (1)),
                       self.local_network.selected_action_placeholder: 
                       np.reshape(a, (1, self.num_actions))}
            result = self.session.run(self.local_network.get_gradients + 
                                      [self.local_network.loss, 
                                       self.local_network.loss_summary], 
                                       feed_dict=feed_dict)
            
            grads = result[0:-2]
            loss_value = result[-2]
            summary_str1 =  result[-1]
            
            self.local_step += 1
            
            # Accumulate gradients
            for i in xrange(len(grads)):
                self.accumulated_grads[i] += grads[i]
                
            # Prepare next state for the actor
            if self.terminal_state:
                self.update_max_score(total_episode_reward)                
                total_episode_reward = 0   
                self.s = self.emulator.get_initial_state()
                a = self.choose_next_action(self.s, 'e-greedy')                
            else:
                self.s = s_prime
            
            self.lock.acquire()
            self.global_step = self.session.run(self.increase_global_step_op)
            self.lock.release()

            if self.global_step % self.q_target_update_steps == 0:
                self.session.run(
                    self.target_network.sync_parameters_w_shared_network)
                logger.debug("Actor {} updating target at Step {}".format(
                    self.getName(), self.global_step))
            
            # Asynchronously apply accumulated gradients to the shared network
            if ((self.local_step % self.grads_update_steps == 0) 
                or self.terminal_state):
                self.apply_gradients_to_shared_network()
                
            for i in xrange(len(self.accumulated_grads)):
                self.accumulated_grads[i] = 0.0

            if self.local_step % 100 == 0:
                self.summary_writer.add_summary(summary_str1, self.global_step)
                self.summary_writer.flush()
