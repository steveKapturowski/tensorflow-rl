from actor_learner import *

class A3CLearner(ActorLearner):

    def __init__(self, session, optimizer_conf, emulator_conf, alg_conf, 
        summary_conf):
        
        super(A3CLearner, self).__init__(session, optimizer_conf, emulator_conf, 
            alg_conf, summary_conf)
        
        self.local_step_start = 1
        
        self.max_local_steps = alg_conf["max_local_steps"]
        
    def run(self):
        """ Main actor learner loop for advantage actor critic learning. """
        self.s = self.emulator.get_initial_state()
        total_episode_reward = 0

        logger.debug("Actor {} resuming at Step {}".format(self.getName(), 
            self.global_step))
        
        self.terminal_state = False

        while (self.global_step < self.max_global_steps):

            for i in xrange(len(self.actor_accumulated_grads)):
                self.actor_accumulated_grads[i] = 0.0
            for i in xrange(len(self.critic_accumulated_grads)):
                self.critic_accumulated_grads[i] = 0.0
            
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
                a = self.choose_next_action(self.s, 'direct')
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
                # Calculate the value offered by critic in the new state.
                R = self.session.run(
                    self.local_network.output_layer_v, 
                    feed_dict={self.local_network.input_placeholder: 
                        states[-1]})[0]
                   
            for i in reversed(xrange(len(states))):
                R = rewards[i] + self.gamma * R

                # Compute gradients on the local (or shared) policy/V network  
                feed_dict={self.local_network.input_placeholder: states[i], 
                           self.local_network.critic_target_placeholder: 
                           np.reshape(R, (1)),
                           self.local_network.selected_action_placeholder: 
                           np.reshape(actions[i], (1, self.num_actions))}
                result = self.session.run(
                                    self.local_network.get_actor_gradients + 
                                    self.local_network.get_critic_gradients +
                                    [self.local_network.actor_objective, 
                                    self.local_network.critic_loss, 
                                    self.local_network.critic_loss_summary,
                                    self.local_network.actor_objective_summary], 
                                    feed_dict=feed_dict)

                n_actor_grads = len(self.local_network.get_actor_gradients)
                n_critic_grads = len(self.local_network.get_critic_gradients)

                actor_grads = result[0:n_actor_grads]
                critic_grads = result[n_actor_grads:n_actor_grads + n_critic_grads]

                actor_objective_value = result[-4]
                critic_loss_value = result[-3]
                summary_str1 =  result[-2]
                summary_str2 =  result[-1]
                
                # Accumulate gradients
                for j in xrange(len(actor_grads)):
                    self.actor_accumulated_grads[j] += actor_grads[j]
                for j in xrange(len(critic_grads)):
                    self.critic_accumulated_grads[j] += critic_grads[j]
                
            self.apply_gradients_to_shared_network()

            if self.local_step % 100 == 0:
                self.summary_writer.add_summary(summary_str1, self.global_step)
                self.summary_writer.add_summary(summary_str2, self.global_step)
                self.summary_writer.flush()              
            
            # Start a new game on reaching terminal state
            if self.terminal_state:
                self.update_max_score(total_episode_reward)
                total_episode_reward = 0
                self.s = self.emulator.get_initial_state()
                self.terminal_state = False
