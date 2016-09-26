from actor_learner import *
from q_network import *
import time
import sys
from hogupdatemv import copy
#import cProfile

class ValueBasedLearner(ActorLearner):

    def __init__(self, args):
        
        super(ValueBasedLearner, self).__init__(args)
        
        # Shared mem vars
        self.target_vars = args.target_vars
        self.target_update_flags = args.target_update_flags
        self.q_target_update_steps = args.q_target_update_steps 
        
        
        conf_learning = {'name': "local_learning_{}".format(self.actor_id),
                         'num_act': self.num_actions,
                         'args': args}

        conf_target = {'name': "local_target_{}".format(self.actor_id),
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = QNetwork(conf_learning)
        self.target_network = QNetwork(conf_target)
        
        if self.actor_id == 0:
            var_list = self.local_network.params + self.target_network.params            
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)
        

    def choose_next_action(self, state):
        """ Epsilon greedy """
        new_action = np.zeros([self.num_actions])
        
        network_output = self.session.run(
                    self.local_network.output_layer, 
                    feed_dict={self.local_network.input_ph: [state]})[0]
            
        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(0,self.num_actions)
        else:
            #network_output = self.session.run(
            #    self.local_network.output_layer, 
            #    feed_dict={self.local_network.input_ph: [state]})[0]
            action_index = np.argmax(network_output)
                
        new_action[action_index] = 1
        self.reduce_thread_epsilon()
        
        return new_action, network_output


    def update_target(self):
        copy(np.frombuffer(self.target_vars.vars, ctypes.c_float),
              np.frombuffer(self.learning_vars.vars, ctypes.c_float))
        
        # Set shared flags
        for i in xrange(len(self.target_update_flags.updated)):
            self.target_update_flags.updated[i] = 1





class OneStepQLearner(ValueBasedLearner):

    def __init__(self, args):
        
        super(OneStepQLearner, self).__init__(args)
        
    def run(self):
        super(OneStepQLearner, self).run()
        #cProfile.runctx('self._run()', globals(), locals(), 'profile-{}.out'.format(self.actor_id))
        self._run()
    
    def _run(self):
        """ Main actor learner loop for 1-step Q learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id, 
            self.global_step.value()))

        s = self.emulator.get_initial_state()

        s_batch = []
        a_batch = []
        y_batch = []
        sel_actions = []
        
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
            # Choose next action and execute it
            a, readout_t = self.choose_next_action(s)
            s_prime, reward, episode_over = self.emulator.next(a)
            total_episode_reward += reward
            
            # Rescale or clip immediate reward
            reward = self.rescale_reward(reward)
            
            ep_t += 1
            episode_ave_max_q += np.max(readout_t)
            
            if episode_over:
                y = reward
            else:
                # Q_target in the new state
                q_target_values_new_state = self.session.run(
                    self.target_network.output_layer, 
                    feed_dict={self.target_network.input_ph: [s_prime]})
                y = reward + self.gamma * np.max(q_target_values_new_state)

            a_batch.append(a)
            s_batch.append(s)
            y_batch.append(y)
            
            sel_actions.append(np.argmax(a)) # For debugging purposes
                                   
            self.local_step += 1
            global_step, update_target = self.global_step.increment(
                self.q_target_update_steps)
            
            # Compute grads and asynchronously apply them to shared memory
            if ((self.local_step % self.grads_update_steps == 0) 
                or episode_over):

                # Compute gradients on the local Q network     
                feed_dict={self.local_network.input_ph: s_batch,
                           self.local_network.target_ph: y_batch,
                           self.local_network.selected_action_ph: a_batch}
                           
                grads = self.session.run(self.local_network.get_gradients,
                                          feed_dict=feed_dict)
                    
                self.apply_gradients_to_shared_memory_vars(grads)
                
                # Obs! This step is not in Alg. 1 of http://arxiv.org/pdf/1602.01783.pdf,
                # since that algorithm does not use local network replicas. 
                # However, when using them, we need to synchronize the local network parameters
                # with the shared parameters (equivalent to the "synchronize 
                # thread-specific parameters" step in Alg. 2)
                self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
                self.save_vars()
                
                s_batch = []
                a_batch = []
                y_batch = []
                
            
            # Copy shared learning network params to shared target network params 
            if update_target:
                self.update_target()
                #logger.debug("Actor {} updating target at Global Step {}".format(
                #    self.actor_id, global_step))

            # Sync local tensorflow target network params with shared target network params
            if self.target_update_flags.updated[self.actor_id] == 1:
                self.sync_net_with_shared_memory(self.target_network, self.target_vars)
                self.target_update_flags.updated[self.actor_id] = 0
                #logger.debug("Actor {} syncing target at Global Step {}".format(
                #    self.actor_id, global_step))

            # Prepare next state for the actor
            if episode_over:
                T = self.global_step.value()
                t = self.local_step
                episode_ave_max_q = episode_ave_max_q/float(ep_t)
                s1 = "Q_MAX {0:.4f}".format(episode_ave_max_q)
                s2 = "EPS {0:.4f}".format(self.epsilon)
                logger.info("T{} / STEP {} / REWARD {} / {} / {} / ACTIONS {}".format(self.actor_id, T, total_episode_reward, s1, s2, np.unique(sel_actions)))
                
                if (self.actor_id == 0):
                    stats = [total_episode_reward, episode_ave_max_q, self.epsilon]
                    feed_dict = {}
                    for i in range(len(stats)):
                        feed_dict[self.summary_ph[i]] = float(stats[i])
                    res = self.session.run(self.update_ops + [self.summary_op], feed_dict = feed_dict)
    
                    self.summary_writer.add_summary(res[-1], self.global_step.value())
                
                ep_t = 0
                total_episode_reward = 0
                episode_ave_max_q = 0
                episode_over = False
                sel_actions = []
                s = self.emulator.get_initial_state()
            else:
                s = s_prime

        
class NStepQLearner(ValueBasedLearner):

    def __init__(self, args):
        
        super(NStepQLearner, self).__init__(args)
        
    def run(self):
        super(NStepQLearner, self).run()
        #cProfile.runctx('self._run()', globals(), locals(), 'profile-{}.out'.format(self.actor_id))
        self._run()

    def _run(self):
        """ Main actor learner loop for n-step Q learning. """    

        logger.debug("Actor {} resuming at Step {}, {}".format(self.actor_id, 
            self.global_step.value(), time.ctime()))

        s = self.emulator.get_initial_state()
        
        s_batch = []
        a_batch = []
        y_batch = []
        
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
                
#                 if np.isnan(np.min(readout_t)):
#                     logger.debug("READOUT is NAN")
#                     self.max_global_steps = 0
#                     return #sys.exit()
                
                new_s, reward, episode_over = self.emulator.next(a)
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
                
                global_step, update_target = self.global_step.increment(
                    self.q_target_update_steps)

                if update_target:
                    update_target = False
                    exec_update_target = True
                
                self.local_network.global_step = global_step         

            if episode_over:
                R = 0
            else:
                # Q_target in the new state
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
            feed_dict={self.local_network.input_ph: s_batch,
                        self.local_network.target_ph: y_batch,
                        self.local_network.selected_action_ph: a_batch}
            
            grads = self.session.run(self.local_network.get_gradients,
                                          feed_dict=feed_dict)
            self.apply_gradients_to_shared_memory_vars(grads)
            
            s_batch = []
            a_batch = []
            y_batch = []
            
            if exec_update_target:
                self.update_target()
                exec_update_target = False
                #logger.debug("Actor {} updating target at Global Step {}".format(
                #    self.actor_id, global_step))

            # Sync local tensorflow target network params with shared target network params
            if self.target_update_flags.updated[self.actor_id] == 1:
                self.sync_net_with_shared_memory(self.target_network, self.target_vars)
                self.target_update_flags.updated[self.actor_id] = 0
                #logger.debug("Actor {} syncing target at Global Step {}".format(
                #    self.actor_id, global_step))

            # Start a new game on reaching terminal state
            if episode_over:
                T = self.global_step.value()
                t = self.local_step
                e_prog = float(t)/self.epsilon_annealing_steps
                episode_ave_max_q = episode_ave_max_q/float(ep_t)
                s1 = "Q_MAX {0:.4f}".format(episode_ave_max_q)
                s2 = "EPS {0:.4f}".format(self.epsilon)
                logger.info("T{} / STEP {} / REWARD {} / {} / {}".format(self.actor_id, T, total_episode_reward, s1, s2))
                
                if (self.actor_id == 0):
                    stats = [total_episode_reward, episode_ave_max_q, self.epsilon]
                    feed_dict = {}
                    for i in range(len(stats)):
                        feed_dict[self.summary_ph[i]] = float(stats[i])
                    res = self.session.run(self.update_ops + [self.summary_op], feed_dict = feed_dict)
    
                    self.summary_writer.add_summary(res[-1], self.global_step.value())
                
                ep_t = 0
                total_episode_reward = 0
                episode_ave_max_q = 0
                episode_over = False
                s = self.emulator.get_initial_state()



class OneStepSARSALearner(ValueBasedLearner):

    def __init__(self, args):
        
        super(OneStepSARSALearner, self).__init__(args)
        
    def run(self):
        super(OneStepSARSALearner, self).run()
        #cProfile.runctx('self._run()', globals(), locals(), 'profile-{}.out'.format(self.actor_id))
        self._run()

    def _run(self):
        """ Main actor learner loop for 1-step SARSA learning. """
        logger.debug("Actor {} resuming at Step {}, {}".format(self.actor_id, 
            self.global_step.value(), time.ctime()))
        
        s = self.emulator.get_initial_state()

        s_batch = []
        a_batch = []
        y_batch = []
        sel_actions = []
        
        exec_update_target = False
        total_episode_reward = 0
        episode_ave_max_q = 0
        episode_over = False
        qmax_down = 0
        qmax_up = 0
        prev_qmax = -10*6
        low_qmax = 0
        ep_t = 0

        # Choose next action
        a, readout_t = self.choose_next_action(s)
        
        while (self.global_step.value() < self.max_global_steps):
            s_prime, reward, episode_over = self.emulator.next(a)
            total_episode_reward += reward
            
            ep_t += 1
            episode_ave_max_q += np.max(readout_t)

            a_batch.append(a)
            s_batch.append(s)
            sel_actions.append(np.argmax(a)) # For debugging purposes

            # Rescale or clip immediate reward
            reward = self.rescale_reward(reward)
            if episode_over:
                y = reward
            else:
                # Choose action that we will execute in the next step 
                a_prime, readout_t = self.choose_next_action(s_prime)
                # Q_target in the new state for the next step action 
                q_target_values_new_state = self.session.run(
                    self.target_network.output_layer, 
                    feed_dict={self.target_network.input_ph: [s_prime]})[0]
                y = reward + self.gamma * \
                    q_target_values_new_state[np.argmax(a_prime)]
                a = a_prime

            y_batch.append(y)
            
            self.local_step += 1
            global_step, update_target = self.global_step.increment(
                self.q_target_update_steps)
            
            # Compute grads and asynchronously apply them to shared memory
            if ((self.local_step % self.grads_update_steps == 0) 
                or episode_over):

                # Compute gradients on the local Q network     
                feed_dict={self.local_network.input_ph: s_batch,
                           self.local_network.target_ph: y_batch,
                           self.local_network.selected_action_ph: a_batch}
                           
                grads = self.session.run(self.local_network.get_gradients,
                                          feed_dict=feed_dict)
                    
                self.apply_gradients_to_shared_memory_vars(grads)

                # Obs! This step is not in Alg. 1 of http://arxiv.org/pdf/1602.01783.pdf,
                # since that algorithm does not use local network replicas. 
                # However, when using them, we need to synchronize the local network parameters
                # with the shared parameters (equivalent to the "synchronize 
                # thread-specific parameters" step in Alg. 2)
                self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
                self.save_vars()
                
                s_batch = []
                a_batch = []
                y_batch = []
                
            # Copy shared learning network params to shared target network params 
            if update_target:
                self.update_target()
                #logger.debug("Actor {} updating target at Global Step {}".format(
                #    self.actor_id, global_step))

            # Sync local tensorflow target network params with shared target network params
            if self.target_update_flags.updated[self.actor_id] == 1:
                self.sync_net_with_shared_memory(self.target_network, self.target_vars)
                self.target_update_flags.updated[self.actor_id] = 0
                #logger.debug("Actor {} syncing target at Global Step {}".format(
                #    self.actor_id, global_step))
            
            # Prepare next state for the actor
            if episode_over:
                T = self.global_step.value()
                t = self.local_step
                episode_ave_max_q = episode_ave_max_q/float(ep_t)
                s1 = "Q_MAX {0:.4f}".format(episode_ave_max_q)
                s2 = "EPS {0:.4f}".format(self.epsilon)
                logger.info("T{} / STEP {} / REWARD {} / {} / {} / ACTIONS {}".format(self.actor_id, T, total_episode_reward, s1, s2, np.unique(sel_actions)))
                
                if (self.actor_id == 0):
                    stats = [total_episode_reward, episode_ave_max_q, self.epsilon]
                    feed_dict = {}
                    for i in range(len(stats)):
                        feed_dict[self.summary_ph[i]] = float(stats[i])
                    res = self.session.run(self.update_ops + [self.summary_op], feed_dict = feed_dict)
    
                    self.summary_writer.add_summary(res[-1], self.global_step.value())
                
                ep_t = 0
                total_episode_reward = 0
                episode_ave_max_q = 0
                episode_over = False
                sel_actions = []
                s = self.emulator.get_initial_state()
                a, readout_t = self.choose_next_action(s)
            else:
                s = s_prime
