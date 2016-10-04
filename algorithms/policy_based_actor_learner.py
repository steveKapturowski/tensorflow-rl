# A3C -- unfinished!
from actor_learner import *
from policy_v_network import *
import time
import utils




class A3CLearner(ActorLearner):

    def __init__(self, args):
        
        super(A3CLearner, self).__init__(args)
        
        # Shared mem vars
        self.learning_vars = args.learning_vars

        conf_learning = {'name': "local_learning_{}".format(self.actor_id),
                         'num_act': self.num_actions,
                         'args': args}
        
        self.local_network = PolicyVNetwork(conf_learning)
        self.reset_hidden_state()
            
        if self.actor_id == 0:
            var_list = self.local_network.params
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=3, 
                                        keep_checkpoint_every_n_hours=2)
    

    def reset_hidden_state(self):
        if self.local_network.use_recurrent:
            self.lstm_state_out = np.zeros([1, self.local_network.hidden_state_size])


    def choose_next_action(self, state):
        new_action = np.zeros([self.num_actions])
        network_output_v, network_output_pi = self.session.run(
                [self.local_network.output_layer_v,
                 self.local_network.output_layer_pi], 
                feed_dict={self.local_network.input_ph: [state]})
            
        network_output_pi = network_output_pi.reshape(-1)
        network_output_v = np.asscalar(network_output_v)
            
        action_index = self.sample_policy_action(network_output_pi)
        
        new_action[action_index] = 1

        return new_action, network_output_v, network_output_pi



        # new_action = np.zeros([self.num_actions])
        # network_output_v, network_output_pi, self.lstm_state_out = self.session.run(
        #         [
        #             self.local_network.output_layer_v,
        #             self.local_network.output_layer_pi,
        #             # self.local_network.lstm_state,
        #         ],
        #         feed_dict={
        #             self.local_network.input_ph: [state],
        #             # self.local_network.step_size: [1],
        #             # self.local_network.initial_lstm_state: self.lstm_state_out,
        #         })
            
        # network_output_pi = network_output_pi.reshape(-1)
        # network_output_v = np.asscalar(network_output_v)
            

        # action_index = self.sample_policy_action(network_output_pi)
        
        # new_action[action_index] = 1

        # return new_action, network_output_v, network_output_pi

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
        super(A3CLearner, self).run()
        #cProfile.runctx('self._run()', globals(), locals(), 'profile-{}.out'.format(self.actor_id))
        self._run()

    def _run(self):
        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id, 
            self.global_step.value()))

        s = self.emulator.get_initial_state()
        total_episode_reward = 0

        s_batch = []
        a_batch = []
        y_batch = []
        adv_batch = []
        
        episode_over = False
        start_time = time.time()
        
        while (self.global_step.value() < self.max_global_steps):

            # Sync local learning net with shared mem
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            self.save_vars()

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
                
            #self.save_state(states)
            
            # Calculate the value offered by critic in the new state.
            if episode_over:
                R = 0
            else:
                R = self.session.run(
                    self.local_network.output_layer_v, 
                    #feed_dict={self.local_network.input_ph:[states[-1]]})[0][0]
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
                # self.local_network.step_size : [len(s_batch)],
                # self.local_network.initial_lstm_state: self.lstm_state_out,
            }


            grads = self.session.run(
                                self.local_network.get_gradients,
                                feed_dict=feed_dict)

            self.apply_gradients_to_shared_memory_vars(grads)     
            
            s_batch = []
            a_batch = []
            y_batch = []          
            adv_batch = []
            
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
                self.reset_hidden_state()
                if self.emulator.env.ale.lives() == 0:
                    s = self.emulator.get_initial_state()


    @utils.only_on_train()
    def log_summary(self, total_episode_reward):
        if (self.actor_id == 0):
            feed_dict = {self.summary_ph[0]: total_episode_reward}
            res = self.session.run(self.update_ops + [self.summary_op], feed_dict=feed_dict)
            self.summary_writer.add_summary(res[-1], self.global_step.value())

