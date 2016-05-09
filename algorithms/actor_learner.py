import numpy as np
from threading import Thread
import sys
import logging_utils
from emulator import Emulator

logger = logging_utils.getLogger('actor_learner')

class ActorLearner(Thread):
    
    def __init__(self, session, optimizer_conf, emulator_conf, alg_conf, 
        summary_conf):
        
        super(ActorLearner, self).__init__()
        
        self.local_step = 0
        
        self.global_step, self.epsilon, self.epsilon_limit = session.run(
            [alg_conf['global_step'], 
            alg_conf['epsilon'], 
            alg_conf['epsilon_limit']])
        
        self.epsilon_init = 1.0
        self.epsilon_placeholder = alg_conf['epsilon_placeholder']
        self.update_thread_epsilon_op = alg_conf['update_thread_epsilon_op']
        self.session = session
        
        # Can be local copy of the shared network, or the shared network itself
        self.local_network = alg_conf['local_network']
        
        self.shared_network = alg_conf['shared_network']
        self.target_network = alg_conf['target_network']

        self.actor_learner_type = alg_conf['actor_learner_type']
        self.optimizer_type = optimizer_conf["type"]
        self.optimizer_mode = optimizer_conf["mode"] or "local"

        # Q/sarsa
        if self.actor_learner_type == 0 or self.actor_learner_type == 1:
            
            self.accumulated_grads = [0.0 for _ in self.local_network.params]       
            
            if ((self.optimizer_type == "momentum") 
                and (self.optimizer_mode == "local")): 
                self.m = []
                for var in self.local_network.params:
                    self.m.append(np.zeros(var.get_shape().as_list()))
            
            if ((self.optimizer_type == "rmsprop") 
                and (self.optimizer_mode == "local")):
                self.g = []
                for var in self.local_network.params:
                    self.g.append(np.zeros(var.get_shape().as_list()))

        # actor critic
        elif self.actor_learner_type == 2:

            self.actor_accumulated_grads = [0.0 
                for _ in self.local_network.actor_params]

            self.critic_accumulated_grads = [0.0 
                for _ in self.local_network.critic_params]

            if ((self.optimizer_type == "momentum") 
                and (self.optimizer_mode == "local")): 
                self.actor_m = []
                for var in self.local_network.actor_params:
                    self.actor_m.append(np.zeros(var.get_shape().as_list()))
                self.critic_m = []
                for var in self.local_network.critic_params:
                    self.critic_m.append(np.zeros(var.get_shape().as_list()))

            if ((self.optimizer_type == "rmsprop") 
                and (self.optimizer_mode == "local")):
                self.actor_g = []
                for var in self.local_network.actor_params:
                    self.actor_g.append(np.zeros(var.get_shape().as_list()))
                self.critic_g = []
                for var in self.local_network.critic_params:
                    self.critic_g.append(np.zeros(var.get_shape().as_list()))            

        self.increase_global_step_op = alg_conf['increase_global_step_op']
        self.actor_id = alg_conf['actor_id']
        self.summary_op = summary_conf['summary_op']
        self.summary_writer = summary_conf['summary_writer']
        
        self.emulator = Emulator(
            emulator_conf["rom_path"], 
            emulator_conf["game"], 
            emulator_conf["visualize"], 
            self.actor_id)

        self.num_actions = len(self.emulator.legal_actions)
        self.s = None
        self.q_target_update_steps = alg_conf["q_target_update_steps"] 
        self.grads_update_steps = alg_conf["grads_update_steps"]
        self.max_global_steps = alg_conf["max_global_steps"]
        
        self.max_epsilon_annealing_steps = alg_conf[
            'max_epsilon_annealing_steps']
        
        self.gamma = alg_conf["gamma"]
        self.terminal_state = False
        self.global_score = alg_conf['global_score']
        self.global_score_placeholder = alg_conf['global_score_placeholder']
        self.update_global_score_op = alg_conf['update_global_score_op']
        self.global_score_summary = summary_conf['global_score_summary']
        self.thread_score = alg_conf['thread_score']
        self.thread_score_placeholder = alg_conf['thread_score_placeholder']
        self.update_thread_score_op = alg_conf['update_thread_score_op']

        self.rescale_rewards = alg_conf['rescale_rewards']
        if self.rescale_rewards:
            self.thread_max_reward = alg_conf['thread_max_reward']
            self.thread_max_reward_placeholder = \
                alg_conf['thread_max_reward_placeholder']
            self.update_max_reward_op = alg_conf['update_thread_max_reward_op']
            self.max_reward = self.session.run(self.thread_max_reward)

        # Updating target network at regular intervals w.r.t. global step, 
        # global step, and global scores requires locking! Otherwise, global 
        # step and score are handled asynchronously by tensorflow. They ought 
        # to be in lock step.
        self.lock = alg_conf['lock']

    def reduce_thread_epsilon(self):
        """ Linear annealing """
        if self.global_step <= self.max_epsilon_annealing_steps:
            
            self.epsilon = self.epsilon_init - ((self.global_step * 
                (self.epsilon_init - self.epsilon_limit)) / 
                self.max_epsilon_annealing_steps)
            
            self.session.run(self.update_thread_epsilon_op, 
                feed_dict={self.epsilon_placeholder: self.epsilon})

    def choose_next_action(self, state, policy_type):
        """ Epsilon greedy/direct policy """
        new_action = np.zeros([self.num_actions])
        if policy_type == 'e-greedy':
            if np.random.rand() <= self.epsilon:
                action_index = np.random.randint(0,self.num_actions)
            else:
                network_output = self.session.run(
                    self.local_network.output_layer, 
                    feed_dict={self.local_network.input_placeholder: state})[0]
                
                action_index = np.argmax(network_output)                   
            self.reduce_thread_epsilon()
        
        elif policy_type == 'direct':
            network_output = self.session.run(
                self.local_network.output_layer_p, 
                feed_dict={self.local_network.input_placeholder: state})[0]
            # print('softmax output:', network_output)
            action_index = np.random.choice(
                range(self.num_actions), p=network_output) 
        
        new_action[action_index] = 1
        return new_action
    
    def execute_action(self, a):
        """ Returns the next state, reward and whether or not the next state 
        is terminal. """
        return self.emulator.next(a)
    
    def apply_gradients_to_shared_network(self):
        """ Apply accumulated gradients to the shared network and clear 
        accumulated gradients. """
        feed_dict = {}
        # Q/sarsa
        if self.actor_learner_type == 0 or self.actor_learner_type == 1:
            for i in xrange(len(self.accumulated_grads)):          
                if (self.optimizer_mode == "local"):
                    if (self.optimizer_type == "momentum"):
                        self.m[i] = self.m[i] * 0.9 + self.accumulated_grads[i]
                        
                        feed_dict[self.shared_network.gradient_placeholders[i]] \
                            = self.m[i]
                                
                    elif (self.optimizer_type == "rmsprop"):
                        self.g[i] = self.g[i] * 0.9 + (self.accumulated_grads[i] * 
                            self.accumulated_grads[i]) * 0.1
                        
                        feed_dict[self.shared_network.gradient_placeholders[i]] \
                            = self.accumulated_grads[i] / \
                                np.sqrt(self.g[i] + 1e-6)
            
                else:
                    feed_dict[self.shared_network.gradient_placeholders[i]] \
                        = self.accumulated_grads[i]

        # actor critic
        elif self.actor_learner_type == 2:
            for i in xrange(len(self.actor_accumulated_grads)):          
                if (self.optimizer_mode == "local"):
                    if (self.optimizer_type == "momentum"):
                        self.actor_m[i] = self.actor_m[i] * 0.9 + \
                            self.actor_accumulated_grads[i] # * 0.1
                        feed_dict[self.shared_network.actor_gradient_placeholders[i]] \
                            = self.actor_m[i]
                                
                    elif (self.optimizer_type == "rmsprop"):
                        self.actor_g[i] = self.actor_g[i] * 0.9 + \
                            (self.actor_accumulated_grads[i] * 
                                self.actor_accumulated_grads[i]) * 0.1
                        feed_dict[self.shared_network.actor_gradient_placeholders[i]] \
                            = self.actor_accumulated_grads[i] / \
                                np.sqrt(self.actor_g[i] + 1e-6)
    
                else:
                    feed_dict[self.shared_network.actor_gradient_placeholders[i]] \
                        = self.actor_accumulated_grads[i]

            for i in xrange(len(self.critic_accumulated_grads)):          
                if (self.optimizer_mode == "local"):
                    if (self.optimizer_type == "momentum"):
                        self.critic_m[i] = self.critic_m[i] * 0.9 + \
                            self.critic_accumulated_grads[i] # * 0.1
                        feed_dict[self.shared_network.critic_gradient_placeholders[i]] \
                            = self.critic_m[i]
                                
                    elif (self.optimizer_type == "rmsprop"):
                        self.critic_g[i] = self.critic_g[i] * 0.9 + \
                            (self.critic_accumulated_grads[i] * 
                                self.critic_accumulated_grads[i]) * 0.1
                        feed_dict[self.shared_network.critic_gradient_placeholders[i]] \
                            = self.critic_accumulated_grads[i] / \
                                np.sqrt(self.critic_g[i] + 1e-6)
    
                else:
                    feed_dict[self.shared_network.critic_gradient_placeholders[i]] \
                        = self.critic_accumulated_grads[i]
        
        self.session.run(self.shared_network.apply_gradients, 
            feed_dict=feed_dict)

    def update_max_score(self, total_episode_reward):
        """ Record game score (total_episode_reward) per thread and globally 
        (across threads), if > current. """
        max_thread_score = self.session.run(self.thread_score)
        if total_episode_reward > max_thread_score:
            self.session.run(self.update_thread_score_op, 
                feed_dict={self.thread_score_placeholder: total_episode_reward})
        summarize = False
        self.lock.acquire()
        max_global_score = self.session.run(self.global_score)
        if total_episode_reward > max_global_score:
            self.session.run(self.update_global_score_op, 
                feed_dict={self.global_score_placeholder: total_episode_reward})
            summary_str = self.session.run(self.global_score_summary)
            summarize = True
        self.lock.release()
        if summarize:
            self.summary_writer.add_summary(summary_str, self.global_step)
            self.summary_writer.flush()

    def clip_reward(self, reward):
        """ Clip immediate reward """
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0
        return reward

    def rescale_reward(self, reward):
        """ Rescale immediate reward by max reward encountered thus far. """
        if reward > self.max_reward:
            _, self.max_reward = self.session.run([self.update_max_reward_op, 
                self.thread_max_reward], 
                feed_dict={self.thread_max_reward_placeholder: reward})
        return reward/self.max_reward