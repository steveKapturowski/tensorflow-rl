# -*- encoding: utf-8 -*-
import numpy as np
import utils.logger
import tensorflow as tf
import ctypes
import pyximport; pyximport.install()
import time
import checkpoint_utils
import tempfile
from multiprocessing import Process 
from hogupdatemv import copy, apply_grads_mom_rmsprop, apply_grads_adam


CHECKPOINT_INTERVAL = 100000
ONE_LIFE_GAMES = [
    'Bowling-v0',
    'Boxing-v0',
    'Carnival-v0',
    'DoubleDunk-v0',
    'Enduro-v0',
    'FishingDerby-v0',
    'Freeway-v0',
    'IceHockey-v0',
    'JourneyEscape-v0',
    'Pong-v0',
    'PrivateEye-v0',
    'Skiing-v0',
    'Tennis-v0',
]
 
logger = utils.logger.getLogger('actor_learner')

def generate_final_epsilon():
    """ Generate lower limit for decaying epsilon. """
    epsilon = {'limits': [0.1, 0.01, 0.5], 'probs': [0.4, 0.3, 0.3]}
    return np.random.choice(epsilon['limits'], p=epsilon['probs']) 


class ActorLearner(Process):
    
    def __init__(self, args):
        
        super(ActorLearner, self).__init__()
       
        self.summ_base_dir = args.summ_base_dir
        
        self.local_step = 0
        self.global_step = args.global_step
        self.last_saving_step = 0

        self.actor_id = args.actor_id
        self.alg_type = args.alg_type
        self.max_local_steps = args.max_local_steps
        self.optimizer_type = args.opt_type
        self.optimizer_mode = args.opt_mode
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps
        self.num_actor_learners = args.num_actor_learners
        self.is_train = args.is_train
        
        # Shared mem vars
        self.learning_vars = args.learning_vars
        size = self.learning_vars.size
        self.flat_grads = np.empty(size, dtype = ctypes.c_float)
            
        if (self.optimizer_mode == "local"):
            if (self.optimizer_type == "rmsprop"):
                self.opt_st = np.ones(size, dtype = ctypes.c_float)
            else:
                self.opt_st = np.zeros(size, dtype = ctypes.c_float)
        elif (self.optimizer_mode == "shared"):
                self.opt_st = args.opt_state

        # rmsprop/momentum
        self.alpha = args.alpha
        # adam
        self.b1 = args.b1
        self.b2 = args.b2
        self.e = args.e
        
        if args.env == "GYM":
            from atari_environment import AtariEnvironment
            self.emulator = AtariEnvironment(
                args.game,
                args.visualize,
                frame_skip=args.frame_skip,
                single_life_episodes=args.single_life_episodes,
            )
        else:
            from emulator import Emulator
            self.emulator = Emulator(
                args.rom_path, 
                args.game, 
                args.visualize, 
                self.actor_id,
                args.random_seed,
                args.single_life_episodes)
            
        self.grads_update_steps = args.grads_update_steps
        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma

        # Exploration epsilons 
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = generate_final_epsilon()
        self.epsilon_annealing_steps = args.epsilon_annealing_steps

        self.rescale_rewards = args.rescale_rewards
        self.max_achieved_reward = -1000000
        if self.rescale_rewards:
            self.thread_max_reward = 1.0

        # Barrier to synchronize all actors after initialization is done
        self.barrier = args.barrier
        
        self.summary_ph, self.update_ops, self.summary_ops = self.setup_summaries()
        self.game = args.game
        

    def reset_hidden_state(self):
        '''
        Override in subclass if needed
        '''
        pass


    def test(self, num_episodes=100):
        '''
        Run test monitor for `num_episodes`
        '''
        log_dir = tempfile.mkdtemp()
        self.emulator.env.monitor.start(log_dir)
        self.sync_net_with_shared_memory(self.local_network, self.learning_vars)

        rewards = list()
        logger.info('writing monitor log to {}'.format(log_dir))
        for episode in range(num_episodes):
            s = self.emulator.get_initial_state()
            self.reset_hidden_state()
            total_episode_reward = 0
            episode_over = False

            while not episode_over:
                a = self.choose_next_action(s)[0]
                s, reward, episode_over = self.emulator.next(a)

                total_episode_reward += reward

            else:
                rewards.append(total_episode_reward)
                logger.info("EPISODE {0} -- REWARD: {1}, RUNNING AVG: {2:.0f}±{3:.0f}, BEST: {4}".format(
                    episode,
                    total_episode_reward,
                    np.array(rewards).mean(),
                    2*np.array(rewards).std(),
                    max(rewards),
                ))

        self.emulator.env.monitor.close()


    def run(self):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.99/self.num_actor_learners)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


        if (self.actor_id==0):
            #Initizlize Tensorboard summaries
            self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.train.SummaryWriter(
                            "{}/{}".format(self.summ_base_dir, self.actor_id), self.session.graph) 

            # Initialize network parameters
            g_step = checkpoint_utils.restore_vars(self.saver, self.session, self.game, self.alg_type, self.max_local_steps)
            self.global_step.val.value = g_step
            self.last_saving_step = g_step   
            logger.debug("T{}: Initializing shared memory...".format(self.actor_id))
            self.init_shared_memory()

        # Wait until actor 0 finishes initializing shared memory
        self.barrier.wait()
        
        if self.actor_id > 0:
            logger.debug("T{}: Syncing with shared memory...".format(self.actor_id))
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)  
            if hasattr(self, 'target_vars'):
                self.sync_net_with_shared_memory(self.target_network, self.target_vars)

        # Wait until all actors are ready to start 
        self.barrier.wait()
        
        # Introduce a different start delay for each actor, so that they do not run in synchronism.
        # This is to avoid concurrent updates of parameters as much as possible 
        time.sleep(0.1877 * self.actor_id)

    def save_vars(self):
        if self.global_step.value()-self.last_saving_step >= CHECKPOINT_INTERVAL:
            self.last_saving_step = self.global_step.value()
            checkpoint_utils.save_vars(self.saver, self.session, self.game, self.alg_type, self.max_local_steps, self.last_saving_step) 
    
    def init_shared_memory(self):
        # Initialize shared memory with tensorflow var values
        params = self.session.run(self.local_network.params)   
     
        # Merge all param matrices into a single 1-D array
        params = np.hstack([p.reshape(-1) for p in params])
        np.frombuffer(self.learning_vars.vars, ctypes.c_float)[:] = params
        if hasattr(self, 'target_vars'):
            np.frombuffer(self.target_vars.vars, ctypes.c_float)[:] = params
        #memoryview(self.learning_vars.vars)[:] = params
        #memoryview(self.target_vars.vars)[:] = memoryview(self.learning_vars.vars)
    
    def reduce_thread_epsilon(self):
        """ Linear annealing """
        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_annealing_steps
            
    
    @checkpoint_utils.only_on_train(return_val=0.0)
    def decay_lr(self):
        if self.global_step.value() <= self.lr_annealing_steps:            
            return self.initial_lr - (self.global_step.value() * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

    def apply_gradients_to_shared_memory_vars(self, grads):
        self._apply_gradients_to_shared_memory_vars(grads, self.opt_st)


    @checkpoint_utils.only_on_train()
    def _apply_gradients_to_shared_memory_vars(self, grads, opt_st):
            #Flatten grads
            offset = 0
            for g in grads:
                self.flat_grads[offset:offset + g.size] = g.reshape(-1)
                offset += g.size
            g = self.flat_grads
            
            if self.optimizer_type == "adam" and self.optimizer_mode == "shared":
                p = np.frombuffer(self.learning_vars.vars, ctypes.c_float)
                p_size = self.learning_vars.size
                m = np.frombuffer(opt_st.ms, ctypes.c_float)
                v = np.frombuffer(opt_st.vs, ctypes.c_float)
                T = self.global_step.value() 
                opt_st.lr.value =  1.0 * opt_st.lr.value * (1 - self.b2**T)**0.5 / (1 - self.b1**T) 
                
                apply_grads_adam(m, v, g, p, p_size, opt_st.lr.value, self.b1, self.b2, self.e)
                    
            else: #local or shared rmsprop/momentum
                lr = self.decay_lr()
                if (self.optimizer_mode == "local"):
                    m = opt_st
                else: #shared 
                    m = np.frombuffer(opt_st.vars, ctypes.c_float)
                
                p = np.frombuffer(self.learning_vars.vars, ctypes.c_float)
                p_size = self.learning_vars.size
                _type = 0 if self.optimizer_type == "momentum" else 1
                    
                #print "BEFORE", "RMSPROP m", m[0], "GRAD", g[0], self.flat_grads[0], self.flat_grads2[0]
                apply_grads_mom_rmsprop(m, g, p, p_size, _type, lr, self.alpha, self.e)
                #print "AFTER", "RMSPROP m", m[0], "GRAD", g[0], self.flat_grads[0], self.flat_grads2[0]

    def rescale_reward(self, reward):
        if self.rescale_rewards:
            """ Rescale immediate reward by max reward encountered thus far. """
            if reward > self.thread_max_reward:
                self.thread_max_reward = reward
            return reward/self.thread_max_reward
        else:
            """ Clip immediate reward """
            if reward > 1.0:
                reward = 1.0
            elif reward < -1.0:
                reward = -1.0
            return reward
            

    def sync_net_with_shared_memory(self, dest_net, shared_mem_vars):
        feed_dict = {}
        offset = 0
        params = np.frombuffer(shared_mem_vars.vars, 
                                  ctypes.c_float)
        for i in xrange(len(dest_net.params)):
            shape = shared_mem_vars.var_shapes[i]
            size = np.prod(shape)
            feed_dict[dest_net.params_ph[i]] = \
                    params[offset:offset+size].reshape(shape)
            offset += size
        
        self.session.run(dest_net.sync_with_shared_memory, 
                feed_dict=feed_dict)

    
    def setup_summaries(self):
        episode_reward = tf.Variable(0.)
        s1 = tf.scalar_summary("Episode Reward " + str(self.actor_id), episode_reward)
        if not hasattr(self, 'target_vars'):
            summary_vars = [episode_reward]
        else:
            episode_ave_max_q = tf.Variable(0.)
            s2 = tf.scalar_summary("Max Q Value " + str(self.actor_id), episode_ave_max_q)
            logged_epsilon = tf.Variable(0.)
            s3 = tf.scalar_summary("Epsilon " + str(self.actor_id), logged_epsilon)
            summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]

        summary_placeholders = [tf.placeholder("float") for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        with tf.control_dependencies(update_ops):
            summary_ops = tf.merge_all_summaries()
        return summary_placeholders, update_ops, summary_ops
    
