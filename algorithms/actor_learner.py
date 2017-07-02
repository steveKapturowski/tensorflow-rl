# -*- encoding: utf-8 -*-
import gym
import time
import ctypes
import tempfile
import utils.logger
import multiprocessing
import tensorflow as tf
import numpy as np

from utils import checkpoint_utils
from utils.decorators import only_on_train
from utils.hogupdatemv import apply_grads_mom_rmsprop, apply_grads_adam, apply_grads_adamax
from contextlib import contextmanager
from multiprocessing import Process


CHECKPOINT_INTERVAL = 100000
ONE_LIFE_GAMES = [
    #Atari
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
    #Classic Control
    'CartPole-v0',
    'CartPole-v1',
    'Pendulum-v0',
    'MountainCar-v0',
    'Acrobot-v1',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
    #Box2D
    'LunarLander-v2',
    'LunarLanderContinuous-v2',
    'BipedalWalker-v2',
    'BipedalWalkerHardcore-v2',
    'CarRacing-v0',
    #MuJoCo
    'InvertedPendulum-v1',
    'IvertedDoublePendulum-v1',
    'Reacher-v1',
    'HalfCheetah-v1',
    'Swimmer-v1',
    'Hopper-v1',
    'Walker2d-v1',
    'Ant-v1',
    'Humanoid-v1',
    'HumanoidStandup-v1',
]
 
logger = utils.logger.getLogger('actor_learner')


class ActorLearner(Process):
    
    def __init__(self, args):
        super(ActorLearner, self).__init__()
       
        self.summ_base_dir = args.summ_base_dir
        
        self.local_step = 0
        self.global_step = args.global_step
        self.local_episode = 0
        self.last_saving_step = 0

        self.saver = None
        self.actor_id = args.actor_id
        self.alg_type = args.alg_type
        self.use_monitor = args.use_monitor
        self.max_local_steps = args.max_local_steps
        self.optimizer_type = args.opt_type
        self.optimizer_mode = args.opt_mode
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps
        self.num_actor_learners = args.num_actor_learners
        self.is_train = args.is_train
        self.input_shape = args.input_shape
        self.reward_clip_val = args.reward_clip_val
        self.q_update_interval = args.q_update_interval
        self.restore_checkpoint = args.restore_checkpoint
        self.random_seed = args.random_seed
        
        # Shared mem vars
        self.learning_vars = args.learning_vars
            
        if self.optimizer_mode == 'local':
            if self.optimizer_type == 'rmsprop':
                self.opt_st = np.ones(self.learning_vars.size, dtype=ctypes.c_float)
            else:
                self.opt_st = np.zeros(self.learning_vars.size, dtype=ctypes.c_float)
        elif self.optimizer_mode == 'shared':
                self.opt_st = args.opt_state

        # rmsprop/momentum
        self.alpha = args.momentum
        # adam
        self.b1 = args.b1
        self.b2 = args.b2
        self.e = args.e
        
        if args.env == 'GYM':
            from environments.atari_environment import AtariEnvironment
            self.emulator = AtariEnvironment(
                args.game,
                self.random_seed,
                args.visualize,
                use_rgb=args.use_rgb,
                frame_skip=args.frame_skip,
                agent_history_length=args.history_length,
                max_episode_steps=args.max_episode_steps,
                single_life_episodes=args.single_life_episodes,
            )
        elif args.env == 'ALE':
            from environments.emulator import Emulator
            self.emulator = Emulator(
                args.rom_path, 
                args.game, 
                args.visualize, 
                self.actor_id,
                self.random_seed,
                args.single_life_episodes)
        else:
            raise Exception('Invalid environment `{}`'.format(args.env))
            
        self.grads_update_steps = args.grads_update_steps
        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma

        self.rescale_rewards = args.rescale_rewards
        self.max_achieved_reward = -float('inf')
        if self.rescale_rewards:
            self.thread_max_reward = 1.0

        # Barrier to synchronize all actors after initialization is done
        self.barrier = args.barrier
        self.game = args.game

        # Initizlize Tensorboard summaries
        self.summary_ph, self.update_ops, self.summary_ops = self.setup_summaries()
        self.summary_op = tf.summary.merge_all()


    def compute_targets(self, rewards, R):
        size = len(rewards)
        y_batch = list()

        for i in reversed(xrange(size)):
            R = rewards[i] + self.gamma * R
            y_batch.append(R)

        y_batch.reverse()
        return y_batch


    def reset_hidden_state(self):
        """
        Override in subclass if needed
        """
        pass


    def is_master(self):
        return self.actor_id == 0


    def test(self, num_episodes=100):
        """
        Run test monitor for `num_episodes`
        """
        rewards = list()
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
                logger.info("EPISODE {0} -- REWARD: {1}, RUNNING AVG: {2:.1f}±{3:.1f}, BEST: {4}".format(
                    episode,
                    total_episode_reward,
                    np.array(rewards).mean(),
                    2*np.array(rewards).std(),
                    max(rewards),
                ))


    def synchronize_workers(self):
        if self.is_master():
            # Initialize network parameters
            g_step = checkpoint_utils.restore_vars(self.saver, self.session, self.game, self.alg_type, self.max_local_steps, self.restore_checkpoint)
            self.global_step.val.value = g_step
            self.last_saving_step = g_step   
            logger.debug("T{}: Initializing shared memory...".format(self.actor_id))
            self.update_shared_memory()

        # Wait until actor 0 finishes initializing shared memory
        self.barrier.wait()

        if not self.is_master():
            logger.debug("T{}: Syncing with shared memory...".format(self.actor_id))
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)  
            if hasattr(self, 'target_network'):
                self.sync_net_with_shared_memory(self.target_network, self.learning_vars)
            elif hasattr(self, 'batch_network'):
                self.sync_net_with_shared_memory(self.batch_network, self.learning_vars)

        # Ensure we don't add any more nodes to the graph
        self.session.graph.finalize()
        self.start_time = time.time()


    def get_gpu_options(self):
        return tf.GPUOptions(allow_growth=True)


    @contextmanager
    def monitored_environment(self):
        if self.use_monitor:
            self.log_dir = tempfile.mkdtemp()
            self.emulator.env = gym.wrappers.Monitor(self.emulator.env, self.log_dir)

        yield
        self.emulator.env.close()


    def run(self):
        #set random seeds so we can reproduce runs
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)

        num_cpus = multiprocessing.cpu_count()
        self.supervisor = tf.train.Supervisor(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.global_variables_initializer(),
            logdir=self.summ_base_dir,
            saver=self.saver,
            summary_op=None)
        session_context = self.supervisor.managed_session(config=tf.ConfigProto(
            intra_op_parallelism_threads=num_cpus,
            inter_op_parallelism_threads=num_cpus,
            gpu_options=self.get_gpu_options(),
            allow_soft_placement=True))

        with self.monitored_environment(), session_context as self.session:
            self.synchronize_workers()

            if self.is_train:
                self.train()
            else:
                self.test()


    def save_vars(self):
        if self.is_master() and self.global_step.value()-self.last_saving_step >= CHECKPOINT_INTERVAL:
            self.last_saving_step = self.global_step.value()
            checkpoint_utils.save_vars(self.saver, self.session, self.game, self.alg_type, self.max_local_steps, self.last_saving_step) 
    
    def update_shared_memory(self):
        # Initialize shared memory with tensorflow var values
        params = self.session.run(self.local_network.params)

        # Merge all param matrices into a single 1-D array
        params = np.hstack([p.reshape(-1) for p in params])
        np.frombuffer(self.learning_vars.vars, ctypes.c_float)[:] = params
        # if hasattr(self, 'target_vars'):
            # target_params = self.session.run(self.target_network.params)
            # np.frombuffer(self.target_vars.vars, ctypes.c_float)[:] = params
                
    
    @only_on_train(return_val=0.0)
    def decay_lr(self):
        if self.global_step.value() <= self.lr_annealing_steps:            
            return self.initial_lr - (self.global_step.value() * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

    def apply_gradients_to_shared_memory_vars(self, grads):
        self._apply_gradients_to_shared_memory_vars(grads, self.learning_vars)


    @only_on_train()
    def _apply_gradients_to_shared_memory_vars(self, grads, shared_vars):
            opt_st = self.opt_st
            self.flat_grads = np.empty(shared_vars.size, dtype=ctypes.c_float)

            #Flatten grads
            offset = 0
            for g in grads:
                self.flat_grads[offset:offset + g.size] = g.reshape(-1)
                offset += g.size
            g = self.flat_grads

            shared_vars.step.value += 1
            T = shared_vars.step.value

            if self.optimizer_type == "adam" and self.optimizer_mode == "shared":
                p = np.frombuffer(shared_vars.vars, ctypes.c_float)
                p_size = shared_vars.size
                m = np.frombuffer(opt_st.ms, ctypes.c_float)
                v = np.frombuffer(opt_st.vs, ctypes.c_float)
                opt_st.lr.value =  1.0 * opt_st.lr.value * (1 - self.b2**T)**0.5 / (1 - self.b1**T) 
                
                apply_grads_adam(m, v, g, p, p_size, opt_st.lr.value, self.b1, self.b2, self.e)

            elif self.optimizer_type == "adamax" and self.optimizer_mode == "shared":
                beta_1 = .9
                beta_2 = .999
                lr = opt_st.lr.value

                p = np.frombuffer(shared_vars.vars, ctypes.c_float)
                p_size = shared_vars.size
                m = np.frombuffer(opt_st.ms, ctypes.c_float)
                u = np.frombuffer(opt_st.vs, ctypes.c_float)

                apply_grads_adamax(m, u, g, p, p_size, lr, beta_1, beta_2, T)
                    
            else: #local or shared rmsprop/momentum
                lr = self.decay_lr()
                if (self.optimizer_mode == "local"):
                    m = opt_st
                else: #shared
                    m = np.frombuffer(opt_st.vars, ctypes.c_float)
                
                p = np.frombuffer(shared_vars.vars, ctypes.c_float)
                p_size = shared_vars.size
                _type = 0 if self.optimizer_type == "momentum" else 1
                
                apply_grads_mom_rmsprop(m, g, p, p_size, _type, lr, self.alpha, self.e)

    def rescale_reward(self, reward):
        if self.rescale_rewards:
            # Rescale immediate reward by max reward encountered thus far
            if np.abs(reward) > self.thread_max_reward:
                self.thread_max_reward = np.abs(reward)
            return reward/self.thread_max_reward
        else:
            # Clip immediate reward
            return np.sign(reward) * np.minimum(self.reward_clip_val, np.abs(reward))
            

    def assign_vars(self, dest_net, params):
        feed_dict = {}
        offset = 0

        for i, var in enumerate(dest_net.params):
            shape = var.get_shape().as_list()
            size = np.prod(shape)
            if type(params) == list:
                feed_dict[dest_net.params_ph[i]] = params[i]
            else:
                feed_dict[dest_net.params_ph[i]] = \
                    params[offset:offset+size].reshape(shape)
            offset += size
        
        self.session.run(dest_net.sync_with_shared_memory, 
            feed_dict=feed_dict)


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

    
    def _get_summary_vars(self):
        episode_reward = tf.Variable(0., name='episode_reward')
        s1 = tf.summary.scalar('Episode_Reward_{}'.format(self.actor_id), episode_reward)

        mean_value = tf.Variable(0., name='mean_value')
        s2 = tf.summary.scalar('Mean_Value_{}'.format(self.actor_id), mean_value)

        mean_entropy = tf.Variable(0., name='mean_entropy')
        s3 = tf.summary.scalar('Mean_Entropy_{}'.format(self.actor_id), mean_entropy)

        return [episode_reward, mean_value, mean_entropy]


    def setup_summaries(self):
        summary_vars = self._get_summary_vars()

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        with tf.control_dependencies(update_ops):
            summary_ops = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_ops


    @only_on_train()
    def log_summary(self, *args):
        if self.is_master():
            feed_dict = {ph: val for ph, val in zip(self.summary_ph, args)}
            summaries = self.session.run(self.update_ops + [self.summary_op], feed_dict=feed_dict)[-1]
            self.supervisor.summary_computed(self.session, summaries, global_step=self.global_step.value())
    

