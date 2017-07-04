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
from environments.atari_environment import AtariEnvironment
from contextlib import contextmanager


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


class ActorLearner(object):
    
    def __init__(self, args):
        super(ActorLearner, self).__init__()
       
        self.summ_base_dir = args.summ_base_dir
        
        self.local_step = 0
        self.local_episode = 0
        self.last_saving_step = 0

        self.saver = None
        self.task_index = args.task_index
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
        self.game = args.game
        
        # rmsprop/momentum
        self.alpha = args.alpha
        # adam
        self.b1 = args.b1
        self.b2 = args.b2
        self.e = args.e
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

        self.grads_update_steps = args.grads_update_steps
        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma

        self.rescale_rewards = args.rescale_rewards
        self.max_achieved_reward = -float('inf')
        if self.rescale_rewards:
            self.thread_max_reward = 1.0

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
        return self.task_index == 0


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


    def _build_optimizer(self):
        with tf.variable_scope('optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.global_step_increment = self.global_step.assign_add(1, use_locking=True)
            self.learning_rate = tf.train.polynomial_decay(
                self.initial_lr,
                self.global_step,
                self.max_global_steps,
                end_learning_rate=1e-6)
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate,
                decay=self.alpha,
                momentum=0.0,
                epsilon=1e-1,
                use_locking=False,
                centered=False,
                name='RMSProp')
            # optimizer = tf.train.SyncReplicasOptimizer(
            #     optimizer,
            #     replicas_to_aggregate=self.num_actor_learners,
            #     total_num_replicas=self.num_actor_learners)

            gradients = [
                e[0] for e in self.optimizer.compute_gradients(
                self.local_network.loss, self.local_network.params)]
            gradients = self.local_network._clip_grads(gradients)

            self.local_network.get_gradients = tf.group(*[
                g.assign_add(l, use_locking=False)
                for g, l in zip(self.global_network.params, self.local_network.params)
            ])
            # self.local_network.get_gradients = self.optimizer.apply_gradients(
            #     zip(gradients, self.local_network.params))


    def get_gpu_options(self):
        return tf.GPUOptions(allow_growth=True)


    @contextmanager
    def monitored_environment(self):
        if self.use_monitor:
            self.log_dir = tempfile.mkdtemp()
            self.emulator.env = gym.wrappers.Monitor(self.emulator.env, self.log_dir)

        yield
        self.emulator.env.close()


    def run(self, target):
        #set random seeds so we can reproduce runs
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)
        num_cpus = multiprocessing.cpu_count()

        self._build_optimizer()
        # local_init_op = self.optimizer.chief_init_op if self.is_master() else optimizer.local_step_init_op
        # chief_queue_runner = self.optimizer.get_chief_queue_runner()
        # sync_init_op = self.optimizer.get_init_tokens_op()

        self.supervisor = tf.train.Supervisor(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.global_variables_initializer(),
            # local_init_op=local_init_op,
            # ready_for_local_init_op=self.optimizer.ready_for_local_init_op,
            global_step=self.global_step,
            is_chief=self.is_master(),
            logdir=self.summ_base_dir,
            save_model_secs=3600,
            saver=self.saver,
            summary_op=None)
        session_context = self.supervisor.prepare_or_wait_for_session(target, config=tf.ConfigProto(
            intra_op_parallelism_threads=num_cpus,
            inter_op_parallelism_threads=num_cpus,
            gpu_options=self.get_gpu_options(),
            allow_soft_placement=True))

        with self.monitored_environment(), session_context as self.session:
            # if self.is_master():
            #     self.session.run(sync_init_op)
            #     self.supervisor.start_queue_runners(self.session, [chief_queue_runner])

            self.session.graph.finalize()
            self.start_time = time.time()

            if self.is_train:
                self.train()
            else:
                self.test()


    def apply_gradients_to_shared_memory_vars(self, grads):
        pass
        # self._apply_gradients_to_shared_memory_vars(grads, self.learning_vars)


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

    
    def _get_summary_vars(self):
        episode_reward = tf.Variable(0., trainable=False, name='episode_reward')
        s1 = tf.summary.scalar('Episode_Reward_{}'.format(self.actor_id), episode_reward)

        mean_value = tf.Variable(0., trainable=False, name='mean_value')
        s2 = tf.summary.scalar('Mean_Value_{}'.format(self.actor_id), mean_value)

        mean_entropy = tf.Variable(0., trainable=False, name='mean_entropy')
        s3 = tf.summary.scalar('Mean_Entropy_{}'.format(self.actor_id), mean_entropy)

        return [episode_reward, mean_value, mean_entropy]


    def setup_summaries(self):
        with tf.variable_scope('summaries'):
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
            self.supervisor.summary_computed(self.session, summaries, global_step=self.global_step.eval(self.session))
    

