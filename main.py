# -*- coding: utf-8 -*-
import os
import sys
import time
import yaml
import argparse
import numpy as np
import utils.logger
import multiprocessing
import tensorflow as tf

from networks.q_network import QNetwork
from multiprocessing import Process, Queue
from networks.dueling_network import DuelingNetwork
from networks.continuous_actions import ContinuousPolicyNetwork, ContinuousPolicyValueNetwork
from networks.policy_v_network import PolicyNetwork, PolicyValueNetwork, PolicyRepeatNetwork, SequencePolicyVNetwork
from utils.shared_memory import SharedCounter, SharedVars, SharedFlags, Barrier
from algorithms.policy_based_actor_learner import A3CLearner, A3CLSTMLearner
from algorithms.sequence_decoder_actor_learner import ActionSequenceA3CLearner, ARA3CLearner
from algorithms.value_based_actor_learner import NStepQLearner, DuelingLearner, OneStepSARSALearner
from algorithms.intrinsic_motivation_actor_learner import PseudoCountA3CLearner, PseudoCountA3CLSTMLearner, PseudoCountQLearner
from algorithms.trpo_actor_learner import TRPOLearner
from algorithms.pgq_actor_learner import PGQLearner
from algorithms.cem_actor_learner import CEMLearner

logger = utils.logger.getLogger('main')


ALGORITHMS = {
    'q': (NStepQLearner, QNetwork),
    'sarsa': (OneStepSARSALearner, QNetwork),
    'dueling': (DuelingLearner, DuelingNetwork),
    'a3c': (A3CLearner, PolicyValueNetwork),
    'a3c-lstm': (A3CLSTMLearner, PolicyValueNetwork),
    'a3c-sequence-decoder': (ActionSequenceA3CLearner, SequencePolicyVNetwork),
    'pgq': (PGQLearner, PolicyValueNetwork),
    'trpo': (TRPOLearner, PolicyNetwork),
    'cem': (CEMLearner, PolicyNetwork),
    'dqn-cts': (PseudoCountQLearner, QNetwork),
    'a3c-cts': (PseudoCountA3CLearner, PolicyValueNetwork),
    'a3c-lstm-cts': (PseudoCountA3CLSTMLearner, PolicyValueNetwork),
    'a3c-repeat': (ARA3CLearner, PolicyRepeatNetwork),
    'a3c-continuous': (A3CLearner, ContinuousPolicyValueNetwork),
    'a3c-lstm-continuous': (A3CLSTMLearner, ContinuousPolicyValueNetwork),
    'cem-continuous': (CEMLearner, ContinuousPolicyNetwork),
    'trpo-continuous': (TRPOLearner, ContinuousPolicyNetwork),
}

def get_num_actions(rom_path, rom_name):
    from ale_python_interface import ALEInterface
    filename = '{0}/{1}.bin'.format(rom_path, rom_name)
    ale = ALEInterface()
    ale.loadROM(filename)
    return len(ale.getMinimalActionSet())

def main(args):
    args.batch_size = None
    logger.debug('CONFIGURATION: {}'.format(args))
    
    """ Set up the graph, the agents, and run the agents in parallel. """
    if args.env == 'GYM':
        from environments import atari_environment
        num_actions, action_space, _ = atari_environment.get_actions(args.game)
        input_shape = atari_environment.get_input_shape(args.game)
    else:
        num_actions = get_num_actions(args.rom_path, args.game)
    
    args.action_space = action_space
    args.summ_base_dir = '/tmp/summary_logs/{}/{}'.format(args.game, time.strftime('%m.%d/%H.%M'))
    logger.info('logging summaries to {}'.format(args.summ_base_dir))

    Learner, Network = ALGORITHMS[args.alg_type]
    args.network = Network
    
    # if args.alg_type.endswith('cts'):
    #     args.density_model_update_flags = SharedFlags(args.num_actor_learners)

    args.barrier = Barrier(args.num_actor_learners)
    args.global_step = SharedCounter(0)
    args.num_actions = num_actions

    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    num_gpus = 0
    if cuda_visible_devices:
        num_gpus = len(cuda_visible_devices.split())


    seed = args.seed or np.random.randint(2**32)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    args.actor_id = 0
    args.device = '/gpu:{}'.format(args.task_index % num_gpus) if num_gpus else '/cpu:0'
    args.random_seed = seed + args.task_index
    args.input_shape = input_shape


    cluster = tf.train.ClusterSpec({
        'ps': ['localhost:2048'],
        'worker': ['localhost:{}'.format(4096+i)
            for i in range(args.num_actor_learners)]})
    server = tf.train.Server(
        cluster,
        job_name=args.job_name,
        task_index=args.task_index)


    if args.job_name == 'ps':
        server.join()
    elif args.job_name == 'worker':

        with tf.device(tf.train.replica_device_setter(
            worker_device='/job:worker/task:{}'.format(args.task_index),
            # ps_device='/job:localhost/task:0',
            cluster=cluster)):

            learner = Learner(args)
            learner.run(server.target)


def get_validated_params(args):
    #validate param
    if args.env=='ALE' and args.rom_path is None:
        raise argparse.ArgumentTypeError('Need to specify the directory where the game roms are located, via --rom_path')         
    if args.reward_clip_val <= 0:
        raise argparse.ArgumentTypeError('value of --reward_clip_val option must be non-negative')
    if args.alg_type not in ALGORITHMS:
        raise argparse.ArgumentTypeError('alg_type `{}` not implemented'.format(args.alg_type))

    # fix up frame_skip depending on whether it was an int or tuple 
    if len(args.frame_skip) == 1:
        args.frame_skip = args.frame_skip[0]
    elif len(args.frame_skip) > 2:
        raise argparse.ArgumentTypeError('Expected tuple of length two or int for param `frame_skip`')

    return args


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', default=None, help='load config defaults from yaml file', dest='config_path')
    args, unprocessed_args = parser.parse_known_args()

    #override defaults
    parser.add_argument('game', help='Name of game')
    parser.add_argument('--alg_type', default="a3c", help='Type of algorithm: q (for Q-learning), sarsa, a3c (for actor-critic)', dest='alg_type')
    parser.add_argument('--arch', default='NIPS', help='Which network architecture to use: NIPS, NATURE, ATARI-TRPO, or FC (fully connected)', dest='arch')
    parser.add_argument('--env', default='GYM', help='Type of environment: ALE or GYM', dest='env')
    parser.add_argument('--rom_path', help='Directory where the game roms are located (needed for ALE environment)', dest='rom_path')
    parser.add_argument('-n', '--num_actor_learners', default=8, type=int, help='number of actors (processes)', dest='num_actor_learners')
    parser.add_argument('-v', '--visualize', default=0, type=int, help='0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized', dest='visualize')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor', dest='gamma')
    parser.add_argument('--frame_skip', default=[4], type=int, nargs='+', help='number of frames to repeat action', dest='frame_skip')
    parser.add_argument('--history_length', default=4, type=int, help='number of frames to stack as input state', dest='history_length')
    parser.add_argument('--single_life_episodes', action='store_true', help='if true, training episodes will be terminated when a life is lost (for games)', dest='single_life_episodes')
    parser.add_argument('--max_decoder_steps', default=20, type=int, help='max number of steps that sequence decoder will be allowed to take', dest='max_decoder_steps')
    parser.add_argument('--test', action='store_false', help='if not set train agents in parallel, otherwise follow optimal policy with single agent', dest='is_train')
    parser.add_argument('--restore_checkpoint', action='store_true', help='resume training from last checkpoint', dest='restore_checkpoint')
    parser.add_argument('--use_monitor', action='store_true', help='Record video / episode stats if set', dest='use_monitor')
    parser.add_argument('--pgq_fraction', default=0.5, type=float, help='fraction by which to multiply q gradients', dest='pgq_fraction')
    parser.add_argument('--activation', default='relu', type=str, help='specify relu, softplus, or tanh activations', dest='activation')
    parser.add_argument('--use_rgb', action='store_true', help='If set use rgb image channels instead of stacked luninance frames', dest='use_rgb')
    parser.add_argument('--no_share_weights', action='store_false', help='If set don\'t share parameters between policy and value function', dest='share_encoder_weights')
    parser.add_argument('--fc_layer_sizes', default=[60, 60], type=int, nargs='+', help='width of layers in fully connected architecture', dest='fc_layer_sizes')
    parser.add_argument('--seed', default=None, type=int, help='Specify random seed. Each process will get its own unique seed computed as seed+actor_id. Due to race conditions only 1 worker process should be used to get deterministic results', dest='seed')
    parser.add_argument('--job_name', default='ps', type=str, help='job name in cluster spec', dest='job_name')
    parser.add_argument('--task_index', default=0, type=int, help='index of task in cluster spec', dest='task_index')

    #optimizer args
    parser.add_argument('--opt_type', default='rmsprop', help='Type of optimizer: rmsprop, momentum, adam, adamax', dest='opt_type')
    parser.add_argument('--opt_mode', default='shared', help='Whether to use \"local\" or \"shared\" vector(s) for the momemtum/optimizer statistics', dest='opt_mode')
    parser.add_argument('--b1', default=0.9, type=float, help='Beta1 for the Adam optimizer', dest='b1')
    parser.add_argument('--b2', default=0.999, type=float, help='Beta2 for the Adam optimizer', dest='b2')
    parser.add_argument('--e', default=0.1, type=float, help='Epsilon for the Rmsprop and Adam optimizers', dest='e')
    parser.add_argument('--alpha', default=0.99, type=float, help='Discount factor for the history/coming gradient, for the Rmsprop optimizer', dest='alpha')
    parser.add_argument('-lr', '--initial_lr', default=0.001, type=float, help='Initial value for the learning rate. Default = LogUniform(10**-4, 10**-2)', dest='initial_lr')
    parser.add_argument('-lra', '--lr_annealing_steps', default=200000000, type=int, help='Nr. of global steps during which the learning rate will be linearly annealed towards zero', dest='lr_annealing_steps')
    parser.add_argument('--max_episode_steps', default=None, type=int, help='max rollout steps per trpo episode', dest='max_episode_steps')

    #clipping args
    parser.add_argument('--clip_loss', default=0.0, type=float, help='If bigger than 0.0, the loss will be clipped at +/-clip_loss', dest='clip_loss_delta')
    parser.add_argument('--clip_norm', default=40, type=float, help='If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm', dest='clip_norm')
    parser.add_argument('--clip_norm_type', default='global', help='Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)', dest='clip_norm_type')
    parser.add_argument('--rescale_rewards', action='store_true', help='If True, rewards will be rescaled (dividing by the max. possible reward) to be in the range [-1, 1]. If False, rewards will be clipped to be in the range [-REWARD_CLIP, REWARD_CLIP]', dest='rescale_rewards')
    parser.add_argument('--reward_clip_val', default=1.0, type=float, help='Clip rewards outside of [-REWARD_CLIP, REWARD_CLIP]', dest='reward_clip_val')
    
    #q-learning args
    parser.add_argument('-ea', '--epsilon_annealing_steps', default=1000000, type=int, help='Nr. of global steps during which the exploration epsilon will be annealed', dest='epsilon_annealing_steps')
    parser.add_argument('--final_epsilon', default=0.1, type=float, help='Final epsilon after annealing is complete. Only used for dqn-cts', dest='final_epsilon')
    parser.add_argument('--grads_update_steps', default=5, type=int, help='Nr. of local steps during which grads are accumulated before applying them to the shared network parameters (needed for 1-step Q/Sarsa learning)', dest='grads_update_steps')
    parser.add_argument('--q_target_update_steps', default=10000, type=int, help='Interval (in nr. of global steps) at which the parameters of the Q target network are updated (obs! 1 step = 4 video frames) (needed for Q-learning and Sarsa)', dest='q_target_update_steps') 
    parser.add_argument('--replay_size', default=100000, type=int, help='Maximum capacity of replay memory', dest='replay_size')
    parser.add_argument('--batch_update_size', default=32, type=int, help='Minibatch size for q-learning updates', dest='batch_update_size')
    parser.add_argument('--exploration_strategy', default='epsilon-greedy', type=str, help='boltzmann or epsilon-greedy', dest='exploration_strategy')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature to use for boltzmann exploration', dest='bolzmann_temperature')

    #a3c args 
    parser.add_argument('--entropy', default=0.01, type=float, help='Strength of the entropy regularization term (needed for actor-critic)', dest='entropy_regularisation_strength')
    parser.add_argument('--max_global_steps', default=200000000, type=int, help='Max. number of training steps', dest='max_global_steps')
    parser.add_argument('--max_local_steps', default=5, type=int, help='Number of steps to gain experience from before every update for the Q learning/A3C algorithm', dest='max_local_steps')
    
    #trpo args
    parser.add_argument('--num_epochs', default=1000, type=int, help='number of epochs for which to run TRPO', dest='num_epochs')
    parser.add_argument('--episodes_per_batch', default=50, type=int, help='number of episodes to batch for TRPO updates', dest='episodes_per_batch')
    parser.add_argument('--trpo_max_rollout', default=1000, type=int, help='max rollout steps per trpo episode', dest='max_rollout')
    parser.add_argument('--cg_subsample', default=0.1, type=float, help='rate at which to subsample data for TRPO conjugate gradient iteration', dest='cg_subsample')
    parser.add_argument('--cg_damping', default=0.001, type=float, help='conjugate gradient damping weight', dest='cg_damping')   
    parser.add_argument('--max_kl', default=0.01, type=float, help='max kl divergence for TRPO updates', dest='max_kl')
    parser.add_argument('--td_lambda', default=1.0, type=float, help='lambda parameter for GAE', dest='td_lambda')
    
    #cts args
    parser.add_argument('--cts_bins', default=8, type=int, help='number of bins to assign pixel values', dest='cts_bins')
    parser.add_argument('--cts_rescale_dim', default=42, type=int, help='rescaled image size to use with cts density model', dest='cts_rescale_dim')
    parser.add_argument('--cts_beta', default=.05, type=float, help='weight by which to scale novelty bonuses', dest='cts_beta')
    parser.add_argument('--cts_eta', default=.9, type=float, help='mixing param between 1-step TD-Error and Monte-Carlo Error', dest='cts_eta')
    parser.add_argument('--density_model', default='cts', type=str, help='density model to use for generating novelty bonuses: cts, or pixel-counts', dest='density_model')
    parser.add_argument('--q_update_interval', default=4, type=int, help='Number of steps between successive batch q-learning updates', dest='q_update_interval')

    if args.config_path:
        with open(args.config_path, 'r') as f:
            parser.set_defaults(**yaml.load(f))

    args = parser.parse_args(unprocessed_args)
    return get_validated_params(args)


if __name__ == '__main__':
    main(get_config())



 