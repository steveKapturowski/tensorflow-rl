import random
import sys
import os
import logging_utils
import time
import ctypes
import argparse
import numpy as np
import tensorflow as tf

from multiprocessing import Process, RawArray
from value_based_actor_learner import NStepQLearner, DuelingLearner, OneStepSARSALearner
from policy_based_actor_learner import A3CLearner, A3CLSTMLearner, ActionSequenceA3CLearner
from shared_utils import SharedCounter, SharedVars, SharedFlags, Barrier

logger = logging_utils.getLogger('main')


def get_learning_rate(low, high):
    """ Return LogUniform(low, high) learning rate. """
    lr = np.exp(random.uniform(np.log(low), np.log(high)))
    return lr

def get_num_actions(rom_path, rom_name):
    from ale_python_interface import ALEInterface
    filename = '{0}/{1}.bin'.format(rom_path, rom_name)
    ale = ALEInterface()
    ale.loadROM(filename)
    return len(ale.getMinimalActionSet())


def main(args):
    logger.debug('CONFIGURATION: {}'.format(args))
    
    """ Set up the graph, the agents, and run the agents in parallel. """
    if args.env == 'GYM':
        import atari_environment 
        num_actions = atari_environment.get_num_actions(args.game)
    else:
        num_actions = get_num_actions(args.rom_path, args.game)
    
    args.summ_base_dir = '/tmp/summary_logs/{}/{}'.format(args.game, time.time())

    algorithms = {
        'q': NStepQLearner,
        'sarsa': OneStepSARSALearner,
        'dueling': DuelingLearner,
        'a3c': A3CLearner,
        'a3c-lstm': A3CLSTMLearner,
        'a3c-sequence-decoder': ActionSequenceA3CLearner,
    }

    assert args.alg_type in algorithms, 'alg_type `{}` not implemented'.format(args.alg_type)
    Learner = algorithms[args.alg_type]

    T = SharedCounter(0)
    args.learning_vars = SharedVars(num_actions, args.alg_type, arch=args.arch)
    
    args.opt_state = SharedVars(
        num_actions, args.alg_type, arch=args.arch, opt_type=args.opt_type, lr=args.initial_lr
    ) if args.opt_mode == 'shared' else None

    if args.alg_type in ['q', 'sarsa', 'dueling']:
        args.target_vars = SharedVars(num_actions, args.alg_type, arch=args.arch)
        args.target_update_flags = SharedFlags(args.num_actor_learners)
    
    args.barrier = Barrier(args.num_actor_learners)
    args.global_step = T
    args.num_actions = num_actions


    if (args.visualize == 2): args.visualize = 0        
    actor_learners = []
    for i in xrange(args.num_actor_learners):
        if (args.visualize == 2) and (i == args.num_actor_learners - 1):
            args.args.visualize = 1

        args.actor_id = i
        
        rng = np.random.RandomState(int(time.time()))
        args.random_seed = rng.randint(1000)
            
        #pass in gpu name to learner here and wrap each learner in device context
        actor_learners.append(Learner(args))
        actor_learners[-1].start()


    for t in actor_learners:
        t.join()
    
    logger.debug('All training threads finished')
    logger.debug('All threads stopped')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('game', help='Name of game')
    parser.add_argument('--env', default='GYM', help='Type of environment: ALE or GYM', dest='env')
    parser.add_argument('--rom_path', help='Directory where the game roms are located (needed for ALE environment)', dest='rom_path')
    parser.add_argument('-v', '--visualize', default=0, type=int, help='0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized', dest='visualize')
    parser.add_argument('--opt_type', default='rmsprop', help='Type of optimizer: rmsprop, momentum, adam', dest='opt_type')
    parser.add_argument('--opt_mode', default='shared', help='Whether to use \"local\" or \"shared\" vector(s) for the momemtum/optimizer statistics', dest='opt_mode')
    parser.add_argument('--b1', default=0.9, type=float, help='Beta1 for the Adam optimizer', dest='b1')
    parser.add_argument('--b2', default=0.999, type=float, help='Beta2 for the Adam optimizer', dest='b2')
    parser.add_argument('--e', default=0.1, type=float, help='Epsilon for the Rmsprop and Adam optimizers', dest='e')
    parser.add_argument('--alpha', default=0.99, type=float, help='Discount factor for the history/coming gradient, for the Rmsprop optimizer', dest='alpha')
    parser.add_argument('-lr', '--initial_lr', default=0.001, type=float, help='Initial value for the learning rate. Default = LogUniform(10**-4, 10**-2)', dest='initial_lr')
    parser.add_argument('-lra', '--lr_annealing_steps', default=640000000, type=int, help='Nr. of global steps during which the learning rate will be linearly annealed towards zero', dest='lr_annealing_steps')
    parser.add_argument('--clip_loss', default=0.0, type=float, help='If bigger than 0.0, the loss will be clipped at +/-clip_loss', dest='clip_loss_delta')
    parser.add_argument('--entropy', default=0.01, type=float, help='Strength of the entropy regularization term (needed for actor-critic)', dest='entropy_regularisation_strength')
    parser.add_argument('--clip_norm', default=40, type=float, help='If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm', dest='clip_norm')
    parser.add_argument('--clip_norm_type', default='global', help='Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)', dest='clip_norm_type')
    parser.add_argument('--alg_type', default="a3c", help='Type of algorithm: q (for Q-learning), sarsa, a3c (for actor-critic)', dest='alg_type') 
    parser.add_argument('-n', '--num_actor_learners', default=16, type=int, help='number of actors (processes)', dest='num_actor_learners')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor', dest='gamma')
    parser.add_argument('--q_target_update_steps', default=10000, type=int, help='Interval (in nr. of global steps) at which the parameters of the Q target network are updated (obs! 1 step = 4 video frames) (needed for Q-learning and Sarsa)', dest='q_target_update_steps') 
    parser.add_argument('--grads_update_steps', default=10, type=int, help='Nr. of local steps during which grads are accumulated before applying them to the shared network parameters (needed for 1-step Q/Sarsa learning)', dest='grads_update_steps')
    parser.add_argument('--max_global_steps', default=640000000, type=int, help='Max. number of training steps', dest='max_global_steps')
    parser.add_argument('-ea', '--epsilon_annealing_steps', default=5000000, type=int, help='Nr. of global steps during which the exploration epsilon will be annealed', dest='epsilon_annealing_steps')
    parser.add_argument('--max_local_steps', default=5, type=int, help='Number of steps to gain experience from before every update for the Q learning/A3C algorithm', dest='max_local_steps')
    parser.add_argument('--max_decoder_steps', default=20, type=int, help='max number of steps that sequence decoder will be allowed to take', dest='max_decoder_steps')
    parser.add_argument('--rescale_rewards', action='store_true', help='If True, rewards will be rescaled (dividing by the max. possible reward) to be in the range [-1, 1]. If False, rewards will be clipped to be in the range [-1, 1]', dest='rescale_rewards')  
    parser.add_argument('--arch', default='NIPS', help='Which network architecture to use: from the NIPS or NATURE paper', dest='arch')
    parser.add_argument('--single_life_episodes', action='store_true', help='if true, training episodes will be terminated when a life is lost (for games)', dest='single_life_episodes')
    parser.add_argument('--frame_skip', default=[4], type=int, nargs='+', help='number of frames to repeat action', dest='frame_skip')
    parser.add_argument('--test', action='store_false', help='if not set train agents in parallel, otherwise follow optimal policy with single agent', dest='is_train')

    args = parser.parse_args()
    if (args.env=='ALE' and args.rom_path is None):
        raise argparse.ArgumentTypeError('Need to specify the directory where the game roms are located, via --rom_path')         
    
    # fix up frame_skip depending on whether it was an int or tuple 
    if len(args.frame_skip) == 1:
        args.frame_skip = args.frame_skip[0]
    elif len(args.frame_skip) > 2:
        raise TypeError('Expected tuple of length two or int for param `frame_skip`')


    main(args)

