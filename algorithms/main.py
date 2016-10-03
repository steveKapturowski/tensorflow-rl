from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import os
import numpy as np
from multiprocessing import Process, RawArray
import tensorflow as tf
import logging_utils
import time
import importlib
from value_based_actor_learner import *
from policy_based_actor_learner import *
import math
from shared_utils import SharedCounter, SharedVars, SharedFlags, Barrier
import ctypes
import argparse

logger = logging_utils.getLogger('main')


def bool_arg(string):
    value = string.lower
    if value == 'true': 
        return True
    elif value == 'false': 
        return False
    else:
        raise argparse.ArgumentTypeError('Expected True or False, but got {}'.format(string))
    
def get_learning_rate(low, high):
    """ Return LogUniform(low, high) learning rate. """
    lr = math.exp(random.uniform(math.log(low), math.log(high)))
    return lr

def get_num_actions(rom_path, rom_name):
    from ale_python_interface import ALEInterface
    filename = '{0}/{1}.bin'.format(rom_path, rom_name)
    ale = ALEInterface()
    ale.loadROM(filename)
    return len(ale.getMinimalActionSet())


def play(args):
    ckpt_dir = 'checkpoints/{0}/q_5_steps/'.format(args.game)
    ckpt = os.listdir(ckpt_dir)[-1].split('.meta')[0]


def main(args):
    logger.debug('CONFIGURATION: {}'.format(args))

    if not args.is_train:
        play(args)
        return

    
    """ Set up the graph, the agents, and run the agents in parallel. """
    if args.env == 'GYM':
        import atari_environment 
        num_actions = atari_environment.get_num_actions(args.game)
    else:
        num_actions = get_num_actions(args.rom_path, args.game)
    
    args.summ_base_dir = '/tmp/summary_logs/{}/{}'.format(args.game, time.time())

    if args.alg_type == 'q':
        if args.max_local_steps > 1:
            Learner = NStepQLearner
        else:
            Learner = OneStepQLearner
    elif args.alg_type == 'sarsa':
        if args.max_local_steps > 1:
            print('n-step SARSA not implemented!')
            sys.exit()
        else:
            Learner = OneStepSARSALearner
    else:
        Learner = A3CLearner

    T = SharedCounter(0)
    args.learning_vars = SharedVars(num_actions, args.alg_type, arch=args.arch, use_recurrent=args.use_recurrent)
    if args.opt_mode == 'shared':
        args.opt_state = SharedVars(num_actions, args.alg_type, arch=args.arch, use_recurrent=args.use_recurrent, opt_type=args.opt_type, lr=args.initial_lr)
    else:
        args.opt_state = None
    if args.alg_type in ['q', 'sarsa']:
        args.target_vars = SharedVars(num_actions, args.alg_type, arch=args.arch, use_recurrent=args.use_recurrent)
        args.target_update_flags = SharedFlags(args.num_actor_learners)
    
    args.barrier = Barrier(args.num_actor_learners)
    args.global_step = T
    args.num_actions = num_actions

#    vars_to_save = {'global_step': T, 
#                    'learning_vars': learning_vars,
#                    'target_vars': target_vars,
#                    'opt_state': opt_st}
#
#     saving = Process(target=save_shared_mem_vars, 
#            args=(vars_to_save, args.game, args.alg_type, 
#             args.max_local_steps))

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
    parser.add_argument('--env', default='ALE', help='Type of environment: ALE or GYM', dest='env')
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
    parser.add_argument('-ea', '--epsilon_annealing_steps', default=1000000, type=int, help='Nr. of global steps during which the exploration epsilon will be annealed', dest='epsilon_annealing_steps')
    parser.add_argument('--max_local_steps', default=5, type=int, help='Number of steps to gain experience from before every update for the Q learning/A3C algorithm', dest='max_local_steps')
    parser.add_argument('--rescale_rewards', default=False, type=bool_arg, help='If True, rewards will be rescaled (dividing by the max. possible reward) to be in the range [-1, 1]. If False, rewards will be clipped to be in the range [-1, 1]', dest='rescale_rewards')  
    parser.add_argument('--arch', default='NIPS', help='Which network architecture to use: from the NIPS or NATURE paper', dest='arch')
    parser.add_argument('--single_life_episodes', action='store_true', help='if true, training episodes will be terminated when a life is lost (for games)', dest='single_life_episodes')
    parser.add_argument('--use_recurrent', action='store_true', help='if true, use recurrent layer in a3c model', dest='use_recurrent')
    parser.add_argument('--train', default=True, type=bool_arg, help='if true train agents in parallel, otherwise follow optimal policy with single agent', dest='is_train')

    args = parser.parse_args()
    if (args.env=='ALE' and args.rom_path is None):
        raise argparse.ArgumentTypeError('Need to specify the directory where the game roms are located, via --rom_path')         
    
    main(args)
