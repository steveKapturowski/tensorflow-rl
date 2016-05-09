from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import os
import numpy as np
from threading import Thread, Lock
import tensorflow as tf
import logging_utils
import time
from emulator import get_num_actions
import importlib
from q_network import *
from policy_v_network import *
from value_based_actor_learner import *
from policy_based_actor_learner import *
import math

logger = logging_utils.getLogger('main')

def generate_epsilon():
    """ Generate lower limit for decaying epsilon. """
    epsilon = {'limits': [0.1, 0.01, 0.5], 'probs': [0.4, 0.3, 0.3]}
    return np.random.choice(epsilon['limits'], p=epsilon['probs']) 

def check_or_create_checkpoint_dir(checkpoint_dir):
    """ Create checkpoint directory if it does not exist """
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

def restore_vars(saver, sess, game_name, actor_learner_type, 
    actor_learner_max_local_steps):
    """ Restore saved net, global score and step, and epsilons OR 
    create checkpoint directory for later storage. """
    sess.run(tf.initialize_all_variables())
    
    checkpoint_dir = 'checkpoints/' + game_name + '/' + \
        {'0': 'Q/', '1': 'sar''sa/', '2': 'a3c/'}[str(actor_learner_type)] + \
        str(actor_learner_max_local_steps) + '_step' + '/'
    
    check_or_create_checkpoint_dir(checkpoint_dir)
    path = tf.train.latest_checkpoint(checkpoint_dir)
    if path is None:
        return False
    else:
        saver.restore(sess, path)
        return True

def save_vars(saver, sess, game_name, actor_learner_type, 
    actor_learner_max_local_steps, global_step, coord):
    """ Checkpoint shared net params, global score and step, and epsilons. """
    checkpoint_dir = 'checkpoints/' + game_name + '/' + \
        {'0': 'Q/', '1': 'sarsa/', '2': 'a3c/'}[str(actor_learner_type)] + \
        str(actor_learner_max_local_steps) + '_step' + '/'
    
    check_or_create_checkpoint_dir(checkpoint_dir)
    while not coord.should_stop():
        step = sess.run(global_step)
        if step % 10000 == 0:
            saver.save(sess, checkpoint_dir + 'net-score-step-epsilons', 
                global_step=step)

def get_learning_rate(low, high):
    """ Return LogUniform(low, high) learning rate. """
    lr = math.exp(random.uniform(math.log(low), math.log(high)))
    return lr

def main(optimizer_conf, emulator_conf, alg_conf):
    """ Set up the graph, the agents, and run the agents in parallel. """
    num_actions = len(get_num_actions(emulator_conf["rom_path"], 
        emulator_conf["game"]))
    local_replicas = alg_conf["local_replicas"]
    num_actor_learners = alg_conf["num_actor_learners"]
    actor_learner_type = alg_conf['actor_learner_type']
    actor_learner_max_local_steps = alg_conf['max_local_steps']
    rescale_rewards = alg_conf['rescale_rewards']

    if actor_learner_type == 0:
        Net = QNetwork
        if actor_learner_max_local_steps > 1:
            Learner = NStepQLearner
        else:
            Learner = OneStepQLearner
    elif actor_learner_type == 1:
        Net = QNetwork
        if actor_learner_max_local_steps > 1:
            print("n-step SARSA not implemented!")
            sys.exit()
        else:
            Learner = OneStepSARSALearner
    elif actor_learner_type == 2:
        Net = PolicyVNetwork
        Learner = A3CLearner

    with tf.Graph().as_default():
        sess = tf.Session()
        # Instantiate global step on the graph for 
        # checkpointing before passing to Saver()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        increase_global_step_op = global_step.assign_add(1)

        conf_shared_nw = {'name': "shared_network",
                            'optimizer_conf': optimizer_conf,
                            'shared_network': None,
                            'local_replicas': local_replicas,
                            'num_act': num_actions,
                            'actor_learner_type': actor_learner_type,
                            'global_step': global_step}
        shared_network = Net(conf_shared_nw)

        if actor_learner_type == 0 or actor_learner_type == 1:
            conf_target_nw = {'name': "target_network",
                                'optimizer_conf': optimizer_conf,
                                'shared_network': shared_network,
                                'local_replicas': local_replicas,
                                'num_act': num_actions,
                                'actor_learner_type': actor_learner_type,
                                'global_step': global_step}
            target_network = Net(conf_target_nw)
        elif actor_learner_type == 2:
            target_network = None

        network_replicas = []
        for i in xrange(num_actor_learners):
            if local_replicas:
                conf_shared_nw_replica = {'name': "local_replica_network_{}".format(i),
                                            'optimizer_conf': optimizer_conf,
                                            'shared_network': shared_network,
                                            'local_replicas': local_replicas,
                                            'num_act': num_actions,
                                            'actor_learner_type': actor_learner_type,
                                            'global_step': global_step}
                network_replicas.append(Net(conf_shared_nw_replica))
            else:
                network_replicas.append(shared_network)

        # Instantiate score and epsilon variables on the graph for 
        # checkpointing before passing to Saver()
        # Global score
        global_score_placeholder = tf.placeholder(tf.int64)
        global_score = tf.Variable(tf.cast(0, tf.int64), 
            name='max_global_score', trainable=False)
        update_global_score_op = global_score.assign(global_score_placeholder)
        global_score_summary = tf.scalar_summary("Global score", global_score)
        
        # Thread scores
        thread_score_placeholders = [tf.placeholder(tf.int64) 
            for i in xrange(num_actor_learners)]
        thread_scores = [tf.Variable(tf.cast(0, tf.int64), 
            name='max_thread_' + str(i) + '_score', trainable=False) 
            for i in xrange(num_actor_learners)]
        update_thread_score_ops = [
            thread_scores[i].assign(thread_score_placeholders[i]) 
            for i in xrange(num_actor_learners)]

        # Exploration epsilons
        thread_epsilon_placeholders = [tf.placeholder(tf.float32) 
            for i in xrange(num_actor_learners)]
        thread_epsilons = [tf.Variable(tf.cast(1.0, tf.float32), 
            name='thread_' + str(i) + '_epsilon', trainable=False) 
            for i in xrange(num_actor_learners)]
        thread_epsilon_limits = [tf.Variable(tf.cast(generate_epsilon(), 
            tf.float32), name='thread_' + str(i) + '_epsilon_limits', 
            trainable=False) for i in xrange(num_actor_learners)]
        update_thread_epsilon_ops = [
            thread_epsilons[i].assign(thread_epsilon_placeholders[i]) 
            for i in xrange(num_actor_learners)]

        if actor_learner_type == 0 or actor_learner_type == 1:
            var_list = shared_network.params + target_network.params            
            var_list.extend(thread_epsilons)
            var_list.extend(thread_epsilon_limits)
        elif actor_learner_type == 2:
            var_list = shared_network.actor_params + \
                shared_network.critic_params

        var_list.append(global_step)
        var_list.append(global_score)
        var_list.extend(thread_scores)

        # Reward rescaling
        if rescale_rewards:
            thread_max_reward_placeholders = [tf.placeholder(tf.float32)
                for i in xrange(num_actor_learners)]
            thread_max_rewards = [tf.Variable(tf.cast(1.0, tf.float32),
                name='thread_' + str(i) + '_r_max', trainable=False)
                for i in xrange(num_actor_learners)]
            update_thread_max_reward_ops = [
                thread_max_rewards[i].assign(thread_max_reward_placeholders[i])
                for i in xrange(num_actor_learners)]
            var_list.extend(thread_max_rewards)
        
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10, 
            keep_checkpoint_every_n_hours=2)

        # Initialise if checkpoint does not exist
        restore_vars(saver, sess, emulator_conf["game"], actor_learner_type, 
            actor_learner_max_local_steps)   
            
        # Merge summaries and initialize writer 
        summary_op = tf.merge_all_summaries()  
        summary_writer = tf.train.SummaryWriter(
            "/tmp/summary_logs/{}".format(int(time.time())), sess.graph_def)
        
        # Thread coordinator
        coord = tf.train.Coordinator()
        
        # Checkpoint thread
        saving = Thread(target=save_vars, 
            args=(saver, sess, emulator_conf["game"], actor_learner_type, 
                actor_learner_max_local_steps, global_step, coord))

        summary_conf = {'summary_op': summary_op, 
            'summary_writer': summary_writer,
            'global_score_summary': global_score_summary}

        alg_conf['shared_network'] = shared_network
        alg_conf['target_network'] = target_network
        alg_conf['global_step'] = global_step
        alg_conf['increase_global_step_op'] = increase_global_step_op
        alg_conf['global_score'] = global_score
        alg_conf['global_score_placeholder'] = global_score_placeholder
        alg_conf['update_global_score_op'] = update_global_score_op

        alg_conf['lock'] = Lock()

        visualize = emulator_conf["visualize"]
        if (visualize == 2): emulator_conf["visualize"] = 0        
        actor_learners = []
        for i in xrange(num_actor_learners):
            if (visualize == 2) and (i == num_actor_learners - 1):
                    emulator_conf["visualize"] = 1
            
            alg_conf['actor_id'] = i
            alg_conf['epsilon'] = thread_epsilons[i]
            alg_conf['epsilon_limit'] = thread_epsilon_limits[i]
            alg_conf['epsilon_placeholder'] = thread_epsilon_placeholders[i]
            alg_conf['update_thread_epsilon_op'] = update_thread_epsilon_ops[i]
            alg_conf['thread_score'] = thread_scores[i]
            alg_conf['thread_score_placeholder'] = thread_score_placeholders[i]
            alg_conf['update_thread_score_op'] = update_thread_score_ops[i]

            if rescale_rewards:
                alg_conf['thread_max_reward'] = thread_max_rewards[i]
                alg_conf['thread_max_reward_placeholder'] = \
                    thread_max_reward_placeholders[i]
                alg_conf['update_thread_max_reward_op'] = \
                    update_thread_max_reward_ops[i]
            
            alg_conf['local_network'] = network_replicas[i]

            actor_learners.append(Learner(sess, optimizer_conf, emulator_conf, 
                alg_conf, summary_conf))

        saving.start()

        for t in actor_learners:
            t.start()

        for t in actor_learners:
            t.join()
        logger.debug('All training threads finished')

        coord.request_stop()
        logger.debug('All threads stopped')


if __name__ == '__main__':
    
    # Visualize can take 3 values:
    # 0: no visualization of emulator
    # 1: all emulators, for all actors, are visualized
    # 2: only 1 emulator (for one of the actors) is visualized
    
    _exit = False
    optimizer_conf = {}
    emulator_conf = {}
    alg_conf = {}
    if len(sys.argv) == 23:
        emulator_conf["game"] = sys.argv[1]
        emulator_conf["rom_path"] = sys.argv[2] # "../atari_roms"
        emulator_conf["visualize"] = int(sys.argv[3])
        optimizer_conf["type"] = sys.argv[4]
        optimizer_conf["mode"] = sys.argv[5]
        optimizer_conf["base_learning_rate"] = float(sys.argv[6])
        optimizer_conf["clip_delta"] = float(sys.argv[7])
        optimizer_conf['lr_decay_step'] = int(sys.argv[8])
        optimizer_conf['lr_decay_rate'] = float(sys.argv[9])
        optimizer_conf['lr_staircase'] = {'True': True, 'False': False}[sys.argv[10]]
        optimizer_conf['entropy_regularisation_strength'] = float(sys.argv[11])
        optimizer_conf['clip_norm'] = float(sys.argv[12])
        optimizer_conf['clip_norm_type'] = sys.argv[13]
        alg_conf['actor_learner_type'] = {'Q': 0, 'sarsa': 1, 'a3c': 2}[sys.argv[14]]
        alg_conf['num_actor_learners'] = int(sys.argv[15])
        alg_conf['gamma'] = float(sys.argv[16])
        alg_conf['q_target_update_steps'] = int(sys.argv[17])
        alg_conf['grads_update_steps'] = int(sys.argv[18])
        alg_conf['max_global_steps'] = int(sys.argv[19])
        alg_conf['max_epsilon_annealing_steps'] = int(sys.argv[20])
        alg_conf['local_replicas'] = {'True': True, 'False': False}[sys.argv[21]]
        alg_conf['max_local_steps'] = int(sys.argv[22])
        alg_conf['rescale_rewards'] = {'True': True, 'False': False}[sys.argv[23]]
    elif len(sys.argv) == 4:
        emulator_conf["game"] = sys.argv[1]
        emulator_conf["rom_path"] = sys.argv[2] # "../atari_roms"
        emulator_conf["visualize"] = int(sys.argv[3])
        optimizer_conf["type"] = "rmsprop"
        optimizer_conf["mode"] = "shared"
        optimizer_conf["base_learning_rate"] = get_learning_rate(10**-4, 10**-2)
        optimizer_conf["clip_delta"] = 1.0
        optimizer_conf['lr_decay_step'] = 100000
        optimizer_conf['lr_decay_rate'] = 0.95
        optimizer_conf['lr_staircase'] = False
        optimizer_conf['entropy_regularisation_strength'] = 0.01
        optimizer_conf['clip_norm'] = 40.0 # Max gradient norm for clipping
        optimizer_conf['clip_norm_type'] = 'global' # global/local/ignore
        alg_conf['actor_learner_type'] = {'Q': 0, 'sarsa': 1, 'a3c': 2}['Q']
        alg_conf['num_actor_learners'] = 16
        alg_conf['gamma'] = 0.99
        alg_conf['q_target_update_steps'] = 10000 # 40000 frames / 4
        alg_conf['grads_update_steps'] = 5
        alg_conf['max_global_steps'] = 2147483647
        alg_conf['max_epsilon_annealing_steps'] = 1000000 # 4 million frames / 4
        alg_conf['local_replicas'] = False # Must be True for n-step and a3c
        alg_conf['max_local_steps'] = 1 # The n in n-step
        alg_conf['rescale_rewards'] = False
    else:
        print("You must provide at least 3 arguments! Try:\n" \
            "python main.py <game-name> ../atari_roms/ 1")
        _exit = True
        
    if not _exit:
        main(optimizer_conf, emulator_conf, alg_conf)
