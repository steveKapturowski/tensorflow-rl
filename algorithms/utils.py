import tensorflow as tf 
import os

def restore_vars(saver, sess, game, alg_type, max_local_steps):
    """ Restore saved net, global step, and epsilons OR 
    create checkpoint directory for later storage. """

    alg = alg_type + "{}/".format("_" + str(max_local_steps) + "_steps" if alg_type == 'q' else "") 
    checkpoint_dir = 'checkpoints/' + game + '/' + alg
    
    check_or_create_checkpoint_dir(checkpoint_dir)
    path = tf.train.latest_checkpoint(checkpoint_dir)
    if path is None:
        sess.run(tf.initialize_all_variables())
        return 0
    else:
        saver.restore(sess, path)
        global_step = int(path[path.rfind("-") + 1:])
        return global_step 

def save_vars(saver, sess, game, alg_type, max_local_steps, global_step):
    """ Checkpoint shared net params, global score and step, and epsilons. """

    alg = alg_type + "{}/".format("_" + str(max_local_steps) + "_steps" if alg_type == 'q' else "") 
    checkpoint_dir = 'checkpoints/' + game + '/' + alg
    
    check_or_create_checkpoint_dir(checkpoint_dir)
    saver.save(sess, checkpoint_dir + "model", global_step=global_step)


def check_or_create_checkpoint_dir(checkpoint_dir):
    """ Create checkpoint directory if it does not exist """
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

# def save_shared_mem_vars(shared_mem_vars, game_name, alg_type, 
#     max_local_steps):
#     checkpoint_dir = 'checkpoints/' + game_name + '/' + \
#         {'0': 'Q/', '1': 'sarsa/', '2': 'a3c/'}[str(alg_type)] + \
#         str(max_local_steps) + '_step' + '/'
# 
#     check_or_create_checkpoint_dir(checkpoint_dir)
#     while True:
#         g_step = shared_mem_vars['global_step'].val.value
#         if g_step % 1000000 == 0:
#             path = checkpoint_dir + 'vars-opt-' + str(g_step)
#             np.save(path + '-learning', np.frombuffer(shared_mem_vars['learning_vars.vars'], ctypes.c_float))
#             np.save(path + '-target', np.frombuffer(shared_mem_vars['target_vars.vars'], ctypes.c_float)) 
#             for i in xrange(len(shared_mem_vars['opt_state.vars'])):
#                 np.save(path + '-opt' + str(i), 
#                         np.frombuffer(shared_mem_vars['opt_state'].vars[i], ctypes.c_float))
