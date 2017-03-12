# -*- encoding: utf-8 -*-
import tensorflow as tf


def gumbel_noise(shape, epsilon=1e-30):
    U = tf.random_uniform(shape)
    return -tf.log(-tf.log(U + epsilon) + epsilon)


def sample_gumbel_softmax(logits, allow_stop, temperature):
    y = logits + gumbel_noise(tf.shape(logits))

    mask = tf.concat(axis=1, values=[
        tf.ones_like(logits[:, :-1]),
        tf.zeros_like(logits[:, -1:])
    ])
    identity = tf.ones_like(logits)
    mask = tf.cond(allow_stop, lambda: identity, lambda: mask)

    exp_logits = tf.exp(y) * mask
    Z = tf.expand_dims(tf.reduce_sum(exp_logits, 1), 1)
    return exp_logits / Z



def gumbel_softmax(logits, use_fixed_action, fixed_action, allow_stop, temperature=0.5, hard=True):
    y = sample_gumbel_softmax(logits, allow_stop, temperature)
    y.set_shape(logits.get_shape().as_list())

    if hard:
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y_hard = tf.cond(use_fixed_action, lambda: fixed_action, lambda: y_hard)
        y = tf.stop_gradient(y_hard - y) + y
    
    return y


def loop_gumbel_softmax(temperature=0.5):
    def loop_function(logits, use_fixed_action, fixed_action, allow_stop):
        return gumbel_softmax(logits, use_fixed_action, fixed_action, allow_stop, temperature=temperature)

    return loop_function


def decoder(decoder_inputs, initial_state, cell, sequence_lengths, W_actions, b_actions,
            max_decoder_steps, scope, use_fixed_action, action_outputs,
            loop_function=None, output_size=None, dtype=tf.float32):

    assert decoder_inputs.get_shape().ndims == 3, 'Decoder inputs must have rank 3'

    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(scope):
        batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        states = [initial_state]
        outputs = []
        prev = None

        t0 = tf.constant(0, dtype=tf.int32)
        max_seq_len = tf.reduce_max(sequence_lengths)
        loop_condition = lambda t, state, logits, action, s_array, l_array, a_array: tf.less(t, max_seq_len, name='loop_condition')

        state_array = tf.TensorArray(dtype=tf.float32, size=max_seq_len, infer_shape=True, dynamic_size=True, name='state_array')
        logits_array = tf.TensorArray(dtype=tf.float32, size=max_seq_len, infer_shape=True, dynamic_size=True, name='logits_array')
        action_array = tf.TensorArray(dtype=tf.float32, size=max_seq_len, infer_shape=True, dynamic_size=True, name='action_array')

        def body(t, hidden_state, logits, action, state_array, logits_array, action_array):
            o, s = cell(action, hidden_state)
            l = tf.nn.xw_plus_b(o, W_actions, b_actions)
            a = loop_function(
                l,
                use_fixed_action,
                action_outputs[:, t, :],
                tf.greater(t, 0))

            update = tf.less(t, sequence_lengths, name='update_cond')

            s_out = tf.where(update, s, hidden_state)
            l_out = tf.where(update, l, logits)
            a_out = tf.where(update, a, action)

            state_array = state_array.write(t, s_out)
            logits_array = logits_array.write(t, l_out)
            action_array = action_array.write(t, a_out)

            return t+1, s, l, a, state_array, logits_array, action_array


        go_action = decoder_inputs[:, 0, :]
        final_const, states, logits, actions, state_array, logits_array, action_array = tf.while_loop(
            loop_condition,
            body,
            loop_vars=[t0,
                       initial_state,
                       decoder_inputs[:, 0, :],
                       go_action,
                       state_array,
                       logits_array,
                       action_array])

        return (
            tf.transpose(state_array.stack(), perm=[1, 0, 2]),
            tf.transpose(logits_array.stack(), perm=[1, 0, 2]),
            tf.transpose(action_array.stack(), perm=[1, 0, 2]),
        )

