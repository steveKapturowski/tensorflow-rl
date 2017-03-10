# -*- encoding: utf-8 -*-
import layers
import numpy as np
import tensorflow as tf
from network import Network
from custom_lstm import CustomBasicLSTMCell


class PolicyVNetwork(Network):
 
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient 
        compute and apply ops, network parameter synchronization ops, and 
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)
        
        self.beta = conf['args'].entropy_regularisation_strength
        self.use_recurrent = conf['args'].alg_type == 'a3c-lstm'
                
        with tf.name_scope(self.name):

            self.critic_target_ph = tf.placeholder(
                'float32', [None], name='target')
            self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

            if self.use_recurrent:
                layer_name = 'lstm_layer'
                self.hidden_state_size = 256
                with tf.variable_scope(self.name+'/'+layer_name) as vs:
                    self.lstm_cell = CustomBasicLSTMCell(self.hidden_state_size, forget_bias=1.0)

                    self.step_size = tf.placeholder(tf.float32, [1], name='step_size')
                    self.initial_lstm_state = tf.placeholder(
                        tf.float32, [1, 2*self.hidden_state_size], name='initital_state')
                    
                    ox_reshaped = tf.reshape(self.ox, [1,-1,256])
                    lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                        self.lstm_cell,
                        ox_reshaped,
                        initial_state=self.initial_lstm_state,
                        sequence_length=self.step_size,
                        time_major=False,
                        scope=vs)

                    self.ox = tf.reshape(lstm_outputs, [-1,256])

                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in 
                        tf.trainable_variables() if v.name.startswith(vs.name)]


            # Final actor layer
            layer_name = 'softmax_policy4'            
            self.wpi, self.bpi, self.output_layer_pi, self.log_output_layer_pi = layers.softmax_and_log_softmax(
                layer_name, self.ox, self.num_actions)
            
            # Entropy: ∑_a[-p_a ln p_a]
            self.output_layer_entropy = tf.reduce_sum(
                - 1.0 * tf.multiply(
                    self.output_layer_pi,
                    self.log_output_layer_pi
                ), axis=1)
            self.entropy = tf.reduce_mean(self.output_layer_entropy)

            
            # Final critic layer
            self.wv, self.bv, self.output_layer_v = layers.fc(
                'fc_value4', self.ox, 1, activation='linear')

            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


            # Advantage critic
            self.adv_critic = tf.subtract(self.critic_target_ph, tf.reshape(self.output_layer_v, [-1]))
            
            # Actor objective
            # Multiply the output of the network by a one hot vector, 1 for the 
            # executed action. This will make the non-regularised objective 
            # term for non-selected actions to be zero.
            self.log_output_selected_action = tf.reduce_sum(
                self.log_output_layer_pi*self.selected_action_ph, 
                axis=1
            )
            actor_objective_advantage_term = tf.multiply(
                self.log_output_selected_action, self.adv_actor_ph
            )
            actor_objective_entropy_term = self.beta * self.output_layer_entropy
            self.actor_objective = -tf.reduce_mean(
                actor_objective_advantage_term
                + actor_objective_entropy_term
            )
            
            # Critic loss
            if self.clip_loss_delta > 0:
                quadratic_part = tf.reduce_mean(tf.pow(
                    tf.minimum(
                        tf.abs(self.adv_critic), self.clip_loss_delta
                    ), 2))
                linear_part = tf.subtract(tf.abs(self.adv_critic), quadratic_part)
                #OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = tf.multiply(tf.constant(0.5), tf.nn.l2_loss(quadratic_part) + \
                    self.clip_loss_delta * linear_part)
            else:
                self.critic_loss = 0.5 * tf.reduce_mean(tf.pow(self.adv_critic, 2))          
            
            self.loss = self.actor_objective + self.critic_loss
            
            # Optimizer
            grads = tf.gradients(self.loss, self.params)

            if self.clip_norm_type == 'ignore':
                # Unclipped gradients
                self.get_gradients = grads
            elif self.clip_norm_type == 'global':
                # Clip network grads by network norm
                self.get_gradients = tf.clip_by_global_norm(
                            grads, self.clip_norm)[0]
            elif self.clip_norm_type == 'local':
                # Clip layer grads by layer norm
                self.get_gradients = [tf.clip_by_norm(
                            g, self.clip_norm) for g in grads]

            # Placeholders for shared memory vars
            self.params_ph = []
            for p in self.params:
                self.params_ph.append(tf.placeholder(tf.float32, 
                    shape=p.get_shape(), 
                    name="shared_memory_for_{}".format(
                        (p.name.split("/", 1)[1]).replace(":", "_"))))
            
            # Ops to sync net with shared memory vars
            self.sync_with_shared_memory = []
            for i in xrange(len(self.params)):
                self.sync_with_shared_memory.append(
                    self.params[i].assign(self.params_ph[i]))



class PolicyRepeatNetwork(Network):
 
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient 
        compute and apply ops, network parameter synchronization ops, and 
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)
        
        self.beta = conf['args'].entropy_regularisation_strength
        self.use_recurrent = conf['args'].alg_type == 'a3c-lstm'
                
        with tf.name_scope(self.name):

            self.critic_target_ph = tf.placeholder(
                'float32', [None], name='target')
            self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

            if self.use_recurrent:
                layer_name = 'lstm_layer'
                self.hidden_state_size = 256
                with tf.variable_scope(self.name+'/'+layer_name) as vs:
                    self.lstm_cell = CustomBasicLSTMCell(self.hidden_state_size, forget_bias=1.0)

                    self.step_size = tf.placeholder(tf.float32, [1], name='step_size')
                    self.initial_lstm_state = tf.placeholder(
                        tf.float32, [1, 2*self.hidden_state_size], name='initital_state')
                    
                    ox_reshaped = tf.reshape(self.ox, [1,-1,256])
                    lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                        self.lstm_cell,
                        ox_reshaped,
                        initial_state=self.initial_lstm_state,
                        sequence_length=self.step_size,
                        time_major=False,
                        scope=vs)

                    self.ox = tf.reshape(lstm_outputs, [-1,256])

                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in 
                        tf.trainable_variables() if v.name.startswith(vs.name)]


            # Final actor layer
            layer_name = 'softmax_policy4'            
            self.wpi, self.bpi, self.output_layer_pi, self.log_output_layer_pi = layers.softmax_and_log_softmax(
                layer_name, self.ox, self.num_actions)

            # Compute Poisson x Discrete Gaussian
            self.t = tf.placeholder(tf.float32, num_actions)
            self.l = tf.placeholder(tf.float32, num_actions)
            k = np.array(range(10))
            t_hat = tf.log(1+tf.exp(self.t))
            l_hat = tf.log(1+tf.exp(self.l))
            self.action_repeat_probs = tf.exp(-(k-l_hat)**2/(2*t_hat) - l_hat - tf.lgamma(k+1)) * l_hat**k
            self.log_action_repeat_probs = -(k-l_hat)**2/(2*t_hat) - l_hat - tf.lgamma(k+1) + k*tf.log(l_hat)

            self.selected_repeat = self.placeholder(tf.int32)
            self.selected_repeat_prob = self.action_repeat_probs[self.selected_repeat]
            
            # Entropy: ∑_a[-p_a ln p_a]
            self.output_layer_entropy = tf.reduce_sum(
                - 1.0 * tf.multiply(
                    self.output_layer_pi * self.action_repeat_probs,
                    self.log_output_layer_pi + self.log_action_repeat_probs
                ), axis=1)
            self.entropy = tf.reduce_mean(self.output_layer_entropy)

            
            # Final critic layer
            self.wv, self.bv, self.output_layer_v = layers.fc(
                'fc_value4', self.ox, 1, activation='linear')

            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


            # Advantage critic
            self.adv_critic = tf.subtract(self.critic_target_ph, tf.reshape(self.output_layer_v, [-1]))
            
            # Actor objective
            # Multiply the output of the network by a one hot vector, 1 for the 
            # executed action. This will make the non-regularised objective 
            # term for non-selected actions to be zero.
            self.log_output_selected_action = tf.reduce_sum(
                (self.log_output_layer_pi + self.selected_repeat_prob) * self.selected_action_ph, 
                axis=1
            )
            actor_objective_advantage_term = tf.multiply(
                self.log_output_selected_action, self.adv_actor_ph
            )
            actor_objective_entropy_term = self.beta * self.output_layer_entropy
            self.actor_objective = -tf.reduce_mean(
                actor_objective_advantage_term
                + actor_objective_entropy_term
            )
            
            # Critic loss
            if self.clip_loss_delta > 0:
                quadratic_part = tf.reduce_mean(tf.pow(
                    tf.minimum(
                        tf.abs(self.adv_critic), self.clip_loss_delta
                    ), 2))
                linear_part = tf.subtract(tf.abs(self.adv_critic), quadratic_part)
                #OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = tf.multiply(tf.constant(0.5), tf.nn.l2_loss(quadratic_part) + \
                    self.clip_loss_delta * linear_part)
            else:
                self.critic_loss = 0.5 * tf.reduce_mean(tf.pow(self.adv_critic, 2))          
            
            self.loss = self.actor_objective + self.critic_loss
            
            # Optimizer
            grads = tf.gradients(self.loss, self.params)

            if self.clip_norm_type == 'ignore':
                # Unclipped gradients
                self.get_gradients = grads
            elif self.clip_norm_type == 'global':
                # Clip network grads by network norm
                self.get_gradients = tf.clip_by_global_norm(
                            grads, self.clip_norm)[0]
            elif self.clip_norm_type == 'local':
                # Clip layer grads by layer norm
                self.get_gradients = [tf.clip_by_norm(
                            g, self.clip_norm) for g in grads]

            # Placeholders for shared memory vars
            self.params_ph = []
            for p in self.params:
                self.params_ph.append(tf.placeholder(tf.float32, 
                    shape=p.get_shape(), 
                    name="shared_memory_for_{}".format(
                        (p.name.split("/", 1)[1]).replace(":", "_"))))
            
            # Ops to sync net with shared memory vars
            self.sync_with_shared_memory = []
            for i in xrange(len(self.params)):
                self.sync_with_shared_memory.append(
                    self.params[i].assign(self.params_ph[i]))


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


#This is still experimental
class SequencePolicyVNetwork(Network):

    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient 
        compute and apply ops, network parameter synchronization ops, and 
        summary ops. """

        super(SequencePolicyVNetwork, self).__init__(conf)
        
        self.beta = conf['args'].entropy_regularisation_strength
        self.max_decoder_steps = conf['args'].max_decoder_steps

        self.max_local_steps = conf['args'].max_local_steps
        self.use_recurrent = conf['args'].alg_type == 'a3c-lstm'
        
        with tf.name_scope(self.name):

            self.critic_target_ph = tf.placeholder(
                'float32', [self.batch_size], name='target')
            self.adv_actor_ph = tf.placeholder("float", [self.batch_size], name='advantage')

            if self.arch == 'NIPS':
                self.ox = self.o3
            else: #NATURE
                self.ox = self.o4

            with tf.variable_scope(self.name+'/lstm_decoder') as vs:
                self.action_outputs = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='action_outputs')
                self.action_inputs = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='action_inputs')
                
                self.decoder_seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name='decoder_seq_lengths')
                self.allowed_actions = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='allowed_actions')
                self.use_fixed_action = tf.placeholder(tf.bool, name='use_fixed_action')
                self.temperature = tf.placeholder(tf.float32, name='temperature')

                self.decoder_hidden_state_size = 256
                self.decoder_lstm_cell = CustomBasicLSTMCell(self.decoder_hidden_state_size, forget_bias=1.0)
                self.decoder_initial_state = tf.placeholder(tf.float32, [self.batch_size, 2*self.decoder_hidden_state_size], name='decoder_initial_state')

                self.network_state = tf.concat(axis=1, values=[
                    tf.zeros_like(self.ox), self.ox
                    # self.ox, tf.zeros_like(self.ox)
                ])

                self.W_actions = tf.get_variable('W_actions', shape=[self.decoder_hidden_state_size, self.num_actions+1], dtype='float32', initializer=tf.contrib.layers.xavier_initializer())
                self.b_actions = tf.get_variable('b_actions', shape=[self.num_actions+1], dtype='float32', initializer=tf.zeros_initializer())


                self.decoder_state, self.logits, self.actions = decoder(
                    self.action_inputs,
                    self.network_state,
                    self.decoder_lstm_cell,
                    self.decoder_seq_lengths,
                    self.W_actions,
                    self.b_actions,
                    self.max_decoder_steps,
                    vs,
                    self.use_fixed_action,
                    self.action_outputs,
                    loop_function=loop_gumbel_softmax(self.temperature),
                )

                self.decoder_trainable_variables = [
                    v for v in tf.trainable_variables()
                    if v.name.startswith(vs.name)
                ]


            print 'Decoder out: s,l,a=', self.decoder_state.get_shape(), self.logits.get_shape(), self.actions.get_shape()


            #mask softmax by allowed actions
            exp_logits = tf.exp(self.logits) * self.allowed_actions
            Z = tf.expand_dims(tf.reduce_sum(exp_logits, 2), 2)
            self.action_probs = exp_logits / Z
            log_action_probs = self.logits - tf.log(Z)

            sequence_probs = tf.reduce_prod(tf.reduce_sum(self.action_probs * self.action_outputs, 2), 1)
            log_sequence_probs = tf.reduce_sum(tf.reduce_sum(log_action_probs * self.action_outputs, 2), 1)

            # ∏a_i * ∑ log a_i
            self.output_layer_entropy = - tf.reduce_mean(tf.stop_gradient(1 + log_sequence_probs) * log_sequence_probs)
            self.entropy = - tf.reduce_mean(log_sequence_probs)

            print 'sp, lsp:', sequence_probs.get_shape(), log_sequence_probs.get_shape()



            # Final critic layer
            self.wv, self.bv, self.output_layer_v = layers.fc(
                'fc_value4', self.ox, 1, activation='linear')

            # Advantage critic
            self.adv_critic = self.critic_target_ph - tf.reshape(self.output_layer_v, [-1])

            self.actor_advantage_term = tf.reduce_sum(log_sequence_probs[:self.max_local_steps] * self.adv_actor_ph)
            self.actor_entropy_term = self.beta * self.output_layer_entropy
            self.actor_objective = - (
                self.actor_advantage_term
                + self.actor_entropy_term
            )
            
            # Critic loss
            if self.clip_loss_delta > 0:
                quadratic_part = tf.minimum(tf.abs(self.adv_critic), 
                    tf.constant(self.clip_loss_delta))
                linear_part = tf.abs(self.adv_critic) - quadratic_part
                #OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = 0.5*tf.nn.l2_loss(quadratic_part) + \
                    self.clip_loss_delta*linear_part
            else:
                #OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = 0.5*tf.nn.l2_loss(self.adv_critic)     
            
            self.loss = self.actor_objective + self.critic_loss
            

            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)



            # Optimizer
            grads = tf.gradients(self.loss, self.params)

            # This is not really an operation, but a list of gradient Tensors. 
            # When calling run() on it, the value of those Tensors 
            # (i.e., of the gradients) will be calculated
            if self.clip_norm_type == 'ignore':
                self.get_gradients = grads
            elif self.clip_norm_type == 'global':
                self.get_gradients = tf.clip_by_global_norm(grads, self.clip_norm)[0]
            elif self.clip_norm_type == 'avg':
                self.get_gradients = tf.clip_by_average_norm(grads, self.clip_norm)[0]
            elif self.clip_norm_type == 'local':
                self.get_gradients = [
                    tf.clip_by_norm(g, self.clip_norm) for g in grads
                ]

            # Placeholders for shared memory vars
            self.params_ph = []
            for p in self.params:
                self.params_ph.append(tf.placeholder(tf.float32, 
                    shape=p.get_shape(), 
                    name="shared_memory_for_{}".format(
                        (p.name.split("/", 1)[1]).replace(":", "_"))))
            
            # Ops to sync net with shared memory vars
            self.sync_with_shared_memory = []
            for i in xrange(len(self.params)):
                self.sync_with_shared_memory.append(
                    self.params[i].assign(self.params_ph[i]))

            
