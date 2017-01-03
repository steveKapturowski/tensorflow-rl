# -*- encoding: utf-8 -*-
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
                    
                    o3_reshaped = tf.reshape(self.o3, [1,-1,256])
                    lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                        self.lstm_cell,
                        o3_reshaped,
                        initial_state=self.initial_lstm_state,
                        sequence_length=self.step_size,
                        time_major=False,
                        scope=vs)

                    self.ox = tf.reshape(lstm_outputs, [-1,256])

                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in 
                        tf.trainable_variables() if v.name.startswith(vs.name)]
            else:
                if self.arch == 'NIPS':
                    self.ox = self.o3
                else: #NATURE
                    self.ox = self.o4

            # Final actor layer
            layer_name = 'softmax_policy4'            
            self.wpi, self.bpi, self.output_layer_pi, self.log_output_layer_pi = self._softmax_and_log_softmax(
                layer_name, self.ox, self.num_actions)
            
            # Entropy: sum_a (-p_a ln p_a)
            self.output_layer_entropy = tf.reduce_sum(
                - 1.0 * tf.mul(
                    self.output_layer_pi,
                    self.log_output_layer_pi
                ), reduction_indices=1)

            
            # Final critic layer
            self.wv, self.bv, self.output_layer_v = self._fc(
                'fc_value4', self.ox, 1, activation='linear')

            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


            # Advantage critic
            self.adv_critic = tf.sub(self.critic_target_ph, tf.reshape(self.output_layer_v, [-1]))
            
            # Actor objective
            # Multiply the output of the network by a one hot vector, 1 for the 
            # executed action. This will make the non-regularised objective 
            # term for non-selected actions to be zero.
            self.log_output_selected_action = tf.reduce_sum(
                self.log_output_layer_pi*self.selected_action_ph, 
                reduction_indices=1
            )
            actor_objective_advantage_term = tf.mul(
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
                linear_part = tf.sub(tf.abs(self.adv_critic), quadratic_part)
                #OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = tf.mul(tf.constant(0.5), tf.nn.l2_loss(quadratic_part) + \
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
    U = tf.random_uniform(shape, min=0, max=1)
    return -tf.log(-tf.log(U + epsilon) + epsilon)


def sample_gumbel_softmax(logits, temperature):
    y = logits + gumbel_noise(tf.shape(logits))
    return tf.nn.softmax(y/temperature)


def gumbel_softmax(logits, temperature=0.5, hard=True):
    y = sample_gumbel_softmax(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


def sample_gumbel_softmax_and_embed(embedding, temperature=0.5, output_projection=None):
    def loop_function(prev, _):
        if output_projection is not None:
            logits = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
            
        return gumbel_softmax(logits, temperature=temperature)

    return loop_function


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
                'float32', [None], name='target')
            self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

            if self.arch == 'NIPS':
                self.ox = self.o3
            else: #NATURE
                self.ox = self.o4

            with tf.variable_scope(self.name+'/lstm_decoder') as vs:
                self.decoder_seq_lengths = tf.placeholder(tf.float32, [None], name='decoder_seq_lengths')
                self.action_outputs = tf.placeholder(tf.float32, [None, self.max_decoder_steps, self.num_actions+1], name='action_outputs')
                self.action_inputs = tf.placeholder(tf.float32, [None, self.max_decoder_steps, self.num_actions+1], name='action_inputs')
                self.allowed_actions = tf.placeholder(tf.float32, [None, self.max_decoder_steps, self.num_actions+1], name='allowed_actions')

                self.decoder_hidden_state_size = 256
                self.decoder_lstm_cell = CustomBasicLSTMCell(self.decoder_hidden_state_size, forget_bias=1.0)

                self.network_state = tf.concat(1, [
                    tf.fill(tf.shape(self.ox), 0.0), self.ox
                ])
                self.decoder_initial_state = tf.placeholder(tf.float32, [None, 2*self.decoder_hidden_state_size], name='decoder_initial_state')

                self.modify_state = tf.placeholder(tf.bool, name='modify_state')
                initial_state_op = tf.cond(
                    self.modify_state,
                    lambda: self.decoder_initial_state,
                    lambda: self.network_state,
                    name='decode_initial_state_conditional')
                decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(
                    self.decoder_lstm_cell,
                    self.action_inputs,
                    initial_state=initial_state_op,
                    sequence_length=self.decoder_seq_lengths,
                    time_major=False,
                    scope=vs)


                self.decoder_trainable_variables = [
                    v for v in tf.trainable_variables()
                    if v.name.startswith(vs.name)
                ]

            fan_in = self.decoder_hidden_state_size
            fan_out = self.num_actions+1
            d = np.sqrt(6. / (fan_in + fan_out))
            initial = tf.random_uniform([fan_in, fan_out], minval=-d, maxval=d)
            self.W_pi = tf.Variable(initial, name='W_pi', dtype='float32')
            self.b_pi = tf.Variable(tf.zeros(fan_out, name='b_pi', dtype='float32'))


            logits = tf.einsum('ijk,kl->ijl', decoder_outputs, self.W_pi) + self.b_pi
            
            #mask softmax by allowed actions
            exp_logits = tf.exp(logits) * self.allowed_actions
            Z = tf.expand_dims(tf.reduce_sum(exp_logits, 2), 2)
            self.action_probs = exp_logits / Z
            log_action_probs = logits - tf.log(Z)



            sequence_probs = tf.reduce_prod(tf.reduce_sum(self.action_probs * self.action_outputs, 2), 1)
            log_sequence_probs = tf.reduce_sum(tf.reduce_sum(log_action_probs * self.action_outputs, 2), 1)

            # ∏a_i * ∑ log a_i
            # self.output_layer_entropy = - tf.reduce_mean(tf.stop_gradient(1 + log_sequence_probs) * log_sequence_probs)
            # self.entropy = - tf.reduce_mean(log_sequence_probs)

            self.output_layer_entropy = - tf.reduce_mean(
                tf.expand_dims(tf.reduce_sum(self.action_outputs, 2), 2) * self.action_probs * log_action_probs)
            self.entropy = self.output_layer_entropy

            # Final critic layer
            self.wv, self.bv, self.output_layer_v = self._fc(
                'fc_value4', self.ox, 1, activation='linear')

            # Advantage critic
            self.adv_critic = self.critic_target_ph - tf.reshape(self.output_layer_v, [-1])


            self.actor_advantage_term = tf.reduce_sum(log_sequence_probs[:self.max_local_steps] * self.adv_actor_ph)
            self.actor_entropy_term = self.beta * self.output_layer_entropy
            self.actor_objective = - (
                self.actor_advantage_term + self.actor_entropy_term
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

            
