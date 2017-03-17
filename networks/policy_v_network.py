# -*- encoding: utf-8 -*-
import layers
import numpy as np
import tensorflow as tf
from network import Network
from custom_lstm import CustomBasicLSTMCell
from sequence_decoder import decoder, loop_gumbel_softmax


class PolicyValueNetwork(Network):
 
    def __init__(self, conf, use_policy_head=True, use_value_head=True):
        super(PolicyValueNetwork, self).__init__(conf)
        
        self.beta = conf['args'].entropy_regularisation_strength
                
        with tf.name_scope(self.name):
            self.loss = 0.0
            if use_policy_head:
                self.loss += self._build_policy_head()

            if use_value_head:
                self.loss += self._build_value_head()

            self._build_gradient_ops()

    def _build_policy_head(self):
        self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')       
        self.wpi, self.bpi, self.output_layer_pi, self.log_output_layer_pi = layers.softmax_and_log_softmax(
            'softmax_policy4', self.ox, self.num_actions)
            
        # Entropy: ∑_a[-p_a ln p_a]
        self.output_layer_entropy = tf.reduce_sum(
            - 1.0 * tf.multiply(
                self.output_layer_pi,
                self.log_output_layer_pi
            ), axis=1)
        self.entropy = tf.reduce_mean(self.output_layer_entropy)
            
        self.log_output_selected_action = tf.reduce_sum(
            self.log_output_layer_pi*self.selected_action_ph, 
            axis=1)

        self.actor_objective = -tf.reduce_mean(
            self.log_output_selected_action * self.adv_actor_ph
            + self.beta * self.output_layer_entropy)

        return self.actor_objective

    def _build_value_head(self):
        self.critic_target_ph = tf.placeholder('float32', [None], name='target')
        self.wv, self.bv, self.output_layer_v = layers.fc(
            'fc_value4', self.ox, 1, activation='linear')

        # Advantage critic
        self.adv_critic = tf.subtract(self.critic_target_ph, tf.reshape(self.output_layer_v, [-1]))
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

        return self.critic_loss


class PolicyNetwork(PolicyValueNetwork):
    def __init__(self, conf,):
        super(PolicyNetwork, self).__init__(conf, use_value_head=False)


class PolicyRepeatNetwork(PolicyValueNetwork):
 
    def __init__(self, conf):
        '''
        Extends action space to parametrize a discrete distribution over repetion lengths
        for each original action
        '''
        super(PolicyRepeatNetwork, self).__init__(conf)
        

    def _build_policy_head(self):
        self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

        self.wpi, self.bpi, self.output_layer_pi, self.log_output_layer_pi = layers.softmax_and_log_softmax(
            'softmax_policy4', self.ox, self.num_actions)

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


        self.log_output_selected_action = tf.reduce_sum(
            (self.log_output_layer_pi + self.selected_repeat_prob) * self.selected_action_ph,
            axis=1)

        self.actor_objective = -tf.reduce_mean(
            self.log_output_selected_action * self.adv_actor_ph
            + self.beta * self.output_layer_entropy)

        return self.actor_objective


#This is still experimental
class SequencePolicyVNetwork(PolicyValueNetwork):

    def __init__(self, conf):
        '''Uses lstm decoder to output action sequences'''
        self.max_decoder_steps = conf['args'].max_decoder_steps
        self.max_local_steps = conf['args'].max_local_steps
        super(SequencePolicyVNetwork, self).__init__(conf)

        
    def _build_policy_head(self):
        self.adv_actor_ph = tf.placeholder("float", [self.batch_size], name='advantage')

        with tf.variable_scope(self.name+'/lstm_decoder') as vs:
            self.action_outputs = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='action_outputs')
            self.action_inputs = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='action_inputs')
                
            self.decoder_seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name='decoder_seq_lengths')
            self.allowed_actions = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='allowed_actions')
            self.use_fixed_action = tf.placeholder(tf.bool, name='use_fixed_action')
            self.temperature = tf.placeholder(tf.float32, name='temperature')

            self.decoder_hidden_state_size = self.ox.get_shape().as_list()[-1]
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


        self.actor_advantage_term = tf.reduce_sum(log_sequence_probs[:self.max_local_steps] * self.adv_actor_ph)
        self.actor_entropy_term = self.beta * self.output_layer_entropy
        self.actor_objective = - (
            self.actor_advantage_term
            + self.actor_entropy_term
        )

        return self.actor_objective
            

