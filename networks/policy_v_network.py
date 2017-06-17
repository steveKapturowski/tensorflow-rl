# -*- encoding: utf-8 -*-
import layers
import numpy as np
import tensorflow as tf
from network import Network
from utils.distributions import Discrete
from custom_lstm import CustomBasicLSTMCell
from sequence_decoder import decoder, loop_gumbel_softmax


class PolicyValueNetwork(Network):
 
    def __init__(self, conf, use_policy_head=True, use_value_head=True):
        super(PolicyValueNetwork, self).__init__(conf)
        self.beta = conf['args'].entropy_regularisation_strength
        self.share_encoder_weights = conf['args'].share_encoder_weights

        encoded_state = self._build_encoder()
                
        with tf.variable_scope(self.name):
            self.loss = 0.0
            if use_policy_head:
                self.loss += self._build_policy_head(encoded_state)

            if use_value_head:
                if not self.share_encoder_weights:
                    with tf.variable_scope('value_encoder'):
                        encoded_state = self._build_encoder()

                self.loss += self._build_value_head(encoded_state)
            self.loss *= tf.cast(tf.shape(self.input_ph)[0], tf.float32) / self.max_local_steps
            self._build_gradient_ops(self.loss)

    def _build_policy_head(self, input_state):
        self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')       
        self.wpi, self.bpi, self.logits = layers.fc(
            'logits', input_state, self.num_actions, activation='linear')
        self.dist = Discrete(self.logits)

        self.output_layer_entropy = self.dist.entropy()
        self.entropy = tf.reduce_sum(self.output_layer_entropy)
        
        self.output_layer_pi = self.dist.probs
        self.log_output_layer_pi = self.dist.log_probs
        self.log_output_selected_action = self.dist.log_likelihood(self.selected_action_ph)
        self.sample_action = self.dist.sample()

        self.actor_objective = -tf.reduce_sum(
            self.log_output_selected_action * self.adv_actor_ph
            + self.beta * self.output_layer_entropy
        )
        return self.actor_objective

    def _build_value_head(self, input_state):
        self.critic_target_ph = tf.placeholder('float32', [None], name='target')
        self.wv, self.bv, self.output_layer_v = layers.fc(
            'fc_value4', input_state, 1, activation='linear')
        # Advantage critic
        self.adv_critic = tf.subtract(self.critic_target_ph, tf.reshape(self.output_layer_v, [-1]))
        # Critic loss
        self.critic_loss = self._value_function_loss(self.adv_critic)
        return self.critic_loss

    def get_action(self, session, state, lstm_state=None):
        feed_dict = {self.input_ph: [state]}
        if lstm_state is not None:
            feed_dict[self.step_size] = [1]
            feed_dict[self.initial_lstm_state] = lstm_state

            action, logits, lstm_state = session.run([
                self.sample_action,
                self.logits,
                self.lstm_state], feed_dict=feed_dict)

            return action, logits[0], lstm_state
        else:
            action, logits = session.run([
                self.sample_action,
                self.logits], feed_dict=feed_dict)

            return action, logits[0]


    def get_action_and_value(self, session, state, lstm_state=None):
        feed_dict = {self.input_ph: [state]}
        if lstm_state is not None:
            feed_dict[self.step_size] = [1]
            feed_dict[self.initial_lstm_state] = lstm_state

            action, logits, v, lstm_state = session.run([
                self.sample_action,
                self.logits,
                self.output_layer_v,
                self.lstm_state], feed_dict=feed_dict)

            return action, v[0, 0], logits[0], lstm_state

        else:
            action, logits, v = session.run([
                self.sample_action,
                self.logits,
                self.output_layer_v], feed_dict=feed_dict)

            return action, v[0, 0], logits[0]


class PolicyNetwork(PolicyValueNetwork):
    def __init__(self, conf,):
        super(PolicyNetwork, self).__init__(conf, use_value_head=False)


class PolicyRepeatNetwork(PolicyValueNetwork):
 
    def __init__(self, conf):
        '''
        Extends action space to parametrize a discrete distribution over repetion lengths
        for each original action
        '''
        self.max_repeat = 5
        super(PolicyRepeatNetwork, self).__init__(conf)
        

    def _build_policy_head(self, input_state):
        self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

        self.wpi, self.bpi, self.output_layer_pi, self.log_output_layer_pi = layers.softmax_and_log_softmax(
            'action_policy', input_state, self.num_actions)

        self.w_ar, self.b_ar, self.action_repeat_probs, self.log_action_repeat_probs = layers.softmax_and_log_softmax(
            'repeat_policy', input_state, self.max_repeat)

        self.selected_repeat = tf.placeholder(tf.int32, [None], name='selected_repeat_placeholder')
        self.selected_repeat_onehot = tf.one_hot(self.selected_repeat, self.max_repeat)

        self.selected_repeat_prob = tf.reduce_sum(self.action_repeat_probs * self.selected_repeat_onehot, 1)
        self.log_selected_repeat_prob = tf.reduce_sum(self.log_action_repeat_probs * self.selected_repeat_onehot, 1)

        # Entropy: ∑_a[-p_a ln p_a]
        self.output_layer_entropy = tf.reduce_sum(
            - 1.0 * tf.multiply(
                tf.expand_dims(self.output_layer_pi, 2) * tf.expand_dims(self.action_repeat_probs, 1),
                tf.expand_dims(self.log_output_layer_pi, 2) + tf.expand_dims(self.log_action_repeat_probs, 1)
            ), axis=[1, 2])

        self.entropy = tf.reduce_sum(self.output_layer_entropy)

        self.log_output_selected_action = tf.reduce_sum(
            self.log_output_layer_pi*self.selected_action_ph,
            axis=1) + self.log_selected_repeat_prob

        self.actor_objective = -tf.reduce_sum(
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

        
    def _build_policy_head(self, input_state):
        self.adv_actor_ph = tf.placeholder("float", [self.batch_size], name='advantage')

        with tf.variable_scope(self.name+'/lstm_decoder') as vs:
            self.action_outputs = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='action_outputs')
            self.action_inputs = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='action_inputs')
                
            self.decoder_seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name='decoder_seq_lengths')
            self.allowed_actions = tf.placeholder(tf.float32, [self.batch_size, None, self.num_actions+1], name='allowed_actions')
            self.use_fixed_action = tf.placeholder(tf.bool, name='use_fixed_action')
            self.temperature = tf.placeholder(tf.float32, name='temperature')

            self.decoder_hidden_state_size = input_state.get_shape().as_list()[-1]
            self.decoder_lstm_cell = CustomBasicLSTMCell(self.decoder_hidden_state_size, forget_bias=1.0)
            self.decoder_initial_state = tf.placeholder(tf.float32, [self.batch_size, 2*self.decoder_hidden_state_size], name='decoder_initial_state')

            self.network_state = tf.concat(axis=1, values=[
                tf.zeros_like(input_state), input_state
                # input_state, tf.zeros_like(input_state)
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
        self.output_layer_entropy = - tf.reduce_sum(tf.stop_gradient(1 + log_sequence_probs) * log_sequence_probs)
        self.entropy = - tf.reduce_sum(log_sequence_probs)

        print 'sp, lsp:', sequence_probs.get_shape(), log_sequence_probs.get_shape()


        self.actor_advantage_term = tf.reduce_sum(log_sequence_probs[:self.max_local_steps] * self.adv_actor_ph)
        self.actor_entropy_term = self.beta * self.output_layer_entropy
        self.actor_objective = - (
            self.actor_advantage_term
            + self.actor_entropy_term
        )

        return self.actor_objective
            

