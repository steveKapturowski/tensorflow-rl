# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import layers
from custom_lstm import CustomBasicLSTMCell


class Network(object):

    def __init__(self, conf):
        """ Initialize hyper-parameters, set up optimizer and network 
        layers common across Q and Policy/V nets. """

        self.name = conf['name']
        self.num_actions = conf['num_act']
        self.arch = conf['args'].arch
        self.batch_size = conf['args'].batch_size
        self.optimizer_type = conf['args'].opt_type
        self.optimizer_mode = conf['args'].opt_mode
        self.clip_loss_delta = conf['args'].clip_loss_delta
        self.clip_norm = conf['args'].clip_norm
        self.clip_norm_type = conf['args'].clip_norm_type
        self.input_shape = conf['input_shape']
        self.use_recurrent = conf['args'].alg_type.endswith('-lstm')

        with tf.name_scope(self.name):
            self.selected_action_ph = tf.placeholder(
                'float32', [self.batch_size, self.num_actions], name='selected_action')
                
            if self.arch == 'FC':
                self.input_ph = tf.placeholder('float32', [self.batch_size]+self.input_shape+[4], name='input')
                self.w1, self.b1, self.o1 = layers.fc('fc1', layers.flatten(self.input_ph), 40, activation='relu')
                self.w2, self.b2, self.o2 = layers.fc('fc2', self.o1, 40, activation='relu')
                self.ox = self.o2
            elif self.arch == 'ATARI-TRPO':
                self.input_ph = tf.placeholder('float32',[self.batch_size, 84, 84, 4], name='input')
                self.w1, self.b1, self.o1 = layers.conv2d('conv1', self.input_ph, 16, 4, 4, 2)
                self.w2, self.b2, self.o2 = layers.conv2d('conv2', self.o1, 16, 4, 16, 2)
                self.w3, self.b3, self.o3 = layers.fc('fc3', layers.flatten(self.o2), 20, activation='relu')
                self.ox = self.o3
            elif self.arch == 'NIPS':
                self.input_ph = tf.placeholder('float32',[self.batch_size, 84, 84, 4], name='input')
                self.w1, self.b1, self.o1 = layers.conv2d('conv1', self.input_ph, 16, 8, 4, 4)
                self.w2, self.b2, self.o2 = layers.conv2d('conv2', self.o1, 32, 4, 16, 2)
                self.w3, self.b3, self.o3 = layers.fc('fc3', layers.flatten(self.o2), 256, activation='relu')
                self.ox = self.o3
            elif self.arch == 'NATURE':
                self.input_ph = tf.placeholder('float32',[self.batch_size, 84, 84, 4], name='input')
                self.w1, self.b1, self.o1 = layers.conv2d('conv1', self.input_ph, 32, 8, 4, 4)
                self.w2, self.b2, self.o2 = layers.conv2d('conv2', self.o1, 64, 4, 32, 2)
                self.w3, self.b3, self.o3 = layers.conv2d('conv3', self.o2, 64, 3, 64, 1)
                self.w4, self.b4, self.o4 = layers.fc('fc4', layers.flatten(self.o3), 512, activation='relu')
                self.ox = self.o4
            else:
                raise Exception('Invalid architecture `{}`'.format(self.arch))


            if self.use_recurrent:
                layer_name = 'lstm_layer'
                self.hidden_state_size = 256
                with tf.variable_scope(self.name+'/'+layer_name) as vs:
                    self.lstm_cell = CustomBasicLSTMCell(self.hidden_state_size, forget_bias=1.0)

                    self.step_size = tf.placeholder(tf.float32, [None], name='step_size')
                    self.initial_lstm_state = tf.placeholder(
                        tf.float32, [None, 2*self.hidden_state_size], name='initital_state')
                    
                    batch_size = tf.shape(self.step_size)[0]
                    ox_reshaped = tf.reshape(self.ox,
                        [batch_size, -1, self.ox.get_shape().as_list()[-1]])

                    lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                        self.lstm_cell,
                        ox_reshaped,
                        initial_state=self.initial_lstm_state,
                        sequence_length=self.step_size,
                        time_major=False,
                        scope=vs)

                    self.ox = tf.reshape(lstm_outputs, [-1,256], name='reshaped_lstm_outputs')

                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in 
                        tf.trainable_variables() if v.name.startswith(vs.name)]


    def _huber_loss(self, diff):
        # DEFINE HUBER LOSS
        if self.clip_loss_delta > 0:
            quadratic_term = tf.minimum(tf.abs(diff), self.clip_loss_delta)
            linear_term = tf.abs(diff) - quadratic_term
            return tf.nn.l2_loss(quadratic_term) + self.clip_loss_delta*linear_term
        else:
            return tf.nn.l2_loss(diff)


    def _clip_grads(self, grads):
        if self.clip_norm_type == 'ignore':
            return grads
        elif self.clip_norm_type == 'global':
            return tf.clip_by_global_norm(grads, self.clip_norm)[0]
        elif self.clip_norm_type == 'avg':
            return tf.clip_by_average_norm(grads, self.clip_norm)[0]
        elif self.clip_norm_type == 'local':
            return [tf.clip_by_norm(g, self.clip_norm)
                    for g in grads]


    def _setup_shared_memory_ops(self):
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


    def _build_gradient_ops(self):
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        grads = tf.gradients(self.loss, self.params)
        self.get_gradients = self._clip_grads(grads)
        self._setup_shared_memory_ops()

        
