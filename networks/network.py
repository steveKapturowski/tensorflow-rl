# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import layers
import utils.ops
from custom_lstm import CustomBasicLSTMCell


class Network(object):

    def __init__(self, conf):
        '''
        Initialize hyper-parameters, set up optimizer and network 
        layers common across Q and Policy/V nets
        '''
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
        self.activation = conf['args'].activation
        self.max_local_steps = conf['args'].max_local_steps
        self.input_channels = 3 if conf['args'].use_rgb else conf['args'].history_length
        self.use_recurrent = 'lstm' in conf['args'].alg_type
        self.fc_layer_sizes = conf['args'].fc_layer_sizes
        self._init_placeholders()


    def _init_placeholders(self):
        with tf.variable_scope(self.name):
            if self.arch == 'FC':
                self.input_ph = tf.placeholder('float32', [self.batch_size]+self.input_shape+[self.input_channels], name='input')
            else: #assume image input
                self.input_ph = tf.placeholder('float32',[self.batch_size, 84, 84, self.input_channels], name='input')

            if self.use_recurrent:
                self.hidden_state_size = 256
                self.step_size = tf.placeholder(tf.float32, [None], name='step_size')
                self.initial_lstm_state = tf.placeholder(
                    tf.float32, [None, 2*self.hidden_state_size], name='initital_state')

            self.selected_action_ph = tf.placeholder(
                'float32', [self.batch_size, self.num_actions], name='selected_action')


    def _build_encoder(self):
        with tf.variable_scope(self.name):
            if self.arch == 'FC':
                layer_i = layers.flatten(self.input_ph)
                for i, layer_size in enumerate(self.fc_layer_sizes):
                    layer_i = layers.fc('fc{}'.format(i+1), layer_i, layer_size, activation=self.activation)[-1]
                self.ox = layer_i
            elif self.arch == 'ATARI-TRPO':
                self.w1, self.b1, self.o1 = layers.conv2d('conv1', self.input_ph, 16, 4, self.input_channels, 2, activation=self.activation)
                self.w2, self.b2, self.o2 = layers.conv2d('conv2', self.o1, 16, 4, 16, 2, activation=self.activation)
                self.w3, self.b3, self.o3 = layers.fc('fc3', layers.flatten(self.o2), 20, activation=self.activation)
                self.ox = self.o3
            elif self.arch == 'NIPS':
                self.w1, self.b1, self.o1 = layers.conv2d('conv1', self.input_ph, 16, 8, self.input_channels, 4, activation=self.activation)
                self.w2, self.b2, self.o2 = layers.conv2d('conv2', self.o1, 32, 4, 16, 2, activation=self.activation)
                self.w3, self.b3, self.o3 = layers.fc('fc3', layers.flatten(self.o2), 256, activation=self.activation)
                self.ox = self.o3
            elif self.arch == 'NATURE':
                self.w1, self.b1, self.o1 = layers.conv2d('conv1', self.input_ph, 32, 8, self.input_channels, 4, activation=self.activation)
                self.w2, self.b2, self.o2 = layers.conv2d('conv2', self.o1, 64, 4, 32, 2, activation=self.activation)
                self.w3, self.b3, self.o3 = layers.conv2d('conv3', self.o2, 64, 3, 64, 1, activation=self.activation)
                self.w4, self.b4, self.o4 = layers.fc('fc4', layers.flatten(self.o3), 512, activation=self.activation)
                self.ox = self.o4
            else:
                raise Exception('Invalid architecture `{}`'.format(self.arch))

            if self.use_recurrent:
                with tf.variable_scope(self.name+'/lstm_layer') as vs:
                    self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                        self.hidden_state_size, state_is_tuple=True, forget_bias=1.0)
                    
                    batch_size = tf.shape(self.step_size)[0]
                    self.ox_reshaped = tf.reshape(self.ox,
                        [batch_size, -1, self.ox.get_shape().as_list()[-1]])
                    state_tuple = tf.contrib.rnn.LSTMStateTuple(
                        *tf.split(self.initial_lstm_state, 2, 1))

                    self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                        self.lstm_cell,
                        self.ox_reshaped,
                        initial_state=state_tuple,
                        sequence_length=self.step_size,
                        time_major=False,
                        scope=vs)

                    self.lstm_state = tf.concat(self.lstm_state, 1)
                    self.ox = tf.reshape(self.lstm_outputs, [-1,self.hidden_state_size], name='reshaped_lstm_outputs')

                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in 
                        tf.trainable_variables() if v.name.startswith(vs.name)]

            return self.ox


    def _value_function_loss(self, diff):
        if self.clip_loss_delta > 0:
            # DEFINE HUBER LOSS
            return 0.5 * tf.reduce_sum(tf.where(
                tf.abs(self.clip_loss_delta) < 1.0, tf.square(diff), tf.sqrt(2*self.clip_loss_delta)*tf.abs(diff)))
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


    def _build_gradient_ops(self, loss):
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.flat_vars = utils.ops.flatten_vars(self.params)

        grads = tf.gradients(loss, self.params)
        self.get_gradients = self._clip_grads(grads)
        self._setup_shared_memory_ops()


    def get_input_shape(self):
        return self.input_ph.get_shape().as_list()[1:]
        
