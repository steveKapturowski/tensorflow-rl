import tensorflow as tf
import numpy as np


class Network(object):

    def __init__(self, conf):
        """ Initialize hyper-parameters, set up optimizer and network 
        layers common across Q and Policy/V nets. """
        
        self.name = conf['name']
        self.num_actions = conf['num_act']
        self.arch = conf['args'].arch
        self.optimizer_type = conf['args'].opt_type
        self.optimizer_mode = conf['args'].opt_mode
        self.clip_loss_delta = conf['args'].clip_loss_delta
        self.clip_norm = conf['args'].clip_norm
        self.clip_norm_type = conf['args'].clip_norm_type

        with tf.name_scope(self.name):
            
            self.input_ph = tf.placeholder(
                'float32',[None,84,84,4], name = 'input')
            self.selected_action_ph = tf.placeholder(
                'float32', [None, self.num_actions], name = 'selected_action')

            if self.optimizer_type == 'adam':
                init= 'glorot_uniform'
            else: 
                init = 'torch'
                
            if self.arch == 'NIPS':
                #conv1
                self.w1, self.b1, self.o1 = self._conv2d('conv1', self.input_ph, 16, 8, 4, 4, init=init)
    
                #conv2
                self.w2, self.b2, self.o2 = self._conv2d('conv2', self.o1, 32, 4, 16, 2, init=init)
    
                #fc3
                self.w3, self.b3, self.o3 = self._fc('fc3', self._flatten(self.o2), 256, activation='relu', init=init)
            else: #NATURE
                #conv1
                self.w1, self.b1, self.o1 = self._conv2d('conv1', self.input_ph, 32, 8, 4, 4, init=init)
    
                #conv2
                self.w2, self.b2, self.o2 = self._conv2d('conv2', self.o1, 64, 4, 32, 2, init=init)
    
                #conv3
                self.w3, self.b3, self.o3 = self._conv2d('conv3', self.o2, 64, 3, 64, 1, init=init)
    
                #fc4
                self.w4, self.b4, self.o4 = self._fc('fc4', self._flatten(self.o3), 512, activation='relu', init=init)
                

    
    
    
    def _flatten(self, _input):
        shape = _input.get_shape().as_list() 
        dim = shape[1]*shape[2]*shape[3] 
        return tf.reshape(_input, [-1,dim], name='_flattened')
            
    def _conv2d(self, name, _input, filters, size, channels, stride, padding='VALID', init="torch"):
        w = self._conv_weight_variable([size,size,channels,filters], 
                                                 name + '_weights', init = init)
        b = self._conv_bias_variable([filters], size, size, channels,
                                               name + '_biases', init = init)
        conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1], 
                padding=padding, name=name + '_convs')
        out = tf.nn.relu(tf.add(conv, b), 
                name='' + name + '_activations')
        
        return w, b, out


    def _conv_weight_variable(self, shape, name, init = "torch"):
        if init == "glorot_uniform":
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
            d = np.sqrt(6. / (fan_in + fan_out))            
        else:
            w = shape[0]
            h = shape[1]
            input_channels = shape[2]
            d = 1.0 / np.sqrt(input_channels * w * h)
        
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')



    def _conv_bias_variable(self, shape, w, h, input_channels, name, init="torch"):
        if init=="glorot_uniform":
            initial = tf.zeros(shape)
        else:
            d = 1.0 / np.sqrt(input_channels * w * h)
            initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')

    def _fc(self, name, _input, output_dim, activation="relu", init="torch"):
        input_dim = _input.get_shape().as_list()[1]
        w = self._fc_weight_variable([input_dim, output_dim], 
                                               name + '_weights', init=init)
        b = self._fc_bias_variable([output_dim], input_dim,
                                               '' + name + '_biases', init=init)
        out = tf.add(tf.matmul(_input, w), b, name= name + '_out')
        
        if activation == "relu":
            out = tf.nn.relu(out, name='' + name + '_relu')

        return w, b, out
    
    def _fc_weight_variable(self, shape, name, init = "torch"):
        if init == "glorot_uniform":
            fan_in = shape[0]
            fan_out = shape[1]
            d = np.sqrt(6. / (fan_in + fan_out))            
        else:
            input_channels = shape[0]
            d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')
    
    def _fc_bias_variable(self, shape, input_channels, name, init= "torch"):
        if init=="glorot_uniform":
            initial = tf.zeros(shape, dtype='float32')
        else:
            d = 1.0 / np.sqrt(input_channels)
            initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')  
    
    def _softmax(self, name, _input, output_dim):
        input_dim = _input.get_shape().as_list()[1]
        w = self._fc_weight_variable([input_dim, output_dim], name + '_weights')
        b = self._fc_bias_variable([output_dim], input_dim, name + '_biases')
        out = tf.nn.softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
 
        return w, b, out
    
    def _softmax_and_log_softmax(self, name, _input, output_dim):
        input_dim = _input.get_shape().as_list()[1]
        w = self._fc_weight_variable([input_dim, output_dim], name + '_weights')
        b = self._fc_bias_variable([output_dim], input_dim, name + '_biases')
        xformed = tf.matmul(_input, w) + b
        out = tf.nn.softmax(xformed, name= name + '_policy')
        log_out = tf.nn.log_softmax(xformed, name= name + '_log_policy')
 
        return w, b, out, log_out
 
  
        
