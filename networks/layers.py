import tensorflow as tf
import numpy as np


def flatten(_input):
    shape = _input.get_shape().as_list()
    dim = reduce(lambda a, b: a*b, shape[1:])
    return tf.reshape(_input, [-1, dim], name='_flattened')
            
def conv2d(name, _input, filters, size, channels, stride, padding='VALID'):
    w = conv_weight_variable([size,size,channels,filters], name+'_weights')
    b = conv_bias_variable([filters], size, size, channels, name+'_biases')
    conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1], 
            padding=padding, name=name+'_convs') + b

    out = tf.nn.relu(conv, name=name+'_activations')
        
    return w, b, out


def conv_weight_variable(shape, name):
    receptive_field_size = np.prod(shape[:2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
    d = np.sqrt(6. / (fan_in + fan_out))            
        
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def conv_bias_variable(shape, w, h, input_channels, name):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name=name, dtype='float32')

def fc(name, _input, output_dim, activation='relu'):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name+'_weights')
    b = fc_bias_variable([output_dim], input_dim, name+'_biases')
    out = tf.matmul(_input, w) + b

    if activation == 'relu':
        out = tf.nn.relu(out, name=name+'_relu')
    elif activation == 'softplus':
        out = tf.nn.softplus(out, name=name+'_softplus')
    elif activation == 'tanh':
        out = tf.nn.tanh(out, name=name+'_tanh')
    #else assume linear

    return w, b, out
    
def fc_weight_variable(shape, name):
    fan_in = shape[0]
    fan_out = shape[1]
    d = np.sqrt(6. / (fan_in + fan_out))            
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')
    
def fc_bias_variable(shape, input_channels, name):
    initial = tf.zeros(shape, dtype='float32')
    return tf.Variable(initial, name=name, dtype='float32')  
    
def softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name+'_weights')
    b = fc_bias_variable([output_dim], input_dim, name+'_biases')
    out = tf.nn.softmax(tf.add(tf.matmul(_input, w), b), name=name+'_policy')
 
    return w, b, out
    
def softmax_and_log_softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name+'_weights')
    b = fc_bias_variable([output_dim], input_dim, name+'_biases')
    xformed = tf.matmul(_input, w) + b
    out = tf.nn.softmax(xformed, name=name+'_policy')
    log_out = tf.nn.log_softmax(xformed, name=name+'_log_policy')

    return w, b, out, log_out


