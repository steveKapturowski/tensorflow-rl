import tensorflow as tf
import numpy as np


def flatten(_input):
    shape = _input.get_shape().as_list()
    dim = reduce(lambda a, b: a*b, shape[1:])
    return tf.reshape(_input, [-1, dim], name='_flattened')

def apply_activation(out, name, activation):
    if activation == 'relu':
        return tf.nn.relu(out, name=name+'_relu')
    elif activation == 'softplus':
        return tf.nn.softplus(out, name=name+'_softplus')
    elif activation == 'tanh':
        return tf.nn.tanh(out, name=name+'_tanh')
    elif activation == 'selu':
        return selu(out, name=name+'_selu')
    elif activation == 'linear':
        return out
    else:
        raise Exception('Invalid activation type \'{}\''.format(activation))

def selu(x, name):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>0.0, x, alpha*tf.exp(x)-alpha)

def conv2d(name, _input, filters, size, channels, stride, activation='relu', padding='VALID', data_format='NHWC'):
    if data_format == 'NHWC':
        strides = [1, stride, stride, 1]
    else:
        strides = [1, 1, stride, stride]

    w = conv_weight_variable([size, size, channels, filters], name+'_weights')
    b = conv_bias_variable([filters], name+'_biases')
    conv = tf.nn.conv2d(_input, w, strides=strides,
            padding=padding, data_format=data_format, name=name+'_convs') + b

    out = apply_activation(conv, name, activation)
    return w, b, out

def conv_weight_variable(shape, name):
    # initializer=tf.contrib.layers.xavier_initializer()
    # initializer = tf.truncated_normal_initializer(0, 0.02)
    d = 1.0 / np.sqrt(np.prod(shape[:-1]))
    initializer = tf.random_uniform_initializer(-d, d)
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=initializer)

def conv_bias_variable(shape, name):
    initializer = tf.zeros_initializer()
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=initializer)

def fc(name, _input, output_dim, activation='relu'):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name+'_weights')
    b = fc_bias_variable([output_dim], input_dim, name+'_biases')
    out = tf.matmul(_input, w) + b

    out = apply_activation(out, name, activation)
    return w, b, out

def fc_weight_variable(shape, name):
    # initializer = tf.contrib.layers.xavier_initializer()
    # initializer = tf.random_normal_initializer(stddev=0.02)
    d = 1.0 / np.sqrt(shape[0])
    initializer = tf.random_uniform_initializer(-d, d)
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=initializer)

def fc_bias_variable(shape, input_channels, name):
    initializer = tf.zeros_initializer()
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=initializer)

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