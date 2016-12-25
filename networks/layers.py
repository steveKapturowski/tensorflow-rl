from tensorflow import tf


def _flatten(_input):
    shape = _input.get_shape().as_list() 
    dim = shape[1]*shape[2]*shape[3] 
    return tf.reshape(_input, [-1,dim], name='_flattened')


def _conv2d(name, _input, filters, size, channels, stride, padding='VALID', init='torch'):
    w = _conv_weight_variable([size,size,channels,filters], 
                                             name + '_weights', init = init)
    b = _conv_bias_variable([filters], size, size, channels,
	                                       name + '_biases', init = init)
	conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1], 
	        padding=padding, name=name + '_convs')
	out = tf.nn.relu(tf.add(conv, b), 
            name='' + name + '_activations')
        
    return w, b, out


def _conv_weight_variable(shape, name, init = 'torch'):
    if init == 'glorot_uniform':
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


def _conv_bias_variable(shape, w, h, input_channels, name, init='torch'):
    if init=='glorot_uniform':
        initial = tf.zeros(shape)
    else:
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def _fc(name, _input, output_dim, activation='relu', init='torch'):
    input_dim = _input.get_shape().as_list()[1]
    w = _fc_weight_variable([input_dim, output_dim], 
                                           name+'_weights', init=init)
    b = _fc_bias_variable([output_dim], input_dim,
                                           name+'_biases', init=init)
    out = tf.add(tf.matmul(_input, w), b, name=name+'_out')
        
    if activation == 'relu':
        out = tf.nn.relu(out, name='' + name + '_relu')

    return w, b, out


def _fc_weight_variable(shape, name, init = 'torch'):
    if init == 'glorot_uniform':
        fan_in = shape[0]
        fan_out = shape[1]
        d = np.sqrt(6. / (fan_in + fan_out))            
    else:
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def _fc_bias_variable(shape, input_channels, name, init='torch'):
    if init=="glorot_uniform":
        initial = tf.zeros(shape, dtype='float32')
    else:
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')  


def _softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = _fc_weight_variable([input_dim, output_dim], name+'_weights')
    b = _fc_bias_variable([output_dim], input_dim, name+'_biases')
    out = tf.nn.softmax(tf.add(tf.matmul(_input, w), b), name= name+'_policy')
 
    return w, b, out


def _softmax_and_log_softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = _fc_weight_variable([input_dim, output_dim], name+'_weights')
    b = _fc_bias_variable([output_dim], input_dim, name+'_biases')
    xformed = tf.matmul(_input, w) + b
    out = tf.nn.softmax(xformed, name=name+'_policy')
    log_out = tf.nn.log_softmax(xformed, name=name+'_log_policy')
 
    return w, b, out, log_out





# loop_function = sample_gumbel_softmax_and_embed((W_pi, b_pi), temperature=0.5, output_projection=None):
def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
    '''
    RNN decoder for the sequence-to-sequence model.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor with shape [batch_size x cell.state_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      loop_function: If not None, this function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
              * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
          (Note that in some cases, like basic RNN cell or GRU cell, outputs and
           states can be the same. They are different for LSTM cells though.)
    '''
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output

    return outputs, state





 