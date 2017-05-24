import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicLSTMCell
from network import Network


class FeudalNetwork(Network):
	def __init__(self, conf):
		super(PolicyValueNetwork, self).__init__(conf)

		with tf.variable_scope('manager'):
			self.build_manager()
		with tf.variable_scope('worker'):
			self.build_worker()

	def build_manager(self, input_state):
		self.manager_initial_lstm_state, self.manager_lstm_state, self.manager_out = self._build_lstm()


	def build_worker(self, input_state):
		self.worker_initial_lstm_state, self.worker_lstm_state, self.worker_out = self._build_lstm(input_state)


	def _build_lstm(self, input_state):
		hidden_state_size = 256
		initial_lstm_state = tf.placeholder(
			tf.float32, [None, 2*hidden_state_size], name='initital_state')
		lstm_cell = BasicLSTMCell(hidden_state_size, forget_bias=1.0, state_is_tuple=True)
		
		batch_size = tf.shape(self.step_size)[0]
		ox_reshaped = tf.reshape(input_state,
			batch_size, -1, input_state.get_shape().as_list()[-1]])

		lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
			lstm_cell,
			ox_reshaped,
			initial_state=initial_lstm_state,
			sequence_length=self.step_size,
			time_major=False)

		out = tf.reshape(lstm_outputs, [-1,hidden_state_size], name='reshaped_lstm_outputs')
		return initial_lstm_state, lstm_state, out


class DilatedLSTM(object):
	def __init__(self, inputs, max_steps, num_cores=10, pool_size=10, args*, **kwargs):
		self.shared_cell = BasicLSTMCell(*args, **kwargs)
		self.max_steps = max_steps
		self.num_cores = num_cores
		self.pool_size = pool_size
		self.inputs = inputs
		self._build_ops()


	def _build_ops(self):
		i0 = tf.constant(0, dtype=tf.int32)
		loop_condition = lambda i, inputs, state: tf.less(i, self.max_steps)

		def body(i, inputs, full_state):
			idx = i % self.num_cores
			prev_state = full_state[idx]
			inputs, full_state[idx] = self.shared_cell(inputs, prev_state)

			return i+1, inputs, full_state

		_, inputs, full_state = tf.while_loop(
			loop_condition,
			body,
			loop_vars=[i0,
					   self.inputs,
					   self.get_zero_state()])


		# tf.max_pool(outputs)


	def get_zero_state(self):
		return [self.shared_cell.zero_state(
					tf.shape(self.max_steps)[0],
					tf.float32) for _ in range(self.stride)]
		



