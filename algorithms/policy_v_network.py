from network import *

class PolicyVNetwork(Network):
 
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient 
        compute and apply ops, network parameter synchronization ops, and 
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)
        
        self.entropy_regularisation_strength = \
                conf['optimizer_conf']['entropy_regularisation_strength']
        
        # Toggle additional recurrent layer
        recurrent_layer = False
                
        with tf.name_scope(self.name):

            self.critic_target_placeholder = tf.placeholder(
                "float32", [None], name = 'target')

            # LSTM layer with 256 cells
            # f = sigmoid(Wf * [h-, x] + bf) 
            # i = sigmoid(Wi * [h-, x] + bi) 
            # C' = sigmoid(Wc * [h-, x] + bc) 
            # o = sigmoid(Wo * [h-, x] + bo)
            # C = f * C_ +  i x C'
            # h = o * tan C
            # state = C
            # o4 = x
            if recurrent_layer:
                layer_name = 'lstm_layer' ; hiddens = 256 ; dim = 256
                with tf.variable_scope(self.name+'/'+layer_name) as vs:
                    self.lstm_cell = tf.nn.rnn_cell.LSTMCell(hiddens, dim)
                    self.lstm_cell_state = tf.Variable(
                        tf.zeros([1, self.lstm_cell.state_size]))
                    self.ox, self.lstm_cell_state = self.lstm_cell(
                        self.o3, self.lstm_cell_state)
                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in 
                        tf.trainable_variables() if v.name.startswith(vs.name)]
            else:
                self.ox = self.o3

            # softmax actor
            layer_name = 'softmax_policy4' 
            hiddens = self.num_actions; dim = 256
            self.w4p = tf.Variable(
                tf.random_normal([dim, hiddens], stddev=0.01), 
                name='' + layer_name + '_weights')
            self.b4p = tf.Variable(
                tf.constant(0.1, shape=[hiddens]), 
                name='' + layer_name + '_biases')
            self.output_layer_p = tf.nn.softmax(
                tf.matmul(self.ox, self.w4p) + self.b4p, 
                name='' + layer_name + '_policy') 
            # Avoiding log(0) by adding a very small quantity (1e-30) to output. 
            self.log_output_layer_p = tf.log(
                tf.add(self.output_layer_p, 
                    tf.constant(1e-30, shape=[hiddens])), 
                name= '' + layer_name + '_log_policy')
            # Entropy: sum_a (-p_a ln p_a)
            self.output_layer_entropy = tf.reduce_sum(
                tf.mul(tf.constant(-1.0, shape=[hiddens]), 
                    tf.mul(self.output_layer_p, 
                        tf.log(tf.add(self.output_layer_p, 
                            tf.constant(1e-30, shape=[hiddens]))))))

            # fc/linear value critic
            layer_name = 'fc_value4'
            hiddens = 1; dim = 256
            self.w4v = tf.Variable(
                tf.random_normal([dim, hiddens], stddev=0.01), 
                name='' + layer_name + '_weights')
            self.b4v = tf.Variable(
                tf.constant(0.1, shape=[hiddens]), 
                name='' + layer_name + '_biases')
            self.output_layer_v = tf.add(
                tf.matmul(self.ox, self.w4v), self.b4v, 
                name= '' + layer_name + '_value')

            self.actor_params = [self.w1, self.b1, self.w2, self.b2, self.w3, 
                self.b3, self.w4p, self.b4p]
            self.critic_params = [self.w1, self.b1, self.w2, self.b2, self.w3, 
                self.b3, self.w4v, self.b4v]

            if recurrent_layer:
                self.actor_params += self.lstm_trainable_variables
                self.critic_params += self.lstm_trainable_variables
 
            # Advantage
            advantage = tf.sub(self.critic_target_placeholder, 
                self.output_layer_v)
            
            # Actor objective
            # Multiply the output of the network by a one hot vector, 1 for the 
            # executed action. This will make the non-regularised objective 
            # term for non-selected actions to be zero.
            log_output_selected_action = tf.reduce_sum(
                tf.mul(self.log_output_layer_p, 
                    self.selected_action_placeholder), 
                reduction_indices = 1)
            actor_objective_advantage_term = tf.mul(
                log_output_selected_action, advantage)
            actor_objective_entropy_term = tf.mul(
                self.entropy_regularisation_strength, self.output_layer_entropy)
            self.actor_objective = tf.mul(
                -1.0, tf.add(actor_objective_advantage_term, 
                    actor_objective_entropy_term))
            
            # Critic loss
            if self.clip_delta > 0:
                quadratic_part = tf.minimum(tf.abs(advantage), 
                    tf.constant(self.clip_delta))
                linear_part = tf.sub(tf.abs(advantage), quadratic_part)
                self.critic_loss = 0.5 * tf.pow(quadratic_part, 2) + \
                    self.clip_delta * linear_part
            else:
                self.critic_loss = tf.mul(tf.constant(0.5), 
                    tf.pow(advantage, 2))
            
            # Optimizer
            self.actor_grads_and_vars = self.optimizer.compute_gradients(
                self.actor_objective, self.actor_params)
            self.critic_grads_and_vars = self.optimizer.compute_gradients(
                self.critic_loss, self.critic_params)

            # This is not really an operation, but a list of gradient Tensors. 
            # When calling run() on it, the value of those Tensors 
            # (i.e., of the gradients) will be calculated
            if self.clip_norm_type == 'ignore':
                # Unclipped gradients
                self.get_actor_gradients = [g 
                    for g, _ in self.actor_grads_and_vars]
                self.get_critic_gradients = [g 
                    for g, _ in self.critic_grads_and_vars]
            elif self.clip_norm_type == 'global':
                # Clip network grads by network norm
                self.get_actor_gradients = tf.clip_by_global_norm(
                    [g for g, _ in self.actor_grads_and_vars], self.clip_norm)[0]
                self.get_critic_gradients = tf.clip_by_global_norm(
                    [g for g, _ in self.critic_grads_and_vars], self.clip_norm)[0]
            elif self.clip_norm_type == 'local':
                # Clip layer grads by layer norm
                self.get_actor_gradients = [tf.clip_by_norm(
                    g, self.clip_norm) for g, _ in self.actor_grads_and_vars]
                self.get_critic_gradients = [tf.clip_by_norm(
                    g, self.clip_norm) for g, _ in self.critic_grads_and_vars]

            self.actor_gradient_placeholders = []
            for var in self.actor_params:
                self.actor_gradient_placeholders.append(
                    tf.placeholder(tf.float32, 
                        shape=var.get_shape(), 
                        name='gradient_of_{}'.format(
                            (var.name.split("/",1)[1]).replace(":","_"))))
            self.apply_actor_gradients = self.optimizer.apply_gradients(
                zip(self.actor_gradient_placeholders, self.actor_params))

            self.critic_gradient_placeholders = []
            for var in self.critic_params:
                self.critic_gradient_placeholders.append(
                    tf.placeholder(tf.float32, 
                        shape=var.get_shape(), 
                        name = 'gradient_of_{}'.format(
                            (var.name.split("/",1)[1]).replace(":","_"))))
            self.apply_critic_gradients = self.optimizer.apply_gradients(
                zip(self.critic_gradient_placeholders, self.critic_params))
            
            # Group gradient application ops
            with tf.device("/cpu:0"):
                self.apply_gradients = tf.group(*[self.apply_actor_gradients, 
                    self.apply_critic_gradients], 
                    name='' + "_apply_gradients_op")

            # Sync
            # The parameters of the target network and the local network 
            # replicas, if used, need to be regularly synchronized with the 
            # parameters of the shared network. We create the operations 
            # for that here.
            if self.shared_network is not None:
                self.sync_parameters_w_shared_network = []
                for i in xrange(len(self.actor_params)):
                    self.sync_parameters_w_shared_network.append(
                        self.actor_params[i].assign(
                            self.shared_network.actor_params[i]))
                for i in xrange(len(self.critic_params)):
                    self.sync_parameters_w_shared_network.append(
                        self.critic_params[i].assign(
                            self.shared_network.critic_params[i]))
          
            # Summary
            if ((self.local_replicas and "replica" in self.name) 
                or ((not self.local_replicas) and "shared" in self.name)): 
                self.critic_loss_summary = tf.scalar_summary(
                    "Critic Loss " + self.name, 
                    tf.reshape(self.critic_loss, []))
                self.actor_objective_summary = tf.scalar_summary(
                    "Actor Objective " + self.name, 
                    tf.reshape(self.actor_objective, []))
                self.learning_rate_summary = tf.scalar_summary(
                    "Learning rate for " + self.name, self.learning_rate)
                self.summary_op = tf.merge_summary(
                    [self.critic_loss_summary.name, 
                    self.actor_objective_summary.name, 
                    self.learning_rate_summary.name])
