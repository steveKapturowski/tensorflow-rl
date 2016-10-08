# A3C -- in progress!
from network import *
from custom_lstm import CustomBasicLSTMCell

class PolicyVNetwork(Network):
 
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient 
        compute and apply ops, network parameter synchronization ops, and 
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)
        
        self.entropy_regularisation_strength = \
                conf['args'].entropy_regularisation_strength
        

        self.use_recurrent = conf['args'].alg_type == 'a3c-lstm'
                
        with tf.name_scope(self.name):

            self.critic_target_ph = tf.placeholder(
                'float32', [None], name='target')
            self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

            # LSTM layer with 256 cells
            # f = sigmoid(Wf * [h-, x] + bf) 
            # i = sigmoid(Wi * [h-, x] + bi) 
            # C' = sigmoid(Wc * [h-, x] + bc) 
            # o = sigmoid(Wo * [h-, x] + bo)
            # C = f * C_ +  i x C'
            # h = o * tan C
            # state = C
            # o4 = x
            if self.use_recurrent:
                layer_name = 'lstm_layer'
                self.hidden_state_size = 256
                with tf.variable_scope(self.name+'/'+layer_name) as vs:
                    self.lstm_cell = CustomBasicLSTMCell(self.hidden_state_size, forget_bias=1.0)

                    self.step_size = tf.placeholder(tf.float32, [1], name='step_size')
                    self.initial_lstm_state = tf.placeholder(
                        tf.float32, [1, 2*self.hidden_state_size], name='initital_state')
                    
                    o3_reshaped = tf.reshape(self.o3, [1,-1,256])
                    lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                        self.lstm_cell,
                        o3_reshaped,
                        initial_state=self.initial_lstm_state,
                        sequence_length=self.step_size,
                        time_major=False,
                        scope=vs)

                    self.ox = tf.reshape(lstm_outputs, [-1,256])

                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in 
                        tf.trainable_variables() if v.name.startswith(vs.name)]
            else:
                if self.arch == 'NIPS':
                    self.ox = self.o3
                else: #NATURE
                    self.ox = self.o4

            # Final actor layer
            layer_name = 'softmax_policy4'            
            self.wpi, self.bpi, self.output_layer_pi = self._softmax(
                layer_name, self.ox, self.num_actions)
            
            # Avoiding log(0) by adding a very small quantity (1e-30) to output.
            self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi, 
                tf.constant(1e-30)), name=layer_name+'_log_policy')
            
            # Entropy: sum_a (-p_a ln p_a)
            self.output_layer_entropy = tf.reduce_sum(tf.mul(
                tf.constant(-1.0), 
                tf.mul(self.output_layer_pi, self.log_output_layer_pi)), reduction_indices=1)
            
            # Final critic layer
            self.wv, self.bv, self.output_layer_v = self._fc(
                'fc_value4', self.ox, 1, activation='linear')


            if self.arch == 'NIPS':
                self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, 
                    self.b3, self.wpi, self.bpi, self.wv, self.bv]
            else: #NATURE
                self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, 
                    self.b3, self.w4, self.b4, self.wpi, self.bpi, self.wv, self.bv]
                

            if self.use_recurrent:
                self.params += self.lstm_trainable_variables
 

            # Advantage critic
            self.adv_critic = tf.sub(self.critic_target_ph, tf.reshape(self.output_layer_v, [-1]))
            
            # Actor objective
            # Multiply the output of the network by a one hot vector, 1 for the 
            # executed action. This will make the non-regularised objective 
            # term for non-selected actions to be zero.
            log_output_selected_action = tf.reduce_sum(
                tf.mul(self.log_output_layer_pi, self.selected_action_ph), 
                reduction_indices=1
            )
            actor_objective_advantage_term = tf.mul(
                log_output_selected_action, self.adv_actor_ph
            )
            actor_objective_entropy_term = tf.mul(
                self.entropy_regularisation_strength, self.output_layer_entropy
            )
            self.actor_objective = -tf.reduce_sum(
                actor_objective_advantage_term
                + actor_objective_entropy_term
            )
            
            # Critic loss
            if self.clip_loss_delta > 0:
                quadratic_part = tf.minimum(tf.abs(self.adv_critic), 
                    tf.constant(self.clip_loss_delta))
                linear_part = tf.sub(tf.abs(self.adv_critic), quadratic_part)
                #OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = tf.mul(tf.constant(0.5), tf.nn.l2_loss(quadratic_part) + \
                    self.clip_loss_delta * linear_part)
            else:
                #OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = tf.mul(tf.constant(0.5), tf.nn.l2_loss(self.adv_critic))               
            
            self.loss = self.actor_objective + self.critic_loss
            
            # Optimizer
            grads = tf.gradients(self.loss, self.params)

            # This is not really an operation, but a list of gradient Tensors. 
            # When calling run() on it, the value of those Tensors 
            # (i.e., of the gradients) will be calculated
            if self.clip_norm_type == 'ignore':
                # Unclipped gradients
                self.get_gradients = grads
            elif self.clip_norm_type == 'global':
                # Clip network grads by network norm
                self.get_gradients = tf.clip_by_global_norm(
                            grads, self.clip_norm)[0]
            elif self.clip_norm_type == 'local':
                # Clip layer grads by layer norm
                self.get_gradients = [tf.clip_by_norm(
                            g, self.clip_norm) for g in grads]

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


            
