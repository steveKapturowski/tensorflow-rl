from network import *

class QNetwork(Network):
 
    
    def __init__(self, conf):
        """ Set up remaining layers, loss function, gradient compute and apply 
        ops, network parameter synchronization ops, and summary ops. """

        super(QNetwork, self).__init__(conf)
                
        with tf.name_scope(self.name):
        
            self.target_ph = tf.placeholder(
                "float32", [None], name='target')
    
            if self.arch == "NIPS":
                #fc4
                self.w4, self.b4, self.output_layer = self._fc('fc4', self.o3, self.num_actions, activation="linear")
            
                self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, 
                                           self.w4, self.b4]
            else: #NATURE
                #fc5
                self.w5, self.b5, self.output_layer = self._fc('fc5', self.o4, self.num_actions, activation="linear")
            
                self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, 
                                           self.w4, self.b4, self.w5, self.b5]
                   

            # Loss
            # Multiply the output of the network by a one hot vector 1 for the 
            # executed action. This will make the loss due to non-selected 
            # actions to be zero.
            if "target" not in self.name:
                output_selected_action = tf.reduce_sum(tf.mul(self.output_layer, 
                                                              self.selected_action_ph), reduction_indices = 1)
                
                diff = tf.sub(self.target_ph, output_selected_action)
            
                # HUBER LOSS
                # If we simply take the squared clipped diff as our loss,
                # then the gradient will be zero whenever the diff exceeds
                # the clip bounds. To avoid this, we extend the loss
                # linearly past the clip point to keep the gradient constant
                # in that regime.
                # 
                # This is equivalent to declaring d loss/d q_vals to be
                # equal to the clipped diff, then backpropagating from
                # there, which is what the DeepMind implementation does.
                if self.clip_loss_delta > 0:
                    quadratic_part = tf.minimum(tf.abs(diff), 
                        tf.constant(self.clip_loss_delta))
                    linear_part = tf.sub(tf.abs(diff), quadratic_part)
                    #self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + 
                    #    self.clip_loss_delta * linear_part)
                    self.loss = tf.add(tf.nn.l2_loss(quadratic_part),
                                              tf.mul(tf.constant(self.clip_loss_delta), linear_part))
                else:
                    #self.loss = tf.reduce_mean(0.5 * tf.square(diff))
                    self.loss = tf.nn.l2_loss(diff)
                      
                # Operations to compute gradients
                with tf.control_dependencies(None):
                    grads = tf.gradients(self.loss, self.params)
                    
                    # This is not really an operation, but a list of gradient Tensors 
                    # When calling run() on it, the value of those Tensors 
                    # (i.e., of the gradients) will be calculated.
                    self.clipped_grad_hist_op = None
                    if self.clip_norm_type == 'ignore':
                        self.get_gradients = grads
                    elif self.clip_norm_type == 'global':
                        self.get_gradients = tf.clip_by_global_norm(grads, self.clip_norm)[0]
                    elif self.clip_norm_type == 'avg':
                        self.get_gradients = tf.clip_by_average_norm(grads, self.clip_norm)[0]
                    elif self.clip_norm_type == 'local':
                        self.get_gradients = [
                            tf.clip_by_norm(g, self.clip_norm) for g in grads
                        ]

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

