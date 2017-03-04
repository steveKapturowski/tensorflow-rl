# -*- coding: utf-8 -*-
import layers
import tensorflow as tf
from network import Network


class DuelingNetwork(Network):
 
    def __init__(self, conf):
        """ Set up remaining layers, loss function, gradient compute and apply 
        ops, network parameter synchronization ops, and summary ops. """

        super(DuelingNetwork, self).__init__(conf)
                
        with tf.name_scope(self.name):
        
            self.target_ph = tf.placeholder('float32', [None], name='target')
    
            o_conv = self.o3 if self.arch == 'NIPS' else self.o4


            self.w_value, self.b_value, self.value = layers.fc(
                'fc5', o_conv, 1, activation='linear')
            self.w_adv, self.b_adv, self.advantage = layers.fc(
                'fc6', o_conv, self.num_actions, activation='linear')
            
            self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            
            self.output_layer = (
                self.value + self.advantage
                - tf.reduce_mean(
                    self.advantage,
                    axis=1,
                    keep_dims=True
                )
            )

            # Loss
            # Multiply the output of the network by a one hot vector 1 for the 
            # executed action. This will make the loss due to non-selected 
            # actions to be zero.
            if 'target' not in self.name:
                output_selected_action = tf.reduce_sum(tf.multiply(self.output_layer, 
                                                              self.selected_action_ph), axis=1)
                
                diff = tf.subtract(self.target_ph, output_selected_action)
            
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
                    linear_part = tf.subtract(tf.abs(diff), quadratic_part)
                    #self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + 
                    #    self.clip_loss_delta * linear_part)
                    self.loss = tf.add(tf.nn.l2_loss(quadratic_part),
                                              tf.multiply(tf.constant(self.clip_loss_delta), linear_part))
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
                        # Unclipped gradients
                        self.get_gradients = grads
                        #self.get_gradients = [g for g, _ in self.grads_and_vars]
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