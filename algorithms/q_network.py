from network import *

class QNetwork(Network):
 
    def __init__(self, conf):
        """ Set up remaining layers, loss function, gradient compute and apply 
        ops, network parameter synchronization ops, and summary ops. """

        super(QNetwork, self).__init__(conf)
                
        with tf.name_scope(self.name):
        
            self.target_placeholder = tf.placeholder(
                "float32", [None], name = 'target')
    
            #fc4
            layer_name = 'fc4'
            hiddens = self.num_actions ; dim = 256
            self.w4 = tf.Variable(
                tf.random_normal([dim, hiddens], stddev=0.01), 
                name='' + layer_name + '_weights')
            self.b4 = tf.Variable(
                tf.constant(0.1, shape=[hiddens]), 
                name='' + layer_name + '_biases')
            self.output_layer = tf.add(tf.matmul(self.o3, self.w4), self.b4, 
                name='' + layer_name + '_outputs')
    
            self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, 
                self.w4, self.b4]
                   
            # Loss
            # Multiply the output of the network by a one hot vector 1 for the 
            # executed action. This will make the loss due to non-selected 
            # actions to be zero.
            output_selected_action = tf.reduce_sum(tf.mul(self.output_layer, 
                self.selected_action_placeholder), reduction_indices = 1)
            diff = tf.sub(self.target_placeholder, output_selected_action)

            if self.clip_delta > 0:
                quadratic_part = tf.minimum(tf.abs(diff), 
                    tf.constant(self.clip_delta))
                linear_part = tf.sub(tf.abs(diff), quadratic_part)
                self.loss = 0.5 * tf.pow(quadratic_part,2)
                #\ + self.clip_delta * linear_part
            else:
                self.loss = tf.mul(tf.constant(0.5),tf.pow(diff, 2))
                        
            # Optimizer
            self.grads_and_vars = self.optimizer.compute_gradients(
                self.loss, self.params)
            # This is not really an operation, but a list of gradient Tensors 
            # When calling run() on it, the value of those Tensors 
            # (i.e., of the gradients) will be calculated.
            if self.clip_norm_type == 'ignore':
                # Unclipped gradients
                self.get_gradients = [g for g, _ in self.grads_and_vars]
            elif self.clip_norm_type == 'global':
                # Clip network grads by network norm
                self.get_gradients = tf.clip_by_global_norm(
                    [g for g, _ in self.grads_and_vars], self.clip_norm)[0]
            elif self.clip_norm_type == 'local':
                # Clip layer grads by layer norm
                self.get_gradients = [tf.clip_by_norm(
                    g, self.clip_norm) for g, _ in self.grads_and_vars]

            self.gradient_placeholders = []
            for var in self.params:
                self.gradient_placeholders.append(
                    tf.placeholder(tf.float32, 
                        shape=var.get_shape(), 
                        name = 'gradient_of_{}'.format(
                            (var.name.split("/",1)[1]).replace(":","_"))))
            self.apply_gradients = self.optimizer.apply_gradients(
                zip(self.gradient_placeholders, self.params))

            # Sync
            # The parameters of the target network and the local network 
            # replicas, if used, need to be regularly synchronized with the 
            # parameters of the shared network. We create the operations 
            # for that here.
            if self.shared_network is not None:
                self.sync_parameters_w_shared_network = []
                for i in xrange(len(self.params)):
                    self.sync_parameters_w_shared_network.append(
                        self.params[i].assign(self.shared_network.params[i]))

            # Summary
            if ((self.local_replicas and "replica" in self.name) 
                or ((not self.local_replicas) and "shared" in self.name)): 
                self.loss_summary = tf.scalar_summary(
                    "Loss " + self.name, tf.reshape(self.loss, []))
                self.learning_rate_summary = tf.scalar_summary(
                    "Learning rate for " + self.name, self.learning_rate)
                self.summary_op = tf.merge_summary(
                    [self.loss_summary.name, self.learning_rate_summary.name])
