import tensorflow as tf

class Network(object):

    def __init__(self, conf):
        """ Initialize hyper-parameters, set up optimizer and network 
        layers common across Q and Policy/V nets. """
        
        optimizer_conf = conf['optimizer_conf']
        self.name = conf['name']
        self.local_replicas = conf['local_replicas']
        self.shared_network = conf['shared_network']
        self.num_actions = conf['num_act']
        self.clip_delta = optimizer_conf['clip_delta']
        self.base_learning_rate = optimizer_conf['base_learning_rate']
        self.lr_decay_step = optimizer_conf['lr_decay_step']
        self.lr_decay_rate = optimizer_conf['lr_decay_rate']
        self.lr_staircase = optimizer_conf['lr_staircase']
        self.clip_norm = optimizer_conf['clip_norm']
        self.clip_norm_type = optimizer_conf['clip_norm_type']
        self.optimizer_type = optimizer_conf["type"]
        self.optimizer_mode = optimizer_conf["mode"] or "local"
        self.global_step = conf['global_step']

        with tf.name_scope(self.name):
            
            self.input_placeholder = tf.placeholder(
                'float32',[None,84,84,4], name = 'input')
            self.selected_action_placeholder = tf.placeholder(
                "float", [None, self.num_actions], name = "selected_action")

            self.learning_rate = tf.train.exponential_decay(
                                                    self.base_learning_rate,
                                                    self.global_step,  
                                                    self.lr_decay_step,     
                                                    self.lr_decay_rate,     
                                                    staircase=self.lr_staircase)

            if self.optimizer_type == "standard": 
                self.optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)
            if self.optimizer_type == "momentum":
                if self.optimizer_mode == "shared": 
                    self.optimizer = tf.train.MomentumOptimizer(
                        self.learning_rate, momentum=0.9)
                else: # Distributed mode
                    self.optimizer = tf.train.GradientDescentOptimizer(
                        self.learning_rate)
            if self.optimizer_type == "rmsprop":
                if self.optimizer_mode == "shared":
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, decay=0.99, epsilon=1e-6)
                else:
                    self.optimizer = tf.train.GradientDescentOptimizer(
                        self.learning_rate)
            
            #conv1
            layer_name = 'conv1'
            size = 8 ; channels = 4 ; filters = 16 ; stride = 4
            self.w1 = tf.Variable(
                tf.random_normal([size,size,channels,filters], stddev=0.01), 
                name='' + layer_name + '_weights')
            self.b1 = tf.Variable(tf.constant(0.1, 
                shape=[filters]), 
                name='' + layer_name + '_biases')
            self.c1 = tf.nn.conv2d(self.input_placeholder, self.w1, 
                strides=[1, stride, stride, 1], 
                padding='VALID', 
                name='' + layer_name + '_convs')
            self.o1 = tf.nn.relu(tf.add(self.c1, self.b1), 
                name='' + layer_name + '_activations')

            #conv2
            layer_name = 'conv2'
            size = 4 ; channels = 16 ; filters = 32 ; stride = 2
            self.w2 = tf.Variable(
                tf.random_normal([size,size,channels,filters], stddev=0.01), 
                name='' + layer_name + '_weights')
            self.b2 = tf.Variable(tf.constant(0.1, 
                shape=[filters]), 
                name='' + layer_name + '_biases')
            self.c2 = tf.nn.conv2d(self.o1, self.w2, 
                strides=[1, stride, stride, 1], 
                padding='VALID', 
                name='' + layer_name + '_convs')
            self.o2 = tf.nn.relu(tf.add(self.c2, self.b2), 
                name='' + layer_name + '_activations')

            # flatten
            o2_shape = self.o2.get_shape().as_list()
    
            #fc3
            layer_name = 'fc3'
            hiddens = 256 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3]
            self.o2_flat = tf.reshape(self.o2, [-1,dim], 
                name='' + layer_name + '_input_flat')
            self.w3 = tf.Variable(
                tf.random_normal([dim, hiddens], stddev=0.01), 
                name='' + layer_name + '_weights')
            self.b3 = tf.Variable(
                tf.constant(0.1, shape=[hiddens]), 
                name='' + layer_name + '_biases')
            self.ip3 = tf.add(tf.matmul(self.o2_flat, self.w3), self.b3, 
                name='' + layer_name + '_ips')
            self.o3 = tf.nn.relu(self.ip3, 
                name='' + layer_name + '_activations')
