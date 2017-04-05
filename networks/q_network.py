# -*- coding: utf-8 -*-
import layers
import tensorflow as tf
from network import Network


class QNetwork(Network):

    def __init__(self, conf):
        """ Set up remaining layers, loss function, gradient compute and apply 
        ops, network parameter synchronization ops, and summary ops. """

        super(QNetwork, self).__init__(conf)
                
        with tf.variable_scope(self.name):
            self.target_ph = tf.placeholder('float32', [None], name='target')
            self.loss = self._build_q_head()
            self._build_gradient_ops()


    def _build_q_head(self):
        self.w_out, self.b_out, self.output_layer = layers.fc('fc_out', self.ox, self.num_actions, activation="linear")
        self.q_selected_action = tf.reduce_sum(self.output_layer * self.selected_action_ph, axis=1)

        diff = tf.subtract(self.target_ph, self.q_selected_action)
        return self._huber_loss(diff)

