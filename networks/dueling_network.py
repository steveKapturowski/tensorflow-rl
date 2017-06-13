# -*- coding: utf-8 -*-
from networks import layers
import tensorflow as tf
from networks.q_network import QNetwork


class DuelingNetwork(QNetwork):
 
    def _build_q_head(self, input_state):
        self.w_value, self.b_value, self.value = layers.fc('fc_value', input_state, 1, activation='linear')
        self.w_adv, self.b_adv, self.advantage = layers.fc('fc_advantage', input_state, self.num_actions, activation='linear')

        self.output_layer = (
            self.value + self.advantage
            - tf.reduce_mean(
                self.advantage,
                axis=1,
                keep_dims=True
            )
        )

        q_selected_action = tf.reduce_sum(self.output_layer * self.selected_action_ph, axis=1)
        diff = tf.subtract(self.target_ph, q_selected_action)
        return self._huber_loss(diff)

