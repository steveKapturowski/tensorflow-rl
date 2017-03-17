# -*- coding: utf-8 -*-
import layers
import tensorflow as tf
from q_network import QNetwork


class DuelingNetwork(QNetwork):
 
    def _build_q_head(self):
        self.w_value, self.b_value, self.value = layers.fc('fc_value', self.ox, 1, activation='linear')
        self.w_adv, self.b_adv, self.advantage = layers.fc('fc_advantage', self.ox, self.num_actions, activation='linear')

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

