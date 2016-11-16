import tensorflow as tf

import bit_utils


class BitGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, w_bit, f_bit, input_size=None, activation=tf.sigmoid):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._activation = activation
        self._w_bit = w_bit
        self._f_bit = f_bit

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        def replace_w(x):
            if x.op.name.endswith('Matrix'):
                return bit_utils.quantize_w(tf.tanh(x), bit=self._w_bit)
            else:
                return x
        with bit_utils.replace_variable(replace_w):
            with tf.variable_scope(scope or type(self).__name__):  # "BitGRUCell"
                with tf.variable_scope("Gates"):  # Reset gate and update gate.
                    r, u = tf.split(1, 2, tf.nn.rnn_cell._linear([inputs, state],
                                    2 * self._num_units, True, 1.0))
                    r, u = tf.sigmoid(r), tf.sigmoid(u)
                with tf.variable_scope("Candidate"):
                    c = self._activation(tf.nn.rnn_cell._linear([
                            inputs, bit_utils.round_bit(r * state, bit=self._f_bit)],
                            self._num_units, True))
                    c = bit_utils.round_bit(c, bit=self._f_bit)
                new_h = bit_utils.round_bit(u * state + (1 - u) * c, bit=self._f_bit)
        return new_h, new_h
