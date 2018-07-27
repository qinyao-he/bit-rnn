import tensorflow as tf
#from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

import bit_utils


class BitGRUCell(tf.nn.rnn_cell.GRUCell):

    def __init__(self,
                 num_units,
                 w_bit,
                 f_bit,
                 activation=tf.sigmoid,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(BitGRUCell, self).__init__(num_units, activation,
                                         reuse, kernel_initializer, bias_initializer)
        self._w_bit = w_bit
        self._f_bit = f_bit
        self._gate_linear = None
        self._candidate_linear = None

    def call(self, inputs, state):
        def replace_w(x):
            if x.op.name.endswith('kernel'):
                return bit_utils.quantize_w(tf.tanh(x), bit=self._w_bit)
            else:
                return x

        with bit_utils.replace_variable(replace_w):
            if self._gate_linear is None:
                bias_ones = self._bias_initializer
                if self._bias_initializer is None:
                    bias_ones = tf.constant_initializer(
                        1.0, dtype=inputs.dtype)
                with tf.variable_scope("gates"):  # Reset gate and update gate.
                    # self._gate_linear = rnn_cell_impl._Linear(
                    self._gate_linear = core_rnn_cell._Linear(
                        [inputs, state],
                        2 * self._num_units,
                        True,
                        bias_initializer=bias_ones,
                        kernel_initializer=self._kernel_initializer)

            value = tf.sigmoid(self._gate_linear([inputs, state]))
            r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

            r_state = bit_utils.round_bit(r * state, bit=self._f_bit)
            if self._candidate_linear is None:
                with tf.variable_scope("candidate"):
                    # self._candidate_linear = rnn_cell_impl._Linear(
                    self._candidate_linear = core_rnn_cell._Linear(
                        [inputs, r_state],
                        self._num_units,
                        True,
                        bias_initializer=self._bias_initializer,
                        kernel_initializer=self._kernel_initializer)
            c = self._activation(self._candidate_linear([inputs, r_state]))
            c = bit_utils.round_bit(c, bit=self._f_bit)
            new_h = bit_utils.round_bit(
                u * state + (1 - u) * c, bit=self._f_bit)
        return new_h, new_h


class BitLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):

    def __init__(self, num_units, w_bit, f_bit, forget_bias=1.0,
                 state_is_tuple=True, activation=tf.sigmoid, reuse=None):
        super(BitLSTMCell, self).__init__(
            num_units, forget_bias, state_is_tuple, activation, reuse)
        self._w_bit = w_bit
        self._f_bit = f_bit
        self._linear = None

    def call(self, inputs, state):
        def replace_w(x):
            if x.op.name.endswith('kernel'):
                return bit_utils.quantize_w(tf.tanh(x), bit=self._w_bit)
            else:
                return x

        with bit_utils.replace_variable(replace_w):
            sigmoid = tf.sigmoid
            # Parameters of gates are concatenated into one multiply for
            # efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

            if self._linear is None:
                # self._linear = rnn_cell_impl._Linear(
                self._linear = core_rnn_cell._Linear(
                    [inputs, h], 4 * self._num_units, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(
                value=self._linear([inputs, h]), num_or_size_splits=4, axis=1)

            new_c = (
                c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            new_h = bit_utils.round_bit(self._activation(
                new_c) * sigmoid(o), bit=self._f_bit)

            if self._state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state
