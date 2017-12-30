import time
import functools

import numpy as np
import tensorflow as tf

import reader

import bit_utils
from bit_rnn_cell import BitGRUCell, BitLSTMCell


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        if 'cell_type' not in dir(config) or config.cell_type == 'gru':
            cell = BitGRUCell(size, w_bit=config.w_bit, f_bit=config.f_bit)
        elif config.cell_type == 'lstm':
            cell = BitLSTMCell(size, w_bit=config.w_bit,
                               f_bit=config.f_bit, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self._initial_state = bit_utils.round_bit(
            tf.sigmoid(self._initial_state), bit=config.f_bit)

        embedding = tf.get_variable(
            "embedding",
            [vocab_size, size],
            initializer=tf.random_uniform_initializer())
        inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        inputs = bit_utils.round_bit(tf.nn.relu(inputs), bit=config.f_bit)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(value=inputs, num_or_size_splits=num_steps, axis=1)]
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                                                   initial_state=self._initial_state)

        output = tf.reshape(tf.concat(values=outputs, axis=1), [-1, size])
        with bit_utils.replace_variable(
                lambda x: bit_utils.quantize_w(tf.tanh(x), bit=config.w_bit)):
            softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
