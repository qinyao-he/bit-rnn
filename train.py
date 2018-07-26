import time
import functools
import importlib

import numpy as np
import tensorflow as tf

import reader

import bit_utils
from bit_rnn_cell import BitGRUCell
from model import PTBModel

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('data_path', None, 'data_path')
flags.DEFINE_string('config', None, 'config')

FLAGS = flags.FLAGS


def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(
            reader.ptb_iterator(data, m.batch_size, m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    return importlib.import_module(FLAGS.config).Config()


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.initializers.variance_scaling(distribution='uniform')
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        tf.global_variables_initializer().run()

        def get_learning_rate(epoch, config):
            base_lr = config.learning_rate
            if epoch <= config.nr_epoch_first_stage:
                return base_lr
            elif epoch <= config.nr_epoch_second_stage:
                return base_lr * 0.1
            else:
                return base_lr * 0.01

        for i in range(config.max_epoch):
            m.assign_lr(session, get_learning_rate(i, config))

            print("Epoch: %d Learning rate: %f"
                  % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(
                session, m, train_data, m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f"
                  % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(
                session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f"
                  % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(
            session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()
