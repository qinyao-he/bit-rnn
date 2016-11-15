import functools

import tensorflow as tf
from tensorflow.python.ops import variable_scope


__origin_get_variable = tf.get_variable


class TFVariableReplaceHelper(object):
    object_stack = []

    def __init__(self, fn):
        self._old_get_variable = None
        self._fn = fn

    @classmethod
    def new_get_variable(*args, **kwargs):
        v = __origin_get_variable(*args, **kwargs)
        return object_stack[-1]._fn(v)

    def __enter__(self):
        object_stack.append(self)
        self._old_get_variable = tf.get_variable
        tf.get_variable = new_get_variable
        variable_scope.get_variable = new_get_variable

    def __exit__(self, *args):
        object_stack.pop()
        tf.get_variable = self._old_get_variable
        variable_scope.get_variable = self._old_get_variable


def replace_variable(fn):
    return TFVariableReplaceHelper(fn)


def round_bit(x, bit):
    k = 2**bit - 1
    with tf.get_default_graph().gradient_override_map({'Floor': 'Identity'}):
        return tf.round(x * k) / k


def quantize_w(x, bit):
    scale = tf.reduce_mean(tf.abs(x)) * 2
    with tf.get_default_graph().gradient_override_map({'clip_by_value': 'Identity'}):
        return (round_bit(tf.clip_by_value(x / scale, -0.5, 0.5) + 0.5,
                          bit=bit) - 0.5) * scale



round_bit_1bit = functools.partial(round_bit, bit=1)
round_bit_2bit = functools.partial(round_bit, bit=2)
round_bit_3bit = functools.partial(round_bit, bit=3)
round_bit_4bit = functools.partial(round_bit, bit=4)

quantize_w_1bit = functools.partial(quantize_w, bit=1)
quantize_w_2bit = functools.partial(quantize_w, bit=2)
quantize_w_3bit = functools.partial(quantize_w, bit=3)
quantize_w_4bit = functools.partial(quantize_w, bit=4)
