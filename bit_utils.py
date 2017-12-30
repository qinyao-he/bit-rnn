import functools

import tensorflow as tf
from tensorflow.python.ops import variable_scope


_origin_get_variable = tf.get_variable
_object_stack = []


def _new_get_variable(*args, **kwargs):
    v = _origin_get_variable(*args, **kwargs)
    if len(_object_stack) != 0:
        return _object_stack[-1]._fn(v)
    else:
        return v


class TFVariableReplaceHelper(object):

    def __init__(self, fn):
        self._old_get_variable = None
        self._fn = fn

    def __enter__(self):
        global _object_stack
        _object_stack.append(self)
        self._old_get_variable = tf.get_variable
        tf.get_variable = _new_get_variable
        variable_scope.get_variable = _new_get_variable

    def __exit__(self, *args):
        global _object_stack
        _object_stack.pop()
        tf.get_variable = self._old_get_variable
        variable_scope.get_variable = self._old_get_variable


def replace_variable(fn):
    return TFVariableReplaceHelper(fn)


def round_bit(x, bit):
    if bit == 32:
        return x
    g = tf.get_default_graph()
    k = 2**bit - 1
    with g.gradient_override_map({'Round': 'Identity'}):
        return tf.round(x * k) / k


_grad_defined = False
if not _grad_defined:
    @tf.RegisterGradient("IdentityMaxMinGrad")
    def _identigy_max_min_grad(op, grad):
        return grad, None


def quantize_w(x, bit):
    if bit == 32:
        return x
    g = tf.get_default_graph()
    # do not compute gradient with respect to scale
    scale = tf.stop_gradient(tf.reduce_mean(tf.abs(x)) * 2.5)
    with g.gradient_override_map({'Minimum': 'IdentityMaxMinGrad'}):
        with g.gradient_override_map({'Maximum': 'IdentityMaxMinGrad'}):
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
