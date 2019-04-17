import tensorflow as tf
import numpy as np
from .initializations import *


def h_lambda(c, x):
    return 2. / (1. - c * tf.square(tf.norm(x)))


def mobius_add(c, x, y):
    numerator = (
        (1. + 2. * c * tf.reduce_sum(x * y, axis = -1) +
         c * tf.square(tf.norm(y))) * x +
        (1. - c * tf.square(tf.norm(x))) * y
    )
    denominator = (
        1. + 2. * c * tf.reduce_sum(x * y, axis = -1) +
        c ** 2. * tf.square(tf.norm(x)) * tf.square(tf.norm(y))
    )

    return numerator / denominator


def h_log(c, x, input):

    xpy = mobius_add(c, -x, input)
    print(c)

    output = (
        2. / (np.sqrt(c) * h_lambda(c, x)) *
        tf.atanh(np.sqrt(c) * tf.norm(xpy)) *
        xpy / tf.norm(xpy)
    )
    output = tf.identity(output, name = 'output')

    return input


def h_exp(c, x, input):

    tanh = (
        tf.tanh(np.sqrt(c) * h_lambda(c, x) * tf.norm(input) / 2.) *
        input / (np.sqrt(c) * tf.norm(input))
    )
    
    output = mobius_add(c, x, tanh)


    return output


def mobius(f, c, input):
    return h_exp(
        c, 0.,
        f(
            h_log(c, 0., input)
        )
    )


def h_matmul(c, M, x):
    Mx = tf.matmul(M, x)

    output = (
        1. / np.sqrt(c) *
        tf.tanh(tf.norm(Mx) / tf.norm(x) * tf.atanh(np.sqrt(c) * tf.norm(x))) *
        Mx / tf.norm(Mx)
    )

    return output


def h_linear(input, **params):
    n_inputs = input.shape[-1].value
    # n_outputs = params['units']
    n_outputs = 10
    c = params['c']

    kernel = tf.Variable(
        initial_value = default_initialization(shape = [n_inputs, n_outputs]),
        name = 'kernel',
        dtype = tf.float32
    )
    f = lambda x: h_matmul(c, x, kernel)

    output = mobius(f, params['c'], input)

    return output



