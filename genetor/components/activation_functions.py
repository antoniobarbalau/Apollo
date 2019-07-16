import tensorflow as tf
import numpy as np


def gelu(input, **params):
    output = (
        .5 * input * (1. + tf.tanh(
            np.sqrt(2. / np.pi) * (input + 0.044715 * tf.pow(input, 3))
        ))
    )
    output = tf.identity(output, name = 'gelu')

    return output


def prelu(input, **params):
    alpha = tf.Variable(
        initial_value = tf.zeros(shape = input.shape[1:]),
        name = 'alpha',
        dtype = tf.float32
    )

    positive = tf.nn.relu(input)
    negative = alpha * (input - tf.abs(input)) * .5

    output = tf.add(
        positive, negative,
        name = params['name']
    )

    return output


