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


