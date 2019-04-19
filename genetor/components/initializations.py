import tensorflow as tf
import numpy as np


def default_initialization(shape):
    fan_in = np.prod(shape[:-1])
    fan_out = shape[-1]
    limit = np.sqrt(2 / (fan_in + fan_out))
    # return tf.random_uniform(
    #     shape = shape,
    #     minval = -limit,
    #     maxval = limit,
    #     dtype = tf.float32
    # )
    return tf.random_normal(
        shape = shape,
        stddev = limit,
        dtype = tf.float32
    )

