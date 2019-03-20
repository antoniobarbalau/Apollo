import tensorflow as tf
import numpy as np


def xavier_over_2(shape, variable_type):
    if variable_type == 'fc_kernel':
        return tf.random_normal(shape = shape,
                                dtype = tf.float32) / tf.sqrt(float(shape[0]) / 2)

    if variable_type == 'fc_bias':
        return tf.random_normal(shape = shape,
                                dtype = tf.float32,
                                stddev = 0.01)

    return None


def default_initialization(shape):
    return tf.random_normal(shape = shape,
                            dtype = tf.float32,
                            stddev = 0.05)
    # limit = np.sqrt(6 / (np.prod(shape[:-1]) + shape[-1]) )
    # return tf.random_uniform(
    #     shape = shape,
    #     minval = -limit,
    #     maxval = limit,
    # )
    # return tf.random_normal(shape = shape,
    #                         mean = 0.0,
    #                         stddev = 5e-2,
    #                         dtype = tf.float32) #/ np.sqrt(np.prod(shape[:-1]) / 2)

