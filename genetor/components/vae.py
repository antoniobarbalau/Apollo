import tensorflow as tf


def vae_encoding(input, **params):
    enc_size = params['units']

    means = tf.layers.dense(input, units = enc_size, activation = None)
    logvar = tf.layers.dense(input, units = enc_size, activation = None)
    stds = 1e-6 + tf.nn.softplus(logvar)

    stds = tf.identity(stds, name = 'stds')
    means = tf.identity(means, name = 'means')

    encoding = means + stds * tf.random_normal(
        tf.shape(means), 0, 1, dtype = tf.float32
    )

    encoding = tf.placeholder_with_default(
        input = encoding,
        shape = [None, enc_size],
        name = 'encoding'
    )

    return encoding

