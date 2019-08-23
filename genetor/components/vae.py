import tensorflow as tf


def vae_encoding(input, **params):
    enc_size = params['units']
    beta = params.get('beta', 1.)

    if len(input.shape) > 2:
        input = tf.layers.flatten(input)
    means = tf.layers.dense(input, units = enc_size, activation = None)
    logvar = tf.layers.dense(input, units = enc_size, activation = None)
    # stds = 1e-6 + tf.nn.softplus(logvar)

    stds = tf.exp(logvar / 2., name = 'stds')
    means = tf.identity(means, name = 'means')

    # encoding = means + stds * tf.random_normal(
    #     tf.shape(means), 0, 1, dtype = tf.float32
    # )
    encoding = means + stds * tf.random_normal(
        tf.shape(means), 0., 1., dtype = tf.float32
    )

    encoding = tf.placeholder_with_default(
        input = encoding,
        shape = [None, enc_size],
        name = 'encoding'
    )


    # kl = .5 * tf.reduce_sum(
    #     tf.square(means) + tf.square(stds) - tf.log(1e-8 + tf.square(stds)) - 1,
    #     axis = 1
    # )
    kl = tf.reduce_sum(
        tf.square(means) * .5 + tf.square(stds) * .5 - tf.log(stds) - .5,
        name = 'kl'
    )
    # kl = tf.reduce_sum(
    #     .5 * (logvar + tf.sqrt(means) + tf.exp(logvar) - 1.),
    #     name = 'kl'
    # )
    # kl = beta * tf.reduce_mean(kl)
    # kl = tf.reduce_mean(kl, name = 'kl')

    return encoding


