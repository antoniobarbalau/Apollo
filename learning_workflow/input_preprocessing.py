import tensorflow as tf


def augment(im):
    shape = [48, 48, 1]
    im = tf.image.decode_png(im, channels = shape[-1])
    im = tf.image.resize_images(im, shape[:-1])

    im = im / 255. - .5

    im = tf.image.random_flip_left_right(im)

    im = random_crop(im)

    im = random_bc(im)

    return im


def random_crop(im):
    min_percentage = .7
    n_l = tf.random_uniform(
        shape = [],
        minval = int(min_percentage * im.shape[0].value),
        maxval = im.shape[0],
        dtype = tf.int64
    )
    n_c = tf.random_uniform(
        shape = [],
        minval = int(min_percentage * im.shape[1].value),
        maxval = im.shape[1],
        dtype = tf.int64
    )
    im = tf.cond(
        tf.random_uniform(shape = []) < .7,
        lambda: tf.image.resize_images(
            tf.random_crop(im, [n_l, n_c, 1]),
            [48, 48]
        ),
        lambda: im
    )

    return im


def random_bc(im):
    b_margin = .25
    c_margin = .125

    r = tf.random_uniform(shape = ())
    im = tf.cond(
        r < .1,
        lambda: tf.image.adjust_brightness(im, tf.random_uniform([], -b_margin, b_margin)),
        lambda: tf.cond(
            r < .2,
            lambda: tf.image.adjust_contrast(im, tf.random_uniform([], -c_margin, c_margin)),
            lambda: tf.cond(
                r < .3,
                lambda: tf.image.adjust_brightness(
                    tf.image.adjust_contrast(im, tf.random_uniform([], -c_margin, c_margin)),
                    tf.random_uniform([], -b_margin, b_margin)
                ),
                lambda: im
            )
        )
    )

    return im


