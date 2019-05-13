import tensorflow as tf
import numpy as np
from .basic import to_tensor


def l2_loss(input, **params):
    with tf.variable_scope(params['name']):
        if type(params['target']) is str:
            target = tf.get_default_graph().get_tensor_by_name(params['target'])
        else:
            target = params['target']

        input_flat = tf.layers.flatten(input)
        target_flat = tf.layers.flatten(target)

        output = tf.reduce_mean(tf.square(input_flat - target_flat),
                                name = 'output')

    return output


def sigmoid_cross_entropy(input, **params):
    target = to_tensor(params['target'])
    
    with tf.variable_scope(params['name']):
        reconstruction = tf.nn.sigmoid(input, name = 'reconstruction')

        output = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = target, logits = input
        )

        output = tf.reduce_mean(output, name = 'output')

    return output


def cross_entropy(input, **params):
    with tf.variable_scope(params['name']):
        target = to_tensor(params['target'])
        n_classes = input.shape[-1].value

        predicted_softmax = tf.nn.softmax(input,
                                          name = 'predicted_softmax')
        predicted_classes = tf.argmax(predicted_softmax,
                                      axis = -1,
                                      name = 'predicted_classes')

        correct_predictions = tf.equal(predicted_classes, target)
        correct_predictions = tf.cast(correct_predictions, tf.float32,
                                      name = 'correct_predictions')
        accuracy_sum = tf.reduce_sum(correct_predictions,
                                     name = 'accuracy_sum')
        accuracy_mean = tf.reduce_mean(correct_predictions,
                                       name = 'accuracy_mean')

        target_one_hot = tf.one_hot(target, depth = n_classes,
                                    name = 'target_one_hot')
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = target_one_hot,
                                                          logits = input)
        loss = tf.reduce_mean(loss,
                              name = 'output')

    return loss


def gan_loss(input, **params):
    with tf.variable_scope(params['name']):
        generator_scope = params['generator_scope']
        discriminator_scope = params['discriminator_scope']
        discriminator_output = tf.reshape(input, [-1])

        is_real = tf.placeholder(
            shape = [None],
            dtype = tf.float32,
            name = 'is_real'
        )
        generator_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope = generator_scope
        )
        generator_loss = tf.reduce_sum(
            -tf.log(discriminator_output + 1e-5) * (1. - is_real),
            name = 'generator_loss'
        )
        generator_optimizer = tf.train.AdamOptimizer()

        discriminator_loss = (
            -tf.reduce_sum(tf.log(discriminator_output + 1e-5) * is_real) -
            tf.reduce_sum(tf.log(1.0 - discriminator_output - 1e-5) * (1.0 - is_real))
        )
        discriminator_loss = tf.identity(discriminator_loss,
                                         name = 'discriminator_loss')
        discriminator_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope = discriminator_scope
        )
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = 1e-5)

    generator_optimizer = generator_optimizer.minimize(
        generator_loss,
        var_list = generator_variables,
        name = 'generator_optimizer'
    )
    discriminator_optimizer = discriminator_optimizer.minimize(
        discriminator_loss,
        var_list = discriminator_variables,
        name = 'discriminator_optimizer'
    )


def contrastive_center_loss(input, **params):
    with tf.variable_scope(params['name']):
        target = to_tensor(params['target'])

        n_dims = input.shape[-1].value

        if 'centroid' in params:
            centroid = to_tensor(params['centroid'])
            n_classes = centroid.shape[-2].value
        else:
            n_classes = params['n_classes']
            centroid = tf.Variable(initial_value = tf.random_normal(shape = [n_classes,
                                                                             n_dims]),
                                   dtype = tf.float32,
                                   name = 'centroid')

        target_one_hot = tf.one_hot(target, depth = n_classes)

        input_tiled = tf.expand_dims(input, axis = 1)
        input_tiled = tf.tile(input_tiled, [1, n_classes, 1],
                              name = 'input_tiled')

        distances = tf.square(tf.subtract(input_tiled, centroid))
        distances = tf.reduce_sum(distances, axis = -1,
                                  name = 'distances')

        predicted_classes = tf.argmin(distances, axis = -1,
                                      name = 'predicted_classes')
        correct_predictions = tf.cast(tf.equal(predicted_classes, target),
                                      tf.float32)
        accuracy_sum = tf.reduce_sum(correct_predictions,
                                     name = 'accuracy_sum')

        center_loss = tf.multiply(distances, target_one_hot)
        center_loss = tf.reduce_sum(center_loss,
                                    axis = -1,
                                    name = 'center_loss')

        reverse_target = 1.0 - target_one_hot
        contrastive_loss = tf.multiply(distances, reverse_target)
        contrastive_loss = tf.reduce_sum(contrastive_loss,
                                         axis = -1,
                                         name = 'contrastive_loss')

        loss = tf.div(center_loss, contrastive_loss + 1e-7)
        loss = tf.reduce_sum(loss,
                              name = 'output')

    return loss


def siamese_contrastive_loss(input, **params):
    with tf.variable_scope(params['name']):
        m = params.get('m', 0.3)
        target = to_tensor(params['target'])

        encoding_size = input.shape[-1].value

        x1, x2 = tf.unstack(tf.reshape(input, [-1, 2, encoding_size]), 2, 1)

        energy = tf.reduce_sum(
            tf.square(tf.subtract(x1, x2)),
            axis = -1,
            keepdims = True
        )

        loss = tf.reduce_mean(
            target * .5 * energy +
            (1. - target) * .5 * tf.square(tf.maximum(0., m - tf.sqrt(energy))),
            name = 'output'
        )

    return loss


def siamese_margin_loss(input, **params):
    with tf.variable_scope(params['name']):
        target = to_tensor(params['target'])

        encoding_size = input.shape[-1].value
        x1, x2 = tf.unstack(tf.reshape(input, [-1, 2, encoding_size]), 2, 1)

        energy = tf.reduce_sum(
            tf.abs(tf.subtract(x1, x2)),
            axis = -1,
            keepdims = True
        )
        energy = 2. * (tf.nn.sigmoid(energy) - .5)
        energy = tf.reshape(energy, [-1], name = 'energy')

        m_1 = 0.3
        m_2 = 0.7
        loss = (target * tf.maximum(energy - m_1, 0.) +
                (1. - target) * tf.maximum(m_2 - energy, 0.))
        loss = tf.reduce_mean(loss,
                              name = 'output')

    return loss


def proto_loss(input, **params):
    ways = params['ways']
    shots_s = params['shots_s']
    shots_q = params['shots_q']

    enc_size = input.shape[-1].value

    samples = tf.reshape(input, [-1, ways, shots_s + shots_q, enc_size])

    s = samples[:, :, :shots_s, :]
    q = samples[:, :, shots_s:, :]

    c = tf.reduce_mean(s, axis = -2)
    # print(c.shape)
    c_tiled = tf.expand_dims(c, axis = 2)
    c_tiled = tf.tile(c_tiled, [1, 1, shots_q, 1])

    intra_cluster = tf.reduce_sum(tf.square(q - c_tiled), axis = -1)

    loss_intra = tf.reduce_mean(intra_cluster)

    c = tf.expand_dims(c, axis = -2)
    # print(c.shape)
    c = tf.tile(c, [1, 1, shots_q, 1])
    # print(c.shape)
    # c = tf.reshape(c, [-1])
    # c = tf.reshape(c, [1, ways, shots_q, enc_size])
    c = tf.expand_dims(c, axis = 1)
    c = tf.tile(c, [1, ways, 1, 1, 1])
    # c = tf.reshape(c, [-1, ways, ways, shots_q, enc_size])
    # c = tf.reshape(c, [-1])
    # c = tf.transpose(c, [0, 3, 1, 2, 4])
    # c = tf.transpose(c, [0, 2, 1, 3])
    # print(c)

    # print(q)
    q = tf.expand_dims(q, axis = 2)
    q = tf.tile(q, [1, 1, ways, 1, 1])
    # q = tf.reshape(q, [-1, ways, shots_q, ways, enc_size])
    # q = tf.reshape(q, [-1])
    # q = tf.reshape(q, [-1, ways, ways, shots_q, enc_size])
    # print(q[:, 1, 1, :, :])
    # print(q)

    # print(loss_inter)


    target = np.zeros([1, ways, ways, shots_q])
    for i in range(ways):
        target[:, i, i, :] = 1.
    target = tf.constant(target, dtype = tf.float32)
    # target = np.arange(J
    # target = np.expand_dims(target, axis = -1)
    # target = np.tile(target, [shots_q, 1])
    # target = np.ravel(target)
    # target = tf.constant(target)

    # print(target.shape)
    # target = tf.one_hot(target, depth = ways)
    # target = tf.reshape(target, [ways, ways, shots_q])
    batch_size = tf.shape(c)[0]
    # target = tf.expand_dims(target, axis = 0)
    target = tf.tile(target, [batch_size, 1, 1, 1])
    # target = 1. - target
    # target = tf.expand_dims(target, axis = -1)
    # print(target.shape)
    # print(loss_inter.shape)

    loss_inter = tf.sqrt(
        tf.reduce_sum(tf.square(c - q), axis = -1) + 1e-6
    )
    loss_inter = loss_inter * target
    loss_inter = tf.reduce_sum(loss_inter)
    # loss_inter = tf.reshape(loss_inter, [-1, 1])
    # loss_inter = (tf.nn.sigmoid(loss_inter) - .5) * 2
    # m = 0.7
    # loss_inter = tf.square(tf.maximum(0., m - loss_inter))
    # loss_inter = tf.reduce_mean(loss_inter)
    # loss_inter = tf.exp(loss_inter)
    # loss_inter = tf.log(tf.reduce_sum(loss_inter))


    output = loss_intra / loss_inter

    return output


def few_shot_loss(input, **params):
    with tf.variable_scope(params['name']):
        ways = params.get('ways', 5)
        enc_size = input.shape[-1].value

        samples = tf.reshape(input, [-1, ways + 1, enc_size])
        query = samples[:, 0, :]
        support = samples[:, 1:, :]

        query = tf.expand_dims(query, axis = 1)
        query = tf.tile(query, [1, ways, 1])

        distances = tf.reduce_sum(
            tf.square(query - support),
            axis = -1
        )
        predicted_classes = tf.argmin(
            distances,
            axis = -1,
            name = 'predicted_classes'
        )

        correct_predictions = tf.cast(
            tf.equal(predicted_classes, 0),
            tf.float32,
            name = 'correct_predictions'
        )
        accuracy_sum = tf.reduce_sum(
            correct_predictions,
            name = 'accuracy_sum'
        )
        accuracy_mean = tf.reduce_mean(
            correct_predictions,
            name = 'accuracy_mean'
        )


    return accuracy_sum

