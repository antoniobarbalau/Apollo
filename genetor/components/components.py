from .. import builder
import json
import numpy as np
import tensorflow as tf


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
                            mean = 0.0,
                            stddev = 1e-2,
                            dtype = tf.float32)


def input(input, **params):
    return input


def conv(input, **params):
    with tf.variable_scope(params['name']):
        kernel_size = params['kernel_size']
        filters = params['filters']
        strides = [1, params['stride'], params['stride'], 1]
        padding = params['padding']
        initialization = params['initialization']

        kernel = tf.Variable(
            initial_value = initialization(
                shape = [kernel_size, kernel_size, input.shape[-1].value, filters]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [filters]),
            name = 'bias',
            dtype = tf.float32)

        filtered_input = tf.nn.conv2d(
            input = input,
            filter = kernel,
            strides = strides,
            padding = padding,
            name = 'filtered_input')
        output_raw = tf.add(filtered_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def dilated_conv(input, **params):
    with tf.variable_scope(params['name']):
        kernel_size = params['kernel_size']
        filters = params['filters']
        rate = params['rate']
        padding = params['padding']
        initialization = params['initialization']

        kernel = tf.Variable(
            initial_value = initialization(
                shape = [kernel_size, kernel_size, input.shape[-1].value, filters]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [filters]),
            name = 'bias',
            dtype = tf.float32)

        filtered_input = tf.nn.atrous_conv2d(
            value = input,
            filters = kernel,
            rate = rate,
            padding = padding,
            name = 'filtered_input')
        output_raw = tf.add(filtered_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def upsampling(input, **params):
    output = tf.layers.conv2d_transpose(input, **params)
    return output


def sigmoid(input, **params):
    output = tf.nn.sigmoid(input)
    return output


def reshape(input, **params):
    output = tf.reshape(input, params['shape'])
    return output


def up_conv(input, **params):
    with tf.variable_scope(params['name']):
        kernel_size = params['kernel_size']
        filters = params['filters']
        strides = [1, params['stride'], params['stride'], 1]
        padding = params['padding']
        output_shape = tf.shape(to_tensor(params['output_shape']))
        initialization = params['initialization']
        kernel = tf.Variable(
            initial_value = initialization(
                shape = [kernel_size, kernel_size, filters, input.shape[-1].value]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [filters]),
            name = 'bias',
            dtype = tf.float32)

        filtered_input = tf.nn.conv2d_transpose(
            value = input,
            filter = kernel,
            strides = strides,
            padding = padding,
            output_shape = output_shape,
            name = 'filtered_input')
        output_raw = tf.add(filtered_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def resize_up_conv(input, **params):
    with tf.variable_scope(params['name']):
        kernel_size = params['kernel_size']
        filters = params['filters']
        strides = [1, params['stride'], params['stride'], 1]
        padding = params['padding']
        output_shape = tf.shape(to_tensor(params['output_shape']))
        initialization = params['initialization']

        print(input.shape)
        input = tf.image.resize_images(input,
                                       size = output_shape[1:3])
        print(input.shape)

        kernel = tf.Variable(
            initial_value = initialization(
                shape = [kernel_size, kernel_size, input.shape[-1].value, filters]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [filters]),
            name = 'bias',
            dtype = tf.float32)

        filtered_input = tf.nn.conv2d(
            input = input,
            filter = kernel,
            strides = strides,
            padding = padding,
            name = 'filtered_input')
        output_raw = tf.add(filtered_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')
        print(output.shape)

    return output


def to_tensor(input):
    if type(input) is str:
        return tf.get_default_graph().get_tensor_by_name(input)
    return input


def concat(input, **params):
    tensor_list = [input] + [to_tensor(t) for t in params['other']]
    return tf.concat(tensor_list, axis = params['axis'])


def max_pool(input, **params):
    output, switches = tf.nn.max_pool_with_argmax(input, **params)

    with tf.variable_scope(params['name']):
        switches = tf.identity(switches,
                               name = 'switches')
        output = tf.identity(output,
                             name = 'output')

    return output


def avg_pool(input, **params):
    output = tf.layers.average_pooling2d(input,
                                       pool_size = [input.shape[1].value,
                                                    input.shape[2].value],
                                       strides = 100)
    output = tf.reshape(output, [-1, output.shape[-1].value])
    return output


def flatten(input, **params):
    return tf.contrib.layers.flatten(input)


def fc(input, **params):
    with tf.variable_scope(params.get('name', '')):
        n_inputs = input.shape[-1].value
        n_outputs = params['units']
        initialization = params['initialization']

        kernel = tf.Variable(
            initial_value = initialization(shape = [n_inputs, n_outputs]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [n_outputs]),
            # initial_value = initialization(shape = [n_outputs]),
            name = 'bias',
            dtype = tf.float32)

        weighted_input = tf.matmul(input, kernel,
                                   name = 'weighted_input')
        output_raw = tf.add(weighted_input, bias,
                            name = 'output_raw')

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


def reduce_max(input, **params):
    return tf.reduce_max(input, **params)


def argmax(input, **params):
    return tf.argmax(input, **params)


def dropout(input, **params):
    return tf.nn.dropout(input, **params)


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


def squash(input, axis = -1, epsilon = 1e-7, name = 'squashed'):
    squared_norm = tf.reduce_sum(tf.square(input),
                                 axis = axis,
                                 keepdims = True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = input / safe_norm
    squashed_input = tf.multiply(squash_factor, unit_vector,
                                 name = name)
    return squashed_input


def primary_caps(input, **params):
    with tf.variable_scope(params['name']):
        n_caps_layers = params['n_caps_layers']
        caps_dim = input.shape[3].value // n_caps_layers
        n_caps = (input.shape[1].value * input.shape[2].value * 
                  n_caps_layers)

        output_raw = tf.reshape(input,
                                 [-1, n_caps, caps_dim],
                                 name = 'output_raw')

        output = squash(output_raw,
                        name = 'output')
    
    return output


def routing_by_agreement(input, predicted_output, **params):
    with tf.variable_scope('routing'):
        batch_size = tf.shape(input)[0]
        input_n_caps = input.shape[1].value
        output_n_caps = params['n_caps']
        output_caps_dim = params['caps_dim']

        raw_weights = tf.zeros([batch_size, input_n_caps, output_n_caps, 1, 1])

        for iter_n in range(params['routing']['n_iterations']):
            softmax_weights = tf.nn.softmax(raw_weights, axis = 2,
                                            name = 'weights_{}'.format(iter_n))

            weighted_prediction = tf.multiply(predicted_output, softmax_weights)
            weighted_sum = tf.reduce_sum(weighted_prediction, axis = 1)

            output_raw_reshaped = tf.reshape(weighted_sum,
                                             [batch_size, output_n_caps,
                                             output_caps_dim, 1])
            output_reshaped = squash(output_raw_reshaped, axis = -2)

            output_expanded = tf.expand_dims(output_reshaped, 1)
            output_tiled = tf.tile(output_expanded, [1, input_n_caps, 1, 1, 1])

            agreement = tf.matmul(predicted_output, output_tiled)
            
            raw_weights = tf.add(raw_weights, agreement,
                                 name = 'weights_raw_{}'.format(iter_n))

        softmax_weights = tf.nn.softmax(raw_weights, axis = 2,
                                        name = 'output')

    return softmax_weights


def caps(input, **params):
    with tf.variable_scope(params['name']):
        input_n_caps = input.shape[1].value
        input_caps_dim = input.shape[2].value
        output_n_caps = params['n_caps']
        output_caps_dim = params['caps_dim']

        kernel_initial_value = tf.random_normal(
            shape = [1,
                     input_n_caps, output_n_caps,
                     input_caps_dim, output_caps_dim],
            stddev = 0.1,
            dtype = tf.float32)
        kernel = tf.Variable(initial_value = kernel_initial_value,
                             name = 'kernel',
                             dtype = tf.float32)
        batch_size = tf.shape(input)[0]
        kernel_tiled = tf.tile(kernel, [batch_size, 1, 1, 1, 1],
                               name = 'kernel_tiled')

        input_tiled = tf.expand_dims(input, -2)
        input_tiled = tf.expand_dims(input_tiled, 2)
        input_tiled = tf.tile(input_tiled, [1, 1, output_n_caps, 1, 1],
                               name = 'input_tiled')

        predicted_output = tf.matmul(input_tiled, kernel_tiled,
                                     name = 'predicted_output')

        routing_weights = routing_by_agreement(input,
                                               predicted_output,
                                               **params)

        weighted_prediction = tf.multiply(predicted_output, routing_weights,
                                          name = 'weighted_prediction')
        weighted_sum = tf.reduce_sum(weighted_prediction, axis = 1,
                                     name = 'weighted_sum')

        raw_output = tf.reshape(weighted_sum,
                                [batch_size, output_n_caps, output_caps_dim],
                                name = 'output_raw')

        output = squash(raw_output, name = 'output')

    return output


def safe_norm(input, axis = -1, epsilon = 1e-7, keep_dims = False, name = 'norm'):
    squared_norm = tf.reduce_sum(tf.square(input),
                                 axis = axis,
                                 keepdims = keep_dims)
    norm = tf.sqrt(squared_norm + epsilon,
                   name = name)

    return norm


def caps_margin_loss(caps, **params):
    with tf.variable_scope(params['name']):
        absent_threshold = params['absent_threshold']
        absent_weight = params['absent_weight']
        present_threshold = params['present_threshold']

        target_classes = params['target_classes']
        n_classes = caps.shape[1].value

        target_one_hot = tf.one_hot(target_classes, depth = n_classes,
                                    name = 'target_one_hot')

        caps_activation = safe_norm(caps,
                                    name = 'caps_activation')

        predicted_classes = tf.argmax(caps_activation, axis = -1,
                                      name = 'predicted_classes')
        correct_predictions = tf.equal(predicted_classes, target_classes)
        correct_predictions = tf.cast(correct_predictions, tf.float32,
                                      name = 'correct_predictions')
        accuracy_sum = tf.reduce_sum(correct_predictions,
                                     name = 'accuracy_sum')
        accuracy_mean = tf.reduce_mean(correct_predictions,
                                       name = 'accuracy_mean')

        present_error = tf.square(tf.maximum(0., present_threshold - caps_activation),
                                  name = 'present_error')
        absent_error = tf.square(tf.maximum(0., caps_activation - absent_threshold),
                                 name = 'absent_error')

        margin_loss = tf.add(target_one_hot * present_error,
                             absent_weight * (1.0 - target_one_hot) * absent_error)
        margin_loss = tf.reduce_sum(margin_loss, axis = 1)
        margin_loss = tf.reduce_mean(margin_loss,
                                     name = 'output')

    return margin_loss


def mask_capsules_with_labels(caps, **params):
    with tf.variable_scope(params['name']):
        mask = tf.one_hot(params['labels'],
                          depth = caps.shape[1].value)
        mask = tf.reshape(mask,
                          shape = [-1, caps.shape[1].value, 1],
                          name = 'mask')

        caps_masked = tf.multiply(caps, mask,
                                  name = 'output')

    return caps_masked


def caps_reconstruction(generated_caps, **params):
    with tf.variable_scope(params['name']):
        reshaped_caps = tf.layers.flatten(generated_caps)

        input = tf.placeholder_with_default(input = reshaped_caps,
                                            shape = reshaped_caps.shape,
                                            name = 'input') 

        layer = input
        for units in params['units']:
            layer = tf.layers.dense(layer,
                                    units = units,
                                    activation = tf.nn.relu)
        reconstruction_flat = tf.layers.dense(layer,
                                              units = np.prod(params['output_shape']),
                                              activation = tf.nn.sigmoid)

        reconstruction = tf.reshape(reconstruction_flat,
                                    shape = [-1, *params['output_shape']],
                                    name = 'output')

    return reconstruction


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


def fire(input, **params):
    with tf.variable_scope(params['name']):
        params['squeeze'] = params.get('squeeze', {})
        params['squeeze'] = dict(builder.DEFAULT_PARAMS['conv'],
                                 **params['squeeze'],
                                 kernel_size = 1,
                                 name = 'squeeze')
        squeezed_input = conv(input,
                              **params['squeeze'])

        params['expand_obo'] = params.get('expand_obo', {})
        params['expand_obo'] = dict(builder.DEFAULT_PARAMS['conv'],
                                    **params['expand_obo'],
                                    kernel_size = 1,
                                    name = 'expand_obo')
        obo_conv = conv(squeezed_input,
                        **params['expand_obo'])

        params['expand_tbt'] = params.get('expand_tbt', {})
        params['expand_tbt'] = dict(builder.DEFAULT_PARAMS['conv'],
                                    **params['expand_tbt'],
                                    kernel_size = 3,
                                    name = 'expand_tbt')
        tbt_conv = conv(squeezed_input,
                        **params['expand_tbt'])

        output = tf.concat([obo_conv, tbt_conv],
                           axis = -1,
                           name = 'output')

    return output


def bypass_fire(input, **params):
    with tf.variable_scope(params['name']):
        fire_output = fire(input, **dict(params, name = 'fire'))

        output = tf.add(input, fire_output,
                         name = 'output')

    return output


def siamese_contrastive_loss(input, **params): # not finalized
    target = to_tensor(params['target'])

    encoding_size = input.shape[-1].value
    x1, x2 = tf.unstack(tf.reshape(input, [-1, 2, encoding_size]), 2, 1)

    energy = tf.reduce_sum(tf.abs(tf.subtract(x1, x2)),
                           axis = -1)
    
    Q = 100.
    loss_impostor = 2 * Q * tf.exp(-2.77 * energy / Q)
    loss_genuine = 2. * tf.square(energy) / Q

    loss = (1. - target) * loss_genuine + target * loss_genuine
    loss = tf.reduce_mean(loss,
                          name = 'output')

    return loss


def siamese_margin_loss(input, **params):
    target = to_tensor(params['target'])

    encoding_size = input.shape[-1].value
    x1, x2 = tf.unstack(tf.reshape(input, [-1, 2, encoding_size]), 2, 1)

    energy = tf.reduce_sum(tf.abs(tf.subtract(x1, x2)),
                           axis = -1,
                           keepdims = True)
    energy = 2. * (tf.nn.sigmoid(energy) - .5)
    energy = tf.reshape(energy, [-1])

    m_1 = 0.3
    m_2 = 0.7
    loss = (target * tf.maximum(energy - m_1, 0.) +
            (1. - target) * tf.maximum(m_2 - energy, 0.))
    loss = tf.reduce_mean(loss,
                          name = 'output')

    return loss


def skip(input, **params):
    with tf.variable_scope(params['name']):
        other = to_tensor(params['other'])

        if input.shape[1].value == other.shape[1].value:
            output = tf.add(input, other, name = 'output')
        else:
            conv_other = conv(other,
                              name = 'skip_conv',
                              filters = input.shape[-1].value,
                              kernel_size = 1, stride = 2,
                              padding = 'SAME', activation = tf.nn.relu,
                              initialization = default_initialization)

            output = tf.add(input, conv_other, name = 'output')

    return output


def batch_norm(input, **params):
    return tf.layers.batch_normalization(
        input,
        training = to_tensor(params['is_training'])
    )


def parse_image_with_shape(im, shape):
    output = tf.image.decode_png(im, channels = shape[-1])
    output = tf.image.resize_images(output, shape[:-1]) / 255.

    return output


def parse_image(shape):
    return lambda im: parse_image_with_shape(im, shape)


def tf_data(input, **params):
    data_format = json.load(open(params['meta_path'], 'r'))

    tf_type = {
        'bytes': tf.FixedLenFeature([], tf.string),
        'int': tf.FixedLenFeature([], tf.int64),
        'float': tf.FixedLenFeature([], tf.float32)
    }
    feature_description = {
        feature_name: tf_type[feature_type]
        for feature_name, feature_type in data_format.items()
    }
    def parse_sample(sample_proto):
        sample = tf.parse_single_example(sample_proto, feature_description)

        for feature_name, feature_parser in params['parsers'].items():
            sample[feature_name] = feature_parser(sample[feature_name])

        return sample

    record_paths_tensor = tf.placeholder(
        dtype = tf.string,
        shape = [None],
        name = 'record_paths'
    )
    dataset = tf.data.TFRecordDataset(record_paths_tensor)
    dataset = dataset.map(parse_sample)
    dataset = dataset.repeat()
    batch_size = tf.placeholder(
        dtype = tf.int64,
        shape = [],
        name = 'batch_size'
    )
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    iterator_initializer = iterator.make_initializer(
        dataset,
        name = 'iterator_initializer'
    )
    next_batch = iterator.get_next(name = 'next_batch')

    placeholders = dict()
    for feature_name in params['create_placeholders_for']:
        placeholders[feature_name] = tf.placeholder_with_default(
            input = next_batch[feature_name],
            shape = next_batch[feature_name].shape,
            name = feature_name
        )

    if 'return' in params:
        return placeholders[params['return']]
    return next_batch


