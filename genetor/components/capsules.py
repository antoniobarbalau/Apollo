import tensorflow as tf
import numpy as np
from .basic import to_tensor


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


def safe_norm(input, axis = -1, epsilon = 1e-7, keep_dims = False, name = 'norm', keepdims = False):
    if keepdims == True:
        keep_dims = keepdims
    if type(input) is float:
        return tf.sqrt(tf.square(input) + epsilon)
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

        target_classes = to_tensor(params['target_classes'])
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

