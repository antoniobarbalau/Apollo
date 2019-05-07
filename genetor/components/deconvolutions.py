import tensorflow as tf
from .basic import to_tensor


def conv2d_transpose(input, **params):
    output = tf.layers.conv2d_transpose(input, **params)
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
        # strides = [1, params['stride'], params['stride'], 1]
        strides = [1, 1, 1, 1]
        padding = params['padding']
        # output_shape = tf.shape(to_tensor(params['output_shape']))
        # output_shape = [int(input.shape[1].value * params['stride']), int(input.shape[2].value * params['stride'])]
        if 'output_shape' in params:
            output_shape = params['output_shape']
        else:
            output_shape = [
                int(input.shape[1].value * params['stride']),
                int(input.shape[2].value * params['stride'])
            ]

        initialization = params['initialization']

        input = tf.image.resize_images(input,
                                       size = output_shape)

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

