import tensorflow as tf

MOMENTUM = .9


def substract(input, **params):
    other = params.get('other', 0)
    return input - other


def input(input, **params):
    return input


def sigmoid(input, **params):
    output = tf.nn.sigmoid(input)
    return output


def relu(input, **params):
    output = tf.nn.relu(input)
    return output


def reshape(input, **params):
    output = tf.reshape(input, params['shape'])
    return output


def to_tensor(input):
    if type(input) is str:
        return tf.get_default_graph().get_tensor_by_name(input)
    return input


def concat(input, **params):
    tensor_list = [input] + [to_tensor(t) for t in params['other']]
    return tf.concat(tensor_list, axis = params['axis'])


def flatten(input, **params):
    return tf.contrib.layers.flatten(input)


def reduce_max(input, **params):
    return tf.reduce_max(input, **params)


def argmax(input, **params):
    return tf.argmax(input, **params)


def dropout(input, **params):
    return tf.nn.dropout(input, **params)


def batch_norm(input, **params):
    activation = params.get('activation', None)
    output = tf.layers.batch_normalization(
        input,
        training = to_tensor(params['is_training'])
    )
    if activation is None:
        return output
    return activation(output)


def fc(input, **params):
    with tf.variable_scope(params.get('name', '')):
        n_inputs = input.shape[-1].value
        n_outputs = params['units']
        initialization = params['initialization']
        batch_norm_is_training = params.get('batch_norm_is_training', None)

        kernel = tf.Variable(
            initial_value = initialization(shape = [n_inputs, n_outputs]),
            name = 'kernel',
            dtype = tf.float32)
        bias = tf.Variable(
            initial_value = tf.zeros(shape = [n_outputs]),
            name = 'bias',
            dtype = tf.float32)

        weighted_input = tf.matmul(input, kernel,
                                   name = 'weighted_input')
        output_raw = tf.add(weighted_input, bias,
                            name = 'output_raw')

        if batch_norm_is_training is not None:
            output_raw = tf.contrib.layers.batch_norm(
                output_raw,
                is_training = batch_norm_is_training,
                decay = MOMENTUM
            )

        activation = params.get('activation', None)
        if activation is None:
            activation = tf.identity
        output = activation(output_raw,
                            name = 'output')

    return output


