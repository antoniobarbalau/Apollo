import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf

session = tf.Session()

# architecture = [{
#     'type': 'tf_data',
#     'params': {
#         'meta_path': '../data/tf_records/mnist/train/meta.json',
#         'parsers': {
#             'input': genetor.components.parse_image(shape = [28, 28, 1])
#         },
#         'create_placeholders_for': ['input', 'target'],
#         'return': 'input'
#     }
input = tf.placeholder(
    shape = [None, 28, 28, 1],
    dtype = tf.float32,
    name = 'input'
)
architecture = [{
    'type': 'input',
    'input': input,
}, *genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [16, 32],
        'kernels': [5] * 3,
        # 'units': [500]
    }
), {
    'type': 'input',
    'output_label': 'conv_output'
}, {
    'type': 'flatten',
    'output_label': 'conv_output_flattened'
}]
conv_output_flattened = genetor.builder.new_graph(
     architecture = architecture
)
reshape_shape = tf.get_default_graph().get_tensor_by_name('conv_output:0').shape
reshape_shape = [-1, *[dim.value for dim in reshape_shape[1:]]]
architecture = [{
    'type': 'fc',
    'output_label': 'means',
    'params': {
        'units': 20
    }
}, {
    'type': 'fc',
    'output_label': 'stds_raw',
    'input': conv_output_flattened,
    'params': {
        'units': 20
    }
}]
stds = genetor.builder.new_graph(
    architecture = architecture,
    input = conv_output_flattened
)
stds = 1e-6 + tf.nn.softplus(stds)
stds = tf.identity(stds, name = 'stds')
means = tf.get_default_graph().get_tensor_by_name('means:0')
kl = .5 * tf.reduce_sum(
    tf.square(means) + tf.square(stds) - tf.log(1e-8 + tf.square(stds)) - 1,
    axis = 1
)
kl = tf.reduce_mean(kl)
encoding = means + stds * tf.random_normal(
    tf.shape(means), 0, 1, dtype = tf.float32
)
encoding = tf.placeholder_with_default(
    input = encoding,
    shape = [None, 20],
    name = 'encoding'
)
architecture = [{
    'type': 'input',
    'input': encoding
}, {
    'type': 'fc',
    'output_label': 'encoding',
    'params': {
        'units': 2
    }
}, {
    'type': 'fc',
    'params': {
        'units': conv_output_flattened.shape[1].value
    }
}, {
    'type': 'reshape',
    'params': {
        'shape': reshape_shape
    }
# }, *genetor.builder.new_architecture(
#     model = 'cnn',
#     structure = {
#         'units': [500]
#     }
# ), {
}, *genetor.builder.new_architecture(
    model = 'deconv',
    structure = {
        'filters': [32],
        'kernels': [5] * 3,
        'strides': [2] * 3,
        'type': 'resize_up_conv'
    }
), {
    'type': 'resize_up_conv',
    'output_label': 'reconstruction',
    'params': {
        'kernel_size': 5,
        'stride': 2,
        'filters': 1,
        # 'units': 784,
        'activation': tf.nn.sigmoid
    }
}]




reconstruction = genetor.builder.new_graph(architecture = architecture)
reconstruction = tf.layers.flatten(reconstruction) + 1e-8
input = tf.layers.flatten(input)
loss = tf.reduce_sum(
    input * tf.log(reconstruction) + (1. - input) * tf.log(1. - reconstruction),
    axis = 1
)
loss = tf.add(
    -tf.reduce_mean(loss), kl,
    name = 'loss'
)
# print(loss.shape)
optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


