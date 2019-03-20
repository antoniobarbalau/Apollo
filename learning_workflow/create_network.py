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
    'type': 'input',
    'input': conv_output_flattened
}, {
    'type': 'fc',
    'output_label': 'encoding',
    'params': {
        'units': 10
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
}, *genetor.builder.new_architecture(
    model = 'deconv',
    structure = {
        'filters': [32, 1],
        'kernels': [5] * 3,
        'strides': [2] * 3,
        'type': 'resize_up_conv'
    }
), {
    'type': 'sigmoid',
    'output_label': 'reconstruction'
}, {
    'type': 'l2_loss',
    'params': {
        'target': input
    }
}]


loss = genetor.builder.new_graph(architecture = architecture)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


