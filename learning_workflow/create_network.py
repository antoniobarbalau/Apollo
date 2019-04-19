import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
target = tf.placeholder(
    shape = [None],
    dtype = tf.int64,
    name = 'target'
)
architecture = [{
    'type': 'input',
    'input': input,
}, *genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [20, 50],
        'kernels': [5] * 2,
        'units': [500]
    }
), {
    'type': 'fc',
    'params': {
        'units': 100,
        'activation': None
    }
}, {
# x = tf.layers.conv2d(input, filters = 20, kernel_size = 5, activation = tf.nn.relu,
#                      padding = 'same')
# x = tf.layers.max_pooling2d(x, pool_size = (2, 2), strides = (2, 2), padding = 'same')
# x = tf.layers.conv2d(x, filters = 50, kernel_size = 5, activation = tf.nn.relu,
#                      padding = 'same')
# x = tf.layers.max_pooling2d(x, pool_size = (2, 2), strides = (2, 2), padding = 'same')
# x = tf.layers.flatten(x)
# x = tf.layers.dense(x, units = 500, activation = tf.nn.relu)
# x = tf.layers.dense(x, units = 100)
# architecture = [{
    'type': 'to_poincare',
    # 'input': x,
    'params': {
        # 'c': 0.05
        'c': 1.
    }
}, {
    'type': 'h_mlr',
    'params': {
        # 'c': 0.05,
        'c': 1.,
        'ball_dim': 100,
        'n_classes': 10
    }
}, {
    'type': 'cross_entropy',
    'params': {
        'target': target
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)

optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


