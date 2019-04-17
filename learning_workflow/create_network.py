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
    'type': 'flatten',
    'input': input,
}, {
    'type': 'h_linear'
}, {
    'type': 'cross_entropy',
    'output_label': 'loss',
    'params': {
        'target': target
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)

optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


