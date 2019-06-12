import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf
import os


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

session = tf.Session()

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
margin = tf.placeholder(
    shape = [],
    dtype = tf.float32,
    name = 'margin'
)
architecture = [{
    'type': 'input',
    'input': input,
}, {
    'type': 'conv',
    'params': {
        'kernel_size': 5,
        'filters': 32,
        'stride': 2
    }
}, {
    'type': 'conv_caps_primary'
}, {
    'type': 'conv_caps'
}, {
    'type': 'conv_caps',
    'params': {
        'stride': 1
    }
}, {
    'type': 'class_capsules'
}, {
    'type': 'spread_loss',
    'output_label': 'loss',
    'params': {
        'target': target,
        'margin': margin
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)

optimizer = tf.train.AdamOptimizer(
    # learning_rate = 1e-4
).minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


