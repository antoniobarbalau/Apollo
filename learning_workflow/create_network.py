import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf
import os


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
architecture = [{
    'type': 'input',
    'input': input,
}, *genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [16, 36],
        'kernels': [5] * 2,
        'units': [256]
    }
), {
    'type': 'fc',
    'output_label': 'encoding',
    'params': {
        'units': 256,
        'activation': None
    }
}, {
    'type': 'proto_loss',
    'output_label': 'loss',
    'params': {
        'ways': 5,
        'shots_q': 5,
        'shots_s': 5
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)

optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


