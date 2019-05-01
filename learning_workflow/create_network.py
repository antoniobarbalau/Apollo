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
learning_rate = tf.placeholder(
    shape = [],
    dtype = tf.float32,
    name = 'learning_rate'
)
architecture = [{
    'type': 'input',
    'input': input,
}, *genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [128, 128],
        'kernels': 5,
        'activation': genetor.components.prelu,
        'units': [256, 256]
    }
), {
    'type': 'h_projection'
}, {
    'type': 'h_exp_map',
    'output_label': 'encoding'
}, {
    'type': 'h_proto_loss',
    'output_label': 'loss',
    'params': {
        'ways': 5,
        'shots_q': 5,
        'shots_s': 5
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)

optimizer = tf.train.AdamOptimizer(
    learning_rate = learning_rate
).minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


