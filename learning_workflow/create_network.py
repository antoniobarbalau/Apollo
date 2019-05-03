import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf
import os


session = tf.Session()

input = tf.placeholder(
    shape = [None, 256, 256, 3],
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
        'filters': [64, 64, 128, 256, 512, 1024],
        'kernels': 5,
        'activation': genetor.components.prelu,
        'units': [1024, 1024],
        'output_label': 'encoding'
    }
), {
    'type': 'proto_loss',
    'output_label': 'loss',
    'params': {
        'ways': 2,
        'shots_q': 1,
        'shots_s': 1
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)

optimizer = tf.train.AdamOptimizer(
    learning_rate = learning_rate
).minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


