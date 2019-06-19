import sys
sys.path.append('..')

from input_preprocessing import augment
import genetor
import glob
import tensorflow as tf
import os


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

session = tf.Session()

learning_rate = tf.placeholder(
    shape = [],
    dtype = tf.float32,
    name = 'learning_rate'
)
batch_norm_is_training = tf.placeholder(
    shape = [],
    dtype = tf.bool,
    name = 'batch_norm_is_training'
)
architecture = [{
    'type': 'tf_data',
    'params': {
        'meta_path': '../data/tf_records/fer/train/meta.json',
        'create_placeholders_for': ['input', 'target'],
        'parsers': {
            'input': augment
        },
        'return': 'input'
    }
}, *genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [256, 512, 512, 512],
        'kernels': 5,
        'units': [128, 7],
        'batch_norm_is_training': batch_norm_is_training
    }
), {
    'type': 'cross_entropy',
    'params': {
        'target': 'target:0'
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)

optimizer = tf.train.AdamOptimizer(
    learning_rate = learning_rate
)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = optimizer.minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


