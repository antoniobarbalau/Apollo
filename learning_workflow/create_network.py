import sys
sys.path.append('..')
import genetor
from genetor.components.initializations import default_initialization
import glob
import tensorflow as tf

session = tf.Session()

architecture = [{
    'type': 'tf_data',
    'params': {
        'meta_path': '../data/tf_records/fer/train/meta.json',
        'parsers': {
            'input': genetor.components.parse_image(shape = [48, 48, 1])
        },
        'create_placeholders_for': ['input', 'target'],
        'return': 'input'
    }
}, *genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [64, 128, 256],
        'kernels': [5] * 3,
        'units': [512, 8],
        'biasless': True
    }
), {
    'type': 'cross_entropy',
    'params': {
        'target': 'target:0'
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)
optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/fer/ckpt')


