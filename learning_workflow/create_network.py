import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf

session = tf.Session()

architecture = [{
    'type': 'tf_data',
    'params': {
        'meta_path': '../data/tf_records/mnist/train/meta.json',
        'parsers': {
            'input': genetor.components.components.parse_image(shape = [28, 28, 1])
        },
        'create_placeholders_for': ['input', 'target'],
        'return': 'input'
    }
}]
architecture += genetor.builder.new_architecture(
    model = 'cnn',
    structure = {
        'filters': [16, 36],
        'kernels': [5, 5],
        'units': [128, 10]
    }
)
architecture += [{
    'type': 'cross_entropy',
    'params': {
        'target': 'target:0'
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)
optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')


saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/mnist_base/ckpt')


