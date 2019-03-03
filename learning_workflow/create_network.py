import sys
sys.path.append('..')
import genetor
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
        'filters': [64, 128, 256, 256],
        'kernels': [9] * 4,
    }
), {
    'type': 'primary_caps',
    'params': {
        'n_caps_layers': 32
    }
}, {
    'type': 'caps',
    'params': {
        'n_caps': 7,
        'caps_dim': 8
    }
}, {
    'type': 'caps_margin_loss',
    'params': {
        'target_classes': 'target:0'
    }
}]

loss = genetor.builder.new_graph(architecture = architecture)
optimizer = tf.train.AdamOptimizer(1e-5).minimize(loss, name = 'optimizer')

saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/fer/ckpt')


