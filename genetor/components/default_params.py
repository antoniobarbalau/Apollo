import tensorflow as tf
from .initializations import default_initialization

DEFAULT_PARAMS = {
    'resize_up_conv': {
        'initialization': default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'up_conv': {
        'initialization': default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'conv': {
        'initialization': default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'stride': 1,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'dilated_conv': {
        'initialization': default_initialization,
        'filters': 32,
        'kernel_size': 3,
        'rate': 2,
        'padding': 'SAME',
        'activation': tf.nn.relu
    },
    'upsampling': {
        'filters': 32,
        'kernel_size': 3,
        'strides': ( 2, 2 ),
        'padding': 'same'
    },
    'concat': {
        'axis': -1
    },
    'max_pool': {
        'strides': [1, 2, 2, 1],
        'ksize': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'fc': {
        'initialization': default_initialization,
        'units': 512,
        'activation': tf.nn.relu
    },
    'dropout': {
        'keep_prob': 0.5
    },
    'caps': {
        'routing': {
            'method': 'agreement',
            'n_iterations': 2
        }
    },
    'caps_margin_loss': {
        'absent_threshold': 0.1,
        'absent_weight': 0.5,
        'present_threshold': 0.9
    },
    'argmax': {
        'axis': -1
    },
    'reduce_max': {
        'axis': -1
    },
    'gan_loss': {
        'generator_scope': 'generator',
        'discriminator_scope': 'discriminator'
    },
    'h_linear': {
        'c': 0.05
    }
}

