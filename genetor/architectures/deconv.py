from .. import components
import tensorflow as tf


def generate_architecture(structure):
    filters = structure['filters']
    strides = structure['strides']
    kernels = structure['kernels']
    deconv_type = structure['type']
    final_activation = structure.get('final_activation', None)

    architecture = []
    for f, s, k in zip(filters, strides, kernels):
        architecture += [{
            'type': deconv_type,
            'params': {
                'filters': f,
                'strides': (s, s),
                'kernel_size': k,
                'padding': 'same',
                'activation': tf.nn.relu
            }
        }]
    architecture[-1]['params']['activation'] = final_activation

    return architecture

