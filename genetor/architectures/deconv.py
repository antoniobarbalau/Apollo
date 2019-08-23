from .. import components
import tensorflow as tf


def generate_architecture(structure):
    filters = structure['filters']
    strides = structure.get('strides', 2)
    kernels = structure.get('kernels', 5)
    activation = structure.get('activation', tf.nn.relu)
    final_activation = structure.get('final_activation', None)
    batch_norm_is_training = structure.get('batch_norm_is_training', None)
    units = structure.get('units', None)
    if units is not None:
        initial_conv_shape = structure['initial_conv_shape']
    output_label = structure.get('output_label', None)

    if type(strides) is int:
        strides = [strides] * len(filters)
    if type(kernels) is int:
        kernels = [kernels] * len(filters)

    conv_params = dict()
    if batch_norm_is_training is not None:
        conv_params['batch_norm_is_training'] = batch_norm_is_training

    architecture = []
    if units is not None:
        for u in units:
            architecture += [{
                'type': 'fc',
                'params': {
                    'units': u,
                    'activation': activation,
                    **conv_params
                }
            }]
        architecture += [{
            'type': 'reshape',
            'params': {
                'shape': [-1, *initial_conv_shape]
            }
        }]

    for i, (f, s, k) in enumerate(zip(filters, strides, kernels)):
        if batch_norm_is_training is None or i == len(filters) - 1:
            architecture += [{
                'type': 'conv2d_transpose',
                'params': {
                    'filters': f,
                    'strides': (s, s),
                    'kernel_size': k,
                    'padding': 'same',
                    'activation': activation
                }
            }]
        else:
            architecture += [{
                'type': 'conv2d_transpose',
                'params': {
                    'filters': f,
                    'strides': (s, s),
                    'kernel_size': k,
                    'padding': 'same',
                    'activation': None
                }
            }, {
                'type': 'batch_norm',
                'params': {
                    'is_training': batch_norm_is_training,
                    'activation': activation
                }
            }]
    architecture[-1]['params']['activation'] = final_activation

    if output_label:
        architecture[-1]['output_label'] = output_label

    return architecture

