from .. import components
import tensorflow as tf


def generate_architecture(structure):
    filters = structure.get('filters', [])
    kernels = structure.get('kernels', 5)
    units = structure.get('units', [])
    strides = structure.get('strides', [])
    biasless = structure.get('biasless', False)
    conv_dropout_rate = structure.get('conv_dropout_rate', None)
    batch_norm_is_training = structure.get('batch_norm_is_training', None)
    activation = structure.get('activation', tf.nn.relu)
    final_activation = structure.get('final_activation', None)
    output_label = structure.get('output_label', None)

    if type(kernels) is not list:
        kernels = [kernels] * len(filters)

    conv_params = dict()

    if conv_dropout_rate is not None:
        conv_params['dropout_rate'] = conv_dropout_rate
    if batch_norm_is_training is not None:
        conv_params['batch_norm_is_training'] = batch_norm_is_training

    architecture = []
    for i, (f, k) in enumerate(zip(filters, kernels)):
        architecture += [{
            'type': 'conv',
            'params': {
                'filters': f,
                'kernel_size': k,
                'stride': 1 if not strides else strides[i],
                'biasless': biasless,
                'activation': activation,
                **conv_params
            }
        }, {
            'type': 'max_pool'
        }]

    if not units:
        return architecture
    architecture += [{
        'type': 'flatten'
    }]

    for u in units:
        architecture += [{
            'type': 'fc',
            'params': {
                'units': u,
                'activation': activation,
                **conv_params
            }
        }]
    architecture[-1]['params']['activation'] = final_activation
    architecture[-1]['params']['batch_norm_is_training'] = None

    if output_label:
        architecture[-1]['output_label'] = output_label

    return architecture

