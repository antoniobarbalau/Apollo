from .. import components
import tensorflow as tf


def generate_architecture(structure):
    filters = structure.get('filters', [])
    kernels = structure.get('kernels', [])
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

    architecture = [{
        'type': 'conv',
        'params': {
            'filters': filters[0],
            'kernel_size': 7,
            # 'stride': 1 if not strides else strides[i],
            'biasless': biasless,
            'activation': activation,
            **conv_params
        }
    }, {
        'type': 'max_pool',
        'output_label': 'add_0',
    }]
    for i, f in enumerate(filters):
        architecture += [{
            'type': 'conv',
            'params': {
                'filters': f,
                'stride': 1 if i == 0 or filters[i - 1] == f else 2,
                # 'stride': 1 if not strides else strides[i],
                'biasless': biasless,
                'activation': activation,
                **conv_params
            }
        }, {
            'type': 'conv',
            'output_label': f'add_{i + 1}',
            'params': {
                'filters': f,
                # 'stride': 1 if not strides else strides[i],
                'biasless': biasless,
                'activation': activation,
                **conv_params
            }
        }, {
            'type': 'skip',
            'params': {
                'other': f'add_{i}:0'
            }
        }]

    architecture += [{
        'type': 'avg_pool'
    }, {
        'type': 'flatten'
    }]


    if not units:
        return architecture
    architecture += [{
        'type': 'flatten'
    }]

    units_final = units[-1]
    units = units[:-1]
    for u in units:
        architecture += [{
            'type': 'fc',
            'params': {
                'units': u,
                'activation': activation
            }
        }]
    architecture += [{
        'type': 'fc',
        'params': {
            'units': units_final,
            'activation': final_activation
        }
    }]

    if output_label:
        architecture[-1]['output_label'] = output_label


    return architecture

