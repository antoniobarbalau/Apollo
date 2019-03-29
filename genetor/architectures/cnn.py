from .. import components


def generate_architecture(structure):
    filters = structure.get('filters', [])
    kernels = structure.get('kernels', [])
    units = structure.get('units', [])
    strides = structure.get('strides', [])
    biasless = structure.get('biasless', False)
    conv_dropout_rate = structure.get('conv_dropout_rate', None)
    batch_norm_is_training = structure.get('batch_norm_is_training', None)

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
                'activation': components.gelu,
                **conv_params
            }
        }, {
            'type': 'max_pool'
        # }, {
        #     'type': 'batch_norm',
        #     'params': {
        #         'is_training': True
        #     }
        }]
    if units:
        architecture += [{
            'type': 'flatten'
        }]
    for u in units:
        architecture += [{
            'type': 'fc',
            'params': {
                'units': u,
                'activation': components.gelu
            }
        }]

    return architecture

