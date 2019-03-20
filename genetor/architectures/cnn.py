from .. import components


def generate_architecture(structure):
    filters = structure['filters']
    kernels = structure['kernels']
    units = structure.get('units', [])
    strides = structure.get('strides', [])
    biasless = structure.get('biasless', False)

    architecture = []
    for i, (f, k) in enumerate(zip(filters, kernels)):
        architecture += [{
            'type': 'conv',
            'params': {
                'filters': f,
                'kernel_size': k,
                'stride': 1 if not strides else strides[i],
                'biasless': biasless
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
                'units': u
            }
        }]

    return architecture

