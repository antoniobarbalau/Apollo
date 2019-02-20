from .. import components


def generate_architecture(structure):
    filters = structure['filters']
    kernels = structure['kernels']
    units = structure['units']
    biasless = structure.get('biasless', False)

    architecture = []
    for f, k in zip(filters, kernels):
        architecture += [{
            'type': 'conv',
            'params': {
                'filters': f,
                'kernel_size': k,
                'biasless': biasless
            }
        }, {
            'type': 'max_pool'
        }]
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

