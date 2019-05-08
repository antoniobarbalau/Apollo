from .. import components


def generate_architecture(structure):
    filters = structure['filters']
    strides = structure['strides']
    kernels = structure['kernels']
    deconv_type = structure['type']

    architecture = []
    for f, s, k in zip(filters, strides, kernels):
        architecture += [{
            'type': deconv_type,
            'params': {
                'filters': f,
                'strides': (s, s),
                'kernel_size': k,
                'padding': 'same'
            }
        }]

    return architecture

