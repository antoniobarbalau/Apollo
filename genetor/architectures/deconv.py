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
                'stride': s,
                'kernel_size': k
            }
        }]

    return architecture

