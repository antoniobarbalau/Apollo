def generate_architecture(structure):
    filters = structure['filters']
    scope = structure['scope']

    architecture = []
    for f in filters[:-1]:
        architecture += [{
            'type': 'conv',
            'params': {
                'filters': f,
            }
        }, {
            'type': 'max_pool',
            'params': {
                'padding': 'SAME'
            }
        }]
    architecture += [{
        'type': 'conv',
        'params': {
            'filters': filters[-1],
        }
    }]
    for i, f in reversed(list(enumerate(filters[:-1]))):
        architecture += [{
            'type': 'up_conv',
            'params': {
                'filters': f,
                'output_shape': f'{scope}/conv_{i}/output:0',
                'stride': 2
            }
        }, {
            'type': 'concat',
            'params': {
                'other': [f'{scope}/conv_{i}/output:0']
            }
        }, {
            'type': 'conv',
            'params': {
                'filters': f,
            }
        }]
    
    return architecture


