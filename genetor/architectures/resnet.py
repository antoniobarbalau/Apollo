from .. import components


def generate_architecture(structure):
    filters = structure['filters']
    is_training = components.to_tensor(structure['is_training'])

    architecture = [{
        'type': 'conv',
        'params': {
            'filters': filters[0],
            'kernel_size': 7
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
                'stride': 1 if i == 0 or filters[i - 1] == f else 2
            }
        }, {
            'type': 'batch_norm',
            'params': {
                'is_training': is_training
            }
        }, {
            'type': 'conv',
            'params': {
                'filters': f
            }
        }, {
            'type': 'batch_norm',
            'output_label': f'add_{i + 1}',
            'params': {
                'is_training': is_training
            }
        }, {
            'type': 'skip',
            'params': {
                'other': f'add_{i}:0'
            }
        }]

    return architecture

