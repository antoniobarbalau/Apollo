import sys
sys.path.append('..')

import cv2
import genetor
import numpy as np
import os
import tensorflow as tf


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

learning_rate_vect = np.concatenate([
    np.ones(17) * 1e-3,
    np.linspace(1e-3, 1e-5, 27),
    np.ones(6) * 1e-5
])

trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    record_paths = ['../data/tf_records/fer/train/0.tfrecords'],
    batch_size = 10,
    placeholders = {
        'learning_rate:0': lambda a, b: learning_rate,
        'batch_norm_is_training:0': lambda a, b: True
    },
    return_values = [
        'cross_entropy_0/accuracy_sum:0'
    ]
)

for e in range(50):
    learning_rate = learning_rate_vect[e]

    losses = trainer.train_epoch()
    trainer.save()

    print(
        f'Epoch {e}: ' +
        f'{np.round(np.sum(losses) / trainer.n_samples, 2)}'
    )


