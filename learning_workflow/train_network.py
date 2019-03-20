import sys
sys.path.append('..')

import cv2
import genetor
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import random

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

train_filenames = glob.glob('../data/raw/mnist/train/*')
random.shuffle(train_filenames)

def input_feeder(iteration_n, batch_size):
    return [
        (np.expand_dims(
            cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
            axis = -1
        ) - 128.) / 255.
        for filepath in train_filenames[
            iteration_n * batch_size: (iteration_n + 1) * batch_size
        ]
    ]
def target_feeder(iteration_n, batch_size):
    return [
        int(os.path.basename(filepath).split('_')[0])
        for filepath in train_filenames[
            iteration_n * batch_size: (iteration_n + 1) * batch_size
        ]
    ]


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    # record_paths = glob.glob(f'../data/tf_records/mnist/train/*.tfrecords'),
    batch_size = 100,
    optimizers = ['optimizer'],
    n_samples = len(train_filenames),
    placeholders = {
        'input:0': input_feeder,
        'target:0': target_feeder
    },
    summary = {
        'path': '../trained_models/summaries/mnist',
        'scalars': ['cross_entropy_0/accuracy_mean:0'],
    },
    return_values = ['cross_entropy_0/accuracy_sum:0']
)

# trainer.train_iteration()
while(1):
    val = trainer.train_epoch()
    random.shuffle(train_filenames)
    print(np.sum(val[:, 0]) / trainer.n_samples)



