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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_filepaths = glob.glob('../data/raw/fer/test/*')


def input_feeder(iteration_n, batch_size):
    output = [
        np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis = -1) / 255. - .5
        for filepath in train_filepaths[
            iteration_n * batch_size:
            (1 + iteration_n) * batch_size
        ]
    ]
    return output


def target_feeder(iteration_n, batch_size):
    output = [
        int(os.path.basename(filepath)[0])
        for filepath in train_filepaths[
            iteration_n * batch_size:
            (1 + iteration_n) * batch_size
        ]
    ]
    return output


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    optimizers = [],
    n_samples = len(train_filepaths),
    batch_size = 10,
    placeholders = {
        'input:0': input_feeder,
        'target:0': target_feeder,
        'batch_norm_is_training:0': lambda a, b: False
    },
    return_values = [
        'cross_entropy_0/accuracy_sum:0'
    ]
)

losses = trainer.train_epoch()
print(np.round(np.sum(losses) / len(train_filepaths), 5))


