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

train_filepaths = glob.glob('../data/raw/mnist/train/*')


def input_feeder(iteration_n, batch_size):
    output = [
        np.expand_dims(
            cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis = -1
        ) / 255. - .5
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
    print(output)
    return output


margin = 0.2
def margin_feeder(a, b):
    return margin


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    batch_size = 10,
    optimizers = ['optimizer'],
    n_samples = len(train_filepaths),
    placeholders = {
        'input:0': input_feeder,
        'target:0': target_feeder,
        'margin:0': margin_feeder
    },
    return_values = [
        'spread_loss_0/accuracy_sum:0'
    ]
)

for _ in range(2000):
    np.random.shuffle(train_filepaths)
    losses = trainer.train_iteration()
    print(losses)
    # trainer.save()
    margin = np.minimum(trainer.epoch_n, 7) * .1 + 0.2
    
    # print(np.mean(losses))

