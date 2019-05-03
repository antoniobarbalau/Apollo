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


WAYS = 2
SHOTS_S = 1
SHOTS_Q = 1
IM_SHAPE = [256, 256, 3]

train_dir = '../../datasets/named_logos_04_22_split/train'
train_filenames = glob.glob(train_dir + '/**/*')
n_classes = len(os.listdir(train_dir))
train_dict = {
    cluster: [
        filename
        for filename in train_filenames
        if int(os.path.basename(os.path.dirname(filename))) == cluster
    ]
    for cluster in range(n_classes)
}


def extract_from_class(class_n, n_samples):
    filenames = np.random.choice(train_dict[class_n], n_samples)
    return [
        cv2.resize(
            cv2.imread(filepath),
            tuple(reversed(IM_SHAPE[:-1]))
        ) / 255.
        for filepath in filenames
    ]


def input_feeder(iteration_n, batch_size):
    output = []
    for _ in range(batch_size):
        classes = np.random.choice(range(n_classes), WAYS)
        output += [
            extract_from_class(class_n, SHOTS_S + SHOTS_Q)
            for class_n in classes
        ]
    output = np.reshape(output, [-1, *IM_SHAPE])
    return output


lr = 1e-3
def lr_feeder(a, b):
    return lr


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    batch_size = 10,
    optimizers = ['optimizer'],
    n_samples = 1,
    placeholders = {
        'input:0': input_feeder,
        'learning_rate:0': lr_feeder
    },
    return_values = [
        'loss:0'
    ]
)

old_loss = 1.
n_equal = 0
for _ in range(2000):
    losses = trainer.train_epoch()
    trainer.save()

    current_loss = np.mean(losses)
    if np.abs(current_loss - old_loss) < 1e-4:
        n_equal += 1
    if n_equal == 10:
        lr /= 2.
        lr = np.min([lr, 1e-8])
        n_equal = 0
    old_loss = current_loss

    print(current_loss)
    print(lr)


