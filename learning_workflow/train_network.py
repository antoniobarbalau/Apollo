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


WAYS = 5
SHOTS_S = 5
SHOTS_Q = 5

train_filenames = glob.glob('../data/raw/mnist/train/*')
train_dict = {
    cluster: [
        filename
        for filename in train_filenames
        if int(os.path.basename(filename)[0]) == cluster
    ]
    for cluster in range(10)
}


def extract_from_class(class_n, n_samples):
    filenames = np.random.choice(train_dict[class_n], n_samples)
    return [
        np.expand_dims(
            cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
            axis = -1
        ) /  255. #/ 0.3081 - 0.1307
        for filepath in filenames
    ]


def input_feeder_proto(iteration_n, batch_size):
    output = []
    for _ in range(batch_size):
        classes = np.random.choice(range(10), WAYS)
        output += [
            extract_from_class(class_n, SHOTS_S + SHOTS_Q)
            for class_n in classes
        ]
    output = np.reshape(output, [-1, 28, 28, 1])
    return output


def input_feeder(iteration_n, batch_size):
    return [
        np.expand_dims(
            cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
            axis = -1
        ) / 255. #/ 0.3081 - 0.1307
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
    batch_size = 10,
    optimizers = ['optimizer'],
    n_samples = 1000,
    placeholders = {
        'input:0': input_feeder_proto,
        'target:0': target_feeder
    },
    return_values = [
        'loss:0'
    ]
)

for _ in range(100):
    losses = trainer.train_epoch()
    # print(losses)
    print(np.mean(losses))
    # print('ok')
    random.shuffle(train_filenames)
    trainer.save()
# recons = trainer.train_iteration()
# print(recons[0].shape)
# print(np.max(recons[0]))
# import cv2
# im = np.array(
        # np.squeeze(recons[0][0], axis = -1) * 255.,
        # dtype = np.uint8
    # )
# print(im)
# cv2.imwrite(
    # './haha.png',
    # im
# )




