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
        np.expand_dims(
            cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
            axis = -1
        ) /  255.
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
    batch_size = 20,
    optimizers = [],
    n_samples = len(train_filenames),
    placeholders = {
        'input:0': input_feeder,
        'target:0': target_feeder
    },
    # summary = {
    #     'path': '../trained_models/summaries/mnist',
    #     'images': [{
    #         'tensor': 'reconstruction:0',
    #         'max_outputs': 4,
    #         'name': 'reconstruction'
    #     }]
    # },
    return_values = [
        'output:0'
    ]
)

for _ in range(40):
    losses = trainer.train_epoch()
    print(losses)
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




