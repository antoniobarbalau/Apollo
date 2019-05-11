import sys
sys.path.append('..')

from functools import reduce
import cv2
import genetor
import glob
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import random
import tensorflow as tf


WAYS = 5
n_tests = 1000

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

get_class = lambda filepath: int('1' + os.path.basename(filepath)[:4])

train_dir = '/home/mpopescu/tonio/lake_omniglot/python/images_evaluation'
train_filenames = glob.glob(train_dir + '/**/**/*')
classes = list(set([
    get_class(filepath)
    for filepath in train_filenames
]))
train_dict = {
    cluster: [
        filename
        for filename in train_filenames
        if get_class(filename) == cluster
    ]
    for cluster in classes
}

def extract_from_class(class_n, n_samples):
    filenames = np.random.choice(
        train_dict[class_n], n_samples, replace = False
    )
    return [
        np.expand_dims(
            cv2.resize(
                cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
                (32, 32)
            ), axis = -1
        ) / 255.
        for filepath in filenames
    ]

def input_feeder(iteration_n, batch_size):
    samples_classes = np.random.choice(
        classes, WAYS,
        replace = False
    )
    output = reduce(
        operator.add,
        [
            extract_from_class(samples_classes[0], 2),
            *[
                extract_from_class(class_n, 1)
                for class_n in samples_classes[1:]
            ]
        ]
    )

    return output

trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    batch_size = 1,
    optimizers = [],
    n_samples = n_tests,
    placeholders = {
        'input:0': input_feeder,
    },
    return_values = [
        'few_shot_loss_0/accuracy_sum:0'
    ]
)

accs = trainer.train_epoch()
print(np.sum(accs) / 1000.)


