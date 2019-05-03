import sys
sys.path.append('..')

import cv2
import genetor
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import genetor
import os
import pickle
import random
import tensorflow as tf

# WAYS = 2
# SHOTS_S = 5
# SHOTS_Q = 5
IM_SHAPE = [256, 256, 3]
ENC_SIZE = 1024
global_class_n = 0
iteration_n = 0
batch_size = 1
n_iterations = 0

train_dir = '../../datasets/named_logos_04_22_split/train'
n_classes = len(os.listdir(train_dir))
train_filenames = glob.glob(train_dir + '/**/*')
train_dir = '../../datasets/named_logos_04_22_split/test_new_classes'
n_classes += len(os.listdir(train_dir))
train_filenames += glob.glob(train_dir + '/**/*')
train_dict = {
    cluster: [
        filename
        for filename in train_filenames
        if int(os.path.basename(os.path.dirname(filename))) == cluster
    ]
    for cluster in range(n_classes)
}


def input_feeder(a, b):
    filenames = train_dict[global_class_n]
    output = [
        cv2.resize(
            cv2.imread(filepath),
            tuple(reversed(IM_SHAPE[:-1]))
        ) / 255.
        for filepath in filenames[
            iteration_n * batch_size: (iteration_n + 1) * batch_size
        ]
    ]
    return output


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    batch_size = 8,
    optimizers = [],
    n_samples = 100,
    placeholders = {
        'input:0': input_feeder,
    },
    return_values = [
        'encoding:0'
    ]
)

proto = []
for class_n in range(n_classes):
    global_class_n = class_n
    n_samples = len(train_dict[global_class_n])
    iteration_n = 0
    batch_size = 1
    n_iterations = math.ceil(n_samples / batch_size)
    outputs = []
    for iteration_n in range(n_iterations):
        outputs += trainer.train_iteration()
    outputs = np.array(outputs)
    outputs = np.reshape(outputs, [-1, ENC_SIZE])
    # outputs = tf.constant(outputs)
    # p = genetor.components.h_avg(outputs)
    # p = trainer.session.run(p)
    p = np.mean(outputs, axis = 0)
    proto.append(p)

proto = np.stack(proto)
pickle.dump(proto, open('./protos.pkl', 'wb'))

