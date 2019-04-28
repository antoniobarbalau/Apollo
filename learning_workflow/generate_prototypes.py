import sys
sys.path.append('..')

import cv2
import genetor
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import tensorflow as tf

WAYS = 5
SHOTS_S = 5
SHOTS_Q = 5
global_class_n = 0
iteration_n = 0
batch_size = 1000
n_iterations = 0

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
        ) / 255. #/ 0.3081 - 0.1307
        for filepath in filenames
    ]


def input_feeder(a, b):
    # output = []
    # for _ in range(batch_size):
    #     classes = np.random.choice(range(10), WAYS)
    #     output += [
    #         extract_from_class(class_n, SHOTS_S + SHOTS_Q)
    #         for class_n in classes
    #     ]
    # output = np.reshape(output, [-1, 28, 28, 1])
    # print(iteration_n)
    filenames = train_dict[global_class_n]
    output = [
        np.expand_dims(
            cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
            axis = -1
        ) / 255. #/ 0.3081 - 0.1307
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
for class_n in range(10):
    global_class_n = class_n
    n_samples = len(train_dict[global_class_n])
    iteration_n = 0
    batch_size = 1000
    n_iterations = math.ceil(n_samples // batch_size)
    outputs = []
    for iteration_n in range(n_iterations):
        outputs += trainer.train_iteration()
    outputs = np.array(outputs)
    outputs = np.reshape(outputs, [-1, 256])
    p = np.mean(outputs, axis = 0)
    proto.append(p)

proto = np.stack(proto)
pickle.dump(proto, open('./protos.pkl', 'wb'))

print(proto.shape)

# for _ in range(100):
    # losses = trainer.train_epoch()
    # print(losses)
    # print(np.mean(losses))
    # print('ok')
    # random.shuffle(train_filenames)
    # trainer.save()
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




