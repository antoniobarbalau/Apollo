import sys
sys.path.append('..')

import cv2
import genetor
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf


train_filenames = glob.glob('../data/raw/mnist/test/*')
random.shuffle(train_filenames)

protos = pickle.load(open('./protos.pkl', 'rb'))
knn = KNeighborsClassifier(
    n_neighbors = 1
)
knn.fit(protos, range(10))

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
    batch_size = 1,
    optimizers = [],
    n_samples = len(train_filenames),
    placeholders = {
        'input:0': input_feeder
    },
    return_values = [
        'encoding:0'
    ]
)

encodings = trainer.train_epoch()
encodings = np.reshape(encodings, [-1, 256])
labels = np.array([
    int(os.path.basename(filepath)[0])
    for filepath in train_filenames
])
predictions = knn.predict(encodings)
print(np.mean(predictions))
print(predictions)
print(np.mean(predictions == labels))
# print(encodings.shape)




