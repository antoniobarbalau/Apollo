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
epoch_n = 0
margin_vect = np.concatenate([
    np.ones(1) * .2,
    np.linspace(.2, .9, 27),
    np.ones(22) * .9
])
learning_rate_vect = np.concatenate([
    np.ones(17) * 1e-3,
    np.linspace(1e-3, 1e-5, 27),
    np.ones(6) * 1e-5
])


def normalize(im):
    return im / 255 - .5


def adjust_bc(im, brightness, contrast):
    if brightness != 0:
        shadow = brightness if brightness > 0 else 0
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
 
        output = cv2.addWeighted(im, alpha_b, im, 0, gamma_b)
    else:
        output = im.copy()
 
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
 
        output = cv2.addWeighted(output, alpha_c, output, 0, gamma_c)
 
    return output


def crop(im, min_percentage):
    n_l = np.random.randint(im.shape[0] * min_percentage, im.shape[0])
    l_start = np.random.randint(0, im.shape[0] - n_l)
    n_c = np.random.randint(im.shape[1] * min_percentage, im.shape[1])
    c_start = np.random.randint(0, im.shape[1] - n_c)
    return cv2.resize(im[
        l_start: l_start + n_l,
        c_start: c_start + n_c
    ], (48, 48))


def augment(im):
    if epoch_n < 10:
        return im

    if epoch_n < 20:
        if np.random.uniform() < .5:
            return np.fliplr(im)
        return im

    if epoch_n < 35:
        crop_prob = np.linspace(.5, .7, 15)[epoch_n - 20]
        min_crop = np.linspace(.95, .7, 15)[epoch_n - 20]
        if np.random.uniform() < crop_prob:
            im = crop(im, min_crop)
        if np.random.uniform() < .5:
            im = np.fliplr(im)
        return im

    if np.random.uniform() < .7:
        im = crop(im, .7)
    if np.random.uniform() < .5:
        im = np.fliplr(im)
    e = np.minimum(epoch_n, 39) - 35
    brightness_limit = np.linspace(0, 100, 5)[e]
    contrast_limit = np.linspace(0, 50, 5)[e]
    brightness = np.random.uniform(-brightness_limit, brightness_limit)
    contrast = np.random.uniform(-contrast_limit, contrast_limit)
    r = np.random.uniform()
    if r < .1:
        return adjust_bc(im, brightness, 0)
    if r < .2:
        return adjust_bc(im, 0, contrast)
    if r < .3:
        return adjust_bc(im, brightness, contrast)
    return im


def input_feeder(iteration_n, batch_size):
    output = [
        np.expand_dims(
            normalize(
                cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            ),
            axis = -1
        )
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


def margin_feeder(a, b):
    return margin


def learning_rate_feeder(a, b):
    return learning_rate


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    batch_size = 10,
    optimizers = [],
    n_samples = len(train_filepaths),
    placeholders = {
        'input:0': input_feeder,
        'target:0': target_feeder,
        # 'margin:0': margin_feeder,
        # 'learning_rate:0': learning_rate_feeder,
        'batch_norm_is_training:0': lambda a, b: False
    },
    return_values = [
        'cross_entropy_0/accuracy_sum:0'
    ]
)

margin = 0
losses = trainer.train_epoch()
print(np.round(np.sum(losses) / len(train_filepaths), 5))

