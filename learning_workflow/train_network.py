import sys
sys.path.append('..')

import cv2
import genetor
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/mnist/ckpt.meta',
    record_paths = glob.glob(f'../data/tf_records/mnist/train/*.tfrecords'),
    batch_size = 100,
    optimizers = ['optimizer'],
    summary = {
        'path': '../trained_models/summaries/mnist',
        'scalars': ['cross_entropy_0/accuracy_mean:0'],
    }
)

while(1):
    trainer.train_epoch()
    # trainer.save()



