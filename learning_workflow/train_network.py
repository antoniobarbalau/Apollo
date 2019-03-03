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
    ckpt_meta_path = '../trained_models/checkpoints/fer/ckpt.meta',
    record_paths = glob.glob(f'../data/tf_records/fer/train/*.tfrecords'),
    batch_size = 200,
    optimizers = ['optimizer'],
    summary = {
        'path': '../trained_models/summaries/fer',
        # 'scalars': ['caps_margin_loss_0/accuracy_mean:0'],
        'scalars': ['cross_entropy_0/accuracy_mean:0'],
    },
    # return_values = ['caps_margin_loss_0/accuracy_sum:0']
    return_values = ['cross_entropy_0/accuracy_sum:0']
)

while(1):
    val = trainer.train_epoch()
    print(np.sum(val[:, 0]) / trainer.n_samples)



