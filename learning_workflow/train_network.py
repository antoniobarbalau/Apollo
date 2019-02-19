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
    batch_size = 100,
    optimizers = ['optimizer'],
    summary = {
        'path': '../trained_models/summaries/fer',
        'scalars': ['cross_entropy_0/output:0'],
    }
)

trainer.train_epoch()



