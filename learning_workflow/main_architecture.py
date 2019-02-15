import tensorflow as tf
import numpy as np
import os
import graph as nicolae


session = tf.Session()


video = tf.placeholder(shape = [None, 224, 224, 3],
                       dtype = tf.float32,
                       name = 'video')
spectrogram = tf.placeholder(shape = [None, 512, 256, 1],
                             dtype = tf.float32,
                             name = 'spectrogram')
is_training = tf.placeholder_with_default(False,
                                          shape = [],
                                          name = 'is_training')

K = 10
architecture = nicolae.builder.new_architecture(
    model = 'resnet',
    input = video,
    structure = {
        'is_training': is_training
    }
)
n_found = 0
for i in reversed(range(len(architecture))):
    if architecture[i]['type'] == 'conv':
        architecture[i]['type'] = 'dilated_conv'
        n_found += 1
        if n_found == 2:
            break
architecture += [{
    'type': 'conv',
    'params': {
        'filters': K
    }
}]
output_video = nicolae.builder.new_graph(architecture = architecture)

architecture = nicolae.builder.new_architecture(
    model = 'unet',
    input = spectrogram,
    structure = {
        'filters': [K, 32, 64, 128, 256, 512, 512],
    }
)
with tf.variable_scope('unet'):
    output_spectrogram = nicolae.builder.new_graph(architecture = architecture)



saver = tf.train.Saver()
session.run(tf.global_variables_initializer())

saver.save(session, './base_ckpt/model')

# n_frames = 3
# 11kHz
# audio sample 6 seconds
# stft window size 1022
# hop length 1022

