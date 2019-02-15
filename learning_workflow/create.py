import tensorflow as tf
import numpy as np
import os
import graph as nicolae
from graph import components


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


n_frames = 3
output_video = tf.reshape(output_video,
                          [-1, n_frames, output_video.shape[1].value,
                           output_video.shape[2].value, K])
output_video = tf.layers.max_pooling3d(
    output_video,
    pool_size = [n_frames, 1, 1],
    strides = [1, 1, 1],
    padding = 'valid',
    data_format = 'channels_last'
)
output_video = tf.reshape(output_video, [-1, output_video.shape[2].value,
                           output_video.shape[3].value, K])



kernel = tf.Variable(
    initial_value = components.default_initialization(shape = [K]),
    dtype = tf.float32)
bias = tf.Variable(
    initial_value = tf.zeros(shape = [K]),
    dtype = tf.float32)

reshaped_im = tf.reshape(
    output_video,
    [-1, output_video.shape[1] * output_video.shape[2], K]
)
reshaped_im = tf.transpose(reshaped_im, [0, 2, 1])
reshaped_kernel = tf.reshape(
    kernel,
    [K, 1]
)
tiled_kernel = tf.tile(
    reshaped_kernel,
    [1, output_video.shape[1] * output_video.shape[2]]
)
conv_kernel = tiled_kernel * reshaped_im
conv_kernel = tf.expand_dims(conv_kernel, axis = 1)
conv_kernel = tf.expand_dims(conv_kernel, axis = 1)

batch_size = tf.shape(output_video)[0]
video_height = output_video.shape[1].value 
video_width = output_video.shape[2].value
n_pixels = video_width * video_height

conv_kernel = tf.transpose(conv_kernel, [1, 2, 0, 3, 4])
conv_kernel = tf.reshape(conv_kernel, [1, 1, K * batch_size, n_pixels])

output_video = tf.transpose(output_video, [1, 2, 0, 3])
output_video = tf.reshape(output_video,
                          [1, video_height, video_width, batch_size * K])

pixel_spectrograms = tf.nn.depthwise_conv2d(output_video,
                                            filter = conv_kernel,
                                            strides = [1, 1, 1, 1],
                                            padding = 'VALID')
pixel_spectrograms = tf.reshape(pixel_spectrograms,
                                [video_height, video_width, batch_size, K, n_pixels])
pixel_spectrograms = tf.transpose(pixel_spectrograms, [2, 0, 1, 3, 4])
pixel_spectrograms = tf.reduce_sum(pixel_spectrograms, axis = 3)


saver = tf.train.Saver()
session.run(tf.global_variables_initializer())

saver.save(session, './base_ckpt/model')

# n_frames = 3
# 11kHz
# audio sample 6 seconds
# stft window size 1022
# hop length 256

