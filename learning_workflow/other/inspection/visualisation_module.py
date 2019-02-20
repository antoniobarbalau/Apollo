import sys
sys.path.append('../../../')

import genetor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


ckpt_meta_path = '../../../trained_models/checkpoints/mnist/ckpt.meta'
session = tf.Session()
saver = tf.train.import_meta_graph(ckpt_meta_path)
saver.restore(session, tf.train.latest_checkpoint(
    os.path.dirname(ckpt_meta_path)
))
graph = tf.get_default_graph()

input = graph.get_tensor_by_name('input:0')

i = 1

last_layer = graph.get_tensor_by_name(f'max_pool_{i}_1/output:0')

while i >= 0:
    conv_output_shape = graph.get_tensor_by_name(f'conv_{i}/output:0')
    conv_output_shape = [
        conv_output_shape.shape[i].value
        for i in range(1, 4)
    ]

    switches = graph.get_tensor_by_name(f'max_pool_{i}_1/switches:0')
    switches = tf.reshape(switches, [-1])
    unpooled = tf.Variable(initial_value = tf.zeros(
        shape = [np.prod(conv_output_shape)],
        dtype = tf.float32
    ), dtype = tf.float32)
    session.run(unpooled.initializer)
    tf.scatter_update(
        unpooled,
        indices = switches,
        updates = tf.reshape(last_layer, [-1])
    )

    unpooled = tf.reshape(unpooled, [-1, *conv_output_shape])
    rectified = tf.nn.relu(unpooled)
    deconv = genetor.components.deconv_kernel(
        input = rectified,
        kernel = graph.get_tensor_by_name(f'conv_{i}/kernel:0'),
        padding = 'SAME',
        stride = 1
    )

    last_layer = deconv

    i -= 1

act = session.run(
    [deconv],
    feed_dict = {
        input: [
            np.expand_dims(cv2.imread(
                '../../../data/raw/mnist/train/0_10.png',
                cv2.IMREAD_GRAYSCALE
            ), axis = -1) / 255.
        ]
    }
)
act = np.reshape(act[0], [28, 28])
plt.imshow(act, cmap = 'gray')
plt.show()
    





