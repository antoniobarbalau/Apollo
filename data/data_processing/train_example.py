import tensorflow as tf
import cv2
import math
import numpy as np
import glob
import os
import shutil
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = False, reshape = [28, 28, 1])

session = tf.Session()
saver = tf.train.import_meta_graph('./base_ckpt/model.meta')
saver.restore(session, tf.train.latest_checkpoint('./base_ckpt'))
graph = tf.get_default_graph()


noise = graph.get_tensor_by_name('noise:0')
generator_output = graph.get_tensor_by_name('generator/output:0')

# summary_images = tf.reshape(generator_output, [-1, 1025, 30, 1])
tf.summary.image(name = 'generated',
                 tensor = generator_output * 255.,
                 max_outputs = 4)

generator_loss = graph.get_tensor_by_name('generator_loss:0')
tf.summary.scalar(name = 'generator_loss',
                 tensor = generator_loss)
discriminator_loss = graph.get_tensor_by_name('discriminator_loss:0')
tf.summary.scalar(name = 'discriminator_loss',
                 tensor = discriminator_loss)
# tf.summa
summary_merged = tf.summary.merge_all()
if os.path.exists('./summary'):
    shutil.rmtree('./summary')
summary_writer = tf.summary.FileWriter(logdir = './summary/')

real_input = graph.get_tensor_by_name('real_input:0')
discriminator_input = graph.get_tensor_by_name('discriminator_input:0')
is_real = graph.get_tensor_by_name('is_real:0')
discriminator_optimizer = graph.get_operation_by_name('discriminator_optimizer')
generator_optimizer = graph.get_operation_by_name('generator_optimizer')
discriminator_output = graph.get_tensor_by_name('discriminator/output:0')
discriminator_loss = graph.get_tensor_by_name('discriminator_loss:0')

session.run(tf.global_variables_initializer())

def read_image(path):
    im = np.expand_dims(cv2.imread(path, cv2.IMREAD_GRAYSCALE), axis = -1)
    # print(np.mean(im))
    return im
paths = glob.glob('../../datasets/manele_samples/*')

n_train_samples = len(paths)
n_train_samples = len(mnist.train.labels)
print(n_train_samples)

epoch_n = -1
last_dl = 100.

def train():

    global last_dl
    if last_dl < 1.:
        return

    global epoch_n
    epoch_n += 1

    batch_size = 128
    n_iterations = int(math.ceil(n_train_samples / batch_size))
    for iteration_n in range(n_iterations):
        final = min(n_train_samples, (iteration_n + 1) * batch_size)
        n_samples = final - batch_size * iteration_n
        r_noise = np.random.uniform(-0.5, .5,
                                    size = [n_samples, 100])
        # print(iteration_n)
        # print(mnist.train.images[iteration_n * batch_size:
        #                                        (iteration_n + 1) * batch_size].shape)

        # go, = session.run(
        #     [generator_output],
        _, _, sm, dl = session.run(
            [generator_optimizer, discriminator_optimizer, summary_merged, discriminator_loss],
            feed_dict = {
                noise: r_noise,
                # real_input: list(map(read_image, paths[iteration_n * batch_size:
                #                                (iteration_n + 1) * batch_size])),
                real_input: mnist.train.images[iteration_n * batch_size:
                                               (iteration_n + 1) * batch_size].reshape(
                                                   [-1, 28, 28, 1]),
                is_real: [0.0] * n_samples + [1.0] * n_samples
            })
        last_dl = dl
        # go = go[0, :, :, 0]
        summary_writer.add_summary(sm, n_iterations * epoch_n + iteration_n)
        # print(np.mean(go))
        # cv2.imwrite('./manea.jpg', go * 255.)

        # print(go.shape)
        # saver.save(session, './ckpt/model')
        


train()
while(1):
    train()
    # saver.save(session, './ckpt/model')


