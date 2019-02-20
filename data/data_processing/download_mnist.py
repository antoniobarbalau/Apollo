import cv2
import os
import shutil

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../raw/MNIST_data', one_hot = False)


folders = ['../raw/mnist/train', '../raw/mnist/val', '../raw/mnist/test']
sets = [mnist.train, mnist.validation, mnist.test]

for folder, data in zip(folders, sets):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, (im, label) in enumerate(zip(data.images, data.labels)):
        cv2.imwrite(
            os.path.join(folder, f'{label}_{i}.png'),
            (im * 255).reshape([28, 28])
        )



