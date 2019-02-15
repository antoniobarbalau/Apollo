import sys

sys.path.append('..')
import cv2
import matplotlib.pyplot as plt
import glob
import os
import tensorflow as tf
import subprocess
import genetor
import numpy as np
import librosa

n_frames = 3
n_seconds = 6


video_path = '../data/raw/piano.webm'
start_time = 70
has_audio_extracted = True
audio_path = '../data/raw/piano.mp3'

output_path = '../data/raw/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)


frame_indexes = np.linspace(
    start = fps * start_time,
    stop = fps * start_time + (fps * n_seconds),
    num = n_frames
)


def preprocess_frame(frame):
    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = cv2.resize(output, (224, 224))

    return output

frames = []
i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret == False:
        break

    if i in frame_indexes:
        frames.append(preprocess_frame(frame))
        if len(frames) == len(frame_indexes):
            break

    i += 1

# for frame in frames:
#     plt.imshow(frame)
#     plt.show()


def extract_audio(filepath):
    global output_path
    filename = os.path.basename(filepath).split('.')[:-1]
    filename = '.'.join(filename)
    output_file = os.path.join(output_path, filename + '.mp3')

    subprocess.run(['ffmpeg', '-i', filepath, '-codec:a', 'libmp3lame',
                   '-qscale:a', '0', output_file])

    return output_file
if not has_audio_extracted:
    audio_path = extract_audio(video_path)

def get_spectrogram(audio_path):
    audio, sr = librosa.core.load(audio_path, sr = 11000)
    sample = audio[sr * start_time: sr * start_time + 256 * 252 + 1022] # verificat din nou

    stft = librosa.core.stft(sample, n_fft = 1022, hop_length = 256)


    p2c = lambda p: p[0] * np.exp( 1j * p[1] )
    c2p = lambda z: ( np.abs(z), np.angle(z) )
    polar = np.array(list(map(
        lambda line: list(map(
            c2p,
            line
        )),
        stft
    )))
    powers = polar[:, :, 0]
    angles = polar[:, :, 1]

    powers = np.log(powers)

    polar = np.concatenate(
        [np.expand_dims(powers, axis = -1), np.expand_dims(angles, axis = -1)],
        axis = -1
    )
    cart = np.array(list(map(
        lambda line: list(map(
            p2c,
            line
        )),
        polar
    )))

    return powers

s1 = get_spectrogram('../data/raw/cello.mp3')
s2 = get_spectrogram('../data/raw/piano.mp3')
s = s1 + s2
target_mask = s2 / s


# def noise_generator(iteration_n, batch_size):
#     return np.random.randn(batch_size, 100)

# def is_real_generator(iteration_n, batch_size):
#     return [1.] * batch_size + [0.] * batch_size

#todo: save
# constant placeholders
# get('attr' peste tot
# de scris la consola

trainer = genetor.train.Coordinator(
    ckpt_meta_path = '../trained_models/checkpoints/tsop/ckpt.meta',
    placeholders = {
        'batch_size:0': 1,
        'n_samples:0': 1,
        'video:0': lambda x, y: frames,
        'spectrogram:0': lambda x, y: [np.expand_dims(s, axis = -1)],
        'target_mask:0': lambda x, y: [target_mask],
    },
    optimizers = ['optimizer'],
    summary = {
        'scalars': ['loss:0'],
        'path': '../trained_models/summaries/tsop'
    }
)

while 1:
    trainer.train_epoch()
    trainer.save()






