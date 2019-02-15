import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import tensorflow as tf


# session = tf.Session()
# saver = tf.train.import_meta_graph('./base_ckpt/model.meta')
# saver.restore(session, tf.train.latest_checkpoint('./base_ckpt'))
# graph = tf.get_default_graph()


N_FRAMES = 3
N_SECONDS = 6
VIDEO_DIR = './dataset/dummy_dataset/'
AUDIO_DIR = './dataset/dummy_audio/'


def extract_audio_to_filepath(video_path, audio_path):
    subprocess.run(['ffmpeg', '-i', video_path, '-codec:a', 'libmp3lame',
                   '-qscale:a', '0', audio_path])


def extract_audio_to_hdd(video_path, audio_dir):
    filename = os.path.basename(video_path).split('.')[:-1]
    filename = '.'.join(filename)
    audio_path = os.path.join(audio_dir, filename + '.mp3')

    if not os.path.exists(output_file):
        extract_audio_to_filepath(
            video_path = video_path,
            audio_path = audio_path
        )

    return audio_path


def preprocess_frame(frame):
    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = cv2.resize(output, (224, 244))

    return output


def extract_frames(video_path, start_time, n_frames,
                   debug = False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_indexes = np.linspace(
        start = fps * start_time,
        stop = fps * start_time + (fps * n_seconds),
        num = n_frames
    )

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

    if debug:
        for frame in frames:
            plt.imshow(frame)
            plt.show()

    return frames


def extract_spectrogram(audio_path, start_time, n_seconds):
    return audio_path


def gen_im_and_sound(video_path, start_time,
                     n_seconds = N_SECONDS,
                     n_frames = N_FRAMES,
                     audio_dir = AUDIO_DIR,
                     debug = False):
    frames = extract_frames(
        video_path = video_path,
        start_time = start_time,
        n_frames = n_frames
    )

    audio_path = extract_audio_to_hdd(
        video_path = video_path,
        audio_dir = AUDIO_DIR
    )

    spectrogram = extract_spectrogram(
        audio_path = audio_path,
        start_time = start_time,
        n_seconds = n_seconds
    )

    return frames, spectrogram
        








gen_im_and_sound('./dataset/dummy_dataset/cello.mp4', 60)





