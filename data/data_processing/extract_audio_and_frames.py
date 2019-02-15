import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess


n_frames = 3
n_seconds = 6


video_path = './piano.webm'
start_time = 70
has_audio_extracted = True
audio_path = './dataset/piano.mp3'

output_path = './dataset'
if not os.path.exists(output_path):
    os.makedirs(output_path)


# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)


# frame_indexes = np.linspace(
#     start = fps * start_time,
#     stop = fps * start_time + (fps * n_seconds),
#     num = n_frames
# )


# def preprocess_frame(frame):
#     output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     output = cv2.resize(output, (224, 244))

#     return output

# frames = []
# i = 0
# while cap.isOpened():
#     ret, frame = cap.read()

#     if ret == False:
#         break

#     if i in frame_indexes:
#         frames.append(preprocess_frame(frame))
#         if len(frames) == len(frame_indexes):
#             break

#     i += 1

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

print(cart.shape)

# reconstructed = librosa.core.istft(cart)

# resample = librosa.core.stft(reconstructed,
#                              n_fft = 1022,
#                              hop_length = 256)
# print(resample.shape)

# print(reconstructed.shape)

# librosa.output.write_wav('./reconstructed.mp3', reconstructed, sr = 11000)


