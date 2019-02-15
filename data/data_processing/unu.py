import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa


# p2z = lambda p: p[0] * np.exp( 1j * p[1] )
# z2p = lambda z: ( np.abs(z), np.angle(z) )


# audio, sample_rate = librosa.load('./audio.mp3')
# audio = audio[10: 10 + sample_rate]
# # print(audio.shape)

# f = librosa.core.stft(audio)
# # print(f.shape)
# polar = np.array([
#     [z2p(x) for x in a]
#     for a in f
# ])
# spec = polar[:, :, 0]
# phase = polar[:, :, 1]
# spec = np.log(spec)
# # print(np.min(spec), np.max(spec), np.mean(spec))

# print(spec.shape)
# # print(phase.shape)
# rs = np.concatenate([np.expand_dims(spec, axis = -1), 
#               np.expand_dims(phase, axis = -1)], axis = -1)
# # print(rs.shape)
# o = np.array([
#     [p2z(x) for x in a]
#     for a in rs
# ])

# # plt.imshow(spec, cmap = 'gray')
# # plt.show()
# o = librosa.core.istft(f)

# librosa.output.write_wav('./reconstructed.wav', o, sr = sample_rate)



cap = cv2.VideoCapture('./video.webm')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    i += 1

    if i < 200:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame.shape)
    frame = cv2.resize(frame, (960, 528))
    plt.imshow(frame)
    plt.show()

    break






