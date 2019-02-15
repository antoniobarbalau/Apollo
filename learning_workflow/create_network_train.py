import sys
sys.path.append('..')
import genetor
from genetor.components.initializations import default_initialization
import glob
import tensorflow as tf

K = 16
N_FRAMES = 3

session = tf.Session()

video = tf.placeholder(shape = [None, 224, 224, 3],
                       dtype = tf.float32,
                       name = 'video')
spectrogram = tf.placeholder(shape = [None, 512, 256, 1],
                             dtype = tf.float32,
                             name = 'spectrogram')
target_mask = tf.placeholder(shape = [None, 512, 256],
                             dtype = tf.float32,
                             name = 'target_mask')
is_training = tf.placeholder_with_default(False,
                                          shape = [],
                                          name = 'is_training')

architecture = [{
    'type': 'input',
    'input': video
}, *genetor.builder.new_architecture(
    model = 'resnet',
    structure = {
        'is_training': is_training
    }
)]
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
        'filters': K,
        'activation': tf.nn.sigmoid
    }
}]
output_video = genetor.builder.new_graph(architecture = architecture)

architecture = [{
    'type': 'input',
    'input': spectrogram
}, *genetor.builder.new_architecture(
    model = 'unet',
    structure = {
        'filters': [K, 32, 64, 128, 256, 512, 512]
    }
)]
with tf.variable_scope('unet'):
    output_spectrogram = genetor.builder.new_graph(architecture = architecture)



output_video = tf.reshape(
    output_video,
    shape = [-1, N_FRAMES, output_video.shape[1], output_video.shape[2], K]
)
frames = tf.unstack(
    output_video,
    axis = 1
)
stmp_frames = tf.concat(frames, axis = 1) # spatio temporal max pooling
stmp = tf.nn.max_pool(
    stmp_frames,
    ksize = [1, stmp_frames.shape[1], stmp_frames.shape[2], 1],
    strides = [1, 1, 1, 1],
    padding = 'VALID'
)

kernel = tf.Variable(
    initial_value = default_initialization(shape = [K]),
    dtype = tf.float32,
    name = 'audio_synthesizer_kernel'
)
bias = tf.Variable(
    initial_value = tf.zeros(shape = [K]),
    dtype = tf.float32,
    name = 'audio_synthesizer_bias'
)

stmp = tf.tile(stmp, [1, output_spectrogram.shape[1], output_spectrogram.shape[2], 1])
mask = stmp * output_spectrogram

mask = tf.reduce_sum(mask, axis = -1,
                     name = 'output_mask')


loss = tf.reduce_mean(tf.abs(mask - target_mask), name = 'loss')
optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')


saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
saver.save(session, '../trained_models/checkpoints/tsop/ckpt')



