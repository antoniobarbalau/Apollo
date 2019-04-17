import sys
sys.path.append('..')
import genetor
import glob
import tensorflow as tf

session = tf.Session()

# architecture = [{
#     'type': 'tf_data',
#     'params': {
#         'meta_path': '../data/tf_records/mnist/train/meta.json',
#         'parsers': {
#             'input': genetor.components.parse_image(shape = [28, 28, 1])
#         },
#         'create_placeholders_for': ['input', 'target'],
#         'return': 'input'
#     }
input = tf.placeholder(
    shape = [None, 28, 28, 1],
    dtype = tf.float32,
    name = 'input'
)
architecture = [{
    'type': 'input',
    'input': input,
}, {
    'type': 'hyperbolic_linear'
}]

loss = genetor.builder.new_graph(architecture = architecture)
print(loss.shape)

# optimizer = tf.train.AdamOptimizer().minimize(loss, name = 'optimizer')

# saver = tf.train.Saver()
# session.run(tf.global_variables_initializer())
# saver.save(session, '../trained_models/checkpoints/mnist/ckpt')


