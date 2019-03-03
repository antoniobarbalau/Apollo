import json
import tensorflow as tf


def parse_image_with_shape(im, shape):
    output = tf.image.decode_png(im, channels = shape[-1])
    output = tf.image.resize_images(output, shape[:-1]) / 255.

    return output


def parse_image(shape):
    return lambda im: parse_image_with_shape(im, shape)


def tf_data(input, **params):
    data_format = json.load(open(params['meta_path'], 'r'))

    tf_type = {
        'bytes': tf.FixedLenFeature([], tf.string),
        'int': tf.FixedLenFeature([], tf.int64),
        'float': tf.FixedLenFeature([], tf.float32)
    }
    feature_description = {
        feature_name: tf_type[feature_type]
        for feature_name, feature_type in data_format.items()
    }
    def parse_sample(sample_proto):
        sample = tf.parse_single_example(sample_proto, feature_description)

        for feature_name, feature_parser in params['parsers'].items():
            sample[feature_name] = feature_parser(sample[feature_name])

        return sample

    record_paths_tensor = tf.placeholder(
        dtype = tf.string,
        shape = [None],
        name = 'record_paths'
    )
    dataset = tf.data.TFRecordDataset(record_paths_tensor)
    dataset = dataset.map(parse_sample)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(100000)

    batch_size = tf.placeholder(
        dtype = tf.int64,
        shape = [],
        name = 'batch_size'
    )
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    iterator_initializer = iterator.make_initializer(
        dataset,
        name = 'iterator_initializer'
    )
    next_batch = iterator.get_next(name = 'next_batch')

    placeholders = dict()
    for feature_name in params['create_placeholders_for']:
        placeholders[feature_name] = tf.placeholder_with_default(
            input = next_batch[feature_name],
            shape = next_batch[feature_name].shape,
            name = feature_name
        )

    if 'return' in params:
        return placeholders[params['return']]
    return next_batch

