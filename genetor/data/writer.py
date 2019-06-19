import json
import os
import tensorflow as tf


class Writer(object):


    def __init__(self, dir, format):
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.format = format
        self.write_meta()

        self.shard_n = -1
        self.create_new_shard()


    def write_meta(self):
        json.dump(self.format, open(os.path.join(self.dir, 'meta.json'), 'w'))


    def create_new_shard(self):
        if self.shard_n != -1:
            self.writer.close()
        self.shard_n = self.shard_n + 1
        self.shard_path = os.path.join(self.dir, f'{self.shard_n}.tfrecords')
        self.writer = tf.python_io.TFRecordWriter(self.shard_path)


    def close_writer(self):
        self.writer.close()


    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))


    @staticmethod
    def _int_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


    @staticmethod
    def read_im(path):
        return open(path, 'rb').read()


    def apply_processor(self, feature_name, feature):
        data_type = self.format[feature_name]
        return getattr(Writer, f'_{data_type}_feature')(feature)


    def write_sample(self, sample):
        feature = {
            feature_name: self.apply_processor(feature_name, feature)
            for feature_name, feature in sample.items()
        }

        sample_proto = tf.train.Example(
          features = tf.train.Features(feature = feature)
        )
        sample_serialized = sample_proto.SerializeToString()

        self.writer.write(sample_serialized)


