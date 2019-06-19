import math
import numpy as np
import os
import shutil
import tensorflow as tf
import time

class Coordinator(object):


    def __init__(
            self,
            ckpt_meta_path,
            record_paths = None,
            summary = None,
            batch_size = None,
            n_samples = None,
            placeholders = dict(),
            optimizers = ['optimizer'],
            record_paths_placeholder = 'record_paths:0',
            batch_size_tensor = 'batch_size:0',
            iterator_initializer = 'iterator_initializer',
            next_batch = 'next_batch',
            return_values = []
    ):

        self.ckpt_meta_path = ckpt_meta_path
        self.record_paths = record_paths
        self.record_paths_placeholder = record_paths_placeholder
        self.batch_size = batch_size
        self.batch_size_tensor = batch_size_tensor
        self.placeholders = placeholders
        self.summary = summary
        self.optimizers = optimizers
        self.iterator_initializer = iterator_initializer
        self.next_batch = next_batch
        self.return_values = return_values

        self.load_session()
        self.convert_placeholders_generators()
        self.load_tensors()
        self.load_operations()
        self.summary_is_non_empty = self.summary and len(list(self.summary.keys())) > 1
        if self.summary:
            self.create_summary()

        if self.record_paths:
            self.initialize_iterators()
        else:
            self.n_samples = n_samples

        self.epoch_n = -1
        self.n_iterations = math.ceil(self.n_samples / self.batch_size)
        self.iteration_n = self.n_iterations - 1


    def train_iteration(self):
        self.iteration_n += 1
        if self.iteration_n == self.n_iterations:
            self.epoch_n += 1
            self.iteration_n = 0

        load_data = [self.operations[self.next_batch]] if self.record_paths else []
        summary = [self.summary_merged] if self.summary_is_non_empty else []
        return_values = [
            self.tensors[tensor_name]
            for tensor_name in self.return_values
        ]
        feed_dict = {
            name: generator(self.iteration_n, self.batch_size)
            for name, generator in self.placeholders.items()
        }

        results = self.session.run(
            return_values +
            load_data +
            [
                self.operations[optimizer_name]
                for optimizer_name in self.optimizers
            ] +
            summary,
            feed_dict = feed_dict
        )

        if self.summary_is_non_empty:
            self.summary_writer.add_summary(
                results[-1],
                (self.epoch_n * self.n_iterations) + self.iteration_n
            )

        return results[:len(return_values)]


    def train_epoch(self):
        def format_seconds(s):
            s = int(s)
            h = s // 3600
            s = s % 3600
            m = s // 60
            s = s % 60
            return f'{h}:{m}:{s}         '

        self.epoch_n += 1
        self.iteration_n = -1

        return_values = []

        return_values.append(self.train_iteration())
        iteration_d_estimation = 0

        for iteration_n in range(1, self.n_iterations):
            remaining_time = (
                (self.n_iterations - iteration_n) * iteration_d_estimation
            )
            print(
                f'{np.round(iteration_n / self.n_iterations, 2)} -- ' +
                f'{format_seconds(remaining_time)}',
                end = '\r'
            )
            iteration_start_time = time.time()
            return_values.append(self.train_iteration())
            iteration_end_time = time.time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_d_estimation += (iteration_n == 1) * iteration_duration
            iteration_d_estimation = (
                .95 * iteration_d_estimation +
                .05 * iteration_duration
            )

        return np.array(return_values)


    def convert_placeholders_generators(self):
        for elem in self.placeholders:
            if type(self.placeholders[elem]) != type(lambda: 0):
                self.placeholders[elem] = lambda a, b: self.placeholders[elem]


    def save(self):
        self.saver.save(self.session, self.ckpt_meta_path.replace('.meta', ''))


    def create_summary(self):
        for tensor_name in self.summary.get('scalars', []):
            tf.summary.scalar(tensor_name, self.tensors[tensor_name])

        for image in self.summary.get('images', []):
            tf.summary.image(
                image['name'],
                self.tensors[image['tensor']],
                max_outputs = image['max_outputs']
            )

        for tensor_name in self.summary.get('text', []):
            tf.summary.text(tensor_name, self.tensors[tensor_name])

        if os.path.exists(self.summary['path']):
            shutil.rmtree(self.summary['path'])

        if self.summary_is_non_empty:
            self.summary_merged = tf.summary.merge_all()

        self.summary_writer = tf.summary.FileWriter(
            self.summary['path'],
            self.graph
        )


    def load_session(self):
        self.session = tf.Session()
        self.saver = tf.train.import_meta_graph(self.ckpt_meta_path)
        self.saver.restore(self.session, tf.train.latest_checkpoint(
            os.path.dirname(self.ckpt_meta_path)
        ))
        self.graph = tf.get_default_graph()


    def load_tensors(self):
        self.tensors = dict()
        tensor_names = list(self.placeholders.keys())
        if self.record_paths:
            tensor_names += [
                self.record_paths_placeholder,
                self.batch_size_tensor
            ]
        if self.return_values:
            tensor_names += self.return_values
        if self.summary:
            tensor_names += self.summary.get('scalars', [])
            tensor_names += self.summary.get('text', [])
            tensor_names += [
                image['tensor']
                for image in self.summary.get('images', [])
            ]
        for tensor_name in tensor_names:
            self.tensors[tensor_name] = self.graph.get_tensor_by_name(
                tensor_name
            )


    def load_operations(self):
        self.operations = dict()
        operation_names = [
            *self.optimizers
        ]
        if self.record_paths:
            operation_names += [
                self.next_batch,
                self.iterator_initializer
            ]
        for operation_name in operation_names:
            self.operations[operation_name] = self.graph.get_operation_by_name(
                operation_name
            )


    def initialize_iterators(self):
        self.n_samples = sum(
            sum(1 for _ in tf.python_io.tf_record_iterator(filename))
            for filename in self.record_paths
        )
        self.session.run(
            self.operations[self.iterator_initializer],
            feed_dict = {
                self.tensors[self.record_paths_placeholder]: self.record_paths,
                self.tensors[self.batch_size_tensor]: self.batch_size
            }
        )
 

