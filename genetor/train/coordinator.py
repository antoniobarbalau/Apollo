import math
import os
import shutil
import tensorflow as tf

class Coordinator(object):

    def __init__(self,
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
                 next_batch = 'next_batch'):

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

        self.load_session()
        self.convert_placeholders_generators()
        self.load_tensors()
        self.load_operations()
        if self.summary:
            self.create_summary()

        self.session.run(tf.global_variables_initializer())

        self.epoch_n = -1


    def train_epoch(self):
        self.epoch_n += 1

        if self.record_paths:
            self.initialize_iterators()
        n_iterations = math.ceil(self.n_samples / self.batch_size)
        load_data = [self.operations[self.next_batch]] if self.record_paths else []
        summary = [self.summary_merged] if self.summary else []
        for iteration_n in range(n_iterations):
            feed_dict = {
                name: generator(iteration_n, self.batch_size)
                for name, generator in self.placeholders.items()
            }
            results = self.session.run(
                load_data +
                [
                    self.operations[optimizer_name]
                    for optimizer_name in self.optimizers
                ] +
                summary,
                feed_dict = feed_dict
            )
            if self.summary:
                self.summary_writer.add_summary(
                    results[-1],
                    (self.epoch_n * n_iterations) + iteration_n
                )


    def eval(self, tensor_names):
        if self.record_paths:
            self.initialize_iterators()
        load_data = [self.operations[self.next_batch]] if self.record_paths else []

        feed_dict = {
            name: generator(0, 0)
            for name, generator in self.placeholders.items()
            if name in self.valid_placeholders
        }
        results = self.session.run(
            to_eval + load_data +
            [
                self.operations[optimizer_name]
                for optimizer_name in self.optimizers
            ],
            feed_dict = feed_dict
        )

        return results[:len(tensor_names)]


    def convert_placeholders_generators(self):
        for elem in self.placeholders:
            if type(self.placeholders[elem]) != type(lambda: 0):
                self.placeholders[elem] = lambda a, b: self.placeholders[elem]


    def save(self):
        self.saver.save(self.session, self.ckpt_meta_path.replace('.meta', ''))


    def create_summary(self):
        for tensor_name in self.summary.get('scalars', []):
            tf.summary.scalar('haha', self.tensors[tensor_name])

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
        else:
            self.n_samples = self.placeholders['n_samples:0']
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
        
    
