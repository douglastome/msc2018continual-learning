# Copyright 2018 Douglas Feitosa Tome All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import cv2

import numpy as np
import tensorflow as tf

import vggish_slim
import vggish_params

from network import Network
from datasets import *
from utils import *


class Task(object):
    def __init__(self, dataset, fisher_batches, lam, number, model_dir):
        self.fisher_batches = fisher_batches
        self.lam = lam
        self.number = number
        self.name = dataset[0]
        self.create_dataset(dataset[1])
        self.create_file_writers(model_dir)

        self.predictions = None
        self.labels = None
        self.rev_labels = None
        self.auc = None
        self.auc_sum = None
        self.accumulate_fisher_op = None
        self.average_fisher_op = None
        self.loss = None
        self.loss_sum = None
        self.train_op = None

    def create_dataset(self, parameters):
        if self.name in ['MNIST',
                         'SplitMNIST',
                         'Music',
                         'HumanSounds',
                         'Animal',
                         'SourceAmbiguousSounds',
                         'SoundsOfThings',
                         'NaturalSounds',
                         'ChannelEnvironmentBackground',
                         'Miscellaneous',
                         'MusicHumanSounds',
                         'SVHN',
                         'ExtraSVHN',
                         'ESC50',
                         'URBANSED',
                         'CIFAR100',
                         'CIFAR10']:
            self.dataset = load_pickled_object(parameters['dataset_path'])
        if 'permutation_path' in parameters:
            permutation = load_pickled_object(parameters['permutation_path'])
            self.dataset.permutate(permutation)
            self.name = 'Permuted ' + self.name
        if 'split_classes' in parameters:
            self.dataset.split(parameters['split_classes'])
            self.name = 'Split ' + self.name
        print('Task %d: %s' % (self.number, self.name))
        self.dataset.print_summary()

    def create_file_writers(self, model_dir):
        self.train_writer = tf.summary.FileWriter(model_dir + '/' + str(self.number) + '/train')
        self.validation_writer = tf.summary.FileWriter(model_dir + '/' + str(self.number) + '/validation')
        self.test_writer = tf.summary.FileWriter(model_dir + '/' + str(self.number) + '/test')

    def get_num_classes(self):
        return self.dataset.get_num_classes()

    def get_set_samples(self, set):
        # Number of samples of set
        return getattr(self.dataset, set).get_num_samples()

    def get_set_batches_per_epoch(self, set):
        # Number of batches of set per epoch
        return int(self.get_set_samples(set) / vggish_params.BATCH_SIZE)


class Classifier(Network):

    def __init__(self, sess, model_dir, model_name, vggish_checkpoint, num_train_batches, datasets, fisher_batches, lambdas, checkpoint_log_freq, validation_log_freq, test_log_freq, has_validation_log, has_test_log, *args, **kwargs):
        self.tasks = []
        for (d, f, l, n) in zip(datasets, fisher_batches, lambdas, range(len(datasets))):
            self.tasks.append(Task(d, f, l, n + 1, model_dir))
        super(Classifier, self).__init__(*args, **kwargs)
        self.sess = sess
        self.model_dir = model_dir
        self.model_name = model_name
        self.vggish_checkpoint = vggish_checkpoint
        self.num_train_batches = num_train_batches   #total number of train batches per task
        self.checkpoint_log_freq = checkpoint_log_freq
        self.validation_log_freq = validation_log_freq
        self.test_log_freq = test_log_freq
        self.has_validation_log = has_validation_log
        self.has_test_log = has_test_log

        self.create_merged_summaries()
        self.create_saver()
        self.locate_features_tensor()
        self.create_file_writers()

        # Initialize all variables in the model and save graph.
        self.initialize_global_variables()
        self.initialize_local_variables()
        self.save_graph()

    def get_checkpoint_path(self):
        return self.model_dir + '/' + self.model_name + '.ckpt'

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def copy_variables(self, vars):
        return self.sess.run([v for v in vars])

    def create_merged_summaries(self):
        if self.has_ewc and self.trainable:
            self.ewc_merged = tf.summary.merge([self.mean_lambda_fisher_sum,  self.mean_lambda_fisher_theta_sum, self.mean_lambda_fisher_theta2_sum, self.nonzero_lambda_fisher_sum, self.nonzero_lambda_fisher_theta_sum, self.nonzero_lambda_fisher_theta2_sum])

    def create_saver(self):
        self.saver = tf.train.Saver()

    def create_file_writers(self):
        self.graph_writer = tf.summary.FileWriter(self.model_dir + '/graph')
        self.fisher_writer = tf.summary.FileWriter(self.model_dir + '/fisher')

    def locate_features_tensor(self):
        self.features = self.sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

    def initialize_global_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def initialize_local_variables(self):
        self.sess.run(tf.local_variables_initializer())

    def initialize_classifier(self):
        if not os.path.exists(os.path.join(os.getcwd(), self.model_dir)):
            os.mkdir(os.path.join(os.getcwd(), self.model_dir))
        # Load pre-trained VGGish
        vggish_slim.load_vggish_slim_checkpoint(self.sess, self.vggish_checkpoint)
        # Save model checkpoint
        self.save_variables()

    def has_checkpoint(self):
        return os.path.exists(os.path.join(os.getcwd(), self.model_dir, 'checkpoint'))

    def restore_variables(self):
        self.saver.restore(self.sess, self.get_checkpoint_path())
        print('Restored checkpoint')

    def save_variables(self):
        self.saver.save(self.sess, self.get_checkpoint_path())
        print('Saved chekpoint')

    def save_graph(self):
        global_steps = self.sess.run(self.global_step)
        self.graph_writer.add_graph(self.sess.graph, global_steps)
        print('Global Step %d - Saved Graph' % (global_steps))

    def save_mean_fisher(self, task):
        summary, mean_fisher, global_steps = self.sess.run([self.mean_fisher_sum, self.mean_fisher, self.global_step])
        self.fisher_writer.add_summary(summary, global_steps)
        print('Global Step %d - Task %d: Mean Fisher %g' % (global_steps, task.number, mean_fisher))

    def save_min_fisher(self, task):
        summary, min_fisher, global_steps = self.sess.run([self.min_fisher_sum, self.min_fisher, self.global_step])
        self.fisher_writer.add_summary(summary, global_steps)
        print('Global Step %d - Task %d: Min Fisher %g' % (global_steps, task.number, min_fisher))

    def save_nonzero_fisher(self, task):
        summary, nonzero_fisher, global_steps = self.sess.run([self.nonzero_fisher_sum, self.nonzero_fisher, self.global_step])
        self.fisher_writer.add_summary(summary, global_steps)
        print('Global Step', global_steps, '- Task', task.number, 'Nonzero Fisher', nonzero_fisher, 'Total Fisher', self.abs_fisher.shape.as_list())

    def save_stats_fisher(self, task):
        self.save_mean_fisher(task)
        self.save_min_fisher(task)
        self.save_nonzero_fisher(task)

    def save_stats_ewc_terms(self, task):
        summary, mean_lambda_fisher, mean_lambda_fisher_theta, mean_lambda_fisher_theta2, nonzero_lambda_fisher, nonzero_lambda_fisher_theta, nonzero_lambda_fisher_theta2, global_steps = self.sess.run([self.ewc_merged, self.mean_lambda_fisher, self.mean_lambda_fisher_theta, self.mean_lambda_fisher_theta2, self.nonzero_lambda_fisher, self.nonzero_lambda_fisher_theta, self.nonzero_lambda_fisher_theta2, self.global_step])
        self.fisher_writer.add_summary(summary, global_steps)
        print('Global Step %d - Task %d: Mean Lambda*Fisher %g' % (global_steps, task.number, mean_lambda_fisher))
        print('Global Step %d - Task %d: Mean Lambda*Fisher*Theta %g' % (global_steps, task.number, mean_lambda_fisher_theta))
        print('Global Step %d - Task %d: Mean Lambda*Fisher*Theta2 %g' % (global_steps, task.number, mean_lambda_fisher_theta2))
        print('Global Step %d - Task %d: Nonzero Lambda*Fisher %g' % (global_steps, task.number, nonzero_lambda_fisher))
        print('Global Step %d - Task %d: Nonzero Lambda*Fisher*Theta %g' % (global_steps, task.number, nonzero_lambda_fisher_theta))
        print('Global Step %d - Task %d: Nonzero Lambda*Fisher*Theta2 %g' % (global_steps, task.number, nonzero_lambda_fisher_theta2))

    def reset_fisher(self):
        self.sess.run(self.fisher_reset_op)

    def accumulate_fisher(self, task):
        (feature, label, rev_label) = task.dataset.validation.next_fisher_batch()
        self.sess.run(task.accumulate_fisher_op, feed_dict={self.features: feature, task.labels: label, task.rev_labels: rev_label})

    def average_fisher(self, task):
        self.sess.run(task.average_fisher_op)

    def update_fisher(self, task):
        print('Computation of Fisher Diagonal for Task %d' % task.number)
        self.reset_fisher()
        task.dataset.validation.fisher_randomize()
        print('Start of Fisher Accumulation')
        for batch in range(task.fisher_batches):
            self.accumulate_fisher(task)
        print('End of Fisher Accumulation')
        self.average_fisher(task)
        print('End of Fisher Computation')

    def update_ewc_terms(self, task):
        self.sess.run([self.accumulate_lambda_fisher_op, self.accumulate_lambda_fisher_theta_op, self.accumulate_lambda_fisher_theta2_op], feed_dict={self.lam: task.lam})
        print('End of EWC Regularization Update')

    def update_loss(self, task):
        self.update_fisher(task)
        self.update_ewc_terms(task)
        print('End of Loss Update')

    def train_on_batch(self, task, task_batch):
        #Train the model on a single batch of the training set of task
        (features, labels) = task.dataset.train.next_batch()
        [loss_sum, global_steps, loss, _] = self.sess.run([task.loss_sum, self.global_step, task.loss, task.train_op], feed_dict={self.features: features, task.labels: labels})
        task.train_writer.add_summary(loss_sum, global_steps)
        task_batch += 1
        print('Global Step %d - Training: Task %d Batch %d => Loss %g' % (global_steps, task.number, task_batch, loss))
        return task_batch

    @staticmethod
    def reshape_features(features):
        if len(features) > vggish_params.MAX_IMAGES_PER_SUMMARY:
            features = features[:vggish_params.MAX_IMAGES_PER_SUMMARY]
        for i in range(len(features)):
            features[i] = np.reshape(features[i], (vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1))
        return features

    def compute_auc(self, features, labels, task, set, show_image):
        if show_image:
            auc_sum, auc, global_steps, preds = self.sess.run([task.auc_sum, task.auc, self.global_step, task.predictions], feed_dict={self.features: features, task.labels: labels})
            features = self.reshape_features(features)
            image_sum = self.sess.run(self.labeled_image_sum, feed_dict={self.labeled_images: features})
            getattr(task, set + '_writer').add_summary(auc_sum, global_steps)
            getattr(task, set + '_writer').add_summary(image_sum, global_steps)
            return auc
        return self.sess.run(task.auc, feed_dict={self.features: features, task.labels: labels})

    def evaluate_on_batch(self, task, set, batch, show_image=False):
        # Evaluate classifier on a batch of set of task
        (features, labels) = getattr(task.dataset, set).next_batch()
        auc = self.compute_auc(features, labels, task, set, show_image)
        batch += 1
        return (batch, auc)

    def evaluate_on_set(self, task, set):
        print('Evaluate Classifier on %s Set of Task %d' % (set.title(), task.number))
        self.initialize_local_variables()
        getattr(task.dataset, set).randomize()
        batch = 0
        while batch < task.get_set_batches_per_epoch(set) - 1:
            (batch, auc) = self.evaluate_on_batch(task, set, batch)
        (batch, auc) = self.evaluate_on_batch(task, set, batch, True)

    def evaluate_on_sets(self, task, set):
        print('Evaluate Classifier on %s Sets of Tasks 1 to %d' % (set.title(), task.number))
        for t in self.tasks[:task.number]:
            self.evaluate_on_set(t, set)

    def train(self):
        # Training loop
        global_step = self.get_global_step()
        print('Start Training Session from Global Step: %d' % global_step)
        current_task = int(global_step / self.num_train_batches)
        task_batch = global_step % self.num_train_batches
        for task in self.tasks[current_task:]:
            print('Task %d Batch %d' % (task.number, task_batch))
            if self.has_validation_log:
                self.evaluate_on_set(task, 'validation')
            if self.has_test_log:
                self.evaluate_on_set(task, 'test')
            if task.number == 1:
                self.save_variables()
                if self.has_ewc:
                    self.save_stats_fisher(task)
                    self.save_stats_ewc_terms(task)
            while task_batch < self.num_train_batches:
                task_batch = self.train_on_batch(task, task_batch)
                if self.has_validation_log and task_batch % self.validation_log_freq == 0:
                    self.evaluate_on_sets(task, 'validation')
                if self.has_test_log and task_batch % self.test_log_freq == 0:
                    self.evaluate_on_sets(task, 'test')
                if task_batch % self.checkpoint_log_freq == 0:
                    self.save_variables()
            if self.has_validation_log:
                self.evaluate_on_sets(task, 'validation')
            if self.has_test_log:
                self.evaluate_on_sets(task, 'test')
            if self.has_ewc:
                self.update_loss(task)
            self.save_variables()
            if self.has_ewc:
                self.save_stats_fisher(task)
                self.save_stats_ewc_terms(task)
            task_batch = 0

    def evaluate(self):
        # Evaluation loop
        global_step = self.get_global_step()
        print('Evaluation Session at Step: %d' % global_step)
        current_task = int(global_step / self.num_train_batches)
        for task in range(current_task + 1 if current_task < self.num_tasks else current_task):
            self.initialize_local_variables()
            test_batch = 0
            while test_batch < task.get_test_batches_per_epoch():
                (test_batch, auc) = self.evaluate_on_batch(task, test_batch, 'test')
            print('Task %d - Total Batches: %d - AUC: %g' % (task + 1, test_batch, auc))
