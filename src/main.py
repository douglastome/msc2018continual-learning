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

from __future__ import print_function

import numpy as np
import tensorflow as tf

import vggish_params

from classifier import Classifier
from datasets import *


flags = tf.app.flags

flags.DEFINE_string(
                    'model_dir',
                    'model',
                    'Path to the model directory where checkpoints and event files '
                    'will be written.'
                    )

flags.DEFINE_string(
                    'model_name',
                    'model',
                    'Name of the model. '
                    'Checkpoint files will have this name prepended.'
                    )

flags.DEFINE_boolean(
                     'train',
                     True,
                     'If True, train the model. '
                     'If False, evaluate the model.'
                     )

flags.DEFINE_boolean(
                     'has_ewc',
                     True,
                     'If True, train the model with EWC. '
                     'If False, train the model without EWC.'
                     )

flags.DEFINE_integer(
                     'num_train_batches',
                     10000,
                     'Number of batches to train the model per task.'
                     )

flags.DEFINE_integer(
                     'num_tasks',
                     2,
                     'Number of tasks.'
                     )

flags.DEFINE_list(
                  'datasets',
                  [('CIFAR10', {'dataset_path': '/my/data/path/CIFAR10/cifar10.pkl'}), ('CIFAR100', {'dataset_path': '/my/data/path/CIFAR100/cifar100.pkl'}) ],
                  'List of datasets used to train the model from first to last.'
                  'Each item in the list is tuple (Dataset, Parameters).'
                  'Length of the list must match num_tasks.'
                  )

flags.DEFINE_list(
                  'fisher_batches',
                  [2, 2],
                  'List of number of batches of validation examples used to compute '
                  'the Fisher information matrix diagonal for tasks. Length of the '
                  'list must match num_tasks.'
                  )

flags.DEFINE_list(
                  'lambdas',
                  [10, 10],
                  'List of lambdas for tasks. Length of the list must match '
                  'num_tasks.'
                  )

flags.DEFINE_string(
                    'vggish_checkpoint',
                    '/my/data/path/vggish_model.ckpt',
                    'Path to the pre-trained VGGish checkpoint file.'
                    )

flags.DEFINE_integer(
                     'checkpoint_log_freq',
                     100000,
                     'Number of batches between consecutive checkpoint file save '
                     'operations during trainig.'
                     )

flags.DEFINE_integer(
                     'validation_log_freq',
                     100000,
                     'Number of batches between consecutive validation set '
                     'evaluations during trainig.'
                     )

flags.DEFINE_integer(
                     'test_log_freq',
                     100000,
                     'Number of batches between consecutive test set evaluations '
                     'during trainig.'
                     )

flags.DEFINE_boolean(
                     'has_validation_log',
                     True,
                     'If True, evaluate the model on the validation sets at the '
                     'start and end of training as well as every validation_log_freq '
                     'during training.'
                     )

flags.DEFINE_boolean(
                     'has_test_log',
                     False,
                     'If True, evaluate the model on the test sets at the start '
                     'and end of training as well as every test_log_freq during '
                     'training.'
                     )

FLAGS = flags.FLAGS


def main(_):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    classifier = Classifier(
                            sess=sess,
                            model_dir=FLAGS.model_dir,
                            model_name=FLAGS.model_name,
                            vggish_checkpoint=FLAGS.vggish_checkpoint,
                            num_train_batches=FLAGS.num_train_batches,
                            datasets = FLAGS.datasets,
                            fisher_batches=FLAGS.fisher_batches,
                            lambdas=FLAGS.lambdas,
                            checkpoint_log_freq=FLAGS.checkpoint_log_freq,
                            validation_log_freq=FLAGS.validation_log_freq,
                            test_log_freq=FLAGS.test_log_freq,
                            has_validation_log=FLAGS.has_validation_log,
                            has_test_log=FLAGS.has_test_log,
                            has_ewc=FLAGS.has_ewc,
                            num_tasks=FLAGS.num_tasks,
                            trainable=FLAGS.train,
                            )

    # Restore variables and permutations if checkpoint exists.
    # If not, initialize model with pre-trained VGGish checkpoint.
    if classifier.has_checkpoint():
        classifier.restore_variables()
    else:
        classifier.initialize_classifier()

    if FLAGS.train:
        classifier.train()
    else:
        classifier.evaluate()


if __name__ == '__main__':
  tf.app.run()
