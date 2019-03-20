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

import numpy as np
import tensorflow as tf

import vggish_params
import vggish_slim

slim = tf.contrib.slim


class Network(object):
    """
    Defines a neural network consisting of VGGish followed by one classification
    layer per task.
    """
    def __init__(self, has_ewc, num_tasks, trainable):
        self.has_ewc = has_ewc
        self.num_tasks = num_tasks
        self.trainable = trainable

        self.build_vggish_graph()
        self.build_global_tensors()
        
        if self.has_ewc and self.trainable:
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.build_fisher_graph()
            self.build_ewc_reg()
            print('trainable_vars before building tasks', len(self.trainable_vars))
            for v in self.trainable_vars:
                print(v.name)
        
        for t in self.tasks:
            self.build_task_graph(t)
            if self.trainable:
                self.build_task_optimizer(t)

        if self.has_ewc and self.trainable:
            test_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print('trainable_vars after building tasks', len(test_trainable_vars))
            for v in test_trainable_vars:
                print(v.name)

    def build_vggish_graph(self):
        # Define VGGish.
        self.embeddings = vggish_slim.define_vggish_slim(training=self.trainable)

    def build_global_tensors(self):
        with tf.variable_scope('global_tensors'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
            # Add image summary
            self.labeled_images = tf.placeholder(tf.float32, shape=(None, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1), name='labeled_images')
            self.labeled_image_sum = tf.summary.image('labeled_image_sum', self.labeled_images, vggish_params.MAX_IMAGES_PER_SUMMARY)
    
    def build_task_graph(self, task):
        # Define a shallow classification model and associated training ops on top
        # of VGGish.
        with tf.variable_scope('task' + str(task.number)):
            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            with tf.variable_scope('classifier'):
                logits = slim.fully_connected(self.embeddings, task.get_num_classes(), activation_fn=None, trainable=self.trainable, scope='logits')
                predictions = tf.sigmoid(logits, name='predictions')
                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                labels = tf.placeholder(tf.float32, shape=(None, task.get_num_classes()), name='labels')
                task.predictions = predictions
                task.labels = labels
            
            # Add performance metric ops.
            with tf.variable_scope('performance'):
                auc_update, auc_value = tf.metrics.auc(labels, predictions, name='performance_metric')
                auc = tf.identity(auc_value, name='auc_value')
                auc_sum = tf.summary.scalar('auc', auc)
                task.auc = auc
                task.auc_sum = auc_sum
        
            if self.has_ewc and self.trainable:
                # Add log-likelihood ops.
                with tf.variable_scope('likelihood'):
                    # Define placeholder and operations to compute Fisher diagonal
                    rev_labels = tf.placeholder(tf.float32, shape=(None, task.get_num_classes()), name='reversed_labels')
                    rev_predictions = 1 - predictions
                    pos_class_probs = tf.multiply(labels, predictions, name='pos_class_probs')
                    neg_class_probs = tf.multiply(rev_labels, rev_predictions, name='neg_class_probs')
                    probs = tf.add(pos_class_probs, neg_class_probs, name='probs')
                    log_probs = tf.log(probs, name='log_probs')
                    log_likelihood = tf.reduce_sum(log_probs, axis=1, name='log_likelihood')
                    task.rev_labels = rev_labels
                    
                    # Define operations to compute Fisher of task
                    # Operations in the next block are only valid if sample size is fixed
                    unstacked_log_likelihood = tf.unstack(log_likelihood ,name='unstacked_log_likelihood')
                    gradients = []
                    for sll in unstacked_log_likelihood:
                        gradients.append(tf.gradients(sll, self.trainable_vars))
                    squared_gradients_sum = []
                    for i in range(len(self.trainable_vars)):
                        grad2_sum = tf.add_n([tf.square(grad[i]) for grad in gradients])
                        squared_gradients_sum.append(grad2_sum)
                    accumulate_fisher_op = [tf.assign_add(f, g) for f, g in zip(self.fisher_diagonal, squared_gradients_sum)]
                    average_fisher_op = [tf.assign(f, f / float(vggish_params.BATCH_SIZE * task.fisher_batches)) for f in self.fisher_diagonal]
                    task.accumulate_fisher_op = accumulate_fisher_op
                    task.average_fisher_op = average_fisher_op
            
            if self.trainable:
                # Add loss ops.
                with tf.variable_scope('loss'):
                    # Cross-entropy label loss.
                    xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
                    loss = tf.reduce_mean(xent, name='loss_op')
                    task.loss = loss

    def build_fisher_graph(self):
        with tf.variable_scope('fisher'):
            # Create Fisher diagonal and running sums of the regularization term
            self.fisher_diagonal = []
            self.lambda_fisher = []
            self.lambda_fisher_theta = []
            self.lambda_fisher_theta2 = []
            for i in range(len(self.trainable_vars)):
                v = self.trainable_vars[i]
                self.fisher_diagonal.append(tf.Variable(tf.constant(0.0, shape=v.get_shape().as_list()), trainable=False, name='fisher_diagonal_' + str(i)))
                self.lambda_fisher.append(tf.Variable(tf.constant(0.0, shape=v.get_shape().as_list()), trainable=False, name='lambda_fisher_sum' + str(i)))
                self.lambda_fisher_theta.append(tf.Variable(tf.constant(0.0, shape=v.get_shape().as_list()), trainable=False, name='lambda_fisher_theta_sum' + str(i)))
                self.lambda_fisher_theta2.append(tf.Variable(tf.constant(0.0, shape=v.get_shape().as_list()), trainable=False, name='lambda_fisher_theta2_sum' + str(i)))

            # Define operations to monitor the Fisher diagonal
            self.fisher_diag_flat = tf.concat([tf.reshape(f, [-1]) for f in self.fisher_diagonal], 0)
            self.mean_fisher = tf.reduce_mean(self.fisher_diag_flat, name='mean_fisher_op')
            self.mean_fisher_sum = tf.summary.scalar('mean_fisher', self.mean_fisher)
            self.abs_fisher = tf.abs(self.fisher_diag_flat, name='abs_fisher_op')
            self.nonzero_fisher = tf.count_nonzero(self.abs_fisher, name='nonzero_fisher_op')
            self.nonzero_fisher_sum = tf.summary.scalar('nonzero_fisher', self.nonzero_fisher)
            self.min_fisher = tf.reduce_min(self.abs_fisher, name='min_fisher_op')
            self.min_fisher_sum = tf.summary.scalar('min_fisher', self.min_fisher)

            # Define operations to monitor the coefficients of the regularization term
            self.lambda_fisher_flat = tf.concat([tf.reshape(f, [-1]) for f in self.lambda_fisher], 0)
            self.mean_lambda_fisher = tf.reduce_mean(self.lambda_fisher_flat, name='mean_lambda_fisher_op')
            self.mean_lambda_fisher_sum = tf.summary.scalar('mean_lambda_fisher', self.mean_lambda_fisher)
            self.nonzero_lambda_fisher = tf.count_nonzero(self.lambda_fisher_flat, name='nonzero_lambda_fisher_op')
            self.nonzero_lambda_fisher_sum = tf.summary.scalar('nonzero_lambda_fisher', self.nonzero_lambda_fisher)

            self.lambda_fisher_theta_flat = tf.concat([tf.reshape(f, [-1]) for f in self.lambda_fisher_theta], 0)
            self.mean_lambda_fisher_theta = tf.reduce_mean(self.lambda_fisher_theta_flat, name='mean_lambda_fisher_theta_op')
            self.mean_lambda_fisher_theta_sum = tf.summary.scalar('mean_lambda_fisher_theta', self.mean_lambda_fisher_theta)
            self.nonzero_lambda_fisher_theta = tf.count_nonzero(self.lambda_fisher_theta_flat, name='nonzero_lambda_fisher_theta_op')
            self.nonzero_lambda_fisher_theta_sum = tf.summary.scalar('nonzero_lambda_fisher_theta', self.nonzero_lambda_fisher_theta)

            self.lambda_fisher_theta2_flat = tf.concat([tf.reshape(f, [-1]) for f in self.lambda_fisher_theta2], 0)
            self.mean_lambda_fisher_theta2 = tf.reduce_mean(self.lambda_fisher_theta2_flat, name='mean_lambda_fisher_theta2_op')
            self.mean_lambda_fisher_theta2_sum = tf.summary.scalar('mean_lambda_fisher_theta2', self.mean_lambda_fisher_theta2)
            self.nonzero_lambda_fisher_theta2 = tf.count_nonzero(self.lambda_fisher_theta2_flat, name='nonzero_lambda_fisher_theta2_op')
            self.nonzero_lambda_fisher_theta2_sum = tf.summary.scalar('nonzero_lambda_fisher_theta2', self.nonzero_lambda_fisher_theta2)
            
            # Define operation to reset the Fisher diagonal
            self.fisher_reset_op = [tf.assign(f, tf.zeros_like(f)) for f in self.fisher_diagonal]

            # Operations to update EWC terms
            self.lam = tf.placeholder(tf.float32, shape=(), name='lambda')
            self.accumulate_lambda_fisher_op = [tf.assign_add(lf, (self.lam / 2.0) * f) for lf, f in zip(self.lambda_fisher, self.fisher_diagonal)]
            self.accumulate_lambda_fisher_theta_op = [tf.assign_add(lft, self.lam * tf.multiply(f, t)) for lft, f, t in zip(self.lambda_fisher_theta, self.fisher_diagonal, self.trainable_vars)]
            self.accumulate_lambda_fisher_theta2_op = [tf.assign_add(lft2, (self.lam / 2.0) * tf.multiply(f, tf.square(t))) for lft2, f, t in zip(self.lambda_fisher_theta2, self.fisher_diagonal, self.trainable_vars)]

    def build_ewc_reg(self):
        with tf.variable_scope('ewc'):
            self.ewc_reg = tf.add_n([tf.reduce_sum(tf.multiply(lf, tf.square(t)) - tf.multiply(lft, t) + lft2) for (t, lf, lft, lft2) in list(zip(self.trainable_vars, self.lambda_fisher, self.lambda_fisher_theta, self.lambda_fisher_theta2))])
    
    def build_task_optimizer(self, task):
        with tf.variable_scope('optimizer' + str(task.number)):
            # Same optimizer and hyperparameters as used to train VGGish.
            if self.has_ewc and task.number > 1:
                task.loss += self.ewc_reg
            loss_sum = tf.summary.scalar('loss', task.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=vggish_params.LEARNING_RATE, epsilon=vggish_params.ADAM_EPSILON)
            train_op = optimizer.minimize(task.loss, global_step=self.global_step, name='train_op')
            task.loss_sum = loss_sum
            task.train_op = train_op
