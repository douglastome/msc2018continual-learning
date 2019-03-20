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
import sys
import csv
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf

import vggish_params
import vggish_input
from utils import *


def generate_permutation(perm_path, dataset_path, ver_seed=None, hor_seed=None):
    dataset = load_pickled_object(dataset_path)
    print('Loaded dataset')
    permutation = dataset.generate_permutation(ver_seed=ver_seed, hor_seed=hor_seed)
    print('Generated permutation')
    save_pickled_object(permutation, perm_path)
    print('Saved permutation')

def generate_permutations(perms_path, dataset_path, perm_name, num_permutations, seeds=None):
    for i in range(num_permutations):
        print('Permutation', i+1)
        perm_path = os.path.join(perms_path, perm_name + str(i+1) + '.pkl')
        if seeds is None:
            ver_seed = None
            hor_seed = None
        else:
            ver_seed = seeds[i][0]
            hor_seed = seeds[i][1]
        generate_permutation(perm_path, dataset_path, ver_seed, hor_seed)

def build_svhn(data_path, batch_size, num_features=None):
    svhn = SVHN(data_path=data_path, batch_size=batch_size)
    print('SVHN')
    svhn.print_summary()
    save_pickled_object(svhn, os.path.join(data_path, 'svhn.pkl'))
    print('Saved SVHN')
    svhn2 = load_pickled_object(os.path.join(data_path, 'svhn.pkl'))
    print('Loaded SVHN')
    print('SVHN')
    svhn2.print_summary()
    if num_features is not None:
        svhn2.plot_features(num_features)

def build_extra_svhn(data_path, batch_size, num_features=None):
    extra_svhn = ExtraSVHN(data_path=data_path, batch_size=batch_size)
    print('ExtraSVHN')
    extra_svhn.print_summary()
    save_pickled_object(extra_svhn, os.path.join(data_path, 'extra_svhn.pkl'))
    print('Saved ExtraSVHN')
    extra_svhn2 = load_pickled_object(os.path.join(data_path, 'extra_svhn.pkl'))
    print('Loaded ExtraSVHN')
    print('ExtraSVHN')
    extra_svhn2.print_summary()
    if num_features is not None:
        extra_svhn2.plot_features(num_features)

def build_cifar100(data_path, batch_size, num_features=None):
    cifar100 = CIFAR100(batch_size=batch_size)
    print('CIFAR100')
    cifar100.print_summary()
    save_pickled_object(cifar100, os.path.join(data_path, 'cifar100.pkl'))
    print('Saved CIFAR100')
    cifar100_2 = load_pickled_object(os.path.join(data_path, 'cifar100.pkl'))
    print('Loaded CIFAR100')
    print('CIFAR100')
    cifar100_2.print_summary()
    if num_features is not None:
        cifar100_2.plot_features(num_features)

def build_cifar10(data_path, batch_size, num_features=None):
    cifar10 = CIFAR10(batch_size=batch_size)
    print('CIFAR10')
    cifar10.print_summary()
    save_pickled_object(cifar10, os.path.join(data_path, 'cifar10.pkl'))
    print('Saved CIFAR10')
    cifar10_2 = load_pickled_object(os.path.join(data_path, 'cifar10.pkl'))
    print('Loaded CIFAR10')
    print('CIFAR10')
    cifar10_2.print_summary()
    if num_features is not None:
        cifar10_2.plot_features(num_features)

def build_mnist(data_path, batch_size, num_features=None):
    mnist = MNIST(batch_size=batch_size)
    print('MNIST')
    mnist.print_summary()
    save_pickled_object(mnist, os.path.join(data_path, 'mnist.pkl'))
    print('Saved MNIST')
    mnist2 = load_pickled_object(os.path.join(data_path, 'mnist.pkl'))
    print('Loaded MNIST')
    print('MNIST')
    mnist2.print_summary()
    if num_features is not None:
        mnist2.plot_features(num_features)

def build_esc50(data_path, train_folds, validation_fold, test_fold, batch_size, num_features=None):
    esc50 = ESC50(data_path=data_path, train_folds=train_folds, validation_fold=validation_fold, test_fold=test_fold, batch_size=batch_size)
    print('ESC-50')
    esc50.print_summary()
    save_pickled_object(esc50, os.path.join(data_path, 'esc50.pkl'))
    print('Saved ESC-50')
    esc50_2 = load_pickled_object(os.path.join(data_path, 'esc50.pkl'))
    print('Loaded ESC-50')
    print('ESC-50')
    esc50_2.print_summary()
    if num_features is not None:
        esc50_2.plot_features(num_features)

def build_urbansed(data_path, batch_size, num_features=None):
    urbansed = URBANSED(data_path=data_path, batch_size=batch_size)
    print('URBAN-SED')
    urbansed.print_summary()
    save_pickled_object(urbansed, os.path.join(data_path, 'urbansed.pkl'))
    print('Saved URBAN-SED')
    urbansed_2 = load_pickled_object(os.path.join(data_path, 'urbansed.pkl'))
    print('Loaded URBAN-SED')
    print('URBAN-SED')
    urbansed_2.print_summary()
    if num_features is not None:
        urbansed_2.plot_features(num_features)

def build_audioset_branches(has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, batch_size):
    """Build one dataset for each branch or combination of branches of AudioSet"""
    #Music: /m/04rlf
    #Human sounds: /m/0dgw9r
    #Animal: /m/0jbk
    #Source-ambiguous sounds: /t/dd00098
    #Sounds of things: /t/dd00041
    #Natural sounds: /m/059j3w
    #Channel, environment and background: /t/dd00123
    music = Music(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('Music')
    music.print_summary()
    save_pickled_object(music, os.path.join(music.data_path, 'music.pkl'))
    
    human_sounds = HumanSounds(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('Human Sounds')
    human_sounds.print_summary()
    save_pickled_object(human_sounds, os.path.join(human_sounds.data_path, 'human_sounds.pkl'))
    
    animal = Animal(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('Animal')
    animal.print_summary()
    save_pickled_object(animal, os.path.join(animal.data_path, 'animal.pkl'))
    
    source_ambiguous = SourceAmbiguousSounds(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('SourceAmbiguousSounds')
    source_ambiguous.print_summary()
    save_pickled_object(source_ambiguous, os.path.join(source_ambiguous.data_path, 'source_ambiguous.pkl'))
    
    sounds_of_things = SoundsOfThings(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('SoundsOfThings')
    sounds_of_things.print_summary()
    save_pickled_object(sounds_of_things, os.path.join(sounds_of_things.data_path, 'sounds_of_things.pkl'))
    
    natural_sounds = NaturalSounds(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('NaturalSounds')
    natural_sounds.print_summary()
    save_pickled_object(natural_sounds, os.path.join(natural_sounds.data_path, 'natural_sounds.pkl'))
    
    background = ChannelEnvironmentBackground(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('ChannelEnvironmentBackground')
    background.print_summary()
    save_pickled_object(background, os.path.join(background.data_path, 'background.pkl'))
    
    miscellaneous = Miscellaneous(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('Miscellaneous')
    miscellaneous.print_summary()
    save_pickled_object(miscellaneous, os.path.join(miscellaneous.data_path, 'miscellaneous.pkl'))
    music_human = MusicHumanSounds(has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, batch_size=batch_size)
    print('MusicHumanSounds')
    music_human.print_summary()
    save_pickled_object(music_human, os.path.join(music_human.data_path, 'music_human.pkl'))


class Data(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.images = None
        self.labels = None
        self.indices = None
        self.index = None

        # Only for validation
        self.rev_labels = None
        self.fisher_indices = None
        self.fisher_index = None

    def get_num_samples(self):
        return self.images.shape[0]
    
    def set_rev_labels(self):
        self.rev_labels = np.ones(self.labels.shape) - self.labels
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.randomize()
        if self.rev_labels is not None:
            self.fisher_randomize()

    def initialize_indices(self):
        self.indices = list(range(self.images.shape[0]))

    def randomize(self):
        shuffle(self.indices)
        self.index = 0

    def next_batch(self, permutation=None):
        if self.index + self.batch_size > self.images.shape[0]:
            self.randomize()
        batch = []
        for i in range(self.batch_size):
            batch.append(self.indices[self.index])
            self.index += 1
        features = [self.images[s] for s in batch]
        labels = [self.labels[s] for s in batch]
        return (features, labels)

    def initialize_fisher_indices(self):
        self.fisher_indices = list(range(self.images.shape[0]))

    def fisher_randomize(self):
        shuffle(self.fisher_indices)
        self.fisher_index = 0

    def next_fisher_batch(self, permutation=None):
        if self.fisher_index + self.batch_size > self.images.shape[0]:
            self.fisher_randomize()
        fisher_batch = []
        for i in range(self.batch_size):
            fisher_batch.append(self.fisher_indices[self.fisher_index])
            self.fisher_index += 1
        features = [self.images[s] for s in fisher_batch]
        labels = [self.labels[s] for s in fisher_batch]
        rev_labels = [self.rev_labels[s] for s in fisher_batch]
        return (features, labels, rev_labels)

    @staticmethod
    def show_img_plt(img, label=None):
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if label is not None:
            plt.title('-'.join([str(l) for l in label]))
        plt.show()

    @staticmethod
    def show_img_cv2(img):
        cv2.imshow('Feature', img)
        cv2.waitKey()

    def plot_features(self, features, lib='plt', labels=None):
        for i in range(len(features)):
            if lib == 'cv2':
                if labels is not None:
                    print('labels', np.where(labels[i] == 1)[0], labels[i])
                features[i] = np.reshape(features[i], (vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1))
                self.show_img_cv2(features[i])
            elif lib == 'plt':
                if labels is None:
                    self.show_img_plt(features[i])
                else:
                    print('labels', np.where(labels[i] == 1)[0], labels[i])
                    self.show_img_plt(features[i], np.where(labels[i] == 1)[0])

    @staticmethod
    def generate_axis_permutation(axis_len, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.permutation(axis_len)
    
    def generate_permutation(self, ver_seed=None, hor_seed=None):
        ver_perm = self.generate_axis_permutation(vggish_params.NUM_BANDS, ver_seed)
        hor_perm = self.generate_axis_permutation(vggish_params.NUM_FRAMES, hor_seed)
        return (ver_perm, hor_perm)

    def permutate(self, permutation):
        if self.images is not None:
            self.images = self.images[:, :, permutation[0]]
            self.images = self.images[:, permutation[1], :]

    @staticmethod
    def matches_classes(label, classes):
        # Check if a given label is consistent with the given subset of classes
        for idx in np.where(label == 1)[1]:
            if idx not in classes:
                return False
        return True
    
    @staticmethod
    def generate_new_label(label, classes):
        new_label = np.zeros((1, len(classes)))
        for idx in np.where(label == 1)[1]:
            new_label[0, classes.index(idx)] = 1
        return new_label

    def split(self, classes):
        images = [img for (img, lbl) in zip(np.split(self.images, self.images.shape[0]), np.split(self.labels, self.labels.shape[0])) if self.matches_classes(lbl, classes)]
        labels = [self.generate_new_label(lbl, classes) for lbl in np.split(self.labels, self.labels.shape[0]) if self.matches_classes(lbl, classes)]
        self.images = np.concatenate(images)
        self.labels = np.concatenate(labels)
        self.initialize_indices()
        self.randomize()
        if self.rev_labels is not None:
            self.set_rev_labels()
            self.initialize_fisher_indices()
            self.fisher_randomize()

    def pickle_data(self, dir):
        os.mkdir(os.path.join(dir, 'images'))
        os.mkdir(os.path.join(dir, 'labels'))
        for i in range(self.images.shape[0]):
            save_pickled_object(self.images[i], os.path.join(dir, 'images', 'image-' + str(i) + '.pkl'))
            save_pickled_object(self.labels[i], os.path.join(dir, 'labels', 'label-' + str(i) + '.pkl'))


class Dataset(object):
    def __init__(self, batch_size):
        self.train = Data(batch_size)
        self.validation = Data(batch_size)
        self.test = Data(batch_size)

    def get_num_classes(self):
        return self.train.labels[0].shape[0]
    
    def initialize_indices(self):
        # Initialize datasets
        self.train.initialize_indices()
        self.train.randomize()
        self.validation.initialize_indices()
        self.validation.randomize()
        self.validation.initialize_fisher_indices()
        self.validation.fisher_randomize()
        self.test.initialize_indices()
        self.test.randomize()
    
    def split_data(self, data, train_split, validation_split, test_split):
        """
        Shuffle data and split it among train, validation, and test sets
        Parameters:
            data: list of (sample, label) where sample is of shape (num_spectrograms, 96, 64) and label of shape (num_spectrograms, num_classes)
            train_split: percentage of train samples
            validation_split: percentage of validation samples
            test_split: percentage of test samples
        """
        shuffle(data)
        print('#data', len(data))
        total_validation_samples = int(len(data) * validation_split/100)
        total_test_samples = int(len(data) * test_split/100)
        self.validation.images = np.concatenate([sample for (sample, _) in data[:total_validation_samples]])
        self.validation.labels = np.concatenate([label for (_, label) in data[:total_validation_samples]])
        self.validation.set_rev_labels()
        self.test.images = np.concatenate([sample for (sample, _) in data[total_validation_samples:total_validation_samples + total_test_samples]])
        self.test.labels = np.concatenate([label for (_, label) in data[total_validation_samples:total_validation_samples + total_test_samples]])
        self.train.images = np.concatenate([sample for (sample, _) in data[total_validation_samples + total_test_samples:]])
        self.train.labels = np.concatenate([label for (_, label) in data[total_validation_samples + total_test_samples:]])

    def set_batch_size(self, batch_size):
        for s in ['train', 'validation', 'test']:
            set = getattr(self, s) # shallow copy
            set.set_batch_size(batch_size)

    def generate_permutation(self, ver_seed=None, hor_seed=None):
        return self.train.generate_permutation(ver_seed, hor_seed)

    def permutate(self, permutation):
        self.permutation = permutation
        for s in ['train', 'validation', 'test']:
            set = getattr(self, s) # shallow copy
            set.permutate(self.permutation)

    def split(self, classes):
        self.classes = classes
        for s in ['train', 'validation', 'test']:
            set = getattr(self, s) # shallow copy
            set.split(self.classes)

    def pickle_data(self, dir):
        for s in ['train', 'validation', 'test']:
            os.mkdir(os.path.join(dir, s))
            set = getattr(self, s) # shallow copy
            set.pickle_data(os.path.join(dir, s))

    def retrieve_pickled_data(self, pickled_data_dir):
        for s in ['train', 'validation', 'test']:
            set = getattr(self, s) # shallow copy
            set.images = np.stack([load_pickled_object(os.path.join(pickled_data_dir, s, 'images', f)) for f in os.listdir(os.path.join(pickled_data_dir, s, 'images')) if os.path.isfile(os.path.join(pickled_data_dir, s, 'images', f)) and f.split('.')[-1] == 'pkl'])
            set.labels = np.stack([load_pickled_object(os.path.join(pickled_data_dir, s, 'labels', f)) for f in os.listdir(os.path.join(pickled_data_dir, s, 'labels')) if os.path.isfile(os.path.join(pickled_data_dir, s, 'labels', f)) and f.split('.')[-1] == 'pkl'])

    def print_summary(self):
        print('Dataset')
        print('Train')
        print('Batch Size:', self.train.batch_size, 'Index:', self.train.index, 'Indices:', type(self.train.indices))
        print('Images:', type(self.train.images), 'Shape:', self.train.images.shape)
        print('Labels:', type(self.train.labels), 'Shape:', self.train.labels.shape)
        print('Validation')
        print('Batch Size:', self.validation.batch_size, 'Index:', self.validation.index, 'Indices:', type(self.validation.indices))
        print('Images:', type(self.validation.images), 'Shape:', self.validation.images.shape)
        print('Labels:', type(self.validation.labels), 'Shape:', self.validation.labels.shape)
        print('Reversed Labels:', type(self.validation.rev_labels), 'Shape:', self.validation.rev_labels.shape)
        print('Test')
        print('Batch Size:', self.test.batch_size, 'Index:', self.test.index, 'Indices:', type(self.test.indices))
        print('Images:', type(self.test.images), 'Shape:', self.test.images.shape)
        print('Labels:', type(self.test.labels), 'Shape:', self.test.labels.shape)

    def plot_features(self, num_features, lib='plt'):
        for set in ['train', 'validation', 'test']:
            print(set, 'set')
            (features, labels) = getattr(self, set).next_batch()
            getattr(self, set).plot_features(features[:num_features], lib, labels[:num_features])
            getattr(self, set).randomize()
            if set == 'validation':
                print('Fisher batch')
                (features, labels, rev_labels) = getattr(self, set).next_fisher_batch()
                getattr(self, set).plot_features(features[:num_features], lib, labels[:num_features])
                getattr(self, set).fisher_randomize()

    @staticmethod
    def build_spectrograms(audio_dir, file):
        return vggish_input.wavfile_to_examples(os.path.join(audio_dir, file))

    @staticmethod
    def build_label(classes, num_classes, num_spectrograms):
        # Generate a list of labels for a given sample labelled with classes
        label = np.zeros(num_classes)
        for c in classes:
            label[c] = 1
        try:
            return  np.stack([label] * num_spectrograms)
        except:
            print('label', label)
            print('num_classes', num_classes)
            print('num_spectrograms', num_spectrograms)
            print('Unexpected error:', sys.exc_info()[0])
            raise

class MNIST(Dataset):
    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)
        # Retrieve MNIST
        mnist = read_data_sets('MNIST_data/', one_hot=True)
        # Collect the names of the datasets in Dataset objects
        sets = ['train', 'validation', 'test']
        # Populate the Dataset object with resized and reshaped MNIST
        for set in sets:
            original_set = getattr(mnist, set) # shallow copy
            resized_set = getattr(self, set) # shallow copy
            resized_set.images = np.stack([cv2.resize(np.reshape(image, (28,28)), (vggish_params.NUM_BANDS, vggish_params.NUM_FRAMES), interpolation=cv2.INTER_CUBIC) for image in np.split(original_set.images, original_set.images.shape[0])])
            resized_set.labels = np.stack([np.reshape(label, 10) for label in np.split(original_set.labels, original_set.labels.shape[0])])
            resized_set.initialize_indices()
            resized_set.randomize()
            if set == 'validation':
                resized_set.set_rev_labels()
                resized_set.initialize_fisher_indices()
                resized_set.fisher_randomize()

class SplitMNIST(Dataset):
    def __init__(self, classes, *args, **kwargs):
        super(SplitMNIST, self).__init__(*args, **kwargs)
        # Retrieve MNIST
        mnist = read_data_sets('MNIST_data/', one_hot=True)
        # Collect the names of the datasets in Dataset objects
        sets = ['train', 'validation', 'test']
        # Populate the Dataset object with resized and reshaped MNIST
        for set in sets:
            original_set = getattr(mnist, set) # shallow copy
            resized_set = getattr(self, set) # shallow copy
            resized_set.images = np.stack([cv2.resize(np.reshape(image, (28,28)), (vggish_params.NUM_BANDS, vggish_params.NUM_FRAMES), interpolation=cv2.INTER_CUBIC) for (image, label) in zip(np.split(original_set.images, original_set.images.shape[0]), np.split(original_set.labels, original_set.labels.shape[0])) if np.argmax(label) in classes])
            resized_set.labels = np.stack([self.generate_label(np.argmax(label), classes) for (image, label) in zip(np.split(original_set.images, original_set.images.shape[0]), np.split(original_set.labels, original_set.labels.shape[0])) if np.argmax(label) in classes])
            resized_set.initialize_indices()
            resized_set.randomize()
            if set == 'validation':
                resized_set.set_rev_labels()
                resized_set.initialize_fisher_indices()
                resized_set.fisher_randomize()

    @staticmethod
    def generate_label(label, classes):
        l = np.zeros(len(classes))
        l[classes.index(label)] = 1
        return l

class CIFAR100(Dataset):
    original_height = 32
    original_width = 32
    original_channels = 3
    num_classes = 100
    fine_label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    def __init__(self, *args, **kwargs):
        super(CIFAR100, self).__init__(*args, **kwargs)
        # Retrieve CIFAR100
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        # Format and split train and validation data
        train_images, train_labels = self.format_samples(x_train, y_train)
        train_data = list(zip(train_images, train_labels))
        shuffle(train_data)
        total_validation_images = int(len(train_data)/5)
        self.validation.images = np.stack([image for (image, _) in train_data[:total_validation_images]])
        self.validation.labels = np.stack([label for (_, label) in train_data[:total_validation_images]])
        self.validation.set_rev_labels()
        self.train.images = np.stack([image for (image, _) in train_data[total_validation_images:]])
        self.train.labels = np.stack([label for (_, label) in train_data[total_validation_images:]])
        # Format test data
        test_images, test_labels = self.format_samples(x_test, y_test)
        self.test.images = np.stack(test_images)
        self.test.labels = np.stack(test_labels)
        self.initialize_indices()

    def format_samples(self, images, labels):
        new_images = []
        new_labels = []
        for (img, lbl) in zip(np.split(images, images.shape[0]), np.split(labels, labels.shape[0])):
            image = np.reshape(img, (self.original_height, self.original_width, self.original_channels))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray, (vggish_params.NUM_BANDS, vggish_params.NUM_FRAMES), interpolation=cv2.INTER_CUBIC)
            new_images.append(resized_gray)
            label = np.zeros(self.num_classes)
            label[lbl[0][0]] = 1
            new_labels.append(label)
        return (new_images, new_labels)


class CIFAR10(Dataset):
    original_height = 32
    original_width = 32
    original_channels = 3
    num_classes = 10
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        # Retrieve CIFAR10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # Format and split train and validation data
        train_images, train_labels = self.format_samples(x_train, y_train)
        train_data = list(zip(train_images, train_labels))
        shuffle(train_data)
        total_validation_images = int(len(train_data)/5)
        self.validation.images = np.stack([image for (image, _) in train_data[:total_validation_images]])
        self.validation.labels = np.stack([label for (_, label) in train_data[:total_validation_images]])
        self.validation.set_rev_labels()
        self.train.images = np.stack([image for (image, _) in train_data[total_validation_images:]])
        self.train.labels = np.stack([label for (_, label) in train_data[total_validation_images:]])
        # Format test data
        test_images, test_labels = self.format_samples(x_test, y_test)
        self.test.images = np.stack(test_images)
        self.test.labels = np.stack(test_labels)
        self.initialize_indices()
    
    def format_samples(self, images, labels):
        new_images = []
        new_labels = []
        for (img, lbl) in zip(np.split(images, images.shape[0]), np.split(labels, labels.shape[0])):
            image = np.reshape(img, (self.original_height, self.original_width, self.original_channels))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray, (vggish_params.NUM_BANDS, vggish_params.NUM_FRAMES), interpolation=cv2.INTER_CUBIC)
            new_images.append(resized_gray)
            label = np.zeros(self.num_classes)
            label[lbl[0][0]] = 1
            new_labels.append(label)
        return (new_images, new_labels)


class SVHN(Dataset):
    def __init__(self, data_path, *args, **kwargs):
        super(SVHN, self).__init__(*args, **kwargs)
        # Load training data
        (train_images, train_labels) = load_svhn_data(data_path, 'train')
        # Shuffle train data and split it between train (90%) and validation sets (10%)
        train_data = list(zip(train_images, train_labels))
        shuffle(train_data)
        total_validation_images = int(len(train_data)/10)
        self.validation.images = np.stack([image for (image, _) in train_data[:total_validation_images]])
        self.validation.labels = np.stack([label for (_, label) in train_data[:total_validation_images]])
        self.validation.set_rev_labels()
        self.train.images = np.stack([image for (image, _) in train_data[total_validation_images:]])
        self.train.labels = np.stack([label for (_, label) in train_data[total_validation_images:]])
        # Load test data
        test_images, test_labels = load_svhn_data(data_path, 'test')
        self.test.images = np.stack(test_images)
        self.test.labels = np.stack(test_labels)
        self.initialize_indices()

class ExtraSVHN(Dataset):
    def __init__(self, data_path, *args, **kwargs):
        super(ExtraSVHN, self).__init__(*args, **kwargs)
        # Load train, test, and extra data
        (train_images, train_labels) = load_svhn_data(data_path, 'train')
        print('#train_images', len(train_images))
        print('#train_labels', len(train_labels))
        (test_images, test_labels) = load_svhn_data(data_path, 'test')
        print('#test_images', len(test_images))
        print('#test_labels', len(test_labels))
        (extra_images, extra_labels) = load_svhn_data(data_path, 'extra')
        print('#extra_images', len(extra_images))
        print('#extra_labels', len(extra_labels))
        all_images = train_images + test_images + extra_images
        all_labels = train_labels + test_labels + extra_labels
        # Merge all data and shuffle them
        all_data = list(zip(all_images, all_labels))
        shuffle(all_data)
        #Split all data 60%/20%/20% among train/validation/test sets
        total_validation_images = int(len(all_data)/5)
        total_test_images = int(len(all_data)/5)
        self.validation.images = np.stack([image for (image, _) in all_data[:total_validation_images]])
        self.validation.labels = np.stack([label for (_, label) in all_data[:total_validation_images]])
        self.validation.set_rev_labels()
        self.test.images = np.stack([image for (image, _) in all_data[total_validation_images:total_validation_images + total_test_images]])
        self.test.labels = np.stack([label for (_, label) in all_data[total_validation_images:total_validation_images + total_test_images]])
        self.train.images = np.stack([image for (image, _) in all_data[total_validation_images + total_test_images:]])
        self.train.labels = np.stack([label for (_, label) in all_data[total_validation_images + total_test_images:]])
        self.initialize_indices()

class ESC50(Dataset):
    def __init__(self, data_path, train_folds, validation_fold, test_fold, *args, **kwargs):
        super(ESC50, self).__init__(*args, **kwargs)
        self.num_classes = 50
        self.train.images, self.train.labels = self.retrieve_samples(data_path, train_folds)
        self.validation.images, self.validation.labels = self.retrieve_samples(data_path, validation_fold)
        self.test.images, self.test.labels = self.retrieve_samples(data_path, test_fold)
        self.initialize_indices()
        self.validation.set_rev_labels()

    def retrieve_samples(self, data_path, folds):
        print(folds)
        # Retrieve the audio samples
        audio_dir = os.path.join(data_path, 'ESC-50-master', 'audio')
        files = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f)) and f.split('.')[-1] == 'wav' and int(f.split('-')[0]) in folds]
        print('#files', len(files))
        files_with_bad_shape = []
        all_labels = []
        valid_samples = []
        valid_samples_labels = []
        for f in files:
            label =int(f.split('-')[3].split('.')[0])
            all_labels.append(label)
            spectrograms = self.build_spectrograms(audio_dir, f)
            if spectrograms.shape[0] > 0 and spectrograms.shape[1] == vggish_params.NUM_FRAMES and spectrograms.shape[2] == vggish_params.NUM_BANDS:
                valid_samples.append(spectrograms)
                valid_samples_labels.append(self.build_label([label], self.num_classes, spectrograms.shape[0]))
            else:
                files_with_bad_shape.append(f)
        all_labels = list(set(all_labels))
        print('#files', len(files))
        print('#files_with_bad_shape', len(files_with_bad_shape))
        print('#distinct labels', len(all_labels))
        print('#valid_samples', len(valid_samples))
        print('valid_samples[0].shape', valid_samples[0].shape)
        print('#valid_samples_labels', len(valid_samples_labels))
        print('valid_samples_labels[0].shape', valid_samples_labels[0].shape)
        return (np.concatenate(valid_samples), np.concatenate(valid_samples_labels))

class URBANSED(Dataset):
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    
    def __init__(self, data_path, *args, **kwargs):
        super(URBANSED, self).__init__(*args, **kwargs)
        for (dir, s) in zip(['train', 'validate', 'test'], ['train', 'validation', 'test']):
            set = getattr(self, s) # shallow copy
            set.images, set.labels = self.retrieve_set_samples(data_path, dir)
        self.initialize_indices()
        self.validation.set_rev_labels()

    def retrieve_set_samples(self, data_path, dir):
        audio_dir = os.path.join(data_path, 'URBAN-SED_v2.0.0', 'audio', dir)
        label_dir = os.path.join(data_path, 'URBAN-SED_v2.0.0', 'annotations', dir)
        audio_files = sorted(list(set([f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f)) and f.split('.')[-1] == 'wav'])))
        label_files = sorted(list(set([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f)) and f.split('.')[-1] == 'txt'])))
        files_with_bad_shape = []
        valid_samples = []
        valid_samples_labels = []
        for a, l in zip(audio_files, label_files):
            assert a.split('.')[0] == l.split('.')[0], 'Audio and label files do not match'
            spectrograms = self.build_spectrograms(audio_dir, a)
            if spectrograms.shape[0] > 0 and spectrograms.shape[1] == vggish_params.NUM_FRAMES and spectrograms.shape[2] == vggish_params.NUM_BANDS:
                annotations = self.get_file_annotations(os.path.join(label_dir, l))
                labels = self.build_labels(annotations, len(self.classes), spectrograms.shape[0])
                labeled_spectrograms = self.get_labeled_spectrograms(labels)
                valid_samples.append(spectrograms[labeled_spectrograms])
                valid_samples_labels.append(labels[labeled_spectrograms])
            else:
                files_with_bad_shape.append(f)
        print('#audio_files', len(audio_files))
        print('#label_files', len(label_files))
        print('#files_with_bad_shape', len(files_with_bad_shape))
        print('#valid_samples', len(valid_samples))
        print('valid_samples[0].shape', valid_samples[0].shape)
        print('#valid_samples_labels', len(valid_samples_labels))
        print('valid_samples_labels[0].shape', valid_samples_labels[0].shape)
        return (np.concatenate(valid_samples), np.concatenate(valid_samples_labels))

    def get_label_number(self, label):
        return self.classes.index(label)

    def get_file_annotations(self, file):
        annotations = []
        f = open(file, 'r')
        for line in f:
            start, end, label = line.split()
            start = int(math.floor(float(start)))
            end = min(int(math.ceil(float(end))), len(self.classes) - 1)
            label = self.get_label_number(label)
            annotations.append({'start': start, 'end': end, 'label': label})
        return annotations
        
    def build_labels(self, annotations, num_classes, num_spectrograms):
        # Generate labels for each spectrogram
        labels = np.zeros((num_spectrograms, num_classes))
        for a in annotations:
            for i in range(a['start'], a['end']):
                labels[i, a['label']] = 1
        return labels

    def get_labeled_spectrograms(self, labels):
        # Identify which spectrograms have at least one positive label
        return [i for i in range(labels.shape[0]) if np.sum(labels[i]) > 0]


class AudioSet(Dataset):
    def __init__(self, top_category_ids, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, pickled_data_dir=None, *args, **kwargs):
        super(AudioSet, self).__init__(*args, **kwargs)
        self.top_category_ids = top_category_ids
        self.has_quality_filter = has_quality_filter
        self.min_quality = min_quality
        self.has_depth_filter = has_depth_filter
        self.depths = depths
        self.has_rerated_filter = has_rerated_filter
        self.ontology_path = ontology_path
        self.config_files_path = config_files_path
        self.data_path = data_path
        self.pickled_data_dir = pickled_data_dir

        self.ontology = None
        self.ontology_ids = None
        self.branch_categories = None
        self.branch_ids = None
        self.categories_of_depth = None
        self.ids_of_depth = None
        self.rerated_ytids = None
        self.unique_labels = None

        self.retrieve_ontology()
        self.retrieve_branch_categories()
        if self.has_quality_filter:
            self.apply_quality_filter()
        if self.has_depth_filter:
            self.apply_depth_filter()
        if self.has_rerated_filter:
            self.retrieve_rerated_videos()
        self.retrieve_labels()
        self.apply_label_filter()
        if self.pickled_data_dir is None:
            self.make_dataset()
        else:
            print('Retrieving pickled data')
            self.retrieve_pickled_data(pickled_data_dir)
            print('Finished retrieving pickled data')
            self.validation.set_rev_labels()
            self.initialize_indices()

    def print_categories(self, categories):
        # Print the index, id, and name of each category present in categories
        for i, cat in enumerate(categories):
            print(i, cat['id'], cat['name'])

    def check_repeated_ids(self, ids):
        # Check whether there is any repeated id in ids
        repeated_ids = []
        total_repeats = 0
        for id in ids:
            repeats = ids.count(id)
            if repeats > 1 and id not in repeated_ids:
                repeated_ids.append(id)
                total_repeats += repeats
        print('#repeated_ids', len(repeated_ids))
        print('total_repeats', total_repeats)

    def find_category(self, category_id):
        # Find category whose id is category_id in the ontology
        return self.ontology[self.ontology_ids.index(category_id)]

    def find_branch_categories(self, category_id):
        # Find categories in the ontology starting with category_id (depth-first search)
        branch_categories = []
        category = self.find_category(category_id)
        if len(category['restrictions']) == 0:
            branch_categories.append(category)
        if len(category['child_ids']) == 0:
            return branch_categories
        for child_id in category['child_ids']:
            branch_categories += self.find_branch_categories(child_id)
        return branch_categories

    def find_categories_of_depth(self, category_id, depth):
        # Find categories in the ontology starting with category_id (depth-first search)
        # whose depth is in the given depth list
        categories_of_depth = []
        category = self.find_category(category_id)
        if (len(category['restrictions']) == 0) and ((depth in self.depths) or (self.depths == [-1] and len(category['child_ids']) == 0)):
            categories_of_depth.append(category)
        if len(category['child_ids']) == 0:
            return categories_of_depth
        for child_id in category['child_ids']:
            categories_of_depth += self.find_categories_of_depth(child_id, depth + 1)
        return categories_of_depth

    def make_label(self, labels, num_spectrograms):
        # Generate a numpy array label for sample with labels
        if self.has_depth_filter:
            label = np.zeros(len(self.ids_of_depth))
        else:
            label = np.zeros(len(self.branch_ids))
        for id in labels:
            if self.has_depth_filter and id in self.ids_of_depth:
                label[self.ids_of_depth.index(id)] = 1
            elif not self.has_depth_filter:
                label[self.branch_ids.index(id)] = 1
        try:
            return  np.stack([label] * num_spectrograms)
        except:
            print('label', label)
            print('num_spectrograms', num_spectrograms)
            print('Unexpected error:', sys.exc_info()[0])
            raise

    def retrieve_ontology(self):
        # Retrieve AudioSet ontology
        self.ontology = load_json(os.path.join(self.ontology_path, 'ontology-master', 'ontology.json'))
        self.ontology_ids = [category['id'] for category in self.ontology]
        print('#ontology', len(self.ontology))
        print('#ontology_ids', len(self.ontology_ids))

    def retrieve_branch_categories(self, has_print=False):
        # Retrieve categories in the given branches of AudioSet
        self.branch_categories = []
        for category_id in self.top_category_ids:
            self.branch_categories += self.find_branch_categories(category_id)
        self.branch_ids = [category['id'] for category in self.branch_categories]
        # Extract unique ids in branches because of nodes with multiple parents in the ontology
        self.branch_ids = sorted(list(set(self.branch_ids)))
        self.branch_categories = [self.find_category(id) for id in self.branch_ids]
        print('#branch_categories', len(self.branch_categories))
        print('#branch_ids', len(self.branch_ids))
        print('check branch_ids')
        self.check_repeated_ids(self.branch_ids)
        if has_print:
            self.print_categories(self.branch_categories)

    def apply_quality_filter(self, has_print=False):
        # Retrieve the quality assessment of categories
        quality = {}
        with open(os.path.join(self.config_files_path, 'qa_true_counts_copy.csv'), 'rt') as csvfile:
            quality_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in quality_reader:
                if int(row[1]) > 0:
                    quality[row[0]] = float(row[2]) / float(row[1])
        print('#quality', len(quality.keys()))
        assessed_branch_ids = [branch_id for branch_id in self.branch_ids if branch_id in quality.keys()]
        print('#assessed_branch_ids', len(assessed_branch_ids))
        # Filter branches by min quality (inclusive)
        print('Apply quality filter')
        self.branch_categories = [category for category in self.branch_categories if category['id'] in quality and quality[category['id']] >= self.min_quality]
        self.branch_ids = [category['id'] for category in self.branch_categories]
        print('#branch_categories', len(self.branch_categories))
        print('#branch_ids', len(self.branch_ids))
        print('check branch_ids')
        self.check_repeated_ids(self.branch_ids)
        if has_print:
            self.print_categories(self.branch_categories)

    def apply_depth_filter(self, has_print=False):
        # Retrieve categories in the given branches of AudioSet of specified depths
        categories_of_depth = []
        for category_id in self.top_category_ids:
            categories_of_depth += self.find_categories_of_depth(category_id, 0)
        ids_of_depth = [category['id'] for category in categories_of_depth]
        # Extract unique ids in categories_of_depth because of nodes with multiple parents in the ontology
        ids_of_depth = sorted(list(set(ids_of_depth)))
        categories_of_depth = [self.find_category(id) for id in ids_of_depth]
        print('#categories_of_depth', len(categories_of_depth))
        print('#ids_of_depth', len(ids_of_depth))
        print('check ids_of_depth')
        self.check_repeated_ids(ids_of_depth)
        # Update branch_categories and branch_ids
        print('Apply depth filter')
        self.ids_of_depth = [id for id in ids_of_depth if id in self.branch_ids]
        self.categories_of_depth = [self.find_category(id) for id in self.ids_of_depth]
        print('#categories_of_depth', len(self.categories_of_depth))
        print('#ids_of_depth', len(self.ids_of_depth))
        print('check ids_of_depth')
        self.check_repeated_ids(self.ids_of_depth)
        if has_print:
            self.print_categories(self.categories_of_depth)

    def retrieve_rerated_videos(self, check_rerated_partition=False):
        # Retrieve the rerated YouTube video IDs
        with open(os.path.join(self.config_files_path, 'rerated_video_ids.txt'), 'rt') as csvfile:
            rerated_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            self.rerated_ytids = [row[0] for row in rerated_reader]
        print('#rerated_ytids', len(self.rerated_ytids))

        if check_rerated_partition:
            # Retrieve the balanced train segments YouTube video IDS and extract the rerated ones
            with open(os.path.join(self.config_files_path, 'balanced_train_segments_copy.csv'), 'rt') as csvfile:
                train_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                train_ytids = [row[0] for row in train_reader]
            rerated_train_ytids = [ytid for ytid in train_ytids if ytid in self.rerated_ytids]
            print('#train_ytids', len(train_ytids))
            print('#rerated_train_ytids', len(rerated_train_ytids))

            # Retrieve the eval segments YouTube video IDS and extract the rerated ones
            with open(os.path.join(self.config_files_path, 'eval_segments_copy.csv'), 'rt') as csvfile:
                eval_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                eval_ytids = [row[0] for row in eval_reader]
            rerated_eval_ytids = [ytid for ytid in eval_ytids if ytid in self.rerated_ytids]
            print('#eval_ytids', len(eval_ytids))
            print('#rerated_eval_ytids', len(rerated_eval_ytids))

            # Retrieve the unbalanced train segments YouTube video IDS and extract the rerated ones
            with open(os.path.join(self.config_files_path, 'unbalanced_train_segments_copy.csv'), 'rt') as csvfile:
                unbalanced_train_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                unbalanced_train_ytids = [row[0] for row in unbalanced_train_reader]
            rerated_unbalanced_train_ytids = [ytid for ytid in unbalanced_train_ytids if ytid in self.rerated_ytids]
            print('#unbalanced_train_ytids', len(unbalanced_train_ytids))
            print('#rerated_unbalanced_train_ytids', len(rerated_unbalanced_train_ytids))

            # Check # of rerated videos
            print('#rerated balanced_train + eval + unbalanced_train', len(rerated_train_ytids) + len(rerated_eval_ytids) + len(rerated_unbalanced_train_ytids))

    def retrieve_segments_labels(self, segments_config_file):
        # Retrieve the unique labels of segments_config_file
        with open(os.path.join(self.config_files_path, segments_config_file), 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            all_labels = []
            for row in reader:
                # Retrieve labels
                labels = [label.replace(' ', '').replace('"', '') for label in row[3:]]
                all_labels += labels
        return sorted(list(set(all_labels)))

    def retrieve_labels(self):
        balanced_train_labels = self.retrieve_segments_labels('balanced_train_segments_copy.csv')
        eval_labels = self.retrieve_segments_labels('eval_segments_copy.csv')
        self.unique_labels = sorted(list(set(balanced_train_labels + eval_labels)))
        print('#balanced_train_labels', len(balanced_train_labels))
        print('#eval_labels', len(eval_labels))
        print('#unique_labels', len(self.unique_labels))

    def apply_label_filter(self, has_print=False):
        # Check whether any branch_id is not present in any sample label
        absent_branch_ids =[id for id in self.branch_ids if id not in self.unique_labels]
        if len(absent_branch_ids) > 0:
            print('Apply label filter on branch_ids')
            self.branch_ids = [id for id in self.branch_ids if id not in absent_branch_ids]
            self.branch_categories = [self.find_category(id) for id in self.branch_ids]
            absent_branch_ids =[id for id in self.branch_ids if id not in self.unique_labels]
            print('#absent_branch_ids', len(absent_branch_ids))
            assert len(absent_branch_ids) == 0, 'branch_ids have one or more ids not present in any sample label'
            print('#branch_categories', len(self.branch_categories))
            print('#branch_ids', len(self.branch_ids))
            print('check branch_ids')
            self.check_repeated_ids(self.branch_ids)
            if has_print:
                print('branch_categories')
                self.print_categories(self.branch_categories)
        if self.has_depth_filter:
            # Check whether any id_of_depth is not present in any sample label
            absent_ids_of_depth =[id for id in self.ids_of_depth if id not in self.unique_labels]
            if len(absent_ids_of_depth) > 0:
                print('Apply label filter on ids_of_depth')
                self.ids_of_depth = [id for id in self.ids_of_depth if id not in absent_ids_of_depth]
                self.categories_of_depth = [self.find_category(id) for id in self.ids_of_depth]
                absent_ids_of_depth =[id for id in self.ids_of_depth if id not in self.unique_labels]
                print('#absent_ids_of_depth', len(absent_ids_of_depth))
                assert len(absent_ids_of_depth) == 0, 'ids_of_depth have one or more ids not present in any sample label'
                print('#categories_of_depth', len(self.categories_of_depth))
                print('#ids_of_depth', len(self.ids_of_depth))
                print('check ids_of_depth')
                self.check_repeated_ids(self.ids_of_depth)
                if has_print:
                    print('categories_of_depth')
                    self.print_categories(self.categories_of_depth)

    def retrieve_samples(self, segments_dir, segments_config_file):
        # Retrieve the samples of segments_config_file compatible with branch_categories (and categories_of_depth if has_depth_filter is set to True)
        segments_files = [f for f in os.listdir(os.path.join(self.data_path, segments_dir)) if os.path.isfile(os.path.join(self.data_path, segments_dir, f)) and f.split('.')[-1] == 'wav']
        with open(os.path.join(self.config_files_path, segments_config_file), 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            all_samples = []
            shared_top = []
            samples_with_at_least_one_matching_label = []
            samples_with_at_least_one_matching_and_one_extraneous_labels = []
            samples_with_bad_shape = []
            valid_samples = []
            valid_sample_labels = []
            for row in reader:
                # Retrieve labels
                labels = [label.replace(' ', '').replace('"', '') for label in row[3:]]
                # Check whether sample has multiple top categories
                top = 0
                for t in self.top_category_ids:
                    if t in labels:
                        top += 1
                if top > 1:
                    shared_top.append(row[0])
                # Check whether sample has any label not in branch_ids
                extraneous_labels = 0
                for label in labels:
                    if label not in self.branch_ids:
                        extraneous_labels += 1
                # Check whether sample has any label in branch_ids
                for id in self.branch_ids:
                    if id in labels:
                        samples_with_at_least_one_matching_label.append(row[0])
                        if extraneous_labels > 0:
                            samples_with_at_least_one_matching_and_one_extraneous_labels.append(row[0])
                        break
                # Check whether there is any label of specified depths
                if self.has_depth_filter:
                    labels_of_depth = 0
                    for label in labels:
                        if label in self.ids_of_depth:
                            labels_of_depth += 1
                all_samples.append(row[0])
                file = row[0] + '_' + row[1].strip() + '.wav'
                if (extraneous_labels) == 0 and (not self.has_depth_filter or labels_of_depth > 0) and (not self.has_rerated_filter or row[0] in self.rerated_ytids) and (file in segments_files):
                    samples = self.build_spectrograms(os.path.join(self.data_path, segments_dir), file)
                    if samples.shape[0] > 0 and samples.shape[1] == vggish_params.NUM_FRAMES and samples.shape[2] == vggish_params.NUM_BANDS:
                        valid_samples.append(samples)
                        valid_sample_labels.append(self.make_label(labels, samples.shape[0]))
                    else:
                        samples_with_bad_shape.append(row[0])
                        #print('Bad sample shape %s of file %s' % (samples.shape, file))
        print('#all_samples', len(all_samples))
        print('#shared_top', len(shared_top))
        print('#samples_with_at_least_one_matching_label', len(samples_with_at_least_one_matching_label))
        print('#samples_with_at_least_one_matching_and_one_extraneous_labels', len(samples_with_at_least_one_matching_and_one_extraneous_labels))
        print('#samples_with_bad_shape', len(samples_with_bad_shape))
        print('#valid_samples', len(valid_samples))
        print('valid_samples[0].shape', valid_samples[0].shape)
        print('#valid_sample_labels', len(valid_sample_labels))
        print('valid_sample_labels[0].shape', valid_sample_labels[0].shape)
        return (valid_samples, valid_sample_labels)

    def make_dataset(self):
        print('')
        print('retrieve balanced train samples')
        balanced_train_samples, balanced_train_labels = self.retrieve_samples('balanced_train_segments', 'balanced_train_segments_copy.csv')
        # Shuffle train data and split it between train (90%) and validation sets (10%)
        balanced_train_data = list(zip(balanced_train_samples, balanced_train_labels))
        shuffle(balanced_train_data)
        print('#balanced_train_data', len(balanced_train_data))
        total_validation_samples = int(len(balanced_train_data)/10)
        self.validation.images = np.concatenate([sample for (sample, _) in balanced_train_data[:total_validation_samples]])
        self.validation.labels = np.concatenate([label for (_, label) in balanced_train_data[:total_validation_samples]])
        self.validation.set_rev_labels()
        self.train.images = np.concatenate([sample for (sample, _) in balanced_train_data[total_validation_samples:]])
        self.train.labels = np.concatenate([label for (_, label) in balanced_train_data[total_validation_samples:]])
        print('')
        print('retrieve eval samples')
        eval_samples, eval_labels = self.retrieve_samples('eval_segments', 'eval_segments_copy.csv')
        self.test.images = np.concatenate(eval_samples)
        self.test.labels = np.concatenate(eval_labels)
        self.initialize_indices()

class Music(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(Music, self).__init__(top_category_ids=['/m/04rlf'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class HumanSounds(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(HumanSounds, self).__init__(top_category_ids=['/m/0dgw9r'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class Animal(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(Animal, self).__init__(top_category_ids=['/m/0jbk'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class SourceAmbiguousSounds(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(SourceAmbiguousSounds, self).__init__(top_category_ids=['/t/dd00098'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class SoundsOfThings(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(SoundsOfThings, self).__init__(top_category_ids=['/t/dd00041'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class NaturalSounds(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(NaturalSounds, self).__init__(top_category_ids=['/m/059j3w'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class ChannelEnvironmentBackground(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(ChannelEnvironmentBackground, self).__init__(top_category_ids=['/t/dd00123'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class Miscellaneous(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(Miscellaneous, self).__init__(top_category_ids=['/m/0jbk', '/t/dd00098', '/m/059j3w', '/t/dd00123'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)

class MusicHumanSounds(AudioSet):
    def __init__(self, has_quality_filter, min_quality, has_depth_filter, depths, has_rerated_filter, ontology_path, config_files_path, data_path, *args, **kwargs):
        super(MusicHumanSounds, self).__init__(top_category_ids=['/m/04rlf', '/m/0dgw9r'], has_quality_filter=has_quality_filter, min_quality=min_quality, has_depth_filter=has_depth_filter, depths=depths, has_rerated_filter=has_rerated_filter, ontology_path=ontology_path, config_files_path=config_files_path, data_path=data_path, *args, **kwargs)
