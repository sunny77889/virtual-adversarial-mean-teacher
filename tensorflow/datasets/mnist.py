# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os

import numpy as np
import scipy.io

from .utils import random_balanced_partitions, random_partitions


class Datafile:
    def __init__(self, path, n_examples):
        self.path = path
        self.n_examples = n_examples
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._load()
        return self._data

    def _load(self):
        data = np.zeros(self.n_examples, dtype=[
            ('x', np.uint8, (28, 28, 1)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        dictionary = scipy.io.loadmat(self.path)
        temp = np.transpose(dictionary['x'], [2, 0, 1])
        temp = temp[:,:,:,np.newaxis]
        data['x'] = temp
        data['y'] = dictionary['y'].reshape((-1))
        #data['y'][data['y'] == 10] = 0
        self._data = data


class MNIST:
    DIR = os.path.join('data', 'images', 'mnist')
    FILES = {
        'train': Datafile(os.path.join(DIR, 'mnist_train.mat'), 60000),
        'test': Datafile(os.path.join(DIR, 'mnist_test.mat'),10000)
    }
    VALIDATION_SET_SIZE = 6000  # 10% of the training set
    UNLABELED = -1

    def __init__(self, data_seed=0, n_labeled='all', n_extra_unlabeled=0, test_phase=False):
        random = np.random.RandomState(seed=data_seed)

        if test_phase:
            self.evaluation, self.training = self._test_and_training()
        else:
            self.evaluation, self.training = self._validation_and_training(random)

        if n_labeled != 'all':
            self.training = self._unlabel(self.training, n_labeled, random)

        if n_extra_unlabeled > 0:
            self.training = self._add_extra_unlabeled(self.training, n_extra_unlabeled, random)

    def _validation_and_training(self, random):
        return random_partitions(self.FILES['train'].data, self.VALIDATION_SET_SIZE, random)

    def _test_and_training(self):
        return self.FILES['test'].data, self.FILES['train'].data

    def _unlabel(self, data, n_labeled, random):
        labeled, unlabeled = random_balanced_partitions(
            data, n_labeled, labels=data['y'], random=random)
        unlabeled['y'] = self.UNLABELED
        return np.concatenate([labeled, unlabeled])

    def _add_extra_unlabeled(self, data, n_extra_unlabeled, random):
        extra_unlabeled, _ = random_partitions(self.FILES['extra'].data, n_extra_unlabeled, random)
        extra_unlabeled['y'] = self.UNLABELED
        return np.concatenate([data, extra_unlabeled])
