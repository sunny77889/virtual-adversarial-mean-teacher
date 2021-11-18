# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import struct

import numpy as np
import scipy.io

from .utils import random_balanced_partitions, random_partitions


class Datafile:
    def __init__(self, data_path,label_path):
        self.images_path = data_path
        self.labels_path = label_path
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._load()
        return self._data
    
    def _load(self):
        
        
        #加载‘idx1_ubyte'文件
        with open(self.labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                    lbpath.read(8))
            labels = np.fromfile(lbpath,
                                dtype=np.uint8)
        #加载'idx3_ubyte'文件
        with open(self.images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                                imgpath.read(16))
            images = np.fromfile(imgpath,
                                dtype=np.uint8).reshape(len(labels), 784)
        #根据实际数据，将类别转换为二类：0（正常）和 1（异常）
        labels[labels<10]=0
        labels[labels>=10]=1
        n_examples = len(images)
        data = np.zeros(n_examples, dtype=[
            ('x', np.uint8, (28, 28, 1)),
            ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
        data['x']=images.reshape((len(images), 28,28,1))
        data['y']=labels
        self._data = data


class COMPARE:
    DIR = os.path.join('tensorflow','data', 'images', '5_Mnist')
    # 5897是训练集中样本的数量， 64355是测试集样本数量
    FILES = {
        'train': Datafile(os.path.join(DIR, 'train-images-idx3-ubyte'),os.path.join(DIR, 'train-labels-idx1-ubyte')),
        'test': Datafile(os.path.join(DIR, 't10k-images-idx3-ubyte'),os.path.join(DIR, 't10k-labels-idx1-ubyte'))
    }
    VALIDATION_SET_SIZE = 589  # 10% of the training set
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
