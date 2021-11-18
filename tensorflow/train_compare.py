# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Train ConvNet Mean Teacher on SVHN training set and evaluate against a validation set

This runner converges quickly to a fairly good accuracy.
On the other hand, the runner experiments/svhn_final_eval.py
contains the hyperparameters used in the paper, and converges
much more slowly but possibly to a slightly better accuracy.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
from datetime import datetime

from datasets.compare1 import COMPARE
from experiments.run_context import RunContext
from mean_teacher import minibatching
from mean_teacher.model import Model

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')


def MT_train(data_seed=0):
    n_labeled = 320
    n_extra_unlabeled = 0

    model = Model(RunContext(__file__, 0))
    model['rampdown_length'] = 0
    model['rampup_length'] = 4000
    model['training_length'] = 6000
    model['max_consistency_cost'] = 50.0

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    trojan = COMPARE(data_seed, n_labeled, n_extra_unlabeled, True)
    training_batches = minibatching.training_batches(trojan.training, n_labeled_per_batch=50)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(trojan.evaluation)
    model.train(training_batches, evaluation_batches_fn)

def MT_test(data_seed=0):
    n_labeled = "all"
    n_extra_unlabeled = 0

    model = Model(RunContext(__file__, 0))
    model['rampdown_length'] = 0
    model['rampup_length'] = 4000
    model['training_length'] = 6000
    model['max_consistency_cost'] = 50.0


    # tensorboard_dir = model.save_tensorboard_graph()
    
    # LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    trojan = COMPARE(data_seed, n_labeled, n_extra_unlabeled, True)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(trojan.evaluation)
    model.load("tensorflow/results/train_compare/savedModel/0/transient/")
    preds = model.pred(evaluation_batches_fn)  # 预测的输出：0，1序列， 0：正常  1:异常
    

import os
import struct

import numpy as np

if __name__ == "__main__":
    # MT_train()
    MT_test()
