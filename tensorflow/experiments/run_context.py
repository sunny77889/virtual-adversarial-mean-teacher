# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime

from pandas import DataFrame


class TrainLog:
    """Saves training logs in Pandas msgpacks"""

    INCREMENTAL_UPDATE_TIME = 300

    def __init__(self, directory, name):
        self.log_file_path = "{}/{}.msgpack".format(directory, name)
        self._log = defaultdict(dict)
        self._log_lock = threading.RLock()
        self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record_single(self, step, column, value):
        self._record(step, {column: value})

    def record(self, step, col_val_dict):
        self._record(step, col_val_dict)

    def save(self):
        df = self._as_dataframe()
        print(self.log_file_path)
        df.to_pickle(self.log_file_path)

    def _record(self, step, col_val_dict):
        with self._log_lock:
            self._log[step].update(col_val_dict)
            if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
                self._last_update_time = time.time()
                self.save()

    def _as_dataframe(self):
        with self._log_lock:
            return DataFrame.from_dict(self._log, orient='index')


class RunContext:
    """Creates directories and files for the run"""

    def __init__(self, runner_file, run_idx):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        runner_name = os.path.basename(runner_file).split(".")[0]
        self.result_dir="./modelSaved/meanTeacher/train_compare/savedModel/0"
        self.transient_dir = self.result_dir + "/transient"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        if not os.path.exists(self.transient_dir):
            os.makedirs(self.transient_dir)

    def create_train_log(self, name):
        return TrainLog(self.result_dir, name)
