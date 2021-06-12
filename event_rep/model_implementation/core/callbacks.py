"""
File: event_rep/model_implementation/core/callbacks.py
Author: Mughilan Muthupari
Creation Date: 2021-05-27

This file contains any custom callbacks outside of Tensorflow's built-in callbacks. One is the
MetricCallback, which saves the training metrics from epoch to epoch. This can be used to generate
graphs after the training is over.
"""
import numpy as np
from collections import defaultdict
import json
import os

from model_implementation.architecture.models import MTRFv4Res
from tensorflow.keras.callbacks import Callback


class MetricCallback(Callback):
    """
    This class will write general info
    about the model, including the past validation
    costs to a JSON metric file.
    """
    def __init__(self, model_odj: MTRFv4Res, save_dir, save_freq=1, past_metrics=None):
        super(MetricCallback, self).__init__()
        self.model_obj = model_odj
        self.save_dir = save_dir
        self.save_freq = save_freq

        if past_metrics is None:
            self.history = defaultdict(list)
            self.metrics = dict()
            self.metrics['best_validation_cost'] = 0
            self.metrics['best_epcoh'] = 0
            self.metrics['total_epochs'] = 0
        else:
            self.metrics = past_metrics
            self.history = self.metrics['history']

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history[k].append(float(v))
        self.metrics['history'] = self.history
        # Find index of minimum validation loss and extract
        # the actual values
        min_loc = int(np.argmin(self.history['val_loss']))
        self.metrics['best_validation_cost'] = self.history['val_loss'][min_loc]
        self.metrics['best_epcoh'] = min_loc + 1
        self.metrics['total_epochs'] += 1

        # Write the file to the save directory on the save frequency
        if self.metrics['total_epochs'] % self.save_freq == 0:
            with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
                json.dump(self.metrics, f, sort_keys=True, indent=2)