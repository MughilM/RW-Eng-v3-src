"""
File: event_rep/evaluation/tasks.py
Author: Mughilan Muthupari
Creation Date: 2021-05-29

This file contains our evaluation tasks. We have a base class that defines the basic
functions, while the rest are subclassed.
"""
import tensorflow as tf
import numpy as np
import os
from typing import Dict, Type, Tuple
from model_implementation.architecture.models import MTRFv4Res
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet


class EvaluationTask:
    def __init__(self, SRC_DIR, model_name, experiment_dir):
        # This is needed so the correct model structure is used
        # when loading the model from the checkpoint.
        PARAM_TO_MODEL: Dict[str, Type[MTRFv4Res]] = {
            'v4': MTRFv4Res
        }
        self.SRC_DIR = SRC_DIR
        self.model_name = model_name
        self.experiment_dir = experiment_dir
        # Because this is a subclassed model we can't load the model all together,
        # due to unstable serialization for subclassed models. Instead, we can use
        # the checkpoints to load the weights. On the flip side, we have to make the
        # correct model structure, using the hyperparameters. This is why the model_name
        # is needed. The hyperparameters are saved in the experiment directory.
        self.model_hp_set = HyperparameterSet(os.path.join(self.experiment_dir, 'hyperparameters.json'))
        # Load the model using the hyperparameters
        self.model: MTRFv4Res = PARAM_TO_MODEL[self.model_name](self.model_hp_set)
        print('Loaded model:')
        print(self.model.build().summary())
        # This should be used to save any metrics, that can later be written in
        # the generate report method.
        self.metrics = {}

    def _preprocess(self) -> Tuple[Type[np.ndarray], Type[np.ndarray], Type[np.ndarray], Type[np.ndarray]]:
        """
        This function should output the correctly formatted numpy arrays
        that are designed to be passed in the model. Any variables that require state
        and need to be accessed when generating the report should be saved in metrics dictionary.
        :return: A 4-tuple of np.arrays in the correct order, for passing through the model.
        """
        raise NotImplementedError('Please implement a way to preprocess the data.')

    def _calc_score(self, word_output, role_output):
        """
        This internal method uses the output of the model to calculate the 'score' for
        the evaluation task, whether that be accuracy, correlation, or anything else.
        The score and anything related to it should be saved in the metrics dictionary for reading
        later and writing to a report.
        Ideally, the
        :param word_output: The word outputs from the model.
        :param role_output: The role outputs from the model.
        :return:
        """
        raise NotImplementedError('Please implement a way to calculate a score for this task.')

    def _generate_report(self):
        """
        This method should be implemented to generate a report, whether that is a txt
        or any other file of the evaluation results. It should use the values saved in
        self.metrics dictionary.
        :return:
        """
        raise NotImplementedError('Please implement a way to generate the report and save to the experiment '
                                  'directory.')

    def run_task(self):
        """
        This is a method that should be run externally. It will simply call the preprocess
        and score functions in turn, and output the result of _calc_score(). This
        method SHOULD NOT BE IMPLEMENTED in subclasses.
        :return:
        """
        input_words, input_roles, target_word, target_role = self._preprocess()
        word_outputs, role_outputs = self.model([input_words, input_roles, target_word, target_role])
        self._calc_score(word_outputs, role_outputs)
        self._generate_report()

