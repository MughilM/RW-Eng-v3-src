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
    def __init__(self, model_name, experiment_dir):
        # This is needed so the correct model structure is used
        # when loading the model from the checkpoint.
        PARAM_TO_MODEL: Dict[str, Type[MTRFv4Res]] = {
            'v4': MTRFv4Res
        }
        self.model_name = model_name
        self.experiment_dir = experiment_dir
        # Because this is a subclassed model we can't load the model all together,
        # due to unstable serialization for subclassed models. Instead, we can use
        # the checkpoints to load the weights. On the flip side, we have to make the
        # correct model structure, using the hyperparameters. This is why the model_name
        # is needed. The hyperparameters are saved in the experiment directory.
        model_hp_set = HyperparameterSet(os.path.join(self.experiment_dir, 'hyperparameters.json'))
        # Load the model using the hyperparameters
        self.model: MTRFv4Res = PARAM_TO_MODEL[self.model_name](model_hp_set)
        print('Loaded model:')
        print(self.model.build().summary())

    def _preprocess(self) -> Tuple[Type[np.ndarray], Type[np.ndarray], Type[np.ndarray], Type[np.ndarray]]:
        """
        This function should output the correctly formatted numpy arrays
        that are designed to be passed in the model.
        :return: A 4-tuple of np.arrays in the correct order, for passing through the model.
        """
        raise NotImplementedError('Please implement a way to preprocess the data.')

    def _calc_score(self, word_output, role_output):
        """
        This internal method uses the output of the model to calculate the 'score' for
        the evaluation task, whether that be accuracy, correlation, or anything else.
        :param word_output: The word outputs from the model.
        :param role_output: The role outputs from the model.
        :return: A type of score specific to the evaluation task
        """
        raise NotImplementedError('Please implement a way to calculate a score for this task.')

    def run_task(self):
        """
        This is a method that should be run externally. It will simply call the preprocess
        and score functions in turn, and output the result of _calc_score(). This
        method SHOULD NOT BE IMPLEMENTED in subclasses.
        :return: The score for this evaluation task.
        """
        input_words, input_roles, target_word, target_role = self._preprocess()
        word_outputs, role_outputs = self.model([input_words, input_roles, target_word, target_role])
        return self._calc_score(word_outputs, role_outputs)


