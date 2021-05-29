"""
File: event_rep/evaluation/tasks.py
Author: Mughilan Muthupari
Creation Date: 2021-05-29

This file contains our evaluation tasks. We have a base class that defines the basic
functions, while the rest are subclassed.
"""
import tensorflow as tf
import numpy as np

class EvaluationTask:
    def __init__(self, model_name, experiment_name):
        self.model_name = model_name
        self.experiment_name = experiment_name

    def preprocess(self):
        """
        This function should output the correctly formatted numpy arrays
        that are designed to be passed in the model.
        :return: A 4-tuple of np.arrays in the correct order, for passing through the model.
        """
        raise NotImplementedError('Please implement a way to preprocess the data.')

    def score(self):
        """
        This method is specific to the class. The point is that you shouldn't
        need to pass throg
        :return:
        """
        pass
