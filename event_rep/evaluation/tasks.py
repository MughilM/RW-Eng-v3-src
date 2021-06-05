"""
File: event_rep/evaluation/tasks.py
Author: Mughilan Muthupari
Creation Date: 2021-05-29

This file contains our evaluation tasks. We have a base class that defines the basic
functions, while the rest are subclassed.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from typing import Dict, Type, Tuple
from model_implementation.architecture.models import MTRFv4Res
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet


class EvaluationTask:
    def __init__(self, SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name):
        # This is needed so the correct model structure is used
        # when loading the model from the checkpoint.
        PARAM_TO_MODEL: Dict[str, Type[MTRFv4Res]] = {
            'v4': MTRFv4Res
        }
        self.SRC_DIR = SRC_DIR
        self.model_name = model_name
        self.experiment_name = experiment_name
        # Because this is a subclassed model we can't load the model all together,
        # due to unstable serialization for subclassed models. Instead, we can use
        # the checkpoints to load the weights. On the flip side, we have to make the
        # correct model structure, using the hyperparameters. This is why the model_name
        # is needed. The hyperparameters are saved in the experiment directory.
        self.hp_set = HyperparameterSet(os.path.join(self.SRC_DIR, EXPERIMENT_DIR, self.experiment_name, 'hyperparameters.json'))
        # Load the model using the hyperparameters
        self.model: MTRFv4Res = PARAM_TO_MODEL[self.model_name](self.hp_set)
        print('Loaded model:')
        print(self.model.build().summary())
        # This should be used to save any metrics, that can later be written in
        # the generate report method.
        self.metrics = {}

    def _preprocess(self) -> Dict[str, np.ndarray]:
        """
        This function should output the correctly formatted numpy arrays
        that are designed to be passed in the model. Any variables that require state
        and need to be accessed when generating the report should be saved in metrics dictionary.
        :return: A dictionary mapping the input layer names to the matrices.
        """
        raise NotImplementedError('Please implement a way to preprocess the data.')

    def _calc_score(self, outputs: dict):
        """
        This internal method uses the output of the model to calculate the 'score' for
        the evaluation task, whether that be accuracy, correlation, or anything else.
        The score and anything related to it should be saved in the metrics dictionary for reading
        later and writing to a report.
        Ideally, the
        :param outputs; The dictionary outputted from the model's call method.
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
        all_inputs = self._preprocess()
        outputs = self.model(all_inputs)
        self._calc_score(outputs)
        self._generate_report()


class CorrelateTFScores(EvaluationTask):
    """
    This class handles all the evaluation tasks where we are given a verb, noun, the role the
    noun is supposed to fill, and a thematic fit judgement score. This data should be in
    csv format with columns as verb,noun,role,score.
    """
    def __init__(self, SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, dataset_name):
        super().__init__(SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name)
        self.dataset_input_file = os.path.join(self.SRC_DIR, f'evaluation/task_data/{dataset_name}.csv')
        self.dataset = pd.read_csv(self.dataset_input_file)
        self.dataset_ids = self.dataset.copy()

    def _preprocess(self) -> Dict[str, np.ndarray]:
        ROLE_MAP = {
            'subj': 'A0',
            'obj': 'A1',
            'instrument': 'AM-MNR',
            'location': 'AM-LOC'
        }
        # dataset_ids will have our words all be converted to IDs,
        # including the roles, in the 3rd column. If we have
        # a word that's not in the vocabulary, then put the unknown word ID (2nd arg in get())
        # These are obtained from the hyperparameter set that was loaded.
        word_to_id = lambda word: self.hp_set.word_vocabulary.get(word, self.hp_set.unk_word_id)
        # Doing both columns will not work, as we will run into the Series hash issue.
        self.dataset_ids['verb'] = self.dataset_ids['verb'].apply(word_to_id)
        self.dataset_ids['noun'] = self.dataset_ids['noun'].apply(word_to_id)
        # self.dataset_ids[['verb', 'noun']] = self.dataset_ids[['verb', 'noun']]\
        #     .apply(lambda word: self.hp_set.word_vocabulary.get(word, self.hp_set.unk_word_id))
        # Convert our role column to role IDs, using the ROLE_MAP and the role vocabulary
        self.dataset_ids['role'] = self.dataset_ids['role'].\
            apply(lambda role: self.hp_set.role_vocabulary[ROLE_MAP[role]])
        # Now we have our inputs in terms of IDs, and we can make the metrics.
        # The important thing is that even though we only have 1 input word (the verb), the
        # order must be preserved. To keep things simple, each input_role sample
        # will be [0, 1, 2, 3, 4, 5]. Using the role vocabulary, we can get what
        # ID the verb maps, and adjust that column.
        input_roles = np.repeat(np.arange(6, dtype=int)[np.newaxis, :], repeats=self.dataset.shape[0], axis=0)
        # First make the input_words be filled with missing words...
        input_words = np.full(shape=(self.dataset.shape[0], 6), fill_value=self.hp_set.missing_word_id, dtype=int)
        # This column will not change
        verb_id = self.hp_set.role_vocabulary['V']
        # Change that column in input_words...
        input_words[:, verb_id] = self.dataset_ids['verb'].values
        # Next, the target role is the role column...needs to have 1 column
        target_role = self.dataset_ids['role'].values[:, np.newaxis]
        # The target word doesn't matter, because we aren't worried about the predicted roles.
        target_word = np.full(shape=(self.dataset.shape[0], 1), fill_value=1729, dtype=int)
        # These are our inputs to the model.
        return {
            'input_roles': input_roles,
            'input_words': input_words,
            'target_role': target_role,
            'target_word': target_word
        }

    def _calc_score(self, outputs):
        word_out = outputs['word_output'].numpy()
        # We want to see what are the word prediction probabilities for the words
        # in the 'word' column and correlate them with the thematic fit scores.
        # For now, we will just append the probabilities as a column, and
        # the report will generate metrics based on missing words.
        self.dataset['noun_probs'] = word_out[np.arange(self.dataset.shape[0]), self.dataset_ids['noun']]

    def _generate_report(self):
        print(self.dataset)
