"""
File: event_rep/evaluation/tasks.py
Author: Mughilan Muthupari
Creation Date: 2021-05-29

This file contains our evaluation tasks. We have a base class that defines the basic
functions, while the rest are subclassed.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os
from typing import Dict, Type
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model_implementation.architecture.models import *
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet
import logging

from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


class EvaluationTask:
    def __init__(self, SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, get_embedding=False):
        # This is needed so the correct model structure is used
        # when loading the model from the checkpoint.
        PARAM_TO_MODEL: Dict[str, Type[MTRFv4Res]] = {
            'v4': MTRFv4Res,
            'v5': MTRFv5Res,
            'v6': MTRFv6Res,
            'v8': MTRFv8Res,
            'v9': MTRFv9Res
        }
        self.SRC_DIR = SRC_DIR
        self.EXPERIMENT_DIR = EXPERIMENT_DIR
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.lemmatizer = WordNetLemmatizer()
        # Because this is a subclassed model we can't load the model all together,
        # due to unstable serialization for subclassed models. Instead, we can use
        # the checkpoints to load the weights. On the flip side, we have to make the
        # correct model structure, using the hyperparameters. This is why the model_name
        # is needed. The hyperparameters are saved in the experiment directory.
        self.hp_set = HyperparameterSet(os.path.join(self.EXPERIMENT_DIR, self.experiment_name, 'hyperparameters.json'))
        # Create the model layers using the hyperparameters
        if self.model_name == 'v9':
            model_obj = MTRFv9Res(self.hp_set, os.path.join(self.EXPERIMENT_DIR, self.experiment_name)[:-3])
        else:
            model_obj: MTRFv4Res = PARAM_TO_MODEL[self.model_name](self.hp_set)
        # Now, we build with training=False, and if we need the embeddings. At this point,
        # we don't have weights...
        self.model = model_obj.build_model(training=False, get_embedding=get_embedding)
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # NOW WE LOAD THE WEIGHTS. Now, since we don't have bias, we will at least
        # get a warning, saying that a layer's weights was not loaded. THIS IS FINE.
        checkpoint_dir = os.path.join(SRC_DIR, EXPERIMENT_DIR, experiment_name, 'checkpoints')
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        # We won't use all the weights (bias or half the model will be missing),
        # so do expect_partial() to remove the warnings...
        self.model.load_weights(latest_checkpoint).expect_partial()
        logger.info(f'Loaded model and hyperparameters from {os.path.join(self.EXPERIMENT_DIR, self.experiment_name)}')
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
        outputs = self.model(all_inputs, training=False)  # This is a different training argument (changes behaviour
                                                          # of certain layers, like BatchNormalization)
        self._calc_score(outputs)
        self._generate_report()


class CorrelateTFScores(EvaluationTask):
    """
    This class handles all the evaluation tasks where we are given a verb, noun, the role the
    noun is supposed to fill, and a thematic fit judgement score. This data should be in
    csv format with columns as verb,noun,role,score.
    """
    def __init__(self, SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, dataset_name: str, get_embedding=False):
        super().__init__(SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, get_embedding=get_embedding)
        self.dataset_name = dataset_name
        self.dataset_input_file = os.path.join(self.SRC_DIR, f'evaluation/task_data/{dataset_name}.csv')
        self.dataset = pd.read_csv(self.dataset_input_file)
        # Lemmatize the words...
        self.dataset['verb'] = self.dataset['verb'].apply(lambda w: self.lemmatizer.lemmatize(w, pos='v'))
        self.dataset['noun'] = self.dataset['noun'].apply(lambda w: self.lemmatizer.lemmatize(w, pos='n'))
        self.dataset_ids = self.dataset.copy()

    def _preprocess(self) -> Dict[str, np.ndarray]:
        # dataset_ids will have our words all be converted to IDs,
        # including the roles, in the 3rd column. If we have
        # a word that's not in the vocabulary, then put the unknown word ID (2nd arg in get())
        # These are obtained from the hyperparameter set that was loaded.
        word_to_id = lambda word: self.hp_set.word_vocabulary.get(word, self.hp_set.unk_word_id)
        # Doing both columns will not work, as we will run into the Series hash issue.
        self.dataset_ids['verb'] = self.dataset_ids['verb'].apply(word_to_id)
        self.dataset_ids['noun'] = self.dataset_ids['noun'].apply(word_to_id)
        # Convert our role column to role IDs.
        # If the model was NOT TRAINED ON the A2 beneficiary role, then we must replace
        # A2 with <OTHER>, so the mapping doesn't error out on encounter.
        self.dataset_ids['role'] = self.dataset_ids['role'].\
            apply(lambda role: self.hp_set.role_vocabulary.get(role, self.hp_set.role_vocabulary['<OTHER>']))
        # Now we have our inputs in terms of IDs, and we can make the metrics.
        # The important thing is that we need to emulate how we trained the model. There,
        # given N roles, each sample had N - 1 input roles, with one removed for the target role.
        # In correlation, our target role is given by the role column. Therefore, this role MUST
        # NOT BE PRESENT in the input roles row.
        input_roles = np.repeat(np.arange(self.hp_set.role_vocab_count, dtype=int)[np.newaxis, :],
                                repeats=self.dataset.shape[0], axis=0)
        # From the full vocab set, figure out which ones to remove based on our 'role' column values
        target_role_indices = self.hp_set.role_vocab_count * np.arange(input_roles.shape[0]) + self.dataset_ids['role']
        # After dropping, the matrix flattens, reshape it...
        input_roles = np.delete(input_roles, target_role_indices).reshape((-1, self.hp_set.role_vocab_count - 1))
        # First make the input_words be filled with missing words... (missing means a role does not have a word)
        input_words = np.full(shape=(self.dataset.shape[0], 6), fill_value=self.hp_set.missing_word_id, dtype=int)
        # Get the location in each row which corresponds to V (because one role is missing in each row,
        # the column will change in each row) and set them to the verb word IDs.
        input_words[np.where(input_roles == self.hp_set.role_vocabulary['V'])] = self.dataset_ids['verb'].values
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
        word_out = outputs['w_out'].numpy()
        # We want to see what are the word prediction probabilities for the words
        # in the 'word' column and correlate them with the thematic fit scores.
        # For now, we will just append the probabilities as a column, and
        # the report will generate metrics based on missing words.
        self.dataset['noun_probs'] = word_out[np.arange(self.dataset.shape[0]), self.dataset_ids['noun']]
        # We will save the unknown words, and calculate a correlation for the whole
        # dataset as well as a correlation with the unknown words removed.
        unknown_verbs = self.dataset['verb'][self.dataset_ids['verb'] == self.hp_set.unk_word_id]
        unknown_nouns = self.dataset['noun'][self.dataset_ids['noun'] == self.hp_set.unk_word_id]
        self.metrics['unknown_verbs'] = np.unique(unknown_verbs.values)
        self.metrics['unknown_nouns'] = np.unique(unknown_nouns.values)
        # Combine the two indices so we can easily remove from the main df in one line
        total_unknown_index = unknown_verbs.index.union(unknown_nouns.index)
        self.metrics['unknown_n'] = len(total_unknown_index)
        # Create a dataframe with the unknown verbs + nouns dropped
        self.dataset_dropped = self.dataset.drop(index=total_unknown_index)
        rho, p = spearmanr(self.dataset['score'], self.dataset['noun_probs'])
        self.metrics['rho'] = rho
        self.metrics['p'] = p
        # Calculate again with the dropped dataframe
        rho, p = spearmanr(self.dataset_dropped['score'], self.dataset_dropped['noun_probs'])
        self.metrics['rho_m'] = rho
        self.metrics['p_m'] = p

    def _generate_report(self):
        os.makedirs(os.path.join(self.EXPERIMENT_DIR, self.experiment_name, 'evaluation_results'), exist_ok=True)
        # Short helper method to draw a box around a piece of text given padding...
        make_box = lambda text, spaces: f"╔{'=' * (len(text) + 2 * spaces)}╗\n║{' ' * spaces}{text}{' ' * spaces}║\n" \
                                        f"╚{'=' * (len(text) + 2 * spaces)}╝\n"
        # Make the header
        header = make_box(f'{self.dataset_name.upper()} RESULTS', 12) + '\n'

        # Unknown word label...
        uk_report = make_box('UNKNOWN WORD REPORT', 4)
        uk_report += f"There are {len(self.metrics['unknown_verbs'])} unknown verbs and " \
                   f"{len(self.metrics['unknown_nouns'])} unknown nouns for a total of " \
                   f"{self.metrics['unknown_n']} unknown samples.\n\n"
        # Apparently I can't put backslashes in f-strings :(
        if len(self.metrics['unknown_verbs']) > 0:
            uk_report += f"UNKNOWN VERBS\n{'=' * len(max(self.metrics['unknown_verbs'], key=lambda x: len(x)))}\n"
            uk_report += '\n'.join(self.metrics['unknown_verbs']) + '\n'
        if len(self.metrics['unknown_nouns']) > 0:
            uk_report += f"UNKNOWN NOUNS\n{'=' * len(max(self.metrics['unknown_nouns'], key=lambda x: len(x)))}\n"
            uk_report += '\n'.join(self.metrics['unknown_nouns']) + '\n\n'

        # Final report on correlations...
        fin_report = make_box('FINAL SPEARMAN CORRELATIONS', 6) + '\n'
        # Multiply correlation by 100, round to 3 decimal places, round p-value to 6 decimal places
        # Show number of samples in entire dataset and with missing samples removed.
        fin_report += f"ENTIRE DATASET ({self.dataset.shape[0]}): {self.metrics['rho'] * 100:.3f}%, " \
                      f"P-value of {self.metrics['p']:.6f}\n"
        fin_report += f"UNKNOWN REMOVED ({self.dataset_dropped.shape[0]}): {self.metrics['rho_m'] * 100:.3f}%, " \
                      f"P-value of {self.metrics['p_m']:.6f}"
        output_file_path = os.path.join(self.EXPERIMENT_DIR, self.experiment_name,
                                        'evaluation_results', f'{self.dataset_name}.txt')
        with open(output_file_path,'w', encoding='utf-8') as f:
            f.write(header)
            f.write(uk_report)
            f.write(fin_report)
        logger.info(f"Evaluation finished! Results saved to {output_file_path}")


class BicknellTask(EvaluationTask):
    def __init__(self, SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, get_embedding=False):
        super().__init__(SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, get_embedding=get_embedding)
        self.dataset_name = 'bicknell'
        self.dataset_input_file = os.path.join(SRC_DIR, f'evaluation/task_data/{self.dataset_name}.csv')
        self.dataset = pd.read_csv(self.dataset_input_file)
        # Lemmatize the verb and two patient columns. The patients are nouns
        self.dataset['verb'] = self.dataset['verb'].apply(lambda w: self.lemmatizer.lemmatize(w, pos='v'))
        self.dataset['cong_patient'] = self.dataset['cong_patient'].apply(lambda w: self.lemmatizer.lemmatize(w, pos='n'))
        self.dataset['incong_patient'] = self.dataset['incong_patient'].apply(lambda w: self.lemmatizer.lemmatize(w, pos='n'))
        self.dataset_ids = self.dataset.copy()

    def _preprocess(self) -> Dict[str, np.ndarray]:
        """
        We have 4 columns in the bicknell.csv. The input roles are A0 and V,
        while the output role is A1.
        :return:
        """
        # Convert the 4 columns of words to integers. Unfortunately, doing it
        # all at once leads to error, which is unusual...
        word_to_id = lambda word: self.hp_set.word_vocabulary.get(word, self.hp_set.unk_word_id)
        for col in self.dataset.columns:
            self.dataset_ids[col] = self.dataset_ids[col].apply(word_to_id)
        # With the input word IDs on hand, let us create the input role IDs.
        # The important thing here is that even though there is data for only 2 input roles,
        # the order must be preserved. To see which ID maps to which role, we use
        # the role vocabulary. The input roles will still be 0-5, except the corresponding
        # columns in the input word matrix will be adjusted accordingly.
        input_roles = np.repeat(np.arange(6, dtype=int)[np.newaxis, :], repeats=self.dataset.shape[0], axis=0)
        # Initially fill the input words with missing words
        input_words = np.full(shape=(self.dataset.shape[0], 6), fill_value=self.hp_set.missing_word_id, dtype=int)
        # Get the agent and verb IDs from the role vocabulary so we know what column to change...
        agent_id = self.hp_set.role_vocabulary['A0']
        verb_id = self.hp_set.role_vocabulary['V']
        # Change the correct columns in input_words...
        input_words[:, agent_id] = self.dataset_ids['agent'].values
        input_words[:, verb_id] = self.dataset_ids['verb'].values
        # Patient is the target role, corresponds to A1...
        target_role = np.repeat([[self.hp_set.role_vocabulary['A1']]], repeats=self.dataset.shape[0], axis=0)
        # target word doesn't matter, since we are not interested in the predicted roles...
        target_word = np.full(shape=(self.dataset.shape[0], 1), fill_value=1729, dtype=int)
        # These are our inputs to the model.
        return {
            'input_roles': input_roles,
            'input_words': input_words,
            'target_role': target_role,
            'target_word': target_word
        }

    def _calc_score(self, outputs: dict):
        """
        For Bicknell, we compare the probabilities of the congruent and
        incongruent patients.
        :param outputs:
        :return:
        """
        word_out = outputs['w_out'].numpy()
        # Index the congruent and incongruent patients
        self.dataset['cong_prob'] = word_out[np.arange(self.dataset.shape[0]), self.dataset_ids['cong_patient']]
        self.dataset['incong_prob'] = word_out[np.arange(self.dataset.shape[0]), self.dataset_ids['incong_patient']]
        # Find the missing words...Use dataset id columns because we just added 2 cols to dataset
        # Don't do unique yet, because we need the index.
        for col in self.dataset_ids.columns:
            self.metrics[f'unknown_{col}s'] = self.dataset[col][self.dataset_ids[col] == self.hp_set.unk_word_id]
        # Union the index of the 4 columns to find the total missing #
        # and unique the missing words...
        total_unknown_index = self.metrics['unknown_agents'].index
        self.metrics['unknown_agents'] = np.unique(self.metrics['unknown_agents'].values)
        for col in ['verb', 'cong_patient', 'incong_patient']:
            total_unknown_index = total_unknown_index.union(self.metrics[f'unknown_{col}s'].index)
            self.metrics[f'unknown_{col}s'] = np.unique(self.metrics[f'unknown_{col}s'].values)
        self.metrics['unknown_n'] = len(total_unknown_index)
        # Create a dataset which drops the missing values. Calculate
        # accuracy on this one and the full dataset...
        self.dataset_dropped = self.dataset.drop(index=total_unknown_index)
        self.metrics['num_correct_m'] = self.dataset_dropped.loc[self.dataset_dropped['cong_prob'] >
                                                                 self.dataset_dropped['incong_prob'], :].shape[0]
        self.metrics['num_incorrect_m'] = self.dataset_dropped.shape[0] - self.metrics['num_correct_m']
        self.metrics['accuracy_m'] = self.metrics['num_correct_m'] / self.dataset_dropped.shape[0]
        # Now for the full dataset...
        self.metrics['num_correct'] = self.dataset.loc[self.dataset['cong_prob'] >
                                                                 self.dataset['incong_prob'], :].shape[0]
        self.metrics['num_incorrect'] = self.dataset.shape[0] - self.metrics['num_correct_m']
        self.metrics['accuracy'] = self.metrics['num_correct_m'] / self.dataset.shape[0]

    def _generate_report(self):
        os.makedirs(os.path.join(self.EXPERIMENT_DIR, self.experiment_name, 'evaluation_results'), exist_ok=True)
        # Short helper method to draw a box around a piece of text given padding...
        make_box = lambda text, spaces: f"╔{'=' * (len(text) + 2 * spaces)}╗\n║{' ' * spaces}{text}{' ' * spaces}║\n" \
                                        f"╚{'=' * (len(text) + 2 * spaces)}╝\n"
        # Make the header
        header = make_box(f'{self.dataset_name.upper()} RESULTS', 12) + '\n'

        # Unknown word label...
        uk_report = make_box('UNKNOWN WORD REPORT', 4)
        uk_report += f'The breakdown of unknown words are as follows:\n'
        for col in self.dataset_ids.columns:
            uk_report += f"  - {len(self.metrics[f'unknown_{col}s'])} {col}s\n"
        uk_report += f"This contributes to a total of {self.metrics['unknown_n']} unknown samples.\n\n"
        # Now for the actual words that are missing...
        for col in self.dataset_ids.columns:
            if len(self.metrics[f'unknown_{col}s']) > 0:
                uk_report += f"UNKNOWN {col.upper()}S\n" \
                             f"{'=' * len(max(self.metrics[f'unknown_{col}s'], key=lambda x: len(x)))}\n"
                uk_report += '\n'.join(self.metrics[f'unknown_{col}s']) + '\n\n'

        # Final metrics reprot at the end
        fin_report = make_box('FINAL ACCURACY METRICS', 6) + '\n'
        # Report accuracy as correct / total and as percentage
        fin_report += f"ENTIRE DATASET ({self.dataset.shape[0]}): {self.metrics['num_correct']} / " \
                      f"{self.dataset.shape[0]} = {self.metrics['accuracy'] * 100:.3f}%\n"
        fin_report += f"UNKNOWN REMOVED ({self.dataset_dropped.shape[0]}): {self.metrics['num_correct_m']} / " \
                      f"{self.dataset_dropped.shape[0]} = {self.metrics['accuracy_m'] * 100:.3f}%"
        # Write to the file
        output_file_path = os.path.join(self.EXPERIMENT_DIR, self.experiment_name,
                                        'evaluation_results', f'{self.dataset_name}.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(uk_report)
            f.write(fin_report)
        logger.info(f'Bicknell evaluation finished. Results saved to {output_file_path}.')


class GS2013Task(EvaluationTask):
    def __init__(self, SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, get_embedding=False):
        super().__init__(SRC_DIR, EXPERIMENT_DIR, model_name, experiment_name, get_embedding=get_embedding)
        self.gs = pd.read_csv(os.path.join(SRC_DIR, 'evaluation/task_data/GS2013.csv'))
        # Lemmatize...
        for noun_col in ['subject', 'object']:
            self.gs[noun_col] = self.gs[noun_col].apply(lambda w: self.lemmatizer.lemmatize(w, pos='n'))
        for verb_col in ['base_verb', 'landmark_verb']:
            self.gs[verb_col] = self.gs[verb_col].apply(lambda w: self.lemmatizer.lemmatize(w, pos='v'))
        # Don't copy it yet, because we need some preprocessing on the original data...
        self.gs_ids = None
        self.word_cols = ['subject', 'base_verb', 'object', 'landmark_verb']

    def _preprocess(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Because we have two inputs for each sample in GS
        (the base verb, and landmark verb). we will return
        as two separate inputs we pass into the model...
        :return: A dictionary with 'base_verb_input' and 'landmark_verb_input'
        that is regular input for the base verbs and landmark verbs.
        """
        # First thing, we have data from 50 participants. Average the
        # scores for all 50 for each sample...
        # Group the data by the words...
        grouped = self.gs.groupby(by=self.word_cols + ['hilo'])
        # Call mean(), and reset the index so the groups turn into the index like before
        self.gs = grouped.mean().reset_index()
        # We don't need the id column anymore...
        self.gs.drop(columns=['id'], inplace=True)
        self.gs_ids = self.gs.copy()
        # Now convert the words to IDs...
        word_to_id = lambda word: self.hp_set.word_vocabulary.get(word, self.hp_set.unk_word_id)
        for col in self.word_cols:
            self.gs_ids[col] = self.gs_ids[col].apply(word_to_id)
        # Now to actually create the input arrays...The input words will be the only
        # thing that differs between the two sets of input...
        input_roles = np.repeat(np.arange(6, dtype=int)[np.newaxis, :], repeats=self.gs.shape[0], axis=0)
        base_verb_words = np.full(shape=(self.gs.shape[0], 6), fill_value=self.hp_set.missing_word_id, dtype=int)
        # Get the corresponding columns. We use the A0, V, and A1 as the roles...
        subject_id = self.hp_set.role_vocabulary['A0']
        verb_id = self.hp_set.role_vocabulary['V']
        object_id = self.hp_set.role_vocabulary['A1']
        # Set the values in the base_verb_words to their corresponding values. Copy and set the
        # landmark verbs...
        base_verb_words[:, subject_id] = self.gs_ids['subject'].values
        base_verb_words[:, verb_id] = self.gs_ids['base_verb'].values
        base_verb_words[:, object_id] = self.gs_ids['object'].values

        landmark_verb_words = base_verb_words.copy()
        landmark_verb_words[:, verb_id] = self.gs_ids['landmark_verb'].values

        # The target word and role can be whatever, since we are not predicting
        # anything, jsut looking at the embeddings...
        target_role = np.full(shape=(self.gs.shape[0], 1), fill_value=0, dtype=int)
        target_word = np.full(shape=(self.gs.shape[0], 1), fill_value=1729, dtype=int)

        return {
            'base_verb_input': {
                'input_roles': input_roles,
                'input_words': base_verb_words,
                'target_role': target_role,
                'target_word': target_word
            },
            'landmark_verb_input': {
                'input_roles': input_roles,
                'input_words': landmark_verb_words,
                'target_role': target_role,
                'target_word': target_word
            }
        }

    def _calc_score(self, base_out, land_out):
        """
        Given the base verb context, and landmark context, we run
        a cosine similarity between all the context embedding pairs.
        We can then update the dataset with this info...
        :param base_out: The context embedding associated with the base verbs
        :param land_out: The context embedding associated with the landmark verbs
        :return:
        """
        # Cosine similarity is just cosine of the angle between the vectors
        # i.e. dot product divided by product of magnitudes..
        # For vectorized row-wise dot product, use einsum...
        dot_products = np.einsum('ij,ij->i', base_out, land_out)
        mag_products = np.linalg.norm(base_out, axis=1) * np.linalg.norm(land_out, axis=1)
        cosine_similarities = dot_products / mag_products
        # Order was preserved when we converted to numpy arrays, so just append the column...
        self.gs['cosine'] = cosine_similarities

        # Unknown words... (subjects, base verbs, objects, landmark verbs)
        for col in self.word_cols:
            self.metrics[f'unknown_{col}s'] = self.gs[col][self.gs_ids[col] == self.hp_set.unk_word_id]
        # Union the index of the 4 columns to find the total unknown #
        # and unique the unknown words...
        total_unknown_index = self.metrics['unknown_subjects'].index
        self.metrics['unknown_subjects'] = np.unique(self.metrics['unknown_subjects'].values)
        for col in ['base_verb', 'object', 'landmark_verb']:
            total_unknown_index = total_unknown_index.union(self.metrics[f'unknown_{col}s'].index)
            self.metrics[f'unknown_{col}s'] = np.unique(self.metrics[f'unknown_{col}s'].values)
        self.metrics['unknown_n'] = len(total_unknown_index)

        self.gs_dropped = self.gs.drop(index=total_unknown_index)
        # Put metrics for dropped dataset...
        rho, p = spearmanr(self.gs_dropped['score'], self.gs_dropped['cosine'])
        self.metrics['rho_m'] = rho
        self.metrics['p_m'] = p
        # Also correlate with LOW and HIGH samples...
        lows = self.gs_dropped[self.gs_dropped['hilo'] == 'LOW']
        highs = self.gs_dropped[self.gs_dropped['hilo'] == 'HIGH']
        rho, p = spearmanr(lows['score'], lows['cosine'])
        self.metrics['low_rho_m'] = rho
        self.metrics['low_p_m'] = p
        rho, p = spearmanr(highs['score'], highs['cosine'])
        self.metrics['high_rho_m'] = rho
        self.metrics['high_p_m'] = p

        # We correlate the scores with the similariites
        rho, p = spearmanr(self.gs['score'], self.gs['cosine'])
        self.metrics['rho'] = rho
        self.metrics['p'] = p
        # Also correlate with LOW and HIGH samples...
        lows = self.gs[self.gs['hilo'] == 'LOW']
        highs = self.gs[self.gs['hilo'] == 'HIGH']
        rho, p = spearmanr(lows['score'], lows['cosine'])
        self.metrics['low_rho'] = rho
        self.metrics['low_p'] = p
        rho, p = spearmanr(highs['score'], highs['cosine'])
        self.metrics['high_rho'] = rho
        self.metrics['high_p'] = p

    def _generate_report(self):
        """
        Nothing different here, just the normal report...
        :return:
        """
        os.makedirs(os.path.join(self.EXPERIMENT_DIR, self.experiment_name, 'evaluation_results'), exist_ok=True)
        # Short helper method to draw a box around a piece of text given padding...
        make_box = lambda text, spaces: f"╔{'=' * (len(text) + 2 * spaces)}╗\n║{' ' * spaces}{text}{' ' * spaces}║\n" \
                                        f"╚{'=' * (len(text) + 2 * spaces)}╝\n"
        # Make the header
        header = make_box('GS RESULTS', 12) + '\n'

        # Unknown word label...
        uk_report = make_box('UNKNOWN WORD REPORT', 4)
        uk_report += f'The breakdown of unknown words are as follows:\n'
        for col in self.word_cols:
            uk_report += f"  - {len(self.metrics[f'unknown_{col}s'])} {col}s\n"
        uk_report += f"This contributes to a total of {self.metrics['unknown_n']} unknown samples.\n\n"
        # Now for the actual words that are unknown...
        for col in self.word_cols:
            if len(self.metrics[f'unknown_{col}s']) > 0:
                uk_report += f"UNKNOWN {col.upper()}S\n" \
                             f"{'=' * len(max(self.metrics[f'unknown_{col}s'], key=lambda x: len(x)))}\n"
                uk_report += '\n'.join(self.metrics[f'unknown_{col}s']) + '\n\n'

        # Report spearman correlation total, low, and high for both full dataset
        # and missing data...
        fin_report = make_box('FINAL SPEARMAN CORRELATIONS', 6) + '\n'
        fin_report += f"ENTIRE DATASET ({self.gs.shape[0]}): {self.metrics['rho'] * 100:.3f}%, " \
                      f"P-value of {self.metrics['p']:.6f}\n"
        fin_report += f"  - LOW DATASET ({self.gs[self.gs['hilo'] == 'LOW'].shape[0]}): " \
                      f"{self.metrics['low_rho'] * 100:.3f}%, P-value of {self.metrics['low_p']:.6f}\n"
        fin_report += f"  - HIGH DATASET ({self.gs[self.gs['hilo'] == 'HIGH'].shape[0]}):" \
                      f" {self.metrics['high_rho'] * 100:.3f}%, P-value of {self.metrics['high_p']:.6f}\n\n"
        fin_report += f"UNKNOWN REMOVED ({self.gs_dropped.shape[0]}): {self.metrics['rho_m'] * 100:.3f}%, " \
                      f"P-value of {self.metrics['p_m']:.6f}\n"
        fin_report += f"  - LOW DATASET ({self.gs_dropped[self.gs_dropped['hilo'] == 'LOW'].shape[0]}): " \
                      f"{self.metrics['low_rho_m'] * 100:.3f}%, P-value of {self.metrics['low_p_m']:.6f}\n"
        fin_report += f"  - HIGH DATASET ({self.gs_dropped[self.gs_dropped['hilo'] == 'HIGH'].shape[0]}): " \
                      f"{self.metrics['high_rho_m'] * 100:.3f}%, P-value of {self.metrics['high_p_m']:.6f}\n"
        output_file_path = os.path.join(self.EXPERIMENT_DIR, self.experiment_name, 'evaluation_results', 'gs.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(uk_report)
            f.write(fin_report)
        logger.info(f'GS2013 evaluation finished. Results saved to {output_file_path}.')

    def run_task(self):
        """
        We need to do a slight override the run_task because we need to pass both
        the base and landmark context
        :return:
        """
        all_inputs = self._preprocess()
        base_context = self.model(all_inputs['base_verb_input'], training=False)['context_embedding']
        landmark_context = self.model(all_inputs['landmark_verb_input'], training=False)['context_embedding']
        self._calc_score(base_context, landmark_context)
        self._generate_report()
