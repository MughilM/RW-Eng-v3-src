"""
File: event_rep/model_implementation/core/batcher.py
Author: Mughilan Muthupari
Creation Date: 2021-05-15

This module holds our batcher classes, which handle all of the data pipelining from
data reading, transforming, to feeding them into the model. The core functionality revolves
around the tf.Dataset, which can be fed directly model.fit(), and as a result, Tensorflow
can optimize and parallel process the data preparation and model training to save time.
"""

import tensorflow as tf
import numpy as np
from itertools import zip_longest
import jsonlines
import os
import glob
import time
import logging
import sys


class WordRoleWriter:
    def __init__(self, input_data_dir, output_csv_path, batch_size,
                 MISSING_WORD_ID, N_ROLES, N_NEG=0, train_dir='train',
                 val_dir='dev', test_dir='test', chunk_size=250000, overwrite=False):
        # Directories
        self.input_dir = input_data_dir
        self.output_dir = output_csv_path
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        # Other IDs...
        self.MISSING_WORD_ID = MISSING_WORD_ID
        self.N_ROLES = N_ROLES
        self.N_NEG = N_NEG

        # Misc.
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.overwrite = overwrite

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Make the directories in the output path for train, dev, and test
        for directory in [train_dir, val_dir, test_dir]:
            os.makedirs(os.path.join(output_csv_path, directory), exist_ok=True)

    def write_csv_pieces(self):
        """
        Uses the input train, dev, and test files to create multiple CSVs that each
        have chunk_size data rows in them. Creating multiple files, especially with
        a very large amount of data, helps speed up Tensorflow preprocessing.
        :return:
        """
        total_time_start = time.perf_counter()
        for directory in [self.train_dir, self.val_dir, self.test_dir]:
            type_time_start = time.perf_counter()
            # First check if there are files. Don't run the raw processing
            # if we are not supposed to overwrite. Otherwise, warn the user...
            # If there are no files, then by default go ahead with the processing.
            if len(glob.glob(os.path.join(self.output_dir, directory, '*.csv'))) > 0:
                if not self.overwrite:
                    self.logger.info(f'Not overwriting anything in {os.path.join(self.output_dir, directory)}/')
                    continue
                else:
                    self.logger.warning(f'Overwrite is set to true. Files in {os.path.join(self.output_dir, directory)}/'
                                    f'will be deleted!')
            # Currently, files are not in the current format,
            # as keys are integers. Get around it by providing a custom
            # loads, which runs eval() on the line.
            with jsonlines.open(os.path.join(self.input_dir, f'NN_{directory}'), loads=lambda x: eval(x)) as reader:
                for file_num, next_chunk in enumerate(zip_longest(*[reader] * self.chunk_size), 1):
                    file_time_start = time.perf_counter()
                    # We need to "unzip" the dictionaries and group the words
                    # and roles together...
                    # We also can't assume the order of the keys, since it's a
                    # dictionary.
                    # The if statement is necessary because if it's the last
                    # chunk and it's smaller than the chunk size, then
                    # zip_longest pads with Nones..
                    all_roles_words = np.asarray([list(zip(*sample.items())) for sample in next_chunk
                                                  if sample is not None], dtype=int)
                    # The shape is (numOfSamples, 2, N_ROLES)
                    # Sub-index on the second dimension to separate roles and words...
                    roles, words = all_roles_words[:, 0, :], all_roles_words[:, 1, :]
                    full_data = np.concatenate((roles, words), axis=1)
                    # Save as csv using txt
                    np.savetxt(os.path.join(self.output_dir, directory, f'{directory}{file_num}.csv'),
                               full_data,
                               fmt='%u',
                               delimiter=',')
                    file_time_end = time.perf_counter()
                    self.logger.info(f'{directory} {file_num} ==> {full_data.shape} ==>'
                                 f' {file_time_end - file_time_start} SECONDS.')
            type_time_end = time.perf_counter()
            self.logger.info(f'TOTAL TIME FOR {directory} FILES ==> {type_time_end - type_time_start} SECONDS.')
        total_time_end = time.perf_counter()
        self.logger.info(f'TOTAL TIME FOR ALL FILES ==> {total_time_end - total_time_start} SECONDS.')

    def get_csv_np_piece(self, data_type):
        """
        Method used for debugging. Will retrieve the first csv piece and return
        as numpy ndarray. During regular training, please use get_tf_dataset
        :param data_type:
        :return:
        """
        self.logger.info(f'Grabbing the {data_type}1.csv piece.')

        data = np.genfromtxt(os.path.join(self.output_dir, data_type, f'{data_type}1.csv'), delimiter=',', dtype=int)
        roles, words = data[:, :self.N_ROLES], data[:, self.N_ROLES:]
        samples = data.shape[0]
        presentRow = np.sum(words != self.MISSING_WORD_ID, axis=1)
        targets = np.where(words != self.MISSING_WORD_ID)
        target_role = roles[targets].reshape((-1, 1))
        target_word = words[targets].reshape((-1, 1))
        role_output = np.ravel(target_role)
        word_output = np.ravel(target_word)
        elements_to_remove = targets[1]

        words = np.repeat(words, axis=0, repeats=presentRow)
        roles = np.repeat([roles[0]], axis=0, repeats=words.shape[0])

        flattened_indices = self.N_ROLES * np.arange(words.shape[0]) + elements_to_remove
        words = np.delete(words, flattened_indices).reshape((-1, self.N_ROLES - 1))
        roles = np.delete(roles, flattened_indices).reshape((-1, self.N_ROLES - 1))
        return (
            {
                'input_roles': roles,
                'input_words': words,
                'target_role': target_role,
                'target_word': target_word
            },
            {
                'r_out': role_output,
                'w_out': word_output
            }
        )

    def get_tf_dataset(self, data_type):
        """
        Gets a Tensorflow Dataset object based on the data type given.
        :param data_type:
        :return:
        """
        self.logger.info(f'Grabbing the {data_type} CSV files')

        # Helper method to convert the CSV reading into the dataset we need
        def _parse_csv_batch(*args):
            # We get each row as 14 separate arguments, so stack them
            # to create the roles and words for this batch
            roles = tf.stack(args[: self.N_ROLES], axis=1)
            words = tf.stack(args[self.N_ROLES:], axis=1)
            return self._produce_dupes(roles, words)
        # Get the list of csv files and create a regular Dataset out of it...
        files = tf.io.matching_files(os.path.join(self.output_dir, data_type, '*.csv'))
        shards = tf.data.Dataset.from_tensor_slices(files)
        # Interleave the file list, and decode csv on the samples.
        # The decode_csv method is faster when working on batches.
        # Finally, parse the batch and put it into a form for the model...
        dataset = shards.interleave(tf.data.TextLineDataset, cycle_length=5,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Tell TensorFlow that each column is an integer...
        default_data = [int()] * (2 * self.N_ROLES)
        # Batch, map the CSV decoder (which extracts the data),
        # then map our preprocessor, shuffle, and prefetch 3 batches...
        dataset = dataset.batch(self.batch_size). \
            map(lambda x: tf.io.decode_csv(x, record_defaults=default_data),
                num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            map(_parse_csv_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            shuffle(self.batch_size * 2). \
            prefetch(3)
        return dataset

    def _produce_dupes(self, roles, words):
        """
        Internal method to produce the duplicates as required. Written with only
        Tensorflow operations for maximum efficiency.
        :param roles:
        :param words:
        :return:
        """
        # Get the locations where we do not have missing words
        present_pair_indices = tf.where(words != self.MISSING_WORD_ID)
        # gather_nd returns a 1D tensor, which is what the
        # output needs to be. However, our input needs
        # an extra dimension...
        role_target_output = tf.gather_nd(roles, present_pair_indices)
        word_target_output = tf.gather_nd(words, present_pair_indices)
        target_role_input = tf.expand_dims(role_target_output, axis=1)
        target_word_input = tf.expand_dims(word_target_output, axis=1)
        # Count how many dupes we need
        num_dupes = tf.shape(present_pair_indices)[0]
        # Grab the 1st column (rows) and count how many
        # times each row appears...
        rows = tf.gather(present_pair_indices, indices=0, axis=1)
        _, _, counts = tf.unique_with_counts(rows)
        # Repeat each row according to counts...
        roles = tf.repeat(roles, repeats=counts, axis=0)
        words = tf.repeat(words, repeats=counts, axis=0)
        # Create a mask for each row that tells TensorFlow where the
        # words are not missing. Then, using a 2D of Trues with the
        # eventual shape, scatter the mask into it.
        # Finally, mask our roles and words out using the 2D 'scattered'
        # tensor, and reshape with one less column.
        row = tf.range(num_dupes, dtype=tf.int64)
        mask_for_each_row = tf.stack([row, present_pair_indices[:, 1]], axis=1)
        scattered = ~tf.scatter_nd(mask_for_each_row, tf.ones((num_dupes,), dtype=tf.bool),
                                   (num_dupes, self.N_ROLES))
        role_input = tf.reshape(tf.boolean_mask(roles, scattered), (num_dupes, self.N_ROLES - 1))
        word_input = tf.reshape(tf.boolean_mask(words, scattered), (num_dupes, self.N_ROLES - 1))

        # THESE KEYS ARE IMPORTANT!
        # They need to match the input and output layer names of the
        # model we are running.
        return (
            {'input_roles': role_input,
             'input_words': word_input,
             'target_role': target_role_input,
             'target_word': target_word_input},
            {'r_out': role_target_output,
             'w_out': word_target_output}
        )