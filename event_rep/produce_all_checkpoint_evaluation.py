"""
File: event_rep/produce_all_checkpoint_evaluation.py
Author: Mughilan Muthupari
Creation Date: 2022-01-14

This file is simple, it will take ALL the checkpoints produced in a single
experiment and do the evaluation tasks on each one and save it to a csv.
This is to see the variance in the scores across all the checkpoints, because
for training, we see if there is an improvement in the validation loss, not the
evaluation task scores. Only needed arguments are the model type, data version,
and experiment name.
"""

import os.path
import logging
import sys
from typing import Dict, Type
import datetime
import argparse
import glob
import re

import numpy as np
import pandas as pd
import nltk

from tensorflow.keras.optimizers import Adam

from model_implementation.architecture.hp.hyperparameters import HyperparameterSet
from model_implementation.core.batcher import WordRoleWriter
from model_implementation.architecture.models import *
from evaluation.tasks import CorrelateTFScores, BicknellTask, GS2013Task

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Directory locations
# The absolute path where main is being run. Should end in RW-Eng-v3-src/event_rep
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory where experiments are saved
EXPERIMENT_DIR = os.path.join(SRC_DIR, 'experiments')
# Pretrained embeddings
PRETRAINED_DIR = os.path.join(SRC_DIR, 'pretrained_embeddings')
# Directory where raw processed CSVs go when we need to split up a large file
# for efficient tensorflow reading.
CSV_PIECE_PATH = os.path.join(SRC_DIR, 'csv_piece_output')
# Primary data path for data...
DATA_PATH = os.path.join(SRC_DIR, 'processed_data')
# This is a dictionary which maps the user-provided parameter of the model to
# the corresponding class
# TODO: Add models here as necessary, as the argument enforcement is on the keys
PARAM_TO_MODEL: Dict[str, Type[MTRFv4Res]] = {
    'v4': MTRFv4Res,
    'v5': MTRFv5Res,
    'v6': MTRFv6Res,
    'v8': MTRFv8Res
}

# The list of all evaluation tasks
# ALL_EVAL_TASKS = ['pado', 'mcrae', 'greenberg', 'bicknell', 'gs', 'ferretti_instrument', 'ferretti_location']
ALL_EVAL_TASKS = ['ferretti_instrument', 'ferretti_location', 'greenberg', 'mcrae', 'pado', 'bicknell', 'gs']

# Download wordnet packages from NLTK
nltk.download('wordnet')

# Start the clock, this allows all the methods to access
# the time...
total_time_start = datetime.datetime.now(datetime.timezone.utc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints')

    # Required parameters
    parser.add_argument('model_name', choices=PARAM_TO_MODEL.keys(), type=str,
                        help='The name of the model to train. Must be one of the model classes'
                             'defined in models.py')
    parser.add_argument('data_version', type=str,
                        help='The folder in the data directory that contains NN_train, NN_test, etc.')
    parser.add_argument('experiment_name', type=str,
                        help='The name of the experiment. ALl model artifacts get saved in a subdirectory of'
                             'this name. Will default to a concatenation of the time, model name, and data version.')

    args = parser.parse_args()
    # Check for existence...
    if not os.path.exists(os.path.join(DATA_PATH, args.data_version)):
        parser.error(f'Invalid data version: {os.path.join(DATA_PATH, args.data_version)} does not exist.')
        sys.exit(1)
    if not os.path.exists(os.path.join(EXPERIMENT_DIR, args.experiment_name)):
        parser.error(f'Invalid experiment name: {os.path.join(EXPERIMENT_DIR, args.experiment_name)} does note exist.')

    logger.info(f'Evaluation tasks to start on experiment {args.experiment_name}')

    # Get the list of checkpoints in the directory
    experiment_name_dir = os.path.join(EXPERIMENT_DIR, args.experiment_name)
    check_files = glob.glob(os.path.join(experiment_name_dir, 'checkpoints', '*.ckpt.index'))
    logger.info(f'Checkpoint Files - {check_files}')
    # EXtract the checkpoint numbers from them
    checkpoint_nums = [int(re.search(r'(\d+)', os.path.basename(check))[0]) for check in check_files]
    # Sort them, so that the resulting csv is easy to read (won't affect actual evaluation)
    checkpoint_nums = sorted(checkpoint_nums)
    logger.info(f'Checkpoint Numbers - {checkpoint_nums}')

    # Get the dataset as well
    data_writer = WordRoleWriter(input_data_dir=os.path.join(DATA_PATH, args.data_version),
                                 output_csv_path=os.path.join(CSV_PIECE_PATH, args.data_version),
                                 batch_size=512,
                                 MISSING_WORD_ID=50001,
                                 N_ROLES=7)
    data_writer.write_csv_pieces()
    test_data = data_writer.get_tf_dataset('test')

    # For each checkpoint number, run the evaluation tasks, BUT DON'T SAVE THEM
    # to the directory.
    data = []
    for check_num in checkpoint_nums:
        results = [check_num]
        logger.info(f'Running evaluation for checkpoint {check_num}...')
        hp_set = HyperparameterSet(os.path.join(EXPERIMENT_DIR, args.experiment_name, 'hyperparameters.json'))
        checkpoint_dir = os.path.join(SRC_DIR, EXPERIMENT_DIR, args.experiment_name, 'checkpoints')
        model_obj = MTRFv5Res(hp_set)
        model = model_obj.build_model()
        model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(os.path.join(checkpoint_dir, f'cp_{check_num:03}.ckpt'))
        # Evaluate our test data on this epoch...
        test_metrics = model.evaluate(test_data)
        test_keys = ['Loss', 'Role Loss', 'Word Loss', 'Role Accuracy', 'Word Accuracy']
        test_metrics_json = {k: v for k, v in zip(test_keys, test_metrics)}
        results.extend([test_metrics_json['Role Accuracy'], test_metrics_json['Word Accuracy']])
        for task in ALL_EVAL_TASKS:
            logger.info(f'Running {task}')
            if task == 'bicknell':
                evaluator = BicknellTask(SRC_DIR, EXPERIMENT_DIR, args.model_name, args.experiment_name,
                                         checkpoint_epoch=check_num, write_report=False)
            elif task == 'gs':
                evaluator = GS2013Task(SRC_DIR, EXPERIMENT_DIR, args.model_name, args.experiment_name,
                                       get_embedding=True, checkpoint_epoch=check_num, write_report=False)
            else:
                evaluator = CorrelateTFScores(SRC_DIR, EXPERIMENT_DIR, args.model_name, args.experiment_name,
                                              task, checkpoint_epoch=check_num, write_report=False)
            evaluator.run_task()
            # Now, since write report is False, extract the metrics from the dictionary
            if task == 'bicknell':
                logger.info(f'BICKNELL (epoch {check_num}) - {evaluator.metrics["accuracy"] * 100:.3f}%')
                results.append(evaluator.metrics['accuracy'])
            elif task == 'gs':
                logger.info(f'GS2013 (Full, Low, High) (epoch {check_num} - {evaluator.metrics["rho"] * 100:.3f}%,'
                            f'{evaluator.metrics["low_rho"] * 100:.3f}%, {evaluator.metrics["high_rho"] * 100:.3f}%')
                # Add the full, low, and high
                results.extend([evaluator.metrics['rho'], evaluator.metrics['low_rho'], evaluator.metrics['high_rho']])
            else:
                logger.info(f'{task.upper()} (epoch {check_num}) - {evaluator.metrics["rho"] * 100:.3f}%')
                results.append(evaluator.metrics['rho'])
        data.append(results)

    # Create a data frame
    evaluation_df = pd.DataFrame(data, columns=['Checkpoint Epoch', 'Role Acc', 'Word Acc'] + ALL_EVAL_TASKS + ['low_gs', 'high_gs'])
    evaluation_df.set_index(['Checkpoint Epoch'], inplace=True)
    # Save as csv in the experiment directory
    evaluation_df.to_csv(os.path.join(EXPERIMENT_DIR, args.experiment_name, 'all_eval_results.csv'))

