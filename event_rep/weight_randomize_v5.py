"""
File: event_rep/weight_randomize_v5.py
Author: Mughilan Muthupari
Creation Date: 2021-12-31

This is a one-off file to perform the experiment where we read in
a v5 model under GloVe tuned embeddings, and run two evaluation experiments.
  1. We first create a new model where only the embeddings have been passed over,
     and the rest of the layers have been randomly initialized while also mimicking
     the weight distribution in each layer. We then run the evaluation tasks to see
     the differences in performance.
  2. Repeat the experiment, only this time, randomly initialize the embeddings, while
     keeping the other layers the same.

The reason this file is one-off is that the method by which the weight transfer
happens means we need to access to each layer's variable. This differs from model to model.
Ideally, each model class would have a method which creates these two extra models.
"""
import inspect

import numpy as np
import pandas as pd
import os
import sys
import logging
import pprint

from model_implementation.architecture.models import MTRFv5Res
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet
from evaluation.tasks import CorrelateTFScores, BicknellTask, GS2013Task
from model_implementation.core.batcher import WordRoleWriter

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Dense

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

# The experiment with the model we want to grab from
EXPERIMENT_NAME = '3a_plus_03_v5_tunespacyavg_1perc'

hp_set = HyperparameterSet(os.path.join(EXPERIMENT_DIR, EXPERIMENT_NAME, 'hyperparameters.json'))


def create_dataset_objects(data_ver):
    data_writer = WordRoleWriter(input_data_dir=os.path.join(DATA_PATH, data_ver),
                                 output_csv_path=os.path.join(CSV_PIECE_PATH, data_ver),
                                 batch_size=hp_set.batch_size,
                                 MISSING_WORD_ID=hp_set.missing_word_id,
                                 N_ROLES=hp_set.role_vocab_count)
    data_writer.write_csv_pieces()
    train_data = data_writer.get_tf_dataset('train')
    vali_data = data_writer.get_tf_dataset('dev')
    test_data = data_writer.get_tf_dataset('test')

    return train_data, vali_data, test_data


def randomize_network(experiment_name, randomize_embedding=False):
    # Generate the huperparameter set from the SRC DIR
    # hp_set = HyperparameterSet(os.path.join(EXPERIMENT_DIR, experiment_name, 'hyperparameters.json'))

    # Create the model, and pull the weights from the latest checkpoint
    model_obj = MTRFv5Res(hp_set)
    model = model_obj.build_model(training=False, get_embedding=False)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Grab the checkpoint file
    checkpoint_dir = os.path.join(EXPERIMENT_DIR, experiment_name, 'checkpoints')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    # Load the weights
    model.load_weights(latest_checkpoint).expect_partial()
    logger.info(f'Loaded model and hyperparameters from {os.path.join(EXPERIMENT_DIR, experiment_name)}')

    # First create a map of the layer names to weights and biases
    # from the original model
    original_weights_biases = {}
    for i, layer in enumerate(model.layers):
        original_weights_biases[layer.name] = {}
        if len(layer.weights) == 0:
            original_weights_biases[layer.name]['weights'] = None
        else:
            original_weights_biases[layer.name]['weights'] = layer.weights[0]
        try:
            original_weights_biases[layer.name]['bias'] = layer.bias.numpy()
        except AttributeError:
            original_weights_biases[layer.name]['bias'] = None
        # print('=' * 50)
        # print(layer.name, layer)
        # print('Weights:', layer.weights)
        # try:
        #     print('Bias', layer.bias.numpy())
        # except AttributeError:
        #     print('Bias', 'No Bias for this layer!')

    print(original_weights_biases)

    # Now create map for the class attributes. We want to see what layer
    # name corresponds to what class attribute name. That way we can access the saem layer.
    # Unfortunately, the output names can have two different attributes, BUT we are in evaluation,
    # which means it will always be target_word_output_no_bias.
    layer_name_cls_att_map = {}
    for member in inspect.getmembers(model_obj):
        # Remove private functions and object-level (__len__, __call__, etc.)
        if not member[0].startswith('_') and not inspect.ismethod(member[1]) and issubclass(member[1].__class__, Layer):
            if member[0] not in ['target_word_output', 'target_role_output']:
                layer_name_cls_att_map[member[1].name] = member[0]

    pprint.pprint(layer_name_cls_att_map)

    # print(model_obj.target_word_output_no_bias.weights[0].shape)
    # print(model_obj.target_word_output_no_bias.weights[0].numpy())
    # random_weights = np.random.random(size=(300, 50002))
    # print('What the weights should be:', random_weights)
    # l: Layer = getattr(model_obj, 'target_word_output_no_bias')
    # l.set_weights([random_weights])
    # print('What the weights are after attempted change')
    # print(model_obj.target_word_output_no_bias.weights[0].numpy())

    # Go through each extracted layer in the model file,
    # and see if it has weights and biases. If they do, adjust the corresponding
    # layer in the model class. If it's an embedding layer, then copy it, otherwise,
    # put something random.
    print('Changing weights')
    print('===============================================')
    for layer_name, data in original_weights_biases.items():
        print(layer_name)
        weights = data['weights']
        bias = data['bias']
        weights_to_set = []
        if layer_name not in layer_name_cls_att_map:
            continue
        l: Layer = getattr(model_obj, layer_name_cls_att_map[layer_name])
        if weights is not None:
            # See if we are randomizing the embeddings or rest of the network
            if randomize_embedding:
                if 'embedding' not in layer_name:
                    weights_to_set.append(weights.numpy())
                else:
                    mean = np.mean(weights.numpy())
                    stdev = np.std(weights.numpy())
                    random_weights = np.random.normal(size=weights.numpy().shape, loc=mean, scale=stdev)
                    weights_to_set.append(random_weights)
            else:
                if 'embedding' in layer_name:
                    weights_to_set.append(weights.numpy())
                else:
                    mean = np.mean(weights.numpy())
                    stdev = np.std(weights.numpy())
                    random_weights = np.random.normal(size=weights.numpy().shape, loc=mean, scale=stdev)
                    # random_weights = np.random.random(size=weights.numpy().shape) * (maxi - mini) + mini
                    weights_to_set.append(random_weights)
        if bias is not None and not randomize_embedding:
            # Embeddings never have biases so don't need to check if we are randomizing embedding
            # This
            mean = np.mean(weights.numpy())
            stdev = np.std(weights.numpy())
            random_weights = np.random.normal(size=weights.numpy().shape, loc=mean, scale=stdev)
            # random_weights = np.random.random(size=bias.numpy().shape) * (maxi - mini) + mini
            weights_to_set.append(random_weights)
        # Now set the layer with the saved weights
        if len(weights_to_set) > 0:
            print('Weights before:', weights)
            print('Mean:', np.mean(weights.numpy()))
            print('Stdev:', np.std(weights.numpy()))
            l.set_weights(weights_to_set)
            print('Weights after:', getattr(model_obj, layer_name_cls_att_map[layer_name]).weights)
            print('Mean:', np.mean(weights_to_set[0]))
            print('Stdev:', np.std(weights_to_set[0]))

    # Once we have set the correct weights, then build the model...
    return model_obj
    # new_model = model_obj.build_model(training=False, get_embedding=False)


# Now, create the Evaluation Task classes, and provide this model
# object to read from. Also say to NOT generate reports, since that would
# override the original scores.
_, _, test_data = create_dataset_objects('v2')
accuracies = ['Role Acc', 'Word Acc']
task_order = ['ferretti_instrument', 'ferretti_location', 'greenberg', 'mcrae', 'pado', 'bicknell', 'gs']
runs = 3
randomize_network_results = pd.DataFrame(columns=accuracies + task_order[:-1] + ['gs_full', 'gs_low', 'gs_high'],
                                         index=[f'Run {run}' for run in range(1, runs + 1)])
randomize_embedding_results = pd.DataFrame(columns=accuracies + task_order[:-1] + ['gs_full', 'gs_low', 'gs_high'],
                                           index=[f'Run {run}' for run in range(1, runs + 1)])
for run in range(1, runs + 1):
    network_randomize = randomize_network(EXPERIMENT_NAME)
    embedding_randomize = randomize_network(EXPERIMENT_NAME, randomize_embedding=True)
    for task in task_order:
        logger.info(f'Running {task}...')
        if task == 'bicknell':
            network_evaluator = BicknellTask(SRC_DIR, EXPERIMENT_DIR, 'v5', EXPERIMENT_NAME,
                                             model_obj=network_randomize,
                                             write_report=False)
            embedding_evaluator = BicknellTask(SRC_DIR, EXPERIMENT_DIR, 'v5', EXPERIMENT_NAME,
                                               model_obj=embedding_randomize,
                                               write_report=False)
        elif task == 'gs':
            # Load the context embedding for the GS2013 task!!
            network_evaluator = GS2013Task(SRC_DIR, EXPERIMENT_DIR, 'v5', EXPERIMENT_NAME, get_embedding=True,
                                           model_obj=network_randomize, write_report=False)
            embedding_evaluator = GS2013Task(SRC_DIR, EXPERIMENT_DIR, 'v5', EXPERIMENT_NAME, get_embedding=True,
                                             model_obj=embedding_randomize,
                                             write_report=False)
        else:
            network_evaluator = CorrelateTFScores(SRC_DIR, EXPERIMENT_DIR, 'v5', EXPERIMENT_NAME, task,
                                                  model_obj=network_randomize, write_report=False)
            embedding_evaluator = CorrelateTFScores(SRC_DIR, EXPERIMENT_DIR, 'v5', EXPERIMENT_NAME, task,
                                                    model_obj=embedding_randomize,
                                                    write_report=False)
        network_evaluator.run_task()
        embedding_evaluator.run_task()
        # Once the task is done, the scores are saved in self.metrics, so we
        # can just read the scores from there.
        if task == 'bicknell':
            logger.info(f'BICKNELL (network): {network_evaluator.metrics["accuracy"] * 100:.3f}%')
            randomize_network_results.loc[f'Run {run}', 'bicknell'] = network_evaluator.metrics['accuracy']
            logger.info(f'BICKNELL (embedding): {embedding_evaluator.metrics["accuracy"] * 100:.3f}%')
            randomize_embedding_results.loc[f'Run {run}', 'bicknell'] = embedding_evaluator.metrics['accuracy']
        elif task == 'gs':
            logger.info(f'GS2013 (Full, Low, High) (network): {network_evaluator.metrics["rho"] * 100:.3f}%, '
                        f'{network_evaluator.metrics["low_rho"] * 100:.3f}%, {network_evaluator.metrics["high_rho"] * 100:.3f}')
            randomize_network_results.loc[f'Run {run}', ['gs_full', 'gs_low', 'gs_high']] = \
                [network_evaluator.metrics['rho'], network_evaluator.metrics['low_rho'],
                 network_evaluator.metrics['high_rho']]
            logger.info(f'GS2013 (Full, Low, High) (embedding): {embedding_evaluator.metrics["rho"] * 100:.3f}%, '
                        f'{embedding_evaluator.metrics["low_rho"] * 100:.3f}%, {embedding_evaluator.metrics["high_rho"] * 100:.3f}')
            randomize_embedding_results.loc[f'Run {run}', ['gs_full', 'gs_low', 'gs_high']] = \
                [embedding_evaluator.metrics['rho'], embedding_evaluator.metrics['low_rho'],
                 embedding_evaluator.metrics['high_rho']]
        else:
            logger.info(f'{task.upper()} (network): {network_evaluator.metrics["rho"] * 100:.3f}%')
            randomize_network_results.loc[f'Run {run}', task] = network_evaluator.metrics['rho']
            logger.info(f'{task.upper()} (embedding): {embedding_evaluator.metrics["rho"] * 100:.3f}%')
            randomize_embedding_results.loc[f'Run {run}', task] = embedding_evaluator.metrics['rho']
        # Do our test metrics
        model = network_randomize.build_model(training=True, get_embedding=False)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        test_metrics = model.evaluate(test_data)
        randomize_network_results.loc[f'Run {run}', 'Role Acc'] = test_metrics[3]
        randomize_network_results.loc[f'Run {run}', 'Word Acc'] = test_metrics[4]
        model = embedding_randomize.build_model(training=True, get_embedding=False)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        test_metrics = model.evaluate(test_data)
        randomize_embedding_results.loc[f'Run {run}', 'Role Acc'] = test_metrics[3]
        randomize_embedding_results.loc[f'Run {run}', 'Word Acc'] = test_metrics[4]

# Write the randomization results to the
randomize_network_results.to_csv(os.path.join(SRC_DIR, EXPERIMENT_DIR, EXPERIMENT_NAME, 'random_network_results.csv'))
randomize_embedding_results.to_csv(
    os.path.join(SRC_DIR, EXPERIMENT_DIR, EXPERIMENT_NAME, 'random_embedding_results.csv'))
