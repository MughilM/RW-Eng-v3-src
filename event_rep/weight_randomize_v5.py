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
import os
import sys
import logging
import pprint

from model_implementation.architecture.models import MTRFv5Res
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet
from evaluation.tasks import CorrelateTFScores, BicknellTask, GS2013Task

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

# The experiment with the model we want to grab from
EXPERIMENT_NAME = '3a_plus_03_v5_tunespacyavg_1perc'

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
# Generate the huperparameter set from the SRC DIR
hp_set = HyperparameterSet(os.path.join(EXPERIMENT_DIR, EXPERIMENT_NAME, 'hyperparameters.json'))

# Create the model, and pull the weights from the latest checkpoint
model_obj = MTRFv5Res(hp_set)
model = model_obj.build_model(training=False, get_embedding=False)
model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Grab the checkpoint file
checkpoint_dir = os.path.join(EXPERIMENT_DIR, EXPERIMENT_NAME, 'checkpoints')
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
# Load the weights
model.load_weights(latest_checkpoint).expect_partial()
logger.info(f'Loaded model and hyperparameters from {os.path.join(EXPERIMENT_DIR, EXPERIMENT_NAME)}')

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
    print('=' * 50)
    print(layer.name, layer)
    print('Weights:', layer.weights)
    try:
        print('Bias', layer.bias.numpy())
    except AttributeError:
        print('Bias', 'No Bias for this layer!')

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

print(model_obj.target_word_output_no_bias.weights[0].shape)
print(model_obj.target_word_output_no_bias.weights[0].numpy())
random_weights = np.random.random(size=(300, 50002))
print('What the weights should be:', random_weights)
l: Layer = getattr(model_obj, 'target_word_output_no_bias')
l.set_weights([random_weights])
print('What the weights are after attempted change')
print(model_obj.target_word_output_no_bias.weights[0].numpy())

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
        if 'embedding' in layer_name:
            weights_to_set.append(weights.numpy())
        else:
            mini = np.min(weights.numpy())
            maxi = np.max(weights.numpy())
            random_weights = np.random.random(size=weights.numpy().shape) * (maxi - mini) + mini
            weights_to_set.append(random_weights)
    if bias is not None:
        # Embeddings never have biases
        mini = np.min(bias.numpy())
        maxi = np.max(bias.numpy())
        random_weights = np.random.random(size=bias.numpy().shape) * (maxi - mini) + mini
        weights_to_set.append(random_weights)
    # Now set the layer with the saved weights
    if len(weights_to_set) > 0:
        print('Weights before:', weights)
        print('Min:', np.min(weights.numpy()))
        print('Max:', np.max(weights.numpy()))
        print('Weights after:', weights_to_set)
        print('Min:', np.min(weights_to_set[0]))
        print('Max:', np.max(weights_to_set[0]))
        l.set_weights(weights_to_set)

# Once we have set the correct weights, then build the model...
new_model = model_obj.build_model(training=False, get_embedding=False)



