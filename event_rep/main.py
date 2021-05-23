"""
File: event_rep/main.py
Author: Mughilan Muthupari
Creation Date: 2021-05-15

This file will be equivalent in function to the main.py in the older
codebase. The difference is that this is not in model_implementation
anymore, as this is not strictly deals in those matters. Having in the
top level also makes running everything easier.
"""

import argparse
import os
import sys
import time
import shutil
from typing import Dict, Type
import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

from model_implementation.architecture.hp.hyperparameters import HyperparameterSet
from model_implementation.core.roles import *
from model_implementation.architecture.models import *
from model_implementation.core.batcher import WordRoleWriter

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
# Generate the huperparameter set from the SRC DIR
hp_set = HyperparameterSet(SRC_DIR)
# This is a dictionary which maps the user-provided parameter of the model to
# the corresponding class
# TODO: Add models here as necessary, as the argument enforcement is on the keys
PARAM_TO_MODEL: Dict[str, Type[MTRFv4Res]] = {
    'v4': MTRFv4Res
}

# Make the directories if they don't already exist.
for directory in [EXPERIMENT_DIR, PRETRAINED_DIR, CSV_PIECE_PATH, DATA_PATH]:
    os.makedirs(directory, exist_ok=True)

# This the role set we will be using...
ROLE_SET = Roles2Args3Mods

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a thematic fit model')
    # Create the hyperparameter set, which sets all of our default parameters.

    # Required parameters
    parser.add_argument('model_name', choices=PARAM_TO_MODEL.keys(), type=str,
                        help='The naem of the model to train. Must be one of the model classes'
                             'defined in models.py')
    parser.add_argument('data_version', type=str,
                        help='The folder in the data directory that contains NN_train, NN_test, etc.')

    # Optional parameters
    parser.add_argument('--experiment_name', type=str, default='',
                        help='The name of the experiment. ALl model artifacts get saved in a subdirectory of'
                             'this name. Will default to a concatenation of the time, model name, and data version.')
    parser.add_argument('--load_previous', dest='load_previous', action='store_true',
                        help='Whether to load a previously trained model.')
    parser.add_argument('--batch_size', type=int, default=hp_set.batch_size,
                        help='The batch size for training')
    parser.add_argument('--dropout_rate', type=float, default=hp_set.dropout_rate,
                        help='The dropout rate for the Dropout layers. Should be between 0 and 1.')
    parser.add_argument('--embedding_type', choices=['random', 'spacy_0', 'spacy_avg', 'fasttext_0',
                                                     'fasttext_avg', 'w2v_0', 'w2v_avg'],
                        help='The type of embedding to use for WORDS. Can be random, or a type of'
                             'pretrained embedding (spaCy, Fasttext, Word2Vec) along with an OOV initialization'
                             '(one of 0 or avg)')
    parser.add_argument('--epochs', type=int, default=hp_set.epochs,
                        help='The MAXIMUM number of epochs to train. It will likely not reach this amount due'
                             'to early stopping criteria.')
    parser.add_argument('--freeze_role_embeddings', dest='freeze_role_embeddings',
                        action='store_true', default=hp_set.freeze_role_embeddings,
                        help='If listed, then freezes the role embeddings i.e. prevents training')
    parser.add_argument('--freeze_word_embeddings', dest='freeze_word_embeddings',
                        action='store_true', default=hp_set.freeze_word_embeddings,
                        help='If listed, then freezes the word embeddings i.e. prevents training. This does'
                             'NOT affect the type of word embedding initialization from embedding_type.')
    parser.add_argument('--hidden_neurons', type=int, default=hp_set.hidden_neurons,
                        help='The number of hidden neurons in the layer where the context embeddings merge'
                             'into the target word and role embeddings.')
    parser.add_argument('--l1_regularization', type=float, default=hp_set.l1_regularization,
                        help='The amount of L1 regularization to use')
    parser.add_argument('--l2_regularization', type=float, default=hp_set.l2_regularization,
                        help='The amount of L2 regularization to use')
    # TODO: Add more language models as they are added
    parser.add_argument('--language_model', choices=['null', 'xlnet', 'bert'], default=hp_set.language_model,
                        help='The type of language model to use for the word embeddings. This OVERRIDES the'
                             'selection made in embedding_type')
    parser.add_argument('--learning_rate', type=float, default=hp_set.learning_rate,
                        help='The learning rate for model training')
    parser.add_argument('--learning_rate_decay', type=float, default=hp_set.learning_rate_decay,
                        help='The amount to decay the learning rate throughout training')
    parser.add_argument('--loss_weight_role', type=float, default=hp_set.loss_weight_role,
                        help='The amount to weight the role when calculating loss')
    parser.add_argument('--n_factors_cls', type=int, default=hp_set.n_factors_cls,
                        help='n_factors_cls')
    parser.add_argument('--pretrained_embedding_size', type=int, default=hp_set.pretrained_embedding_size,
                        help='The embedding size to use for pretrained embeddings. Utilized for when'
                             'embedding_type is anything other than random.')
    parser.add_argument('--role_embedding_dimension', type=int, default=hp_set.role_embedding_dimension,
                        help='The embedding size for the roles')
    parser.add_argument('--word_embedding_dimension', type=int, default=hp_set.word_embedding_dimension,
                        help='The embedding size for the words')
    parser.add_argument('--avg_out_miss_embedding', dest='zero_out_miss_embedding',
                        action='store_false', default=hp_set.zero_out_miss_embedding,
                        help='If listed, then substitutes the average + noise for pretrained embedding vectors'
                             'where those words are not listed in our vocabulary. By default, substitutes them'
                             'with all 0 vectors.')
    # Extra parameter, role set, generally not touched at all.
    parser.add_argument('--role_set', type=Roles, default=Roles2Args3Mods,
                        help='The role set to use. Default Roless2Args3Mods')

    args = parser.parse_args()

    if args.experiment_name == '':
        experiment_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}_{args.model_name}_" \
                          f"{args.data_version}"
    else:
        experiment_name = args.experiment_name
    # Convert arguments into dictionary
    opts = vars(args)
    print(opts)
    # Pop the irrelevant attributes from the dictionary, and pass them to the hyperparameter
    # set so the values get updated.
    irrel_keys = ['model_name', 'data_version', 'role_set', 'experiment_name']
    for key in irrel_keys:
        if key in opts:
            opts.pop(key)
    hp_set.update_parameters(opts)

    # Now that the parameters are updated, let's generate the dataset object we need
    data_writer = WordRoleWriter(input_data_dir=os.path.join(DATA_PATH, args.data_version),
                                 output_csv_path=CSV_PIECE_PATH,
                                 batch_size=hp_set.batch_size,
                                 MISSING_WORD_ID=hp_set.missing_word_id,
                                 N_ROLES=hp_set.role_vocab_count)
    data_writer.write_csv_pieces()
    train_data = data_writer.get_tf_dataset('train')
    vali_data = data_writer.get_tf_dataset('dev')
    test_data = data_writer.get_tf_dataset('test')

    # Make the model object!
    model: Type[MTRFv4Res] = PARAM_TO_MODEL[args.model_name]
    logging.info('Clean model summary:')
    model.summary()


    model_artifact_dir = os.path.join(EXPERIMENT_DIR, experiment_name)
    checkpoint_dir = os.path.join(EXPERIMENT_DIR, experiment_name, 'checkpoints')
    if os.path.exists(model_artifact_dir):
        if args.load_previous:
            logging.info(f'Attempting to continue train from a checkpoint at {checkpoint_dir}...')
            if not os.path.exists(checkpoint_dir):
                logging.error('No checkpoints present! Quitting...')
                sys.exit(1)
            # TODO: Add reading from a description/metrics file...
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info(f'Found latest checkpoint {latest_checkpoint}! Loading weights...')
            model.load_weights(latest_checkpoint)
            initial_epoch = 0
        # If the path exists, but we are not loading the previous,
        # then delete whatever is there...
        else:
            shutil.rmtree(model_artifact_dir)
            os.makedirs(model_artifact_dir, exist_ok=True)
            existing_description = None
            initial_epoch = 0

    # Make callbacks...
    checkpointer = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'cp_{epoch:03d}.ckpt'),
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=0,
                                   save_weights_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=1)
    nanChecker = TerminateOnNaN()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-3)
    # TODO: Add description/metric callback

    start_time = datetime.datetime.now(datetime.timezone.utc)
    logging.info(f'TRAINING STARTED AT {start_time} UTC...')

    # FIT OUR MODEL!
    model.fit(train_data,
              epochs=initial_epoch + args.epochs,
              workers=1,
              verbose=1,
              initial_epoch=initial_epoch,
              validation_data=vali_data,
              callbacks=[checkpointer, stopper, nanChecker, reduce_lr])

    # Report time taken, and metrics on the testing dataset...
    end_time = datetime.datetime.now(datetime.timezone.utc)
    print(f'TRAINING ENDED AT {end_time} UTC...')

    # For testing, we need to load the latest checkpoint,
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(f'Testing with latest checkpoint: {latest}')
    model.model.load_weights(latest)
    model.model.evaluate(test_data, workers=1)

    saved_model_dir = os.path.join(EXPERIMENT_DIR, experiment_name, 'model')
    print(f'Saving model in SavedModel format at {saved_model_dir}...')
    model.model.save(saved_model_dir)
    print('Done!')

    print(f'EXPERIMENT SAVED AT {model_artifact_dir}.')

    end_time = datetime.datetime.now(datetime.timezone.utc)
    print(f'PROGRAM ENDED AT {end_time} UTC...')
    print(f'TOTAL TIME: {str(end_time - start_time)}')