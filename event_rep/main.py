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
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet

# Directory locations
# The absolute path where main is being run. Should end in RW-Eng-v3-src/event_rep
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory where experiments are saved
EXPERIMENT_DIR = os.path.join(SRC_DIR, 'experiments')
# Pretrained embeddings
PRETRAINED_DIR = os.path.join(SRC_DIR, 'pretrained_embeddings')
# Primary data path for data...
DATA_PATH = os.path.join(SRC_DIR, 'processed_data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a thematic fit model')
    # Create the hyperparameter set, which sets all of our default parameters.
    hp_set = HyperparameterSet(SRC_DIR)

    # Required parameters
    # TODO: Add models to choices
    parser.add_argument('model_name', choices=['MTRFv4Res', 'MTRFv5Res', 'MTRFv6Res'],
                        help='The naem of the model to train. Must be one of the model classes'
                             'defined in models.py')
    parser.add_argument('data_version', type=str,
                        help='The folder in the data directory that contains NN_train, NN_test, etc.')

    # Optional parameters
    parser.add_argument('--experiment_name', type=str,
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

    opts = vars(parser.parse_args())

    print(opts)
