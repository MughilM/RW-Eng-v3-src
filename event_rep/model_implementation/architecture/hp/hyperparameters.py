"""
File: event_rep/model_implementation/architecture/hp/hyperparameters.py
Author: Mughilan Muthupari
Creation Date: 2021-05-15

This file holds all of the hyperparameter classes necessary for model building.
The values themselves are held in JSON files for easy reading. The design implementation
is for the module to ALWAYS write the meta values (epochs, learning rate, etc.)
to the experiment directory, while the user can choose which architect values
(number of hidden nodes, embedding dimensions, etc.) when writing.
"""
import json
import os
import pickle
import logging

logger = logging.getLogger(__name__)


class HyperparameterSet:
    def __init__(self, SRC_DIR, output_dir='./', default_hp_file='default_params_all.json'):
        """
        Initializer for the default hyperparameter set. The implementation shouldn't be changed
        unless hyperparameters are added. Anyone looking at the source code, do not pay attention
        to the values defined here, as they are immediately overwritten by the values in the JSONs.
        :param output_dir: The directory to output the final hyperparameter JSON
        :param description_file: The path to the data description file that contains vocabulary data
        :param default_hp_file: The JSON that contains the default hyperparameters
        """
        self.output_dir = output_dir
        self.default_hp_file = os.path.join(SRC_DIR, 'model_implementation/architecture/hp', default_hp_file)
        # META parameters. Make sure the name matches exactly with JSON keys,
        # as we'll be using setattr to transfer the values over.
        self.batch_size = 0
        self.epochs = 0
        self.learning_rate = 0
        self.learning_rate_decay = 0
        self.l1_regularization = 0
        self.l2_regularization = 0
        self.loss_weight_role = 0
        self.dropout_rate = 0
        self.freeze_word_embeddings = False
        self.freeze_role_embeddings = False
        # Architecture parameters. As before, make sure names match.
        self.embedding_type = ''
        self.word_embedding_dimension = 0
        self.role_embedding_dimension = 0
        self.n_factors_cls = 0
        self.hidden_neurons = 0
        self.pretrained_embedding_size = 0
        self.language_model = ''
        self.zero_out_miss_embedding = False
        # Vocabulary parameters.
        self.missing_role_id = 0
        self.missing_word_id = 0
        self.unk_word_id = 0
        self.unk_role_id = 0
        self.word_vocabulary = {}
        self.role_vocabulary = {}
        # Now read in the JSONs and use setattr to transfer values...
        with open(self.default_hp_file, 'r') as f:
            params = json.load(f)
            update_object_params_dict(self, params)
        # with open(description_file, 'rb') as f:
        #     des = pickle.load(f)
        #     for paramName, value in des.items():
        #         if hasattr(self, paramName):
        #             setattr(self, paramName, value)
        self.word_vocab_count = 0
        self.role_vocab_count = 0

    def read_description_params(self, description_file):
        """
        We need a separate method to read the parameters from the description, because
        this depends on what the user inputs for their dataset.
        :param description_file: The path to the description file (non-json).
        :return: Nothing, but sets all of the class variables
        """
        with open(description_file, 'rb') as f:
            des = pickle.load(f)
            update_object_params_dict(self, des)
        # As of this moment, the unknown word and role are NOT included
        # in the vocabulary dictionary, so we must add them separately.
        # TODO: Ideally, these should be included in the vocabulary dictionaries in description itself.
        if self.missing_word_id != self.unk_word_id:
            self.word_vocabulary['<NULL>'] = self.missing_word_id
        self.word_vocabulary['<UNKNOWN>'] = self.unk_word_id
        self.role_vocabulary['<OTHER>'] = self.unk_role_id
        self.word_vocab_count = len(self.word_vocabulary)
        self.role_vocab_count = len(self.role_vocabulary)

    def set_output_dir(self, output_dir):
        """
        Sets the output directory on where to write the full hyperparameter JSON.
        This can differ due to the overlying experiment name, which isn't known until
        all the user parameters are read in.
        :param output_dir:
        :return:
        """
        self.output_dir = output_dir

    def update_parameters(self, new_params: dict):
        """
        Updates the hyperparameters based on a new dictionary.
        :param new_params:
        :return:
        """
        update_object_params_dict(self, new_params)

    def write_hp(self, ignore_hp: list = None):
        """
        Writes the hyperparameter set to a JSON in the output directory.
        :return:
        """
        # We get all of the class variables. dir() grabs the functions as well,
        # so exclude those...
        if ignore_hp is None:
            ignore_hp = []
        full_ignore_list = ['output_dir', 'default_hp_file', 'write_hp'] + ignore_hp
        attributes = [k for k in dir(self) if not k.startswith('_') and k not in full_ignore_list]
        with open(os.path.join(self.output_dir, 'hyperparameters.json'), 'w') as f:
            json.dump({k: getattr(self, k) for k in attributes}, fp=f, indent=2, separators=(',', ': '))


def update_object_params_dict(obj, d: dict):
    """
    Function to prevent repeated code. Updates the object with
    parameter and values from d.
    :param obj: The object whose variables to update
    :param d: The dictionary that contains the values. The keys translate to object attributes.
    :return:
    """
    for paramName, value in d.items():
        if hasattr(obj, paramName):
            setattr(obj, paramName, value)
        else:
            logger.warning(f'{paramName} not in class variables. Please update class if variable is needed.'
                           f'Skipping for now...')
