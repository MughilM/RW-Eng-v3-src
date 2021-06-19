"""
File: event_rep/model_implementation/architecture/models.py
Author: Mughilan Muthupari
Creation Date: 2021-05-15

This module holds all of the model classes that we utilize, starting from the
original MTRFv4Res baseline model. Any extra models that are needed should
be implemented here. Each model class should be paired with a hyperparameter class
from hp/hyperparameters.py that define only the hyperparameters necessary.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet

# Embedding Packages
import spacy
import fasttext
import fasttext.util
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors

import os
import requests
import logging


class MTRFv4Res(Model):
    def __init__(self, hp_set: HyperparameterSet, pretrained_emb_dir=None, **kwargs):
        super(MTRFv4Res, self).__init__(**kwargs)
        self.hp_set = hp_set
        self.PRETRAINED_DIR = pretrained_emb_dir
        self.logger = logging.getLogger(__name__)
        # Input layers, ONLY USED IN build().
        self.input_words = Input(shape=(self.hp_set.role_vocab_count - 1), dtype=tf.uint32, name='input_words')
        self.input_roles = Input(shape=(self.hp_set.role_vocab_count - 1), dtype=tf.uint32, name='input_roles')
        self.target_word = Input(shape=(1,), dtype=tf.uint32, name='target_word')
        self.target_role = Input(shape=(1,), dtype=tf.uint32, name='target_role')
        # Add the embedding layers for the input words and roles
        self.word_embedding = self.get_embedding()
        self.role_embedding = Embedding(input_dim=self.hp_set.role_vocab_count,
                                        output_dim=self.hp_set.role_embedding_dimension,
                                        embeddings_initializer=glorot_uniform(), name='role_embedding',
                                        trainable=not self.hp_set.freeze_role_embeddings)
        # Make a mask for the embedding to zero out the embedding values
        # for where the word is missing. This is the cleanest solution for
        # right now, but open to more elegant solutions in the future...
        self.embedding_mask = Lambda(lambda x: tf.cast(
            tf.expand_dims(
                tf.not_equal(x, self.hp_set.missing_word_id, name='missing_word_reverse_mask'),
                axis=-1, name='expand'),
            dtype=tf.float32, name='cast_to_float'), name='create_mask_for_missing_words')

        self.word_embed_multi = Multiply(name='apply_mask_to_word')
        self.role_embed_multi = Multiply(name='apply_mask_to_role')
        # Dropout layers for both word and role
        self.word_dropout = Dropout(self.hp_set.dropout_rate)
        self.role_dropout = Dropout(self.hp_set.dropout_rate)
        # Multiplication of embeddings layer
        self.embedding_multiply = Multiply(name='multiply_embeddings')
        # Next, is the residual block, which takes the combined embeddings and
        # adds it to a PReLU forwarded version of itself
        self.lin_proj1 = Dense(self.hp_set.word_embedding_dimension, activation='linear',
                               use_bias=False, name='linear_proj_1')
        self.prelu = PReLU(alpha_initializer='ones', name='prelu')
        self.lin_proj2 = Dense(self.hp_set.word_embedding_dimension, activation='linear',
                               use_bias=False, name='linear_proj_2')
        self.prelu_proj_add = Add(name='residual')
        self.average_across_input = Lambda(lambda x: tf.reduce_mean(x, axis=1), name='context_embedding')
        # Finally we need to produce the embeddings for the target role and word...
        # Our target word and role will be (batch_size, 1, 512)...
        # Note that embeddings cross as inputs into the other target e.g.
        # the target word embedding inputs to the ROLE and vice versa.
        self.target_word_embedding = Embedding(self.hp_set.word_vocab_count, self.hp_set.n_factors_cls,
                                               embeddings_initializer=glorot_uniform(),
                                               name='target_word_embedding')
        self.target_role_embedding = Embedding(self.hp_set.role_vocab_count, self.hp_set.n_factors_cls,
                                               embeddings_initializer=glorot_uniform(),
                                               name='target_role_embedding')
        # Create two dropout layers for both, and a reshape layer as there is an extra dimension.
        self.target_word_drop = Dropout(self.hp_set.dropout_rate, name='target_word_dropout')
        self.target_role_drop = Dropout(self.hp_set.dropout_rate, name='target_role_dropout')
        self.target_word_flatten = Flatten(name='flatten_target_word_embedding')
        self.target_role_flatten = Flatten(name='flatten_target_role_embedding')
        # Create two weighted context embeddings that will pass into each target embedding
        self.weight_context_word = Dense(self.hp_set.n_factors_cls, activation='linear', use_bias=False,
                                         name='weighted_context_word')
        self.weight_context_role = Dense(self.hp_set.n_factors_cls, activation='linear', use_bias=False,
                                         name='weighted_context_role')
        # Create two Multiply layers for when we multiply the weighted context with the embeddings...
        # Remember to cross! These will be enforced during call.
        self.target_word_hidden = Multiply(name='cont_x_tre')  # target ROLE embedding for target word hidden
        self.target_role_hidden = Multiply(name='cont_x_twe')  # target WORD embedding for target role hidden

        ### FINALLY THE OUTPUTS
        # The reason the activation is separated is because it makes it easy
        # to NOT USE THE BIAS during the evaluation tasks.
        self.target_word_output_raw = Dense(self.hp_set.word_vocab_count, name='word_output_raw')
        self.target_role_output_raw = Dense(self.hp_set.role_vocab_count, name='role_output_raw')
        self.target_word_act = Activation('softmax', name='word_output_act')
        self.target_role_act = Activation('softmax', name='role_output_act')

    def call(self, inputs, training=True, mask=None, get_embedding=False):
        # WE ASSUME THAT THE INPUTS ARE PASSED IN AS DICTIONARIES WITH THE KEYS:
        # [input_words, input_roles, target_word, target_role]
        # Though of course, when the Model is defined and data is fed through, it is much easier
        # to simply provide a tf.dataset where the input layer names map properly...
        # input_words, input_roles, target_word, target_role = inputs[0], inputs[1], inputs[2], inputs[3]
        input_words, input_roles, target_word, target_role = inputs['input_words'], inputs['input_roles'], \
                                                             inputs['target_word'], inputs['target_role']
        # Pass inputs through embedding...
        word_embedding_out = self.word_embedding(input_words)
        role_embedding_out = self.role_embedding(input_roles)
        # Now zero out the missing word embeddings if need be.
        if self.hp_set.zero_out_miss_embedding:
            mask = self.embedding_mask(input_words)
            # Update the embeddings
            word_embedding_out = self.word_embed_multi([word_embedding_out, mask])
        # Apply dropout (obviously if dropout_rate = 0, then this layer does nothing...
        # ...but our default is nonzero...
        word_embedding_out = self.word_dropout(word_embedding_out)
        role_embedding_out = self.role_dropout(role_embedding_out)
        # Concatenate the two embeddings
        total_embeddings = self.embedding_multiply([word_embedding_out, role_embedding_out])
        # Pass the total embeddings through the residual block next
        residual = self.prelu_proj_add([
            total_embeddings,
            self.lin_proj2(self.prelu(self.lin_proj1(total_embeddings)))
        ])
        context_embedding = self.average_across_input(residual)
        if get_embedding:
            return context_embedding
        # Create target role and word embeddings multiplied with the context.
        # Note the crossing of inputs into the other's hidden layer.
        twh_out = self.target_word_hidden([
            self.weight_context_word(context_embedding),
            self.target_role_flatten(self.target_role_drop(self.target_role_embedding(target_role)))
        ])
        trh_out = self.target_role_hidden([
            self.weight_context_role(context_embedding),
            self.target_word_flatten(self.target_word_drop(self.target_word_embedding(target_word)))
        ])
        # Forward through the output layers.
        # If training is False, then DO NOT USE THE BIAS IN THE OUTPUT LAYERS...
        if not training:
            tw_out_raw = tf.tensordot(twh_out, self.target_word_output_raw.weights[0], axes=1)
            tr_out_raw = tf.tensordot(trh_out, self.target_role_output_raw.weights[0], axes=1)
        else:
            tw_out_raw = self.target_word_output_raw(twh_out)
            tr_out_raw = self.target_role_output_raw(trh_out)
        tw_out = self.target_word_act(tw_out_raw)
        tr_out = self.target_role_act(tr_out_raw)
        return {'w_out': tw_out, 'r_out': tr_out}

    def build(self, input_shapes=None):
        """
        This differs from the normal build because input_shapes is not required,
        since the class defines the input layers in-house. But the parameter has
        to stay, otherwise model.fit, evaluate, etc. would cause errors.
        :param input_shapes: The input_shapes. NOT USED, but used by model.fit() during training,
        so removing would cause errors.
        :return: Model object with inputs and outputs defined.
        """
        input_list = {'input_words': self.input_words,
                      'input_roles': self.input_roles,
                      'target_word': self.target_word,
                      'target_role': self.target_role}
        return Model(inputs=input_list, outputs=self.call(input_list))

    def get_embedding(self):
        """
        Method that gets the correct embedding, depending on what sort of embedding is chosen.
        spaCy and Fasttext models are saved in their respective package directory, while
        the embedding for word2vec will be downloaded to a separate directory if needed.
        :return:
        """
        # First check the embedding type. If random, then simply random a
        # randomly initilized embedding...
        if self.hp_set.embedding_type == 'random':
            return Embedding(input_dim=self.hp_set.word_vocab_count,
                             output_dim=self.hp_set.word_embedding_dimension,
                             embeddings_initializer=glorot_uniform(), name='word_embedding_random',
                             trainable=not self.hp_set.freeze_word_embeddings)
        # Otherwise we are looking at one of 3 embeddings, and one of 2 initilaizers
        # for OOV...
        embedding_source, oov_init = tuple(self.hp_set.embedding_type.split('_'))
        layer_name = f'word_embedding_{embedding_source}_{oov_init}'
        # We essentially want to merge a blank embedding matrix with whatever
        # words we have in teh pretrained embedding model, and either leave the missing
        # word alone or impute them with the average embedding (plus some noise)
        full_embedding_matrix = np.zeros(shape=(self.hp_set.word_vocab_count, self.hp_set.word_embedding_dimension))
        # Because we plan to match our word IDs with the indices of our pretrained embedding matrices,
        # it will helpful if we separate the words and the IDs and sort them.
        words = np.array(list(self.hp_set.word_vocabulary.keys()))
        IDs = np.array(list(self.hp_set.word_vocabulary.values()))
        words = words[np.argsort(IDs)]
        # Check each of the three and merge embeddings...
        if embedding_source == 'spacy':
            nlp = spacy.load('en_core_web_lg')
            # Get spaCy's embedding matrix
            spacy_matrix = nlp.vocab.vectors.data
            # Use the find function to get the indices in the spaCy matrix
            # that corresponds to our words. If the word is NOT there, then
            # the function returns -1 for that word. Adjust accordingly...
            spacy_IDs = nlp.vocab.vectors.find(keys=words)
            full_embedding_matrix = spacy_matrix[spacy_IDs]
            # Whereever the spaCy ID is -1, set it to 0...
            full_embedding_matrix[spacy_IDs == -1] = 0
        elif embedding_source == 'fasttext':
            # Download fasttext embedding to the directory,
            # We can't control download location with utils.download_model
            fasttext_emb_path = os.path.join(self.PRETRAINED_DIR, 'cc.en.300.bin.gz')
            if not os.path.exists(fasttext_emb_path):
                self.logger.info(f'Fasttext embeddings not present. '
                                 f'Downloading file (~4.2 GB) to {fasttext_emb_path}...', end='')
                fasttext_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
                r = requests.get(fasttext_url, allow_redirects=True)
                open(fasttext_emb_path, 'wb').write(r.content)
                self.logger.info('Done!')
            self.logger.info(f'Reading Fasttext embeddings from {fasttext_emb_path}')
            fasttext_model = fasttext.load_model(fasttext_emb_path)
            # Unfortunately, there is no way to grab the fasttext embedding matrix,
            # we need to do it one at a time.
            vect_get_vector = np.vectorize(fasttext_model.get_word_vector)
            vect_get_id = np.vectorize(fasttext_model.get_word_id)
            # Apply it to our vocabulary. With fasttext, it does NOT return a 0 vector if the
            # word is absent, but get_word_id from fasttext can help with that.
            full_embedding_matrix = vect_get_vector(words)
            fasttext_ids = vect_get_id(words)
            full_embedding_matrix[fasttext_ids == -1] = 0
            # We don't need the fasttext model anymore, delete it, it's taking up 4 GB of memory...
            del fasttext_model
        # Last is Word2Vec
        elif embedding_source == 'w2v':
            w2v_emb_path = os.path.join(self.PRETRAINED_DIR, 'GoogleNews-vectors-negative300.bin.gz')
            if not os.path.exists(w2v_emb_path):
                self.logger.info(f'Word2Vec embeddings not present. Downloading file (~1.5 GB) to {w2v_emb_path}...', end='')
                w2v_url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
                r = requests.get(w2v_url, allow_redirects=True)
                open(w2v_emb_path, 'wb').write(r.content)
                self.logger.info('Done!')
            self.logger.info(f'Reading Word2Vec embeddings from {w2v_emb_path}')
            w2v_model = KeyedVectors.load_word2vec_format(w2v_emb_path, binary=True)
            # A Gensim 4.0.0 special. Note that if Gensim 3.x is installed, the following will not work.
            # Due to the annoying KeyError, we have to change the function a tad...
            get_any_word = np.vectorize(lambda w: w2v_model.key_to_index[w] if w in w2v_model.key_to_index else -1)
            # Apply our vectorized function to the word list
            gensim_IDs = get_any_word(words)
            full_embedding_matrix = w2v_model.vectors[gensim_IDs]
            # Set any -1 gensim_IDs that are -1 to 0...
            full_embedding_matrix[gensim_IDs == -1] = 0
        else:
            self.logger.error(f'Embedding source of {embedding_source} is invalid.')
        # Now check to see if we need to keep OOV at 0 or we need to average the
        # ones we have...
        if oov_init == 'avg':
            # By default, we have 0s for OOV. Get the rows of OOV...
            oov_rows = np.where(np.sum(full_embedding_matrix, axis=1) == 0)[0]
            # Easier to impute the mean if the 0s were nan.
            full_embedding_matrix[full_embedding_matrix == 0] = np.nan
            # Compute the imputed mean, down the columns this time
            imputed_mean = np.nanmean(full_embedding_matrix, axis=0)
            # Assign to matrix
            full_embedding_matrix[oov_rows] = imputed_mean
            # Now add some noise...
            # TODO: Hardcoded limit bounds here (-0.05, 0.05), maybe make as a hyperparameter?
            noise = np.random.random((len(oov_rows), self.hp_set.word_embedding_dimension)) * 0.1 - 0.05
            full_embedding_matrix[oov_rows] += noise

        # Make the embedding layer and return it, and we are done!
        return Embedding(input_dim=self.hp_set.word_vocab_count,
                         output_dim=self.hp_set.word_embedding_dimension,
                         embeddings_initializer=full_embedding_matrix,
                         name=layer_name,
                         trainable=not self.hp_set.freeze_word_embeddings)


class MTRFv5Res(MTRFv4Res):
    """
    In v5 of the model, we use a 'shared embedding', in the sense
    that the TARGET words and roles will use the same embedding as the
    input words and roles. This is to see if we can preserve model
    performance even when drastically reducing the number of parameters.

    The only thing we need to change in init, are the weighted context
    layers, since they need to be multiplied with the flattened embedding,
    because the dimensions have changed now. We also need to re-implement
    call, for a such a small change. This is a downside of model subclassing.
    We could use helper methods, but that introduces more links we need to manage.
    Plus, we have no idea how the functions would link from one to the next.
    """
    def __init__(self, hp_set: HyperparameterSet, **kwargs):
        super().__init__(hp_set, **kwargs)
        # Weighted context word will multiply with target role embedding,
        # so output nodes should be the same as the role embedding dimension.
        # Likewise for the weighted context role and target word embedding.
        self.weight_context_word = Dense(self.hp_set.role_embedding_dimension, activation='linear', use_bias=False,
                                         name='weighted_context_word')
        self.weight_context_role = Dense(self.hp_set.word_embedding_dimension, activation='linear', use_bias=False,
                                         name='weighted_context_role')

    def call(self, inputs, training=True, mask=None, get_embedding=False):
        input_words, input_roles, target_word, target_role = inputs['input_words'], inputs['input_roles'], \
                                                             inputs['target_word'], inputs['target_role']
        # Pass inputs through embedding...
        word_embedding_out = self.word_embedding(input_words)
        role_embedding_out = self.role_embedding(input_roles)
        # Now zero out the missing word embeddings if need be.
        if self.hp_set.zero_out_miss_embedding:
            mask = self.embedding_mask(input_words)
            # Update the embeddings
            word_embedding_out = self.word_embed_multi([word_embedding_out, mask])
        # Apply dropout (obviously if dropout_rate = 0, then this layer does nothing...
        # ...but our default is nonzero...
        word_embedding_out = self.word_dropout(word_embedding_out)
        role_embedding_out = self.role_dropout(role_embedding_out)
        # Concatenate the two embeddings
        total_embeddings = self.embedding_multiply([word_embedding_out, role_embedding_out])
        # Pass the total embeddings through the residual block next
        residual = self.prelu_proj_add([
            total_embeddings,
            self.lin_proj2(self.prelu(self.lin_proj1(total_embeddings)))
        ])
        context_embedding = self.average_across_input(residual)
        if get_embedding:
            return context_embedding
        # Create target role and word embeddings multiplied with the context.
        # Note the crossing of inputs into the other's hidden layer.
        ###### THE CHANGE IS HERE! INSTEAD OF USING target_word/role_embedding,
        ###### we use word/role embedding...The flattening stands...
        twh_out = self.target_word_hidden([
            self.weight_context_word(context_embedding),
            self.target_role_flatten(self.target_role_drop(self.role_embedding(target_role)))
        ])
        trh_out = self.target_role_hidden([
            self.weight_context_role(context_embedding),
            self.target_word_flatten(self.target_word_drop(self.word_embedding(target_word)))
        ])
        # Forward through the output layers.
        # If training is False, then DO NOT USE THE BIAS IN THE OUTPUT LAYERS...
        if not training:
            tw_out_raw = tf.tensordot(twh_out, self.target_word_output_raw.weights[0], axes=1)
            tr_out_raw = tf.tensordot(trh_out, self.target_role_output_raw.weights[0], axes=1)
        else:
            tw_out_raw = self.target_word_output_raw(twh_out)
            tr_out_raw = self.target_role_output_raw(trh_out)
        tw_out = self.target_word_act(tw_out_raw)
        tr_out = self.target_role_act(tr_out_raw)
        return {'w_out': tw_out, 'r_out': tr_out}