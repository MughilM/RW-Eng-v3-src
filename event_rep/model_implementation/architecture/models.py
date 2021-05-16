"""
File: event_rep/model_implementation/architecture/models.py
Author: Mughilan Muthupari
Creation Date: 2021-05-15

This module holds all of the model classes that we utilize, starting from the
original MTRFv4Res baseline model. Any extra models that are needed should
be implemented here. Each model class should be paired with a hyperparameter class
from hp/hyperparameters.py that define only the hyperparameters necessary.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_uniform
from model_implementation.architecture.hp.hyperparameters import HyperparameterSet


class MTRFv4Res(Model):
    def __init__(self, hp_set: HyperparameterSet):
        super(MTRFv4Res, self).__init__()
        self.hp_set = hp_set
        # Set up the 4 inputs necessary...
        self.input_words = Input(shape=(self.hp_set.role_vocab_count - 1), dtype=tf.uint32, name='Input Words')
        self.input_roles = Input(shape=(self.hp_set.role_vocab_count - 1), dtype=tf.uint32, name='Input Roles')
        self.target_word = Input(shape=(1,), dtype=tf.uint32, name='Target Word')
        self.target_role = Input(shape=(1,), dtype=tf.uint32, name='Target Role')
        # Add the embedding layers for the input words and roles
        self.word_embedding = Embedding(input_dim=self.hp_set.word_vocab_count,
                                        output_dim=self.hp_set.word_embedding_dimension,
                                        embeddings_initializer=glorot_uniform(), name='Word Embedding',
                                        trainable=not self.hp_set.freeze_word_embeddings)
        self.role_embedding = Embedding(input_dim=self.hp_set.role_vocab_count,
                                        output_dim=self.hp_set.role_embedding_dimension,
                                        embeddings_initializer=glorot_uniform(), name='Role Embedding',
                                        trainable=not self.hp_set.freeze_role_embeddings)
        # Make a mask for the embedding to zero out the embedding values
        # for where the word is missing. This is the cleanest solution for
        # right now, but open to more elegant solutions in the future...
        self.embedding_mask = Lambda(lambda x: tf.cast(
            tf.expand_dims(
                tf.not_equal(x, self.hp_set.missing_word_id, name='Missing Word Reverse Mask'),
                axis=-1, name='Expand'),
            dtype=tf.float32, name='Cast to Float'), name='Create Mask for Missing Words')

        self.word_embed_multi = Multiply(name='Apply Mask to Word')
        self.role_embed_multi = Multiply(name='Apply Mask to Role')
        # Dropout layers for both word and role
        self.word_dropout = Dropout(self.hp_set.dropout_rate)
        self.role_dropout = Dropout(self.hp_set.dropout_rate)
        # Concatenation of embeddings layer
        self.embedding_concat = Concatenate(name='Combine Embeddings')
        # Next, is the residual block, which takes the combined embeddings and
        # adds it to a PReLU forwarded version of itself
        self.lin_proj1 = Dense(self.hp_set.word_embedding_dimension, activation='linear',
                               use_bias=False, name='Linear Proj 1')
        self.prelu = PReLU(alpha_initializer='ones', name='PReLU')
        self.lin_proj2 = Dense(self.hp_set.word_embedding_dimension, activation='linear',
                               use_bias=False, name='Linear Proj 2')
        self.prelu_proj_add = Add(name='Residual')
        self.average_across_input = Lambda(lambda x: tf.reduce_mean(x, axis=1), name='Context Embedding')
        # Finally we need to produce the embeddings for the target role and word...
        # Our target word and role will be (batch_size, 1, 512)...
        # Note that embeddings cross as inputs into the other target e.g.
        # the target word embedding inputs to the ROLE and vice versa.
        self.target_word_embedding = Embedding(self.hp_set.word_vocab_count, self.hp_set.n_factors_cls,
                                               embeddings_initializer=glorot_uniform(),
                                               name='Target Word Embedding')
        self.target_role_embedding = Embedding(self.hp_set.role_vocab_count, self.hp_set.n_factors_cls,
                                               embeddings_initializer=glorot_uniform(),
                                               name='Target Role Embedding')
        # Create two dropout layers for both, and a reshape layer as there is an extra dimension.
        self.target_word_drop = Dropout(self.hp_set.dropout_rate, name='Target Word Dropout')
        self.target_role_drop = Dropout(self.hp_set.dropout_rate, name='Target Role Dropout')
        self.target_word_reshape = Reshape((self.hp_set.n_factors_cls,), name='Reshape Target Word Embedding')
        self.target_role_reshape = Reshape((self.hp_set.n_factors_cls,), name='Reshape Target Role Embedding')
        # Create two weighted context embeddings that will pass into each target embedding
        self.weight_context_word = Dense(self.hp_set.n_factors_cls, activation='linear', use_bias=False,
                                         name='Weighted Context 1')
        self.weight_context_role = Dense(self.hp_set.n_factors_cls, activation='linear', use_bias=False,
                                         name='Weighted Context 2')
        # Create two Multiply layers for when we multiply the weighted context with the embeddings...
        # Remember to cross! These will be enforced during call.
        self.target_word_hidden = Multiply(name='Cont * TRE')  # target ROLE embedding for target word hidden
        self.target_role_hidden = Multiply(name='Cont * TWE')  # target WORD embedding for target role hidden

        ### FINALLY THE OUTPUTS
        self.target_word_output = Dense(self.hp_set.word_vocab_count, activation='softmax', name='Word Output')
        self.target_role_output = Dense(self.hp_set.role_vocab_count, activation='softmax', name='Role Output')

    def call(self, inputs, training=None, mask=None):
        # WE ASSUME THAT THE INPUTS ARE ORDERED LIKE SO:
        # [input_words, input_roles, target_word, target_role]
        # Though of course, when the Model is defined and data is fed through, it is much easier
        # to simply provide a tf.dataset where the input layer names map properly...
        input_words, input_roles, target_word, target_role = tuple(inputs)
        # Pass inputs through embedding...
        word_embedding_out = self.word_embedding(input_words)
        role_embedding_out = self.role_embedding(input_roles)
        # Now zero out the missing word embeddings if need be.
        if self.hp_set.zero_out_miss_embedding:
            mask = self.embedding_mask(input_words)
            # Update the embeddings
            word_embedding_out = self.word_embed_multi([word_embedding_out, mask])
            role_embedding_out = self.role_embed_multi([role_embedding_out, mask])
        # Apply dropout (obviously if dropout_rate = 0, then this layer does nothing...
        # ...but our default is nonzero...
        word_embedding_out = self.word_dropout(word_embedding_out)
        role_embedding_out = self.role_dropout(role_embedding_out)
        # Concatenate the two embeddings
        total_embeddings = self.embedding_concat([word_embedding_out, role_embedding_out])
        # Pass the total embeddings through the residual block next
        residual = self.prelu_proj_add([
            total_embeddings,
            self.lin_proj2(self.prelu(self.lin_proj1(total_embeddings)))
        ])
        context_embedding = self.average_across_input(residual)
        # Create target role and word embeddings multiplied with the context.
        # Note the crossing of inputs into the other's hidden layer.
        twh_out = self.target_word_hidden([
            self.weight_context_word(context_embedding),
            self.target_role_reshape(self.target_role_drop(self.target_role_embedding(target_role)))
        ])
        trh_out = self.target_role_hidden([
            self.weight_context_role(context_embedding),
            self.target_word_reshape(self.target_word_drop(self.target_word_embedding(target_word)))
        ])
        # Forward through the output layers and we are done!
        tw_out = self.target_word_output(twh_out)
        tr_out = self.target_role_output(trh_out)
        return tw_out, tr_out
