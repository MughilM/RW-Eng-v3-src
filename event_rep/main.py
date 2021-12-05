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
import datetime
import json
import logging
import os.path
import shutil
import sys
import time
from typing import Dict, Type

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

from model_implementation.architecture.models import *
from model_implementation.core.batcher import WordRoleWriter
from model_implementation.core.color_logging import ColorHandler
from model_implementation.core.roles import *
from model_implementation.core.callbacks import MetricCallback
from evaluation.tasks import CorrelateTFScores, BicknellTask, GS2013Task

logger = logging.getLogger(__name__)

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
hp_set = HyperparameterSet(os.path.join(SRC_DIR, 'model_implementation/architecture/hp/default_params_all.json'))
# This is a dictionary which maps the user-provided parameter of the model to
# the corresponding class
# TODO: Add models here as necessary, as the argument enforcement is on the keys
PARAM_TO_MODEL: Dict[str, Type[MTRFv4Res]] = {
    'v4': MTRFv4Res,
    'v5': MTRFv5Res,
    'v6': MTRFv6Res
}

# Make the directories if they don't already exist.
for directory in [EXPERIMENT_DIR, PRETRAINED_DIR, CSV_PIECE_PATH, DATA_PATH]:
    os.makedirs(directory, exist_ok=True)

# This the role set we will be using...
ROLE_SET = Roles2Args3Mods

# The list of all evaluation tasks
ALL_EVAL_TASKS = ['pado', 'mcrae', 'greenberg', 'bicknell', 'gs', 'ferretti_instrument', 'ferretti_location']

# Start the clock, this allows all the methods to access
# the time...
total_time_start = datetime.datetime.now(datetime.timezone.utc)


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


def train_test_eval(model_name,
                    experiment_name,
                    train_dataset: tf.data.Dataset,
                    vali_dataset: tf.data.Dataset,
                    test_dataset: tf.data.Dataset,
                    load_previous: bool):
    # Make the model object!
    model_obj: MTRFv4Res = PARAM_TO_MODEL[model_name](hp_set, pretrained_emb_dir=PRETRAINED_DIR)
    model = model_obj.build_model()
    logger.info('Clean model summary:')
    # Extra parentheses for build() because input_shapes are not required.
    model.summary()
    # Same model image to experiment directory.
    os.makedirs(os.path.join(EXPERIMENT_DIR, experiment_name), exist_ok=True)

    model_artifact_dir = os.path.join(EXPERIMENT_DIR, experiment_name)
    checkpoint_dir = os.path.join(EXPERIMENT_DIR, experiment_name, 'checkpoints')
    past_metrics = None
    initial_epoch = 0
    if os.path.exists(model_artifact_dir):
        if load_previous:
            logger.info(f'Attempting to continue train from a checkpoint at {checkpoint_dir}...')
            if not os.path.exists(checkpoint_dir):
                logger.error('No checkpoints present! Quitting...')
                sys.exit(1)
            # TODO: Add reading from a description/metrics file...
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            logger.info(f'Found latest checkpoint {latest_checkpoint}! Loading weights...')
            model.load_weights(latest_checkpoint)
            with open(os.path.join(model_artifact_dir, 'metrics.json'), 'r') as f:
                past_metrics = json.load(f)
            initial_epoch = past_metrics['total_epochs'] + 1
        # If the path exists, but we are not loading the previous,
        # then delete whatever is there...
        else:
            shutil.rmtree(model_artifact_dir)
            os.makedirs(model_artifact_dir, exist_ok=True)
    # Put model image in the model artifacts for reference afterwards.
    tf.keras.utils.plot_model(model, to_file=os.path.join(model_artifact_dir, f'{model_name}.png'),
                              show_shapes=True)
    # Make callbacks...
    checkpointer = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'cp_{epoch:03d}.ckpt'),
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=0,
                                   save_weights_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1)
    nanChecker = TerminateOnNaN()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-3)
    metric_callback = MetricCallback(model_odj=model, save_dir=model_artifact_dir, save_freq=1,
                                     past_metrics=past_metrics)

    total_time_start = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f'TRAINING STARTED AT {total_time_start} UTC...')

    # COMPILE AND FIT OUR MODEL!
    model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_dataset,
              epochs=initial_epoch + args.epochs,
              workers=1,
              verbose=2,
              initial_epoch=initial_epoch,
              validation_data=vali_dataset,
              callbacks=[checkpointer, stopper, nanChecker, reduce_lr, metric_callback])

    # Report time taken, and metrics on the testing dataset...
    end_time = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f'TRAINING ENDED AT {end_time} UTC...')

    # For testing, we need to load the latest checkpoint,
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    logger.info(f'Testing with latest checkpoint: {latest}')
    model.load_weights(latest)
    # Evaluate on test set, and create dictionary for testing metrics
    test_metrics = model.evaluate(test_dataset)
    test_keys = ['Loss', 'Role Loss', 'Word Loss', 'Role Accuracy', 'Word Accuracy']
    test_metrics_json = {k: v for k, v in zip(test_keys, test_metrics)}
    # Also save the total time...
    test_metrics_json['Total Training Time'] = str(end_time - total_time_start)
    # Save the test metrics in the corresponding experiment folder
    with open(os.path.join(model_artifact_dir, 'final_metrics.json'), 'w') as f:
        json.dump(test_metrics_json, f, indent=2)
    logger.info('Testing done. To resume training, please use the checkpoint directory.')
    logger.info(f'EXPERIMENT AND HYPERPARAMETERS SAVED AT {model_artifact_dir}.\n')

    return model, model_artifact_dir


def run_thematic_evaluation(tasks: list, model, experiment):
    logger.info(f'Evaluation Tasks to run: {tasks}')
    for task in tasks:
        logger.info(f'Running {task}...')
        if task == 'bicknell':
            evaluator = BicknellTask(SRC_DIR, EXPERIMENT_DIR, model, experiment)
        elif task == 'gs':
            # Load the context embedding for the GS2013 task!!
            evaluator = GS2013Task(SRC_DIR, EXPERIMENT_DIR, model, experiment, get_embedding=True)
        else:
            evaluator = CorrelateTFScores(SRC_DIR, EXPERIMENT_DIR, model, experiment, task)
        evaluator.run_task()


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
    parser.add_argument('--load_previous', dest='load_previous', action='store_true', default=False,
                        help='Whether to load a previously trained model.')
    parser.add_argument('--batch_size', type=int, default=hp_set.batch_size,
                        help='The batch size for training')
    parser.add_argument('--dropout_rate', type=float, default=hp_set.dropout_rate,
                        help='The dropout rate for the Dropout layers. Should be between 0 and 1.')
    parser.add_argument('--embedding_type', choices=['random', 'glove', 'fasttext', 'w2v'], default='random',
                        help='The type of embedding to use for WORDS. Can be random, or a type of'
                             'pretrained embedding (GloVe (from spaCy), Fasttext, Word2Vec)')
    parser.add_argument('--oov', choices=['0', 'avg'], default='avg',
                        help='The type of out-of-vocabulary initialization for the pretrained embeddings. '
                             'Either initialize OOV to 0 vectors, or to the average of the rest. Does not '
                             'get used for random embeddings, since there are no OOV. Default avg.')
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
    parser.add_argument('--word_role_aggregation', choices=['multiply', 'concat', 'null'], default='multiply',
                        help='The aggregation method to combine the roles with the words. "multiply" will be applied'
                             'if the role and word have the same embedding dimension. Otherwise, they will '
                             'concatenated. The "drop" option means to completely REMOVE the roles from '
                             'aggregation i.e. the model will only use the words.')
    parser.add_argument('--use_ortho_roles', dest='use_ortho_roles', action='store_true', default=False,
                        help='Whether to initialize the role embedding with orthogonal vectors. Completely '
                             'depends on the number of roles in the model. Default False.')
    # Extra parameter, role set, generally not touched at all.
    parser.add_argument('--role_set', type=Roles, default=Roles2Args3Mods,
                        help='The role set to use. Default Roless2Args3Mods')
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument('--do_eval', action='store_true', default=False,
                            help='If specified, runs evaluation. See help for evaluation task options.')
    eval_group.add_argument('--eval_only', dest='eval_only', action='store_true', default=False,
                            help='If specified, then DOES NOT RUN TRAINING! It will ONLY run the evaluation '
                                 'tasks. In this way it differs from do_eval, which allows for both training '
                                 'and evaluation in the same program run.')
    # eval_task_group = parser.add_mutually_exclusive_group()
    parser.add_argument('--evaluation_tasks', choices=ALL_EVAL_TASKS + ['all'],
                        nargs='*', default=['all'],
                        help='The specific evaluation tasks to run. Must specify at least one.')
    # eval_task_group.add_argument('--run_all_tasks', action='store_true', default=False,
    #                              help='If specified, runs ALL thematic fit evaluation tasks.')

    args = parser.parse_args()
    # Check the choices for word_role_aggregation...
    if (args.word_role_aggregation == 'multiply') and (args.word_embedding_dimension != args.role_embedding_dimension):
        logger.warning('You chose multiply for the aggregation method, but the word and role dimensions '
                       'are different. Aggregation method will change to "concat".')
        args.word_role_aggregation = 'concat'
    # Small checks for contradictions.
    if args.eval_only:
        if args.experiment_name == '' or not os.path.exists(os.path.join(EXPERIMENT_DIR, args.experiment_name)):
            parser.error('You specified to only run evaluation, but did not provide an experiment name or '
                         'the experiment does not exist.')
            sys.exit(1)
        if args.load_previous:
            parser.error('You specified to only run evaluation, but also to load a previous model for '
                         'continued training.')
            sys.exit(1)
        if args.evaluation_tasks == 'none':
            parser.error('You specified to only run evaluation, but no evaluation tasks were specified.')
            sys.exit(1)
    # Special case, make sure specific evaluation tasks if evaluation is enabled and
    # run_all_tasks is False.
    if (args.do_eval or args.eval_only) and 'all' in args.evaluation_tasks and len(args.evaluation_tasks) > 1:
        parser.error('Evaluation is enabled, but the "all" option is included along with other'
                     'evaluation tasks. Please double check the task list.')
        sys.exit(1)

    if args.experiment_name == '':
        experiment_name = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}_{args.model_name}_" \
                          f"{args.data_version}"
    else:
        experiment_name = args.experiment_name
    # Convert arguments into dictionary
    opts = vars(args)
    # Don't run the training if we are only evaluating
    if not args.eval_only:
        hp_set.update_parameters(opts)
        # Next, we also need to read in the description file from the input data directory
        hp_set.read_description_params(os.path.join(DATA_PATH, args.data_version, 'description'))
        # Small argument check. If we are using orthogonal roles, then change value
        # of role embedding dimension to match the input role size
        # (This also helps adjust some other layers, other than role embedding, during build)
        if hp_set.use_ortho_roles:
            hp_set.role_embedding_dimension = hp_set.role_vocab_count
        train, vali, test = create_dataset_objects(args.data_version)
        model, model_artifact_dir = train_test_eval(args.model_name, experiment_name,
                                                    train, vali, test, args.load_previous)

        # Update the output directory for our hyperparameters,
        # and write the JSON to the model output directory.
        hp_set.set_output_dir(model_artifact_dir)
        hp_set.write_hp()
    # Run the evaluation if one of the parameters is specified
    if args.do_eval or args.eval_only:
        if args.evaluation_tasks == ['all']:
            run_thematic_evaluation(ALL_EVAL_TASKS, args.model_name, experiment_name)
        else:
            run_thematic_evaluation(args.evaluation_tasks, args.model_name, experiment_name)

    end_time = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f'PROGRAM ENDED AT {end_time} UTC...')
    logger.info(f'TOTAL TIME: {str(end_time - total_time_start)}\n')
