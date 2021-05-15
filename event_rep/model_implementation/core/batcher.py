"""
File: event_rep/model_implementation/core/batcher.py
Author: Mughilan Muthupari
Creation Date: 2021-05-15

This module holds our batcher classes, which handle all of the data pipelining from
data reading, transforming, to feeding them into the model. The core functionality revolves
around the tf.Dataset, which can be fed directly model.fit(), and as a result, Tensorflow
can optimize and parallel process the data preparation and model training to save time.
"""