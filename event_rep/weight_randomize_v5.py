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

The reason this file is one-off is because the method by which the weight transfer
happens means we need to access to each layer's variable. This differs from model to model.
Ideally, each model class would have a method which creates these two extra models.
"""

