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