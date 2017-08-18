#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a sample script to demonstrate how to use LNP module in ncupy

@author: juniti-y
"""

######################################################################
###
### Import modules
###
######################################################################

import numpy as np           # For numpy
from ncupy import lnp        # For lnp

######################################################################
###
### Parameters for data generation
###
######################################################################

# Number of samples per class
N = 100

# Dimensionality of predictors
P = 2

# Difference between means of two classes
MU1 = 1.0

# Variance of each classes
SGM = 0.5

######################################################################
###
### Generate the training data
###
######################################################################


# Draw class-1 data from N(mu_diff,sgm)
X1 = SGM * np.random.randn(N, P) + MU1
# Add the class label to each sample
Y1 = np.ones((N, 1))
# Draw class-0 data from N(0,sgm)
X0 = SGM * np.random.randn(N, P)
# Add the class label to each sample
Y0 = np.zeros((N, 1))
# Concatenate all samples
X = np.r_[X0, X1]
Y = np.r_[Y0, Y1]


######################################################################
###
### Constructing GLM that fits the training data
###
######################################################################

# Create the object of GLM
MODEL = lnp.LNP()

# Change the options of the object
MODEL.set_options(method='MLE2', delta_t=0.001)

# Run the fitting algorithm
MODEL.fit(X, Y)

# Predict the responses to given predictors
PY = MODEL.predict(X)
