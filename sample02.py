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

import numpy as np               # For numpy
from ncupy import lnp            # For lnp
import matplotlib.pyplot as plt  # For pyplot

######################################################################
###
### Parameters for data generation
###
######################################################################

# Sampling condition
MAX_T = 100.0    # Duration (unit: [sec.])
DELTA_T = 0.001   # Sampling interval (unit: [sec.])

# Signal property
FR1 = 13.0
FR2 = 17.0

# Composition Weight
W1 = 1.0
W2 = -1.0

# Baseline
W0 = 2.0

# Window size for moving average
MV_W = 10

######################################################################
###
### Generate the training data
###
######################################################################

# Generate the time bins
T_BIN = np.linspace(0.0, MAX_T, np.int(np.round(MAX_T/DELTA_T))+1)

# Memory allocation of response
# Y = np.zeros((T_BIN.size, 1))

# Generate the sin signal 1
X1 = np.sin(2 * np.pi * T_BIN/FR1)

# Generate the sin signal 2
X2 = np.sin(2 * np.pi * T_BIN/FR2)

# Generate the composition signal
Z = W1*X1 + W2*X2

# Compute the firing rate (Unit: [sec.])
R = np.exp(Z + W0)

# Compute the spike probability for each bin
S_P = DELTA_T * R

# Generate the random number that follows the uniform dist.
RND = np.random.rand(np.size(T_BIN))

# Generate the spike
S = np.double(RND < S_P)

# Get the moving average of S
CEF_AVG = np.ones(MV_W)/MV_W
ES = np.convolve(S, CEF_AVG, 'same')

# Compute the empirical firing rate
EFR = ES/(DELTA_T*MV_W)

# Generate the data matrix for predictors
X = np.c_[np.reshape(X1, (X1.size, 1)), np.reshape(X2, (X2.size, 1))]

# Generate the data matrix for responses
Y = np.reshape(S, (S.size, 1))

######################################################################
###
### Constructing LNP model that fits the training data
###
######################################################################

# Create the object of LNP class
MODEL = lnp.LNP()

# Change the options of the object
MODEL.set_options(method='MLE2', delta_t=DELTA_T)

# Run the fitting algorithm
MODEL.fit(X, Y)

# Predict the responses to given predictors
PY = MODEL.predict(X)

######################################################################
###
### Compare the true process and its estimation (just for demonstration)
###
######################################################################

plt.plot(T_BIN, R, T_BIN, PY, T_BIN, EFR)
plt.title('Comparison among true, model, empirical')
plt.legend(['true', 'model', 'empirical'])
