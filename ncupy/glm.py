#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:08:42 2017

@author: juniti-y
"""

## Import modules
import numpy as np

class Logit:
    """
    Explanation for class
    """
    def __init__(self):
        """
        Constructor
        """
        ### Initialize the private variables
        self.xdata = None
        self.ydata = None
        self.weight = None
        self.bias = None
        self.llh = None
        return
    def status(self):
        """
        To monitor the status of the model
        """
        print(self.xdata)
        print(self.ydata)
    def fit(self, xdata, ydata):
        """
        Fit the model parameters to the training data set (x,y)
        """
        ####################################################################
        # Get the dimensionality of training data set
        ####################################################################
        ndata = np.size(xdata, axis=0)        
        ####################################################################
        # Store the training data into private variables
        ####################################################################
        self.xdata = np.c_[xdata, np.ones((ndata, 1))]
        self.ydata = ydata
        ####################################################################
        # Intialize the parameters at random
        ####################################################################
        wdim = np.size(self.xdata, axis=1)
        weight = np.zeros((wdim,1))
        
        for k in range(10):
            ####################################################################
            # Evaluate the log-likelihood
            ####################################################################
            llh = self.__log_likelihood(weight)
            print(llh)
            ####################################################################
            # Evaluate the gradient of log-likelihood
            ####################################################################
            grad_w = self.__grad_log_likelihood(weight)
            ####################################################################
            # Evaluate the Hessian of log-likelihood
            ####################################################################
            hes_w = self.__hessian_log_likelihood(weight)
            ####################################################################
            # Update the weight
            ####################################################################
            weight = weight - np.linalg.solve(hes_w, grad_w)
            #np.dot(np.linalg.inv(hes_w),grad_w)
            print(weight)
            #weight = weight + 0.001*grad_w

            
        
    def __log_likelihood(self, weight):
        """
        Compute the log-likelihood
        """
        xdata = self.xdata
        ydata = self.ydata
        zdata = np.dot(xdata, weight)
        log_mu1 = -np.log(1.0+np.exp(-zdata))
        log_mu0 = -np.log(1.0+np.exp(+zdata))
        z_pos_idx = np.where(zdata > 0)
        z_pos_value = zdata[z_pos_idx]
        z_neg_idx = np.where(zdata < 0)
        z_neg_value = zdata[z_neg_idx]
        log_mu1[z_neg_idx] = z_neg_value - np.log(1.0+np.exp(+z_neg_value))
        log_mu0[z_pos_idx] = -z_pos_value - np.log(1.0+np.exp(-z_pos_value))
        llh = np.dot(ydata[:,0], log_mu1[:,0])+ np.dot(1.0-ydata[:,0], log_mu0[:,0])
        return llh
    def __grad_log_likelihood(self, weight):
        """
        Compute the gradient of the log-likelihood
        """
        xdata = self.xdata
        ydata = self.ydata
        zdata = np.dot(xdata, weight)
        ypred = logit(zdata)
        grad_w = np.dot(xdata.T, ydata-ypred)
        return grad_w
    def __hessian_log_likelihood(self, weight):
        """
        Compute Hessian matrix (second-order derivative) of the log-likelihood
        """
        wdim = np.size(weight, axis=0)
        xdata = self.xdata
        zdata = np.dot(xdata, weight)
        ypred = logit(zdata)
        lmbd = - np.tile(ypred * (1.0-ypred),(1,wdim))
        lmbd_x = lmbd * xdata
        hes_w= np.dot(xdata.T, lmbd_x)
        return hes_w
#
#
#
def logit(value):
    """
    Compute the output of logistic regression: logit(Xw + b)
    """
    return 1.0/(1.0+np.exp(-value))
