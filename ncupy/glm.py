#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:08:42 2017

@author: juniti-y
"""

## Import modules
import time
import numpy as np

class Logit:
    """
    Class: Logit
    =====
    
    Provides tools for estimation and prediction using Logit model
    """
    def __init__(self):
        """
        Constructor
        """
        ### For training data set
        self.xdata = None  ## x~=(x,1)
        self.ydata = None  ## y

        ### For estimation options
        self.__method = 'MLE'   ## Method for parameter estimation
        self.__max_iter = 1000  ## maximum number of iterations
        self.__th_imp = 1.0e-4  ## threshould of improvement
        self.__monitor = True   ## feedback or not

        ### For estimation results
        self.weight = None ## w~=(w,b)
        self.llh = None    ## log-likelihood


    def set_options(self, method='MLE', max_iter=1000, th_imp=1.0e-4, monitor=True):
        """
        Set options for estimation algorithms
        """
        self.__method = method
        self.__max_iter = max_iter
        self.__th_imp = th_imp
        self.__monitor = monitor

    def status(self):
        """
        To monitor the status of the model
        """
        print(self.xdata)
        print(self.ydata)
        
        return
    

    def fit(self, xdata, ydata):
        """
        Estimate the model parameters using the training data set (x,y)
        """

        # Store the training data into private variables
        self.xdata = np.c_[xdata, np.ones((xdata.shape[0], 1))]
        self.ydata = ydata

        # Intialize the parameters by zero-vector
        weight = np.zeros((self.xdata.shape[1], 1))

        # Evaluate the initial log-likelihood
        llh = self.__log_likelihood(weight)

        # Feedback the status to users
        if self.__monitor:
            print('========== Start: Maximum likelihood Estimation ==========')
            start_time = time.time()

        # Initialize the counter for judging the convergence condition
        cnt = 0
        for k in range(self.__max_iter):
            # Evaluate the gradient of log-likelihood
            grad_w = self.__grad_log_likelihood(weight)
            # Evaluate the Hessian of log-likelihood
            hes_w = self.__hessian_log_likelihood(weight)
            # Compute (Hessian)^{-1}*(gradient)
            dldw = np.linalg.solve(hes_w, grad_w)
            eta = 1.0
            # Compute the new parameter with step-size adjustment
            while True:
                # tentatively update the parameters
                w_new = weight - eta * dldw
                # Evaluate the log-likelihood of the updated parameter
                llh_new = self.__log_likelihood(w_new)
                if llh_new >= llh:
                    break
                else:
                    eta = 0.5*eta
            # Check if improvement threshold is satisfied
            if llh_new - llh < self.__th_imp:
                cnt = cnt + 1
            # Update the parameter & its log-likelihood
            llh = llh_new
            weight = w_new
            # Feedback the status to users
            if self.__monitor:
                print('k = %d:\t log-likelihood = %.4f' % (k, llh))
            # Escape from the loop if the stopping considion is satisfied
            if cnt >= 5:
                break

        # Feedback the status to users
        if self.__monitor:
            end_time = time.time()
            print('========== End ==========')
            print('processing time = {:.4}'.format(end_time - start_time) + '[sec]')
        # Store the estimation results
        self.weight = weight
        self.llh = llh


    def predict(self, xdata):
        """
        Predict the responses for given predictors
        """
        xpred = np.c_[xdata, np.ones((xdata.shape[0], 1))]
        zpred = np.dot(xpred, self.weight)
        ypred = logit(zpred)
        return ypred

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
        llh = np.dot(ydata[:, 0], log_mu1[:, 0]) + np.dot(1.0-ydata[:, 0], log_mu0[:, 0])
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
        lmbd = - np.tile(ypred * (1.0-ypred), (1, wdim))
        lmbd_x = lmbd * xdata
        hes_w = np.dot(xdata.T, lmbd_x)
        return hes_w
#
#
#
def logit(value):
    """
    Compute the output of logistic regression: logit(Xw + b)
    """
    return 1.0/(1.0+np.exp(-value))
