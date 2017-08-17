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
        self.__th_imp = 1.0e-8  ## threshould of improvement
        self.__lambda = 1.0e-4  ## Regularization parameter
        self.__verbose = True   ## feedback or not


        ### For estimation results
        self.weight = None     ## w~=(w,b)
        self.llh = None        ## log-likelihood
        self.pnlt = None       ## Penalty term
        self.err = None        ## Empirical error for training data
        self.__converge = None ## For judging the convergence
        self.__ptime = None    ## processing time for estimation


    def set_options(self, method='MLE', max_iter=1000, th_imp=1.0e-8, lmbd=1.0e-4, verbose=True):
        """
        Set options for estimation algorithms
        """
        self.__method = method
        self.__max_iter = max_iter
        self.__th_imp = th_imp
        self.__verbose = verbose
        self.__lambda = lmbd

    def __start_verbose(self):
        """
        Report the start of estimation procedure the users
        """
        if self.__verbose:
            print('========== Start: Parameter estimation ==========')
            if self.__method == 'MLE':
                print('Method: Maximum Likelihood Estimation (MLE)')
            if self.__method == 'MLE2':
                print('Method: Maximum Likelihood Estimation (MLE)')
                str_add = '\t with L2-regularizer (lambda = '\
                + str(self.__lambda) + ')'
                print(str_add)
        return
    
    def __end_verbose(self):
        """
        Report the end of estimation procedure to users
        """
        if self.__verbose:
            print('========== End: Parameter estimation ==========')
            print('')
            print('========= Report of estimation result =========')
            print('- Convergence: %r' % self.__converge)
            print('- Processing time: {:.4}'.format(self.__ptime) + '[sec]')
            print('')
            print('- Coeffients = %s' % str(self.weight[:-1, 0]))
            print('- Intercept: %.4f' % self.weight[-1, 0])
            print('')
            print('- Empirical error: %.4e' % self.err)
            print('--- Normalized log-likelihood: %.4e' % (self.llh/self.xdata.shape[0]))
            print('--- Regularization term: %.4e' % self.pnlt)
            print('===============================================')
        return
        
    
    def status(self):
        """
        To monitor the status of the model
        """
        print(self.xdata)
        print(self.ydata)

    def fit(self, xdata, ydata):
        """
        Estimate the model parameters using the training data set (x,y)
        """
        ### Selecting the estimation algorithm
        if self.__method == 'MLE':
            self.run_mle(xdata, ydata)
        elif self.__method == 'MLE2':
            self.run_mle2(xdata, ydata)
        return

    def run_mle(self, xdata, ydata):
        """
        Estimate the model parameters that fit the training data
        (xdata,ydata) by maximum likelihood estimation method.
        This is implemented by the special case of L2-regularization
        with lambda = 0.0
        """
        self.set_options(lmbd=0.0)
        self.run_mle2(xdata, ydata)
        return

    def run_mle2(self, xdata, ydata):
        """
        Estimate the model parameters that fit the training data
        (xdata,ydata) by L2-reguralized maximum likelihood estimation method
        """

        # Get the number of training samples
        ndata = xdata.shape[0]

        # Store the training data into private variables
        self.xdata = np.c_[xdata.copy(), np.ones((ndata, 1))]
        self.ydata = ydata.copy()

        # Get the regularization parameter
        lmbd = self.__lambda

        # Intialize the parameters by zero-vector
        weight = np.zeros((self.xdata.shape[1], 1))
        # Evaluate the initial log-likelihood
        llh = self.__log_likelihood(weight)
        # Evaluate the initial l2-penalty
        pnlt = 0.5 * lmbd * np.dot(weight[:-1, 0], weight[:-1, 0])
        # Evaluate the regularized error function
        err = - llh/ndata + pnlt

        # Feedback the status to users
        self.__start_verbose()

        # Initialize the counter for judging the convergence condition
        self.__converge = False
        cnt = 0
        # Start the process timer
        start_time = time.time()
        for k in range(self.__max_iter):
            # Evaluate the gradient of log-likelihood
            grad_l = self.__grad_log_likelihood(weight)
            # Evaluate the gradient of l2-penalty term
            grad_p = lmbd * weight.copy()
            grad_p[-1] = 0.0
            # Evaluate the gradient of error function
            grad_err = - grad_l/ndata + grad_p

            # Evaluate the Hessian of log-likelihood
            hes_l = self.__hessian_log_likelihood(weight)
            # Evaluate the Hessian of l2-penelty term
            hes_p = lmbd * np.eye(weight.shape[0])
            hes_p[-1] = 0.0
            # Evaluate the Hessian of error function
            hes_err = - hes_l/ndata + hes_p

            # Compute (Hessian)^{-1}*(gradient)
            dw_vec = np.linalg.solve(hes_err, grad_err)
            eta = 1.0
            # Compute the new parameter with step-size adjustment
            while True:
                # tentatively update the parameters
                w_new = weight - eta * dw_vec
                # Evaluate the log-likelihood of the updated parameter
                llh_new = self.__log_likelihood(w_new)
                # Evaluate the l2-penalty of the updated parameter
                pnlt_new = 0.5 * lmbd * np.dot(w_new[:-1, 0], w_new[:-1, 0])
                # Evaluate the regularized error function after updating
                err_new = -llh_new/ndata + pnlt_new
                if err_new <= err:
                    break
                else:
                    eta = 0.5*eta
            # Check if improvement threshold is satisfied
            if err - err_new < self.__th_imp:
                cnt = cnt + 1
            # Update the parameter & its log-likelihood
            llh = llh_new
            pnlt = pnlt_new
            err = err_new
            weight = w_new
            # Feedback the status to users
            if self.__verbose:
                print('k = %d:\t Err = %.4e\t Log-likelihood = %.4e' % (k, err, llh))
            # Escape from the loop if the stopping considion is satisfied
            if cnt >= 5:
                self.__converge = True
                break

        # Stop the process timer
        end_time = time.time()
        self.__ptime = end_time - start_time

        # Store the estimation results
        self.weight = weight
        self.llh = llh
        self.pnlt = pnlt
        self.err = err

        # Feedback the status to users
        self.__end_verbose()


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
        llh = np.dot(ydata[:, 0], log_mu1[:, 0]) + \
        np.dot(1.0-ydata[:, 0], log_mu0[:, 0])
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

##############################################################################
##
## Utility functions
##
##############################################################################

def logit(value):
    """
    Compute the output of logistic regression: logit(Xw + b)
    """
    return 1.0/(1.0+np.exp(-value))
