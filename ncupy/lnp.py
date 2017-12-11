#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:55:28 2017

@author: juniti-y
"""

## Import modules
import time
import numpy as np
import scipy as sp

class LNP:
    """
    Class: LNP
    =====

    Provides tools for estimation and prediction
    using linear nonlinear poisson (LNP) cascade model
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
        self.__dt = 1.0         ## resolution for time-discretization
        self.__verbose = True   ## feedback or not


        ### For estimation results
        self.weight = None     ## w~=(w,b)
        self.llh = None        ## log-likelihood
        self.pnlt = None       ## Penalty term
        self.err = None        ## Empirical error for training data
        self.__converge = None ## For judging the convergence
        self.__ptime = None    ## processing time for estimation


    def set_options(self, method='MLE', max_iter=1000, th_imp=1.0e-8,
                    lmbd=1.0e-4, delta_t=1.0, verbose=True):
        """
        Set options for estimation algorithms
        """
        self.__method = method
        self.__max_iter = max_iter
        self.__th_imp = th_imp
        self.__lambda = lmbd
        self.__dt = delta_t
        self.__verbose = verbose

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
            print('- Coeffients = %s' % str(self.weight[:-1]))
            print('- Intercept: %.4f' % self.weight[-1])
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

    def fit(self, xdata, ydata, dist=None):
        """
        Estimate the model parameters using the training data set (xdata, ydata)
        """
        ### Selecting the estimation algorithm
        if self.__method == 'MLE':
            self.run_mle(xdata, ydata, dist)
        elif self.__method == 'MLE2':
            self.run_mle2(xdata, ydata, dist)
        return

    def run_mle(self, xdata, ydata, dist=None):
        """
        Estimate the model parameters that fit the training data
        (xdata,ydata) by maximum likelihood estimation method.
        This is implemented by the special case of L2-regularization
        with lambda = 0.0
        """
        self.set_options(lmbd=0.0)
        self.run_mle2(xdata, ydata, dist)
        return

    def run_mle2(self, xdata, ydata, dist=None):
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

        # Set [dist] by one if unset
        if dist is None:
            dist = np.ones(self.xdata.shape[1]-1)

        # Intialize the parameters by zero-vector
        weight = np.zeros(self.xdata.shape[1])

        # Evaluate the initial log-likelihood
        llh = self.__log_likelihood(weight)
        # Evaluate the initial l2-penalty
        pnlt = 0.5 * lmbd * np.dot(dist, weight[:-1]*weight[:-1])

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
            grad_p = lmbd * np.append(dist, 0) * weight
            # Evaluate the gradient of error function
            grad_err = - grad_l/ndata + grad_p

            # Evaluate the Hessian of log-likelihood
            hes_l = self.__hessian_log_likelihood(weight)
            # Evaluate the Hessian of l2-penelty term
            hes_p = lmbd * np.diag(np.append(dist, 0))
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
                pnlt_new = 0.5 * lmbd * np.dot(dist, w_new[:-1]*w_new[:-1])
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
        Predict the instantaneous firing rate per unit time
        """
        # Construct x=(x,1)
        xpred = np.c_[xdata, np.ones((xdata.shape[0], 1))]
        # Compute z=<x,w>
        zpred = np.dot(xpred, self.weight)
        # Compute the firing rate: r = exp(z)
        ypred = np.exp(zpred)

        return ypred

    def get_fit_result(self):
        """
        Get the result of parameter fitting
        """
        return self.weight, self.llh, self.pnlt

    def __log_likelihood(self, weight):
        """
        Compute the log-likelihood
        """

        # Get the training data set
        xdata = self.xdata
        ydata = self.ydata

        # Get the time-resolution
        delta_t = self.__dt

        # Compute z=<x,w>
        zdata = np.dot(xdata, weight)
        # Compute the log of the expected num. of spikes: log(r) = z + log(dt)
        log_r = zdata + np.log(delta_t)
        # Compute the expected number of spikes: r = dt * exp(z)
        rval = np.exp(log_r)

        # Compute \sum_k y_k \ln r_k
        sum_ylr = np.dot(ydata, log_r)
        # Compute \sum_k r_k
        sum_r = np.sum(rval)
        # Compute \sum loggamma(y_k+1)
        sum_lgr = np.sum(sp.special.gammaln(ydata+1))

        # Compute the log-likelihood
        llh = sum_ylr - sum_r - sum_lgr

        return llh

    def __grad_log_likelihood(self, weight):
        """
        Compute the gradient of the log-likelihood
        """
        # Get the training data set
        xdata = self.xdata
        ydata = self.ydata

        # Get the time-resolution
        delta_t = self.__dt

        # Compute z=<x,w>
        zdata = np.dot(xdata, weight)
        # Compute the log of the expected num. of spikes: log(r) = z + log(dt)
        log_r = zdata + np.log(delta_t)
        # Compute the expected number of spikes: r = dt * exp(z)
        rval = np.exp(log_r)

        # Compute the gradient of log-likelihood
        grad_w = np.dot(ydata-rval, xdata)
        return grad_w

    def __hessian_log_likelihood(self, weight):
        """
        Compute Hessian matrix (second-order derivative) of the log-likelihood
        """
        # Get the training data set
        xdata = self.xdata

        # Get the time-resolution
        delta_t = self.__dt

        # Compute z=<x,w>
        zdata = np.dot(xdata, weight)
        # Compute the log of the expected num. of spikes: log(r) = z + log(dt)
        log_r = zdata + np.log(delta_t)
        # Compute the expected number of spikes: r = dt * exp(z)
        rval = np.exp(log_r)

        # Compute the Hessian of log-likelihood
        # wdim = np.size(weight, axis=0)
        wdim = weight.size
        r_diag = - np.matlib.repmat(rval, wdim, 1)
        # r_diag = - np.tile(rval, (1, wdim))
        r_x = r_diag.T * xdata
        #r_x = r_diag * xdata
        hes_w = np.dot(xdata.T, r_x)
        return hes_w
