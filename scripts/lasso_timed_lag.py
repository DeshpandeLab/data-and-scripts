import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy
import time
from gaussian_kernel import *
from glmnet_lasso import *


def lasso_timed_lag(expression_data, time_data, target_gene, lags, lambda_val, sigma):
    
    #INPUTS:
    #expression_data: NxM matrix of expression data of N genes and M cells, ordered temporally with index matching time_data
    #time_data: 1xM matrix of time points each expression data corresponds to index-wise
    #target_gene: integer between 0 and N-1 that represents the target gene's index in the expression_data matrix
    #lags: a list of integers of time lags to be included in the Am design matrix. Example: [5,10,50]
    #lambda_val: lambda value for lasso regression, adjusted based on data fit results
    #sigma: kernel smoothing parameter, set by user
    
    #OUTPUTS:
    #bm: nNx1 matrix of predicted coefficients, where n is the number of lags (i.e. len(lags)) and N genes
    
    start_time = time.time()

    X_design = np.asarray(expression_data)
    time_index = np.asarray(time_data)
    
    n_lags = len(lags)
    max_lag = max(lags)
    N = X_design.shape[0]
    M = X_design.shape[1]
    
    #find index of largest time lag (time>50, for instance)
    max_time_index_t = (np.where(time_index[0,:]>max_lag))[0]
    max_time_index=max_time_index_t[0]
    
    y = X_design[target_gene, max_time_index:] #dim: 1x(M-max_time_index+1)
    y = y.transpose() #dim: (M-max_time_index+1)x1
    #print(max_time_index)
    
    
    #create Am matrix, (essentially the X matrix in lasso regression)
    Am = np.empty(shape=[M-max_time_index, n_lags*N])
    Am_kernelized = np.empty(shape=[M-max_time_index, n_lags*N])
    #print("Am shape: ",Am_kernelized.shape)
    for i in range(n_lags):
        #find the non-smoothed version first
        Am[:,i*N:(i+1)*N] = X_design[:,max_time_index-lags[i]:M-lags[i]].transpose()
        
        #find kernel matrix for X design matrix (different for each lag)
        dimen = N #(M-lags[i]+1) - (max_time_index-lags[i])
        #time_stamp = np.arange(0,dimen).reshape(dimen,1)
        #FIX time_stamp
        time_stamp = time_index[0,np.where(time_index>lags[i])] - lags[i]
        #print(time_stamp)
        X_kernel = gaussian_kernel(time_data, time_stamp, sigma)
        #print(X_kernel[0,0])
        #print("kernel: ",np.where(X_kernel[0]>0.9)[0])
        #print(X_kernel[0,np.where(X_kernel[0]>0.9)])
        
        #apply normalized kernal matrix to Am
        X_kernel_sec = X_kernel[:,max_time_index-lags[i]:M-lags[i]]
        Am_kernelized[:,i*N:(i+1)*N] = np.matmul(X_kernel_sec, Am[:,i*N:(i+1)*N])
        #Am_kernelized[:,i*N:(i+1)*N] = np.matmul(Am[:,i*N:(i+1)*N], X_kernel[:,i*N:(i+1)*N])
    
    #print(Am_kernelized)
    #print("Am shape: ", Am_kernelized.shape)
    
    
    #run glmnet regression
    bm = glmnet_lasso(y, Am_kernelized, lambda_val)
    
    end_time = time.time()
    run_time = end_time - start_time
    print("program run time: ", run_time)
    
    return bm
