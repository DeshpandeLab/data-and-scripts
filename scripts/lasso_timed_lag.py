import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy
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
        dimen = 100 #(M-lags[i]+1) - (max_time_index-lags[i])
        time_stamp = np.arange(0,dimen).reshape(dimen,1)
        X_kernel = gaussian_kernel(time_data, time_stamp, sigma)
        
        #apply normalized kernal matrix to Am
        X_kernel_sec = X_kernel[:,max_time_index-lags[i]:M-lags[i]+1].transpose()
        
        Am_kernelized[:,i*N:(i+1)*N] = np.matmul(Am[:,i*N:(i+1)*N], X_kernel[:,i*N:(i+1)*N])
    
    #print(Am_kernelized.shape)
    #print(y.shape)
    
    
    #run glmnet regression
    bm = glmnet_lasso(y, Am_kernelized, 0.1)
    
    
    return bm
