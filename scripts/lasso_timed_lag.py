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
    #print(y.shape)
    
    
    #create Am matrix, (essentially the X matrix in lasso regression)
    Am_kernelized = np.empty(shape=[M-max_time_index, n_lags*N])
    for i in range(n_lags):
        #find kernel matrix for X design matrix (different for each lag)
        dimen = N #(M-lags[i]+1) - (max_time_index-lags[i])
        #time_stamp = np.arange(0,dimen).reshape(dimen,1)
        #FIX time_stamp
        #time_stamp = time_index[0,np.where(time_index>lags[i])] - lags[i]
        time_stamp = time_index - lags[i]
        #time_stamp = time_stamp.reshape((1,time_stamp.shape[0]))
      #  print(i)
      #  print(time_stamp)
        kernel = gaussian_kernel(time_data, time_stamp, sigma)
      #  print(Am_kernelized.shape)
        #apply normalized kernal matrix to Am
        kernel = kernel[max_time_index:M,:]
        #print(kernel.shape)
        #print(kernel[0:6,0:6])
        total = np.sum(kernel, axis=1)
            #standardize matrix (NO LOOP)
        #print("Max Time index:",max_time_index)
        #print("Total shape:",total.shape)
        #print("Total head:",total[0:6])

        Am_kernelized[:,i*N:(i+1)*N] = np.matmul(kernel,X_design.T)/total[:,None]
        #Am_kernelized[:,i*N:(i+1)*N] = np.matmul(Am[:,i*N:(i+1)*N], X_kernel[:,i*N:(i+1)*N])
    #print(X_kernel[0:5,0:5])
    print("Am",Am_kernelized[0:6,0:6])
    print(Am_kernelized[0:6,100:106])
    print(Am_kernelized[0:6,200:206])
    print("Am shape: ", Am_kernelized.shape)
   
   #compare with MATLAB Am values
    Am_mat = pd.read_csv("./scripts/Am_matlab.csv", header=None)
    Am_mat = Am_mat.to_numpy()
    #Mat Am structured differently, convert to python arrangement for comparison
    Am_arr = Am_mat[:,n_lags-1::n_lags]
    for i in range(n_lags-1)[::-1]:
      Am1 = Am_mat[:,i::n_lags]
      Am_arr = np.concatenate([Am_arr,Am1],axis=1)
     
    #print("Am_arr: ",Am_arr[0:6,0:6])
    #print("Am_arr shape: ", Am_arr.shape) 
      
    Am_diff = np.abs(Am_kernelized-Am_arr)
    print("difference in Am between Mat & Python: ", Am_diff[0:6,0:6])

    
    #run glmnet regression
    bm = glmnet_lasso(y, Am_kernelized, lambda_val)
    
    end_time = time.time()
    run_time = end_time - start_time
    print("program run time: ", run_time)
    
    return bm
