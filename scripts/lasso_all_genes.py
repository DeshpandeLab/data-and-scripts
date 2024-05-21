
import numpy as np
import pandas as pd
import scipy as scipy
import time
from lasso_timed_lag import *


# lasso_all_genes.py
# using lasso_timed_lag to build matrices for all genes in the dataset

def lasso_all_genes(expression_data, time_data, lags, lambda_val, sigma):

#INPUTS:
#expression_data: NxM matrix of expression data of N genes and M cells, ordered temporally with index matching time_data
#time_data: 1xM matrix of time points each expression data corresponds to index-wise
#lags: a list of integers of time lags to be included in the Am design matrix. Example: [5,10,50]
#lambda_val: lambda value for lasso regression, adjusted based on data fit results
#sigma: kernel smoothing parameter, set by user
    
#OUTPUTS:
#bm: a list of n NxN matrices of predicted coefficients, where n is the number of lags (i.e. len(lags)) and N genes, 
#    in the order of the list of lags provided as input
#  note: every column of bm represents the coefficients for a single gene, from 1 to N
#const: a numpy array of Nx1 constants from the lasso regression, where each row is the constant coef of 1 gene
    
    start_time_program = time.time()
    #create output list of matrices
    X_design = np.asarray(expression_data)
    N = X_design.shape[0]
    n_lags = len(lags)
    const = np.empty(shape=[N,1])
    bm_test = (np.empty(shape=[N,N,n_lags]))
    #print("all gene bm shape [gene, gene, n_lags]: ", bm_test.shape)

    start_time_loop = time.time()
    for i in range(N):
        bm_one = lasso_timed_lag(expression_data,time_data,i,lags,lambda_val,sigma)
        const[i,0]=bm_one[0,0]
        for j in range(n_lags):
            bm_test[:,i,j]=bm_one[j*N+1:(j+1)*N+1,0]

    end_time = time.time()
    run_time_loop = end_time - start_time_loop
    run_time_total = end_time - start_time_program
    #print("loop run time: ", run_time_loop, "seconds")
    #print("program run time: ", run_time_total, "seconds")

    return (bm_test,const)