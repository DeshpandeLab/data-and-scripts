import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy
from glmnet_lasso import *


def lasso_lag_input(expression_data, target_gene, lags, lambda_val):
    
    #INPUTS:
    #expression_data: NxM matrix of expression data of N genes and M cells, ordered temporally
    #target_gene: integer between 0 and N-1 that represents the target gene's index in the expression_data matrix
    #lags: a list of integers of time lags to be included in the Am design matrix. Example: [5,10,50]
    #lambda_val: lambda value for lasso regression, adjusted based on data fit results
    
    #OUTPUTS:
    #bm: nNx1 matrix of predicted coefficients, where n is the number of lags (i.e. len(lags)) and N genes
    
    X_design = np.asarray(expression_data)
    
    n_lags = len(lags)
    max_lag = max(lags)
    N = X_design.shape[0]
    M = X_design.shape[1]
    #print(N, M)
    
    y = X_design[target_gene, max_lag:] #dim: 1x(M-max_lag+1)
    y = y.transpose() #dim: (M-max_lag+1)x1
    #print(y.shape)
    
    #create Am matrix (essentially the X matrix in lasso regression)
    Am = np.empty(shape=[M-max_lag, n_lags*N])
    for i in range(n_lags):
        Am[:,i*N:(i+1)*N] = X_design[:,max_lag-lags[i]:M-lags[i]].transpose()
    #print(Am.shape)
    
    #run glmnet regression
    bm = glmnet_lasso(y, Am, 0.1)
    
    
    return bm