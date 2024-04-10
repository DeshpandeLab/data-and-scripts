import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy


def gaussian_kernel(time_data, time_lag_interval, sigma):
    
    #INPUTS:
    #time_data: 1xM matrix of time points each expression data corresponds to index-wise
    #time_lag_interval: the time interval on which a kernel is needed (y's time interval with lag)
    #sigma: kernel smoothing parameter, set by user
    
    #OUTPUTS:
    #kernal_matrix: time_lag_interval x M matrix, corresponding to each lagged time value and time point in the data
    
    r = time_lag_interval.shape[0]
    c = time_data.shape[1]
    kernel_matrix = np.empty(shape=[r, c])
    
    #build matrix
    for i in range(r):
        for j in range(c):
            distance = (time_lag_interval[i,0]-time_data[0,j])**2
            kernel_matrix[i,j] = np.exp(-distance/sigma**2)
    
    #standardize matrix
    for i in range(r):
        total = np.sum(kernel_matrix[i,:])
        for j in range(c):
            kernel_matrix[i,j] = kernel_matrix[i,j]/total
    
    
    return kernel_matrix