import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy
import time


def gaussian_kernel(time_data, time_lag_interval, sigma):
    
    #INPUTS:
    #time_data: 1xM matrix of time points each expression data corresponds to index-wise
    #time_lag_interval: 1xN matrix of the time interval on which a kernel is needed (y's time interval with lag)
    #sigma: kernel smoothing parameter, set by user
    
    #OUTPUTS:
    #kernal_matrix: time_lag_interval x M matrix, corresponding to each lagged time value and time point in the data
    
    start_time = time.time()

    r = time_lag_interval.shape[1]
    c = time_data.shape[1]
    kernel_matrix = np.empty(shape=[r, c])
    

    #change loops into nonloops
    #build matrix (NO LOOP)
    #for i in range(r):
        #for j in range(c):
            #distance = (time_lag_interval[0,i]-time_data[0,j])**2
            #kernel_matrix[i,j] = np.exp(-distance/sigma)
    distance_mx = (time_lag_interval.T - time_data)**2
    kernel_matrix = np.exp(-distance_mx / sigma)
    
    #print(kernel_matrix[31:35,0])

    #standardize matrix (NO LOOP)
    total = np.sum(kernel_matrix, axis=1)
    kernel_matrix = kernel_matrix/total[:,np.newaxis]
    
    #print("1st col ", kernel_matrix[0,:])
    
    end_time = time.time()
    run_time = end_time - start_time
    #print("gaussian kernal program runtime: ", run_time)

    return kernel_matrix