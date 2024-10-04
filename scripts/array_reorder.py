import numpy as np
import scipy as scipy
import pandas as pd

def array_reorder(array, n_lags):

    # function that reorders Matlab SINGE matrix outputs to the Python SINGE configuration.
    # INPUTS:
    # array: numpy array of a variable from MATLAB SINGE output, 
    #        arranged by lag3, lag2, lag1, lag3, etc. stacked columns
    # n_lags: integer number representing number of lags used

    # OUTPUT:
    # arr_reordered: numpy array of the variable rearranged to Python SINGE output format,
    #                arranged by lag1, lag2, lag3 stacked matrices (instead of stacked columns)
    # array now follows the order of: [lag1][lag2][lag3] in order

    array = array.to_numpy()

    arr_reordered = array[:,n_lags-1::n_lags]
    for i in range(n_lags-1)[::1]:
        arr1 = array[:,i::n_lags]
        arr_reordered = np.concatenate([arr_reordered,arr1], axis=1)


    return arr_reordered