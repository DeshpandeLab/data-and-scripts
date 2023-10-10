import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy
import glmnet_python
from glmnet import glmnet
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef


#a function that takes two matrices as input and run the glmnet lasso regression

#gene_list: Nx1 matrix of target cell data
#expression_data: NxM matrix of cell x gene expression data
#y_pred: Mx1 matrix of what we're looking for, fitted values by the regression
def glmnet_lasso(gene_list, expression_data):
    
    #check if parameters are of suitable size matrices
    
    
    #create variables for regression
    X = gene_list
    y = expression_data
    lambdas = np.logspace(np.log10(1e-4), np.log10(35), 100)

    #regression
    fit = glmnet(x = X, y = y, family='gaussian', nlambda=100, alpha=0.2)
    glmnetPrint(fit)
    glmnetCoef(fit, s=scipy.float64([0.5]), exact=False)

    #decide which model did a better job of fitting and predicting matrix A


    y_pred = y_pred1 #or y_pred2, depending on which is more optimal


    return y_pred