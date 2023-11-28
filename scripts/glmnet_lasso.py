import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scipy
#import glmnet
import glmnet_python
from glmnet_python import glmnet
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef
from scipy.sparse import csr_matrix
#from glmnet import ElasticNet



#glmnet_lasso
#a function that takes two matrices as input and run the glmnet lasso regression

#target_gene: Nx1 matrix of target gene expression information
#expression_data: NxM matrix of cell x gene expression data, corresponding to the index of gene_list
#lambda_value: lambda value (penalty), can be adjusted based on data and initial fit results
#gene_list: Nx1 matrix of a list of gene names
#A:Mx1 matrix of model coefficients (what we're looking for), fitted values by the regression
def glmnet_lasso(target_gene, expression_data, lambda_value, gene_list=None):
    
    #check if parameters are of suitable size matrices
    #if len(gene_list) != len(expression_data):
        #raise ValueError("Matrix must have the same number of rows.")
    #elif len(gene_list[0]!=1):
        #raise ValueError("Gene matrix contains more than one column.")
    
    #create variables for regression
    # X: NxM gene expressions
    X = expression_data
    # y: Nx1 sparse matrix of the target cells
    y = target_gene
    lambdas = np.logspace(np.log10(1e-4), np.log10(35), 100)

    #regression
    #m = ElasticNet()
    #m = m.fit(X, y)
    fit = glmnet(x = X, y = y, family='gaussian', nlambda=100, alpha=0.2)
    ##glmnetPrint(fit)
    coef_pred = glmnetCoef(fit, s=scipy.float64([lambda_value]), exact=False)

    #decide which model did a better job of fitting and predicting matrix A
    #y_pred = y_pred1 #or y_pred2, depending on which is more optimal


    return coef_pred