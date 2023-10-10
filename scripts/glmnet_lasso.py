import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glmnet


#a function that takes two matrices as input and run the glmnet lasso regression

#gene_list: Nx1 matrix of target cell data
#expression_data: NxM matrix of cell x gene expression data
#y_pred: Mx1 matrix of what we're looking for, fitted values by the regression
def glmnet_lasso(gene_list, expression_data):
    
    X = gene_list
    y = expression_data
    lambdas = np.logspace(np.log10(1e-4), np.log10(35), 100)

    #regression
    enet = glmnet.ElasticNet(lambdas = lambdas)
    y_pred1 = enet.fit(X, y).predict(X)

    lasso = glmnet.Lasso(lambdas = lambdas)
    y_pred2 = lasso.fit(X, y).predict(X)

    #decide which model did a better job of fitting and predicting matrix A

    y_pred = y_pred1 #or y_pred2, depending on which is more optimal


    return y_pred