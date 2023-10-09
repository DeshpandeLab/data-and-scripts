import numpy as np
import pandas as pd
from glmnet import LogitNet
from glmnet import ElasticNet


#a function that takes two matrices as input and run the glmnet lasso regression

#gene_list: Nx1 matrix of target cell data
#expression_data: NxM matrix of cell x gene expression data
#A: Mx1 matrix of what we're looking for
def glmnet_lasso(gene_list, expression_data):
    


    return A