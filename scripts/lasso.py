
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection


#gene_list: Nx1 matrix of target cell data
#expression_data: NxM matrix of cell x gene expression data
#A: Mx1 matrix of what we're looking for
def lasso_reg(gene_list, expression_data):
    X = gene_list
    y = expression_data
    model = linear_model.Lasso()

    #search for best lambda value for data
    cv = model_selection.RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid=dict()
    grid['alpha'] = np.arange(0, 1, 0.01)  #parameter lambda is named alpha in this model
    search = model_selection.GridSearchCV(model, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    search_results = search.fit(X, y)

    #print best lambda value
    print(search_results.best_params_)

    #get the coefficients for the fitted estimator
    A = search_results.coef_

    return A

