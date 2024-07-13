
import numpy as np
import pandas as pd
import time
from lasso_lag_input import *
from lasso_timed_lag import *
from lasso_all_genes import *

#testing data
file_path1 = "/Users/emilyxie/Downloads/data-and-scripts/data/gene_list.txt"
gene_list = pd.read_csv(file_path1, header=None, delim_whitespace=True)
#print(gene_list)

file_path2 = "/Users/emilyxie/Downloads/data-and-scripts/data/X_SCODE_data.csv"
expression_data = pd.read_csv(file_path2, header=None)
#print(expression_data)
expression_data = expression_data.to_numpy()

target_gene=99

bm = lasso_lag_input(expression_data, target_gene, [5,10,50], 0.3)
#print(bm)
#print("bm without time lag shape: ", bm.shape)





#testing next step
time_data = pd.read_csv("/Users/emilyxie/Downloads/data-and-scripts/scripts/ptime.txt", header=None)

time_data = time_data.to_numpy()
#print(time_data.shape)
#test1 = time_data[0,91:]
#print(test1.shape)

new_bm = lasso_timed_lag(expression_data, time_data, target_gene, [30], 0.01, 2)

#print(new_bm)
#print("bm with kernel shape: ", new_bm.shape)


#testing lasso_all_genes
#[bm_test, const] = lasso_all_genes(expression_data,time_data,[10,20,50],0.1,1)
#print(bm_test)
#print(const)