
import numpy as np
import pandas as pd
import time
from lasso_lag_input import *

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
print(bm)
print("bm shape: ", bm.shape)