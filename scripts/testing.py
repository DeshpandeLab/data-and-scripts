
import numpy as np
import pandas as pd
from lasso import *
import time

#create Nx1 matrix of target cell data
file_path1 = "/Users/emilyxie/Downloads/SINGE/gene_list.txt"
gene_list = pd.read_csv(file_path1, header=None, delim_whitespace=True)
gene_list.to_csv('gene_list.csv', index=None)

#print(gene_list)

#create NxM matrix of cell & expression data
file_path2 = "/Users/emilyxie/Downloads/SINGE/X_SCODE_data.csv"
expression_data = pd.read_csv(file_path2, header=None)
#print(gene_cell_data)

#run lasso regression
start = time.time()
lasso_reg(gene_list, expression_data)
end = time.time()
elapsed = end - start
print(f'Time elapsed: {elapsed:.8f} seconds.')