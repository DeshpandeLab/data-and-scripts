import pandas as pd 
import scipy.io 


#filename must be in a string with format "filename.mat"
def mat_to_csv(filename):
    mat = scipy.io.loadmat(filename)
    data = pd.DataFrame.from_dict(mat, orient='index')
    name = filename[0:-3]+"csv"
    data.to_csv(name)
    


#mat_to_csv("gene_list.mat")