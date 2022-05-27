from sortedcontainers import SortedDict
from numpy import array
from numpy import argmax
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string

def find_lack_vars_and_gen_var_dict(var_descr_file):
    
    lack_vars, descrs = list(), dict()
    lines = open(var_descr_file, 'r').readlines()
    for line in lines:

        if '(' in line and not line.startswith(' '):

            temp = line.split('(')
            curr_var = temp[0].rstrip()
            var_type = temp[1].split(')')[0].rstrip()
            descrs[curr_var] = var_type

        if 'NA' in line:

            lack_vars.append(curr_var)
            
    return lack_vars, descrs


def print_table(col_titles, cols):
    
    buffer, num_cols, title_str = 16, len(col_titles), str()
    
    for i in range(num_cols):
        
        title = col_titles[i]
        title_str += title
        
        if i != num_cols-1:
            title_str += ''.join([ ' ' for i in range(buffer-len(title))])
            
    print(title_str)
    
    for j in range(len(cols[0])):
        
        row_str = str()
        for k in range(num_cols):
            
            entry = cols[k][j]
            
            if k != num_cols-1:
                row_str += ' ' + str(entry) + ''.join([' ' for l in range(buffer-len(entry))])
                
            else:
                row_str += ' ' + str(entry)
            
        print(row_str)