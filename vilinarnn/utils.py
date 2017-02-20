'''
Created on Feb 11, 2017

@author: mozhu
'''

import numpy as np

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


