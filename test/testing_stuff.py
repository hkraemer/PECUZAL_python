import sys 
import os
import numpy as np
import random

# load data from ./data-directory
os.chdir(os.path.join(os.getcwd(), 'data'))
data = np.genfromtxt('roessler_test_series.csv')

# change path and import pecora_embedding functions
os.chdir(os.path.realpath('../../src/Hauke'))

from pecora_embedding import *

theiler = 2
Tw = 40
Y = data[:1000,:]
sample_size = 1
norm = 'chebyshev'
K = 8

# L , L_loc = uzal_cost(Y, K = 4, Tw = Tw, theiler = theiler , sample_size = sample_size, norm = norm)
# print(L)

eps_star = continuity_statistic(Y[:,0], [0], [0], delays = range(50), sample_size = 1, K = 13, theiler = 5,
        norm = 'euclidean', alpha = 0.05, p = 0.5)


