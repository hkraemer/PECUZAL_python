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
Y = data[:100,:]
sample_size = 1
norm = 'euclidean'
K = 8

L = uzal_cost(Y, K = K, Tw = Tw, theiler = theiler , sample_size = sample_size, norm = norm)
display(L)
# D = np.size(Y,1)
# NN = len(Y)-Tw
# NNN = int(np.floor(sample_size*NN))
# ns = random.sample(list(np.arange(NN)),NNN) # the fiducial point indices
# ns = np.array(np.arange(0,100))
# vs = Y[ns[:]] # the fiducial points in the data set

# vtree = KDTree(Y[:-Tw], metric = norm)
# allNNidxs, _ = all_neighbors(vtree, vs, ns, K, theiler) 

# NNidxs = allNNidxs[0]


# E = comp_Ek2(Y, ns[0], NNidxs, 1, K, norm)
