import sys 
import os
import numpy as np
import random
import matplotlib.pyplot as plt

# load data from ./data-directory
os.chdir(os.path.join(os.getcwd(), 'data'))
data = np.genfromtxt('roessler_test_series.csv')

# change path and import pecora_embedding functions
os.chdir(os.path.realpath('../../src/Hauke'))

from pecuzal_embedding import *

theiler = 8
Tw = 40
Y = data[:1000,:2]
sample_size = 1
metric = 'chebyshev'
K = 9


# pecuzal_embedding(Y)
# pecuzal_embedding(data[:1000,:])

Y2 = Y[:,1]

L , L_loc = uzal_cost(Y2, K = 4, Tw = Tw, theiler = theiler , sample_size = sample_size, norm = metric)
print(L)

# eps_star = continuity_statistic(Y, [0], [0], delays = range(101), sample_size = 1, K = K, theiler = theiler,
#         norm = metric, alpha = 0.01, p = 0.5)


# plt.figure()
# plt.plot(eps_star[:,0])
# plt.plot(eps_star[:,1])

# plt.grid()

