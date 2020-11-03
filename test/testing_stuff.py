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

sample_size = 1
metric = 'chebyshev'
K = 13

Y = data[:1000,:2]
Y2 = Y[:,1]

# L , L_loc = uzal_cost(Y2, K = K, Tw = Tw, theiler = theiler , sample_size = sample_size, norm = metric)
# print(L)

# eps_star = continuity_statistic(Y, [0], [0], delays = range(101), sample_size = 1, K = K, theiler = theiler,
#         norm = metric, alpha = 0.01, p = 0.5)

Y_final, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(Y2, taus = range(100))


# plt.figure()
# plt.plot(eps_star[:,0])
# plt.plot(eps_star[:,1])

# plt.grid()

s = data[:1000,2]

Y = hcat_lagged_values(s, s, 2)


test = np.array([5, 4, 4])

test = np.empty(shape=(1,3))

Y_test = Y[:3,:]

np.append(Y_test,test,axis=1)