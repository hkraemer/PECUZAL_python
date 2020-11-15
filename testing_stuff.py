import sys 
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from scipy.stats import binom, zscore

# load data from ./data-directory
os.chdir(os.path.join(os.getcwd(), 'data'))
data = np.genfromtxt('roessler_test_series.csv')

# change path and import pecora_embedding functions
os.chdir(os.path.realpath('../../src'))


from pecuzal_embedding import *

data = data[:5000,:]

N = len(data)
mis = np.empty(shape=(50,3))
for i in range(3):
    mis[:,i], lags = mi(data[:,i])    # compute mutual information up to default maximum time lag

plt.figure(figsize=(14., 8,))

ts_str = ['x','y','z']

cnt = 0
for i in range(0,6,2):
    plt.subplot(3,2,i+1)
    plt.plot(range(N),data[:,cnt])
    plt.grid()
    if i == 4:
        plt.xlabel('time [in sampling units]')
    plt.title(ts_str[cnt]+'-component of Roessler test time series')

    plt.subplot(3,2,i+2)
    plt.plot(lags,mis[:,cnt])
    plt.grid()
    plt.ylabel('MI')
    if i == 4:
        plt.xlabel('time lag [in sampling units]')
    plt.title('Mutual information for '+ts_str[cnt]+'-component of Roessler test time series')
    cnt +=1
plt.subplots_adjust(hspace=.3)


plt.figure()
plt.plot(eps)

t0 = time.time()
Y_final, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(100),theiler=30)
t1 = time.time()
print(t1-t0)

from mpl_toolkits import mplot3d

ts_labels = ['x','y','z']

fig = plt.figure(figsize=(14., 8.))
ax = plt.subplot(121, projection='3d')
ax.plot(Y_final[:,0], Y_final[:,1], Y_final[:,2], 'gray')
ax.grid()
ax.set_xlabel('{}(t+{})'.format(ts_labels[ts_vals[0]],tau_vals[0]))
ax.set_ylabel('{}(t+{})'.format(ts_labels[ts_vals[1]],tau_vals[1]))
ax.set_zlabel('{}(t+{})'.format(ts_labels[ts_vals[2]],tau_vals[2]))
ax.set_title('PECUZAL reconstructed Roessler system (multivariate)')
ax.view_init(5, -80)

ax = plt.subplot(122, projection='3d')
ax.plot(data[:5000,0], data[:5000,1], data[:5000,2], 'gray')
ax.grid()
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
ax.set_title('Original Roessler system')


def ar_process(u0, alpha, p, N):
    '''Generate `N`-sample data from an auto regressive process of order 1 with autocorrelation-parameter 
    `alpha` and amplitude `p` for an intial condition value `u0`.
    '''
    x = np.zeros(N+10)
    x[0] = u0
    for i in range(1,N+10):
        x[i] = alpha*x[i-1] + p*np.random.randn()
    
    return x[10:]

u0 = .2
data = ar_process(u0, .9, .2, 2000)

Y_reconstruct, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data)

plt.figure()
plt.plot(data)
plt.title('AR(1) process')
plt.xlabel('sample')
plt.grid()

plt.figure(figsize=(8., 5.))
plt.plot(eps[:,0], label='1st embedding cycle')
plt.scatter([tau_vals[1]], [eps[tau_vals[1],0]])
plt.plot(eps[:,1], label='2nd embedding cycle')
plt.scatter([tau_vals[2]], [eps[tau_vals[2],1]])
plt.plot(eps[:,2], label='3rd embedding cycle')
plt.title('Continuity statistics for PECUZAL embedding of Roessler y-component')
plt.xlabel('delay')
plt.ylabel(r'$\langle \varepsilon^\star \rangle$')
plt.legend(loc='upper right')
plt.grid()

plt.figure()
plt.plot(eps_star)
plt.grid()

s = data[:1000,2]

Y = hcat_lagged_values(s, s, 2)


test = np.array([5, 4, 4])

test = np.empty(shape=(1,3))

Y_test = Y[:3,:]

np.append(Y_test,test,axis=1)