# This script includes all code shown in the documentation
# and is purely made for reproducibility reasons
#
# Before running this script, make sure you have already installed
# the pecuzal-embedding package by typing `pip install pecuzal-embedding`
# into you console.
#
# A note regarding the reproducibility of the shown figures:
# If you run this code on the command line, i.e. not embedded in any
# Python clinet on some Editor (e.g. Atom or VS Code), then you would need
# to open the saved `.png`-files from the folder you run this code from.
#
# K.H.Kraemer Nov 2020

#' Univariate example
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from pecuzal_embedding import pecuzal_embedding, mi

# integrate Roessler system on standard parameters
def roessler(x,t):
   return [-x[1]-x[2], x[0]+.2*x[1], .2+x[2]*(x[0]-5.7)]

x0 = [1., .5, 0.5] # define initial conditions
tspan = np.arange(0., 5000.*.2, .2) # time span
data = odeint(roessler, x0, tspan, hmax = 0.01)

data = data[2500:,:]    # remove transients

y = data[:,1]   # bind only y-component
muinf, lags = mi(y)    # compute mutual information up to default maximum time lag

plt.figure(figsize=(6., 8,))
plt.subplot(2,1,1)
plt.plot(range(len(y[:1000])),y[:1000])
plt.grid()
plt.xlabel('time [in sampling units]')
plt.title('y-component of Roessler test time series')

plt.subplot(2,1,2)
plt.plot(lags,muinf)
plt.grid()
plt.ylabel('MI')
plt.xlabel('time lag [in sampling units]')
plt.title('Mutual information for y-component of Roessler test time series')

plt.subplots_adjust(hspace=.3)
plt.savefig('mi_and_timeseries_y_comp.png')

Y_reconstruct, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(y, taus = range(100), theiler = 7)

fig = plt.figure(figsize=(14., 8.))
ax = plt.subplot(121, projection='3d')
ax.plot(Y_reconstruct[:,0], Y_reconstruct[:,1], Y_reconstruct[:,2], 'gray')
ax.grid()
ax.set_xlabel('y(t+{})'.format(tau_vals[0]))
ax.set_ylabel('y(t+{})'.format(tau_vals[1]))
ax.set_zlabel('y(t+{})'.format(tau_vals[2]))
ax.set_title('PECUZAL reconstructed Roessler system')

ax = plt.subplot(122, projection='3d')
ax.plot(data[:5000,0], data[:5000,1], data[:5000,2], 'gray')
ax.grid()
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
ax.set_title('Original Roessler system')
plt.savefig('reconstruction_y_comp.png')

print(tau_vals)
print(ts_vals)

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
plt.savefig('continuity_univariate.png')

print(Ls)


## Multivariate example

N = len(data)
mis = np.empty(shape=(50,3))
for i in range(3):
    mis[:,i], lags = mi(data[:,i])    # compute mutual information up to default maximum time lag

plt.figure(figsize=(14., 8,))

ts_str = ['x','y','z']

cnt = 0
for i in range(0,6,2):
    plt.subplot(3,2,i+1)
    plt.plot(range(1000),data[:1000,cnt])
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
plt.savefig('mi_and_timeseries_multi.png')

Y_reconstruct, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(100), theiler = 7)

print(tau_vals)
print(ts_vals)

print(Ls)
L_total = np.sum(Ls[:-1])
print(L_total)

ts_labels = ['x','y','z']

fig = plt.figure(figsize=(14., 8.))
ax = plt.subplot(121, projection='3d')
ax.plot(Y_reconstruct[:,0], Y_reconstruct[:,1], Y_reconstruct[:,2], 'gray')
ax.grid()
ax.set_xlabel('{}(t+{})'.format(ts_labels[ts_vals[0]],tau_vals[0]))
ax.set_ylabel('{}(t+{})'.format(ts_labels[ts_vals[1]],tau_vals[1]))
ax.set_zlabel('{}(t+{})'.format(ts_labels[ts_vals[2]],tau_vals[2]))
ax.set_title('PECUZAL reconstructed Roessler system (multivariate)')
ax.view_init(-115, 30)

ax = plt.subplot(122, projection='3d')
ax.plot(data[:5000,0], data[:5000,1], data[:5000,2], 'gray')
ax.grid()
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
ax.set_title('Original Roessler system')
plt.savefig('reconstruction_multi.png')


## Stochastic example

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

plt.figure()
plt.plot(data)
plt.title('AR(1) process')
plt.xlabel('sample')
plt.grid()
plt.savefig('ar_ts.png')

Y_reconstruct, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data)