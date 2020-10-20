import pandas as pd
import numpy as np
import scipy.stats
import pyplot as plt

# open roessler time series
x = pd.read_csv('roessler_test_series.csv', names =['xvalues'])

plt.figure()
plt.plot(x)

# normalize time series
x = (x-np.mean(x))/np.std(x)

# taus
taus = np.arange(1,51)

# js
js = 0 # univariative test

# sample size
sample_size = 0.1

# theiler
theiler = 1

# undersamp
undersamp = True

# alpha
alpha = 0.05

# p_val
p_val = 0.5

# beta
beta = 0.05

# normalization
norm = 'euclidean'

############
def get_binomial_table(p,alpha):
    '''
    for the Pecora-FNN method we need to estimate how many points from the delta neighborhood need to fall outside the epsilon neighborhood to reject the Null Hypothesis at a certain significance level alpha. Here we compute a table storing these numbers of points.

    :param p:
    :param alpha:
    :return:
    '''
    prob = 1 - alpha

    # suport
    x = np.arange(0,21)

    # set range for trials
    n_min = 8
    n_max = 15

    prob_table = np.zeros([8,3])
    prob_table[:,0] = np.arange(n_min,n_max+1)

    # loop over the different n values
    for n in iter(range(n_min,n_max+1)):
        col = n-n_min
        y = scipy.stats.binom.cdf(x, n, p_val)

        for i in iter(range(0,20)):
            # if np.sum(y[0:i]) > prob:
            if y[i] > prob:

                prob_table[col,1] = i
                break
            else:
                pass

    prob_table[:, 2] = prob_table[:, 0] - prob_table[:, 1]

    return prob_table



# def pecora_embed_ts(Y,x,tau):
#     '''
#
#     :return: epsilon_mins,gammas,dist_old_,dist_,Y_old,fiducials
#     '''
#     N = len(Y)
#     timespan_diff = tau
#     M = N - timespan_diff
#
#     Y2 = np.zeros([M, len(Y) + 1])
#     Y2(:, 1: size(Y, 2)) = Y(1: M,:);
#     Y2(:, size(Y, 2) + 1) = x(1 + tau: N);
#
# # normalize time series
# x = (x - np.mean(x)) / np.std(x)
#
# # table for the continuity statistic
# bino_table = get_binomial_table(p_val, alpha)
# delta_points = bino_table[:, 0]
# epsilon_points = bino_table[:, 1]
#
# # considered neighbours
# neighbours = delta_points[-1]


### uzal cost func
def uzal_cost(Y, k = 3, Tw = 40, theiler =1 , sample_size = 0.2, norm = 'euclidean'):
    '''

    :param Y: state space vector.
    :param k: number of nearest neighbors to be considered.
    :param Tw: Time forward parameter.
    :param theiler: Theiler window for excluding serial correlated points from neighbourhood.
    :param sample_size: Number of considered fiducial points as a fraction of input time series length.
    :param norm: The norm used for distance computations.
    :return:
            L: The value of the proposed cost-function,
            L_local: The local value of the proposed cost-function. Note that this output can be given only, if you
            have set `sample_size` = 1, i.e. considering all points of the trajectory. The length of `L_local` is
            length(`Y`)-`Tw`.
    '''

    NN = np.zeros(len(Y)-Tw)
    NNN = np.floor(sample_size*NN)
    data_samps = np.arange(0,len(NNN),sample_size)

    E_k2_avrg = np.arange(1, len(NNN),sample_size)*0
    epsilon_k2 = np.arange(1, len(NNN),sample_size)*0

    distances =[]
    for ks in iter(range(0,len(NNN))):
        fiducial_point = data_samps[ks]

        fid_point = Y[fiducial_point]
        distances.append(all_distances(Y[fiducial_point:], Tw, fid_point))

    ind = np.sort(distances)

    eps_idx = np.zeros(k+1)
    eps_idx[0] = fiducial_point
    neighborhood = np.zeros(k+1)
    neighborhood[0] = Y[fiducial_point]

    l = 2

    for nei in iter(range(0,k)):

        if (ind[l] > fiducial_point + theiler) | (ind[l] < fiducial_point - theiler):
            eps_idx[nei + 1] = ind[l]

            neighborhood[nei + 1] = Y[ind[l]]
            l = l + 1

        else:
            l = l + 1
        if l > len(ind):
            print('not enough neighbours')


    pd = scipy.spatial.distance.pdist(neighborhood, norm)
    epsilon_k2[ks] = (2 / (k * (k + 1))) * np.sum(pd**2) # Eq.16

    E_k2 = np.zeros([1, Tw])

    for T in iter(range(1,Tw)):
        eps_ball = Y[eps_idx + T]
        u_k = np.sum(eps_ball) / (k + 1) # Eq.14
        E_k2[T] = np.sum((np.sqrt(np.sum((eps_ball - u_k)** 2)))** 2) / (k + 1) # Eq.13

    E_k2_avrg[ks] = np.mean(E_k2)

    sigma_k2 = E_k2_avrg/ epsilon_k2 # Eq.17
    sigma_k2_avrg = np.mean(sigma_k2) # Eq.18
    alpha_k2 = 1 / np.sum(1/ epsilon_k2) # Eq.21

    L = np.log10(np.sqrt(sigma_k2_avrg) * np.sqrt(alpha_k2)) # Eq.26

    L_local = np.log10(np.sqrt(sigma_k2) * np.sqrt(alpha_k2))

    return L, L_local

x = pd.read_csv('roessler_test_series.csv', sep='\t', names=['x','y','z'])
uzal_cost(x['x'], k = 3, Tw = 40, theiler =1 , sample_size = 1, norm = 'euclidean')

def all_distances(Y, Tw,fid_point):
    Y_new = Y[0:- Tw]
    YY = np.matlib.repmat(fid_point, len(Y_new), 1)

    return np.sqrt(np.sum((YY[:, 0] - Y_new) ** 2))



Y = x['x']
KNN = 3
Tw = 12
L1 =-3.292232428563244