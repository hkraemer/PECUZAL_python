#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:48:16 2020

@author: hkraemer
"""

# this is the actual function. The aim is to make this function work 
# automatically for any time series input (uni- or multivariate)
# the desired input should look like this:

import math
import numpy as np
import scipy


# def embed_pecora_uzal(var,taus=np.arange(0,50),datasample=0.5,theiler_window=1,alpha=0.05,p=0.5,k=3,Tw=50,norm="euclidean"):
def all_neighbors(vtree, vs, ns, K, w):
    pass


def fiducial_pairwise_dist_sqrd(data, NNidxs, v, metric):
    pass


def comp_Ek2(e_ball, u_k, Y, param, NNidxs, T, K, metric):
    pass

def to_evaluate(metric, v1, v):
    pass


from scipy.stats import zscore
import numpy as np


def embed_pecora_uzal(Y, Tw=40, K=3, Int=1, samplesize=0.5, norm="euclidean"):
    # `var` is the input-variable. This can be a single time series or a multi-
    # dimensional array, containing many time series. `taus` is a range-object 
    # indicating the delay values which the algorithm consideres. `datasample` is a
    # floating number from the interval (0,1] and determines the number of considered
    # trajectory points (fiducial points) as a fraction from all points. Lets say 
    # the input time series `var`is of length 10,000 and `datasample=0.5`, then 
    # 5,000 fiducial points get randomly chosen and the corresponding continuity-
    # statistic gets computed. `alpha` is the significance level for the continuity-
    # statistic, `p` is the probability for the binomial distribution. `k` is the 
    # number of considered nearest neighbors in Uzal's L-statistic and `Tw` is 
    # the maximum time horizon for the L-statistic. `norm` is the metric for 
    # distance computations in phase space. This can be restricted to `euclidean` 
    # and `chebychev`. 

    # the desired ouput should be
    # =============================================================================
    #  return Y,tau_vals,ts_vals,epsilon_mins,FNNs,Ls
    # =============================================================================

    # `Y` is the final embedded trajectory. `tau_vals` are the chosen delay 
    # values for each embedding cycle. Thus, the length of `tau_vals` corresponds
    # to the dimensionality of `Y`. `ts_vals` is a list storing the chosen time 
    # series for each delay value in `tau_vals`. In case of univariate embedding,
    # i.e. when there is just a single time series as input `var`, then `ts_vals`
    # is a list of same length as `tau_vals` just containing Ones (or Zeros, 
    # depending on the indexing style you prefer), since there is just one time 
    # series to choose from. `epsilon_mins` is a list containing the continuity-
    # statistic for each embedding cycle. `FNNs` contains the amount of false
    # nearest neighbors for each embedding cycle. `Ls` contains Uzal's L-statistic
    # for each embedding cycle.

    # I suggest a function body looking roughly like this (but feel free to do 
    # whatever you want and what seems to make sense, also performance-wise):

    # =============================================================================
    #     z-standardize all time series in `var``
    Y = Y.apply(zscore)

    #     preallocate empty lists for `tau_vals`, `ts_vals`, `epsilon_mins`, `FNNs`, `Ls``
    tau_vals =np.zeros(Y.shape)
    ts_vals = =np.zeros(Y.shape)
    epislon_mins = np.zeros(Y.shape)
    FNNs = np.zeros(Y.shape)
    Ls = np.zeros(Y.shape)

    for i in iter(range(0,len(Y))):
        for j in iter(range(0, len(Y))):
            print(i,j)
            epsilons = continuity_statistic(Y[i],Y[j])


    #     
    #     start while-loop over embedding cycles:
    #     
    #         in the first embedding cycle:
    #             
    #             loop over all time series in `var`, i:
    #             
    #                 loop over all time series in `var`, j:
    #             
    #                     for each (i,j) combination compute the continuity statistic:
    #                         epsilons = continuity_statistic(timeseries i, timeseries j, kwargs)
    #                     
    #                         for each peak in the continuity statistic `epsilons`:
    #                             make an embedding `Y_trial` with this delay `delay_trial`, 
    #                             and compute the L-statistic L = uzal_cost(`Y_trial`,`delay_trial`,`k`,`Tw`)
    #                             also compute the FNN-statistic fnns = fnn(`Y_trial-distances`, `former-distances)
    #                             
    #                             save the peak/delay value `delay_trial`, which has the
    #                             lowest `L`.
    #                     
    #             compare all L-statistics `L` for all (i,j)-combinations and take the
    #             one with the minimum `L`. Then ts_vals[0] = i and ts_vals[1] = j,
    #             tau_vals[0] = 0 and tau_vals[1] = `delay_trial`, which corresponds 
    #             to the minimum `L`
    #             
    #                 
    #         in all consecutive embedding cycles:
    #             
    #             loop over all time series in `var`, i:
    #                 
    #                 for each time series i, compute the continuity statistic:
    #                     epsilons = continuity_statistic(`Y`, timeseries i, kwargs)
    #                 
    #                     for each peak in the continuity statistic `epsilons`:
    #                         make an embedding `Y_trial` with this delay `delay_trial`, 
    #                         and compute the L-statistic L = uzal_cost(`Y_trial`,`theiler_window`,`k`,`Tw`,`datasample`)
    #                         also compute the FNN-statistic fnns = fnn(`Y_trial-distances`, `former-distances)
    #                         
    #                     save the peak/delay value `delay_trial`, which has the
    #                     lowest `L` in `tau_vals` and save the corresponding time 
    #                     series in `ts_vals`
    #                     
    #                     
    #         construct the actual phase space trajectory `Y` with the values in
    #         `tau_vals` and `ts_vals`
    #         
    #         compute the L-statistic for `Y`: L = uzal_cost(`Y`,`theiler_window`,`k`,`Tw`,`datasample`)
    #         
    #         check break criterions:
    #             
    #         if there hasn't been any valid peak to choose from in the continuity statistic, 
    #         the last saved `delay_value` is NaN, break
    #     
    #         if `L` is higher than in the former embedding cycle, break:
    #             
    #     
    #     return all return-values
    # =============================================================================

    D = Y['D']
    ET = Y['ET']

    NN = len(Y) - Tw
    NNN = math.floor(Int)
    ns = np.random.choice((np.arange(1, NN), NNN)  # the fiducial point indices

    vs = Y[ns] # the fiducial points in the data set

    vtree = scipy.spatial.KDTree(Y[0:-Tw])
    vtree.count_neighbors(vs)
    allNNidxs, allNNdist = all_neighbors(vtree, vs, ns, K, w)
    e_squared = np.zeros(NNN)  # neighborhood size
    E_squared_avrg = np.zeros(NNN)  # averaged conditional variance
    E_squared = np.zeros(Tw)
    e_ball = np.zeros(ET, K + 1, D)  # preallocation
    u_k = np.zeros(ET, D)

    # loop over each fiducial point
    for (i, v) in iter(range((vs)):
        NNidxs = allNNidxs[i]  # indices of k nearest neighbors to v
    # pairwise distance of fiducial points and `v`
    pdsqrd = fiducial_pairwise_dist_sqrd(Y.data, NNidxs, v, metric)
    e_squared[i] = (2 / (K * (K + 1))) * pdsqrd  # Eq. 16
    # loop over the different time horizons
    for T in iter(range(1, Tw)):
        E_squared[T] = comp_Ek2(e_ball, u_k, Y, ns[i], NNidxs, T, K, metric)  # Eqs. 13 & 14

    # Average E²[T] over all prediction horizons
    E_squared_avrg[i] = np.mean(E_squared)  # Eq. 15

    sigma_squared = E_squared_avrg. / e_squared  # noise amplification σ², Eq. 17
    sigma_squared_avrg = np.mean(sigma_squared)  # averaged value of the noise amplification, Eq. 18
    alpha_squared = 1 / np.sum(e_squared. ^ (-1))  # for normalization, Eq. 21
    L = np.log10(np.sqrt(sigma_squared_avrg) * np.sqrt(alpha_squared))




def continuity_statistic(var1, time_series_x, `taus`, `datasample`, `theiler_window`, `alpha`, `p`, `norm`):


# function body

# according to `p` and `alpha`, compute how many points fall outside the
# epsilon neighborhood and how many can be tolerated inside to reject the Null
# =============================================================================
#     bino_table = get_binomial_table(p_val,alpha);
# =============================================================================

# perform continuity statistic

def uzal_cost(Y, theiler_window = 17, k = 3, Tw = 40 , datasample = 0.5, metric = 'euclidean'):
    Y = np.zeros([1000, 2])
    Y[:, 0] = np.random.rand(1000)
    Y[:, 1] = np.random.rand(1000) * 1.0005
    Y = Y.transpose()

    D = Y[0]
    ET = Y[1]


    # select a random phase space vector sample according to input samplesize
    NN = len(Y[0,:]) - Tw
    NNN = int(math.floor(datasample * NN))
    ns = np.random.choice((np.arange(0, NN), NNN),replace = False)

    vs = Y[0][ns]  # the fiducial points in the data set

    from sklearn.neighbors import KDTree
    vtree = KDTree(Y.reshape(-1, 1))

    allNNidxs = vtree.query_radius(Y.reshape(-1, 1), r =  k)[0]
    e_squared = np.zeros(NNN)  # neighborhood size
    E_squared_avrg = np.zeros(NNN)  # averaged conditional variance
    E_squared = np.zeros(Tw)
    e_ball = np.zeros([len(ET), K + 1, len(D)])  # preallocation
    u_k = np.zeros([len(ET), len(D)])

    # loop over each fiducial point
    for (i, v) in enumerate(vs)
        NNidxs = allNNidxs[i]  # indices of k nearest neighbors to v
        # pairwise distance of fiducial points and `v`
        pdsqrd = fiducial_pairwise_dist_sqrd(view(Y.data, NNidxs), v, metric)
        e_squared[i] = (2 / (K * (K + 1))) * pdsqrd  # Eq. 16
        # loop over the different time horizons
        for T = 1:Tw
        E_squared[T] = comp_Ek2(e_ball, u_k, Y, ns[i], NNidxs, T, K, metric)  # Eqs. 13 & 14

    # Average E²[T] over all prediction horizons
    E_squared_avrg[i] = np.mean(E_squared)  # Eq. 15

    sigma_squared = E_squared_avrg./ e_squared  # noise amplification σ², Eq. 17
    alpha_squared = 1 / sum(e_squared**(-1))  # for normalization, Eq. 21
    L_local = np.log10.(np.sqrt.(sigma_squared).*np.sqrt(sigma_squared))
    return L_local




# function body


def get_binomial_table(p_val, alpha):


# for the Pecora-FNN method we need to estimate how many points from the
#    delta neighborhood need to fall outside the epsilon neighborhood to
#    reject the Null Hypothesis at a certain significance level alpha. Here we
#    compute a table storing these numbers of points.

def fnn(`new_distances`, `old_distances`, r=2):
# for a list of `old_distances` and `new_distances`, compute the false
# nearest neighbor fraction under a threshold `r` after
# Kantz & Schreiber 2004, formula 3.8 on page 37.
# and
# Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor
# method to detect determinism in time series data. Physical Review E 60, 4970


# other helper functions ....