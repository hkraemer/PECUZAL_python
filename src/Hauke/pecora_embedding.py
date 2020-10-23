#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:48:16 2020

@author: hkraemer
"""

import math
import numpy as np
import scipy
import random
from scipy.stats import zscore
from sklearn.neighbors import KDTree
from scipy.stats import binom


# def pecuzal_embedding(Y, Tw=40, K=3, Int=1, samplesize=0.5, norm="euclidean"):
#     # `var` is the input-variable. This can be a single time series or a multi-
#     # dimensional array, containing many time series. `taus` is a range-object 
#     # indicating the delay values which the algorithm consideres. `datasample` is a
#     # floating number from the interval (0,1] and determines the number of considered
#     # trajectory points (fiducial points) as a fraction from all points. Lets say 
#     # the input time series `var`is of length 10,000 and `datasample=0.5`, then 
#     # 5,000 fiducial points get randomly chosen and the corresponding continuity-
#     # statistic gets computed. `alpha` is the significance level for the continuity-
#     # statistic, `p` is the probability for the binomial distribution. `k` is the 
#     # number of considered nearest neighbors in Uzal's L-statistic and `Tw` is 
#     # the maximum time horizon for the L-statistic. `norm` is the metric for 
#     # distance computations in phase space. This can be restricted to `euclidean` 
#     # and `chebychev`. 

#     # the desired ouput should be
#     # =============================================================================
#     #  return Y,tau_vals,ts_vals,epsilon_mins,FNNs,Ls
#     # =============================================================================

#     # `Y` is the final embedded trajectory. `tau_vals` are the chosen delay 
#     # values for each embedding cycle. Thus, the length of `tau_vals` corresponds
#     # to the dimensionality of `Y`. `ts_vals` is a list storing the chosen time 
#     # series for each delay value in `tau_vals`. In case of univariate embedding,
#     # i.e. when there is just a single time series as input `var`, then `ts_vals`
#     # is a list of same length as `tau_vals` just containing Ones (or Zeros, 
#     # depending on the indexing style you prefer), since there is just one time 
#     # series to choose from. `epsilon_mins` is a list containing the continuity-
#     # statistic for each embedding cycle. `FNNs` contains the amount of false
#     # nearest neighbors for each embedding cycle. `Ls` contains Uzal's L-statistic
#     # for each embedding cycle.

#     # I suggest a function body looking roughly like this (but feel free to do 
#     # whatever you want and what seems to make sense, also performance-wise):

#     # =============================================================================
#     #     z-standardize all time series in `var``
#     Y = Y.apply(zscore)

#     #     preallocate empty lists for `tau_vals`, `ts_vals`, `epsilon_mins`, `FNNs`, `Ls``
#     tau_vals = np.zeros(Y.shape)
#     ts_vals = np.zeros(Y.shape)
#     epislon_mins = np.zeros(Y.shape)
#     FNNs = np.zeros(Y.shape)
#     Ls = np.zeros(Y.shape)

#     for i in iter(range(0,len(Y))):
#         for j in iter(range(0, len(Y))):
#             print(i,j)
#             epsilons = continuity_statistic(Y[i],Y[j])


#     #     
#     #     start while-loop over embedding cycles:
#     #     
#     #         in the first embedding cycle:
#     #             
#     #             loop over all time series in `var`, i:
#     #             
#     #                 loop over all time series in `var`, j:
#     #             
#     #                     for each (i,j) combination compute the continuity statistic:
#     #                         epsilons = continuity_statistic(timeseries i, timeseries j, kwargs)
#     #                     
#     #                         for each peak in the continuity statistic `epsilons`:
#     #                             make an embedding `Y_trial` with this delay `delay_trial`, 
#     #                             and compute the L-statistic L = uzal_cost(`Y_trial`,`delay_trial`,`k`,`Tw`)
#     #                             also compute the FNN-statistic fnns = fnn(`Y_trial-distances`, `former-distances)
#     #                             
#     #                             save the peak/delay value `delay_trial`, which has the
#     #                             lowest `L`.
#     #                     
#     #             compare all L-statistics `L` for all (i,j)-combinations and take the
#     #             one with the minimum `L`. Then ts_vals[0] = i and ts_vals[1] = j,
#     #             tau_vals[0] = 0 and tau_vals[1] = `delay_trial`, which corresponds 
#     #             to the minimum `L`
#     #             
#     #                 
#     #         in all consecutive embedding cycles:
#     #             
#     #             loop over all time series in `var`, i:
#     #                 
#     #                 for each time series i, compute the continuity statistic:
#     #                     epsilons = continuity_statistic(`Y`, timeseries i, kwargs)
#     #                 
#     #                     for each peak in the continuity statistic `epsilons`:
#     #                         make an embedding `Y_trial` with this delay `delay_trial`, 
#     #                         and compute the L-statistic L = uzal_cost(`Y_trial`,`theiler_window`,`k`,`Tw`,`datasample`)
#     #                         also compute the FNN-statistic fnns = fnn(`Y_trial-distances`, `former-distances)
#     #                         
#     #                     save the peak/delay value `delay_trial`, which has the
#     #                     lowest `L` in `tau_vals` and save the corresponding time 
#     #                     series in `ts_vals`
#     #                     
#     #                     
#     #         construct the actual phase space trajectory `Y` with the values in
#     #         `tau_vals` and `ts_vals`
#     #         
#     #         compute the L-statistic for `Y`: L = uzal_cost(`Y`,`theiler_window`,`k`,`Tw`,`datasample`)
#     #         
#     #         check break criterions:
#     #             
#     #         if there hasn't been any valid peak to choose from in the continuity statistic, 
#     #         the last saved `delay_value` is NaN, break
#     #     
#     #         if `L` is higher than in the former embedding cycle, break:
#     #             
#     #     
#     #     return all return-values
#     # =============================================================================

#     D = Y['D']
#     ET = Y['ET']

#     NN = len(Y) - Tw
#     NNN = math.floor(Int)
#     ns = np.random.choice((np.arange(1, NN), NNN)  # the fiducial point indices

#     vs = Y[ns] # the fiducial points in the data set

#     vtree = scipy.spatial.KDTree(Y[0:-Tw])
#     vtree.count_neighbors(vs)
#     allNNidxs, allNNdist = all_neighbors(vtree, vs, ns, K, w)
#     e_squared = np.zeros(NNN)  # neighborhood size
#     E_squared_avrg = np.zeros(NNN)  # averaged conditional variance
#     E_squared = np.zeros(Tw)
#     e_ball = np.zeros(ET, K + 1, D)  # preallocation
#     u_k = np.zeros(ET, D)

#     # loop over each fiducial point
#     for (i, v) in iter(range((vs)):
#         NNidxs = allNNidxs[i]  # indices of k nearest neighbors to v
#     # pairwise distance of fiducial points and `v`
#     pdsqrd = fiducial_pairwise_dist_sqrd(Y.data, NNidxs, v, metric)
#     e_squared[i] = (2 / (K * (K + 1))) * pdsqrd  # Eq. 16
#     # loop over the different time horizons
#     for T in iter(range(1, Tw)):
#         E_squared[T] = comp_Ek2(e_ball, u_k, Y, ns[i], NNidxs, T, K, metric)  # Eqs. 13 & 14

#     # Average E²[T] over all prediction horizons
#     E_squared_avrg[i] = np.mean(E_squared)  # Eq. 15

#     sigma_squared = E_squared_avrg. / e_squared  # noise amplification σ², Eq. 17
#     sigma_squared_avrg = np.mean(sigma_squared)  # averaged value of the noise amplification, Eq. 18
#     alpha_squared = 1 / np.sum(e_squared. ^ (-1))  # for normalization, Eq. 21
#     L = np.log10(np.sqrt(sigma_squared_avrg) * np.sqrt(alpha_squared))




def continuity_statistic(s, taus, js, delays = range(50), sample_size = 0.1, K = 13, theiler = 1, norm = 'euclidean', alpha = 0.05, p = 0.5):
    '''
    :return:
            The continuity statistic `avrg_eps_star` for each time series in `s` for the given trajectory
            specified by the general embedding parameters `taus` and `js`.
    :param s: Input time series as numpy array. This can be a multivariate set, where the timeseries
    are stored in the columns.
    :param taus: denotes what delay times will be used for each of the entries of the delay vector. 
    It is recommended that `taus[0] = 0`.
    :param js: denotes which of the timeseries contained in `s` will be used for the entries of the 
    delay vector. `js` can contain duplicate indices.
    :param delays = range(50): Possible time delay values `delays` (in sampling time units). For each 
    of the `tau`'s in `delays` the continuity-statistic `avrg_eps_star` gets computed.
    :param sample_size = 0.1: Number of considered fiducial points as a fraction of input time series length.
    :param K = 13: the amount of nearest neighbors in the Delta-ball. Must be at least 8 (in order to guarantee 
    a valid statistic). `avrg_eps_star` is computed taking the minimum result over all `k in K`. 
    :param theiler = 1: Theiler window for excluding serial correlated points from neighbourhood.
    :param norm: The norm used for distance computations. Must be either `'euclidean'` (Default) or `'chebyshev'`
    :param alpha: Significance level (Default is 0.05)
    :param p: Binominal p (Default is 0.5)
    '''
    if np.ndim(s)>1:
        assert (np.size(s,0) > np.size(s,1)), "You must provide a numpy array storing the time series in its columns."
        D = np.size(s,1)
        all_eps_star = np.empty(shape=(len(delays), D))
    else:
        D = 1
    assert (K >= 8) and (type(K) is int) and (K < len(s)) , "You must provide a delta-neighborhood size consisting of at least 8 neighbors."
    assert (sample_size > 0) and (sample_size <= 1), "sample_size must be in (0 1]"
    assert (theiler >= 0) and (type(theiler) is int) and (theiler < len(s)), "Theiler window must be a positive integer smaller than the time series length."
    assert (alpha >= 0) and (alpha < 1), "Significance level alpha must be in (0 1)"
    assert (p >= 0) and (p < 1), "Binomial p parameter must be in (0 1)"
    assert (type(norm) is str) and (norm == 'euclidean' or norm == 'chebyshev')

    vspace = genembed(s, taus, js)
    vtree = KDTree(vspace[:-np.amax(delays)], metric = norm)

    # pick fiducial points
    if sample_size == 1:
        N = len(vspace)-delays[-1]
        ns = np.arange(N)
    else:
        N = int(np.floor(sample_size*(len(vspace)-delays[-1])))
        ns = random.sample(list(np.arange(len(vspace)-delays[-1])),N) # the fiducial point indices

    vs = vspace[ns]
    allNNidxs, _ = all_neighbors(vtree, vs, ns, K, theiler, len(vspace[:-np.amax(delays)]))

    # Loop over potential timeseries to use in new embedding
    for i in range(D):
        if D == 1:
            x = (s-np.mean(s))/np.std(s) # so that different timeseries can be compared
            all_eps_star = continuity_per_timeseries(x, ns, allNNidxs, delays, K, alpha, p)
        else:
            x = s[:,i]
            x = (x-np.mean(x))/np.std(x) # so that different timeseries can be compared
            all_eps_star[:,i] = continuity_per_timeseries(x, ns, allNNidxs, delays, K, alpha, p)

    return all_eps_star


def genembed(s, taus, js):
    '''
    :param s: Input time series as numpy array. This can be a multivariate set, where the timeseries
    are stored in the columns.
    :param taus: denotes what delay times will be used for each of the entries of the delay vector. 
    It is recommended that `taus[0] = 0`.
    :param js: denotes which of the timeseries contained in `s` will be used for the entries of the 
    delay vector. `js` can contain duplicate indices.
    :return:
            generalized embedding of `s` which can be a uni- or multivariate and return the result 
            as a new numpy.array.

    The generalized embedding works as follows:
    `taus, js` are tuples (or vectors) of length `D`, which also coincides with the embedding
    dimension. For example, imagine input trajectory `s = [x, y, z]` where `x, y, z` are
    timeseries (the columns of `s`).
    If `js = (0, 2, 1)` and `taus = (0, 2, 7)` the created delay vector at each step `t` will be
    .. math::
        (x(t), z(t+2), y(t+7))
    '''
    assert np.amax(js) <= np.ndim(s)
    if np.ndim(s) == 1:
        assert js[0] == 0 
    N = len(s) - np.amax(taus)
    data = np.empty(shape=(N,len(taus)))
    for (i, tau) in enumerate(taus):
        if np.ndim(s) == 1:
            data[:,i] = s[tau:(N+tau)]
        else:
            data[:,i] = s[tau:(N+tau), js[i]]
    return data
    

def get_binomial_table(p = 0.5, alpha = 0.05, trial_range = 8):
    '''
    :param p: Binominal p (Default is 0.5)
    :param alpha: Significance level (Default is 0.05)
    :param trial_range: number of considered delta-neighborhood-points (Default is 8)
    :return:
            `delta_to_epsilon_amount`, a dictionary with `delta_points` as keys and the 
            corresponding number of points in order to reject the Null, `epsilon_points`, 
            constitute the values. 
    Compute the numbers of points from the delta-neighborhood, which need to fall outside
    the eosilon-neighborhood, in order to reject the Null Hypothesis at a significance
    level `alpha`. One parameter of the binomial distribution is `p`, the other one would be 
    the number of trials, i.e. the considered number of points of the delta-neighborhood. 
    `trial_range` determines the number of considered delta-neighborhood-points, always 
    starting from 8. For instance, if `trial_range=8`, then delta-neighborhood sizes from 
    8 up to 15 are considered.
    '''
    assert trial_range >= 1
    delta_to_epsilon_amount = dict()
    for key in range(8,8+trial_range):
        delta_to_epsilon_amount[key] = int(binom.ppf(1-alpha, key, p))
    return delta_to_epsilon_amount


def continuity_per_timeseries(x, ns, allNNidxs, delays, K, alpha, p):
    avrg_eps_star = np.zeros(np.size(delays))
    Ks = [k for k in range(8,K+1)]
    delta_to_epsilon_amount = get_binomial_table(p, alpha, len(Ks))
    for (l, tau) in enumerate(delays): # Loop over the different delays
        c = 0
        for (i, n) in enumerate(ns): # Loop over fiducial points
            NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
            # calculate minimum ε
            avrg_eps_star[l] += eps_star(x, n, tau, NNidxs, delta_to_epsilon_amount, Ks)
            c += 1
        avrg_eps_star[l] /= c
    return avrg_eps_star


def eps_star(x, n, tau, NNidxs, delta_to_epsilon_amount, Ks):
    a = x[n+tau] # fiducial point in epsilon-space
    dis = np.array([abs(a - x[i+tau]) for i in NNidxs])
    eps = np.zeros(len(Ks))
    for (i, k) in enumerate(Ks):
        sortedds = np.sort(dis[:k])
        l = delta_to_epsilon_amount[k]
        eps[i] = sortedds[l-1]
    return np.amin(eps)


# uzal cost function
def uzal_cost(Y, K = 3, Tw = 40, theiler = 1 , sample_size = 1.0, norm = 'euclidean'):
    '''
    :param Y: state space vector.
    :param K: number of nearest neighbors to be considered.
    :param Tw: Time forward parameter.
    :param theiler: Theiler window for excluding serial correlated points from neighbourhood.
    :param sample_size: Number of considered fiducial points as a fraction of input time series length.
    :param norm: The norm used for distance computations. Must be either `'euclidean'` or `'chebyshev'`
    :return:
            L: The value of the proposed cost-function,
            L_local: The local value of the proposed cost-function. Note that this output can be given only, if you
            have set `sample_size` = 1, i.e. considering all points of the trajectory. The length of `L_local` is
            length(`Y`)-`Tw`.
    '''
    assert (theiler >= 0) and (type(theiler) is int) and (theiler < len(Y))
    assert (K >= 0) and (type(K) is int) and (K < len(Y))
    assert (sample_size > 0) and (sample_size <= 1)
    assert (Tw >= 0) and (type(Tw) is int) and (Tw < len(Y))
    assert (type(norm) is str) and (norm == 'euclidean' or norm == 'chebyshev')

    D = np.size(Y,1)
    NN = len(Y)-Tw
    NNN = int(np.floor(sample_size*NN))
    ns = random.sample(list(np.arange(NN)),NNN) # the fiducial point indices

    vs = Y[ns[:]] # the fiducial points in the data set

    vtree = KDTree(Y[:-Tw], metric = norm)
    allNNidxs, _ = all_neighbors(vtree, vs, ns, K, theiler, (len(vs)-Tw)) 

    eps2 = np.empty(NNN)             # neighborhood size
    E2_avrg = np.empty(NNN)          # averaged conditional variance
    E2 = np.empty(Tw)                # condition variance 
    neighborhood_v = np.empty(shape=(K+1,D))

    # loop over each fiducial point
    for (i,v) in enumerate(vs):
        NNidxs = allNNidxs[i] # indices of K nearest neighbors to v

        # construct local neighborhood   
        neighborhood_v[0] = v            
        neighborhood_v[1:] = Y[NNidxs]

        # pairwise distance of fiducial points and `v`
        pdsqrd = scipy.spatial.distance.pdist(neighborhood_v, norm)
        eps2[i] = (2/(K*(K+1))) * np.sum(pdsqrd**2)  # Eq. 16

        # loop over the different time horizons
        for T in range(Tw):
            E2[T] = comp_Ek2(Y, ns[i], NNidxs, T+1, K, norm) # Eqs. 13 & 14

        # Average E²[T] over all prediction horizons
        E2_avrg[i] = np.mean(E2)                   # Eq. 15
    
    sigma2 = E2_avrg / eps2 # noise amplification σ², Eq. 17
    sigma2_avrg = np.mean(sigma2) # averaged value of the noise amplification, Eq. 18
    alpha2 = 1 / np.sum(eps2**(-1)) # for normalization, Eq. 21
    L = np.log10(np.sqrt(sigma2_avrg)*np.sqrt(alpha2))
    L_local = np.log10(np.sqrt(sigma2)*np.sqrt(alpha2))
    return L, L_local




def all_neighbors(vtree, vs, ns, K, theiler, k_max):
    '''
    :return:  The `K`-th nearest neighbors for all input points `vs`, with indices `ns` in
    original data, while respecting the `theiler` window.
    '''
    dists = np.empty(shape=(len(vs),K))  
    idxs = np.empty(shape=(len(vs),K),dtype=int)

    dist_, ind_ = vtree.query(vs[:], k=k_max)

    for i in range(np.size(dist_,0)):    
        cnt = 0
        for j in range(1,np.size(dist_,1)):
            if ind_[i,j] < ns[i]-theiler or ind_[i,j] > ns[i]+theiler:
                dists[i,cnt], idxs[i,cnt] = dist_[i,j], ind_[i,j]
                if cnt == K-1:
                    break
                else:
                    cnt += 1
    return idxs, dists
    


def comp_Ek2(Y, ns, NNidxs, T, K, norm):
    '''
    :return:  The approximated conditional variance for a specific point in state space
    `ns` (index value) with its `K`-nearest neighbors, which indices are stored in
    `NNidxs`, for a time horizon `T`. This corresponds to Eqs. 13 & 14 in Uzal et al. 2011.
    The specified `norm` is used for distance computations.
    '''
    D = np.size(Y,1)
    # determine neighborhood `T` time steps ahead
    eps_ball = np.empty(shape=(K+1,D))
    eps_ball[0] = Y[ns+T]
    for (i, j) in enumerate(NNidxs):
        eps_ball[i+1] = Y[j+T]

    # compute center of mass
    u_k = np.sum(eps_ball,axis=0) / (K+1)   # Eq. 14

    E2_sum = 0
    for j in range(K+1):
        if norm == 'euclidean':
            E2_sum += scipy.spatial.distance.euclidean(eps_ball[j], u_k)**2
        elif norm == 'chebyshev':
            E2_sum += scipy.spatial.distance.chebyshev(eps_ball[j], u_k)**2
    
    E2 = E2_sum / (K+1)  # Eq. 13
    return E2       
