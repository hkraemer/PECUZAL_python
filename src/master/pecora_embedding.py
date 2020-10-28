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
from sklearn.neighbors import KDTree
from scipy.stats import binom


def pecuzal_embedding(s, taus = range(50), theiler = 1, sample_size = 1., K = 13, KNN = 3, Tw = 4*theiler, alpha = 0.05, p = 0.5, norm="euclidean", max_cycles = 50):
    '''Performs an embedding of time series using the PECUZAL method

    Parameters
    ----------
    s : 'numpy.ndarray' (N, M)
        Input time series of length N as numpy array. This can be a multivariate set, where the M timeseries are stored in the columns.
    taus : `iterable`, optional
        Possible delay values in sampling time units (Default is `taus=range(50)`). For each of the `tau`'s in `taus` the continuity statistic 
        `avrg_eps_star` gets computed and further processed in order to find optimal delays for each embedding cycle.
    theiler : `int`, optional
        Theiler window for excluding serial correlated points from neighbourhood. In sampling time units, Default is `theiler = 1`.
    sample_size : `float`, optional
        Number of considered fiducial points as a fraction of input time series length, i.e. a float from interval (0,1.] (Default is 
        `sample_size = 1.0`, i.e., all points of the acutal trajectory get considered).
    K : `int`, optional
        The amount of nearest neighbors in the Delta-ball. Must be at least 8 (in order to guarantee a valid statistic) and the Default is
        `K = 13`. The continuity statistic `avrg_eps_star` is computed in each embedding cycle, taking the minimum result over all `k in K`.
    KNN : `int`, optional
        The number of nearest neighbors to be considered in the L-statistic, Default is `KNN = 3`.  
    Tw : `int`, optional
        The maximal considered time horizon for obtaining the L-statistic. Default is `Tw = 4*theiler`.
    alpha : `float`, optional
        Significance level for obtaining the continuity statistic `avrg_eps_star` in each embedding cycle (Default is `alpha = 0.05`).
    p : `float`, optional
        Binominal p for obtaining the continuity statistic `avrg_eps_star` in each embedding cycle (Default is `p = 0.5`).
    norm : `str`, optional
        The norm used for distance computations. Must be either `'euclidean'` (Default) or `'chebyshev'`
    max_cycles : `int`, optional
        The algorithm will stop after that many cycles no matter what. Default is `max_cycles = 50`.
    
    Returns
    -------
    Y : 'numpy.ndarray' (N', m)
        The trajectory from the embedding of length `N' = N-sum(tau_vals)` of dimension `m` (embedding dimension)
    tau_vals : 'list' [`int`]
        The chosen delay values for each embedding cycle, `len(tau_vals) = m`.
    ts_vals : 'list' [`int`]
        The according time series number (index) chosen for each delay value in `tau_vals`, `len(ts_vals) = m`. For univariate embedding
        `ts_vals` is a vector of zeros of length `tau_vals`, because there is simply just one time series to choose from, i.e. index 0. 
    Ls : 'list'
        The L-statistic for each embedding cycle. The minimum of these values corresponds to the L-value for the returned
        trajectory `Y`.
    avrg_eps_stars : 'list' [`list`]
        The continuity statistics for each embedding cycle. Contains `avrg_eps_star` of each embedding cycle.
    
    See also
    --------
    uzal_cost
    continuity_statistic

    Notes
    -----
    The method works iteratively and gradually builds the final embedding vectors
    `Y`, as proposed in [1]_. Based on the continuity statistic `avrg_eps_star` [2]_ the algorithm picks an
    optimal delay value `tau_i` for each embedding cycle `i`. For achieving that, we take the inpute time series 
    `s` and compute the continuity statistic `avrg_eps_star`. 1. Each local maxima in `avrg_eps_star` is used 
    for constructing a candidate embedding trajectory `Y_trial` with a delay corresponding to that
    specific peak in `avrg_eps_star`. 2. We then compute the `L`-statistic [3]_ for `Y_trial`. 3. We pick the 
    peak/`tau`-value, for which `L` is minimal and construct the actual embedding trajectory `Y_actual` (steps 
    1.-3. correspond to an embedding cycle). 4. We repeat steps 1.-3. with `Y_actual` as input and stop the
    algorithm when `L` can not be reduced anymore. `Y_actual` -> `Y`.

    In case of multivariate embedding, i.e. when embedding a set of `M` time series, in each embedding cycle 
    `avrg_eps_star` gets computed for all `M` time series available. The optimal delay value `tau_i` in each 
    embedding cycle `i` is chosen as the peak/`tau`-value for which `L` is minimal under all available peaks 
    and under all M `avrg_eps_star`'s. In the first embedding cycle there will be M**2 different `avrg_eps_star`'s
    to consider, since it is not clear a priori which time series of the input should consitute the first component 
    of the embedding vector and form `Y_actual`.

    References
    ----------
    .. [1] Kraemer et al., "A unified and automated approach to attractor reconstruction", arXiv, vol. 22,
        pp. 585-588, 2020.
    .. [2] Pecora et al., "A unified approach to attractor reconstruction", Chaos, vol. 17, 013110, 2007.
    .. [3] Uzal et al., "Optimal reconstruction of dynamical systems: A noise amplification approach", Physical Review E,
        vol. 84, 016223, 2011.
    '''    
    pass




def continuity_statistic(s, taus, js, delays = range(50), sample_size = 1.0, K = 13, theiler = 1, norm = 'euclidean', alpha = 0.05, p = 0.5):
    '''Compute the continuity statistic for a trajectory defined by `s`, `taus` and `js` and all time series stored in `s` after [1]_.

    Parameters
    ----------
    s : `numpy.ndarray` (N, M)
        Input time series of length `N`. This can be a multivariate set, consisting of `M`time series, which are stored in the columns.
    taus : `list` or `numpy.ndarray`
        Denotes what delay times will be used for constructing the trajectory for which the continuity statistic to all time series in
        `s` will be computed.
    js : `list` or `numpy.ndarray`
        Denotes which of the timeseries contained in `s` will be used for constructing the trajectory using the delay values stored in
        `taus`. `js` can contain duplicate indices.
    delays : `iterable`, optional
        Possible delay values in sampling time units (Default is `delays = range(50)`). The continuity statistic `avrg_eps_star` is a 
        function of these delay values.
    sample_size : `float`, optional
        Number of considered fiducial points as a fraction of input time series length, i.e. a float from interval (0,1.] (Default is 
        `sample_size = 1.0`, i.e., all points of the acutal trajectory get considered).
    K : `int`, optional
        The amount of nearest neighbors in the Delta-ball. Must be at least 8 (in order to guarantee a valid statistic) and the Default is
        `K = 13`. The continuity statistic `avrg_eps_star` is computed by taking the minimum result over all `k in K`.
    theiler : `int`, optional
        Theiler window for excluding serial correlated points from neighbourhood. In sampling time units, Default is `theiler = 1`.
    norm : `str`, optional
        The norm used for distance computations. Must be either `'euclidean'` (Default) or `'chebyshev'`
    alpha : `float`, optional
        Significance level for obtaining the continuity statistic `avrg_eps_star` in each embedding cycle (Default is `alpha = 0.05`).
    p : `float`, optional
        Binominal p for obtaining the continuity statistic `avrg_eps_star` in each embedding cycle (Default is `p = 0.5`).

    Returns
    -------
    avrg_eps_star : `numpy.ndarray` (len(delays), M)
        The continuity statistic `avrg_eps_star` for each of the `M` time series in `s` for the given trajectory
        specified by the general embedding parameters `taus` and `js` and for all delay values specified in `delays`.

    See also
    --------
    pecuzal_embedding

    Notes
    -----
    The full algorithm is too large to discuss here and written in detail in [1]_ and in summary in [2].

    References
    ----------
    .. [1] Pecora et al., "A unified approach to attractor reconstruction", Chaos, vol. 17, 013110, 2007.
    .. [2] Kraemer et al., "A unified and automated approach to attractor reconstruction", arXiv, vol. 22,
        pp. 585-588, 2020.
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
    '''Perform an embedding with delays `taus` and time series stored in `s`, specified by their indices `js`

    Parameters
    ----------
    s : `numpy.ndarray` (N, M)
        Input time series of length `N`. This can be a multivariate set, consisting of `M`time series, which are stored in the columns.
    taus : `list` or `numpy.ndarray`
        Denotes what delay times will be used for constructing the trajectory for which the continuity statistic to all time series in
        `s` will be computed.
    js : `list` or `numpy.ndarray`
        Denotes which of the timeseries contained in `s` will be used for constructing the trajectory using the delay values stored in
        `taus`. `js` can contain duplicate indices.

    Returns
    -------
    Y : `numpy.ndarray` (N', d)
        The trajectory from the embedding of length `N' = N-sum(taus)` of dimension `d = len(taus)`.

    Notes
    -----
    The generalized embedding works as follows:
    `taus, js` are `list`'s (or `numpy.ndarray`'s) of length `d`, which also coincides with the embedding dimension. For example, imagine 
    input trajectory :math:`s = [x, y, z]` where :math:`x, y, z` are timeseries (the columns of `s`).
    If :math:`js = (0, 2, 1)` and :math:`taus = (0, 2, 7)` the created delay vector at each step `t` will be
    .. math:: (x(t), z(t+2), y(t+7))
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
    '''Compute the numbers of points from the delta-neighborhood, which need to fall outside the epsilon-neighborhood, in order to reject 
    the Null Hypothesis at a significance level `alpha`.

    Parameters
    ----------
    p : `float`, optional
        Binominal p (Default is `p = 0.5`).
    alpha : `float`, optional
        Significance level in order to be able to reject the Null on the basis of the binomial distribution (Default is `alpha = 0.05`).
    trial_range : `int`, optional
        Number of considered delta-neighborhood-points (Default is `trial_range = 8`).
    
    Returns
    -------
    delta_to_epsilon_amount : `dict`
        A dictionary with `delta_points` as keys and the corresponding number of points in order to reject the Null, `epsilon_points`, 
        constitute the values.

    Notes
    -----
    One parameter of the binomial distribution is `p`, the other one would be the number of trials, i.e. the considered number of points 
    of the delta-neighborhood. `trial_range` determines the number of considered delta-neighborhood-points, always starting from 8. For 
    instance, if `trial_range = 8`, then delta-neighborhood sizes from 8 up to 15 are considered.
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
    '''Compute the L-statistic for the trajectory `Y` after [1]_

    Parameters
    ----------
    Y : `numpy.ndarray` (N, d)
        State space vectors of length `N` and dimensionality `d`. This is usually an ouput from an embedding of some time series.
    K : `int`, optional
        The number of nearest neighbors to be considered in the L-statistic, Default is `K = 3`.
    Tw : `int`, optional
        The maximal considered time horizon for obtaining the L-statistic. Default is `Tw = 1`.
    theiler : `int`, optional
        Theiler window for excluding serial correlated points from neighbourhood. In sampling time units, Default is `theiler = 1`.
    sample_size : `float`, optional
        Number of considered fiducial points as a fraction of input trajectory length, i.e. a float from interval (0,1.] (Default is 
        `sample_size = 1.0`, i.e., all points of the acutal trajectory get considered).
    norm : `str`, optional
        The norm used for distance computations. Must be either `'euclidean'` (Default) or `'chebyshev'`
 
    Returns
    -------
    L : `float`
        The value of the proposed cost-function.
    L_local : `numpy.ndarray` (N',) 
        The local value of the proposed cost-function. Note that this output is only meaningful, if you have set `sample_size` = 1.0, 
        i.e. considering all points of the trajectory. The length of `L_local` is `N' = len(Y)-Tw`.

    Notes
    -----
    The `L`-statistic is based on theoretical arguments on noise amplification, the complexity of the reconstructed attractor and a 
    direct measure of local stretch, which constitutes an irrelevance measure [1]_. Technically, it is the logarithm of the product of the
    :math:`\sigma`-statistic and a normalization statistic :math:`\alpha`:
    .. math:: L = log10(\sigma*\alpha)

    The :math:`\sigma`-statistic is computed as follows. :math:`\sigma = \sqrt{\sigma^2} = \sqrt{E^2/\varepsilon^2}`.
    :math:`E^2` approximates the conditional variance at each point in state space and for a time horizon :math:`T \in Tw`, using :math:`K` 
    nearest neighbors. For each reference point of the state space trajectory :math:`Y`, the neighborhood consists of the reference point 
    itself and its :math:`K+1` nearest neighbors. :math:`E^2` measures how strong a neighborhood expands during :math:`T` time steps. 
    :math:`E^2` is averaged over many time horizons :math:`T = 1:Tw`. Consequently, :math:`\varepsilon^2` is the size of the neighborhood at 
    the reference point itself and is defined as the mean pairwise distance of the neighborhood. Finally, :math:`\sigma^2` gets averaged over 
    a range of reference points on the attractor, which is controlled by :math:`samplesize`. This is just for performance reasons and the most 
    accurate result will obviously be gained when setting :math:`sample_size=1.0`.

    The :math:`\alpha`-statistic is a normalization factor, such that :math:`\sigma`'s from different embeddings can be compared. :math:`\alpha^2`
    is defined as the inverse of the sum of the inverse of all :math:`\varepsilon^2`'s for all considered reference points.

    References
    ----------
    .. [1] Uzal et al., "Optimal reconstruction of dynamical systems: A noise amplification approach", Physical Review E,
        vol. 84, 016223, 2011.
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
    '''Compute `K` nearest neighbours for the points `vs` (having indices `ns`) from the tree `vtree`, while respecting the `theiler`-window.

    Returns
    -------
    indices : `numpy.ndarray` (len(vs),K)
        The indices of the K-nearest neighbours of all points `vs` (having indices `ns`)
    dists : `numpy.ndarray` (len(vs),K)
        The distances to the K-nearest neighbours of all points `vs` (having indices `ns`)
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
    '''Compute the approximated conditional variance for a specific point in `Y` (with index `ns`) for time horizon `T`.

    Returns
    -------
    E2 : `float``
        The approximated conditional variance for a specific point in state space `ns` (index value) with its `K`-nearest neighbors, 
        which indices are stored in `NNidxs`, for a time horizon `T`. This corresponds to Eqs. 13 & 14 in [1]_.

    References
    ----------
    .. [1] Uzal et al., "Optimal reconstruction of dynamical systems: A noise amplification approach", Physical Review E,
        vol. 84, 016223, 2011.
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
