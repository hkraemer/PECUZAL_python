#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In the following the main function :py:func:`pecuzal_embedding.pecuzal_embedding` and all its auxilary functions are listed. 
"""

import math
import numpy as np
import scipy
import random
from sklearn.neighbors import KDTree
from scipy.stats import binom, zscore

def pecuzal_embedding(s, taus = range(50), theiler = 1, sample_size = 1., K = 13, KNN = 3, Tw_factor = 4, alpha = 0.05, p = 0.5, max_cycles = 50):
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
    Tw_factor : `int`, optional
        The maximal considered time horizon for obtaining the L-statistic as a factor getting multiplied with the given `theiler`:
        `Tw = Tw_factor * theiler` and Default is `Tw_factor = 4`. If `theiler` is set to, say, `theiler = 15` and `Tw_factor` is on its Default
        the maximal considered time horizon `Tw` for obtaining the L-statistic is `Tw = Tw_factor * 15 = 4 * 15 = 60`.
    alpha : `float`, optional
        Significance level for obtaining the continuity statistic `avrg_eps_star` in each embedding cycle (Default is `alpha = 0.05`).
    p : `float`, optional
        Binominal p for obtaining the continuity statistic `avrg_eps_star` in each embedding cycle (Default is `p = 0.5`).
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
    `Y`, as proposed in [kraemer2020]_ . Based on the continuity statistic `avrg_eps_star` [pecora2007]_ the algorithm picks an
    optimal delay value `tau_i` for each embedding cycle `i`. For achieving that, we take the inpute time series 
    `s` and compute the continuity statistic `avrg_eps_star`. 1. Each local maxima in `avrg_eps_star` is used 
    for constructing a candidate embedding trajectory `Y_trial` with a delay corresponding to that
    specific peak in `avrg_eps_star`. 2. We then compute the `L`-statistic [uzal2011]_ for `Y_trial`. 3. We pick the 
    peak/`tau`-value, for which `L` is minimal and construct the actual embedding trajectory `Y_actual` (steps 
    1.-3. correspond to an embedding cycle). 4. We repeat steps 1.-3. with `Y_actual` as input and stop the
    algorithm when `L` can not be reduced anymore. `Y_actual` -> `Y`.

    In case of multivariate embedding, i.e. when embedding a set of `M` time series, in each embedding cycle 
    `avrg_eps_star` gets computed for all `M` time series available. The optimal delay value `tau_i` in each 
    embedding cycle `i` is chosen as the peak/`tau`-value for which `L` is minimal under all available peaks 
    and under all M `avrg_eps_star`'s. In the first embedding cycle there will be M**2 different `avrg_eps_star`'s
    to consider, since it is not clear a priori which time series of the input should consitute the first component 
    of the embedding vector and form `Y_actual`.

    For distance computations the Euclidean norm is used.

    References
    ----------
    .. [pecora2007] Pecora et al., "A unified approach to attractor reconstruction", Chaos, vol. 17, 013110, 2007.  https://doi.org/10.1063/1.2430294
    .. [uzal2011] Uzal et al., "Optimal reconstruction of dynamical systems: A noise amplification approach", Physical Review E,
        vol. 84, 016223, 2011. https://doi.org/10.1103/PhysRevE.84.016223
    '''    
    if np.ndim(s)>1:
        assert (np.size(s,0) > np.size(s,1)), "You must provide a numpy array storing the time series in its columns."
        D = np.size(s,1)
    else:
        D = 1
    assert (K >= 8) and (type(K) is int) and (K < len(s)) , "You must provide a delta-neighborhood size consisting of at least 8 neighbors."
    assert (KNN >= 1) and (type(KNN) is int), "You must provide a valid integer number of considered nearest neighbours for the computation of the L-statistic." 
    assert (sample_size > 0) and (sample_size <= 1), "sample_size must be in (0 1]"
    assert (theiler >= 0) and (type(theiler) is int) and (theiler < len(s)), "Theiler window must be a positive integer smaller than the time series length."
    assert (alpha >= 0) and (alpha < 1), "Significance level alpha must be in (0 1)"
    assert (p >= 0) and (p < 1), "Binomial p parameter must be in (0 1)"
    norm = 'euclidean' 
    Tw = Tw_factor*theiler # set time horizon for L-statistic

    s_orig = s
    s = zscore(s) # especially important for comparative L-statistics
    # define actual phase space trajectory
    Y_act = []
    # compute initial L values for each time series
    if D>1:
        L_inits = np.zeros(D)
        for i in range(D):
            L_inits[i], _ = uzal_cost(s[:,i], sample_size = sample_size, K = KNN, norm = norm, theiler = theiler, Tw = Tw)
        L_init = np.amin(L_inits)
    else:
        L_init, _ = uzal_cost(s, sample_size = sample_size, K = KNN, norm = norm, theiler = theiler, Tw = Tw)

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag, counter = True, 0

    # preallocate output variables
    tau_vals = [0]
    ts_vals = []
    Ls = []
    eps = np.empty(shape=(len(taus), max_cycles))

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag:
        Y_act, tau_vals, ts_vals, Ls, eps = pecuzal_multivariate_embedding_cycle(
                Y_act, flag, s, taus, theiler, counter, eps, tau_vals, norm,
                Ls, ts_vals, sample_size, K, alpha, p, Tw, KNN)

        flag = pecuzal_break_criterion(Ls, counter, max_cycles, L_init)
        counter += 1
    
    # construct final reconstruction vector
    if D > 1:
        Y_final = s_orig[:,ts_vals[0]]
        for i in range(len(tau_vals[:-2])):
            Y_final = hcat_lagged_values(Y_final,s_orig[:,ts_vals[i+1]],tau_vals[i+1])
    else:
        Y_final = s_orig
        for i in range(len(tau_vals[:-2])):
            Y_final = hcat_lagged_values(Y_final,s_orig,tau_vals[i+1])       
    
    return Y_final, tau_vals[:-1], ts_vals[:-1], Ls, eps[:,:counter]


def pecuzal_multivariate_embedding_cycle(Y_act, flag, Ys, taus, theiler, counter, eps, tau_vals, norm,
        Ls, ts_vals, sample_size, K, alpha, p, Tw, KNN):
    '''Perform one embedding cycle on `Y_act` with a multivariate set Ys
    '''
    if np.ndim(Ys)>1:
        M = np.size(Ys,1)
    else:
        M = 1

    # in the 1st cycle we have to check all (size(Y,2)^2 combinations and pick
    # the tau according to minimial xi = (peak height * resulting L-statistic)
    if counter == 0:
        Y_act, tau_vals, ts_vals, Ls, eps = first_embedding_cycle_pecuzal(Ys, M, taus, theiler, sample_size, K,
                                norm, alpha, p, Tw, KNN, tau_vals, ts_vals, Ls, eps)
    # in all other cycles we just have to check (size(Y,2)) combinations and pick
    # the tau according to minimal resulting L-statistic
    else:
        Y_act, tau_vals, ts_vals, Ls, eps = embedding_cycle_pecuzal(Y_act, Ys, counter, M, taus, theiler, sample_size,
                            K, norm, alpha, p, Tw, KNN, tau_vals, ts_vals, Ls, eps)

    return Y_act, tau_vals, ts_vals, Ls, eps



def first_embedding_cycle_pecuzal(Ys, M, taus, theiler, sample_size, K,
                        norm, alpha, p, Tw, KNN, tau_vals, ts_vals, Ls, eps):
    '''Perform the first embedding cycle of the multivariate embedding.
    '''
    counter = 0

    if M > 1:
        L_min = np.zeros(M)
        L_min_idx = np.zeros(M, dtype=int)
        idx = np.zeros(M, dtype=int)
        xi_min = np.zeros(M)
        estar = np.zeros(shape=(len(taus), M*M))
        for ts in range(M):
            estar[:,(M*ts):(M*(ts+1))] = continuity_statistic(Ys, [0], [ts], delays = taus, sample_size = sample_size, 
                                                K = K, theiler = theiler, norm = norm, alpha = alpha, p = p)
            
            L_min[ts], L_min_idx[ts], idx[ts], xi_min[ts] = choose_right_embedding_params_first(
                                            estar[:,(M*ts):(M*(ts+1))], Ys[:,ts],
                                            Ys, taus, Tw, KNN, theiler, sample_size,
                                            norm)
  
        min_idx = np.argmin(xi_min)
        if np.ndim(min_idx) > 0:
            min_idx = min_idx[0]
        L_mini = L_min[min_idx]
        # update tau_vals, ts_vals, Ls
        tau_vals.append(taus[L_min_idx[min_idx]])
        ts_vals.append(min_idx) # time series to start with
        ts_vals.append(idx[min_idx]) # result of 1st embedding cycle
        Ls.append(L_mini) 
        eps[:,counter] = estar[:,(M*ts_vals[0])+ts_vals[1]]
        # create phase space vector for this embedding cycle
        Y_act = hcat_lagged_values(Ys[:,ts_vals[0]],
                                    Ys[:,ts_vals[1]],tau_vals[1])
    else:
        estar = continuity_statistic(Ys, [0], [0], delays = taus, sample_size = sample_size, 
                                        K = K, theiler = theiler, norm = norm, alpha = alpha, p = p)
        L_min, L_min_idx, idx, _ = choose_right_embedding_params_first(
                                        estar, Ys, Ys, taus, Tw, KNN, theiler, sample_size, norm)
        # update tau_vals, ts_vals, Ls
        tau_vals.append(taus[L_min_idx])
        ts_vals.append(0) # time series to start with
        ts_vals.append(idx) # result of 1st embedding cycle
        Ls.append(L_min)
        eps[:,counter] = estar
        # create phase space vector for this embedding cycle
        Y_act = hcat_lagged_values(Ys,Ys,tau_vals[1])   

    return Y_act, tau_vals, ts_vals, Ls, eps


def embedding_cycle_pecuzal(Y_act, Ys, counter, M, taus, theiler, sample_size,
                    K, norm, alpha, p, Tw, KNN, tau_vals, ts_vals, Ls, eps):
    """Perform an embedding cycle of the multivariate embedding, but the first one.
    """
    
    estar = continuity_statistic(Ys, tau_vals, ts_vals, delays = taus, sample_size = sample_size, 
                                    K = K, theiler = theiler, norm = norm, alpha = alpha, p = p)
    # update tau_vals, ts_vals, Ls, eps
    L_min, L_min_idx, idx = choose_right_embedding_params(estar, Y_act, Ys, taus, Tw, KNN, theiler, sample_size, norm)

    tau_vals.append(taus[L_min_idx])
    ts_vals.append(idx)
    Ls.append(L_min)
    if np.ndim(Ys)>1:
        eps[:,counter] = estar[:,ts_vals[-1]]
        # create phase space vector for this embedding cycle
        Y_act = hcat_lagged_values(Y_act, Ys[:, ts_vals[-1]], tau_vals[-1])
    else:
        eps[:,counter] = estar
        # create phase space vector for this embedding cycle
        Y_act = hcat_lagged_values(Y_act, Ys, tau_vals[-1])  

    return Y_act, tau_vals, ts_vals, Ls, eps



def choose_right_embedding_params_first(estar, Y_act, s, taus, Tw, KNN, theiler, sample_size, norm):
    '''Choose the right embedding parameters of the estar-statistic in the first
    embedding cycle on the basis of minimal `xi` = (peak height * resulting `L`-statistic).
    
    Parameters
    ----------
    estar : `numpy.ndarray` (len(taus), M)
        The M continuity statistic(s) for delays `taus`.
    Y_act : `numpy.ndarray`
        The actual phase space trajectory.
    s : numpy.ndarray` (N, M)
        The `M` time series of length `N`.
    taus : `iterable`
        Possible delay values in sampling time units.
    Tw : `int`
        The maximal considered time horizon for obtaining the L-statistic.
    KNN : `int`
        The number of nearest neighbors to be considered in the L-statistic.
    theiler : `int`
        Theiler window for excluding serial correlated points from neighbourhood, in sampling time units.
    sample_size : `float`
        Number of considered fiducial points as a fraction of input time series length, i.e. a float from interval (0,1.].
    norm : `str`
        The norm used for distance computations.

    Returns
    -------
    L : `float`
        `L`-value of chosen peak in `estar`
    tau_idx : `int`
        The corresponding index value of the chosen peak 
    ts_idx : `int`
        The number of the chosen time series to start with
    xi_min : `float`
        The minimum `xi = (peak height * resulting L-statistic)` according to the chosen peak.

    See also
    --------
    continuity_statistic
    '''
    D = np.ndim(s)
    if D > 1:
        xi_min_ = np.zeros(D)
        L_min_ = np.zeros(D)
        tau_idx = np.zeros(D, dtype=int)
        for ts in range(D):
            # zero-padding of estar in order to also cover tau=0 (important for the multivariate case)
            # get the L-statistic for each peak in estar and take the one according to L_min
            L_trials_, max_idx_, xi_trials_ = local_L_statistics(np.insert(estar[:,ts],0,0), Y_act, s[:,ts],
                                            taus, Tw, KNN, theiler, sample_size, norm)
            min_idx_ = np.argmin(xi_trials_)
            if np.ndim(min_idx_)>0:
                min_idx_ = min_idx_[0]
            xi_min_[ts] = xi_trials_[min_idx_]
            L_min_[ts] = L_trials_[min_idx_]
            tau_idx[ts] = max_idx_[min_idx_]-1

        idx = np.argmin(xi_min_)
        if np.ndim(idx)>0:
            idx = idx[0]

        return L_min_[idx], tau_idx[idx], idx, xi_min_[idx]
    else:
        L_trials_, max_idx_, xi_trials_ = local_L_statistics(np.insert(estar,0,0), Y_act, s,
                                    taus, Tw, KNN, theiler, sample_size, norm)
        min_idx_ = np.argmin(xi_trials_)
        if np.ndim(min_idx_)>0:
            min_idx_ = min_idx_[0]
        xi_min_ = xi_trials_[min_idx_]
        L_min_ = L_trials_[min_idx_]
        tau_idx = max_idx_[min_idx_]-1

        return L_min_, tau_idx, 0, xi_min_



def choose_right_embedding_params(estar, Y_act, s, taus, Tw, KNN, theiler, sample_size, norm):
    '''Choose the right embedding parameters of the estar-statistic in any embedding cycle, but the
    first one, on the basis of minimal `L`.
    
    Parameters
    ----------
    estar : `numpy.ndarray` (len(taus), M)
        The M continuity statistic(s) for delays `taus`.
    Y_act : `numpy.ndarray`
        The actual phase space trajectory.
    s : numpy.ndarray` (N, M)
        The `M` time series of length `N`.
    taus : `iterable`
        Possible delay values in sampling time units.
    Tw : `int`
        The maximal considered time horizon for obtaining the L-statistic.
    KNN : `int`
        The number of nearest neighbors to be considered in the L-statistic.
    theiler : `int`
        Theiler window for excluding serial correlated points from neighbourhood, in sampling time units.
    sample_size : `float`
        Number of considered fiducial points as a fraction of input time series length, i.e. a float from interval (0,1.].
    norm : `str`
        The norm used for distance computations.

    Returns
    -------
    L : `float`
        `L`-value of chosen peak in `estar`
    tau_idx : `int`
        The corresponding index value of the chosen peak 
    ts_idx : `int`
        The number of the chosen time series to start with

    See also
    --------
    continuity_statistic
    '''
    D = np.ndim(s)
    if D > 1:
        L_min_ = np.zeros(D)
        tau_idx = np.zeros(D, dtype=int)
        for ts in range(D):
            # zero-padding of estar in order to also cover tau=0 (important for the multivariate case)
            # get the L-statistic for each peak in estar and take the one according to L_min
            L_trials_, max_idx_, _ = local_L_statistics(np.insert(estar[:,ts],0,0), Y_act, s[:,ts],
                                            taus, Tw, KNN, theiler, sample_size, norm)
            min_idx_ = np.argmin(L_trials_)
            if np.ndim(min_idx_)>0:
                min_idx_ = min_idx_[0]
            L_min_[ts] = L_trials_[min_idx_]
            tau_idx[ts] = max_idx_[min_idx_]-1

        idx = np.argmin(L_min_)
        if np.ndim(idx)>0:
            idx = idx[0]

        return L_min_[idx], tau_idx[idx], idx
    else:
        L_trials_, max_idx_, _ = local_L_statistics(np.insert(estar,0,0), Y_act, s,
                                    taus, Tw, KNN, theiler, sample_size, norm)
        min_idx_ = np.argmin(L_trials_)
        if np.ndim(min_idx_)>0:
            min_idx_ = min_idx_[0]
        L_min_ = L_trials_[min_idx_]
        tau_idx = max_idx_[min_idx_]-1

        return L_min_, tau_idx, int(0)



def local_L_statistics(estar, Y_act, s, taus, Tw, KNN, theiler, sample_size, norm):
    '''Return the L-statistic `L` and indices `max_idx` and weighted peak height
    `xi = peak-height * L` for all local maxima in `estar`.
    '''
    maxima, max_idx = get_maxima(estar) # determine local maxima in estar
    if np.ndim(max_idx)>0:
        L_trials = np.zeros(len(max_idx), dtype=float)
        xi_trials = np.zeros(len(max_idx), dtype=float)

    for (i,tau_idx) in enumerate(max_idx):
        # create candidate phase space vector for this peak/τ-value
        Y_trial = hcat_lagged_values(Y_act,s,taus[int(tau_idx)-1])
        # compute L-statistic
        if np.ndim(max_idx)>0:
            L_trials[i], _ = uzal_cost(Y_trial, Tw = Tw, K = KNN, theiler = theiler,
                    sample_size = sample_size, norm = norm)
            xi_trials[i] = L_trials[i]*maxima[i]
        else:
            L_trials, _ = uzal_cost(Y_trial, Tw = Tw, K = KNN, theiler = theiler,
                    sample_size = sample_size, norm = norm)
            xi_trials = L_trials*maxima          
    
    return L_trials, max_idx, xi_trials



def get_maxima(s):
    '''Return the maxima of the given time series `s` and its indices
    '''
    maximas = []
    maximas_idx = []
    N = len(s)
    flag = False
    first_point = 0
    for i in range(N-1):
        if (s[i-1] < s[i]) and (s[i+1] < s[i]):
            flag = False
            maximas.append(s[i])
            maximas_idx.append(i)
        
        # handling constant values
        if flag:
            if s[i+1] < s[first_point]:
                flag = False
                maximas.append(s[first_point])
                maximas_idx.append(first_point)
                
            elif s[i+1] > s[first_point]:
                flag = False
            
        if (s[i-1] < s[i]) and (s[i+1] == s[i]):
            flag = True
            first_point = i
        
    # make sure there is no empty vector returned
    if np.size(maximas) == 0:
        maximas_idx = np.argmax(s)
        if np.ndim(maximas_idx)>0:
            maximas_idx = maximas_idx[0]
        maximas = s[maximas_idx]
    
    return maximas, maximas_idx



def hcat_lagged_values(Y, s, tau):
    '''Add the `tau` lagged values of the timeseries `s` as additional component to `Y`
    (`np.ndarray`), in order to form a higher embedded dataset `Z`. 
    The dimensionality of `Z` is thus equal to that of `Y` + 1.
    '''
    assert tau >= 0
    N = len(Y)
    try:
        D = np.size(Y,1)
    except IndexError:
        D = 1
    assert N <= len(s)
    M = N - tau
    data = np.empty(shape=(M,D+1))

    if D > 1:
        data[:,:-1] = Y[:M,:]
    else:
        data[:,0] = Y[:M]

    for i in range(M):
        data[i,-1] = s[i+tau]
         
    return data


def pecuzal_break_criterion(Ls, counter, max_num_of_cycles, L_init):
    '''Checks whether some break criteria are fulfilled
    '''
    flag = True
    if counter == 0:
        if Ls[-1] > L_init:
            print("Algorithm stopped due to increasing L-values. Valid embedding NOT achieved.")
            flag = False
        
    if counter > 0 and Ls[-1]>Ls[-2]:
        print("Algorithm stopped due to minimum L-value reached. VALID embedding achieved.")
        flag = False

    if max_num_of_cycles == counter:
        print("Algorithm stopped due to hitting max cycle number. Valid embedding NOT achieved.")
        flag = False

    return flag

def continuity_statistic(s, taus, js, delays = range(50), sample_size = 1.0, K = 13, theiler = 1, norm = 'euclidean', alpha = 0.05, p = 0.5):
    '''Compute the continuity statistic for a trajectory defined by `s`, `taus` and `js` and all time series stored in `s`.

    Parameters
    ----------
    s : `numpy.ndarray` (N, M)
        Input time series of length `N`. This can be a multivariate set, consisting of `M` time series, which are stored in the columns.
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
        Number of considered fiducial points as a fraction of input time series length, i.e. a float :math:`\\in (0,1.]` (Default is 
        `sample_size = 1.0`, i.e., all points of the acutal trajectory get considered).
    K : `int`, optional
        The amount of nearest neighbors in the :math:`\\delta`-ball. Must be at least 8 (in order to guarantee a valid statistic) and the Default is
        `K = 13`. The continuity statistic `avrg_eps_star` is computed by taking the minimum result over all :math:`k \\in K`.
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
    The full algorithm is too large to discuss here and written in detail in Ref. [pecora2007]_ and in summary in Ref. [kraemer2020]_ .
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
    If `js = (0, 2, 1)` and `taus = (0, 2, 7)` the created delay vector at each step `t` will be

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
    '''Compute the numbers of points from the :math:`\\delta`-neighborhood, which need to fall outside the :math:`\\varepsilon`-neighborhood, in order to reject 
    the Null Hypothesis at a significance level :math:`\\alpha`.

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
    of the :math:`\\delta`-neighborhood. `trial_range` determines the number of considered :math:`\\delta`-neighborhood-points, always starting from 8. For 
    instance, if `trial_range = 8`, then :math:`\\delta`-neighborhood sizes from 8 up to 15 are considered.
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
    '''Compute the L-statistic for the trajectory `Y`.

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
    direct measure of local stretch, which constitutes an irrelevance measure [uzal2011]_. Technically, it is the logarithm of the product of the
    :math:`\\sigma`-statistic and a normalization statistic :math:`\\alpha` :

    .. math:: 
        L = log_{10}(\\sigma\\alpha)

    The :math:`\\sigma`-statistic is computed as follows. :math:`\\sigma = \\sqrt{\\sigma^2} = \\sqrt{E^2/\\varepsilon^2}`.
    :math:`E^2` approximates the conditional variance at each point in state space and for a time horizon :math:`T \\in Tw`, using :math:`K` 
    nearest neighbors. For each reference point of the state space trajectory `Y`, the neighborhood consists of the reference point 
    itself and its `K+1` nearest neighbors. :math:`E^2` measures how strong a neighborhood expands during `T` time steps. 
    :math:`E^2` is averaged over many time horizons :math:`T = 1:Tw`. Consequently, :math:`\\varepsilon^2` is the size of the neighborhood at 
    the reference point itself and is defined as the mean pairwise distance of the neighborhood. Finally, :math:`\\sigma^2` gets averaged over 
    a range of reference points on the attractor, which is controlled by `sample_size`. This is just for performance reasons and the most 
    accurate result will obviously be gained when setting `sample_size=1.0`.

    The :math:`\\alpha`-statistic is a normalization factor, such that :math:`\\sigma`'s from different embeddings can be compared. :math:`\\alpha^2`
    is defined as the inverse of the sum of the inverse of all :math:`\\varepsilon^2`'s for all considered reference points.
    '''

    assert (theiler >= 0) and (type(theiler) is int) and (theiler < len(Y))
    assert (K >= 0) and (type(K) is int) and (K < len(Y))
    assert (sample_size > 0) and (sample_size <= 1)
    assert (Tw >= 0) and (type(Tw) is int) and (Tw < len(Y))
    assert (type(norm) is str) and (norm == 'euclidean' or norm == 'chebyshev')

    if np.ndim(Y)>1:
        assert (np.size(Y,0) > np.size(Y,1)), "You must provide a numpy array storing the time series in its columns."
        D = np.size(Y,1)
    else:
        D = 1
    NN = len(Y)-Tw
    if sample_size == 1:
        NNN = NN
        ns = [i for i in range(NN)]
    else:
        NNN = int(np.floor(sample_size*NN))
        ns = random.sample(range(NN),NNN) # the fiducial point indices

    vs = Y[ns[:]] # the fiducial points in the data set

    if D == 1:
        allNNidxs, _ = all_neighbors_1dim(vs, ns, K, theiler)
        neighborhood_v = np.empty(K+1)
    else:
        vtree = KDTree(Y[:-Tw], metric = norm)
        allNNidxs, _ = all_neighbors(vtree, vs, ns, K, theiler, (len(vs)-Tw))
        neighborhood_v = np.empty(shape=(K+1,D)) 
        
    eps2 = np.empty(NNN)             # neighborhood size
    E2_avrg = np.empty(NNN)          # averaged conditional variance
    E2 = np.empty(Tw)                # condition variance 
    

    # loop over each fiducial point
    for (i,v) in enumerate(vs):
        NNidxs = allNNidxs[i] # indices of K nearest neighbors to v

        # construct local neighborhood   
        neighborhood_v[0] = v            
        neighborhood_v[1:] = Y[NNidxs]

        # pairwise distance of fiducial points and `v`
        if D == 1:
            neighborhood_v_ = np.empty(shape=(len(neighborhood_v),2))
            neighborhood_v_[:,0] = neighborhood_v
            neighborhood_v_[:,1] = neighborhood_v
            pdsqrd = scipy.spatial.distance.pdist(neighborhood_v_, 'chebyshev')
        else:
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
    
def all_neighbors_1dim(vs, ns, K, theiler):
    '''Compute `K` nearest neighbours for the points `vs` (having indices `ns`), while respecting the `theiler`-window for 1-d arrays.

    Returns
    -------
    indices : `numpy.ndarray` (len(vs),K)
        The indices of the K-nearest neighbours of all points `vs` (having indices `ns`)
    dists : `numpy.ndarray` (len(vs),K)
        The distances to the K-nearest neighbours of all points `vs` (having indices `ns`)
    '''
    dists = np.empty(shape=(len(vs),K))  
    idxs = np.empty(shape=(len(vs),K),dtype=int)

    for (i,v) in enumerate(vs):
        dis = np.array([abs(v - vs[j]) for j in range(len(vs))])
        idx = np.argsort(dis)
        cnt = 0
        for j in range(len(idxs)):
            if idx[j] < ns[i]-theiler or idx[j] > ns[i]+theiler:
                dists[i,cnt], idxs[i,cnt] = dis[idx[j]], idx[j]
                if cnt == K-1:
                    break
                else:
                    cnt += 1
    return idxs, dists


def comp_Ek2(Y, ns, NNidxs, T, K, norm):
    '''Compute the approximated conditional variance for a specific point in `Y` (with index `ns`) for time horizon `T`.

    Returns
    -------
    E2 : `float`
        The approximated conditional variance for a specific point in state space `ns` (index value) with its `K`-nearest neighbors, 
        which indices are stored in `NNidxs`, for a time horizon `T`. This corresponds to Eqs. 13 & 14 in [uzal2011]_.
    '''
    if np.ndim(Y)>1:
        assert (np.size(Y,0) > np.size(Y,1)), "You must provide a numpy array storing the time series in its columns."
        D = np.size(Y,1)
        eps_ball = np.empty(shape=(K+1,D))
    else:
        D = 1
        eps_ball = np.empty(K+1)
    # determine neighborhood `T` time steps ahead
    
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


def mi(x, maxlag = 50):
    """Compute the auto mutual information of a time series `x` up to a lag `maxlag`.

    Parameters
    ----------
    x : `numpy.ndarray`
        Numpy array storing the time series values.
    maxlag : `int`, optional
        The maximum lag in sampling units, i.e. an integer value

    Returns
    -------
    mi : `numpy.ndarray` (len(range(maxlag)))
        The auto mutual information of the given time series at each considered lag.
    lags : `numpy.ndarray` (len(range(maxlag)))
        The considered lags
    """

    assert (type(maxlag) is int) and (maxlag > 0)
    # initialize variables
    binrule="fd"
    x = zscore(x)
    n = len(x)
    lags = np.arange(0, maxlag, dtype="int")
    mi = np.zeros(len(lags))
    # loop over lags and get MI
    for i, lag in enumerate(lags):
        # extract lagged data
        y1 = x[:n - lag].copy()
        y2 = x[lag:].copy()
        # use np.histogram to get individual entropies
        H1, be1 = entropy1d(y1, binrule)
        H2, be2 = entropy1d(y2, binrule)
        H12, _, _ = entropy2d(y1, y2, [be1, be2])
        # use the entropies to estimate MI
        mi[i] = H1 + H2 - H12

    return mi, lags


def entropy1d(x, binrule):
    """
    Returns the Shannon entropy according to the bin rule specified.
    """
    p, be = np.histogram(x, bins=binrule, density=True)
    r = be[1:] - be[:-1]
    P = p * r
    H = -(P * np.log2(P)).sum()

    return H, be


def entropy2d(x, y, bin_edges):
    """
    Returns the Shannon entropy according to the bin rule specified.
    """
    p, bex, bey = np.histogram2d(x, y, bins=bin_edges, normed=True)
    r = np.outer(bex[1:] - bex[:-1], bey[1:] - bey[:-1])
    P = p * r
    H = np.zeros(P.shape)
    i = ~np.isinf(np.log2(P))
    H[i] = -(P[i] * np.log2(P[i]))
    H = H.sum()

    return H, bex, bey

# """Example NumPy style docstrings.

# This module demonstrates documentation as specified by the `NumPy
# Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
# are created with a section header followed by an underline of equal length.

# Example
# -------
# Examples can be given using either the ``Example`` or ``Examples``
# sections. Sections support any reStructuredText formatting, including
# literal blocks::

#     $ python example_numpy.py


# Section breaks are created with two blank lines. Section breaks are also
# implicitly created anytime a new section starts. Section bodies *may* be
# indented:

# Notes
# -----
#     This is an example of an indented section. It's like any other section,
#     but the body is indented to help it stand out from surrounding text.

# If a section is indented, then a section break is created by
# resuming unindented text.

# Attributes
# ----------
# module_level_variable1 : int
#     Module level variables may be documented in either the ``Attributes``
#     section of the module docstring, or in an inline docstring immediately
#     following the variable.

#     Either form is acceptable, but the two should not be mixed. Choose
#     one convention to document module level variables and be consistent
#     with it.


# .. _NumPy Documentation HOWTO:
#    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

# """

# module_level_variable1 = 12345

# module_level_variable2 = 98765
# """int: Module level variable documented inline.

# The docstring may span multiple lines. The type may optionally be specified
# on the first line, separated by a colon.
# """

# def function_with_types_in_docstring(param1, param2):
#     """Example function with types documented in the docstring.

#     `PEP 484`_ type annotations are supported. If attribute, parameter, and
#     return types are annotated according to `PEP 484`_, they do not need to be
#     included in the docstring:

#     Parameters
#     ----------
#     param1 : int
#         The first parameter.
#     param2 : str
#         The second parameter.

#     Returns
#     -------
#     bool
#         True if successful, False otherwise.

#     .. _PEP 484:
#         https://www.python.org/dev/peps/pep-0484/

#     """


# def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
#     """Example function with PEP 484 type annotations.

#     The return type must be duplicated in the docstring to comply
#     with the NumPy docstring style.

#     Parameters
#     ----------
#     param1
#         The first parameter.
#     param2
#         The second parameter.

#     Returns
#     -------
#     bool
#         True if successful, False otherwise.

#     """


# def module_level_function(param1, param2=None, *args, **kwargs):
#     """This is an example of a module level function.

#     Function parameters should be documented in the ``Parameters`` section.
#     The name of each parameter is required. The type and description of each
#     parameter is optional, but should be included if not obvious.

#     If ``*args`` or ``**kwargs`` are accepted,
#     they should be listed as ``*args`` and ``**kwargs``.

#     The format for a parameter is::

#         name : type
#             description

#             The description may span multiple lines. Following lines
#             should be indented to match the first line of the description.
#             The ": type" is optional.

#             Multiple paragraphs are supported in parameter
#             descriptions.

#     Parameters
#     ----------
#     param1 : int
#         The first parameter.
#     param2 : :obj:`str`, optional
#         The second parameter.
#     *args
#         Variable length argument list.
#     **kwargs
#         Arbitrary keyword arguments.

#     Returns
#     -------
#     bool
#         True if successful, False otherwise.

#         The return type is not optional. The ``Returns`` section may span
#         multiple lines and paragraphs. Following lines should be indented to
#         match the first line of the description.

#         The ``Returns`` section supports any reStructuredText formatting,
#         including literal blocks::

#             {
#                 'param1': param1,
#                 'param2': param2
#             }

#     Raises
#     ------
#     AttributeError
#         The ``Raises`` section is a list of all exceptions
#         that are relevant to the interface.
#     ValueError
#         If `param2` is equal to `param1`.

#     """
#     if param1 == param2:
#         raise ValueError('param1 may not be equal to param2')
#     return True


# def example_generator(n):
#     """Generators have a ``Yields`` section instead of a ``Returns`` section.

#     Parameters
#     ----------
#     n : int
#         The upper limit of the range to generate, from 0 to `n` - 1.

#     Yields
#     ------
#     int
#         The next number in the range of 0 to `n` - 1.

#     Examples
#     --------
#     Examples should be written in doctest format, and should illustrate how
#     to use the function.

#     >>> print([i for i in example_generator(4)])
#     [0, 1, 2, 3]

#     """
#     for i in range(n):
#         yield i


# class ExampleError(Exception):
#     """Exceptions are documented in the same way as classes.

#     The __init__ method may be documented in either the class level
#     docstring, or as a docstring on the __init__ method itself.

#     Either form is acceptable, but the two should not be mixed. Choose one
#     convention to document the __init__ method and be consistent with it.

#     Note
#     ----
#     Do not include the `self` parameter in the ``Parameters`` section.

#     Parameters
#     ----------
#     msg : str
#         Human readable string describing the exception.
#     code : :obj:`int`, optional
#         Numeric error code.

#     Attributes
#     ----------
#     msg : str
#         Human readable string describing the exception.
#     code : int
#         Numeric error code.

#     """

#     def __init__(self, msg, code):
#         self.msg = msg
#         self.code = code


# class ExampleClass:
#     """The summary line for a class docstring should fit on one line.

#     If the class has public attributes, they may be documented here
#     in an ``Attributes`` section and follow the same formatting as a
#     function's ``Args`` section. Alternatively, attributes may be documented
#     inline with the attribute's declaration (see __init__ method below).

#     Properties created with the ``@property`` decorator should be documented
#     in the property's getter method.

#     Attributes
#     ----------
#     attr1 : str
#         Description of `attr1`.
#     attr2 : :obj:`int`, optional
#         Description of `attr2`.

#     """

#     def __init__(self, param1, param2, param3):
#         """Example of docstring on the __init__ method.

#         The __init__ method may be documented in either the class level
#         docstring, or as a docstring on the __init__ method itself.

#         Either form is acceptable, but the two should not be mixed. Choose one
#         convention to document the __init__ method and be consistent with it.

#         Note
#         ----
#         Do not include the `self` parameter in the ``Parameters`` section.

#         Parameters
#         ----------
#         param1 : str
#             Description of `param1`.
#         param2 : list(str)
#             Description of `param2`. Multiple
#             lines are supported.
#         param3 : :obj:`int`, optional
#             Description of `param3`.

#         """
#         self.attr1 = param1
#         self.attr2 = param2
#         self.attr3 = param3  #: Doc comment *inline* with attribute

#         #: list(str): Doc comment *before* attribute, with type specified
#         self.attr4 = ["attr4"]

#         self.attr5 = None
#         """str: Docstring *after* attribute, with type specified."""

#     @property
#     def readonly_property(self):
#         """str: Properties should be documented in their getter method."""
#         return "readonly_property"

#     @property
#     def readwrite_property(self):
#         """list(str): Properties with both a getter and setter
#         should only be documented in their getter method.

#         If the setter method contains notable behavior, it should be
#         mentioned here.
#         """
#         return ["readwrite_property"]

#     @readwrite_property.setter
#     def readwrite_property(self, value):
#         value

#     def example_method(self, param1, param2):
#         """Class methods are similar to regular functions.

#         Note
#         ----
#         Do not include the `self` parameter in the ``Parameters`` section.

#         Parameters
#         ----------
#         param1
#             The first parameter.
#         param2
#             The second parameter.

#         Returns
#         -------
#         bool
#             True if successful, False otherwise.

#         """
#         return True

#     def __special__(self):
#         """By default special members with docstrings are not included.

#         Special members are any methods or attributes that start with and
#         end with a double underscore. Any special member with a docstring
#         will be included in the output, if
#         ``napoleon_include_special_with_doc`` is set to True.

#         This behavior can be enabled by changing the following setting in
#         Sphinx's conf.py::

#             napoleon_include_special_with_doc = True

#         """
#         pass

#     def __special_without_docstring__(self):
#         pass

#     def _private(self):
#         """By default private members are not included.

#         Private members are any methods or attributes that start with an
#         underscore and are *not* special. By default they are not included
#         in the output.

#         This behavior can be changed such that private members *are* included
#         by changing the following setting in Sphinx's conf.py::

#             napoleon_include_private_with_doc = True

#         """
#         pass

#     def _private_without_docstring(self):
#         pass