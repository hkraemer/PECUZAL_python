#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:26:16 2020

@author: hkraemer
"""

import numpy as np
from scipy import spatial
import scipy.spatial.distance as dist
from scipy.stats import iqr


def stf2d(data, name):

    #save data to file named name
    N = len(data[:,0])
    M = len(data[0,:])

    f = open(name, 'w')
    for counterN in range(N): 
        for counterM in range(M):
            f.write(str(data[counterN,counterM]))
            f.write("\t")
        f.write("\n")
    f.close()


def stf1d(data, name):

    #save data to file named name
    N = len(data)

    f = open(name, 'w')
    for counterN in range(N): 
        f.write(str(data[counterN]))
        f.write("\n")
    f.close()


def compute_distances(X):
    
    # compute distance matrix of data
    distv = dist.pdist(X, 'euclidean')
    distm = dist.squareform(distv)

    return distm


def nn_scalar(idx, ts, Nnn):

    distts = np.abs(ts-ts[idx])
    nnidx = np.zeros(Nnn)
    nnidxtemp = np.argsort(distts)

    # exclude nearest temporal neighbours to each side
    countn = 0
    counti = 1
    while countn < Nnn:
        if nnidxtemp[counti] < idx-5 or nnidxtemp[counti] > idx+5:
            nnidx[countn] = nnidxtemp[counti]
            counti = counti + 1
            countn = countn + 1
        else:
            counti = counti + 1

    nnidx = nnidx.astype(int)

    return nnidx


def nn_vec(idx, ts, Nnn):

    distts = np.sqrt(np.sum((ts-ts[idx,:])**2, axis=1))
    nnidx = np.zeros(Nnn)
    nnidxtemp = np.argsort(distts)

    # exclude nearest temporal neighbours to each side
    countn = 0
    counti = 1
    while countn < Nnn:
        if nnidxtemp[counti] < idx-5 or nnidxtemp[counti] > idx+5:
            nnidx[countn] = nnidxtemp[counti]
            counti = counti + 1
            countn = countn + 1
        else:
            counti = counti + 1

    nnidx = nnidx.astype(int)

    return nnidx



def get_tau_distances(var, tau, fidx, nnidxs):

    Nnn = len(nnidxs)
    mdists = np.zeros(Nnn)
    ftau = var[fidx+tau]

    for count in range(Nnn):
        mdists[count] = np.abs(ftau - var[nnidxs[count]+tau])

    return mdists


def determine_max(taus, aveps):

    prevmax = 0
    currmax = 0
    follmax = max(aveps)

    maxcount = 0

    for count in range(1,len(taus)-1):

        #print maxcount

        if currmax > prevmax and currmax > follmax:
            break
        else:
            if aveps[count] >= aveps[count-1] and aveps[count] >= aveps[count+1]:

                if maxcount == 0:         
                    prevmax = aveps[count]
                    maxcount = maxcount +1
                elif maxcount == 1:
                    currmax = aveps[count]
                    restau = taus[count]
                    maxcount = maxcount + 1
                elif maxcount == 2:
                    follmax = aveps[count]
                    folltau = taus[count]
                    maxcount = maxcount + 1
                else:
                    prevmax = currmax
                    currmax = follmax
                    follmax = aveps[count]
                    restau = folltau
                    folltau = taus[count]
                    maxcount = maxcount + 1

    return restau


def determine_end(restaus, taus, aveps, avepsprev):

    # set thresholds
    thrslope = 0.
    thrmean = 0.1
    thrdim = 5 # set to very high number if manual break at specific dimension not wanted/required
    diffmean = 0.1
    diffslope = 0.01

    # get slope
    A = np.vstack([taus, np.ones(len(taus))]).T
    slope, n = np.linalg.lstsq(A, aveps)[0]

    # get mean 
    mean = np.mean(aveps)

    # get dimension
    dim = len(restaus) + 1

    #print slope
    #print mean
    #print dim

    if slope <= thrslope or mean <= thrmean or dim >= thrdim:
        end = 1
    else:
        end = 0    

    # check similarity of aveps to previous iteration
    meanprev = np.mean(avepsprev)

    B = np.vstack([taus, np.ones(len(taus))]).T
    slopeprev, nprev = np.linalg.lstsq(B, avepsprev)[0]

    #print meanprev
    #print slopeprev

    if abs(mean-meanprev) <= diffmean and abs(slope-slopeprev) <= diffslope:
        end = -1

    return end 



def nonuniform_embedding(var, t, restaus, resdim):

    maxtau = max(restaus)
    var_emb = np.zeros((len(var)-maxtau, resdim))

    var_emb[:,0] = var[:-maxtau]
    time = t[:-maxtau]
    
    for count in range(resdim-1):
        if restaus[count] == maxtau:
            var_emb[:,count+1] = var[maxtau:]
        else:
            var_emb[:,count+1] = var[restaus[count]:-(maxtau-restaus[count])]

    return [time, var_emb]




def embed_scalar(var, t, taus, Nfid):

    N = len(var)

    Ndl = np.arange(5,14)
    Nil = np.array([5,6,7,7,8,9,9,9,10])

    aveps = np.zeros((len(taus),len(Ndl)))

    # loop over taus
    tauc = 0

    for tau in taus:

        # get indices of fiducial points and loop over them
        mineps = np.zeros(Nfid)
        fidx = np.random.choice(N-1-tau,Nfid,replace=False)

        Ndlc = 0

        for Ndelta in Ndl:

            for cfid in range(Nfid):

                # get indices of Ndelta nearest neighbours of fiducial point
                nnidx = nn_scalar(fidx[cfid], var[:-tau], Ndelta)

                # get distances of neighbouring points to fiducial point in mappings
                mdists = get_tau_distances(var, tau, fidx[cfid], nnidx)

                rejectnull = 1. 
                epsilon = 5. # chosse automatically depending on range of time series
                de = 0.1

                while rejectnull == 1.:

                    # check how many of the nearest neighbours are mapped into epsilon range
                    ineps = 0
                    for count in range(Ndelta):
                        if mdists[count] <= epsilon: ##print mdists to check!!!
                            ineps = ineps + 1

                    if ineps >= Nil[Ndlc]:  
                        rejectnull = 1.
                        epsilon = epsilon - de
                    else:
                        rejectnull = 0.
                        mineps[cfid] = epsilon + de


            aveps[tauc,Ndlc] = np.mean(mineps)

            Ndlc = Ndlc + 1
            
        print(tauc)
        tauc = tauc + 1


    return [aveps, taus]




def embed_vec(var, t, taus, Nfid):

    N, dim = var.shape

    Ndl = np.arange(5,14)
    Nil = np.array([5,6,7,7,8,9,9,9,10])

    aveps = np.zeros((len(taus),len(Ndl)))

    # loop over taus
    tauc = 0

    for tau in taus:

        # get indices of fiducial points and loop over them
        mineps = np.zeros(Nfid)
        fidx = np.random.choice(N-1-tau,Nfid,replace=False)

        Ndlc = 0

        for Ndelta in Ndl:

            for cfid in range(Nfid):

                # get indices of Ndelta nearest neighbours of fiducial point
                nnidx = nn_vec(fidx[cfid], var[:-tau,:], Ndelta)

                # get distances of neighbouring points to fiducial point in mappings
                mdists = get_tau_distances(var[:,0], tau, fidx[cfid], nnidx)

                rejectnull = 1. 
                epsilon = 5.
                de = 0.1

                while rejectnull == 1.:

                    # check how many of the nearest neighbours are mapped into epsilon range
                    ineps = 0
                    for count in range(Ndelta):
                        if mdists[count] <= epsilon:
                            ineps = ineps + 1

                    if ineps >= Nil[Ndlc]:
                        rejectnull = 1.
                        epsilon = epsilon - de
                    else:
                        rejectnull = 0.
                        mineps[cfid] = epsilon + de


            aveps[tauc,Ndlc] = np.mean(mineps)
            Ndlc = Ndlc + 1
            
        print(tauc)
        tauc = tauc + 1


    return [aveps, taus]


def embed_pecora(var, t, taus, Nfid, name, nametaus, namedata):

    # first embedding cycle with scalar time series (and save params to file)
    avepsprev = np.zeros(len(taus))
    aveps, taus = embed_scalar(var, t, taus, Nfid)
    saat = np.zeros((len(taus),len(aveps[0,:])+1))
    saat[:,0] = taus
    saat[:,1:] = aveps
    stf2d(saat, './nutde/params-1-%s.txt' % name)

    #aveps = np.min(aveps,axis=1)
    aveps = np.max(aveps,axis=1)

    taumax = determine_max(taus, aveps)
    restaus = np.array([taumax])
    
    end = determine_end(restaus, taus, aveps, avepsprev)

    avepsprev = aveps

    time, var_emb = nonuniform_embedding(var, t, restaus, len(restaus)+1)

    # final embedding cycle(s) with vector time series (and save params to file)
    while end == 0:
        aveps, taus = embed_vec(var_emb, time, taus, Nfid)
        saat = np.zeros((len(taus),len(aveps[0,:])+1))
        saat[:,0] = taus
        saat[:,1:] = aveps
        stf2d(saat, './nutde/params-%s-%s.txt' % (len(restaus)+1, name))

        #aveps = np.min(aveps,axis=1)
        aveps = np.max(aveps,axis=1)
    
        taumax = determine_max(taus, aveps)
        restaus = np.append(restaus, taumax)    

        end = determine_end(restaus, taus, aveps, avepsprev)

        if end != 0:
            restaus = restaus[:-1]
        else: 
            avepsprev = aveps

        time, var_emb = nonuniform_embedding(var, t, restaus, len(restaus)+1)

    # print restaus and final embedded vector to file
    stf1d(restaus, nametaus)
    embt = np.zeros((len(time),len(restaus)+2))
    embt[:,0] = time
    embt[:,1:] = var_emb    
    stf2d(embt, namedata)

    return embt   