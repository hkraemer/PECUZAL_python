""" Functions for project "false positives in wRNA" """

import numpy as np
import diffEmb
import mviAAFT
from pyunicorn.timeseries import RecurrenceNetwork
from pyunicorn.timeseries import Surrogates

def tde(time,var,dim,tau):

    #time delay embedding
    N = len(time)
    Nemb = N - (dim-1)*tau
    x_tde = np.zeros((Nemb, dim))

    for dimcounter in range(dim):
        x_tde[:, dimcounter] = var[dimcounter*tau:N-(dim-1-dimcounter)*tau]

    return x_tde


def dLp(time,var,dim,p):

    #differential embedding - discrete Legendre polynomials
    glc = diffEmb.legendre_coordinates(var, dim=dim, t=time, p=p)

    return glc


def scale(emb_ts):
    
    #scale all coordinates of (embedded) time series to unit variance
    N = len(emb_ts[:,0])
    dim = len(emb_ts[0,:])

    scaled_ts = np.zeros((N,dim))

    for dimcounter in range(dim):
        scaled_ts[:,dimcounter] = emb_ts[:,dimcounter]/np.std(emb_ts[:,dimcounter])

    return scaled_ts


def wrna(emb_ts, W, dW, RR):

    #windowed recurrence network analysis
    N = len(emb_ts[:,0])
    Nwind = int(((N-W)/dW))

    T = np.zeros(Nwind)
    L = np.zeros(Nwind)

    for mu in range(Nwind):
        rn = RecurrenceNetwork(emb_ts[mu*dW:mu*dW+W,:], recurrence_rate=RR, silence_level=2)
        T[mu] = rn.transitivity()
        L[mu] = rn.average_path_length()

    return [T, L]


def wrna_surr(emb_ts, W, RR, Nsurr):

    #surrogates from embedded time series to get RNA significance bounds
    N = len(emb_ts[:,0])

    T = np.zeros(Nsurr)
    L = np.zeros(Nsurr)

    for surr in range(Nsurr):
        indices = np.arange(N)
        np.random.shuffle(indices)
        rand = indices[:W]

        rn = RecurrenceNetwork(emb_ts[rand,:], recurrence_rate=RR, silence_level=2)

        T[surr] = rn.transitivity()
        L[surr] = rn.average_path_length()

    return [T, L]


def wrna_mv_iAAFTsurr(emb_ts, W, RR, Nsurr, n_iterations):

    #surrogates from embedded time series to get RNA significance bounds
    N = len(emb_ts[:,0])

    T = np.zeros(Nsurr)
    L = np.zeros(Nsurr)

    for surr in range(Nsurr):

        surrogates = mviAAFT.mv_iAAFT_surrogates(emb_ts, n_iterations, output="true_amplitudes")

        rn = RecurrenceNetwork(surrogates[:W,:], recurrence_rate=RR, silence_level=2)

        T[surr] = rn.transitivity()
        L[surr] = rn.average_path_length()

    return [T, L]


def wrna_eval(res, surr, cb1, cb2):

    #calculate fraction of significant points
    Nw = len(res)

    sig1 = np.percentile(surr,cb1)
    sig2 = np.percentile(surr,cb2)

    surrbin = np.zeros(Nw)

    for counter in range(Nw):
        if res[counter] < sig1 or res[counter] > sig2:
            surrbin[counter] = 1

    return np.sum(surrbin)/Nw


def iAAFT(var, Niter):

    #create iAAFT surrogate data from (univariate) time series
    origData = np.array([var])
    surr = Surrogates(origData,silence_level=2)
    surr.clear_cache()
    surro = surr.refined_AAFT_surrogates(origData, Niter)
    surrogates = surro[0,:]

    return surrogates



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




