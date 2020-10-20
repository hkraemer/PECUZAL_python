import numpy as np
from scipy.integrate import odeint


def lorenz_system(vec,t):
    sigma, rho, beta = [10.,60.,8./3.]
    x, y, z = vec
    return[sigma*(y-x), x*(rho-z)-y, x*y-beta*z]


def lorenz(ini, t):
    ts = odeint(lorenz_system, ini, t)
    xx, yy, zz = ts.T
    return xx







