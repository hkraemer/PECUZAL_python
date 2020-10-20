import numpy as np
import matplotlib.pyplot as pyplot
import inpts
import psrpec

# Lorenz system x-component time series (and save to file)
ini = [10.0,10.0,10.0]
t = np.arange(0., 1280.0, 0.01)
var = inpts.lorenz(ini, t)
N = len(var)

f = open('./data/data-lorenz-hr.txt', 'w')
for counter in range(N): 
    f.write(str(t[counter]))
    f.write("\t")
    f.write(str(var[counter]))
    f.write("\n")
f.close()

pyplot.figure(1)
pyplot.plot(t,var)
pyplot.figure(2)
pyplot.plot(t[:5000],var[:5000])
pyplot.show()


"""
# embedding cycles as proposed by Pecora et al. 2007
Nfid = 500
Ndelta = 13
taus = np.arange(1,401,dtype=int)

# normalize time series
var = var[:5000]
t = t[:5000]

var = var - np.mean(var)
var = var/np.std(var,ddof=1)

print np.mean(var)
print np.std(var,ddof=1)

embt = psrpec.embed_pecora(var, t, taus, Nfid, Ndelta, 'lorenz', './nutde/restaus-lorenz.txt', './nutde/psr-lorenz.txt')

print embt
"""
