import numpy as np
import matplotlib.pyplot as pyplot


# load results
aveps1 = np.loadtxt('./nutde/test10/params-1-lorenz.txt')
#aveps2 = np.loadtxt('./nutde/params-2-lorenz.txt')
#aveps3 = np.loadtxt('./nutde/params-3-lorenz.txt')
#aveps4 = np.loadtxt('./nutde/params-4-lorenz.txt')
#aveps5 = np.loadtxt('./nutde/params-5-lorenz.txt')
#aveps6 = np.loadtxt('./nutde/params-6-lorenz.txt')

taus1 = aveps1[:,0]
#taus2 = aveps2[:,0]
#taus3 = aveps3[:,0]
#taus4 = aveps4[:,0]
#taus5 = aveps5[:,0]
#taus6 = aveps6[:,0]


pyplot.figure(1)
pyplot.plot(taus1, aveps1[:,1:], color = 'darkblue', label='[0]')
#pyplot.plot(taus2, aveps2, color = 'darkorange', label='[0,18]')
#pyplot.plot(taus3, aveps3, color = 'yellow', label='[0,18,46]')
#pyplot.plot(taus4, aveps4, color = 'darkviolet', label='[0,18,46,64]')
#pyplot.plot(taus5, aveps5, color = 'darkcyan', label='[0,]')
#pyplot.plot(taus6, aveps6, color = 'darkgreen', label='[0,]')
pyplot.xlabel('delays')
pyplot.ylabel('average epsilon')
pyplot.legend()
pyplot.show()
