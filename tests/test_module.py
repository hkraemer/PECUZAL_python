import sys 
import os
import unittest
import numpy as np
from pecuzal_embedding import *


class TestModule(unittest.TestCase):

    # Test case for univariate example
    def test_pecuzal_univariate_example(self):
        data = np.genfromtxt('lorenz_pecora_uni_x.csv')
        data = data[:500]
        theiler = 21
        Tmax = 100
        K = 14
        KNN = 3

        Y, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler, sample_size = 1., K = K, KNN = KNN)

        self.assertTrue(-1.68 < Ls[0] < -1.67)
        self.assertTrue(-1.72 < Ls[1] < -1.71)
        self.assertTrue(-1.71 < Ls[2] < -1.70)

        self.assertEqual(tau_vals[1],58)
        self.assertEqual(tau_vals[2],12)

        self.assertTrue(len(ts_vals) == 3)


    # Test case for multivariate example
    def test_pecuzal_multivariate_example(self):
        data = np.genfromtxt('lorenz_pecora_multi.csv')
        data = data[:500,:]
        theiler = 15
        Tmax = 100

        Y, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler)

        self.assertTrue(tau_vals[0] == tau_vals[1] == 0)

        self.assertEqual(ts_vals[0], 1)
        self.assertEqual(ts_vals[1], 2)
        self.assertEqual(len(ts_vals),2)

        self.assertTrue(-1.69 < Ls[0] < -1.68)
        self.assertTrue(-1.68 < Ls[1] < -1.67)


if __name__ == '__main__':
    unittest.main()



data = np.genfromtxt('lorenz_pecora_uni_x.csv')
data = data[:500]
theiler = 21
Tmax = 100
K = 14
KNN = 3

Y, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler, sample_size = 1., K = K, KNN = KNN)
