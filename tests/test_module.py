import sys 
import os
import unittest
import numpy as np
from pecuzal_embedding import *


class TestModule(unittest.TestCase):

    # Test case for univariate example
    def test_pecuzal_univariate_example(self):
        data = np.genfromtxt('./data/lorenz_pecora_uni_x.csv')
        data = data[:500]
        theiler = 21
        Tmax = 100

        Y, tau_vals, ts_vals, Ls, _ = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler, econ = True)

        self.assertTrue(-0.5 < Ls[0] < -0.496)
        self.assertTrue(-0.012 < Ls[2] < -0.011)
        self.assertTrue(-0.6078 < np.sum(Ls) < -0.6077)
        self.assertEqual(np.size(Y,1),4)
        self.assertEqual(tau_vals[1],21)
        self.assertEqual(tau_vals[2],13)
        self.assertEqual(tau_vals[3],78)

        Y, tau_vals, ts_vals, Ls, _ = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler, L_threshold = 0.05, econ = True)

        self.assertTrue(-0.5 < Ls[0] < -0.496)
        self.assertEqual(np.size(Y,1),3)
        self.assertEqual(tau_vals[1],21)


    # Test case for multivariate example
    def test_pecuzal_multivariate_example(self):
        data = np.genfromtxt('./data/lorenz_pecora_multi.csv')
        data = data[:500,:2]
        theiler = 15
        Tmax = 100

        Y, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler, econ = True)

        self.assertTrue(tau_vals[0] == tau_vals[1] == 0)

        self.assertEqual(ts_vals[0], 1)
        self.assertEqual(ts_vals[1], 0)
        self.assertEqual(len(ts_vals),2)

        self.assertTrue(np.sum(Ls) < -0.544942)


if __name__ == '__main__':
    unittest.main()

