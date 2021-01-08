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

        Y, tau_vals, ts_vals, Ls, _ = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler)

        self.assertTrue(-0.4974 < Ls[0] < -0.4973)
        self.assertTrue(-0.10126 < Ls[1] < -0.10125)
        self.assertTrue(-0.0145 < Ls[2] < -0.01449)
        self.assertEqual(np.size(Y,1),4)
        self.assertEqual(tau_vals[1],21)
        self.assertEqual(tau_vals[2],78)
        self.assertEqual(len(ts_vals),4)


    # Test case for multivariate example
    def test_pecuzal_multivariate_example(self):
        data = np.genfromtxt('./data/lorenz_pecora_multi.csv')
        data = data[:500,:]
        theiler = 15
        Tmax = 100

        Y, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(Tmax), theiler = theiler)

        self.assertTrue(tau_vals[0] == tau_vals[1] == 0)

        self.assertEqual(ts_vals[0], 1)
        self.assertEqual(ts_vals[1], 0)
        self.assertEqual(len(ts_vals),2)

        self.assertTrue(np.sum(Ls[:-1]) < -0.5505736)


if __name__ == '__main__':
    unittest.main()

