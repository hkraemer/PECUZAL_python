.. _sec_multivariate:

Multivariate example
====================

Similar to the approach in the :ref:`sec_univariate`, we now highlight the capability of the
proposed embedding method for a multivariate input. Again, we load all three time series of the
Roessler system, which are stored in `roessler_test_series.csv`, and restrict ourselves to the
first 5,000 samples, in order to save computation time.

.. code-block:: python
   
    import numpy as np
    data = np.genfromtxt('roessler_test_series.csv')
    data = data[:5000,:]

The idea is now to feed in all three time series to the algorithm, even though this is a very
far-from-reality example. We already have an adequate representation of the system we want to
reconstruct, namely the three time series from the numerical integration. But let us see what
PECUZAL suggests for a reconstruction.

Since we have to deal with three time series now, let us estimate the Theiler window as the
maximum of all Theiler windows of each time series. Again, we estimate such a Theiler window
by taking the first minimum of the auto mutual information.

.. code-block:: python
   
   import matplotlib.pyplot as plt
   from pecuzal_embedding import *

   N = len(data)
   mis = np.empty(shape=(50,3))
   for i in range(3):
      mis[:,i], lags = mi(data[:,i])    # compute mutual information up to default maximum time lag

   plt.figure(figsize=(14., 8,))

   ts_str = ['x','y','z']

   cnt = 0
   for i in range(0,6,2):
      plt.subplot(3,2,i+1)
      plt.plot(range(N),data[:,cnt])
      plt.grid()
      if i == 4:
         plt.xlabel('time [in sampling units]')
      plt.title(ts_str[cnt]+'-component of Roessler test time series')

      plt.subplot(3,2,i+2)
      plt.plot(lags,mis[:,cnt])
      plt.grid()
      plt.ylabel('MI')
      if i == 4:
         plt.xlabel('time lag [in sampling units]')
      plt.title('Mutual information for '+ts_str[cnt]+'-component of Roessler test time series')
      cnt +=1
   plt.subplots_adjust(hspace=.3)

.. _fig_mi_multi:

.. image:: ./docsource/images/mi_and_timeseries_multi.png

Due to the spikyness of the `z`-component the according auto mutual information yields `nan`-values as
a result of empty bins in the histograms. So we stick to the choice of `theiler = 30`, here and 
call the PECUZAL algorithm :py:func:`pecuzal_embedding.pecuzal_embedding` with default `kwargs` 
and possible delays ranging from `0:100`.

.. code-block:: python

   Y_reconstruct, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data, taus = range(100), theiler = 30)

which leads to the following note in the console (see also the note on :ref:`performance <note_performance>`):

::

   Algorithm stopped due to minimum L-value reached. VALID embedding achieved.


The suggested embedding parameters...

::

   tau_vals = [0, 3, 11]
   ts_vals = [0, 1, 0]

... reveal that PECUZAL builds the reconstructed trajectory `Y_reconstruct` from the unlagged time series, having
index `0`, i.e. the `x`-component, the `y`-component lagged by 3 samples, and finally again the `x`-component lagged
by 11 samples. As expected the total `L`-value is smaller here than in the :ref:`univariate case <l_uni>`:

.. code-block:: python

   L_total = np.amin(Ls)

   -3.419176812657791


The reconstructed attractor looks also quite similar to the original one, even though that is not a proper evaluation
criterion for the goodness of a reconstruction, see [kraemer2020]_.

.. code-block:: python
   
   from mpl_toolkits import mplot3d
   
   ts_labels = ['x','y','z']

   fig = plt.figure(figsize=(14., 8.))
   ax = plt.subplot(121, projection='3d')
   ax.plot(Y_reconstruct[:,0], Y_reconstruct[:,1], Y_reconstruct[:,2], 'gray')
   ax.grid()
   ax.set_xlabel('{}(t+{})'.format(ts_labels[ts_vals[0]],tau_vals[0]))
   ax.set_ylabel('{}(t+{})'.format(ts_labels[ts_vals[1]],tau_vals[1]))
   ax.set_zlabel('{}(t+{})'.format(ts_labels[ts_vals[2]],tau_vals[2]))
   ax.set_title('PECUZAL reconstructed Roessler system (multivariate)')
   ax.view_init(38, -25)

   ax = plt.subplot(122, projection='3d')
   ax.plot(data[:5000,0], data[:5000,1], data[:5000,2], 'gray')
   ax.grid()
   ax.set_xlabel('x(t)')
   ax.set_ylabel('y(t)')
   ax.set_zlabel('z(t)')
   ax.set_title('Original Roessler system')


.. _fig_rec_multi:

.. image:: ./docsource/images/reconstruction_multi.png

