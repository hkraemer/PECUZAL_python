.. _sec_noise:

Stochastic source example
=========================

If you want to run the following example on your local machine, you are welcome to download the code
`here <https://github.com/hkraemer/PECUZAL_python/blob/docs-config/docs/compute_documentation_examples.py>`_ 
and run it (after having pip-installed the package).

Finally we demonstrate how the PECUZAL method deals with non-deterministic input
data. Therefore, we create a simple AR(1)-process:

.. code-block:: python

    import numpy as np
    import random
    import matplotlib.pyplot as plt

    def ar_process(u0, alpha, p, N):
        '''Generate `N`-sample data from an auto regressive process of order 1 with autocorrelation-parameter 
        `alpha` and amplitude `p` for an intial condition value `u0`.
        '''
        x = np.zeros(N+10)
        x[0] = u0
        for i in range(1,N+10):
            x[i] = alpha*x[i-1] + p*np.random.randn()
        
        return x[10:]

    u0 = .2
    data = ar_process(u0, .9, .2, 2000)

    plt.figure()
    plt.plot(data)
    plt.title('AR(1) process')
    plt.xlabel('sample')
    plt.grid()

.. _fig_ar:

.. image:: ./docsource/images/ar_ts.png


When we now call the PECUZAL algorithm :py:func:`pecuzal_embedding.pecuzal_embedding`

.. code-block:: python

    from pecuzal_embedding import *

    Y_reconstruct, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(data)

we'll get the following note in the console:

::

    Algorithm stopped due to increasing L-values. Valid embedding NOT achieved.


The algorithm did not obtain any valid embedding, thus, it values the input data as a non-deterministic
source.