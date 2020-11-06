.. PECUZAL automatic embedding for Python documentation master file, created by
   sphinx-quickstart on Wed Nov  4 15:35:18 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PECUZAL automatic embedding for Python's documentation!
==================================================================

Here we introduce the PECUZAL automatic embedding method blabla


.. todo::
   * Insert correct hyperlink to the `pecuzal_embedding`-function in the Julia-Docs
   * Insert correct Licence at the end of the page
   * Insert correct citation link at the end of the page
   * Insert correct installation guide


Getting started
===============

Go and install that stuff by doing...

.. note::
   This implementation is not profiled well. We recommend to use the implementation
   in the `Julia language <https://juliadynamics.github.io/DynamicalSystems.jl/dev/>`_,
   in order to get fast results, especially in the multivariate case. Moreover,
   it is well documented and embedded in the 
   `DynamicalSystems.jl <https://juliadynamics.github.io/DynamicalSystems.jl/dev/>`_ ecosystem.
   For instance, the compuations made in the :ref:`sec_univariate` and the :ref:`sec_multivariate`
   in this documentation took approximately `500s` and `1680s`, respectively. In the Julia implementation
   the exact same computation took `3s` and `20s`, respectively! (running on a 2.8GHz Quad-Core i7,  16GB 1600 MHz DDR3)

.. toctree::
   :maxdepth: 1
   :caption: Usage and examples

   Embedding of a univariate time series <univariate_example>
   Embedding of multivariate time series <multivariate_example>
   Embedding of non-deterministic data <noise_example>


.. toctree::
   :maxdepth: 1
   :caption: Source functions

   pecuzal_embedding

Citing and reference
====================
If you enjoy this tool and find it valuable for your research please cite

.. [kraemer2020] Kraemer et al., "A unified and automated approach to attractor reconstruction", New Journal of Physics, vol. 22, pp. 585-588, 2020.

or as BiBTeX-entry:

::

  @article{kraemer2020,
  doi = {10.21105/joss.00598},
  url = {https://doi.org/10.21105/joss.00598},
  year  = {2020},
  month = {nov},
  volume = {3},
  number = {23},
  pages = {598},
  author = {K. Hauke Kraemer},
  title = {A unified and automated approach to attractor reconstruction},
  journal = {New Journal of Physics}
  }


Licence
=======
This is under blabla Licence


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
