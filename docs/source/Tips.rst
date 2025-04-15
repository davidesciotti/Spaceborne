Tips
====


+++++
Speed
+++++

``Spaceborne`` offers the possibility to cache the results of the most time-consuming 
operations for later use. In particular, these are the survey covariance 
:math:`\sigma^2_b(z_1, z_2)` and the connected trispectrum 
:math:`T^{ABCD}(k_1, k_2, a)`. When running the code for the first time, set: 

.. code-block:: yaml

   covariance:
      load_cached_sigma2_b: False
      
   PyCCL:
      load_cached_tkka: False 

to avoid running into errors (the files do not exist yet). If you rerun the code 
**with consistent settings**, you can load these in later runs changing the 
configurations above to ``True``. The code will not check the consistency of, 
say, the cosmological parameters you used to compute the cached quantities.

Note that these and other expensive operations are run in parallel, so the code will 
run faster simply by increasing the number of threads used:

.. code-block:: yaml

   misc:
      num_threads: 40
