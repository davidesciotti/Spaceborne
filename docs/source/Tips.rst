Tips
====


+++++
Speed
+++++

``Spaceborne`` offers the possibility to cache the result of the most time-consuming 
operation -- the computation of the cNG connected trispectrum 
:math:`T^{ABCD}(k_1, k_2, a)` -- for later use. When running the code for the first time, set: 

.. code-block:: yaml
      
   PyCCL:
      load_cached_tkka: False 

To avoid running into errors (since the file does not exist yet). If you rerun the code 
**with consistent settings**, you can load this in later runs by simply changing the 
above flag to ``True``. 

.. warning:: 
   The code will **not** check the consistency of the configuration used to generate 
   the cached files when loading them for subsequent runs. If, for example, 
   run 2 loads the cached trispectrum from round 1 but has a different halo model,
   the code will produce inconsistent results.

Note: these and other expensive operations are run in parallel, so the code will 
run faster simply by increasing the number of threads used:

.. code-block:: yaml

   misc:
      num_threads: 40
