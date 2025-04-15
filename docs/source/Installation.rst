Installation
============

==========
Quickstart
==========

We recommend using ``Spaceborne`` in a dedicated ``conda`` environment. 
This ensures all dependencies are properly managed. Start by cloning the 
`repository <https://github.com/davidesciotti/Spaceborne>`_,
then run

.. code-block:: bash
   
   $ conda env create -f environment.yaml
   $ conda activate spaceborne
   $ pip install .


``Spaceborne`` leverages the `julia language <https://julialang.org/>`_ for 
computationally intensive tasks. 
We recommend installing ``Julia`` via `juliaup <https://github.com/JuliaLang/juliaup>`_:

.. code-block:: bash

   $ curl -fsSL https://install.julialang.org | sh  # Install juliaup
   $ juliaup default 1.10                           # Install Julia version 1.10

Then, install the required ``Julia`` packages:

..  code-block:: bash

   $ julia -e 'using Pkg; Pkg.add("LoopVectorization"); Pkg.add("YAML"); Pkg.add("NPZ")'

and you are ready to go!

Notes
_____

* Using ``mamba`` instead of ``conda`` in the first line will *significantly* 
  speed up the environment creation. To install ``mamba``, 
  run ``conda install mamba`` in your ``base`` environment.
* The environment is designed to already include all the packages 
  required by the ``Spaceborne`` dependencies (see below), so this should be the only 
  step needed. If you decide not to use a given package,
  you can remove its dependencies from the ``environment.yaml`` file (comments are provided).
  If, on the other hand, you encounter installation issues with a particular package 
  (both ``CCL`` and ``NaMaster`` have a C engine, which may require some additional 
  libraries), please refer to the official documentation of the package in question. 
  As an example, some installation issues with ``CCL`` can be solved by running

   .. code-block:: bash

      $ sudo apt-get install gfortran cmake build-essential autoconf bison

   on Linux and 

   .. code-block:: bash

      $ brew install gfortran cmake build-essential autoconf bison

   on OSX. 

* Finally, please note that these instructions have only been tested for Linux and OSX 
  systems.


=================
Main dependencies
=================

``Spaceborne`` leverages several public packages for some of its functionalities.
As much as possible, the dependencies are isolated and the relevant packages are only
imported when needed. A notable exception is ``CCL``, which is always imported and 
therefore needs to be installed for the code to run.


* ``CCL``. The code relies heavily on the ``CCL`` library for the computation of the background 
  evolution, radial kernels, angular power spectra, halo model, and connected trispectrum.
  This is the main dependency of the code, and it is mandatory to run it.
* ``CLOE``. [WORK IN PROGRESS]. Alternatively, the code can use the ``CLOE`` library as 
  a backend for the calculation of the background evolution, radial kernels and 
  angular power spectra. Note that also in this case, the halo mode quantities and 
  connected trispectra will still be computed with ``CCL``.
* ``pylevin`` [`GitHub <https://github.com/rreischke/levin_bessel>`_]. This package 
  is used to  perform the integrals involving Bessel functions, namely the survey 
  covariance :math:`\sigma_b^2(z_1, z_2)` and the real-space covariance, using the Levin 
  method (see `2502.12142 <https://arxiv.org/abs/2502.12142>`_) 
  The use of this package is optional, and the code will fall back to a simpson 
  integration if ``pylevin`` is not installed.
* ``NaMaster`` [`GitHub <https://github.com/LSSTDESC/NaMaster>`_]. This package is used 
  to compute the partial-sky Gaussian covariance in the NKA or iNKA approximation, using
  the pseudo-:math:`C_\ell` framework (see `1809.09603 <https://arxiv.org/abs/1809.09603>`_).






