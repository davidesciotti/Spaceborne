.. Spaceborne documentation master file, created by
   sphinx-quickstart on Tue Apr  8 15:57:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/sb_logo.png
   :width: 30%
   :align: center

Welcome to Spaceborne Documentation!
====================================

Spaceborne is a Python package for the computation of the covariance matrix of
photometric observables (weak lensing, photometric galaxy clustering and their 
cross-correlation). It can produce the covariance matrix both for harmonic- and 
real-space two-point statistics, and it allows the inclusion of the non-Gaussian
contributions (super-sample and connected non-Gaussian covariance) to the covariance matrix.
The code includes a self-explanatory configuration file; further details can be found in 
this documentation, as well as in the accompanying paper [Sciotti et al., in prep].

Features
--------

* Calculation of covariance matrices for various cosmological probes
* Support for different ordering schemes in the output
* Flexible configuration through ``YAML`` files
* Integration with Julia for computational efficiency

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   Installation
   Usage
   IO
   Tips