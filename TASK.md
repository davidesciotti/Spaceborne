I want to update the namaster Gaussian covariance computation according to their new catalog-based approach. This is not live at the moment, but it's almost finished in the catalog_based_covariance branch: https://github.com/LSSTDESC/NaMaster/blob/catalog_based_covariances/

I also found an example in https://github.com/LSSTDESC/NaMaster/blob/catalog_based_covariances/doc/4Catalogs.ipynb

I want to implement this method in Spaceborne. The natural place is the cov_partial_sky.py module. Let's start by planning the implementation. Use Sonnet 5 subagents when possible to save tokens.