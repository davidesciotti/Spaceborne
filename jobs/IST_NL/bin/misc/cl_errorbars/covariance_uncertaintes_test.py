import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

"""
script to test whether the presence of non-diagonal elements in a covariance matrix affects the uncertainties.

- The uncertainties computed as the variance of the samples - i.e. the marginal uncertainties - are equal to 
  (within sampling noise) the ones computed as the square root of the diagonal elements of the covariance. 
  This is true both for the diagonal and non-diagonal covariance, as should be.
- The conditional uncertainties differ in the non-diagonal case, and are equal to the marginal uncertainties 
  in the diagonal case
  
Bottom line: the presence of non-diagonal elements in the covariance matrix only affects the marginal uncertainties, 
"""

cov_diag = np.asarray(((1.25, 0), (0, 1.25))).reshape(2, 2)
cov_nondiag = np.asarray(((1.25, 0.99), (0.99, 1.25))).reshape(2, 2)
# cov_nondiag = np.ones((2, 2)) # this one is singular, the determinant is 0

print(cov_diag)
print(cov_nondiag)

datavector = (0, 0)
n_samples = 10_000

rng = np.random.default_rng()
samples_diag = rng.multivariate_normal(mean=datavector, cov=cov_diag, size=n_samples)
samples_nondiag = rng.multivariate_normal(mean=datavector, cov=cov_nondiag, size=n_samples)

# plot the samples of 2 components of the datavector (say, x and y)
plt.scatter(samples_diag[:, 0], samples_diag[:, 1], alpha=0.6, label='diagonal cov')
plt.scatter(samples_nondiag[:, 0], samples_nondiag[:, 1], alpha=0.6, label='non-diagonal cov')
plt.grid()
plt.legend()


# marginal uncertainties - same for diag and non-diag
marginal_diag = np.sqrt(np.diag(cov_diag))
marginal_nondiag = np.sqrt(np.diag(cov_nondiag))
print(f'marginal_diag =      {marginal_diag}')
print(f'marginal_nondiag =   {marginal_nondiag}')

# variance of the samples - the same as the marginal uncertainty
marginal_diag_sampled = np.sqrt(np.var(samples_diag, axis=0))
marginal_nondiag_sampled = np.sqrt(np.var(samples_nondiag, axis=0))
print(f'marginal_diag_sampled =      {marginal_diag_sampled}')
print(f'marginal_nondiag_sampled =   {marginal_nondiag_sampled}')

# marginal uncertainties - different for diag and non-diag
conditional_diag = np.sqrt(1/(np.diag(np.linalg.inv(cov_diag))))
conditional_nondiag = np.sqrt(1/(np.diag(np.linalg.inv(cov_nondiag))))  # different!!
print(f'conditional_diag =     {conditional_diag}')
print(f'conditional_nondiag =  {conditional_nondiag}')
