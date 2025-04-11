"""Simple script to run different tests and comparison on two covariance matrices.
11/04/2025: As of now, I just dumped some code not to lose it, no real development yet.
"""

import numpy as np
import matplotlib.pyplot as plt
from spaceborne import sb_lib as sl



cov = (
    cov_dict['cov_3x2pt_g_2D']
    + cov_dict['cov_3x2pt_ssc_2D']
    + cov_dict['cov_3x2pt_cng_2D']
)
cov_inv = np.linalg.inv(cov)

# test simmetry
sl.compare_arrays(
    cov, cov.T, 'cov', 'cov.T', abs_val=True, log_diff=False, plot_diff_threshold=1
)

identity = cov @ cov_inv
identity_true = np.eye(cov.shape[0])

tol = 1e-4
mask = np.abs(identity) < tol
masked_identity = np.ma.masked_where(mask, identity)
sl.matshow(
    masked_identity, abs_val=True, title=f'cov @ cov_inv\n mask below {tol}', log=True
)

sl.compare_arrays(
    cov @ cov_inv,
    cov_inv @ cov,
    'cov @ cov_inv',
    'cov_inv @ cov',
    abs_val=True,
    log_diff=True,
    plot_diff_threshold=1,
)

plt.semilogy(np.linalg.eigvals(cov))
