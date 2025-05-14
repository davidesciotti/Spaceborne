"""Simple script to run different tests and comparison on two covariance matrices.
"""

import matplotlib.pyplot as plt
import numpy as np
from spaceborne import sb_lib as sl

common_path = '/home/cosmo/davide.sciotti/data/Spaceborne/output'
cov_a = np.load(f'{common_path}/nmt_cov_test/cov_G_3x2pt_2D.npz')['arr_0']
cov_b = np.load(f'{common_path}/sample_cov_test/cov_G_3x2pt_2D.npz')['arr_0']


# # test simmetry
# sl.compare_arrays(
#     cov, cov.T, 'cov', 'cov.T', abs_val=True, log_diff=False, plot_diff_threshold=1
# )

# identity = cov @ cov_inv
# identity_true = np.eye(cov.shape[0])

# tol = 1e-4
# mask = np.abs(identity) < tol
# masked_identity = np.ma.masked_where(mask, identity)
# sl.matshow(
#     masked_identity, abs_val=True, title=f'cov @ cov_inv\n mask below {tol}', log=True
# )

# visual comparison: covariance
sl.compare_arrays(
    cov_a,
    cov_b,
    'cov nmt',
    'cov sample',
    log_array=True,
    abs_val=False,
    log_diff=False,
    plot_diff_threshold=10,
)

# visual comparison: correlation
corr_a = sl.cov2corr(cov_a)
corr_b = sl.cov2corr(cov_b)
sl.plot_correlation_matrix(corr_a)
sl.plot_correlation_matrix(corr_b)

# main diagonal
sl.compare_funcs(
    x=None,
    y={
        'nmt': np.diag(cov_a),
        'sample': np.diag(cov_b),
    },
    logscale_y=(True, False),
    title='diag',
)

# spectrum
sl.compare_funcs(
    x=None,
    y={
        'diag nmt': np.linalg.eigvals(cov_a),
        'diag sample': np.linalg.eigvals(cov_b),
    },
    logscale_y=(True, False),
    title='spectrum',
)
