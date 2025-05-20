"""Simple script to run different tests and comparison on two covariance matrices."""

import matplotlib.pyplot as plt
import numpy as np
from spaceborne import sb_lib as sl


common_path = '/home/cosmo/davide.sciotti/data/common_data/cov_sb_for_marco/output'
name_a = 'iNKATrue_coupledTrue_nmtCov'
name_b = 'iNKATrue_coupledTrue_sampleCov'
cov_a = np.load(f'{common_path}/{name_a}/cov_G_3x2pt_2D.npz')['arr_0']
cov_b = np.load(f'{common_path}/{name_b}/cov_G_3x2pt_2D.npz')['arr_0']


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

# visual comparison: correlation
corr_a = sl.cov2corr(cov_a)
corr_b = sl.cov2corr(cov_b)
sl.plot_correlation_matrix(corr_a)
sl.plot_correlation_matrix(corr_b)

# visual comparison: covariance
sl.compare_arrays(
    cov_a,
    cov_b,
    name_a,
    name_b,
    log_array=True,
    abs_val=False,
    log_diff=False,
    plot_diff_threshold=10,
)


# main diagonal
sl.compare_funcs(
    x=None,
    y={
        name_a: np.diag(cov_a),
        name_b: np.diag(cov_b),
    },
    logscale_y=(True, False),
    title='diag',
)

# spectrum
sl.compare_funcs(
    x=None,
    y={
        f'{name_a}': np.linalg.eigvals(cov_a),
        f'{name_b}': np.linalg.eigvals(cov_b),
    },
    logscale_y=(True, False),
    title='spectrum',
)
