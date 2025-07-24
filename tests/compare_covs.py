import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

from spaceborne import sb_lib as sl


# cov_a = np.load(
#     '/u/dsciotti/code/common_data/RR2/Jose/output/cov_check_SB_run/cov_G_3x2pt_2D.npz'
# )['arr_0']
# cov_b = np.load(
#     '/u/dsciotti/code/common_data/RR2/Jose/output/cov_check_OC_run/cov_G_3x2pt_2D.npz'
# )['arr_0']


ROOT = '/u/dsciotti/code/common_data/RR2/Jose/output/cov_check_from_jose'

cov_a = np.load(f'{ROOT}/cov_TOT_3x2pt_2D.npz')['arr_0']
cov_b = fits.open(f'{ROOT}/sim_cells_forCLOE_1507.fits')
cov_b = cov_b['COVMAT'].data

name_a = 'SB'
name_b = 'CS'

sl.compare_arrays(cov_a, cov_b, name_a, name_b, log_diff=True, plot_diff_threshold=10)

sl.compare_funcs(
    x=None,
    y={name_a: np.diag(cov_a), name_b: np.diag(cov_b)},
    logscale_y=[True, False],
    title='cov diag',
)

sl.compare_funcs(
    x=None,
    y={name_a: cov_a.flatten(), name_b: cov_b.flatten()},
    logscale_y=[True, False],
    title='cov flat',
)

eig_a, _ = np.linalg.eig(cov_a)
eig_b, _ = np.linalg.eig(cov_b)
sl.compare_funcs(
    x=None,
    y={name_a: eig_a, name_b: eig_b},
    logscale_y=[True, False],
    title='cov eig',
)
