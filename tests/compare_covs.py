import numpy as np
import matplotlib.pyplot as plt

from spaceborne import sb_lib as sl


cov_a = np.load(
    '/u/dsciotti/code/common_data/RR2/Jose/output/cov_check_SB_run/cov_G_3x2pt_2D.npz'
)['arr_0']
cov_b = np.load(
    '/u/dsciotti/code/common_data/RR2/Jose/output/cov_check_OC_run/cov_G_3x2pt_2D.npz'
)['arr_0']

name_a = 'SB'
name_b = 'OC'


sl.compare_arrays(cov_a, cov_b, name_a, name_b, log_diff=True, plot_diff_threshold=10)

sl.compare_funcs(
    x=None,
    y={name_a: np.diag(cov_a), name_b: np.diag(cov_b)},
    logscale_y=[True, False],
    title='cov diag',
)

eig_sb, _ = np.linalg.eig(cov_a)
eig_oc, _ = np.linalg.eig(cov_b)
sl.compare_funcs(
    x=None,
    y={name_a: eig_sb, name_b: eig_oc},
    logscale_y=[True, False],
    title='cov eig',
)
