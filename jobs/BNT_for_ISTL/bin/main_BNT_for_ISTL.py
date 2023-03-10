import bz2
import gc
import pickle
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import warnings

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

# general libraries
sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

# general configurations
sys.path.append(f'{project_path.parent}/common_data/common_config')
import mpl_cfg
import ISTF_fid_params as ISTFfid

# project modules
sys.path.append(f'{project_path}/bin')
import ell_values as ell_utils
import cl_preprocessing as cl_utils
import compute_Sijkl as Sijkl_utils
import covariance as covmat_utils
import fisher_matrix as FM_utils

# job configuration and modules
sys.path.append(f'{project_path}/jobs')
import BNT_for_ISTL.config.config_BNT_for_ISTL as cfg

mpl.use('Qt5Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()


# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks
# TODO super check that things work with different # of z bins


def check_against_cl_15gen_func():
    if not check_against_cl_15gen:
        return

    cl_15gen_path = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/Cij-NonLin-eNLA_15gen'
    cl_LL_2D_vinc = np.genfromtxt(f'{cl_15gen_path}/CijLL-LCDM-NonLin-eNLA.dat')
    cl_LG_2D_vinc = np.genfromtxt(f'{cl_15gen_path}/CijLG-LCDM-NonLin-eNLA.dat')
    cl_GG_2D_vinc = np.genfromtxt(f'{cl_15gen_path}/CijGG-LCDM-NonLin-eNLA.dat')

    ell_vinc = cl_LL_2D_vinc[:, 0]
    cl_LL_2D_vinc = cl_LL_2D_vinc[:, 1:]
    cl_LG_2D_vinc = cl_LG_2D_vinc[:, 1:]
    cl_GG_2D_vinc = cl_GG_2D_vinc[:, 1:]

    cl_LL_3D_vinc = mm.cl_2D_to_3D_symmetric(cl_LL_2D_vinc, len(ell_vinc), zpairs_auto, zbins)
    cl_GL_3D_vinc = mm.cl_2D_to_3D_asymmetric(cl_LG_2D_vinc, len(ell_vinc), zbins, order='C').transpose(0, 2, 1)
    cl_GG_3D_vinc = mm.cl_2D_to_3D_symmetric(cl_GG_2D_vinc, len(ell_vinc), zpairs_auto, zbins)

    plt.figure()
    for zi in range(zbins - 1):
        plt.plot(ell_vinc, np.abs(cl_GL_3D_vinc[:, zi, zi + 1]), c=colors[zi])
        plt.plot(ells, np.abs(cl_GL_3d_BNT_DEMO[:, zi, zi + 1]), c=colors[zi], ls='--')


def compare_diagonal_cls(cl_3d_A, cl_3d_B):
    plt.figure()
    for zi in range(zbins):
        plt.plot(ells, np.abs(cl_3d_A)[:, zi, zi], label=f'({zi}, {zi})', c=colors[zi], alpha=0.6)
        plt.plot(ells, np.abs(cl_3d_B)[:, zi, zi], c=colors[zi], ls='--')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg
Sijkl_cfg = cfg.Sijkl_cfg
FM_cfg = cfg.FM_cfg

# utils.consistency_checks(general_cfg, covariance_cfg)

# some variables used for I/O naming, just to make things more readable
zbins = general_cfg['zbins']
EP_or_ED = general_cfg['EP_or_ED']
triu_tril = covariance_cfg['triu_tril']
row_col_major = covariance_cfg['row_col_major']

# ! other options
bnt_transform = False
check_against_cl_15gen = False
plot_BNT_vs_nonBNT_cls = False  # this is a "successful" check: the plot looks the same as in the BNT-demo
# ! end other options

# some checks
assert EP_or_ED == 'EP' and zbins == 10, 'ISTF uses 10 equipopulated bins'
assert covariance_cfg['GL_or_LG'] == 'GL', 'Cij_14may uses GL, also for the probe responses'

zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

# build ind files and store it into the covariance dictionary
ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
covariance_cfg['ind'] = ind

cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, zbins)]

from_DEMO_870_path = f'{job_path}/data/from_BNT_DEMO_870'

# ! get the cls from the 870 BNT DEMO; these are already in 3D
cl_LL_3d = np.load(f'{from_DEMO_870_path}/cC_LL_arr.npy')
cl_GL_3d = np.load(f'{from_DEMO_870_path}/cC_GL_arr.npy').transpose(0, 2, 1)
cl_GG_3d = np.load(f'{from_DEMO_870_path}/cC_GG_arr.npy')

# note that this name is quite verbose, the BNT-ized cls must come from the BNT demo!
cl_LL_BNT_3d = np.load(f'{from_DEMO_870_path}/cC_LL_BNT.npy')
cl_GL_BNT_3d = np.load(f'{from_DEMO_870_path}/cC_GL_BNT.npy').transpose(0, 2, 1)
cl_GG_BNT_3d = np.load(f'{from_DEMO_870_path}/cC_GG_BNT.npy')

# ! check GL against the cls from Vincenzo (very bad agreement...)
check_against_cl_15gen_func()

ells = np.load(f'{from_DEMO_870_path}/ell_values.npy')
delta_ells = np.load(f'{from_DEMO_870_path}/delta_ells.npy')
nbl = len(ells)

# ! build BNT covariance by transforming the Gaussian one
bnt_matrix = np.genfromtxt(f'{from_DEMO_870_path}/BNT_matrix.txt')

cl_LL_BNTdark_3d = cl_utils.cl_BNT_transform(cl_LL_3d, bnt_matrix, 'L', 'L')
cl_GL_BNTdark_3d = cl_utils.cl_BNT_transform(cl_GL_3d, bnt_matrix, 'G', 'L')
cl_GG_BNTdark_3d = cl_utils.cl_BNT_transform(cl_GG_3d, bnt_matrix, 'G', 'G')

np.testing.assert_allclose(cl_LL_BNTdark_3d, cl_LL_BNT_3d, atol=0, rtol=1e-10)
np.testing.assert_allclose(cl_GL_BNTdark_3d, cl_GL_BNT_3d, atol=0, rtol=1e-10)
np.testing.assert_allclose(cl_GG_BNTdark_3d, cl_GG_BNT_3d, atol=0, rtol=1e-10)

cl_3x2pt_5D = cl_utils.build_3x2pt_datavector_5D(cl_LL_3d, cl_GL_3d, cl_GG_3d, nbl, zbins)
cl_3x2pt_BNT_5D = cl_utils.build_3x2pt_datavector_5D(cl_LL_BNT_3d, cl_GL_BNT_3d, cl_GG_BNT_3d, nbl, zbins)

ell_dict = {
    'ell_WL': ells,
    'ell_GC': ells,
    'ell_WA': ells,
}

delta_dict = {
    'delta_l_WL': delta_ells,
    'delta_l_GC': delta_ells,
    'delta_l_WA': delta_ells,
}

cl_dict_3D = {
    'cl_LL_3D': cl_LL_3d,
    'cl_GL_3D': cl_GL_3d,
    'cl_GG_3D': cl_GG_3d,
    'cl_WA_3D': cl_LL_3d,  # ! not used
    'cl_3x2pt_5D': cl_3x2pt_5D,
}

cl_dict_BNT_3D = {
    'cl_LL_3D': cl_LL_BNT_3d,
    'cl_GL_3D': cl_GL_BNT_3d,
    'cl_GG_3D': cl_GG_BNT_3d,
    'cl_WA_3D': cl_LL_BNT_3d,  # ! not used
    'cl_3x2pt_5D': cl_3x2pt_BNT_5D,
}

rl_dict_3D = {}

# ! compute covariance matrix
# dummy sijkl matrix, I am not computing covSSC...
sijkl = np.random.rand(2 * zbins, 2 * zbins, 2 * zbins, 2 * zbins)

# ! compute covariance matrix with and without BNT
covariance_cfg['cov_BNT_transform'] = True
cov_dict_BNT_True = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                             ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl, BNT_matrix=bnt_matrix)

covariance_cfg['cov_BNT_transform'] = False
cov_dict_BNT_False = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                              ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl)

# this is the BNT-covariance computed with the BNT-transformed cls:
covariance_cfg['cov_BNT_transform'] = False
cov_dict_BNT_True_with_cls = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                                      ell_dict, delta_dict, cl_dict_BNT_3D, rl_dict_3D, sijkl)

cov_3x2pt_GO_BNT_True = cov_dict_BNT_True['cov_3x2pt_GO_2DCLOE']
cov_3x2pt_GO_BNT_False = cov_dict_BNT_False['cov_3x2pt_GO_2DCLOE']
cov_3x2pt_GO_BNT_True_with_cls = cov_dict_BNT_True_with_cls['cov_3x2pt_GO_2DCLOE']

warnings.warn('you have to be in branch #870 for this import to work')
cov_3x2pt_GO_BNT_False_benchmark = np.load(
    '/Users/davide/Documents/Lavoro/Programmi/likelihood-implementation/data/'
    'ExternalBenchmark/Photometric/data/'
    'CovMat-3x2pt-Gauss-20Bins-probe_ell_zpair.npy')


del cov_dict_BNT_True, cov_dict_BNT_False, cov_dict_BNT_True_with_cls
gc.collect()

# ! check that the BNT-transformed covariance is the same as the one computed with the BNT-transformed cls
mm.compare_arrays(cov_3x2pt_GO_BNT_True, cov_3x2pt_GO_BNT_True_with_cls,
                  'cov_3x2pt_GO_BNT_True', 'cov_3x2pt_GO_BNT_True_with_cls',
                  plot_array=True, log_array=True,
                  plot_diff=True, log_diff=True, plot_diff_threshold=10)

# ! check that the Gaussian covariance matrices are the same
mm.compare_arrays(cov_3x2pt_GO_BNT_False, cov_3x2pt_GO_BNT_False_benchmark,
                  'cov_3x2pt_GO_BNT_False', 'cov_3x2pt_GO_BNT_False_benchmark',
                  plot_array=True, log_array=True,
                  plot_diff=True, log_diff=True, plot_diff_threshold=20)

assert 1 > 2



# diff = mm.percent_diff(cov_3x2pt_GO_BNT_True[-1100:, -1100:], cov_3x2pt_GO_benchmark[-1100:, -1100:])
# mm.matshow(diff, log=False, abs_val=True, title='WL diff')

# diff = mm.percent_diff(cov_3x2pt_GO_BNT_True, cov_3x2pt_GO_BNT_False)
# mm.matshow(np.abs(diff), title='diff', log=False, abs_val=False)
#
# mm.compare_arrays(cov_3x2pt_GO_BNT_True, cov_3x2pt_GO_BNT_False,
#                   name_A='cov_3x2pt_GO_BNT_True', name_B='cov_3x2pt_GO_BNT_False', plot_diff=True, log_diff=True,
#                   plot_array=True, log_array=True)

# ! ok but not perfect there is still a (small) number of outliers; maybe check:
# ! also, why do I have to recompute the covariance?
# - cls (undo the modifications to simulate_data to have the triu elements, then unpack...)
# - deltas
# - ell values?

np.save(f'{job_path}/output/CovMat-3x2pt-Gauss-BNT-20Bins.npy', cov_3x2pt_GO_BNT_True)

print('done')
