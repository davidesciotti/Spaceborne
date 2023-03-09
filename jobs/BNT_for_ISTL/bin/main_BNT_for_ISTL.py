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

# ! import CLOE's ells, delta_ell and cls
old_data_path = f'{job_path}/data/874-BNT-cls/from_870_DEMO-BNT'
from_DEMO_870_path = f'{job_path}/data/from_BNT_DEMO_870'
data_path = f'{job_path}/data'
cl_branch_870_path = '/Users/davide/Documents/Lavoro/Programmi/likelihood-implementation/data/ExternalBenchmark/' \
                     'Photometric/data'

# save the cls as it's done in simulate_data.py, then unpack them
# cl_LL_2d_bench = np.genfromtxt(f'{data_path}/Cls_zNLA_ShearShear_new.dat')
# cl_GL_2d_bench = np.genfromtxt(f'{data_path}/Cls_zNLA_PosShear_new.dat')
# cl_GG_2d_bench = np.genfromtxt(f'{data_path}/Cls_zNLA_PosPos_new.dat')
#
# # get the cls directly from the benchmarks folder in the repo
# cl_LL_2d_bench = np.genfromtxt(f'{cl_branch_870_path}/Cls_zNLA_ShearShear.dat')
# cl_GL_2d_bench = np.genfromtxt(f'{cl_branch_870_path}/Cls_zNLA_PosShear.dat')
# cl_GG_2d_bench = np.genfromtxt(f'{cl_branch_870_path}/Cls_zNLA_PosPos.dat')
#
# assert np.all(cl_LL_2d_bench[:, 0] == cl_GL_2d_bench[:, 0]), 'ell values are not the same for all the cls'
# assert np.all(cl_LL_2d_bench[:, 0] == cl_GG_2d_bench[:, 0]), 'ell values are not the same for all the cls'
#
# ells = cl_LL_2d_bench[:, 0]
# nbl = len(ells)
#
# # remove ell column
# cl_LL_2d_bench = cl_LL_2d_bench[:, 1:]
# cl_GL_2d_bench = cl_GL_2d_bench[:, 1:]
# cl_GG_2d_bench = cl_GG_2d_bench[:, 1:]


# reshape
# cl_LL_3d_bench = mm.cl_2D_to_3D_symmetric(cl_LL_2d_bench, nbl, zpairs_auto, zbins)
# cl_GL_3d_bench = mm.cl_2D_to_3D_asymmetric(cl_GL_2d_bench, nbl, zbins, 'row-major')
# cl_GG_3d_bench = mm.cl_2D_to_3D_symmetric(cl_GG_2d_bench, nbl, zpairs_auto, zbins)

# ! get the cls from the 870 BNT DEMO; these are already in 3D
cl_LL_3d = np.load(f'{from_DEMO_870_path}/cC_LL_arr.npy')
cl_GL_3d = np.load(f'{from_DEMO_870_path}/cC_GL_arr.npy').transpose(0, 2, 1)
cl_GG_3d = np.load(f'{from_DEMO_870_path}/cC_GG_arr.npy')

# note that this name is quite verbose, the BNT-ized cls must come from the BNT demo!
cl_LL_BNT_3d = np.load(f'{from_DEMO_870_path}/cC_LL_BNT.npy')
cl_GL_BNT_3d = np.load(f'{from_DEMO_870_path}/cC_GL_BNT.npy')
cl_GG_BNT_3d = np.load(f'{from_DEMO_870_path}/cC_GG_BNT.npy')

# ! check that the cls are the same (they are not, by a small amount)
# try:
#     np.testing.assert_array_equal(cl_LL_3d_bench, cl_LL_3d, verbose=False)
#     np.testing.assert_array_equal(cl_GL_3d_bench, cl_GL_3d_BNT_DEMO, verbose=False)
#     np.testing.assert_array_equal(cl_GG_3d_bench, cl_GG_3d_BNT_DEMO, verbose=False)
# except AssertionError as e:
#     print(e)

# TODO check that the cov with these cls is the same as the bnt-transformed covariance
# choose which cls to use, whether the benchmarks or the BNT-DEMO ones (they are slightly different)
# cl_LL_3d = cl_LL_3d_BNT_DEMO
# cl_GL_3d = cl_GL_3d_BNT_DEMO
# cl_GG_3d = cl_GG_3d_BNT_DEMO

# ! check GL against the cls from Vincenzo (very bad agreement...)
check_against_cl_15gen_func()

# ! build BNT covariance by transforming the Gaussian one
bnt_matrix = np.genfromtxt(f'{from_DEMO_870_path}/BNT_matrix.txt')

cl_LL_BNT_3d_dark = cl_utils.cl_BNT_transform(cl_LL_3d, bnt_matrix, 'L', 'L')
cl_GL_BNT_3d_dark = cl_utils.cl_BNT_transform(cl_GL_3d, bnt_matrix, 'G', 'L')
cl_GG_BNT_3d_dark = cl_utils.cl_BNT_transform(cl_GG_3d, bnt_matrix, 'G', 'G')

# compare_diagonal_cls(cl_LL_BNT_3d_dark, cl_LL_BNT_3d_BNT_DEMO)
# compare_diagonal_cls(cl_GL_BNT_3d_dark, cl_GL_BNT_3d_BNT_DEMO)
# compare_diagonal_cls(cl_LL_BNT_3d_dark, cl_LL_BNT_3d_BNT_DEMO)

np.testing.assert_array_equal(cl_GL_BNT_3d_dark, cl_GL_BNT_3d_BNT_DEMO, verbose=False)

plt.figure()
diff = mm.percent_diff(cl_GL_BNT_3d_dark, cl_GL_BNT_3d_BNT_DEMO)
for zi in range(zbins):
    plt.plot(ells, diff[:, zi, zi])

ell_idx = 0
mm.matshow(diff[ell_idx, :, :])
# mm.compare_arrays(cl_GL_BNT_3d_dark, cl_GL_BNT_3d_BNT_DEMO)


assert 1 > 2

cl_3x2pt_5D = cl_utils.build_3x2pt_datavector_5D(cl_LL_3d, cl_GL_3d, cl_GG_3d, nbl, zbins)
cl_3x2pt_BNT_5D_BNT_DEMO = cl_utils.build_3x2pt_datavector_5D(cl_LL_BNT_3d, cl_GL_BNT_3d, cl_GG_BNT_3d, nbl, zbins)

# the ell values are all the same!
ells = np.load(f'{old_data_path}/ell_values.npy')
delta_ells = np.load(f'{old_data_path}/delta_ells.npy')

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

cl_BNT_dict_3D = {
    'cl_LL_3D': cl_LL_BNT_3d_BNT_DEMO,
    'cl_GL_3D': cl_GL_BNT_3d_BNT_DEMO,
    'cl_GG_3D': cl_GG_BNT_3d_BNT_DEMO,
    'cl_WA_3D': cl_LL_BNT_3d_BNT_DEMO,  # ! not used
    'cl_3x2pt_5D': cl_3x2pt_BNT_5D_BNT_DEMO,
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
                                                      ell_dict, delta_dict, cl_BNT_dict_3D, rl_dict_3D, sijkl)

cov_3x2pt_GO_BNT_True = cov_dict_BNT_True['cov_3x2pt_GO_2DCLOE']
cov_3x2pt_GO_BNT_False = cov_dict_BNT_False['cov_3x2pt_GO_2DCLOE']
cov_3x2pt_GO_BNT_True_with_cls = cov_dict_BNT_True_with_cls['cov_3x2pt_GO_2DCLOE']

del cov_dict_BNT_True, cov_dict_BNT_False, cov_dict_BNT_True_with_cls
gc.collect()

# ! check that the BNT-transformed covariance is the same as the one computed with the BNT-transformed cls
mm.compare_arrays(cov_3x2pt_GO_BNT_True, cov_3x2pt_GO_BNT_True_with_cls,
                  'cov_3x2pt_GO_BNT_True', 'cov_3x2pt_GO_BNT_True_with_cls', plot_array=True,
                  log_array=True, log_diff=False,
                  plot_diff=True)

assert 1 > 2

# ! check that the Gaussian covariance matrix are the same
if bnt_transform:
    # cov_3x2pt_GO_2DCLOE_bnt = ...
    raise NotImplementedError('I still have to output the BNT-transformed cov by CLOE')
    # print('are the two covariance matrices equal?', np.array_equal(cov_3x2pt_GO_BNT_True_2DCLOE, cov_3x2pt_GO_2DCLOE_benchmark))

else:
    warnings.warn('you have to be in branch #870 for this import to work')
    cov_3x2pt_GO_BNT_False_benchmark = np.load(
        '/Users/davide/Documents/Lavoro/Programmi/likelihood-implementation/data/'
        'ExternalBenchmark/Photometric/data/'
        'CovMat-3x2pt-Gauss-20Bins-probe_ell_zpair.npy')

    try:
        print('are the two covariance matrices equal?',
              np.testing.assert_allclose(cov_3x2pt_GO_BNT_False, cov_3x2pt_GO_BNT_False_benchmark, atol=0,
                                         rtol=1e-3))
    except AssertionError:
        print('covariance matrices are not close')

    mm.compare_arrays(cov_3x2pt_GO_BNT_False, cov_3x2pt_GO_BNT_False_benchmark,
                      plot_array=True, log_array=True,
                      plot_diff=True, log_diff=False, plot_diff_threshold=10)

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
