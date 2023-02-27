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
ell_min = general_cfg['ell_min']
ell_max_WL = general_cfg['ell_max_WL']
ell_max_GC = general_cfg['ell_max_GC']
ell_max_XC = general_cfg['ell_max_XC']
triu_tril = covariance_cfg['triu_tril']
row_col_major = covariance_cfg['row_col_major']
n_probes = general_cfg['n_probes']
nbl_WL = general_cfg['nbl_WL']
nbl_GC = general_cfg['nbl_GC']
nbl = nbl_WL
bIA = ISTFfid.IA_free['beta_IA']

variable_specs = {
    'zbins': zbins,
    'EP_or_ED': EP_or_ED,
    'triu_tril': triu_tril,
    'row_col_major': row_col_major,
}

# ! other options
bnt_transform = False
covariance_cfg['compute_covmat'] = True
# ! end other options

# some checks
assert EP_or_ED == 'EP' and zbins == 10, 'ISTF uses 10 equipopulated bins'
assert covariance_cfg['GL_or_LG'] == 'GL', 'Cij_14may uses GL, also for the probe responses'
assert nbl_GC == nbl_WL, 'for ISTF we are using the same number of ell bins for WL and GC'

zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

# import the ind files and store it into the covariance dictionary
ind_folder = covariance_cfg['ind_folder'].format(**variable_specs)
ind_filename = covariance_cfg['ind_filename'].format(**variable_specs)
ind = np.genfromtxt(f'{ind_folder}/{ind_filename}', dtype=int)
covariance_cfg['ind'] = ind

# ! import CLOE's ells, delta_ell and cls
data_path = '/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT'

cl_LL_3d = np.load(f'{data_path}/cC_LL_arr.npy')
cl_GL_3d = np.load(f'{data_path}/cC_GL_arr.npy').transpose(0, 2, 1)  # ! check against cij14may to make sure it's GL
cl_GG_3d = np.load(f'{data_path}/cC_GG_arr.npy')
# if bnt_transform:
#     cl_LL_3d = np.load(f'{data_path}/cC_LL_BNT.npy')
#     cl_GL_3d = np.load(f'{data_path}/cC_GL_BNT.npy')
#     cl_GG_3d = np.load(f'{data_path}/cC_GG_BNT.npy')

bnt_matrix = np.load(f'{data_path}/mat_BNT.npy')

cl_3x2pt_5D = cl_utils.build_3x2pt_datavector_5D(cl_LL_3d, cl_GL_3d, cl_GG_3d, nbl_GC, zbins, n_probes)

# the ell values are all the same!
ells = np.load(f'{data_path}/ell_values.npy')
delta_ells = np.load(f'{data_path}/delta_ells.npy')

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

rl_dict_3D = {}

# ! compute covariance matrix

# ! compute or load Sijkl
# if Sijkl exists, load it; otherwise, compute it and save it
Sijkl_folder = Sijkl_cfg['Sijkl_folder']
Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(nz=Sijkl_cfg['nz'])

# if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):
#
#     print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
#     sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')

# else:
#
#     # ! load kernels
#     # TODO this should not be done if Sijkl is loaded; I have a problem with nz, which is part of the file name...
#     nz = Sijkl_cfg["nz"]
#     wf_folder = Sijkl_cfg["wf_input_folder"].format(nz=nz)
#     wil_filename = Sijkl_cfg["wil_filename"].format(normalization=Sijkl_cfg['wf_normalization'],
#                                                     has_IA=str(Sijkl_cfg['has_IA']), nz=nz, bIA=bIA)
#     wig_filename = Sijkl_cfg["wig_filename"].format(normalization=Sijkl_cfg['wf_normalization'], nz=nz)
#     wil = np.genfromtxt(f'{wf_folder}/{wil_filename}')
#     wig = np.genfromtxt(f'{wf_folder}/{wig_filename}')
#
#     # preprocess (remove redshift column)
#     z_arr, wil = Sijkl_utils.preprocess_wf(wil, zbins)
#     z_arr_2, wig = Sijkl_utils.preprocess_wf(wig, zbins)
#     assert np.array_equal(z_arr, z_arr_2), 'the redshift arrays are different for the GC and WL kernels'
#     assert nz == z_arr.shape[0], 'nz is not the same as the number of redshift points in the kernels'
#
#     # transpose and stack, ordering is important here!
#     transp_stacked_wf = np.vstack((wil.T, wig.T))
#     sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
#                                       Sijkl_cfg['WF_normalization'])
#     np.save(f'{Sijkl_folder}/{Sijkl_filename}', sijkl)

# dummy sijkl matrix, I am not computing covSSC...
sijkl = np.random.rand(2 * zbins, 2 * zbins, 2 * zbins, 2 * zbins)

# ! compute covariance matrix
covariance_cfg['cov_BNT_transform'] = True
cov_dict_BNT = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                        ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl, BNT_matrix=bnt_matrix)

covariance_cfg['cov_BNT_transform'] = False
cov_dict_noBNT = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                          ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl)

# check
cov_3x2pt_GO_BNT_2DCLOE = cov_dict_BNT['cov_3x2pt_GO_2DCLOE']
cov_3x2pt_GO_noBNT_2DCLOE = cov_dict_noBNT['cov_3x2pt_GO_2DCLOE']

del cov_dict_BNT, cov_dict_noBNT
gc.collect()

# TODO: check that the gaussian covariance agrees with the file in the repo

if bnt_transform:
    # cov_3x2pt_GO_2DCLOE_bnt = ...
    raise NotImplementedError('I still have to output the BNT-transformed cov by CLOE')
    # print('are the two covariance matrices equal?', np.array_equal(cov_3x2pt_GO_BNT_2DCLOE, cov_3x2pt_GO_2DCLOE_benchmark))

else:
    warnings.warn('you have to be in branch #870 for this import to work')
    cov_3x2pt_GO_2DCLOE_noBNT_benchmark = np.load(
        '/Users/davide/Documents/Lavoro/Programmi/likelihood-implementation/data/'
        'ExternalBenchmark/Photometric/data/'
        'CovMat-3x2pt-Gauss-20Bins-probe_ell_zpair.npy')

    try:
        print('are the two covariance matrices equal?',
              np.testing.assert_allclose(cov_3x2pt_GO_noBNT_2DCLOE, cov_3x2pt_GO_2DCLOE_noBNT_benchmark, atol=0, rtol=1e-3))
    except AssertionError:
        print('covariance matrices are not close')

    mm.compare_arrays(cov_3x2pt_GO_noBNT_2DCLOE, cov_3x2pt_GO_2DCLOE_noBNT_benchmark,
                      plot_array=True, log_array=True,
                      plot_diff=True, log_diff=True)

    # diff = mm.percent_diff(cov_3x2pt_GO_BNT_2DCLOE[-1100:, -1100:], cov_3x2pt_GO_2DCLOE_benchmark[-1100:, -1100:])
    # mm.matshow(diff, log=False, abs_val=True, title='WL diff')

# diff = mm.percent_diff(cov_3x2pt_GO_BNT_2DCLOE, cov_3x2pt_GO_noBNT_2DCLOE)
# mm.matshow(np.abs(diff), title='diff', log=False, abs_val=False)
#
# mm.compare_arrays(cov_3x2pt_GO_BNT_2DCLOE, cov_3x2pt_GO_noBNT_2DCLOE,
#                   name_A='cov_3x2pt_GO_BNT_2DCLOE', name_B='cov_3x2pt_GO_noBNT_2DCLOE', plot_diff=True, log_diff=True,
#                   plot_array=True, log_array=True)

# ! ok but not perfect there is still a (small) number of outliers; maybe check:
# ! also, why do I have to recompute the covariance?
# - cls (undo the modifications to simulate_data to have the triu elements, then unpack...)
# - deltas
# - ell values?

np.save(f'{job_path}/output/CovMat-3x2pt-Gauss-BNT-20Bins.npy', cov_3x2pt_GO_BNT_2DCLOE)

print('done')
