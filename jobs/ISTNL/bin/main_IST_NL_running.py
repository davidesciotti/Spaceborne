import sys
import time
from pathlib import Path

import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as spar
import numpy as np
import pandas as pd

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
job_name = job_path.parts[-1]

# general libraries
sys.path.append(f'{project_path}/lib')
import my_module as mm
import cosmo_lib as csmlib

# general config
sys.path.append(f'{project_path}/config')
import mpl_cfg
import ISTF_fid_params as ISTFfid

# job configuration
sys.path.append(f'{job_path}/config')
import config_IST_NL as cfg

# project libraries
sys.path.append(f'{project_path}/bin')
import ell_values_running as ell_utils
import Cl_preprocessing_running as cl_utils
import compute_Sijkl as Sijkl_utils
import covariance_running as covmat_utils
import FM_running as FM_utils
import utils_running as utils
import unit_test

matplotlib.use('Qt5Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

# import the configuration dictionaries from common_config.py
general_config = cfg.general_config
covariance_config = cfg.covariance_config
FM_config = cfg.FM_cfg
plot_config = cfg.plot_config

# plot settings:
params = plot_config['params']
markersize = plot_config['markersize']
dpi = plot_config['dpi']
pic_format = plot_config['pic_format']
plt.rcParams.update(params)

# consistency checks:
utils.consistency_checks(general_config, covariance_config)

# for the time being, I/O is manual and from the main
# load inputs (job-specific)
ind_ordering = covariance_config['ind_ordering']
ind_2 = np.genfromtxt(f"{project_path}/input/ind_files/indici_{ind_ordering}_like_int.dat", dtype=int)
covariance_config['ind'] = ind

# this is the string indicating the flattening, or "block", convention
# TODO raise exception here?
which_flattening_str = None  # initialize to a value
if covariance_config['block_index'] in ['ell', 'vincenzo', 'C-style']:
    which_flattening_str = 'Cstyle'
elif covariance_config['block_index'] in ['ij', 'sylvain', 'F-style']:
    which_flattening_str = 'Fstyle'

# convention = 0 # Lacasa & Grain 2019, incorrect (using IST WFs)
convention = 1  # Euclid, correct

bia = 0.0  # the one used by CLOE at the moment
# bia = 2.17

assert bia == 0.0, 'bia must be 0!'
assert convention == 1, 'convention must be 1 for Euclid!'

# normalization = 'IST'
# normalization = 'PySSC'
# from new SSC code (need to specify the 'convention' needed)
Sijkl_marco = np.load(
    job_path / f"input/CosmoCentral_outputs/Sijkl/Sijkl_WFmarco_nz10000_bia{bia}_conv{convention}_gen22.npy")

# from old PySSC code (so no 'convention' parameter needed)
# Sijkl_marco = np.load(path / f"data/CosmoCentral_outputs/Sijkl/Sijkl_WFmarco_nz10000_bia{bia}_{normalization}normaliz_oldCode.npy") 

# old Sijkl matrices
# Sijkl_sylv = np.load(path.parent / "common_data/Sijkl/sylvain_cosmologyFixe_IAFixe_Sijkl_GCph_WL_XC_3may.npy") 
# Sijkl_dav = np.load(path.parent / "common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy") 


Sijkl = Sijkl_marco

# a couple more checks
assert np.all(Sijkl == Sijkl_marco), 'Sijkl should be Sijkl_marco'
assert bia == 0., 'IST_NL uses bia = 0, or the "zNLA model" (for the moment)'

###############################################################################
######################### FORECAST COMPUTATION ################################
###############################################################################

for NL_flag in range(5):

    # linear case has a different file name (no '-NL_flag_{NL_flag} suffix')
    if NL_flag == 0:
        NL_flag_string_cl = ''
        NL_flag_string_cov = ''
    else:
        # the only difference is an underscore/hyphen at the beginning of the string
        NL_flag_string_cl = f'_NL_flag_{NL_flag}'
        NL_flag_string_cov = f'-NL_flag_{NL_flag}'

    # import Cl
    C_LL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data'
                                        f'/Cls_zNLA_ShearShear{NL_flag_string_cl}.dat')
    C_GL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data'
                                        f'/Cls_zNLA_PosShear{NL_flag_string_cl}.dat')
    C_GG_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data'
                                        f'/Cls_zNLA_PosPos{NL_flag_string_cl}.dat')

    # remove ell column
    C_LL_2D = C_LL_2D[:, 1:]
    C_GL_2D = C_GL_2D[:, 1:]
    C_GG_2D = C_GG_2D[:, 1:]

    # set ells and deltas
    ell_bins = np.linspace(np.log(10.), np.log(5000.), 21)
    ells = (ell_bins[:-1] + ell_bins[1:]) / 2.
    ells = np.exp(ells)
    deltas = np.diff(np.exp(ell_bins))

    # store everything in dicts
    ell_dict = {
        'ell_WL': ells,
        'ell_GC': ells,
        'ell_WA': ells}  # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
    delta_dict = {
        'delta_l_WL': deltas,
        'delta_l_GC': deltas,
        'delta_l_WA': deltas}  # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
    cl_dict_2D = {
        'C_LL_2D': C_LL_2D,
        'C_LL_WLonly_2D': C_LL_2D,
        # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
        'C_XC_2D': C_GL_2D,
        'C_GG_2D': C_GG_2D,
        'C_WA_2D': C_LL_2D}  # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything

    ###############################################################################

    # ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_config)
    # Cl_dict = cl_utils.generate_Cls(general_config, ell_dict, cl_dict_2D)

    cl_dict_3D = cl_utils.reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D)
    cov_dict = covmat_utils.compute_cov(general_config, covariance_config, ell_dict, delta_dict, cl_dict_3D, Sijkl)

    # save
    # smarter save: loop over the names (careful of the ordering, tuples are better than lists because immutable)
    probe_CLOE_list = ('PosPos', 'ShearShear', '3x2pt')
    probe_script_list = ('GC', 'WL', '3x2pt')
    # ! settle this once we solve the ordering issue!! If we go for fstyle there will no longer be need for 2DCLOE
    # ndim_list = ('2D', '2D', '2DCLOE')  # in the 3x2pt case the name is '2DCLOE', not '2D'
    # no 2DCLOE
    ndim_list = ('2D', '2D', '2D')  # in the 3x2pt case the name is '2DCLOE', not '2D'

    GO_or_GS_CLOE_list = ('Gauss', 'GaussSSC')
    GO_or_GS_script_list = ('GO', 'GS')

    # probes and dimensions
    for probe_CLOE, probe_script, ndim in zip(probe_CLOE_list, probe_script_list, ndim_list):
        for GO_or_GS_CLOE, GO_or_GS_script in zip(GO_or_GS_CLOE_list, GO_or_GS_script_list):
            # save sparse (.npz)
            spar.save_npz(
                job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins{NL_flag_string_cov}-{ndim}-'
                           f'{which_flattening_str}-Sparse.npz',
                spar.csr_matrix(cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_{ndim}']))

            # save normal (.npy)
            np.save(
                job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins{NL_flag_string_cov}-{ndim}-'
                           f'{which_flattening_str}.npy',
                cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_{ndim}'])

            # save 4D in npy (sparse only works for arrays with dimension <= 2)
            np.save(
                job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins{NL_flag_string_cov}_4D.npy',
                cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_4D'])

    # more readable and more error prone?, probably...
    # np.save(job_path / f'output/covmat/CovMat-ShearShear-Gauss-20bins{NL_flag_string_cov}_2D.npy', cov_dict[f'cov_WL_GO_2D'])
    # np.save(job_path / f'output/covmat/CovMat-PosPos-Gauss-20bins{NL_flag_string_cov}_2D.npy', cov_dict[f'cov_GC_GO_2D'])
    # np.save(job_path / f'output/covmat/CovMat-3x2pt-Gauss-20bins{NL_flag_string_cov}_2D.npy', cov_dict[f'cov_3x2pt_GO_2D'])

    # np.save(job_path / f'output/covmat/CovMat-ShearShear-GaussSSC-20bins{NL_flag_string_cov}_2D.npy', cov_dict[f'cov_WL_GS_2D'])
    # np.save(job_path / f'output/covmat/CovMat-PosPos-GaussSSC-20bins{NL_flag_string_cov}_2D.npy', cov_dict[f'cov_GC_GS_2D'])
    # np.save(job_path / f'output/covmat/CovMat-3x2pt-GaussSSC-20bins{NL_flag_string_cov}_2D.npy', cov_dict[f'cov_3x2pt_GS_2D'])

    np.save(job_path / f'output/covmat/CovMat-ShearShear-SSC-20bins{NL_flag_string_cov}_4D.npy',
            cov_dict[f'cov_WL_SS_4D'])
    np.save(job_path / f'output/covmat/CovMat-PosPos-SSC-20bins{NL_flag_string_cov}_4D.npy', cov_dict[f'cov_GC_SS_4D'])
    np.save(job_path / f'output/covmat/CovMat-3x2pt-SSC-20bins{NL_flag_string_cov}_4D.npy',
            cov_dict[f'cov_3x2pt_SS_4D'])

    # compute FM and pot, just to get an idea (the derivatives are still from Vincenzo!)
    # FM_dict = FM_utils.compute_FM(general_config, covariance_config, FM_config, ell_dict, cov_dict)
    # plt.figure()
    # plot_utils.plot_FM(general_config, covariance_config, plot_config, FM_dict)

    # some tests
    # TODO put these tests in another module
    # probe = '3x2pt'

    # tests

    # cov_WL_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-ShearShear-Gauss-20Bins.npy')
    # cov_GC_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-PosPos-Gauss-20Bins.npy')
    # cov_3x2pt_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-3x2pt-Gauss-20Bins.npy')

    # cov_d_4D = cov_dict[f'cov_{probe}_G_4D']
    # cov_d_2D = cov_dict[f'cov_{probe}_G_2D']
    # cov_d_2DCLOE = cov_dict[f'cov_{probe}_G_2DCLOE']

    # cov_d = cov_d_2D
    # cov_s = cov_3x2pt_benchmark2

    # limit = 210
    # mm.matshow(cov_d[:limit,:limit], f'Davide {probe} 1st', log=True, abs_val=True)
    # mm.matshow(cov_s[:limit,:limit], f'Santiago {probe} 1st', log=True, abs_val=True)
    # mm.matshow(cov_d[-limit:,-limit:], f'Davide {probe} last', log=True, abs_val=True)
    # mm.matshow(cov_s[-limit:,-limit:], f'Santiago {probe} last', log=True, abs_val=True)
    # mm.matshow(cov_d, f'Davide {probe}', log=True, abs_val=True)
    # mm.matshow(cov_s, f'Santiago {probe}', log=True, abs_val=True)

    # # mm.matshow(np.abs(cov_d_2DCLOE), f'Davide {probe} 2DCLOE (wrong?)', log=True)

    # diff = mm.percent_diff(cov_s, cov_d)
    # # diff = np.where(np.abs(diff) > 5, diff, 1)
    # mm.matshow(diff[:limit,:limit], 'diff 1st block', log=False, abs_val=True)
    # mm.matshow(diff[-limit:,-limit:], 'diff last block', log=False, abs_val=True)
    # mm.matshow(diff, 'diff', log=False, abs_val=True)

stop_time = time.perf_counter() - start_time
print(f'done in {stop_time:.2f} s')
