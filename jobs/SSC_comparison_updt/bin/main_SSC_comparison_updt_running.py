import sys
import time
from pathlib import Path
import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

# general libraries
sys.path.append(f'{project_path}/lib')
import my_module as mm
import cosmo_lib as csmlib

# general configurations
sys.path.append(f'{project_path}/config')
import mpl_cfg

# job configuration
sys.path.append(f'{job_path}/config')
import config_SSC_comparison_updt as cfg

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

start_time = time.perf_counter()

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

# import the configuration dictionaries from config.py
general_cfg = cfg.general_config
covariance_cfg = cfg.covariance_config
Sijkl_cfg = cfg.Sijkl_config
FM_cfg = cfg.FM_config

# consistency checks:
# utils.consistency_checks(general_cfg, covariance_cfg)

# load inputs (job-specific)
zbins = general_cfg['zbins']

ind = np.genfromtxt(
    f'{project_path}/input/ind_files/variable_zbins/{covariance_cfg["ind_ordering"]}_like/indici_{covariance_cfg["ind_ordering"]}_like_zbins{zbins}.dat',
    dtype=int)
covariance_cfg['ind'] = ind

if Sijkl_cfg['use_precomputed_sijkl']:
    sijkl = np.load(
        f'{job_path}/output/sijkl/sijkl_wf{Sijkl_cfg["input_WF"]}_nz7000_zbins{zbins}_hasIA{Sijkl_cfg["has_IA"]}.npy')

else:
    start_time = time.perf_counter()
    sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, Sijkl_cfg, zbins=zbins)
    print(f'zbins {zbins}: Sijkl computation took {time.perf_counter() - start_time:.2} seconds')

    if Sijkl_cfg['save_Sijkl']:
        np.save(
            f'{job_path}/output/sijkl/sijkl_wf{Sijkl_cfg["input_WF"]}_nz7000_zbins{zbins}_hasIA{Sijkl_cfg["has_IA"]}.npy',
            sijkl)

###############################################################################
######################### FORECAST COMPUTATION ################################
###############################################################################


for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((5000, 3000), (1500, 750)):

    which_probe_response = covariance_cfg['which_probe_response']
    # set the string, just for the file names
    if which_probe_response == 'constant':
        which_probe_response_str = 'const'
    elif which_probe_response == 'variable':
        which_probe_response_str = 'var'
    else:
        raise ValueError('which_probe_response must be either constant or variable')

    # some variables used for I/O naming and to compute Sylvain's deltas
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    ell_max_XC = ell_max_GC
    nbl = general_cfg['nbl']

    # compute ell and delta ell values
    ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_cfg)
    nbl_WA = ell_dict['ell_WA'].shape[0]

    # import and interpolate the cls
    cl_dict_2D, Rl_dict_2D = cl_utils.import_and_interpolate_cls(general_cfg, covariance_cfg, ell_dict)
    # reshape them to 3D
    cl_dict_3D, Rl_dict_3D = cl_utils.reshape_cls_2D_to_3D(general_cfg, ell_dict, cl_dict_2D, Rl_dict_2D)

    if new_responses:
        # take the ell values for the interpolation and the number of ell bins
        nbl_WL_spv3 = 32
        ell_WL_spv3, _ = ell_utils.compute_ells(nbl_WL_spv3, general_cfg['ell_min'],
                                                general_cfg['ell_max_WL'], recipe='ISTF')

        ell_dict_spv3 = {}
        ell_dict_spv3['ell_WL'] = ell_WL_spv3
        ell_dict_spv3['ell_GC'] = np.copy(ell_WL_spv3[ell_WL_spv3 < ell_max_GC])
        ell_dict_spv3['ell_WA'] = np.copy(ell_WL_spv3[ell_WL_spv3 > ell_max_GC])
        ell_dict_spv3['ell_XC'] = np.copy(ell_dict_spv3['ell_GC'])

        nbl_GC_spv3 = ell_dict_spv3['ell_GC'].shape[0]
        nbl_WA_spv3 = ell_dict_spv3['ell_WA'].shape[0]
        nbl_3x2pt_spv3 = nbl_GC_spv3

        rl_ll_3d = cl_utils.get_spv3_cls_3d('WL', nbl_WL_spv3, general_cfg['zbins'], ell_max_WL=ell_max_WL,
                                            cls_or_responses='responses')
        rl_gg_3d = cl_utils.get_spv3_cls_3d('GC', nbl_GC_spv3, general_cfg['zbins'], ell_max_WL=ell_max_WL,
                                            cls_or_responses='responses')
        rl_wa_3d = cl_utils.get_spv3_cls_3d('WA', nbl_WA_spv3, general_cfg['zbins'], ell_max_WL=ell_max_WL,
                                            cls_or_responses='responses')
        rl_3x2pt_5d = cl_utils.get_spv3_cls_3d('3x2pt', nbl_3x2pt_spv3, general_cfg['zbins'], ell_max_WL=ell_max_WL,
                                               cls_or_responses='responses')

        rl_ll_3d_fn = interp1d(ell_WL_spv3, rl_ll_3d, axis=0)
        rl_ll_3d_interp = rl_ll_3d_fn(ell_dict['ell_WL'])

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        my_resp = np.load(
            '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/Variable_response/output/R_LL_WLonly_3D.npy')
        ell_LL_my_rl = np.load(
            '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/Variable_response/output/ell_LL.npy')

        # for i in range(general_cfg['zbins']):
        i = 0
        plt.plot(ell_WL_spv3, rl_ll_3d[:, i, i], label='new', c=colors[i])
        plt.plot(ell_dict['ell_WL'], rl_ll_3d_interp[:, i, i], label='interp', c=colors[i])
        # plt.plot(10**ell_dict['ell_WL'], Rl_dict_3D['R_LL_WLonly_3D'][:, i, i], '--', label='old', c=colors[i])
        # plt.plot(ell_LL_my_rl, my_resp[:, i, i], '-.', label='davide', c=colors[i])
        plt.legend()
        plt.show()
        plt.grid()

        assert 1 > 2, 'this is not implemented yet'

    # compute covariance matrix
    cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                        ell_dict, delta_dict, cl_dict_3D, Rl_dict_3D, sijkl)
    # compute Fisher Matrix
    FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict)

    assert 1 > 2

    # save:
    if covariance_cfg['save_covariance']:
        np.save(f'{job_path}/output/covmat/covmat_GO_WL_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WL_GO_2D'])
        np.save(f'{job_path}/output/covmat/covmat_GO_GC_lmaxGC{ell_max_GC}_nbl{nbl}_2D.npy', cov_dict['cov_GC_GO_2D'])
        np.save(f'{job_path}/output/covmat/covmat_GO_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_2D.npy',
                cov_dict['cov_3x2pt_GO_2D'])
        np.save(f'{job_path}/output/covmat/covmat_GO_WA_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WA_GO_2D'])

        np.save(
            f'{job_path}/output/covmat/covmat_GS_WL_lmaxWL{ell_max_WL}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_WL_GS_2D'])
        np.save(
            f'{job_path}/output/covmat/covmat_GS_GC_lmaxGC{ell_max_GC}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_GC_GS_2D'])
        np.save(
            f'{job_path}/output/covmat/covmat_GS_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_3x2pt_GS_2D'])
        np.save(
            f'{job_path}/output/covmat/covmat_GS_WA_lmaxWL{ell_max_WL}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_WA_GS_2D'])

    if FM_cfg['save_FM']:
        np.savetxt(f"{job_path}/output/FM/FM_WL_GO_lmaxWL{ell_max_WL}_nbl{nbl}.txt", FM_dict['FM_WL_GO'])
        np.savetxt(f"{job_path}/output/FM/FM_GC_GO_lmaxGC{ell_max_GC}_nbl{nbl}.txt", FM_dict['FM_GC_GO'])
        np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GO_lmaxXC{ell_max_XC}_nbl{nbl}.txt", FM_dict['FM_3x2pt_GO'])

        np.savetxt(f"{job_path}/output/FM/FM_WL_GS_lmaxWL{ell_max_WL}_nbl{nbl}_Rl{which_probe_response_str}.txt",
                   FM_dict['FM_WL_GS'])
        np.savetxt(f"{job_path}/output/FM/FM_GC_GS_lmaxGC{ell_max_GC}_nbl{nbl}_Rl{which_probe_response_str}.txt",
                   FM_dict['FM_GC_GS'])
        np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GS_lmaxXC{ell_max_XC}_nbl{nbl}_Rl{which_probe_response_str}.txt",
                   FM_dict['FM_3x2pt_GS'])

    if FM_cfg['save_FM_as_dict']:
        sio.savemat(f'{job_path}/output/FM/FM_dict.mat', FM_dict)

    if general_cfg['save_cls']:
        for key in cl_dict_3D.keys():
            np.save(f"{job_path}/output/cl_3D/{key}.npy", cl_dict_3D[f'{key}'])

    # test

print('done')

unit_test.FM_check(job_path / f"output/FM", general_cfg, covariance_cfg)
