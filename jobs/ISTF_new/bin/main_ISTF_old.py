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
sys.path.append(f'{project_path.parent}/config')
import mpl_cfg

# job configuration
sys.path.append(f'{job_path}/config')
import config_ISTF_old as cfg

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
FM_cfg = cfg.FM_config

# consistency checks:
utils.consistency_checks(general_cfg, covariance_cfg)

# for the time being, I/O is manual and from the main
# load inputs (job-specific)
ind = np.genfromtxt(f"{project_path}/input/ind_files/indici_vincenzo_like_int.dat", dtype=int)
covariance_cfg['ind'] = ind

Sijkl_dav = np.load(f"{project_path}/input/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy")  # davide, eNLA
# Sijkl_marco = np.load(project_path / "input/Sijkl/Sijkl_WFmarco_nz10000_zNLA_gen22.npy")  # marco, zNLA
# Sijkl_sylv = np.load(project_path / "input/Sijkl/sylvain_cosmologyFixe_IAFixe_Sijkl_GCph_WL_XC_3may.npy")  # sylvain, eNLA

Sijkl = Sijkl_dav

assert np.array_equal(Sijkl, Sijkl_dav), 'Sijkl should be Sijkl_dav'

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

    # ! deltas are different for Sylvain! overwrite the standard delta_dict
    delta_dict['delta_l_WL'] = mm.delta_l_Sylvain(nbl, 10 ** ell_dict['ell_WL'])
    delta_dict['delta_l_GC'] = mm.delta_l_Sylvain(nbl, 10 ** ell_dict['ell_GC'])
    delta_dict['delta_l_WA'] = mm.delta_l_Sylvain(nbl_WA, 10 ** ell_dict['ell_WA'])

    # ! ################################################ MAIN BODY #####################################################
    # import and interpolate the cls
    cl_dict_2D, Rl_dict_2D = cl_utils.import_and_interpolate_cls(general_cfg, covariance_cfg, ell_dict)

    # reshape them to 3D
    cl_dict_3D, Rl_dict_3D = cl_utils.reshape_cls_2D_to_3D(general_cfg, ell_dict, cl_dict_2D, Rl_dict_2D)

    # compute covariance matrix
    cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                        ell_dict, delta_dict, cl_dict_3D, Rl_dict_3D, Sijkl)
    # compute Fisher Matrix
    FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict)
    # ! ############################################## END MAIN BODY ###################################################

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

print('done')

# test
unit_test.check_FMs_against_oldSSCscript(job_path / f"output/FM", general_cfg, covariance_cfg)
