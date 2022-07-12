import sys
import time
from pathlib import Path

import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Qt5Agg')

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

# general libraries
sys.path.append(str(project_path.parent / 'common_data/common_lib'))
import my_module as mm
import cosmo_lib as csmlib

# job configuration
sys.path.append(str(project_path / 'jobs'))
import SPV3.configs.config_SPV3 as cfg

# project libraries
sys.path.append(str(project_path / 'bin'))
import ell_values_running as ell_utils
import Cl_preprocessing_running as Cl_utils
import compute_Sijkl as Sijkl_utils
import covariance_running as covmat_utils
import FM_running as FM_utils
import utils_running as utils
import unit_test

start_time = time.perf_counter()

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

# import the configuration dictionaries from config.py
general_config = cfg.general_config
covariance_config = cfg.covariance_config
Sijkl_config = cfg.Sijkl_config
FM_config = cfg.FM_config
cosmo_params_dict = csmlib.cosmo_par_dict_classy

# consistency checks:
utils.consistency_checks(general_config, covariance_config)

# for the time being, I/O is manual and from the main
# load inputs (job-specific)
ind = np.genfromtxt(project_path / "config/common_data/ind/indici_vincenzo_like.dat").astype(int) - 1
covariance_config['ind'] = ind

Sijkl_dav = np.load(project_path / "config/common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy")  # davide, eNLA
# Sijkl_marco = np.load(project_path / "config/common_data/Sijkl/Sijkl_WFmarco_nz10000_zNLA_gen22.npy")  # marco, zNLA
# Sijkl_sylv = np.load(project_path / "config/common_data/Sijkl/sylvain_cosmologyFixe_IAFixe_Sijkl_GCph_WL_XC_3may.npy")  # sylvain, eNLA

Sijkl = Sijkl_dav

assert np.array_equal(Sijkl, Sijkl_dav), 'Sijkl should be Sijkl_dav'

###############################################################################
######################### FORECAST COMPUTATION ################################
###############################################################################

# for (general_config['ell_max_WL'], general_config['ell_max_GC']) in ((5000, 3000), (1500, 750)):

which_probe_response = covariance_config['which_probe_response']
# set the string, just for the file names
if which_probe_response == 'constant':
    which_probe_response_str = 'const'
elif which_probe_response == 'variable':
    which_probe_response_str = 'var'
else:
    raise ValueError('which_probe_response must be either constant or variable')


# some variables used for I/O naming and to compute Sylvain's deltas
ell_max_WL = general_config['ell_max_WL']
ell_max_GC = general_config['ell_max_GC']
ell_max_XC = ell_max_GC
nbl = general_config['nbl']

# compute ell and delta ell values
ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_config)

nbl_WA = ell_dict['ell_WA'].shape[0]

# ! deltas are different for Sylvain! overwrite the standard delta_dict
delta_dict['delta_l_WL'] = mm.delta_l_Sylvain(nbl, 10 ** ell_dict['ell_WL'])
delta_dict['delta_l_GC'] = mm.delta_l_Sylvain(nbl, 10 ** ell_dict['ell_GC'])
delta_dict['delta_l_WA'] = mm.delta_l_Sylvain(nbl_WA, 10 ** ell_dict['ell_WA'])

# import and interpolate the cls
cl_dict_2D, Rl_dict_2D = Cl_utils.import_and_interpolate_cls(general_config, covariance_config, ell_dict)
# reshape them to 3D
cl_dict_3D, Rl_dict_3D = Cl_utils.reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D, Rl_dict_2D)
# Sijkl from PySSC
Sijkl = Sijkl_utils.compute_Sijkl(cosmo_params_dict, Sijkl_config, zbins=general_config['zbins'])

mm.matshow(Sijkl[0,0,:,:])
mm.matshow(Sijkl_dav[0,0,:,:])

if cfg.Sijkl_config['save_Sijkl']:
    np.save(project_path / f'output/sijkl/sijkl_wf{cfg.Sijkl_config["input_WF"]}.npy')


assert 1 > 2



# compute covariance matrix
cov_dict = covmat_utils.compute_cov(general_config, covariance_config,
                                    ell_dict, delta_dict, cl_dict_3D, Rl_dict_3D, Sijkl)
# compute Fisher Matrix
FM_dict = FM_utils.compute_FM(general_config, covariance_config, FM_config, ell_dict, cov_dict)

# save:
if covariance_config['save_covariance']:
    np.save(job_path / f'output/covmat/covmat_GO_WL_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WL_GO_2D'])
    np.save(job_path / f'output/covmat/covmat_GO_GC_lmaxGC{ell_max_GC}_nbl{nbl}_2D.npy', cov_dict['cov_GC_GO_2D'])
    np.save(job_path / f'output/covmat/covmat_GO_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_2D.npy', cov_dict['cov_3x2pt_GO_2D'])
    np.save(job_path / f'output/covmat/covmat_GO_WA_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WA_GO_2D'])

    np.save(job_path / f'output/covmat/covmat_GS_WL_lmaxWL{ell_max_WL}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_WL_GS_2D'])
    np.save(job_path / f'output/covmat/covmat_GS_GC_lmaxGC{ell_max_GC}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_GC_GS_2D'])
    np.save(job_path / f'output/covmat/covmat_GS_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_3x2pt_GS_2D'])
    np.save(job_path / f'output/covmat/covmat_GS_WA_lmaxWL{ell_max_WL}_nbl{nbl}_Rl{which_probe_response_str}_2D.npy',
            cov_dict['cov_WA_GS_2D'])

if FM_config['save_FM']:
    np.savetxt(job_path / f"output/FM/FM_WL_GO_lmaxWL{ell_max_WL}_nbl{nbl}.txt", FM_dict['FM_WL_GO'])
    np.savetxt(job_path / f"output/FM/FM_GC_GO_lmaxGC{ell_max_GC}_nbl{nbl}.txt", FM_dict['FM_GC_GO'])
    np.savetxt(job_path / f"output/FM/FM_3x2pt_GO_lmaxXC{ell_max_XC}_nbl{nbl}.txt", FM_dict['FM_3x2pt_GO'])

    np.savetxt(job_path / f"output/FM/FM_WL_GS_lmaxWL{ell_max_WL}_nbl{nbl}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_WL_GS'])
    np.savetxt(job_path / f"output/FM/FM_GC_GS_lmaxGC{ell_max_GC}_nbl{nbl}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_GC_GS'])
    np.savetxt(job_path / f"output/FM/FM_3x2pt_GS_lmaxXC{ell_max_XC}_nbl{nbl}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_3x2pt_GS'])

if FM_config['save_FM_as_dict']:
    sio.savemat(job_path / f'output/FM/FM_dict.mat', FM_dict)

if general_config['save_cls']:
    for key in cl_dict_3D.keys():
        np.save(job_path / f"output/cl_3D/{key}.npy", cl_dict_3D[f'{key}'])



print('done')
