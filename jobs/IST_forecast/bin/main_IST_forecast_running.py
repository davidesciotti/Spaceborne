import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/lib')
import my_module as mm

# general config
sys.path.append(f'{project_path}/config')
import mpl_cfg
import ISTF_fid_params as ISTFfid

sys.path.append(f'{project_path}/bin')
import ell_values_running as ell_utils
import Cl_preprocessing_running as Cl_utils
import covariance_running as covmat_utils
import FM_running as FM_utils
import plots_FM_running as plot_utils
import utils_running as utils
import unit_test as tests

# job configuration
sys.path.append(f'{job_path}/config')
import config_IST_forecast as cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)

start_time = time.perf_counter()





###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

# import the configuration dictionaries from common_config.py
general_config = cfg.general_config
covariance_config = cfg.covariance_config
FM_config = cfg.FM_cfg
plot_config = cfg.plot_config

for general_config['ell_max_WL'], general_config['ell_max_GC'] in (5000, 3000), (1500, 750):
    for covariance_config['which_probe_response'] in ['constant', 'variable']:

        which_probe_response = covariance_config['which_probe_response']
        # set the string, just for the file names
        if which_probe_response == 'constant':
            which_probe_response_str = 'const'
        elif which_probe_response == 'variable':
            which_probe_response_str = 'var'
        else:
            raise ValueError('which_probe_response must be either constant or variable')



        # consistency checks:
        utils.consistency_checks(general_config, covariance_config)

        # for the time being, I/O is manual and from the main
        # load inputs (job-specific)
        ind_ordering = covariance_config['ind_ordering']
        ind = np.genfromtxt(project_path / f"input/ind_files/indici_{ind_ordering}_like_int.dat", dtype=int)
        covariance_config['ind'] = ind

        Sijkl_dav = np.load(project_path / "input/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy")  # davide, eNLA
        # Sijkl_marco = np.load(project_path / "input/Sijkl/Sijkl_WFmarco_nz10000_zNLA_gen22.npy")  # marco, zNLA
        # Sijkl_sylv = np.load(project_path / "input/Sijkl/sylvain_cosmologyFixe_IAFixe_Sijkl_GCph_WL_XC_3may.npy")  # sylvain, eNLA

        Sijkl = Sijkl_dav

        assert np.all(Sijkl == Sijkl_dav), 'Sijkl should be Sijkl_dav'

        ###############################################################################
        ######################### FORECAST COMPUTATION ################################
        ###############################################################################

        # some variables used for I/O naming and to compute Sylvain's deltas
        ell_max_WL = general_config['ell_max_WL']
        ell_max_GC = general_config['ell_max_GC']
        ell_max_XC = ell_max_GC
        ell_max_WL = general_config['ell_max_WL']
        nbl = general_config['nbl']

        # MAIN BODY OF THE SCRIPT
        # compute ell and delta ell values
        ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_config)
        # import and interpolate the cls
        cl_dict_2D, rl_dict_2D = Cl_utils.import_and_interpolate_cls(general_config, covariance_config, ell_dict)
        # reshape them to 3D
        cl_dict_3D, rl_dict_3D = Cl_utils.reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D, rl_dict_2D)
        # compute covariance matrix
        cov_dict = covmat_utils.compute_cov(general_config, covariance_config, ell_dict, delta_dict, cl_dict_3D, rl_dict_3D,
                                            Sijkl)
        # compute Fisher Matrix
        FM_dict = FM_utils.compute_FM(general_config, covariance_config, FM_config, ell_dict, cov_dict)
        # plot forecasts
        # plt.figure()
        # for plot_config['GO_or_GS'] in ['GO', 'GS']:
        #     for plot_config['probe'] in ['WL', 'GC', '3x2pt']:
        # plot_utils.plot_FM(general_config, covariance_config, plot_config, FM_dict)

        # ! new, delete
        """
        probe = 'WL'
        if ell_max_WL == 5000:
            case = 'opt'
        else:
            case = 'pes'
        
        if probe == 'WL':
            nparams = 8
        
        FM_LCDM = np.delete(FM_dict[f'FM_{probe}_GO'], (2, 3), axis=0)
        FM_LCDM = np.delete(FM_LCDM, (2, 3), axis=1)
        
        uncert_dav_wCDM = mm.uncertainties_FM(FM_dict[f'FM_{probe}_GO'])[:7]
        uncert_dav_LCDM = mm.uncertainties_FM(FM_LCDM, nparams)[:5]
        uncert_ISTF_wCDM = ISTFfid.forecasts[f'{probe}_{case}_w0waCDM_flat']
        uncert_ISTF_LCDM = ISTFfid.forecasts[f'{probe}_{case}_LCDM_flat']
        
        print('uncert_dav_wCDM', uncert_dav_wCDM)
        print('uncert_ISTF_wCDM', uncert_ISTF_wCDM)
        print('uncert_dav_LCDM', uncert_dav_LCDM)
        print('uncert_ISTF_LCDM', uncert_ISTF_LCDM)
        
        diff = mm.percent_diff(uncert_dav_LCDM, uncert_ISTF_LCDM)
        print('diff', diff)
        """
        # ! end new


        # save:
        if covariance_config['save_covariance']:
            np.save(f'{job_path}/output/covmat/covmat_GO_WL_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WL_GO_2D'])
            np.save(f'{job_path}/output/covmat/covmat_GO_GC_lmaxGC{ell_max_GC}_nbl{nbl}_2D.npy', cov_dict['cov_GC_GO_2D'])
            np.save(f'{job_path}/output/covmat/covmat_GO_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_2D.npy', cov_dict['cov_3x2pt_GO_2D'])
            np.save(f'{job_path}/output/covmat/covmat_GO_WA_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WA_GO_2D'])

            np.save(f'{job_path}/output/covmat/covmat_GS_WL_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WL_GS_2D'])
            np.save(f'{job_path}/output/covmat/covmat_GS_GC_lmaxGC{ell_max_GC}_nbl{nbl}_2D.npy', cov_dict['cov_GC_GS_2D'])
            np.save(f'{job_path}/output/covmat/covmat_GS_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl}_2D.npy', cov_dict['cov_3x2pt_GS_2D'])
            np.save(f'{job_path}/output/covmat/covmat_GS_WA_lmaxWL{ell_max_WL}_nbl{nbl}_2D.npy', cov_dict['cov_WA_GS_2D'])

        if FM_config['save_FM']:
            np.savetxt(f"{job_path}/output/FM/FM_WL_GO_lmaxWL{ell_max_WL}_nbl{nbl}.txt", FM_dict['FM_WL_GO'])  # WLonly
            np.savetxt(f"{job_path}/output/FM/FM_GC_GO_lmaxGC{ell_max_GC}_nbl{nbl}.txt", FM_dict['FM_GC_GO'])  # GConly
            np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GO_lmaxXC{ell_max_XC}_nbl{nbl}.txt", FM_dict['FM_3x2pt_GO'])  # ALL

            np.savetxt(f"{job_path}/output/FM/FM_WL_GS_lmaxWL{ell_max_WL}_nbl{nbl}_Rl{which_probe_response_str}.txt", FM_dict['FM_WL_GS'])
            np.savetxt(f"{job_path}/output/FM/FM_GC_GS_lmaxGC{ell_max_GC}_nbl{nbl}_Rl{which_probe_response_str}.txt", FM_dict['FM_GC_GS'])
            np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GS_lmaxXC{ell_max_XC}_nbl{nbl}_Rl{which_probe_response_str}.txt", FM_dict['FM_3x2pt_GS'])


########### TESTS #############################################
general_config['ell_max_WL'], general_config['ell_max_GC'] = 5000, 3000
covariance_config['which_probe_response'] = 'constant'
tests.check_FMs_against_oldSSCscript(f'{job_path}/output/FM', general_config, covariance_config, tolerance=0.0001)
