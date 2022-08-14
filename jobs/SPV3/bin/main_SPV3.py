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
import config_SPV3 as cfg

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

# TODO ind will be different for the different number of z bins ✅
# TODO finish exploring the cls ✅
# TODO make sure you changed fsky ✅
# TODO change sigma_eps2? ✅
# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks
# TODO super check that things work with different # of z bins


###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_cfg = cfg.general_config
covariance_cfg = cfg.covariance_config
Sijkl_cfg = cfg.Sijkl_config
FM_cfg = cfg.FM_config

which_probe_response = covariance_cfg['which_probe_response']
# set the string, just for the file names
if which_probe_response == 'constant':
    which_probe_response_str = 'const'
elif which_probe_response == 'variable':
    which_probe_response_str = 'var'
else:
    raise ValueError('which_probe_response must be either constant or variable')

zbins_SPV3 = (3, 5, 7, 9, 10, 11, 13, 15, 17)
zbins_SPV3 = (3, 5, 7, 9, 10)
# zbins_SPV3 = (10,)

for general_cfg['zbins'] in zbins_SPV3:
    for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((1500, 750), (5000, 3000)):

        # utils.consistency_checks(general_cfg, covariance_cfg)

        zbins = general_cfg['zbins']
        EP_or_ED = general_cfg['EP_or_ED']

        ind = np.genfromtxt(f'{project_path}/input/ind_files/variable_zbins/{covariance_cfg["ind_ordering"]}_like/'
                            f'indici_{covariance_cfg["ind_ordering"]}_like_zbins{zbins}.dat', dtype=int)
        covariance_cfg['ind'] = ind

        # some variables used for I/O naming
        ell_max_WL_opt = general_cfg['ell_max_WL_opt']
        ell_max_WL = general_cfg['ell_max_WL']
        ell_max_GC = general_cfg['ell_max_GC']
        ell_max_XC = ell_max_GC
        nbl_WL_32 = general_cfg['nbl_WL_32']

        assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
            'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

        # compute ell and delta ell values in the reference (optimistic) case
        ell_WL_nbl32, delta_l_WL_nbl32 = ell_utils.compute_ells(general_cfg['nbl_WL_32'], general_cfg['ell_min'],
                                                                general_cfg['ell_max_WL_opt'], recipe='ISTF')

        ell_WL_nbl32 = np.log10(ell_WL_nbl32)

        # perform the cuts
        ell_dict = {}
        ell_dict['ell_WL'] = np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_WL])
        ell_dict['ell_GC'] = np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_GC])
        ell_dict['ell_WA'] = np.copy(
            ell_WL_nbl32[(10 ** ell_WL_nbl32 > ell_max_GC) & (10 ** ell_WL_nbl32 < ell_max_WL)])
        ell_dict['ell_XC'] = np.copy(ell_dict['ell_GC'])

        # set corresponding # of ell bins
        nbl_WL = ell_dict['ell_WL'].shape[0]
        nbl_GC = ell_dict['ell_GC'].shape[0]
        nbl_WA = ell_dict['ell_WA'].shape[0]
        nbl_3x2pt = nbl_GC
        general_cfg['nbl_WL'] = nbl_WL

        # ! not super sure about these deltas
        delta_dict = {}
        delta_dict['delta_l_WL'] = np.copy(delta_l_WL_nbl32[:nbl_WL])
        delta_dict['delta_l_GC'] = np.copy(delta_l_WL_nbl32[:nbl_GC])
        delta_dict['delta_l_WA'] = np.copy(delta_l_WL_nbl32[nbl_GC:])

        # set # of nbl in the opt case, import and reshape, then cut the reshaped datavectors in the pes case
        nbl_WL_opt = 32
        nbl_GC_opt = 29
        nbl_WA_opt = 3
        nbl_3x2pt_opt = 29

        if ell_max_WL == ell_max_WL_opt:
            assert (nbl_WL_opt, nbl_GC_opt, nbl_WA_opt, nbl_3x2pt_opt) == (nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt), \
                'nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt don\'t match with the expected values for the optimistic case'

        cl_ll_3d = cl_utils.get_spv3_cls_3d('WL', nbl_WL_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                            cls_or_responses='cls', specs=general_cfg['specs'], EP_or_ED=EP_or_ED)
        cl_gg_3d = cl_utils.get_spv3_cls_3d('GC', nbl_GC_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                            cls_or_responses='cls', specs=general_cfg['specs'], EP_or_ED=EP_or_ED)
        cl_wa_3d = cl_utils.get_spv3_cls_3d('WA', nbl_WA_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                            cls_or_responses='cls', specs=general_cfg['specs'], EP_or_ED=EP_or_ED)
        cl_3x2pt_5d = cl_utils.get_spv3_cls_3d('3x2pt', nbl_3x2pt_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                               cls_or_responses='cls', specs=general_cfg['specs'], EP_or_ED=EP_or_ED)

        rl_ll_3d = cl_utils.get_spv3_cls_3d('WL', nbl_WL_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                            cls_or_responses='responses', specs=general_cfg['specs'], EP_or_ED=EP_or_ED)
        rl_gg_3d = cl_utils.get_spv3_cls_3d('GC', nbl_GC_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                            cls_or_responses='responses', specs=general_cfg['specs'], EP_or_ED=EP_or_ED)
        rl_wa_3d = cl_utils.get_spv3_cls_3d('WA', nbl_WA_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                            cls_or_responses='responses', specs=general_cfg['specs'], EP_or_ED=EP_or_ED)
        rl_3x2pt_5d = cl_utils.get_spv3_cls_3d('3x2pt', nbl_3x2pt_opt, general_cfg['nbl_WL_32'], zbins, ell_max_WL_opt,
                                               cls_or_responses='responses', specs=general_cfg['specs'],
                                               EP_or_ED=EP_or_ED)

        if ell_max_WL == ell_max_WL_opt:
            assert (
                np.array_equal(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :])), 'cl_wa_3d should be obtainable from cl_ll_3d!'

        # cut datavectors and responses; be carful of WA, because it does not start from ell_min
        if ell_max_WL == 1500:
            cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
            cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
            cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
            cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

            rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
            rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
            rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
            rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

        cl_dict_3D = {
            'C_LL_WLonly_3D': cl_ll_3d,
            'C_GG_3D': cl_gg_3d,
            'C_WA_3D': cl_wa_3d,
            'C_3x2pt_5D': cl_3x2pt_5d}

        rl_dict_3D = {
            'R_LL_WLonly_3D': rl_ll_3d,
            'R_GG_3D': rl_gg_3d,
            'R_WA_3D': rl_wa_3d,
            'R_3x2pt_5D': rl_3x2pt_5d}

        if Sijkl_cfg['use_precomputed_sijkl']:
            sijkl = np.load(
                f'{job_path}/output/sijkl/sijkl_wf{Sijkl_cfg["input_WF"]}_nz7000_zbins{zbins:02}_hasIA{Sijkl_cfg["has_IA"]}.npy')

        else:
            start_time = time.perf_counter()
            sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, Sijkl_cfg, zbins=zbins)
            print(f'zbins {zbins}: Sijkl computation took {time.perf_counter() - start_time:.2} seconds')

            if Sijkl_cfg['save_Sijkl']:
                np.save(
                    f'{job_path}/output/sijkl/sijkl_wf{Sijkl_cfg["input_WF"]}_nz7000_zbins{zbins:02}_hasIA{Sijkl_cfg["has_IA"]}.npy',
                    sijkl)

        # compute covariance matrix
        cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                            ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl)

        # compute Fisher Matrix
        # FM_dict = FM_utils.compute_FM(general_config, covariance_cfg, FM_config, ell_dict, cov_dict)

        # ! SAVE:
        # this is just to set the correct probe names
        probe_dav_dict = {
            'WL': 'LL_WLonly_3D',
            'GC': 'GG_3D',
            'WA': 'WA_3D',
            '3x2pt': '3x2pt_5D',
        }

        cl_rl_path = f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022'
        covmat_path = f'{job_path}/output/covmat/zbins{zbins:02}'
        if general_cfg['save_cls_3d']:
            for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
                np.save(f'{cl_rl_path}/DataVectors/3D_reshaped/{probe_vinc}/'
                        f'dv-{probe_vinc}-{nbl_WL}-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.npy',
                        cl_dict_3D[f'C_{probe_dav_dict[probe_dav]}'])

                if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                    np.savetxt(
                        f'{cl_rl_path}/DataVectors/3D_reshaped/{probe_vinc}/ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                        10 ** ell_dict[f'ell_{probe_dav}'])

        if general_cfg['save_rls_3d']:
            for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
                np.save(f'{cl_rl_path}/ResFunTabs/3D_reshaped/{probe_vinc}/'
                        f'rf-{probe_vinc}-{nbl_WL}-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.npy',
                        rl_dict_3D[f'R_{probe_dav_dict[probe_dav]}'])

                if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                    np.savetxt(
                        f'{cl_rl_path}/ResFunTabs/3D_reshaped/{probe_vinc}/ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                        10 ** ell_dict[f'ell_{probe_dav}'])

        if covariance_cfg['save_covariance'] and ell_max_WL == 5000:
            np.save(f'{covmat_path}/covmat_GO_WL_lmaxWL{ell_max_WL}_nbl{nbl_WL}_zbins{zbins:02}_2D.npy',
                    cov_dict['cov_WL_GO_2D'])
            np.save(f'{covmat_path}/covmat_GO_GC_lmaxGC{ell_max_GC}_nbl{nbl_GC}_zbins{zbins:02}_2D.npy',
                    cov_dict['cov_GC_GO_2D'])
            np.save(f'{covmat_path}/covmat_GO_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl_3x2pt}_zbins{zbins:02}_2D.npy',
                    cov_dict['cov_3x2pt_GO_2D'])
            np.save(f'{covmat_path}/covmat_GO_WA_lmaxWL{ell_max_WL}_nbl{nbl_WA}_zbins{zbins:02}_2D.npy',
                    cov_dict['cov_WA_GO_2D'])

            np.save(f'{covmat_path}/covmat_GS_WL_lmaxWL{ell_max_WL}_nbl{nbl_WL}'
                    f'_zbins{zbins:02}_Rl{which_probe_response_str}_2D.npy', cov_dict['cov_WL_GS_2D'])
            np.save(f'{covmat_path}/covmat_GS_GC_lmaxGC{ell_max_GC}_nbl{nbl_GC}'
                    f'_zbins{zbins:02}_Rl{which_probe_response_str}_2D.npy', cov_dict['cov_GC_GS_2D'])
            np.save(f'{covmat_path}/covmat_GS_3x2pt_lmaxXC{ell_max_XC}_nbl{nbl_3x2pt}'
                    f'_zbins{zbins:02}_Rl{which_probe_response_str}_2D.npy', cov_dict['cov_3x2pt_GS_2D'])
            np.save(f'{covmat_path}/covmat_GS_WA_lmaxWL{ell_max_WL}_nbl{nbl_WA}'
                    f'_zbins{zbins:02}_Rl{which_probe_response_str}_2D.npy', cov_dict['cov_WA_GS_2D'])

        elif covariance_cfg['save_covariance'] and ell_max_WL == 1500:
            # in this case, save only WA
            np.save(f'{covmat_path}/covmat_GO_WA_lmaxWL{ell_max_WL}_nbl{nbl_WA}'
                    f'_zbins{zbins:02}_2D.npy', cov_dict['cov_WA_GO_2D'])
            np.save(f'{covmat_path}/covmat_GS_WA_lmaxWL{ell_max_WL}_nbl{nbl_WA}'
                    f'_zbins{zbins:02}_Rl{which_probe_response_str}_2D.npy', cov_dict['cov_WA_GS_2D'])

        if covariance_cfg['save_covariance_dat']:
            path_vinc_fmt = f'{job_path}/output/covmat/vincenzos_format'
            for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
                for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):

                    # this is just because the 3x2pt folder is called "All" instead of "3x2pt"
                    if probe == '3x2pt':
                        folder_probe_vinc = 'All'
                    else:
                        folder_probe_vinc = probe_vinc

                    np.savetxt(f'{path_vinc_fmt}/{GOGS_folder}/{folder_probe_vinc}/cm-{probe_vinc}-{nbl_WL}'
                               f'-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.dat',
                               cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.10e')

assert 1 == 0, 'stop here'

if FM_cfg['save_FM']:
    np.savetxt(f"{job_path}/output/FM/FM_WL_GO_lmaxWL{ell_max_WL}_nbl{nbl_WL}.txt", FM_dict['FM_WL_GO'])
    np.savetxt(f"{job_path}/output/FM/FM_GC_GO_lmaxGC{ell_max_GC}_nbl{nbl_WL}.txt", FM_dict['FM_GC_GO'])
    np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GO_lmaxXC{ell_max_XC}_nbl{nbl_WL}.txt", FM_dict['FM_3x2pt_GO'])
    np.savetxt(f"{job_path}/output/FM/FM_WL_GS_lmaxWL{ell_max_WL}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_WL_GS'])
    np.savetxt(f"{job_path}/output/FM/FM_GC_GS_lmaxGC{ell_max_GC}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_GC_GS'])
    np.savetxt(f"{job_path}/output/FM/FM_3x2pt_GS_lmaxXC{ell_max_XC}_nbl{nbl_WL}_Rl{which_probe_response_str}.txt",
               FM_dict['FM_3x2pt_GS'])

if FM_cfg['save_FM_as_dict']:
    sio.savemat(job_path / f'output/FM/FM_dict.mat', FM_dict)

print('done')
