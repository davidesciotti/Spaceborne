import pickle
import sys
import time
from pathlib import Path
import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import warnings
import scipy.sparse as spar
import _pickle as cPickle

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

# job configuration
sys.path.append(f'{job_path}/config')
import config_ISTF as cfg

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


def save_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'wb') as handle:
        cPickle.dump(data, handle)


def load_compressed_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


# todo move this to my_module
def dC_dict_to_4D_array(param_names, dC_dict_3D, nbl, zbins, is_3x2pt=False, n_probes=2):
    # param_names should be params_tot in all cases, because when the derivative dows not exist
    # in dC_dict_3D the output array will remain null
    if is_3x2pt:
        dC_4D = np.zeros((nbl, n_probes, n_probes, zbins, zbins, len(param_names)))
    else:
        dC_4D = np.zeros((nbl, zbins, zbins, len(param_names)))

    if not dC_dict_3D:
        warnings.warn('The input dictionary is empty')

    for idx, paramname in enumerate(param_names):
        for key, value in dC_dict_3D.items():
            if f'dDVd{paramname}' in key:
                dC_4D[..., idx] = value

        # a check, if the derivative wrt the param is not in the folder at all
        if not any(f'dDVd{paramname}' in key for key in dC_dict_3D.keys()):
            print(f'WARNING: derivative dDVd{paramname} not found in dC_dict_3D')
    return dC_4D


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

which_probe_response = covariance_cfg['which_probe_response']
# set the string, just for the file names
if which_probe_response == 'constant':
    which_probe_response_str = 'const'
elif which_probe_response == 'variable':
    which_probe_response_str = 'var'
else:
    raise ValueError('which_probe_response must be either constant or variable')

for covariance_cfg['triu_tril'] in ['triu', 'tril']:
    for covariance_cfg['row_col_wise'] in ['row-wise', 'col-wise']:
        for general_cfg['zbins'] in general_cfg['zbins_list']:
            # for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((5000, 3000), (1500, 750)):
            for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((5000, 3000),):
                for general_cfg['EP_or_ED'] in general_cfg['zbins_type_list']:

                    # utils.consistency_checks(general_cfg, covariance_cfg)

                    # some variables used for I/O naming, just to make things more readable
                    zbins = general_cfg['zbins']
                    EP_or_ED = general_cfg['EP_or_ED']
                    ell_min = general_cfg['ell_min']
                    ell_max_WL = general_cfg['ell_max_WL']
                    ell_max_GC = general_cfg['ell_max_GC']
                    ell_max_XC = ell_max_GC
                    triu_tril = covariance_cfg['triu_tril']
                    row_col_wise = covariance_cfg['row_col_wise']
                    n_probes = general_cfg['n_probes']
                    nbl_WL = general_cfg['nbl_WL']
                    nbl_GC = general_cfg['nbl_GC']

                    # which cases to save: GO, GS or GO, GS and SS
                    cases_tosave = ['GO', 'GS']
                    if covariance_cfg[f'save_cov_GS']:
                        cases_tosave.append('GS')
                    if covariance_cfg[f'save_cov_SS']:
                        cases_tosave.append('SS')

                    variable_specs = {
                        'zbins': zbins,
                        'EP_or_ED': EP_or_ED,
                        'triu_tril': triu_tril,
                        'row_col_wise': row_col_wise,
                    }

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

                    # ! compute ell and delta ell values
                    ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_cfg)
                    nbl_WA = ell_dict['ell_WA'].shape[0]
                    ell_WL, ell_GC, ell_WA = ell_dict['ell_WL'], ell_dict['ell_GC'], ell_dict['ell_WA']


                    # import
                    cl_folder = general_cfg['cl_folder'].format(**variable_specs)
                    cl_filename = general_cfg['cl_filename']
                    cl_LL_1D = np.genfromtxt(f'{cl_folder}/{cl_filename.format(probe="LL")}')
                    cl_GL_1D = np.genfromtxt(f'{cl_folder}/{cl_filename.format(probe="GL")}')
                    cl_GG_1D = np.genfromtxt(f'{cl_folder}/{cl_filename.format(probe="GG")}')

                    rl_folder = general_cfg['rl_folder'].format(**variable_specs)
                    rl_filename = general_cfg['rl_filename']
                    rl_LL_1D = np.genfromtxt(f'{rl_folder}/{rl_filename.format(probe="ll")}')
                    rl_GL_1D = np.genfromtxt(f'{rl_folder}/{rl_filename.format(probe="gl")}')
                    rl_GG_1D = np.genfromtxt(f'{rl_folder}/{rl_filename.format(probe="ll")}')

                    # interpolate
                    cl_dict_2D = {}
                    cl_dict_2D['cl_LL_2D'] = mm.Cl_interpolator(zpairs_auto, cl_LL_1D, ell_WL, nbl_WL)
                    cl_dict_2D['cl_GG_2D'] = mm.Cl_interpolator(zpairs_auto, cl_GG_1D, ell_GC, nbl_GC)
                    cl_dict_2D['cl_WA_2D'] = mm.Cl_interpolator(zpairs_auto, cl_LL_1D, ell_WA, nbl_WA)
                    cl_dict_2D['cl_GL_2D'] = mm.Cl_interpolator(zpairs_cross, cl_GL_1D, ell_GC, nbl_GC)
                    cl_dict_2D['cl_LLfor3x2pt_2D'] = mm.Cl_interpolator(zpairs_auto, cl_LL_1D, ell_GC, nbl_GC)

                    rl_dict_2D = {}
                    rl_dict_2D['rl_LL_2D'] = mm.Cl_interpolator(zpairs_auto, rl_LL_1D, ell_WL, nbl_WL)
                    rl_dict_2D['rl_GG_2D'] = mm.Cl_interpolator(zpairs_auto, rl_GG_1D, ell_GC, nbl_GC)
                    rl_dict_2D['rl_WA_2D'] = mm.Cl_interpolator(zpairs_auto, rl_LL_1D, ell_WA, nbl_WA)
                    rl_dict_2D['rl_GL_2D'] = mm.Cl_interpolator(zpairs_cross, rl_GL_1D, ell_GC, nbl_GC)
                    rl_dict_2D['rl_LLfor3x2pt_2D'] = mm.Cl_interpolator(zpairs_auto, rl_LL_1D, ell_GC, nbl_GC)

                    # reshape to 3D
                    cl_dict_3D = {}
                    cl_dict_3D['cl_LL_3D'] = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_LL_2D'], nbl_WL, zpairs_auto, zbins)
                    cl_dict_3D['cl_GG_3D'] = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_GG_2D'], nbl_GC, zpairs_auto, zbins)
                    cl_dict_3D['cl_WA_3D'] = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_WA_2D'], nbl_WA, zpairs_auto, zbins)
                    cl_dict_3D['cl_GL_3D'] = mm.cl_2D_to_3D_asymmetric(cl_dict_2D['cl_GL_2D'], nbl_GC, zbins)
                    cl_dict_3D['cl_LLfor3x2pt_3D'] = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_LLfor3x2pt_2D'], nbl_GC, zpairs_auto, zbins)

                    rl_dict_3D = {}
                    rl_dict_3D['rl_LL_3D'] = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_LL_2D'], nbl_WL, zpairs_auto, zbins)
                    rl_dict_3D['rl_GG_3D'] = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_GG_2D'], nbl_GC, zpairs_auto, zbins)
                    rl_dict_3D['rl_WA_3D'] = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_WA_2D'], nbl_WA, zpairs_auto, zbins)
                    rl_dict_3D['rl_GL_3D'] = mm.cl_2D_to_3D_asymmetric(rl_dict_2D['rl_GL_2D'], nbl_GC, zbins)
                    rl_dict_3D['rl_LLfor3x2pt_3D'] = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_LLfor3x2pt_2D'], nbl_GC, zpairs_auto, zbins)
                    # end split import and


                    # ! load, interpolate, reshape cls and responses
                    general_cfg['nbl'] = nbl_WL
                    general_cfg['cl_folder'] = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/14may/CijDers'
                    general_cfg['rl_folder'] = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/Pk_responses_2D/EP10'
                    cl_dict_2D_old, rl_dict_2D_old = cl_utils.import_and_interpolate_cls(general_cfg, covariance_cfg, ell_dict)
                    cl_dict_3D_old, rl_dict_3D_old = cl_utils.reshape_cls_2D_to_3D(general_cfg, ell_dict, cl_dict_2D, rl_dict_2D)

                    assert 1 > 2
                    # ! compute covariance matrix
                    sijkl = np.load(f"{project_path}/input/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy")  # davide, eNLA
                    if covariance_cfg['compute_covmat']:
                        cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                                            ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl)

                    # ! compute Fisher Matrix
                    if FM_cfg['compute_FM']:
                        FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict)

                    # ! save:
                    # this is just to set the correct probe names
                    probe_dav_dict = {
                        'WL': 'LL_WLonly_3D',
                        'GC': 'GG_3D',
                        'WA': 'WA_3D',
                        '3x2pt': '3x2pt_5D'}

                    # just a dict for the output file names
                    clrl_dict = {
                        'cl_inputname': 'dv',
                        'rl_inputname': 'rf',
                        'cl_dict_3D': cl_dict_3D,
                        'rl_dict_3D': rl_dict_3D,
                        'cl_dict_key': 'C',
                        'rl_dict_key': 'R',
                    }
                    for cl_or_rl in ['cl', 'rl']:
                        folder = general_cfg[f'{cl_or_rl}_folder']
                        if general_cfg[f'save_{cl_or_rl}s_3d']:

                            for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'],
                                                             ['WL', 'GC', '3x2pt', 'WA']):
                                # save cl and/or response
                                np.save(f'{folder}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}/'
                                        f'{clrl_dict[f"{cl_or_rl}_inputname"]}-{probe_vinc}-{nbl_WL}-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.npy',
                                        clrl_dict[f"{cl_or_rl}_dict_3D"][
                                            f'{clrl_dict[f"{cl_or_rl}_dict_key"]}_{probe_dav_dict[probe_dav]}'])

                                # save ells and deltas
                                if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                                    np.savetxt(
                                        f'{folder}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}/ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                                        10 ** ell_dict[f'ell_{probe_dav}'])
                                    np.savetxt(
                                        f'{folder}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}/delta_ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                                        delta_dict[f'delta_l_{probe_dav}'])

                    covmat_path = covariance_cfg["cov_folder"].format(zbins=zbins,
                                                                      triu_tril=covariance_cfg['triu_tril'],
                                                                      row_col_wise=covariance_cfg['row_col_wise'])
                    for ndim in (2, 4, 6):
                        if covariance_cfg[f'save_cov_{ndim}D']:

                            # save GO, GS or GO, GS and SS
                            which_cov_list = ['GO', 'GS']
                            if covariance_cfg[f'save_cov_SS']:
                                which_cov_list.append('SS')

                            # set probes to save; the ndim == 6 case is different
                            probe_list = ['WL', 'GC', '3x2pt', 'WA']
                            ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
                            nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]
                            # in this case, 3x2pt is saved in 10D as a dictionary
                            if ndim == 6:
                                probe_list = ['WL', 'GC', 'WA']
                                ellmax_list = [ell_max_WL, ell_max_GC, ell_max_WL]
                                nbl_list = [nbl_WL, nbl_GC, nbl_WA]

                            # save all covmats in the optimistic case
                            if ell_max_WL == 5000:

                                for which_cov in which_cov_list:
                                    for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                                        np.save(f'{covmat_path}/'
                                                f'covmat_{which_cov}_{probe}_lmax{ell_max}_nbl{nbl}_zbins{EP_or_ED}{zbins:02}_{ndim}D.npy',
                                                cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'])

                                    # in this case, 3x2pt is saved in 10D as a dictionary
                                    if ndim == 6:
                                        filename = f'{covmat_path}/covmat_{which_cov}_3x2pt_lmax{ell_max_XC}_nbl{nbl_3x2pt}_zbins{EP_or_ED}{zbins:02}_10D.pickle'
                                        with open(filename, 'wb') as handle:
                                            pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)

                            # in the pessimistic case, save only WA
                            elif ell_max_WL == 1500:
                                for which_cov in which_cov_list:
                                    np.save(
                                        f'{covmat_path}/covmat_{which_cov}_WA_lmax{ell_max_WL}_nbl{nbl_WA}_zbins{EP_or_ED}{zbins:02}_{ndim}D.npy',
                                        cov_dict[f'cov_WA_{which_cov}_{ndim}D'])

                    # save in .dat for Vincenzo, only in the optimistic case and in 2D
                    if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
                        path_vinc_fmt = f'{job_path}/output/covmat/vincenzos_format'
                        for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
                            for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):
                                np.savetxt(f'{path_vinc_fmt}/{GOGS_folder}/{probe_vinc}/cm-{probe_vinc}-{nbl_WL}'
                                           f'-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.dat',
                                           cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.10e')

                    # check for Stefano
                    if covariance_cfg['save_cov_6D']:
                        print('GHOST CODE BELOW')
                        npairs = (zbins * (zbins + 1)) // 2
                        cov_WL_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GO_6D'], nbl_WL, npairs, ind[:npairs, :])
                        cov_GC_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GO_6D'], nbl_GC, npairs, ind[:npairs, :])
                        cov_WL_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GS_6D'], nbl_WL, npairs, ind[:npairs, :])
                        cov_GC_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GS_6D'], nbl_GC, npairs, ind[:npairs, :])
                        assert np.array_equal(cov_WL_GO_4D, cov_dict[f'cov_WL_GO_4D'])
                        assert np.array_equal(cov_GC_GO_4D, cov_dict[f'cov_GC_GO_4D'])
                        assert np.allclose(cov_WL_GS_4D, cov_dict[f'cov_WL_GS_4D'], rtol=1e-9, atol=0)
                        assert np.allclose(cov_GC_GS_4D, cov_dict[f'cov_GC_GS_4D'], rtol=1e-9, atol=0)

                    if FM_cfg['save_FM']:
                        probe_list = ['WL', 'GC', '3x2pt', 'WA']
                        ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
                        nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]

                        for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                            for which_cov in which_cov_list:
                                np.savetxt(f'{FM_cfg["FM_output_folder"]}/'
                                           f'FM_{probe}_{which_cov}_lmax{ell_max}_nbl{nbl}_zbins{EP_or_ED}{zbins:02}.txt',
                                           FM_dict[f'FM_{probe}_{which_cov}'])

                    if FM_cfg['save_FM_as_dict']:
                        sio.savemat(job_path / f'output/FM/FM_dict.mat', FM_dict)

print('done')
