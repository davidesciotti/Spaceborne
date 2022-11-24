import bz2
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
import config_SPV3_magcut_zcut as cfg

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

    # for general_cfg['zbins'] in general_cfg['zbins_list']:
    # for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((5000, 3000),):
    # for general_cfg['EP_or_ED'] in ('ED',):
for general_cfg['magcut_lens'] in general_cfg['magcut_lens_list']:
    for general_cfg['magcut_source'] in general_cfg['magcut_source_list']:
        for general_cfg['zcut_lens'] in general_cfg['zcut_lens_list']:
            for general_cfg['zcut_source'] in general_cfg['zcut_source_list']:

                # utils.consistency_checks(general_cfg, covariance_cfg)

                # some variables used for I/O naming, just to make things more readable
                zbins = general_cfg['zbins']
                EP_or_ED = general_cfg['EP_or_ED']
                ell_max_WL = general_cfg['ell_max_WL']
                ell_max_GC = general_cfg['ell_max_GC']
                ell_max_XC = ell_max_GC
                nbl_WL_32 = general_cfg['nbl_WL_32']
                magcut_source = general_cfg['magcut_source']
                magcut_lens = general_cfg['magcut_lens']
                zcut_source = general_cfg['zcut_source']
                zcut_lens = general_cfg['zcut_lens']
                zmax = int(general_cfg['zmax'] * 10)
                triu_tril = covariance_cfg['triu_tril']
                row_col_wise = covariance_cfg['row_col_wise']
                n_probes = general_cfg['n_probes']
                use_stefano_BNT_ingredients = general_cfg['use_stefano_BNT_ingredients']

                # which cases to save: GO, GS or GO, GS and SS
                cases_tosave = ['GO', ]
                if covariance_cfg[f'save_cov_GS']:
                    cases_tosave.append('GS')
                if covariance_cfg[f'save_cov_SS']:
                    cases_tosave.append('SS')

                assert general_cfg['flagship_version'] == 2, 'The input files used in this job for flagship version 2!'

                # import the ind files and store it into the covariance dictionary
                ind_folder = covariance_cfg['ind_folder'].format(triu_tril=triu_tril,
                                                                 row_col_wise=row_col_wise)
                ind_filename = covariance_cfg['ind_filename'].format(triu_tril=triu_tril,
                                                                     row_col_wise=row_col_wise,
                                                                     zbins=zbins)
                ind = np.genfromtxt(f'{ind_folder}/{ind_filename}', dtype=int)
                covariance_cfg['ind'] = ind

                assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
                    'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

                # compute ell and delta ell values in the reference (optimistic) case
                ell_WL_nbl32, delta_l_WL_nbl32 = ell_utils.compute_ells(general_cfg['nbl_WL_32'],
                                                                        general_cfg['ell_min'],
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

                delta_dict = {}
                delta_dict['delta_l_WL'] = np.copy(delta_l_WL_nbl32[:nbl_WL])
                delta_dict['delta_l_GC'] = np.copy(delta_l_WL_nbl32[:nbl_GC])
                delta_dict['delta_l_WA'] = np.copy(delta_l_WL_nbl32[nbl_GC:])

                # set # of nbl in the opt case, import and reshape, then cut the reshaped datavectors in the pes case
                nbl_WL_opt = 32
                nbl_GC_opt = 29
                nbl_WA_opt = 3
                nbl_3x2pt_opt = 29

                if ell_max_WL == general_cfg['ell_max_WL_opt']:
                    assert (nbl_WL_opt, nbl_GC_opt, nbl_WA_opt, nbl_3x2pt_opt) == (nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt), \
                        'nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt don\'t match with the expected values for the optimistic case'

                # ! import datavectors (cl) and response functions (rl)
                # this is just to make the .format() more compact
                variable_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins,
                                  'magcut_lens': magcut_lens, 'zcut_lens': zcut_lens,
                                  'magcut_source': magcut_source, 'zcut_source': zcut_source,
                                  'zmax': zmax}

                cl_fld = general_cfg['cl_folder']
                cl_filename = general_cfg['cl_filename']
                cl_ll_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='WLO', **variable_specs)}")
                cl_gg_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='GCO', **variable_specs)}")
                cl_wa_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='WLA', **variable_specs)}")
                cl_3x2pt_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='3x2pt', **variable_specs)}")

                rl_fld = general_cfg['rl_folder']
                rl_filename = general_cfg['rl_filename']
                rl_ll_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLO', **variable_specs)}")
                rl_gg_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='GCO', **variable_specs)}")
                rl_wa_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLA', **variable_specs)}")
                rl_3x2pt_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='3x2pt', **variable_specs)}")

                # ! reshape to 3 dimensions
                cl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(cl_ll_1d, 'WL', nbl_WL_opt, zbins)
                cl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(cl_gg_1d, 'GC', nbl_GC_opt, zbins)
                cl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(cl_wa_1d, 'WA', nbl_WA_opt, zbins)
                cl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(cl_3x2pt_1d, '3x2pt', nbl_3x2pt_opt, zbins)

                rl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(rl_ll_1d, 'WL', nbl_WL_opt, zbins)
                rl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(rl_gg_1d, 'GC', nbl_GC_opt, zbins)
                rl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(rl_wa_1d, 'WA', nbl_WA_opt, zbins)
                rl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(rl_3x2pt_1d, '3x2pt', nbl_3x2pt_opt, zbins)

                # ! BNT transform if the corresponding flag is set to True
                if general_cfg['cl_BNT_transform']:
                    assert general_cfg['EP_or_ED'] == 'ED', 'cl_BNT_transform is only available for ED'
                    assert general_cfg['zbins'] == 13, 'cl_BNT_transform is only available for zbins=13'
                    BNT_matrix = np.genfromtxt(f'{general_cfg["BNT_matrix_path"]}/{general_cfg["BNT_matrix_filename"]}',
                                               delimiter=',')
                    cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, BNT_matrix)
                    cl_gg_3d = cl_utils.cl_BNT_transform(cl_gg_3d, BNT_matrix)
                    cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, BNT_matrix)
                    cl_3x2pt_5d = cl_utils.cl_BNT_transform(cl_3x2pt_5d, BNT_matrix)
                    print('you shuld BNT transform the responses do this with the responses too!')

                # check that cl_wa is equal to cl_ll in the last nbl_WA_opt bins
                if ell_max_WL == general_cfg['ell_max_WL_opt']:
                    if not np.array_equal(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :]):
                        rtol = 1e-5
                        # plt.plot(ell_dict['ell_WL'], cl_ll_3d[:, 0, 0])
                        # plt.plot(ell_dict['ell_WL'][nbl_GC:nbl_WL], cl_wa_3d[:, 0, 0])
                        assert (np.allclose(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :], rtol=rtol, atol=0)), \
                            'cl_wa_3d should be obtainable from cl_ll_3d!'
                        print(f'cl_wa_3d and cl_ll_3d[nbl_GC:nbl_WL, :, :] are not exactly equal, but have a relative '
                              f'difference of less than {rtol}')

                # cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
                if ell_max_WL == 1500:
                    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
                    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
                    cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
                    cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

                    rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
                    rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
                    rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
                    rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

                # store cls and responses in a dictionary
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

                warnings.warn("not the nicest way, this if is ugly")
                if covariance_cfg['compute_covmat']:
                    # ! compute or load Sijkl
                    # ! load kernels
                    # TODO this should not be done if Sijkl is loaded; I have a problem with nz, which is part of the file name...
                    WF_folder = Sijkl_cfg["wf_input_folder"].format(**variable_specs)
                    WF_filename = Sijkl_cfg["wf_input_filename"]
                    wil = np.genfromtxt(f'{WF_folder}/{WF_filename.format(which_WF="WiWL", **variable_specs)}')
                    wig = np.genfromtxt(f'{WF_folder}/{WF_filename.format(which_WF="WiGC", **variable_specs)}')

                    # preprocess (remove redshift column)
                    z_arr, wil = Sijkl_utils.preprocess_wf(wil, zbins)
                    z_arr_2, wig = Sijkl_utils.preprocess_wf(wig, zbins)
                    assert np.array_equal(z_arr, z_arr_2), 'the redshift arrays are different for the GC and WL kernels'

                    # transpose and stack, ordering is important here!
                    transp_stacked_wf = np.vstack((wil.T, wig.T))

                    nz = z_arr.shape[0]  # get number of z points in nz to name the Sijkl file
                    Sijkl_folder = Sijkl_cfg['Sijkl_folder']
                    Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(
                        flagship_version=general_cfg['flagship_version'],
                        nz=nz, EP_or_ED=EP_or_ED, zbins=zbins,
                        IA_flag=Sijkl_cfg['IA_flag'],
                        magcut_source=magcut_source,
                        zcut_source=zcut_source)
                    # if Sijkl exists, load it; otherwise, compute it and save it
                    if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):
                        print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
                        Sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')
                    else:
                        Sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                                          Sijkl_cfg['WF_normalization'])
                        np.save(f'{Sijkl_folder}/{Sijkl_filename}', Sijkl)

                # ! compute covariance matrix
                if covariance_cfg['compute_covmat']:
                    ng_filename = f'{covariance_cfg["ng_filename"].format(**variable_specs)}'
                    # ! first column, in this case 👇
                    covariance_cfg['ng'] = np.genfromtxt(f'{covariance_cfg["ng_folder"]}/'f'{ng_filename}')[:, 1]
                    cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                                        ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl)

                    # now overwrite the WL GS entries with Stefano's BNT covmats:
                    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_pairs(zbins)

                    GOGS_list = ['GO', 'GS']
                    probe_list = ['WL', 'GC', '3x2pt', 'WA']
                    ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
                    nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]
                    zpairs_list = [zpairs_auto, zpairs_auto, zpairs_3x2pt, zpairs_auto]

                    cov_folder_stef = f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022' \
                                      f'/Flagship_2/CovMats/BNT_True/produced_by_stefano/magcut_zcut'

                    if use_stefano_BNT_ingredients:
                        for probe, ellmax, nbl, zpairs in zip(probe_list, ellmax_list, nbl_list, zpairs_list):
                            for GO_or_GS in GOGS_list:
                                cov_filename = f'BNT_covmat_{GO_or_GS}_{probe}_lmax{ellmax}_nbl{nbl}_' \
                                               f'zbins{EP_or_ED}{zbins:02d}_ML{magcut_lens:03d}_' \
                                               f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                               f'ZS{zcut_source:02d}_6D.npy'
                                if os.path.isfile(f'{cov_folder_stef}/{cov_filename}'):
                                    # ! this will give an error, Stefano's covariance is not in 6D but in 3D!
                                    print(
                                        f'cov_{probe}_{GO_or_GS} already exists in folder\n{cov_folder_stef}; loading it'
                                        f'\np.s.: this print should appear for all the files in the folder!')
                                    cov_dict[f'cov_{probe}_{GO_or_GS}_6D'] = np.load(
                                        f'{cov_folder_stef}/{cov_filename}')
                                    cov_dict[f'cov_{probe}_{GO_or_GS}_4D'] = mm.cov_6D_to_4D(
                                        cov_dict[f'cov_{probe}_{GO_or_GS}_6D'], nbl, zpairs, ind[:zpairs, :])
                                    cov_dict[f'cov_{probe}_{GO_or_GS}_2D'] = mm.cov_4D_to_2D(
                                        cov_dict[f'cov_{probe}_{GO_or_GS}_4D'], 'ell')

                # ! compute Fisher matrix
                if FM_cfg['compute_FM']:

                    # import derivatives and store them in dictionary
                    derivetives_folder_stef = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_2/Derivatives/BNT_True'
                    derivatives_folder = FM_cfg['derivatives_folder'].format(magcut_lens=magcut_lens,
                                                                             zcut_lens=zcut_lens,
                                                                             magcut_source=magcut_source,
                                                                             zcut_source=zcut_source)
                    dC_dict_1D = dict(mm.get_kv_pairs(derivatives_folder, "dat"))
                    dC_dict_BNT_WLO_1D = dict(mm.get_kv_pairs(derivetives_folder_stef, "dat"))

                    # reshape them (no interpolation needed in this case)
                    dC_dict_LL_3D = {}
                    dC_dict_GG_3D = {}
                    dC_dict_WA_3D = {}
                    dC_dict_3x2pt_5D = {}

                    if use_stefano_BNT_ingredients:
                        for key in dC_dict_BNT_WLO_1D.keys():
                            if 'WLO' in key:
                                dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_BNT_WLO_1D[key], probe='WL',
                                                                               nbl=nbl_WL, zbins=zbins)

                    elif not use_stefano_BNT_ingredients:
                        for key in dC_dict_1D.keys():
                            if 'WLO' in key:
                                dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], probe='WL', nbl=nbl_WL,
                                                                               zbins=zbins)
                            if 'GCO' in key:
                                dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], probe='GC', nbl=nbl_GC,
                                                                               zbins=zbins)
                            elif 'WLA' in key:
                                dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], probe='WA', nbl=nbl_WA,
                                                                               zbins=zbins)
                            elif '3x2pt' in key:
                                dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], probe='3x2pt',
                                                                                  nbl=nbl_3x2pt, zbins=zbins)

                    # now turn the dict. into npy array
                    paramnames_cosmo = ["Om", "Ob", "wz", "wa", "h", "ns", "s8"]
                    paramnames_IA = ["Aia", "eIA", "bIA"]
                    paramnames_galbias = [f'b{zbin_idx:02d}' for zbin_idx in range(zbins)]
                    paramnames_shearbias = [f'b{zbin_idx:02d}' for zbin_idx in range(zbins)]
                    paramnames_dz = [f'b{zbin_idx:02d}' for zbin_idx in range(zbins)]
                    paramnames_3x2pt = paramnames_cosmo + paramnames_IA + paramnames_galbias + paramnames_shearbias + paramnames_dz
                    FM_cfg['paramnames_3x2pt'] = paramnames_3x2pt  # save them to pass to FM_utils module

                    dC_LL_4D = dC_dict_to_4D_array(paramnames_3x2pt, dC_dict_LL_3D, nbl_WL, zbins)
                    dC_GG_4D = dC_dict_to_4D_array(paramnames_3x2pt, dC_dict_GG_3D, nbl_GC, zbins)
                    dC_WA_4D = dC_dict_to_4D_array(paramnames_3x2pt, dC_dict_WA_3D, nbl_WA, zbins)
                    dC_3x2pt_5D = dC_dict_to_4D_array(paramnames_3x2pt, dC_dict_3x2pt_5D, nbl_3x2pt, zbins,
                                                      is_3x2pt=True)

                    # store the derivatives in a dictionary (of dictionaries)
                    deriv_dict = {'dC_LL_4D': dC_LL_4D,
                                  'dC_GG_4D': dC_GG_4D,
                                  'dC_WA_4D': dC_WA_4D,
                                  'dC_3x2pt_5D': dC_3x2pt_5D}
                    # TODO save 3D derivatives to file

                    FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict)

                    # ! save cls and responses:
                    # this is just to set the correct probe names
                    probe_dav_dict = {'WL': 'LL_WLonly_3D',
                                      'GC': 'GG_3D',
                                      'WA': 'WA_3D',
                                      '3x2pt': '3x2pt_5D'}

                    # just a dict for the output file names
                    clrl_dict = {'cl_dict_3D': cl_dict_3D,
                                 'rl_dict_3D': rl_dict_3D,
                                 'cl_inputname': 'dv',
                                 'rl_inputname': 'rf',
                                 'cl_dict_key': 'C',
                                 'rl_dict_key': 'R'}
                    for cl_or_rl in ['cl', 'rl']:
                        if general_cfg[f'save_{cl_or_rl}s_3d']:

                            for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'],
                                                             ['WL', 'GC', '3x2pt', 'WA']):
                                # save cl and/or response; not very readable but it works, plus all the cases are in the for loop

                                filepath = f'{general_cfg[f"{cl_or_rl}_folder"]}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}'
                                filename = general_cfg[f'{cl_or_rl}_filename'].format(probe=probe_vinc, **variable_specs
                                                                                      ).replace(".dat", "_3D.npy")
                                file = clrl_dict[f"{cl_or_rl}_dict_3D"][
                                    f'{clrl_dict[f"{cl_or_rl}_dict_key"]}_{probe_dav_dict[probe_dav]}']
                                np.save(f'{filepath}/{filename}', file)

                                # save ells and deltas
                                if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                                    filepath = f'{general_cfg[f"{cl_or_rl}_folder"]}/' \
                                               f'3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}'
                                    ells_filename = f'ell_{probe_dav}_ellmaxWL{ell_max_WL}'
                                    np.savetxt(f'{filepath}/{ells_filename}.txt', 10 ** ell_dict[f'ell_{probe_dav}'])
                                    np.savetxt(f'{filepath}/delta_{ells_filename}.txt',
                                               delta_dict[f'delta_l_{probe_dav}'])

                # ! save covariance:
                if covariance_cfg['cov_file_format'] == 'npy':
                    save_funct = np.save
                    extension = 'npy'
                elif covariance_cfg['cov_file_format'] == 'npz':
                    save_funct = np.savez_compressed
                    extension = 'npz'
                else:
                    raise ValueError('cov_file_format not recognized: must be "npy" or "npz"')

                # TODO skip the computation and saving if the file already exists
                cov_folder = covariance_cfg["cov_folder"].format(zbins=zbins)
                for ndim in (2, 4, 6):
                    if covariance_cfg[f'save_cov_{ndim}D']:

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

                            for which_cov in cases_tosave:
                                for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                                    cov_filename = f'covmat_{which_cov}_{probe}_lmax{ell_max}_nbl{nbl}_' \
                                                   f'zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_' \
                                                   f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                                   f'ZS{zcut_source:02d}_{ndim}D.{extension}'
                                    save_funct(f'{cov_folder}/{cov_filename}',
                                               cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'])  # save in .npy or .npz

                                # in this case, 3x2pt is saved in 10D as a dictionary
                                if ndim == 6:
                                    cov_filename = f'covmat_{which_cov}_3x2pt_lmax{ell_max_XC}_' \
                                                   f'nbl{nbl_3x2pt}_zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_' \
                                                   f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                                   f'ZS{zcut_source:02d}_10D'
                                    start = time.perf_counter()
                                    with open(f'{cov_folder}/{cov_filename}', 'wb') as handle:
                                        pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)
                                    print(f'covmat 3x2pt {which_cov} saved in {time.perf_counter() - start:.2f} s')

                        # in the pessimistic case, save only WA
                        elif ell_max_WL == 1500:
                            for which_cov in cases_tosave:
                                cov_filename = f'covmat_{which_cov}_WA_lmax{ell_max_WL}_nbl{nbl_WA}_' \
                                               f'zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_' \
                                               f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                               f'ZS{zcut_source:02d}_{ndim}D.npy'
                                np.save(f'{cov_folder}/{cov_filename}', cov_dict[f'cov_WA_{which_cov}_{ndim}D'])

                # save in .dat for Vincenzo, only in the optimistic case and in 2D
                if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
                    raise NotImplementedError('not implemented yet')
                    path_vinc_fmt = f'{job_path}/output/covmat/vincenzos_format'
                    for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
                        for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):
                            np.savetxt(f'{path_vinc_fmt}/{GOGS_folder}/{probe_vinc}/cm-{probe_vinc}-{nbl_WL}'
                                       f'-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.dat',
                                       cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.10e')

                # check for Stefano
                if covariance_cfg['save_cov_6D']:
                    warnings.warn('old checks below, you could probably discard...')
                    npairs = (zbins * (zbins + 1)) // 2
                    cov_WL_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GO_6D'], nbl_WL, npairs, ind[:npairs, :])
                    cov_GC_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GO_6D'], nbl_GC, npairs, ind[:npairs, :])
                    cov_WL_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GS_6D'], nbl_WL, npairs, ind[:npairs, :])
                    cov_GC_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GS_6D'], nbl_GC, npairs, ind[:npairs, :])
                    assert np.array_equal(cov_WL_GO_4D, cov_dict[f'cov_WL_GO_4D'])
                    assert np.array_equal(cov_GC_GO_4D, cov_dict[f'cov_GC_GO_4D'])
                    assert np.allclose(cov_WL_GS_4D, cov_dict[f'cov_WL_GS_4D'], rtol=1e-9, atol=0)
                    assert np.allclose(cov_GC_GS_4D, cov_dict[f'cov_GC_GS_4D'], rtol=1e-9, atol=0)

                # ! save FM
                if FM_cfg['save_FM']:
                    probe_list = ['WL', 'GC', '3x2pt', 'WA']
                    ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
                    nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]
                    header = 'parameters\' ordering:' + str(paramnames_3x2pt)

                    for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                        for which_cov in cases_tosave:
                            FM_folder = FM_cfg["FM_folder"]
                            FM_filename = FM_cfg["FM_filename"].format(probe=probe, which_cov=which_cov,
                                                                       ell_max=ell_max, nbl=nbl,
                                                                       **variable_specs)
                            np.savetxt(f'{FM_folder}/{FM_filename}', FM_dict[f'FM_{probe}_{which_cov}'], header=header)

                # if FM_cfg['save_FM_as_dict']:
                #     sio.savemat(job_path / f'output/FM/FM_dict.mat', FM_dict)

print('done')
