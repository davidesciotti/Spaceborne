import bz2
import glob
import pickle
import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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
import ISTF_fid_params as ISTF_fid

# job configuration
sys.path.append(f'{job_path}/config')
import config_SPV3_magcut_zcut as cfg

# project libraries
sys.path.append(f'{project_path}/bin')
import ell_values as ell_utils
import cl_preprocessing as cl_utils
import compute_Sijkl as Sijkl_utils
import covariance as covmat_utils
import fisher_matrix as FM_utils
import utils_running as utils
import unit_test as ut

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


def load_build_3x2pt_BNT_cov_dict_stef(cov_BNTstef_folder, probe_ordering, variable_specs, GO_or_GS, cov_dict,
                                       nbl_3x2pt):
    """transforms dictionary of 3x2pt cov blocks into a dictionary of the usual form (i.e., with the usual keys)"""

    cov_3x2pt_BNT_imported_dict = dict(mm.get_kv_pairs(cov_BNTstef_folder))

    # select only the ones corresponding to the current configuration of MS, ML, ZS, ZL values
    str_start = f'BNT_covmat_{GO_or_GS}_3x2pt_'
    str_end = '_lmax3000_nbl29_zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_ZL{zcut_lens:02}' \
              '_MS{magcut_source:03d}_ZS{zcut_source:02}_6D'.format(**variable_specs)
    cov_3x2pt_BNT_imported_dict = {key: value
                                   for key, value in cov_3x2pt_BNT_imported_dict.items()
                                   if key.startswith(str_start) and key.endswith(str_end)}
    if not cov_3x2pt_BNT_imported_dict:
        raise ValueError('cov_3x2pt_BNT_imported_dict is empty')

    # second cleanup - I could include this in the previous comprehension, but it's more readable this way
    cov_3x2pt_BNT_imported_dict = {key.replace(str_start, '').replace(str_end, ''): value
                                   for key, value in cov_3x2pt_BNT_imported_dict.items()}

    # initialize the keys of the new dictionary to 0
    zbins = variable_specs['zbins']
    cov_3x2pt_BNT_dict = {}
    for probe_A, probe_B in probe_ordering:
        for probe_C, probe_D in probe_ordering:
            try:
                cov_3x2pt_BNT_dict[probe_A, probe_B, probe_C, probe_D] = cov_3x2pt_BNT_imported_dict[
                    probe_A + probe_B + probe_C + probe_D]
            except KeyError:  # if the key is not present, the transposed keys probably are
                print(probe_A + probe_B + probe_C + probe_D, 'not present in cov_3x2pt_BNT_imported_dict')
                try:
                    cov_3x2pt_BNT_dict[probe_A, probe_B, probe_C, probe_D] = cov_3x2pt_BNT_imported_dict[
                        probe_C + probe_D + probe_A + probe_B].transpose((0, 1, 4, 5, 2, 3))
                except KeyError:  # the GGGG key is not present (no BNT transform in this case)
                    cov_3x2pt_BNT_dict[probe_A, probe_B, probe_C, probe_D] = \
                        np.zeros((nbl_3x2pt, nbl_3x2pt, zbins, zbins, zbins, zbins))

    if not cov_3x2pt_BNT_dict:
        raise ValueError('cov_3x2pt_BNT_dict is empty')

    cov_3x2pt_BNT_dict['G', 'G', 'G', 'G'] = cov_dict[f'cov_3x2pt_{GO_or_GS}_10D']['G', 'G', 'G', 'G']

    return cov_3x2pt_BNT_dict


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

ML_list = [230, 230, 245, 245]
ZL_list = [0, 2, 0, 2]
MS_list = [245, 245, 245, 245]
ZS_list = [0, 0, 0, 2]

ML_list = [245]
ZL_list = [0]
MS_list = [245]
ZS_list = [0]

warnings.warn('restore the ML, Zl, ... lists')
warnings.warn('restore nbl_WL = 32, or call it nbl_WL_opt instead of nbl_WL_32...')

for general_cfg['magcut_lens'], general_cfg['zcut_lens'], \
        general_cfg['magcut_source'], general_cfg['zcut_source'] in \
        zip(ML_list, ZL_list, MS_list, ZS_list):

    # for general_cfg['magcut_lens'] in general_cfg['magcut_lens_list']:
    #     for general_cfg['magcut_source'] in general_cfg['magcut_source_list']:
    #         for general_cfg['zcut_lens'] in general_cfg['zcut_lens_list']:
    #             for general_cfg['zcut_source'] in general_cfg['zcut_source_list']:

    # utils.consistency_checks(general_cfg, covariance_cfg)

    # some variables used for I/O naming, just to make things more readable
    zbins = general_cfg['zbins']
    EP_or_ED = general_cfg['EP_or_ED']
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    ell_max_XC = ell_max_GC
    magcut_source = general_cfg['magcut_source']
    magcut_lens = general_cfg['magcut_lens']
    zcut_source = general_cfg['zcut_source']
    zcut_lens = general_cfg['zcut_lens']
    zmax = int(general_cfg['zmax'] * 10)
    triu_tril = covariance_cfg['triu_tril']
    row_col_major = covariance_cfg['row_col_major']
    n_probes = general_cfg['n_probes']
    whos_BNT = general_cfg['whos_BNT']
    GL_or_LG = covariance_cfg['GL_or_LG']
    probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]

    # some checks
    assert general_cfg['flagship_version'] == 2, 'The input files used in this job for flagship version 2!'
    assert general_cfg['use_WA'] is False, 'We do not use Wadd for SPV3 at the moment'

    if general_cfg['BNT_transform']:
        assert general_cfg['EP_or_ED'] == 'ED', 'BNT matrices are only available for ED case'
        assert general_cfg['zbins'] == 13, 'BNT matrices are only available for zbins=13'

    # which cases to save: GO, GS or GO, GS and SS
    cases_tosave = ['GO', 'GS']
    if covariance_cfg[f'save_cov_GS']:
        cases_tosave.append('GS')
    if covariance_cfg[f'save_cov_SS']:
        cases_tosave.append('SS')

    # import the ind files and store it into the covariance dictionary
    ind_folder = covariance_cfg['ind_folder'].format(triu_tril=triu_tril,
                                                     row_col_major=row_col_major)
    ind_filename = covariance_cfg['ind_filename'].format(triu_tril=triu_tril,
                                                         row_col_major=row_col_major,
                                                         zbins=zbins)
    ind = np.genfromtxt(f'{ind_folder}/{ind_filename}', dtype=int)
    covariance_cfg['ind'] = ind

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    # convenience vectors
    ind_auto = ind[:zpairs_auto, :].copy()
    # ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()

    assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
        'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

    # compute ell and delta ell values in the reference (optimistic) case
    ell_WL_nbl32, delta_l_WL_nbl32 = ell_utils.compute_ells(general_cfg['nbl_WL_opt'],
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

    # this is just to make the .format() more compact
    variable_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins,
                      'magcut_lens': magcut_lens, 'zcut_lens': zcut_lens,
                      'magcut_source': magcut_source, 'zcut_source': zcut_source,
                      'zmax': zmax}
    BNT_matrix_filename = general_cfg["BNT_matrix_filename"].format(**variable_specs)
    BNT_matrix = np.load(f'{general_cfg["BNT_matrix_path"]}/{BNT_matrix_filename}')

    # ! import datavectors (cl) and response functions (rl)
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

    if general_cfg['cl_BNT_transform']:
        assert general_cfg['cov_BNT_transform'] is False, 'the BNT transform should be applied either to the Cls ' \
                                                          'or to the covariance'

        cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, BNT_matrix, 'L', 'L')
        cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, BNT_matrix, 'L', 'L')
        cl_3x2pt_5d = cl_utils.cl_BNT_transform_3x2pt(cl_3x2pt_5d, BNT_matrix)
        warnings.warn('you should probebly BNT-transform the responses too!')

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
        'cl_LL_3D': cl_ll_3d,
        'cl_GG_3D': cl_gg_3d,
        'cl_WA_3D': cl_wa_3d,
        'cl_3x2pt_5D': cl_3x2pt_5d}

    rl_dict_3D = {
        'rl_LL_3D': rl_ll_3d,
        'rl_GG_3D': rl_gg_3d,
        'rl_WA_3D': rl_wa_3d,
        'rl_3x2pt_5D': rl_3x2pt_5d}

    if covariance_cfg['compute_covmat']:
        # ! load kernels
        # TODO this should not be done if Sijkl is loaded; I have a problem with nz, which is part of the file name...
        wf_folder = Sijkl_cfg["wf_input_folder"].format(**variable_specs)
        wf_WL_filename = Sijkl_cfg["wf_WL_input_filename"]
        wf_GC_filename = Sijkl_cfg["wf_GC_input_filename"]
        wil = np.genfromtxt(f'{wf_folder}/{wf_WL_filename.format(**variable_specs)}')
        wig = np.genfromtxt(f'{wf_folder}/{wf_GC_filename.format(**variable_specs)}')

        # preprocess (remove redshift column)
        z_arr, wil = Sijkl_utils.preprocess_wf(wil, zbins)
        z_arr_2, wig = Sijkl_utils.preprocess_wf(wig, zbins)
        assert np.array_equal(z_arr, z_arr_2), 'the redshift arrays are different for the GC and WL kernels'

        # transpose and stack, ordering is important here!
        transp_stacked_wf = np.vstack((wil.T, wig.T))

        # ! compute or load Sijkl
        nz = z_arr.shape[0]  # get number of z points in nz to name the Sijkl file
        Sijkl_folder = Sijkl_cfg['Sijkl_folder']
        warnings.warn('Sijkl_folder is set to BNT_False in all cases, so as not to have to recompute the Sijkl matrix'
                      'in the BNT_True case - for which I use Stefanos files')
        Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(flagship_version=general_cfg['flagship_version'],
                                                            nz=nz, IA_flag=Sijkl_cfg['IA_flag'], **variable_specs)
        # if Sijkl exists, load it; otherwise, compute it and save it
        if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):
            print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
            Sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')
        else:
            Sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                              Sijkl_cfg['WF_normalization'])
            np.save(f'{Sijkl_folder}/{Sijkl_filename}', Sijkl)

        # ! compute covariance matrix
        # TODO: if already existing, don't compute the covmat, like done above for Sijkl
        ng_folder = covariance_cfg["ng_folder"]
        ng_filename = f'{covariance_cfg["ng_filename"].format(**variable_specs)}'
        # the ng values are in the second column, for these input files 👇
        covariance_cfg['ng'] = np.genfromtxt(f'{ng_folder}/'f'{ng_filename}')[:, 1]
        cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                            ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl)

        if general_cfg['cov_BNT_transform']:
            assert general_cfg['cl_BNT_transform'] is False, 'the BNT transform should be applied either to the Cls ' \
                                                              'or to the covariance'
            assert general_cfg['deriv_BNT_transform'], 'you should BNT transform the derivatives as well'

            if whos_BNT == '/davide':

                X_dict = covmat_utils.build_X_matrix_BNT(BNT_matrix)

                cov_WL_GO_BNT_6D = covmat_utils.cov_BNT_transform(cov_dict['cov_WL_GO_6D'], X_dict, 'L', 'L')
                cov_WA_GO_BNT_6D = covmat_utils.cov_BNT_transform(cov_dict['cov_WA_GO_6D'], X_dict, 'L', 'L')
                cov_3x2pt_GO_BNT_dict = covmat_utils.cov_3x2pt_BNT_transform(cov_dict['cov_3x2pt_GO_10D'], X_dict)

                cov_WL_GS_BNT_6D = covmat_utils.cov_BNT_transform(cov_dict['cov_WL_GS_6D'], X_dict, 'L', 'L')
                cov_WA_GS_BNT_6D = covmat_utils.cov_BNT_transform(cov_dict['cov_WA_GS_6D'], X_dict, 'L', 'L')
                cov_3x2pt_GS_BNT_dict = covmat_utils.cov_3x2pt_BNT_transform(cov_dict['cov_3x2pt_GS_10D'], X_dict)

                # reshape to 4D
                cov_WL_GO_BNT_4D = mm.cov_6D_to_4D(cov_WL_GO_BNT_6D, nbl_WL, zpairs_auto, ind_auto)
                cov_WL_GS_BNT_4D = mm.cov_6D_to_4D(cov_WL_GS_BNT_6D, nbl_WL, zpairs_auto, ind_auto)
                cov_WA_GO_BNT_4D = mm.cov_6D_to_4D(cov_WA_GO_BNT_6D, nbl_WA, zpairs_auto, ind_auto)
                cov_WA_GS_BNT_4D = mm.cov_6D_to_4D(cov_WA_GS_BNT_6D, nbl_WA, zpairs_auto, ind_auto)

                # reshape to 2D
                cov_WL_GO_BNT_2D = mm.cov_4D_to_2D(cov_WL_GO_BNT_4D, block_index=covariance_cfg['block_index'])
                cov_WL_GS_BNT_2D = mm.cov_4D_to_2D(cov_WL_GS_BNT_4D, block_index=covariance_cfg['block_index'])
                cov_WA_GO_BNT_2D = mm.cov_4D_to_2D(cov_WA_GO_BNT_4D, block_index=covariance_cfg['block_index'])
                cov_WA_GS_BNT_2D = mm.cov_4D_to_2D(cov_WA_GS_BNT_4D, block_index=covariance_cfg['block_index'])

            elif whos_BNT == '/stefano':
                warnings.warn('deprecate use of stefanos input files')

                cov_BNTstef_folder_GO = covariance_cfg['cov_BNTstef_folder'].format(GO_or_GS='GO', probe='3x2pt')
                cov_BNTstef_folder_GS = covariance_cfg['cov_BNTstef_folder'].format(GO_or_GS='GS', probe='3x2pt')

                cov_3x2pt_GO_BNT_dict = load_build_3x2pt_BNT_cov_dict_stef(cov_BNTstef_folder_GO, probe_ordering,
                                                                           variable_specs, 'GO', cov_dict, nbl_3x2pt)
                cov_3x2pt_GS_BNT_dict = load_build_3x2pt_BNT_cov_dict_stef(cov_BNTstef_folder_GS, probe_ordering,
                                                                           variable_specs, 'GS', cov_dict, nbl_3x2pt)

            # transform from dict of 6D arrays to single 4D array
            cov_3x2pt_GO_BNT_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_GO_BNT_dict, probe_ordering, nbl_3x2pt,
                                                              zbins, ind.copy(), GL_or_LG)
            cov_3x2pt_GS_BNT_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_GS_BNT_dict, probe_ordering, nbl_3x2pt,
                                                              zbins, ind.copy(), GL_or_LG)

            # reshape to 2D
            cov_3x2pt_GO_BNT_2D = mm.cov_4D_to_2D(cov_3x2pt_GO_BNT_4D, block_index=covariance_cfg['block_index'])
            cov_3x2pt_GS_BNT_2D = mm.cov_4D_to_2D(cov_3x2pt_GS_BNT_4D, block_index=covariance_cfg['block_index'])

            # overwrite the non-BNT value
            cov_dict['cov_WL_GO_2D'] = cov_WL_GO_BNT_2D
            cov_dict['cov_WL_GS_2D'] = cov_WL_GS_BNT_2D
            cov_dict['cov_WA_GO_2D'] = cov_WA_GO_BNT_2D
            cov_dict['cov_WA_GS_2D'] = cov_WA_GS_BNT_2D
            cov_dict['cov_3x2pt_GO_2D'] = cov_3x2pt_GO_BNT_2D
            cov_dict['cov_3x2pt_GS_2D'] = cov_3x2pt_GS_BNT_2D

    # ! compute Fisher matrix
    if FM_cfg['compute_FM']:

        # declare the set of parameters under study
        paramnames_cosmo = ["Om", "Ox", "Ob", "wz", "wa", "h", "ns", "s8"]
        paramnames_IA = ["Aia", "eIA", "bIA"]
        paramnames_galbias = [f'bG{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_shearbias = [f'm{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_dzWL = [f'dzWL{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_dzGC = [f'dzGC{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]
        paramnames_3x2pt = paramnames_cosmo + paramnames_IA + paramnames_galbias + paramnames_shearbias + \
                           paramnames_dzWL + paramnames_dzGC
        FM_cfg['paramnames_3x2pt'] = paramnames_3x2pt  # save them to pass to FM_utils module

        # fiducial values
        fid_cosmo = [0.32, 0.68, 0.05, -1.0, 0.0, 0.67, 0.96, 0.816]  # TODO import from ISTfid
        fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
        fid_galaxy_bias = np.genfromtxt(f'{ng_folder}/{ng_filename}')[:, 2]
        fid_shear_bias = np.zeros((zbins,))
        fid_dzWL = np.zeros((zbins,))
        fid_dzGC = np.zeros((zbins,))
        fid_3x2pt = np.concatenate((fid_cosmo, fid_IA, fid_galaxy_bias, fid_shear_bias, fid_dzWL, fid_dzGC))
        assert len(fid_3x2pt) == len(
            paramnames_3x2pt), 'the fiducial values list and parameter names should have the same length'

        # import derivatives and store them in one big dictionary
        derivatives_folder = FM_cfg['derivatives_folder'].format(**variable_specs)
        der_prefix = FM_cfg['derivatives_prefix']
        dC_dict_1D = dict(mm.get_kv_pairs(derivatives_folder, "dat"))
        # check if dictionary is empty
        if not dC_dict_1D:
            raise ValueError(f'No derivatives found in folder {derivatives_folder}')

        # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
        dC_dict_LL_3D = {}
        dC_dict_GG_3D = {}
        dC_dict_WA_3D = {}
        dC_dict_3x2pt_5D = {}
        for key in dC_dict_1D.keys():
            if 'WLO' in key:
                dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WL', nbl=nbl_WL, zbins=zbins)
            elif 'GCO' in key:
                dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'GC', nbl=nbl_GC, zbins=zbins)
            elif 'WLA' in key:
                dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WA', nbl=nbl_WA, zbins=zbins)
            elif '3x2pt' in key:
                dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], '3x2pt', nbl=nbl_3x2pt, zbins=zbins)

        # ! BNT transform stefano
        # in this case, overwrite part of the dictionary entries (the 3x2pt, in particular)
        if general_cfg['deriv_BNT_transform'] and whos_BNT == '/stefano':
            warnings.warn('deprecate use of stefanos input files')

            # import in one big dictionary
            derivatives_BNTstef_folder = FM_cfg['derivatives_BNTstef_folder'].format(probe='3x2pt')
            dC_dict_3x2pt_BNT_1D = dict(mm.get_kv_pairs(derivatives_BNTstef_folder, "dat"))

            # select only items witht he correct magnitude and redshift cuts
            cuts_specs_str = f'ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}'
            dC_dict_3x2pt_BNT_1D = {key: dC_dict_3x2pt_BNT_1D[key] for key in dC_dict_3x2pt_BNT_1D.keys()
                                    if cuts_specs_str in key}

            # check that the dict is not empty
            if not dC_dict_3x2pt_BNT_1D:
                raise ValueError(f'No derivatives found in folder {derivatives_BNTstef_folder}')

            # separate in 4 different dictionaries, reshape and remove the probe-specific suffix in the keys:
            # in this way, referring both to the 3x2pt, dC_dict_3x2pt_BNT_LL_3D and dC_dict_3x2pt_BNT_LG_3D will have
            # the same keys, and the same for the other probes
            dC_dict_3x2pt_BNT_LL_3D = {}
            dC_dict_3x2pt_BNT_LG_3D = {}
            for key in dC_dict_3x2pt_BNT_1D.keys():
                if '3x2pt_LL_' in key:
                    dC_dict_3x2pt_BNT_LL_3D[key.replace('_LL_', '')] = cl_utils.cl_SPV3_1D_to_3D(
                        dC_dict_3x2pt_BNT_1D[key], probe='WL', nbl=nbl_3x2pt, zbins=zbins)
                elif '3x2pt_LG_' in key:
                    dC_dict_3x2pt_BNT_LG_3D[key.replace('_LG_', '')] = cl_utils.cl_SPV3_1D_to_3D(
                        dC_dict_3x2pt_BNT_1D[key], probe='XC', nbl=nbl_3x2pt, zbins=zbins)

            # a check on the keys, they now must be the same
            assert dC_dict_3x2pt_BNT_LL_3D.keys() == dC_dict_3x2pt_BNT_LG_3D.keys(), \
                'The keys of the dictionaries are not the same'

            # now finish building the derivatives 5D vector with non-BNT derivatives:
            # instantiate a dict of 5D numpy arrays
            dC_dict_3x2pt_BNT_5D = {}
            allkeys = {key for key in dC_dict_3x2pt_BNT_LL_3D}
            for key in allkeys:
                dC_dict_3x2pt_BNT_5D[key] = np.zeros((nbl_3x2pt, n_probes, n_probes, zbins, zbins))

            # fill it with the various 3D arrays in the different dictionaries
            for key in dC_dict_3x2pt_BNT_5D.keys():
                dC_dict_3x2pt_BNT_5D[key][:, 0, 0, :, :] = dC_dict_3x2pt_BNT_LL_3D[key]
                dC_dict_3x2pt_BNT_5D[key][:, 0, 1, :, :] = dC_dict_3x2pt_BNT_LG_3D[key]
                dC_dict_3x2pt_BNT_5D[key][:, 1, 0, :, :] = dC_dict_3x2pt_BNT_LG_3D[key].transpose(0, 2, 1)
                # for GG, I use Vincenzo's (i.e., non-BNT) derivatives, picking the keys of dC_dict_3x2pt_BNT_5D (which
                # are a subset of the keys of dC_dict_3x2pt_5D)
                dC_dict_3x2pt_BNT_5D[key][:, 1, 1, :, :] = dC_dict_3x2pt_5D[key.lstrip('BNT_')][:, 1, 1, :, :]

            # overwrite the non-BNT derivatives with the BNT ones
            dC_dict_3x2pt_5D = dC_dict_3x2pt_BNT_5D

        # turn the dictionaries of derivatives into npy array
        dC_LL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LL_3D, paramnames_3x2pt, nbl_WL, zbins, der_prefix)
        dC_GG_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GG_3D, paramnames_3x2pt, nbl_GC, zbins, der_prefix)
        dC_WA_4D = FM_utils.dC_dict_to_4D_array(dC_dict_WA_3D, paramnames_3x2pt, nbl_WA, zbins, der_prefix)
        dC_3x2pt_5D = FM_utils.dC_dict_to_4D_array(dC_dict_3x2pt_5D, paramnames_3x2pt, nbl_3x2pt, zbins, der_prefix,
                                                   is_3x2pt=True)

        # ! BNT transform derivatives - Davide
        if general_cfg['deriv_BNT_transform'] and whos_BNT == '/davide':

            assert general_cfg['cov_BNT_transform'], 'you should BNT transform the covariance as well'

            for alf in range(len(paramnames_3x2pt)):
                dC_LL_4D[:, :, :, alf] = cl_utils.cl_BNT_transform(dC_LL_4D[:, :, :, alf], BNT_matrix, 'L', 'L')
                dC_WA_4D[:, :, :, alf] = cl_utils.cl_BNT_transform(dC_WA_4D[:, :, :, alf], BNT_matrix, 'L', 'L')
                dC_3x2pt_5D[:, :, :, :, :, alf] = cl_utils.cl_BNT_transform_3x2pt(dC_3x2pt_5D[:, :, :, :, :, alf],
                                                                                  BNT_matrix)

            # TODO finish this, it's defined only for WL...
            transformed_derivs_folder = FM_cfg['transformed_derivs_folder']
            transformed_derivs_filename = f'dDV-BNTdav_WLO-wzwaCDM-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-' \
                                          f'{EP_or_ED}{zbins:02}-ML{magcut_lens:03d}-ZL{zcut_lens:02d}-' \
                                          f'MS{magcut_source:03d}-ZS{zcut_source:02d}.npy'

            # save BNT-transformed derivatives
            readme = 'shape: (ell_bins, z_bins, z_bins, num_parameters); parameters order:' + str(paramnames_3x2pt)
            with open(f'{transformed_derivs_folder}/README_transformed_derivs.txt', "w") as text_file:
                text_file.write(readme)

            np.save(f'{transformed_derivs_folder}/{transformed_derivs_filename}', dC_LL_4D)

        # store the derivatives arrays in a dictionary
        deriv_dict = {'dC_LL_4D': dC_LL_4D,
                      'dC_WA_4D': dC_WA_4D,
                      'dC_GG_4D': dC_GG_4D,
                      'dC_3x2pt_5D': dC_3x2pt_5D}

        FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict)
        FM_dict['parameters'] = paramnames_3x2pt
        FM_dict['fiducial_values'] = fid_3x2pt

        ########################################################################################################################
        ######################################################### SAVE #########################################################
        ########################################################################################################################

        # ! save cls and responses:
        # this is just to set the correct probe names
        probe_dav_dict = {'WL': 'LL_3D',
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
                        np.savetxt(f'{filepath}/delta_{ells_filename}.txt', delta_dict[f'delta_l_{probe_dav}'])

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
                        cov_3x2pt_filename = f'covmat_{which_cov}_3x2pt_lmax{ell_max_XC}_' \
                                             f'nbl{nbl_3x2pt}_zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_' \
                                             f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                             f'ZS{zcut_source:02d}_10D.pickle'
                        with open(f'{cov_folder}/{cov_3x2pt_filename}', 'wb') as handle:
                            pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)

            # in the pessimistic case, save only WA
            elif ell_max_WL == 1500:
                for which_cov in cases_tosave:
                    cov_WA_filename = f'covmat_{which_cov}_WA_lmax{ell_max_WL}_nbl{nbl_WA}_' \
                                      f'zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_' \
                                      f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                      f'ZS{zcut_source:02d}_{ndim}D.npy'
                    np.save(f'{cov_folder}/{cov_WA_filename}', cov_dict[f'cov_WA_{which_cov}_{ndim}D'])

    # save in .dat for Vincenzo, only in the optimistic case and in 2D
    if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
        raise NotImplementedError('not implemented yet')
        path_vinc_fmt = f'{job_path}/output/covmat/vincenzos_format'
        for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
            for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):
                np.savetxt(f'{path_vinc_fmt}/{GOGS_folder}/{probe_vinc}/cm-{probe_vinc}-{nbl_WL}'
                           f'-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.dat',
                           cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.10e')

    # ! save FM
    FM_folder = FM_cfg["FM_folder"]
    if FM_cfg['save_FM']:
        probe_list = ['WL', 'GC', '3x2pt', 'WA']
        ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
        nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]
        header = 'parameters\' ordering:' + str(paramnames_3x2pt)

        for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
            for which_cov in cases_tosave:
                FM_filename = FM_cfg["FM_filename"].format(probe=probe, which_cov=which_cov,
                                                           ell_max=ell_max, nbl=nbl,
                                                           **variable_specs)
                np.savetxt(f'{FM_folder}/{FM_filename}', FM_dict[f'FM_{probe}_{which_cov}'], header=header)
                print('FM saved')

    if FM_cfg['save_FM_as_dict']:
        mm.save_pickle(f'{FM_folder}/FM_dict_ML{magcut_lens:03d}-ZL{zcut_lens:02d}-'
                       f'MS{magcut_source:03d}-ZS{zcut_source:02d}.pickle', FM_dict)

cov_output_path = f'{job_path}/output/Flagship_{general_cfg["flagship_version"]}/covmat/BNT_{general_cfg["BNT_transform"]}{whos_BNT}/zbins{zbins}'
cov_benchmark_path = cov_output_path + '/benchmarks'

FM_output_path = f'{job_path}/output/Flagship_{general_cfg["flagship_version"]}/FM/BNT_{general_cfg["BNT_transform"]}{whos_BNT}/zbins{zbins}'
FM_benchmark_path = FM_output_path + '/benchmarks'
ut.test_cov_FM(FM_output_path, FM_benchmarks_path)

print('done')
