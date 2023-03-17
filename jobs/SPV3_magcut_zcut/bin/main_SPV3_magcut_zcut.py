import sys
import time
from pathlib import Path
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import gc

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

matplotlib.use('Qt5Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)

start_time = time.perf_counter()

# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks
# TODO super check that things work with different # of z bins

# TODO invert by nulling the elements of the noise vector with the right indices, then compute covmat in this way and compare the results
# TODO check the cut in the derivatives
# TODO reorder all these cutting functions...
# TODO loop over kmax_list
# TODO careful! the 3x2pt has ell_XC for all probes, see get_idxs_3x2pt function
# TODO recompute Sijkl to be safe
# TODO ell values in linear scale!!!


###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg
Sijkl_cfg = cfg.Sijkl_cfg
FM_cfg = cfg.FM_cfg


def load_ell_cuts(kmax_h_over_Mpc=None):
    """loads ell_cut valeus, rescales them and load into a dictionary"""
    if kmax_h_over_Mpc is None:
        kmax_h_over_Mpc = general_cfg['kmax_h_over_Mpc_ref']

    ell_cuts_fldr = general_cfg['ell_cuts_folder']
    ell_cuts_filename = general_cfg['ell_cuts_filename']
    ell_cuts_LL = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="WL", **variable_specs)}')
    ell_cuts_GG = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="GC", **variable_specs)}')
    warnings.warn('I am not sure this ell_cut file is for GL, the filename is "XC"')
    ell_cuts_GL = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="XC", **variable_specs)}')
    ell_cuts_LG = ell_cuts_GL.T

    # ! linearly rescale ell cuts
    ell_cuts_LL *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
    ell_cuts_GG *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
    ell_cuts_GL *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
    ell_cuts_LG *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref

    ell_cuts_dict = {
        'LL': ell_cuts_LL,
        'GG': ell_cuts_GG,
        'GL': ell_cuts_GL,
        'LG': ell_cuts_LG}

    return ell_cuts_dict


def cl_ell_cut_wrap(ell_dict, cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d, kmax_h_over_Mpc=None):
    """Wrapper for the ell cuts. Avoids the 'if general_cfg['cl_ell_cuts']' in the main loop
    (i.e., we use extraction)"""

    if not general_cfg['cl_ell_cuts']:
        return cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d

    raise Exception('I decided to implement the cuts in 1dim, this function should not be used')

    print('Performing the cl ell cuts...')

    ell_cuts_dict = load_ell_cuts(kmax_h_over_Mpc=kmax_h_over_Mpc)

    cl_ll_3d = cl_utils.cl_ell_cut(cl_ll_3d, ell_cuts_dict['WL'], ell_dict['ell_WL'])
    cl_wa_3d = cl_utils.cl_ell_cut(cl_wa_3d, ell_cuts_dict['WL'], ell_dict['ell_WA'])
    cl_gg_3d = cl_utils.cl_ell_cut(cl_gg_3d, ell_cuts_dict['GC'], ell_dict['ell_GC'])
    cl_3x2pt_5d = cl_utils.cl_ell_cut_3x2pt(cl_3x2pt_5d, ell_cuts_dict, ell_dict)

    return cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d


def get_idxs_to_delete(ell_values, ell_cuts, is_auto_spectrum):
    if is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_idx, ell_val in enumerate(ell_values):
            for zi in range(zbins):
                for zj in range(zi, zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1

    elif not is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_idx, ell_val in enumerate(ell_values):
            for zi in range(zbins):
                for zj in range(zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1
    else:
        raise ValueError('is_auto_spectrum must be True or False')

    return idxs_to_delete


def get_idxs_to_delete_3x2pt(ell_dict):
    idxs_to_delete_LL = get_idxs_to_delete(10 ** ell_dict['ell_XC'], ell_dict['ell_cuts_dict']['LL'], True)
    idxs_to_delete_GL = get_idxs_to_delete(10 ** ell_dict['ell_XC'], ell_dict['ell_cuts_dict']['GL'], False)
    idxs_to_delete_GG = get_idxs_to_delete(10 ** ell_dict['ell_XC'], ell_dict['ell_cuts_dict']['GG'], True)
    idxs_to_delete_3x2pt = idxs_to_delete_LL + idxs_to_delete_GL + idxs_to_delete_GG
    return idxs_to_delete_3x2pt


# ======================================================================================================================


ML_list = general_cfg['magcut_lens_list']
ZL_list = general_cfg['zcut_lens_list']
MS_list = general_cfg['magcut_source_list']
ZS_list = general_cfg['zcut_source_list']

for general_cfg['magcut_lens'], general_cfg['zcut_lens'], general_cfg['magcut_source'], general_cfg['zcut_source'] in \
        zip(ML_list, ZL_list, MS_list, ZS_list):
    # TODO implement this for loop!
    for kmax_h_over_Mpc in general_cfg['kmax_h_over_Mpc_list']:

        # without zip, i.e. for all the possible combinations (aka, a nightmare)
        # for general_cfg['magcut_lens'] in general_cfg['magcut_lens_list']:
        #     for general_cfg['magcut_source'] in general_cfg['magcut_source_list']:
        #         for general_cfg['zcut_lens'] in general_cfg['zcut_lens_list']:
        #             for general_cfg['zcut_source'] in general_cfg['zcut_source_list']:

        # some convenence variables, just to make things more readable
        zbins = general_cfg['zbins']
        EP_or_ED = general_cfg['EP_or_ED']
        ell_max_WL = general_cfg['ell_max_WL']
        ell_max_GC = general_cfg['ell_max_GC']
        ell_max_XC = general_cfg['ell_max_XC']
        magcut_source = general_cfg['magcut_source']
        magcut_lens = general_cfg['magcut_lens']
        zcut_source = general_cfg['zcut_source']
        zcut_lens = general_cfg['zcut_lens']
        zmax = int(general_cfg['zmax'] * 10)
        triu_tril = covariance_cfg['triu_tril']
        row_col_major = covariance_cfg['row_col_major']
        n_probes = general_cfg['n_probes']
        GL_or_LG = covariance_cfg['GL_or_LG']
        probe_ordering = [['L', 'L'], [GL_or_LG[0], GL_or_LG[1]], ['G', 'G']]
        kmax_h_over_Mpc_ref = general_cfg['kmax_h_over_Mpc_ref']

        # some checks
        assert general_cfg['flagship_version'] == 2, 'The input files used in this job for flagship version 2!'
        assert general_cfg['use_WA'] is False, 'We do not use Wadd for SPV3 at the moment'
        if general_cfg['BNT_transform']:
            assert general_cfg['EP_or_ED'] == 'ED', 'BNT matrices are only available for ED case'
            assert general_cfg['zbins'] == 13, 'BNT matrices are only available for zbins=13'

        if covariance_cfg['cov_BNT_transform']:
            assert general_cfg['cl_BNT_transform'] is False, 'the BNT transform should be applied either to the Cls ' \
                                                             'or to the covariance'
            assert FM_cfg['derivatives_BNT_transform'], 'you should BNT transform the derivatives as well'

        # which cases to save: GO, GS or GO, GS and SS
        cases_tosave = ['GO', 'GS']
        if covariance_cfg[f'save_cov_GS']:
            cases_tosave.append('GS')
        if covariance_cfg[f'save_cov_SSC']:
            cases_tosave.append('SS')

        # build the ind array and store it into the covariance dictionary
        ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
        covariance_cfg['ind'] = ind

        # convenience vectors
        zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
        ind_auto = ind[:zpairs_auto, :].copy()
        # ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()

        assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
            'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

        # compute ell and delta ell values in the reference (optimistic) case
        ell_WL_nbl32, delta_l_WL_nbl32 = ell_utils.compute_ells(general_cfg['nbl_WL_opt'], general_cfg['ell_min'],
                                                                general_cfg['ell_max_WL_opt'], recipe='ISTF')
        ell_WL_nbl32 = np.log10(ell_WL_nbl32)

        # perform the cuts
        # TODO take the 10**!!!!
        ell_dict = {
            'ell_WL': np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_WL]),
            'ell_GC': np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_GC]),
            'ell_WA': np.copy(ell_WL_nbl32[(10 ** ell_WL_nbl32 > ell_max_GC) & (10 ** ell_WL_nbl32 < ell_max_WL)])}
        ell_dict['ell_XC'] = np.copy(ell_dict['ell_GC'])

        # set corresponding number of ell bins
        nbl_WL = ell_dict['ell_WL'].shape[0]
        nbl_GC = ell_dict['ell_GC'].shape[0]
        nbl_WA = ell_dict['ell_WA'].shape[0]
        nbl_3x2pt = nbl_GC
        general_cfg['nbl_WL'] = nbl_WL

        delta_dict = {'delta_l_WL': np.copy(delta_l_WL_nbl32[:nbl_WL]),
                      'delta_l_GC': np.copy(delta_l_WL_nbl32[:nbl_GC]),
                      'delta_l_WA': np.copy(delta_l_WL_nbl32[nbl_GC:])}

        # set # of nbl in the opt case, import and reshape, then cut the reshaped datavectors in the pes case
        # TODO this should not be hardcoded! if so, it should go in the config file...
        nbl_WL_opt = 32
        nbl_GC_opt = 29
        nbl_WA_opt = 3
        nbl_3x2pt_opt = 29

        if ell_max_WL == general_cfg['ell_max_WL_opt']:
            assert (nbl_WL_opt, nbl_GC_opt, nbl_WA_opt, nbl_3x2pt_opt) == (nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt), \
                'nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt don\'t match with the expected values for the optimistic case'

        # this is just to make the .format() more compact
        variable_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins, 'magcut_lens': magcut_lens, 'zcut_lens': zcut_lens,
                          'magcut_source': magcut_source, 'zcut_source': zcut_source, 'zmax': zmax,
                          'ell_max_WL': ell_max_WL, 'ell_max_GC': ell_max_GC, 'ell_max_XC': ell_max_XC,
                          'nbl_WL': nbl_WL, 'nbl_GC': nbl_GC, 'nbl_WA': nbl_WA, 'nbl_3x2pt': nbl_3x2pt}

        ng_folder = covariance_cfg["ng_folder"]
        ng_filename = f'{covariance_cfg["ng_filename"].format(**variable_specs)}'

        BNT_matrix_filename = general_cfg["BNT_matrix_filename"].format(**variable_specs)
        BNT_matrix = np.load(f'{general_cfg["BNT_matrix_path"]}/{BNT_matrix_filename}')

        # ! import and reshape datavectors (cl) and response functions (rl)
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

        # reshape to 3 dimensions
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

        # ! BNT transform the cls (and responses?)
        if general_cfg['cl_BNT_transform']:
            print('BNT-transforming the Cls...')
            assert covariance_cfg['cov_BNT_transform'] is False, \
                'the BNT transform should be applied either to the Cls or to the covariance, not both'
            cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, BNT_matrix, 'L', 'L')
            cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, BNT_matrix, 'L', 'L')
            cl_3x2pt_5d = cl_utils.cl_BNT_transform_3x2pt(cl_3x2pt_5d, BNT_matrix)
            warnings.warn('you should probebly BNT-transform the responses too!')

        # ! cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
        if ell_max_WL == 1500:
            warnings.warn(
                'you are cutting the datavectors and responses in the pessimistic case, but is this compatible '
                'with the redshift-dependent ell cuts?')
            assert 1 > 2, 'you should check this'
            cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
            cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
            cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
            cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

            rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
            rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
            rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
            rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

        # ! 3d cl ell cuts (*after* BNT!!)
        cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d = cl_ell_cut_wrap(
            ell_dict, cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d, kmax_h_over_Mpc=None)

        # this is to pass the ll cuts to the covariance module
        warnings.warn('restore kmax_h_over_Mpc in the next line')
        ell_cuts_dict = load_ell_cuts(kmax_h_over_Mpc=None)
        ell_dict['ell_cuts_dict'] = ell_cuts_dict  # rename for better readability

        # ! try vincenzo's method for cl_ell_cuts: get the idxs to delete for the flattened 1d cls
        ell_dict['idxs_to_delete_dict'] = {
            'LL': get_idxs_to_delete(10 ** ell_dict['ell_WL'], ell_cuts_dict['LL'], is_auto_spectrum=True),
            'GG': get_idxs_to_delete(10 ** ell_dict['ell_GC'], ell_cuts_dict['GG'], is_auto_spectrum=True),
            'WA': get_idxs_to_delete(10 ** ell_dict['ell_WA'], ell_cuts_dict['LL'], is_auto_spectrum=True),
            'GL': get_idxs_to_delete(10 ** ell_dict['ell_XC'], ell_cuts_dict['GL'], is_auto_spectrum=False),
            'LG': get_idxs_to_delete(10 ** ell_dict['ell_XC'], ell_cuts_dict['LG'], is_auto_spectrum=False),
            '3x2pt': get_idxs_to_delete_3x2pt(ell_dict)
        }

        # TODO here you could implement 1d cl ell cuts (but we are cutting at the covariance and derivatives level)

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
            z_arr_wil, wil = Sijkl_utils.preprocess_wf(wil, zbins)
            z_arr_wig, wig = Sijkl_utils.preprocess_wf(wig, zbins)
            assert np.array_equal(z_arr_wil, z_arr_wig), 'the redshift arrays are different for the GC and WL kernels'
            z_arr = z_arr_wil

            # transpose and stack, ordering is important here!
            transp_stacked_wf = np.vstack((wil.T, wig.T))

            # ! compute or load Sijkl
            nz = z_arr.shape[0]  # get number of z points in nz to name the Sijkl file
            Sijkl_folder = Sijkl_cfg['Sijkl_folder']
            assert general_cfg['cl_BNT_transform'] is False, 'for SSC, at the moment the BNT transform should not be ' \
                                                             'applied to the cls, but to the covariance matrix (how ' \
                                                             'should we deal with the responses in the former case?)'
            Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(flagship_version=general_cfg['flagship_version'],
                                                                nz=nz, IA_flag=Sijkl_cfg['has_IA'], **variable_specs)

            # if Sijkl exists, load it; otherwise, compute it and save it
            if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):
                print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
                Sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')
            else:
                Sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                                  Sijkl_cfg['wf_normalization'])
                np.save(f'{Sijkl_folder}/{Sijkl_filename}', Sijkl)

            # ! compute covariance matrix
            # TODO: if already existing, don't compute the covmat, like done above for Sijkl
            # the ng values are in the second column, for these input files 👇
            covariance_cfg['ng'] = np.genfromtxt(f'{ng_folder}/'f'{ng_filename}')[:, 1]
            cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                                ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, BNT_matrix)

            # save covariance matrix and test against benchmarks
            cov_folder = covariance_cfg['cov_folder'].format(ell_cuts=str(general_cfg['ell_cuts']),
                                                             **variable_specs)
            covmat_utils.save_cov(cov_folder, covariance_cfg, cov_dict, **variable_specs)

            cov_benchmark_folder = f'{cov_folder}/benchmarks'
            mm.test_folder_content(cov_folder, cov_benchmark_folder, covariance_cfg['cov_file_format'])

        # ! compute Fisher matrix
        if FM_cfg['compute_FM']:

            # set the fiducial values in a dictionary and a list
            fiducials_dict = {
                'cosmo': [ISTF_fid.primary['Om_m0'], ISTF_fid.extensions['Om_Lambda0'], ISTF_fid.primary['Om_b0'],
                          ISTF_fid.primary['w_0'], ISTF_fid.primary['w_a'],
                          ISTF_fid.primary['h_0'], ISTF_fid.primary['n_s'], ISTF_fid.primary['sigma_8']],
                'IA': np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()]),
                'galaxy_bias': np.genfromtxt(f'{ng_folder}/{ng_filename}')[:, 2],  # ! it needs to be set in the main!
                'shear_bias': np.zeros((zbins,)),
                'dzWL': np.zeros((zbins,)),
                'dzGC': np.zeros((zbins,)),
            }

            fiducials_3x2pt = np.concatenate(
                (fiducials_dict['cosmo'], fiducials_dict['IA'], fiducials_dict['galaxy_bias'],
                 fiducials_dict['shear_bias'], fiducials_dict['dzWL'], fiducials_dict['dzGC']))

            # set parameters' names, as a dict and as a list
            param_names_dict = FM_cfg['param_names_dict']
            param_names_3x2pt = FM_cfg['param_names_3x2pt']

            assert len(fiducials_3x2pt) == len(param_names_3x2pt), \
                'the fiducial values list and parameter names should have the same length'

            # ! preprocess derivatives
            # import and store them in one big dictionary
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
                    dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WL', nbl_WL, zbins)
                elif 'GCO' in key:
                    dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'GC', nbl_GC, zbins)
                elif 'WLA' in key:
                    dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WA', nbl_WA, zbins)
                elif '3x2pt' in key:
                    dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], '3x2pt', nbl_3x2pt, zbins)

            # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
            dC_LL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix)
            dC_GG_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix)
            dC_WA_4D = FM_utils.dC_dict_to_4D_array(dC_dict_WA_3D, param_names_3x2pt, nbl_WA, zbins, der_prefix)
            dC_3x2pt_6D = FM_utils.dC_dict_to_4D_array(dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins,
                                                       der_prefix, is_3x2pt=True)

            # free up memory
            del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_WA_3D, dC_dict_3x2pt_5D
            gc.collect()

            # store the derivatives arrays in a dictionary
            deriv_dict = {'dC_LL_4D': dC_LL_4D,
                          'dC_WA_4D': dC_WA_4D,
                          'dC_GG_4D': dC_GG_4D,
                          'dC_3x2pt_6D': dC_3x2pt_6D}

            # ! compute and save fisher matrix
            FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict,
                                          BNT_matrix)
            FM_dict['param_names_dict'] = param_names_dict
            FM_dict['fiducial_values_dict'] = fiducials_dict

            fm_folder = FM_cfg['fm_folder'].format(ell_cuts=str(general_cfg['ell_cuts']))
            FM_utils.save_FM(fm_folder, FM_dict, FM_cfg, FM_cfg['save_FM_txt'], FM_cfg['save_FM_dict'],
                             **variable_specs)

            del cov_dict
            gc.collect()

        # ! unit test: check that the outputs have not changed
        fm_benchmark_folder = f'{fm_folder}/benchmarks'
        mm.test_folder_content(fm_folder, fm_benchmark_folder, 'txt')

    """
    # ! save cls and responses:
    # TODO this should go inside a function too
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

            for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
                # save cl and/or response; not very readable but it works, plus all the cases are in the for loop

                filepath = f'{general_cfg[f"{cl_or_rl}_folder"]}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}'
                filename = general_cfg[f'{cl_or_rl}_filename'].format(
                    probe=probe_vinc, **variable_specs).replace(".dat", "_3D.npy")
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

    """

print('done')
