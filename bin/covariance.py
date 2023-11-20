import gc
import pdb
import sys
import time
import warnings
from pathlib import Path
import pickle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange
from scipy.integrate import simps

import cl_preprocessing
import pyccl_cov
import sigma2_SSC

matplotlib.use('Qt5Agg')

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(str(project_path_here.parent / 'cl_v2/bin'))
import wf_cl_lib

###############################################################################
################ CODE TO COMPUTE THE G AND SSC COVMATS ########################
###############################################################################

probe_names_dict = {'LL': 'WL', 'GG': 'GC', '3x2pt': '3x2pt', }


def get_ellmax_nbl(probe, general_cfg):
    if probe == 'LL':
        ell_max = general_cfg['ell_max_WL']
        nbl = general_cfg['nbl_WL']
    elif probe == 'GG':
        ell_max = general_cfg['ell_max_GC']
        nbl = general_cfg['nbl_GC']
    elif probe == '3x2pt':
        ell_max = general_cfg['ell_max_XC']
        nbl = general_cfg['nbl_3x2pt']
    else:
        raise ValueError('probe must be LL or GG or 3x2pt')
    return ell_max, nbl


def ssc_with_exactSSC(general_cfg, covariance_cfg, return_format_3x2pt):
    # this actually just imports the precomputed ssc. It can also compute deltab, quite useless at the moment
    print('computing SSC covariance with exactSSC...')
    warnings.warn('I am dividing by fsky in the import here below (8d dict), should be correct but be careful...')

    probe = covariance_cfg['exactSSC_cfg']['probe']
    zbins = general_cfg['zbins']
    ell_max, nbl = get_ellmax_nbl(probe, general_cfg)
    probe_ordering = covariance_cfg['probe_ordering']
    ind_dict = covariance_cfg['ind_dict']

    z_steps_sigma2 = covariance_cfg['exactSSC_cfg']['z_steps_sigma2']
    k_txt_label = covariance_cfg['exactSSC_cfg']['k_txt_label']
    cl_integral_convention = covariance_cfg['exactSSC_cfg']['cl_integral_convention']
    path = covariance_cfg['exactSSC_cfg']['path']

    general_suffix = f'nbl{nbl}_ellmax{ell_max}_zbins{zbins}_' \
                     f'zsteps{z_steps_sigma2}_k{k_txt_label}_convention{cl_integral_convention}'

    if not covariance_cfg['exactSSC_cfg']['use_precomputed_sigma2']:
        # this part should be finished, what should I do with the array? save it?
        cosmo_ccl = csmlib.istantiate_cosmo_ccl_obj(general_cfg['fid_pars_dict'],
                                                    general_cfg['fid_pars_dict']['other_params']['extra_parameters'])

        sigma2, z_grid_sigma2 = sigma2_SSC.compute_sigma2(covariance_cfg['exactSSC_cfg'], cosmo_ccl)

        z_steps_sigma2 = len(z_grid_sigma2)
        np.savez_compressed(f'/Users/davide/Documents/Lavoro/Programmi/exact_SSC/output/integrand_arrays/sigma2/'
                            f'sigma2_zsteps{z_steps_sigma2}.npz')

    # single-probe case
    if probe in ('LL', 'GG'):
        cov_exactSSC_SS_4D = np.load(f'{path}/cov_SSC_{probe}{probe}_4D_nbl{nbl}_ellmax{ell_max}_zbins{zbins}_'
                                     f'zsteps{z_steps_sigma2}_k{k_txt_label}_convention{cl_integral_convention}.npy')
    # populate 3x2pt dictionary
    elif probe == '3x2pt':
        cov_exactSSC_3x2pt_dict_8D = {}
        cov_exactSSC_3x2pt_dict_10D = {}
        for probe_A, probe_B in probe_ordering:
            for probe_C, probe_D in probe_ordering:

                try:
                    cov_exactSSC_3x2pt_dict_8D[probe_A, probe_B, probe_C, probe_D] = np.load(
                        f'{path}/cov_SSC_{probe_A}{probe_B}{probe_C}{probe_D}_4D_{general_suffix}.npy')
                except FileNotFoundError:
                    # for 3x2pt, I have the files with 32 bins, ie with lmax = 5000.
                    general_suffix_nbl29 = general_suffix.replace('nbl29', 'nbl32')
                    general_suffix_nbl29 = general_suffix_nbl29.replace('ellmax3000', 'ellmax5000')

                    # cut the covariance to 29 bins
                    cov_exactSSC_3x2pt_dict_8D[probe_A, probe_B, probe_C, probe_D] = np.load(
                        f'{path}/cov_SSC_{probe_A}{probe_B}{probe_C}{probe_D}_'
                        f'4D_{general_suffix_nbl29}.npy')[:nbl, :nbl, :, :] / covariance_cfg['fsky']

                    cov_exactSSC_3x2pt_dict_10D[probe_A, probe_B, probe_C, probe_D] = mm.cov_4D_to_6D_blocks(
                        cov_exactSSC_3x2pt_dict_8D[probe_A, probe_B, probe_C, probe_D],
                        nbl, zbins, ind_dict[probe_A, probe_B], ind_dict[probe_C, probe_D])

        assert probe == '3x2pt', ('probe must be 3x2pt at the moment, messing around with return dimension for BNT. to '
                                  'be implemented better later')
        if return_format_3x2pt == 'dict_8d':
            return cov_exactSSC_3x2pt_dict_8D
        elif return_format_3x2pt == 'dict_10d':
            return cov_exactSSC_3x2pt_dict_10D
        elif return_format_3x2pt == '4d_array':
            cov_exactSSC_SS_4D = mm.cov_3x2pt_8D_dict_to_4D(cov_exactSSC_3x2pt_dict_8D, probe_ordering)
            return cov_exactSSC_SS_4D
        else:
            raise ValueError('return_format_3x2pt must be "dict_8d" or "dict_10d" or 4d_array')

    else:
        raise ValueError('probe must be LL or GG or 3x2pt')


def ssc_with_pyccl(general_cfg, covariance_cfg, ell_dict):
    print('computing SSC covariance with PyCCL')
    warnings.warn('input nofz for ccl, or better the kernels!')

    probe = covariance_cfg['PyCCL_cfg']['probe']
    zbins = general_cfg['zbins']
    ell_max, nbl = get_ellmax_nbl(probe, general_cfg)
    ell_grid = ell_dict['ell_' + probe_names_dict[probe]]
    path_ccl = covariance_cfg['PyCCL_cfg']['path']
    probe_ordering = covariance_cfg['probe_ordering']
    use_hod_for_gcph = covariance_cfg['PyCCL_cfg']['use_HOD_for_GCph']
    ind_dict = covariance_cfg['ind_dict']

    general_suffix = f'nbl{nbl}_ellmax{ell_max}_zbins{zbins}'
    cov_8D_dict_filename = f'cov_PyCCL_SSC_{probe}_{general_suffix}' \
                           f'_8D_dict_useHOD{use_hod_for_gcph}.pickle'

    if covariance_cfg['PyCCL_cfg']['load_precomputed_cov']:

        # all the covs of interest are in the 3x2pt picke file
        cov_8D_dict_filename = cov_8D_dict_filename.replace(probe, '3x2pt')
        # cov_PyCCL_dict_8D = mm.load_pickle(f'{path_ccl}/{cov_8D_dict_filename}')

        if probe in ('LL', 'GG'):
            cov_PyCCL_SS_4D = cov_PyCCL_dict_8D[probe[0], probe[1], probe[0], probe[1]]
        else:
            path = '/Users/davide/Desktop/pyccl_cov_spv3_test'
            # cov_PyCCL_SS_4D = mm.cov_3x2pt_8D_dict_to_4D(cov_PyCCL_dict_8D, probe_ordering)
            cov_ccl_3x2pt_dict_8D = {}
            cov_ccl_3x2pt_dict_10D = {}
            for probe_A, probe_B in probe_ordering:
                for probe_C, probe_D in probe_ordering:
                    cov_ccl_3x2pt_dict_8D[probe_A, probe_B, probe_C, probe_D] = np.load(
                        f'{path}/cov_ssc_3x2pt_dict_8D_{probe_A}{probe_B}{probe_C}{probe_D}.npy')
                    cov_ccl_3x2pt_dict_10D[probe_A, probe_B, probe_C, probe_D] = mm.cov_4D_to_6D_blocks(
                        cov_ccl_3x2pt_dict_8D[probe_A, probe_B, probe_C, probe_D],
                        nbl, zbins, ind_dict[probe_A, probe_B], ind_dict[probe_C, probe_D])
                    print(f'ssc block {probe_A}{probe_B}{probe_C}{probe_D} loaded')

            return cov_ccl_3x2pt_dict_10D

        # old, only for LL and GG  # cov_PyCCL_SS_4D = np.load(f'{path_ccl}/cov_PyCCL_SSC_{probe}_{general_suffix}_4D.npz')['arr_0']

    else:

        cov_PyCCL_SS_8D_dict = pyccl_cov.compute_cov_ng_with_pyccl(general_cfg['fid_pars_dict'], probe,
                                                                   'SSC', ell_grid, general_cfg, covariance_cfg)
        if covariance_cfg['PyCCL_cfg']['save_cov']:

            # not the best way, dict vs 4d array as output...
            if probe == '3x2pt' and not covariance_cfg['PyCCL_cfg']['get_3x2pt_cov_in_4D']:
                # in this case, save the whole dictionary, then revert to 4d array as output
                cov_8D_dict = cov_PyCCL_SS_4D
                mm.save_pickle(f'{path_ccl}/{cov_8D_dict_filename}', cov_PyCCL_SS_4D)
                cov_PyCCL_SS_4D = mm.cov_3x2pt_8D_dict_to_4D(cov_PyCCL_SS_4D, probe_ordering)

            else:
                np.savez_compressed(f'{path_ccl}/cov_PyCCL_SSC_{probe}_{general_suffix}_4D.npz', cov_PyCCL_SS_4D)

    if covariance_cfg['PyCCL_cfg']['compute_cng']:
        warnings.warn('computing cNG with PyCCL. This is then added to SSC in a rudimental way; the code has to be '
                      'improved')

        warnings.warn('SKIPPING COMPUTATION OF CNG FOR THE MOMENT, restore the lines below')
        # cov_PyCCL_cng_8D_dict = pyccl_cov.compute_cov_ng_with_pyccl(probe, 'cNG', ell_grid, z_grid_nofz=None,
        #                                                        n_of_z=None,
        #                                                        general_cfg=general_cfg,
        #                                                        covariance_cfg=covariance_cfg)
        # mm.save_pickle(f'{path_ccl}/{cov_8D_dict_filename.replace("SSC", "cNG")}', cov_PyCCL_cng_8D_dict)

        cov_PyCCL_cng_8D_dict = mm.load_pickle(f'{path_ccl}/{cov_8D_dict_filename.replace("SSC", "cNG")}')
        cov_PyCCL_cng_4D = mm.cov_3x2pt_8D_dict_to_4D(cov_PyCCL_cng_8D_dict, probe_ordering)

        cov_PyCCL_SS_4D += cov_PyCCL_cng_4D

    return cov_PyCCL_SS_4D


def compute_cov(general_cfg, covariance_cfg, ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, BNT_matrix):
    """
    This code computes the Gaussian-only, SSC-only and Gaussian+SSC
    covariance matrices, for different ordering options
    """

    # import settings:
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    ell_max_3x2pt = general_cfg['ell_max_XC']
    zbins = general_cfg['zbins']
    n_probes = general_cfg['n_probes']
    triu_tril = covariance_cfg['triu_tril']
    rowcol_major = covariance_cfg['row_col_major']
    SSC_code = covariance_cfg['SSC_code']

    fsky = covariance_cfg['fsky']
    GL_or_LG = covariance_cfg['GL_or_LG']
    # ! must copy the array! Otherwise, it gets modified and changed at each call
    ind = covariance_cfg['ind'].copy()
    block_index = covariance_cfg['block_index']
    which_probe_response = covariance_cfg['which_probe_response']
    probe_ordering = covariance_cfg['probe_ordering']

    # (not the best) check to ensure that the (LL, XC, GG) ordering is respected
    assert probe_ordering[0] == ('L', 'L'), 'the XC probe should be in position 1 (not 0) of the datavector'
    assert probe_ordering[2] == ('G', 'G'), 'the XC probe should be in position 1 (not 0) of the datavector'

    # import ell values
    ell_WL, nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
    ell_GC, nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
    ell_WA, nbl_WA = ell_dict['ell_WA'], ell_dict['ell_WA'].shape[0]
    ell_3x2pt, nbl_3x2pt = ell_GC, nbl_GC

    cov_dict = {}

    # sanity checks
    if general_cfg['nbl_WL'] is None:
        assert nbl_WL == general_cfg['nbl'], 'nbl_WL != general_cfg["nbl"], there is a discrepancy'

    if general_cfg['nbl_WL'] is not None:
        assert nbl_WL == general_cfg['nbl_WL'], 'nbl_WL != general_cfg["nbl_WL"], there is a discrepancy'

    if nbl_WL == nbl_GC == nbl_3x2pt:
        print('all probes (but WAdd) have the same number of ell bins')

    # nbl for Wadd
    if ell_WA.size == 1:
        nbl_WA = 1  # in the case of just one bin it would give error
    else:
        nbl_WA = ell_WA.shape[0]

    # ell values in linear scale:
    if ell_WL.max() < 15:  # very rudimental check of whether they're in lin or log scale
        raise ValueError('looks like the ell values are in log scale. You should use linear scale instead.')

    # load deltas
    delta_l_WL = delta_dict['delta_l_WL']
    delta_l_GC = delta_dict['delta_l_GC']
    delta_l_WA = delta_dict['delta_l_WA']
    delta_l_3x2pt = delta_l_GC

    # load set correct output folder, get number of pairs
    # output_folder = mm.get_output_folder(ind_ordering, which_forecast)
    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    # if C_XC is C_LG, switch the ind.dat ordering for the correct rows
    if GL_or_LG == 'LG':
        print('\nAttention! switching columns in the ind array (for the XC part)')
        ind[zpairs_auto:(zpairs_auto + zpairs_cross), [2, 3]] = ind[zpairs_auto:(zpairs_auto + zpairs_cross), [3, 2]]

    # sanity check: the last 2 columns of ind_auto should be equal to the last two of ind_auto
    assert np.array_equiv(ind[:zpairs_auto, 2:], ind[-zpairs_auto:, 2:])

    # convenience vectors, used for the cov_4D_to_6D function
    ind_auto = ind[:zpairs_auto, :].copy()
    ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()
    ind_dict = {('L', 'L'): ind_auto,
                ('G', 'L'): ind_cross,
                ('G', 'G'): ind_auto}
    covariance_cfg['ind_dict'] = ind_dict

    # load Cls
    cl_LL_3D = cl_dict_3D['cl_LL_3D']
    cl_GG_3D = cl_dict_3D['cl_GG_3D']
    cl_WA_3D = cl_dict_3D['cl_WA_3D']
    cl_3x2pt_5D = cl_dict_3D['cl_3x2pt_5D']

    if which_probe_response == 'constant':
        rl_value = covariance_cfg['response_const_value']
        rl_LL_3D = np.full(cl_LL_3D.shape, rl_value)
        rl_GG_3D = np.full(cl_GG_3D.shape, rl_value)
        rl_WA_3D = np.full(cl_WA_3D.shape, rl_value)
        rl_3x2pt_5D = np.full(cl_3x2pt_5D.shape, rl_value)
    elif which_probe_response == 'variable':
        rl_LL_3D = rl_dict_3D['rl_LL_3D']
        rl_GG_3D = rl_dict_3D['rl_GG_3D']
        rl_WA_3D = rl_dict_3D['rl_WA_3D']
        rl_3x2pt_5D = rl_dict_3D['rl_3x2pt_5D']
    else:
        raise ValueError("which_probe_response must be 'constant' or 'variable'")

    # print settings
    print(f'\ncheck: \nind_ordering = {triu_tril}, {rowcol_major} \nblock_index = {block_index}\n'
          f'zbins: {general_cfg["EP_or_ED"]}{zbins}\n'
          f'nbl_WA: {nbl_WA} nbl_WL: {nbl_WL} nbl_GC:  {nbl_GC}, nbl_3x2pt:  {nbl_3x2pt}\n'
          f'ell_max_WL = {ell_max_WL} \nell_max_GC = {ell_max_GC}\nGL_or_LG: {GL_or_LG}\n')

    # ! ======================================= COMPUTE GAUSS ONLY COVARIANCE =======================================

    # build noise vector
    warnings.warn('which folder should I use for ngbTab? lenses or sources? Flagship or Redbook?')
    noise_3x2pt_4D = mm.build_noise(zbins, n_probes, sigma_eps2=covariance_cfg['sigma_eps2'], ng=covariance_cfg['ng'],
                                    EP_or_ED=general_cfg['EP_or_ED'])

    # create dummy ell axis, the array is just repeated along it
    nbl_max = np.max((nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA))
    noise_5D = np.zeros((n_probes, n_probes, nbl_max, zbins, zbins))
    for probe_A in (0, 1):
        for probe_B in (0, 1):
            for ell_idx in range(nbl_WL):
                noise_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

    # remember, the ell axis is a dummy one for the noise, is just needs to be of the
    # same length as the corresponding cl one
    noise_LL_5D = noise_5D[0, 0, :nbl_WL, :, :][np.newaxis, np.newaxis, ...]
    noise_GG_5D = noise_5D[1, 1, :nbl_GC, :, :][np.newaxis, np.newaxis, ...]
    noise_WA_5D = noise_5D[0, 0, :nbl_WA, :, :][np.newaxis, np.newaxis, ...]
    noise_3x2pt_5D = noise_5D[:, :, :nbl_3x2pt, :, :]

    if general_cfg['cl_BNT_transform']:
        print('BNT-transforming the noise spectra...')
        noise_LL_5D = cl_preprocessing.cl_BNT_transform(noise_LL_5D[0, 0, ...], BNT_matrix, 'L', 'L')[None, None, ...]
        noise_WA_5D = cl_preprocessing.cl_BNT_transform(noise_WA_5D[0, 0, ...], BNT_matrix, 'L', 'L')[None, None, ...]
        noise_3x2pt_5D = cl_preprocessing.cl_BNT_transform_3x2pt(noise_3x2pt_5D, BNT_matrix)

    start = time.perf_counter()
    cl_LL_5D = cl_LL_3D[np.newaxis, np.newaxis, ...]
    cl_GG_5D = cl_GG_3D[np.newaxis, np.newaxis, ...]
    cl_WA_5D = cl_WA_3D[np.newaxis, np.newaxis, ...]
    rl_LL_5d = rl_LL_3D[np.newaxis, np.newaxis, ...]
    rl_GG_5d = rl_GG_3D[np.newaxis, np.newaxis, ...]
    rl_WA_5d = rl_WA_3D[np.newaxis, np.newaxis, ...]

    # 5d versions of auto-probe spectra
    cov_WL_GO_6D = mm.covariance_einsum(cl_LL_5D, noise_LL_5D, fsky, ell_WL, delta_l_WL)[0, 0, 0, 0, ...]
    cov_GC_GO_6D = mm.covariance_einsum(cl_GG_5D, noise_GG_5D, fsky, ell_GC, delta_l_GC)[0, 0, 0, 0, ...]
    cov_WA_GO_6D = mm.covariance_einsum(cl_WA_5D, noise_WA_5D, fsky, ell_WA, delta_l_WA)[0, 0, 0, 0, ...]
    cov_3x2pt_GO_10D = mm.covariance_einsum(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_3x2pt, delta_l_3x2pt)
    print("Gauss. cov. matrices computed in %.2f seconds" % (time.perf_counter() - start))

    ######################## COMPUTE SSC COVARIANCE ###############################

    # compute the covariance with PySSC anyway, not to have problems with WA
    start = time.perf_counter()

    # preprocess Sijkl by expanding the probe dimensions
    s_ABCD_ijkl = mm.expand_dims_sijkl(Sijkl, zbins)
    s_LLLL_ijkl = s_ABCD_ijkl[0, 0, 0, 0, ...][np.newaxis, np.newaxis, np.newaxis, np.newaxis, ...]
    s_GGGG_ijkl = s_ABCD_ijkl[1, 1, 1, 1, ...][np.newaxis, np.newaxis, np.newaxis, np.newaxis, ...]

    # compute 6d SSC
    cov_WL_SS_6D = mm.covariance_SSC_einsum(cl_LL_5D, rl_LL_5d, s_LLLL_ijkl, fsky)[0, 0, 0, 0, ...]
    cov_GC_SS_6D = mm.covariance_SSC_einsum(cl_GG_5D, rl_GG_5d, s_GGGG_ijkl, fsky)[0, 0, 0, 0, ...]
    cov_WA_SS_6D = mm.covariance_SSC_einsum(cl_WA_5D, rl_WA_5d, s_LLLL_ijkl, fsky)[0, 0, 0, 0, ...]
    cov_3x2pt_SS_10D = mm.covariance_SSC_einsum(cl_3x2pt_5D, rl_3x2pt_5D, s_ABCD_ijkl, fsky)
    print("SS cov. matrices computed in %.2f s with PySSC" % (time.perf_counter() - start))

    # these are needed by PyCCL and exactSSC:
    if SSC_code != 'PySSC':

        probe_ssc_code = covariance_cfg[SSC_code + '_cfg']['probe']

        if probe_ssc_code == 'LL':
            ell_grid = ell_WL
            ell_max = ell_max_WL
        elif probe_ssc_code == 'GG':
            ell_grid = ell_GC
            ell_max = ell_max_GC
        elif probe_ssc_code == '3x2pt':
            ell_grid = ell_3x2pt
            ell_max = ell_max_3x2pt
        else:
            raise ValueError(f'probe_ssc_code must be LL or GG or 3x2pt')
        nbl_ssc_code = len(ell_grid)

    start_time = time.perf_counter()
    if SSC_code == 'exactSSC':
        warnings.warn('the name of this function should be changed...')
        cov_exactSSC_SS_dict_10D = ssc_with_exactSSC(general_cfg, covariance_cfg, return_format_3x2pt='dict_10d')
        cov_3x2pt_SS_10D = mm.cov_10D_dict_to_array(cov_exactSSC_SS_dict_10D, nbl_3x2pt, zbins, n_probes)

    elif SSC_code == 'PyCCL':
        cov_ccl_3x2pt_dict_10D = ssc_with_pyccl(general_cfg, covariance_cfg, ell_dict)
        cov_3x2pt_SS_10D = mm.cov_10D_dict_to_array(cov_ccl_3x2pt_dict_10D, nbl_3x2pt, zbins, n_probes)


    elif SSC_code not in ('PySSC', 'PyCCL', 'exactSSC'):
        raise ValueError('covariance_cfg["SSC_code"] must be PySSC or PyCCL or exactSSC')

    print(f'SSC covariance computed with {SSC_code} in {(time.perf_counter() - start_time):.2f} seconds')

    # sum GO and SS in 6D (or 10D), not in 4D (it's the same)
    cov_WL_GS_6D = cov_WL_GO_6D + cov_WL_SS_6D
    cov_GC_GS_6D = cov_GC_GO_6D + cov_GC_SS_6D
    cov_WA_GS_6D = cov_WA_GO_6D + cov_WA_SS_6D
    cov_3x2pt_GS_10D = cov_3x2pt_GO_10D + cov_3x2pt_SS_10D

    # ! BNT transform
    if covariance_cfg['cov_BNT_transform']:
        print('BNT-transforming the covariance matrix...')

        # turn to dict for the BNT function
        cov_3x2pt_GO_10D_dict = mm.cov_10D_array_to_dict(cov_3x2pt_GO_10D, probe_ordering)
        cov_3x2pt_GS_10D_dict = mm.cov_10D_array_to_dict(cov_3x2pt_GS_10D, probe_ordering)

        X_dict = build_X_matrix_BNT(BNT_matrix)
        cov_WL_GO_6D = cov_BNT_transform(cov_WL_GO_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_WA_GO_6D = cov_BNT_transform(cov_WA_GO_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_3x2pt_GO_10D_dict = cov_3x2pt_BNT_transform(cov_3x2pt_GO_10D_dict, X_dict)

        cov_WL_GS_6D = cov_BNT_transform(cov_WL_GS_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_WA_GS_6D = cov_BNT_transform(cov_WA_GS_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_3x2pt_GS_10D_dict = cov_3x2pt_BNT_transform(cov_3x2pt_GS_10D_dict, X_dict)

        # revert to 10D arrays
        cov_3x2pt_GO_10D = mm.cov_10D_dict_to_array(cov_3x2pt_GO_10D_dict, nbl_3x2pt, zbins, n_probes=2)
        cov_3x2pt_GS_10D = mm.cov_10D_dict_to_array(cov_3x2pt_GS_10D_dict, nbl_3x2pt, zbins, n_probes=2)

    # ! transform everything in 4D
    start = time.perf_counter()
    cov_WL_GO_4D = mm.cov_6D_to_4D(cov_WL_GO_6D, nbl_WL, zpairs_auto, ind_auto)
    cov_GC_GO_4D = mm.cov_6D_to_4D(cov_GC_GO_6D, nbl_GC, zpairs_auto, ind_auto)
    cov_WA_GO_4D = mm.cov_6D_to_4D(cov_WA_GO_6D, nbl_WA, zpairs_auto, ind_auto)
    cov_3x2pt_GO_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_GO_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(), GL_or_LG)

    cov_WL_GS_4D = mm.cov_6D_to_4D(cov_WL_GS_6D, nbl_WL, zpairs_auto, ind_auto)
    cov_GC_GS_4D = mm.cov_6D_to_4D(cov_GC_GS_6D, nbl_GC, zpairs_auto, ind_auto)
    cov_WA_GS_4D = mm.cov_6D_to_4D(cov_WA_GS_6D, nbl_WA, zpairs_auto, ind_auto)
    cov_3x2pt_GS_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_GS_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(), GL_or_LG)
    print('covariance matrices reshaped (6D -> 4D) in {:.2f} s'.format(time.perf_counter() - start))

    # ! ========================= plug the 4D covariances into the pipeline ============================================
    # ! i commented this because now I'm plugging exactSSC as a 10d dict (10 array, to be precise)
    # if SSC_code in ('PyCCL', 'exactSSC'):
    #
    #     print(f'adding SSC cov from {SSC_code} directly in 4D. This creates some problems with the BNT, TODO')
    #
    #     if SSC_code == 'PyCCL':
    #         cov_SS_4D = cov_PyCCL_SS_4D
    #     elif SSC_code == 'exactSSC':
    #         cov_SS_4D = cov_exactSSC_SS_4D
    #     else:
    #         raise ValueError('SSC_code must be PyCCL or exactSSC')
    #
    #     if probe_ssc_code == 'LL':
    #         cov_WL_GS_4D = cov_WL_GO_4D + cov_SS_4D
    #     elif probe_ssc_code == 'GG':
    #         cov_GC_GS_4D = cov_GC_GO_4D + cov_SS_4D
    #     elif probe_ssc_code == '3x2pt':
    #         cov_3x2pt_GS_4D = cov_3x2pt_GO_4D + cov_SS_4D

    # ! transform everything in 2D
    start = time.perf_counter()
    cov_WL_GO_2D = mm.cov_4D_to_2D(cov_WL_GO_4D, block_index=block_index)
    cov_GC_GO_2D = mm.cov_4D_to_2D(cov_GC_GO_4D, block_index=block_index)
    cov_WA_GO_2D = mm.cov_4D_to_2D(cov_WA_GO_4D, block_index=block_index)
    cov_3x2pt_GO_2D = mm.cov_4D_to_2D(cov_3x2pt_GO_4D, block_index=block_index)

    cov_WL_GS_2D = mm.cov_4D_to_2D(cov_WL_GS_4D, block_index=block_index)
    cov_GC_GS_2D = mm.cov_4D_to_2D(cov_GC_GS_4D, block_index=block_index)
    cov_WA_GS_2D = mm.cov_4D_to_2D(cov_WA_GS_4D, block_index=block_index)
    cov_3x2pt_GS_2D = mm.cov_4D_to_2D(cov_3x2pt_GS_4D, block_index=block_index)
    print('covariance matrices reshaped (4D -> 2D) in {:.2f} s'.format(time.perf_counter() - start))

    if covariance_cfg['cov_ell_cuts']:
        # perform the cuts on the 2D covs (way faster!)
        print('Performing ell cuts on the 2d covariance matrix...')
        cov_WL_GO_2D = mm.remove_rows_cols_array2D(cov_WL_GO_2D, ell_dict['idxs_to_delete_dict']['LL'])
        cov_GC_GO_2D = mm.remove_rows_cols_array2D(cov_GC_GO_2D, ell_dict['idxs_to_delete_dict']['GG'])
        cov_WA_GO_2D = mm.remove_rows_cols_array2D(cov_WA_GO_2D, ell_dict['idxs_to_delete_dict']['WA'])
        cov_3x2pt_GO_2D = mm.remove_rows_cols_array2D(cov_3x2pt_GO_2D, ell_dict['idxs_to_delete_dict']['3x2pt'])

        cov_WL_GS_2D = mm.remove_rows_cols_array2D(cov_WL_GS_2D, ell_dict['idxs_to_delete_dict']['LL'])
        cov_GC_GS_2D = mm.remove_rows_cols_array2D(cov_GC_GS_2D, ell_dict['idxs_to_delete_dict']['GG'])
        cov_WA_GS_2D = mm.remove_rows_cols_array2D(cov_WA_GS_2D, ell_dict['idxs_to_delete_dict']['WA'])
        cov_3x2pt_GS_2D = mm.remove_rows_cols_array2D(cov_3x2pt_GS_2D, ell_dict['idxs_to_delete_dict']['3x2pt'])

    ############################### save in dictionary ########################
    probe_names = ('WL', 'GC', '3x2pt', 'WA')

    covs_GO_4D = (cov_WL_GO_4D, cov_GC_GO_4D, cov_3x2pt_GO_4D, cov_WA_GO_4D)
    covs_GS_4D = (cov_WL_GS_4D, cov_GC_GS_4D, cov_3x2pt_GS_4D, cov_WA_GS_4D)

    covs_GO_2D = (cov_WL_GO_2D, cov_GC_GO_2D, cov_3x2pt_GO_2D, cov_WA_GO_2D)
    covs_GS_2D = (cov_WL_GS_2D, cov_GC_GS_2D, cov_3x2pt_GS_2D, cov_WA_GS_2D)

    if covariance_cfg['save_cov_SSC']:
        warnings.warn('this is not correct for exatcSSC and PySSC, the dimensionality reduction is done earlier!')
        cov_WL_SS_4D = mm.cov_6D_to_4D(cov_WL_SS_6D, nbl_WL, zpairs_auto, ind_auto)
        cov_GC_SS_4D = mm.cov_6D_to_4D(cov_GC_SS_6D, nbl_GC, zpairs_auto, ind_auto)
        cov_WA_SS_4D = mm.cov_6D_to_4D(cov_WA_SS_6D, nbl_WA, zpairs_auto, ind_auto)
        cov_3x2pt_SS_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SS_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(),
                                                 GL_or_LG)

        cov_WL_SS_2D = mm.cov_4D_to_2D(cov_WL_SS_4D, block_index=block_index)
        cov_GC_SS_2D = mm.cov_4D_to_2D(cov_GC_SS_4D, block_index=block_index)
        cov_WA_SS_2D = mm.cov_4D_to_2D(cov_WA_SS_4D, block_index=block_index)
        cov_3x2pt_SS_2D = mm.cov_4D_to_2D(cov_3x2pt_SS_4D, block_index=block_index)

        # covs_SS_4D = (cov_WL_SS_4D, cov_GC_SS_4D, cov_3x2pt_SS_4D, cov_WA_SS_4D)
        covs_SS_2D = (cov_WL_SS_2D, cov_GC_SS_2D, cov_3x2pt_SS_2D, cov_WA_SS_2D)

        for probe_name, cov_SS_2D in zip(probe_names, covs_SS_2D):
            cov_dict[f'cov_{probe_name}_SS_2D'] = cov_SS_2D  # cov_dict[f'cov_{probe_name}_SS_4D'] = cov_SS_4D

    for probe_name, cov_GO_4D, cov_GO_2D, cov_GS_4D, cov_GS_2D in zip(probe_names, covs_GO_4D, covs_GO_2D, covs_GS_4D,
                                                                      covs_GS_2D):
        # save 4D
        # cov_dict[f'cov_{probe_name}_GO_4D'] = cov_GO_4D
        # cov_dict[f'cov_{probe_name}_GS_4D'] = cov_GS_4D
        # if covariance_cfg['save_cov_SSC']:

        # save 2D
        cov_dict[f'cov_{probe_name}_GO_2D'] = cov_GO_2D
        cov_dict[f'cov_{probe_name}_GS_2D'] = cov_GS_2D

    # '2DCLOE', i.e. the 'multi-diagonal', non-square blocks ordering, only for 3x2pt
    # note: we found out that this is not actually used in CLOE...
    if covariance_cfg['save_2DCLOE']:
        cov_dict[f'cov_3x2pt_GO_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_GO_4D, nbl_3x2pt, zbins)
        cov_dict[f'cov_3x2pt_SS_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_SS_4D, nbl_3x2pt, zbins)
        cov_dict[f'cov_3x2pt_GS_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_GS_4D, nbl_3x2pt, zbins)

    return cov_dict


def build_X_matrix_BNT(BNT_matrix):
    """
    Builds the X matrix for the BNT transform, according to eq.
    :param BNT_matrix:
    :return:
    """
    X = {}
    delta_kron = np.eye(BNT_matrix.shape[0])
    X['L', 'L'] = np.einsum('ae, bf -> aebf', BNT_matrix, BNT_matrix)
    X['G', 'G'] = np.einsum('ae, bf -> aebf', delta_kron, delta_kron)
    X['G', 'L'] = np.einsum('ae, bf -> aebf', delta_kron, BNT_matrix)
    X['L', 'G'] = np.einsum('ae, bf -> aebf', BNT_matrix, delta_kron)
    return X


def cov_BNT_transform(cov_noBNT_6D, X_dict, probe_A, probe_B, probe_C, probe_D, optimize=True):
    """same as above, but only for one probe (i.e., LL or GL: GG is not modified by the BNT)"""
    cov_BNT_6D = np.einsum('aebf, cgdh, LMefgh -> LMabcd', X_dict[probe_A, probe_B], X_dict[probe_C, probe_D],
                           cov_noBNT_6D, optimize=optimize)
    return cov_BNT_6D


def cov_3x2pt_BNT_transform(cov_3x2pt_dict_10D, X_dict, optimize=True):
    """in np.einsum below, L and M are the ell1, ell2 indices, which are not touched by the BNT transform"""

    cov_3x2pt_BNT_dict_10D = {}

    for probe_A, probe_B, probe_C, probe_D in cov_3x2pt_dict_10D.keys():
        cov_3x2pt_BNT_dict_10D[probe_A, probe_B, probe_C, probe_D] = cov_BNT_transform(
            cov_3x2pt_dict_10D[probe_A, probe_B, probe_C, probe_D], X_dict, probe_A, probe_B, probe_C, probe_D,
            optimize=optimize)

    return cov_3x2pt_BNT_dict_10D


# @njit
def cov_ell_cut(cov_6d, ell_cuts_idxs_AB, ell_cuts_idxs_CD, zbins):
    # TODO pythonize this
    for zi in range(zbins):
        for zj in range(zbins):
            for zk in range(zbins):
                for zl in range(zbins):
                    for ell1 in ell_cuts_idxs_AB[zi, zj]:
                        for ell2 in ell_cuts_idxs_CD[zk, zl]:
                            if ell1 < cov_6d.shape[0] and ell2 < cov_6d.shape[1]:
                                cov_6d[ell1, ell2, zi, zj, zk, zl] = 0

    # pythonic version?
    # ell_idxs_tocut = np.array(ell_cuts_idxs_LL)  # convert list of lists to numpy array
    # idx_pairs = itertools.product(range(zbins), repeat=4)
    # ell_pairs = [(ell1, ell2) for ell1, ell2 in zip(*np.where(ell_idxs_tocut))]
    # for (zi, zj, zk, zl), (ell1, ell2) in zip(idx_pairs, ell_pairs):
    #     covariance_matrix[ell1, ell2, zi, zj, zk, zl] = 0

    return cov_6d


def compute_BNT_matrix(zbins, zgrid_n_of_z, n_of_z_arr, cosmo_ccl, plot_nz=True):
    """
    Computes the BNT matrix. Shamelessly stolen from Santiago's implementation in CLOE
    :param zbins:
    :param zgrid_n_of_z:
    :param n_of_z_arr:
    :param plot_nz:
    :return: BNT matrix, of shape (zbins x zbins)
    """

    if n_of_z_arr is None:
        assert zbins == 10, 'Only 10 zbins are currently supported, because the analytical n(z) is the ISTF one' \
                            '(i.e., with IST:F edges)'
        z_grid = np.linspace(1e-5, 3, 1000)
        n_of_z_arr = np.array([wf_cl_lib.niz_unnormalized_simps(z_grid, zbin_idx) for zbin_idx in range(zbins)])
        n_of_z_arr = np.array(
            [wf_cl_lib.normalize_niz_simps(n_of_z_arr[zbin_idx], z_grid) for zbin_idx in range(zbins)])
    elif (n_of_z_arr is None) ^ (zgrid_n_of_z is None):
        raise ValueError('Either both n_of_z_arr and zgrid_n_of_z must be None, or both must be not None')
    else:
        assert n_of_z_arr.shape[0] == len(zgrid_n_of_z), 'n_of_z must have zgrid_n_of_z rows'
        assert n_of_z_arr.shape[1] == zbins, 'n_of_z must have zbins columns'
        z_grid = zgrid_n_of_z
        if z_grid[0] == 0:
            warnings.warn('z_grid starts at 0, which gives a null comoving distance. '
                          'Removing the first element from the grid')
            z_grid = z_grid[1:]
            n_of_z_arr = n_of_z_arr[1:, :]

    chi = csmlib.ccl_comoving_distance(z_grid, use_h_units=False, cosmo_ccl=cosmo_ccl)

    if plot_nz:
        plt.figure()
        for zi in range(zbins):
            plt.plot(z_grid, n_of_z_arr[:, zi], label=f'zbin {zi}')
        plt.title('n(z) used for BNT computation')
        plt.grid()
        plt.legend()

    A_list = np.zeros(zbins)
    B_list = np.zeros(zbins)
    for zbin_idx in range(zbins):
        n_of_z = n_of_z_arr[:, zbin_idx]
        A_list[zbin_idx] = simps(y=n_of_z, x=z_grid)
        B_list[zbin_idx] = simps(y=n_of_z / chi, x=z_grid)

    bnt_matrix = np.eye(zbins)
    bnt_matrix[1, 0] = -1.
    for i in range(2, zbins):
        mat = np.array([[A_list[i - 1], A_list[i - 2]], [B_list[i - 1], B_list[i - 2]]])
        A = -1. * np.array([A_list[i], B_list[i]])
        soln = np.dot(np.linalg.inv(mat), A)
        bnt_matrix[i, i - 1] = soln[0]
        bnt_matrix[i, i - 2] = soln[1]

    return bnt_matrix


# @njit(parallel=True)
# def cov_ell_cut(cov_6d, ell_cuts_idxs_AB, ell_cuts_idxs_CD, zbins):
#     for zi in prange(zbins):
#         for zj in prange(zbins):
#             for zk in prange(zbins):
#                 for zl in prange(zbins):
#                     for ell1 in ell_cuts_idxs_AB[zi, zj]:
#                         for ell2 in ell_cuts_idxs_CD[zk, zl]:
#                             if ell1 < cov_6d.shape[0] and ell2 < cov_6d.shape[1]:
#                                 cov_6d[ell1, ell2, zi, zj, zk, zl] = 0
#
#     return cov_6d


def save_cov(cov_folder, covariance_cfg, cov_dict, cases_tosave, **variable_specs):
    # TODO skip the computation and saving if the file already exists
    if not covariance_cfg['save_cov']:
        return

    ell_max_WL = variable_specs['ell_max_WL']
    ell_max_GC = variable_specs['ell_max_GC']
    ell_max_XC = variable_specs['ell_max_XC']
    nbl_WL = variable_specs['nbl_WL']
    nbl_GC = variable_specs['nbl_GC']
    nbl_3x2pt = variable_specs['nbl_3x2pt']
    nbl_WA = variable_specs['nbl_WA']

    # which file format to use
    if covariance_cfg['cov_file_format'] == 'npy':
        save_funct = np.save
        extension = 'npy'
    elif covariance_cfg['cov_file_format'] == 'npz':
        save_funct = np.savez_compressed
        extension = 'npz'
    else:
        raise ValueError('cov_file_format not recognized: must be "npy" or "npz"')

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

            for which_cov in cases_tosave:

                for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                    cov_filename = covariance_cfg['cov_filename'].format(which_cov=which_cov, probe=probe,
                                                                         ell_max=ell_max, nbl=nbl, ndim=ndim,
                                                                         **variable_specs)
                    save_funct(f'{cov_folder}/{cov_filename}.{extension}',
                               cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'])  # save in .npy or .npz

                # in this case, 3x2pt is saved in 10D as a dictionary
                # TODO these pickle files are too heavy, probably it's best to revert to npz
                if ndim == 6:
                    cov_3x2pt_filename = covariance_cfg['cov_filename'].format(which_cov=which_cov, probe='3x2pt',
                                                                               ell_max=ell_max_XC, nbl=nbl_3x2pt,
                                                                               ndim=10, **variable_specs)
                    with open(f'{cov_folder}/{cov_3x2pt_filename}.pickle', 'wb') as handle:
                        pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)

            print(f'Covariance matrices saved in {covariance_cfg["cov_file_format"]}')

    # save in .dat for Vincenzo, only in the optimistic case and in 2D
    if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
        for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
            for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):
                cov_filename_vincenzo = covariance_cfg['cov_filename_vincenzo'].format(probe_vinc=probe_vinc,
                                                                                       GOGS_filename=GOGS_filename,
                                                                                       **variable_specs)
                np.savetxt(f'{cov_folder_vincenzo}/{GOGS_folder}/{cov_filename_vincenzo}',
                           cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.9e')
        print('Covariance matrices saved')
