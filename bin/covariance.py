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

matplotlib.use('Qt5Agg')

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm
import cosmo_lib

sys.path.append(str(project_path_here.parent / 'cl_v2/bin'))
import wf_cl_lib


###############################################################################
################ CODE TO COMPUTE THE G AND SSC COVMATS ########################
###############################################################################


def compute_cov(general_cfg, covariance_cfg, ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, BNT_matrix):
    """
    This code computes the Gaussian-only, SSC-only and Gaussian+SSC
    covariance matrices, for different ordering options
    """

    # import settings:
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
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

    # this is a check to make sure that XC has the ordering (L, G) or (G, L) specified by GL_or_LG, and it
    # only works for the (LL, XC, GG) ordering
    probe_ordering = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))

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

    # # delete the 6D matrices to free memory
    # del cov_WL_GO_6D, cov_GC_GO_6D, cov_WA_GO_6D, cov_3x2pt_GO_10D
    # gc.collect()

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

    if covariance_cfg['test_exact_SSC']:
        warnings.warn('Computing GS with exact SSC covariance by dadde\'s code')
        cov_WL_SS_6D = np.load(f'/Users/davide/Documents/Lavoro/Programmi/exact_SSC/output/SSC_matrix/'
                               f'cov_SSC_LLLL_6D_zbins{zbins}_ellbins{nbl_WL}'
                               f'_julia_conventionPySSC.npy')
        cov_WL_SS_6D = np.load(f'/Users/davide/Documents/Lavoro/Programmi/PyCCL_SSC/output/covmat/after_script_update/cov_PyCCL_SSC_LL_nbl20_ellmax3000_HMrecipeKrause2017_6D.npy')


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

    # ! 6d cov ell cuts, deprecated
    # if covariance_cfg['cov_ell_cuts']:
    #     assert False, 'Cov ell cuts in 6D are deprecated'
    #     print('Performing ell cuts on covariance matrix...')
    #     # ! get the ell indices which will be set to 0 for each zi, zj
    #     ell_cuts_dict = ell_dict['ell_cuts_dict']
    #     ell_cuts_idxs_LL = cl_preprocessing.get_ell_cuts_indices(ell_WL, ell_cuts_dict['WL'], zbins)
    #     ell_cuts_idxs_WA = cl_preprocessing.get_ell_cuts_indices(ell_WA, ell_cuts_dict['WL'], zbins)
    #     ell_cuts_idxs_GG = cl_preprocessing.get_ell_cuts_indices(ell_GC, ell_cuts_dict['GC'], zbins)
    #     ell_cuts_idxs_GL = cl_preprocessing.get_ell_cuts_indices(ell_GC, ell_cuts_dict['GL'], zbins)
    #
    #     # ! perform the cuts: single-probe
    #     cov_WL_GO_6D = cov_ell_cut(cov_WL_GO_6D, ell_cuts_idxs_LL, ell_cuts_idxs_LL, zbins)
    #     cov_GC_GO_6D = cov_ell_cut(cov_GC_GO_6D, ell_cuts_idxs_GG, ell_cuts_idxs_GG, zbins)
    #     cov_WA_GO_6D = cov_ell_cut(cov_WA_GO_6D, ell_cuts_idxs_WA, ell_cuts_idxs_WA, zbins)
    #
    #     cov_WL_GS_6D = cov_ell_cut(cov_WL_GS_6D, ell_cuts_idxs_LL, ell_cuts_idxs_LL, zbins)
    #     cov_GC_GS_6D = cov_ell_cut(cov_GC_GS_6D, ell_cuts_idxs_GG, ell_cuts_idxs_GG, zbins)
    #     cov_WA_GS_6D = cov_ell_cut(cov_WA_GS_6D, ell_cuts_idxs_WA, ell_cuts_idxs_WA, zbins)
    #
    #     # ! perform the cuts: 3x2pt (define a dictionary of ell_cuts_idxs to be able to use a loop)
    #     ell_cuts_idxs_dict = {
    #         ('L', 'L'): ell_cuts_idxs_LL,
    #         ('G', 'L'): ell_cuts_idxs_GL,
    #         ('G', 'G'): ell_cuts_idxs_GG,
    #     }
    #     for A, B in probe_ordering:
    #         for C, D in probe_ordering:
    #             cov_3x2pt_GO_10D_dict[A, B, C, D] = cov_ell_cut(cov_3x2pt_GO_10D_dict[A, B, C, D], ell_cuts_idxs_dict[A, B], ell_cuts_idxs_dict[C, D], zbins)
    #             cov_3x2pt_GS_10D_dict[A, B, C, D] = cov_ell_cut(cov_3x2pt_GS_10D_dict[A, B, C, D], ell_cuts_idxs_dict[A, B], ell_cuts_idxs_dict[C, D], zbins)

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

    # TODO finish this PYCCL stuff
    # cov_WL_SS_4D_pyssc = np.copy(cov_WL_SS_4D)
    # cov_GC_SS_4D_pyssc = np.copy(cov_GC_SS_4D)
    # cov_WL_SS_2D_pyssc = mm.cov_4D_to_2D(cov_WL_SS_4D_pyssc, block_index=block_index)
    # cov_GC_SS_2D_pyssc = mm.cov_4D_to_2D(cov_GC_SS_4D_pyssc, block_index=block_index)

    if SSC_code == 'PyCCL':
        print('Computing GS with PyCCL SSC covariance')
        # assert covariance_cfg['compute_cov_6D'] is False, 'compute_cov_6D must be False when using, because cov_GS_4D' \
        #                                                   ' gets overwritten below. Fix this.'

        # TODO for now, load the existing files; then, compute the SSC cov properly
        fldr = covariance_cfg["cov_SSC_PyCCL_folder"]
        filename = covariance_cfg["cov_SSC_PyCCL_filename"]

        cov_WL_SS_6D = np.load(f'{fldr}/{filename.format(probe="LL", nbl=nbl_WL, ell_max=ell_max_WL)}')
        cov_GC_SS_6D = np.load(f'{fldr}/{filename.format(probe="GG", nbl=nbl_GC, ell_max=ell_max_GC)}')
        cov_3x2pt_SS_10D_arr = np.load(f'{fldr}/{filename.format(probe="3x2pt", nbl=nbl_GC, ell_max=ell_max_GC)}')
        # ! transform into a dict to be able to reshape to 4D, this is a very ugly way to do it
        cov_3x2pt_SS_10D_dict = {
            ('L', 'L', 'L', 'L'): cov_3x2pt_SS_10D_arr[0, 0, 0, 0],
            ('L', 'L', 'G', 'L'): cov_3x2pt_SS_10D_arr[0, 0, 1, 0],
            ('L', 'L', 'G', 'G'): cov_3x2pt_SS_10D_arr[0, 0, 1, 1],
            ('G', 'L', 'L', 'L'): cov_3x2pt_SS_10D_arr[1, 0, 0, 0],
            ('G', 'L', 'G', 'L'): cov_3x2pt_SS_10D_arr[1, 0, 1, 0],
            ('G', 'L', 'G', 'G'): cov_3x2pt_SS_10D_arr[1, 0, 1, 1],
            ('G', 'G', 'L', 'L'): cov_3x2pt_SS_10D_arr[1, 1, 0, 0],
            ('G', 'G', 'G', 'L'): cov_3x2pt_SS_10D_arr[1, 1, 1, 0],
            ('G', 'G', 'G', 'G'): cov_3x2pt_SS_10D_arr[1, 1, 1, 1],
        }
        # cov_3x2pt_SS_6D = mm.load_pickle(f'{fldr}/{filename.format(probe="3x2pt", nbl=nbl_GC, ell_max=ell_max_GC)}')

        # reshape to 4D
        cov_WL_SS_4D = mm.cov_6D_to_4D(cov_WL_SS_6D, nbl_WL, zpairs_auto, ind=ind_auto)
        cov_GC_SS_4D = mm.cov_6D_to_4D(cov_GC_SS_6D, nbl_GC, zpairs_auto, ind=ind_auto)
        cov_3x2pt_SS_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SS_10D_dict, probe_ordering, nbl_GC, zbins, ind.copy(),
                                                 GL_or_LG)

        # ! better way, to be tested:
        # if it works, delete the v2 and give as input the array (you can delete this ugly dict creation)
        cov_3x2pt_SS_4D_v2 = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SS_10D_arr, probe_ordering, nbl_GC, zbins, ind.copy(),
                                                    GL_or_LG)
        assert np.array_equal(cov_3x2pt_SS_4D,
                              cov_3x2pt_SS_4D_v2), 'cov_3x2pt_SS_4D and cov_3x2pt_SS_4D_v2 are not equal'

    if covariance_cfg['compute_cov_6D']:
        assert False, 'now I compute the covariance in 6D (with einsum) in all cases!'

        # compute 3x2pt covariance in 10D, potentially with whichever probe ordering, and the WL, GS and WA cov in 6D

        # store the input datavector and noise spectra in a dictionary
        cl_dict_3x2pt = mm.build_3x2pt_dict(cl_3x2pt_5D)
        rl_dict_3x2pt = mm.build_3x2pt_dict(rl_3x2pt_5D)
        noise_dict_3x2pt = mm.build_3x2pt_dict(noise_3x2pt_4D)
        Sijkl_dict = mm.build_Sijkl_dict(Sijkl, zbins)

        # probe ordering
        # the function should be able to work with whatever 
        # ordering of the probes; (TODO check this)

        # print as a check
        print('check: datavector probe ordering:', probe_ordering)
        print('check: probe combinations:')
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                print(A, B, C, D)

        # compute the 10D covariance only for the blocks which will actually be used (GO and SS)
        start = time.perf_counter()
        cov_3x2pt_GO_10D_dict = mm.cov_G_10D_dict(cl_dict_3x2pt, noise_dict_3x2pt, nbl_3x2pt, zbins, ell_3x2pt,
                                                  delta_l_3x2pt, fsky, probe_ordering)
        print(f'cov_3x2pt_GO_10D_dict computed in {(time.perf_counter() - start):.2f} s')

        start = time.perf_counter()
        cov_3x2pt_SS_10D_dict = mm.cov_SS_10D_dict(cl_dict_3x2pt, rl_dict_3x2pt, Sijkl_dict, nbl_3x2pt, zbins, fsky,
                                                   probe_ordering)
        print(f'cov_3x2pt_SS_10D_dict computed in {(time.perf_counter() - start):.2f} s')

        # sum GO and SS
        cov_3x2pt_GS_10D_dict = {}
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                cov_3x2pt_GS_10D_dict[A, B, C, D] = np.zeros((nbl_3x2pt, nbl_3x2pt, zbins, zbins, zbins, zbins))

        for A, B in probe_ordering:
            for C, D in probe_ordering:
                cov_3x2pt_GS_10D_dict[A, B, C, D][...] = cov_3x2pt_GO_10D_dict[A, B, C, D][...] + \
                                                         cov_3x2pt_SS_10D_dict[A, B, C, D][...]

        # this is to revert from 10D to 4D, which is trickier for the 3x2pt (each block has to be converted to 4D and
        # stacked to make the 4D_3x2pt)
        """
        # note: I pass ind_copy because the LG-GL check and inversion is performed in the function (otherwise it would be
        # performed twice!)
        # ! careful of passing clean copies of ind to both functions!!!
        cov_3x2pt_GO_4D_new = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_GO_10D_dict, probe_ordering, nbl_3x2pt, zbins,
                                                          ind.copy(), GL_or_LG)
        cov_3x2pt_SS_4D_new = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_SS_10D_dict, probe_ordering, nbl_3x2pt, zbins,
                                                          ind.copy(), GL_or_LG)
        cov_3x2pt_GS_4D_new = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_GS_10D_dict, probe_ordering, nbl_3x2pt, zbins,
                                                          ind.copy(), GL_or_LG)

        # check with old result and show the arrays 
        print('check: is cov_3x2pt_GO_4D from covariance_10D_dict function == old one?',
              np.array_equal(cov_3x2pt_GO_4D_new, cov_3x2pt_GO_4D))
        print('check: is cov_3x2pt_SS_4D from covariance_10D_dict function == old one?',
              np.array_equal(cov_3x2pt_SS_4D_new, cov_3x2pt_SS_4D))
        print('check: is cov_3x2pt_GS_4D from covariance_10D_dict function == old one?',
              np.array_equal(cov_3x2pt_GS_4D_new, cov_3x2pt_GS_4D))
        """

        # TODO implement the other covmats in this module!
        # if use_PyCCL_SS
        # if use_PyCCL_cNG:

        # save the 6D-10D covs in the dictionary
        cov_dict['cov_3x2pt_GO_10D_dict'] = cov_3x2pt_GO_10D_dict
        cov_dict['cov_3x2pt_GS_10D_dict'] = cov_3x2pt_GS_10D_dict
        cov_dict['cov_3x2pt_SS_10D_dict'] = cov_3x2pt_SS_10D_dict

        # this is the 1st way to compute cov_6D: simply transform the cov_4D array (note that cov_4D_to_6D does not
        # work for 3x2pt, althought it should be easy to implement). Quite slow for GS or SS matrices.
        # example:
        # cov_dict['cov_WL_GO_6D'] = mm.cov_4D_to_6D(cov_WL_GO_4D, nbl_WL, zbins, probe='LL', ind=ind_auto)

        # the cov_G_10D_dict function takes as input a dict, not an array: this is just to create them.
        # note: the only reason why I cannot pass the pre-built cl_dict_3x2pt dictionary is that it contains the probes
        # up to ell_max_XC, so WL (and WA?) will have a different ell_max than WL_only.
        cl_dict_LL = {('L', 'L'): cl_LL_3D}
        cl_dict_GG = {('G', 'G'): cl_GG_3D}
        cl_dict_WA = {('L', 'L'): cl_WA_3D}

        rl_dict_LL = {('L', 'L'): rl_LL_3D}
        rl_dict_GG = {('G', 'G'): rl_GG_3D}
        rl_dict_WA = {('L', 'L'): rl_WA_3D}

        # ! cov_G_6D
        start_time = time.perf_counter()
        cov_dict['cov_WL_GO_6D'] = mm.cov_G_10D_dict(cl_dict_LL, noise_dict_3x2pt,
                                                     nbl_WL, zbins, ell_WL, delta_l_WL, fsky,
                                                     probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        cov_dict['cov_GC_GO_6D'] = mm.cov_G_10D_dict(cl_dict_GG, noise_dict_3x2pt,
                                                     nbl_GC, zbins, ell_GC, delta_l_GC, fsky,
                                                     probe_ordering=[['G', 'G'], ])['G', 'G', 'G', 'G']
        cov_dict['cov_WA_GO_6D'] = mm.cov_G_10D_dict(cl_dict_WA, noise_dict_3x2pt,
                                                     nbl_WA, zbins, ell_WA, delta_l_WA, fsky,
                                                     probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        print(f'cov_GO_6D computed in {(time.perf_counter() - start_time):.2f} s')

        # ! cov_SSC_6D
        start_time = time.perf_counter()
        cov_WL_SS_6D = mm.cov_SS_10D_dict(cl_dict_LL, rl_dict_LL, Sijkl_dict, nbl_WL, zbins, fsky,
                                          probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        cov_GC_SS_6D = mm.cov_SS_10D_dict(cl_dict_GG, rl_dict_GG, Sijkl_dict, nbl_GC, zbins, fsky,
                                          probe_ordering=[['G', 'G'], ])['G', 'G', 'G', 'G']
        cov_WA_SS_6D = mm.cov_SS_10D_dict(cl_dict_WA, rl_dict_WA, Sijkl_dict, nbl_WA, zbins, fsky,
                                          probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        print(f'cov_SS_6D computed in {(time.perf_counter() - start_time):.2f} s')


        if covariance_cfg['save_cov_SSC']:
            cov_dict['cov_WL_SS_6D'] = cov_WL_SS_6D
            cov_dict['cov_GC_SS_6D'] = cov_GC_SS_6D
            cov_dict['cov_WA_SS_6D'] = cov_WA_SS_6D

        # ! cov_GS_6D
        cov_dict['cov_WL_GS_6D'] = cov_dict['cov_WL_GO_6D'] + cov_WL_SS_6D
        cov_dict['cov_GC_GS_6D'] = cov_dict['cov_GC_GO_6D'] + cov_GC_SS_6D
        cov_dict['cov_WA_GS_6D'] = cov_dict['cov_WA_GO_6D'] + cov_WA_SS_6D

        # ! BNT transform
        if covariance_cfg['cov_BNT_transform']:
            print('BNT-transforming the covariance matrix...')

            X_dict = build_X_matrix_BNT(BNT_matrix)
            cov_dict['cov_WL_GO_6D'] = cov_BNT_transform(cov_dict['cov_WL_GO_6D'], X_dict, 'L', 'L', 'L', 'L')
            cov_dict['cov_WA_GO_6D'] = cov_BNT_transform(cov_dict['cov_WA_GO_6D'], X_dict, 'L', 'L', 'L', 'L')
            cov_dict['cov_3x2pt_GO_10D_dict'] = cov_3x2pt_BNT_transform(cov_dict['cov_3x2pt_GO_10D_dict'], X_dict)

            cov_dict['cov_WL_GS_6D'] = cov_BNT_transform(cov_dict['cov_WL_GS_6D'], X_dict, 'L', 'L', 'L', 'L')
            cov_dict['cov_WA_GS_6D'] = cov_BNT_transform(cov_dict['cov_WA_GS_6D'], X_dict, 'L', 'L', 'L', 'L')
            cov_dict['cov_3x2pt_GS_10D_dict'] = cov_3x2pt_BNT_transform(cov_dict['cov_3x2pt_GS_10D_dict'], X_dict)

        # if covariance_cfg['cov_ell_cuts']:

        # print('Performing ell cuts on covariance matrix...')
        # # ! get the ell indices which will be set to 0 for each zi, zj
        # ell_cuts_dict = ell_dict['ell_cuts_dict']
        # ell_cuts_idxs_LL = cl_preprocessing.get_ell_cuts_indices(l_lin_WL, ell_cuts_dict['WL'], zbins)
        # ell_cuts_idxs_WA = cl_preprocessing.get_ell_cuts_indices(l_lin_WA, ell_cuts_dict['WL'], zbins)
        # ell_cuts_idxs_GG = cl_preprocessing.get_ell_cuts_indices(l_lin_GC, ell_cuts_dict['GC'], zbins)
        # ell_cuts_idxs_GL = cl_preprocessing.get_ell_cuts_indices(l_lin_GC, ell_cuts_dict['GL'], zbins)
        #
        # # ! perform the cuts: single-probe
        # cov_dict['cov_WL_GO_6D'] = cov_ell_cut(cov_dict['cov_WL_GO_6D'], ell_cuts_idxs_LL, ell_cuts_idxs_LL, zbins)
        # cov_dict['cov_WA_GO_6D'] = cov_ell_cut(cov_dict['cov_WA_GO_6D'], ell_cuts_idxs_WA, ell_cuts_idxs_WA, zbins)
        # cov_dict['cov_GC_GO_6D'] = cov_ell_cut(cov_dict['cov_GC_GO_6D'], ell_cuts_idxs_GG, ell_cuts_idxs_GG, zbins)
        #
        # # ! perform the cuts: 3x2pt (define a dictionary of ell_cuts_idxs to be able to use a loop)
        # ell_cuts_idxs_dict = {
        #     ('L', 'L'): ell_cuts_idxs_LL,
        #     ('G', 'L'): ell_cuts_idxs_GL,
        #     ('G', 'G'): ell_cuts_idxs_GG,
        # }
        # for A, B in probe_ordering:
        #     for C, D in probe_ordering:
        #         cov_dict['cov_3x2pt_GO_10D_dict'][A, B, C, D] = cov_ell_cut(
        #             cov_dict['cov_3x2pt_GO_10D_dict'][A, B, C, D],
        #             ell_cuts_idxs_dict[A, B], ell_cuts_idxs_dict[C, D], zbins)

        # if not converted in 4D, only the 6D covs will be overwritten by the BNT-transofrmed version!
        cov_WL_GO_4D = mm.cov_6D_to_4D(cov_dict['cov_WL_GO_6D'], nbl_WL, zpairs_auto, ind_auto)
        cov_GC_GO_4D = mm.cov_6D_to_4D(cov_dict['cov_GC_GO_6D'], nbl_GC, zpairs_auto, ind_auto)
        cov_WA_GO_4D = mm.cov_6D_to_4D(cov_dict['cov_WA_GO_6D'], nbl_WA, zpairs_auto, ind_auto)
        cov_3x2pt_GO_4D = mm.cov_3x2pt_10D_to_4D(cov_dict['cov_3x2pt_GO_10D_dict'], probe_ordering, nbl_GC, zbins,
                                                 ind.copy(), GL_or_LG)

        if covariance_cfg['compute_SSC']:
            cov_WL_GS_4D = mm.cov_6D_to_4D(cov_dict['cov_WL_GS_6D'], nbl_WL, zpairs_auto, ind_auto)
            cov_GC_GS_4D = mm.cov_6D_to_4D(cov_dict['cov_GC_GS_6D'], nbl_GC, zpairs_auto, ind_auto)
            cov_WA_GS_4D = mm.cov_6D_to_4D(cov_dict['cov_WA_GS_6D'], nbl_WA, zpairs_auto, ind_auto)
            cov_3x2pt_GS_4D = mm.cov_3x2pt_10D_to_4D(cov_dict['cov_3x2pt_GS_10D_dict'], probe_ordering, nbl_GC, zbins,
                                                     ind.copy(), GL_or_LG)
        else:
            warnings.warn('SSC not computed, setting SSC covariances to identity...')
            cov_WL_GS_4D = np.eye(cov_WL_GO_4D.shape)
            cov_GC_GS_4D = np.eye(cov_GC_GO_4D.shape)
            cov_WA_GS_4D = np.eye(cov_WA_GO_4D.shape)
            cov_3x2pt_GS_4D = np.eye(cov_3x2pt_GO_4D.shape)

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

    ############################### save in dictionary  ########################
    probe_names = ('WL', 'GC', '3x2pt', 'WA')

    covs_GO_4D = (cov_WL_GO_4D, cov_GC_GO_4D, cov_3x2pt_GO_4D, cov_WA_GO_4D)
    covs_GS_4D = (cov_WL_GS_4D, cov_GC_GS_4D, cov_3x2pt_GS_4D, cov_WA_GS_4D)

    covs_GO_2D = (cov_WL_GO_2D, cov_GC_GO_2D, cov_3x2pt_GO_2D, cov_WA_GO_2D)
    covs_GS_2D = (cov_WL_GS_2D, cov_GC_GS_2D, cov_3x2pt_GS_2D, cov_WA_GS_2D)

    if covariance_cfg['save_cov_SSC']:
        cov_WL_SS_4D = mm.cov_6D_to_4D(cov_WL_SS_6D, nbl_WL, zpairs_auto, ind_auto)
        cov_GC_SS_4D = mm.cov_6D_to_4D(cov_GC_SS_6D, nbl_GC, zpairs_auto, ind_auto)
        cov_WA_SS_4D = mm.cov_6D_to_4D(cov_WA_SS_6D, nbl_WA, zpairs_auto, ind_auto)
        cov_3x2pt_SS_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SS_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(),
                                                 GL_or_LG)

        cov_WL_SS_2D = mm.cov_4D_to_2D(cov_WL_SS_4D, block_index=block_index)
        cov_GC_SS_2D = mm.cov_4D_to_2D(cov_GC_SS_4D, block_index=block_index)
        cov_WA_SS_2D = mm.cov_4D_to_2D(cov_WA_SS_4D, block_index=block_index)
        cov_3x2pt_SS_2D = mm.cov_4D_to_2D(cov_3x2pt_SS_4D, block_index=block_index)

        covs_SS_4D = (cov_WL_SS_4D, cov_GC_SS_4D, cov_3x2pt_SS_4D, cov_WA_SS_4D)
        covs_SS_2D = (cov_WL_SS_2D, cov_GC_SS_2D, cov_3x2pt_SS_2D, cov_WA_SS_2D)

        for probe_name, cov_SS_2D in zip(probe_names, covs_SS_2D):
            cov_dict[f'cov_{probe_name}_SS_2D'] = cov_SS_2D
            #     cov_dict[f'cov_{probe_name}_SS_4D'] = cov_SS_4D

    for probe_name, cov_GO_4D, cov_GO_2D, cov_GS_4D, cov_GS_2D \
            in zip(probe_names, covs_GO_4D, covs_GO_2D, covs_GS_4D, covs_GS_2D):
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
        cov_3x2pt_BNT_dict_10D[probe_A, probe_B, probe_C, probe_D] = \
            cov_BNT_transform(cov_3x2pt_dict_10D[probe_A, probe_B, probe_C, probe_D], X_dict,
                              probe_A, probe_B, probe_C, probe_D, optimize=optimize)

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


def compute_BNT_matrix(zbins, zgrid_n_of_z, n_of_z_arr, plot_nz=True):
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

    warnings.warn('I am assuming an IST:F fiducial cosmology to compute the comoving distance')
    chi = cosmo_lib.ccl_comoving_distance(z_grid, use_h_units=False)

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
        A_list[zbin_idx] = simps(n_of_z, z_grid)
        B_list[zbin_idx] = simps(n_of_z / chi, z_grid)

    bnt_matrix = np.eye(zbins)
    bnt_matrix[1, 0] = -1.
    for i in range(2, zbins):
        mat = np.array([[A_list[i - 1], A_list[i - 2]],
                        [B_list[i - 1], B_list[i - 2]]])
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


def save_cov(cov_folder, covariance_cfg, cov_dict, **variable_specs):
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

    # which cases to save: GO, GS or GO, GS and SS
    cases_tosave = ['GO', 'GS']
    if covariance_cfg[f'save_cov_GS']:
        cases_tosave.append('GS')
    if covariance_cfg[f'save_cov_SSC']:
        cases_tosave.append('SS')

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

                # save all covmats in the optimistic case
                if ell_max_WL == 5000:
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

                # in the pessimistic case, save only WA
                elif ell_max_WL == 1500:
                    cov_WA_filename = covariance_cfg['cov_filename'].format(which_cov=which_cov, probe='WA',
                                                                            ell_max=ell_max_WL, nbl=nbl_WA, ndim=ndim,
                                                                            **variable_specs)
                    np.save(f'{cov_folder}/{cov_WA_filename}.{extension}', cov_dict[f'cov_WA_{which_cov}_{ndim}D'])
            print('Covariance matrices saved')

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
