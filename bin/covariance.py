import pdb
import sys
import time
import warnings
from pathlib import Path
import pickle

import matplotlib
import numpy as np
from numba import njit, prange

import cl_preprocessing

matplotlib.use('Qt5Agg')

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm


###############################################################################
################ CODE TO COMPUTE THE G AND SSC COVMATS ########################
###############################################################################


def compute_cov(general_cfg, covariance_cfg, ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, BNT_matrix=None):
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
    probe_ordering = [['L', 'L'], [None, None], ['G', 'G']]

    # (not the best) check to ensure that the (LL, XC, GG) ordering is respected
    assert probe_ordering[0] == ['L', 'L'], 'the XC probe should be in position 1 (not 0) of the datavector'
    assert probe_ordering[2] == ['G', 'G'], 'the XC probe should be in position 1 (not 0) of the datavector'

    # this overwrites the 1st axis, the one describing XC
    probe_ordering[1][0] = GL_or_LG[0]
    probe_ordering[1][1] = GL_or_LG[1]

    start = time.perf_counter()

    # import ell values
    ell_WL, nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
    ell_GC, nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
    ell_WA, nbl_WA = ell_dict['ell_WA'], ell_dict['ell_WA'].shape[0]
    ell_XC, nbl_3x2pt = ell_GC, nbl_GC

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
        print('looks like the ell values are in log scale. Switching to linear scale.')
        l_lin_WL = 10 ** ell_WL
        l_lin_GC = 10 ** ell_GC
        l_lin_WA = 10 ** ell_WA
        l_lin_XC = l_lin_GC
    else:
        l_lin_WL = ell_WL
        l_lin_GC = ell_GC
        l_lin_WA = ell_WA
        l_lin_XC = l_lin_GC

    # load deltas
    delta_l_WL = delta_dict['delta_l_WL']
    delta_l_GC = delta_dict['delta_l_GC']
    delta_l_WA = delta_dict['delta_l_WA']
    delta_l_XC = delta_l_GC

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
    print(
        f'\ncheck: \nind_ordering = {triu_tril}, {rowcol_major} \nblock_index = {block_index}\n'
        f'zbins: {general_cfg["EP_or_ED"]}{zbins}\n'
        f'nbl_WA: {nbl_WA} nbl_WL: {nbl_WL} nbl_GC:  {nbl_GC}, nbl_3x2pt:  {nbl_3x2pt}\n'
        f'ell_max_WL = {ell_max_WL} \nell_max_GC = {ell_max_GC}\nGL_or_LG: {GL_or_LG}\n')

    # build noise vector

    warnings.warn('which folder should I use for ngbTab? lenses or sources? Flagship or Redbook?')
    noise = mm.build_noise(zbins, n_probes, sigma_eps2=covariance_cfg['sigma_eps2'], ng=covariance_cfg['ng'],
                           EP_or_ED=general_cfg['EP_or_ED'])

    ################### COMPUTE GAUSS ONLY COVARIANCE #########################

    # WL only covariance
    cov_WL_GO_4D = mm.covariance(nbl=nbl_WL, npairs=zpairs_auto, start_index=0, stop_index=zpairs_auto,
                                 Cij=cl_LL_3D, noise=noise, l_lin=l_lin_WL,
                                 delta_l=delta_l_WL, fsky=fsky, ind=ind)
    # GC only covariance
    starting_GC_index = zpairs_auto + zpairs_cross
    cov_GC_GO_4D = mm.covariance(nbl=nbl_GC, npairs=zpairs_auto, start_index=starting_GC_index, stop_index=zpairs_3x2pt,
                                 Cij=cl_GG_3D, noise=noise, l_lin=l_lin_GC,
                                 delta_l=delta_l_GC, fsky=fsky, ind=ind)
    # WA covariance
    cov_WA_GO_4D = mm.covariance_WA(nbl_WA, zpairs_auto, start_index=0, stop_index=zpairs_auto,
                                    Cij=cl_WA_3D, noise=noise, l_lin=l_lin_WA,
                                    delta_l=delta_l_WA, fsky=fsky, ind=ind, ell_WA=ell_WA)
    # ALL covariance
    cov_3x2pt_GO_4D = mm.covariance_ALL(nbl=nbl_3x2pt, npairs=zpairs_3x2pt,
                                        Cij=cl_3x2pt_5D, noise=noise, l_lin=l_lin_XC,
                                        delta_l=delta_l_XC, fsky=fsky, ind=ind)
    print("Gauss. cov. matrices computed in %.2f seconds" % (time.perf_counter() - start))

    ######################## COMPUTE SSC COVARIANCE ###############################

    # compute the covariance with PySSC anyway, not to have problems with WA
    start = time.perf_counter()
    # TODO the 4d computation should not be repeated ic compute_cov_6d is True!
    cov_WL_SS_4D = mm.cov_SSC(nbl_WL, zpairs_auto, ind, cl_LL_3D, Sijkl, fsky, "WL", zbins, rl_LL_3D)
    cov_GC_SS_4D = mm.cov_SSC(nbl_GC, zpairs_auto, ind, cl_GG_3D, Sijkl, fsky, "GC", zbins, rl_GG_3D)
    cov_WA_SS_4D = mm.cov_SSC(nbl_WA, zpairs_auto, ind, cl_WA_3D, Sijkl, fsky, "WA", zbins, rl_WA_3D)
    cov_3x2pt_SS_4D = mm.cov_SSC_ALL(nbl_3x2pt, zpairs_3x2pt, ind, cl_3x2pt_5D, Sijkl, fsky, zbins, rl_3x2pt_5D)
    print("SS cov. matrices computed in %.2f seconds with PySSC" % (time.perf_counter() - start))

    cov_WL_SS_4D_pyssc = np.copy(cov_WL_SS_4D)
    cov_GC_SS_4D_pyssc = np.copy(cov_GC_SS_4D)
    cov_WL_SS_2D_pyssc = mm.cov_4D_to_2D(cov_WL_SS_4D_pyssc, block_index=block_index)
    cov_GC_SS_2D_pyssc = mm.cov_4D_to_2D(cov_GC_SS_4D_pyssc, block_index=block_index)

    if SSC_code == 'PyCCL':
        print('Computing GS with PyCCL SSC covariance')
        assert covariance_cfg['compute_cov_6D'] is False, 'compute_cov_6D must be False when using, because cov_GS_4D' \
                                                          ' gets overwritten below. Fix this.'

        # TODO for now, load the existing files; then, compute the SSC cov properly
        fldr = covariance_cfg["cov_SSC_PyCCL_folder"]
        filename = covariance_cfg["cov_SSC_PyCCL_filename"]

        cov_WL_SS_6D = np.load(f'{fldr}/{filename.format(probe="WL", nbl=nbl_WL, ell_max=ell_max_WL)}')
        cov_GC_SS_6D = np.load(f'{fldr}/{filename.format(probe="GC", nbl=nbl_GC, ell_max=ell_max_GC)}')
        # TODO re-establish the 3x2pt
        cov_3x2pt_SS_10D_arr = np.load(f'{fldr}/{filename.format(probe="3x2pt", nbl=nbl_GC, ell_max=ell_max_GC)}')
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
        cov_3x2pt_SS_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_SS_10D_dict, probe_ordering, nbl_GC,
                                                      zbins, ind.copy(), GL_or_LG)

    ############################## SUM G + SSC ################################
    cov_WL_GS_4D = cov_WL_GO_4D + cov_WL_SS_4D
    cov_GC_GS_4D = cov_GC_GO_4D + cov_GC_SS_4D
    cov_WA_GS_4D = cov_WA_GO_4D + cov_WA_SS_4D
    cov_3x2pt_GS_4D = cov_3x2pt_GO_4D + cov_3x2pt_SS_4D

    if covariance_cfg['compute_cov_6D']:

        # compute 3x2pt covariance in 10D, potentially with whichever probe ordering, and the WL, GS and WA cov in 6D

        # store the input datavector and noise spectra in a dictionary
        cl_dict_3x2pt = mm.build_3x2pt_dict(cl_3x2pt_5D)
        rl_dict_3x2pt = mm.build_3x2pt_dict(rl_3x2pt_5D)
        noise_dict_3x2pt = mm.build_3x2pt_dict(noise)
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
        cov_3x2pt_GO_10D_dict = mm.cov_G_10D_dict(cl_dict_3x2pt, noise_dict_3x2pt, nbl_3x2pt, zbins, l_lin_XC,
                                                  delta_l_XC, fsky, probe_ordering)
        print(f'cov_3x2pt_GO_10D_dict computed in {(time.perf_counter() - start):.2f} seconds')

        start = time.perf_counter()
        cov_3x2pt_SS_10D_dict = mm.cov_SS_10D_dict(cl_dict_3x2pt, rl_dict_3x2pt, Sijkl_dict, nbl_3x2pt, zbins, fsky,
                                                   probe_ordering)
        print(f'cov_3x2pt_SS_10D_dict computed in {(time.perf_counter() - start):.2f} seconds')

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

        # TODO use pandas dataframe?
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
                                                     nbl_WL, zbins, l_lin_WL, delta_l_WL, fsky,
                                                     probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        cov_dict['cov_GC_GO_6D'] = mm.cov_G_10D_dict(cl_dict_GG, noise_dict_3x2pt,
                                                     nbl_GC, zbins, l_lin_GC, delta_l_GC, fsky,
                                                     probe_ordering=[['G', 'G'], ])['G', 'G', 'G', 'G']
        cov_dict['cov_WA_GO_6D'] = mm.cov_G_10D_dict(cl_dict_WA, noise_dict_3x2pt,
                                                     nbl_WA, zbins, l_lin_WA, delta_l_WA, fsky,
                                                     probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        print(f'cov_GO_6D computed in {(time.perf_counter() - start_time):.2f} seconds')

        # ! NEW BIT: compute the Gaussian covariance with einsum
        """
        noise_LL_5d = np.zeros((1, 1, nbl_WL, zbins, zbins))  # ! nbl changes from probe to probe!
        noise_WA_5d = np.zeros((1, 1, nbl_WA, zbins, zbins))
        noise_GG_5d = np.zeros((1, 1, nbl_GC, zbins, zbins))
        noise_3x2pt_5d = np.zeros((2, 2, nbl_GC, zbins, zbins))
        cl_3x2pt_5d = cl_3x2pt_5D.transpose(1, 2, 0, 3, 4)

        for ell_idx in range(nbl_WL):
            noise_LL_5d[0, 0, ell_idx, :, :] = noise_dict_3x2pt['L', 'L']
        for ell_idx in range(nbl_WA):
            noise_WA_5d[0, 0, ell_idx, :, :] = noise_dict_3x2pt['L', 'L']
        for ell_idx in range(nbl_GC):
            noise_GG_5d[0, 0, ell_idx, :, :] = noise_dict_3x2pt['G', 'G']
            noise_3x2pt_5d[0, 0, ell_idx, :, :] = noise_dict_3x2pt['L', 'L']
            noise_3x2pt_5d[0, 1, ell_idx, :, :] = noise_dict_3x2pt['L', 'G']
            noise_3x2pt_5d[1, 0, ell_idx, :, :] = noise_dict_3x2pt['G', 'L']
            noise_3x2pt_5d[1, 1, ell_idx, :, :] = noise_dict_3x2pt['G', 'G']

        cl_LL_5d = cl_LL_3D[np.newaxis, np.newaxis, ...]
        cl_WA_5d = cl_WA_3D[np.newaxis, np.newaxis, ...]
        cl_GG_5d = cl_GG_3D[np.newaxis, np.newaxis, ...]

        cov_dict['cov_GO_WL_6D'] = mm.covariance_einsum(cl_LL_5d, noise_LL_5d, fsky, l_lin_WL, delta_l_WL)[0, 0, 0, 0, ...]
        cov_dict['cov_GO_WA_6D'] = mm.covariance_einsum(cl_WA_5d, noise_WA_5d, fsky, l_lin_WA, delta_l_WA)[0, 0, 0, 0, ...]
        cov_dict['cov_GO_GC_6D'] = mm.covariance_einsum(cl_GG_5d, noise_GG_5d, fsky, l_lin_GC, delta_l_GC)[0, 0, 0, 0, ...]
        cov_3x2pt_GO_10D_arr = mm.covariance_einsum(cl_3x2pt_5d, noise_3x2pt_5d, fsky, l_lin_GC, delta_l_GC)
        cov_dict['cov_3x2pt_GO_10D_dict'] = mm.cov_10D_array_to_dict(cov_3x2pt_GO_10D_arr)
        """
        # ! end cov_einsum. Now you might as well just use this, but it's kind of a big update

        # ! cov_SSC_6D
        start_time = time.perf_counter()
        cov_WL_SS_6D = mm.cov_SS_10D_dict(cl_dict_LL, rl_dict_LL, Sijkl_dict, nbl_WL, zbins, fsky,
                                          probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        cov_GC_SS_6D = mm.cov_SS_10D_dict(cl_dict_GG, rl_dict_GG, Sijkl_dict, nbl_GC, zbins, fsky,
                                          probe_ordering=[['G', 'G'], ])['G', 'G', 'G', 'G']
        cov_WA_SS_6D = mm.cov_SS_10D_dict(cl_dict_WA, rl_dict_WA, Sijkl_dict, nbl_WA, zbins, fsky,
                                          probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        print(f'cov_SS_6D computed in {(time.perf_counter() - start_time):.2f} seconds')

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
        cov_3x2pt_GO_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_dict['cov_3x2pt_GO_10D_dict'], probe_ordering, nbl_GC,
                                                      zbins, ind.copy(), GL_or_LG)

        cov_WL_GS_4D = mm.cov_6D_to_4D(cov_dict['cov_WL_GS_6D'], nbl_WL, zpairs_auto, ind_auto)
        cov_GC_GS_4D = mm.cov_6D_to_4D(cov_dict['cov_GC_GS_6D'], nbl_GC, zpairs_auto, ind_auto)
        cov_WA_GS_4D = mm.cov_6D_to_4D(cov_dict['cov_WA_GS_6D'], nbl_WA, zpairs_auto, ind_auto)
        cov_3x2pt_GS_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_dict['cov_3x2pt_GS_10D_dict'], probe_ordering, nbl_GC,
                                                      zbins, ind.copy(), GL_or_LG)

    ############################### 4D to 2D ##################################
    # Here an ordering convention ('block_index') is needed as well
    cov_WL_GO_2D = mm.cov_4D_to_2D(cov_WL_GO_4D, block_index=block_index)
    cov_GC_GO_2D = mm.cov_4D_to_2D(cov_GC_GO_4D, block_index=block_index)
    cov_WA_GO_2D = mm.cov_4D_to_2D(cov_WA_GO_4D, block_index=block_index)
    cov_3x2pt_GO_2D = mm.cov_4D_to_2D(cov_3x2pt_GO_4D, block_index=block_index)

    cov_WL_GS_2D = mm.cov_4D_to_2D(cov_WL_GS_4D, block_index=block_index)
    cov_GC_GS_2D = mm.cov_4D_to_2D(cov_GC_GS_4D, block_index=block_index)
    cov_WA_GS_2D = mm.cov_4D_to_2D(cov_WA_GS_4D, block_index=block_index)
    cov_3x2pt_GS_2D = mm.cov_4D_to_2D(cov_3x2pt_GS_4D, block_index=block_index)

    cov_WL_SS_2D = mm.cov_4D_to_2D(cov_WL_SS_4D, block_index=block_index)
    cov_GC_SS_2D = mm.cov_4D_to_2D(cov_GC_SS_4D, block_index=block_index)
    cov_WA_SS_2D = mm.cov_4D_to_2D(cov_WA_SS_4D, block_index=block_index)
    cov_3x2pt_SS_2D = mm.cov_4D_to_2D(cov_3x2pt_SS_4D, block_index=block_index)

    if covariance_cfg['cov_ell_cuts']:
        # perform the cuts on the 2D covs (way faster!)
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
    covs_SS_4D = (cov_WL_SS_4D, cov_GC_SS_4D, cov_3x2pt_SS_4D, cov_WA_SS_4D)

    covs_GO_2D = (cov_WL_GO_2D, cov_GC_GO_2D, cov_3x2pt_GO_2D, cov_WA_GO_2D)
    covs_GS_2D = (cov_WL_GS_2D, cov_GC_GS_2D, cov_3x2pt_GS_2D, cov_WA_GS_2D)
    covs_SS_2D = (cov_WL_SS_2D, cov_GC_SS_2D, cov_3x2pt_SS_2D, cov_WA_SS_2D)

    for probe_name, cov_GO_4D, cov_GO_2D, cov_GS_4D, cov_GS_2D, cov_SS_4D, cov_SS_2D \
            in zip(probe_names, covs_GO_4D, covs_GO_2D, covs_GS_4D, covs_GS_2D, covs_SS_4D, covs_SS_2D):

        # save 4D
        # cov_dict[f'cov_{probe_name}_GO_4D'] = cov_GO_4D
        # cov_dict[f'cov_{probe_name}_GS_4D'] = cov_GS_4D
        # if covariance_cfg['save_cov_SSC']:
        #     cov_dict[f'cov_{probe_name}_SS_4D'] = cov_SS_4D

        # save 2D
        cov_dict[f'cov_{probe_name}_GO_2D'] = cov_GO_2D
        cov_dict[f'cov_{probe_name}_GS_2D'] = cov_GS_2D
        if covariance_cfg['save_cov_SSC']:
            cov_dict[f'cov_{probe_name}_SS_2D'] = cov_SS_2D

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
    # todo it's nicer if you sandwitch the covariance, maybe?
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
