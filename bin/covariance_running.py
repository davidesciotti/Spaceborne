import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm


###############################################################################
################ CODE TO COMPUTE THE G AND SSC COVMATS ########################
###############################################################################


def compute_cov(general_cfg, covariance_cfg, ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl):
    """
    This code computes the Gaussian-only, SSC-only and Gaussian+SSC
    covariance matrices, for different ordering options
    """

    # import settings:
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    zbins = general_cfg['zbins']
    n_probes = general_cfg['n_probes']
    which_forecast = general_cfg['which_forecast']
    EP_or_ED = general_cfg['EP_or_ED']

    fsky = covariance_cfg['fsky']
    GL_or_LG = covariance_cfg['GL_or_LG']
    ind_ordering = covariance_cfg['ind_ordering']
    # ! must copy the array! Otherwise, it gets modified and changed at each call
    ind = covariance_cfg['ind'].copy()
    block_index = covariance_cfg['block_index']
    which_probe_response = covariance_cfg['which_probe_response']
    print('DEBUG', which_probe_response)

    start = time.perf_counter()


    # import ell values
    ell_WL, nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
    ell_GC, nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
    ell_WA, nbl_WA = ell_dict['ell_WA'], ell_dict['ell_WA'].shape[0]
    ell_XC, nbl_3x2pt = ell_GC, nbl_GC

    cov_dict = {}

    # sanity checks
    if general_cfg['nbl_WL'] is None:
        assert nbl_WL == general_cfg['nbl'], 'WARNING: nbl_WL != general_cfg["nbl"], there is a discrepancy'

    if general_cfg['nbl_WL'] is not None:
        assert nbl_WL == general_cfg['nbl_WL'], 'WARNING: nbl_WL != general_cfg["nbl_WL"], there is a discrepancy'

    if nbl_WL == nbl_GC == nbl_3x2pt:
        print('all probes (but WAdd) have the same number of ell bins')

    # nbl for Wadd
    if ell_WA.size == 1:
        nbl_WA = 1  # in the case of just one bin it would give error
    else:
        nbl_WA = ell_WA.shape[0]

    # ell values in linear scale:
    if ell_WL.max() < 15:  # very rudimental check of whether they're in lin or log scale
        print('looks like the ell values are already in linear scale')
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

    # sanity check: the last 2 columns of ind_LL should be equal to the last two of ind_GG
    assert np.array_equiv(ind[:zpairs_auto, 2:], ind[-zpairs_auto:, 2:])

    # convenience vectors, used for the cov_4D_to_6D function
    ind_LL = ind[:zpairs_auto, :].copy()
    ind_GG = ind_LL.copy()
    ind_XC = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()

    # load Cls
    cl_LL_3D = cl_dict_3D['cl_LL_3D']
    cl_GG_3D = cl_dict_3D['cl_GG_3D']
    cl_WA_3D = cl_dict_3D['cl_WA_3D']
    cl_3x2pt_5D = cl_dict_3D['cl_3x2pt_5D']

    if which_probe_response == 'constant':
        rl_value = covariance_cfg['rl_value']
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
        f'\ncheck: \nwhich_forecast = {which_forecast} \nind_ordering = {ind_ordering} \nblock_index = {block_index}\n'
        f'zbins: {general_cfg["EP_or_ED"]}{zbins}\n'
        f'nbl_WA: {nbl_WA} nbl_WL: {nbl_WL} nbl_GC:  {nbl_GC}, nbl_3x2pt:  {nbl_3x2pt}\n'
        f'ell_max_WL = {ell_max_WL} \nell_max_GC = {ell_max_GC}\n'
        f'computing the covariance in blocks? {covariance_cfg["save_cov_6D"]}\n')

    # build noise vector
    print('which folder should I use for ngbTab? lenses or sources? Flagship or Redbook?')
    # a couple rough checks
    if general_cfg['EP_or_ED'] == 'ED':
        assert general_cfg['which_forecast'] == 'SPV3', 'data for the ED case is only available for SPV3'

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

    ######################## COMPUTE SS COVARIANCE ###############################

    start = time.perf_counter()
    cov_WL_SS_4D = mm.cov_SSC(nbl_WL, zpairs_auto, ind, cl_LL_3D, Sijkl, fsky, "WL", zbins, rl_LL_3D)
    cov_GC_SS_4D = mm.cov_SSC(nbl_GC, zpairs_auto, ind, cl_GG_3D, Sijkl, fsky, "GC", zbins, rl_GG_3D)
    cov_WA_SS_4D = mm.cov_SSC(nbl_WA, zpairs_auto, ind, cl_WA_3D, Sijkl, fsky, "WA", zbins, rl_WA_3D)
    cov_3x2pt_SS_4D = mm.cov_SSC_ALL(nbl_3x2pt, zpairs_3x2pt, ind, cl_3x2pt_5D, Sijkl, fsky, zbins, rl_3x2pt_5D)
    print("SS cov. matrices computed in %.2f seconds" % (time.perf_counter() - start))

    ############################## SUM G + SSC ################################
    cov_WL_GS_4D = cov_WL_GO_4D + cov_WL_SS_4D
    cov_GC_GS_4D = cov_GC_GO_4D + cov_GC_SS_4D
    cov_WA_GS_4D = cov_WA_GO_4D + cov_WA_SS_4D
    cov_3x2pt_GS_4D = cov_3x2pt_GO_4D + cov_3x2pt_SS_4D

    if covariance_cfg['save_cov_6D']:

        # compute 3x2pt covariance in 10D, potentially with whichever probe ordering, and the WL, GS and WA cov in 6D

        # store the input datavector and noise spectra in a dictionary
        cl_dict_3x2pt = mm.build_3x2pt_dict(cl_3x2pt_5D)
        rl_dict_3x2pt = mm.build_3x2pt_dict(rl_3x2pt_5D)
        noise_dict_3x2pt = mm.build_3x2pt_dict(noise)
        Sijkl_dict = mm.build_Sijkl_dict(Sijkl, zbins)

        # probe ordering
        # the function should be able to work with whatever 
        # ordering of the probes; (TODO check this)

        # this is a check to make sure that XC has the ordering (L, G) or (G, L) specified by GL_or_LG, and it
        # only works for the (LL, XC, GG) ordering
        probe_ordering = [['L', 'L'], [None, None], ['G', 'G']]

        # (not the best) check to ensure that the (LL, XC, GG) ordering is respected
        assert probe_ordering[0] == ['L', 'L'], 'the XC probe should be in position 1 (not 0) of the datavector'
        assert probe_ordering[2] == ['G', 'G'], 'the XC probe should be in position 1 (not 0) of the datavector'

        # this overwrites the 1st axis, the one describing XC
        probe_ordering[1][0] = GL_or_LG[0]
        probe_ordering[1][1] = GL_or_LG[1]

        # print as a check
        print('check: datavector probe ordering:', probe_ordering)
        print('check: GL_or_LG:', GL_or_LG)
        print('check: probe combinations:')
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                print(A, B, C, D)

        # compute the 10D covariance only for the blocks which will actually be used (GO and SS)
        start = time.perf_counter()
        cov_3x2pt_GO_10D = mm.cov_G_10D_dict(cl_dict_3x2pt, noise_dict_3x2pt, nbl_3x2pt, zbins, l_lin_XC,
                                             delta_l_XC, fsky, probe_ordering)
        print(f'cov_3x2pt_GO_10D computed in {(time.perf_counter() - start):.2f} seconds')

        start = time.perf_counter()
        cov_3x2pt_SS_10D = mm.cov_SS_10D_dict(cl_dict_3x2pt, rl_dict_3x2pt, Sijkl_dict, nbl_3x2pt, zbins, fsky,
                                              probe_ordering)
        print(f'cov_3x2pt_SS_10D computed in {(time.perf_counter() - start):.2f} seconds')

        # sum GO and SS
        cov_3x2pt_GS_10D = {}
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                cov_3x2pt_GS_10D[A, B, C, D] = np.zeros((nbl_3x2pt, nbl_3x2pt, zbins, zbins, zbins, zbins))

        for A, B in probe_ordering:
            for C, D in probe_ordering:
                cov_3x2pt_GS_10D[A, B, C, D][...] = cov_3x2pt_GO_10D[A, B, C, D][...] + \
                                                    cov_3x2pt_SS_10D[A, B, C, D][...]

        # this is to revert from 10D to 4D, which is trickier for the 3x2pt (each block has to be converted to 4D and
        # stacked to make the 4D_3x2pt)
        """
        # note: I pass ind_copy because the LG-GL check and inversion is performed in the function (otherwise it would be
        # performed twice!)
        # ! careful of passing clean copies of ind to both functions!!!
        cov_3x2pt_GO_4D_new = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_GO_10D, probe_ordering, nbl_3x2pt, zbins,
                                                          ind.copy(), GL_or_LG)
        cov_3x2pt_SS_4D_new = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_SS_10D, probe_ordering, nbl_3x2pt, zbins,
                                                          ind.copy(), GL_or_LG)
        cov_3x2pt_GS_4D_new = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_GS_10D, probe_ordering, nbl_3x2pt, zbins,
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
        cov_dict['cov_3x2pt_GO_10D'] = cov_3x2pt_GO_10D
        cov_dict['cov_3x2pt_GS_10D'] = cov_3x2pt_GS_10D
        cov_dict['cov_3x2pt_SS_10D'] = cov_3x2pt_SS_10D

        # this is the 1st way to compute cov_6D: simply transform the cov_4D array (note that cov_4D_to_6D does not
        # work for 3x2pt, althought it should be easy to implement). Quite slow for GS or SS matrices.
        # example:
        # cov_dict['cov_WL_GO_6D'] = mm.cov_4D_to_6D(cov_WL_GO_4D, nbl_WL, zbins, probe='LL', ind=ind_LL)

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
        print(f'cov_GO_6D new computed in {(time.perf_counter() - start_time):.2f} seconds')


        # ! cov_SS_6D
        start_time = time.perf_counter()
        cov_WL_SS_6D = mm.cov_SS_10D_dict(cl_dict_LL, rl_dict_LL, Sijkl_dict, nbl_WL, zbins, fsky,
                                          probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        cov_GC_SS_6D = mm.cov_SS_10D_dict(cl_dict_GG, rl_dict_GG, Sijkl_dict, nbl_GC, zbins, fsky,
                                          probe_ordering=[['G', 'G'], ])['G', 'G', 'G', 'G']
        cov_WA_SS_6D = mm.cov_SS_10D_dict(cl_dict_WA, rl_dict_WA, Sijkl_dict, nbl_WA, zbins, fsky,
                                          probe_ordering=[['L', 'L'], ])['L', 'L', 'L', 'L']
        print(f'cov_SS_6D new computed in {(time.perf_counter() - start_time):.2f} seconds')

        if covariance_cfg['save_cov_SS']:
            cov_dict['cov_WL_SS_6D'] = cov_WL_SS_6D
            cov_dict['cov_GC_SS_6D'] = cov_GC_SS_6D
            cov_dict['cov_WA_SS_6D'] = cov_WA_SS_6D

        # ! cov_GS_6D
        cov_dict['cov_WL_GS_6D'] = cov_dict['cov_WL_GO_6D'] + cov_WL_SS_6D
        cov_dict['cov_GC_GS_6D'] = cov_dict['cov_GC_GO_6D'] + cov_GC_SS_6D
        cov_dict['cov_WA_GS_6D'] = cov_dict['cov_WA_GO_6D'] + cov_WA_SS_6D

        # test that they are equal to the 4D ones; this is quite slow, so I check only some arrays
        print('check: is cov_4D == mm.cov_6D_to_4D(cov_6D)?')
        assert np.allclose(cov_WL_GO_4D, mm.cov_6D_to_4D(cov_dict['cov_WL_GO_6D'], nbl_WL, zpairs_auto, ind_LL), rtol=1e-7, atol=0)
        # assert np.allclose(cov_GC_GO_4D, mm.cov_6D_to_4D(cov_dict['cov_GC_GO_6D'], nbl_GC, zpairs_auto, ind_GG), rtol=1e-7, atol=0)
        # assert np.allclose(cov_WA_GO_4D, mm.cov_6D_to_4D(cov_dict['cov_WA_GO_6D'], nbl_WA, zpairs_auto, ind_LL), rtol=1e-7, atol=0)

        # assert np.allclose(cov_WL_GS_4D, mm.cov_6D_to_4D(cov_dict['cov_WL_GS_6D'], nbl_WL, zpairs_auto, ind_LL), rtol=1e-7, atol=0)
        assert np.allclose(cov_GC_GS_4D, mm.cov_6D_to_4D(cov_dict['cov_GC_GS_6D'], nbl_GC, zpairs_auto, ind_GG), rtol=1e-7, atol=0)
        # assert np.allclose(cov_WA_GS_4D, mm.cov_6D_to_4D(cov_dict['cov_WA_GS_6D'], nbl_WA, zpairs_auto, ind_LL), rtol=1e-7, atol=0)

    print('checks passed')

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
        cov_dict[f'cov_{probe_name}_GO_4D'] = cov_GO_4D
        cov_dict[f'cov_{probe_name}_GS_4D'] = cov_GS_4D
        cov_dict[f'cov_{probe_name}_SS_4D'] = cov_SS_4D
        # save 2D
        cov_dict[f'cov_{probe_name}_GO_2D'] = cov_GO_2D
        cov_dict[f'cov_{probe_name}_GS_2D'] = cov_GS_2D
        cov_dict[f'cov_{probe_name}_SS_2D'] = cov_SS_2D

    # '2DCLOE', i.e. the 'multi-diagonal', non-square blocks ordering, only for 3x2pt
    # note: we found out that this is not actually used in CLOE...
    if covariance_cfg['save_2DCLOE']:
        cov_dict[f'cov_3x2pt_GO_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_GO_4D, nbl_3x2pt, zbins)
        cov_dict[f'cov_3x2pt_GS_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_GS_4D, nbl_3x2pt, zbins)

    return cov_dict
