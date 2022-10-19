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


def compute_cov(general_cfg, covariance_cfg, ell_dict, delta_dict, cl_dict_3D, Rl_dict_3D, Sijkl):
    """
    This code computes the Gaussian-only, SSC-only and Gaussian+SSC
    covariance matrices, for different ordering options
    """

    # import settings:
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    zbins = general_cfg['zbins']
    cl_folder = general_cfg['cl_folder']
    nProbes = general_cfg['nProbes']
    which_forecast = general_cfg['which_forecast']

    fsky = covariance_cfg['fsky']
    GL_or_LG = covariance_cfg['GL_or_LG']
    ind_ordering = covariance_cfg['ind_ordering']
    # ! must copy the array! Otherwise, it gets modifiesd and changed at each call
    ind = covariance_cfg['ind'].copy()
    block_index = covariance_cfg['block_index']
    which_probe_response = covariance_cfg['which_probe_response']

    print('TODO: find a better way to copy the ind file?')
    ind_copy = ind.copy()
    ind_copy_2 = ind.copy()

    start = time.perf_counter()

    ell_max_XC = ell_max_GC
    ell_max_WA = ell_max_GC

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
    npairs_auto, npairs_cross, npairs_tot = mm.get_pairs(zbins)

    # if C_XC is C_LG, switch the ind.dat ordering for the correct rows
    if GL_or_LG == 'LG':
        print('\nAttention! switching columns in the ind array (for the XC part)')
        ind[npairs_auto:(npairs_auto + npairs_cross), [2, 3]] = ind[npairs_auto:(npairs_auto + npairs_cross), [3, 2]]

    # sanity check: the last 2 columns of ind_LL should be equal to the last two of ind_GG
    assert np.array_equiv(ind[:npairs_auto, 2:], ind[-npairs_auto:, 2:])

    # convenience vectors, used for the cov_4D_to_6D function
    ind_LL = ind[:npairs_auto, :].copy()
    ind_GG = ind_LL.copy()
    ind_XC = ind[npairs_auto:npairs_cross + npairs_auto, :].copy()

    # load Cls
    C_LL_3D = cl_dict_3D['C_LL_WLonly_3D']
    C_GG_3D = cl_dict_3D['C_GG_3D']
    C_WA_3D = cl_dict_3D['C_WA_3D']
    C_3x2pt_5D = cl_dict_3D['C_3x2pt_5D']

    if which_probe_response == 'constant':
        Rl = covariance_cfg['Rl']
        R_LL_3D = np.full(C_LL_3D.shape, Rl)
        R_GG_3D = np.full(C_GG_3D.shape, Rl)
        R_WA_3D = np.full(C_WA_3D.shape, Rl)
        R_3x2pt_5D = np.full(C_3x2pt_5D.shape, Rl)
    elif which_probe_response == 'variable':
        R_LL_3D = Rl_dict_3D['R_LL_WLonly_3D']
        R_GG_3D = Rl_dict_3D['R_GG_3D']
        R_WA_3D = Rl_dict_3D['R_WA_3D']
        R_3x2pt_5D = Rl_dict_3D['R_3x2pt_5D']
    else:
        raise ValueError("which_probe_response must be 'constant' or 'variable'")

    # print settings
    print(
        f'\ncheck: \nwhich_forecast = {which_forecast} \nind_ordering = {ind_ordering} \nblock_index = {block_index}\n'
        f'zbins: {zbins} {general_cfg["EP_or_ED"]}\n'
        f'nbl_WA: {nbl_WA} nbl_WL: {nbl_WL} nbl_GC:  {nbl_GC}, nbl_3x2pt:  {nbl_3x2pt}\n'
        f'ell_max_WL = {ell_max_WL} \nell_max_GC = {ell_max_GC}\n'
        f'computing the covariance in blocks? {covariance_cfg["save_cov_6D"]}\n')

    # build noise vector
    if general_cfg['EP_or_ED'] == 'EP':
        ng = covariance_cfg['ng']
    elif general_cfg['EP_or_ED'] == 'ED':
        print('lenses or sources? Flagship or Redbook?')
        ng = np.genfromtxt(
            f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/InputNz/Lenses/Flagship/ngbTab-ED{zbins:02}.dat')[
             0, :]
        # a couple rough checks
        assert 28. < np.sum(ng) < 29., 'the sum of ng over all bins is not 28.73'
        assert general_cfg['which_forecast'] == 'SPV3', 'data for the ED case is only available for SPV3'
    else:
        raise ValueError('EP_or_ED must be "EP" or "ED"')

    noise = mm.build_noise(zbins, nProbes, sigma_eps2=covariance_cfg['sigma_eps2'], ng=ng,
                           EP_or_ED=general_cfg['EP_or_ED'])

    ################### COMPUTE GAUSS ONLY COVARIANCE #########################

    # WL only covariance
    cov_WL_GO_4D = mm.covariance(nbl=nbl_WL, npairs=npairs_auto, start_index=0, stop_index=npairs_auto,
                                 Cij=C_LL_3D, noise=noise, l_lin=l_lin_WL,
                                 delta_l=delta_l_WL, fsky=fsky, ind=ind)
    # GC only covariance
    starting_GC_index = npairs_auto + npairs_cross
    cov_GC_GO_4D = mm.covariance(nbl=nbl_GC, npairs=npairs_auto, start_index=starting_GC_index, stop_index=npairs_tot,
                                 Cij=C_GG_3D, noise=noise, l_lin=l_lin_GC,
                                 delta_l=delta_l_GC, fsky=fsky, ind=ind)
    # WA covariance
    cov_WA_GO_4D = mm.covariance_WA(nbl_WA, npairs_auto, start_index=0, stop_index=npairs_auto,
                                    Cij=C_WA_3D, noise=noise, l_lin=l_lin_WA,
                                    delta_l=delta_l_WA, fsky=fsky, ind=ind, ell_WA=ell_WA)
    # ALL covariance
    cov_3x2pt_GO_4D = mm.covariance_ALL(nbl=nbl_3x2pt, npairs=npairs_tot,
                                        Cij=C_3x2pt_5D, noise=noise, l_lin=l_lin_XC,
                                        delta_l=delta_l_XC, fsky=fsky, ind=ind)
    print("Gauss. cov. matrices computed in %.2f seconds" % (time.perf_counter() - start))

    ######################## COMPUTE SS COVARIANCE ###############################

    start = time.perf_counter()
    cov_WL_SS_4D = mm.cov_SSC(nbl_WL, npairs_auto, ind, C_LL_3D, Sijkl, fsky, "WL", zbins, R_LL_3D)
    cov_GC_SS_4D = mm.cov_SSC(nbl_GC, npairs_auto, ind, C_GG_3D, Sijkl, fsky, "GC", zbins, R_GG_3D)
    cov_WA_SS_4D = mm.cov_SSC(nbl_WA, npairs_auto, ind, C_WA_3D, Sijkl, fsky, "WA", zbins, R_WA_3D)
    cov_3x2pt_SS_4D = mm.cov_SSC_ALL(nbl_3x2pt, npairs_tot, ind, C_3x2pt_5D, Sijkl, fsky, zbins, R_3x2pt_5D)
    print("SS cov. matrices computed in %.2f seconds" % (time.perf_counter() - start))

    ############################## SUM G + SSC ################################
    cov_WL_GS_4D = cov_WL_GO_4D + cov_WL_SS_4D
    cov_GC_GS_4D = cov_GC_GO_4D + cov_GC_SS_4D
    cov_WA_GS_4D = cov_WA_GO_4D + cov_WA_SS_4D
    cov_3x2pt_GS_4D = cov_3x2pt_GO_4D + cov_3x2pt_SS_4D

    if covariance_cfg['save_cov_6D']:
        # compute 3x2pt covariance in 10D, potentially with whichever probe ordering, and the WL, GS and WA cov in 6D

        # store the input datavector and noise spectra in a dictionary
        cl_dict = mm.build_3x2pt_dict(C_3x2pt_5D)
        rl_dict = mm.build_3x2pt_dict(R_3x2pt_5D)
        noise_dict = mm.build_3x2pt_dict(noise)
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
        cov_3x2pt_GO_10D = mm.cov_G_10D_dict(cl_dict, noise_dict, nbl_3x2pt, zbins, l_lin_XC,
                                             delta_l_XC, fsky, probe_ordering)
        print(f'cov_3x2pt_GO_10D computed in {(time.perf_counter() - start):.2f} seconds')

        start = time.perf_counter()
        cov_3x2pt_SS_10D = mm.cov_SS_10D_dict(cl_dict, rl_dict, Sijkl_dict, nbl_3x2pt, zbins, fsky, probe_ordering)
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
        # # TODO implement the other covmats in this module!
        # if use_PyCCL_SS
        # if use_PyCCL_cNG:

        # save the 6D-10D covs in the dictionary
        cov_dict['cov_3x2pt_GO_10D'] = cov_3x2pt_GO_10D
        cov_dict['cov_3x2pt_GS_10D'] = cov_3x2pt_GS_10D
        cov_dict['cov_3x2pt_SS_10D'] = cov_3x2pt_SS_10D

        cov_dict['cov_WL_GO_6D'] = mm.cov_4D_to_6D(cov_WL_GO_4D, nbl_WL, zbins, probe='LL', ind=ind_LL)
        cov_dict['cov_GC_GO_6D'] = mm.cov_4D_to_6D(cov_GC_GO_4D, nbl_GC, zbins, probe='GG', ind=ind_GG)
        cov_dict['cov_WA_GO_6D'] = mm.cov_4D_to_6D(cov_WA_GO_4D, nbl_WA, zbins, probe='LL', ind=ind_LL)

        cov_dict['cov_WL_GS_6D'] = mm.cov_4D_to_6D(cov_WL_GS_4D, nbl_WL, zbins, probe='LL', ind=ind_LL)
        cov_dict['cov_GC_GS_6D'] = mm.cov_4D_to_6D(cov_GC_GS_4D, nbl_GC, zbins, probe='GG', ind=ind_GG)
        cov_dict['cov_WA_GS_6D'] = mm.cov_4D_to_6D(cov_WA_GS_4D, nbl_WA, zbins, probe='LL', ind=ind_LL)

        cov_dict['cov_WL_SS_6D'] = mm.cov_4D_to_6D(cov_WL_SS_4D, nbl_WL, zbins, probe='LL', ind=ind_LL)
        cov_dict['cov_GC_SS_6D'] = mm.cov_4D_to_6D(cov_GC_SS_4D, nbl_GC, zbins, probe='GG', ind=ind_GG)
        cov_dict['cov_WA_SS_6D'] = mm.cov_4D_to_6D(cov_WA_SS_4D, nbl_WA, zbins, probe='LL', ind=ind_LL)

        # test that they are equal to the 4D ones; this is quite slow, so I check only some of the arrays
        print('checks: is cov_4D == mm.cov_6D_to_4D(cov_6D)?')
        assert np.array_equal(cov_WL_GO_4D, mm.cov_6D_to_4D(cov_dict['cov_WL_GO_6D'], nbl_WL, npairs_auto, ind_LL))
        # assert np.array_equal(cov_GC_GO_4D, mm.cov_6D_to_4D(cov_dict['cov_GC_GO_6D'], nbl_GC, npairs_auto, ind_GG))
        # assert np.array_equal(cov_WA_GO_4D, mm.cov_6D_to_4D(cov_dict['cov_WA_GO_6D'], nbl_WA, npairs_auto, ind_LL))

        # assert np.array_equal(cov_WL_GS_4D, mm.cov_6D_to_4D(cov_dict['cov_WL_GS_6D'], nbl_WL, npairs_auto, ind_LL))
        assert np.array_equal(cov_GC_GS_4D, mm.cov_6D_to_4D(cov_dict['cov_GC_GS_6D'], nbl_GC, npairs_auto, ind_GG))
        # assert np.array_equal(cov_WA_GS_4D, mm.cov_6D_to_4D(cov_dict['cov_WA_GS_6D'], nbl_WA, npairs_auto, ind_LL))

        # assert np.array_equal(cov_WL_SS_4D, mm.cov_6D_to_4D(cov_dict['cov_WL_SS_6D'], nbl_WL, npairs_auto, ind_LL))
        # assert np.array_equal(cov_GC_SS_4D, mm.cov_6D_to_4D(cov_dict['cov_GC_SS_6D'], nbl_GC, npairs_auto, ind_GG))
        assert np.array_equal(cov_WA_SS_4D, mm.cov_6D_to_4D(cov_dict['cov_WA_SS_6D'], nbl_WA, npairs_auto, ind_LL))
        print('checks passed')

    ############################### 4D to 2D ##################################
    # Here an ordering convention ('block_index') is needed as well
    cov_WL_GO_2D = mm.cov_4D_to_2D(cov_WL_GO_4D, nbl_WL, npairs_auto, block_index=block_index)
    cov_GC_GO_2D = mm.cov_4D_to_2D(cov_GC_GO_4D, nbl_GC, npairs_auto, block_index=block_index)
    cov_WA_GO_2D = mm.cov_4D_to_2D(cov_WA_GO_4D, nbl_WA, npairs_auto, block_index=block_index)
    cov_3x2pt_GO_2D = mm.cov_4D_to_2D(cov_3x2pt_GO_4D, nbl_3x2pt, npairs_tot, block_index=block_index)

    cov_WL_GS_2D = mm.cov_4D_to_2D(cov_WL_GS_4D, nbl_WL, npairs_auto, block_index=block_index)
    cov_GC_GS_2D = mm.cov_4D_to_2D(cov_GC_GS_4D, nbl_GC, npairs_auto, block_index=block_index)
    cov_WA_GS_2D = mm.cov_4D_to_2D(cov_WA_GS_4D, nbl_WA, npairs_auto, block_index=block_index)
    cov_3x2pt_GS_2D = mm.cov_4D_to_2D(cov_3x2pt_GS_4D, nbl_3x2pt, npairs_tot, block_index=block_index)

    cov_WL_SS_2D = mm.cov_4D_to_2D(cov_WL_SS_4D, nbl_WL, npairs_auto, block_index=block_index)
    cov_GC_SS_2D = mm.cov_4D_to_2D(cov_GC_SS_4D, nbl_GC, npairs_auto, block_index=block_index)
    cov_WA_SS_2D = mm.cov_4D_to_2D(cov_WA_SS_4D, nbl_WA, npairs_auto, block_index=block_index)
    cov_3x2pt_SS_2D = mm.cov_4D_to_2D(cov_3x2pt_SS_4D, nbl_3x2pt, npairs_tot, block_index=block_index)

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
