import sys
import time
import warnings
from pathlib import Path
import numpy as np

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm

script_name = sys.argv[0]
start = time.perf_counter()


###############################################################################
################## CODE TO COMPUTE THE FISHER MATRIX ##########################
###############################################################################


###########################################

# XXX attention! the dC_LL matrix should have (nParams - 10) as dimension,
# since WL has no bias. This would complicate the structure of the datacector 
# and taking nParams instead seems to have ho impact on the final result.

def dC_4D_to_3D(dC_4D, nbl, zpairs, nparams_tot, ind):
    print('most likely, I already wrote this function...')
    dC_3D = np.zeros((nbl, zpairs, nparams_tot))
    for ell in range(nbl):
        for alf in range(nparams_tot):
            dC_3D[ell, :, alf] = mm.array_2D_to_1D_ind(dC_4D[ell, :, :, alf], zpairs, ind)
    return dC_3D


def compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict):
    # shorten names
    # nbl = general_cfg['nbl']
    zbins = general_cfg['zbins']
    use_WA = general_cfg['use_WA']
    GL_or_LG = covariance_cfg['GL_or_LG']
    ind = covariance_cfg['ind']
    block_index = covariance_cfg['block_index']
    nparams_tot = FM_cfg['nparams_tot']

    # import ell values
    ell_WL, nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
    ell_GC, nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
    ell_WA, nbl_WA = ell_dict['ell_WA'], ell_dict['ell_WA'].shape[0]
    ell_XC, nbl_3x2pt = ell_GC, nbl_GC


    # set the flattening convention for the derivatives vector, based on the setting used to reduce the covariance
    # matrix' dimensions
    # TODO review this
    if block_index in ['ell', 'vincenzo', 'C-style']:
        which_flattening = 'C'
    elif block_index in ['ij', 'sylvain', 'F-style']:
        which_flattening = 'F'
    else:
        raise ValueError("block_index should be either 'ell', 'vincenzo', 'C-style', 'ij', 'sylvain' or 'F-style'")

    # check to see if ell values are in linear or log scale
    if np.max(ell_WL) > 30:
        print('switching to log scale for the ell values')
        ell_WL = np.log10(ell_WL)
        ell_GC = np.log10(ell_GC)
        ell_WA = np.log10(ell_WA)
        ell_XC = ell_GC

    # nbl for Wadd: in the case of just one bin it would give error
    if ell_WA.size == 1:
        nbl_WA = 1
    else:
        nbl_WA = ell_WA.shape[0]

    warnings.warn('nParams_WL needs to be defined as len(paramnames_LL?)')
    nParams_bias = zbins
    nParams_WL = nparams_tot - nParams_bias

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_pairs(zbins)

    if GL_or_LG == 'LG':
        print('\nAttention! switching columns in the ind array (for the XC part)')
        ind[zpairs_auto:(zpairs_auto + zpairs_cross), [2, 3]] = ind[zpairs_auto:(zpairs_auto + zpairs_cross), [3, 2]]

    ############################################

    # invert GO covmats
    warnings.warn('de-comment these lines')
    # start1 = time.perf_counter()
    # cov_WL_GO_2D_inv = np.linalg.inv(cov_dict['cov_WL_GO_2D'])
    # cov_GC_GO_2D_inv = np.linalg.inv(cov_dict['cov_GC_GO_2D'])
    # cov_WA_GO_2D_inv = np.linalg.inv(cov_dict['cov_WA_GO_2D'])
    # cov_3x2pt_GO_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_GO_2D'])
    # print(f'GO covmats inverted in {(time.perf_counter() - start1):.2f} s')
    #
    # # invert GS covmats
    # start2 = time.perf_counter()
    # cov_WL_GS_2D_inv = np.linalg.inv(cov_dict['cov_WL_GS_2D'])
    # cov_GC_GS_2D_inv = np.linalg.inv(cov_dict['cov_GC_GS_2D'])
    # cov_WA_GS_2D_inv = np.linalg.inv(cov_dict['cov_WA_GS_2D'])
    # cov_3x2pt_GS_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_GS_2D'])
    # print(f'GO covmats inverted in {(time.perf_counter() - start2):.2f} s')

    # set parameters names for the different probes

    """
    # initialize derivatives arrays
    dC_LL_WLonly = np.zeros((nbl, zpairs_auto, nparams_tot))
    dC_LL = np.zeros((nbl, zpairs_auto, nparams_tot))
    dC_XC = np.zeros((nbl, zpairs_cross, nparams_tot))
    dC_GG = np.zeros((nbl, zpairs_auto, nparams_tot))
    dC_WA = np.zeros((nbl_WA, zpairs_auto, nparams_tot))

    # create dict to store interpolated Cij arrays
    dC_WLonly_interpolated_dict = {}
    dC_GConly_interpolated_dict = {}
    dC_3x2pt_interpolated_dict = {}
    dC_WA_interpolated_dict = {}

    # call the function to interpolate: PAY ATTENTION TO THE PARAMETERS PASSED!
    # WLonly
    dC_WLonly_interpolated_dict = mm.interpolator(probe_code=probe_code_LL,
                                                  dC_interpolated_dict=dC_WLonly_interpolated_dict,
                                                  dC_dict=dC_dict, params_names=paramnames_LL, nbl=nbl,
                                                  npairs=zpairs_auto, ell_values=ell_WL, suffix=suffix)
    # GConly
    dC_GConly_interpolated_dict = mm.interpolator(probe_code=probe_code_GG,
                                                  dC_interpolated_dict=dC_GConly_interpolated_dict,
                                                  dC_dict=dC_dict, params_names=paramnames_GG, nbl=nbl,
                                                  npairs=zpairs_auto, ell_values=ell_XC, suffix=suffix)
    # LL for 3x2pt
    dC_3x2pt_interpolated_dict = mm.interpolator(probe_code=probe_code_LL,
                                                 dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                                                 dC_dict=dC_dict, params_names=paramnames_LL, nbl=nbl,
                                                 npairs=zpairs_auto, ell_values=ell_XC, suffix=suffix)
    # XC for 3x2pt
    dC_3x2pt_interpolated_dict = mm.interpolator(probe_code=probe_code_XC,
                                                 dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                                                 dC_dict=dC_dict, params_names=paramnames_XC, nbl=nbl,
                                                 npairs=zpairs_cross, ell_values=ell_XC, suffix=suffix)
    # GG for 3x2pt
    dC_3x2pt_interpolated_dict = mm.interpolator(probe_code=probe_code_GG,
                                                 dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                                                 dC_dict=dC_dict, params_names=paramnames_GG, nbl=nbl,
                                                 npairs=zpairs_auto, ell_values=ell_XC, suffix=suffix)
    # LL for WA
    dC_WA_interpolated_dict = mm.interpolator(probe_code=probe_code_LL,
                                              dC_interpolated_dict=dC_WA_interpolated_dict,
                                              dC_dict=dC_dict, params_names=paramnames_LL, nbl=nbl_WA,
                                              npairs=zpairs_auto, ell_values=ell_WA, suffix=suffix)

    # fill the dC array using the interpolated dictionary
    # WLonly
    dC_LL_WLonly = mm.fill_dC_array(params_names=paramnames_LL,
                                    dC_interpolated_dict=dC_WLonly_interpolated_dict,
                                    probe_code=probe_code_LL, dC=dC_LL_WLonly, suffix=suffix)
    # LL for 3x2pt
    dC_LL = mm.fill_dC_array(params_names=paramnames_LL,
                             dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                             probe_code=probe_code_LL, dC=dC_LL, suffix=suffix)
    # XC for 3x2pt
    dC_XC = mm.fill_dC_array(params_names=paramnames_XC,
                             dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                             probe_code=probe_code_XC, dC=dC_XC, suffix=suffix)
    # GG for 3x2pt and GConly
    dC_GG = mm.fill_dC_array(params_names=paramnames_GG,
                             dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                             probe_code=probe_code_GG, dC=dC_GG, suffix=suffix)
    # LL for WA
    dC_WA = mm.fill_dC_array(params_names=paramnames_LL,
                             dC_interpolated_dict=dC_WA_interpolated_dict,
                             probe_code=probe_code_LL, dC=dC_WA, suffix=suffix)

    # ! reshape dC from (nbl, zpairs, nparams_tot) to (nbl, zbins, zbins, nparams) - i.e., go from '2D' to '3D'
    # (+ 1 "excess" dimension). Note that Vincenzo uses np.triu to reduce the dimensions of the cl arrays,
    # but ind_vincenzo to organize the covariance matrix.

    dC_LL_4D = np.zeros((nbl, zbins, zbins, nparams_tot))
    dC_GG_4D = np.zeros((nbl, zbins, zbins, nparams_tot))
    dC_LL_WLonly_4D = np.zeros((nbl, zbins, zbins, nparams_tot))
    dC_WA_4D = np.zeros((nbl_WA, zbins, zbins, nparams_tot))

    # fill symmetric
    triu_idx = np.triu_indices(zbins)
    for ell in range(nbl):
        for alf in range(nparams_tot):
            for i in range(zpairs_auto):
                dC_LL_4D[ell, triu_idx[0][i], triu_idx[1][i], alf] = dC_LL[ell, i, alf]
                dC_GG_4D[ell, triu_idx[0][i], triu_idx[1][i], alf] = dC_GG[ell, i, alf]
                dC_LL_WLonly_4D[ell, triu_idx[0][i], triu_idx[1][i], alf] = dC_LL_WLonly[ell, i, alf]
    # Wadd
    for ell in range(nbl_WA):
        for alf in range(nparams_tot):
            for i in range(zpairs_auto):
                dC_WA_4D[ell, triu_idx[0][i], triu_idx[1][i]] = dC_WA[ell, i]

    # symmetrize
    for alf in range(nparams_tot):
        dC_LL_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_LL_4D[:, :, :, alf], nbl, zbins)
        dC_GG_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_GG_4D[:, :, :, alf], nbl, zbins)
        dC_WA_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_WA_4D[:, :, :, alf], nbl_WA, zbins)
        dC_LL_WLonly_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_LL_WLonly_4D[:, :, :, alf], nbl, zbins)

    # fill asymmetric
    dC_XC_4D = np.reshape(dC_XC, (nbl, zbins, zbins, nparams_tot))

    """

    # load reshaped derivatives, with shape (nbl, zbins, zbins, nparams)
    dC_LL_4D = deriv_dict['dC_LL_4D']
    dC_GG_4D = deriv_dict['dC_GG_4D']
    dC_WA_4D = deriv_dict['dC_WA_4D']

    # separate the different 3x2pt contributions
    # ! delicate point, double check
    if GL_or_LG == 'GL':
        probe_A, probe_B = 1, 0
    elif GL_or_LG == 'LG':
        probe_A, probe_B = 0, 1
    else:
        raise ValueError('GL_or_LG must be "GL" or "LG"')

    dC_LLfor3x2pt_4D = deriv_dict['dC_3x2pt_5D'][:, 0, 0, :, :, :]
    dC_XCfor3x2pt_4D = deriv_dict['dC_3x2pt_5D'][:, probe_A, probe_B, :, :, :]
    dC_GGfor3x2pt_4D = deriv_dict['dC_3x2pt_5D'][:, 1, 1, :, :, :]

    assert np.array_equal(dC_GGfor3x2pt_4D, dC_GG_4D), "dC_GGfor3x2pt_4D and dC_GG_4D are not equal"
    assert nbl_3x2pt == nbl_GC, 'nbl_3x2pt and nbl_GC are not equal'

    # separate the ind for the different probes
    ind_LL = ind[:zpairs_auto, :]
    ind_GG = ind[:zpairs_auto, :]
    ind_XC = ind[zpairs_auto:zpairs_cross, :]  # ! watch out for the ind switch!!

    # flatten z indices, obviously following the ordering given in ind
    dC_LL_3D_v2 = dC_4D_to_3D(dC_LL_4D, nbl_WL, zpairs_auto, nparams_tot, ind_LL)
    dC_GG_3D_v2 = dC_4D_to_3D(dC_GG_4D, nbl_GC, zpairs_auto, nparams_tot, ind_GG)
    dC_WA_3D_v2 = dC_4D_to_3D(dC_WA_4D, nbl_WA, zpairs_auto, nparams_tot, ind_LL)
    dC_LLfor3x2pt_3D = dC_4D_to_3D(dC_LLfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_LL)
    dC_XCfor3x2pt_3D = dC_4D_to_3D(dC_XCfor3x2pt_4D, nbl_3x2pt, zpairs_cross, nparams_tot, ind_XC)
    dC_GGfor3x2pt_3D = dC_GG_3D_v2.copy()  # the GG component of the 3x2pt is equal to the GConly case (same ell_max)

    ######################### FILL DATAVECTOR #####################################

    # fill 3D datavector
    D_WA_3D = dC_WA_3D
    D_WLonly_3D = dC_LL_WLonly_3D
    D_GConly_3D = dC_GG_3D
    D_3x2pt_3D = np.concatenate((dC_LL_3D, dC_XC_3D, dC_GG_3D), axis=1)

    # build the 3x2pt datavector, getting rid of the unnecessary cross-term
    # ! delicate point, double check
    if GL_or_LG == 'GL':
        dC_XC_3D = dC_3x2pt_5D[:, 1, 0, :, :]
    elif GL_or_LG == 'LG':
        dC_XC_3D = dC_3x2pt_5D[:, 0, 1, :, :]

    D_3x2pt_3D = np.concatenate((dC_3x2pt_5D[:, 0, 0, :, :], dC_XC_3D, dC_3x2pt_5D[:, 1, 1, :, :]), axis=1)

    # collapse ell and zpair - ATTENTION: np.reshape, like ndarray.flatten, accepts an 'ordering' parameter, which works
    # in the same way
    # not with the old datavector, which was ordered in a different way...
    D_WA_2D = np.reshape(D_WA_3D, (nbl_WA * zpairs_auto, nparams_tot), order=which_flattening)
    D_WLonly_2D = np.reshape(D_WLonly_3D, (nbl * zpairs_auto, nparams_tot), order=which_flattening)
    D_GConly_2D = np.reshape(D_GConly_3D, (nbl * zpairs_auto, nparams_tot), order=which_flattening)
    D_3x2pt_2D = np.reshape(D_3x2pt_3D, (nbl * zpairs_3x2pt, nparams_tot), order=which_flattening)

    ######################### COMPUTE FM #####################################

    # COMPUTE FM GO
    start3 = time.perf_counter()
    FM_WL_GO = mm.compute_FM_2D(nbl, zpairs_auto, nparams_tot, cov_WL_GO_2D_inv, D_WLonly_2D)
    FM_GC_GO = mm.compute_FM_2D(nbl, zpairs_auto, nparams_tot, cov_GC_GO_2D_inv, D_GConly_2D)
    FM_WA_GO = mm.compute_FM_2D(nbl_WA, zpairs_auto, nparams_tot, cov_WA_GO_2D_inv, D_WA_2D)
    FM_3x2pt_GO = mm.compute_FM_2D(nbl, zpairs_3x2pt, nparams_tot, cov_3x2pt_GO_2D_inv, D_3x2pt_2D)
    print(f'GO FM done in {(time.perf_counter() - start3):.2f} s')

    # COMPUTE FM GS
    start4 = time.perf_counter()
    FM_WL_GS = mm.compute_FM_2D(nbl, zpairs_auto, nparams_tot, cov_WL_GS_2D_inv, D_WLonly_2D)
    FM_GC_GS = mm.compute_FM_2D(nbl, zpairs_auto, nparams_tot, cov_GC_GS_2D_inv, D_GConly_2D)
    FM_WA_GS = mm.compute_FM_2D(nbl_WA, zpairs_auto, nparams_tot, cov_WA_GS_2D_inv, D_WA_2D)
    FM_3x2pt_GS = mm.compute_FM_2D(nbl, zpairs_3x2pt, nparams_tot, cov_3x2pt_GS_2D_inv, D_3x2pt_2D)
    print(f'GS FM done in {(time.perf_counter() - start4):.2f} s')

    # sum WA, this is the actual FM_3x2pt
    if use_WA:
        FM_3x2pt_GO += FM_WA_GO
        FM_3x2pt_GS += FM_WA_GS

    # delete null rows and columns to avoid having singular matrices
    # WL for 3x2pt has no bias
    FM_WL_GO_toSave = FM_WL_GO[:nParams_WL, :nParams_WL]
    FM_WL_GS_toSave = FM_WL_GS[:nParams_WL, :nParams_WL]
    # GConly has no IA parameters
    FM_GC_GO_toSave = np.delete(FM_GC_GO, (7, 8, 9), 0)
    FM_GC_GO_toSave = np.delete(FM_GC_GO_toSave, (7, 8, 9), 1)
    FM_GC_GS_toSave = np.delete(FM_GC_GS, (7, 8, 9), 0)
    FM_GC_GS_toSave = np.delete(FM_GC_GS_toSave, (7, 8, 9), 1)

    # save dictionary
    probe_names = ['WL', 'GC', '3x2pt']
    FMs_GO = [FM_WL_GO_toSave, FM_GC_GO_toSave, FM_3x2pt_GO]
    FMs_GS = [FM_WL_GS_toSave, FM_GC_GS_toSave, FM_3x2pt_GS]

    FM_dict = {}
    for probe_name, FM_GO, FM_GS in zip(probe_names, FMs_GO, FMs_GS):
        FM_dict[f'FM_{probe_name}_GO'] = FM_GO
        FM_dict[f'FM_{probe_name}_GS'] = FM_GS

    print("FMs computed in %.2f seconds" % (time.perf_counter() - start))

    return FM_dict

    # TODO: create pd dataframe
