import sys
import warnings
from pathlib import Path
import numpy as np


project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib_and_cfg'))
import common_lib.my_module as mm


def import_and_interpolate_cls(general_config, covariance_config, ell_dict):
    """
    This code imports and interpolates and rearranges the Cls
    """

    # import and rename settings:
    nbl = general_config['nbl']
    zbins = general_config['zbins']
    cl_folder = general_config['cl_folder']
    rl_folder = general_config['rl_folder']
    zbins = general_config['zbins']
    zbin_type = general_config['EP_or_ED']

    # import ell values:
    ell_WL = ell_dict['ell_WL']
    ell_GC = ell_dict['ell_GC']
    ell_WA = ell_dict['ell_WA']
    ell_XC = ell_GC

    # nbl for Wadd
    if ell_WA.size == 1:
        nbl_WA = 1  # in the case of just one bin it would give error
    else:
        nbl_WA = ell_WA.shape[0]

    npairs, npairs_asimm, npairs_tot = mm.get_zpairs(zbins)

    # import Vincenzo's (different versions of) Cls
    # also implements a further consistency check on GL/LG
    if 'Cij_thesis' in cl_folder:
        assert covariance_config['GL_or_LG'] == 'LG', 'Cij_thesis uses LG'
        cl_LL_import = np.genfromtxt(f'{cl_folder}/CijGG-N4TB-GR-eNLA.dat')
        cl_XC_import = np.genfromtxt(f'{cl_folder}/CijDG-N4TB-GR-eNLA.dat')
        cl_GG_import = np.genfromtxt(f'{cl_folder}/CijDD-N4TB-GR-eNLA.dat')

    elif 'Cij_15gen' in cl_folder:  # Cij-NonLin-eNLA_15gen
        assert covariance_config['GL_or_LG'] == 'LG', 'Cij_14may uses LG'
        cl_LL_import = np.genfromtxt(f'{cl_folder}/CijLL-LCDM-NonLin-eNLA.dat')
        cl_XC_import = np.genfromtxt(f'{cl_folder}/CijLG-LCDM-NonLin-eNLA.dat')
        cl_GG_import = np.genfromtxt(f'{cl_folder}/CijGG-LCDM-NonLin-eNLA.dat')
        cl_LL_import[:, 0] = np.log10(cl_LL_import[:, 0])
        cl_XC_import[:, 0] = np.log10(cl_XC_import[:, 0])
        cl_GG_import[:, 0] = np.log10(cl_GG_import[:, 0])

    elif '14may' in cl_folder:
        assert covariance_config['GL_or_LG'] == 'GL', 'Cij_14may uses GL'
        cl_LL_import = np.genfromtxt(f'{cl_folder.format(EP_or_ED=zbin_type, zbins=zbins)}/CijLL-GR-Flat-eNLA-NA.dat')
        cl_XC_import = np.genfromtxt(f'{cl_folder.format(EP_or_ED=zbin_type, zbins=zbins)}/CijGL-GR-Flat-eNLA-NA.dat')
        cl_GG_import = np.genfromtxt(f'{cl_folder.format(EP_or_ED=zbin_type, zbins=zbins)}/CijGG-GR-Flat-eNLA-NA.dat')

    elif 'Cij_SPV3' in cl_folder:
        assert 1 > 2, 'Cij_SPV3 is not implemented'
        assert covariance_config['GL_or_LG'] == 'GL', 'Cij_SPV3 uses GL'
        cl_LL_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijLL-GR-Flat-eNLA-NA.dat')
        cl_XC_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijGL-GR-Flat-eNLA-NA.dat')
        cl_GG_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijGG-GR-Flat-eNLA-NA.dat')

    else:
        raise ValueError('cl_folder must contain the string Cij_15gen, Cij_thesis or Cij_14may')

    # import responses
    R_LL_import = np.genfromtxt(f'{rl_folder.format(EP_or_ED=zbin_type, zbins=zbins)}/rijllcorr-istf-alex.dat')
    R_GL_import = np.genfromtxt(f'{rl_folder.format(EP_or_ED=zbin_type, zbins=zbins)}/rijglcorr-istf-alex.dat')
    R_GG_import = np.genfromtxt(f'{rl_folder.format(EP_or_ED=zbin_type, zbins=zbins)}/rijggcorr-istf-alex.dat')

    ###########################################################################
    # interpolate Vincenzo's Cls in ell values
    # careful, this part is a bit tricky. Pay attention to the ell_WL,
    # ell_XC arguments in e.g. fLL(ell_XC) vs fLL(ell_WL)
    cl_dict_2D = {}
    cl_dict_2D['cl_LL_2D'] = mm.cl_interpolator(cl_LL_import, npairs, ell_WL, nbl)
    cl_dict_2D['cl_GG_2D'] = mm.cl_interpolator(cl_GG_import, npairs, ell_XC, nbl)
    cl_dict_2D['cl_WA_2D'] = mm.cl_interpolator(cl_LL_import, npairs, ell_WA, nbl_WA)
    cl_dict_2D['cl_XC_2D'] = mm.cl_interpolator(cl_XC_import, npairs_asimm, ell_XC, nbl)
    cl_dict_2D['cl_LLfor3x2pt_2D'] = mm.cl_interpolator(cl_LL_import, npairs, ell_XC, nbl)

    rl_dict_2D = {}
    rl_dict_2D['rl_LL_2D'] = mm.cl_interpolator(R_LL_import, npairs, ell_WL, nbl)
    rl_dict_2D['rl_GG_2D'] = mm.cl_interpolator(R_GG_import, npairs, ell_XC, nbl)
    rl_dict_2D['rl_WA_2D'] = mm.cl_interpolator(R_LL_import, npairs, ell_WA, nbl_WA)
    rl_dict_2D['rl_XC_2D'] = mm.cl_interpolator(R_GL_import, npairs_asimm, ell_XC, nbl)
    rl_dict_2D['rl_LLfor3x2pt_2D'] = mm.cl_interpolator(R_LL_import, npairs, ell_XC, nbl)

    return cl_dict_2D, rl_dict_2D


def reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D, rl_dict_2D):
    # fill the 3D (nbl x zbins x zbins) matrices, or equivalently nbl (zbins x zbins) matrices

    print(general_config)

    print('note: this function makes no sense, generalize it to work with responses OR cls')
    nbl = general_config['nbl']
    ell_max_WL = general_config['ell_max_WL']
    ell_max_GC = general_config['ell_max_GC']
    zbins = general_config['zbins']
    cl_folder = general_config['cl_folder']
    n_probes = general_config['n_probes']

    # import ell values:
    ell_WL = ell_dict['ell_WL']
    ell_GC = ell_dict['ell_GC']
    ell_WA = ell_dict['ell_WA']
    ell_XC = ell_GC

    cl_LL_2D = cl_dict_2D['cl_LL_2D']
    cl_GG_2D = cl_dict_2D['cl_GG_2D']
    cl_WA_2D = cl_dict_2D['cl_WA_2D']
    cl_GL_2D = cl_dict_2D['cl_GL_2D']
    cl_LLfor3x2pt_2D = cl_dict_2D['cl_LLfor3x2pt_2D']

    rl_LL_2D = rl_dict_2D['rl_LL_2D']
    rl_GG_2D = rl_dict_2D['rl_GG_2D']
    rl_WA_2D = rl_dict_2D['rl_WA_2D']
    rl_GL_2D = rl_dict_2D['rl_GL_2D']
    rl_LLfor3x2pt_2D = rl_dict_2D['rl_LLfor3x2pt_2D']

    # compute n_zpairs
    npairs, npairs_asimm, npairs_tot = mm.get_zpairs(zbins)

    # nbl for Wadd
    if np.asanyarray(ell_WA).size == 1:
        nbl_WA = 1  # in the case of just one bin it would give error
    else:
        nbl_WA = ell_WA.shape[0]

    # initialize cls arrays
    cl_LL_3D = np.zeros((nbl, zbins, zbins))  # 3D, for WLonly
    cl_LLfor3x2pt_3D = np.zeros((nbl, zbins, zbins))  # 3D, for the datavector
    cl_GG_3D = np.zeros((nbl, zbins, zbins))  # 3D, for GConly
    cl_WA_3D = np.zeros((nbl_WA, zbins, zbins))  # 3D, ONLY for the datavector (there's no Wadd_only case)
    cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))

    rl_LL_3D = np.zeros((nbl, zbins, zbins))
    rl_LLfor3x2pt_3D = np.zeros((nbl, zbins, zbins))
    rl_GG_3D = np.zeros((nbl, zbins, zbins))
    rl_WA_3D = np.zeros((nbl_WA, zbins, zbins))
    rl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))

    # fill upper triangle: LL, GG, WLonly
    triu_idx = np.triu_indices(zbins)
    for ell in range(nbl):
        for i in range(npairs):
            cl_LL_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_LL_2D[ell, i]
            cl_GG_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_GG_2D[ell, i]
            cl_LLfor3x2pt_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_LLfor3x2pt_2D[ell, i]

            rl_LL_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_LL_2D[ell, i]
            rl_GG_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_GG_2D[ell, i]
            rl_LLfor3x2pt_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_LLfor3x2pt_2D[ell, i]

    # Wadd
    for ell in range(nbl_WA):
        for i in range(npairs):
            cl_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_WA_2D[ell, i]
            rl_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_WA_2D[ell, i]

    # fill asymmetric
    cl_XC_3D = np.reshape(cl_GL_2D, (nbl, zbins, zbins))
    rl_XC_3D = np.reshape(rl_GL_2D, (nbl, zbins, zbins))

    # symmetrize
    cl_LL_3D = mm.fill_3D_symmetric_array(cl_LL_3D, nbl, zbins)
    cl_LLfor3x2pt_3D = mm.fill_3D_symmetric_array(cl_LLfor3x2pt_3D, nbl, zbins)
    cl_GG_3D = mm.fill_3D_symmetric_array(cl_GG_3D, nbl, zbins)
    cl_WA_3D = mm.fill_3D_symmetric_array(cl_WA_3D, nbl_WA, zbins)

    rl_LL_3D = mm.fill_3D_symmetric_array(rl_LL_3D, nbl, zbins)
    rl_LLfor3x2pt_3D = mm.fill_3D_symmetric_array(rl_LLfor3x2pt_3D, nbl, zbins)
    rl_GG_3D = mm.fill_3D_symmetric_array(rl_GG_3D, nbl, zbins)
    rl_WA_3D = mm.fill_3D_symmetric_array(rl_WA_3D, nbl_WA, zbins)

    # fill datavector correctly:
    print('is this way of filling the datavector agnostic to LG, GL???')
    # ! pay attention to LG, GL...
    cl_3x2pt_5D[0, 0, :, :, :] = cl_LLfor3x2pt_3D
    cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_3D
    cl_3x2pt_5D[0, 1, :, :, :] = np.transpose(cl_XC_3D, (0, 2, 1))
    cl_3x2pt_5D[1, 0, :, :, :] = cl_XC_3D

    # ! pay attention to LG, GL...
    rl_3x2pt_5D[0, 0, :, :, :] = rl_LLfor3x2pt_3D
    rl_3x2pt_5D[1, 1, :, :, :] = rl_GG_3D
    rl_3x2pt_5D[0, 1, :, :, :] = np.transpose(rl_XC_3D, (0, 2, 1))
    rl_3x2pt_5D[1, 0, :, :, :] = rl_XC_3D

    # create dict with results:
    cl_dict_3D = {
        'cl_LL_3D': cl_LL_3D,
        'cl_GG_3D': cl_GG_3D,
        'cl_WA_3D': cl_WA_3D,
        'cl_3x2pt_5D': cl_3x2pt_5D}

    Rl_dict_3D = {
        'rl_LL_3D': rl_LL_3D,
        'rl_GG_3D': rl_GG_3D,
        'rl_WA_3D': rl_WA_3D,
        'rl_3x2pt_5D': rl_3x2pt_5D}

    print('Cls and responses reshaped')

    return cl_dict_3D, Rl_dict_3D


def build_3x2pt_datavector_5D(dv_LLfor3x2pt_3D, dv_GL_3D, dv_GG_3D, nbl, zbins, n_probes=2):
    dv_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
    dv_3x2pt_5D[0, 0, :, :, :] = dv_LLfor3x2pt_3D
    dv_3x2pt_5D[1, 0, :, :, :] = dv_GL_3D
    dv_3x2pt_5D[0, 1, :, :, :] = np.transpose(dv_GL_3D, (0, 2, 1))
    dv_3x2pt_5D[1, 1, :, :, :] = dv_GG_3D
    return dv_3x2pt_5D


def get_spv3_cls_3d(probe: str, nbl: int, general_cfg: dict, zbins: int, cl_or_rl: str,
                    EP_or_ED: str):
    warnings.warn('THIS FUNCTION SHOULD BE DEPRECATED')
    """This function imports and interpolates the SPV3 cls, which have a different format than the usual input files"""

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
    specs = general_cfg['specs']
    nbl_WL_32 = general_cfg['nbl_WL_32']
    input_folder = general_cfg[f'{cl_or_rl}_folder']

    # default values, changed only for the 3x2pt case
    zpairs = zpairs_auto

    if probe == 'WL':
        probe_here = 'WLO'
    elif probe == 'WA':
        probe_here = 'WLA'
    elif probe == 'GC':
        probe_here = 'GCO'
    elif probe == '3x2pt':
        probe_here = probe
        zpairs = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, WA, GC or 3x2pt')

    if cl_or_rl == 'cl':
        name = 'dv'
    elif cl_or_rl == 'rl':
        name = 'rf'
    else:
        raise ValueError('cl_or_rl must be "cl" or "rl"')

    if 'SPV3_07_2022/Flagship_1' in input_folder:
        input_folder = f'{input_folder}/{probe_here}'
        filename = f'{name}-{probe_here}-{nbl_WL_32}-{specs}-{EP_or_ED}{zbins:02}.dat'
    elif 'SPV3_07_2022/Flagship_2' in input_folder:
        filename = f'{name}-{probe_here}-Opt-{EP_or_ED}{zbins:02}-FS2.dat'
    else:
        raise ValueError('input_folder should contain "SPV3_07_2022/Flagship_1" or "SPV3_07_2022/Flagship_2"')

    cl_1d = np.genfromtxt(f'{input_folder}/{filename}')

    # ! delete below
    # this check can only be done for the optimistic case, since these are the only datavectors I have (from which
    # I can obtain the pessimistic ones simply by removing some ell bins)
    assert zpairs == int(cl_1d.shape[0] / nbl), 'the number of elements in the datavector is incompatible with ' \
                                                'the number of ell bins for this case/probe'

    cl_3d = cl_SPV3_1D_to_3D(cl_1d, probe, nbl, zbins)
    return cl_3d


# @njit
def cl_SPV3_1D_to_3D(cl_1d, probe: str, nbl: int, zbins: int):
    """This function reshapes the SPV3 cls, which have a different format wrt the usual input files, from 1 to 3
    dimensions (5 dimensions for the 3x2pt case)"""

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    # the checks on zpairs in the if statements can only be done for the optimistic case, since these are the only
    # datavectors I have (from which I can obtain the pessimistic ones simply by removing some ell bins).

    # This case switch is not to repeat the assert below for each case
    if probe in ['WL', 'WA', 'GC']:
        zpairs = zpairs_auto
        is_symmetric = True
    elif probe == 'XC':
        zpairs = zpairs_cross
        is_symmetric = False
    elif probe == '3x2pt':
        zpairs = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, WA, XC, GC or 3x2pt')

    assert zpairs == int(cl_1d.shape[0] / nbl), 'the number of elements in the datavector is incompatible ' \
                                                'with the number of ell bins for this case/probe'

    if probe != '3x2pt':
        cl_3d = mm.cl_1D_to_3D(cl_1d, nbl, zbins, is_symmetric=is_symmetric)

        # if cl is not a cross-spectrum, symmetrize
        if probe != 'XC':
            cl_3d = mm.fill_3D_symmetric_array(cl_3d, nbl, zbins)
        return cl_3d

    elif probe == '3x2pt':
        cl_2d = np.reshape(cl_1d, (nbl, zpairs_3x2pt))

        # split into 3 2d datavectors
        cl_ll_3x2pt_2d = cl_2d[:, :zpairs_auto]
        cl_gl_3x2pt_2d = cl_2d[:, zpairs_auto:zpairs_auto + zpairs_cross]
        cl_gg_3x2pt_2d = cl_2d[:, zpairs_auto + zpairs_cross:]

        # reshape them individually - the symmetrization is done within the function
        cl_ll_3x2pt_3d = mm.cl_2D_to_3D_symmetric(cl_ll_3x2pt_2d, nbl=nbl, zpairs=zpairs_auto, zbins=zbins)
        cl_gl_3x2pt_3d = mm.cl_2D_to_3D_asymmetric(cl_gl_3x2pt_2d, nbl=nbl, zbins=zbins, order='C')
        cl_gg_3x2pt_3d = mm.cl_2D_to_3D_symmetric(cl_gg_3x2pt_2d, nbl=nbl, zpairs=zpairs_auto, zbins=zbins)

        # use them to populate the datavector
        cl_3x2pt = np.zeros((2, 2, nbl, zbins, zbins))
        cl_3x2pt[0, 0, :, :, :] = cl_ll_3x2pt_3d
        cl_3x2pt[1, 1, :, :, :] = cl_gg_3x2pt_3d
        cl_3x2pt[1, 0, :, :, :] = cl_gl_3x2pt_3d
        cl_3x2pt[0, 1, :, :, :] = np.transpose(cl_gl_3x2pt_3d, (0, 2, 1))
        return cl_3x2pt  # in this case, return the datavector (I could name it "cl_3d" and avoid this return statement,
        # but it's not 3d!)


def cl_BNT_transform(cl_3D, BNT_matrix, probe_A, probe_B):
    assert cl_3D.ndim == 3, 'cl_3D must be 3D'
    assert BNT_matrix.ndim == 2, 'BNT_matrix must be 2D'
    assert cl_3D.shape[1] == BNT_matrix.shape[0], 'the number of ell bins in cl_3D and BNT_matrix must be the same'

    BNT_transform_dict = {
        'L': BNT_matrix,
        'G': np.eye(BNT_matrix.shape[0]),
    }

    cl_3D_BNT = np.zeros(cl_3D.shape)
    for ell_idx in range(cl_3D.shape[0]):
        cl_3D_BNT[ell_idx, :, :] = BNT_transform_dict[probe_A] @ \
                                   cl_3D[ell_idx, :, :] @ \
                                   BNT_transform_dict[probe_B].T

    return cl_3D_BNT


def cl_BNT_transform_3x2pt(cl_3x2pt_5D, BNT_matrix):
    """wrapper function to quickly implement the cl (or derivatives) BNT transform for the 3x2pt datavector"""

    cl_3x2pt_5D_BNT = np.zeros(cl_3x2pt_5D.shape)
    cl_3x2pt_5D_BNT[0, 0, :, :, :] = cl_BNT_transform(cl_3x2pt_5D[0, 0, :, :, :], BNT_matrix, 'L', 'L')
    cl_3x2pt_5D_BNT[0, 1, :, :, :] = cl_BNT_transform(cl_3x2pt_5D[0, 1, :, :, :], BNT_matrix, 'L', 'G')
    cl_3x2pt_5D_BNT[1, 0, :, :, :] = cl_BNT_transform(cl_3x2pt_5D[1, 0, :, :, :], BNT_matrix, 'G', 'L')
    cl_3x2pt_5D_BNT[1, 1, :, :, :] = cl_3x2pt_5D[1, 1, :, :, :]  # no need to transform the GG part

    return cl_3x2pt_5D_BNT


def get_ell_cuts_indices(ell_values, ell_cuts_2d_array, zbins):
    """ creates an array of lists containing the ell indices to cut (to set to 0) for each zi, zj)"""
    ell_idxs_tocut = np.zeros((zbins, zbins), dtype=list)
    for zi in range(zbins):
        for zj in range(zbins):
            ell_cut = ell_cuts_2d_array[zi, zj]
            if np.any(ell_values > ell_cut):  # i.e., if you need to do a cut at all
                ell_idxs_tocut[zi, zj] = np.where(ell_values > ell_cut)[0]
            else:
                ell_idxs_tocut[zi, zj] = np.array([])

    return ell_idxs_tocut


def cl_ell_cut(cl_3D, ell_values, ell_cuts_matrix):
    """cut (sets to zero) the cl_3D array at the ell values specified in ell_cuts_matrix"""

    # TODO call get_ell_cuts_indices function here not to repeat code

    nbl = cl_3D.shape[0]
    zbins = cl_3D.shape[1]

    assert cl_3D.ndim == 3, 'cl_3D must be 3D'
    assert ell_cuts_matrix.ndim == 2, 'ell_cuts_matrix must be 2D'
    assert cl_3D.shape[1] == cl_3D.shape[2], 'the last two axes\' dimensions do not coincide'
    assert nbl == ell_values.shape[0], 'the number of ell bins in cl_3D and ell_values must be the same'
    assert zbins == ell_cuts_matrix.shape[0], 'the number of zbins in cl_3D and ell_cuts_matrix axes\' length ' \
                                              'must be the same'

    cl_3D_ell_cut = cl_3D.copy()
    for zi in range(zbins):
        for zj in range(zbins):
            ell_cut = ell_cuts_matrix[zi, zj]
            if np.any(ell_values > ell_cut):  # i.e., if you need to do a cut at all
                ell_idxs_tocut = np.where(ell_values > ell_cut)[0]
                cl_3D_ell_cut[ell_idxs_tocut, zi, zj] = 0

    return cl_3D_ell_cut


def cl_ell_cut_v2(cl_3D, ell_values, ell_cuts_matrix):
    """cut (sets to zero) the cl_3D array at the ell values specified in ell_cuts_matrix.
    Smarter version, without for loops - only marginally faster"""

    nbl = cl_3D.shape[0]
    zbins = cl_3D.shape[1]

    assert cl_3D.ndim == 3, 'cl_3D must be 3D'
    assert ell_cuts_matrix.ndim == 2, 'ell_cuts_matrix must be 2D'
    assert cl_3D.shape[1] == cl_3D.shape[2], 'the last two axes\' dimensions do not coincide'
    assert nbl == ell_values.shape[0], 'the number of ell bins in cl_3D and ell_values must be the same'
    assert zbins == ell_cuts_matrix.shape[0], 'the number of zbins in cl_3D and ell_cuts_matrix axes\' length ' \
                                              'must be the same'

    # Create a 3D mask of the same shape as cl_3D where the
    # elements that should be cut are marked as True
    ell_cuts_matrix_3D = np.expand_dims(ell_cuts_matrix, 0)
    mask = (ell_values[:, None, None] > ell_cuts_matrix_3D)

    # Use the mask to set the corresponding elements of cl_3D to zero
    cl_3D_ell_cut = np.where(mask, 0, cl_3D)

    return cl_3D_ell_cut


def cl_ell_cut_3x2pt(cl_3x2pt_5D, ell_cuts_dict, ell_values_3x2pt):
    """wrapper function to quickly implement the cl (or derivatives) ell cut for the 3x2pt datavector"""

    cl_LLfor3x2pt_3D = cl_3x2pt_5D[0, 0, :, :, :]
    cl_LGfor3x2pt_3D = cl_3x2pt_5D[0, 1, :, :, :]
    cl_GLfor3x2pt_3D = cl_3x2pt_5D[1, 0, :, :, :]
    cl_GGfor3x2pt_3D = cl_3x2pt_5D[1, 1, :, :, :]

    cl_LLfor3x2pt_3D_ell_cut = cl_ell_cut(cl_LLfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['LL'])
    cl_LGfor3x2pt_3D_ell_cut = cl_ell_cut(cl_LGfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['LG'])
    cl_GLfor3x2pt_3D_ell_cut = cl_ell_cut(cl_GLfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['GL'])
    cl_GGfor3x2pt_3D_ell_cut = cl_ell_cut(cl_GGfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['GG'])

    cl_3x2pt_5D_ell_cut = np.zeros(cl_3x2pt_5D.shape)
    cl_3x2pt_5D_ell_cut[0, 0, :, :, :] = cl_LLfor3x2pt_3D_ell_cut
    cl_3x2pt_5D_ell_cut[0, 1, :, :, :] = cl_LGfor3x2pt_3D_ell_cut
    cl_3x2pt_5D_ell_cut[1, 0, :, :, :] = cl_GLfor3x2pt_3D_ell_cut
    cl_3x2pt_5D_ell_cut[1, 1, :, :, :] = cl_GGfor3x2pt_3D_ell_cut

    return cl_3x2pt_5D_ell_cut
