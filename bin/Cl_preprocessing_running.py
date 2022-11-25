import sys
from pathlib import Path
import numpy as np
# from numba import njit

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm


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
        C_LL_import = np.genfromtxt(f'{cl_folder}/CijGG-N4TB-GR-eNLA.dat')
        C_XC_import = np.genfromtxt(f'{cl_folder}/CijDG-N4TB-GR-eNLA.dat')
        C_GG_import = np.genfromtxt(f'{cl_folder}/CijDD-N4TB-GR-eNLA.dat')

    elif 'Cij_15gen' in cl_folder:  # Cij-NonLin-eNLA_15gen
        assert covariance_config['GL_or_LG'] == 'LG', 'Cij_14may uses LG'
        C_LL_import = np.genfromtxt(f'{cl_folder}/CijLL-LCDM-NonLin-eNLA.dat')
        C_XC_import = np.genfromtxt(f'{cl_folder}/CijLG-LCDM-NonLin-eNLA.dat')
        C_GG_import = np.genfromtxt(f'{cl_folder}/CijGG-LCDM-NonLin-eNLA.dat')
        C_LL_import[:, 0] = np.log10(C_LL_import[:, 0])
        C_XC_import[:, 0] = np.log10(C_XC_import[:, 0])
        C_GG_import[:, 0] = np.log10(C_GG_import[:, 0])

    elif '14may' in cl_folder:
        assert covariance_config['GL_or_LG'] == 'GL', 'Cij_14may uses GL'
        C_LL_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijLL-GR-Flat-eNLA-NA.dat')
        C_XC_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijGL-GR-Flat-eNLA-NA.dat')
        C_GG_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijGG-GR-Flat-eNLA-NA.dat')

    elif 'Cij_SPV3' in cl_folder:
        assert 1 > 2, 'Cij_SPV3 is not implemented'
        assert covariance_config['GL_or_LG'] == 'GL', 'Cij_SPV3 uses GL'
        C_LL_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijLL-GR-Flat-eNLA-NA.dat')
        C_XC_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijGL-GR-Flat-eNLA-NA.dat')
        C_GG_import = np.genfromtxt(f'{cl_folder}/{zbin_type}{zbins:02}/CijGG-GR-Flat-eNLA-NA.dat')

    else:
        raise ValueError('cl_folder must contain the string Cij_15gen, Cij_thesis or Cij_14may')

    # import responses
    R_LL_import = np.genfromtxt(f'{rl_folder}/rijllcorr-istf-alex.dat')
    R_GL_import = np.genfromtxt(f'{rl_folder}/rijglcorr-istf-alex.dat')
    R_GG_import = np.genfromtxt(f'{rl_folder}/rijggcorr-istf-alex.dat')

    ###########################################################################
    # interpolate Vincenzo's Cls in ell values
    # careful, this part is a bit tricky. Pay attention to the ell_WL,
    # ell_XC arguments in e.g. fLL(ell_XC) vs fLL(ell_WL)
    cl_dict_2D = {}
    cl_dict_2D['C_LL_2D'] = mm.Cl_interpolator(npairs, C_LL_import, ell_XC, nbl)
    cl_dict_2D['C_GG_2D'] = mm.Cl_interpolator(npairs, C_GG_import, ell_XC, nbl)
    cl_dict_2D['C_WA_2D'] = mm.Cl_interpolator(npairs, C_LL_import, ell_WA, nbl_WA)
    cl_dict_2D['C_XC_2D'] = mm.Cl_interpolator(npairs_asimm, C_XC_import, ell_XC, nbl)
    cl_dict_2D['C_LL_WLonly_2D'] = mm.Cl_interpolator(npairs, C_LL_import, ell_WL, nbl)

    Rl_dict_2D = {}
    Rl_dict_2D['R_LL_2D'] = mm.Cl_interpolator(npairs, R_LL_import, ell_XC, nbl)
    Rl_dict_2D['R_GG_2D'] = mm.Cl_interpolator(npairs, R_GG_import, ell_XC, nbl)
    Rl_dict_2D['R_WA_2D'] = mm.Cl_interpolator(npairs, R_LL_import, ell_WA, nbl_WA)
    Rl_dict_2D['R_XC_2D'] = mm.Cl_interpolator(npairs_asimm, R_GL_import, ell_XC, nbl)
    Rl_dict_2D['R_LL_WLonly_2D'] = mm.Cl_interpolator(npairs, R_LL_import, ell_WL, nbl)

    return cl_dict_2D, Rl_dict_2D


def reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D, Rl_dict_2D):
    # fill the 3D (nbl x zbins x zbins) matrices, or equivalently nbl (zbins x zbins) matrices

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
    cl_XC_2D = cl_dict_2D['cl_XC_2D']
    cl_LLfor3x2pt_2D = cl_dict_2D['cl_LLfor3x2pt_2D']

    rl_LL_2D = Rl_dict_2D['rl_LL_2D']
    rl_GG_2D = Rl_dict_2D['rl_GG_2D']
    rl_WA_2D = Rl_dict_2D['rl_WA_2D']
    rl_XC_2D = Rl_dict_2D['rl_XC_2D']
    rl_LLfor3x2pt_2D = Rl_dict_2D['rl_LLfor3x2pt_2D']

    # compute n_zpairs
    npairs, npairs_asimm, npairs_tot = mm.get_zpairs(zbins)

    # nbl for Wadd
    if np.asanyarray(ell_WA).size == 1:
        nbl_WA = 1  # in the case of just one bin it would give error
    else:
        nbl_WA = ell_WA.shape[0]

    # initialize cls arrays
    C_LL_WLonly_3D = np.zeros((nbl, zbins, zbins))  # 3D, for WLonly
    C_LL_3D = np.zeros((nbl, zbins, zbins))  # 3D, for the datavector
    C_GG_3D = np.zeros((nbl, zbins, zbins))  # 3D, for GConly
    C_WA_3D = np.zeros((nbl_WA, zbins, zbins))  # 3D, ONLY for the datavector (there's no Wadd_only case)
    C_3x2pt_5D = np.zeros((nbl, n_probes, n_probes, zbins, zbins))

    R_LL_WLonly_3D = np.zeros((nbl, zbins, zbins))
    R_LL_3D = np.zeros((nbl, zbins, zbins))
    R_GG_3D = np.zeros((nbl, zbins, zbins))
    R_WA_3D = np.zeros((nbl_WA, zbins, zbins))
    R_3x2pt_5D = np.zeros((nbl, n_probes, n_probes, zbins, zbins))

    # fill upper triangle: LL, GG, WLonly
    triu_idx = np.triu_indices(zbins)
    for ell in range(nbl):
        for i in range(npairs):
            C_LL_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_LLfor3x2pt_2D[ell, i]
            C_GG_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_GG_2D[ell, i]
            C_LL_WLonly_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_LL_2D[ell, i]

            R_LL_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_LLfor3x2pt_2D[ell, i]
            R_GG_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_GG_2D[ell, i]
            R_LL_WLonly_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_LL_2D[ell, i]

    # Wadd
    for ell in range(nbl_WA):
        for i in range(npairs):
            C_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = cl_WA_2D[ell, i]
            R_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = rl_WA_2D[ell, i]

    # fill asymmetric
    C_XC_3D = np.reshape(cl_XC_2D, (nbl, zbins, zbins))
    R_XC_3D = np.reshape(rl_XC_2D, (nbl, zbins, zbins))

    # symmetrize
    C_LL_WLonly_3D = mm.fill_3D_symmetric_array(C_LL_WLonly_3D, nbl, zbins)
    C_LL_3D = mm.fill_3D_symmetric_array(C_LL_3D, nbl, zbins)
    C_GG_3D = mm.fill_3D_symmetric_array(C_GG_3D, nbl, zbins)
    C_WA_3D = mm.fill_3D_symmetric_array(C_WA_3D, nbl_WA, zbins)

    R_LL_WLonly_3D = mm.fill_3D_symmetric_array(R_LL_WLonly_3D, nbl, zbins)
    R_LL_3D = mm.fill_3D_symmetric_array(R_LL_3D, nbl, zbins)
    R_GG_3D = mm.fill_3D_symmetric_array(R_GG_3D, nbl, zbins)
    R_WA_3D = mm.fill_3D_symmetric_array(R_WA_3D, nbl_WA, zbins)

    # fill datavector correctly:
    print('is this way of filling the datavector agnostic to LG, GL???')
    # ! pay attention to LG, GL...
    C_3x2pt_5D[:, 0, 0, :, :] = C_LL_3D
    C_3x2pt_5D[:, 1, 1, :, :] = C_GG_3D
    C_3x2pt_5D[:, 0, 1, :, :] = np.transpose(C_XC_3D, (0, 2, 1))
    C_3x2pt_5D[:, 1, 0, :, :] = C_XC_3D

    # ! pay attention to LG, GL...
    R_3x2pt_5D[:, 0, 0, :, :] = R_LL_3D
    R_3x2pt_5D[:, 1, 1, :, :] = R_GG_3D
    R_3x2pt_5D[:, 0, 1, :, :] = np.transpose(R_XC_3D, (0, 2, 1))
    R_3x2pt_5D[:, 1, 0, :, :] = R_XC_3D

    # create dict with results:
    cl_dict_3D = {
        'C_LL_WLonly_3D': C_LL_WLonly_3D,
        'C_GG_3D': C_GG_3D,
        'C_WA_3D': C_WA_3D,
        'C_3x2pt_5D': C_3x2pt_5D}

    Rl_dict_3D = {
        'R_LL_WLonly_3D': R_LL_WLonly_3D,
        'R_GG_3D': R_GG_3D,
        'R_WA_3D': R_WA_3D,
        'R_3x2pt_5D': R_3x2pt_5D}

    print('Cls and responses reshaped')

    return cl_dict_3D, Rl_dict_3D


def get_spv3_cls_3d(probe: str, nbl: int, general_cfg: dict, zbins: int, cl_or_rl: str,
                    EP_or_ED: str):
    print('THIS FUNCTION SHOULD BE DEPRECATED')
    """This function imports and interpolates the CPV3 cls, which have a different format wrt the usual input files"""

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

    assert probe in ['WL', 'WA', 'GC', '3x2pt'], 'probe must be WL, WA, GC or 3x2pt'

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    # the checks on zpairs in the if statements can only be done for the optimistic case, since these are the only
    # datavectors I have (from which I can obtain the pessimistic ones simply by removing some ell bins)
    if probe != '3x2pt':
        assert zpairs_auto == int(cl_1d.shape[0] / nbl), 'the number of elements in the datavector is incompatible ' \
                                                         'with the number of ell bins for this case/probe'
        cl_3d = mm.cl_1D_to_3D(cl_1d, nbl, zbins, is_symmetric=True)
        cl_3d = mm.fill_3D_symmetric_array(cl_3d, nbl, zbins)
        return cl_3d

    else:
        assert zpairs_3x2pt == int(cl_1d.shape[0] / nbl), 'the number of elements in the datavector is incompatible ' \
                                                          'with the number of ell bins for this case/probe'
        cl_2d = np.reshape(cl_1d, (nbl, zpairs_3x2pt))

        # split into 3 2d datavectors
        cl_ll_3x2pt_2d = cl_2d[:, :zpairs_auto]
        cl_lg_3x2pt_2d = cl_2d[:, zpairs_auto:zpairs_auto + zpairs_cross]  # ! is it really gl? or lg?
        cl_gg_3x2pt_2d = cl_2d[:, zpairs_auto + zpairs_cross:]

        # reshape them individually - the symmetrization is done within the function
        cl_ll_3x2pt_3d = mm.cl_2D_to_3D_symmetric(cl_ll_3x2pt_2d, nbl=nbl, npairs=zpairs_auto, zbins=zbins)
        cl_lg_3x2pt_3d = mm.cl_2D_to_3D_asymmetric(cl_lg_3x2pt_2d, nbl=nbl, zbins=zbins)
        cl_gg_3x2pt_3d = mm.cl_2D_to_3D_symmetric(cl_gg_3x2pt_2d, nbl=nbl, npairs=zpairs_auto, zbins=zbins)

        # use them to populate the datavector
        cl_3x2pt = np.zeros((nbl, 2, 2, zbins, zbins))
        cl_3x2pt[:, 0, 0, :, :] = cl_ll_3x2pt_3d
        cl_3x2pt[:, 1, 1, :, :] = cl_gg_3x2pt_3d
        cl_3x2pt[:, 0, 1, :, :] = cl_lg_3x2pt_3d
        cl_3x2pt[:, 1, 0, :, :] = np.transpose(cl_lg_3x2pt_3d, (0, 2, 1))
        return cl_3x2pt  # in this case, return the datavector (I could name it "cl_3d" and avoid this return statement,
        # but it's not 3d!)


def cl_BNT_transform(cl_3D, BNT_matrix):
    cl_3D_BNT = np.zeros(cl_3D.shape)
    if cl_3D.ndim == 3:  # WL, GC
        for ell_idx in range(cl_3D.shape[0]):
            cl_3D_BNT[ell_idx, :, :] = BNT_matrix @ cl_3D[ell_idx, :, :] @ BNT_matrix.T

    elif cl_3D.ndim == 5:  # 3x2pt
        for ell_idx in range(cl_3D.shape[0]):
            for probe_A in range(cl_3D.shape[1]):
                for probe_B in range(cl_3D.shape[2]):
                    cl_3D_BNT[ell_idx, probe_A, probe_B, :, :] = BNT_matrix @ \
                                                                 cl_3D[ell_idx, probe_A, probe_B, :, :] @ \
                                                                 BNT_matrix.T
    else:
        raise ValueError('input Cl array should be 3-dim or 5-dim')

    return cl_3D_BNT
