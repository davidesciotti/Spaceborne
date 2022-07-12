import sys
from pathlib import Path
import numpy as np

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

    npairs, npairs_asimm, npairs_tot = mm.get_pairs(zbins)

    # import Vincenzo's (different versions of) Cls
    # also implements a further consistency check on GL/LG
    if cl_folder == "Cij_thesis":
        assert covariance_config['GL_or_LG'] == 'LG', 'Cij_thesis uses LG'
        C_LL_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/thesis_data/Cij_tesi/CijGG-N4TB-GR-eNLA.dat')
        C_XC_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/thesis_data/Cij_tesi/CijDG-N4TB-GR-eNLA.dat')
        C_GG_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/thesis_data/Cij_tesi/CijDD-N4TB-GR-eNLA.dat')

    elif cl_folder == "Cij_15gen":  # Cij-NonLin-eNLA_15gen
        assert covariance_config['GL_or_LG'] == 'LG', 'Cij_14may uses LG'
        C_LL_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/Cij-NonLin-eNLA_15gen/CijLL-LCDM-NonLin-eNLA.dat')
        C_XC_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/Cij-NonLin-eNLA_15gen/CijLG-LCDM-NonLin-eNLA.dat')
        C_GG_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/Cij-NonLin-eNLA_15gen/CijGG-LCDM-NonLin-eNLA.dat')
        C_LL_import[:, 0] = np.log10(C_LL_import[:, 0])
        C_XC_import[:, 0] = np.log10(C_XC_import[:, 0])
        C_GG_import[:, 0] = np.log10(C_GG_import[:, 0])

    elif cl_folder == "Cij_14may":
        assert covariance_config['GL_or_LG'] == 'GL', 'Cij_14may uses GL'
        C_LL_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/CijDers/14may/EP10/CijLL-GR-Flat-eNLA-NA.dat')
        C_XC_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/CijDers/14may/EP10/CijGL-GR-Flat-eNLA-NA.dat')
        C_GG_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/CijDers/14may/EP10/CijGG-GR-Flat-eNLA-NA.dat')

    elif cl_folder == "Cij_SPV3":
        assert 1 > 2, 'Cij_SPV3 is not implemented'
        assert covariance_config['GL_or_LG'] == 'GL', 'Cij_SPV3 uses GL'
        C_LL_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/SPV3_07_2022/DataVecTabs/EP10/CijLL-GR-Flat-eNLA-NA.dat')
        C_XC_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/SPV3_07_2022/DataVecTabs/EP10/CijGL-GR-Flat-eNLA-NA.dat')
        C_GG_import = np.genfromtxt(
            project_path_here / 'config/common_data/vincenzo/SPV3_07_2022/DataVecTabs/EP10/CijGG-GR-Flat-eNLA-NA.dat')



    else:
        raise ValueError('cl_folder must be Cij_15gen, Cij_thesis or Cij_14may')

    # import responses
    R_LL_import = np.genfromtxt(
        project_path_here / 'config/common_data/vincenzo/Pk_responses_2D/rijllcorr-istf-alex.dat')
    R_GL_import = np.genfromtxt(
        project_path_here / 'config/common_data/vincenzo/Pk_responses_2D/rijglcorr-istf-alex.dat')
    R_GG_import = np.genfromtxt(
        project_path_here / 'config/common_data/vincenzo/Pk_responses_2D/rijggcorr-istf-alex.dat')

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

    nbl = general_config['nbl']
    ell_max_WL = general_config['ell_max_WL']
    ell_max_GC = general_config['ell_max_GC']
    zbins = general_config['zbins']
    cl_folder = general_config['cl_folder']
    nProbes = general_config['nProbes']

    # import ell values:
    ell_WL = ell_dict['ell_WL']
    ell_GC = ell_dict['ell_GC']
    ell_WA = ell_dict['ell_WA']
    ell_XC = ell_GC

    C_LL_2D = cl_dict_2D['C_LL_2D']
    C_GG_2D = cl_dict_2D['C_GG_2D']
    C_WA_2D = cl_dict_2D['C_WA_2D']
    C_XC_2D = cl_dict_2D['C_XC_2D']
    C_LL_WLonly_2D = cl_dict_2D['C_LL_WLonly_2D']

    R_LL_2D = Rl_dict_2D['R_LL_2D']
    R_GG_2D = Rl_dict_2D['R_GG_2D']
    R_WA_2D = Rl_dict_2D['R_WA_2D']
    R_XC_2D = Rl_dict_2D['R_XC_2D']
    R_LL_WLonly_2D = Rl_dict_2D['R_LL_WLonly_2D']

    # compute n_zpairs
    npairs, npairs_asimm, npairs_tot = mm.get_pairs(zbins)

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
    D_3x2pt = np.zeros((nbl, nProbes, nProbes, zbins, zbins))

    R_LL_WLonly_3D = np.zeros((nbl, zbins, zbins))
    R_LL_3D = np.zeros((nbl, zbins, zbins))
    R_GG_3D = np.zeros((nbl, zbins, zbins))
    R_WA_3D = np.zeros((nbl_WA, zbins, zbins))
    R_3x2pt = np.zeros((nbl, nProbes, nProbes, zbins, zbins))

    # fill upper triangle: LL, GG, WLonly
    triu_idx = np.triu_indices(zbins)
    for ell in range(nbl):
        for i in range(npairs):
            C_LL_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_LL_2D[ell, i]
            C_GG_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_GG_2D[ell, i]
            C_LL_WLonly_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_LL_WLonly_2D[ell, i]

            R_LL_3D[ell, triu_idx[0][i], triu_idx[1][i]] = R_LL_2D[ell, i]
            R_GG_3D[ell, triu_idx[0][i], triu_idx[1][i]] = R_GG_2D[ell, i]
            R_LL_WLonly_3D[ell, triu_idx[0][i], triu_idx[1][i]] = R_LL_WLonly_2D[ell, i]

    # Wadd
    for ell in range(nbl_WA):
        for i in range(npairs):
            C_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_WA_2D[ell, i]
            R_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = R_WA_2D[ell, i]

    # fill asymmetric
    C_XC_3D = np.reshape(C_XC_2D, (nbl, zbins, zbins))
    R_XC_3D = np.reshape(R_XC_2D, (nbl, zbins, zbins))

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
    D_3x2pt[:, 0, 0, :, :] = C_LL_3D
    D_3x2pt[:, 1, 1, :, :] = C_GG_3D
    D_3x2pt[:, 0, 1, :, :] = np.transpose(C_XC_3D, (0, 2, 1))
    D_3x2pt[:, 1, 0, :, :] = C_XC_3D

    # ! pay attention to LG, GL...
    R_3x2pt[:, 0, 0, :, :] = R_LL_3D
    R_3x2pt[:, 1, 1, :, :] = R_GG_3D
    R_3x2pt[:, 0, 1, :, :] = np.transpose(R_XC_3D, (0, 2, 1))
    R_3x2pt[:, 1, 0, :, :] = R_XC_3D

    # create dict with results:
    cl_dict_3D = {
        'C_LL_WLonly_3D': C_LL_WLonly_3D,
        'C_GG_3D': C_GG_3D,
        'C_WA_3D': C_WA_3D,
        'D_3x2pt': D_3x2pt}

    Rl_dict_3D = {
        'R_LL_WLonly_3D': R_LL_WLonly_3D,
        'R_GG_3D': R_GG_3D,
        'R_WA_3D': R_WA_3D,
        'R_3x2pt': R_3x2pt}

    print('Cls and responses reshaped')

    return cl_dict_3D, Rl_dict_3D


# ! teeeeeest
import matplotlib.pyplot as plt

base_path = '/Users/davide/Documents/Lavoro/Programmi'

cl_ll_1d = np.genfromtxt(f'{base_path}/common_data/vincenzo/SPV3_07_2022/DataVecTabs/WLO/dv-WLO-Opt-EP10.dat')
cl_gg_1d = np.genfromtxt(f'{base_path}/common_data/vincenzo/SPV3_07_2022/DataVecTabs/GCO/dv-GCO-Opt-EP10.dat')

nbl = 32
zbins = 10

cl_ll_3d = mm.cl_1D_to_3D(cl_ll_1d, nbl, zbins)
cl_gg_3d = mm.cl_1D_to_3D(cl_gg_1d, nbl, zbins)

cl_ll_3d = mm.fill_3D_symmetric_array(cl_ll_3d, nbl, zbins)
cl_gg_3d = mm.fill_3D_symmetric_array(cl_gg_3d, nbl, zbins)

for ell in range(nbl):
    if not mm.check_symmetric(cl_ll_3d[ell, ...], rtol=1e-05, atol=1e-08) or \
            not mm.check_symmetric(cl_gg_3d[ell, ...], rtol=1e-05, atol=1e-08):
        print('not symmetric!')

probe = 'WLO'
case = 'Opt'
if case == 'Opt':
    nbl_GC = 29
path = f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/DataVecTabs/{probe}'

zbins_list = (7, 9, 10, 11, 13, 15)

for zbins in zbins_list:
    if zbins in [7, 9]:
        string_0 = '0'
    else:
        string_0 = ''
    file = np.genfromtxt(f'{path}/dv-{probe}-{case}-EP{string_0}{zbins}.dat')

    print(file.shape, file.shape[0]/32, zbins*(zbins+1)/2)

mydict = dict(mm.get_kv_pairs(path))
for key in mydict.keys():
    print(key, mydict[key].shape)







ell = 0
iz, jz = 0, 0
# load old 3D array
cl_ll_3d_old = np.load(f'{base_path}/SSC_restructured_v2/jobs/SSC_comparison/output/cl_3D/C_LL_WLonly_3D.npy')
cl_gg_3d_old = np.load(f'{base_path}/SSC_restructured_v2/jobs/SSC_comparison/output/cl_3D/C_GG_3D.npy')


plt.plot(np.logspace(np.log10(10), np.log10(5000), 30), cl_gg_3d_old[:, iz, jz])
plt.plot(np.logspace(np.log10(10), np.log10(5000), nbl), cl_gg_3d[:, iz, jz])
plt.xscale('log')
# plt.yscale('log')

mm.matshow(cl_gg_3d[ell, :, :], title='new')
mm.matshow(cl_gg_3d_old[ell, :, :], title='old')

plt.plot(cl_gg_1d)