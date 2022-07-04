import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# get project directory
path = Path.cwd().parent.parent
# import configuration and functions modules
sys.path.append(str(path.parent / 'common_data'))
import my_config
sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

###############################################################################
###############################################################################
###############################################################################


# XXX TODO implement ind dictionary like Luca!
# note: old ordering is triU, rows-wise (column-wise? not sure...)
# note: ROW-wise triangular Lower is SEYFERT



def generate_Cls(general_settings, ell_dict):
    """
    this code imports, interpolates and rearranges the Cls, ordering them in 
    (nbl x zbins x zbins) matrices. 
    Note: no need to import and interpolate Vincenzo's Cls, just rearrange Santiago's Cls!
    """
    
    # import settings:
    nbl = general_settings['nbl']
    ell_max_WL = general_settings['ell_max_WL']
    ell_max_GC = general_settings['ell_max_GC']
    zbins = general_settings['zbins']
    Cij_folder = general_settings['Cij_folder']
    nProbes = general_settings['nProbes']
            
    # import ell values:
    ell_WL = ell_dict['ell_WL']
    ell_GC = ell_dict['ell_GC']
    ell_WA = ell_dict['ell_WA']
    ell_XC = ell_GC
        
    # nbl for Wadd
    if np.asanyarray(ell_WA).size == 1: nbl_WA = 1 # in the case of just one bin it would give error
    else: nbl_WA = ell_WA.shape[0]
        
    
    # create Cls arrays
    C_LL_WLonly_3D = np.zeros((nbl, zbins, zbins)) # 3D, for WLonly
    C_LL_3D = np.zeros((nbl, zbins, zbins)) # 3D, for the datavector
    C_GG_3D = np.zeros((nbl, zbins, zbins)) # 3D, for GConly
    C_WA_3D = np.zeros((nbl_WA, zbins, zbins)) # 3D, ONLY for the datavector (there's no Wadd_only case)
    D_3x2pt = np.zeros((nbl, nProbes, nProbes, zbins, zbins))
    
    npairs, npairs_asimm, npairs_tot = mm.get_pairs(zbins)    
    
    
    # import Vincenzo's Cls    
    if Cij_folder == "Cij_thesis":
        C_LL_import = np.genfromtxt(path.parent / 'common_data/vincenzo/thesis_data/Cij_tesi/CijGG-N4TB-GR-eNLA.dat')
        C_XC_import = np.genfromtxt(path.parent / 'common_data/vincenzo/thesis_data/Cij_tesi/CijDG-N4TB-GR-eNLA.dat')
        C_GG_import = np.genfromtxt(path.parent / 'common_data/vincenzo/thesis_data/Cij_tesi/CijDD-N4TB-GR-eNLA.dat')
    elif Cij_folder == "Cij_15gen": # Cij-NonLin-eNLA_15gen
        C_LL_import = np.genfromtxt(path.parent / 'common_data/vincenzo/Cij-NonLin-eNLA_15gen/CijLL-LCDM-NonLin-eNLA.dat')
        C_XC_import = np.genfromtxt(path.parent / 'common_data/vincenzo/Cij-NonLin-eNLA_15gen/CijLG-LCDM-NonLin-eNLA.dat')
        C_GG_import = np.genfromtxt(path.parent / 'common_data/vincenzo/Cij-NonLin-eNLA_15gen/CijGG-LCDM-NonLin-eNLA.dat')
        C_LL_import[:,0] = np.log10(C_LL_import[:,0])
        C_XC_import[:,0] = np.log10(C_XC_import[:,0])
        C_GG_import[:,0] = np.log10(C_GG_import[:,0])
        print('ATTENTION! XC is LG, not GL!!')
    if Cij_folder == "Cij_14may":
        C_LL_import = np.genfromtxt(path.parent / 'common_data/vincenzo/14may/CijDers/EP10/CijLL-GR-Flat-eNLA-NA.dat')
        C_XC_import = np.genfromtxt(path.parent / 'common_data/vincenzo/14may/CijDers/EP10/CijGL-GR-Flat-eNLA-NA.dat') # XXX GL, not LG!!
        C_GG_import = np.genfromtxt(path.parent / 'common_data/vincenzo/14may/CijDers/EP10/CijGG-GR-Flat-eNLA-NA.dat')

    
    ###########################################################################
    # interpolate Vincenzo's Cls in ell values
    # careful, this part is a bit tricky. Pay attention to the ell_WL,
    # ell_XC arguments in e.g. fLL(ell_XC) vs fLL(ell_WL)
    C_LL_2D = mm.Cl_interpolator(npairs, C_LL_import, ell_XC, nbl)
    C_GG_2D = mm.Cl_interpolator(npairs, C_GG_import, ell_XC, nbl)
    C_WA_2D = mm.Cl_interpolator(npairs, C_LL_import, ell_WA, nbl_WA)
    C_XC_2D = mm.Cl_interpolator(npairs_asimm, C_XC_import, ell_XC, nbl)
    C_LL_WLonly_2D = mm.Cl_interpolator(npairs, C_LL_import, ell_WL, nbl)
    
    
    
    
    ###########################################################################
    # fill the 3D (nbl x zbins x zbins) matrices, or equivalently nbl (zbins x zbins) matrices
    
        
    
    # fill upper triangle: LL, GG, WLonly        
    triu_idx = np.triu_indices(zbins)
    for ell in range(nbl):
        for i in range(npairs):
            C_LL_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_LL_2D[ell, i]
            C_GG_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_GG_2D[ell, i]
            C_LL_WLonly_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_LL_WLonly_2D[ell, i]
    
    # Wadd
    for ell in range(nbl_WA):
        for i in range(npairs):
            C_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_WA_2D[ell, i]
            
    # fill asymmetric
    C_XC_3D = np.reshape(C_XC_2D, (nbl, zbins, zbins))
    
    # symmetrize
    C_LL_WLonly_3D = mm.fill_3D_symmetric_array(C_LL_WLonly_3D, nbl, zbins)
    C_LL_3D = mm.fill_3D_symmetric_array(C_LL_3D, nbl, zbins)
    C_GG_3D = mm.fill_3D_symmetric_array(C_GG_3D, nbl, zbins)
    C_WA_3D = mm.fill_3D_symmetric_array(C_WA_3D, nbl_WA, zbins)
    
    
    # fill datavector correctly:
    D_3x2pt[:, 0, 0, :, :] = C_LL_3D
    D_3x2pt[:, 1, 1, :, :] = C_GG_3D
    D_3x2pt[:, 0, 1, :, :] = np.transpose(C_XC_3D, (0,2,1)) # XXX pay attention to LG, GL...
    D_3x2pt[:, 1, 0, :, :] = C_XC_3D # XXX pay attention to LG, GL...
    

    # create dict with results:
    Cl_dict = {}
    Cl_dict['C_LL_WLonly_3D'] = C_LL_WLonly_3D
    Cl_dict['C_GG_3D'] = C_GG_3D
    Cl_dict['C_WA_3D'] = C_WA_3D
    Cl_dict['D_3x2pt'] = D_3x2pt
    
    print(f'\ncase ell_max_WL = {ell_max_WL} done')


    return Cl_dict
