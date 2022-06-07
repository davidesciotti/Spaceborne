import numpy as np
import sys
from pathlib import Path
# get project directory
path = Path.cwd().parent.parent
# import configuration and functions modules
sys.path.append(str(path / 'lib'))
import my_module as mm

###############################################################################
###############################################################################
###############################################################################


# XXX TODO implement ind dictionary like Luca!
# note: old ordering is triU, rows-wise (column-wise? not sure...)
# note: ROW-wise triangular Lower is SEYFERT



    
    ###########################################################################
    # fill the 3D (nbl x zbins x zbins) matrices, or equivalently nbl (zbins x zbins) matrices
    
        
def reshape_cls_2D_to_3D(general_config, ell_dict, cl_2D_dict):
    
    nbl = general_config['nbl']
    ell_max_WL = general_config['ell_max_WL']
    zbins = general_config['zbins']
    nProbes = general_config['nProbes']
    
    # import ell values:
    ell_WA = ell_dict['ell_WA']
    
    C_LL_2D = cl_2D_dict['C_LL_2D']
    C_GG_2D = cl_2D_dict['C_GG_2D']
    C_WA_2D = cl_2D_dict['C_WA_2D']
    C_XC_2D = cl_2D_dict['C_XC_2D']
    C_LL_WLonly_2D = cl_2D_dict['C_LL_WLonly_2D']
    
    # compute n_zpairs
    npairs, npairs_asimm, npairs_tot = mm.get_pairs(zbins)    
    
    # nbl for Wadd
    if np.asanyarray(ell_WA).size == 1: nbl_WA = 1 # in the case of just one bin it would give error
    else: nbl_WA = ell_WA.shape[0]
    
    # initialize cls arrays
    C_LL_WLonly_3D = np.zeros((nbl, zbins, zbins)) # 3D, for WLonly
    C_LL_3D = np.zeros((nbl, zbins, zbins)) # 3D, for the datavector
    C_GG_3D = np.zeros((nbl, zbins, zbins)) # 3D, for GConly
    C_WA_3D = np.zeros((nbl_WA, zbins, zbins)) # 3D, ONLY for the datavector (there's no Wadd_only case)
    D_3x2pt = np.zeros((nbl, nProbes, nProbes, zbins, zbins))


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
    cl_3D_dict = {}
    cl_3D_dict['C_LL_WLonly_3D'] = C_LL_WLonly_3D
    cl_3D_dict['C_GG_3D'] = C_GG_3D
    cl_3D_dict['C_WA_3D'] = C_WA_3D
    cl_3D_dict['D_3x2pt'] = D_3x2pt
    
    print(f'\ncase ell_max_WL = {ell_max_WL} done')

    return cl_3D_dict
