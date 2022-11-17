import numpy as np

"""
routine to reshape the covariance matrix from 4D to 6D. The "ind"
 array to be passed as argument to cov_4D_to_6D should have number of rows == npairs 
 (npaurs = number of independent redshift pairs). It is then
 
 ind_WL = ind[:npairs_auto, :] 
 ind_GC = ind_WL
 ind_LG = ind[npairs_auto:(npairs_auto + npairs_cross), :]
 ind_GL = ind_LG[:, [0, 1, 3, 2]]: switch the last two columns of ind_LG
 ind_3x2pt = ind
 
 the first 2 columns of the array indicate the probe (0 = 'L', 1 = 'G').
 It is possible to delete them and modify the function cov_4D_to_6D changing
 i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3] 
 into
 i, j, k, l = ind[ij, 0], ind[ij, 1], ind[kl, 0], ind[kl, 1] 
 
 """

def get_pairs(zbins):
    """
    returns the number of redshift bins pairs for the different probes
    """
    npairs_auto  = (zbins*(zbins+1))//2
    npairs_cross = zbins**2
    npairs_3x2pt = 2*npairs_auto + npairs_cross
    return npairs_auto, npairs_cross, npairs_3x2pt 

def symmetrize_ij(cov_6D, zbins=10):
    """
    manually impose the i <-> j, k <-> l symmetry
    """
    for i in range(zbins):
        for j in range(zbins):
            cov_6D[:, :, i, j, :, :] = cov_6D[:, :, j, i, :, :]
            cov_6D[:, :, :, :, i, j] = cov_6D[:, :, :, :, j, i]
    return cov_6D
       

def cov_4D_to_6D(cov_4D, nbl, zbins, probe, ind):
    """transform the cov from shape (nbl, nbl, npairs, npairs) 
    to (nbl, nbl, zbins, zbins, zbins, zbins)"""
    
    assert probe == 'LL' or \
    probe == 'GG' or \
    probe == '3x2pt' or \
    probe == 'LG' or \
    probe == 'GL', 'probe must be "LL", "GG" or "3x2pt"'
    

    npairs_auto, npairs_cross, npairs_3x2pt = get_pairs(zbins)    
    
    if probe == '3x2pt': npairs = npairs_3x2pt
    elif probe == 'LL' or probe == 'GG': npairs = npairs_auto
    elif probe == 'GL' or probe == 'LG': npairs = npairs_cross
    
    assert ind.shape[0] == npairs, 'ind.shape[0] != npairs'
    
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ij in range(npairs):
        for kl in range(npairs):
            # rename for better readability
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3] 
            # reshape
            cov_6D[:, :, i, j, k, l] = cov_4D[:, :, ij, kl]
            
    # GL, LG and 3x2pt are not symmetric
    if probe == 'LL' or probe == 'GG':  
        cov_6D = symmetrize_ij(cov_6D, zbins)
            
    return cov_6D








