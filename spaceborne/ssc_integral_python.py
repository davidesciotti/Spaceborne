from numba import njit, prange
import numpy as np
    

@njit(parallel=True)
def numba_integral_trapz_4d(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]
    num_col = ind_AB.shape[1]

    dz = z_array[1] - z_array[0]

    result = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))

    for ell1 in prange(nbl):
        for ell2 in range(nbl):  # this could be further optimized by computing only upper triangular ells, but not with prange
            for zij in range(zpairs_AB):
                for zkl in range(zpairs_CD):
                    for z1_idx in range(z_steps):
                        for z2_idx in range(z_steps):

                            zi = ind_AB[zij, num_col - 2]
                            zj = ind_AB[zij, num_col - 1]
                            zk = ind_CD[zkl, num_col - 2]
                            zl = ind_CD[zkl, num_col - 1]

                            result[ell1, ell2, zij, zkl] += (cl_integral_prefactor[z1_idx] *
                                                                cl_integral_prefactor[z2_idx] *
                                                                d2ClAB_dVddeltab[ell1, zi, zj, z1_idx] *
                                                                d2ClCD_dVddeltab[ell2, zk, zl, z2_idx] *
                                                                sigma2[z1_idx, z2_idx])

    result *= dz**2
    return result

@njit(parallel=True)
def numba_integral_trapz_6d(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array):
    
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]
    zbins = d2ClCD_dVddeltab.shape[1]

    dz = z_array[1] - z_array[0]

    result = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))

    for ell1 in prange(nbl):
        for ell2 in range(nbl):
            for zi in range(zbins):
                for zj in range(zbins):
                    for zk in range(zbins):
                        for zl in range(zbins):
                            for z1_idx in range(z_steps):
                                for z2_idx in range(z_steps):

                                    result[ell1, ell2, zi, zj, zk, zl] += (cl_integral_prefactor[z1_idx] *
                                                                            cl_integral_prefactor[z2_idx] *
                                                                            d2ClAB_dVddeltab[ell1, zi, zj, z1_idx] *
                                                                            d2ClCD_dVddeltab[ell2, zk, zl, z2_idx] *
                                                                            sigma2[z1_idx, z2_idx])

    result *= dz**2
    return result