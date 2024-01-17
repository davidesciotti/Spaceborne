import numpy as np
import sys

from numba import njit

sys.path.append("/home/cosmo/davide.sciotti/data/common_data/common_lib")
import my_module as mm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

cov_GS_BNT_ste_4D = np.load('/home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/Flagship_2/CovMats/BNT_True/BNT_covmat_GS_WL_lmax5000_nbl32_zbins13_ED_Rlvar_6D_stef.npy')
cov_GS_BNT_dav_6D = np.load('/home/cosmo/davide.sciotti/data/SSC_restructured_v2/jobs/SPV3/output/Flagship_2/BNT_True/covmat/zbins13/covmat_GS_WL_lmax5000_nbl32_zbins13_ED_Rlvar_6D.npy')
zbins = 13
nbl = 32

# reshape cov_ste: put ell1, ell2 axes first
cov_GS_BNT_ste_4D = cov_GS_BNT_ste_4D.transpose(2, 3, 0, 1)


# now explode the p, q into i, j, k, l
tril_indices = np.tril_indices(zbins)

@njit
def reshape_ste_cm(cov_ste_4D):
    cov_ste_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(zbins):
                for q in range(zbins):
                    i, j, k, l = tril_indices[0][p], tril_indices[1][p], tril_indices[0][q], tril_indices[1][q]
                    cov_ste_6D[ell1, ell2, i, j, k, l] = cov_ste_4D[ell1, ell2, p, q]
    return cov_ste_6D

cov_GS_BNT_ste_6D = reshape_ste_cm(cov_GS_BNT_ste_4D)

# reshape to 4D to have a look
zpairs = int(zbins * (zbins + 1) / 2)
ind_tril_indices = np.hstack((tril_indices[0].reshape(-1, 1), tril_indices[1].reshape(-1, 1)))
cov_GS_BNT_ste_4D = mm.cov_6D_to_4D(cov_GS_BNT_ste_6D, nbl=32, npairs=zpairs, ind=ind_tril_indices)
cov_GS_BNT_dav_4D = mm.cov_6D_to_4D(cov_GS_BNT_dav_6D, nbl=32, npairs=zpairs, ind=ind_tril_indices)

mm.matshow(cov_GS_BNT_ste_4D[0, 0, ...], log=True, title='cov_GS_BNT_ste_4D[0, 0, ...]')
mm.matshow(cov_GS_BNT_dav_4D[0, 0, ...], log=True, title='cov_GS_BNT_dav_4D[0, 0, ...]')




mm.compare_arrays(cov_GS_BNT_ste_6D, cov_GS_BNT_dav_6D, 'ste', 'dav', log_arr=True)

print('done')