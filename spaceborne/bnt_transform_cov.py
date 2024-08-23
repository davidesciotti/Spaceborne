"""
This is a script to quickly BNT-transform an input covariance matrix, given an input covariance and a BNT matrix
"""

import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import yaml
import os
import warnings
import numpy as np
ROOT = os.getenv('ROOT')

sys.path.append(f'{ROOT}/Spaceborne')
import spaceborne.covariance as covariance
import spaceborne.pyccl_cov_class as pyccl_cov_class
import spaceborne.my_module as mm

%matplotlib inline


# nofz = np.genfromtxt(f'{ROOT}/CLOE_benchmarks/nzTabSPV3.dat')
# zgrid_nz = nofz[:, 0]
# nofz = nofz[:, 1:]

# with open(f'{ROOT}/Spaceborne/config_release.yaml') as f:
#     cfg = yaml.safe_load(f)

# fid_pars_dict = cfg['cosmology']
# flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)
# cfg['general_cfg']['flat_fid_pars_dict'] = flat_fid_pars_dict

# h = flat_fid_pars_dict['h']
# cfg['general_cfg']['fid_pars_dict'] = fid_pars_dict

# ccl_obj = pyccl_cov_class.PycclClass(fid_pars_dict)


# # ! compute BNT
# bnt_mat_dav = covariance.compute_BNT_matrix(zbins, zgrid_nz, nofz, ccl_obj.cosmo_ccl, plot_nz=True)
# bnt_mat_cloe = np.load(f'{ROOT}/CLOE_benchmarks/BNT_matrix.npy')
# mm.compare_arrays(bnt_mat_dav, bnt_mat_cloe)

nbl = 32
probe_ordering = (('L', 'L'), ('G', 'L'), ('G', 'G'))
GL_or_LG = 'GL'
ml, ms = 245, 245
cm_folder_sesto = f'{ROOT}/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/covmat/Sesto'

for zbins in (3, 5, 7, 9, 10, 11, 13, 15):
    for which_cov in ('OC', ):
        for ep_ed in ['ED', 'EP']:

            ind = mm.build_full_ind('triu', 'row-major', zbins)

            suffix = f'{ep_ed}{zbins:02d}-zedMin02-zedMax25-ML{ml}-MS{ms}'

            bnt_mat = np.genfromtxt(
                f'{cm_folder_sesto}/BNTmat-{suffix}.dat')

            # ! reshape cov to 6D to BNT-trasform it
            if which_cov == 'OC':
                cov_2d = np.genfromtxt(
                    f'{cm_folder_sesto}/_old_filenames/covmat_GSSCcNG_OneCovariance_3x2pt_zbins{ep_ed}{zbins:02d}_lmax5000_ML245_ZL02_MS245_ZS02_pkHMCodeBar_13245deg2_2D.dat')
            else:
                cov_2d = np.genfromtxt(
                    f'{cm_folder_sesto}/cmfull-{suffix}-{which_cov}.dat')

            # reshape to 10d
            cov_4d = mm.cov_2D_to_4D(cov_2D=cov_2d, nbl=nbl, block_index='vincenzo', optimize=True)
            cov_10d_dict = mm.cov_3x2pt_4d_to_10d_dict(cov_4d, zbins, probe_ordering, nbl, ind.copy(), optimize=False)

            # bnt transform
            X_dict = covariance.build_X_matrix_BNT(bnt_mat)
            cov_10d_dict_bnt = covariance.cov_3x2pt_BNT_transform(cov_10d_dict, X_dict)

            # re-reshape to 2d
            cov_4d_bnt = mm.cov_3x2pt_10D_to_4D(cov_10d_dict_bnt, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
            cov_2d_bnt = mm.cov_4D_to_2D(cov_4D=cov_4d_bnt, block_index='vincenzo', optimize=True)

            mm.matshow(cov_2d[:, :], log=True, title='cov_2d')
            mm.matshow(cov_2d_bnt[:, :], log=True, title='cov_2d_bnt')

            np.savetxt(
                f'{cm_folder_sesto}/cmfull-{suffix}-{which_cov}-BNT.dat', cov_2d, fmt='%.7e')

            del cov_10d_dict_bnt, cov_10d_dict
            gc.collect()

            print(f'zbins {ep_ed}{zbins}, {which_cov} done')

print('done')
