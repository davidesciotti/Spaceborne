import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path}/common_config')
import ISTF_fid_params
import mpl_cfg


plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# ! options
zbins = 13
zbin_type = 'ED'
nbl = 29
# ! end options

path_stef = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_2/CovMats/BNT_True/produced_by_stefano'
cov_GGGL = np.load(f'{path_stef}/BNT_covmat_GGGL_3x2pt_lmax3000_nbl{nbl}_zbins{zbins}_{zbin_type}_Rlvar_10D.npy')
cov_GGLL = np.load(f'{path_stef}/BNT_covmat_GGLL_3x2pt_lmax3000_nbl{nbl}_zbins{zbins}_{zbin_type}_Rlvar_10D.npy')
cov_LLLL = np.load(f'{path_stef}/BNT_covmat_LLLL_3x2pt_lmax3000_nbl{nbl}_zbins{zbins}_{zbin_type}_Rlvar_10D.npy')
cov_GLGL = np.load(f'{path_stef}/BNT_covmat_GLGL_3x2pt_lmax3000_nbl{nbl}_zbins{zbins}_{zbin_type}_Rlvar_10D.npy')
cov_LLGL = np.load(f'{path_stef}/BNT_covmat_LLGL_3x2pt_lmax3000_nbl{nbl}_zbins{zbins}_{zbin_type}_Rlvar_10D.npy')
cov_GGGG_6D = np.load(f'/home/davide/Documenti/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3/output/Flagship_2/BNT_False/covmat/zbins{zbins}/covmat_GS_GC_lmax3000_nbl{nbl}_zbins{zbins}_{zbin_type}_Rlvar_6D.npy')

zpairs = int(zbins * (zbins + 1) / 2)
tril_indices = np.tril_indices(zbins)
ind_tril_indices = np.hstack((tril_indices[0].reshape(-1, 1), tril_indices[1].reshape(-1, 1)))
cov_GGGG_4D = mm.cov_6D_to_4D(cov_GGGG_6D, nbl=nbl, npairs=zpairs, ind=ind_tril_indices)
cov_GGGG_2D = mm.cov_4D_to_2D(cov_GGGG_4D, nbl=nbl, npairs_AB=zpairs, npairs_CD=None, block_index='vincenzo')



# cov_3x2pt_6D_dict = {
#     ('L', 'L', 'L', 'L',):
# }


# def cov_3x2pt_dict_10D_to_4D(cov_3x2pt_dict_10D, probe_ordering, nbl, zbins, ind_copy, GL_or_LG):
