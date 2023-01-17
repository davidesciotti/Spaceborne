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

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# 'block_index': 'ell',
# 'GL_or_LG': GL_or_LG,

triu_tril = 'triu'
row_col_wise = 'row-wise'
zbins = 13
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

ind = np.genfromtxt(f'{project_path.parent}/common_data/ind_files/variable_zbins/{triu_tril:s}_{row_col_wise:s}/indices_{triu_tril:s}_{row_col_wise:s}_zbins{zbins:02d}.dat', dtype=int)

cov_BNT_stef_6D_dict = dict(mm.get_kv_pairs_npy(
    f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_2/CovMats/BNT_True/produced_by_stefano/magcut_zcut'))

cov_BNT_stef_4D_dict = {}
cov_BNT_stef_2D_dict = {}
for key in cov_BNT_stef_6D_dict.keys():
    # cov_BNT_stef_4D_dict[key] = mm.cov_6D_to_4D(cov_BNT_stef_6D_dict[key], 32, zpairs_auto, ind[:zpairs_auto, :])
    print(cov_BNT_stef_6D_dict[key].shape)