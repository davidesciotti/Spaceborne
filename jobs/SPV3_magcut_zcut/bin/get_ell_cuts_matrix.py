import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(f'/home/davide/Documenti/Lavoro/Programmi/Spaceborne')
import bin.my_module as mm
import common_cfg.mpl_cfg as mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

pk_path = ('/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/InputFiles/InputPS/'
           'HMcodeBar/InFiles/Flat/w0/PddVsZedLogK-w0_-1.000e+00.dat')
pk_2d = mm.pk_vinc_file_to_2d_npy(pk_path, plot_pk_z0=True)

# cl_ll_3d = csmlib.project_pk(pab_k_z_interp_func, kernel_a, kernel_b, z_grid, ell_grid, cl_integral_convention,
#                              use_h_units, cosmo_ccl)


