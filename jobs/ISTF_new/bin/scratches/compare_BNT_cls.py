import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('/home/cosmo/davide.sciotti/data/common_data/common_lib')
import my_module as mm

cl_ste = np.load(
    f'/home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/Flagship_2/DataVectors/3D_reshaped_BNT_True/stefano/Cl_WL_BNT_stefano_3D.npy')
cl_dav = np.load(
    f'/home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/Flagship_2/DataVectors/3D_reshaped_BNT_True/WLO/dv-WLO-32-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-ED13.npy')
ell_values = np.genfromtxt(
    f'/home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/Flagship_2/DataVectors/3D_reshaped_BNT_True/WLO/ell_WL_ellmaxWL5000.txt')

diff = mm.percent_diff(cl_ste, cl_dav)
mm.compare_arrays(cl_ste, cl_dav, rtol=1e-4)

i = 0
j = 0
# fix, axs = plt.figure()
plt.figure()
# plt.plot(ell_values, cl_ste[:, i, j], label='stefano')
# plt.plot(ell_values, cl_dav[:, i, j], label='davide', ls='--')
plt.plot(ell_values, diff[:, i, j], label='diff', ls='--')
plt.grid()
plt.legend()
# plt.yscale('log')

# mm.compare_arrays(cl_ste.reshape((13*16, 13*16)), cl_dav.reshape((13*16, 13*16)))

print('done')
