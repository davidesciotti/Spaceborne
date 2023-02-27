import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# ells and deltas are the same (between the DEMO-BNT notebook and simulate_data.py)
ell_values = np.load('/Users/davide/Desktop/874-BNT-cls/ell_values.npy')
delta_ells = np.load('/Users/davide/Desktop/874-BNT-cls/delta_ells.npy')

# these have been output via the #870 DEMO-BNT
cC_LL_arr = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cC_LL_arr.npy')
cC_GL_arr = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cC_GL_arr.npy')
cC_GG_arr = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cC_GG_arr.npy')
cC_LL_BNT = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cC_LL_BNT.npy')
cC_GL_BNT = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cC_GL_BNT.npy')
cC_GG_BNT = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cC_GG_BNT.npy')

# these via simulate_data.py?
# cl_WL_BNTTrue = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cl_WL_BNTTrue.npy')
# cl_GC_BNTTrue = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cl_GC_BNTTrue.npy')
# cl_XC_BNTTrue = np.load('/Users/davide/Desktop/874-BNT-cls/from_870_DEMO-BNT/cl_XC_BNTTrue.npy')

zbins = cC_GG_BNT.shape[1]

# define colormap
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, zbins)]

plt.figure()
for zj in range(zbins):
    zi = zj
    plt.plot(ell_values, cC_LL_arr[:, zi, zj], label='BNT False', c=colors[zj], ls='-')
    plt.plot(ell_values, cC_LL_BNT[:, zi, zj], label='BNT True', c=colors[zj], ls='--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell$')
plt.legend()
plt.grid()
plt.show()
