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

zbins = 13
MS = 230
ZS = 0
probe = 'GC'
wil_new = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_2/KernelFun/magcut_zcut/Wi{probe:s}-ED{zbins:02}-MS{MS}-ZS{ZS:02}.dat')
wil_new = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_2/KernelFun/Wi{probe}-ED{zbins:02}-FS2.dat')


