import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path}/common_config')
import ISTF_fid_params
import mpl_cfg

matplotlib.use('Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

zbins = 10
niz_flagship = np.genfromtxt(f'/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/'
                             f'SPV3_07_2022/InputNz/Lenses/Flagship/niTab-EP{zbins}.dat')
niz_redbook = np.genfromtxt(f'/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/'
                            f'SPV3_07_2022/InputNz/Lenses/RedBook/niTab-EP{zbins}.dat')

z_flagship = niz_flagship[:, 0]
z_redbook = niz_redbook[:, 0]

color = cm.rainbow(np.linspace(0, 1, zbins))

for zbin in range(zbins):
    plt.plot(z_flagship, niz_flagship[:, zbin + 1], '--', color=color[zbin])
    plt.plot(z_redbook, niz_redbook[:, zbin + 1], color=color[zbin])
plt.yscale('log')
plt.grid()