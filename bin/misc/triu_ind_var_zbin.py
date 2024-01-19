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

matplotlib.use('Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

"""
this script produces the ind file for an arbitrary number of zbins, for the triu index ordering
"""

for zbins in range(2, 25):

    triu_idx = np.triu_indices(zbins)
    xc_idx = np.indices((zbins, zbins))

    i_wl = triu_idx[0]
    j_wl = triu_idx[1]

    i_xc = xc_idx[0].flatten()
    j_xc = xc_idx[1].flatten()

    i_gc = i_wl
    j_gc = j_wl

    npairs_auto = int(zbins * (zbins + 1) / 2)
    npairs_cross = zbins ** 2

    ind = np.stack([
        np.concatenate([np.zeros(npairs_auto), np.ones(npairs_cross), np.ones(npairs_auto)]),
        np.concatenate([np.zeros(npairs_auto), np.zeros(npairs_cross), np.ones(npairs_auto)]),
        np.concatenate([i_wl, i_xc, i_gc]),  # tomographic row index
        np.concatenate([j_wl, j_xc, j_gc])  # tomographic col index
    ]).T

    ind = ind.astype('int')

    ind_bench = np.genfromtxt(
        f'{project_path}/input/ind_files/indici_triu_like_int.dat', dtype='int')

    np.savetxt(
        f'/home/davide/Documenti/Lavoro/Programmi/common_data/ind_files/variable_zbins/triu_like/indici_triu_like_zbins{zbins}.dat',
        ind, fmt='%i', delimiter='\t')
