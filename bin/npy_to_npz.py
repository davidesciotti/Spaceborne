import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import common_cfg.mpl_cfg as mpl_cfg
from . import my_module as mm

matplotlib.use('Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

folder_path = '/home/cosmo/davide.sciotti/data/Spaceborne/jobs/SPV3_magcut_zcut_thesis/output/Flagship_2/covmat/PyCCL'
npy_dict = dict(mm.get_kv_pairs(folder_path, extension='npy'))

for key in npy_dict.keys():
    np.savez_compressed(f'{folder_path}/{key}.npz', npy_dict[key])
