import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
import common_cfg.mpl_cfg as mpl_cfg


matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

output = '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/SSC_matrix/julia'
benchmark = '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/SSC_matrix/julia/halo_model'
extension = 'npy'

mm.test_folder_content(output, benchmark, extension, verbose=True, rtol=1e-3)
