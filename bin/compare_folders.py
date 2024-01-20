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

folder_a = '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/ISTF/jan_2024/SSC_matrix/nbl20_wrong'
folder_b = '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/ISTF/jan_2024/SSC_matrix'
extension = 'npy'

mm.test_folder_content(folder_a, folder_b, extension, verbose=True, rtol=1e-3)
