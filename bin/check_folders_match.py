import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = '/Users/davide/Documents/Lavoro/Programmi'

sys.path.append(f'{ROOT}/Spaceborne/bin')
import my_module as mm

sys.path.append(f'{ROOT}/Spaceborne/common_cfg')
import mpl_cfg as mpl_cfg


matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

output = '/Users/davide/Documents/Lavoro/Programmi/exact_SSC/output/ISTF/new_test/d2ClAB_dVddeltab'
benchmark = '/Users/davide/Documents/Lavoro/Programmi/exact_SSC/output/ISTF/d2ClAB_dVddeltab'
extension = 'npy'

mm.test_folder_content(output, benchmark, extension, verbose=True, rtol=1e-3)
