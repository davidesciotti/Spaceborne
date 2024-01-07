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

output = '/Users/davide/Documents/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut_thesis/output/Flagship_2/covmat/PyCCL'
benchmark = '/Users/davide/Downloads/PyCCL'
extension = 'npz'

mm.test_folder_content(output, benchmark, extension, verbose=True, rtol=1e-7)
