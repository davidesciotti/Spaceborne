import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg')
import common_lib.my_module as mm
import common_lib.wf_cl_lib as wf_cl_lib
import common_cfg.mpl_cfg as mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

output = '/Users/davide/Documents/Lavoro/Programmi/exact_SSC/output/SSC_matrix/julia'
benchmark = '/Users/davide/Documents/Lavoro/Programmi/exact_SSC/output/SSC_matrix/julia/flounder'
extension = 'npy'

mm.test_folder_content(output, benchmark, extension, verbose=True, rtol=1e-7)
