import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ROOT = '/home/cosmo/davide.sciotti/data'

sys.path.append(f'{ROOT}/Spaceborne/bin')
import my_module as mm

sys.path.append(f'{ROOT}/Spaceborne/common_cfg')
import mpl_cfg as mpl_cfg


matplotlib.use('Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

output = '/home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/OutputFiles/DataVecDers/Flat/Bacco/'
benchmark = '/home/davide/Scaricati/drive-download-20240112T092248Z-001/DataVecDers/Flat/All/Bacco/EP13/'
extension = 'dat'

mm.test_folder_content(output, benchmark, extension, verbose=True, rtol=1e-3)
