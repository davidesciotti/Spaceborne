import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent.parent
job_path = Path.cwd().parent.parent

sys.path.append(str(project_path / 'lib'))
import my_module as mm

matplotlib.use('Qt5Agg')

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8)
          # 'backend': 'Qt5Agg'
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()

cov_4D = np.load('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/output/covmat/CovMat-3x2pt'
                 '-Gauss-20bins-NL_flag_1_4D.npy')

cov_2D_ellblock = mm.cov_4D_to_2D(cov_4D, nbl=20, npairs_AB=210, block_index='ell')
cov_2D_zblock = mm.cov_4D_to_2D(cov_4D, nbl=20, npairs_AB=210, block_index='ij')
cov_2DCLOE_ellblock = mm.cov_4D_to_2DCLOE_3x2pt(cov_4D, nbl=20, zbins=10)

cov_2DCLOE_zblock = mm.cov_4D_to_2DCLOE_3x2pt_new(cov_4D, nbl=20, zbins=10, block_index='ij')

print(np.array_equal(cov_2D_zblock, cov_2DCLOE_zblock))


mm.matshow(cov_2D_zblock, log=True)
mm.matshow(cov_2DCLOE_zblock, log=True)

