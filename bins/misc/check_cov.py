import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

# get project directory
path = Path.cwd().parent.parent

# import configuration and functions modules
sys.path.append(str(path.parent / 'common_data'))
import my_config as my_config
sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

start_time = time.perf_counter()

params = {'lines.linewidth' : 3.5,
          'font.size' : 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral'
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
###############################################################################
###############################################################################

bia = 0.0
cov_WL_G_bia0 = np.load(path / f'output/covmat/CovMat-ShearShear-Gauss-20bins-Cl_marco_bia{bia}.npy')
cov_WL_GpSSC_bia0 = np.load(path / f'output/covmat/CovMat-ShearShear-GaussSSC-20bins-Cl_marco_bia{bia}.npy')
bia = 2.17
cov_WL_G_bia2 = np.load(path / f'output/covmat/CovMat-ShearShear-Gauss-20bins-Cl_marco_bia{bia}.npy')
cov_WL_GpSSC_bia2 = np.load(path / f'output/covmat/CovMat-ShearShear-GaussSSC-20bins-Cl_marco_bia{bia}.npy')


# mm.matshow(cov_WL_G_bia0)
# mm.matshow(cov_WL_G_bia2)
diff_G = mm.percent_diff(cov_WL_G_bia0, cov_WL_G_bia2)
mm.matshow(diff_G, 'percent diff, Gauss', log=True, abs_val=True)

# mm.matshow(cov_WL_GpSSC_bia0, 'bia0', log=True)
# mm.matshow(cov_WL_GpSSC_bia2, 'bia2', log=True)
# diff_GpSSC = mm.percent_diff(cov_WL_GpSSC_bia2, cov_WL_GpSSC_bia0)
# mm.matshow(diff_GpSSC, 'percent diff btw. bia=0 and bia=2.17, G+SSC', log=True, abs_val=True)
diff_GpSSC = mm.percent_diff(cov_WL_GpSSC_bia0, cov_WL_GpSSC_bia2)
mm.matshow(diff_GpSSC, 'percent diff btw. bia=0 and bia=2.17, G+SSC', log=True, abs_val=True)

# diff_GpSSC = mm.percent_diff_mean(cov_WL_GpSSC_bia2, cov_WL_GpSSC_bia0)
# mm.matshow(diff_GpSSC, 'percent diff btw. bia=0 and bia=2.17, G+SSC', log=False, abs_val=False)
# diff_GpSSC = mm.percent_diff_mean(cov_WL_GpSSC_bia0, cov_WL_GpSSC_bia2)
# mm.matshow(diff_GpSSC, 'percent diff btw. bia=0 and bia=2.17, G+SSC', log=False, abs_val=False)






