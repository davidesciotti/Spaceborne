import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

# get project directory
path = Path.cwd().parent.parent.parent.parent

# import configuration and functions modules
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
          'font.family': 'STIXGeneral',
          'figure.figsize': (10, 10)
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
###############################################################################
###############################################################################
"""
just a check of the covariance matrices computed with the old pipeline and the 
new one (SSC_restructured)
"""



# get project directory
project_path = Path.cwd().parent.parent.parent.parent
job_path = Path.cwd().parent.parent
home_path = Path.home()

cov_dict_new = {}
cov_dict_old = {}
# here I don't get the same covariance matrices, but just because the Cls in the repo changed
path_old = '/Users/davide/Documents/Lavoro/Programmi/SSC_for_ISTNL/output/covmat'
# this imports the Cls directly from the repo, more recent: with this I obtain the same covmats (from
# the new and from the old pipeline)
path_old = '/Users/davide/Documents/Lavoro/Programmi/SSC_for_ISTNL/output/check_SSC_restructured/covmat'
for NL_flag in range(1, 5):

    # new (from SSC_restructured)
    # GO
    cov_dict_new['cov_GC_GO_2D'] = np.load(job_path / f'output/covmat/CovMat-PosPos-Gauss-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_new['cov_WL_GO_2D'] = np.load(job_path / f'output/covmat/CovMat-ShearShear-Gauss-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_new['cov_3x2pt_GO_2D'] = np.load(job_path / f'output/covmat/CovMat-3x2pt-Gauss-20bins-NL_flag_{NL_flag}.npy')
    # GS
    cov_dict_new['cov_GC_GS_2D'] = np.load(job_path / f'output/covmat/CovMat-PosPos-GaussSSC-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_new['cov_WL_GS_2D'] = np.load(job_path / f'output/covmat/CovMat-ShearShear-GaussSSC-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_new['cov_3x2pt_GS_2D'] = np.load(job_path / f'output/covmat/CovMat-3x2pt-GaussSSC-20bins-NL_flag_{NL_flag}.npy')

    # old (from SSC_for_ISTNL)
    cov_dict_old['cov_GC_GO_2D'] = np.load(f'{path_old}/CovMat-PosPos-Gauss-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_old['cov_WL_GO_2D'] = np.load(f'{path_old}/CovMat-ShearShear-Gauss-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_old['cov_3x2pt_GO_2D'] = np.load(f'{path_old}/CovMat-3x2pt-Gauss-20bins-NL_flag_{NL_flag}.npy')
    # GS
    cov_dict_old['cov_GC_GS_2D'] = np.load(f'{path_old}/CovMat-PosPos-GaussSSC-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_old['cov_WL_GS_2D'] = np.load(f'{path_old}/CovMat-ShearShear-GaussSSC-20bins-NL_flag_{NL_flag}.npy')
    cov_dict_old['cov_3x2pt_GS_2D'] = np.load(f'{path_old}/CovMat-3x2pt-GaussSSC-20bins-NL_flag_{NL_flag}.npy')


    for key in cov_dict_new.keys():
        are_equals = np.all(cov_dict_old[key] == cov_dict_new[key])
        print(f'NL_flag = {NL_flag}, key = {key}, are the covariance matrices equal? {are_equals}')

    



