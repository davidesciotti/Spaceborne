import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time
import pickle


# get project directory
path = Path.cwd().parent.parent

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


path_import = '/home/cosmo/davide.sciotti/data/SSC_restructured/jobs/IST_NL/output/covmat'

NL_flag = 1
cov_dict = {}

for probe in ['WL', '3x2pt']:
    for GO_or_GS in ['GO', 'GS']:
            
        if GO_or_GS == 'GO': GO_or_GS_filename = 'Gauss'
        elif GO_or_GS == 'GS': GO_or_GS_filename = 'GaussSSC'
        
        if probe == 'WL': probe_filename = 'ShearShear'
        else: probe_filename = probe
        
        cov_dict[f'cov_{probe}_{GO_or_GS}_4D'] = np.load(f'{path_import}/6D_for_Santiago/CovMat-{probe_filename}-{GO_or_GS_filename}-20bins-NL_flag_{NL_flag}_4D.npy')
        cov_dict[f'cov_{probe}_{GO_or_GS}_2D'] = np.load(f'{path_import}/CovMat-{probe_filename}-{GO_or_GS_filename}-20bins-NL_flag_{NL_flag}.npy')

probe = 'WL'
GO_or_GS = 'GO'
nbl = 20
ind = np.genfromtxt('/home/cosmo/davide.sciotti/data/SSC_restructured/jobs/IST_NL/input/indici_cloe_like.dat').astype(int) - 1
ind_LL = ind[:55, :]
# i think the XC part has to be switched!!! like this
ind[55:155, [2, 3]] = ind[55:155, [3, 2]]
ind_3x2pt = ind

zbins = 10

# reshape in 6D
for GO_or_GS in ['GO', 'GS']:
    cov_dict[f'cov_WL_{GO_or_GS}_6D'] = mm.cov_4D_to_6D(cov_dict[f'cov_WL_{GO_or_GS}_4D'], nbl, zbins, 'WL', ind_LL)
    cov_dict[f'cov_3x2pt_{GO_or_GS}_6D'] = mm.cov_4D_to_6D(cov_dict[f'cov_3x2pt_{GO_or_GS}_4D'], nbl, zbins, '3x2pt', ind_3x2pt)


# show, this reshaping is jenuuuuus
rand_ell = np.random.randint(nbl)
GO_or_GS = 'GO'
probe_test = '3x2pt'
mm.matshow(cov_dict[f'cov_{probe_test}_{GO_or_GS}_6D'][rand_ell, rand_ell, ...].reshape((zbins**2, zbins**2)), log=True, title = f'{rand_ell}, WL, {GO_or_GS}')


for GO_or_GS in ['GO', 'GS']:
    
    if GO_or_GS == 'GO': GO_or_GS_filename = 'Gauss'
    elif GO_or_GS == 'GS': GO_or_GS_filename = 'GaussSSC'

    np.save(f'{path_import}/6D_for_Santiago/CovMat-ShearShear-{GO_or_GS_filename}-20bins-NL_flag_{NL_flag}_6D.npy', cov_dict[f'cov_WL_{GO_or_GS}_6D'])


path_santiago = '/home/cosmo/davide.sciotti/data/SSC_restructured/jobs/IST_NL/output/covmat/6D_for_Santiago'
with open(f"{path_santiago}/cov_3x2pt_GO_10D.pkl", "rb") as file:
    cov_3x2pt_GO_new_10D = pickle.load(file)

mm.matshow(cov_3x2pt_GO_new_10D['L', 'L', 'L', 'L'][0, 0, ...].reshape((zbins**2, zbins**2)))








