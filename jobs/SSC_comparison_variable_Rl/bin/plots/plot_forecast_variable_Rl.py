# %load_ext autoreload
# %autoreload 2
# %matplotlib widget

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

# get project directory
project_path = Path.cwd().parent.parent.parent.parent
job_path = Path.cwd().parent.parent

# import project libraries
sys.path.append(str(project_path / 'lib'))
import my_module as mm

# import job config
sys.path.append(str(job_path / 'configs'))
import config_SSC_comparison_variable_Rl as config

plot_config = config.plot_config

params = plot_config['params']
markersize = plot_config['markersize']
dpi = plot_config['dpi']
pic_format = plot_config['pic_format']
figsize = plot_config['params']['figure.figsize']
markersize = plot_config['markersize']
markersize = 15
param_names_label = plot_config['param_names_label']
plt.rcParams.update(params)

"""
This script plots the variable Rl GS constraints, for all probes, pessimistic 
and optimistic cases. It also show the % difference w.r.t. the Rl = 4 case, i.e.
the scenario considered in the paper.
"""

# the 3x2pt is the most affected by the change in Rl
probe = '3x2pt'

GO_or_GS = 'GS'  # ! test GO as consistency check
nbl = 30
n_cosmo_params = 7
case = 'opt'

Rl_values = np.linspace(2, 8, 25)[4:21]  # slice to switch to range[2, 7]

uncert = np.zeros((Rl_values.size, n_cosmo_params))

for i, Rl in enumerate(Rl_values):

    # very inelegant way of setting the various lmax/probe names
    if case == 'opt':
        if probe == 'WL':
            ell_max = 5000
            probe_lmax = 'WL'
        elif probe == 'GC':
            ell_max = 3000
            probe_lmax = 'GC'
        elif probe == '3x2pt':
            ell_max = 3000
            probe_lmax = 'XC'

    elif case == 'pes':
        if probe == 'WL':
            ell_max = 1500
            probe_lmax = 'WL'
        elif probe == 'GC':
            ell_max = 750
            probe_lmax = 'GC'
        elif probe == '3x2pt':
            ell_max = 750
            probe_lmax = 'XC'

    # get the FM and compute the uncertainties
    FM = np.genfromtxt(job_path / f"output/FM/FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rl_{Rl:.2f}.txt")
    uncert[i, :] = mm.uncertainties_FM(FM)[:n_cosmo_params] * 100

# compute the % difference with respect to the reference (Rl = 4) case, then plot it
Rl_4_index = np.where(Rl_values == 4.)[0][0]
diff = (uncert / uncert[Rl_4_index, :] - 1) * 100

plt.figure(figsize=figsize)
# plot GS constraints or % difference
for param_idx in range(7):
    # plt.plot(Rl_values, uncert[:, param_idx], '.-', label = "uncert_dav_GS, " + my_xticks[param_idx])    
    plt.plot(Rl_values, diff[:, param_idx], '.-', label=param_names_label[param_idx], markersize=markersize)

# max and min, not very useful
maxes = uncert.max(axis=0)
mins = uncert.min(axis=0)

# plt.title(f"FM forec., {probe}, opt. " "vs  $R$")
plt.ylabel("$ \\bar{\\sigma}_{\\alpha}(R)/ \\bar{\\sigma}_{\\alpha}(R=4) \, [\\%]$")
plt.xlabel('$R$')

plt.grid()
plt.legend()
plt.tight_layout()

# save
plt.savefig(job_path / f'output/plots/constr_vs_R_{probe}.{pic_format}',
            dpi=dpi, format=f'{pic_format}',
            bbox_inches='tight')
