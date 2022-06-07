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
import config_SSC_comparison_variable_ellmax as config


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
probe = 'WL'

GO_or_GS = 'GS' # ! test GO as consistency check
nbl = 30
n_cosmo_params = 7
case = 'opt'

ell_max_values = np.linspace(750, 5000, 10)

uncert_GO = np.zeros((ell_max_values.size, n_cosmo_params))
uncert_GS = np.zeros((ell_max_values.size, n_cosmo_params))

for i, ell_max in enumerate(ell_max_values):
    
    # very inelegant way of setting the various lmax/probe names
    if probe == 'WL': 
        probe_lmax = 'WL'
    elif probe == 'GC': 
        probe_lmax = 'GC'
    elif probe == '3x2pt': 
        probe_lmax = 'XC'
        
    if probe == 'WL': 
        probe_lmax = 'WL'
    elif probe == 'GC': 
        probe_lmax = 'GC'
    elif probe == '3x2pt': 
        probe_lmax = 'XC'

    # get the FM and compute the uncertainties
    FM_GO = np.genfromtxt(job_path / f"output/FM/FM_{probe}_GO_lmax{probe_lmax}{ell_max:.2f}_nbl{nbl}.txt")
    FM_GS = np.genfromtxt(job_path / f"output/FM/FM_{probe}_GS_lmax{probe_lmax}{ell_max:.2f}_nbl{nbl}.txt")

    uncert_GO[i, :] = mm.uncertainties_FM(FM_GO)[:n_cosmo_params] * 100
    uncert_GS[i, :] = mm.uncertainties_FM(FM_GS)[:n_cosmo_params] * 100
    
    
# compute the % difference with respect to the reference (Rl = 4) case, then plot it

diff = mm.percent_diff(uncert_GS, uncert_GO)
plt.figure(figsize=figsize)
# plot GS constraints or % difference
for param_idx in range(7):
    # plt.plot(Rl_values, uncert[:, param_idx], '.-', label = "uncert_dav_GS, " + my_xticks[param_idx])    
    plt.plot(ell_max_values, diff[:, param_idx], '.-', label = param_names_label[param_idx], markersize=markersize)    

# max and min, not very useful
# maxes = uncert.max(axis=0)
# mins = uncert.min(axis=0)

plt.title(f"FM forec., {probe}." " $\sigma^{\\rm GS}/\sigma^{\\rm GO}$ vs  $\\ell_{max}$")
plt.ylabel("$ (\\bar{\\sigma}_{\\alpha}({\\rm GS})/ \\bar{\\sigma}_{\\alpha}({\\rm GO}) - 1) \\times 100 \, [\\%]$")
plt.xlabel('$\\ell_{max}$')

plt.grid()
plt.legend()
plt.tight_layout()

# save
plt.savefig(job_path / f'output/plots/constr_vs_ellmax_{probe}.{pic_format}', 
            dpi=dpi, format=f'{pic_format}',
            bbox_inches='tight')




