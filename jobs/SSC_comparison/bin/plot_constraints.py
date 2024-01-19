import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(str(project_path / 'lib'))
sys.path.append(str(project_path / 'bin/5_plots/plot_FM'))

import my_module as mm

import plots_FM_running as plot_utils
import config_SSC_comparison as config


params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8),
          'backend': 'Qt5Agg'
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()

general_config = config.general_config
covariance_config = config.covariance_config
FM_config = config.FM_cfg
plot_config = config.plot_config

FM_dict = dict(mm.get_kv_pairs(job_path / 'output/FM', extension="txt"))
FM_dict_PyCCL = dict(
    mm.get_kv_pairs('/home/cosmo/davide.sciotti/data/SSC_restructured/jobs/PyCCL_forecast/output/FM',
                    extension="txt"))

params = plot_config['params']
markersize = 7
dpi = plot_config['dpi']
pic_format = plot_config['pic_format']
which_probe_response_str = 'const'

plot_config['which_plot'] = 'constraints_only'
plot_config['plot_sylvain'] = True
plot_config['plot_ISTF'] = False
plot_config['custom_label'] = f'{which_probe_response_str} Rl,'


plt.figure(figsize=plot_config['params']['figure.figsize'])
# plot_utils.plot_FM(general_config, covariance_config, plot_config, FM_dict)

probe = 'GC'
GO_or_GS = 'GS'

if probe == 'GC':
    lmax = 'GC3000'
elif probe == 'WL':
    lmax = 'WL5000'
elif probe == '3x2pt':
    lmax = 'XC3000'

sigma_FM_const = mm.uncertainties_FM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{lmax}_nbl30_Rlconst'])[:7]
sigma_FM_var = mm.uncertainties_FM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{lmax}_nbl30_Rlvar'])[:7]
sigma_FM_GO = mm.uncertainties_FM(FM_dict[f'FM_{probe}_GO_lmax{lmax}_nbl30'])[:7]
if probe == 'WL':
    sigma_FM_PyCCL = mm.uncertainties_FM(FM_dict_PyCCL[f'FM_WL_GS_lmaxWL5000_nbl30_PyCCL'])[:7]
    plt.plot(range(7), sigma_FM_PyCCL, '--', marker='o', markersize=markersize, label=f'{probe} {GO_or_GS} lmax{lmax}, PyCCL')

plt.plot(range(7), sigma_FM_const, '--', marker='o', markersize=markersize, label=f'{probe} {GO_or_GS} lmax{lmax}, const Rl')
plt.plot(range(7), sigma_FM_var, '--', marker='o', markersize=markersize, label=f'{probe} {GO_or_GS} lmax{lmax}, var Rl')
if probe != 'WL':
    plt.plot(range(7), sigma_FM_GO, '--', marker='o', markersize=markersize, label=f'{probe} GO lmax{lmax}')
plt.grid()
plt.legend()
plt.xticks(range(7), plot_config['param_names_label'])



# ! article plot, comparison
# for plot_config['GO_or_GS'] in ['GO', 'GS']:
#     plot_utils.plot_FM(general_config, covariance_config, plot_config, FM_dict)
# plt.grid()
#
# plt.savefig(job_path / f'output/plots/dav_vs_sylv_{plot_config["probe"]}.{pic_format}',
#             dpi=dpi, format=f'{pic_format}',
#             bbox_inches='tight')
