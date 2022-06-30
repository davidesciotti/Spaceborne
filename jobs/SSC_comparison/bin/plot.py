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
sys.path.append(str(project_path / 'bin'))
sys.path.append(str(project_path / 'jobs'))

import my_module as mm
import plots_FM_running as plot_utils
import SSC_comparison.configs.config_SSC_comparison as cfg

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

ell_max_WL = cfg.general_config['ell_max_WL']
ell_max_GC = cfg.general_config['ell_max_GC']
ell_max_XC = ell_max_GC
nbl = cfg.general_config['nbl']

# import all outputs from this script
FM_dict = dict(mm.get_kv_pairs(job_path / 'output/FM', filetype="txt"))

# choose what you want to plot
probe = '3x2pt'
GO_or_GS = 'GS'

if probe == '3x2pt':
    probe_lmax = 'XC'
else:
    probe_lmax = probe

if probe == 'WL':
    ell_max = ell_max_WL
else:
    ell_max = ell_max_GC

keys = [f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlconst',
        f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlvar',
        f'FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl{nbl}']

# cases = ['Rlconst', 'Rlvar']
# for i, key in enumerate(keys):
#     plot_utils.plot_FM_constr(FM_dict[key], label=f'{cases[i]}')

# compute uncertainties
data = np.asarray([mm.uncertainties_FM(FM_dict[key])[:7] for key in keys]) * 100
data[-1, :] = mm.percent_diff_mean(data[0, :], data[1, :])


label_list = [f'{GO_or_GS} Rlconst',
              f'{GO_or_GS} Rlvar',
              f'% diff',]

title = f'{probe}'


plot_utils.bar_plot_v2(data, title, label_list)
