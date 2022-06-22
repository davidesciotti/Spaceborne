import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (10, 10),
          'lines.markersize': 8
          }
plt.rcParams.update(params)

matplotlib.use('Qt5Agg')
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.constrained_layout.use'] = True

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent


path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/PyCCL_forecast'
path_mm = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/lib'
path_plotlib = '/5_plots/plot_FM'
path_config = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/PyCCL_forecast/configs'


sys.path.append(project_path)
sys.path.append(job_path / 'PyCCL_forecast/configs')

import jobs.PyCCL_forecast.configs.config_PyCCL_forecast as config
import lib.my_module as mm


########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################


ell_max_WL = 5000
nbl = 30
GO_or_GS = 'GS'
name = 'diff_PyCCLvsPySSC'

FM_dict = {}
uncert_dict = {}
plot_config = config.plot_config
general_config = config.general_config
covariance_config = config.covariance_config
param_names_label = plot_config['param_names_label']

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex='col')

probe = 'WL'
ell_max = 5000
whos_SSC = 'PyCCL'
GS_or_GScNG = 'GScNG'
to_compare = []

FM_dict = dict(mm.get_kv_pairs(job_path / 'output/FM', filetype="txt"))

for key in FM_dict.keys():
    uncert_dict[key] = mm.uncertainties_FM(FM_dict[key])


# plot constraints
hm_recipes = ['KiDS1000', 'Krause2017']
for hm_recipe in hm_recipes:

    # only PyCCL has different halomodel recipes
    if whos_SSC != 'PyCCL':
        hm_recipe = ''

    FM_name = f'FM_{probe}_{GS_or_GScNG}_lmax{probe}{ell_max}_nbl{nbl}_{whos_SSC}{hm_recipe}'
    ax[0].plot(range(7), uncert_dict[FM_name][:7] * 100, '--', label=f"{whos_SSC} {GS_or_GScNG} {GO_or_GS} {hm_recipe}",
               marker='o')

    # list of constraints to compute the % difference of
    to_compare.append(FM_name)

diff_1 = mm.percent_diff_mean(uncert_dict[f'{to_compare[0]}'], uncert_dict[f'{to_compare[1]}'])
diff_2 = mm.percent_diff_mean(uncert_dict[f'{to_compare[1]}'], uncert_dict[f'{to_compare[0]}'])

ax[1].plot(range(7), diff_1[:7], "--", label=hm_recipes[0], marker='o')
ax[1].plot(range(7), diff_2[:7], "--", label=hm_recipes[1], marker='o')
ax[1].fill_between(range(7), diff_1[:7], diff_2[:7], color='grey', alpha=0.3)

ax[0].legend()
ax[1].legend()

ax[1].set_xticks(range(7), param_names_label)
ax[0].set_ylabel("$ \\sigma_\\alpha/ \\theta_{fid} \; [\\%]$")
ax[1].set_ylabel("diff. w.r.t. mean [%]")
fig.set_title(f"FM forec., {probe}.")

# save fig
fig.savefig(f"{path}/output/plots/{name}.pdf")
