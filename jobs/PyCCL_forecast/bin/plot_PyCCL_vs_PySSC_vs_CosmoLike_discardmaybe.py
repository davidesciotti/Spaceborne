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
          'figure.figsize': (10, 10)
          }
plt.rcParams.update(params)
markersize = 10

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
import lib.my_module_old as mm


########################################################################################################################
########################################################################################################################
########################################################################################################################

def uncertainties_FM(FM):
    fid = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.55, 1, 1)
    # fidmn = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.06, 0.55, 1) # with massive neutrinos
    FM_inv = np.linalg.inv(FM)
    sigma_FM = np.zeros(10)
    for i in range(10):
        sigma_FM[i] = np.sqrt(FM_inv[i, i]) / fid[i]
    return sigma_FM


########################################################################################################################
########################################################################################################################
########################################################################################################################


ell_max_WL = 5000
nbl = 30
GO_or_GS = 'GS'
name = 'diff_PyCCLvsPySSC'

FM_dict = {}
uncert_dict = {}
plot_config = config.plot_cfg
general_config = config.general_cfg
covariance_config = config.covariance_cfg
param_names_label = plot_config['param_names_label']

fig, ax = plt.subplots(2, 1, figsize=(14, 10))

for which_SSC in ['PyCCL', 'CosmoLike', 'PySSC']:
    FM_dict[f'FM_WL_GS_{which_SSC}'] = np.genfromtxt(
        f"{path}/output/FM/FM_WL_GS_lmaxWL{ell_max_WL}_nbl{nbl}_{which_SSC}.txt", )
    uncert_dict[f'uncert_WL_GS_{which_SSC}'] = uncertainties_FM(FM_dict[f'FM_WL_GS_{which_SSC}'])

    ax[0].plot(range(7), uncert_dict[f'uncert_WL_GS_{which_SSC}'][:7] * 100, "o--", label=f"{which_SSC} {GO_or_GS}")

if name == 'diff_PyCCLvsCosmoLike':
    diff = mm.percent_diff_mean(uncert_dict[f'uncert_WL_GS_PyCCL'], uncert_dict[f'uncert_WL_GS_CosmoLike'])
elif name == 'diff_PyCCLvsPySSC':
    diff = mm.percent_diff_mean(uncert_dict[f'uncert_WL_GS_PySSC'], uncert_dict[f'uncert_WL_GS_PyCCL'])



# ax[1].plot(range(7), diff_PyCCLvsCosmoLike[:7], "o--", label=f"(PyCCL/CosmoLike - 1) * 100")
ax[1].plot(range(7), diff[:7], "o--", label=f"{name}")

ax[0].legend()
ax[1].legend()
ax[1].set_xticks(range(7), param_names_label)

ax[0].set_ylabel("$ \\sigma_\\alpha/ \\theta_{fid} [\\%]$")
ax[1].set_ylabel("% diff wrt mean")
# plt.title(f"FM forec., {probe}, {case}.")

# save fig
# fig.savefig(f"{path}/output/plots/{name}.pdf")
