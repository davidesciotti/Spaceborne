import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

mpl_rcParams = {'lines.linewidth': 3.5,
                'font.size': 20,
                'axes.labelsize': 'x-large',
                'axes.titlesize': 'x-large',
                'xtick.labelsize': 'x-large',
                'ytick.labelsize': 'x-large',
                'mathtext.fontset': 'stix',
                'font.family': 'STIXGeneral',
                'figure.figsize': (10, 10),
                'lines.markersize': 8,
                'axes.grid': True,
                'figure.constrained_layout.use': True,
                'backend': 'Qt5Agg'
                }
plt.rcParams.update(mpl_rcParams)

# matplotlib.use('Qt5Agg')
# plt.rcParams['axes.grid'] = True
# plt.rcParams['figure.constrained_layout.use'] = True

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(project_path)
sys.path.append(project_path.parent / 'common_data/common_config')
sys.path.append(job_path / 'PyCCL_forecast/configs')

import jobs.PyCCL_forecast.configs.config_PyCCL_forecast as config
import lib.my_module as mm
# import mpl_rcParams

########################################################################################################################
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################


nbl = 30

uncert_dict = {}
plot_config = config.plot_config
general_config = config.general_config
covariance_config = config.covariance_config
param_names_label = plot_config['param_names_label']

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex='col')

probe = 'WL'
ell_max = 5000
whos_SSC = 'PyCCL'
hm_recipe = 'KiDS1000'
GS_or_GScNG = 'GS'
FM_to_compare = []

FM_dict = dict(mm.get_kv_pairs(job_path / 'output/FM', filetype="txt"))

if whos_SSC == 'PySSC':
    FM_dict_SSCcomp = dict(mm.get_kv_pairs(job_path.parent / 'SSC_comparison/output/FM', filetype="txt"))

for key in FM_dict.keys():
    uncert_dict[key] = mm.uncertainties_FM(FM_dict[key])

# compare cases
cases_to_compare = ['PyCCL', 'CosmoLike']
# cases_to_compare = ['KiDS1000', 'Krause2017']

# abbreviate variable names
case_0 = cases_to_compare[0]
case_1 = cases_to_compare[1]

# set the correct filename
if cases_to_compare == ['PyCCL', 'CosmoLike']:
    var_toloop = whos_SSC  # this has no effect, it is just to remind which variable I should loop over
    name = f'{case_0}_vs_{case_1}_{GS_or_GScNG}_probe_{probe}_{hm_recipe}'
elif cases_to_compare == ['KiDS1000', 'Krause2017']:
    var_toloop = hm_recipe
    name = f'PyCCL_{case_0}_vs_{case_1}_{GS_or_GScNG}_probe_{probe}_{hm_recipe}'
else:
    raise ValueError('cases_to_compare list is undefined')

# remember to choose the correct variable to loop over!
for whos_SSC in cases_to_compare:

    # only PyCCL has different halomodel recipes!
    if whos_SSC != 'PyCCL':
        hm_recipe = ''

    FM_name = f'FM_{probe}_{GS_or_GScNG}_lmax{probe}{ell_max}_nbl{nbl}_{whos_SSC}{hm_recipe}'
    ax[0].plot(range(7), uncert_dict[FM_name][:7] * 100, '--', label=f"{whos_SSC} {GS_or_GScNG} {hm_recipe}",
               marker='o')

    # list of names to compare
    FM_to_compare.append(FM_name)

# perc differences w.r.t. mean
diff_1 = mm.percent_diff_mean(uncert_dict[f'{FM_to_compare[0]}'], uncert_dict[f'{FM_to_compare[1]}'])
diff_2 = mm.percent_diff_mean(uncert_dict[f'{FM_to_compare[1]}'], uncert_dict[f'{FM_to_compare[0]}'])

ax[1].plot(range(7), diff_1[:7], "--", label=case_0, marker='o')
ax[1].plot(range(7), diff_2[:7], "--", label=case_1, marker='o')
ax[1].fill_between(range(7), diff_1[:7], diff_2[:7], color='grey', alpha=0.3)

ax[0].legend()
ax[1].legend()

ax[1].set_xticks(range(7), param_names_label)

ax[0].set_ylabel("$ \\sigma_\\alpha/ \\theta_{fid} \; [\\%]$")
ax[1].set_ylabel("diff. w.r.t. mean [%]")
fig.suptitle(f'FM forec., {probe}., ' '$\ell_{max}$ ' f'{ell_max}')

fig.savefig(f"{job_path}/output/plots/{name}.pdf")
