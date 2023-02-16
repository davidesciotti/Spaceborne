import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from getdist import MCSamples, plots
from matplotlib.cm import get_cmap
from getdist.gaussian_mixtures import GaussianND

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(str(project_path.parent / 'common_data'))
import common_lib.my_module as mm
import common_config.mpl_cfg as mpl_cfg
import common_config.ISTF_fid_params as ISTF_fid

sys.path.append(str(project_path / 'bin'))
import plots_FM_running as plot_utils

sys.path.append(str(project_path / 'jobs'))
import SPV3.config.config_SPV3 as cfg


# plot config
matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')
markersize = 10


nbl = cfg.general_cfg['nbl']

# import all outputs from this script
FM_dict = dict(mm.get_kv_pairs(job_path / 'output/FM', extension="txt"))
FM_dict_PyCCL = dict(mm.get_kv_pairs(job_path.parent / 'PyCCL_forecast/output/FM', extension="txt"))
FM_dict = {**FM_dict, **FM_dict_PyCCL}

# ! options
GO_or_GS = 'GS'
probe = '3x2pt'
which_comparison = 'GO_vs_GS'  # this is just to set the title of the plot
which_Rl = 'var'
which_uncertainty = 'marginal'
ell_max_WL = 5000
ell_max_GC = 3000
nparams = 7
zbins = 10
# ! end options

# fiducial values
fid_cosmo = np.asarray([ISTF_fid.primary[key] for key in ISTF_fid.primary.keys()])[:7]
fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
fid_bias = np.asarray([ISTF_fid.photoz_bias[key] for key in ISTF_fid.photoz_bias.keys()])

assert GO_or_GS == 'GS', 'GO_or_GS should be GS, if not what are you comparing?'

if probe == '3x2pt':
    probe_lmax = 'XC'
else:
    probe_lmax = probe

if probe == 'WL':
    ell_max = ell_max_WL
else:
    ell_max = ell_max_GC

keys = [f'FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl{nbl}',
        # f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlconst',
        f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlvar',
        # f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_PyCCLKiDS1000',
        ]

label_list = [f'GO',
              # f'{GO_or_GS} Rlconst',
              f'{GO_or_GS} Rlvar',
              # f'{GO_or_GS} PyCCL',
              # f'% diff wrt mean, Rlconst vs PyCCL',
              # f'% diff wrt mean, Rlvar vs PyCCL'
              ]

data = []
for i, key in enumerate(keys):
    uncert = np.asarray(mm.uncertainties_FM(FM_dict[key])[:nparams])
    data.append(uncert)

    fom[key] = mm.compute_FoM(FM_dict[key])

# compute percent diff of the cases chosen - careful of the indices!
diff_1 = mm.percent_diff_mean(data[-3], data[-1])
diff_2 = mm.percent_diff_mean(data[-2], data[-1])
data.append(diff_1)
data.append(diff_2)

data = np.asarray(data)

title = f'{probe}, ' '$\\ell_{max} = $' f'{ell_max}'

plot_utils.bar_plot(data, title, label_list, bar_width=0.18)

plt.savefig(
    job_path / f'output/plots/const_vs_var_rl/{probe}_ellmax{ell_max}_{GO_or_GS}_Rlconst_vs_Rlvar_vs_PyCCL.png')

# compute and print FoM
probe = '3x2pt'
probe_lmax = 'XC'
print(mm.compute_FoM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlconst']))
print(mm.compute_FoM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlvar']))
print(mm.compute_FoM(FM_dict[f'FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl{nbl}']))
