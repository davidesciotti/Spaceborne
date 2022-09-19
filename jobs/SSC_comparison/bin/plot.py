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

sys.path.append(f'{project_path}/lib')
sys.path.append(f'{project_path}/bin')
sys.path.append(f'{project_path}/jobs')
sys.path.append(f'{project_path}/config')
sys.path.append(f'{project_path.parent}/common_config')

import my_module as mm
import plots_FM_running as plot_utils
import SSC_comparison.config.config_SSC_comparison as cfg
import mpl_cfg
import ISTF_fid_params as ISTF_fid

# plot config
matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')
markersize = 10

# other params
nbl = cfg.general_config['nbl']
zbins = 10

# import all outputs from this script
FM_dict = dict(mm.get_kv_pairs(job_path / 'output/FM', filetype="txt"))
FM_dict_PyCCL = dict(mm.get_kv_pairs(job_path.parent / 'PyCCL_forecast/output/FM', filetype="txt"))
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
# ! end options

# fiducial values
fid_cosmo = np.asarray([ISTF_fid.primary[key] for key in ISTF_fid.primary.keys()])[:7]
fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
fid_bias = np.asarray([ISTF_fid.photoz_bias[key] for key in ISTF_fid.photoz_bias.keys()])

assert GO_or_GS == 'GS', 'GO_or_GS should be GS, if not what are you comparing?'

if probe == '3x2pt':
    probe_lmax = 'XC'
    param_names_label = mpl_cfg.general_dict['param_names_label'] + mpl_cfg.general_dict['IA_names_label'] + \
                        mpl_cfg.general_dict['bias_names_label']
    fid = np.concatenate((fid_cosmo, fid_IA, fid_bias), axis=0)
else:
    probe_lmax = probe

if probe == 'WL':
    ell_max = ell_max_WL
    param_names_label = mpl_cfg.general_dict['param_names_label'] + mpl_cfg.general_dict['IA_names_label']
    fid = np.concatenate((fid_cosmo, fid_IA), axis=0)
else:
    ell_max = ell_max_GC

if probe == 'GC':
    param_names_label = mpl_cfg.general_dict['param_names_label'] + mpl_cfg.general_dict['bias_names_label']
    fid = np.concatenate((fid_cosmo, fid_bias), axis=0)

elif probe not in ['WL', 'GC', '3x2pt']:
    raise ValueError('probe should be WL, GC or 3x2pt')

fid = fid[:nparams]
param_names_label = param_names_label[:nparams]

keys = [f'FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl{nbl}',
        f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rl{which_Rl}',
        # f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_PyCCLKiDS1000',
        ]

label_list = [f'Gauss-only covmat (GO)',
              # f'{GO_or_GS} Rlconst',
              f'Gauss+SS covmat ({GO_or_GS})',
              # f'{GO_or_GS} PyCCL',
              # f'% diff wrt mean, Rlconst vs PyCCL',
              f'[(GS/GO - 1) $\\times$ 100]']

data = []
fom = {}
for i, key in enumerate(keys):
    uncert = np.asarray(mm.uncertainties_FM(FM_dict[key], nparams=nparams, fiducials=fid,
                                            which_uncertainty=which_uncertainty, normalize=True)[:nparams])
    data.append(uncert)
    fom[key] = mm.compute_FoM(FM_dict[key])

# compute percent diff of the cases chosen - careful of the indices!
print('careful about this absolute value!')
diff_1 = mm.percent_diff(data[-1], data[-2])
# diff_2 = mm.percent_diff_mean(data[-2], data[-1])
data.append(diff_1)
# data.append(diff_2)

data = np.asarray(data)

# title = f'{probe}, ' + '$\\ell_{max} = $' + f'{ell_max}'
title = f'{probe}, ' + '$\\ell_{max} = $' + f'{ell_max}'  # for PhD workshop

# plot_utils.bar_plot(data, title, label_list, nparams=nparams, param_names_label=param_names_label, bar_width=0.18,
#                     second_axis=True)

########################### getdist plot

# plot_utils.triangle_plot(FM=FM_dict[keys[0]], fiducials=fid, param_names_label=param_names_label)

# should I do this?
fid = np.where(fid == 0., 1, fid)  # the fiducial for wa is 0, substitute with 1 to avoid division by zero
fid = np.where(fid == -1, 1, fid)  # the fiducial for wa is -1, substitute with 1 to avoid negative values

# very basic check
ndim = len(param_names_label)
assert ndim == nparams, 'ndim should be equal to nparams'

names = ['$\\Omega_{m,0}$',
         '$\\Omega_{b,0}$',
         '$w_0$',
         '$w_a$',
         '$h$',
         '$n_s$',
         '$\\sigma_8$']

# parameters' covariance matrix
FM_inv_GO = np.linalg.inv(FM_dict[keys[0]])[:nparams, :nparams]
FM_inv_GS = np.linalg.inv(FM_dict[keys[1]])[:nparams, :nparams]

GO_gaussian = GaussianND(fid, FM_inv_GO, names=names)
GS_gaussian = GaussianND(fid, FM_inv_GS, names=names)
g = plots.get_subplot_plotter()
g.settings.linewidth = 2
g.settings.legend_fontsize = 18
g.settings.linewidth_contour = 2.5
g.settings.axes_fontsize = 13
g.settings.axes_labelsize = 17
# g.settings.subplot_size_ratio = 1
g.settings.solid_colors = 'tab10'
g.triangle_plot([GS_gaussian, GO_gaussian], filled=True, contour_lws=1.4, legend_labels=['Gauss + SSC', 'Gauss-only'], ratio=1)
# g.add_legend(['Gauss + SSC', 'Gauss-only'], legend_loc='upper right')


# plt.savefig(
#     job_path / f'output/plots/{which_comparison}/{probe}_ellmax{ell_max}_Rl{which_Rl}_{which_uncertainty}.png')

# compute and print FoM
print('GO FoM:', mm.compute_FoM(FM_dict[f'FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl{nbl}']))
print('Rl const FoM:', mm.compute_FoM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlconst']))
print('Rl var FoM:', mm.compute_FoM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlvar']))
