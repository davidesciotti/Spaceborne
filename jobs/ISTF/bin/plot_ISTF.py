import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.stats as stats
from getdist import MCSamples, plots
from matplotlib import ticker
from matplotlib.cm import get_cmap
from getdist.gaussian_mixtures import GaussianND
import pandas as pd

project_path = Path.cwd().parent.parent.parent

sys.path.append(str(project_path.parent / 'common_data'))
import common_lib.my_module as mm
import common_config.mpl_cfg as mpl_cfg
import common_config.ISTF_fid_params as ISTF_fid

sys.path.append(str(project_path / 'bin'))
import plots_FM_running as plot_utils

sys.path.append(str(project_path / 'jobs/SPV3_magcut_zcut/config'))
import config_SPV3_magcut_zcut as cfg

# plot config
matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')
markersize = 10

########################################################################################################################

# ! options
zbins = 10
zbins_list = np.array((zbins,), dtype=int)
probe = 'WL'
pes_opt_list = ('opt',)
EP_or_ED_list = ('EP',)
which_comparison = 'GO_vs_GS'  # this is just to set the title of the plot
which_Rl = 'var'
nparams_chosen = 7
model = 'flat'
which_diff = 'mean'
which_cfg = 'cl15gen'
flagship_version = 2
check_old_FM = False
pes_opt = 'opt'
which_uncertainty = 'marginal'
# whether to remove the rows/cols for the given nuisance parameters (ie whether to fix them)
params_tofix_dict = {
    'IA': False,
    'gal_bias': False,
}
bar_plot_cosmo = True
triangle_plot = False
plot_prior_contours = False
bar_plot_nuisance = False
pic_format = 'pdf'
BNT_transform = False
dpi = 500
n_cosmo_params = 7
nparams_toplot = n_cosmo_params
EP_or_ED = 'EP'
nbl = 30
SSC_code = 'PySSC'
# ! end options

assert params_tofix_dict['IA'] is False, 'IA parameters should be left free to vary'
assert params_tofix_dict['gal_bias'] is False, 'galaxy bias parameters should be left free to vary'

uncert_ratio_dict = {}
uncert_G_dict = {}
uncert_GS_dict = {}

# these are just for the I/O
lmax = 3000
if probe == 'WL':
    lmax = 5000

probe_sylv = probe
if probe == 'GC':
    probe_sylv = 'GCph'

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
elif which_diff == 'mean':
    diff_funct = mm.percent_diff_mean

uncert_ratio_dict[probe] = {}

fom_df = pd.DataFrame()
# import FM dict
FM_PySSC_dict = mm.load_pickle(
    f'{project_path}/jobs/ISTF/output/{which_cfg}/FM/PySSC/FM_dict_zbins{EP_or_ED}{zbins:02}.pickle')
FM_PyCCL_dict = mm.load_pickle(
    f'{project_path}/jobs/ISTF/output/{which_cfg}/FM/PyCCL/FM_dict_zbins{EP_or_ED}{zbins:02}.pickle')

param_names_dict = FM_PySSC_dict['param_names_dict']
fiducial_values_dict = FM_PySSC_dict['fiducial_values_dict']

# shorten names
FM_PySSC_GO = FM_PySSC_dict[f'FM_{probe}_GO']
FM_PySSC_GS = FM_PySSC_dict[f'FM_{probe}_GS']
FM_PyCCL_GO = FM_PyCCL_dict[f'FM_{probe}_GO']
FM_PyCCL_GS = FM_PyCCL_dict[f'FM_{probe}_GS']

assert np.all(FM_PySSC_GO == FM_PyCCL_GO), 'FM_PySSC_GO and FM_PyCCL_GO should be the same'
FM_GO = FM_PySSC_GO


# fix the desired parameters and remove null rows/columns
FM_GO, param_names, fiducials = mm.mask_FM(FM_GO, param_names_dict, fiducial_values_dict, params_tofix_dict,
                                                 remove_null_rows_cols=True)
FM_PySSC_GS, _, _ = mm.mask_FM(FM_PySSC_GS, param_names_dict, fiducial_values_dict, params_tofix_dict,
                               remove_null_rows_cols=True)
FM_PyCCL_GS, _, _ = mm.mask_FM(FM_PyCCL_GS, param_names_dict, fiducial_values_dict, params_tofix_dict,
                               remove_null_rows_cols=True)

wzwa_idx = [param_names.index('wz'), param_names.index('wa')]

# cases should include all the FM, plus percent differences if you want to show them. the ordering is important, it must
# be the same!
FMs = [FM_GO, FM_PySSC_GS, FM_PyCCL_GS]
cases_to_compute = ['FM_GO', 'FM_PySSC_GS', 'FM_PyCCL_GS']
cases_to_plot = ['FM_GO', 'FM_PySSC_GS', 'FM_PyCCL_GS', 'abs(percent_diff_GS) wrt mean']

# compute uncertainties and store them in a dictionary
uncert_dict = {}
fom_dict = {}
for FM, case in zip(FMs, cases_to_compute):
    uncert_dict[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fiducials[:nparams_toplot],
                                                       which_uncertainty=which_uncertainty, normalize=True))
    fom_dict[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx)
    print(f'FoM({probe}, {case}): {fom_dict[case]}')

# add the percent differences and/or rations to the dictionary
to_compare_A = uncert_dict['FM_PySSC_GS'] - uncert_dict['FM_GO']
to_compare_B = uncert_dict['FM_PyCCL_GS'] - uncert_dict['FM_GO']
# to_compare_A = uncert_dict['FM_PySSC_GS']
# to_compare_B = uncert_dict['FM_PyCCL_GS']
uncert_dict['abs(percent_diff_GS) wrt mean'] = np.abs(diff_funct(to_compare_A, to_compare_B))
# uncert_dict['percent_diff_GS'] = diff_funct(uncert_dict['FM_PyCCL_GS'], uncert_dict['FM_GO'])


# silent check against IST:F (which do not exist for GC alone):
if probe != 'GC':
    uncert_dict['ISTF'] = ISTF_fid.forecasts[f'{probe}_opt_w0waCDM_flat']
    diff = diff_funct(uncert_dict['ISTF'], uncert_dict['FM_GO'])
    assert np.all(np.abs(diff) < 5.0), f'IST:F and G are not consistent! Remember that you are checking against the ' \
                                       f'optimistic case'

# transform dict. into an array
uncert_array = []
for case in cases_to_plot:
    uncert_array.append(uncert_dict[case])
uncert_array = np.asarray(uncert_array)

title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, lmax, EP_or_ED, zbins)
plot_utils.bar_plot(uncert_array[:, :nparams_toplot], title, cases_to_plot, nparams=nparams_toplot,
                    param_names_label=param_names[:nparams_toplot], bar_width=0.12)

mm.matshow(FM_PySSC_GS, 'FM_PySSC_GS')
mm.matshow(FM_PyCCL_GS, 'FM_PyCCL_GS')

array_to_show = np.abs(mm.percent_diff_mean(FM_PyCCL_GS, FM_PySSC_GS))
mm.matshow(array_to_show, 'perc diff wrt mean, PyCCL vs PySSC', log=True)
# Get the Axes object of the plot
ax = plt.gca()

# Loop over the array and add the numbers as annotations
for i in range(array_to_show.shape[0]):
    for j in range(array_to_show.shape[1]):
        ax.annotate('{:.0f}'.format(array_to_show[i, j]), xy=(j, i), ha='center', va='center')

# create list with the quantites you want to keep track of, and add it as row of the df. You will plot outside
# the for loop simply choosing the entries of the df you want.
# fom_list = [probe, SSC_code, BNT_transform,
#             fom_dict[cases[0]], fom_dict[cases[1]],
#             fom_dict[cases[2]], fom_dict[cases[3]]]
#
# fom_df = fom_df.append(pd.DataFrame([fom_list],
#                                     columns=['probe', 'ML', 'ZL', 'MS', 'ZS',
#                                              'kmax_h_over_Mpc', 'kmax_1_over_Mpc', 'BNT',
#                                              'which_cuts',
#                                              'center_or_min',
#                                              'FM_GO_noEllcuts',
#                                              'FM_GO_Ellcuts',
#                                              'FM_GS_noEllcuts',
#                                              'FM_GS_Ellcuts']), ignore_index=True)
#
# diff_FM = diff_funct(FM_PySSC_GS, FM_PyCCL_GS)
# mm.matshow(diff_FM, title=f'percent difference wrt mean between PySSC and PyCCL FMs {probe}, {EP_or_ED}{zbins:02}')
