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

project_path = Path.cwd().parent

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
probe = 'GC'
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
fix_gal_bias = False  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_shear_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_dzWL = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_dzGC = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_IA = False  # whether to remove the rows/cols for the IA nuisance parameters (ie whether to fix them)
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

assert fix_IA is False, 'IA parameters should be left free to vary'
assert fix_gal_bias is False, 'galaxy bias parameters should be left free to vary'

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
else:
    diff_funct = mm.percent_diff_mean

uncert_ratio_dict[probe] = {}

# import FM dict
FM_dict_PySSC = mm.load_pickle(
    f'{project_path}/jobs/ISTF/output/{which_cfg}/FM/PySSC/FM_dict_{EP_or_ED}{zbins:02}.pickle')
_param_names = FM_dict_PySSC['parameters_names']  # this should not change when passed the second time to the function
_fiducials = FM_dict_PySSC['fiducial_values']  # this should not change when passed the second time to the function
FM_PySSC_GO = FM_dict_PySSC[f'FM_{probe}_GO']
FM_PySSC_GS = FM_dict_PySSC[f'FM_{probe}_GS']

FM_PyCCL_dict = mm.load_pickle(
    f'{project_path}/jobs/ISTF/output/{which_cfg}/FM/PyCCL/FM_dict_{EP_or_ED}{zbins:02}.pickle')
FM_PyCCL_GO = FM_PyCCL_dict[f'FM_{probe}_GO']
FM_PyCCL_GS = FM_PyCCL_dict[f'FM_{probe}_GS']

# fix the desired parameters and remove null rows/columns
FM_PySSC_GO, param_names, fiducials = mm.mask_FM(FM_PySSC_GO, _param_names, _fiducials, n_cosmo_params, fix_IA,
                                                 fix_gal_bias)
FM_PySSC_GS, _, _ = mm.mask_FM(FM_PySSC_GS, _param_names, _fiducials, n_cosmo_params, fix_IA, fix_gal_bias)
wzwa_idx = [param_names.index('wz'), param_names.index('wa')]

FM_PyCCL_GO, _, _ = mm.mask_FM(FM_PyCCL_GO, _param_names, _fiducials, n_cosmo_params, fix_IA, fix_gal_bias)
FM_PyCCL_GS, _, _ = mm.mask_FM(FM_PyCCL_GS, _param_names, _fiducials, n_cosmo_params, fix_IA, fix_gal_bias)

# cases should include all the FM, plus percent differences if you want to show them. the ordering is important, it must
# be the same!
FMs = [FM_PySSC_GO, FM_PyCCL_GO, FM_PySSC_GS, FM_PyCCL_GS]
cases_to_compute = ['FM_PySSC_GO', 'FM_PyCCL_GO', 'FM_PySSC_GS', 'FM_PyCCL_GS']
cases_to_plot = ['FM_PySSC_GO', 'FM_PySSC_GS', 'FM_PyCCL_GS', 'abs(percent_diff_GS) wrt mean']

# compute uncertainties and store them in a dictionary
uncert_dict = {}
fom = {}
for FM, case in zip(FMs, cases_to_compute):
    uncert_dict[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fiducials[:nparams_toplot],
                                                       which_uncertainty=which_uncertainty, normalize=True))
    fom[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx)
    print(f'FoM({probe}, {case}): {fom[case]}')

# add the percent differences and/or rations to the dictionary
uncert_dict['abs(percent_diff_GS) wrt mean'] = np.abs(diff_funct(uncert_dict['FM_PyCCL_GS'], uncert_dict['FM_PySSC_GS']))
# uncert_dict['percent_diff_GS'] = diff_funct(uncert_dict['FM_PyCCL_GS'], uncert_dict['FM_PyCCL_GO'])

assert np.array_equal(uncert_dict['FM_PySSC_GO'], uncert_dict['FM_PyCCL_GO']), \
    'the GO uncertainties must be the same, I am only changing the SSC code!'

# silent check against IST:F (which do not exist for GC alone):
if probe != 'GC':
    uncert_dict['ISTF'] = ISTF_fid.forecasts[f'{probe}_opt_w0waCDM_flat']
    diff = diff_funct(uncert_dict['ISTF'], uncert_dict['FM_PySSC_GO'])
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
