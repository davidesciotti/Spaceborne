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
probe = 'WL'
pes_opt_list = ('opt',)
EP_or_ED_list = ('EP',)
which_comparison = 'GO_vs_GS'  # this is just to set the title of the plot
which_Rl = 'var'
nparams_chosen = 7
model = 'flat'
which_diff = 'normal'
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
FM_dict_PySSC = mm.load_pickle(f'{project_path}/jobs/ISTF/output/FM/PySSC/FM_dict_{EP_or_ED}{zbins:02}.pickle')
_param_names = FM_dict_PySSC['parameters']  # this should not change when passed the second time to the function
_fiducials = FM_dict_PySSC['fiducial_values']  # this should not change when passed the second time to the function
FM_PySSC_GO = FM_dict_PySSC[f'FM_{probe}_GO']
FM_PySSC_GS = FM_dict_PySSC[f'FM_{probe}_GS']

FM_PyCCL_dict = mm.load_pickle(f'{project_path}/jobs/ISTF/output/FM/PyCCL/FM_dict_{EP_or_ED}{zbins:02}.pickle')
FM_PyCCL_GO = FM_PyCCL_dict[f'FM_{probe}_GO']
FM_PyCCL_GS = FM_PyCCL_dict[f'FM_{probe}_GS']

# from SSC_paper_jan22_backup_works
# FM_GO_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/SSC_paper_jan22_backup_works/output/FM/'
#                           f'common_ell_and_deltas/Cij_14may/FM_{probe}_G_lmax{probe}{lmax}_nbl{nbl}.txt')
# FM_GS_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/SSC_paper_jan22_backup_works/output/FM/'
#                           f'common_ell_and_deltas/Cij_14may/FM_{probe}_G+SSC_lmax{probe}{lmax}_nbl{nbl}.txt')

# from SSC_restructured_bu_may2022
# FM_GO_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/!archive/SSC_restructured_v2_bu/jobs/'
#                           f'SSC_comparison/output/FM/FM_{probe}_GO_lmax{probe}{lmax}_nbl{nbl}.txt')
# FM_GS_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/!archive/SSC_restructured_v2_bu/jobs/'
#                           f'SSC_comparison/output/FM/FM_{probe}_GS_lmax{probe}{lmax}_nbl{nbl}.txt')

# from common_data/sylvain/FM/common_ell_and_deltas/latest_downloads/renamed
# FM_GO_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/common_data/sylvain/FM/common_ell_and_deltas/'
#                           f'latest_downloads/renamed/FM_{probe}_GO_lmax{probe}{lmax}_nbl{nbl}_ellDavide.txt')
# FM_GS_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/common_data/sylvain/FM/common_ell_and_deltas/'
#                           f'latest_downloads/renamed/FM_{probe}_GS_lmax{probe}{lmax}_nbl{nbl}_ellDavide.txt')

# from SSC_restructured_v2/jobs/SSC_comparison/output/FM/
# FM_GO_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/output/FM'
#                           f'/FM_{probe}_GO_lmax{probe}{lmax}_nbl{nbl}.txt')
# FM_GS_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/output/FM'
#                           f'/FM_{probe}_GS_lmax{probe}{lmax}_nbl{nbl}_Rlconst.txt')
#
#
# FM_GS_PyCCL = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/PyCCL_forecast/output/'
#                             f'FM/FM_{probe}_GS_lmax{probe}{lmax}_nbl{nbl}_PyCCLKiDS1000.txt')

# fix the desired parameters and remove null rows/columns
FM_PySSC_GO, param_names, fiducials = mm.mask_FM(FM_PySSC_GO, _param_names, _fiducials, n_cosmo_params, fix_IA, fix_gal_bias)
FM_PySSC_GS, _, _ = mm.mask_FM(FM_PySSC_GS, _param_names, _fiducials, n_cosmo_params, fix_IA, fix_gal_bias)
wzwa_idx = [param_names.index('wz'), param_names.index('wa')]

FM_PyCCL_GO, param_names, fiducials = mm.mask_FM(FM_PyCCL_GO, _param_names, _fiducials, n_cosmo_params, fix_IA, fix_gal_bias)
FM_PyCCL_GS, _, _ = mm.mask_FM(FM_PyCCL_GS, _param_names, _fiducials, n_cosmo_params, fix_IA, fix_gal_bias)

assert 1 > 2


# FMs = [FM_GO, FM_GS, FM_GS_PyCCL]
# cases = ['G', 'GS', 'GS_PyCCL', 'percent_diff_GS', 'percent_diff_GS_PyCCL']
FMs = [FM_PySSC_GO, FM_GO_old, FM_PySSC_GS, FM_GS_old, FM_GS_PyCCL]
cases = ['G', 'G_old', 'GS', 'GS_old', 'GS_PyCCL', 'percent_diff_GS', 'percent_diff_GS_old']

# compute uncertainties
uncert_dict = {}
fom = {}
for FM, case in zip(FMs, cases):
    uncert_dict[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fiducials[:nparams_toplot],
                                                       which_uncertainty=which_uncertainty, normalize=True))
    fom[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx)
    print(f'FoM({probe}, {case}): {fom[case]}')

uncert_dict['percent_diff_GS'] = diff_funct(uncert_dict['GS'], uncert_dict['G'])
uncert_dict['percent_diff_GS_old'] = diff_funct(uncert_dict['GS_old'], uncert_dict['G_old'])
# uncert_dict['percent_diff_GS_PyCCL'] = diff_funct(uncert_dict['GS_PyCCL'], uncert_dict['G'])
uncert_dict['ratio'] = uncert_dict['GS'] / uncert_dict['G']

# check against IST:F (which do not exist for GC alone):
if probe != 'GC':
    uncert_dict['ISTF'] = ISTF_fid.forecasts[f'{probe}_opt_w0waCDM_flat']
    diff = diff_funct(uncert_dict['ISTF'], uncert_dict['G'])
    assert np.all(np.abs(diff) < 5.0), f'IST:F and G are not consistent!'

# transform dict. into an array
uncert_array = []
for case in cases:
    uncert_array.append(uncert_dict[case])
uncert_array = np.asarray(uncert_array)

title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, lmax, EP_or_ED, zbins)
plot_utils.bar_plot(uncert_array[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
                    param_names_label=param_names[:nparams_toplot], bar_width=0.12)
