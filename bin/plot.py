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

lmax = 3000
if probe == 'WL':
    lmax = 5000

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

uncert_ratio_dict[probe] = {}

# import FM dict and
FM_dict = mm.load_pickle(f'{project_path}/jobs/ISTF/output/FM/{SSC_code}/FM_dict_{EP_or_ED}{zbins:02}.pickle')
_params = FM_dict['parameters']  # this should not change when passed the second time to the function
_fid = FM_dict['fiducial_values']  # this should not change when passed the second time to the function
FM_GO = FM_dict[f'FM_{probe}_GO']
FM_GS = FM_dict[f'FM_{probe}_GS']

# fix the desired parameters and remove null rows/columns
FM_GO, params, fid = mm.mask_FM(FM_GO, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
FM_GS, _, _ = mm.mask_FM(FM_GS, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
wzwa_idx = [params.index('wz'), params.index('wa')]

FMs = [FM_GO, FM_GS]
cases = ['G', 'GS', 'percent_diff']

# compute uncertainties
uncert_dict = {}
fom = {}
for FM, case in zip(FMs, cases):
    uncert_dict[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fid[:nparams_toplot],
                                                       which_uncertainty=which_uncertainty, normalize=True))
    fom[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx)
    print(f'FoM({probe}, {case}): {fom[case]}')

uncert_dict['percent_diff'] = diff_funct(uncert_dict['GS'], uncert_dict['G'])
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
                    param_names_label=params[:nparams_toplot], bar_width=0.12)
