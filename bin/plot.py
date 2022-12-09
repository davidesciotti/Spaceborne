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
FM_dict = mm.load_pickle(f'{project_path}/jobs/ISTF/output/FM/FM_dict_{EP_or_ED}{zbins:02}.pickle')
_params = FM_dict['parameters']  # this should not change when passed the second time to the function
_fid = FM_dict['fiducial_values']
FM_WL_GO = FM_dict['FM_WL_GO']
FM_WL_GS = FM_dict['FM_WL_GS']

FM_WL_GO_old = np.genfromtxt('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/'
                             'output/FM/FM_WL_GO_lmaxWL5000_nbl30.txt')
FM_WL_GO_old_2 = np.genfromtxt('/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/ISTspecs_indVincenzo/Cij_14may/FM_WL_G_lmaxWL5000_nbl30.txt')

# fix the desired parameters and remove null rows/columns
FM_WL_GO, params, fid = mm.mask_FM(FM_WL_GO, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
FM_WL_GS, _, _ = mm.mask_FM(FM_WL_GS, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
wzwa_idx = [params.index('wz'), params.index('wa')]

FMs = [FM_WL_GO, FM_WL_GO_old, FM_WL_GO_old_2]
cases = ['G', 'FM_WL_GO_old', 'FM_WL_GO_old_2', 'ISTF']
probe = 'WL'

# compute uncertainties
uncert_dict = {}
fom = {}
for FM, case in zip(FMs, cases):
    uncert_dict[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fid[:nparams_toplot],
                                                       which_uncertainty=which_uncertainty, normalize=True))
    fom[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx)
    print(f'FoM({probe}, {case}): {fom[case]}')

uncert_dict['percent_diff'] = diff_funct(uncert_dict['G'], ISTF_fid.forecasts['WL_opt_w0waCDM_flat'])
# uncert_dict['ratio'] = uncert_dict['GS'] / uncert_dict['G']
uncert_dict['ISTF'] = ISTF_fid.forecasts['WL_opt_w0waCDM_flat']

np.set_printoptions(precision=2, suppress=True)
print(uncert_dict['G'])
print(ISTF_fid.forecasts['WL_opt_w0waCDM_flat'])
print('percent_diff dav vs ISTF\n', (uncert_dict['G']/ISTF_fid.forecasts['WL_opt_w0waCDM_flat'] - 1)*100)


# transform dict. into an array
uncert_array = []
for case in cases:
    uncert_array.append(uncert_dict[case])
uncert_array = np.asarray(uncert_array)

title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, lmax, EP_or_ED, zbins)
plot_utils.bar_plot(uncert_array[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
                    param_names_label=params[:nparams_toplot],
                    bar_width=0.12,
                    second_axis=False, no_second_axis_bars=0)

# plot


assert 1 > 2

for probe in probes:

    lmax = 3000
    if probe == 'WL':
        lmax = 5000

    title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, ell_max, EP_or_ED, zbins)
    title += f'\nML = {magcut_lens / 10}, MS = {magcut_source / 10}, ZL = {zcut_lens / 10}, ZS = {zcut_source / 10:}, zmax = 2.5'

    # TODO try with pandas dataframes

    # print('3', FM_GO.shape)
    # if model == 'flat':
    #     FM_GO = np.delete(FM_GO, obj=1, axis=0)
    #     FM_GO = np.delete(FM_GO, obj=1, axis=1)
    #     FM_GS = np.delete(FM_GS, obj=1, axis=0)
    #     FM_GS = np.delete(FM_GS, obj=1, axis=1)
    #     cosmo_params = 7
    # elif model == 'nonflat':
    #     w0wa_rows = [3, 4]  # Omega_DE is in position 1, so w0, wa are shifted by 1 position
    #     nparams += 1
    #     cosmo_params = 8
    #     fid = np.insert(arr=fid, obj=1, values=ISTF_fid.extensions['Om_Lambda0'], axis=0)
    # pars_labels_TeX = np.insert(arr=pars_labels_TeX, obj=1, values='$\\Omega_{\\rm DE, 0}$', axis=0)

    # fid = fid[:nparams]
    # pars_labels_TeX = pars_labels_TeX[:nparams]

    ####################################################################################################################

    cases = ('G', 'GS')
    FMs = (FM_GO, FM_GS)

    data = []
    fom = {}
    uncert_dict = {}
    uncert_ratio_dict[probe][ML][ZL][MS][ZS] = []
    uncert_G_dict[probe][ML][ZL][MS][ZS] = []
    uncert_GS_dict[probe][ML][ZL][MS][ZS] = []
    for FM, case in zip(FMs, cases):
        uncert_dict[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams, fiducials=fid,
                                                           which_uncertainty=which_uncertainty, normalize=True))
        fom[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_rows)
        print(f'FoM({probe}, {case}): {fom[case]}')

    uncert_dict['percent_diff'] = diff_funct(uncert_dict['GS'], uncert_dict['G'])
    uncert_dict['ratio'] = uncert_dict['GS'] / uncert_dict['G']
    cases = ['G', 'GS', 'percent_diff']

    for case in cases:
        data.append(uncert_dict[case])

    # store uncertainties in dictionaries to easily retrieve them in the different cases
    uncert_G_dict[probe][ML][ZL][MS][ZS] = uncert_dict['G']
    uncert_GS_dict[probe][ML][ZL][MS][ZS] = uncert_dict['GS']
    uncert_ratio_dict[probe][ML][ZL][MS][ZS] = uncert_dict['ratio']
    # append the FoM values at the end of the array
    uncert_ratio_dict[probe][ML][ZL][MS][ZS] = np.append(
        uncert_ratio_dict[probe][ML][ZL][MS][ZS], fom['GS'] / fom['G'])

    for probe in probes:
        for zbins in zbins_list:
            for pes_opt in ('opt', 'pes'):
                data = np.asarray(data)
                plot_utils.bar_plot(data[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
                                    param_names_label=paramnames_3x2pt[:nparams_toplot],
                                    bar_width=0.12,
                                    second_axis=False, no_second_axis_bars=1)

            # plt.savefig(job_path / f'output/plots/{which_comparison}/'
            #                        f'bar_plot_{probe}_ellmax{ell_max}_zbins{EP_or_ED}{zbins:02}_Rl{which_Rl}_{which_uncertainty}.png')
