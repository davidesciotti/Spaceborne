import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
zbins = 13
zbins_list = np.array((zbins,), dtype=int)
probes = ('WL',)
pes_opt_list = ('opt',)
EP_or_ED_list = ('ED',)
which_comparison = 'GO_vs_GS'  # this is just to set the title of the plot
which_Rl = 'var'
nparams_chosen = 7
which_job = 'SPV3'
model = 'flat'
which_diff = 'normal'
flagship_version = 2
check_old_FM = False
pes_opt = 'opt'
which_uncertainty = 'marginal'
fix_gal_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_shear_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_dzWL = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_dzGC = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_IA = False
w0wa_rows = [2, 3]
bar_plot_cosmo = True
triangle_plot = False
plot_prior_contours = False
bar_plot_nuisance = False
pic_format = 'pdf'
BNT_transform = False
dpi = 500
magcut_lens = 230
magcut_source = 245
zcut_lens = 0
zcut_source = 0
zmax = 25
whos_BNT = 'stefano'
EP_or_ED = 'ED'
n_cosmo_params = 8
nparams_toplot = n_cosmo_params
# ! end options

ML_list = [230, 230, 245, 245]
ZL_list = [0, 2, 0, 2]
MS_list = [245, 245, 245, 245]
ZS_list = [0, 0, 2, 2]

job_path = project_path / f'jobs/{which_job}'
uncert_ratio_dict = {}
uncert_G_dict = {}
uncert_GS_dict = {}

# TODO fix this
if which_job == 'SPV3':
    nbl = 32
else:
    raise ValueError

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

for probe in probes:
    uncert_ratio_dict[probe] = {}
    for ML in ML_list:
        uncert_ratio_dict[probe][ML] = {}
        for ZL in ZL_list:
            uncert_ratio_dict[probe][ML][ZL] = {}
            for MS in MS_list:
                uncert_ratio_dict[probe][ML][ZL][MS] = {}
                for ZS in ZS_list:
                    uncert_ratio_dict[probe][ML][ZL][MS][ZS] = []

for probe in probes:
    for ML, ZL, MS, ZS in zip(ML_list, ZL_list, MS_list, ZS_list):

        lmax = 3000
        nbl = 29
        if probe == 'WL':
            lmax = 5000
            nbl = 32

        FM_path = f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output' \
                  f'/Flagship_{flagship_version}/BNT_{BNT_transform}/{whos_BNT}/FM'
        FM_dict = mm.load_pickle(f'{FM_path}/FM_dict_ML{ML}_ZL{ZL}_MS{MS}_ZS{ZS}.pickle')
        _params = FM_dict['parameters']  # this should not change when passed the second time to the function
        _fid = FM_dict['fiducial_values']  # this should not change when passed the second time to the function
        FM_GO = FM_dict[f'FM_{probe}_GO']
        FM_GS = FM_dict[f'FM_{probe}_GS']

        # fix the desired parameters and remove null rows/columns
        FM_GO, params, fid = mm.mask_FM(FM_GO, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        FM_GS, _, _ = mm.mask_FM(FM_GS, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        wzwa_idx = [params.index('wz'), params.index('wa')]
        assert len(fid) == len(params), 'the fiducial values list and parameter names should have the same length'


        title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, lmax, EP_or_ED, zbins)
        title += f'\nML = {magcut_lens / 10}, MS = {magcut_source / 10}, ZL = {zcut_lens / 10}, ZS = {zcut_source / 10:}, zmax = 2.5'


        cases = ('G', 'GS')
        FMs = (FM_GO, FM_GS)

        data = []
        fom = {}
        uncert = {}
        uncert_ratio_dict[probe][ML][ZL][MS][ZS] = []
        uncert_G_dict[probe][ML][ZL][MS][ZS] = []
        uncert_GS_dict[probe][ML][ZL][MS][ZS] = []
        for FM, case in zip(FMs, cases):
            uncert[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fid[:nparams_toplot],
                                                          which_uncertainty=which_uncertainty, normalize=True))
            fom[case] = mm.compute_FoM(FM, w0wa_idxs=w0wa_rows)
            print(f'FoM({probe}, {case}): {fom[case]}')

        uncert['percent_diff'] = diff_funct(uncert['GS'], uncert['G'])
        uncert['ratio'] = uncert['GS'] / uncert['G']
        cases = ['G', 'GS', 'percent_diff']

        for case in cases:
            data.append(uncert[case])

        # store uncertainties in dictionaries to easily retrieve them in the different cases
        uncert_G_dict[probe][ML][ZL][MS][ZS] = uncert['G']
        uncert_GS_dict[probe][ML][ZL][MS][ZS] = uncert['GS']
        uncert_ratio_dict[probe][ML][ZL][MS][ZS] = uncert['ratio']
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
