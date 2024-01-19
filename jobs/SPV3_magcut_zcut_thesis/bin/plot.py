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
matplotlib.use('Agg')
markersize = 10

########################################################################################################################

# ! options
zbins = 13
zbins_list = np.array((zbins,), dtype=int)
probes = ('3x2pt', 'WL', 'GC')
model = 'flat'
which_diff = 'normal'
flagship_version = 2
check_old_FM = False
pes_opt = 'opt'
which_uncertainty = 'marginal'
fix_IA = False
fix_gal_bias = False  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_shear_bias = False  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_dzWL = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_dzGC = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
bar_plot_cosmo = True
triangle_plot = False
bar_plot_nuisance = False
dpi = 500
zmax = 25
EP_or_ED = 'ED'
n_cosmo_params = 8
nparams_toplot = n_cosmo_params
pic_format = 'pdf'
GO_or_GS = 'GS'
plot_fom = True
# ! end options

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

# ML_list = [230, 230, 245, 245]
# ZL_list = [0, 2, 0, 2]
# MS_list = [245, 245, 245, 245]
# ZS_list = [0, 0, 0, 2]
#
ML_list = [245]
ZL_list = [0]
MS_list = [245]
ZS_list = [0]

uncert_ratio_dict = {}
uncert_G_dict = {}
uncert_GS_dict = {}

# for probe in probes:
#     uncert_ratio_dict[probe] = {}
#     uncert_G_dict[probe] = {}
#     uncert_GS_dict[probe] = {}
#     for ML in ML_list:
#         uncert_ratio_dict[probe][ML] = {}
#         uncert_G_dict[probe][ML] = {}
#         uncert_GS_dict[probe][ML] = {}
#         for ZL in ZL_list:
#             uncert_ratio_dict[probe][ML][ZL] = {}
#             uncert_G_dict[probe][ML][ZL] = {}
#             uncert_GS_dict[probe][ML][ZL] = {}
#             for MS in MS_list:
#                 uncert_ratio_dict[probe][ML][ZL][MS] = {}
#                 uncert_G_dict[probe][ML][ZL][MS] = {}
#                 uncert_GS_dict[probe][ML][ZL][MS] = {}
#                 for ZS in ZS_list:
#                     uncert_ratio_dict[probe][ML][ZL][MS][ZS] = []
#                     uncert_G_dict[probe][ML][ZL][MS][ZS] = []
#                     uncert_GS_dict[probe][ML][ZL][MS][ZS] = []

for probe in probes:
    for ML, ZL, MS, ZS in zip(ML_list, ZL_list, MS_list, ZS_list):

        lmax = 3000
        nbl = 29
        if probe == 'WL':
            lmax = 5000
            nbl = 32

        FM_noBNT_path = f'/home/davide/Documenti/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut/output' \
                        f'/Flagship_{flagship_version}/FM/BNT_False'
        FM_BNT_path = f'/home/davide/Documenti/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut/output' \
                      f'/Flagship_{flagship_version}/FM/BNT_True'
        FM_noBNT_dict = mm.load_pickle(f'{FM_noBNT_path}/FM_dict_ML{ML:03d}-ZL{ZL:02d}-MS{MS:03d}-ZS{ZS:02d}.pickle')
        FM_BNT_dict = mm.load_pickle(f'{FM_BNT_path}/FM_dict_ML{ML:03d}-ZL{ZL:02d}-MS{MS:03d}-ZS{ZS:02d}.pickle')
        _params = FM_noBNT_dict['parameters']  # this should not change when passed the second time to the function
        _fid = FM_noBNT_dict['fiducial_values']  # this should not change when passed the second time to the function
        FM_GO_noBNT = FM_noBNT_dict[f'FM_{probe}_GO']
        FM_GS_noBNT = FM_noBNT_dict[f'FM_{probe}_GS']
        FM_GO_BNT = FM_BNT_dict[f'FM_{probe}_GO']
        FM_GS_BNT = FM_BNT_dict[f'FM_{probe}_GS']

        # fix the desired parameters and remove null rows/columns
        FM_GO_noBNT, param_names, fid = mm.mask_FM(FM_GO_noBNT, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        FM_GS_noBNT, _, _ = mm.mask_FM(FM_GS_noBNT, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        FM_GO_BNT, _, _ = mm.mask_FM(FM_GO_BNT, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        FM_GS_BNT, _, _ = mm.mask_FM(FM_GS_BNT, _params, _fid, n_cosmo_params, fix_IA, fix_gal_bias)
        wzwa_idx = [param_names.index('wz'), param_names.index('wa')]
        assert len(fid) == len(param_names), 'the fiducial values list and parameter names should have the same length'

        title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, lmax, EP_or_ED, zbins)
        title += f'\nML = {ML / 10}, MS = {MS / 10}, ZL = {ZL / 10}, ZS = {ZS / 10}, zmax = {zmax / 10}'

        if GO_or_GS == 'GO':
            FMs = (FM_GO_noBNT, FM_GO_BNT)
        elif GO_or_GS == 'GS':
            FMs = (FM_GS_noBNT, FM_GS_BNT)
        else:
            raise ValueError('GO_or_GS should be either GO or GS')

        # cases = [f'FM_{GO_or_GS}', f'FM_{GO_or_GS}_BNT', 'percent_diff/10']
        cases = [f'noBNT', f'BNT', 'abs(percent_diff)']
        # cases = [f'noBNT', f'BNT']
        # key_to_compare_A, key_to_compare_B = f'FM_{GO_or_GS}', f'FM_{GO_or_GS}_BNT'  # which cases to take the percent diff and ratio of
        key_to_compare_A, key_to_compare_B = f'noBNT', f'BNT'  # which cases to take the percent diff and ratio of

        data = []
        fom = {}
        uncert = {}
        for FM, case in zip(FMs, cases):
            uncert[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fid[:nparams_toplot],
                                                          which_uncertainty=which_uncertainty, normalize=True))
            fom[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx)/10
            print(f'FoM({probe}, {case}): {fom[case]}')

        uncert['abs(percent_diff)'] = np.abs(diff_funct(uncert[key_to_compare_A], uncert[key_to_compare_B]))
        uncert['ratio'] = uncert[key_to_compare_A] / uncert[key_to_compare_B]

        for case in cases:
            data.append(uncert[case])

        # # store uncertainties in dictionaries to easily retrieve them in the different cases
        # uncert_G_dict[probe][ML][ZL][MS][ZS] = uncert['G']
        # uncert_GS_dict[probe][ML][ZL][MS][ZS] = uncert['GS']
        # uncert_ratio_dict[probe][ML][ZL][MS][ZS] = uncert['ratio']
        # # append the FoM values at the end of the array
        # uncert_ratio_dict[probe][ML][ZL][MS][ZS] = np.append(
        #     uncert_ratio_dict[probe][ML][ZL][MS][ZS], fom['GS'] / fom['G'])

        # fom_array = np.array([fom['FM_GO'], fom['FM_GO_BNT'], mm.percent_diff(fom['FM_GO'], fom['FM_GO_BNT'])])/100
        fom_array = np.array([fom['noBNT'], fom['BNT'], np.abs(mm.percent_diff(fom['noBNT'], fom['BNT']))])
        # fom_array = np.array([fom['noBNT'], fom['BNT']])
        if plot_fom:
            param_names_label = param_names[:nparams_toplot] + ['FoM/10']
            nparams_toplot += 1

        data = np.asarray(data)
        data = np.column_stack((data, fom_array))
        plot_utils.bar_plot(data[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
                            param_names_label=param_names_label, bar_width=0.15)

        # plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
        #                        f'bar_plot_{probe}_ellmax{lmax}_zbins{EP_or_ED}{zbins:02}'
        #                        f'_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.png')
