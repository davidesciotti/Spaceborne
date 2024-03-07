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
job_path = Path.cwd().parent

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

markersize = 10


def plot_from_dataframe(fom_df, key_1, key_2, key_3, key_4, constant_fom_idx, plot_hlines, title, save,
                        filename_suffix):
    ellmax3000_TeX = '$\\ell_{\\rm max} = 3000 \\; \\forall z_i, z_j$'
    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.plot(fom_df['kmax_1_over_Mpc'], fom_df[key_1], label='FoM G', marker='o', c='tab:blue')
    plt.plot(fom_df['kmax_1_over_Mpc'], fom_df[key_2], label='FoM GS', marker='o', c='tab:orange')
    if plot_hlines:
        plt.axhline(fom_df[key_3][constant_fom_idx], label=f'FoM G, {ellmax3000_TeX}', ls='--', color='tab:blue')
        plt.axhline(fom_df[key_4][constant_fom_idx], label=f'FoM GS, {ellmax3000_TeX}', ls='--', color='tab:orange')
    plt.xlabel("$k_{\\rm max}[1/Mpc]$")
    plt.ylabel("FoM")
    plt.legend()
    plt.grid()
    plt.show()
    plt.tight_layout()

    if save:
        plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
                               f'FoM_vs_kmax_{probe}_zbins{EP_or_ED}{zbins:02}'
                               f'ML{ML:03d}_MS{MS:03d}_{filename_suffix}.png')


########################################################################################################################

# ! options
h = 0.67
zbins = 13
probe = '3x2pt'
model = 'nonflat'
which_diff = 'normal'
flagship_version = 2
which_uncertainty = 'marginal'
bar_plot_cosmo = True
triangle_plot = False
bar_plot_nuisance = False
dpi = 500
zmax = 25
EP_or_ED = 'ED'
n_cosmo_params = 7
pic_format = 'pdf'
plot_fom = False
divide_fom_by_10 = False
shear_bias_priors = True
params_tofix_dict = {
    'cosmo': False,
    'IA': False,
    'galbias': False,
    'shearbias': False,
    'dzWL': True,
    'dzGC': True,
}
BNT_transform = True
ell_cuts = True
center_or_min = 'ell_center'

kmax_1_over_Mpc_filename = np.array((25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 1000), dtype=int)
# ! end options

ell_cuts_subfolder = f'/{center_or_min}'
if not ell_cuts:
    ell_cuts_subfolder = ''

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

if model == 'nonflat':
    n_cosmo_params += 1

# ML_list = [230, 230, 245, 245]
# ZL_list = [0, 2, 0, 2]
# MS_list = [245, 245, 245, 245]
# ZS_list = [0, 0, 0, 2]  # it think this is wrong
# ZS_list = [0, 2, 0, 2]

# ML_list = [245, 245]
# ZL_list = [0, 2]
# MS_list = [245, 245]
# ZS_list = [0, 2]

ML_list = [245]
ZL_list = [0]
MS_list = [245]
ZS_list = [0]

fom_df = pd.DataFrame()

for ML, ZL, MS, ZS in zip(ML_list, ZL_list, MS_list, ZS_list):
    k_max_counter = 0
    for kmax_h_over_Mpc in cfg.general_cfg['kmax_h_over_Mpc_list']:
        for which_cuts in ['Francis', 'Vincenzo']:
            for BNT_transform in (True, False):
                for center_or_min in ('ell_center', 'ell_min'):
                    # for BNT_transform, which_cuts, center_or_min in cases_done:
                    #     print(BNT_transform, which_cuts, center_or_min)

                    # assert params_tofix_dict['cosmo'] is False and params_tofix_dict['IA'] is False and \
                    #        params_tofix_dict['galbias'] is False and params_tofix_dict['shearbias'] is False and \
                    #        params_tofix_dict['dzWL'] is True and params_tofix_dict['dzGC'] is True, \
                    #     'the other cases are not implemented yet'

                    # these have to be initialized at every iteration
                    cases = []
                    nparams_toplot = n_cosmo_params

                    lmax = 3000
                    nbl = 29
                    if probe == 'WL':
                        lmax = 5000
                        nbl = 32

                    fm_path = f'/home/davide/Documenti/Lavoro/Programmi/Spaceborne/jobs/SPV3_magcut_zcut/' \
                              f'output/Flagship_{flagship_version}/FM'
                    fm_ellcuts_path = f'{fm_path}/BNT_{BNT_transform}/ell_cuts_True/{which_cuts}/{center_or_min}'
                    fm_no_ellcuts_path = f'{fm_path}/BNT_{BNT_transform}/ell_cuts_False'

                    fm_ellcuts_filename = f'FM_zbins{EP_or_ED}{zbins:02d}-ML{ML:03d}-ZL{ZL:02d}-MS{MS:03d}-ZS{ZS:02d}' \
                                          f'_kmax_h_over_Mpc{kmax_h_over_Mpc:03f}.pickle'
                    fm_no_ellcuts_filename = f'FM_zbins{EP_or_ED}{zbins:02d}-ML{ML:03d}-ZL{ZL:02d}-MS{MS:03d}-ZS{ZS:02d}.pickle'

                    fm_ellcuts_dict = mm.load_pickle(f'{fm_ellcuts_path}/{fm_ellcuts_filename}')
                    fm_no_ellcuts_dict = mm.load_pickle(f'{fm_no_ellcuts_path}/{fm_no_ellcuts_filename}')

                    # this is just as a reference; the values roughly match.
                    # FM_GS_kcuts_vinc = np.genfromtxt(
                    #     '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022'
                    #     '/Flagship_2/TestKappaMax/'
                    #     f'fm-3x2pt-wzwaCDM-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-ED13-'
                    #     f'kM{kmax_1_over_Mpc_filename[k_max_counter]:03d}.dat')
                    # brutal cut of last 13 dz params
                    # FM_GS_kcuts_vinc = FM_GS_kcuts_vinc[:-zbins, :-zbins]

                    # parameter names
                    param_names_dict = fm_no_ellcuts_dict['param_names_dict']
                    fiducial_values_dict = fm_no_ellcuts_dict['fiducial_values_dict']

                    assert param_names_dict == fm_ellcuts_dict['param_names_dict'], \
                        'param_names_dict not equal for Ellcuts and noEllcuts'
                    for key in fiducial_values_dict.keys():
                        assert np.all(fiducial_values_dict[key] == fm_ellcuts_dict['fiducial_values_dict'][key]), \
                            'fiducial_values_dict not equal for Ellcuts and noEllcuts'

                    # rename for convenience
                    FM_GO_Ellcuts = fm_ellcuts_dict[f'FM_{probe}_GO']
                    FM_GS_Ellcuts = fm_ellcuts_dict[f'FM_{probe}_GS']
                    FM_GO_noEllcuts = fm_no_ellcuts_dict[f'FM_{probe}_GO']
                    FM_GS_noEllcuts = fm_no_ellcuts_dict[f'FM_{probe}_GS']

                    # fix the desired parameters and remove null rows/columns
                    FM_GO_noEllcuts, param_names_list, fiducials_list = mm.mask_FM(FM_GO_noEllcuts, param_names_dict,
                                                                                   fiducial_values_dict,
                                                                                   params_tofix_dict)
                    FM_GS_noEllcuts, _, _ = mm.mask_FM(FM_GS_noEllcuts, param_names_dict, fiducial_values_dict,
                                                       params_tofix_dict)
                    FM_GO_Ellcuts, _, _ = mm.mask_FM(FM_GO_Ellcuts, param_names_dict, fiducial_values_dict,
                                                     params_tofix_dict)
                    FM_GS_Ellcuts, _, _ = mm.mask_FM(FM_GS_Ellcuts, param_names_dict, fiducial_values_dict,
                                                     params_tofix_dict)

                    wzwa_idx = [param_names_list.index('wz'), param_names_list.index('wa')]

                    assert FM_GO_noEllcuts.shape[0] == len(param_names_list) == len(fiducials_list), \
                        'the number of rows should be equal to the number of parameters and fiducials'

                    FMs = [FM_GO_noEllcuts, FM_GO_Ellcuts, FM_GS_noEllcuts, FM_GS_Ellcuts]
                    cases = ['FM_GO_noEllcuts', 'FM_GO_Ellcuts', 'FM_GS_noEllcuts', 'FM_GS_Ellcuts']
                    # cases += ['abs(percent_diff)']

                    # ! priors on shear bias
                    if shear_bias_priors:
                        shear_bias_1st_idx = param_names_list.index('m01')
                        shear_bias_last_idx = param_names_list.index(f'm{zbins:02}')
                        prior = np.zeros(FM_GO_Ellcuts.shape)
                        for i in range(shear_bias_1st_idx, shear_bias_last_idx + 1):
                            prior[i, i] = 5e-4 ** -2
                        FMs = [FM + prior for FM in FMs]

                    # ! this is to compute uncertaintiens on the cosmo params, and percent differences
                    key_to_compare_A, key_to_compare_B = cases[0], cases[
                        1]  # which cases to take the percent diff and ratio of

                    data = []
                    fom_dict = {}
                    uncert_dict = {}
                    for FM, case in zip(FMs, cases):
                        uncert_dict[case] = np.asarray(
                            mm.uncertainties_FM(FM, nparams=nparams_toplot, fiducials=fiducials_list[:nparams_toplot],
                                                which_uncertainty=which_uncertainty, normalize=True))
                        fom_dict[case] = mm.compute_FoM(FM, w0wa_idxs=wzwa_idx)
                        print(f'FoM({probe}, {case}): {fom_dict[case]}')

                    uncert_dict['abs(percent_diff)'] = np.abs(
                        diff_funct(uncert_dict[key_to_compare_A], uncert_dict[key_to_compare_B]))
                    uncert_dict['ratio'] = uncert_dict[key_to_compare_A] / uncert_dict[key_to_compare_B]

                    for case in cases:
                        data.append(uncert_dict[case])

                    data = np.asarray(data)
                    param_names_label = param_names_list[:nparams_toplot]

                    # add the FoM to the usual bar plot
                    if plot_fom:
                        fom_array = np.array([fom_dict[key_to_compare_A], fom_dict[key_to_compare_B],
                                              np.abs(
                                                  mm.percent_diff(fom_dict[key_to_compare_A],
                                                                  fom_dict[key_to_compare_B]))])

                        if divide_fom_by_10:
                            fom_array[0] /= 10
                            fom_array[1] /= 10
                            param_names_label += ['FoM/10']
                        else:
                            param_names_label += ['FoM']

                        nparams_toplot += 1
                        data = np.column_stack((data, fom_array))

                    # ! here I only plot the FoM as a function of kmax using pd.DataFrame

                    # create list with the quantites you want to keep track of, and add it as row of the df. You will plot outside
                    # the for loop simply choosing the entries of the df you want.
                    fom_list = [probe, ML, ZL, MS, ZS, kmax_h_over_Mpc, kmax_h_over_Mpc * h,
                                BNT_transform, which_cuts, center_or_min,
                                fom_dict[cases[0]], fom_dict[cases[1]],
                                fom_dict[cases[2]], fom_dict[cases[3]]]

                    fom_df = fom_df.append(pd.DataFrame([fom_list],
                                                        columns=['probe', 'ML', 'ZL', 'MS', 'ZS',
                                                                 'kmax_h_over_Mpc', 'kmax_1_over_Mpc', 'BNT',
                                                                 'which_cuts',
                                                                 'center_or_min',
                                                                 'FM_GO_noEllcuts',
                                                                 'FM_GO_Ellcuts',
                                                                 'FM_GS_noEllcuts',
                                                                 'FM_GS_Ellcuts']), ignore_index=True)

                    # title = '%s, $\\ k_{\\rm max}[h/Mpc] = %.2f$, zbins %s%i$' % (probe, kmax_h_over_Mpc, EP_or_ED, zbins)
                    # title += f'\nML = {ML / 10}, MS = {MS / 10}, ZL = {ZL / 10}, ZS = {ZS / 10}, zmax = {zmax / 10}'
                    # plot_utils.bar_plot(data[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
                    #                     param_names_label=param_names_label, bar_width=0.15)

                    # plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
                    #                        f'bar_plot_{probe}_ellmax{lmax}_zbins{EP_or_ED}{zbins:02}'
                    #                        f'_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.png')
                    k_max_counter += 1

fom_df_zmin00 = fom_df.loc[fom_df['ZL'] == 0]
fom_df_zmin02 = fom_df.loc[fom_df['ZL'] == 2]

title = f'{probe} (no GCsp), zbins {EP_or_ED}{zbins}, zmax = {zmax / 10}' \
        '\nprior on $\\sigma(m) = 5 \\times 10^{-4}$' \
        '; ${\\rm dzWL, dzGCph}$ fixed'
# f'\nML = {ML / 10}, MS = {MS / 10}, zmin = 0, ' \

# plot_from_dataframe(fom_df=fom_df_zmin00,
#                     key_1='FM_GO_Ellcuts', key_2='FM_GS_Ellcuts',
#                     key_3='FM_GO_noEllcuts', key_4='FM_GS_noEllcuts',
#                     constant_fom_idx=0, plot_hlines=True, title=title,
#                     save=True, filename_suffix='zmin00')


# if it's better to take the min instead of ell_center (it s!)
fom_df_True_Francis_center = fom_df.loc[
    (fom_df['BNT'] == True) & (fom_df['which_cuts'] == 'Francis') & (fom_df['center_or_min'] == 'ell_center')]
fom_df_False_Francis_center = fom_df.loc[
    (fom_df['BNT'] == False) & (fom_df['which_cuts'] == 'Francis') & (fom_df['center_or_min'] == 'ell_center')]
fom_df_True_Francis_min = fom_df.loc[
    (fom_df['BNT'] == True) & (fom_df['which_cuts'] == 'Francis') & (fom_df['center_or_min'] == 'ell_min')]
fom_df_False_Francis_min = fom_df.loc[
    (fom_df['BNT'] == False) & (fom_df['which_cuts'] == 'Francis') & (fom_df['center_or_min'] == 'ell_min')]
fom_df_True_Vincenzo_center = fom_df.loc[
    (fom_df['BNT'] == True) & (fom_df['which_cuts'] == 'Vincenzo') & (fom_df['center_or_min'] == 'ell_center')]
fom_df_True_Vincenzo_min = fom_df.loc[
    (fom_df['BNT'] == True) & (fom_df['which_cuts'] == 'Vincenzo') & (fom_df['center_or_min'] == 'ell_min')]
fom_df_False_Vincenzo_center = fom_df.loc[
    (fom_df['BNT'] == False) & (fom_df['which_cuts'] == 'Vincenzo') & (fom_df['center_or_min'] == 'ell_center')]
fom_df_False_Vincenzo_min = fom_df.loc[
    (fom_df['BNT'] == False) & (fom_df['which_cuts'] == 'Vincenzo') & (fom_df['center_or_min'] == 'ell_min')]

ellmax3000_TeX = '$\\ell^{\\rm max} = 3000 \\; \\forall z_i, z_j$'
plt.figure(figsize=(12, 10))

# center-min check, Francis True
# plt.plot(fom_df_True_Francis_center['kmax_h_over_Mpc'], fom_df_True_Francis_center['FM_GO_Ellcuts'],
#          label='fom_df_True_Francis_center', marker='o', c='tab:orange')
# plt.plot(fom_df_True_Francis_min['kmax_h_over_Mpc'], fom_df_True_Francis_min['FM_GO_Ellcuts'],
#          label='fom_df_True_Francis_min', marker='o', c='tab:blue')

# ! center-min check, Francis False
# title += f'\nadvanced cuts, no BNT'
# plt.plot(fom_df_False_Francis_center['kmax_h_over_Mpc'], fom_df_False_Francis_center['FM_GO_Ellcuts'],
#          label='$\\ell^{\\rm center} < \\ell^{\\rm max}_{ij}$', marker='o', c='tab:orange')
# plt.plot(fom_df_False_Francis_min['kmax_h_over_Mpc'], fom_df_False_Francis_min['FM_GO_Ellcuts'],
#          label='$\\ell^{\\rm min} < \\ell^{\\rm max}_{ij}$', marker='o', c='tab:blue')

# center-min check, Vincenzo True
# plt.plot(fom_df_True_Vincenzo_center['kmax_h_over_Mpc'], fom_df_True_Vincenzo_center['FM_GO_Ellcuts'],
#          label='fom_df_True_Vincenzo_center', marker='o', c='tab:orange')
# plt.plot(fom_df_True_Vincenzo_min['kmax_h_over_Mpc'], fom_df_True_Vincenzo_min['FM_GO_Ellcuts'],
#          label='fom_df_True_Vincenzo_min', marker='o', c='tab:blue')

# center-min check, Vincenzo False
# plt.plot(fom_df_False_Vincenzo_center['kmax_h_over_Mpc'], fom_df_False_Vincenzo_center['FM_GO_Ellcuts'],
#          label='fom_df_False_Vincenzo_center', marker='o', c='tab:orange')
# plt.plot(fom_df_False_Vincenzo_min['kmax_h_over_Mpc'], fom_df_False_Vincenzo_min['FM_GO_Ellcuts'],
#          label='fom_df_False_Vincenzo_min', marker='o', c='tab:blue')

# BNT check, Francis min
# plt.plot(fom_df_True_Francis_min['kmax_h_over_Mpc'], fom_df_True_Francis_min['FM_GO_Ellcuts'],
#          label='fom_df_True_Francis_min', marker='o', c='tab:orange')
# plt.plot(fom_df_False_Francis_min['kmax_h_over_Mpc'], fom_df_False_Francis_min['FM_GO_Ellcuts'],
#          label='fom_df_False_Francis_min', marker='o', c='tab:blue')

# ! BNT check, Vincenzo center
# title += f'\nlinear cuts'
# perc_diff = (fom_df_True_Vincenzo_center['FM_GO_Ellcuts'].values / fom_df_False_Vincenzo_center[
#     'FM_GO_Ellcuts'].values - 1) * 100
# plt.plot(fom_df_True_Vincenzo_center['kmax_h_over_Mpc'], fom_df_True_Vincenzo_center['FM_GO_Ellcuts'],
#          label='BNT', marker='o', c='tab:orange')
# plt.plot(fom_df_False_Vincenzo_center['kmax_h_over_Mpc'], fom_df_False_Vincenzo_center['FM_GO_Ellcuts'],
#          label='no BNT', marker='o', c='tab:blue')
# plt.plot(fom_df_True_Vincenzo_center['kmax_h_over_Mpc'], perc_diff, label='percent difference', marker='o',
#          c='tab:green')

# BNT check, Vincenzo min
# plt.plot(fom_df_True_Vincenzo_min['kmax_h_over_Mpc'], fom_df_True_Vincenzo_min['FM_GO_Ellcuts'],
#          label='BNT', marker='o', c='tab:orange')
# plt.plot(fom_df_False_Vincenzo_min['kmax_h_over_Mpc'], fom_df_False_Vincenzo_min['FM_GO_Ellcuts'],
#          label='no BNT', marker='o', c='tab:blue')

# ! Vincenzo-Francis check, False center
plt.plot(fom_df_False_Francis_center['kmax_h_over_Mpc'], fom_df_False_Francis_center['FM_GO_Ellcuts'],
         label='advanced cuts', marker='o', c='tab:orange')
plt.plot(fom_df_False_Vincenzo_center['kmax_h_over_Mpc'], fom_df_False_Vincenzo_center['FM_GO_Ellcuts'],
         label='linear cuts', marker='o', c='tab:blue')


# ! SSC check, Francis False center
# title += f'\nadvanced cuts, no BNT'
# fig, ax = plt.subplots()
# perc_diff = (fom_df_False_Francis_center['FM_GS_Ellcuts'].values / fom_df_False_Francis_center[
#     'FM_GO_Ellcuts'].values - 1) * 100
# ax.plot(fom_df_False_Francis_center['kmax_h_over_Mpc'], fom_df_False_Francis_center['FM_GO_Ellcuts'],
#          label='Gauss', marker='o', c='tab:blue')
# ax.plot(fom_df_False_Francis_center['kmax_h_over_Mpc'], fom_df_False_Francis_center['FM_GS_Ellcuts'],
#          label='Gauss + SSC', marker='o', c='tab:orange')
# ax2 = ax.twinx()
# ax2.plot(fom_df_False_Francis_center['kmax_h_over_Mpc'], perc_diff, label='percent difference', marker='o',
#          c='tab:green', ls='--')
#
# # customize the axis labels and legend
# ax.set_xlabel("$k_{\\rm max}[h/Mpc]$")
# ax.set_ylabel("FoM")
# ax.grid()
# ax2.set_ylabel('GS/G - 1 [%]')
# ax.legend()
# ax2.legend(loc='lower right', bbox_to_anchor=(0.999, 0.15))
# ax.axhline(fom_df['FM_GO_noEllcuts'][0], label=f'FoM G, {ellmax3000_TeX}', ls='--', color='k')

# ! SSC check, Vincenzo False center
# title += f'\nlinear cuts, no BNT'
# fig, ax = plt.subplots()
# perc_diff = (fom_df_False_Vincenzo_center['FM_GS_Ellcuts'].values / fom_df_False_Vincenzo_center[
#     'FM_GO_Ellcuts'].values - 1) * 100
# ax.plot(fom_df_False_Vincenzo_center['kmax_h_over_Mpc'], fom_df_False_Vincenzo_center['FM_GO_Ellcuts'],
#          label='Gauss', marker='o', c='tab:blue')
# ax.plot(fom_df_False_Vincenzo_center['kmax_h_over_Mpc'], fom_df_False_Vincenzo_center['FM_GS_Ellcuts'],
#          label='Gauss + SSC', marker='o', c='tab:orange')
# ax2 = ax.twinx()
# ax2.plot(fom_df_False_Vincenzo_center['kmax_h_over_Mpc'], perc_diff, label='percent difference', marker='o',
#          c='tab:green', ls='--')
#
# # customize the axis labels and legend
# ax.set_xlabel("$k_{\\rm max}[h/Mpc]$")
# ax.set_ylabel("FoM")
# ax.grid()
# ax2.set_ylabel('GS/G - 1 [%]')
# ax.legend()
# ax2.legend(loc='lower right', bbox_to_anchor=(0.999, 0.15))
# ax.axhline(fom_df['FM_GO_noEllcuts'][0], label=f'FoM G, {ellmax3000_TeX}', ls='--', color='k')


# ! SSC BNT check, Francis center
# title += f'\nlinear cuts'
# fig, ax = plt.subplots()
# perc_diff = (fom_df_True_Vincenzo_center['FM_GS_Ellcuts'].values / fom_df_False_Vincenzo_center[
#     'FM_GS_Ellcuts'].values - 1) * 100
# ax.plot(fom_df_True_Vincenzo_center['kmax_h_over_Mpc'], fom_df_True_Vincenzo_center['FM_GS_Ellcuts'],
#          label='Gauss + SSC, BNT', marker='o', c='tab:blue')
# ax.plot(fom_df_False_Vincenzo_center['kmax_h_over_Mpc'], fom_df_False_Vincenzo_center['FM_GS_Ellcuts'],
#          label='Gauss + SSC, no BNT', marker='o', c='tab:orange')
# ax2 = ax.twinx()
# ax2.plot(fom_df_True_Francis_min['kmax_h_over_Mpc'], perc_diff, label='percent difference', marker='o',
#          c='tab:green', ls='--')
#
# # customize the axis labels and legend
# ax.set_xlabel("$k_{\\rm max}[h/Mpc]$")
# ax.set_ylabel("FoM")
# ax.grid()
# ax2.set_ylabel('BNT/no BNT - 1 [%]')
# ax.legend(loc='right')
# ax2.legend(loc='lower right', bbox_to_anchor=(0.999, 0.15))
# ax.axhline(fom_df['FM_GS_noEllcuts'][0], label=f'FoM G, {ellmax3000_TeX}', ls='--', color='k')



plt.axhline(fom_df['FM_GO_noEllcuts'][0], label=f'FoM G, {ellmax3000_TeX}', ls='--', color='k')
plt.xlabel("$k_{\\rm max}[h/Mpc]$")
plt.ylabel("FoM")
plt.title(title)
plt.legend()
plt.grid()
plt.show()
plt.tight_layout()
#

# if save:
#     plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
#                            f'FoM_vs_kmax_{probe}_zbins{EP_or_ED}{zbins:02}'
#                            f'ML{ML:03d}_MS{MS:03d}_{filename_suffix}.png')

# GO vs GS as a function of the ell cut
# plt.figure()
# plt.plot(fom_df['kmax_h_over_Mpc'], (fom_df['FM_GS_Ellcuts']/fom_df['FM_GO_Ellcuts'] - 1)*100)
# plt.xlabel("$k_{\\rm max}[1/Mpc]$")
# plt.ylabel("FM_GS/FM_GO - 1 [%]")


# title = '%s (no GCsp), zbins %s%i, BNT {BNT_transform}' \
#         f'\nML = {ML / 10}, MS = {MS / 10}, zmin = 0.2, zmax = {zmax / 10}' \
#         '\nprior on $\\sigma(m) = 5 \\times 10^{-4}$' \
#         '\n ${\\rm dzWL, dzGCph}$ fixed' % (probe, EP_or_ED, zbins)
# plot_from_dataframe(fom_df=fom_df_zmin02,
#                     key_1='FM_GO_Ellcuts', key_2='FM_GS_Ellcuts',
#                     key_3='FM_GO_noEllcuts', key_4='FM_GS_noEllcuts',
#                     constant_fom_idx=0, plot_hlines=True,
#                     title=title, save=True, filename_suffix='zmin02')
