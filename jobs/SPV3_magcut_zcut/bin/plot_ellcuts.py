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
matplotlib.use('Qt5Agg')
markersize = 10

########################################################################################################################

# ! options
h = h
zbins = 13
probes = ('3x2pt',)
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
# ! end options

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
# ZS_list = [0, 0, 0, 2]

ML_list = [245, 245]
ZL_list = [0, 2]
MS_list = [245, 245]
ZS_list = [0, 2]

# ML_list = [245]
# ZL_list = [0]
# MS_list = [245]
# ZS_list = [0]

# create pd dataframe
fom_df = pd.DataFrame()

for probe in probes:
    for ML, ZL, MS, ZS in zip(ML_list, ZL_list, MS_list, ZS_list):
        k_counter = 0
        for kmax_h_over_Mpc in cfg.general_cfg['kmax_list_h_over_Mpc']:

            assert params_tofix_dict['cosmo'] is False and params_tofix_dict['IA'] is False and \
                   params_tofix_dict['galbias'] is False and params_tofix_dict['shearbias'] is False and \
                   params_tofix_dict['dzWL'] is True and params_tofix_dict['dzGC'] is True, \
                'the other cases are not implemented yet'

            # these have to be initialized at every iteration
            cases = []
            nparams_toplot = n_cosmo_params

            lmax = 3000
            nbl = 29
            if probe == 'WL':
                lmax = 5000
                nbl = 32

            FM_Ellcuts_path = f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output' \
                              f'/Flagship_{flagship_version}/FM/BNT_True/ell_cuts_True'
            FM_noEllcuts_path = FM_Ellcuts_path.replace('ell_cuts_True', 'ell_cuts_False')
            FM_filename = f'FM_zbins{EP_or_ED}{zbins:02d}-ML{ML:03d}-ZL{ZL:02d}-MS{MS:03d}-ZS{ZS:02d}' \
                          f'_kmax_h_over_Mpc{kmax_h_over_Mpc:03f}.pickle'
            FM_Ellcuts_dict = mm.load_pickle(f'{FM_Ellcuts_path}/{FM_filename}')
            FM_noEllcuts_dict = mm.load_pickle(f'{FM_noEllcuts_path}/{FM_filename}')

            kmax_1_over_Mpc_filename = np.array((25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 1000), dtype=int)
            FM_GS_kcuts_vinc = np.genfromtxt(
                '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022'
                '/Flagship_2/TestKappaMax/'
                f'fm-3x2pt-wzwaCDM-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-ED13-'
                f'kM{kmax_1_over_Mpc_filename[k_counter]:03d}.dat')

            # brutal cut of last 13 dz params
            FM_GS_kcuts_vinc = FM_GS_kcuts_vinc[:-zbins, :-zbins]

            # remove 'param_names_' from the keys
            FM_noEllcuts_dict['param_names_dict'] = {key.replace('param_names_', ''): value for key, value in
                                                     FM_noEllcuts_dict['param_names_dict'].items()}
            FM_Ellcuts_dict['param_names_dict'] = {key.replace('param_names_', ''): value for key, value in
                                                   FM_Ellcuts_dict['param_names_dict'].items()}
            FM_noEllcuts_dict['fiducial_values_dict'] = {key.replace('fid_', ''): value for key, value in
                                                         FM_noEllcuts_dict['fiducial_values_dict'].items()}
            FM_Ellcuts_dict['fiducial_values_dict'] = {key.replace('fid_', ''): value for key, value in
                                                       FM_Ellcuts_dict['fiducial_values_dict'].items()}

            # remove 3X2pt key from the dict
            if '3x2pt' in FM_noEllcuts_dict['param_names_dict'].keys():
                FM_noEllcuts_dict['param_names_dict'].pop('3x2pt')
                FM_noEllcuts_dict['fiducial_values_dict'].pop('3x2pt')

            if '3x2pt' in FM_Ellcuts_dict['param_names_dict'].keys():
                FM_Ellcuts_dict['param_names_dict'].pop('3x2pt')
                FM_Ellcuts_dict['fiducial_values_dict'].pop('3x2pt')

            # parameter names
            param_names_dict = FM_noEllcuts_dict['param_names_dict']
            fiducial_values_dict = FM_noEllcuts_dict['fiducial_values_dict']

            assert param_names_dict == FM_Ellcuts_dict['param_names_dict'], \
                'param_names_dict not equal for Ellcuts and noEllcuts'
            for key in fiducial_values_dict.keys():
                assert np.all(fiducial_values_dict[key] == FM_Ellcuts_dict['fiducial_values_dict'][key]), \
                    'fiducial_values_dict not equal for Ellcuts and noEllcuts'

            # rename for convenience
            FM_GO_Ellcuts = FM_Ellcuts_dict[f'FM_{probe}_GO']
            FM_GS_Ellcuts = FM_Ellcuts_dict[f'FM_{probe}_GS']
            FM_GO_noEllcuts = FM_noEllcuts_dict[f'FM_{probe}_GO']
            FM_GS_noEllcuts = FM_noEllcuts_dict[f'FM_{probe}_GS']

            # fix the desired parameters and remove null rows/columns
            FM_GO_noEllcuts, param_names_list, fiducials_list = mm.mask_FM(FM_GO_noEllcuts, param_names_dict,
                                                                           fiducial_values_dict, params_tofix_dict)
            FM_GS_noEllcuts, _, _ = mm.mask_FM(FM_GS_noEllcuts, param_names_dict, fiducial_values_dict,
                                               params_tofix_dict)
            FM_GO_Ellcuts, _, _ = mm.mask_FM(FM_GO_Ellcuts, param_names_dict, fiducial_values_dict, params_tofix_dict)
            FM_GS_Ellcuts, _, _ = mm.mask_FM(FM_GS_Ellcuts, param_names_dict, fiducial_values_dict, params_tofix_dict)

            wzwa_idx = [param_names_list.index('wz'), param_names_list.index('wa')]

            assert FM_GO_noEllcuts.shape[0] == len(param_names_list) == len(fiducials_list), \
                'the number of rows should be equal to the number of parameters and fiducials'

            FMs = [FM_GO_noEllcuts, FM_GO_Ellcuts, FM_GS_noEllcuts, FM_GS_Ellcuts, FM_GS_kcuts_vinc]
            cases = ['FM_GO_noEllcuts', 'FM_GO_Ellcuts', 'FM_GS_noEllcuts', 'FM_GS_Ellcuts', 'FM_GS_kcuts_vinc']
            # cases += ['abs(percent_diff)']

            # ! priors on shear bias
            if shear_bias_priors:
                shear_bias_1st_idx = param_names_list.index('m01')
                shear_bias_last_idx = param_names_list.index(f'm{zbins:02}')
                prior = np.zeros(FM_GO_Ellcuts.shape)
                for i in range(shear_bias_1st_idx, shear_bias_last_idx + 1):
                    prior[i, i] = 5e-4 ** -2
                FMs = [FM + prior for FM in FMs]

            key_to_compare_A, key_to_compare_B = cases[0], cases[1]  # which cases to take the percent diff and ratio of

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

            # # store uncertainties in dictionaries to easily retrieve them in the different cases
            # uncert_G_dict[probe][ML][ZL][MS][ZS] = uncert_dict['G']
            # uncert_GS_dict[probe][ML][ZL][MS][ZS] = uncert_dict['GS']
            # uncert_ratio_dict[probe][ML][ZL][MS][ZS] = uncert_dict['ratio']
            # # append the FoM values at the end of the array
            # uncert_ratio_dict[probe][ML][ZL][MS][ZS] = np.append(
            #     uncert_ratio_dict[probe][ML][ZL][MS][ZS], fom_dict['GS'] / fom_dict['G'])

            data = np.asarray(data)
            param_names_label = param_names_list[:nparams_toplot]

            if plot_fom:
                fom_array = np.array([fom_dict[key_to_compare_A], fom_dict[key_to_compare_B],
                                      np.abs(mm.percent_diff(fom_dict[key_to_compare_A], fom_dict[key_to_compare_B]))])

                if divide_fom_by_10:
                    fom_array[0] /= 10
                    fom_array[1] /= 10
                    param_names_label += ['FoM/10']
                else:
                    param_names_label += ['FoM']

                nparams_toplot += 1
                data = np.column_stack((data, fom_array))

            print('kmax, fom_dict[key_to_compare_B]:', kmax_h_over_Mpc, fom_dict[key_to_compare_B])

            fom_list = [probe, ML, ZL, MS, ZS, kmax_h_over_Mpc, kmax_h_over_Mpc / h, fom_dict[cases[0]],
                        fom_dict[cases[1]],
                        fom_dict[cases[2]], fom_dict[cases[3]], fom_dict[cases[4]]]
            fom_df = fom_df.append(
                pd.DataFrame([fom_list], columns=['probe', 'ML', 'ZL', 'MS', 'ZS',
                                                  'kmax_h_over_Mpc', 'kmax_1_over_Mpc',
                                                  'FM_GO_noEllcuts',
                                                  'FM_GO_Ellcuts',
                                                  'FM_GS_noEllcuts',
                                                  'FM_GS_Ellcuts',
                                                  'FM_GS_kcuts_vinc']), ignore_index=True)

            title = '%s, $\\ k_{\\rm max}[h/Mpc] = %.2f$, zbins %s%i$' % (probe, kmax_h_over_Mpc, EP_or_ED, zbins)
            title += f'\nML = {ML / 10}, MS = {MS / 10}, ZL = {ZL / 10}, ZS = {ZS / 10}, zmax = {zmax / 10}'

            # plot_utils.bar_plot(data[:, :nparams_toplot], title, cases, nparams=nparams_toplot,
            #                     param_names_label=param_names_label, bar_width=0.15)

            # plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
            #                        f'bar_plot_{probe}_ellmax{lmax}_zbins{EP_or_ED}{zbins:02}'
            #                        f'_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.png')
            k_counter += 1

fom_df_zmin00 = fom_df.loc[fom_df['ZL'] == 0]
fom_df_zmin02 = fom_df.loc[fom_df['ZL'] == 2]

title = '%s (no GCsp), zbins %s%i, BNT transform' \
        f'\nML = {ML / 10}, MS = {MS / 10}, zmin = 0, zmax = {zmax / 10}' \
        '\nprior on $\\sigma(m) = 5 \\times 10^{-4}$' \
        '\n ${\\rm dzWL, dzGCph}$ fixed' % (probe, EP_or_ED, zbins)


def plot_from_dataframe(fom_df, key_1, key_2, key_3, key_4, title, constant_fom_idx, save):

    ellmax3000_TeX = '$\\ell_{\\rm max} = 3000 \\; \\forall z_i, z_j$'
    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.plot(fom_df['kmax_1_over_Mpc'], fom_df[key_1], label='FoM G', marker='o', c='tab:blue')
    plt.plot(fom_df['kmax_1_over_Mpc'], fom_df[key_2], label='FoM GS', marker='o', c='tab:orange')
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
                               f'ML{ML:03d}_MS{MS:03d}_zmin0.png')

title = '%s (no GCsp), zbins %s%i, BNT transform ' \
        f'\nML = {ML / 10}, MS = {MS / 10}, zmin = 0.2, zmax = {zmax / 10}' \
        '\nprior on $\\sigma(m) = 5 \\times 10^{-4}$' \
        '\n ${\\rm dzWL, dzGCph}$ fixed' % (probe, EP_or_ED, zbins)
plt.figure(figsize=(12, 10))
plt.title(title)
plt.plot(fom_df_zmin02['kmax_h_over_Mpc'] / h, fom_df_zmin02['FM_GO_Ellcuts'], label='FoM G', marker='o', ls='-',
         c='tab:blue')
plt.plot(fom_df_zmin02['kmax_h_over_Mpc'] / h, fom_df_zmin02['FM_GS_Ellcuts'], label='FoM GS', marker='o', ls='-',
         c='tab:orange')
plt.axhline(fom_df_zmin02['FM_GO_noEllcuts'][12], label='FoM G, $\\ell_{\\rm max} = 3000 \\; \\forall z_i, z_j$',
            ls='--', color='tab:blue')
plt.axhline(fom_df_zmin02['FM_GS_noEllcuts'][12], label='FoM GS, $\\ell_{\\rm max} = 3000 \\; \\forall z_i, z_j$',
            ls='--', color='tab:orange')

plt.xlabel("$k_{\\rm max}[1/Mpc]$")
plt.ylabel("FoM")
plt.ylim(0, 340)
plt.legend()
plt.grid()
plt.show()
plt.tight_layout()
plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
                       f'FoM_vs_kmax_{probe}_zbins{EP_or_ED}{zbins:02}'
                       f'ML{ML:03d}_MS{MS:03d}_zmin0.2.png')

# vincenzo's cuts
title = '%s (no GCsp), zbins %s%i, BNT transform ' \
        f'\nML = {ML / 10}, MS = {MS / 10}, zmin = 0, zmax = {zmax / 10}' \
        '\nprior on $\\sigma(m) = 5 \\times 10^{-4}$' \
        '\n ${\\rm dzWL, dzGCph}$ fixed' % (probe, EP_or_ED, zbins)
plt.figure(figsize=(12, 10))
plt.title(title)
plt.plot(fom_df_zmin00['kmax_h_over_Mpc'] / h, fom_df_zmin00['FM_GS_Ellcuts'], label='FoM GS, BNT cuts', marker='o',
         c='tab:blue')
plt.plot(fom_df_zmin00['kmax_h_over_Mpc'] / h, fom_df_zmin00['FM_GS_kcuts_vinc'], label='FoM GS, k cuts', marker='o',
         c='tab:orange')
plt.xlabel("$k_{\\rm max}[1/Mpc]$")
plt.ylabel("FoM")
plt.legend()
plt.grid()
plt.show()
plt.tight_layout()

plt.savefig(job_path / f'output/Flagship_{flagship_version}/plots/'
                       f'FoM_vs_kmax_{probe}_zbins{EP_or_ED}{zbins:02}'
                       f'ML{ML:03d}_MS{MS:03d}_zmin0_kcuts_vincenzo.png')
