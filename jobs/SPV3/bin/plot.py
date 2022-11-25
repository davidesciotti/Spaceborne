import sys
import time
import warnings
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
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from scipy.interpolate import interp2d
from xarray.plot.utils import legend_elements

project_path = Path.cwd().parent.parent.parent

sys.path.append(str(project_path.parent / 'common_data'))
import common_lib.my_module as mm
import common_config.mpl_cfg as mpl_cfg
import common_config.ISTF_fid_params as ISTF_fid

sys.path.append(str(project_path / 'bin'))
import plots_FM_running as plot_utils

# plot config
matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')
markersize = 10

########################################################################################################################

# ! options
zbins_list = np.array((10,), dtype=int)
probes = ('3x2pt',)
pes_opt_list = ('opt',)
EP_or_ED_list = ('EP',)
which_comparison = 'GO_vs_GS'  # this is just to set the title of the plot
which_Rl = 'var'
nparams_chosen = 7
which_job = 'SPV3'
model = 'flat'
which_diff = 'normal'
flagship_version = 1
check_old_FM = False
which_uncertainty = 'marginal'
fix_shear_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_dz_nuisance = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
w0wa_rows = [2, 3]
bar_plot_cosmo = True
triangle_plot = False
plot_ratio_vs_zbins = False
plot_fom_vs_zbins = False
plot_fom_vs_eps_b = False
plot_prior_contours = False
bar_plot_nuisance = False
plot_response = False
plot_ISTF_kernels = False
pic_format = 'pdf'
dpi = 500
# ! end options


job_path = project_path / f'jobs/{which_job}'
uncert_ratio_dict = {}

# TODO fix this
if which_job == 'SPV3':
    nbl = 32
else:
    raise ValueError

# fiducial values
fid_cosmo = np.asarray([ISTF_fid.primary[key] for key in ISTF_fid.primary.keys()])[:7]
fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
fid_galaxy_bias = np.asarray([ISTF_fid.photoz_galaxy_bias[key] for key in ISTF_fid.photoz_galaxy_bias.keys()])
fid_shear_bias = np.asarray([ISTF_fid.photoz_shear_bias[key] for key in ISTF_fid.photoz_shear_bias.keys()])

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

# initialize dict lists
for probe in probes:
    uncert_ratio_dict[probe] = {}
    for zbins in zbins_list:
        uncert_ratio_dict[probe][f'zbins{zbins:02}'] = {}
        for EP_or_ED in ('EP', 'ED'):
            uncert_ratio_dict[probe][f'zbins{zbins:02}'][EP_or_ED] = {}
            for pes_opt in ('opt', 'pes'):
                uncert_ratio_dict[probe][f'zbins{zbins:02}'][EP_or_ED][pes_opt] = []

for probe in probes:
    for zbins in zbins_list:
        for pes_opt in pes_opt_list:
            for EP_or_ED in EP_or_ED_list:

                # some checks
                assert which_diff in ['normal', 'mean'], 'which_diff should be "normal" or "mean"'
                assert which_uncertainty in ['marginal',
                                             'conditional'], 'which_uncertainty should be "marginal" or "conditional"'
                assert which_Rl in ['const', 'var'], 'which_Rl should be "const" or "var"'
                assert model in ['flat', 'nonflat'], 'model should be "flat" or "nonflat"'
                assert probe in ['WL', 'GC', '3x2pt'], 'probe should be "WL" or "GC" or "3x2pt"'
                assert pes_opt in ['opt', 'pes'], 'pes_opt should be "opt" or "pes"'
                assert which_job == 'SPV3', 'which_job should be "SPV3"'

                if bar_plot_nuisance:  # ! fix this
                    assert zbins == 10, 'I have not generalized the numbers below, plus, the gal bias fiducials ' \
                                        'are not defined for zbins != 10'
                    if fix_shear_bias:
                        if probe == '3x2pt':
                            nparams_chosen = 20
                        elif probe == 'GC':
                            nparams_chosen = 17
                        elif probe == 'WL':
                            nparams_chosen = 10
                    elif not fix_shear_bias:
                        if probe == '3x2pt':
                            nparams_chosen = 30
                        elif probe == 'GC':
                            nparams_chosen = 17
                        elif probe == 'WL':
                            nparams_chosen = 20

                nparams = nparams_chosen  # re-initialize at every iteration

                specs = f'NonFlat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-{EP_or_ED}{zbins:02}'

                if pes_opt == 'opt':
                    ell_max_WL = 5000
                    ell_max_GC = 3000
                else:
                    ell_max_WL = 1500
                    ell_max_GC = 750

                if probe == '3x2pt':
                    probe_lmax = 'XC'
                    probe_folder = 'All'
                    probename_vinc = probe
                    pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['IA_labels_TeX'] + \
                                      mpl_cfg.general_dict['galaxy_bias_labels_TeX'] + mpl_cfg.general_dict[
                                          'shear_bias_labels_TeX']
                    fid = np.concatenate((fid_cosmo, fid_IA, fid_galaxy_bias, fid_shear_bias), axis=0)
                else:
                    probe_lmax = probe
                    probe_folder = probe + 'O'
                    probename_vinc = probe + 'O'

                if probe == 'WL':
                    ell_max = ell_max_WL
                    pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['IA_labels_TeX'] + \
                                      mpl_cfg.general_dict['shear_bias_labels_TeX']
                    fid = np.concatenate((fid_cosmo, fid_IA, fid_shear_bias), axis=0)
                else:
                    ell_max = ell_max_GC

                if probe == 'GC':
                    pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict[
                        'galaxy_bias_labels_TeX']
                    fid = np.concatenate((fid_cosmo, fid_galaxy_bias), axis=0)

                title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, ell_max, EP_or_ED, zbins)

                # TODO try with pandas dataframes
                # todo non-tex labels

                # import vincenzo's FM, not in a dictionary because they are all split into different folders
                vinc_FM_folder = f'vincenzo/SPV3_07_2022/Flagship_{flagship_version}/FishMat'
                if pes_opt == 'opt':
                    FM_GO = np.genfromtxt(
                        project_path.parent / f'common_data/{vinc_FM_folder}/GaussOnly/{probe_folder}/'
                                              f'OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')
                    FM_GS = np.genfromtxt(project_path.parent / f'common_data/{vinc_FM_folder}/GaussSSC/{probe_folder}/'
                                                                f'OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')
                elif pes_opt == 'pes':
                    FM_GO = np.genfromtxt(
                        project_path.parent / f'common_data/{vinc_FM_folder}/GaussOnly/Cuts/SetupId1{zbins:02}1100034/'
                                              f'fm-{probename_vinc}-{nbl}-wzwaCDM-NonFlat-GR-TB-Pess.dat')
                    FM_GS = np.genfromtxt(
                        project_path.parent / f'common_data/{vinc_FM_folder}/GaussSSC/Cuts/SetupId1{zbins:02}1100034/'
                                              f'fm-{probename_vinc}-{nbl}-wzwaCDM-NonFlat-GR-TB-Pess.dat')
                else:
                    raise ValueError('pes_opt should be "opt" or "pes"')

                # remove rows/cols for the redshift center nuisance parameters
                if fix_dz_nuisance:
                    FM_GO = FM_GO[:-zbins, :-zbins]
                    FM_GS = FM_GS[:-zbins, :-zbins]

                if probe != 'GC':
                    if fix_shear_bias:
                        assert fix_dz_nuisance, 'the case with free dz_nuisance is not implemented (but it\'s easy; you just need to be more careful with the slicing)'
                        FM_GO = FM_GO[:-zbins, :-zbins]
                        FM_GS = FM_GS[:-zbins, :-zbins]

                if model == 'flat':
                    FM_GO = np.delete(FM_GO, obj=1, axis=0)
                    FM_GO = np.delete(FM_GO, obj=1, axis=1)
                    FM_GS = np.delete(FM_GS, obj=1, axis=0)
                    FM_GS = np.delete(FM_GS, obj=1, axis=1)
                    cosmo_params = 7
                elif model == 'nonflat':
                    w0wa_rows = [3, 4]  # Omega_DE is in position 1, so w0, wa are shifted by 1 position
                    nparams += 1
                    cosmo_params = 8
                    fid = np.insert(arr=fid, obj=1, values=ISTF_fid.extensions['Om_Lambda0'], axis=0)
                    pars_labels_TeX = np.insert(arr=pars_labels_TeX, obj=1, values='$\\Omega_{\\rm DE, 0}$', axis=0)

                fid = fid[:nparams]
                pars_labels_TeX = pars_labels_TeX[:nparams]

                # remove null rows and columns
                idx = mm.find_null_rows_cols_2D(FM_GO)
                idx_GS = mm.find_null_rows_cols_2D(FM_GS)
                assert np.array_equal(idx, idx_GS), 'the null rows/cols indices should be equal for GO and GS'
                FM_GO = mm.remove_null_rows_cols_2D(FM_GO, idx)
                FM_GS = mm.remove_null_rows_cols_2D(FM_GS, idx)

                ####################################################################################################################

                # TODO plot FoM ratio vs # of bins (I don't have the ED FMs!)

                # old FMs (before specs updates)
                if check_old_FM:
                    FM_GO_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/'
                                              f'SSC_comparison/output/FM/FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl30.txt')
                    FM_GS_old = np.genfromtxt(f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/'
                                              f'SSC_comparison/output/FM/FM_{probe}_GS_lmax{probe_lmax}{ell_max}_nbl30_Rlvar.txt')
                    cases = ('GO_old', 'GO_new', 'GS_old', 'GS_new')
                    FMs = (FM_GO_old, FM_GO, FM_GS_old, FM_GS)
                else:
                    cases = ('G', 'GS')
                    FMs = (FM_GO, FM_GS)

                data = []
                fom = {}
                uncert = {}
                for FM, case in zip(FMs, cases):
                    uncert[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams, fiducials=fid,
                                                                  which_uncertainty=which_uncertainty, normalize=True))
                    fom[case] = mm.compute_FoM(FM, w0wa_rows=w0wa_rows)
                    print(f'FoM({probe}, {case}): {fom[case]}')

                # set uncertainties to 0 (or 1? see code) for \Omega_DE in the non-flat case, where Ode was not a free parameter
                if model == 'nonflat' and check_old_FM:
                    for case in ('GO_old', 'GS_old'):
                        uncert[case] = np.insert(arr=uncert[case], obj=1, values=1, axis=0)
                        uncert[case] = uncert[case][:nparams]

                if check_old_FM:
                    uncert['diff_old'] = diff_funct(uncert['GS_old'], uncert['GO_old'])
                    uncert['diff_new'] = diff_funct(uncert['GS_new'], uncert['GO_new'])
                    uncert['ratio_old'] = uncert['GS_old'] / uncert['GO_old']
                    uncert['ratio_new'] = uncert['GS_new'] / uncert['GO_new']
                else:
                    uncert['percent_diff'] = diff_funct(uncert['GS'], uncert['G'])
                    uncert['ratio'] = uncert['GS'] / uncert['G']

                uncert_vinc = {
                    'zbins_EP10': {
                        'flat': {
                            'WL_pes': np.asarray([1.998, 1.001, 1.471, 1.069, 1.052, 1.003, 1.610]),
                            'WL_opt': np.asarray([1.574, 1.013, 1.242, 1.035, 1.064, 1.001, 1.280]),
                            'GC_pes': np.asarray([1.002, 1.002, 1.003, 1.003, 1.001, 1.001, 1.001]),
                            'GC_opt': np.asarray([1.069, 1.016, 1.147, 1.096, 1.004, 1.028, 1.226]),
                            '3x2pt_pes': np.asarray([1.442, 1.034, 1.378, 1.207, 1.028, 1.009, 1.273]),
                            '3x2pt_opt': np.asarray([1.369, 1.004, 1.226, 1.205, 1.018, 1.030, 1.242]),
                        },
                        'nonflat': {
                            'WL_pes': np.asarray([2.561, 1.358, 1.013, 1.940, 1.422, 1.064, 1.021, 1.433]),
                            'WL_opt': np.asarray([2.113, 1.362, 1.004, 1.583, 1.299, 1.109, 1.038, 1.559]),
                            'GC_pes': np.asarray([1.002, 1.001, 1.002, 1.002, 1.003, 1.001, 1.000, 1.001]),
                            'GC_opt': np.asarray([1.013, 1.020, 1.006, 1.153, 1.089, 1.004, 1.039, 1.063]),
                            '3x2pt_pes': np.asarray([1.360, 1.087, 1.043, 1.408, 1.179, 1.021, 1.009, 1.040]),
                            '3x2pt_opt': np.asarray([1.572, 1.206, 1.013, 1.282, 1.191, 1.013, 1.008, 1.156]),
                        },
                        'nonflat_shearbias': {
                            'WL_pes': np.asarray([1.082, 1.049, 1.000, 1.057, 1.084, 1.034, 1.025, 1.003]),
                            'WL_opt': np.asarray([1.110, 1.002, 1.026, 1.022, 1.023, 1.175, 1.129, 1.009]),
                            '3x2pt_pes': np.asarray([1.297, 1.087, 1.060, 1.418, 1.196, 1.021, 1.030, 1.035]),
                            '3x2pt_opt': np.asarray([1.222, 1.136, 1.010, 1.300, 1.206, 1.013, 1.009, 1.164]),
                        },
                    }
                }

                # print my and vincenzo's uncertainties and check that they are sufficiently close
                if zbins == 10 and EP_or_ED == 'EP':
                    with np.printoptions(precision=3, suppress=True):
                        print(f'ratio GS/GO, probe: {probe}')
                        print('dav:', uncert["ratio"])
                        print('vin:', uncert_vinc[f'zbins_{EP_or_ED}{zbins:02}'][model][f"{probe}_{pes_opt}"])

                model_here = model
                if not fix_shear_bias:
                    model_here += '_shearbias'
                if zbins == 10 and EP_or_ED == 'EP' and model_here != 'flat_shearbias' and which_uncertainty == 'marginal':
                    # the tables in the paper, from which these uncertainties have been taken, only include the cosmo params (7 or 8)
                    nparams_vinc = uncert_vinc[f'zbins_{EP_or_ED}{zbins:02}'][model_here][f"{probe}_{pes_opt}"].shape[0]
                    assert np.allclose(uncert["ratio"][:nparams_vinc],
                                       uncert_vinc[f'zbins_{EP_or_ED}{zbins:02}'][model_here][f"{probe}_{pes_opt}"],
                                       atol=0,
                                       rtol=1e-2), 'my uncertainties differ from vincenzos'

                if check_old_FM:
                    cases = ['GO_old', 'GO_new', 'GS_old', 'GS_new', 'diff_old', 'diff_new']
                else:
                    cases = ['G', 'GS', 'percent_diff']

                for case in cases:
                    data.append(uncert[case])

                # store uncertainties in dictionaries to easily retrieve them in the different cases
                uncert_ratio_dict[f'{probe}'][f'zbins{zbins:02}'][EP_or_ED][pes_opt] = uncert['ratio']
                # append the FoM values at the end of the array
                uncert_ratio_dict[f'{probe}'][f'zbins{zbins:02}'][EP_or_ED][pes_opt] = np.append(
                    uncert_ratio_dict[f'{probe}'][f'zbins{zbins:02}'][EP_or_ED][pes_opt], fom['GS'] / fom['G'])

if bar_plot_cosmo:

    for probe in probes:
        for zbins in zbins_list:
            for pes_opt in ('opt', 'pes'):
                plot_utils.bar_plot(data, title, cases, nparams=nparams, param_names_label=pars_labels_TeX,
                                    bar_width=0.12,
                                    second_axis=False, no_second_axis_bars=1)

            plt.savefig(job_path / f'output/plots/{which_comparison}/'
                                   f'bar_plot_{probe}_ellmax{ell_max}_zbins{EP_or_ED}{zbins:02}_Rl{which_Rl}_{which_uncertainty}.png')

if probe == '3x2pt' and triangle_plot:
    plot_utils.triangle_plot(FM_GO, FM_GS, fiducials=fid,
                             title=title, param_names_label=pars_labels_TeX)

if plot_ratio_vs_zbins:

    fontsize = 15

    params = {'lines.linewidth': 2,
              'font.size': fontsize,
              'axes.labelsize': 'small',
              'axes.titlesize': 'small',
              'xtick.labelsize': 'small',
              'ytick.labelsize': 'small',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              }
    plt.rcParams.update(params)
    markersize = 5.5

    rows, cols = 3, 3
    # fig, axs = plt.subplots(rows, cols, sharex=True, subplot_kw=dict(box_aspect=0.75),
    #                         constrained_layout=False, figsize=(15, 6.5), tight_layout={'pad': 0.7})
    fig, axs = plt.subplots(rows, cols, sharex=True, subplot_kw=dict(box_aspect=0.65),
                            constrained_layout=False, figsize=(12, 8), tight_layout={'pad': 0.9})

    # number each axs box: 0 for [0, 0], 1 for [0, 1] and so forth
    axs_idx = np.arange(0, rows * cols, 1).reshape((rows, cols))
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
    linestyle = 'dashed'
    fmt = '%.2f'
    panel_titles_fontsize = fontsize
    pars_labels_TeX = np.append(pars_labels_TeX, '$\\rm{FoM}$')

    # loop over 7 cosmo params + FoM; this iterates over the different plots (boxes)
    for param_idx in range(len(pars_labels_TeX)):
        # take i, j and their "counter" (the param index)
        i, j = np.where(axs_idx == param_idx)[0][0], np.where(axs_idx == param_idx)[1][0]

        # loop over cases within the same plot (box)
        color_idx = 0
        for pes_opt in ('pes', 'opt'):
            for probe in ('WL', '3x2pt'):
                # set xticks on int values
                axs[i, j].xaxis.set_ticks(zbins_list)
                axs[i, j].xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))

                # list to plot on the y-axis
                uncert_ratio_vs_zbins = [uncert_ratio_dict[f'{probe}'][f'zbins{zbins:02}'][EP_or_ED][pes_opt][param_idx]
                                         for zbins in
                                         zbins_list]

                axs[i, j].plot(zbins_list, uncert_ratio_vs_zbins,
                               ls=linestyle, markersize=markersize, marker='o', color=colors[color_idx],
                               label=f'{probe} {pes_opt}')
                axs[i, j].yaxis.set_major_formatter(FormatStrFormatter(f'{fmt}'))

                color_idx += 1

        axs[i, j].grid()
        axs[i, j].set_title(f'{pars_labels_TeX[param_idx]}', pad=10.0, fontsize=panel_titles_fontsize)

    # legend in the bottom:
    # get labels and handles from one of the axis (the first)
    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=[0.531, 1.10],
               bbox_transform=fig.transFigure)
    # fig.subplots_adjust(top=0.1)  # or whatever

    fig.supxlabel('${\\cal N}_b$', x=0.535, y=0.03, fontsize=fontsize)
    fig.supylabel('${\\cal R}(x) = \\sigma_{\\rm GS}(x) \, / \, \\sigma_{\\rm G}(x)$', x=0.03, fontsize=fontsize)

    plt.savefig(job_path / f'output/plots/replot_vincenzo_newspecs/GS_G_ratio_vs_zbins.{pic_format}', dpi=dpi,
                bbox_inches='tight')

if plot_fom_vs_zbins:

    fontsize = 23
    params = {'lines.linewidth': 5,
              'font.size': fontsize,
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'xx-large',
              'ytick.labelsize': 'xx-large',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              }
    plt.rcParams.update(params)
    markersize = 14

    tick_spacing = 0.01

    for probe in ('WL', '3x2pt'):
        plt.figure()
        param_idx = -1  # FoM is at the last place
        pes_opt = 'opt'
        for EP_or_ED in ['EP', 'ED']:
            uncert_ratio_vs_zbins = [uncert_ratio_dict[f'{probe}'][f'zbins{zbins:02}'][EP_or_ED][pes_opt][param_idx] for
                                     zbins in
                                     zbins_list]
            plt.plot(zbins_list, uncert_ratio_vs_zbins, '--', marker='o', label=EP_or_ED, markersize=markersize)

        plt.grid()
        plt.legend(loc='lower right', prop={'size': fontsize})
        plt.ylabel('${\\cal R}({\\rm FoM})$')
        plt.xlabel('${\\cal N}_b$')
        plt.gca().yaxis.set_major_formatter('{x:.2f}')  # 2 significant digits on y axis
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))  # set tick step to 0.01

        plt.savefig(
            job_path / f'output/plots/replot_vincenzo_newspecs/FoM_vs_EPED/FoM_vs_EP-ED_zbins_{probe}.{pic_format}',
            dpi=dpi, bbox_inches='tight')

if plot_fom_vs_eps_b:

    fontsize = 23
    params = {'lines.linewidth': 4,
              'font.size': fontsize,
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'xx-large',
              'ytick.labelsize': 'xx-large',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              }
    plt.rcParams.update(params)
    markersize = 14

    FoM_vs_prior = np.genfromtxt(
        f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/FoMvsPrior/fomvsprior-EP10-Opt.dat')
    FoM_vs_prior[:, 0] = 10 ** FoM_vs_prior[:, 0]  # eps_b
    FoM_vs_prior[:, 1] = 10 ** FoM_vs_prior[:, 1]  # sigma_m

    # find the correct line fot the different sigma_m values, Vincenzo flattens the array
    sigma_m_fixed = (5e-4, 50e-4, 100e-4)
    sigma_m_fixed = (5e-4, 50e-4, 500e-4)
    start_idxs = [np.argmin(np.abs(FoM_vs_prior[:, 1] - sigma_m_value)) for sigma_m_value in sigma_m_fixed]

    eps_b_values = np.unique(FoM_vs_prior[:, 0])
    sigma_m_values = np.unique(FoM_vs_prior[:, 1])
    n_points = eps_b_values.size

    if np.any(max(sigma_m_fixed) > FoM_vs_prior[:, 1]):
        print('extrapolating FoM_vs_eps_b')

        X = eps_b_values
        Y = sigma_m_values
        Z = np.reshape(FoM_vs_prior[:, -3], (n_points, n_points)).T
        FoM_G_extrap = interp2d(x=X, y=Y, z=Z, kind='linear', fill_value='extrapolate')
        FoM_G_extrap_array = FoM_G_extrap(x=eps_b_values, y=sigma_m_fixed).T

        Z = np.reshape(FoM_vs_prior[:, -2], (n_points, n_points)).T
        FoM_GS_extrap = interp2d(x=X, y=Y, z=Z, kind='linear', fill_value='extrapolate')
        FoM_GS_extrap_array = FoM_GS_extrap(x=eps_b_values, y=sigma_m_fixed).T

        Z = np.reshape(FoM_vs_prior[:, -1], (n_points, n_points)).T
        FoM_ratio_extrap = interp2d(x=X, y=Y, z=Z, kind='linear', fill_value='extrapolate')
        FoM_ratio_extrap_array = FoM_ratio_extrap(x=eps_b_values, y=sigma_m_fixed).T

    step = int(np.shape(FoM_vs_prior[:, 0])[0] / np.shape(eps_b_values)[0])  # ratio between total and unique elements
    fsky_correction = 14700 / 14000  # survey area has been changed again

    linestyles = ('solid', 'dashed', 'dotted')
    linestyle_labels = (
        '$\\sigma_m = %i \\times 10^{−4}$' % int(sigma_m_fixed[0] * 1e4),
        '$\\sigma_m = %i \\times 10^{−4}$' % int(sigma_m_fixed[1] * 1e4),
        '$\\sigma_m = %i \\times 10^{−4}$' % int(sigma_m_fixed[2] * 1e4))
    color_labels = ('G', 'GS')

    # without extrapolation
    plt.figure()
    for start, ls, label in zip(start_idxs, linestyles, linestyle_labels):
        plt.plot(FoM_vs_prior[start::step, 0], FoM_vs_prior[start::step, -3] * fsky_correction, color='tab:blue',
                 ls='-')  # ! change back to ls=ls
        plt.plot(FoM_vs_prior[start::step, 0], FoM_vs_prior[start::step, -2] * fsky_correction, color='tab:orange',
                 ls=ls)

    # with extrapolation
    for start, start_2, ls, label in zip(range(3), start_idxs, linestyles, linestyle_labels):
        print('FoM_vs_prior == FoM_G_extrap_array?',
              np.array_equal(FoM_G_extrap_array[:, start], FoM_vs_prior[start_2::step, -3]))

        plt.plot(eps_b_values, FoM_G_extrap_array[:, start] * fsky_correction, ls=ls, color='g')
        # plt.plot(eps_b_values, FoM_G_extrap_array[:, start] / FoM_vs_prior[start_2::step, -3], ls=ls, color='g')
        # plt.plot(eps_b_values, FoM_GS_extrap_array[:, start] * fsky_correction, ls=ls, color='g')

    dummy_lines = []
    for i in range(len(sigma_m_fixed)):
        dummy_lines.append(plt.plot([], [], c="black", ls=linestyles[i])[0])

    dummy_colors = []
    for i in range(len(sigma_m_fixed)):
        dummy_colors.append(plt.plot([], [])[0])

    linestyles_legend = plt.legend(dummy_lines, linestyle_labels, prop={'size': fontsize})
    color_legend = plt.legend(dummy_colors, color_labels, bbox_to_anchor=[1, 0.8], prop={'size': fontsize})
    plt.gca().add_artist(color_legend)
    plt.gca().add_artist(linestyles_legend)

    plt.grid()
    plt.xscale('log')
    plt.xlabel('$\\epsilon_b (\\%)$')
    plt.ylabel('${\\rm FoM}$')
    plt.show()

    plt.savefig(job_path / f'output/plots/replot_vincenzo_newspecs/FoM_vs_epsb/FoM_vs_epsb.{pic_format}', dpi=dpi,
                bbox_inches='tight')

    plt.figure()
    for start, ls, label in zip(start_idxs, linestyles, linestyle_labels):
        plt.plot(FoM_vs_prior[start::step, 0], FoM_vs_prior[start::step, -2] / FoM_vs_prior[start::step, -3],
                 color='black',
                 ls=ls, label=label)

    plt.legend(prop={'size': fontsize})
    plt.grid()
    plt.xscale('log')
    plt.xlabel('$\\epsilon_b (\%)$')
    plt.ylabel('${\\cal R}({\\rm FoM})$')
    plt.show()

    plt.savefig(job_path / f'output/plots/replot_vincenzo_newspecs/FoM_vs_epsb/FoMratio_vs_epsb.{pic_format}', dpi=dpi,
                bbox_inches='tight')

if plot_prior_contours:
    fontsize = 20
    params = {'lines.linewidth': 3,
              'font.size': fontsize,
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              }
    plt.rcParams.update(params)
    markersize = 14

    FoM_vs_prior = np.genfromtxt(
        f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/FoMvsPrior/fomvsprior-EP10-Opt.dat')
    FoM_vs_prior[:, 0] = 10 ** FoM_vs_prior[:, 0]  # eps_b
    FoM_vs_prior[:, 1] = 10 ** FoM_vs_prior[:, 1]  # sigma_m

    # take the epsilon values
    epsb_values = np.unique(FoM_vs_prior[:, 0])
    sigmam_values = np.unique(FoM_vs_prior[:, 1])
    n_points = sigmam_values.size
    FoM_ref = 294.8  # from Vincenzo's email - this is for EP10, non-flat
    FoM_ratio_values = np.arange(0.8, 1.1, 0.05)

    # produce grid and pass Z values
    X = epsb_values
    Y = sigmam_values * 1e4
    X, Y = np.meshgrid(X, Y)
    Z = np.reshape(np.abs(FoM_vs_prior[:, -2] / FoM_ref),
                   (n_points, n_points)).T  # XXX careful of the FoM_vs_prior index!! it's FoM_GS/FoM_ref
    print('min =', Z.min())
    print('max =', Z.max())

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(19, 19))

    # levels of contour plot (set a high xorder to have line on top of legend)
    CS = axs.contour(X, Y, Z, levels=FoM_ratio_values, cmap='plasma', zorder=1)

    # plot adjustments
    axs.set_xlabel('$\\epsilon_b \, (\%)$')
    axs.set_ylabel('$\\sigma_m \\times 10^4$')

    # I want the plot to be square even with different limits on the x and y axes - probably not the smartest way to do it
    xlim_min, xlim_max = 0.5, 3.5
    step = 0.5
    axs.set_xlim(xlim_min, xlim_max)
    axs.set_ylim(0, 10)
    x_length = axs.get_xlim()[1] - axs.get_xlim()[0]
    y_length = axs.get_ylim()[1] - axs.get_ylim()[0]
    axs.set_aspect(x_length / y_length)

    axs.xaxis.set_ticks(np.arange(xlim_min, xlim_max, step))
    axs.grid()

    # legend: from source (see the comment): https://stackoverflow.com/questions/64523051/legend-is-empty-for-contour-plot-is-this-by-design
    handle, _ = CS.legend_elements()
    label = ['${\\rm FoM_{GS}} \, / \, {\\rm FoM}_{\\rm ref}}$ = ' + f'{a:.2f}' for a in CS.levels]
    axs.legend(handle, label, bbox_to_anchor=[1.01, 1.01], loc='upper right')

    plt.savefig(job_path / f'output/plots/replot_vincenzo_newspecs/epsb_sigmam_contour.{pic_format}', dpi=dpi,
                bbox_inches='tight')

if bar_plot_nuisance:
    data = uncert['diff'][cosmo_params:]

    zbins = 10
    case = 'opt'
    data = np.asarray(data)
    plot_utils.bar_plot(data, title, label_list=['% diff'], nparams=nparams - cosmo_params,
                        param_names_label=pars_labels_TeX[cosmo_params:],
                        bar_width=0.17,
                        second_axis=False, no_second_axis_bars=0)

if plot_response:
    fontsize = 20
    params = {'lines.linewidth': 3,
              'font.size': fontsize,
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              }
    plt.rcParams.update(params)
    markersize = 14

    ED_or_EP = 'EP'
    zbins = 10
    path = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_1/ResFunTabs'

    ell_WL = np.genfromtxt(
        f'{path}/480_ell_points_for_paper_plot/elllist-WLO-{ED_or_EP}{zbins}-HR.dat')
    ell_GC = np.genfromtxt(
        f'{path}/480_ell_points_for_paper_plot/elllist-GCO-{ED_or_EP}{zbins}-HR.dat')
    ell_3x2pt = np.genfromtxt(
        f'{path}/480_ell_points_for_paper_plot/elllist-3x2pt-{ED_or_EP}{zbins}-HR.dat')

    rf_WL = np.genfromtxt(
        f'{path}/480_ell_points_for_paper_plot/rf-WLO-Opt-{ED_or_EP}{zbins}-HR.dat')
    rf_GC = np.genfromtxt(
        f'{path}/480_ell_points_for_paper_plot/rf-GCO-Opt-{ED_or_EP}{zbins}-HR.dat')
    rf_3x2pt_1d = np.genfromtxt(
        f'{path}/480_ell_points_for_paper_plot/rf-3x2pt-Opt-{ED_or_EP}{zbins}-HR.dat')

    nbl_WL = ell_WL.shape[0]
    nbl_GC = ell_GC.shape[0]
    nbl_3x2pt = ell_3x2pt.shape[0]

    # reshape 3x2pt
    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
    rf_2d = np.reshape(rf_3x2pt_1d, (nbl_3x2pt, zpairs_3x2pt))

    # split into 3 2d datavectors
    rf_ll_3x2pt_2d = rf_2d[:, :zpairs_auto]
    rf_lg_3x2pt_2d = rf_2d[:, zpairs_auto:zpairs_auto + zpairs_cross]  # ! is it really gl? or lg?
    rf_gg_3x2pt_2d = rf_2d[:, zpairs_auto + zpairs_cross:]

    # reshape them individually - the symmetrization is done within the function
    rf_ll_3x2pt_3d = mm.cl_2D_to_3D_symmetric(rf_ll_3x2pt_2d, nbl=nbl_3x2pt, npairs=zpairs_auto, zbins=zbins)
    rf_lg_3x2pt_3d = mm.cl_2D_to_3D_asymmetric(rf_lg_3x2pt_2d, nbl=nbl_3x2pt, zbins=zbins)
    rf_gg_3x2pt_3d = mm.cl_2D_to_3D_symmetric(rf_gg_3x2pt_2d, nbl=nbl_3x2pt, npairs=zpairs_auto, zbins=zbins)
    warnings.warn('check the order of the 3x2pt response functions, the default is actually F-style!')

    # use them to populate the datavector
    rf_3x2pt = np.zeros((nbl_3x2pt, 2, 2, zbins, zbins))
    rf_3x2pt[:, 0, 0, :, :] = rf_ll_3x2pt_3d
    rf_3x2pt[:, 1, 1, :, :] = rf_gg_3x2pt_3d
    rf_3x2pt[:, 0, 1, :, :] = rf_lg_3x2pt_3d
    rf_3x2pt[:, 1, 0, :, :] = np.transpose(rf_lg_3x2pt_3d, (0, 2, 1))

    # reshape WL and GC
    rf_WL_3d = mm.cl_1D_to_3D(rf_WL, nbl=nbl_WL, zbins=zbins, is_symmetric=True)
    rf_GC_3d = mm.cl_1D_to_3D(rf_GC, nbl=nbl_GC, zbins=zbins, is_symmetric=True)
    rf_WL_3d = mm.fill_3D_symmetric_array(rf_WL_3d, nbl_WL, zbins)
    rf_GC_3d = mm.fill_3D_symmetric_array(rf_GC_3d, nbl_GC, zbins)

    # central z bin
    i, j = 4, 4

    plt.figure()
    plt.plot(ell_WL, rf_WL_3d[:, i, j], ls='-', label='WL')
    plt.plot(ell_GC, rf_GC_3d[:, i, j], ls='-', label='GCph')
    plt.plot(ell_3x2pt, rf_3x2pt[:, 1, 0, i, j], ls='-',
             label='XC')  # ! the paper uses R^{gm}, so it's GCph first and WL second

    plt.xscale('log')
    plt.xlabel('$\ell$')
    plt.ylabel('$R_{%i%i}^{AB}(\ell)$' % (i + 1, j + 1))
    plt.grid()
    plt.legend()
    plt.show()

    plt.savefig(job_path / f'output/plots/replot_vincenzo_newspecs/responses.{pic_format}', dpi=dpi,
                bbox_inches='tight')

if plot_ISTF_kernels:

    fontsize = 17
    params = {'lines.linewidth': 3,
              'font.size': fontsize,
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large',
              'mathtext.fontset': 'stix',
              'font.family': 'STIXGeneral',
              'legend.fontsize': 12.3,
              }
    plt.rcParams.update(params)

    wil = np.load(f'{project_path.parent}/cl_v2/output/WF/WFs_v16_eNLA_may22/wil_IA_IST_nz700.npy')
    wig = np.load(f'{project_path.parent}/cl_v2/output/WF/WFs_v16_eNLA_may22/wig_IST_nz700.npy')
    z_values = wig[:, 0]
    wil = wil[:, 1:]
    wig = wig[:, 1:]

    fig, ax = plt.subplots(1, 2, figsize=(28, 5))
    for zbin_idx in range(zbins):
        label = '$%.3f < z < %.3f$' % (
            ISTF_fid.zbin_edges['z_minus'][zbin_idx], ISTF_fid.zbin_edges['z_plus'][zbin_idx])
        ax[0].plot(z_values, wil[:, zbin_idx], label=label)
        ax[1].plot(z_values, wig[:, zbin_idx], label=label)

    ax[0].set_xlim(0, 3)
    ax[1].set_xlim(0, 3)
    ax[0].set_xlabel('$z$')
    ax[1].set_xlabel('$z$')
    ax[0].set_ylabel('$W^L_i(z) \; [{\\rm Mpc^{-1}}]$')
    ax[1].set_ylabel('$W^G_i(z) \; [{\\rm Mpc^{-1}}]$')
    ax[1].ticklabel_format(style='scientific', axis='y')
    formatter = ticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax[1].yaxis.set_major_formatter(formatter)
    ax[0].legend(loc='upper right', )
    ax[1].legend(loc='upper right', )
    ax[0].grid()
    ax[1].grid()

    plt.savefig(job_path / f'output/plots/replot_vincenzo_newspecs/ISTF_kernels_nobias.{pic_format}', dpi=dpi,
                bbox_inches='tight')

print('*********** done ***********')
