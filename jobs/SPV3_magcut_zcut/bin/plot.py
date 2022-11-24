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

# plot config
matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')
markersize = 10

########################################################################################################################

# ! options
zbins_list = np.array((10,), dtype=int)
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
fix_shear_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
fix_dz_nuisance = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
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
zbins = 13
EP_or_ED = 'ED'
# ! end options


job_path = project_path / f'jobs/{which_job}'
uncert_ratio_dict = {}

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
        for EP_or_ED in EP_or_ED_list:

            lmax = 3000
            nbl = 29
            if probe == 'WL':
                lmax = 5000
                nbl = 32

            FM_path = f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output/Flagship_{flagship_version}/BNT_{BNT_transform}/FM'
            FM_GO_filename = f'FM_{probe}_GO_lmax{lmax}_nbl{nbl}_zbins{EP_or_ED:s}{zbins:02d}' \
                          f'-ML{magcut_lens:d}-ZL{zcut_lens:02d}-MS{magcut_source:d}-ZS{zcut_source:02d}.txt'
            FM_GS_filename = f'FM_{probe}_GS_lmax{lmax}_nbl{nbl}_zbins{EP_or_ED:s}{zbins:02d}' \
                          f'-ML{magcut_lens:d}-ZL{zcut_lens:02d}-MS{magcut_source:d}-ZS{zcut_source:02d}.txt'
            FM_GO = np.genfromtxt(f'{FM_path}/{FM_GO_filename}')
            FM_GS = np.genfromtxt(f'{FM_path}/{FM_GS_filename}')

            # fiducial values
            fid_cosmo = np.asarray([ISTF_fid.primary[key] for key in ISTF_fid.primary.keys()])[:7]
            fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
            ng_folder = f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/InputNz/magcut_zcut'
            ng_filename = f'ngbsTab-{EP_or_ED:s}{zbins:02d}-zedMin{zcut_source:02d}-zedMax{zmax:02d}-mag{magcut_source:03d}.dat'
            fid_galaxy_bias = np.genfromtxt(f'{ng_folder}/{ng_filename}')[:, 2]
            fid_shear_bias = np.zeros((zbins,))

            # some checks
            assert which_diff in ['normal', 'mean'], 'which_diff should be "normal" or "mean"'
            assert which_uncertainty in ['marginal',
                                         'conditional'], 'which_uncertainty should be "marginal" or "conditional"'
            assert which_Rl in ['const', 'var'], 'which_Rl should be "const" or "var"'
            assert model in ['flat', 'nonflat'], 'model should be "flat" or "nonflat"'
            assert probe in ['WL', 'GC', '3x2pt'], 'probe should be "WL" or "GC" or "3x2pt"'
            assert which_job == 'SPV3', 'which_job should be "SPV3"'

            if bar_plot_nuisance:

                if fix_shear_bias:
                    if probe == '3x2pt':
                        nparams_chosen = len(fid_cosmo) + len(fid_IA) + len(fid_galaxy_bias)
                    elif probe == 'GC':
                        nparams_chosen = len(fid_cosmo) + len(fid_galaxy_bias)
                    elif probe == 'WL':
                        nparams_chosen = len(fid_cosmo) + len(fid_IA)

                elif not fix_shear_bias:
                    if probe == '3x2pt':
                        nparams_chosen = len(fid_cosmo) + len(fid_IA) + len(fid_galaxy_bias) + len(fid_shear_bias)
                    elif probe == 'GC':
                        nparams_chosen = len(fid_cosmo) + len(fid_galaxy_bias)
                    elif probe == 'WL':
                        nparams_chosen = len(fid_cosmo) + len(fid_IA) + len(fid_shear_bias)

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
            print(FM_GO.shape)
            # remove rows/cols for the redshift center nuisance parameters
            if fix_dz_nuisance:
                FM_GO = FM_GO[:-zbins, :-zbins]
                FM_GS = FM_GS[:-zbins, :-zbins]

            print('2', FM_GO.shape)
            if probe != 'GC':
                if fix_shear_bias:
                    assert fix_dz_nuisance, 'the case with free dz_nuisance is not implemented (but it\'s easy; you just need to be more careful with the slicing)'
                    FM_GO = FM_GO[:-zbins, :-zbins]
                    FM_GS = FM_GS[:-zbins, :-zbins]

            print('3', FM_GO.shape)
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

            print(FM_GO.shape)
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
                data = np.asarray(data)
                plot_utils.bar_plot(data, title, cases, nparams=nparams, param_names_label=pars_labels_TeX,
                                    bar_width=0.12,
                                    second_axis=True, no_second_axis_bars=1)

            plt.savefig(job_path / f'output/plots/{which_comparison}/'
                                   f'bar_plot_{probe}_ellmax{ell_max}_zbins{EP_or_ED}{zbins:02}_Rl{which_Rl}_{which_uncertainty}.png')


