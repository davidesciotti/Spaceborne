import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from getdist import MCSamples, plots
from matplotlib.cm import get_cmap
from getdist.gaussian_mixtures import GaussianND

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
GO_or_GS = 'GS'
probe = 'WL'
which_comparison = 'GO_vs_GS'  # this is just to set the title of the plot
which_Rl = 'var'
which_uncertainty = 'marginal'
pes_opt = 'opt'
nparams = 7
zbins = 10
which_job = 'SPV3'
model = 'flat'
which_diff = 'normal'
specs = f'NonFlat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-EP{zbins}'
check_old_FM = False
fix_dz_nuisance = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_shear_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
w0wa_rows = [2, 3]
bar_plot = True
# ! end options

job_path = project_path / f'jobs/{which_job}'

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

assert GO_or_GS == 'GS', 'GO_or_GS should be GS, if not what are you comparing?'
assert which_diff in ['normal', 'mean'], 'which_diff should be "normal" or "mean"'
assert which_uncertainty in ['marginal', 'conditional'], 'which_uncertainty should be "marginal" or "conditional"'
assert which_Rl in ['const', 'var'], 'which_Rl should be "const" or "var"'
assert model in ['flat', 'nonflat'], 'model should be "flat" or "nonflat"'
assert probe in ['WL', 'GC', '3x2pt'], 'probe should be "WL" or "GC" or "3x2pt"'
assert pes_opt in ['opt', 'pes'], 'pes_opt should be "opt" or "pes"'
assert which_job == 'SPV3', 'which_job should be "SPV3"'

# compute percent diff of the cases chosen - careful of the indices!
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

for probe in ['WL', 'GC', '3x2pt']:

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
                          mpl_cfg.general_dict['galaxy_bias_labels_TeX']
        fid = np.concatenate((fid_cosmo, fid_IA, fid_galaxy_bias), axis=0)
    else:
        probe_lmax = probe
        probe_folder = probe + 'O'
        probename_vinc = probe + 'O'

    if probe == 'WL':
        ell_max = ell_max_WL
        pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['IA_labels_TeX'] + \
                          mpl_cfg.general_dict['shear_bias_labels_TeX']
        fid = np.concatenate((fid_cosmo, fid_IA), axis=0)
    else:
        ell_max = ell_max_GC

    if probe == 'GC':
        pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['galaxy_bias_labels_TeX']
        fid = np.concatenate((fid_cosmo, fid_galaxy_bias), axis=0)

    title = '%s, $\ell_{\\rm max} = %i$' % (probe, ell_max)

    # import vincenzo's FM, not in a dictionary because they are all split into different folders
    vinc_FM_folder = 'vincenzo/SPV3_07_2022/FishMat'

    # TODO pessimistic case
    # TODO try with pandas dataframes
    FM_GO = np.genfromtxt(
        project_path.parent / f'common_data/{vinc_FM_folder}/GaussOnly/{probe_folder}/OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')
    FM_GS = np.genfromtxt(
        project_path.parent / f'common_data/{vinc_FM_folder}/GaussSSC/{probe_folder}/OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')

    # old FMs (before specs updates)
    FM_GO_old = np.genfromtxt(
        f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/output/FM/FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl30.txt')
    FM_GS_old = np.genfromtxt(
        f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/output/FM/FM_{probe}_GS_lmax{probe_lmax}{ell_max}_nbl30_Rlvar.txt')

    # remove rows/cols for the redshift center nuisance parameters
    if fix_dz_nuisance:
        FM_GO = FM_GO[:-10, :-10]
        FM_GS = FM_GS[:-10, :-10]

    if probe != 'GC':
        if fix_shear_bias:
            assert fix_dz_nuisance, 'the case with free dz_nuisance is not implemented (you just need to be more careful with the slicing)'
            FM_GO = FM_GO[:-10, :-10]
            FM_GS = FM_GS[:-10, :-10]

    if model == 'flat':
        FM_GO = np.delete(FM_GO, obj=1, axis=0)
        FM_GO = np.delete(FM_GO, obj=1, axis=1)
        FM_GS = np.delete(FM_GS, obj=1, axis=0)
        FM_GS = np.delete(FM_GS, obj=1, axis=1)
    elif model == 'nonflat':
        w0wa_rows = [3, 4]  # Omega_DE is in position 1
        nparams += 1
        fid = np.insert(arr=fid, obj=1, values=ISTF_fid.extensions['Om_Lambda0'], axis=0)
        pars_labels_TeX = np.insert(arr=pars_labels_TeX, obj=1, values='$\\Omega_{\\rm DE}$', axis=0)

    fid = fid[:nparams]
    pars_labels_TeX = pars_labels_TeX[:nparams]

    # remove null rows and columns
    idx = mm.find_null_rows_cols_2D(FM_GO)
    idx_GS = mm.find_null_rows_cols_2D(FM_GS)
    assert np.array_equal(idx, idx_GS), 'the null rows/cols indices should be equal for GO and GS'
    FM_GO = mm.remove_null_rows_cols_2D(FM_GO, idx)
    FM_GS = mm.remove_null_rows_cols_2D(FM_GS, idx)

    ########################################################################################################################

    data = []
    fom = {}
    uncert = {}
    cases = ['GO', 'GS']

    for FM, case in zip([FM_GO, FM_GS], cases):
        uncert[case] = np.asarray(mm.uncertainties_FM(FM, nparams=nparams, fiducials=fid,
                                                      which_uncertainty=which_uncertainty, normalize=True))
        fom[case] = mm.compute_FoM(FM, w0wa_rows=w0wa_rows)

    # set uncertainties to 0 for \Omega_DE in the non-flat case, where Ode was not a free parameter
    if model == 'nonflat':
        for case in cases:
            uncert[case] = np.insert(arr=uncert[case], obj=1, values=1, axis=0)
            uncert[case] = uncert[case][:nparams]

    # uncert['diff_old'] = diff_funct(uncert['GS_old'], uncert['GO_old'])
    uncert['diff'] = diff_funct(uncert['GS'], uncert['GO'])
    # uncert['ratio_old'] = uncert['GS_old'] / uncert['GO_old']
    # uncert['ratio'] = uncert['GS'] / uncert['GO']

    uncert_vinc = {
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
        }
    }

    # print my and vincenzo's uncertainties and check that they are sufficiently close
    # with np.printoptions(precision=3, suppress=True):
    # print(f'ratio GS/GO, probe: {probe}')
    # print('dav:', uncert["ratio"])
    # print('vin:', uncert_vinc[model][f"{probe}_{pes_opt}"])

    # model_here = model
    # if not fix_shear_bias:
    #     model_here += '_shearbias'
    # assert np.allclose(uncert["ratio"], uncert_vinc[model_here][f"{probe}_{pes_opt}"], atol=0,
    #                    rtol=1e-2), 'my uncertainties differ from vincenzos'

    if check_old_FM:
        cases = ['GO_old', 'GO_new', 'GS_old', 'GS_new', 'diff_old', 'diff_new']
    else:
        cases.append('diff')

    for case in cases:
        data.append(uncert[case])

    if bar_plot:
        data = np.asarray(data)
        plot_utils.bar_plot(data, title, cases, nparams=nparams, param_names_label=pars_labels_TeX, bar_width=0.12,
                            second_axis=False, no_second_axis_bars=1)

    # if probe == '3x2pt':
    #     plot_utils.triangle_plot(FM_GO_old, FM_GS_old, fiducials=fid,
    #                              title=title, param_names_label=pars_labels_TeX)

    plt.savefig(job_path / f'output/plots/{which_comparison}/'
                           f'{probe}_ellmax{ell_max}_Rl{which_Rl}_{which_uncertainty}.png')

    # compute and print FoM
    print(f'{probe} GO FoM:', mm.compute_FoM(FM_GO))
    print(f'{probe} GS Rl_{which_Rl} FoM:', mm.compute_FoM(FM_GS))

print('*********** done ***********')
