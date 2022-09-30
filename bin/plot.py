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

project_path = Path.cwd().parent

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
ell_max_WL = 5000
ell_max_GC = 3000
nparams = 7
zbins = 10
which_job = 'SPV3'
model = 'flat'
which_diff = 'normal'
check_old_FM = True
fix_dz_nuisance = True  # whether to remove the rows/cols for the dz nuisance parameters (ie whether to fix them)
fix_shear_bias = True  # whether to remove the rows/cols for the shear bias nuisance parameters (ie whether to fix them)
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
fid_bias = np.asarray([ISTF_fid.photoz_bias[key] for key in ISTF_fid.photoz_bias.keys()])

assert GO_or_GS == 'GS', 'GO_or_GS should be GS, if not what are you comparing?'
assert which_diff in ['normal', 'mean'], 'which_diff should be "normal" or "mean"'
assert which_uncertainty in ['marginal', 'conditional'], 'which_uncertainty should be "marginal" or "conditional"'
assert which_Rl in ['const', 'var'], 'which_Rl should be "const" or "var"'
assert model in ['flat', 'nonflat'], 'model should be "flat" or "nonflat"'
assert probe in ['WL', 'GC', '3x2pt'], 'probe should be "WL" or "GC" or "3x2pt"'
assert which_job == 'SPV3', 'which_job should be "SPV3"'

if probe == '3x2pt':
    probe_lmax = 'XC'
    probe_folder = 'All'
    probename_vinc = probe
    pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['IA_labels_TeX'] + \
                      mpl_cfg.general_dict['galaxy_bias_labels_TeX']
    fid = np.concatenate((fid_cosmo, fid_IA, fid_bias), axis=0)
else:
    probe_lmax = probe
    probe_folder = probe + 'O'
    probename_vinc = probe + 'O'

if probe == 'WL':
    ell_max = ell_max_WL
    pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['IA_labels_TeX']
    fid = np.concatenate((fid_cosmo, fid_IA), axis=0)
else:
    ell_max = ell_max_GC

if probe == 'GC':
    pars_labels_TeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['galaxy_bias_labels_TeX']
    fid = np.concatenate((fid_cosmo, fid_bias), axis=0)

# import vincenzo's FM, not in a dictionary because they are all split into different folders
vinc_FM_folder = 'vincenzo/SPV3_07_2022/FishMat'
specs = f'NonFlat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-EP{zbins}'

# TODO pessimistic case
FM_GO = np.genfromtxt(
    project_path.parent / f'common_data/{vinc_FM_folder}/GaussOnly/{probe_folder}/OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')
FM_GS = np.genfromtxt(
    project_path.parent / f'common_data/{vinc_FM_folder}/GaussSSC/{probe_folder}/OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')


# TODO try with pandas dataframes


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

label_list = [f'Gauss-only covmat (GO)',
              # f'{GO_or_GS} Rlconst',
              f'Gauss+SS covmat ({GO_or_GS})',
              # f'{GO_or_GS} PyCCL',
              # f'% diff wrt mean, Rlconst vs PyCCL',
              f'[(GS/GO - 1) $\\times$ 100]',
              'bing',
              'pluf'
              ]

data = []
fom = {}
for FM, case in zip([FM_GO, FM_GS], ('GO', 'GS')):
    uncert = np.asarray(mm.uncertainties_FM(FM, nparams=nparams, fiducials=fid,
                                            which_uncertainty=which_uncertainty, normalize=True))
    data.append(uncert)
    fom[case] = mm.compute_FoM(FM)

if check_old_FM:
    probe_lmax = probe
    if probe == '3x2pt':
        probe_lmax = 'XC'
    FM_GO_test = np.genfromtxt(
        f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/output/FM/FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl30.txt')
    FM_GS_test = np.genfromtxt(
        f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SSC_comparison/output/FM/FM_{probe}_GS_lmax{probe_lmax}{ell_max}_nbl30_Rlvar.txt')

    uncert = np.asarray(mm.uncertainties_FM(FM_GO_test, nparams=nparams, fiducials=fid,
                                            which_uncertainty=which_uncertainty, normalize=True))
    data.append(uncert)

# compute percent diff of the cases chosen - careful of the indices!
print('careful about this absolute value!')
if which_diff == 'normal':
    diff_funct = mm.percent_diff
else:
    diff_funct = mm.percent_diff_mean

diff = diff_funct(data[-1], data[-2])
data.append(diff)

data = np.asarray(data)

if probe == '3x2pt':
    title = '%s, $\ell_{max} = %i$' % (probe, ell_max)
else:
    title = 'FM normalized 1-$\\sigma$ parameter constraints, %s - lower is better' % probe  # for PhD workshop

plot_utils.bar_plot(data, title, label_list, nparams=nparams, param_names_label=pars_labels_TeX, bar_width=0.18,
                    second_axis=False)



assert 1 > 2

if probe == '3x2pt':
    plot_utils.triangle_plot(FM_GO_test, FM_GS_test, fiducials=fid,
                             title=title, param_names_label=pars_labels_TeX)

plt.savefig(job_path / f'output/plots/{which_comparison}/'
                       f'{probe}_ellmax{ell_max}_Rl{which_Rl}_{which_uncertainty}.png')

# compute and print FoM
print('GO FoM:', mm.compute_FoM(FM_GO))
print(f'GS Rl_{which_Rl} FoM:', mm.compute_FoM(FM_GS))

print('*********** done ***********')
