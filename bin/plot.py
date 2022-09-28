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
job = 'SPV3'
model = 'flat'
# ! end options

job_path = project_path / f'jobs/{job}'

# TODO fix this
if job == 'SPV3':
    nbl = 32
else:
    raise ValueError

# fiducial values
fid_cosmo = np.asarray([ISTF_fid.primary[key] for key in ISTF_fid.primary.keys()])[:7]
fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
fid_bias = np.asarray([ISTF_fid.photoz_bias[key] for key in ISTF_fid.photoz_bias.keys()])

assert GO_or_GS == 'GS', 'GO_or_GS should be GS, if not what are you comparing?'

if probe == '3x2pt':
    probe_lmax = 'XC'
    probe_folder = 'All'
    probename_vinc = probe
    param_names_label = mpl_cfg.general_dict['param_names_label_rm'] + mpl_cfg.general_dict['IA_names_label'] + \
                        mpl_cfg.general_dict['bias_names_label']
    fid = np.concatenate((fid_cosmo, fid_IA, fid_bias), axis=0)
else:
    probe_lmax = probe
    probe_folder = probe + 'O'
    probename_vinc = probe + 'O'

if probe == 'WL':
    ell_max = ell_max_WL
    param_names_label = mpl_cfg.general_dict['param_names_label_rm'] + mpl_cfg.general_dict['IA_names_label']
    fid = np.concatenate((fid_cosmo, fid_IA), axis=0)
else:
    ell_max = ell_max_GC

if probe == 'GC':
    param_names_label = mpl_cfg.general_dict['param_names_label_rm'] + mpl_cfg.general_dict['bias_names_label']
    fid = np.concatenate((fid_cosmo, fid_bias), axis=0)

elif probe not in ['WL', 'GC', '3x2pt']:
    raise ValueError('probe should be WL, GC or 3x2pt')

# import vincenzo's FM, not in a dictionary because they are all split into different folders
vinc_FM_folder = 'vincenzo/SPV3_07_2022/FishMat'
specs = f'NonFlat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-EP{zbins}'

# TODO pessimistic case
FM_GO = np.genfromtxt(
    project_path.parent / f'common_data/{vinc_FM_folder}/GaussOnly/{probe_folder}/OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')
FM_GS = np.genfromtxt(
    project_path.parent / f'common_data/{vinc_FM_folder}/GaussSSC/{probe_folder}/OneSample/fm-{probename_vinc}-{nbl}-wzwaCDM-{specs}.dat')

if model == 'flat':
    FM_GO = np.delete(FM_GO, 1, 0)
    FM_GO = np.delete(FM_GO, 1, 1)
    FM_GS = np.delete(FM_GS, 1, 0)
    FM_GS = np.delete(FM_GS, 1, 1)
elif model == 'nonflat':
    # TODO test this
    np.insert(arr=fid, obj=1, values=ISTF_fid.extensions['Om_Lambda0'], axis=0)
    np.insert(arr=param_names_label, obj=1, values='$\Omega_{\rm DE}$', axis=0)
else:
    raise ValueError('model must be either flat or nonflat')

fid = fid[:nparams]
param_names_label = param_names_label[:nparams]

# find null rows and columns
zero_rows_idxs = np.where(np.all(FM_GO == 0, axis=0))[0]
zero_cols_idxs = np.where(np.all(FM_GO == 0, axis=1))[0]
assert np.array_equal(zero_rows_idxs, zero_cols_idxs), 'null rows and columns indices should be the same!'

# print the corresponding indices
if zero_rows_idxs.shape[0] != 0:
    print(f'FM had some null rows and columns, at indices {zero_rows_idxs}')

# delete the null rows and columns
FM_GO = np.delete(FM_GO, zero_rows_idxs, axis=0)
FM_GO = np.delete(FM_GO, zero_rows_idxs, axis=1)
FM_GS = np.delete(FM_GS, zero_rows_idxs, axis=0)
FM_GS = np.delete(FM_GS, zero_rows_idxs, axis=1)

########################################################################################################################

label_list = [f'Gauss-only covmat (GO)',
              # f'{GO_or_GS} Rlconst',
              f'Gauss+SS covmat ({GO_or_GS})',
              # f'{GO_or_GS} PyCCL',
              # f'% diff wrt mean, Rlconst vs PyCCL',
              f'[(GS/GO - 1) $\\times$ 100]']

data = []
fom = {}
for FM, case in zip([FM_GO, FM_GS], ('GO', 'GS')):
    uncert = np.asarray(mm.uncertainties_FM(FM, nparams=nparams, fiducials=fid,
                                            which_uncertainty=which_uncertainty, normalize=True)[:nparams])
    data.append(uncert)
    fom[case] = mm.compute_FoM(FM)

# compute percent diff of the cases chosen - careful of the indices!
print('careful about this absolute value!')
diff_1 = mm.percent_diff(data[-1], data[-2])
# diff_2 = mm.percent_diff_mean(data[-2], data[-1])
data.append(diff_1)
# data.append(diff_2)

data = np.asarray(data)

if probe == '3x2pt':
    title = '%s, $\ell_{max} = %i$' % (probe, ell_max)
else:
    title = 'FM normalized 1-$\\sigma$ parameter constraints, %s - lower is better' % probe  # for PhD workshop

plot_utils.bar_plot(data, title, label_list, nparams=nparams, param_names_label=param_names_label, bar_width=0.18,
                    second_axis=True)
if probe == '3x2pt':
    plot_utils.triangle_plot(FM_GO, FM_GS, fiducials=fid,
                             title=title, param_names_label=param_names_label)

plt.savefig(job_path / f'output/plots/{which_comparison}/'
                       f'{probe}_ellmax{ell_max}_Rl{which_Rl}_{which_uncertainty}.png')

# compute and print FoM
print('GO FoM:', mm.compute_FoM(FM_GO))
print('Rl var FoM:', mm.compute_FoM(FM_GS))

print('*********** done ***********')
