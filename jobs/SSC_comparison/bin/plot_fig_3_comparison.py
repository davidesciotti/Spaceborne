import itertools
import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from getdist import MCSamples, plots
from matplotlib.cm import get_cmap
from getdist.gaussian_mixtures import GaussianND

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(str(project_path.parent / 'common_data'))
import common_lib.my_module as mm
import common_config.mpl_cfg as mpl_cfg
import common_config.ISTF_fid_params as ISTF_fid

sys.path.append(str(project_path / 'bin'))
import plots_FM_running as plot_utils

sys.path.append(str(project_path / 'jobs'))
import SSC_comparison.config.config_SSC_comparison as cfg

# plot config
matplotlib.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')
markersize = 10

# import all outputs from this script
FM_dict = dict(mm.get_kv_pairs(job_path / 'input/FM_plot_3', extension="txt"))
nparams_sylvain = FM_dict['sylvain_GS'].shape[0]

# ! options
probe = '3x2pt'
which_Rl = 'var'
ell_max = 3000
which_uncertainty = 'marginal'
nparams = 7
zbins = 10
# ! end options

# fiducial values
fid_cosmo = np.asarray([ISTF_fid.primary[key] for key in ISTF_fid.primary.keys()])[:7]
fid_IA = np.asarray([ISTF_fid.IA_free[key] for key in ISTF_fid.IA_free.keys()])
fid_bias = np.asarray([ISTF_fid.photoz_galaxy_bias[key] for key in ISTF_fid.photoz_galaxy_bias.keys()])

# param names (plain)
param_names_str = mpl_cfg.general_dict['cosmo_labels'] + mpl_cfg.general_dict['IA_labels'] + \
                  mpl_cfg.general_dict['galaxy_bias_labels']

# param names (latex)
param_names_LaTeX = mpl_cfg.general_dict['cosmo_labels_TeX'] + mpl_cfg.general_dict['IA_labels_TeX'] + \
                    mpl_cfg.general_dict['galaxy_bias_labels_TeX']
fid = np.concatenate((fid_cosmo, fid_IA, fid_bias), axis=0)

# cut the number of parameters
fid = fid[:nparams]
param_names_LaTeX = param_names_LaTeX[:nparams]
param_names_str = param_names_str[:nparams]

keys = list(FM_dict.keys())

for key in FM_dict.keys():
    # this is because My FMs include Omega_x
    if 'davide' in key:
        FM_dict[key] = np.delete(FM_dict[key], 1, axis=0)
        FM_dict[key] = np.delete(FM_dict[key], 1, axis=1)

    # cut my FM and remove null rows and columns
    FM_dict[key] = FM_dict[key][:nparams_sylvain, :nparams_sylvain]

    # remove null rows and columns
    FM_dict[key], cut_param_names_list, cut_fiducials_list = mm.mask_FM(FM_dict[key], param_names_str, fid,
                                                                        params_tofix_dict={},
                                                                        remove_null_rows_cols=True)
    assert FM_dict[key].shape[0] == FM_dict[key].shape[1], 'FM is not square!'
    assert FM_dict[key].shape[0] == 20, 'FM has wrong number of parameters!'

label_list = [f'G',
              'GS'
              ]

columns = [('FM_name',), param_names_str]
columns = list(itertools.chain(*columns))
data = pd.DataFrame(columns=columns)
fom = {}
for i, key in enumerate(keys):
    uncert = np.asarray(mm.uncertainties_FM(FM_dict[key], nparams=nparams, fiducials=fid,
                                            which_uncertainty=which_uncertainty, normalize=True)[:nparams])
    new_row_list = [key, *uncert]
    new_row = pd.DataFrame([new_row_list], columns=data.columns)

    data = pd.concat([data, new_row], ignore_index=True)
    fom[key] = mm.compute_FoM(FM_dict[key])

uncert_dav_GO = data.loc[data['FM_name'] == 'davide_GO'].drop(columns='FM_name').to_numpy()[0]
uncert_dav_GS = data.loc[data['FM_name'] == 'davide_GS'].drop(columns='FM_name').to_numpy()[0]
uncert_syl_GO = data.loc[data['FM_name'] == 'sylvain_GO'].drop(columns='FM_name').to_numpy()[0]
uncert_syl_GS = data.loc[data['FM_name'] == 'sylvain_GS'].drop(columns='FM_name').to_numpy()[0]
diff_GO = mm.percent_diff_mean(uncert_syl_GO, uncert_dav_GO)
diff_GS = mm.percent_diff_mean(uncert_syl_GS, uncert_dav_GS)

data_array = np.vstack((diff_GO, diff_GS))

title = '%s, $\ell_{max} = %i$' % (probe, ell_max)

plot_utils.bar_plot(data_array, title, label_list, nparams=nparams, param_names_label=param_names_LaTeX, bar_width=0.40,
                    second_axis=False, superimpose_bars=True)
# if probe == '3x2pt':
#     plot_utils.triangle_plot(FM_dict[keys[0]], FM_dict[keys[1]], fiducials=fid,
#                              title=title, param_names_LaTeX=param_names_LaTeX)

# plt.savefig(job_path / f'output/plots/{which_comparison}/'
#                        f'{probe}_ellmax{ell_max}_Rl{which_Rl}_{which_uncertainty}.png')

# compute and print FoM
# print('GO FoM:', mm.compute_FoM(FM_dict[f'FM_{probe}_GO_lmax{probe_lmax}{ell_max}_nbl{nbl}']))
# print('Rl const FoM:', mm.compute_FoM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlconst']))
# print('Rl var FoM:', mm.compute_FoM(FM_dict[f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}_Rlvar']))

print('*********** done ***********')
