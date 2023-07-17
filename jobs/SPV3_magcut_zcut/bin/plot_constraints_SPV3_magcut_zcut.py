import pdb
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

sys.path.append('../../bin/plot_FM_running')
import plots_FM_running as plot_utils

sys.path.append('../../../common_lib_and_cfg/common_config')
import mpl_cfg

sys.path.append('../../../common_lib_and_cfg/common_lib')
import my_module as mm

mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
mpl.use('Qt5Agg')

# ! options
specs_str = 'idIA2_idB3_idM3_idR1'
EP_or_ED = 'EP'
zbins = 13
num_params_tokeep = 7
fix_curvature = True
fix_gal_bias = False
fix_shear_bias = False  # this has to be an outer loop if you also want to vary the shear bias prior itself
fix_dz = True
include_fom = True
fid_shear_bias_prior = 1e-4
shear_bias_prior = fid_shear_bias_prior
gal_bias_perc_prior = None
string_columns = ['probe', 'go_or_gs', 'BNT_transform', 'ell_cuts']
triangle_plot = False
use_Wadd = False  # the difference is extremely small
which_pk = 'HMCode2020'
ML = 245
MS = 245
ZL = 2
ZS = 2
probes = ('WL', 'GC', '3x2pt')

# ! options


# TODO understand nan instead of None in the fm_uncert_df
# TODO maybe there is a bettewr way to handle the prior values in relation to the fix flag
# TODO superimpose bars

fm_dict = mm.load_pickle(
    '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/output/Flagship_2/FM/BNT_False/ell_cuts_False/FM_zbinsEP13_ML245_ZL02_MS245_ZS02_idIA2_idB3_idM3_idR1_pkHMCode2020.pickle')

fm_uncert_df = pd.DataFrame()
for go_or_gs in ['GO', ]:
    for probe in probes:
        for BNT_transform in [True, False]:
            for ell_cuts in [False, ]:

                names_params_to_fix = []

                fm_path = f'/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/' \
                          f'output/Flagship_2/FM/BNT_{BNT_transform}/ell_cuts_{ell_cuts}'
                fm_name = f'FM_{go_or_gs}_{probe}_zbins{EP_or_ED}{zbins}_' \
                          f'ML{ML}_ZL{ZL:02d}_MS{MS}_ZS{ZS:02d}_{specs_str}_pk{which_pk}.pickle'
                fm_pickle_name = fm_name.replace('.txt', '.pickle').replace(f'_{go_or_gs}_{probe}', '')
                fm_dict = mm.load_pickle(f'{fm_path}/{fm_pickle_name}')

                fm = fm_dict[f'FM_{probe}_{go_or_gs}']

                if probe == '3x2pt' and use_Wadd:
                    fm_wa = np.genfromtxt(
                        f'{fm_path.replace("3x2pt", "WA")}/{fm_name.replace("3x2pt", "WA")}')
                    fm += fm_wa

                # with open('/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config/'
                #           'fiducial_params_dict_for_FM.yml') as f:
                #     fiducials_dict = yaml.safe_load(f)
                fiducials_dict = fm_dict['fiducials_dict_flattened']  # TODO probably better with a yaml file...

                assert fm.shape[0] == fm.shape[1], 'FM matrix is not square!'
                assert len(fiducials_dict) == fm.shape[0], 'FM matrix and fiducial parameters length do not match!'

                # fix some of the parameters (i.e., which columns to remove)
                # if fix_curvature:
                # print('fixing curvature')
                # names_params_to_fix += ['ODE']
                # else:
                #     num_params_tokeep += 1

                if fix_shear_bias:
                    # print('fixing shear bias parameters')
                    names_params_to_fix += [f'm{(zi + 1):02d}' for zi in range(zbins)]
                    # in this way 👇there is no need for a 'add_shear_bias_prior' (or similar) boolean flag
                    shear_bias_prior = None

                if fix_gal_bias:
                    # print('fixing galaxy bias parameters')
                    names_params_to_fix += [f'bG{(zi + 1):02d}' for zi in range(zbins)]
                    gal_bias_perc_prior = None

                if fix_dz:
                    # print('fixing dz parameters')
                    names_params_to_fix += [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]

                fm, fiducials_dict = mm.mask_fm_v2(fm, fiducials_dict, names_params_to_fix,
                                                   remove_null_rows_cols=True)

                param_names = list(fiducials_dict.keys())

                # ! add prior on shear and/or gal bias
                if shear_bias_prior != None and probe in ['WL', '3x2pt']:
                    shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]
                    shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
                    fm = mm.add_prior_to_fm(fm, fiducials_dict, shear_bias_param_names, shear_bias_prior_values)

                if gal_bias_perc_prior != None and probe in ['GC', '3x2pt']:
                    gal_bias_param_names = [f'b{(zi + 1):02d}_photo' for zi in range(zbins)]

                    # go from sigma_b / b_fid to sigma_b
                    gal_bias_idxs = [param_names.index(gal_bias_param_name)
                                     for gal_bias_param_name in gal_bias_param_names]

                    gal_bias_fid_values = np.array(list(fiducials_dict.values()))[gal_bias_idxs]
                    gal_bias_prior_values = gal_bias_perc_prior * gal_bias_fid_values / 100
                    fm = mm.add_prior_to_fm(fm, fiducials_dict, gal_bias_param_names, gal_bias_prior_values)

                # ! triangle plot
                if triangle_plot:
                    if probe == '3x2pt' and go_or_gs == 'GS' and fix_shear_bias == False:
                        # decide params to show in the triangle plot
                        cosmo_param_names = list(fiducials_dict.keys())[:num_params_tokeep]
                        shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]
                        params_tot_list = cosmo_param_names + shear_bias_param_names

                        trimmed_fid_dict = {param: fiducials_dict[param] for param in params_tot_list}

                        # get the covariance matrix (careful on how you cut the FM!!)
                        fm_idxs_tokeep = [list(fiducials_dict.keys()).index(param) for param in params_tot_list]
                        cov = np.linalg.inv(fm)[fm_idxs_tokeep, :][:, fm_idxs_tokeep]

                        plot_utils.contour_plot_chainconsumer(cov, trimmed_fid_dict)

                # ! compute uncertainties from fm
                uncert_fm = mm.uncertainties_fm_v2(fm, fiducials_dict, which_uncertainty='marginal',
                                                   normalize=True,
                                                   percent_units=True)[:num_params_tokeep]

                # compute the FoM
                w0wa_idxs = param_names.index('wz'), param_names.index('wa')
                fom = mm.compute_FoM(fm, w0wa_idxs)

                df_columns_names = string_columns + [param_name for param_name in fiducials_dict.keys()][
                                                    :num_params_tokeep] + ['FoM']

                # this is a list of lists just to have a 'row list' instead of a 'column list',
                # I still haven't figured out the problem, but in this way it works
                df_columns_values = [[probe, go_or_gs, BNT_transform, ell_cuts] +
                                     uncert_fm.tolist() + [fom]]

                assert len(df_columns_names) == len(df_columns_values[0]), 'Wrong number of columns!'

                fm_uncert_df_to_concat = pd.DataFrame(df_columns_values, columns=df_columns_names)
                fm_uncert_df = pd.concat([fm_uncert_df, fm_uncert_df_to_concat], ignore_index=True)
                fm_uncert_df = fm_uncert_df.drop_duplicates()  # ! drop duplicates from df!!

# ! percent difference between two cases (usually, GO and GS)
key_to_compare = 'BNT_transform'
option_A = True
option_B = False
df_A = fm_uncert_df[fm_uncert_df[key_to_compare] == option_A]
df_B = fm_uncert_df[fm_uncert_df[key_to_compare] == option_B]
arr_A = df_A.iloc[:, len(string_columns):].select_dtypes('number').values
arr_B = df_B.iloc[:, len(string_columns):].select_dtypes('number').values
perc_diff_df = df_A
perc_diff_df.iloc[:, len(string_columns):] = mm.percent_diff(arr_A, arr_B)
perc_diff_df[key_to_compare] = 'perc_diff'
perc_diff_df['FoM'] = np.abs(perc_diff_df['FoM'])
fm_uncert_df = pd.concat([fm_uncert_df, perc_diff_df], axis=0, ignore_index=True)
fm_uncert_df = fm_uncert_df.drop_duplicates()  # ! drop duplicates from df!!

# select cases to show in bar plot
probe_toplot = '3x2pt'
fm_uncert_df_toplot = fm_uncert_df[(fm_uncert_df['probe'] == probe_toplot) &
                                   (fm_uncert_df['go_or_gs'] == 'GO')
                                   ]
# uncert_A = fm_uncert_df_toplot[fm_uncert_df_toplot['go_or_gs'] == 'GO'].iloc[:, len(string_columns):].values[0, :]
# uncert_B = fm_uncert_df_toplot[fm_uncert_df_toplot['go_or_gs'] == 'GS'].iloc[:, len(string_columns):].values[0, :]
# uncert_perc_diff = fm_uncert_df_toplot[fm_uncert_df_toplot['go_or_gs'] == 'perc_diff'].iloc[:,
#                    len(string_columns):].values[0, :]

# check the values in the paper tables
# table_1_values = list(fm_uncert_df_toplot.iloc[0, len(string_columns):].values)
# for table_1_value in table_1_values:
#     table_1_value = table_1_value/100 + 1
#     print(f'{table_1_value:.3f}')


data = fm_uncert_df_toplot.iloc[:3, len(string_columns):].values
label_list = list(fm_uncert_df_toplot['probe'].values)
label_list = ['None' if value is None else value for value in label_list]
title = None

include_fom = False
if include_fom:
    num_params_tokeep += 1
data = data[:, :num_params_tokeep]

ylabel = r'$(\sigma_{\rm GS}/\sigma_{\rm G} - 1) \times 100$ [%]'
plot_utils.bar_plot(data, title, label_list, bar_width=0.2, nparams=num_params_tokeep, param_names_label=None,
                    second_axis=False, no_second_axis_bars=0, superimpose_bars=False, show_markers=False, ylabel=ylabel,
                    include_fom=include_fom, figsize=(10, 8))

# plt.savefig('../output/plots/WL_vs_GC_vs_3x2pt_GOGS_perc_uncert_increase.pdf', bbox_inches='tight', dpi=600)

assert False, 'stop here to check which fiducials I used for the galaxy bias'
