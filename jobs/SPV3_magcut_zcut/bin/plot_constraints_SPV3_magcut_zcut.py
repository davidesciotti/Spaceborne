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

sys.path.append('../../../common_lib_and_cfg/common_cfg')
import mpl_cfg

sys.path.append('../../../common_lib_and_cfg/common_lib')
import my_module as mm

sys.path.append('../config')
import config_SPV3_magcut_zcut as cfg

mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
mpl.use('Qt5Agg')

general_cfg = cfg.general_cfg
FM_cfg = cfg.FM_cfg

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
fid_shear_bias_prior = 5e-4
shear_bias_prior = fid_shear_bias_prior
gal_bias_perc_prior = None
string_columns = ['probe', 'go_or_gs', 'whose_FM', 'BNT_transform', 'ell_cuts', 'which_cuts', 'center_or_min', 'kmax_h_over_Mpc'
                  ]
triangle_plot = False
use_Wadd = False  # the difference is extremely small
which_pk = 'HMCode2020'
ML = 245
MS = 245
ZL = 2
ZS = 2
probes = ('WL', 'GC', '3x2pt')
which_cuts = 'Vincenzo'
center_or_min = 'min'
h = 0.67  # 0.6774?
whose_FM_list = ('davide',)
# ! options

probe_vinc_dict = {
    'WL': 'WLO',
    'GC': 'GCO',
    '3x2pt': '3x2pt',
}

# TODO understand nan instead of None in the fm_uncert_df
# TODO maybe there is a bettewr way to handle the prior values in relation to the fix flag
# TODO superimpose bars

assert fix_curvature, 'Im studyin only flat models'

fm_uncert_df = pd.DataFrame()
for go_or_gs in ['GO', ]:
    for probe in probes:
        for BNT_transform in [False, True]:
            for ell_cuts in [False, True]:
                for kmax_h_over_Mpc in general_cfg['kmax_h_over_Mpc_list']:
                    for whose_FM in whose_FM_list:
                        for center_or_min in ['min']:

                            names_params_to_fix = []

                            if whose_FM == 'davide':
                                fm_path = f'/home/cosmo/davide.sciotti/data/SSC_restructured_v2/jobs/SPV3_magcut_zcut/' \
                                          f'output/Flagship_2/FM/BNT_{BNT_transform}/ell_cuts_{ell_cuts}'
                                fm_name = f'FM_{go_or_gs}_{probe}_zbins{EP_or_ED}{zbins}_' \
                                          f'ML{ML}_ZL{ZL:02d}_MS{MS}_ZS{ZS:02d}_{specs_str}_pk{which_pk}.pickle'

                                if ell_cuts:
                                    fm_path += f'/{which_cuts}/ell_{center_or_min}'
                                    fm_name = fm_name.replace(f'.pickle', f'kmaxhoverMpc{kmax_h_over_Mpc:.03f}.pickle')

                                fm_pickle_name = fm_name.replace('.txt', '.pickle').replace(f'_{go_or_gs}_{probe}', '')
                                fm_dict = mm.load_pickle(f'{fm_path}/{fm_pickle_name}')

                                fm = fm_dict[f'FM_{probe}_{go_or_gs}']

                            elif whose_FM == 'vincenzo':

                                kmax_1_over_Mpc = int(np.round(kmax_h_over_Mpc * h * 100))  # for vincenzo's file names

                                fm_path = '/home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/' \
                                          f'LiFEforSPV3/OutputFiles/FishMat/GaussOnly/Flat/' \
                                          f'{probe_vinc_dict[probe]}/{which_pk}/TestKappaMax'
                                fm_name = f'fm-{probe_vinc_dict[probe]}-{EP_or_ED}{zbins}-ML{ML}-MS{MS}-{specs_str.replace("_", "-")}' \
                                          f'-kM{kmax_1_over_Mpc:03d}.dat'
                                fm = np.genfromtxt(f'{fm_path}/{fm_name}')

                            if probe == '3x2pt' and use_Wadd:
                                assert False, 'import of Wadd not implemented for Vincenzos FM yet'
                                fm_wa = np.genfromtxt(
                                    f'{fm_path.replace("3x2pt", "WA")}/{fm_name.replace("3x2pt", "WA")}')
                                fm += fm_wa

                            # with open('/home/cosmo/davide.sciotti/data/common_lib_and_cfg/common_config/'
                            #           'fiducial_params_dict_for_FM.yml') as f:
                            #     fiducials_dict = yaml.safe_load(f)
                            fiducials_dict = fm_dict['fiducials_dict_flattened']  # TODO probably better a yaml file...

                            assert fm.shape[0] == fm.shape[1], 'FM matrix is not square!'
                            assert len(fiducials_dict) == fm.shape[0], 'FM matrix and fiducial parameters ' \
                                                                       'length do not match!'

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
                            cosmo_param_names = list(fiducials_dict.keys())[:num_params_tokeep]

                            # ! add prior on shear and/or gal bias
                            if shear_bias_prior != None and probe in ['WL', '3x2pt']:
                                shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
                                shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
                                fm = mm.add_prior_to_fm(fm, fiducials_dict, shear_bias_param_names,
                                                        shear_bias_prior_values)

                            if gal_bias_perc_prior != None and probe in ['GC', '3x2pt']:
                                gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(zbins)]

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
                                    shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]
                                    params_tot_list = cosmo_param_names + shear_bias_param_names

                                    trimmed_fid_dict = {param: fiducials_dict[param] for param in params_tot_list}

                                    # get the covariance matrix (careful on how you cut the FM!!)
                                    fm_idxs_tokeep = [list(fiducials_dict.keys()).index(param) for param in
                                                      params_tot_list]
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
                            df_columns_values = [
                                [probe, go_or_gs, whose_FM, BNT_transform, ell_cuts, which_cuts, center_or_min,
                                 kmax_h_over_Mpc] +
                                uncert_fm.tolist() + [fom]]

                            assert len(df_columns_names) == len(df_columns_values[0]), 'Wrong number of columns!'

                            fm_uncert_df_to_concat = pd.DataFrame(df_columns_values, columns=df_columns_names)
                            fm_uncert_df = pd.concat([fm_uncert_df, fm_uncert_df_to_concat], ignore_index=True)
                            fm_uncert_df = fm_uncert_df.drop_duplicates()  # ! drop duplicates from df!!

# ! percent difference between two cases (usually, GO and GS)
key_to_compare = 'BNT_transform'
value_A = True
value_B = False

param_toplot = 'FoM'
# param_toplot = cosmo_param_names
probe_toplot = 'GC'
center_or_min = 'min'
ell_cuts = True

# add percent difference to the dataframe
df_A = fm_uncert_df[fm_uncert_df[key_to_compare] == value_A]
df_B = fm_uncert_df[fm_uncert_df[key_to_compare] == value_B]
arr_A = df_A.iloc[:, len(string_columns):].select_dtypes('number').values
arr_B = df_B.iloc[:, len(string_columns):].select_dtypes('number').values
perc_diff_df = df_A.copy()
perc_diff_df.iloc[:, len(string_columns):] = mm.percent_diff(arr_B, arr_A)  # ! the reference is GO!!
perc_diff_df[key_to_compare] = 'perc_diff'
perc_diff_df['FoM'] = np.abs(perc_diff_df['FoM'])
fm_uncert_df = pd.concat([fm_uncert_df, perc_diff_df], axis=0, ignore_index=True)
fm_uncert_df = fm_uncert_df.drop_duplicates()  # drop duplicates from df

fm_uncert_df_toplot = fm_uncert_df[
    (fm_uncert_df['probe'] == probe_toplot) &
    (fm_uncert_df['go_or_gs'] == 'GO') &
    (fm_uncert_df['whose_FM'] == 'davide') &
    (fm_uncert_df['ell_cuts'] == ell_cuts) &
    (fm_uncert_df['center_or_min'] == center_or_min)
    ]
uncert_A = fm_uncert_df_toplot[fm_uncert_df_toplot[key_to_compare] == value_A][param_toplot].values
uncert_B = fm_uncert_df_toplot[fm_uncert_df_toplot[key_to_compare] == value_B][param_toplot].values
uncert_perc_diff = fm_uncert_df_toplot[fm_uncert_df_toplot[key_to_compare] == 'perc_diff'][param_toplot].values

plt.figure()
title = f'{probe_toplot}, {key_to_compare}={value_A} vs {value_B}'
plt.title(f'{title}')
plt.plot(general_cfg['kmax_h_over_Mpc_list'], uncert_A, label=f'{key_to_compare}={value_A}', marker='o')
plt.plot(general_cfg['kmax_h_over_Mpc_list'], uncert_B, label=f'{key_to_compare}={value_B}', marker='o')
plt.plot(general_cfg['kmax_h_over_Mpc_list'], uncert_perc_diff, label='perc diff', marker='.')
plt.xscale('log')
plt.xlabel(r'$k_{\rm max}$ [h/Mpc]')
plt.ylabel(param_toplot)
plt.legend()

data = fm_uncert_df_toplot.iloc[:3, len(string_columns):].values
label_list = list(fm_uncert_df_toplot['probe'].values)
label_list = ['None' if value is None else value for value in label_list]

include_fom = False
if include_fom:
    num_params_tokeep += 1
data = data[:, :num_params_tokeep]

ylabel = r'$(\sigma_{\rm GS}/\sigma_{\rm G} - 1) \times 100$ [%]'
plot_utils.bar_plot(data, title, label_list, bar_width=0.2, nparams=num_params_tokeep, param_names_label=None,
                    second_axis=False, no_second_axis_bars=0, superimpose_bars=False, show_markers=False, ylabel=ylabel,
                    include_fom=include_fom, figsize=(10, 8))

# plt.savefig('../output/plots/WL_vs_GC_vs_3x2pt_GOGS_perc_uncert_increase.pdf', bbox_inches='tight', dpi=600)


print('done')
