import sys
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

sys.path.append('/home/davide/Documenti/Lavoro/Programmi/Spaceborne')
import bin.plots_FM_running as plot_utils
import bin.my_module as mm
import common_cfg.mpl_cfg as mpl_cfg


mpl.use('Qt5Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
# mpl.use('qt')

# ! options
specs_str = 'idMag0-idRSD0-idFS0-idSysWL3-idSysGC4'
nbl_WL_opt = 32
ep_or_ed = 'EP'
zbins = 10
num_params_tokeep = 7
fix_curvature = True
fix_gal_bias = False
fix_shear_bias = False  # this has to be an outer loop if you also want to vary the shear bias prior itself
fix_dz = True
include_fom = True
fid_shear_bias_prior = 1e-4
shear_bias_prior = None
gal_bias_perc_prior = None
shear_bias_priors = [None, ]
gal_bias_perc_priors = shear_bias_priors
string_columns = ['probe', 'go_or_gs', 'fix_shear_bias', 'fix_gal_bias',
                  'shear_bias_prior', 'gal_bias_perc_prior']
triangle_plot = False
use_Wadd = True  # the difference is extremely small

# these CAN BE used for fixing them or adding priors
shear_bias_param_names = [f'm{(zi + 1):02d}_photo' for zi in range(zbins)]
gal_bias_param_names = [f'b{(zi + 1):02d}_photo' for zi in range(zbins)]
dz_param_names = [f'dz{(zi + 1):02d}_photo' for zi in range(zbins)]

# ! options

# TODO understand nan instead of None in the fm_uncert_df
# TODO maybe there is a bettewr way to handle the prior values in relation to the fix flag
# TODO superimpose bars

go_or_gs_folder_dict = {
    'GO': 'GaussOnly',
    'GS': 'GaussSSC',
}
probes_vinc = ('WLO', 'GCO', '3x2pt')

fm_uncert_df = pd.DataFrame()
for go_or_gs in ['GO', 'GS']:
    for probe_vinc in probes_vinc:
        print(f'****** {probe_vinc}, {go_or_gs} ******')
        for fix_shear_bias in [False, True]:
            for fix_gal_bias in [False, True]:
                for shear_bias_prior in shear_bias_priors:
                    for gal_bias_perc_prior in gal_bias_perc_priors:

                        names_params_to_fix = []
                        num_params_tokeep = 7
                        

                        fm_path = f'/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_1_restored/' \
                                  f'FishMat_restored/{go_or_gs_folder_dict[go_or_gs]}/{probe_vinc}/FS1NoCuts'
                        fm_name = f'fm-{probe_vinc}-{nbl_WL_opt}-wzwaCDM-NonFlat-GR-TB-{specs_str}-{ep_or_ed}{zbins}'
                        fm = np.genfromtxt(f'{fm_path}/{fm_name}.dat')

                        if probe_vinc == '3x2pt' and use_Wadd:
                            fm_wa = np.genfromtxt(
                                f'{fm_path.replace("3x2pt", "WLA")}/{fm_name.replace("3x2pt", "WLA")}.dat')
                            fm += fm_wa

                        with open('/home/davide/Documenti/Lavoro/Programmi/Spaceborne/common_cfg/ISTF_fiducial_params.yml') as f:
                            fiducials_dict = yaml.safe_load(f)['FM_ordered_params']
                            
                        assert len(fiducials_dict) == fm.shape[0] == fm.shape[1], 'Wrong shape of FM matrix!'

                        # fix some of the parameters (i.e., which columns to remove)
                        if fix_curvature:
                            # print('fixing curvature')
                            names_params_to_fix += ['Om_Lambda0']
                        else:
                            num_params_tokeep += 1

                        if fix_shear_bias:
                            # print('fixing shear bias parameters')
                            names_params_to_fix += shear_bias_param_names
                            # in this way 👇there is no need for a 'add_shear_bias_prior' (or similar) boolean flag
                            shear_bias_prior = None

                        if fix_gal_bias:
                            # print('fixing galaxy bias parameters')
                            names_params_to_fix += gal_bias_param_names
                            gal_bias_perc_prior = None

                        if fix_dz:
                            # print('fixing dz parameters')
                            names_params_to_fix += dz_param_names

                        fm, fiducials_dict = mm.mask_fm_v2(fm, fiducials_dict, names_params_to_fix,
                                                           remove_null_rows_cols=True)

                        param_names = list(fiducials_dict.keys())

                        # ! add prior on shear and/or gal bias
                        if shear_bias_prior != None and probe_vinc in ['WLO', '3x2pt']:
                            shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
                            fm = mm.add_prior_to_fm(fm, fiducials_dict, shear_bias_param_names, shear_bias_prior_values)

                        if gal_bias_perc_prior != None and probe_vinc in ['GCO', '3x2pt']:
                            # go from sigma_b / b_fid to sigma_b
                            gal_bias_idxs = [param_names.index(gal_bias_param_name)
                                             for gal_bias_param_name in gal_bias_param_names]

                            warnings.warn('update the fiducial values to FS1!!!!')
                            gal_bias_fid_values = np.array(list(fiducials_dict.values()))[gal_bias_idxs]
                            gal_bias_prior_values = gal_bias_perc_prior * gal_bias_fid_values / 100
                            fm = mm.add_prior_to_fm(fm, fiducials_dict, gal_bias_param_names, gal_bias_prior_values)
                            
                        # plot FM for defense presentation
                        # plt.matshow(fm[:num_params_tokeep, :num_params_tokeep])
                        # plt.xticks(range(num_params_tokeep), mpl_cfg.general_dict['cosmo_labels_TeX'])
                        # plt.yticks(range(num_params_tokeep), mpl_cfg.general_dict['cosmo_labels_TeX'])
                        # plt.colorbar()
                        # plt.title('WL FM')
                        # plt.savefig('/Users/davide/Documents/Science 🛰/Talks/2023_12_22 - Defense/img/FM_WL.pdf', dpi=300, bbox_inches='tight')
                        
                        # ! triangle plot
                        if triangle_plot and fix_shear_bias==True and fix_gal_bias==False:
                            # decide params to show in the triangle plot
                            cosmo_param_names = list(fiducials_dict.keys())[:num_params_tokeep]
                            # params_tot_list = cosmo_param_names + shear_bias_param_names
                            # params_tot_list = cosmo_param_names + gal_bias_param_names
                            params_tot_list = cosmo_param_names

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
                        w0wa_idxs = param_names.index('w_0'), param_names.index('w_a')
                        fom = mm.compute_FoM(fm, w0wa_idxs)

                        df_columns_names = string_columns + [param_name for param_name in fiducials_dict.keys()][
                                                            :num_params_tokeep] + ['FoM']

                        # this is a list of lists just to have a 'row list' instead of a 'column list',
                        # I still haven't figured out the problem, but in this way it works
                        df_columns_values = [[probe_vinc, go_or_gs, fix_shear_bias, fix_gal_bias,
                                              shear_bias_prior, gal_bias_perc_prior] +
                                             uncert_fm.tolist() + [fom]]

                        assert len(df_columns_names) == len(df_columns_values[0]), 'Wrong number of columns!'

                        fm_uncert_df_to_concat = pd.DataFrame(df_columns_values, columns=df_columns_names)
                        fm_uncert_df = pd.concat([fm_uncert_df, fm_uncert_df_to_concat], ignore_index=True)
                        fm_uncert_df = fm_uncert_df.drop_duplicates()  # ! drop duplicates from df!!

# ! percent difference
df_gs = fm_uncert_df[fm_uncert_df["go_or_gs"] == "GS"]
df_go = fm_uncert_df[fm_uncert_df["go_or_gs"] == "GO"]
arr_gs = df_gs.iloc[:, len(string_columns):].select_dtypes('number').values
arr_go = df_go.iloc[:, len(string_columns):].select_dtypes('number').values
perc_diff_df = df_gs
perc_diff_df.iloc[:, len(string_columns):] = mm.percent_diff(arr_gs, arr_go)
perc_diff_df['go_or_gs'] = 'perc_diff'
perc_diff_df['FoM'] = np.abs(perc_diff_df['FoM'])
# perc_diff_df.iloc[:, -1] = mm.percent_diff(arr_gs[:, -1], arr_go[:, -1])
fm_uncert_df = pd.concat([fm_uncert_df, perc_diff_df], axis=0, ignore_index=True)
fm_uncert_df = fm_uncert_df.drop_duplicates()  # ! drop duplicates from df!!

# select cases to show in bar plot
probe_vinc_toplot = '3x2pt'
go_or_gs_toplot = 'perc_diff'
fm_uncert_df_toplot = fm_uncert_df[(fm_uncert_df['probe'] == probe_vinc_toplot) &
                                   (fm_uncert_df['fix_gal_bias'] == False) &
                                   (fm_uncert_df['shear_bias_prior'].isna()) &
                                   (fm_uncert_df['gal_bias_perc_prior'].isna())
                                   ]
uncert_go = fm_uncert_df_toplot[fm_uncert_df_toplot['go_or_gs'] == 'GO'].iloc[:, len(string_columns):].values[0, :]
uncert_gs = fm_uncert_df_toplot[fm_uncert_df_toplot['go_or_gs'] == 'GS'].iloc[:, len(string_columns):].values[0, :]
uncert_perc_diff_df = fm_uncert_df_toplot[fm_uncert_df_toplot['go_or_gs'] == 'GO']
uncert_perc_diff_fix_shear_bias = uncert_perc_diff_df[uncert_perc_diff_df['fix_shear_bias'] == True].iloc[:,
                   len(string_columns):].values[0, :]
uncert_perc_diff_free_shear_bias = uncert_perc_diff_df[uncert_perc_diff_df['fix_shear_bias'] == False].iloc[:,
                   len(string_columns):].values[0, :]

# check the values in the paper tables
# table_1_values = list(fm_uncert_df_toplot.iloc[0, len(string_columns):].values)
# for table_1_value in table_1_values:
#     table_1_value = table_1_value/100 + 1
#     print(f'{table_1_value:.3f}')


data = uncert_perc_diff_df.iloc[:3, len(string_columns):].values
label_list = list(fm_uncert_df_toplot['probe'].values)
label_list = ['None' if value is None else value for value in label_list]
title = probe_vinc_toplot

if include_fom:
    num_params_tokeep += 1
data = data[:, :num_params_tokeep]

warnings.warn('flipping rows to show the fix shear bias case first')
data = np.flip(data, axis=0)
label_list = ['Shear bias fixed', 'Shear bias free']


# data = np.delete(data, [1, 3, 4, 5], axis=1)
# num_params_tokeep = 4
# param_names_label = ['$\\Omega_{{\\rm m},0}$',
#  '$w_0$',
#  '$\\sigma_8$', 'FoM']

param_names_label = None
ylabel = r'$(\sigma_{\rm GS}/\sigma_{\rm G} - 1) \times 100$ [%]'
ylabel = r'relative uncertainty, G [%]'
plot_utils.bar_plot(data, title, label_list, bar_width=0.2, nparams=num_params_tokeep,
                    param_names_label=param_names_label,
                    second_axis=False, no_second_axis_bars=0, superimpose_bars=False, show_markers=False, ylabel=ylabel,
                    include_fom=include_fom, figsize=(10, 8))

# plt.savefig('/Users/davide/Documents/Science 🛰/Talks/2023_12_22 - Defense/img/WL_shear_bias_free_fixed.pdf', dpi=300, bbox_inches='tight')
assert False, 'stop here to check which fiducials I used for the galaxy bias'

# ! study FoM vs priors on shear and gal. bias
probe = 'GCO'
param_toplot = 'w_a'
other_nuisance_value = 1e-4  # for WLO and GCO this has no impact, of course
nuisance_name = 'gal_bias'

# horrible cases selector
if probe == 'GCO':
    nuisance_name = 'gal_bias'
    nuisance_prior_name = 'gal_bias_perc_prior'
    other_nuisance_name = 'shear_bias_prior'
    xlabel = nuisance_prior_name

elif probe == 'WLO':
    nuisance_name = 'shear_bias'
    nuisance_prior_name = 'shear_bias_prior'
    other_nuisance_name = 'gal_bias_perc_prior'
    xlabel = nuisance_prior_name

elif probe == '3x2pt':
    if nuisance_name == 'gal_bias':
        nuisance_prior_name = 'gal_bias_perc_prior'
        other_nuisance_name = 'shear_bias_prior'
    elif nuisance_name == 'shear_bias':
        nuisance_prior_name = 'shear_bias_prior'
        other_nuisance_name = 'gal_bias_perc_prior'
    xlabel = f'{nuisance_name} prior'
    xlabel += f'; {other_nuisance_name} = {other_nuisance_value}'

else:
    raise ValueError('Wrong probe!')

nuisance_vals = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                 (fm_uncert_df['go_or_gs'] == 'GO') &
                                 (fm_uncert_df[f'fix_{nuisance_name}'] == False) &
                                 (fm_uncert_df[other_nuisance_name] == other_nuisance_value)
                                 ][nuisance_prior_name].values
param_toplot_go = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                   (fm_uncert_df['go_or_gs'] == 'GO') &
                                   (fm_uncert_df[f'fix_{nuisance_name}'] == False) &
                                   (fm_uncert_df[other_nuisance_name] == other_nuisance_value)
                                   ][param_toplot].values
param_toplot_gs = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                   (fm_uncert_df['go_or_gs'] == 'GS') &
                                   (fm_uncert_df[f'fix_{nuisance_name}'] == False) &
                                   (fm_uncert_df[other_nuisance_name] == other_nuisance_value)
                                   ][param_toplot].values
param_toplot_perc_diff = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                          (fm_uncert_df['go_or_gs'] == 'perc_diff') &
                                          (fm_uncert_df[f'fix_{nuisance_name}'] == False) &
                                          (fm_uncert_df[other_nuisance_name] == other_nuisance_value)
                                          ][param_toplot].values

# these below are all horizontal lines
param_toplot_fix_nuisance_go = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                                (fm_uncert_df['go_or_gs'] == 'GO') &
                                                (fm_uncert_df[f'fix_{nuisance_name}'] == True) &
                                                (fm_uncert_df[other_nuisance_name] == other_nuisance_value)
                                                ][param_toplot].values
param_toplot_fix_nuisance_gs = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                                (fm_uncert_df['go_or_gs'] == 'GS') &
                                                (fm_uncert_df[f'fix_{nuisance_name}'] == True) &
                                                (fm_uncert_df[other_nuisance_name] == other_nuisance_value)
                                                ][param_toplot].values

fm_uncert_df_toplot_no_prior_go = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                                   (fm_uncert_df['go_or_gs'] == 'GO') &
                                                   (fm_uncert_df[f'fix_{nuisance_name}'] == False) &
                                                   (fm_uncert_df[other_nuisance_name] == other_nuisance_value) &
                                                   (fm_uncert_df[nuisance_prior_name].isna())
                                                   ][param_toplot].values
fm_uncert_df_toplot_no_prior_gs = fm_uncert_df.loc[(fm_uncert_df['probe'] == probe) &
                                                   (fm_uncert_df['go_or_gs'] == 'GS') &
                                                   (fm_uncert_df[f'fix_{nuisance_name}'] == False) &
                                                   (fm_uncert_df[other_nuisance_name] == other_nuisance_value) &
                                                   (fm_uncert_df[nuisance_prior_name].isna())
                                                   ][param_toplot].values

plt.figure()
plt.plot(nuisance_vals, param_toplot_go, ls='-', marker='o', label='GO', color='tab:blue')
plt.plot(nuisance_vals, param_toplot_gs, ls='-', marker='o', label='GS', color='tab:orange')
plt.plot(nuisance_vals, param_toplot_perc_diff, ls='-', marker='o', label='perc. diff.', color='tab:green')
plt.axhline(param_toplot_fix_nuisance_go[0], ls='--', label='fixed nuisance, GO', color='tab:blue')
plt.axhline(param_toplot_fix_nuisance_gs[0], ls='--', label='fixed nuisance, GS', color='tab:orange')
plt.axhline(fm_uncert_df_toplot_no_prior_go[0], ls=':', label='no prior, GO', color='tab:blue')
plt.axhline(fm_uncert_df_toplot_no_prior_gs[0], ls=':', label='no prior, GS', color='tab:orange')
plt.xlabel(xlabel)
plt.ylabel(param_toplot)
plt.xscale('log')
plt.yscale('log')
plt.title(probe)
plt.legend()
