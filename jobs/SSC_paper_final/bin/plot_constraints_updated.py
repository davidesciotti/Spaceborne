import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append('../../bin/plot_FM_running')
import plots_FM_running as plot_utils

sys.path.append('../../../common_lib_and_cfg/common_config')
import mpl_cfg

sys.path.append('../../../common_lib_and_cfg/common_lib')
import my_module as mm

mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
mpl.use('Qt5Agg')

with open('/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config/'
          'fiducial_params_dict_for_FM.yml') as f:
    fiducials_dict = yaml.safe_load(f)

# ! options
specs_str = 'idMag0-idRSD0-idFS0-idSysWL3-idSysGC4'
nbl_WL_opt = 32
ep_or_ed = 'EP'
zbins = 10
params_tokeep = 7
fix_curvature = True
fix_galaxy_bias = False
fix_shear_bias = True
fix_dz = True
include_fom = True
# ! options


go_or_gs_folder_dict = {
    'GO': 'GaussOnly',
    'GS': 'GaussSSC',
}
probes_vinc = ('WLO', 'GCO', '3x2pt')

fm_uncert_df = pd.DataFrame()
for go_or_gs in ['GO', 'GS']:
    for probe_vinc in probes_vinc:

        names_params_to_fix = []

        fm_path = f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_1_restored/' \
                  f'FishMat_restored/{go_or_gs_folder_dict[go_or_gs]}/{probe_vinc}/FS1NoCuts'
        fm_dict = dict(mm.get_kv_pairs(f'{fm_path}', extension="dat"))
        fm_name = f'fm-{probe_vinc}-{nbl_WL_opt}-wzwaCDM-NonFlat-GR-TB-' \
                  f'{specs_str}-{ep_or_ed}{zbins}'

        fm = fm_dict[fm_name]

        assert len(fiducials_dict) == fm.shape[0] == fm.shape[1], 'Wrong shape of FM matrix!'


        # fix some of the parameters (i.e., which columns to remove)
        if fix_curvature:
            print('fixing curvature')
            names_params_to_fix += ['Om_Lambda0']
        else:
            params_to_fix = 8

        if probe_vinc == 'WLO':
            print('fixing shear bias parameters')
            names_params_to_fix += [f'm{zi + 1}_photo' for zi in range(zbins)]

        if fix_galaxy_bias:
            print('fixing galaxy bias parameters')
            names_params_to_fix += [f'b{(zi + 1):02d}_photo' for zi in range(zbins)]

        if fix_dz:
            print('fixing dz parameters')
            names_params_to_fix += [f'dz{zi + 1}_photo' for zi in range(zbins)]

        fm, fiducials_dict_trimmed = mm.mask_fm_v2(fm, fiducials_dict, names_params_to_fix, remove_null_rows_cols=True)
        uncert_fm = mm.uncertainties_fm_v2(fm, fiducials_dict_trimmed, which_uncertainty='marginal', normalize=True,
                                           percent_units=True)[:params_tokeep]

        param_names = list(fiducials_dict_trimmed.keys())
        param_values = list(fiducials_dict_trimmed.values())

        w0wa_idxs = param_names.index('w_0'), param_names.index('w_a')
        fom = mm.compute_FoM(fm, w0wa_idxs)

        string_columns = ['probe', 'go_or_gs']
        df_columns_names = string_columns + [param_name for param_name in fiducials_dict_trimmed.keys()][
                                            :params_tokeep] + ['FoM']

        # this is a list of lists just to have a 'row list' instead of a 'column list',
        # I still haven't figured out the problem...
        df_columns_values = [[probe_vinc, go_or_gs] + uncert_fm.tolist() + [fom]]

        # assert False
        assert len(df_columns_names) == len(df_columns_values[0]), 'Wrong number of columns!'

        fm_uncert_df_to_concat = pd.DataFrame(df_columns_values, columns=df_columns_names)
        fm_uncert_df = pd.concat([fm_uncert_df, fm_uncert_df_to_concat], ignore_index=True)

# assert False

# compute percent differences between GO and GS
for probe_vinc in probes_vinc:
    uncert_go = fm_uncert_df[(fm_uncert_df['go_or_gs'] == 'GO') & (fm_uncert_df['probe'] == probe_vinc)].select_dtypes(
        include=np.number).values[0]
    uncert_gs = fm_uncert_df[(fm_uncert_df['go_or_gs'] == 'GS') & (fm_uncert_df['probe'] == probe_vinc)].select_dtypes(
        include=np.number).values[0]
    diff = mm.percent_diff(uncert_gs, uncert_go)
    df_columns_values = [[probe_vinc, 'perc_diff'] + diff.tolist()]

    new_df_row = pd.DataFrame(df_columns_values, columns=df_columns_names)

    # the FoM is instead GO/GS-1
    fom_go = fm_uncert_df[(fm_uncert_df['go_or_gs'] == 'GO') & (fm_uncert_df['probe'] == probe_vinc)]['FoM'].values[0]
    fom_gs = fm_uncert_df[(fm_uncert_df['go_or_gs'] == 'GS') & (fm_uncert_df['probe'] == probe_vinc)]['FoM'].values[0]
    new_df_row['FoM'] = mm.percent_diff(fom_go, fom_gs)

    fm_uncert_df = pd.concat([fm_uncert_df, new_df_row], ignore_index=True)
#
# diff_df = pd.DataFrame(diff, columns=['diff', ] + [param_name for param_name in fiducials_dict_trimmed.keys()][
#                                                   :params_tokeep])
# fm_uncert_df = pd.concat([fm_uncert_df, diff_df], ignore_index=True)

# ratio_columns = df_columns_names[2:]
# ratios = pd.DataFrame()
# for col in ratio_columns:
#     ratios[col] = fm_uncert_df.apply(lambda row: row[col] / fm_uncert_df.loc[
#         (fm_uncert_df['probe'] == row['probe']) & (fm_uncert_df['go_or_gs'] == 'GO'), col].values[0], axis=1)

# for each probe take the numerical values, divide the GS by the GO and append the result in a new row
# df_ratio = pd.DataFrame()
# numerical_columns = fm_uncert_df.select_dtypes(include=np.number).columns
# for col in numerical_columns:
#     go_values = fm_uncert_df.loc[fm_uncert_df['go_or_gs'] == 'GO', col]
#     gs_values = fm_uncert_df.loc[fm_uncert_df['go_or_gs'] == 'GS', col]
#     df_ratio['go_or_gs' == 'diff'][col] = (gs_values.values / go_values.values - 1) * 100
#     df_ratio['go_or_gs'][col] = 'perc_diff'
# df_ratio['probe'][col] = df.loc[df['probe']].values[0]


# perc_diff = mm.percent_diff(uncert_gs[numeric_columns], uncert_go[numeric_columns])
#
# # Add 'probe' and 'go_or_gs' columns to the ratio DataFrame
# perc_diff[['probe', 'go_or_gs']] = uncert_go[['probe', 'go_or_gs']]
#
# # Reorder the columns to match the original DataFrame
# perc_diff = perc_diff[uncert_go.columns]
#
# perc_diff = mm.percent_diff(uncert_gs, uncert_go)

ylabel = r'$(\sigma_{\rm GS}/\sigma_{\rm G} - 1) \times 100$ [%]'
data = fm_uncert_df[fm_uncert_df['go_or_gs'] == 'perc_diff'].select_dtypes(include=np.number).values
title = None
label_list = ('WL', 'GCph', r'$3\times 2$pt')

if include_fom:
    params_tokeep += 1
plot_utils.bar_plot(data, title, label_list, bar_width=0.2, nparams=params_tokeep, param_names_label=None,
                    second_axis=False, no_second_axis_bars=0, superimpose_bars=False, show_markers=False, ylabel=ylabel,
                    include_fom=include_fom)

plt.savefig('../output/plots/WL_vs_GC_vs_3x2pt_GOGS_perc_uncert_increase.pdf', bbox_inches='tight', dpi=600)
