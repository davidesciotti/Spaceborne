import sys
import time
from operator import itemgetter
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

import matplotlib
import matplotlib.pyplot as plt

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append('../config')
sys.path.append('../../bin/plot_FM_running')

import my_module as mm

import plots_FM_running as plot_utils
import config_SSCpaper_final as cfg

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8),
          'backend': 'Qt5Agg'
          }
plt.rcParams.update(params)
markersize = 10

FM_cfg = cfg.FM_cfg
with open('/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config/'
          'fiducial_params_dict_for_FM.yml') as f:
    fiducials_dict = yaml.safe_load(f)

plot_sylvain = True
plot_ISTF = False
specs_str = 'idMag0-idRSD0-idFS0-idSysWL3-idSysGC4'
dpi = 500
pic_format = 'pdf'

probe_vinc = 'GCO'
nbl_WL_opt = 32
ep_or_ed = 'EP'
zbins = 10
params_tokeep = 7
go_or_gs_folder_dict = {
    'GO': 'GaussOnly',
    'GS': 'GaussSSC',
}

fm_uncert_df = pd.DataFrame()
for go_or_gs in ['GO', 'GS']:
    for probe_vinc in ['WLO', 'GCO', '3x2pt']:
        fm_path = f'/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/Flagship_1_restored/' \
                  f'FishMat_restored/{go_or_gs_folder_dict[go_or_gs]}/{probe_vinc}/FS1NoCuts'
        fm_dict = dict(mm.get_kv_pairs(f'{fm_path}', extension="dat"))
        fm_name = f'fm-{probe_vinc}-{nbl_WL_opt}-wzwaCDM-NonFlat-GR-TB-' \
                  f'{specs_str}-{ep_or_ed}{zbins}'

        fm = fm_dict[fm_name]

        assert len(fiducials_dict) == fm.shape[0] == fm.shape[1], 'Wrong shape of FM matrix!'
        param_names = list(fiducials_dict.keys())
        param_values = list(fiducials_dict.values())

        # fix some of the parameters (i.e., which columns to remove)
        names_params_to_fix = ['Om_Lambda0'] + [f'dz{zi + 1}_photo' for zi in range(zbins)]
        fm, fiducials_dict_trimmed = mm.mask_fm_v2(fm, fiducials_dict, names_params_to_fix, remove_null_rows_cols=True)
        uncert_fm = mm.uncertainties_fm_v2(fm, fiducials_dict_trimmed, which_uncertainty='marginal', normalize=True,
                                           percent_units=True)[:params_tokeep]

        df_columns_names = ['probe', 'go_or_gs'] + [param_name for param_name in fiducials_dict_trimmed.keys()][:params_tokeep]
        df_columns_values = [probe_vinc, go_or_gs] + [uncert for uncert in uncert_fm]
        df_column_values_reshaped = np.array(df_columns_values).reshape(1, -1)

        assert len(df_columns_names) == len(df_columns_values), 'Wrong number of columns!'

        fm_uncert_df_to_concat = pd.DataFrame(df_column_values_reshaped, columns=df_columns_names)
        fm_uncert_df = pd.concat([fm_uncert_df, fm_uncert_df_to_concat], ignore_index=True)


fm_go_GC = fm_uncert_df[(fm_uncert_df['probe'] == 'GCO') & (fm_uncert_df['go_or_gs'] == 'GO')].drop(['probe', 'go_or_gs'], axis=1).values[0]
fm_gs_GC = fm_uncert_df[(fm_uncert_df['probe'] == 'GCO') & (fm_uncert_df['go_or_gs'] == 'GS')].drop(['probe', 'go_or_gs'], axis=1).values[0]
data = np.vstack((fm_go_GC, fm_gs_GC))
title = 'GC'
label_list = ['GO', 'GS']
param_names_label = list(fiducials_dict_trimmed.keys())[:params_tokeep]
plot_utils.bar_plot(data, title, label_list, bar_width=0.18, nparams=params_tokeep, param_names_label=param_names_label,
             second_axis=False, no_second_axis_bars=0, superimpose_bars=True)

