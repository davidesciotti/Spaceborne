import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent

sys.path.append(str(project_path / 'lib'))
import my_module as mm

matplotlib.use('Qt5Agg')

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8)
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()


def FM_check(general_config, FM_dict, tolerance=0.0001):
    # check FMS
    if general_config['which_forecast'] == 'sylvain':
        path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/common_ell_and_deltas/Cij_14may'
    elif general_config['which_forecast'] == 'IST':
        path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/ISTspecs_indVincenzo/Cij_14may'
    elif general_config['which_forecast'] == 'CLOE':
        path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/ISTspecs_indVincenzo/Cij_14may'
    else:
        raise ValueError('this unit test is not suited for this type of forecast')

    FM_d_old = dict(mm.get_kv_pairs(path_import, filetype="txt"))

    for probe in ['WL', 'GC', '3x2pt']:
        for GO_or_GS in ['GO', 'GS']:

            if GO_or_GS == 'GS':
                GO_or_GS_old = 'G+SSC'
            else:
                GO_or_GS_old = 'G'

            if probe == '3x2pt':
                probe_lmax = 'XC'
                probe_lmax2 = 'GC'
            else:
                probe_lmax = probe
                probe_lmax2 = probe

            nbl = general_config['nbl']
            ell_max = general_config[f'ell_max_{probe_lmax2}']

            FM_old = FM_d_old[f'FM_{probe}_{GO_or_GS_old}_lmax{probe_lmax}{ell_max}_nbl{nbl}']
            FM_new = FM_dict[f'FM_{probe}_{GO_or_GS}']

            diff = mm.percent_diff(FM_old, FM_new)
            # mm.matshow(diff, title=f'{probe}, {GO_or_GS}')

            if np.all(np.abs(diff < tolerance)):
                result_symbol = '✅'
            else:
                result_symbol = '❌'

            print(f'is the percent difference between the FM < {tolerance} %?, {probe}, {GO_or_GS}', result_symbol)
