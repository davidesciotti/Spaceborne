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


def check_FMs_against_oldSSCscript(FM_new_path, general_config, covariance_config, tolerance=0.0001):
    """check FMS against old scripts"""

    if covariance_config['which_probe_response'] == 'variable':
        raise Exception('old scripts did not implement variable probe response, nothing to compare the results to')
    if ((general_config['ell_max_WL'], general_config['ell_max_GC']) == (1500, 750)) and general_config[
        'which_forecast'] != 'sylvain':
        raise Exception(
            'old scripts did not implement the pessimistic case unless which_forecasts is "sylvain", apparently')

    if general_config['which_forecast'] == 'sylvain':
        path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/common_ell_and_deltas/Cij_14may'
    elif general_config['which_forecast'] == 'IST':
        path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/ISTspecs_indVincenzo/Cij_14may'
    elif general_config['which_forecast'] == 'CLOE':
        print('WARNING: is this unit test implemented?? path is the same as general_config[which_forecast] == "IST"...')
        path_import = '/Users/davide/Documents/Lavoro/Programmi/SSCcomp_prove/output/FM/ISTspecs_indVincenzo/Cij_14may'
    else:
        raise ValueError('general_config["which_forecast"] must be either "sylvain", "IST" or "CLOE"')

    FM_d_old = dict(mm.get_kv_pairs(path_import, filetype="txt"))
    FM_d_new = dict(mm.get_kv_pairs(FM_new_path, filetype="txt"))

    # TODO plot forecasts to get an idea of the difference: do not just compare the FMs

    for probe in ['WL', 'GC', '3x2pt']:
        for GO_or_GS in ['GO', 'GS']:

            if GO_or_GS == 'GS':
                GO_or_GS_old = 'G+SSC'
                Rl_str = '_Rlconst'
            else:
                GO_or_GS_old = 'G'
                Rl_str = ''

            if probe == '3x2pt':
                probe_lmax = 'XC'
                probe_lmax2 = 'GC'
            else:
                probe_lmax = probe
                probe_lmax2 = probe

            nbl = general_config['nbl']
            ell_max = general_config[f'ell_max_{probe_lmax2}']

            FM_old = FM_d_old[f'FM_{probe}_{GO_or_GS_old}_lmax{probe_lmax}{ell_max}_nbl{nbl}']
            FM_new = FM_d_new[f'FM_{probe}_{GO_or_GS}_lmax{probe_lmax}{ell_max}_nbl{nbl}{Rl_str}']

            diff = mm.percent_diff(FM_old, FM_new)
            # mm.matshow(diff, title=f'{probe}, {GO_or_GS}')

            # perform 2 separate tests, manually computing the difference and using np.allclose
            for test_result_bool in [np.all(np.abs(diff) < tolerance), np.allclose(FM_old, FM_new)]:

                # transform into nicer and easier to understand string (symbol)
                if test_result_bool:
                    result_emoji = '✅'
                else:
                    result_emoji = '❌'

                print(f'is the percent difference between the FM < {tolerance}?, {probe}, {GO_or_GS}, {result_emoji}')


def test_cov_FM(output_path, benchmarks_path):
    """tests that the outputs do not change between the old and the new version"""
    old_dict = dict(mm.get_kv_pairs_npy(benchmarks_path))
    new_dict = dict(mm.get_kv_pairs_npy(output_path))

    assert old_dict.keys() == new_dict.keys(), 'The number of files or their names has changed'

    for key in old_dict.keys():
        assert np.array_equal(old_dict[key], new_dict[key]), f'The file {key} is different'

    print('tests passed successfully ✅')