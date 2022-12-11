import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm

sys.path.append(f'{project_path}/bin')
import ell_values_running as ell_utils

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params
import mpl_cfg

sys.path.append(f'{job_path}/config')
import config_SPV3 as cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()


def test_cov():
    """tests that the outputs do not change between the old and the new version"""
    cov_old_dict = dict(mm.get_kv_pairs_npy(f'{job_path}/output/covmat/test_benchmarks'))
    cov_new_dict = dict(mm.get_kv_pairs_npy(f'{job_path}/output/covmat'))

    assert cov_old_dict.keys() == cov_new_dict.keys(), 'The number of files or theit names has changed'

    for key in cov_old_dict.keys():
        assert np.array_equal(cov_old_dict[key], cov_new_dict[key]), f'The covmat {key} is different'
    print('test_cov passed successfully ✅')


def test_FM():
    """tests that the outputs do not change between the old and the new version"""
    FM_old_dict = dict(mm.get_kv_pairs_npy(f'{job_path}/output/FM/test_benchmarks'))
    FM_new_dict = dict(mm.get_kv_pairs_npy(f'{job_path}/output/FM'))

    assert FM_old_dict.keys() == FM_new_dict.keys(), 'The number of files or theit names has changed'

    for key in FM_old_dict.keys():
        assert np.array_equal(FM_old_dict[key], FM_new_dict[key]), f'The FM {key} is different'

    print('test_FM passed successfully ✅')
