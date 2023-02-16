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
import ell_values as ell_utils

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params
import mpl_cfg

sys.path.append(f'{job_path}/config')
import config_SPV3 as cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()


def test_cov_FM(SSC_code, to_test):
    """tests that the outputs do not change between the old and the new version"""
    old_dict = dict(mm.get_kv_pairs(f'{job_path}/output/{to_test}/test_benchmarks_{SSC_code}'))
    new_dict = dict(mm.get_kv_pairs(f'{job_path}/output/{to_test}/{SSC_code}'))

    assert old_dict.keys() == new_dict.keys(), 'The number of files or their names has changed'

    for key in old_dict.keys():
        assert np.array_equal(old_dict[key], new_dict[key]), f'The file {key} is different'

    print('test_cov passed successfully ✅')