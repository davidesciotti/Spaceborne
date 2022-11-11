from pathlib import Path
import sys
import numpy as np
project_path = Path.cwd().parent.parent.parent

sys.path.append(f'{project_path}/bin')
import utils_running as utils

which_forecast = 'sylvain'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)


general_config = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': 10,
    'nProbes': 2,
    'nbl': 30,
    'nbl_WL': None,  # it's equal for all the other probes
    'which_forecast': which_forecast,  # ie choose whether to have IST's or sylvain's deltas
    'cl_folder': cl_folder,
    'use_WA': True,
    'save_cls': False,
    'EP_or_ED': 'EP',
}

if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_ordering': ind_ordering,
    'GL_or_LG': GL_or_LG,
    'save_SSC_only_covmats': True,
    'compute_covariance_in_blocks': False,
    'fsky': fsky,
    'Rl': 4,
    'save_covariance': False,
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'block_index': 'ell',
    'which_probe_response': 'constant',
    'sigma_eps2': 0.3 ** 2,
    'ng': 30,
}

FM_config = {
    'nParams': 20,
    'save_FM': False,
    'save_FM_as_dict': False
}