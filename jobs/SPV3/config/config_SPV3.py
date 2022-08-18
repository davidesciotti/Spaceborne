from pathlib import Path
import sys
import numpy as np
project_path = Path.cwd().parent.parent.parent

sys.path.append(f'{project_path}/bin')
import utils_running as utils

which_forecast = 'SPV3'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

general_config = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': 10,
    'EP_or_ED': 'EP',
    'nProbes': 2,
    # 'nbl_WL': 32,
    'nbl_WL_32': 32,
    'which_forecast': which_forecast,
    'cl_folder': cl_folder,
    'use_WA': True,
    'save_cls_3d': True,
    'save_rls_3d': True,
    'specs': 'wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4',
}

if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_ordering': ind_ordering,
    'GL_or_LG': GL_or_LG,
    'save_SSC_only_covmats': False,
    'compute_covariance_in_blocks': False,
    'fsky': fsky,  # ! new
    'Rl': 4,
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'block_index': 'ell',
    'which_probe_response': 'variable',
    'sigma_eps2': (0.3 * np.sqrt(2)) ** 2,  # ! new
    'ng': 28.73,  # ! new
    'save_covariance': True,
    'save_covariance_dat': True,
}

Sijkl_config = {
    'input_WF': 'vincenzo_SPV3',
    'WF_normalization': 'IST',
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,
    'save_Sijkl': False,
}

FM_config = {
    'nParams': 20,
    'save_FM': True,
    'save_FM_as_dict': True
}
