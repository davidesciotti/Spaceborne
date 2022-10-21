from pathlib import Path
import sys
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = f'{project_path}/jobs/SPV3'

sys.path.append(f'{project_path}/bin')
import utils_running as utils

which_forecast = 'SPV3'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

general_config = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': None,
    'zbins_list': (13, ),
    'EP_or_ED': 'ED',
    'nProbes': 2,
    # 'nbl_WL': 32,
    'nbl_WL_32': 32,
    'which_forecast': which_forecast,
    'cl_folder': cl_folder,
    'use_WA': True,
    'save_cls_3d': True,
    'save_rls_3d': False,
    'specs': 'wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4',
    'cl_BNT_transform': False,
    'cl_input_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_2/DataVectors',
    'rl_input_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_2/ResFunTabs',
}

if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_ordering': ind_ordering,
    'GL_or_LG': GL_or_LG,
    'fsky': fsky,  # ! new
    # 'Rl': 4,
    'block_index': 'ell',  # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'which_probe_response': 'variable',
    'sigma_eps2': (0.26 * np.sqrt(2)) ** 2,  # ! new
    'ng': 28.73,  # ! new
    'compute_covmat': True,
    'save_cov_2D': False,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_SS': False,
    'save_cov_dat': False,  # this is the format used by Vincenzo
    'save_2DCLOE': False,  # quite useless, this is not the format used by CLOE
    'output_folder': '/cl_BNT',
}

Sijkl_config = {
    'wf_input_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_2/KernelFun',
    'sijkl_folder': f'{job_path}/output/sijkl/Flagship_1',  # this is also an input folder, once the sijkl are computed
    'WF_suffix': 'nzFS2',
    'WF_normalization': 'IST',
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': False,
    'save_Sijkl': True,
}

FM_config = {
    'compute_FM': False,
    'nParams': 20,
    'save_FM': True,
    'save_FM_as_dict': True
}
