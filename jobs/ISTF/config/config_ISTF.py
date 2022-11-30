from pathlib import Path
import sys
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/bin')
import utils_running as utils

which_forecast = 'IST'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

# ! choose the flagship version and whether you want to compute the BNT transformed cls
flagship_version = None
BNT_transform = False

if flagship_version == 1:
    assert BNT_transform is False, 'we are applying the BNT only for Flagship_2'

if BNT_transform:
    assert flagship_version == 2, 'we are applying the BNT only for Flagship_2'

general_config = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,

    'ell_max_GC': 3000,
    'zbins': None,
    'zbins_list': (10,),
    'EP_or_ED': 'EP',
    'zbins_type_list': ('EP', ),
    'nProbes': 2,
    'nbl': 30,
    'nbl_WL': 30,  # useless??
    'nbl_WL_32': 32,
    'which_forecast': which_forecast,
    'use_WA': False,
    'save_cls_3d': False,
    'save_rls_3d': False,
    'specs': None,
    'cl_BNT_transform': BNT_transform,
    'BNT_matrix_path': None,
    'BNT_matrix_filename': None,
    'cl_folder': f'{project_path.parent}/common_data/vincenzo/14may/CijDers',
    'rl_folder': f'{project_path.parent}/common_data/vincenzo/Pk_responses_2D',
}

if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_folder': f'{project_path.parent}/common_data/ind_files/' + '{triu_tril:s}_{row_col_wise:s}',
    'ind_filename': 'indices_{triu_tril:s}_{row_col_wise:s}_zbins{zbins:02d}.dat',
    'ind_ordering': ind_ordering,  # TODO deprecate this
    'triu_tril': 'triu',
    'row_col_wise': 'row-wise',
    'ind_name': 'vincenzo',
    'GL_or_LG': GL_or_LG,
    'fsky': fsky,
    # 'Rl': 4,
    'block_index': 'ell',
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'which_probe_response': 'variable',
    'ng': 30,
    'ng_folder': None,
    'sigma_eps2': 0.3 ** 2,
    'compute_covmat': True,
    'save_cov_2D': True,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_SS': False,
    'save_cov_dat': False,  # this is the format used by Vincenzo
    'save_2DCLOE': False,  # quite useless, this is not the format used by CLOE
    'cov_folder': f'{job_path}/output/covmat' + '/zbins{zbins:02d}' + '/{triu_tril:s}_{row_col_wise:s}',
}

Sijkl_config = {
    'wf_input_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/KernelFun',
    'Sijkl_folder': f'{job_path}/output/Flagship_{flagship_version}/BNT_{BNT_transform}/sijkl',
    # this is also an input folder, once the sijkl are computed
    'WF_suffix': f'FS{flagship_version}',
    'WF_normalization': 'IST',
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,
    'save_Sijkl': False,
}

FM_config = {
    'compute_FM': False,
    'nParams': 20,
    'save_FM': False,
    'save_FM_as_dict': False,
    'derivatives_folder': None,  # TODO: add the derivatives folder
    'FM_output_folder': f'{job_path}/output/Flagship_{flagship_version}/BNT_{BNT_transform}/FM',
}
