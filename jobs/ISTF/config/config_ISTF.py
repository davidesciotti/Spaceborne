from pathlib import Path
import sys
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/bin')
import utils_running as utils

which_forecast = 'IST'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

BNT_transform = False

general_cfg = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': None,
    'zbins_list': (10,),
    'EP_or_ED': 'EP',
    'zbins_type_list': ('EP', ),
    'n_probes': 2,
    'nbl_WL': 30,
    'nbl_GC': 30,
    'which_forecast': which_forecast,
    'use_WA': False,
    'save_cls_3d': False,
    'save_rls_3d': False,
    'cl_BNT_transform': BNT_transform,
    'BNT_matrix_path': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/BNT_matrix',
    'BNT_matrix_filename': 'BNT_mat_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    'cl_folder': f'{project_path.parent}/common_data/vincenzo/14may/CijDers/' + '{EP_or_ED:s}{zbins:02d}',
    'rl_folder': f'{project_path.parent}/common_data/vincenzo/Pk_responses_2D/' + '{EP_or_ED:s}{zbins:02d}',
    'cl_filename': 'Cij{probe:s}-GR-Flat-eNLA-NA.dat',
    'rl_filename': 'rij{probe:s}corr-istf-alex.dat',
}

if general_cfg['ell_max_WL'] == general_cfg['ell_max_GC']:
    general_cfg['use_WA'] = False

covariance_cfg = {
    'ind_folder': f'{project_path.parent}/common_data/ind_files' + '/{triu_tril:s}_{row_col_wise:s}',
    'ind_filename': 'indices_{triu_tril:s}_{row_col_wise:s}_zbins{zbins:02d}.dat',
    'ind_ordering': ind_ordering,  # TODO deprecate this
    'triu_tril': 'triu',
    'row_col_wise': 'row-wise',
    'block_index': 'ell',
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
    'save_cov_GS': False,
    'save_cov_SS': False,
    'save_cov_dat': False,  # this is the format used by Vincenzo
    'save_2DCLOE': False,  # quite useless, this is not the format used by CLOE
    'cov_folder': f'{job_path}/output/covmat' + '/zbins{zbins:02d}' + '/{triu_tril:s}_{row_col_wise:s}',
}

Sijkl_cfg = {
    'wf_input_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_/KernelFun/magcut_zcut',
    'wf_input_filename': '{which_WF:s}-{EP_or_ED:s}{zbins:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'Sijkl_folder': f'{job_path}/output/Flagship_/BNT_{BNT_transform}/sijkl',
    'Sijkl_filename': 'sijkl_WF-FS01d}_nz{nz:d}_zbins{EP_or_ED:s}{zbins:02}_IA{IA_flag:}'
                      '_MS{magcut_source:02d}-ZS{zcut_source:02d}.npy',
    'WF_normalization': 'IST',
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
}

FM_cfg = {
    'compute_FM': True,
    'nparams_tot': 20,  # total (cosmo + nuisance) number of parameters
    'paramnames_3x2pt': None,  # ! for the time being, these are defined in the main and then passed here
    'save_FM': True,
    'save_FM_as_dict': False,
    'derivatives_BNT_transform': True,
    'derivatives_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_/Derivatives/BNT_{BNT_transform}/' +
                          'ML{magcut_lens:03d}ZL{zcut_lens:02d}MS{magcut_source:03d}ZS{zcut_source:02d}',
    'derivatives_filename': '/BNT_dDVd{param:s}-{probe:s}-{specs:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-'
                            'ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'FM_folder': f'{job_path}/output/Flagship_/BNT_{BNT_transform}/FM',
    'FM_filename': 'FM_{probe:s}_{which_cov:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02}-'
                   'ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.txt',
    'params_order': None,
}
