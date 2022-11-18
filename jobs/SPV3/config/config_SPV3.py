from pathlib import Path
import sys
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/bin')
import utils_running as utils

which_forecast = 'SPV3'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

# ! choose the flagship version and whether you want to compute the BNT transformed cls
flagship_version = 1
BNT_transform = False

if flagship_version == 1:
    assert BNT_transform is False, 'we are applying the BNT only for Flagship_2'
    cl_filename = 'dv-{probe:s}-{nbl_WL_opt:02d}-{specs:s}-{EP_or_ED:s}{zbins:02d}',
    rl_filename = 'rf-{probe:s}-{nbl_WL_opt:02d}-{specs:s}-{EP_or_ED:s}{zbins:02d}',

if BNT_transform:
    assert flagship_version == 2, 'we are applying the BNT only for Flagship_2'
    cl_filename = 'dv-{probe:s}-Opt-{specs:s}-{EP_or_ED:s}{zbins:02d}-FS{flagship_version:s}',
    rl_filename = 'rf-{probe:s}-Opt-{specs:s}-{EP_or_ED:s}{zbins:02d}-FS{flagship_version:s}',

general_config = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': None,
    'zbins_list': (13,),
    'EP_or_ED': 'ED',
    'nProbes': 2,
    # 'nbl_WL': 32,
    'nbl_WL_32': 32,
    'which_forecast': which_forecast,
    'use_WA': True,
    'save_cls_3d': True,
    'save_rls_3d': True,
    'specs': 'wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4',
    'cl_BNT_transform': BNT_transform,
    'BNT_matrix_path': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/BNT_matrix',
    'BNT_matrix_filename': f'BNT_matrix_csv_version.txt',
    'cl_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/DataVectors' + '/{probe:s}',
    'rl_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/ResFunTabs' + '/{probe:s}',
    'cl_filename': 'dv-{probe:s}-{nbl_WL_opt:02d}-{specs:s}-{EP_or_ED:s}{zbins:02d}.dat',
    'rl_filename': 'rf-{probe:s}-{nbl_WL_opt:02d}-{specs:s}-{EP_or_ED:s}{zbins:02d}.dat',
    # 'cl_filename_FS2': 'dv-{probe:s}-Opt-{specs:s}-{EP_or_ED:s}{zbins:02d}-FS{flagship_version:s}.dat',
    # 'rl_filename_FS2': 'rf-{probe:s}-Opt-{specs:s}-{EP_or_ED:s}{zbins:02d}-FS{flagship_version:s}.dat',
    'flagship_version': flagship_version,
}

if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_folder': f'{project_path.parent}/common_data/ind_files/variable_zbins/{ind_ordering}_like',
    'ind_filename': 'indices_{ind_ordering:s}_like_zbins{zbins:02d}.dat',
    'ind_ordering': ind_ordering,
    'GL_or_LG': GL_or_LG,
    'fsky': fsky,  # ! new
    # 'Rl': 4,
    'block_index': 'ell',
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'which_probe_response': 'variable',
    'ng': None,  # ! the new value is 28.73 (for Flagship_1), but I'm taking the value from the ngbTab files
    'ng_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/InputNz/Lenses/Flagship',
    'ng_filename': 'ngbTab-{EP_or_ED:s}{zbins:02d}.dat',
    'sigma_eps2': (0.26 * np.sqrt(2)) ** 2,  # ! new
    'compute_covmat': True,
    'cov_file_format': 'npz',  # or npy
    'save_cov_2D': False,
    'save_cov_4D': False,
    'save_cov_6D': True,  # or 10D for the 3x2pt
    'save_cov_SS': False,
    'save_cov_dat': False,  # this is the format used by Vincenzo
    'save_2DCLOE': False,  # quite useless, this is not the format used by CLOE
    'cov_folder': f'{job_path}/output/Flagship_{flagship_version}/BNT_{BNT_transform}/covmat',
}

Sijkl_config = {
    'wf_input_folder': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_{flagship_version}/KernelFun',
    'wf_input_filename': '{which_WF:s}-{EP_or_ED:s}{zbins:02d}.dat',
    'Sijkl_folder': f'{job_path}/output/Flagship_{flagship_version}/BNT_{BNT_transform}/sijkl',
    'Sijkl_filename': 'sijkl_WF-FS{flagship_version:01d}_nz{nz:d}_zbins{EP_or_ED:s}{zbins:02}_IA{IA_flag:}.npy',
    'WF_normalization': 'IST',
    'IA_flag': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
}

FM_config = {
    'compute_FM': False,
    'nParams': 20,
    'save_FM': False,
    'save_FM_as_dict': False,
    'derivatives_folder': None,  # TODO: add the derivatives folder
    'FM_output_folder': f'{job_path}/output/Flagship_{flagship_version}/BNT_{BNT_transform}/FM',
}
