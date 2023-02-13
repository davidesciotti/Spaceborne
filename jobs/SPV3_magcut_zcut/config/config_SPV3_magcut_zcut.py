from pathlib import Path
import sys
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/bin')
import utils_running as utils

which_forecast = 'SPV3'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

SPV3_folder = f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022'

# ! choose the flagship version and whether you want to use the BNT transform
flagship_version = 2

cl_BNT_transform = False
cov_BNT_transform = True
deriv_BNT_transform = True

if cl_BNT_transform or cov_BNT_transform or deriv_BNT_transform:
    BNT_transform = True
else:
    BNT_transform = False

assert flagship_version == 2, 'the files for the multicut case are only available for Flagship_2'

if BNT_transform:
    assert flagship_version == 2, 'we are applying the BNT only for Flagship_2'

general_cfg = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'ell_max_XC': 3000,
    'zbins': 13,
    'zbins_list': None,
    'EP_or_ED': 'ED',
    'n_probes': 2,
    # 'nbl_WL': 32,
    'nbl_WL_opt': 32,  # the case with the largest range, i.e. the reference ell binning from which the cuts are applied
    'which_forecast': which_forecast,
    'use_WA': False,
    'save_cls_3d': False,
    'save_rls_3d': False,
    'ell_cuts': True,
    'ell_cuts_folder': f'{SPV3_folder}/ell_cuts',
    'ell_cuts_filename': 'lmax_cut_{probe:s}_{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-'
                         'ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'kmax_ref_h_over_Mpc': 1.0,
    # 'kmax_list_1_over_Mpc': np.array((0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 3.00, 4.00, 5.00, 10.00)),
    'kmax_list_h_over_Mpc': np.array([0.1675, 0.335, 0.5025, 0.67, 0.8375, 1.005, 1.1725, 1.34, 2.01, 2.68, 3.35, 6.7]),
    'cl_BNT_transform': cl_BNT_transform,
    'BNT_transform': BNT_transform,
    'BNT_matrix_path': f'{SPV3_folder}/BNT_matrix',
    'BNT_matrix_filename': 'BNT_mat_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    'cl_folder': f'{SPV3_folder}/Flagship_{flagship_version}/DataVectors/magcut_zcut_True',
    'rl_folder': f'{SPV3_folder}/Flagship_{flagship_version}/ResFunTabs/magcut_zcut_True',
    'cl_filename': 'dv-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'rl_filename': 'rf-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'magcut_lens_list': (245,),
    'magcut_source_list': (245,),
    'zcut_lens_list': (0,),
    'zcut_source_list': (0, ),
    'zmax': 2.5,
    'magcut_source': None,
    'magcut_lens': None,
    'zcut_source': None,
    'zcut_lens': None,
    'flagship_version': flagship_version,
}

if general_cfg['ell_max_WL'] == general_cfg['ell_max_GC']:
    general_cfg['use_WA'] = False

covariance_cfg = {
    'ind_folder': f'{project_path.parent}/common_data/ind_files' + '/{triu_tril:s}_{row_col_major:s}',
    'ind_filename': 'indices_{triu_tril:s}_{row_col_major:s}_zbins{zbins:02d}.dat',
    'triu_tril': 'triu',
    'row_col_major': 'row-major',
    'block_index': 'ell',
    'GL_or_LG': GL_or_LG,
    'fsky': fsky,  # ! new
    # 'Rl': 4,
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'SSC_code': 'PySSC',
    'which_probe_response': 'variable',
    'ng': None,  # ! the new value is 28.73 (for Flagship_1), but I'm taking the value from the ngbTab files
    'ng_folder': f'{SPV3_folder}/Flagship_{flagship_version}/InputNz/magcut_zcut',
    'ng_filename': 'ngbsTab-{EP_or_ED:s}{zbins:02d}-zedMin{zcut_source:02d}-zedMax{zmax:02d}-mag{magcut_source:03d}.dat',
    'sigma_eps2': (0.26 * np.sqrt(2)) ** 2,  # ! new
    'cov_BNT_transform': cov_BNT_transform,
    'compute_covmat': True,
    'compute_cov_6D': True,  # or 10D for the 3x2pt
    'cov_file_format': 'npz',  # or npy
    'save_cov_2D': False,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_GS': False,
    'save_cov_SS': False,
    'save_2DCLOE': False,  # outermost loop is on the probes
    'save_cov_dat': False,  # this is the format used by Vincenzo
    'cov_folder': f'{job_path}/output/Flagship_{flagship_version}/covmat/BNT_{BNT_transform}' + '/ell_cuts_{ell_cuts:s}/zbins{EP_or_ED:s}{zbins:02d}',
    'cov_filename': 'covmat_{which_cov:s}_{probe:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02d}_'
                    'ML{magcut_lens:03d}_ZL{zcut_lens:02d}_'
                    'MS{magcut_source:03d}_ZS{zcut_source:02d}_{ndim:d}D.{extension:s}',
    'cov_filename_vincenzo': 'cm-{probe_vinc:s}-{GOGS_filename:s}-{nbl_WL:d}-{EP_or_ED:s}{zbins:02d}-'
                             'ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
}

Sijkl_cfg = {
    'wf_input_folder': f'{SPV3_folder}/Flagship_{flagship_version}/KernelFun/magcut_zcut',
    'wf_WL_input_filename': 'WiWL-{EP_or_ED:s}{zbins:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'wf_GC_input_filename': 'WiGC-{EP_or_ED:s}{zbins:02d}-ML{magcut_source:03d}-ZL{zcut_source:02d}.dat',
    'Sijkl_folder': f'{job_path}/output/Flagship_{flagship_version}/sijkl',
    'Sijkl_filename': 'sijkl_WF-FS{flagship_version:01d}_nz{nz:d}_zbins{EP_or_ED:s}{zbins:02}_IA{IA_flag:}'
                      '_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    'WF_normalization': 'IST',
    'IA_flag': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
}

FM_cfg = {
    'compute_FM': True,
    'save_FM_txt': True,
    'save_FM_dict': True,
    'nparams_tot': 20,  # total (cosmo + nuisance) number of parameters
    'param_names_3x2pt': None,  # ! for the time being, these are defined in the main and then passed here
    'save_FM_as_dict': True,
    'derivatives_folder': f'{SPV3_folder}/Flagship_{flagship_version}/Derivatives/BNT_False/' +
                          'ML{magcut_lens:03d}ZL{zcut_lens:02d}MS{magcut_source:03d}ZS{zcut_source:02d}',
    'derivatives_filename': 'BNT_dDVd{param:s}-{probe:s}-{specs:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-'
                            'ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'derivatives_prefix': 'dDVd',
    'deriv_BNT_transform': deriv_BNT_transform,
    # the filename is the same as above
    'FM_folder': f'{job_path}/output/Flagship_{flagship_version}/FM/BNT_{BNT_transform}/ell_cuts_' + '{ell_cuts:s}',
    'FM_txt_filename': 'FM_{probe:s}_{which_cov:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02}-'
                       'ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}'
                       '_kmax_h_over_Mpc{kmax_h_over_Mpc:03f}',
    'FM_dict_filename': 'FM_zbins{EP_or_ED:s}{zbins:02}-ML{magcut_lens:03d}-ZL{zcut_lens:02d}-'
                        'MS{magcut_source:03d}-ZS{zcut_source:02d}_kmax_h_over_Mpc{kmax_h_over_Mpc:03f}',
    'params_order': None,
}
