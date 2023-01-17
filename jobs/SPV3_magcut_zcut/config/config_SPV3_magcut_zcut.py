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

whos_BNT = '/davide'  # "/stefano "or "/davide"

if BNT_transform is False:
    whos_BNT = ''

assert flagship_version == 2, 'the files for the multicut case are only available for Flagship_2'

if BNT_transform:
    assert flagship_version == 2, 'we are applying the BNT only for Flagship_2'

general_cfg = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
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
    'cl_BNT_transform': cl_BNT_transform,
    'cov_BNT_transform': cov_BNT_transform,
    'deriv_BNT_transform': deriv_BNT_transform,
    'BNT_transform': BNT_transform,
    'whos_BNT': whos_BNT,
    'BNT_matrix_path': f'{SPV3_folder}/BNT_matrix',
    'BNT_matrix_filename': 'BNT_mat_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    'cl_folder': f'{SPV3_folder}/Flagship_{flagship_version}/DataVectors/magcut_zcut',
    'rl_folder': f'{SPV3_folder}/Flagship_{flagship_version}/ResFunTabs/magcut_zcut',
    'cl_filename': 'dv-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'rl_filename': 'rf-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'magcut_lens_list': (230, 235, 240, 245, 250),
    'magcut_source_list': (245,),
    'zcut_lens_list': (0, 2),
    'zcut_source_list': (0, 2),
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
    'ind_ordering': ind_ordering,  # TODO deprecate this
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
    'compute_covmat': True,
    'cov_file_format': 'npz',  # or npy
    'save_cov_2D': False,
    'save_cov_4D': False,
    'save_cov_6D': True,  # or 10D for the 3x2pt
    'save_cov_GS': False,
    'save_cov_SS': False,
    'save_cov_dat': False,  # this is the format used by Vincenzo
    'save_2DCLOE': False,  # outermost loop is on the probes
    'cov_folder': f'{job_path}/output/Flagship_{flagship_version}/covmat/BNT_{BNT_transform}{whos_BNT}' + '/zbins{zbins:02d}',
    'cov_BNTstef_folder': f'{SPV3_folder}/Flagship_{flagship_version}/CovMats/BNT_True/stefano/magcut_zcut/' + '{GO_or_GS:s}/{probe:s}',
    'cov_BNTstef_filename': 'BNT_covmat_{GO_or_GS}_{probe}_{block:s}_lmax{ellmax}_nbl{nbl}_zbins{EP_or_ED:s}{zbins:02d}'
                            '_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}_6D.npy'

    # 'cov_folder': f'{SPV3_folder}/Flagship_{flagship_version}/CovMats/BNT_True/produced_by_stefano/magcut_zcut'
    # 'cov_folder': f'/Volumes/4TB/covmat_cuts',
    # 'cov_filename': 'covmat_{GO_or_GS:s}_{probe:s}_lmax{lmax:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02d}'
    #                '_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:03d}',
    # TODO add cov_filename!!
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
    'nparams_tot': 20,  # total (cosmo + nuisance) number of parameters
    'paramnames_3x2pt': None,  # ! for the time being, these are defined in the main and then passed here
    'save_FM': False,
    'save_FM_as_dict': True,
    'transformed_derivs_folder': f'{SPV3_folder}/Flagship_{flagship_version}/Derivatives/BNT_{BNT_transform}{whos_BNT}',
    'derivatives_BNTstef_folder': f'{SPV3_folder}/Flagship_{flagship_version}/Derivatives/BNT_{BNT_transform}/stefano/' + '{probe:s}',
    'derivatives_folder': f'{SPV3_folder}/Flagship_{flagship_version}/Derivatives/BNT_False/' +
                          'ML{magcut_lens:03d}ZL{zcut_lens:02d}MS{magcut_source:03d}ZS{zcut_source:02d}',
    'derivatives_filename': 'BNT_dDVd{param:s}-{probe:s}-{specs:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-'
                            'ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'derivatives_prefix': 'dDVd',
    # the filename is the same as above
    'FM_folder': f'{job_path}/output/Flagship_{flagship_version}/FM/BNT_{BNT_transform}{whos_BNT}',
    'FM_filename': 'FM_{probe:s}_{which_cov:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02}-'
                   'ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.txt',
    'params_order': None,
}
