from pathlib import Path
import sys
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/bin')
import utils_running as utils

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTFfid

cfg_name = 'cl14may'
which_forecast = 'ISTF'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

cl_BNT_transform = False
cov_BNT_transform = False
deriv_BNT_transform = False

if cl_BNT_transform or cov_BNT_transform or deriv_BNT_transform:
    BNT_transform = True
else:
    BNT_transform = False

# settings for SSC comparison (aka 'sylvain'):
# survey_area_deg2 = 15469.86  # deg^2
# use_WA: False
# + different deltas...

general_cfg = {
    'cfg_name': cfg_name,
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'ell_max_XC': 3000,
    'zbins': 10,
    'zbins_list': None,
    'EP_or_ED': 'EP',
    'zbins_type_list': ('EP',),
    'n_probes': 2,
    'nbl_WL': 30,
    'nbl_GC': 30,
    'nbl_3x2pt': 30,
    'use_WA': True,  # ! xxx
    'save_cls_3d': False,
    'save_rls_3d': False,
    'ell_cuts': False,
    'cl_BNT_transform': cl_BNT_transform,
    'BNT_transform': BNT_transform,
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
    'ind_folder': f'{project_path.parent}/common_data/ind_files' + '/{triu_tril:s}_{row_col_major:s}',
    'ind_filename': 'indices_{triu_tril:s}_{row_col_major:s}_zbins{zbins:02d}.dat',
    'triu_tril': 'triu',
    'row_col_major': 'row-major',
    'GL_or_LG': 'GL',
    'fsky': fsky,
    'block_index': 'ell',
    'which_probe_response': 'variable',
    'response_const_value': None,  # it used to be 4 for a constant probe response, which this is wrong
    'SSC_code': 'PySSC',  # PySSC or PyCCL
    'ng': 30,
    'ng_folder': None,
    'ng_filename': None,
    'sigma_eps2': 0.3 ** 2,

    'cov_BNT_transform': cov_BNT_transform,
    'compute_covmat': True,
    'compute_cov_6D': True,
    'save_cov_2D': True,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_GS': False,
    'save_cov_SSC': False,
    'save_2DCLOE': False,
    'cov_file_format': 'npz',
    'save_cov_dat': False,
    'cov_folder': str(job_path) + f'/output/{cfg_name}/' + 'covmat/{SSC_code:s}',
    'cov_filename': 'covmat_{which_cov:s}_{probe:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02d}_{ndim:d}D.npy',
    'cov_SSC_PyCCL_folder': f'{project_path.parent}/PyCCL_SSC/output/covmat',
    'cov_SSC_PyCCL_filename': 'cov_PyCCL_SSC_{probe:s}_nbl{nbl:d}_ellsISTF_ellmax{ell_max:d}_hm_recipeKiDS1000_6D.npy',  # TODO these 2 filenames could be unified...
    # TODO these 2 filenames could be unified...
}

Sijkl_cfg = {
    'wf_input_folder': f'{project_path.parent}/common_data/everyones_WF_from_Gdrive/davide/' + 'nz{nz:d}/gen2022',
    'wf_WL_input_filename': 'wil_dav_IA{has_IA:s}_{normalization:s}_nz{nz:d}_bia{bIA:.02f}.txt',
    'wf_GC_input_filename': 'wig_dav_{normalization:s}_nz{nz:d}.txt',
    'Sijkl_folder': f'{project_path.parent}/common_data/Sijkl',
    'Sijkl_filename': 'Sijkl_WFdavide_nz{nz:d}_IA_3may.npy',
    'wf_normalization': 'IST',
    'nz': 10_000,
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
}

# dictionaries of cosmological parameters' names and values
param_names_dict = {
    'cosmo': ["Om", "Ob", "wz", "wa", "h", "ns", "s8"],
    'IA': ["Aia", "eIA", "bIA"],
    'galaxy_bias': [f'b{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)],
}
# fiducial values
fiducials_dict = {
    'cosmo': [ISTFfid.primary['Om_m0'], ISTFfid.primary['Om_b0'], ISTFfid.primary['w_0'], ISTFfid.primary['w_a'],
              ISTFfid.primary['h_0'], ISTFfid.primary['n_s'], ISTFfid.primary['sigma_8']],
    'IA': [ISTFfid.IA_free['A_IA'], ISTFfid.IA_free['eta_IA'], ISTFfid.IA_free['beta_IA']],
    'galaxy_bias': [ISTFfid.photoz_galaxy_bias[f'bL{zbin:02d}_photo'] for zbin in range(1, general_cfg['zbins'] + 1)],
}


param_names_3x2pt = param_names_dict['cosmo'] + param_names_dict['IA'] + param_names_dict['galaxy_bias']
# this needs to be done outside the dictionary creation
fiducials_3x2pt = np.concatenate((fiducials_dict['cosmo'], fiducials_dict['IA'], fiducials_dict['galaxy_bias']))
assert len(param_names_3x2pt) == len(fiducials_3x2pt), "the fiducial values list and parameter names should have the " \
                                                       "same length"
nparams_tot = len(param_names_3x2pt)

FM_cfg = {
    'compute_FM': True,

    'param_names_dict': param_names_dict,
    'fiducials_dict': fiducials_dict,
    'nparams_tot': nparams_tot,  # total (cosmo + nuisance) number of parameters
    'param_names_3x2pt': param_names_3x2pt,  # ! for the time being, these are defined in the main and then passed here

    'save_FM_txt': True,
    'save_FM_dict': True,
    'FM_folder': str(job_path) + f'/output/{cfg_name}/' + 'FM/{SSC_code:s}',
    'FM_txt_filename': 'FM_{probe:s}_{which_cov:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02}',
    'FM_dict_filename': 'FM_dict_zbins{EP_or_ED:s}{zbins:02}',

    'derivatives_folder': f'{project_path.parent}/common_data/vincenzo/14may/CijDers/' + '{EP_or_ED:s}{zbins:02d}',
    'derivatives_prefix': 'dCij{probe:s}d',
    'derivatives_suffix': '-GR-Flat-eNLA-NA',  # I'd like to use this, but instead:
    'derivatives_BNT_transform': deriv_BNT_transform,
}
