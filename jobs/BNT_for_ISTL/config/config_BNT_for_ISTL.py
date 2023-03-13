from pathlib import Path

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

survey_area_deg2 = 15_000  # deg^2

deg2_in_sphere = 41252.96  # deg^2 in a spere
fsky = survey_area_deg2 / deg2_in_sphere
cfg_name = 'cl15gen'

# settings for SSC comparison (aka 'sylvain'):
# survey_area_deg2 = 15469.86  # deg^2
# use_WA: False

general_cfg = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 5000,
    'ell_max_XC': 5000,
    'zbins': 10,
    'EP_or_ED': 'EP',
    'n_probes': 2,
    'nbl_WL': 20,
    'nbl_GC': 20,
    'use_WA': False,
    'save_cls_3d': False,
    'save_rls_3d': False,
    'cl_BNT_transform': False,
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
    # this is the one used by me and Vincenzo. The blocks in the 2D covmat will be indexed by ell1, ell2
    'which_probe_response': 'constant',
    'response_const_value': 4,  # it used to be 4 for a constant probe response, which this is wrong
    'SSC_code': 'PySSC',  # PySSC or PyCCL
    'cov_BNT_transform': None,
    'ng': 30,
    'ng_folder': None,
    'ng_filename': None,
    'sigma_eps2': 0.3 ** 2,
    'compute_covmat': True,
    'compute_cov_6D': True,
    'save_cov_2D': True,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_GS': False,
    'save_cov_SSC': False,
    'save_2DCLOE': True,
    'cov_file_format': 'npy',
    'cov_folder': str(job_path) + f'/output/{cfg_name}/' + 'covmat/{SSC_code:s}',
    'cov_filename': 'covmat_{which_cov:s}_{probe:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02d}_{ndim:d}D.npy',
    'cov_SSC_PyCCL_folder': f'{project_path.parent}/PyCCL_SSC/output/covmat',
    'cov_SSC_PyCCL_filename': 'cov_PyCCL_SSC_{probe:s}_nbl{nbl:d}_ellsISTF_ellmax{ell_max:d}_hm_recipeKiDS1000_6D.npy',
    # TODO these 2 filenames could be unified...
}

Sijkl_cfg = {
    'wf_input_folder': f'{project_path.parent}/common_data/everyones_WF_from_Gdrive/davide/' + 'nz{nz:d}/gen2022',
    'wil_filename': 'wil_dav_IA{has_IA:s}_{normalization:s}_nz{nz:d}_bia{bIA:.02f}.txt',
    'wig_filename': 'wig_dav_{normalization:s}_nz{nz:d}.txt',
    'Sijkl_folder': f'{project_path.parent}/common_data/Sijkl',
    'Sijkl_filename': 'Sijkl_WFdavide_nz{nz:d}_IA_3may.npy',
    'wf_normalization': 'IST',
    'nz': 10_000,
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
}

# define the parameters outside the dictionary, it's more convenient
paramnames_cosmo = ["Om", "Ob", "wz", "wa", "h", "ns", "s8"]
paramnames_IA = ["Aia", "eIA", "bIA"]
paramnames_galbias = [f'b{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)]
paramnames_3x2pt = paramnames_cosmo + paramnames_IA + paramnames_galbias
nparams_total = len(paramnames_3x2pt)

FM_cfg = {
    'compute_FM': True,
    'nparams_tot': 20,  # total (cosmo + nuisance) number of parameters
    'save_FM': True,
    'save_FM_as_dict': True,
    'derivatives_BNT_transform': True,
    'derivatives_folder': f'{project_path.parent}/common_data/vincenzo/thesis_data/Cij_derivatives_tesi/new_names/',
    'derivatives_prefix': 'dCij{probe:s}d',
    'derivatives_suffix': '-N4TB-GR-eNLA',  # I'd like to use this, but instead:
    'FM_folder': str(job_path) + f'/output/{cfg_name}/' + 'FM/{SSC_code:s}',
    'FM_filename': 'FM_{probe:s}_{which_cov:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02}.txt',
    'params_order': None,
    'paramnames_cosmo': paramnames_cosmo,
    'paramnames_IA': paramnames_IA,
    'paramnames_galbias': paramnames_galbias,
    'paramnames_3x2pt': paramnames_3x2pt,
    'nparams_tot': nparams_total,
}
