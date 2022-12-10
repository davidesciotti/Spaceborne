from pathlib import Path

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

deg2_in_sphere = 41252.96  # deg^2 in a spere
survey_area_deg2 = 15_000  # deg^2
fsky = survey_area_deg2 / deg2_in_sphere

BNT_transform = False

# settings for SSC comparison (aka 'sylvain'):
# survey_area_deg2 = 15469.86  # deg^2
# use_WA: False

general_cfg = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': 10,
    'zbins_list': None,
    'EP_or_ED': 'EP',
    'zbins_type_list': ('EP',),
    'n_probes': 2,
    'nbl_WL': 30,
    'nbl_GC': 30,
    'use_WA': True,  # ! xxx
    'save_cls_3d': False,
    'save_rls_3d': False,
    'cl_BNT_transform': BNT_transform,
    'BNT_matrix_path': f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/BNT_matrix',
    'BNT_matrix_filename': 'BNT_mat_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    # 'cl_folder': f'{project_path.parent}/common_data/vincenzo/14may/CijDers/' + '{EP_or_ED:s}{zbins:02d}',
    'cl_folder': f'{project_path.parent}/common_data/vincenzo/thesis_data/Cij_tesi/new_names',
    'rl_folder': f'{project_path.parent}/common_data/vincenzo/Pk_responses_2D/' + '{EP_or_ED:s}{zbins:02d}',
    # 'cl_filename': 'Cij{probe:s}-GR-Flat-eNLA-NA.dat',
    'cl_filename': 'Cij{probe:s}-N4TB-GR-eNLA.dat',
    'rl_filename': 'rij{probe:s}corr-istf-alex.dat',
}

if general_cfg['ell_max_WL'] == general_cfg['ell_max_GC']:
    general_cfg['use_WA'] = False

covariance_cfg = {
    'ind_folder': f'{project_path.parent}/common_data/ind_files' + '/{triu_tril:s}_{row_col_major:s}',
    'ind_filename': 'indices_{triu_tril:s}_{row_col_major:s}_zbins{zbins:02d}.dat',
    'triu_tril': 'triu',
    'row_col_major': 'row-wise',
    'GL_or_LG': 'GL',
    'fsky': fsky,
    'rl_value': 4,  # it used to be 4 for a constant probe response, which this is wrong
    'block_index': 'ell',
    # this is the one used by me and Vincenzo. The blocks in the 2D covmat will be indexed by ell1, ell2
    'which_probe_response': 'constant',
    'ng': 30,
    'ng_folder': None,
    'ng_filename': None,
    'sigma_eps2': 0.3 ** 2,
    'compute_covmat': True,
    'save_cov_2D': True,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_GS': False,
    'save_cov_SSC': False,
    'save_cov_dat': False,  # this is the format used by Vincenzo
    'save_2DCLOE': False,  # quite useless, this is not the format used by CLOE
    'cov_folder': f'{job_path}/output/covmat',
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
    # 'derivatives_folder': f'{project_path.parent}/common_data/vincenzo/14may/CijDers/' + '{EP_or_ED:s}{zbins:02d}',
    'derivatives_folder': f'{project_path.parent}/common_data/vincenzo/thesis_data/Cij_derivatives_tesi/new_names/',
    # 'derivatives_filename': 'dCij{probe:s}d{param:s}-GR-Flat-eNLA-NA.dat',
    'derivatives_prefix': 'dCij{probe:s}d',
    'derivatives_suffix': '-N4TB-GR-eNLA',  # I'd like to use this, but instead:
    'FM_folder': f'{job_path}/output/FM',
    'FM_filename': 'FM_{probe:s}_{which_cov:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02}.txt',
    'params_order': None,
    'paramnames_cosmo': paramnames_cosmo,
    'paramnames_IA': paramnames_IA,
    'paramnames_galbias': paramnames_galbias,
    'paramnames_3x2pt': paramnames_3x2pt,
    'nparams_total': nparams_total,

}
