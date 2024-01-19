import sys
import numpy as np
import yaml

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
JOB_ROOT = f'{ROOT}/Spaceborne/jobs/ISTF'


sys.path.append(f'{ROOT}/Spaceborne')
import bin.check_specs as utils


with open(
        '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/common_cfg/ISTF_fiducial_params.yml') as f:
    fid_pars_dict = yaml.load(f, Loader=yaml.FullLoader)
fid_pars_dict_for_fm = fid_pars_dict['FM_ordered_params']  # necessary for FM handling

which_input_files = 'cl14may'  # which input files to use
which_forecast = 'ISTF'
fsky, GL_or_LG, ind_ordering, cl_folder = utils.get_specs(which_forecast)

cl_BNT_transform = False
cov_BNT_transform = False
deriv_BNT_transform = False

cl_ell_cuts = False
cov_ell_cuts = False
deriv_ell_cuts = False

if cl_BNT_transform or cov_BNT_transform or deriv_BNT_transform:
    BNT_transform = True
else:
    BNT_transform = False

if cl_ell_cuts or cov_ell_cuts or deriv_ell_cuts:
    ell_cuts = True
else:
    ell_cuts = False

if cl_ell_cuts:
    assert cov_ell_cuts is False, 'if you want to apply ell cuts to the cls, you cannot apply them to the cov'
    assert deriv_ell_cuts, 'if you want to apply ell cuts to the cls, you hould also apply them to the derivatives'

if cov_ell_cuts:
    assert cl_ell_cuts is False, 'if you want to apply ell cuts to the cov, you cannot apply them to the cls'
    assert deriv_ell_cuts, 'if you want to apply ell cuts to the cov, you hould also apply them to the derivatives'

use_sylvains_deltas = False
use_WA = False
if which_input_files == 'cl14may':
    cl_folder = f'{ROOT}/common_data/vincenzo/14may/CijDers/' + '{EP_or_ED:s}{zbins:02d}'
    cl_filename = 'Cij{probe:s}-GR-Flat-eNLA-NA.dat'
    gal_bias_prefix = 'bL'
    derivatives_folder = f'{ROOT}/common_data/vincenzo/14may/CijDers/' + '{EP_or_ED:s}{zbins:02d}'
    derivatives_suffix = '-GR-Flat-eNLA-NA'
elif which_input_files == 'cl15gen':
    cl_folder = f'{ROOT}/common_data/vincenzo/thesis_data/Cij_tesi/new_names'
    cl_filename = 'Cij{probe:s}-N4TB-GR-eNLA.dat'
    gal_bias_prefix = 'b'
    derivatives_folder = f'{ROOT}/common_data/vincenzo/thesis_data/Cij_derivatives_tesi/new_names'
    derivatives_suffix = '-N4TB-GR-eNLA'
elif which_input_files == 'SSC_comparison_updated':
    # settings for SSC comparison (aka 'sylvain'):
    # survey_area_deg2 = 15469.86  # deg^2
    use_WA: False
    use_sylvains_deltas = True
    GL_or_LG = 'GL'
    triu_tril = 'triu'
    cl_folder = 'SPV3'

general_cfg = {
    'fid_yaml_path': f'{ROOT}/Spaceborne/common_cfg/ISTF_fiducial_params.yml',

    'fid_pars_dict': fid_pars_dict,
    'which_input_files': which_input_files,
    'which_forecast': which_forecast,

    'ell_min': 10,
    'ell_max_WL': 3000,
    'ell_max_GC': 3000,
    'ell_max_XC': 3000,
    'ell_max_3x2pt': 3000,
    'zbins': 10,
    'zbins_list': None,
    'EP_or_ED': 'EP',
    'n_probes': 2,
    'nbl_WL': 30,
    'nbl_GC': 30,
    'nbl_3x2pt': 30,
    'use_WA': use_WA,  # ! xxx
    'save_cls_3d': False,
    'save_rls_3d': False,
    
    'bias_model': 'ISTF_bias',
    'has_rsd': False,
    'has_magnification_bias': False,

    'ell_cuts': ell_cuts,

    'cl_BNT_transform': cl_BNT_transform,
    'BNT_transform': BNT_transform,
    'BNT_matrix_path': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/BNT_matrix',
    'BNT_matrix_filename': 'BNT_mat_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    'cl_folder': cl_folder,
    'rl_folder': f'{ROOT}/common_data/vincenzo/Pk_responses_2D/' + '{EP_or_ED:s}{zbins:02d}',
    'cl_filename': cl_filename,
    'rl_filename': 'rij{probe:s}corr-istf-alex.dat',

    'test_against_benchmarks': True,
}

if general_cfg['ell_max_WL'] == general_cfg['ell_max_GC']:
    general_cfg['use_WA'] = False

covariance_cfg = {
    # 'ind_folder': f'{ROOT}/common_data/ind_files' + '/{triu_tril:s}_{row_col_major:s}',
    # 'ind_filename': 'indices_{triu_tril:s}_{row_col_major:s}_zbins{zbins:02d}.dat',
    'triu_tril': 'triu',
    'row_col_major': 'row-major',
    'block_index': 'ell',
    'GL_or_LG': 'GL',
    'probe_ordering': None,  # this is set in the main

    'which_probe_response': 'variable',
    'response_const_value': None,  # it used to be 4 for a constant probe response, which this is wrong

    'fsky': fsky,
    'ng': 30,
    'ng_folder': None,
    'ng_filename': None,
    'sigma_eps2': 0.3 ** 2,
    'use_sylvains_deltas': use_sylvains_deltas,
    
    'nofz_folder': f'{ROOT}/CLOE_validation/data/n_of_z',
    'nofz_filename': 'nzTabISTF.dat',

    'cov_BNT_transform': cov_BNT_transform,
    'cov_ell_cuts': cov_ell_cuts,

    'compute_covmat': True,
    'compute_SSC': True,
    'compute_cov_6D': False,

    'save_cov': False,
    'cov_file_format': 'npz',  # or npy
    'save_cov_dat': False,  # this is the format used by Vincenzo

    # in cov_dict
    'save_cov_2D': False,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_GO': False,
    'save_cov_GS': False,
    'save_cov_SSC': False,
    'save_2DCLOE': False,  # outermost loop is on the probes

    'cov_folder': f'{JOB_ROOT}/output/{which_input_files}/' + 'covmat/{SSC_code:s}',
    'cov_filename': 'covmat_{which_cov:s}_{probe:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02d}_{ndim:d}D',
    'cov_SSC_PyCCL_folder': f'{ROOT}/PyCCL_SSC/output/covmat',
    'cov_SSC_PyCCL_filename': 'cov_PyCCL_SSC_{probe:s}_nbl{nbl:d}_ellsISTF_ellmax{ell_max:d}_HMrecipeKrause2017_6D',
    # TODO these 2 filenames could be unified...

    'SSC_code': 'PyCCL',  # ! PySSC or PyCCL or exactSSC

    'PySSC_cfg': {
        'which_ng_cov': 'SSC',
    },

    'PyCCL_cfg': {
        'probe': '3x2pt',  # TODO deprecate this? probably still useful if I want to compute instead of loading...
        'which_ng_cov': ('SSC',),
        'which_grids': '',
        'get_3x2pt_cov_in_4D': False,
        'save_trispectrum': False,
        'cov_path': '/home/davide/Documenti/Lavoro/Programmi/PyCCL_SSC/output/covmat/ISTF/jan_2024',
        'cov_filename': 'cov_{which_ng_cov:s}_pyccl_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D_'
                        'nbl{nbl:d}_ellmax{lmax:d}_zbins{EP_or_ED:s}{zbins:02d}{which_grids:s}.npz',

        'load_precomputed_cov': False,
        'save_cov': False,
        'use_HOD_for_GCph': True,  # ! this must be True, incorrect results for GCph!!

        # z_grid min and max should probably coincide. play around with steps to find the minimum number
        'z_grid_tkka_min': 0.001,
        'z_grid_tkka_max': 3,
        'z_grid_tkka_steps': 500,
        'z_grid_min': 0.001,
        'z_grid_max': 3,
        'z_grid_steps': 1000,
        'n_samples_wf': 1000,
        'bias_model': 'step-wise',
    },

    'exactSSC_cfg': {
        'which_ng_cov': 'SSC',
        # in this case it is only possible to load precomputed arrays, I have to compute the integral with Julia
        'cov_path': '/home/davide/Documenti/Lavoro/Programmi/exact_SSC/output/ISTF/jan_2024/SSC_matrix',
        'cov_filename': 'cov_{which_ng_cov:s}_spaceborne_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D_nbl{nbl:d}_ellmax{lmax:d}'
                        '_zbins{EP_or_ED:s}{zbins:02d}_zsteps{z_steps_sigma2:d}_k{k_txt_label:s}'
                        '_convention{cl_integral_convention:s}.npy',

        # settings for sigma2
        'cl_integral_convention': 'PySSC',  # or Euclid, but gives same results as it should!!! TODO remove this
        'k_txt_label': '1overMpc',
        'use_precomputed_sigma2': True,  # still need to understand exactly where to call/save this
        'z_min_sigma2': 0.001,
        'z_max_sigma2': 3,
        'z_steps_sigma2': 3000,
        'log10_k_min_sigma2': -4,
        'log10_k_max_sigma2': 1,
        'k_steps_sigma2': 20_000,
    }
}

Sijkl_cfg = {
    'wf_input_folder': f'{ROOT}/common_data/everyones_WF_from_Gdrive/davide/' + 'nz{nz:d}/gen2022',
    'wf_WL_input_filename': 'wil_dav_IA{has_IA:s}_{normalization:s}_nz{nz:d}_bia{bIA:.02f}.txt',
    'wf_GC_input_filename': 'wig_dav_{normalization:s}_nz{nz:d}.txt',
    'wf_normalization': 'IST',
    'nz': 10_000,
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl

    'Sijkl_folder': f'{ROOT}/common_data/Sijkl',
    'Sijkl_filename': 'Sijkl_WFdavide_nz{nz:d}_IA_3may.npy',
    'load_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
    'save_sijkl': False,  # save the computed Sijkl in Sijkl_folder
    # TODO update to new version of pyssc, check if this the agreement
}

# dictionaries of cosmological parameters' names and values
param_names_dict = {
    'cosmo': ["Om", "Ob", "wz", "wa", "h", "ns", "s8"],
    'IA': ["Aia", "eIA", "bIA"],
    'galaxy_bias': [f'{gal_bias_prefix}{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)],
}
# fiducial values
fiducials_dict = {
    'cosmo': [fid_pars_dict_for_fm['Om_m0'], fid_pars_dict_for_fm['Om_b0'], fid_pars_dict_for_fm['w_0'],
              fid_pars_dict_for_fm['w_a'],
              fid_pars_dict_for_fm['h'], fid_pars_dict_for_fm['n_s'], fid_pars_dict_for_fm['sigma_8']],
    'IA': [fid_pars_dict_for_fm['A_IA'], fid_pars_dict_for_fm['eta_IA'],
           fid_pars_dict_for_fm['beta_IA']],
    'galaxy_bias': [fid_pars_dict_for_fm[f'b{zbin:02d}_photo'] for zbin in range(1, general_cfg['zbins'] + 1)],
}

param_names_3x2pt = param_names_dict['cosmo'] + param_names_dict['IA'] + param_names_dict['galaxy_bias']
# this needs to be done outside the dictionary creation
fiducials_3x2pt = np.concatenate((fiducials_dict['cosmo'], fiducials_dict['IA'], fiducials_dict['galaxy_bias']))
assert len(param_names_3x2pt) == len(fiducials_3x2pt), "the fiducial values list and parameter names should have the " \
                                                       "same length"

FM_cfg = {
    'compute_FM': True,

    'param_names_dict': param_names_dict,
    'fiducials_dict': fiducials_dict,
    'nparams_tot': len(param_names_3x2pt),  # total (cosmo + nuisance) number of parameters
    'param_names_3x2pt': param_names_3x2pt,  # ! for the time being, these are defined in the main and then passed here

    'save_FM_txt': False,
    'save_FM_dict': False,

    'load_preprocess_derivatives': False,
    'derivatives_folder': derivatives_folder,
    'derivatives_prefix': 'dCij{probe:s}d',
    'derivatives_suffix': derivatives_suffix,

    'derivatives_BNT_transform': deriv_BNT_transform,
    'deriv_ell_cuts': deriv_ell_cuts,

    'fm_folder': str(JOB_ROOT) + f'/output/{which_input_files}/' + 'FM/jan_2024/{SSC_code:s}',
    'FM_txt_filename': 'FM_{probe:s}_{which_cov:s}_lmax{ell_max:d}_nbl{nbl:d}_zbins{EP_or_ED:s}{zbins:02}',
    'FM_dict_filename': 'FM_dict_zbins{EP_or_ED:s}{zbins:02}',

    'test_against_benchmarks': True,
    'FM_file_format': 'txt',
}
