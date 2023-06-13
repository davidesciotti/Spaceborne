from pathlib import Path
import sys
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path}/bin')
import check_specs as utils

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTFfid

which_forecast = 'SPV3'
fsky, GL_or_LG, ind_ordering, _ = utils.get_specs(which_forecast)

SPV3_folder = f'{project_path.parent}/common_data/vincenzo/SPV3_07_2022/Flagship_1_restored'

# ! choose the flagship version and whether you want to use the BNT transform
flagship_version = 1

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

assert flagship_version == 2, 'the files for the multicut case are only available for Flagship_2'

if BNT_transform:
    assert flagship_version == 2, 'we are applying the BNT only for Flagship_2'

general_cfg = {
    'ell_min': 10,
    'ell_max_WL_opt': 5000,  # this is the value from which the various bin cuts are applied
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'ell_max_XC': 3000,
    'zbins': 10,
    'zbins_list': None,
    'EP_or_ED': 'EP',
    'n_probes': 2,
    'which_forecast': which_forecast,
    'use_WA': False,
    'save_cls_3d': True,
    'save_rls_3d': True,

    'flat_or_nonflat': 'flat',

    # the case with the largest range is nbl_WL_opt.. This is the reference ell binning from which the cuts are applied;
    # in principle, the other binning should be consistent with this one and should not be hardcoded, as long as
    # lmax=5000, 3000 holds
    'nbl_WL_opt': 32,
    'nbl_GC_opt': 29,
    'nbl_WA_opt': 3,
    'nbl_3x2pt_opt': 29,

    'ell_cuts': ell_cuts,
    'which_cuts': 'Vincenzo',
    'center_or_min': 'min',  # cut if the bin *center* or the bin *lower edge* is larger than ell_max[zi, zj]
    'cl_ell_cuts': cl_ell_cuts,
    'ell_cuts_folder': f'{SPV3_folder}/ell_cuts',
    'ell_cuts_filename': 'lmax_cut_{probe:s}_{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:03d}-'
                         'ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
    'kmax_h_over_Mpc_ref': 1.0,
    # 'kmax_list_1_over_Mpc': np.array((0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 3.00, 5.00, 10.00)),
    'kmax_h_over_Mpc_list': np.array([0.37313433, 0.74626866, 1.11940299, 1.49253731, 1.86567164,
                                      2.23880597, 2.6119403, 2.98507463, 4.47761194,
                                      7.46268657]),

    'BNT_matrix_path': f'{SPV3_folder}/BNT_matrix',
    'BNT_matrix_filename': 'BNT_mat_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}.npy',
    'BNT_transform': BNT_transform,  # ! to be deprecated?
    'cl_BNT_transform': cl_BNT_transform,

    'idIA': 2,
    'idB': 3,
    'idM': 3,
    'idR': 1,

    'which_pk': 'HMCode2020',
    'cl_folder': f'{SPV3_folder}' + f'/DataVectors/Noiseless/{probe:s}' + f'/FS{flagship_version}',
    'rl_folder': f'{SPV3_folder}' + f'/ResFunTabs/FS{flagship_version}' + '/{probe:s}',
    'cl_filename': 'dv-{probe:s}-{nbl:d}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-{EP_or_ED:s}{zbins:02d}.dat',
    'rl_filename': 'rf-{probe:s}-{nbl:d}-wzwaCDM-Flat-GR-TB-idMag0-idRSD0-idFS0-idSysWL3-idSysGC4-{EP_or_ED:s}{zbins:02d}.dat',

    'flagship_version': flagship_version,

    'test_against_benchmarks': False,
}

if general_cfg['ell_max_WL'] == general_cfg['ell_max_GC']:
    general_cfg['use_WA'] = False

covariance_cfg = {

    # ordering-related stuff
    'triu_tril': 'triu',
    'row_col_major': 'row-major',
    'block_index': 'ell',
    'GL_or_LG': GL_or_LG,

    'SSC_code': 'PySSC',
    'which_probe_response': 'variable',
    'response_const_value': None,  # it used to be 4 for a constant probe response, which is quite wrong
    'cov_SSC_PyCCL_folder': '/Users/davide/Documents/Lavoro/Programmi/PyCCL_SSC/output/covmat/after_script_update',
    'cov_SSC_PyCCL_filename': 'cov_PyCCL_SSC_{probe:s}_nbl{nbl:s}_ellmax{ell_max:d}_HMrecipeKrause2017_6D.npy',

    # n_gal, sigma_eps, fsky, all entering the covariance matrix
    'fsky': fsky,  # ! new
    'sigma_eps2': (0.26 * np.sqrt(2)) ** 2,  # ! new
    'ng': None,  # ! the new value is 28.73 (for Flagship_1), but I'm taking the value from the ngbTab files
    'ng_folder': f'{SPV3_folder}/InputNz/Lenses/Flagship',
    'ng_filename': 'ngbsTab-{EP_or_ED:s}{zbins:02d}.dat',

    # sources (and lenses) redshift distributions
    'nofz_folder': f'{SPV3_folder}/InputNz/Lenses/Flagship',
    'nofz_filename': 'niTab-{EP_or_ED:s}{zbins:02d}.dat',
    'plot_nz_tocheck': True,

    'cov_BNT_transform': cov_BNT_transform,
    'cov_ell_cuts': cov_ell_cuts,

    'compute_covmat': True,
    'compute_SSC': True,

    'save_cov': True,
    'cov_file_format': 'npz',  # or npy
    'save_cov_dat': False,  # this is the format used by Vincenzo

    'save_cov_2D': True,
    'save_cov_4D': False,
    'save_cov_6D': False,  # or 10D for the 3x2pt
    'save_cov_GS': False,
    'save_cov_SSC': False,
    'save_2DCLOE': False,  # outermost loop is on the probes

    # ! no folders for ell_cut_center or min
    'cov_folder': f'{job_path}/output/covmat',
    'cov_filename': 'covmat_{which_cov:s}_{probe:s}_zbins{EP_or_ED:s}{zbins:02d}_{ndim:d}D',
    'cov_filename_vincenzo': 'cm-{probe_vinc:s}-{GOGS_filename:s}-{nbl_WL:d}-{EP_or_ED:s}{zbins:02d}-'
                             'ML{magcut_lens:03d}-ZL{zcut_lens:02d}-MS{magcut_source:03d}-ZS{zcut_source:02d}.dat',
}
if ell_cuts:
    covariance_cfg['cov_filename'].replace('_{ndim:d}D', 'kmaxhoverMpc{kmax_h_over_Mpc:.03f}_{ndim:d}D')

Sijkl_cfg = {
    'wf_input_folder': f'{SPV3_folder}/Windows/FS1',
    'wf_WL_input_filename': 'WiGamma-{EP_or_ED:s}{zbins:02d}.dat',
    'wf_GC_input_filename': 'WiGC-{EP_or_ED:s}{zbins:02d}.dat',
    'wf_normalization': 'IST',
    'nz': None,  # ! is this used?
    'has_IA': True,  # whether to include IA in the WF used to compute Sijkl

    'Sijkl_folder': f'{job_path}/output/sijkl',
    'Sijkl_filename': 'sijkl_WF-FS{flagship_version:01d}_nz{nz:d}_zbins{EP_or_ED:s}{zbins:02}_IA{IA_flag:}.npy',
    'use_precomputed_sijkl': True,  # try to load precomputed Sijkl from Sijkl_folder, if it altready exists
}


param_names_dict = {
    'cosmo': ["Om", "Ob", "wz", "wa", "h", "ns", "s8", 'logT_AGN'],
    'IA': ["Aia", "eIA"],
    'shear_bias': [f'm{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)],
    'dzWL': [f'dzWL{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)],
    'galaxy_bias': [f'bG{zbin_idx:02d}' for zbin_idx in range(1, 5)],
    'magnification_bias': [f'bM{zbin_idx:02d}' for zbin_idx in range(1, 5)],
    # 'dzGC': [f'dzGC{zbin_idx:02d}' for zbin_idx in range(1, general_cfg['zbins'] + 1)]
}
# declare the set of parameters under study
param_names_3x2pt = list(np.concatenate([param_names_dict[key] for key in param_names_dict.keys()]))

# I cannot define the fiducial values here because I need to import the files for the galaxy bias

ell_cuts_subfolder = f'{general_cfg["which_cuts"]}/ell_{general_cfg["center_or_min"]}'
if not ell_cuts:
    ell_cuts_subfolder = ''

FM_txt_filename = covariance_cfg['cov_filename'].replace('covmat_', 'FM_').replace('_{ndim:d}D', '')
FM_dict_filename = covariance_cfg['cov_filename'].replace('covmat_{which_cov:s}_{probe:s}_', 'FM_').replace(
    '_{ndim:d}D',
    '')
deriv_filename = covariance_cfg['cov_filename'].replace('covmat_', 'dDVd')
FM_cfg = {
    'compute_FM': True,

    'param_names_dict': param_names_dict,
    'fiducials_dict': None,  # this needs to be set in the main, since it depends on the n_gal file
    'param_names_3x2pt': param_names_3x2pt,
    'nparams_tot': len(param_names_3x2pt),  # total (cosmo + nuisance) number of parameters

    'save_FM_txt': True,
    'save_FM_dict': True,

    'load_preprocess_derivatives': False,
    'derivatives_folder': SPV3_folder + '/OutputFiles/DataVecDers/{flat_or_nonflat:s}/{which_pk:s}',
    'derivatives_filename': deriv_filename,
    'derivatives_prefix': 'dDVd',

    'derivatives_BNT_transform': deriv_BNT_transform,
    'deriv_ell_cuts': deriv_ell_cuts,

    'fm_folder': f'{job_path}/output/Flagship_{flagship_version}/FM/BNT_{BNT_transform}' +
                 '/ell_cuts_{ell_cuts:s}' + ell_cuts_subfolder,
    'FM_txt_filename': FM_txt_filename,
    'FM_dict_filename': FM_dict_filename,

    'test_against_benchmarks': False,
    # 'FM_zbins{EP_or_ED:s}{zbins:02}-ML{magcut_lens:03d}-ZL{zcut_lens:02d}-'
    #                 'MS{magcut_source:03d}-ZS{zcut_source:02d}'
    #                 '_kmax_h_over_Mpc{kmax_h_over_Mpc:03f}',
}
