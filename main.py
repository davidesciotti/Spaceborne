import argparse
import os
import pprint
import sys
import time
import warnings
from copy import deepcopy
from functools import partial
from importlib.util import find_spec

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import CubicSpline, RectBivariateSpline
from spaceborne import (
    bnt,
    cl_utils,
    config_checker,
    cosmo_lib,
    mask_utils,
    ell_utils,
    pyccl_interface,
    responses,
    sigma2_SSC,
    wf_cl_lib,
)
from spaceborne import covariance as sb_cov
from spaceborne import onecovariance_interface as oc_interface
from spaceborne import sb_lib as sl

try:
    import pyfiglet

    text = 'Spaceborne'
    ascii_art = pyfiglet.figlet_format(text, font='slant')
    print(ascii_art)
except ImportError:
    pass


# Get the current script's directory
# current_dir = Path(__file__).resolve().parent
# parent_dir = current_dir.parent

warnings.filterwarnings(
    'ignore',
    message='.*FigureCanvasAgg is non-interactive, and thus cannot be shown.*',
    category=UserWarning,
)

pp = pprint.PrettyPrinter(indent=4)
script_start_time = time.perf_counter()

# use the _dev config in the develop branch!
_config_path = 'config.yaml' if os.path.exists('config.yaml') else 'example_config.yaml'

def load_config():
    # Check if we're running in a Jupyter environment (or interactive mode)
    if 'ipykernel_launcher.py' in sys.argv[0]:
        # Running interactively, so use default config file
        config_path = _config_path

    else:
        parser = argparse.ArgumentParser(description='Spaceborne')
        parser.add_argument(
            '--config',
            type=str,
            help='Path to the configuration file',
            required=False,
            default=_config_path,
        )
        parser.add_argument(
            '--show_plots',
            action='store_true',
            help='Show plots if specified',
            required=False,
        )
        args = parser.parse_args()
        config_path = args.config

    # Only switch to Agg if not running interactively and --show_plots is not specified.
    if 'ipykernel_launcher.py' not in sys.argv[0] and '--show_plots' not in sys.argv:
        import matplotlib

        matplotlib.use('Agg')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return cfg


cfg = load_config()


# some convenence variables, just to make things more readable
h = cfg['cosmology']['h']
galaxy_bias_fit_fiducials = np.array(cfg['C_ell']['galaxy_bias_fit_coeff'])
magnification_bias_fit_fiducials = np.array(
    cfg['C_ell']['magnification_bias_fit_coeff']
)
# this has the same length as ngal_sources, as checked below
zbins = len(cfg['nz']['ngal_lenses'])
which_sigma2_b = cfg['covariance']['which_sigma2_b']
probe_ordering = cfg['covariance']['probe_ordering']
GL_OR_LG = probe_ordering[1][0] + probe_ordering[1][1]
output_path = cfg['misc']['output_path']
clr = cm.rainbow(np.linspace(0, 1, zbins))

if not os.path.exists(output_path):
    raise FileNotFoundError(
        f'Output path {output_path} does not exist. '
        'Please create it before running the script.'
    )
for subdir in ['cache', 'cache/trispectrum/SSC', 'cache/trispectrum/cNG']:
    os.makedirs(f'{output_path}/{subdir}', exist_ok=True)

# ! START HARDCODED OPTIONS/PARAMETERS
use_h_units = False  # whether or not to normalize Megaparsecs by little h
nbl_3x2pt_oc = 500  # number of ell bins over which to compute the Cls passed to OC
# for the Gaussian covariance computation
k_steps_sigma2 = 20_000
k_steps_sigma2_levin = 300
shift_nz_interpolation_kind = 'linear'  # TODO this should be spline

# whether or not to symmetrize the covariance probe blocks when
# reshaping it from 4D to 6D.
# Useful if the 6D cov elements need to be accessed directly, whereas if
# the cov is again reduced to 4D or 2D.
# Can be set to False for a significant speedup
symmetrize_output_dict = {
    ('L', 'L'): False,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): False,
}
unique_probe_comb = [
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
]
probename_dict = {0: 'L', 1: 'G'}

# these are configs which should not be visible to the user
cfg['OneCovariance'] = {}
cfg['ell_cuts'] = {}
cfg['covariance']['n_probes'] = 2
cfg['covariance']['G_code'] = 'Spaceborne'
cfg['covariance']['SSC_code'] = 'Spaceborne'
cfg['covariance']['cNG_code'] = 'PyCCL'
cfg['OneCovariance']['precision_settings'] = 'default'
cfg['OneCovariance']['path_to_oc_executable'] = '/home/davide/Documenti/Lavoro/Programmi/OneCovariance/covariance.py'  # fmt: skip
cfg['OneCovariance']['path_to_oc_ini'] = './input/config_3x2pt_pure_Cell_general.ini'
cfg['OneCovariance']['consistency_checks'] = False
cfg['misc']['save_output_as_benchmark'] = False
cfg['misc']['bench_filename'] = (
'../Spaceborne_bench/output_G{g_code:s}_SSC{ssc_code:s}_cNG{cng_code:s}_KE{use_KE:s}_resp{which_pk_responses:s}_b1g{which_b1g_in_resp:s}_newtest'
)
cfg['ell_cuts']['apply_ell_cuts'] = False  # Type: bool
cfg['ell_cuts']['center_or_min'] = (
    'center'  # Type: str. Cut if the bin *center* or the bin *lower edge* is larger than ell_max[zi, zj]
)
cfg['ell_cuts']['cl_ell_cuts'] = False  # Type: bool
cfg['ell_cuts']['cov_ell_cuts'] = False  # Type: bool
cfg['ell_cuts']['kmax_h_over_Mpc_ref'] = (
    1.0  # Type: float. This is used when ell_cuts is False, also...?
)
cfg['ell_cuts']['kmax_h_over_Mpc_list'] = [0.1, 0.16681005, 0.27825594, 0.46415888, 0.77426368, 1.29154967, 2.15443469, 3.59381366, 5.9948425, 10.0,]  # fmt: skip

# if in main branch, set this to False
cfg['nz']['shift_nz'] = True
if sl.is_main_branch():
    cfg['nz']['shift_nz'] = False

cfg['nz']['dzWL'] = [-0.008848, 0.051368, 0.059484]
cfg['nz']['dzGC'] = [-0.008848, 0.051368, 0.059484]
cfg['nz']['normalize_shifted_nz'] = True
cfg['nz']['nz_gaussian_smoothing'] = False
cfg['nz']['nz_gaussian_smoothing_sigma'] = 2
cfg['nz']['plot_nz_tocheck'] = True
if cfg['nz']['shift_nz']:
    warnings.warn('nz is currently being shifted!!', stacklevel=2)
# ! END HARDCODED OPTIONS/PARAMETERS

# some of the configs have been defined here...
dzWL_fiducial = cfg['nz']['dzWL']
dzGC_fiducial = cfg['nz']['dzGC']
shift_nz = cfg['nz']['shift_nz']
normalize_shifted_nz = cfg['nz']['normalize_shifted_nz']
n_probes = cfg['covariance']['n_probes']


# ! set non-gaussian cov terms to compute
cov_terms_list = []
if cfg['covariance']['G']:
    cov_terms_list.append('G')
if cfg['covariance']['SSC']:
    cov_terms_list.append('SSC')
if cfg['covariance']['cNG']:
    cov_terms_list.append('cNG')
cov_terms_str = ''.join(cov_terms_list)

compute_oc_g, compute_oc_ssc, compute_oc_cng = False, False, False
compute_sb_ssc, compute_sb_cng = False, False
compute_ccl_ssc, compute_ccl_cng = False, False
if cfg['covariance']['G'] and cfg['covariance']['G_code'] == 'OneCovariance':
    compute_oc_g = True
if cfg['covariance']['SSC'] and cfg['covariance']['SSC_code'] == 'OneCovariance':
    compute_oc_ssc = True
if cfg['covariance']['cNG'] and cfg['covariance']['cNG_code'] == 'OneCovariance':
    compute_oc_cng = True

if cfg['covariance']['SSC'] and cfg['covariance']['SSC_code'] == 'Spaceborne':
    compute_sb_ssc = True
if cfg['covariance']['cNG'] and cfg['covariance']['cNG_code'] == 'Spaceborne':
    raise NotImplementedError('Spaceborne cNG not implemented yet')
    compute_sb_cng = True

if cfg['covariance']['SSC'] and cfg['covariance']['SSC_code'] == 'PyCCL':
    compute_ccl_ssc = True
if cfg['covariance']['cNG'] and cfg['covariance']['cNG_code'] == 'PyCCL':
    compute_ccl_cng = True

if cfg['covariance']['use_KE_approximation']:
    cl_integral_convention_ssc = 'Euclid_KE_approximation'
    ssc_integration_type = 'simps_KE_approximation'
else:
    cl_integral_convention_ssc = 'Euclid'
    ssc_integration_type = 'simps'

if use_h_units:
    k_txt_label = 'hoverMpc'
    pk_txt_label = 'Mpcoverh3'
else:
    k_txt_label = '1overMpc'
    pk_txt_label = 'Mpc3'

if not cfg['ell_cuts']['apply_ell_cuts']:
    kmax_h_over_Mpc = cfg['ell_cuts']['kmax_h_over_Mpc_ref']


# ! sanity checks on the configs
# TODO update this when cfg are done
cfg_check_obj = config_checker.SpaceborneConfigChecker(cfg)
cfg_check_obj.run_all_checks()

# ! instantiate CCL object
ccl_obj = pyccl_interface.PycclClass(
    cfg['cosmology'],
    cfg['extra_parameters'],
    cfg['intrinsic_alignment'],
    cfg['halo_model'],
    cfg['PyCCL']['spline_params'],
    cfg['PyCCL']['gsl_params'],
)
# set other useful attributes
ccl_obj.p_of_k_a = 'delta_matter:delta_matter'
ccl_obj.zbins = zbins
ccl_obj.output_path = output_path
ccl_obj.which_b1g_in_resp = cfg['covariance']['which_b1g_in_resp']

# get ccl default a and k grids
a_default_grid_ccl = ccl_obj.cosmo_ccl.get_pk_spline_a()
z_default_grid_ccl = cosmo_lib.a_to_z(a_default_grid_ccl)[::-1]
lk_default_grid_ccl = ccl_obj.cosmo_ccl.get_pk_spline_lk()

if cfg['C_ell']['cl_CCL_kwargs'] is not None:
    cl_ccl_kwargs = cfg['C_ell']['cl_CCL_kwargs']
else:
    cl_ccl_kwargs = {}

if cfg['intrinsic_alignment']['lumin_ratio_filename'] is not None:
    ccl_obj.lumin_ratio_2d_arr = np.genfromtxt(
        cfg['intrinsic_alignment']['lumin_ratio_filename']
    )
else:
    ccl_obj.lumin_ratio_2d_arr = None

# define k_limber function
k_limber_func = partial(
    cosmo_lib.k_limber, cosmo_ccl=ccl_obj.cosmo_ccl, use_h_units=use_h_units
)

# ! define k and z grids used throughout the code (k is in 1/Mpc)
# TODO should zmin and zmax be inferred from the nz tables?
# TODO -> not necessarily true for all the different zsteps
z_grid = np.linspace(  # fmt: skip
    cfg['covariance']['z_min'], 
    cfg['covariance']['z_max'], 
    cfg['covariance']['z_steps']
)  # fmt: skip
z_grid_trisp = np.linspace(
    cfg['covariance']['z_min'],
    cfg['covariance']['z_max'],
    cfg['covariance']['z_steps_trisp'],
)
k_grid = np.logspace(
    cfg['covariance']['log10_k_min'],
    cfg['covariance']['log10_k_max'],
    cfg['covariance']['k_steps'],
)
# in this case we need finer k binning because of the bessel functions
k_grid_s2b_simps = np.logspace(  # fmt: skip
    cfg['covariance']['log10_k_min'], 
    cfg['covariance']['log10_k_max'], 
    k_steps_sigma2
)  # fmt: skip
if len(z_grid) < 250:
    warnings.warn(
        'z_grid is small, at the moment it used to compute various '
        'intermediate quantities',
        stacklevel=2,
    )

zgrid_str = (
    f'zmin{cfg["covariance"]["z_min"]}_zmax{cfg["covariance"]["z_max"]}'
    f'_zsteps{cfg["covariance"]["z_steps"]}'
)

# ! do the same for CCL - i.e., set the above in the ccl_obj with little variations
# ! (e.g. a instead of z)
# TODO I leave the option to use a grid for the CCL, but I am not sure if it is needed
z_grid_tkka_SSC = z_grid_trisp
z_grid_tkka_cNG = z_grid_trisp
ccl_obj.a_grid_tkka_SSC = cosmo_lib.z_to_a(z_grid_tkka_SSC)[::-1]
ccl_obj.a_grid_tkka_cNG = cosmo_lib.z_to_a(z_grid_tkka_cNG)[::-1]
ccl_obj.logn_k_grid_tkka_SSC = np.log(k_grid)
ccl_obj.logn_k_grid_tkka_cNG = np.log(k_grid)

# check that the grid is in ascending order
if not np.all(np.diff(ccl_obj.a_grid_tkka_SSC) > 0):
    raise ValueError('a_grid_tkka_SSC is not in ascending order!')
if not np.all(np.diff(ccl_obj.a_grid_tkka_cNG) > 0):
    raise ValueError('a_grid_tkka_cNG is not in ascending order!')
if not np.all(np.diff(z_grid) > 0):
    raise ValueError('z grid is not in ascending order!')
if not np.all(np.diff(z_grid_trisp) > 0):
    raise ValueError('z grid is not in ascending order!')

if cfg['PyCCL']['use_default_k_a_grids']:
    ccl_obj.a_grid_tkka_SSC = a_default_grid_ccl
    ccl_obj.a_grid_tkka_cNG = a_default_grid_ccl
    ccl_obj.logn_k_grid_tkka_SSC = lk_default_grid_ccl
    ccl_obj.logn_k_grid_tkka_cNG = lk_default_grid_ccl

# build the ind array and store it into the covariance dictionary
zpairs_auto, zpairs_cross, zpairs_3x2pt = sl.get_zpairs(zbins)
ind = sl.build_full_ind(
    cfg['covariance']['triu_tril'], cfg['covariance']['row_col_major'], zbins
)
ind_auto = ind[:zpairs_auto, :].copy()
ind_cross = ind[zpairs_auto : zpairs_cross + zpairs_auto, :].copy()
ind_dict = {('L', 'L'): ind_auto, ('G', 'L'): ind_cross, ('G', 'G'): ind_auto}

# ! ====================================================================================
# ! ====================================================================================
# ! ====================================================================================
# ! ================================= BEGIN MAIN BODY ==================================

# ! 1. Mask
mask_obj = mask_utils.Mask(cfg['mask'])
mask_obj.process()
if hasattr(mask_obj, 'mask'):
    import healpy as hp

    hp.mollview(mask_obj.mask, cmap='inferno_r')


# ! 2. Redshift distributions
# The shape of these input files should be `(zpoints, zbins + 1)`, with `zpoints` the
# number of points over which the distribution is measured and zbins the number of
# redshift bins. The first column should contain the redshifts values.
# We also define:
# - `nz_full`: nz table including a column for the z values
# - `nz`:      nz table excluding a column for the z values
# - `nz_original`: nz table as imported (it may be subjected to shifts later on)
nz_src_tab_full = np.genfromtxt(cfg['nz']['nz_sources_filename'])
nz_lns_tab_full = np.genfromtxt(cfg['nz']['nz_lenses_filename'])
zgrid_nz_src = nz_src_tab_full[:, 0]
zgrid_nz_lns = nz_lns_tab_full[:, 0]
nz_src = nz_src_tab_full[:, 1:]
nz_lns = nz_lns_tab_full[:, 1:]

# nz may be subjected to a shift
nz_unshifted_src = nz_src
nz_unshifted_lns = nz_lns

# ! 3. \ell binning
ell_obj = ell_utils.EllBinning(cfg)
ell_obj.build_ell_bins()

# private cfg dictionary. This serves a couple different purposeses:
# 1. To store and pass hardcoded parameters in a convenient way
# 2. To make the .format() more compact
pvt_cfg = {
    'zbins': zbins,
    'ind': ind,
    'probe_ordering': probe_ordering,
    'which_ng_cov': cov_terms_str,
    'cov_terms_list': cov_terms_list,
    'GL_OR_LG': GL_OR_LG,
    'symmetrize_output_dict': symmetrize_output_dict,
    'use_h_units': use_h_units,
    'z_grid': z_grid,
}


# ! START SCALE CUTS: for these, we need to:
# 1. Compute the BNT. This is done with the raw, or unshifted n(z), but only for
# the purpose of computing the ell cuts - the rest of the code uses a BNT matrix
# from the shifted n(z) - see also comment below.
# 2. compute the kernels for the un-shifted n(z) (for consistency)
# 3. bnt-transform these kernels (for lensing, it's only the gamma kernel),
# and use these to:
# 4. compute the z means
# 5. compute the ell cuts

# ! shift and set nz
if shift_nz:
    nz_src = wf_cl_lib.shift_nz(
        zgrid_nz_src,
        nz_unshifted_src,
        dzWL_fiducial,
        normalize=normalize_shifted_nz,
        plot_nz=False,
        interpolation_kind=shift_nz_interpolation_kind,
        bounds_error=False,
        fill_value=0,
    )
    nz_lns = wf_cl_lib.shift_nz(
        zgrid_nz_lns,
        nz_unshifted_lns,
        dzGC_fiducial,
        normalize=normalize_shifted_nz,
        plot_nz=False,
        interpolation_kind=shift_nz_interpolation_kind,
        bounds_error=False,
        fill_value=0,
    )

ccl_obj.set_nz(
    nz_full_src=np.hstack((zgrid_nz_src[:, None], nz_src)),
    nz_full_lns=np.hstack((zgrid_nz_lns[:, None], nz_lns)),
)
ccl_obj.check_nz_tuple(zbins)

# ! IA
ccl_obj.set_ia_bias_tuple(z_grid_src=z_grid, has_ia=cfg['C_ell']['has_IA'])

# ! galaxy bias
# TODO the alternative should be the HOD gal bias already set in the responses class!!
if cfg['C_ell']['which_gal_bias'] == 'from_input':
    gal_bias_input = np.genfromtxt(cfg['C_ell']['gal_bias_table_filename'])
    ccl_obj.gal_bias_2d, ccl_obj.gal_bias_func = sl.check_interpolate_input_tab(
        input_tab=gal_bias_input, z_grid_out=z_grid, zbins=zbins
    )
    ccl_obj.gal_bias_tuple = (z_grid, ccl_obj.gal_bias_2d)
elif cfg['C_ell']['which_gal_bias'] == 'FS2_polynomial_fit':
    ccl_obj.set_gal_bias_tuple_spv3(
        z_grid_lns=z_grid, magcut_lens=None, poly_fit_values=galaxy_bias_fit_fiducials
    )
else:
    raise ValueError('which_gal_bias should be "from_input" or "FS2_polynomial_fit"')

if np.all(
    [
        np.allclose(ccl_obj.gal_bias_2d[:, 0], ccl_obj.gal_bias_2d[:, zi])
        for zi in range(zbins)
    ]
):
    single_b_of_z = True
else:
    single_b_of_z = False

# ! magnification bias
if cfg['C_ell']['has_magnification_bias']:
    if cfg['C_ell']['which_mag_bias'] == 'from_input':
        mag_bias_input = np.genfromtxt(cfg['C_ell']['mag_bias_table_filename'])
        ccl_obj.mag_bias_2d, ccl_obj.mag_bias_func = sl.check_interpolate_input_tab(
            mag_bias_input, z_grid, zbins
        )
        ccl_obj.mag_bias_tuple = (z_grid, ccl_obj.mag_bias_2d)
    elif cfg['C_ell']['which_mag_bias'] == 'FS2_polynomial_fit':
        ccl_obj.set_mag_bias_tuple(
            z_grid_lns=z_grid,
            has_magnification_bias=cfg['C_ell']['has_magnification_bias'],
            magcut_lens=None,
            poly_fit_values=magnification_bias_fit_fiducials,
        )
    else:
        raise ValueError(
            'which_mag_bias should be "from_input" or "FS2_polynomial_fit"'
        )
else:
    ccl_obj.mag_bias_tuple = None

plt.figure()
for zi in range(zbins):
    plt.plot(z_grid, ccl_obj.gal_bias_2d[:, zi], label=f'$z_{{{zi}}}$')
plt.xlabel(r'$z$')
plt.ylabel(r'$b_{g}(z)$')
plt.legend()


# ! set radial kernel arrays and objects
ccl_obj.set_kernel_obj(cfg['C_ell']['has_rsd'], cfg['PyCCL']['n_samples_wf'])
ccl_obj.set_kernel_arr(
    z_grid_wf=z_grid, has_magnification_bias=cfg['C_ell']['has_magnification_bias']
)

gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias!)'
ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_wo_gal_bias_arr

# ! compute BNT and z means
if cfg['BNT']['cl_BNT_transform'] or cfg['BNT']['cov_BNT_transform']:
    bnt_matrix = bnt.compute_bnt_matrix(
        zbins, zgrid_nz_src, nz_src, cosmo_ccl=ccl_obj.cosmo_ccl, plot_nz=False
    )
    wf_gamma_ccl_bnt = (bnt_matrix @ ccl_obj.wf_gamma_arr.T).T
    z_means_ll = wf_cl_lib.get_z_means(z_grid, wf_gamma_ccl_bnt)
else:
    bnt_matrix = None
    z_means_ll = wf_cl_lib.get_z_means(z_grid, ccl_obj.wf_gamma_arr)

z_means_gg = wf_cl_lib.get_z_means(z_grid, ccl_obj.wf_galaxy_arr)


# assert np.all(np.diff(z_means_ll) > 0), 'z_means_ll should be monotonically increasing'
# assert np.all(np.diff(z_means_gg) > 0), 'z_means_gg should be monotonically increasing'
# assert np.all(np.diff(z_means_ll_bnt) > 0), (
#     'z_means_ll_bnt should be monotonically increasing '
#     '(not a strict condition, valid only if we do not shift the n(z) in this part)'
# )

# 5. compute the ell cuts
# TODO need to adapt this to the class structure
# ell_cuts_dict = {}
# ellcuts_kw = {
#     'kmax_h_over_Mpc': kmax_h_over_Mpc,
#     'cosmo_ccl': ccl_obj.cosmo_ccl,
#     'zbins': zbins,
#     'h': h,
#     'kmax_h_over_Mpc_ref': cfg['ell_cuts']['kmax_h_over_Mpc_ref'],
# }
# ell_cuts_dict['LL'] = ell_utils.load_ell_cuts(
#     z_values_a=z_means_ll, z_values_b=z_means_ll, **ellcuts_kw
# )
# ell_cuts_dict['GG'] = ell_utils.load_ell_cuts(
#     z_values_a=z_means_gg, z_values_b=z_means_gg, **ellcuts_kw
# )
# ell_cuts_dict['GL'] = ell_utils.load_ell_cuts(
#     z_values_a=z_means_gg, z_values_b=z_means_ll, **ellcuts_kw
# )
# ell_cuts_dict['LG'] = ell_utils.load_ell_cuts(
#     z_values_a=z_means_ll, z_values_b=z_means_gg, **ellcuts_kw
# )
# ell_dict['ell_cuts_dict'] = (
#     ell_cuts_dict  # this is to pass the ell cuts to the covariance module
# )
# ! END SCALE CUTS

wf_cl_lib.plot_nz_src_lns(zgrid_nz_src, nz_src, zgrid_nz_lns, nz_lns, colors=clr)

# convenience variables
wf_delta = ccl_obj.wf_delta_arr  # no bias here either, of course!
wf_gamma = ccl_obj.wf_gamma_arr
wf_ia = ccl_obj.wf_ia_arr
wf_mu = ccl_obj.wf_mu_arr
wf_lensing = ccl_obj.wf_lensing_arr

# plot
wf_names_list = [
    'delta',
    'gamma',
    'ia',
    'magnification',
    'lensing',
    gal_kernel_plt_title,
]
wf_ccl_list = [
    ccl_obj.wf_delta_arr,
    ccl_obj.wf_gamma_arr,
    ccl_obj.wf_ia_arr,
    ccl_obj.wf_mu_arr,
    ccl_obj.wf_lensing_arr,
    ccl_obj.wf_galaxy_arr,
]

plt.figure()
for wf_idx in range(len(wf_ccl_list)):
    for zi in range(zbins):
        plt.plot(z_grid, wf_ccl_list[wf_idx][:, zi], c=clr[zi], alpha=0.6)
    plt.xlabel('$z$')
    plt.ylabel(r'$W_i^X(z)$')
    plt.suptitle(f'{wf_names_list[wf_idx]}')
    plt.tight_layout()
    plt.show()

# compute cls
ccl_obj.cl_ll_3d = ccl_obj.compute_cls(
    ell_obj.ells_WL,
    ccl_obj.p_of_k_a,
    ccl_obj.wf_lensing_obj,
    ccl_obj.wf_lensing_obj,
    cl_ccl_kwargs,
)
ccl_obj.cl_gl_3d = ccl_obj.compute_cls(
    ell_obj.ells_XC,
    ccl_obj.p_of_k_a,
    ccl_obj.wf_galaxy_obj,
    ccl_obj.wf_lensing_obj,
    cl_ccl_kwargs,
)
ccl_obj.cl_gg_3d = ccl_obj.compute_cls(
    ell_obj.ells_GC,
    ccl_obj.p_of_k_a,
    ccl_obj.wf_galaxy_obj,
    ccl_obj.wf_galaxy_obj,
    cl_ccl_kwargs,
)

sl.write_cl_tab('./input', 'cl_ll', ccl_obj.cl_ll_3d, ell_obj.ells_WL, zbins)
sl.write_cl_tab('./input', 'cl_gl', ccl_obj.cl_gl_3d, ell_obj.ells_XC, zbins)
sl.write_cl_tab('./input', 'cl_gg', ccl_obj.cl_gg_3d, ell_obj.ells_GC, zbins)


if cfg['C_ell']['use_input_cls']:
    print('Using input Cls')
    cl_ll_tab = np.genfromtxt(cfg['C_ell']['cl_LL_path'])
    cl_gl_tab = np.genfromtxt(cfg['C_ell']['cl_GL_path'])
    cl_gg_tab = np.genfromtxt(cfg['C_ell']['cl_GG_path'])

    ells_WL_in, cl_ll_3d_in = sl.import_cl_tab(cl_ll_tab)
    ells_XC_in, cl_gl_3d_in = sl.import_cl_tab(cl_gl_tab)
    ells_GC_in, cl_gg_3d_in = sl.import_cl_tab(cl_gg_tab)

    if not np.allclose(ells_WL_in, ell_obj.ells_WL, atol=0, rtol=1e-5):
        cl_ll_3d_spline = CubicSpline(ells_WL_in, cl_ll_3d_in, axis=0)
        cl_ll_3d_in = cl_ll_3d_spline(ell_obj.ells_WL)

    if not np.allclose(ells_XC_in, ell_obj.ells_XC, atol=0, rtol=1e-5):
        cl_gl_3d_spline = CubicSpline(ells_XC_in, cl_gl_3d_in, axis=0)
        cl_gl_3d_in = cl_gl_3d_spline(ell_obj.ells_XC)

    if not np.allclose(ells_GC_in, ell_obj.ells_GC, atol=0, rtol=1e-5):
        cl_gg_3d_spline = CubicSpline(ells_GC_in, cl_gg_3d_in, axis=0)
        cl_gg_3d_in = cl_gg_3d_spline(ell_obj.ells_GC)

    # save the sb cls for the plot below
    cl_ll_3d_sb = ccl_obj.cl_ll_3d
    cl_gl_3d_sb = ccl_obj.cl_gl_3d
    cl_gg_3d_sb = ccl_obj.cl_gg_3d
    cl_ll_3d_sb, cl_gl_3d_sb = pyccl_interface.apply_mult_shear_bias(
        cl_ll_3d_sb, cl_gl_3d_sb, np.array(cfg['C_ell']['mult_shear_bias']), zbins
    )

    # assign them to ccl_obj
    ccl_obj.cl_ll_3d = cl_ll_3d_in
    ccl_obj.cl_gl_3d = cl_gl_3d_in
    ccl_obj.cl_gg_3d = cl_gg_3d_in


# ! add multiplicative shear bias
# ! THIS SHOULD NOT BE DONE FOR THE OC Cls!! mult shear bias values are passed
# ! in the .ini file
ccl_obj.cl_ll_3d, ccl_obj.cl_gl_3d = pyccl_interface.apply_mult_shear_bias(
    ccl_obj.cl_ll_3d, ccl_obj.cl_gl_3d, np.array(cfg['C_ell']['mult_shear_bias']), zbins
)
# TODO this simple cut will not work for different binning schemes!
ccl_obj.cl_3x2pt_5d = np.zeros((n_probes, n_probes, ell_obj.nbl_3x2pt, zbins, zbins))
ccl_obj.cl_3x2pt_5d[0, 0, :, :, :] = ccl_obj.cl_ll_3d[: ell_obj.nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[1, 0, :, :, :] = ccl_obj.cl_gl_3d[: ell_obj.nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[0, 1, :, :, :] = ccl_obj.cl_gl_3d[
    : ell_obj.nbl_3x2pt, :, :
].transpose(0, 2, 1)
ccl_obj.cl_3x2pt_5d[1, 1, :, :, :] = ccl_obj.cl_gg_3d[: ell_obj.nbl_3x2pt, :, :]

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
plt.tight_layout()
for zi in range(zbins):
    zj = zi
    ax[0].loglog(ell_obj.ells_WL, ccl_obj.cl_ll_3d[:, zi, zj], c=clr[zi], ls='-')
    ax[1].loglog(ell_obj.ells_XC, ccl_obj.cl_gl_3d[:, zi, zj], c=clr[zi], ls='-')
    ax[2].loglog(ell_obj.ells_GC, ccl_obj.cl_gg_3d[:, zi, zj], c=clr[zi], ls='-')

if cfg['C_ell']['use_input_cls']:
    for zi in range(zbins):
        zj = zi
        kw = dict(c=clr[zi], ls='', marker='.')
        ax[0].loglog(ell_obj.ells_WL, cl_ll_3d_sb[:, zi, zj], **kw)
        ax[1].loglog(ell_obj.ells_XC, cl_gl_3d_sb[:, zi, zj], **kw)
        ax[2].loglog(ell_obj.ells_GC, cl_gg_3d_sb[:, zi, zj], **kw)
    # Add style legend only to middle plot
    style_legend = ax[1].legend(
        handles=[
            plt.Line2D([], [], color='gray', ls='-', label='SB'),
            plt.Line2D([], [], color='gray', ls='', marker='.', label='Input'),
        ],
        loc='upper right',
        fontsize=8,
        frameon=False,
    )
    ax[1].add_artist(style_legend)  # Preserve after adding z-bin legend


ax[2].legend(
    [f'$z_{{{zi}}}$' for zi in range(zbins)],
    loc='upper right',
    fontsize=8,
    frameon=False,
)


ax[0].set_title('LL')
ax[1].set_title('GL')
ax[2].set_title('GG')
ax[0].set_xlabel('$\\ell$')
ax[1].set_xlabel('$\\ell$')
ax[2].set_xlabel('$\\ell$')
ax[0].set_ylabel('$C_{\\ell}$')
plt.show()


# ! BNT transform the cls (and responses?) - it's more complex since I also have to
# ! transform the noise spectra, better to transform directly the covariance matrix
if cfg['BNT']['cl_BNT_transform']:
    print('BNT-transforming the Cls...')
    assert cfg['BNT']['cov_BNT_transform'] is False, (
        'the BNT transform should be applied either to the Cls or to the covariance, '
        'not both'
    )
    cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, bnt_matrix, 'L', 'L')
    cl_3x2pt_5d = cl_utils.cl_BNT_transform_3x2pt(cl_3x2pt_5d, bnt_matrix)
    warnings.warn('you should probably BNT-transform the responses too!', stacklevel=2)
    if compute_oc_g or compute_oc_ssc or compute_oc_cng:
        raise NotImplementedError('You should cut also the OC Cls')


# ! cut datavectors and responses in the pessimistic case; be carful of WA,
# ! because it does not start from ell_min
if ell_obj.ell_max_WL == 1500:
    warnings.warn(
        'you are cutting the datavectors and responses in the pessimistic case, but is '
        'this compatible with the redshift-dependent ell cuts? Yes, this is an '
        'old warning; nonetheless, check ',
        stacklevel=2,
    )
    raise ValueError('you should check this')
    cl_ll_3d = cl_ll_3d[: ell_obj.nbl_WL, :, :]
    cl_gg_3d = cl_gg_3d[: ell_obj.nbl_GC, :, :]
    cl_3x2pt_5d = cl_3x2pt_5d[: ell_obj.nbl_3x2pt, :, :]

if cfg['ell_cuts']['center_or_min'] == 'center':
    ell_prefix = 'ell'
elif cfg['ell_cuts']['center_or_min'] == 'min':
    ell_prefix = 'ell_edges'
else:
    raise ValueError(
        'cfg["ell_cuts"]["center_or_min"] should be either "center" or "min"'
    )

# ell_dict['idxs_to_delete_dict'] = {
#     'LL': ell_utils.get_idxs_to_delete(
#         ell_dict[f'{ell_prefix}_WL'],
#         ell_cuts_dict['LL'],
#         is_auto_spectrum=True,
#         zbins=zbins,
#     ),
#     'GG': ell_utils.get_idxs_to_delete(
#         ell_dict[f'{ell_prefix}_GC'],
#         ell_cuts_dict['GG'],
#         is_auto_spectrum=True,
#         zbins=zbins,
#     ),
#     'GL': ell_utils.get_idxs_to_delete(
#         ell_dict[f'{ell_prefix}_XC'],
#         ell_cuts_dict['GL'],
#         is_auto_spectrum=False,
#         zbins=zbins,
#     ),
#     'LG': ell_utils.get_idxs_to_delete(
#         ell_dict[f'{ell_prefix}_XC'],
#         ell_cuts_dict['LG'],
#         is_auto_spectrum=False,
#         zbins=zbins,
#     ),
#     '3x2pt': ell_utils.get_idxs_to_delete_3x2pt(
#         ell_dict[f'{ell_prefix}_3x2pt'], ell_cuts_dict, zbins, cfg['covariance']
#     ),
# }

# ! 3d cl ell cuts (*after* BNT!!)
# TODO here you could implement 1d cl ell cuts (but we are cutting at the covariance
# TODO and derivatives level)
# if cfg['ell_cuts']['cl_ell_cuts']:
#     cl_ll_3d = cl_utils.cl_ell_cut(cl_ll_3d, ell_obj.ells_WL, ell_cuts_dict['LL'])
#     cl_gg_3d = cl_utils.cl_ell_cut(cl_gg_3d, ell_obj.ells_GC, ell_cuts_dict['GG'])
#     cl_3x2pt_5d = cl_utils.cl_ell_cut_3x2pt(
#         cl_3x2pt_5d, ell_cuts_dict, ell_dict['ell_3x2pt']
#     )
#     if compute_oc_g or compute_oc_ssc or compute_oc_cng:
#         raise NotImplementedError('You should cut also the OC Cls')

# re-set cls in the ccl_obj after BNT transform and/or ell cuts
# ccl_obj.cl_ll_3d = cl_ll_3d
# ccl_obj.cl_gg_3d = cl_gg_3d
# ccl_obj.cl_3x2pt_5d = cl_3x2pt_5d

# ! build covariance matrices
cov_obj = sb_cov.SpaceborneCovariance(cfg, pvt_cfg, ell_obj, bnt_matrix)
cov_obj.jl_integrator_path = './spaceborne/julia_integrator.jl'
cov_obj.fsky = mask_obj.fsky
cov_obj.set_ind_and_zpairs(ind, zbins)
cov_obj.symmetrize_output_dict = symmetrize_output_dict
cov_obj.consistency_checks()
cov_obj.set_gauss_cov(
    ccl_obj=ccl_obj, split_gaussian_cov=cfg['covariance']['split_gaussian_cov']
)

# ! =================================== OneCovariance ==================================
if compute_oc_g or compute_oc_ssc or compute_oc_cng:
    if cfg['ell_cuts']['cl_ell_cuts']:
        raise NotImplementedError(
            'TODO double check inputs in this case. This case is untested'
        )

    start_time = time.perf_counter()

    # * 1. save ingredients in ascii format
    oc_path = f'{output_path}/OneCovariance'
    if not os.path.exists(oc_path):
        os.makedirs(oc_path)

    nz_src_ascii_filename = cfg['nz']['nz_sources_filename'].replace(
        '.dat', f'_dzshifts{shift_nz}.ascii'
    )
    nz_lns_ascii_filename = cfg['nz']['nz_lenses_filename'].replace(
        '.dat', f'_dzshifts{shift_nz}.ascii'
    )
    nz_src_ascii_filename = nz_src_ascii_filename.format(**pvt_cfg)
    nz_lns_ascii_filename = nz_lns_ascii_filename.format(**pvt_cfg)
    nz_src_ascii_filename = os.path.basename(nz_src_ascii_filename)
    nz_lns_ascii_filename = os.path.basename(nz_lns_ascii_filename)
    nz_src_tosave = np.column_stack((zgrid_nz_src, nz_src))
    nz_lns_tosave = np.column_stack((zgrid_nz_lns, nz_lns))
    np.savetxt(f'{oc_path}/{nz_src_ascii_filename}', nz_src_tosave)
    np.savetxt(f'{oc_path}/{nz_lns_ascii_filename}', nz_lns_tosave)

    # oc needs finer ell sampling to avoid issues with ell bin edges
    ells_3x2pt_oc = np.geomspace(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_3x2pt'], nbl_3x2pt_oc
    )
    cl_ll_3d_oc = ccl_obj.compute_cls(
        ells_3x2pt_oc,
        ccl_obj.p_of_k_a,
        ccl_obj.wf_lensing_obj,
        ccl_obj.wf_lensing_obj,
        cl_ccl_kwargs,
    )
    cl_gl_3d_oc = ccl_obj.compute_cls(
        ells_3x2pt_oc,
        ccl_obj.p_of_k_a,
        ccl_obj.wf_galaxy_obj,
        ccl_obj.wf_lensing_obj,
        cl_ccl_kwargs,
    )
    cl_gg_3d_oc = ccl_obj.compute_cls(
        ells_3x2pt_oc,
        ccl_obj.p_of_k_a,
        ccl_obj.wf_galaxy_obj,
        ccl_obj.wf_galaxy_obj,
        cl_ccl_kwargs,
    )
    cl_3x2pt_5d_oc = np.zeros((n_probes, n_probes, nbl_3x2pt_oc, zbins, zbins))
    cl_3x2pt_5d_oc[0, 0, :, :, :] = cl_ll_3d_oc
    cl_3x2pt_5d_oc[1, 0, :, :, :] = cl_gl_3d_oc
    cl_3x2pt_5d_oc[0, 1, :, :, :] = cl_gl_3d_oc.transpose(0, 2, 1)
    cl_3x2pt_5d_oc[1, 1, :, :, :] = cl_gg_3d_oc

    cl_ll_ascii_filename = f'Cell_ll_nbl{nbl_3x2pt_oc}'
    cl_gl_ascii_filename = f'Cell_gl_nbl{nbl_3x2pt_oc}'
    cl_gg_ascii_filename = f'Cell_gg_nbl{nbl_3x2pt_oc}'
    sl.write_cl_ascii(
        oc_path, cl_ll_ascii_filename, cl_3x2pt_5d_oc[0, 0, ...], ells_3x2pt_oc, zbins
    )
    sl.write_cl_ascii(
        oc_path, cl_gl_ascii_filename, cl_3x2pt_5d_oc[1, 0, ...], ells_3x2pt_oc, zbins
    )
    sl.write_cl_ascii(
        oc_path, cl_gg_ascii_filename, cl_3x2pt_5d_oc[1, 1, ...], ells_3x2pt_oc, zbins
    )

    ascii_filenames_dict = {
        'cl_ll_ascii_filename': cl_ll_ascii_filename,
        'cl_gl_ascii_filename': cl_gl_ascii_filename,
        'cl_gg_ascii_filename': cl_gg_ascii_filename,
        'nz_src_ascii_filename': nz_src_ascii_filename,
        'nz_lns_ascii_filename': nz_lns_ascii_filename,
    }

    if cfg['covariance']['which_b1g_in_resp'] == 'from_input':
        gal_bias_ascii_filename = f'{oc_path}/gal_bias_table.ascii'
        ccl_obj.save_gal_bias_table_ascii(z_grid, gal_bias_ascii_filename)
        ascii_filenames_dict['gal_bias_ascii_filename'] = gal_bias_ascii_filename
    elif cfg['covariance']['which_b1g_in_resp'] == 'from_HOD':
        warnings.warn(
            'OneCovariance will use the HOD-derived galaxy bias '
            'for the Cls and responses',
            stacklevel=2,
        )

    # * 2. compute cov using the onecovariance interface class
    print('Start NG cov computation with OneCovariance...')
    # initialize object, build cfg file
    oc_obj = oc_interface.OneCovarianceInterface(
        cfg, pvt_cfg, do_g=compute_oc_g, do_ssc=compute_oc_ssc, do_cng=compute_oc_cng
    )
    oc_obj.oc_path = oc_path
    oc_obj.z_grid_trisp_sb = z_grid_trisp
    oc_obj.path_to_config_oc_ini = f'{oc_obj.oc_path}/input_configs.ini'
    oc_obj.ells_sb = ell_obj.ells_3x2pt
    oc_obj.build_save_oc_ini(ascii_filenames_dict, print_ini=True)

    # compute covs
    oc_obj.call_oc_from_bash()
    oc_obj.process_cov_from_list_file()
    oc_obj.output_sanity_check(rtol=1e-4)  # .dat vs .mat

    # This is an alternative method to call OC (more convoluted and more maintanable).
    # I keep the code for optional consistency checks
    if cfg['OneCovariance']['consistency_checks']:
        # store in temp variables for later check
        check_cov_sva_oc_3x2pt_10D = oc_obj.cov_sva_oc_3x2pt_10D
        check_cov_mix_oc_3x2pt_10D = oc_obj.cov_mix_oc_3x2pt_10D
        check_cov_sn_oc_3x2pt_10D = oc_obj.cov_sn_oc_3x2pt_10D
        check_cov_ssc_oc_3x2pt_10D = oc_obj.cov_ssc_oc_3x2pt_10D
        check_cov_cng_oc_3x2pt_10D = oc_obj.cov_cng_oc_3x2pt_10D

        oc_obj.call_oc_from_class()
        oc_obj.process_cov_from_class()

        # a more strict relative tolerance will make this test fail,
        # the number of digits in the .dat and .mat files is lower
        np.testing.assert_allclose(
            check_cov_sva_oc_3x2pt_10D, oc_obj.cov_sva_oc_3x2pt_10D, atol=0, rtol=1e-3
        )
        np.testing.assert_allclose(
            check_cov_mix_oc_3x2pt_10D, oc_obj.cov_mix_oc_3x2pt_10D, atol=0, rtol=1e-3
        )
        np.testing.assert_allclose(
            check_cov_sn_oc_3x2pt_10D, oc_obj.cov_sn_oc_3x2pt_10D, atol=0, rtol=1e-3
        )
        np.testing.assert_allclose(
            check_cov_ssc_oc_3x2pt_10D, oc_obj.cov_ssc_oc_3x2pt_10D, atol=0, rtol=1e-3
        )
        np.testing.assert_allclose(
            check_cov_cng_oc_3x2pt_10D, oc_obj.cov_cng_oc_3x2pt_10D, atol=0, rtol=1e-3
        )

    print(f'Time taken to compute OC: {(time.perf_counter() - start_time) / 60:.2f} m')

else:
    oc_obj = None

# ! ========================================== Spaceborne ==============================


if compute_sb_ssc:
    print('Start SSC computation with Spaceborne...')

    resp_obj = responses.SpaceborneResponses(
        cfg=cfg, k_grid=k_grid, z_grid=z_grid_trisp, ccl_obj=ccl_obj
    )
    resp_obj.use_h_units = use_h_units

    if cfg['covariance']['which_pk_responses'] == 'halo_model':
        which_b1g_in_resp = cfg['covariance']['which_b1g_in_resp']
        include_terasawa_terms = cfg['covariance']['include_terasawa_terms']
        gal_bias_2d_trisp = ccl_obj.gal_bias_func(z_grid_trisp)
        if gal_bias_2d_trisp.ndim == 1:
            assert single_b_of_z, (
                'Galaxy bias should be a single function of redshift for all bins, '
            )
            'there seems to be some inconsistency'
            gal_bias_2d_trisp = np.tile(gal_bias_2d_trisp[:, None], zbins)

        dPmm_ddeltab = np.zeros((len(k_grid), len(z_grid_trisp), zbins, zbins))
        dPgm_ddeltab = np.zeros((len(k_grid), len(z_grid_trisp), zbins, zbins))
        dPgg_ddeltab = np.zeros((len(k_grid), len(z_grid_trisp), zbins, zbins))
        # TODO this can be made more efficient - eg by having a
        # TODO "if_bias_equal_all_bins" flag

        if single_b_of_z:
            resp_obj.set_hm_resp(
                k_grid=k_grid,
                z_grid=z_grid_trisp,
                which_b1g=which_b1g_in_resp,
                b1g_zi=gal_bias_2d_trisp[:, 0],
                b1g_zj=gal_bias_2d_trisp[:, 0],
                include_terasawa_terms=include_terasawa_terms,
            )
            for zi in range(zbins):
                for zj in range(zbins):
                    dPmm_ddeltab[:, :, zi, zj] = resp_obj.dPmm_ddeltab_hm
                    dPgm_ddeltab[:, :, zi, zj] = resp_obj.dPgm_ddeltab_hm
                    dPgg_ddeltab[:, :, zi, zj] = resp_obj.dPgg_ddeltab_hm
                    # TODO check these
                    r_mm = resp_obj.r1_mm_hm
                    r_gm = resp_obj.r1_gm_hm
                    r_gg = resp_obj.r1_gg_hm

        else:
            for zi in range(zbins):
                for zj in range(zbins):
                    resp_obj.set_hm_resp(
                        k_grid=k_grid,
                        z_grid=z_grid_trisp,
                        which_b1g=which_b1g_in_resp,
                        b1g_zi=gal_bias_2d_trisp[:, zi],
                        b1g_zj=gal_bias_2d_trisp[:, zj],
                        include_terasawa_terms=include_terasawa_terms,
                    )
                    dPmm_ddeltab[:, :, zi, zj] = resp_obj.dPmm_ddeltab_hm
                    dPgm_ddeltab[:, :, zi, zj] = resp_obj.dPgm_ddeltab_hm
                    dPgg_ddeltab[:, :, zi, zj] = resp_obj.dPgg_ddeltab_hm
                    # TODO check these
                    r_mm = resp_obj.r1_mm_hm
                    r_gm = resp_obj.r1_gm_hm
                    r_gg = resp_obj.r1_gg_hm

        # reduce dimensionality
        dPmm_ddeltab = dPmm_ddeltab[:, :, 0, 0]
        dPgm_ddeltab = dPgm_ddeltab[:, :, :, 0]

    elif cfg['covariance']['which_pk_responses'] == 'separate_universe':
        resp_obj.set_g1mm_su_resp()
        r_mm = resp_obj.compute_r1_mm()
        resp_obj.set_su_resp(
            b2g_from_halomodel=True, include_b2g=cfg['covariance']['include_b2g']
        )
        r_gm = resp_obj.r1_gm
        r_gg = resp_obj.r1_gg
        b1g_hm = resp_obj.b1g_hm
        b2g_hm = resp_obj.b2g_hm

        dPmm_ddeltab = resp_obj.dPmm_ddeltab
        dPgm_ddeltab = resp_obj.dPgm_ddeltab
        dPgg_ddeltab = resp_obj.dPgg_ddeltab

    else:
        raise ValueError(
            'which_pk_responses must be either "halo_model" or "separate_universe". '
            f' Got {cfg["covariance"]["which_pk_responses"]}.'
        )

    # ! 2. prepare integrands (d2CAB_dVddeltab) and volume element

    # ! test k_max_limber vs k_max_dPk and adjust z_min accordingly
    k_max_resp = np.max(k_grid)
    ell_grid = ell_obj.ells_GC
    kmax_limber = cosmo_lib.get_kmax_limber(
        ell_grid, z_grid, use_h_units, ccl_obj.cosmo_ccl
    )

    z_grid_test = deepcopy(z_grid)
    while kmax_limber > k_max_resp:
        print(
            f'kmax_limber > k_max_dPk '
            f'({kmax_limber:.2f} {k_txt_label} > {k_max_resp:.2f} {k_txt_label}): '
            f'Increasing z_min until kmax_limber < k_max_dPk. '
            f'Alternatively, increase k_max_dPk or decrease ell_max.'
        )
        z_grid_test = z_grid_test[1:]
        kmax_limber = cosmo_lib.get_kmax_limber(
            ell_grid, z_grid_test, use_h_units, ccl_obj.cosmo_ccl
        )
        print(f'Retrying with z_min = {z_grid_test[0]:.3f}')

    dPmm_ddeltab_spline = RectBivariateSpline(
        k_grid, z_grid_trisp, dPmm_ddeltab, kx=3, ky=3
    )
    dPmm_ddeltab_klimb = np.array(
        [
            dPmm_ddeltab_spline(k_limber_func(ell_val, z_grid), z_grid, grid=False)
            for ell_val in ell_obj.ells_WL
        ]
    )

    dPgm_ddeltab_klimb = np.zeros((len(ell_obj.ells_XC), len(z_grid), zbins))
    for zi in range(zbins):
        dPgm_ddeltab_spline = RectBivariateSpline(
            k_grid, z_grid_trisp, dPgm_ddeltab[:, :, zi], kx=3, ky=3
        )
        dPgm_ddeltab_klimb[:, :, zi] = np.array(
            [
                dPgm_ddeltab_spline(k_limber_func(ell_val, z_grid), z_grid, grid=False)
                for ell_val in ell_obj.ells_XC
            ]
        )

    dPgg_ddeltab_klimb = np.zeros((len(ell_obj.ells_GC), len(z_grid), zbins, zbins))
    for zi in range(zbins):
        for zj in range(zbins):
            dPgg_ddeltab_spline = RectBivariateSpline(
                k_grid, z_grid_trisp, dPgg_ddeltab[:, :, zi, zj], kx=3, ky=3
            )
            dPgg_ddeltab_klimb[:, :, zi, zj] = np.array(
                [
                    dPgg_ddeltab_spline(
                        k_limber_func(ell_val, z_grid), z_grid, grid=False
                    )
                    for ell_val in ell_obj.ells_GC
                ]
            )

    # ! integral prefactor
    cl_integral_prefactor = cosmo_lib.cl_integral_prefactor(
        z_grid,
        cl_integral_convention_ssc,
        use_h_units=use_h_units,
        cosmo_ccl=ccl_obj.cosmo_ccl,
    )
    # ! observable densities
    # z: z_grid index (for the radial projection)
    # i, j: zbin index
    d2CLL_dVddeltab = np.einsum(
        'zi,zj,Lz->Lijz', wf_lensing, wf_lensing, dPmm_ddeltab_klimb
    )
    d2CGL_dVddeltab = np.einsum(
        'zi,zj,Lzi->Lijz', wf_delta, wf_lensing, dPgm_ddeltab_klimb
    ) + np.einsum('zi,zj,Lz->Lijz', wf_mu, wf_lensing, dPmm_ddeltab_klimb)
    d2CGG_dVddeltab = (
        np.einsum('zi,zj,Lzij->Lijz', wf_delta, wf_delta, dPgg_ddeltab_klimb)
        + np.einsum('zi,zj,Lzi->Lijz', wf_delta, wf_mu, dPgm_ddeltab_klimb)
        + np.einsum('zi,zj,Lzj->Lijz', wf_mu, wf_delta, dPgm_ddeltab_klimb)
        + np.einsum('zi,zj,Lz->Lijz', wf_mu, wf_mu, dPmm_ddeltab_klimb)
    )

    # ! 3. Compute/load/save sigma2_b
    if cfg['covariance']['load_cached_sigma2_b']:
        sigma2_b = np.load(f'{output_path}/cache/sigma2_b_{zgrid_str}.npy')

    else:
        if cfg['covariance']['use_KE_approximation']:
            # compute sigma2_b(z) (1 dimension) using the existing CCL implementation
            ccl_obj.set_sigma2_b(
                z_grid=z_grid,
                which_sigma2_b=which_sigma2_b,
                mask_obj=mask_obj,
            )
            _a, sigma2_b = ccl_obj.sigma2_b_tuple
            # quick sanity check on the a/z grid
            sigma2_b = sigma2_b[::-1]
            _z = cosmo_lib.a_to_z(_a)[::-1]
            np.testing.assert_allclose(z_grid, _z, atol=0, rtol=1e-8)

        else:
            # depending on the modules installed, integrate with levin or simpson
            # (in the latter case, in parallel or not)
            integration_scheme = 'levin' if find_spec('pylevin') else 'simps'
            parallel = bool(find_spec('pathos'))

            if integration_scheme == 'levin':
                k_grid_s2b = k_grid
            elif integration_scheme == 'simps':
                k_grid_s2b = k_grid_s2b_simps

            sigma2_b = sigma2_SSC.sigma2_z1z2_wrap_parallel(
                z_grid=z_grid,
                k_grid_sigma2=k_grid_s2b,
                cosmo_ccl=ccl_obj.cosmo_ccl,
                which_sigma2_b=which_sigma2_b,
                mask_obj=mask_obj,
                n_jobs=cfg['misc']['num_threads'],
                integration_scheme=integration_scheme,
                batch_size=cfg['misc']['levin_batch_size'],
                parallel=parallel,
            )

    if not cfg['covariance']['load_cached_sigma2_b']:
        np.save(f'{output_path}/cache/sigma2_b_{zgrid_str}.npy', sigma2_b)
        np.save(f'{output_path}/cache/zgrid_sigma2_b_{zgrid_str}.npy', z_grid)

    # ! 4. Perform the integration calling the Julia module
    print('Computing the SSC integral...')
    start = time.perf_counter()
    cov_ssc_3x2pt_dict_8D = cov_obj.ssc_integral_julia(
        d2CLL_dVddeltab=d2CLL_dVddeltab,
        d2CGL_dVddeltab=d2CGL_dVddeltab,
        d2CGG_dVddeltab=d2CGG_dVddeltab,
        cl_integral_prefactor=cl_integral_prefactor,
        sigma2=sigma2_b,
        z_grid=z_grid,
        integration_type=ssc_integration_type,
        probe_ordering=probe_ordering,
        num_threads=cfg['misc']['num_threads'],
    )
    print(f'SSC computed in {(time.perf_counter() - start) / 60:.2f} m')

    # in the full_curved_sky case only, sigma2_b has to be divided by fsky
    # TODO it would make much more sense to divide s2b directly...
    if which_sigma2_b == 'full_curved_sky':
        for key in cov_ssc_3x2pt_dict_8D:
            cov_ssc_3x2pt_dict_8D[key] /= mask_obj.fsky
    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask', 'flat_sky']:
        pass
    else:
        raise ValueError(f'which_sigma2_b = {which_sigma2_b} not recognized')

    cov_obj.cov_ssc_sb_3x2pt_dict_8D = cov_ssc_3x2pt_dict_8D

# TODO integrate this with Spaceborne_covg

# ! ========================================== PyCCL ===================================
if compute_ccl_ssc:
    # Note: this z grid has to be larger than the one requested in the trispectrum
    # (z_grid_tkka in the cfg file). You can probaby use the same grid as the
    # one used in the trispectrum, but from my tests is should be
    # zmin_s2b < zmin_s2b_tkka and zmax_s2b =< zmax_s2b_tkka.
    # if zmin=0 it looks like I can have zmin_s2b = zmin_s2b_tkka
    ccl_obj.set_sigma2_b(
        z_grid=z_default_grid_ccl,  # TODO can I not just pass z_grid here?
        fsky=cfg['mask']['fsky'],
        which_sigma2_b=which_sigma2_b,
        mask_obj=mask_obj,
    )

if compute_ccl_ssc or compute_ccl_cng:
    ccl_ng_cov_terms_list = []
    if compute_ccl_ssc:
        ccl_ng_cov_terms_list.append('SSC')
    if compute_ccl_cng:
        ccl_ng_cov_terms_list.append('cNG')

    for which_ng_cov in ccl_ng_cov_terms_list:
        ccl_obj.initialize_trispectrum(which_ng_cov, probe_ordering, cfg['PyCCL'])
        ccl_obj.compute_ng_cov_3x2pt(
            which_ng_cov,
            ell_obj.ells_GC,
            mask_obj.fsky,
            integration_method=cfg['PyCCL']['cov_integration_method'],
            probe_ordering=probe_ordering,
            ind_dict=ind_dict,
        )

# ! ========================== combine covariance terms ================================
cov_obj.build_covs(ccl_obj=ccl_obj, oc_obj=oc_obj)
cov_dict = cov_obj.cov_dict

# ! ============================ plot & tests ==========================================
for key in cov_dict:
    sl.matshow(cov_dict[key], title=key)

for which_cov in cov_dict:
    probe = which_cov.split('_')[1]
    which_ng_cov = which_cov.split('_')[2]
    ndim = which_cov.split('_')[3]
    cov_filename = cfg['covariance']['cov_filename'].format(
        which_ng_cov=which_ng_cov, probe=probe, ndim=ndim
    )
    cov_filename = cov_filename.replace('_g_', '_G_')
    cov_filename = cov_filename.replace('_ssc_', '_SSC_')
    cov_filename = cov_filename.replace('_cng_', '_cNG_')
    cov_filename = cov_filename.replace('_tot_', '_TOT_')

    if cov_filename.endswith('.npz'):
        save_func = np.savez_compressed
    elif cov_filename.endswith('.npy'):
        save_func = np.save

    save_func(f'{output_path}/{cov_filename}', cov_dict[which_cov])

    if cfg['covariance']['save_full_cov']:
        for a, b, c, d in unique_probe_comb:
            abcd_str = (
                f'{probename_dict[a]}{probename_dict[b]}'
                f'{probename_dict[c]}{probename_dict[d]}'
            )
            cov_tot_6d = (
                cov_obj.cov_3x2pt_g_10D[a, b, c, d, ...]
                + cov_obj.cov_3x2pt_ssc_10D[a, b, c, d, ...]
                + cov_obj.cov_3x2pt_cng_10D[a, b, c, d, ...]
            )
            save_func(
                f'{output_path}/cov_{abcd_str}_G_6D',
                cov_obj.cov_3x2pt_g_10D[a, b, c, d, ...],
            )
            save_func(
                f'{output_path}/cov_{abcd_str}_SSC_6D',
                cov_obj.cov_3x2pt_ssc_10D[a, b, c, d, ...],
            )
            save_func(
                f'{output_path}/cov_{abcd_str}_cNG_6D',
                cov_obj.cov_3x2pt_cng_10D[a, b, c, d, ...],
            )
            save_func(f'{output_path}/cov_{abcd_str}_TOT_6D', cov_tot_6d)

print(f'Covariance matrices saved in {output_path}\n')

# save cfg file
with open(f'{output_path}/run_config.yaml', 'w') as yaml_file:
    yaml.dump(cfg, yaml_file, default_flow_style=False)

# save ell values
header_list = ['ell', 'delta_ell', 'ell_lower_edges', 'ell_upper_edges']

# ells_ref, probably no need to save
# ells_2d_save = np.column_stack((
#     ell_ref_nbl32,
#     delta_l_ref_nbl32,
#     ell_edges_ref_nbl32[:-1],
#     ell_edges_ref_nbl32[1:],
# ))
# sl.savetxt_aligned(f'{output_path}/ell_values_ref.txt', ells_2d_save, header_list)

for probe in ['WL', 'GC', '3x2pt']:
    ells_2d_save = np.column_stack(
        (
            getattr(ell_obj, f'ells_{probe}'),
            getattr(ell_obj, f'delta_l_{probe}'),
            getattr(ell_obj, f'ell_edges_{probe}')[:-1],
            getattr(ell_obj, f'ell_edges_{probe}')[1:],
        )
    )
    sl.savetxt_aligned(
        f'{output_path}/ell_values_{probe}.txt', ells_2d_save, header_list
    )

if cfg['misc']['save_output_as_benchmark']:
    # some of the test quantities are not defined in some cases
    if not compute_sb_ssc:
        sigma2_b = np.array([])
        dPmm_ddeltab = np.array([])
        dPgm_ddeltab = np.array([])
        dPgg_ddeltab = np.array([])
        d2CLL_dVddeltab = np.array([])
        d2CGL_dVddeltab = np.array([])
        d2CGG_dVddeltab = np.array([])

    # better to work with empty arrays than None
    if bnt_matrix is None:
        _bnt_matrix = np.array([])

    # I don't fully remember why I don't save these
    _ell_dict = vars(ell_obj)
    # _ell_dict.pop('ell_cuts_dict')
    # _ell_dict.pop('idxs_to_delete_dict')

    import datetime

    branch, commit = sl.get_git_info()
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'branch': branch,
        'commit': commit,
    }

    bench_filename = cfg['misc']['bench_filename'].format(
        g_code=cfg['covariance']['G_code'],
        ssc_code=cfg['covariance']['SSC_code'] if cfg['covariance']['SSC'] else 'None',
        cng_code=cfg['covariance']['cNG_code'] if cfg['covariance']['cNG'] else 'None',
        use_KE=str(cfg['covariance']['use_KE_approximation']),
        which_pk_responses=cfg['covariance']['which_pk_responses'],
        which_b1g_in_resp=cfg['covariance']['which_b1g_in_resp'],
    )

    if os.path.exists(f'{bench_filename}.npz'):
        raise ValueError(
            'You are trying to overwrite a benchmark file. Please rename the file or '
            'delete the existing one.'
        )

    with open(f'{bench_filename}.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    np.savez_compressed(
        bench_filename,
        backup_cfg=cfg,
        ind=ind,
        z_grid=z_grid,
        z_grid_trisp=z_grid_trisp,
        k_grid=k_grid,
        k_grid_sigma2_b=k_grid_s2b_simps,
        nz_src=nz_src,
        nz_lns=nz_lns,
        **_ell_dict,
        bnt_matrix=_bnt_matrix,
        gal_bias_2d=ccl_obj.gal_bias_2d,
        mag_bias_2d=ccl_obj.mag_bias_2d,
        wf_delta=ccl_obj.wf_delta_arr,
        wf_gamma=ccl_obj.wf_gamma_arr,
        wf_ia=ccl_obj.wf_ia_arr,
        wf_mu=ccl_obj.wf_mu_arr,
        wf_lensing_arr=ccl_obj.wf_lensing_arr,
        cl_ll_3d=ccl_obj.cl_ll_3d,
        cl_gl_3d=ccl_obj.cl_gl_3d,
        cl_gg_3d=ccl_obj.cl_gg_3d,
        cl_3x2pt_5d=ccl_obj.cl_3x2pt_5d,
        sigma2_b=sigma2_b,
        dPmm_ddeltab=dPmm_ddeltab,
        dPgm_ddeltab=dPgm_ddeltab,
        dPgg_ddeltab=dPgg_ddeltab,
        d2CLL_dVddeltab=d2CLL_dVddeltab,
        d2CGL_dVddeltab=d2CGL_dVddeltab,
        d2CGG_dVddeltab=d2CGG_dVddeltab,
        **cov_dict,
        metadata=metadata,
    )

cov = (
    cov_dict['cov_3x2pt_g_2D']
    + cov_dict['cov_3x2pt_ssc_2D']
    + cov_dict['cov_3x2pt_cng_2D']
)
cov_inv = np.linalg.inv(cov)

# test simmetry
sl.compare_arrays(
    cov, cov.T, 'cov', 'cov.T', abs_val=True, log_diff=False, plot_diff_threshold=1
)

identity = cov @ cov_inv
identity_true = np.eye(cov.shape[0])

tol = 1e-4
mask = np.abs(identity) < tol
masked_identity = np.ma.masked_where(mask, identity)
sl.matshow(
    masked_identity, abs_val=True, title=f'cov @ cov_inv\n mask below {tol}', log=True
)

sl.compare_arrays(
    cov @ cov_inv,
    cov_inv @ cov,
    'cov @ cov_inv',
    'cov_inv @ cov',
    abs_val=True,
    log_diff=True,
    plot_diff_threshold=1,
)

plt.semilogy(np.linalg.eigvals(cov))

for which_cov in cov_dict:
    if '3x2pt' in which_cov and 'tot' in which_cov:
        if cfg['misc']['test_condition_number']:
            cond_number = np.linalg.cond(cov_dict[which_cov])
            print(f'Condition number of {which_cov} = {cond_number:.4e}')

        if cfg['misc']['test_cholesky_decomposition']:
            print(f'Performing Cholesky decomposition of {which_cov}...')
            try:
                np.linalg.cholesky(cov_dict[which_cov])
                print('Cholesky decomposition successful')
            except np.linalg.LinAlgError:
                print(
                    'Cholesky decomposition failed. Consider checking the condition '
                    'number or symmetry.'
                )

        if cfg['misc']['test_numpy_inversion']:
            print(f'Computing numpy inverse of {which_cov}...')
            try:
                inv_cov = np.linalg.inv(cov_dict[which_cov])
                print('Numpy inversion successful.')
                # Test correctness of inversion:
                identity_check = np.allclose(
                    np.dot(cov_dict[which_cov], inv_cov),
                    np.eye(cov_dict[which_cov].shape[0]),
                    atol=1e-9,
                    rtol=1e-7,
                )
                if identity_check:
                    print(
                        'Inverse tested successfully (M @ M^{-1} is identity). '
                        'atol=1e-9, rtol=1e-7'
                    )
                else:
                    print(
                        f'Warning: Inverse test failed for {which_cov} (M @ M^{-1} '
                        'deviates from identity). atol=0, rtol=1e-7'
                    )
            except np.linalg.LinAlgError:
                print(
                    f'Numpy inversion failed for {which_cov} : '
                    'Matrix is singular or near-singular.'
                )

        if cfg['misc']['test_symmetry']:
            if not np.allclose(
                cov_dict[which_cov], cov_dict[which_cov].T, atol=0, rtol=1e-7
            ):
                print(
                    f'Warning: Matrix {which_cov} is not symmetric. atol=0, rtol=1e-7'
                )
            else:
                print('Matrix is symmetric. atol=0, rtol=1e-7')

print(f'Finished in {(time.perf_counter() - script_start_time) / 60:.2f} minutes')


# """
# this code snippet can be appended at the end of main.py for a quick and dirty FM
# estimation. It is not intended to be used for a serious analysis.
# """

# FM_ordered_params = {
#     'Om': 0.32,
#     'Ob': 0.05,
#     'wz': -1.0,
#     'wa': 0.0,
#     'h': 0.6737,
#     'ns': 0.966,
#     's8': 0.816,
#     'logT': 7.75,
#     'Aia': 0.16,
#     'eIA': 1.66,
#     'm01': 0.0,
#     'm02': 0.0,
#     'm03': 0.0,
#     'm04': 0.0,
#     'm05': 0.0,
#     'm06': 0.0,
#     'm07': 0.0,
#     'm08': 0.0,
#     'm09': 0.0,
#     'm10': 0.0,
#     'm11': 0.0,
#     'm12': 0.0,
#     'm13': 0.0,
#     'dzWL01': -0.025749,
#     'dzWL02': 0.022716,
#     'dzWL03': -0.026032,
#     'dzWL04': 0.012594,
#     'dzWL05': 0.019285,
#     'dzWL06': 0.008326,
#     'dzWL07': 0.038207,
#     'dzWL08': 0.002732,
#     'dzWL09': 0.034066,
#     'dzWL10': 0.049479,
#     'dzWL11': 0.06649,
#     'dzWL12': 0.000815,
#     'dzWL13': 0.04907,
#     # coefficients for the polynomial magnification and galaxy bias fits
#     'bG01': 1.33291,
#     'bG02': -0.72414,
#     'bG03': 1.0183,
#     'bG04': -0.14913,
#     'bM01': -1.50685,
#     'bM02': 1.35034,
#     'bM03': 0.08321,
#     'bM04': 0.04279,
# }


# fm_cfg = {
#     'GL_or_LG': 'GL',
#     'compute_FM': True,
#     'save_FM_txt': False,
#     'save_FM_dict': True,
#     'load_preprocess_derivatives': False,
#     'which_derivatives': 'Vincenzo',  # Vincenzo or Spaceborne,
#     'derivatives_folder': '{ROOT:s}/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3_may24/OutputFiles/DataVecDers/{flat_or_nonflat:s}/{which_pk:s}/{EP_or_ED:s}{zbins:02d}',
#     'derivatives_filename': 'dDVd{param_name:s}-{probe:s}-ML{magcut_lens:03d}-MS{magcut_source:03d}-{EP_or_ED:s}{zbins:02d}.dat',
#     'derivatives_prefix': 'dDVd',
#     'derivatives_BNT_transform': False,
#     'deriv_ell_cuts': False,
#     'fm_folder': '{ROOT:s}/common_data/Spaceborne/jobs/SPV3/output/Flagship_{flagship_version}/FM/BNT_{BNT_transform:s}/ell_cuts_{ell_cuts:s}',
#     'fm_txt_filename': 'fm_txt_filename',
#     'fm_dict_filename': f'FM_dict_sigma2b_bigchanges.pickle',
#     'test_against_vincenzo': False,
#     'test_against_benchmarks': False,
#     'FM_ordered_params': FM_ordered_params,
#     'ind': ind,
#     'block_index': 'ell',
#     'zbins': zbins,
#     'compute_SSC': True,
# }


# param_names_3x2pt = [param for param in FM_ordered_params.keys() if param != 'ODE']
# nparams_tot = len(param_names_3x2pt)

# flat_or_nonflat = 'Flat'
# magcut_lens = 245  # valid for GCph
# magcut_source = 245  # valid for WL
# zmin_nz_lens = 2  # = 0.2
# zmin_nz_source = 2  # = 0.2
# zmax_nz = 25  # = 2.5
# idIA = 2
# idB = 3
# idM = 3
# idR = 1
# idBM = 3  # for the SU responses
# ep_or_ed = 'EP'
# ROOT = '/home/davide/Documenti/Lavoro/Programmi'

# variable_specs = {
#     'flat_or_nonflat': flat_or_nonflat,
#     'which_pk': 'HMCodeBar',
#     'EP_or_ED': ep_or_ed,
#     'zbins': zbins,
# }

# # list_params_to_vary = list(FM_ordered_params.keys())
# list_params_to_vary = [param for param in FM_ordered_params.keys() if param != 'ODE']
# # list_params_to_vary = ['h', 'wa', 'dzWL01', 'm06', 'bG02', 'bM02']
# # list_params_to_vary = ['bM02', ]


# # Vincenzo's derivatives
# der_prefix = fm_cfg['derivatives_prefix']
# derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs, ROOT=ROOT)
# fm_dict_filename = fm_cfg['fm_dict_filename'].format(**variable_specs, ROOT=ROOT)
# # ! get vincenzo's derivatives' parameters, to check that they match with the yaml file
# # check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
# vinc_filenames = sl.get_filenames_in_folder(derivatives_folder)
# vinc_filenames = [
#     vinc_filename
#     for vinc_filename in vinc_filenames
#     if vinc_filename.startswith(der_prefix)
# ]

# # keep only the files corresponding to the correct magcut_lens, magcut_source and zbins
# vinc_filenames = [
#     filename
#     for filename in vinc_filenames
#     if all(
#         x in filename
#         for x in [f'ML{magcut_lens}', f'MS{magcut_source}', f'{ep_or_ed}{zbins:02d}']
#     )
# ]
# vinc_filenames = [filename.replace('.dat', '') for filename in vinc_filenames]

# vinc_trimmed_filenames = [
#     vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames
# ]
# vinc_trimmed_filenames = [
#     vinc_trimmed_filename[len(der_prefix) :]
#     if vinc_trimmed_filename.startswith(der_prefix)
#     else vinc_trimmed_filename
#     for vinc_trimmed_filename in vinc_trimmed_filenames
# ]
# vinc_param_names = list(set(vinc_trimmed_filenames))
# vinc_param_names.sort()

# # ! get fiducials names and values from the yaml file
# # remove ODE if I'm studying only flat models
# if flat_or_nonflat == 'Flat' and 'ODE' in FM_ordered_params:
#     FM_ordered_params.pop('ODE')
# fm_fid_dict = FM_ordered_params
# param_names_3x2pt = list(fm_fid_dict.keys())
# fm_cfg['param_names_3x2pt'] = param_names_3x2pt
# fm_cfg['nparams_tot'] = len(param_names_3x2pt)

# # sort them to compare with vincenzo's param names
# my_sorted_param_names = param_names_3x2pt.copy()
# my_sorted_param_names.sort()

# for dzgc_param_name in [f'dzGC{zi:02d}' for zi in range(1, zbins + 1)]:
#     if (
#         dzgc_param_name in vinc_param_names
#     ):  # ! added this if statement, not very elegant
#         vinc_param_names.remove(dzgc_param_name)

# # check whether the 2 lists match and print the elements that are in one list but not in the other
# param_names_not_in_my_list = [
#     vinc_param_name
#     for vinc_param_name in vinc_param_names
#     if vinc_param_name not in my_sorted_param_names
# ]
# param_names_not_in_vinc_list = [
#     my_sorted_param_name
#     for my_sorted_param_name in my_sorted_param_names
#     if my_sorted_param_name not in vinc_param_names
# ]

# # Check if the parameter names match
# if not np.all(vinc_param_names == my_sorted_param_names):
#     # Print the mismatching parameters
#     print(
#         f'Params present in input folder but not in the cfg file: {param_names_not_in_my_list}'
#     )
#     print(
#         f'Params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}'
#     )

# # ! preprocess derivatives (or load the alreay preprocessed ones)
# if fm_cfg['load_preprocess_derivatives']:
#     warnings.warn(
#         'loading preprocessed derivatives is faster but a bit more dangerous, make sure all the specs are taken into account'
#     )
#     dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
#     dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
#     dC_3x2pt_6D = np.load(
#         f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy'
#     )

# elif not fm_cfg['load_preprocess_derivatives']:
#     der_prefix = fm_cfg['derivatives_prefix']
#     dC_dict_1D = dict(sl.get_kv_pairs_v2(derivatives_folder, 'dat'))
#     # check if dictionary is empty
#     if not dC_dict_1D:
#         raise ValueError(f'No derivatives found in folder {derivatives_folder}')

#     # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
#     dC_dict_LL_3D = {}
#     dC_dict_GG_3D = {}
#     dC_dict_3x2pt_5D = {}

#     for key in vinc_filenames:  # loop over these, I already selected ML, MS and so on
#         if not key.startswith('dDVddzGC'):
#             if 'WLO' in key:
#                 dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
#                     dC_dict_1D[key], 'WL', cfg['ell_binning']['ell_bins_ref'], zbins
#                 )[:ell_obj.nbl_WL, :, :]
#             elif 'GCO' in key:
#                 dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
#                     dC_dict_1D[key], 'GC', ell_obj.nbl_GC, zbins
#                 )
#             elif '3x2pt' in key:
#                 dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(
#                     dC_dict_1D[key], '3x2pt', ell_obj.nbl_3x2pt, zbins
#                 )

#     # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
#     dC_LL_4D_vin = sl.dC_dict_to_4D_array(
#         dC_dict_LL_3D, param_names_3x2pt, ell_obj.nbl_WL, zbins, der_prefix
#     )
#     dC_GG_4D_vin = sl.dC_dict_to_4D_array(
#         dC_dict_GG_3D, param_names_3x2pt, ell_obj.nbl_GC, zbins, der_prefix
#     )
#     dC_3x2pt_6D_vin = sl.dC_dict_to_4D_array(
#         dC_dict_3x2pt_5D, param_names_3x2pt, ell_obj.nbl_3x2pt, zbins, der_prefix, is_3x2pt=True
#     )

#     # free up memory
#     del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_3x2pt_5D

#     # save these so they can simply be imported!
#     if not os.path.exists(f'{derivatives_folder}/reshaped_into_np_arrays'):
#         os.makedirs(f'{derivatives_folder}/reshaped_into_np_arrays')
#     np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_LL_4D.npy', dC_LL_4D_vin)
#     np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_GG_4D.npy', dC_GG_4D_vin)
#     np.save(
#         f'{derivatives_folder}/reshaped_into_np_arrays/dC_3x2pt_6D.npy', dC_3x2pt_6D_vin
#     )

# deriv_dict_vin = {
#     'dC_LL_4D': dC_LL_4D_vin,
#     'dC_GG_4D': dC_GG_4D_vin,
#     'dC_3x2pt_6D': dC_3x2pt_6D_vin,
# }


# print('Starting covariance matrix inversion...')
# start_time = time.perf_counter()
# cov_WL_GO_2D_inv = np.linalg.inv(cov_dict['cov_WL_g_2D'])
# cov_GC_GO_2D_inv = np.linalg.inv(cov_dict['cov_GC_g_2D'])
# cov_XC_GO_2D_inv = np.linalg.inv(cov_dict['cov_XC_g_2D'])
# cov_3x2pt_GO_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_g_2D'])
# cov_WL_tot_2D_inv = np.linalg.inv(cov_dict['cov_WL_tot_2D'])
# cov_GC_tot_2D_inv = np.linalg.inv(cov_dict['cov_GC_tot_2D'])
# cov_XC_tot_2D_inv = np.linalg.inv(cov_dict['cov_XC_tot_2D'])
# cov_3x2pt_tot_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_tot_2D'])
# print('done in %.2f seconds' % (time.perf_counter() - start_time))


# # load reshaped derivatives, with shape (nbl, zbins, zbins, nparams)
# dC_LL_4D = deriv_dict_vin['dC_LL_4D']
# dC_GG_4D = deriv_dict_vin['dC_GG_4D']
# dC_3x2pt_6D = deriv_dict_vin['dC_3x2pt_6D']


# dC_LLfor3x2pt_4D = dC_3x2pt_6D[0, 0, :, :, :, :]
# dC_XCfor3x2pt_4D = dC_3x2pt_6D[0, 1, :, :, :, :]
# dC_GGfor3x2pt_4D = dC_3x2pt_6D[1, 1, :, :, :, :]

# # flatten z indices, obviously following the ordering given in ind
# # separate the ind for the different probes
# dC_LL_3D = sl.dC_4D_to_3D(dC_LL_4D, nbl_WL, zpairs_auto, nparams_tot, ind_auto)
# dC_GG_3D = sl.dC_4D_to_3D(dC_GG_4D, nbl_GC, zpairs_auto, nparams_tot, ind_auto)
# dC_LLfor3x2pt_3D = sl.dC_4D_to_3D(
#     dC_LLfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto
# )
# dC_XCfor3x2pt_3D = sl.dC_4D_to_3D(
#     dC_XCfor3x2pt_4D, nbl_3x2pt, zpairs_cross, nparams_tot, ind_cross
# )
# dC_GGfor3x2pt_3D = sl.dC_4D_to_3D(
#     dC_GGfor3x2pt_4D, nbl_3x2pt, zpairs_auto, nparams_tot, ind_auto
# )

# # concatenate the flattened components of the 3x2pt datavector
# dC_3x2pt_3D = np.concatenate(
#     (dC_LLfor3x2pt_3D, dC_XCfor3x2pt_3D, dC_GGfor3x2pt_3D), axis=1
# )


# # collapse ell and zpair - ATTENTION: np.reshape, like ndarray.flatten, accepts an 'ordering' parameter, which works
# # in the same way not with the old datavector, which was ordered in a different way...
# block_index = 'ell'
# if block_index in ['ell', 'vincenzo', 'C-style']:
#     which_flattening = 'C'
# elif block_index in ['ij', 'sylvain', 'F-style']:
#     which_flattening = 'F'
# else:
#     raise ValueError(
#         "block_index should be either 'ell', 'vincenzo', 'C-style', 'ij', 'sylvain' or 'F-style'"
#     )

# dC_LL_2D = np.reshape(
#     dC_LL_3D, (nbl_WL * zpairs_auto, nparams_tot), order=which_flattening
# )
# dC_GG_2D = np.reshape(
#     dC_GG_3D, (nbl_GC * zpairs_auto, nparams_tot), order=which_flattening
# )
# dC_XC_2D = np.reshape(
#     dC_XCfor3x2pt_3D, (nbl_3x2pt * zpairs_cross, nparams_tot), order=which_flattening
# )
# dC_3x2pt_2D = np.reshape(
#     dC_3x2pt_3D, (nbl_3x2pt * zpairs_3x2pt, nparams_tot), order=which_flattening
# )

# # ! cut the *flattened* derivatives vector
# # if FM_cfg['deriv_ell_cuts']:
# #     print('Performing the ell cuts on the derivatives...')
# #     dC_LL_2D = np.delete(dC_LL_2D, ell_dict['idxs_to_delete_dict']['LL'], axis=0)
# #     dC_GG_2D = np.delete(dC_GG_2D, ell_dict['idxs_to_delete_dict']['GG'], axis=0)
# #     dC_WA_2D = np.delete(dC_WA_2D, ell_dict['idxs_to_delete_dict']['WA'], axis=0)
# #     dC_XC_2D = np.delete(dC_XC_2D, ell_dict['idxs_to_delete_dict'][GL_or_LG], axis=0)
# #     dC_3x2pt_2D = np.delete(dC_3x2pt_2D, ell_dict['idxs_to_delete_dict']['3x2pt'], axis=0)
# #     # raise ValueError('the above cuts are correct, but I should be careful when defining the 2x2pt datavector/covmat,\
# #         # as n_elem_ll will be lower because of the cuts...')

# # # if the ell cuts removed all WA bins (which is in fact the case)
# # if dC_WA_2D.shape[0] == 0:
# #     dC_WA_2D = np.ones((nbl_WA * zpairs_auto, nparams_tot))

# ######################### COMPUTE FM #####################################

# start = time.perf_counter()
# FM_WL_GO = np.einsum(
#     'ia,ik,kb->ab', dC_LL_2D, cov_WL_GO_2D_inv, dC_LL_2D, optimize='optimal'
# )
# FM_GC_GO = np.einsum(
#     'ia,ik,kb->ab', dC_GG_2D, cov_GC_GO_2D_inv, dC_GG_2D, optimize='optimal'
# )
# FM_XC_GO = np.einsum(
#     'ia,ik,kb->ab', dC_XC_2D, cov_XC_GO_2D_inv, dC_XC_2D, optimize='optimal'
# )
# FM_3x2pt_GO = np.einsum(
#     'ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_GO_2D_inv, dC_3x2pt_2D, optimize='optimal'
# )
# print(f'GO FM done in {(time.perf_counter() - start):.2f} s')

# start = time.perf_counter()
# FM_WL_GS = np.einsum(
#     'ia,ik,kb->ab', dC_LL_2D, cov_WL_tot_2D_inv, dC_LL_2D, optimize='optimal'
# )
# FM_GC_GS = np.einsum(
#     'ia,ik,kb->ab', dC_GG_2D, cov_GC_tot_2D_inv, dC_GG_2D, optimize='optimal'
# )
# FM_XC_GS = np.einsum(
#     'ia,ik,kb->ab', dC_XC_2D, cov_XC_tot_2D_inv, dC_XC_2D, optimize='optimal'
# )
# FM_3x2pt_GS = np.einsum(
#     'ia,ik,kb->ab', dC_3x2pt_2D, cov_3x2pt_tot_2D_inv, dC_3x2pt_2D, optimize='optimal'
# )
# print(f'GS FM done in {(time.perf_counter() - start):.2f} s')


# # store the matrices in the dictionary
# probe_names = ['WL', 'GC', 'XC', '3x2pt']
# FMs_GO = [FM_WL_GO, FM_GC_GO, FM_XC_GO, FM_3x2pt_GO]
# FMs_GS = [FM_WL_GS, FM_GC_GS, FM_XC_GS, FM_3x2pt_GS]


# FM_dict = {}
# for probe_name, FM_GO, FM_GS in zip(probe_names, FMs_GO, FMs_GS):
#     FM_dict[f'FM_{probe_name}_G'] = FM_GO
#     FM_dict[f'FM_{probe_name}_TOT'] = FM_GS

# print('FMs computed in %.2f seconds' % (time.perf_counter() - start))


# # ! ==================================== compute and save fisher matrix ================================================


# fm_dict = FM_dict

# # ordered fiducial parameters entering the FM
# fm_dict['fiducial_values_dict'] = fm_cfg['FM_ordered_params']

# fm_folder = '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/FM/BNT_False/ell_cuts_False'
# from spaceborne import plot_lib

# fm_dict_filename = fm_cfg['fm_dict_filename']
# if fm_cfg['save_FM_dict']:
#     sl.save_pickle(f'{fm_folder}/{fm_dict_filename}', fm_dict)

# # ! plot the results directly, as a quick check
# nparams_toplot = 7
# names_params_to_fix = []
# divide_fom_by_10 = True
# include_fom = True
# which_uncertainty = 'marginal'

# fix_dz = True
# fix_shear_bias = True
# fix_gal_bias = False
# fix_mag_bias = False
# shear_bias_prior = 5e-4
# # dz_prior = np.array(2 * 1e-3 * (1 + np.array(cfg['covariance_cfg']['zbin_centers'])))

# probes = ['WL', 'GC', 'XC', '3x2pt']
# dz_param_names = [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]
# shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
# gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(4)]
# mag_bias_param_names = [f'bM{(zi + 1):02d}' for zi in range(4)]
# param_names_list = list(FM_ordered_params.keys())

# if fix_dz:
#     names_params_to_fix += dz_param_names

# if fix_shear_bias:
#     names_params_to_fix += shear_bias_param_names

# if fix_gal_bias:
#     names_params_to_fix += gal_bias_param_names

# if fix_mag_bias:
#     names_params_to_fix += mag_bias_param_names

# fom_dict = {}
# uncert_dict = {}
# masked_fm_dict = {}
# masked_fid_pars_dict = {}
# perc_diff_probe = {}
# fm_dict_toplot = deepcopy(fm_dict)
# del fm_dict_toplot['fiducial_values_dict']
# for key in list(fm_dict_toplot.keys()):
#     if key != 'fiducial_values_dict' and '_WA_' not in key and '_2x2pt_' not in key:
#         print(key)

#         fm = deepcopy(fm_dict_toplot[key])

#         masked_fm_dict[key], masked_fid_pars_dict[key] = sl.mask_fm_v2(
#             fm,
#             FM_ordered_params,
#             names_params_to_fix=names_params_to_fix,
#             remove_null_rows_cols=True,
#         )

#         if not fix_shear_bias and any(item in key for item in ['WL', 'XC', '3x2pt']):
#             print(f'adding shear bias Gaussian prior to {key}')
#             shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
#             masked_fm_dict[key] = sl.add_prior_to_fm(
#                 masked_fm_dict[key],
#                 masked_fid_pars_dict[key],
#                 shear_bias_param_names,
#                 shear_bias_prior_values,
#             )

#         if not fix_dz:
#             print(f'adding dz Gaussian prior to {key}')
#             masked_fm_dict[key] = sl.add_prior_to_fm(
#                 masked_fm_dict[key], masked_fid_pars_dict[key], dz_param_names, dz_prior
#             )

#         uncert_dict[key] = sl.uncertainties_fm_v2(
#             masked_fm_dict[key],
#             masked_fid_pars_dict[key],
#             which_uncertainty=which_uncertainty,
#             normalize=True,
#             percent_units=True,
#         )[:nparams_toplot]

#         param_names = list(masked_fid_pars_dict[key].keys())
#         cosmo_param_names = list(masked_fid_pars_dict[key].keys())[:nparams_toplot]

#         w0wa_idxs = param_names.index('wz'), param_names.index('wa')
#         fom_dict[key] = sl.compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)

# # compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
# for probe in probes:
#     key_a = f'FM_{probe}_G'
#     key_b = f'FM_{probe}_TOT'

#     uncert_dict[f'perc_diff_{probe}_G'] = sl.percent_diff(
#         uncert_dict[key_b], uncert_dict[key_a]
#     )
#     fom_dict[f'perc_diff_{probe}_G'] = np.abs(
#         sl.percent_diff(fom_dict[key_b], fom_dict[key_a])
#     )

#     nparams_toplot = 7
#     divide_fom_by_10_plt = False if probe in ('WLXC') else divide_fom_by_10

#     cases_to_plot = [
#         f'FM_{probe}_G',
#         f'FM_{probe}_TOT',
#         # f'FM_{probe}_GSSCcNG',
#         f'perc_diff_{probe}_G',
#         #  f'FM_{probe}_{which_ng_cov_suffix}',
#         #  f'perc_diff_{probe}_{which_ng_cov_suffix}',
#     ]

#     # # transform dict. into an array and add the fom
#     uncert_array, fom_array = [], []

#     for case in cases_to_plot:
#         uncert_array.append(uncert_dict[case])
#         if divide_fom_by_10 and 'FM' in case and 'WL' not in case:
#             fom_dict[case] /= 10
#         fom_array.append(fom_dict[case])

#     uncert_array = np.asarray(uncert_array)
#     fom_array = np.asarray(fom_array)

#     uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

#     perc_diff_probe[probe] = np.append(
#         uncert_dict[f'perc_diff_{probe}_G'], fom_dict[f'perc_diff_{probe}_G']
#     )

#     # label and title stuff
#     fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
#     param_names_label = (
#         param_names_list[:nparams_toplot] + [fom_label]
#         if include_fom
#         else param_names_list[:nparams_toplot]
#     )
#     lmax = (
#         cfg['ell_binning'][f'ell_max_{probe}']
#         if probe in ['WL', 'GC']
#         else cfg['ell_binning']['ell_max_3x2pt']
#     )
#     title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i, zsteps %s\n%s uncertainties' % (
#         probe,
#         lmax,
#         ep_or_ed,
#         zbins,
#         len(z_grid),
#         which_uncertainty,
#     )

#     # bar plot
#     if include_fom:
#         nparams_toplot = 8

#     for i, case in enumerate(cases_to_plot):
#         cases_to_plot[i] = case
#         if 'OneCovariance' in cases_to_plot[i]:
#             cases_to_plot[i] = cases_to_plot[i].replace('OneCovariance', 'OneCov')
#         if f'PySSC_{probe}_G' in cases_to_plot[i]:
#             cases_to_plot[i] = cases_to_plot[i].replace(
#                 f'PySSC_{probe}_G', f'{probe}_G'
#             )

#         cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
#         cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
#         cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
#         cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')
#         cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')

#     plot_lib.bar_plot(
#         uncert_array[:, :nparams_toplot],
#         title,
#         cases_to_plot,
#         nparams=nparams_toplot,
#         param_names_label=param_names_label,
#         bar_width=0.13,
#         include_fom=include_fom,
#         divide_fom_by_10_plt=divide_fom_by_10_plt,
#     )

# # ! % diff for the 3 probes - careful about the plot title
# perc_diff_probe.pop('XC')
# plot_lib.bar_plot(
#     np.array(list(perc_diff_probe.values())),
#     title + r', % diff (G + SSC + cNG)/G',
#     (list(perc_diff_probe.keys())),
#     nparams=nparams_toplot,
#     param_names_label=param_names_label,
#     bar_width=0.13,
#     include_fom=include_fom,
#     divide_fom_by_10_plt=False,
# )

# # ! Print tables

# # if include_fom:
# #     nparams_toplot_ref = nparams_toplot
# #     nparams_toplot = nparams_toplot_ref + 1
# # titles = param_names_list[:nparams_toplot_ref] + ['FoM']

# # # for uncert_dict, _, name in zip([uncert_dict, uncert_dict], [fm_dict, fm_dict_vin], ['Davide', 'Vincenzo']):
# # print(f"G uncertainties [%]:")
# # data = []
# # for probe in probes:
# #     uncerts = [f'{uncert:.3f}' for uncert in uncert_dict[f'FM_{probe}_G']]
# #     fom = f'{fom_dict[f"FM_{probe}_G"]:.2f}'
# #     data.append([probe] + uncerts + [fom])
# # print(tabulate(data, headers=titles, tablefmt="pretty"))

# # print(f"GSSC/G ratio  :")
# # data = []
# # table = []  # tor tex
# # for probe in probes:
# #     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'ratio_{probe}_G']]
# #     fom = f'{fom_dict[f"ratio_{probe}_G"]:.2f}'
# #     data.append([probe] + ratios + [fom])
# #     table.append(ratios + [fom])
# # print(tabulate(data, headers=titles, tablefmt="pretty"))

# # print(f"SSC % increase :")
# # data = []
# # for probe in probes:
# #     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'perc_diff_{probe}_G']]
# #     fom = f'{fom_dict[f"perc_diff_{probe}_G"]:.2f}'
# #     data.append([probe] + ratios + [fom])
# # print(tabulate(data, headers=titles, tablefmt="pretty"))

# # ! quickly compare two selected FMs
# # TODO this is misleading, understand better why (comparing GSSC, not perc_diff)

# fm_dict_of_dicts = {
#     'develop': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_bigchanges.pickle'),
#     'levin': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_levindav.pickle'),
#     # 'current': fm_dict,
# }


# labels = list(fm_dict_of_dicts.keys())
# fm_dict_list = list(fm_dict_of_dicts.values())
# keys_toplot_in = ['FM_WL_TOT', 'FM_GC_TOT', 'FM_XC_TOT', 'FM_3x2pt_TOT']
# # keys_toplot = 'all'
# colors = [
#     'tab:blue',
#     'tab:green',
#     'tab:orange',
#     'tab:red',
#     'tab:cyan',
#     'tab:grey',
#     'tab:olive',
#     'tab:purple',
# ]

# reference = 'first_key'
# nparams_toplot_in = 8
# normalize_by_gauss = True

# sl.compare_fm_constraints(
#     *fm_dict_list,
#     labels=labels,
#     keys_toplot_in=keys_toplot_in,
#     normalize_by_gauss=True,
#     which_uncertainty='marginal',
#     reference=reference,
#     colors=colors,
#     abs_FoM=True,
#     save_fig=False,
#     fig_path='/home/davide/Scrivania/',
# )

# assert False, 'stop here'

# fisher_matrices = (
#     fm_dict_of_dicts['SB_hm_simpker']['FM_3x2pt_GSSC'],
#     fm_dict_of_dicts['SB_KEapp_hm_simpker']['FM_3x2pt_GSSC'],
#     fm_dict_of_dicts['OC_simpker']['FM_3x2pt_GSSC'],
#     fm_dict_of_dicts['current']['FM_3x2pt_GSSC'],
# )
# fiducials = list(fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values())
# # fiducials = (
# # fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values(),
# # fm_dict_of_dicts['SB_KEapp_hm_simpker']['fiducial_values_dict'].values(),
# # fm_dict_of_dicts['OC_simpker']['fiducial_values_dict'].values(),
# # fm_dict_of_dicts['current']['fiducial_values_dict'].values(),
# # )
# param_names_list = list(
#     fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].keys()
# )
# param_names_labels_toplot = param_names_list[:8]
# plot_lib.triangle_plot(
#     fisher_matrices,
#     fiducials,
#     title,
#     labels,
#     param_names_list,
#     param_names_labels_toplot,
#     param_names_labels_tex=None,
#     rotate_param_labels=False,
#     contour_colors=None,
#     line_colors=None,
# )


# print(
#     'Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60)
# )
