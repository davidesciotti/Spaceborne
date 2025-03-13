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
from scipy.interpolate import RectBivariateSpline
from spaceborne import (
    bnt,
    cl_utils,
    config_checker,
    cosmo_lib,
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


def load_config():
    # Check if we're running in a Jupyter environment (or interactive mode)
    if 'ipykernel_launcher.py' in sys.argv[0]:
        # Running interactively, so use default config file
        config_path = 'config.yaml'

    else:
        parser = argparse.ArgumentParser(description='Spaceborne')
        parser.add_argument(
            '--config',
            type=str,
            help='Path to the configuration file',
            required=False,
            default='config.yaml',
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
dzWL_fiducial = cfg['nz']['dzWL']
dzGC_fiducial = cfg['nz']['dzGC']
nz_gaussian_smoothing = cfg['nz'][
    'nz_gaussian_smoothing'
]  # does not seem to have a large effect...
nz_gaussian_smoothing_sigma = cfg['nz']['nz_gaussian_smoothing_sigma']
shift_nz = cfg['nz']['shift_nz']
normalize_shifted_nz = cfg['nz']['normalize_shifted_nz']
zbins = len(
    cfg['nz']['ngal_lenses']
)  # this has the same length as ngal_sources, as checked below
ell_max_WL = cfg['ell_binning']['ell_max_WL']
ell_max_GC = cfg['ell_binning']['ell_max_GC']
ell_max_3x2pt = cfg['ell_binning']['ell_max_3x2pt']
nbl_WL_opt = cfg['ell_binning']['nbl_WL_opt']
triu_tril = cfg['covariance']['triu_tril']
row_col_major = cfg['covariance']['row_col_major']
n_probes = cfg['covariance']['n_probes']
which_sigma2_b = cfg['covariance']['which_sigma2_b']
include_ia_in_bnt_kernel_for_zcuts = cfg['BNT']['include_ia_in_bnt_kernel_for_zcuts']
compute_bnt_with_shifted_nz_for_zcuts = cfg['BNT'][
    'compute_bnt_with_shifted_nz_for_zcuts'
]
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
# ! END HARDCODED OPTIONS/PARAMETERS

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
a_default_grid_ccl = ccl_obj.cosmo_ccl.get_pk_spline_a()
z_default_grid_ccl = cosmo_lib.a_to_z(a_default_grid_ccl)[::-1]
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

# ! do the same for CCL - i.e., set the above in the ccl_obj with little variations
# ! (e.g. a instead of z)
# TODO I leave the option to use a grid for the CCL, but I am not sure if it is needed
ccl_obj.z_grid_tkka_SSC = z_grid_trisp
ccl_obj.z_grid_tkka_cNG = z_grid_trisp
ccl_obj.a_grid_tkka_SSC = cosmo_lib.z_to_a(z_grid_trisp)[::-1]
ccl_obj.a_grid_tkka_cNG = cosmo_lib.z_to_a(z_grid_trisp)[::-1]
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

# build the ind array and store it into the covariance dictionary
zpairs_auto, zpairs_cross, zpairs_3x2pt = sl.get_zpairs(zbins)
ind = sl.build_full_ind(triu_tril, row_col_major, zbins)
ind_auto = ind[:zpairs_auto, :].copy()
ind_cross = ind[zpairs_auto : zpairs_cross + zpairs_auto, :].copy()
ind_dict = {('L', 'L'): ind_auto, ('G', 'L'): ind_cross, ('G', 'G'): ind_auto}

# ! Import redshift distributions
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

# ! compute ell values, ell bins and delta ell
# TODO add option to import ell values
# TODO _WL_opt should be called "ref"
ell_dict = {}
if cfg['ell_binning']['binning_type'] == 'unbinned':
    ell_dict['ell_WL'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_WL'] + 1
    )
    ell_dict['ell_GC'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_GC'] + 1
    )
    ell_dict['ell_3x2pt'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_3x2pt'] + 1
    )
    ell_dict['ell_XC'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_3x2pt'] + 1
    )

    # delta_ell values, needed for gaussian covariance (if binned in this way)
    ell_dict['delta_l_WL'] = np.ones(len(ell_dict['ell_WL']))
    ell_dict['delta_l_GC'] = np.ones(len(ell_dict['ell_GC']))
    ell_dict['delta_l_3x2pt'] = np.ones(len(ell_dict['ell_3x2pt']))

    # TODO this is a bit sloppy
    ell_dict['ell_edges_WL'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_WL'] + 2
    )
    ell_dict['ell_edges_GC'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_GC'] + 2
    )
    ell_dict['ell_edges_3x2pt'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_3x2pt'] + 2
    )
    ell_dict['ell_edges_XC'] = np.arange(
        cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_3x2pt'] + 2
    )

else:
    # compute ell and delta ell values in the reference (optimistic) case
    ell_ref_nbl32, delta_l_ref_nbl32, ell_edges_ref_nbl32 = ell_utils.compute_ells(
        nbl=cfg['ell_binning']['nbl_WL_opt'],
        ell_min=cfg['ell_binning']['ell_min'],
        ell_max=cfg['ell_binning']['ell_max_WL_opt'],
        recipe='ISTF',
        output_ell_bin_edges=True,
    )

    # perform the cuts (not the redshift-dependent ones!) on the ell centers and edges
    ell_dict['ell_WL'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_WL])
    ell_dict['ell_GC'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_GC])
    ell_dict['ell_3x2pt'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_3x2pt])
    ell_dict['ell_XC'] = np.copy(ell_dict['ell_3x2pt'])

    # TODO why not save all edges??
    # store edges *except last one for dimensional consistency* in the ell_dict
    mask_wl = (ell_edges_ref_nbl32 < ell_max_WL) | np.isclose(
        ell_edges_ref_nbl32, ell_max_WL, atol=0, rtol=1e-5
    )
    mask_gc = (ell_edges_ref_nbl32 < ell_max_GC) | np.isclose(
        ell_edges_ref_nbl32, ell_max_GC, atol=0, rtol=1e-5
    )
    mask_3x2pt = (ell_edges_ref_nbl32 < ell_max_3x2pt) | np.isclose(
        ell_edges_ref_nbl32, ell_max_3x2pt, atol=0, rtol=1e-5
    )
    ell_dict['ell_edges_WL'] = np.copy(ell_edges_ref_nbl32[mask_wl])
    ell_dict['ell_edges_GC'] = np.copy(ell_edges_ref_nbl32[mask_gc])
    ell_dict['ell_edges_3x2pt'] = np.copy(ell_edges_ref_nbl32[mask_3x2pt])
    ell_dict['ell_edges_XC'] = np.copy(ell_dict['ell_edges_3x2pt'])

    # delta_ell values, needed for gaussian covariance (if binned in this way)
    ell_dict['delta_l_WL'] = np.copy(delta_l_ref_nbl32[: len(ell_dict['ell_WL'])])
    ell_dict['delta_l_GC'] = np.copy(delta_l_ref_nbl32[: len(ell_dict['ell_GC'])])
    ell_dict['delta_l_3x2pt'] = np.copy(delta_l_ref_nbl32[: len(ell_dict['ell_3x2pt'])])


# set the corresponding number of ell bins
nbl_WL = len(ell_dict['ell_WL'])
nbl_GC = len(ell_dict['ell_GC'])
nbl_3x2pt = nbl_GC

# checks
for key in ell_dict:
    if ell_dict[key].size == 0:
        raise ValueError(f'ell values for key {key} must be non-empty')

assert (
    len(ell_dict['ell_3x2pt']) == len(ell_dict['ell_XC']) == len(ell_dict['ell_GC'])
), '3x2pt, XC and GC should  have the same number of ell bins'
assert np.all(ell_dict['ell_3x2pt'] == ell_dict['ell_XC']), (
    '3x2pt and XC should have the same ell values'
)
assert np.all(ell_dict['ell_3x2pt'] == ell_dict['ell_GC']), (
    '3x2pt and GC should have the same ell values'
)
assert nbl_WL == nbl_3x2pt == nbl_GC, 'use the same number of bins for the moment'


# provate cfg dictionary. This serves a couple different purposeses:
# 1. To store and pass hardcoded parameters in a convenient way
# 2. To make the .format() more compact
pvt_cfg = {
    'zbins': zbins,
    'ind': ind,
    'probe_ordering': probe_ordering,
    'ell_min': cfg['ell_binning']['ell_min'],
    'ell_max_WL': ell_max_WL,
    'ell_max_GC': ell_max_GC,
    'ell_max_3x2pt': ell_max_3x2pt,
    'nbl_WL': nbl_WL,
    'nbl_GC': nbl_GC,
    'nbl_3x2pt': nbl_3x2pt,
    'which_ng_cov': cov_terms_str,
    'cov_terms_list': cov_terms_list,
    'GL_OR_LG': GL_OR_LG,
    'symmetrize_output_dict': symmetrize_output_dict,
    'use_h_units': use_h_units,
    'z_grid': z_grid,
    'ells_sb': ell_dict['ell_3x2pt'],
}

# TODO delete this? maybe I still want to print some of these options...
# pp.pprint(pvt_cfg)


# ! START SCALE CUTS: for these, we need to:
# 1. Compute the BNT. This is done with the raw, or unshifted n(z), but only for
# the purpose of computing the ell cuts - the rest of the code uses a BNT matrix
# from the shifted n(z) - see also comment below.
# 2. compute the kernels for the un-shifted n(z) (for consistency)
# 3. bnt-transform these kernels (for lensing, it's only the gamma kernel),
# and use these to:
# 4. compute the z means
# 5. compute the ell cuts

# 1. Compute BNT
assert compute_bnt_with_shifted_nz_for_zcuts is False, (
    'The BNT used to compute the z_means and ell cuts is just for a simple case: '
    'no IA, no dz shift'
)
assert shift_nz is True, (
    'The signal (and BNT used to transform it) is computed with a shifted n(z); '
    'You could use an un-shifted n(z) for the BNT, but'
    'this would be slightly inconsistent (but also what I did so far).'
)
assert include_ia_in_bnt_kernel_for_zcuts is False, (
    'We compute the BNT just for a simple case: no IA, no shift. '
    'This is because we want to compute the z means'
)

# * IMPORTANT NOTE: The BNT should be computed from the same n(z) (shifted or not)
# which is then used to compute the kernels which are then used to get the z_means,
# and finally the ell_cuts, for consistency. In other words, we cannot compute the
# kernels with a shifted n(z) and transform them with a BNT computed from the
# unshifted n(z) and viceversa. If the n(z) are shifted, one of the BNT kernels will
# become negative, but this is just because two of the original kernels get very close
# after the shift: the transformation is correct. Having said that, I leave the code
# below in case we want to change this in the future
if nz_gaussian_smoothing:
    nz_src = wf_cl_lib.gaussian_smmothing_nz(
        zgrid_nz_src, nz_unshifted_src, nz_gaussian_smoothing_sigma, plot=True
    )
    nz_lns = wf_cl_lib.gaussian_smmothing_nz(
        zgrid_nz_lns, nz_unshifted_lns, nz_gaussian_smoothing_sigma, plot=True
    )

if compute_bnt_with_shifted_nz_for_zcuts:
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

bnt_matrix = bnt.compute_bnt_matrix(
    zbins, zgrid_nz_src, nz_src, cosmo_ccl=ccl_obj.cosmo_ccl, plot_nz=False
)

# 2. compute the kernels for the un-shifted n(z) (for consistency)
ccl_obj.set_nz(
    nz_full_src=np.hstack((zgrid_nz_src[:, None], nz_src)),
    nz_full_lns=np.hstack((zgrid_nz_lns[:, None], nz_lns)),
)
ccl_obj.check_nz_tuple(zbins)
ccl_obj.set_ia_bias_tuple(z_grid_src=z_grid, has_ia=cfg['C_ell']['has_IA'])

# ! set galaxy and magnification bias
if cfg['C_ell']['which_gal_bias'] == 'from_input':
    gal_bias_input = np.genfromtxt(cfg['C_ell']['gal_bias_table_filename'])
    ccl_obj.gal_bias_2d, ccl_obj.gal_bias_func = sl.check_interpolate_input_tab(
        input_tab=gal_bias_input, z_grid_out=z_grid, zbins=zbins
    )
    ccl_obj.gal_bias_tuple = (z_grid, ccl_obj.gal_bias_2d)


# TODO the alternative should be the HOD gal bias already set in the responses class!!
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

# 3. ! bnt-transform these kernels (for lensing, it's only the gamma kernel, without IA)
wf_gamma_ccl_bnt = (bnt_matrix @ ccl_obj.wf_gamma_arr.T).T

# 4. compute the z means
z_means_ll = wf_cl_lib.get_z_means(z_grid, ccl_obj.wf_gamma_arr)
z_means_gg = wf_cl_lib.get_z_means(z_grid, ccl_obj.wf_galaxy_arr)
z_means_ll_bnt = wf_cl_lib.get_z_means(z_grid, wf_gamma_ccl_bnt)

# plt.figure()
# for zi in range(zbins):
#     plt.plot(z_grid, ccl_obj.wf_gamma_arr[:, zi], ls='-', c=clr[zi],
#              alpha=0.6, label='wf_gamma_ccl' if zi == 0 else None)
#     plt.plot(z_grid, wf_gamma_ccl_bnt[:, zi], ls='--', c=clr[zi],
#              alpha=0.6, label='wf_gamma_ccl_bnt' if zi == 0 else None)
#     plt.axvline(z_means_ll_bnt[zi], ls=':', c=clr[zi])
# plt.legend()
# plt.xlabel('$z$')
# plt.ylabel(r'$W_i^{\gamma}(z)$')

assert np.all(np.diff(z_means_ll) > 0), 'z_means_ll should be monotonically increasing'
assert np.all(np.diff(z_means_gg) > 0), 'z_means_gg should be monotonically increasing'
assert np.all(np.diff(z_means_ll_bnt) > 0), (
    'z_means_ll_bnt should be monotonically increasing '
    '(not a strict condition, valid only if we do not shift the n(z) in this part)'
)

# 5. compute the ell cuts
ell_cuts_dict = {}
ellcuts_kw = {
    'kmax_h_over_Mpc': kmax_h_over_Mpc,
    'cosmo_ccl': ccl_obj.cosmo_ccl,
    'zbins': zbins,
    'h': h,
    'kmax_h_over_Mpc_ref': cfg['ell_cuts']['kmax_h_over_Mpc_ref'],
}
ell_cuts_dict['LL'] = ell_utils.load_ell_cuts(
    z_values_a=z_means_ll_bnt, z_values_b=z_means_ll_bnt, **ellcuts_kw
)
ell_cuts_dict['GG'] = ell_utils.load_ell_cuts(
    z_values_a=z_means_gg, z_values_b=z_means_gg, **ellcuts_kw
)
ell_cuts_dict['GL'] = ell_utils.load_ell_cuts(
    z_values_a=z_means_gg, z_values_b=z_means_ll_bnt, **ellcuts_kw
)
ell_cuts_dict['LG'] = ell_utils.load_ell_cuts(
    z_values_a=z_means_ll_bnt, z_values_b=z_means_gg, **ellcuts_kw
)
ell_dict['ell_cuts_dict'] = (
    ell_cuts_dict  # this is to pass the ell cuts to the covariance module
)
# ! END SCALE CUTS

# now compute the BNT used for the rest of the code
if shift_nz:
    nz_src = wf_cl_lib.shift_nz(
        zgrid_nz_src,
        nz_unshifted_src,
        dzWL_fiducial,
        normalize=normalize_shifted_nz,
        plot_nz=False,
        interpolation_kind=shift_nz_interpolation_kind,
    )
    nz_lns = wf_cl_lib.shift_nz(
        zgrid_nz_lns,
        nz_unshifted_lns,
        dzGC_fiducial,
        normalize=normalize_shifted_nz,
        plot_nz=False,
        interpolation_kind=shift_nz_interpolation_kind,
    )
    # * this is important: the BNT matrix I use for the rest of the code (so not to
    # * compute the ell cuts) is instead
    # * consistent with the shifted n(z) used to compute the kernels
    bnt_matrix = bnt.compute_bnt_matrix(
        zbins, zgrid_nz_src, nz_src, cosmo_ccl=ccl_obj.cosmo_ccl, plot_nz=False
    )

wf_cl_lib.plot_nz_src_lns(zgrid_nz_src, nz_src, zgrid_nz_lns, nz_lns, colors=clr)

# re-set n(z) used in CCL class, then re-compute kernels
ccl_obj.set_nz(
    nz_full_src=np.hstack((zgrid_nz_src[:, None], nz_src)),
    nz_full_lns=np.hstack((zgrid_nz_lns[:, None], nz_lns)),
)
ccl_obj.set_kernel_obj(cfg['C_ell']['has_rsd'], cfg['PyCCL']['n_samples_wf'])
ccl_obj.set_kernel_arr(
    z_grid_wf=z_grid, has_magnification_bias=cfg['C_ell']['has_magnification_bias']
)

gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias)'
ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_wo_gal_bias_arr

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
    ell_dict['ell_WL'],
    ccl_obj.p_of_k_a,
    ccl_obj.wf_lensing_obj,
    ccl_obj.wf_lensing_obj,
    cl_ccl_kwargs,
)
ccl_obj.cl_gl_3d = ccl_obj.compute_cls(
    ell_dict['ell_XC'],
    ccl_obj.p_of_k_a,
    ccl_obj.wf_galaxy_obj,
    ccl_obj.wf_lensing_obj,
    cl_ccl_kwargs,
)
ccl_obj.cl_gg_3d = ccl_obj.compute_cls(
    ell_dict['ell_GC'],
    ccl_obj.p_of_k_a,
    ccl_obj.wf_galaxy_obj,
    ccl_obj.wf_galaxy_obj,
    cl_ccl_kwargs,
)

# ! add multiplicative shear bias
# ! THIS SHOULD NOT BE DONE FOR THE OC Cls!! mult shear bias values are passed
# ! in the .ini file
mult_shear_bias = np.array(cfg['C_ell']['mult_shear_bias'])
assert len(mult_shear_bias) == zbins, (
    'mult_shear_bias should be a vector of length zbins'
)
if not np.all(mult_shear_bias == 0):
    print('applying multiplicative shear bias')
    print(f'mult_shear_bias = {mult_shear_bias}')
    for ell_idx, _ in enumerate(ccl_obj.cl_ll_3d.shape[0]):
        for zi in range(zbins):
            for zj in range(zbins):
                ccl_obj.cl_ll_3d[ell_idx, zi, zj] *= (1 + mult_shear_bias[zi]) * (
                    1 + mult_shear_bias[zj]
                )

    for ell_idx, _ in enumerate(ccl_obj.cl_gl_3d.shape[0]):
        for zi in range(zbins):
            for zj in range(zbins):
                ccl_obj.cl_gl_3d[ell_idx, zi, zj] *= 1 + mult_shear_bias[zj]

ccl_obj.cl_3x2pt_5d = np.zeros((n_probes, n_probes, nbl_3x2pt, zbins, zbins))
ccl_obj.cl_3x2pt_5d[0, 0, :, :, :] = ccl_obj.cl_ll_3d[:nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[1, 0, :, :, :] = ccl_obj.cl_gl_3d[:nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[0, 1, :, :, :] = ccl_obj.cl_gl_3d[:nbl_3x2pt, :, :].transpose(
    0, 2, 1
)
ccl_obj.cl_3x2pt_5d[1, 1, :, :, :] = ccl_obj.cl_gg_3d[:nbl_3x2pt, :, :]

cl_ll_3d, cl_gl_3d, cl_gg_3d = ccl_obj.cl_ll_3d, ccl_obj.cl_gl_3d, ccl_obj.cl_gg_3d
cl_3x2pt_5d = ccl_obj.cl_3x2pt_5d


fig, ax = plt.subplots(1, 3)
plt.tight_layout()
for zi in range(zbins):
    zj = zi
    ax[0].loglog(ell_dict['ell_WL'], ccl_obj.cl_ll_3d[:, zi, zj], c=clr[zi])
    ax[1].loglog(ell_dict['ell_XC'], ccl_obj.cl_gl_3d[:, zi, zj], c=clr[zi])
    ax[2].loglog(ell_dict['ell_GC'], ccl_obj.cl_gg_3d[:, zi, zj], c=clr[zi])
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
if ell_max_WL == 1500:
    warnings.warn(
        'you are cutting the datavectors and responses in the pessimistic case, but is '
        'this compatible with the redshift-dependent ell cuts? Yes, this is an '
        'old warning; nonetheless, check ',
        stacklevel=2,
    )
    raise ValueError('you should check this')
    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
    cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

if cfg['ell_cuts']['center_or_min'] == 'center':
    ell_prefix = 'ell'
elif cfg['ell_cuts']['center_or_min'] == 'min':
    ell_prefix = 'ell_edges'
else:
    raise ValueError(
        'cfg["ell_cuts"]["center_or_min"] should be either "center" or "min"'
    )

ell_dict['idxs_to_delete_dict'] = {
    'LL': ell_utils.get_idxs_to_delete(
        ell_dict[f'{ell_prefix}_WL'],
        ell_cuts_dict['LL'],
        is_auto_spectrum=True,
        zbins=zbins,
    ),
    'GG': ell_utils.get_idxs_to_delete(
        ell_dict[f'{ell_prefix}_GC'],
        ell_cuts_dict['GG'],
        is_auto_spectrum=True,
        zbins=zbins,
    ),
    'GL': ell_utils.get_idxs_to_delete(
        ell_dict[f'{ell_prefix}_XC'],
        ell_cuts_dict['GL'],
        is_auto_spectrum=False,
        zbins=zbins,
    ),
    'LG': ell_utils.get_idxs_to_delete(
        ell_dict[f'{ell_prefix}_XC'],
        ell_cuts_dict['LG'],
        is_auto_spectrum=False,
        zbins=zbins,
    ),
    '3x2pt': ell_utils.get_idxs_to_delete_3x2pt(
        ell_dict[f'{ell_prefix}_3x2pt'], ell_cuts_dict, zbins, cfg['covariance']
    ),
}

# ! 3d cl ell cuts (*after* BNT!!)
# TODO here you could implement 1d cl ell cuts (but we are cutting at the covariance
# TODO and derivatives level)
if cfg['ell_cuts']['cl_ell_cuts']:
    cl_ll_3d = cl_utils.cl_ell_cut(cl_ll_3d, ell_dict['ell_WL'], ell_cuts_dict['LL'])
    cl_gg_3d = cl_utils.cl_ell_cut(cl_gg_3d, ell_dict['ell_GC'], ell_cuts_dict['GG'])
    cl_3x2pt_5d = cl_utils.cl_ell_cut_3x2pt(
        cl_3x2pt_5d, ell_cuts_dict, ell_dict['ell_3x2pt']
    )
    if compute_oc_g or compute_oc_ssc or compute_oc_cng:
        raise NotImplementedError('You should cut also the OC Cls')

# re-set cls in the ccl_obj after BNT transform and/or ell cuts
ccl_obj.cl_ll_3d = cl_ll_3d
ccl_obj.cl_gg_3d = cl_gg_3d
ccl_obj.cl_3x2pt_5d = cl_3x2pt_5d

# ! build covariance matrices
cov_obj = sb_cov.SpaceborneCovariance(cfg, pvt_cfg, ell_dict, bnt_matrix)
cov_obj.jl_integrator_path = './spaceborne/julia_integrator.jl'
cov_obj.set_ind_and_zpairs(ind, zbins)
cov_obj.symmetrize_output_dict = symmetrize_output_dict
cov_obj.consistency_checks()
cov_obj.set_gauss_cov(
    ccl_obj=ccl_obj, split_gaussian_cov=cfg['covariance']['split_gaussian_cov']
)

# ! ========================================== OneCovariance ===========================
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
    oc_obj.ells_sb = ell_dict['ell_3x2pt']
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
    ell_grid = ell_dict['ell_3x2pt']
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
            for ell_val in ell_dict['ell_WL']
        ]
    )

    dPgm_ddeltab_klimb = np.zeros((len(ell_dict['ell_XC']), len(z_grid), zbins))
    for zi in range(zbins):
        dPgm_ddeltab_spline = RectBivariateSpline(
            k_grid, z_grid_trisp, dPgm_ddeltab[:, :, zi], kx=3, ky=3
        )
        dPgm_ddeltab_klimb[:, :, zi] = np.array(
            [
                dPgm_ddeltab_spline(k_limber_func(ell_val, z_grid), z_grid, grid=False)
                for ell_val in ell_dict['ell_XC']
            ]
        )

    dPgg_ddeltab_klimb = np.zeros((len(ell_dict['ell_GC']), len(z_grid), zbins, zbins))
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
                    for ell_val in ell_dict['ell_GC']
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
        sigma2_b = np.load(f'{output_path}/cache/sigma2_b.npy')

    else:
        if cfg['covariance']['use_KE_approximation']:
            # compute sigma2_b(z) (1 dimension) using the existing CCL implementation
            ccl_obj.set_sigma2_b(
                z_grid=z_grid,
                fsky=cfg['mask']['fsky'],
                which_sigma2_b=which_sigma2_b,
                nside_mask=cfg['mask']['nside_mask'],
                mask_path=cfg['mask']['mask_path'],
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
                area_deg2_in=cfg['mask']['survey_area_deg2'],
                nside_mask=cfg['mask']['nside_mask'],
                mask_path=cfg['mask']['nside_mask'],
                n_jobs=cfg['misc']['num_threads'],
                parallel=parallel,
                integration_scheme=integration_scheme,
                batch_size=cfg['misc']['levin_batch_size'],
            )

    if not cfg['covariance']['load_cached_sigma2_b']:
        np.save(f'{output_path}/cache/sigma2_b.npy', sigma2_b)
        np.save(f'{output_path}/cache/zgrid_sigma2_b.npy', z_grid)

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
            cov_ssc_3x2pt_dict_8D[key] /= cfg['mask']['fsky']
    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask', 'flat_sky']:
        pass
    else:
        raise ValueError(f'which_sigma2_b = {which_sigma2_b} not recognized')

    cov_obj.cov_ssc_sb_3x2pt_dict_8D = cov_ssc_3x2pt_dict_8D

# TODO integrate this with Spaceborne_covg

# ! ========================================== PyCCL ===================================
if compute_ccl_ssc or compute_ccl_cng:
    # Note: this z grid has to be larger than the one requested in the trispectrum
    # (z_grid_tkka in the cfg file). You can probaby use the same grid as the
    # one used in the trispectrum, but from my tests is should be
    # zmin_s2b < zmin_s2b_tkka and zmax_s2b =< zmax_s2b_tkka.
    # if zmin=0 it looks like I can have zmin_s2b = zmin_s2b_tkka
    ccl_obj.set_sigma2_b(
        z_grid=z_default_grid_ccl,
        fsky=cfg['mask']['fsky'],
        which_sigma2_b=which_sigma2_b,
        nside_mask=cfg['mask']['nside_mask'],
        mask_path=cfg['mask']['mask_path'],
    )

    ccl_ng_cov_terms_list = []
    if compute_ccl_ssc:
        ccl_ng_cov_terms_list.append('SSC')
    if compute_ccl_cng:
        ccl_ng_cov_terms_list.append('cNG')

    for which_ng_cov in ccl_ng_cov_terms_list:
        ccl_obj.initialize_trispectrum(which_ng_cov, probe_ordering, cfg['PyCCL'])
        ccl_obj.compute_ng_cov_3x2pt(
            which_ng_cov,
            ell_dict['ell_3x2pt'],
            cfg['mask']['fsky'],
            integration_method=cfg['PyCCL']['cov_integration_method'],
            probe_ordering=probe_ordering,
            ind_dict=ind_dict,
        )

# ! ========================================== combine covariance terms ================
cov_obj.build_covs(ccl_obj=ccl_obj, oc_obj=oc_obj)
cov_dict = cov_obj.cov_dict

# ! ========================================== plot & tests ============================
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
            ell_dict[f'ell_{probe}'],
            ell_dict[f'delta_l_{probe}'],
            ell_dict[f'ell_edges_{probe}'][:-1],
            ell_dict[f'ell_edges_{probe}'][1:],
        )
    )
    sl.savetxt_aligned(
        f'{output_path}/ell_values_{probe}.txt', ells_2d_save, header_list
    )

if cfg['misc']['save_output_as_benchmark']:
    if not compute_sb_ssc:
        sigma2_b = None
        dPmm_ddeltab = None
        dPgm_ddeltab = None
        dPgg_ddeltab = None
        d2CLL_dVddeltab = None
        d2CGL_dVddeltab = None
        d2CGG_dVddeltab = None

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

    _ell_dict = deepcopy(ell_dict)
    _ell_dict.pop('ell_cuts_dict')
    _ell_dict.pop('idxs_to_delete_dict')

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
        nbl_WL=nbl_WL,
        nbl_GC=nbl_GC,
        nbl_3x2pt=nbl_3x2pt,
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


#  TODO This section has been temporarily restored, to be deleted in public branch

FM_ordered_params = {
    'Om': 0.32,
    'Ob': 0.05,
    'wz': -1.0,
    'wa': 0.0,
    'h': 0.6737,
    'ns': 0.966,
    's8': 0.816,
    'logT': 7.75,
    'Aia': 0.16,
    'eIA': 1.66,
    'm01': 0.0,
    'm02': 0.0,
    'm03': 0.0,
    'm04': 0.0,
    'm05': 0.0,
    'm06': 0.0,
    'm07': 0.0,
    'm08': 0.0,
    'm09': 0.0,
    'm10': 0.0,
    'm11': 0.0,
    'm12': 0.0,
    'm13': 0.0,
    'dzWL01': -0.025749,
    'dzWL02': 0.022716,
    'dzWL03': -0.026032,
    'dzWL04': 0.012594,
    'dzWL05': 0.019285,
    'dzWL06': 0.008326,
    'dzWL07': 0.038207,
    'dzWL08': 0.002732,
    'dzWL09': 0.034066,
    'dzWL10': 0.049479,
    'dzWL11': 0.06649,
    'dzWL12': 0.000815,
    'dzWL13': 0.04907,
    # coefficients for the polynomial magnification and galaxy bias fits
    'bG01': 1.33291,
    'bG02': -0.72414,
    'bG03': 1.0183,
    'bG04': -0.14913,
    'bM01': -1.50685,
    'bM02': 1.35034,
    'bM03': 0.08321,
    'bM04': 0.04279,
}


fm_cfg = {
    'GL_or_LG': 'GL',
    'compute_FM': True,
    'save_FM_txt': False,
    'save_FM_dict': True,
    'load_preprocess_derivatives': False,
    'which_derivatives': 'Vincenzo',  # Vincenzo or Spaceborne,
    'derivatives_folder': '{ROOT:s}/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3_may24/OutputFiles/DataVecDers/{flat_or_nonflat:s}/{which_pk:s}/{EP_or_ED:s}{zbins:02d}',
    'derivatives_filename': 'dDVd{param_name:s}-{probe:s}-ML{magcut_lens:03d}-MS{magcut_source:03d}-{EP_or_ED:s}{zbins:02d}.dat',
    'derivatives_prefix': 'dDVd',
    'derivatives_BNT_transform': False,
    'deriv_ell_cuts': False,
    'fm_folder': '{ROOT:s}/common_data/Spaceborne/jobs/SPV3/output/Flagship_{flagship_version}/FM/BNT_{BNT_transform:s}/ell_cuts_{ell_cuts:s}',
    'fm_txt_filename': 'fm_txt_filename',
    'fm_dict_filename': f'FM_dict_sigma2b_simpsdav.pickle',
    'test_against_vincenzo': False,
    'test_against_benchmarks': False,
    'FM_ordered_params': FM_ordered_params,
    'ind': ind,
    'block_index': 'ell',
    'zbins': zbins,
    'compute_SSC': True,
}


from spaceborne import fisher_matrix as fm_utils

flat_or_nonflat = 'Flat'
magcut_lens = 245  # valid for GCph
magcut_source = 245  # valid for WL
zmin_nz_lens = 2  # = 0.2
zmin_nz_source = 2  # = 0.2
zmax_nz = 25  # = 2.5
idIA = 2
idB = 3
idM = 3
idR = 1
idBM = 3  # for the SU responses
ep_or_ed = 'EP'
ROOT = '/home/davide/Documenti/Lavoro/Programmi'

variable_specs = {
    'flat_or_nonflat': flat_or_nonflat,
    'which_pk': 'HMCodeBar',
    'EP_or_ED': ep_or_ed,
    'zbins': zbins,
}

# list_params_to_vary = list(FM_ordered_params.keys())
list_params_to_vary = [param for param in FM_ordered_params.keys() if param != 'ODE']
# list_params_to_vary = ['h', 'wa', 'dzWL01', 'm06', 'bG02', 'bM02']
# list_params_to_vary = ['bM02', ]


# Vincenzo's derivatives
der_prefix = fm_cfg['derivatives_prefix']
derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs, ROOT=ROOT)
fm_dict_filename = fm_cfg['fm_dict_filename'].format(**variable_specs, ROOT=ROOT)
# ! get vincenzo's derivatives' parameters, to check that they match with the yaml file
# check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
vinc_filenames = sl.get_filenames_in_folder(derivatives_folder)
vinc_filenames = [
    vinc_filename
    for vinc_filename in vinc_filenames
    if vinc_filename.startswith(der_prefix)
]

# keep only the files corresponding to the correct magcut_lens, magcut_source and zbins
vinc_filenames = [
    filename
    for filename in vinc_filenames
    if all(
        x in filename
        for x in [f'ML{magcut_lens}', f'MS{magcut_source}', f'{ep_or_ed}{zbins:02d}']
    )
]
vinc_filenames = [filename.replace('.dat', '') for filename in vinc_filenames]

vinc_trimmed_filenames = [
    vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames
]
vinc_trimmed_filenames = [
    vinc_trimmed_filename[len(der_prefix) :]
    if vinc_trimmed_filename.startswith(der_prefix)
    else vinc_trimmed_filename
    for vinc_trimmed_filename in vinc_trimmed_filenames
]
vinc_param_names = list(set(vinc_trimmed_filenames))
vinc_param_names.sort()

# ! get fiducials names and values from the yaml file
# remove ODE if I'm studying only flat models
if flat_or_nonflat == 'Flat' and 'ODE' in FM_ordered_params:
    FM_ordered_params.pop('ODE')
fm_fid_dict = FM_ordered_params
param_names_3x2pt = list(fm_fid_dict.keys())
fm_cfg['param_names_3x2pt'] = param_names_3x2pt
fm_cfg['nparams_tot'] = len(param_names_3x2pt)

# sort them to compare with vincenzo's param names
my_sorted_param_names = param_names_3x2pt.copy()
my_sorted_param_names.sort()

for dzgc_param_name in [f'dzGC{zi:02d}' for zi in range(1, zbins + 1)]:
    if (
        dzgc_param_name in vinc_param_names
    ):  # ! added this if statement, not very elegant
        vinc_param_names.remove(dzgc_param_name)

# check whether the 2 lists match and print the elements that are in one list but not in the other
param_names_not_in_my_list = [
    vinc_param_name
    for vinc_param_name in vinc_param_names
    if vinc_param_name not in my_sorted_param_names
]
param_names_not_in_vinc_list = [
    my_sorted_param_name
    for my_sorted_param_name in my_sorted_param_names
    if my_sorted_param_name not in vinc_param_names
]

# Check if the parameter names match
if not np.all(vinc_param_names == my_sorted_param_names):
    # Print the mismatching parameters
    print(
        f'Params present in input folder but not in the cfg file: {param_names_not_in_my_list}'
    )
    print(
        f'Params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}'
    )

# ! preprocess derivatives (or load the alreay preprocessed ones)
if fm_cfg['load_preprocess_derivatives']:
    warnings.warn(
        'loading preprocessed derivatives is faster but a bit more dangerous, make sure all the specs are taken into account'
    )
    dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
    dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
    dC_3x2pt_6D = np.load(
        f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy'
    )

elif not fm_cfg['load_preprocess_derivatives']:
    der_prefix = fm_cfg['derivatives_prefix']
    dC_dict_1D = dict(sl.get_kv_pairs_v2(derivatives_folder, 'dat'))
    # check if dictionary is empty
    if not dC_dict_1D:
        raise ValueError(f'No derivatives found in folder {derivatives_folder}')

    # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
    dC_dict_LL_3D = {}
    dC_dict_GG_3D = {}
    dC_dict_3x2pt_5D = {}

    for key in vinc_filenames:  # loop over these, I already selected ML, MS and so on
        if not key.startswith('dDVddzGC'):
            if 'WLO' in key:
                dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], 'WL', nbl_WL_opt, zbins
                )[:nbl_WL, :, :]
            elif 'GCO' in key:
                dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], 'GC', nbl_GC, zbins
                )
            elif '3x2pt' in key:
                dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(
                    dC_dict_1D[key], '3x2pt', nbl_3x2pt, zbins
                )

    # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
    dC_LL_4D_vin = fm_utils.dC_dict_to_4D_array(
        dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix
    )
    dC_GG_4D_vin = fm_utils.dC_dict_to_4D_array(
        dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix
    )
    dC_3x2pt_6D_vin = fm_utils.dC_dict_to_4D_array(
        dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins, der_prefix, is_3x2pt=True
    )

    # free up memory
    del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_3x2pt_5D

    # save these so they can simply be imported!
    if not os.path.exists(f'{derivatives_folder}/reshaped_into_np_arrays'):
        os.makedirs(f'{derivatives_folder}/reshaped_into_np_arrays')
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_LL_4D.npy', dC_LL_4D_vin)
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_GG_4D.npy', dC_GG_4D_vin)
    np.save(
        f'{derivatives_folder}/reshaped_into_np_arrays/dC_3x2pt_6D.npy', dC_3x2pt_6D_vin
    )

deriv_dict_vin = {
    'dC_LL_4D': dC_LL_4D_vin,
    'dC_GG_4D': dC_GG_4D_vin,
    'dC_3x2pt_6D': dC_3x2pt_6D_vin,
}

# ! ==================================== compute and save fisher matrix ================================================
fm_dict_vin = fm_utils.compute_FM(
    cfg['covariance'], fm_cfg, ell_dict, cov_dict, deriv_dict_vin, bnt_matrix
)

# TODO finish testing derivatives
# fm_dict_dav = fm_utils.compute_FM(cfg, ell_dict, cov_dict, deriv_dict_dav, bnt_matrix)
# fm_dict_vin_modified = {key + '_vin': value for key, value in fm_dict_vin.items()}
# del fm_dict_vin_modified['fiducial_values_dict_vin']
# fm_dict = {**fm_dict_dav, **fm_dict_vin_modified}

fm_dict = fm_dict_vin

# ordered fiducial parameters entering the FM
fm_dict['fiducial_values_dict'] = fm_cfg['FM_ordered_params']

fm_folder = '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/FM/BNT_False/ell_cuts_False'
from spaceborne import plot_lib

fm_dict_filename = fm_cfg['fm_dict_filename']
if fm_cfg['save_FM_dict']:
    sl.save_pickle(f'{fm_folder}/{fm_dict_filename}', fm_dict)

# ! plot the results directly, as a quick check
nparams_toplot = 7
names_params_to_fix = []
divide_fom_by_10 = True
include_fom = True
which_uncertainty = 'marginal'

fix_dz = True
fix_shear_bias = True
fix_gal_bias = False
fix_mag_bias = False
shear_bias_prior = 5e-4
# dz_prior = np.array(2 * 1e-3 * (1 + np.array(cfg['covariance_cfg']['zbin_centers'])))

probes = ['WL', 'GC', 'XC', '3x2pt']
dz_param_names = [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]
shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(4)]
mag_bias_param_names = [f'bM{(zi + 1):02d}' for zi in range(4)]
param_names_list = list(FM_ordered_params.keys())

if fix_dz:
    names_params_to_fix += dz_param_names

if fix_shear_bias:
    names_params_to_fix += shear_bias_param_names

if fix_gal_bias:
    names_params_to_fix += gal_bias_param_names

if fix_mag_bias:
    names_params_to_fix += mag_bias_param_names

fom_dict = {}
uncert_dict = {}
masked_fm_dict = {}
masked_fid_pars_dict = {}
perc_diff_probe = {}
fm_dict_toplot = deepcopy(fm_dict)
del fm_dict_toplot['fiducial_values_dict']
for key in list(fm_dict_toplot.keys()):
    if key != 'fiducial_values_dict' and '_WA_' not in key and '_2x2pt_' not in key:
        print(key)

        fm = deepcopy(fm_dict_toplot[key])

        masked_fm_dict[key], masked_fid_pars_dict[key] = sl.mask_fm_v2(
            fm,
            FM_ordered_params,
            names_params_to_fix=names_params_to_fix,
            remove_null_rows_cols=True,
        )

        if not fix_shear_bias and any(item in key for item in ['WL', 'XC', '3x2pt']):
            print(f'adding shear bias Gaussian prior to {key}')
            shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
            masked_fm_dict[key] = sl.add_prior_to_fm(
                masked_fm_dict[key],
                masked_fid_pars_dict[key],
                shear_bias_param_names,
                shear_bias_prior_values,
            )

        if not fix_dz:
            print(f'adding dz Gaussian prior to {key}')
            masked_fm_dict[key] = sl.add_prior_to_fm(
                masked_fm_dict[key], masked_fid_pars_dict[key], dz_param_names, dz_prior
            )

        uncert_dict[key] = sl.uncertainties_fm_v2(
            masked_fm_dict[key],
            masked_fid_pars_dict[key],
            which_uncertainty=which_uncertainty,
            normalize=True,
            percent_units=True,
        )[:nparams_toplot]

        param_names = list(masked_fid_pars_dict[key].keys())
        cosmo_param_names = list(masked_fid_pars_dict[key].keys())[:nparams_toplot]

        w0wa_idxs = param_names.index('wz'), param_names.index('wa')
        fom_dict[key] = sl.compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)

# compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
for probe in probes:
    key_a = f'FM_{probe}_G'
    key_b = f'FM_{probe}_Gtot'

    uncert_dict[f'perc_diff_{probe}_G'] = sl.percent_diff(
        uncert_dict[key_b], uncert_dict[key_a]
    )
    fom_dict[f'perc_diff_{probe}_G'] = np.abs(
        sl.percent_diff(fom_dict[key_b], fom_dict[key_a])
    )

    nparams_toplot = 7
    divide_fom_by_10_plt = False if probe in ('WLXC') else divide_fom_by_10

    cases_to_plot = [
        f'FM_{probe}_G',
        f'FM_{probe}_Gtot',
        # f'FM_{probe}_GSSCcNG',
        f'perc_diff_{probe}_G',
        #  f'FM_{probe}_{which_ng_cov_suffix}',
        #  f'perc_diff_{probe}_{which_ng_cov_suffix}',
    ]

    # # transform dict. into an array and add the fom
    uncert_array, fom_array = [], []

    for case in cases_to_plot:
        uncert_array.append(uncert_dict[case])
        if divide_fom_by_10 and 'FM' in case and 'WL' not in case:
            fom_dict[case] /= 10
        fom_array.append(fom_dict[case])

    uncert_array = np.asarray(uncert_array)
    fom_array = np.asarray(fom_array)

    uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

    perc_diff_probe[probe] = np.append(
        uncert_dict[f'perc_diff_{probe}_G'], fom_dict[f'perc_diff_{probe}_G']
    )

    # label and title stuff
    fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
    param_names_label = (
        param_names_list[:nparams_toplot] + [fom_label]
        if include_fom
        else param_names_list[:nparams_toplot]
    )
    lmax = (
        cfg['ell_binning'][f'ell_max_{probe}']
        if probe in ['WL', 'GC']
        else cfg['ell_binning']['ell_max_3x2pt']
    )
    title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i, zsteps %s\n%s uncertainties' % (
        probe,
        lmax,
        ep_or_ed,
        zbins,
        len(z_grid),
        which_uncertainty,
    )

    # bar plot
    if include_fom:
        nparams_toplot = 8

    for i, case in enumerate(cases_to_plot):
        cases_to_plot[i] = case
        if 'OneCovariance' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace('OneCovariance', 'OneCov')
        if f'PySSC_{probe}_G' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace(
                f'PySSC_{probe}_G', f'{probe}_G'
            )

        cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
        cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')
        cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')

    plot_lib.bar_plot(
        uncert_array[:, :nparams_toplot],
        title,
        cases_to_plot,
        nparams=nparams_toplot,
        param_names_label=param_names_label,
        bar_width=0.13,
        include_fom=include_fom,
        divide_fom_by_10_plt=divide_fom_by_10_plt,
    )

# ! % diff for the 3 probes - careful about the plot title
perc_diff_probe.pop('XC')
plot_lib.bar_plot(
    np.array(list(perc_diff_probe.values())),
    title + r', % diff (G + SSC + cNG)/G',
    (list(perc_diff_probe.keys())),
    nparams=nparams_toplot,
    param_names_label=param_names_label,
    bar_width=0.13,
    include_fom=include_fom,
    divide_fom_by_10_plt=False,
)

# ! Print tables

# if include_fom:
#     nparams_toplot_ref = nparams_toplot
#     nparams_toplot = nparams_toplot_ref + 1
# titles = param_names_list[:nparams_toplot_ref] + ['FoM']

# # for uncert_dict, _, name in zip([uncert_dict, uncert_dict], [fm_dict, fm_dict_vin], ['Davide', 'Vincenzo']):
# print(f"G uncertainties [%]:")
# data = []
# for probe in probes:
#     uncerts = [f'{uncert:.3f}' for uncert in uncert_dict[f'FM_{probe}_G']]
#     fom = f'{fom_dict[f"FM_{probe}_G"]:.2f}'
#     data.append([probe] + uncerts + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# print(f"GSSC/G ratio  :")
# data = []
# table = []  # tor tex
# for probe in probes:
#     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'ratio_{probe}_G']]
#     fom = f'{fom_dict[f"ratio_{probe}_G"]:.2f}'
#     data.append([probe] + ratios + [fom])
#     table.append(ratios + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# print(f"SSC % increase :")
# data = []
# for probe in probes:
#     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'perc_diff_{probe}_G']]
#     fom = f'{fom_dict[f"perc_diff_{probe}_G"]:.2f}'
#     data.append([probe] + ratios + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

# ! quickly compare two selected FMs
# TODO this is misleading, understand better why (comparing GSSC, not perc_diff)

fm_dict_of_dicts = {
    'simps': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_simpsdav.pickle'),
    'levin': sl.load_pickle(f'{fm_folder}/FM_dict_sigma2b_levindav.pickle'),
    # 'current': fm_dict,
}


labels = list(fm_dict_of_dicts.keys())
fm_dict_list = list(fm_dict_of_dicts.values())
keys_toplot_in = ['FM_WL_Gtot', 'FM_GC_Gtot', 'FM_XC_Gtot', 'FM_3x2pt_Gtot']
# keys_toplot = 'all'
colors = [
    'tab:blue',
    'tab:green',
    'tab:orange',
    'tab:red',
    'tab:cyan',
    'tab:grey',
    'tab:olive',
    'tab:purple',
]

reference = 'first_key'
nparams_toplot_in = 8
normalize_by_gauss = True

sl.compare_fm_constraints(
    *fm_dict_list,
    labels=labels,
    keys_toplot_in=keys_toplot_in,
    normalize_by_gauss=True,
    which_uncertainty='marginal',
    reference=reference,
    colors=colors,
    abs_FoM=True,
    save_fig=False,
    fig_path='/home/davide/Scrivania/',
)

assert False, 'stop here'

fisher_matrices = (
    fm_dict_of_dicts['SB_hm_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['SB_KEapp_hm_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['OC_simpker']['FM_3x2pt_GSSC'],
    fm_dict_of_dicts['current']['FM_3x2pt_GSSC'],
)
fiducials = list(fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values())
# fiducials = (
# fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['SB_KEapp_hm_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['OC_simpker']['fiducial_values_dict'].values(),
# fm_dict_of_dicts['current']['fiducial_values_dict'].values(),
# )
param_names_list = list(
    fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].keys()
)
param_names_labels_toplot = param_names_list[:8]
plot_lib.triangle_plot(
    fisher_matrices,
    fiducials,
    title,
    labels,
    param_names_list,
    param_names_labels_toplot,
    param_names_labels_tex=None,
    rotate_param_labels=False,
    contour_colors=None,
    line_colors=None,
)


print(
    'Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60)
)
