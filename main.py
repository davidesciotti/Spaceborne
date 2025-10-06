# ruff: noqa: E402 (ignore module import not on top of the file warnings)
import argparse
import contextlib
import gc
import itertools
import os
import sys

import yaml


def load_config(_config_path):
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
            '--show-plots',
            action='store_true',
            help='Show plots if specified',
            required=False,
        )
        args = parser.parse_args()
        config_path = args.config

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return cfg


cfg = load_config('config.yaml')
# Set jax platform
if cfg['misc']['jax_platform'] == 'auto':
    pass
else:
    os.environ['JAX_PLATFORMS'] = cfg['misc']['jax_platform']

# if using the CPU, set the number of threads
num_threads = cfg['misc']['num_threads']
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
os.environ['XLA_FLAGS'] = (
    f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={str(num_threads)}'
)


import pprint
import time
import warnings
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.integrate import simpson as simps
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.ndimage import gaussian_filter1d

from spaceborne import (
    bnt,
    ccl_interface,
    config_checker,
    cosmo_lib,
    cov_harmonic_space,
    ell_utils,
    io_handler,
    mask_utils,
    oc_interface,
    responses,
    wf_cl_lib,
)
from spaceborne import constants as const
from spaceborne import plot_lib as sb_plt
from spaceborne import sb_lib as sl

with contextlib.suppress(ImportError):
    import pyfiglet

    text = 'Spaceborne'
    ascii_art = pyfiglet.figlet_format(text=text, font='slant')
    print(ascii_art)

if 'ipykernel_launcher.py' not in sys.argv[0] and '--show-plots' not in sys.argv:
    matplotlib.use('Agg')

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


def plot_cls():
    _, ax = plt.subplots(1, 3, figsize=(15, 4))
    # plt.tight_layout()

    # cls are (for the moment) in the ccl obj, whether they are imported from input
    # files or not
    for zi in range(zbins):
        zj = zi
        kw = {'c': clr[zi], 'ls': '-', 'marker': '.'}
        ax[0].loglog(ell_obj.ells_WL, ccl_obj.cl_ll_3d[:, zi, zj], **kw)
        ax[1].loglog(ell_obj.ells_XC, ccl_obj.cl_gl_3d[:, zi, zj], **kw)
        ax[2].loglog(ell_obj.ells_GC, ccl_obj.cl_gg_3d[:, zi, zj], **kw)

    # if input cls are used, then overplot the sb predictions on top
    if cfg['C_ell']['use_input_cls']:
        for zi in range(zbins):
            zj = zi
            sb_kw = {'c': clr[zi], 'ls': '', 'marker': 'x'}
            ax[0].loglog(ell_obj.ells_WL, cl_ll_3d_sb[:, zi, zj], **sb_kw)
            ax[1].loglog(ell_obj.ells_XC, cl_gl_3d_sb[:, zi, zj], **sb_kw)
            ax[2].loglog(ell_obj.ells_GC, cl_gg_3d_sb[:, zi, zj], **sb_kw)
        # Add style legend only to middle plot
        style_legend = ax[1].legend(
            handles=[
                plt.Line2D([], [], label='input', **kw),
                plt.Line2D([], [], label='SB', **sb_kw),
            ],
            loc='upper right',
            fontsize=16,
            frameon=False,
        )
        ax[1].add_artist(style_legend)  # Preserve after adding z-bin legend

    ax[2].legend(
        [f'$z_{{{zi}}}$' for zi in range(zbins)],
        loc='upper right',
        fontsize=16,
        frameon=False,
    )

    ax[0].set_title('LL')
    ax[1].set_title('GL')
    ax[2].set_title('GG')
    ax[0].set_xlabel('$\\ell$')
    ax[1].set_xlabel('$\\ell$')
    ax[2].set_xlabel('$\\ell$')
    ax[0].set_ylabel('$C_{\\ell}$')
    # increase font size
    for axi in ax:
        for item in (
            [axi.title, axi.xaxis.label, axi.yaxis.label]
            + axi.get_xticklabels()
            + axi.get_yticklabels()
        ):
            item.set_fontsize(16)
    plt.show()


# ! ====================================================================================
# ! ================================== PREPARATION =====================================
# ! ====================================================================================


# ! set some convenence variables, just to make things more readable
h = cfg['cosmology']['h']
galaxy_bias_fit_fiducials = np.array(cfg['C_ell']['galaxy_bias_fit_coeff'])
magnification_bias_fit_fiducials = np.array(
    cfg['C_ell']['magnification_bias_fit_coeff']
)
# this has the same length as ngal_sources, as checked below
zbins = len(cfg['nz']['ngal_lenses'])
output_path = cfg['misc']['output_path']
clr = cm.rainbow(np.linspace(0, 1, zbins))  # pylint: disable=E1101
shift_nz = cfg['nz']['shift_nz']

obs_space = cfg['probe_selection']['space']


# ! check/create paths
if not os.path.exists(output_path):
    raise FileNotFoundError(
        f'Output path {output_path} does not exist. '
        'Please create it before running the script.'
    )
for subdir in ['cache', 'cache/trispectrum/SSC', 'cache/trispectrum/cNG']:
    os.makedirs(f'{output_path}/{subdir}', exist_ok=True)

# ! START HARDCODED OPTIONS/PARAMETERS
use_h_units = False  # whether or not to normalize Megaparsecs by little h

ell_max_max = max(cfg['binning']['ell_max_WL'], cfg['binning']['ell_max_GC'])
ell_min_unb_oc = 2
ell_max_unb_oc = 5000 if ell_max_max < 5000 else ell_max_max
nbl_3x2pt_oc = 500
# for the Gaussian covariance computation
k_steps_sigma2_simps = 20_000
k_steps_sigma2_levin = 300
shift_nz_interpolation_kind = 'linear'

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


# these are configs which should not be visible to the user
cfg['covariance']['n_probes'] = 2

if 'G_code' not in cfg['covariance']:
    cfg['covariance']['G_code'] = 'Spaceborne'
if 'SSC_code' not in cfg['covariance']:
    cfg['covariance']['SSC_code'] = 'Spaceborne'
if 'cNG_code' not in cfg['covariance']:
    cfg['covariance']['cNG_code'] = 'PyCCL'

if 'OneCovariance' not in cfg:
    cfg['OneCovariance'] = {}
    cfg['OneCovariance']['path_to_oc_executable'] = (
        '/home/cosmo/davide.sciotti/data/OneCovariance/covariance.py'
    )
    cfg['OneCovariance']['consistency_checks'] = False
    cfg['OneCovariance']['oc_output_filename'] = 'cov_rcf_mergetest_v2_'

if 'save_output_as_benchmark' not in cfg['misc'] or 'bench_filename' not in cfg['misc']:
    cfg['misc']['save_output_as_benchmark'] = False
    cfg['misc']['bench_filename'] = (
        '../Spaceborne_bench/output_G{g_code:s}_SSC{ssc_code:s}_cNG{cng_code:s}'
        '_KE{use_KE:s}_resp{which_pk_responses:s}_b1g{which_b1g_in_resp:s}'
        '_devmerge3_nmt'
    )


cfg['ell_cuts'] = {}
cfg['ell_cuts']['apply_ell_cuts'] = False  # Type: bool
# Type: str. Cut if the bin *center* or the bin *lower edge* is
# larger than ell_max[zi, zj]
cfg['ell_cuts']['center_or_min'] = 'center'
cfg['ell_cuts']['cl_ell_cuts'] = False  # Type: bool
cfg['ell_cuts']['cov_ell_cuts'] = False  # Type: bool
# Type: float. This is used when ell_cuts is False, also...?
cfg['ell_cuts']['kmax_h_over_Mpc_ref'] = 1.0
cfg['ell_cuts']['kmax_h_over_Mpc_list'] = [
    0.1, 0.16681005, 0.27825594, 0.46415888, 0.77426368, 1.29154967,
    2.15443469, 3.59381366, 5.9948425, 10.0,
]  # fmt: skip

# Sigma2_b settings, common to Spaceborne and PyCCL. Can be one of:
# - full_curved_sky: Use the full- (curved-) sky expression (for Spaceborne only).
#   In this case, the output covmat
# - from_input_mask: input a mask with path specified by mask_path
# - polar_cap_on_the_fly: generate a polar cap during the run, with nside
#   specified by nside
# - null (None): use the flat-sky expression (valid for PyCCL only)
# - flat_sky: use the flat-sky expression (valid for PyCCL only)
#   has to be rescaled by fsky
cfg['covariance']['which_sigma2_b'] = 'from_input_mask'  # Type: str | None
# Integration scheme used for the SSC survey covariance (sigma2_b) computation. Options:
# - 'simps': uses simpson integration. This is faster but less accurate
# - 'levin': uses levin integration. This is slower but more accurate
cfg['covariance']['sigma2_b_int_method'] = 'fft'  # Type: str.
# Whether to load the previously computed sigma2_b.
# No need anymore since it's quite fast
cfg['covariance']['load_cached_sigma2_b'] = False  # Type: bool.

# How many integrals to compute at once for the  numerical integration of
# the sigma^2_b(z_1, z_2) function with pylevin.
# IMPORTANT NOTE: in case of memory issues, (i.e., if you notice the code crashing
# while computing sigma2_b), decrease this or num_threads.
cfg['misc']['levin_batch_size'] = 1000  # Type: int.

# ordering of the different 3x2pt probes in the covariance matrix
cfg['covariance']['probe_ordering'] = [
    ['L', 'L'],
    ['G', 'L'],
    ['G', 'G'],
]  # Type: list[list[str]]

probe_ordering = cfg['covariance']['probe_ordering']  # TODO deprecate this
GL_OR_LG = probe_ordering[1][0] + probe_ordering[1][1]

# This has been deprecated since i am no longer using Levin integration.
# This variable used to control the number of bins over which to compute the Levin
# RS cov (*without* analytical bin averaging, i.e. using J_mu in place of K_mu).
# From then, the covariance was rebinned to cfg['binning']['theta_bins'].
# This works but is not ideal, as the proper bin averaging is more correct.
# Type: int. Number of theta bins used for the fine grid, after which the covariance is rebinned
cfg['precision']['theta_bins_fine'] = cfg['binning']['theta_bins']

# Integration method for the covariance projection to real space. Options:
# - 'simps': uses simpson integration. This is faster but less accurate
# - 'levin': uses levin integration. This is slower but more accurate
cfg['precision']['cov_rs_int_method'] = 'simps'  # Type: str.
# setting this to False makes the code resort to the less accurate bin averaging method
# mentioned above
cfg['precision']['levin_bin_avg'] = True  # Type: bool.
# ! END HARDCODED OPTIONS/PARAMETERS

# convenence settings that have been hardcoded
n_probes = cfg['covariance']['n_probes']
which_sigma2_b = cfg['covariance']['which_sigma2_b']

# ! probe selection

# * small naming guide for the confused developer:
# - unique_probe_combs: the probe combinations which are actually computed, meaning the
#                       elements of the diagonal and, if requested, the cross-terms.
# - symm_probe_combs: the lower triangle, (or an empty list if cross terms are not
#                     required), which are the blocks filled by symmetry
# - nonreq_probe_combs: the blocks which are not required at all, and are set to 0 in
#                       the 10D/6D matrix. This does *not* include the probes in
#                       symm_probe_combs! "Required" means I want that probe combination
#                       in the final covariance matrix, not that I want to explicitly
#                       compute it
# - req_probe_combs_2d: the probe combinations which need to appear in the final 2D
#                       format of the covmat. This includes the cross-probes whether
#                       they are required or not (aka, they need to be present in the 2D
#                       covmat, either as actual values or as zeros)

# example: I request LL and GL, no cross-terms
# unique_probe_combs = ['LLLL', 'GLGL']
# symm_probe_combs = []
# nonreq_probe_combs = {'GGGG', 'GGGL', 'GGLL', 'GLGG', 'GLLL', 'LLGG', 'LLGL'}
# req_probe_combs_2d = ['LLLL', 'LLGL', 'GLLL', 'GLGL']

unique_probe_names_hs = []
if cfg['probe_selection']['LL']:
    unique_probe_names_hs.append('LL')
if cfg['probe_selection']['GL']:
    unique_probe_names_hs.append('GL')
if cfg['probe_selection']['GG']:
    unique_probe_names_hs.append('GG')

unique_probe_names_rs = []
if cfg['probe_selection']['xip']:
    unique_probe_names_rs.append('xip')
if cfg['probe_selection']['xim']:
    unique_probe_names_rs.append('xim')
if cfg['probe_selection']['gt']:
    unique_probe_names_rs.append('gt')
if cfg['probe_selection']['w']:
    unique_probe_names_rs.append('gg')  # TODO CHANGE TO w!

# add cross terms if requested
unique_probe_combs_hs = sl.build_probe_list(
    unique_probe_names_hs, include_cross_terms=cfg['probe_selection']['cross_cov']
)
unique_probe_combs_rs = sl.build_probe_list(
    unique_probe_names_rs, include_cross_terms=cfg['probe_selection']['cross_cov']
)

# probe combinations to be filled by symmetry or to exclude altogether
symm_probe_combs_hs, nonreq_probe_combs_hs = sl.get_probe_combs(
    unique_probe_combs_hs, space='harmonic'
)
symm_probe_combs_rs, nonreq_probe_combs_rs = sl.get_probe_combs(
    unique_probe_combs_rs, space='real'
)

# required probe combinations to include in the 2d arrays (must include the
# cross-terms!)
_req_probe_combs_hs_2d = sl.build_probe_list(
    unique_probe_names_hs, include_cross_terms=True
)
_req_probe_combs_rs_2d = sl.build_probe_list(
    unique_probe_names_rs, include_cross_terms=True
)
# as req_probe_combs_2d still only contains the upper triangle,
# add the symemtric blocks
symm_probe_combs_hs_2d, _ = sl.get_probe_combs(_req_probe_combs_hs_2d, space='harmonic')
symm_probe_combs_rs_2d, _ = sl.get_probe_combs(_req_probe_combs_rs_2d, space='real')
_req_probe_combs_hs_2d += symm_probe_combs_hs_2d
_req_probe_combs_rs_2d += symm_probe_combs_rs_2d

# reorder!
req_probe_combs_hs_2d = []
for probe in const.HS_ALL_PROBE_COMBS:
    if probe in _req_probe_combs_hs_2d:
        req_probe_combs_hs_2d.append(probe)
req_probe_combs_rs_2d = []
for probe in const.RS_ALL_PROBE_COMBS:
    if probe in _req_probe_combs_rs_2d:
        req_probe_combs_rs_2d.append(probe)

unique_probe_combs_ix_hs = [
    [const.HS_PROBE_NAME_TO_IX_DICT[idx] for idx in comb]
    for comb in unique_probe_combs_hs
]
nonreq_probe_combs_ix_hs = [
    [const.HS_PROBE_NAME_TO_IX_DICT[idx] for idx in comb]
    for comb in nonreq_probe_combs_hs
]
req_probe_combs_2d_ix_hs = [
    [const.HS_PROBE_NAME_TO_IX_DICT[idx] for idx in comb]
    for comb in req_probe_combs_hs_2d
]

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

_condition = 'GLGL' in req_probe_combs_hs_2d or 'gtgt' in req_probe_combs_rs_2d
if compute_ccl_cng and _condition:
    raise ValueError(
        'There seems to be some issue with the symmetry of the GLGL '
        'block in the '
        'CCL cNG covariance, so for the moment it is disabled. '
        'The LLLL and GGGG blocks are not affected, so you can still '
        'compute the single-probe cNG covariances.'
    )

# ! set HS probes to compute depending on RS ones
# Set HS probes depending on RS ones
if obs_space != 'real':
    pass  # nothing to do

elif cfg['covariance']['SSC'] or cfg['covariance']['cNG']:
    # Otherwise switch on HS probes corresponding to selected RS probes
    cfg['probe_selection']['LL'] = (
        cfg['probe_selection']['xip'] or cfg['probe_selection']['xim']
    )
    cfg['probe_selection']['GL'] = cfg['probe_selection']['gt']
    cfg['probe_selection']['GG'] = cfg['probe_selection']['w']

else:
    # If neither SSC nor cNG are active → turn off all HS probes
    cfg['probe_selection']['LL'] = False
    cfg['probe_selection']['GL'] = False
    cfg['probe_selection']['GG'] = False


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
# TODO update this periodically
cfg_check_obj = config_checker.SpaceborneConfigChecker(cfg, zbins)
cfg_check_obj.run_all_checks()

# ! instantiate CCL object
ccl_obj = ccl_interface.CCLInterface(
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
z_grid = np.linspace(
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
k_grid_s2b = np.logspace(
    cfg['covariance']['log10_k_min'],
    cfg['covariance']['log10_k_max'],
    k_steps_sigma2_simps
)  # fmt: skip

if len(z_grid) < 1000:
    warnings.warn(
        'the number of steps in the redshift grid is small, '
        'you may want to consider increasing it',
        stacklevel=2,
    )

zgrid_str = (
    f'zmin{cfg["covariance"]["z_min"]}_'
    f'zmax{cfg["covariance"]["z_max"]}_'
    f'zsteps{cfg["covariance"]["z_steps"]}'
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

# private cfg dictionary. This serves a couple different purposeses:
# 1. To store and pass hardcoded parameters in a convenient way
# 2. To make the .format() more compact
pvt_cfg = {
    'zbins': zbins,
    'ind': ind,
    'n_probes': n_probes,
    'probe_ordering': probe_ordering,
    'unique_probe_combs': unique_probe_combs_hs,
    'probe_comb_idxs': unique_probe_combs_ix_hs,
    'req_probe_combs_2d': req_probe_combs_hs_2d,
    'which_ng_cov': cov_terms_str,
    'cov_terms_list': cov_terms_list,
    'GL_OR_LG': GL_OR_LG,
    'symmetrize_output_dict': symmetrize_output_dict,
    'use_h_units': use_h_units,
    'z_grid': z_grid,
}

# instantiate data handler class
io_obj = io_handler.IOHandler(cfg, pvt_cfg)

# ! ====================================================================================
# ! ================================= BEGIN MAIN BODY ==================================
# ! ====================================================================================

# ! ===================================== \ells ========================================
ell_obj = ell_utils.EllBinning(cfg)
ell_obj.build_ell_bins()
# not always required, but in this way it's simpler
ell_obj.compute_ells_3x2pt_unbinned()
ell_obj._validate_bins()

pvt_cfg['nbl_3x2pt'] = ell_obj.nbl_3x2pt
pvt_cfg['ell_min_3x2pt'] = ell_obj.ell_min_3x2pt


# ! ===================================== Mask =========================================
mask_obj = mask_utils.Mask(cfg['mask'])
mask_obj.process()
if hasattr(mask_obj, 'mask'):
    import healpy as hp

    hp.mollview(mask_obj.mask, cmap='inferno_r', title='Mask - Mollweide view')

# add fsky to pvt_cfg
pvt_cfg['fsky'] = mask_obj.fsky


# ! ===================================== n(z) =========================================
# load
io_obj.get_nz_fmt()
io_obj.load_nz()

# assign to variables
zgrid_nz_src = io_obj.zgrid_nz_src
zgrid_nz_lns = io_obj.zgrid_nz_lns
nz_src = io_obj.nz_src
nz_lns = io_obj.nz_lns

# nz may be subjected to a shift: save the original arrays
nz_unshifted_src = nz_src
nz_unshifted_lns = nz_lns

if shift_nz:
    nz_src = wf_cl_lib.shift_nz(
        zgrid_nz_src,
        nz_unshifted_src,
        cfg['nz']['dzWL'],
        normalize=cfg['nz']['normalize_shifted_nz'],
        plot_nz=True,
        interpolation_kind=shift_nz_interpolation_kind,
        bounds_error=False,
        fill_value=0,
        clip_min=cfg['nz']['clip_zmin'],
        clip_max=cfg['nz']['clip_zmax'],
        plt_title='$n_i(z)$ sources shifts ',
    )
    nz_lns = wf_cl_lib.shift_nz(
        zgrid_nz_lns,
        nz_unshifted_lns,
        cfg['nz']['dzGC'],
        normalize=False,
        plot_nz=True,
        interpolation_kind=shift_nz_interpolation_kind,
        bounds_error=False,
        fill_value=0,
        clip_min=cfg['nz']['clip_zmin'],
        clip_max=cfg['nz']['clip_zmax'],
        plt_title='$n_i(z)$ lenses shifts ',
    )

if cfg['nz']['smooth_nz']:
    for zi in range(zbins):
        nz_src[:, zi] = gaussian_filter1d(
            nz_src[:, zi], sigma=cfg['nz']['sigma_smoothing']
        )
        nz_lns[:, zi] = gaussian_filter1d(
            nz_lns[:, zi], sigma=cfg['nz']['sigma_smoothing']
        )

# check if they are normalised, and if not do so
nz_lns_norm = simps(y=nz_lns, x=zgrid_nz_lns, axis=0)
nz_src_norm = simps(y=nz_src, x=zgrid_nz_src, axis=0)

if not np.allclose(nz_lns_norm, 1, atol=0, rtol=1e-3):
    warnings.warn(
        '\nThe lens n(z) are not normalised. Proceeding to normalise them', stacklevel=2
    )
    nz_lns /= nz_lns_norm

if not np.allclose(nz_src_norm, 1, atol=0, rtol=1e-3):
    warnings.warn(
        '\nThe source n(z) are not normalised. Proceeding to normalise them',
        stacklevel=2,
    )
    nz_src /= nz_src_norm


ccl_obj.set_nz(
    nz_full_src=np.hstack((zgrid_nz_src[:, None], nz_src)),
    nz_full_lns=np.hstack((zgrid_nz_lns[:, None], nz_lns)),
)
ccl_obj.check_nz_tuple(zbins)

wf_cl_lib.plot_nz_src_lns(zgrid_nz_src, nz_src, zgrid_nz_lns, nz_lns, colors=clr)


# ! ========================================= IA =======================================
ccl_obj.set_ia_bias_tuple(z_grid_src=z_grid, has_ia=cfg['C_ell']['has_IA'])


# ! =================================== Galaxy bias ====================================
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

# Check if the galaxy bias is the same in all bins
# Note: the [0] (inside square brackets) means "select column 0 but keep the array
# two-dimensional", for shape consistency
single_b_of_z = np.allclose(ccl_obj.gal_bias_2d, ccl_obj.gal_bias_2d[:, [0]])


# ! ============================ Magnification bias ====================================
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
    plt.plot(z_grid, ccl_obj.gal_bias_2d[:, zi], label=f'$z_{{{zi}}}$', c=clr[zi])
plt.xlabel(r'$z$')
plt.ylabel(r'$b_{g, i}(z)$')
plt.legend()


# ! ============================ Radial kernels ========================================
ccl_obj.set_kernel_obj(cfg['C_ell']['has_rsd'], cfg['PyCCL']['n_samples_wf'])
ccl_obj.set_kernel_arr(
    z_grid_wf=z_grid, has_magnification_bias=cfg['C_ell']['has_magnification_bias']
)

gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias)'
ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_wo_gal_bias_arr


# ! ================================= BNT and z means ==================================
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


# assert np.all(np.diff(z_means_ll) > 0), 'z_means_ll should be monotonically
# increasing'
# assert np.all(np.diff(z_means_gg) > 0), 'z_means_gg should be monotonically
# increasing'
# assert np.all(np.diff(z_means_ll_bnt) > 0), (
#     'z_means_ll_bnt should be monotonically increasing '
#     '(not a strict condition, valid only if we do not shift the n(z) in this part)'
# )


# ! ===================================== \ell cuts ====================================
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
    'IA',
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


# ! ======================================== Cls =======================================
# ! note that the function below includes the multiplicative shear bias
print('Computing Cls...')
t0 = time.perf_counter()
_cl_3x2pt_5d = ccl_interface.compute_cl_3x2pt_5d(
    ccl_obj,
    ells=ell_obj.ells_3x2pt,
    zbins=zbins,
    mult_shear_bias=np.array(cfg['C_ell']['mult_shear_bias']),
    cl_ccl_kwargs=cl_ccl_kwargs,
    n_probes_hs=cfg['covariance']['n_probes'],
)

ccl_obj.cl_ll_3d = _cl_3x2pt_5d[0, 0]
ccl_obj.cl_gl_3d = _cl_3x2pt_5d[1, 0]
ccl_obj.cl_gg_3d = _cl_3x2pt_5d[1, 1]
print(f'done in {time.perf_counter() - t0:.2f} s')


if cfg['C_ell']['use_input_cls']:
    # TODO NMT here you should ask the user for unbinned cls

    # load input cls
    io_obj.get_cl_fmt()
    io_obj.load_cls()

    # check ells before spline interpolation
    io_obj.check_ells_in(ell_obj)

    print(f'Using input Cls for LL from file\n{cfg["C_ell"]["cl_LL_path"]}')
    print(f'Using input Cls for GGL from file\n{cfg["C_ell"]["cl_GL_path"]}')
    print(f'Using input Cls for GG from file\n{cfg["C_ell"]["cl_GG_path"]}')

    ells_WL_in, cl_ll_3d_in = io_obj.ells_WL_in, io_obj.cl_ll_3d_in
    ells_XC_in, cl_gl_3d_in = io_obj.ells_XC_in, io_obj.cl_gl_3d_in
    ells_GC_in, cl_gg_3d_in = io_obj.ells_GC_in, io_obj.cl_gg_3d_in

    # interpolate input Cls on the desired ell grid
    cl_ll_3d_spline = CubicSpline(ells_WL_in, cl_ll_3d_in, axis=0)
    cl_gl_3d_spline = CubicSpline(ells_XC_in, cl_gl_3d_in, axis=0)
    cl_gg_3d_spline = CubicSpline(ells_GC_in, cl_gg_3d_in, axis=0)
    cl_ll_3d_in = cl_ll_3d_spline(ell_obj.ells_3x2pt)
    cl_gl_3d_in = cl_gl_3d_spline(ell_obj.ells_3x2pt)
    cl_gg_3d_in = cl_gg_3d_spline(ell_obj.ells_3x2pt)

    # save the sb cls for the plot comparing sb and input cls
    cl_ll_3d_sb = ccl_obj.cl_ll_3d
    cl_gl_3d_sb = ccl_obj.cl_gl_3d
    cl_gg_3d_sb = ccl_obj.cl_gg_3d

    # assign them to ccl_obj; m-bias is applied right below
    ccl_obj.cl_ll_3d = cl_ll_3d_in
    ccl_obj.cl_gl_3d = cl_gl_3d_in
    ccl_obj.cl_gg_3d = cl_gg_3d_in

# I am creating copies here, not just a view (so modifying ccl_obj.cl_3x2pt_5d will
# not affect ccl_obj.cl_ll_3d, ccl_obj.cl_gl_3d, ccl_obj.cl_gg_3d and vice versa)
ccl_obj.cl_3x2pt_5d = np.zeros((n_probes, n_probes, ell_obj.nbl_3x2pt, zbins, zbins))
ccl_obj.cl_3x2pt_5d[0, 0] = ccl_obj.cl_ll_3d
ccl_obj.cl_3x2pt_5d[1, 0] = ccl_obj.cl_gl_3d
ccl_obj.cl_3x2pt_5d[0, 1] = ccl_obj.cl_gl_3d.transpose(0, 2, 1)
ccl_obj.cl_3x2pt_5d[1, 1] = ccl_obj.cl_gg_3d

# cls plots
plot_cls()

# this is a lil' bit convoluted: the cls used by the code (wither from input or from sb)
# are stored in ccl_obj.cl_xx_3d. The cl_xx_3d_sb are only computed if 'use_input_cls'
# is True and are only plotted in that case
_key = 'input' if cfg['C_ell']['use_input_cls'] else 'SB'
_ell_dict_wl = {_key: ell_obj.ells_3x2pt}
_ell_dict_xc = {_key: ell_obj.ells_3x2pt}
_ell_dict_gc = {_key: ell_obj.ells_3x2pt}
_cl_dict_wl = {_key: ccl_obj.cl_3x2pt_5d[0, 0]}
_cl_dict_xc = {_key: ccl_obj.cl_3x2pt_5d[1, 0]}
_cl_dict_gc = {_key: ccl_obj.cl_3x2pt_5d[1, 1]}
if cfg['C_ell']['use_input_cls']:
    _ell_dict_wl['SB'] = ell_obj.ells_3x2pt
    _ell_dict_xc['SB'] = ell_obj.ells_3x2pt
    _ell_dict_gc['SB'] = ell_obj.ells_3x2pt
    _cl_dict_wl['SB'] = cl_ll_3d_sb
    _cl_dict_xc['SB'] = cl_gl_3d_sb
    _cl_dict_gc['SB'] = cl_gg_3d_sb

if cfg['misc']['cl_triangle_plot']:
    sb_plt.cls_triangle_plot(
        _ell_dict_wl, _cl_dict_wl, is_auto=True, zbins=zbins, suptitle='WL'
    )
    sb_plt.cls_triangle_plot(
        _ell_dict_xc, _cl_dict_xc, is_auto=False, zbins=zbins, suptitle='GGL'
    )
    sb_plt.cls_triangle_plot(
        _ell_dict_gc, _cl_dict_gc, is_auto=True, zbins=zbins, suptitle='GCph'
    )


# ! BNT transform the cls (and responses?) - it's more complex since I also have to
# ! transform the noise spectra, better to transform directly the covariance matrix
if cfg['BNT']['cl_BNT_transform']:
    print('BNT-transforming the Cls...')
    assert cfg['BNT']['cov_BNT_transform'] is False, (
        'the BNT transform should be applied either to the Cls or to the covariance, '
        'not both'
    )
    from spaceborne import bnt

    cl_ll_3d = bnt.cl_bnt_transform(ccl_obj.cl_ll_3d, bnt_matrix, 'L', 'L')
    cl_3x2pt_5d = bnt.cl_bnt_transform_3x2pt(ccl_obj.cl_3x2pt_5d, bnt_matrix)
    warnings.warn('you should probably BNT-transform the responses too!', stacklevel=2)
    if compute_oc_g or compute_oc_ssc or compute_oc_cng:
        raise NotImplementedError('You should cut also the OC Cls')


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


# cov_rs_2d_full = np.hstack((cov_obj.cov_rs_obj.cov_rs_2d_dict[xipxip, xipxim, xipxip]))

# for _, cov in cov_obj.cov_rs_obj.cov_rs_full_2d:
#     print(_)
#     sl.plot_correlation_matrix(corr_dav)
#     plt.title(_)


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

# ! =========================== Unbinned Cls for nmt/sample cov ========================
if cfg['namaster']['use_namaster'] or cfg['sample_covariance']['compute_sample_cov']:
    from spaceborne import cov_partial_sky

    # check that the input cls are computed over a fine enough grid
    if cfg['C_ell']['use_input_cls']:
        for ells_in, ells_out in zip(
            [ells_WL_in, ells_XC_in, ells_GC_in],
            [ell_obj.ells_3x2pt_unb, ell_obj.ells_3x2pt_unb, ell_obj.ells_3x2pt_unb],
        ):
            io_obj.check_ells_in(ells_in, ells_out)

    # initialize nmt_cov_obj and set a couple useful attributes
    nmt_cov_obj = cov_partial_sky.NmtCov(
        cfg=cfg, pvt_cfg=pvt_cfg, ell_obj=ell_obj, mask_obj=mask_obj
    )

    # set unbinned ells in nmt_cov_obj
    nmt_cov_obj.ells_3x2pt_unb = ell_obj.ells_3x2pt_unb
    nmt_cov_obj.nbl_3x2pt_unb = ell_obj.nbl_3x2pt_unb

    if cfg['C_ell']['use_input_cls']:
        cl_3x2pt_5d_unb = np.zeros(
            (n_probes, n_probes, ell_obj.nbl_3x2pt_unb, zbins, zbins)
        )
        cl_3x2pt_5d_unb[0, 0] = cl_ll_3d_spline(ell_obj.ells_3x2pt_unb)
        cl_3x2pt_5d_unb[1, 0] = cl_gl_3d_spline(ell_obj.ells_3x2pt_unb)
        cl_3x2pt_5d_unb[0, 1] = cl_3x2pt_5d_unb[1, 0].transpose(0, 2, 1)
        cl_3x2pt_5d_unb[1, 1] = cl_gg_3d_spline(ell_obj.ells_3x2pt_unb)

    else:
        cl_3x2pt_5d_unb = ccl_interface.compute_cl_3x2pt_5d(
            ccl_obj,
            ells=ell_obj.ells_3x2pt_unb,
            zbins=zbins,
            mult_shear_bias=np.array(cfg['C_ell']['mult_shear_bias']),
            cl_ccl_kwargs=cl_ccl_kwargs,
            n_probes_hs=cfg['covariance']['n_probes'],
        )

    nmt_cov_obj.cl_ll_unb_3d = cl_3x2pt_5d_unb[0, 0]
    nmt_cov_obj.cl_gl_unb_3d = cl_3x2pt_5d_unb[1, 0]
    nmt_cov_obj.cl_gg_unb_3d = cl_3x2pt_5d_unb[1, 1]

else:
    nmt_cov_obj = None


# ! =============== Init real space cov object, put here for simplicity for the moment ==============
if obs_space == 'real':
    from spaceborne import cov_real_space

    # initialize cov_rs_obj and set a couple useful attributes
    cov_rs_obj = cov_real_space.CovRealSpace(cfg, pvt_cfg, mask_obj)
    cov_rs_obj.set_cov_2d_ordering(req_probe_combs_2d=req_probe_combs_rs_2d)
    cov_rs_obj.set_ind_and_zpairs(ind, zbins)
    ell_obj.compute_ells_3x2pt_rs()
    cov_rs_obj.ells = ell_obj.ells_3x2pt_rs
    cov_rs_obj.nbl = len(ell_obj.ells_3x2pt_rs)

    # set 3x2pt cls: recompute cls on the finer ell grid...
    if cfg['C_ell']['use_input_cls']:
        cl_ll_3d_for_rs = cl_ll_3d_spline(cov_rs_obj.ells)
        cl_gl_3d_for_rs = cl_gl_3d_spline(cov_rs_obj.ells)
        cl_gg_3d_for_rs = cl_gg_3d_spline(cov_rs_obj.ells)

    else:
        cl_3x2pt_5d_for_rs = ccl_interface.compute_cl_3x2pt_5d(
            ccl_obj,
            ells=cov_rs_obj.ells,
            zbins=zbins,
            mult_shear_bias=np.array(cfg['C_ell']['mult_shear_bias']),
            cl_ccl_kwargs=cl_ccl_kwargs,
            n_probes_hs=cfg['covariance']['n_probes'],
        )
    # ...and store them in the cov_rs object
    cov_rs_obj.cl_3x2pt_5d = cl_3x2pt_5d_for_rs


# !  =============================== Build Gaussian covs ===============================
cov_hs_obj = cov_harmonic_space.SpaceborneCovariance(
    cfg, pvt_cfg, ell_obj, nmt_cov_obj, bnt_matrix
)
cov_hs_obj.set_ind_and_zpairs(ind)
cov_hs_obj.consistency_checks()
cov_hs_obj.set_gauss_cov(
    ccl_obj=ccl_obj,
    split_gaussian_cov=cfg['covariance']['split_gaussian_cov'],
    nonreq_probe_combs_ix=nonreq_probe_combs_ix_hs,
)

# ! =================================== OneCovariance ================================
if compute_oc_g or compute_oc_ssc or compute_oc_cng:
    if cfg['ell_cuts']['cl_ell_cuts']:
        raise NotImplementedError(
            'TODO double check inputs in this case. This case is untested'
        )

    start_time = time.perf_counter()

    # * 1. save ingredients in ascii format
    # TODO this should me moved to io_handler.py
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
        ell_obj.ell_min_3x2pt, ell_obj.ell_max_3x2pt, nbl_3x2pt_oc
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
    print('Start cov computation with OneCovariance...')
    # initialize object, build cfg file
    oc_obj = oc_interface.OneCovarianceInterface(
        cfg, pvt_cfg, do_g=compute_oc_g, do_ssc=compute_oc_ssc, do_cng=compute_oc_cng
    )
    oc_obj.oc_path = oc_path
    oc_obj.z_grid_trisp_sb = z_grid_trisp
    oc_obj.path_to_config_oc_ini = f'{oc_obj.oc_path}/input_configs.ini'
    oc_obj.ells_sb = ell_obj.ells_3x2pt
    oc_obj.build_save_oc_ini(ascii_filenames_dict, h, print_ini=True)

    # compute covs
    oc_obj.call_oc_from_bash()

    # load output .list file (maybe the .mat format would be better, actually...)
    # and store it into a 6d dictionary
    oc_output_covlist_fname = (
        f'{oc_path}/{cfg["OneCovariance"]["oc_output_filename"]}_list.dat'
    )
    oc_obj.cov_dict_6d = oc_interface.process_cov_from_list_file(
        oc_output_covlist_fname=oc_output_covlist_fname,
        zbins=zbins,
        df_chunk_size=5_000_000,
    )

    # some useful vars to make the cov processing work regardless of the space
    ps = cfg['probe_selection']
    full_cov = False  # True if all probes + the cross-covariance are required
    if obs_space == 'harmonic':
        _valid_probes_oc = const.HS_DIAG_PROBES_OC
        _req_probe_combs_2d = req_probe_combs_hs_2d
        nbx = ell_obj.nbl_3x2pt
        probe_idx_dict = oc_obj.probe_idx_dict_hs
        n_probes_oc = 2
        full_cov = (ps['LL'] + ps['GL'] + ps['GG']) == 3 and ps['cross_cov'] is True
    elif obs_space == 'real':
        _valid_probes_oc = const.RS_DIAG_PROBES_OC
        _req_probe_combs_2d = req_probe_combs_rs_2d
        nbx = cov_rs_obj.nbt_coarse
        probe_idx_dict = cov_rs_obj.probe_idx_dict_short_oc
        n_probes_oc = 4
        full_cov = (ps['xip'] + ps['xim'] + ps['gt'] + ps['w']) == 4 and ps[
            'cross_cov'
        ] is True

    # fill the missing probe combinations (ab, cd -> cd, ab) by symmetry
    oc_obj.cov_dict_6d = oc_interface.symmetrize_probes_dict_6d(
        cov_dict_6d=oc_obj.cov_dict_6d, space=obs_space, valid_probes=_valid_probes_oc
    )

    # turn to 10d arrays, which are still used in the SpaceborneCovariance class
    cov_tot = np.zeros(
        (
            n_probes_oc,
            n_probes_oc,
            n_probes_oc,
            n_probes_oc,
            nbx,
            nbx,
            zbins,
            zbins,
            zbins,
            zbins,
        )
    )
    for term in ['sva', 'sn', 'mix', 'g', 'ssc', 'cng']:
        cov = oc_interface.oc_cov_dict_6d_to_array_10d(
            cov_dict_6d=oc_obj.cov_dict_6d,
            desired_term=term,
            n_probes=n_probes_oc,
            nbx=nbx,
            zbins=zbins,
            probe_idx_dict=probe_idx_dict,
        )

        # finally, store the arrays in the corresponding attributes
        setattr(oc_obj, f'cov_3x2pt_{term}_10d', cov)
        cov_tot += cov

    # set also total covariance
    oc_obj.cov_3x2pt_tot_10d = cov_tot
    # free memory
    del cov_tot
    gc.collect()

    # compare list and mat formats
    if full_cov:
        oc_obj.output_sanity_check(req_probe_combs_2d=_req_probe_combs_2d, rtol=1e-4)

    # This is an alternative method to call OC (more convoluted but more maintanable).
    # I keep the code for optional consistency checks
    if cfg['OneCovariance']['consistency_checks']:
        # store in temp variables for later check
        check_cov_sva_oc_3x2pt_10D = oc_obj.cov_3x2pt_sva_10d
        check_cov_mix_oc_3x2pt_10D = oc_obj.cov_3x2pt_mix_10d
        check_cov_sn_oc_3x2pt_10D = oc_obj.cov_3x2pt_sn_10d
        check_cov_ssc_oc_3x2pt_10D = oc_obj.cov_3x2pt_ssc_10d
        check_cov_cng_oc_3x2pt_10D = oc_obj.cov_3x2pt_cng_10d

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

if compute_sb_ssc:
    # ! ================================= Probe responses ==============================
    resp_obj = responses.SpaceborneResponses(
        cfg=cfg, k_grid=k_grid, z_grid=z_grid_trisp, ccl_obj=ccl_obj
    )
    resp_obj.use_h_units = use_h_units

    if cfg['covariance']['which_pk_responses'] == 'halo_model':
        # convenience variables
        which_b1g_in_resp = cfg['covariance']['which_b1g_in_resp']
        include_terasawa_terms = cfg['covariance']['include_terasawa_terms']

        # recompute galaxy bias on the z grid used to compute the responses/trispectrum
        gal_bias_2d_trisp = ccl_obj.gal_bias_func(z_grid_trisp)
        if gal_bias_2d_trisp.ndim == 1:
            assert single_b_of_z, (
                'Galaxy bias should be a single function of redshift for all bins, '
                'there seems to be some inconsistency'
            )
            gal_bias_2d_trisp = np.tile(gal_bias_2d_trisp[:, None], zbins)

        dPmm_ddeltab = np.zeros((len(k_grid), len(z_grid_trisp), zbins, zbins))
        dPgm_ddeltab = np.zeros((len(k_grid), len(z_grid_trisp), zbins, zbins))
        dPgg_ddeltab = np.zeros((len(k_grid), len(z_grid_trisp), zbins, zbins))
        # TODO this can be made more efficient - eg by having a
        # TODO "if_bias_equal_all_bins" flag

        if single_b_of_z:
            # compute dPAB/ddelta_b
            resp_obj.set_hm_resp(
                k_grid=k_grid,
                z_grid=z_grid_trisp,
                which_b1g=which_b1g_in_resp,
                b1g_zi=gal_bias_2d_trisp[:, 0],
                b1g_zj=gal_bias_2d_trisp[:, 0],
                include_terasawa_terms=include_terasawa_terms,
            )

            # reshape appropriately
            _dPmm_ddeltab_hm = resp_obj.dPmm_ddeltab_hm[:, :, None, None]
            _dPgm_ddeltab_hm = resp_obj.dPgm_ddeltab_hm[:, :, None, None]
            _dPgg_ddeltab_hm = resp_obj.dPgg_ddeltab_hm[:, :, None, None]

            dPmm_ddeltab = np.repeat(_dPmm_ddeltab_hm, zbins, axis=2)
            dPmm_ddeltab = np.repeat(dPmm_ddeltab, zbins, axis=3)
            dPgm_ddeltab = np.repeat(_dPgm_ddeltab_hm, zbins, axis=2)
            dPgm_ddeltab = np.repeat(dPgm_ddeltab, zbins, axis=3)
            dPgg_ddeltab = np.repeat(_dPgg_ddeltab_hm, zbins, axis=2)
            dPgg_ddeltab = np.repeat(dPgg_ddeltab, zbins, axis=3)

            # # TODO check these
            # r_mm = resp_obj.r1_mm_hm
            # r_gm = resp_obj.r1_gm_hm
            # r_gg = resp_obj.r1_gg_hm

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
                    # # TODO check these
                    # r_mm = resp_obj.r1_mm_hm
                    # r_gm = resp_obj.r1_gm_hm
                    # r_gg = resp_obj.r1_gg_hm

        # for mm and gm there are redundant axes: reduce dimensionality (squeeze)
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

    # ! prepare integrands (d2CAB_dVddeltab) and volume element
    # ! - test k_max_limber vs k_max_dPk and adjust z_min accordingly
    k_max_resp = np.max(k_grid)
    ell_grid = ell_obj.ells_GC
    kmax_limber = cosmo_lib.get_kmax_limber(
        ell_grid, z_grid, use_h_units, ccl_obj.cosmo_ccl
    )

    z_grid_test = z_grid.copy()
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

    from spaceborne import cov_ssc

    ssc_obj = cov_ssc.SpaceborneSSC(cfg, ccl_obj, z_grid, ind_dict, zbins, use_h_units)
    ssc_obj.set_sigma2_b(ccl_obj, mask_obj, k_grid_s2b, which_sigma2_b)

    cov_ssc_3x2pt_dict_8D = ssc_obj.compute_ssc(
        d2CLL_dVddeltab_4d=d2CLL_dVddeltab,
        d2CGL_dVddeltab_4d=d2CGL_dVddeltab,
        d2CGG_dVddeltab_4d=d2CGG_dVddeltab,
        unique_probe_combs_hs=unique_probe_combs_hs,
        symm_probe_combs_hs=symm_probe_combs_hs,
        nonreq_probe_combs_hs=nonreq_probe_combs_hs,
    )

    # in the full_curved_sky case only, sigma2_b has to be divided by fsky
    # TODO it would make much more sense to divide s2b directly...
    if which_sigma2_b == 'full_curved_sky':
        for key in cov_ssc_3x2pt_dict_8D:
            cov_ssc_3x2pt_dict_8D[key] /= mask_obj.fsky
    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask', 'flat_sky']:
        pass
    else:
        raise ValueError(f'which_sigma2_b = {which_sigma2_b} not recognized')

    cov_hs_obj.cov_ssc_sb_3x2pt_dict_8D = cov_ssc_3x2pt_dict_8D

# ! ========================================== PyCCL ===================================
if compute_ccl_ssc:
    # Note: this z grid has to be larger than the one requested in the trispectrum
    # (z_grid_tkka in the cfg file). You can probaby use the same grid as the
    # one used in the trispectrum, but from my tests is should be
    # zmin_s2b < zmin_s2b_tkka and zmax_s2b =< zmax_s2b_tkka.
    # if zmin=0 it looks like I can have zmin_s2b = zmin_s2b_tkka
    ccl_obj.set_sigma2_b(
        z_grid=z_default_grid_ccl,  # TODO can I not just pass z_grid here?
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
        ccl_obj.initialize_trispectrum(
            which_ng_cov, unique_probe_combs_hs, cfg['PyCCL']
        )
        ccl_obj.compute_ng_cov_3x2pt(
            which_ng_cov,
            ell_obj.ells_GC,
            mask_obj.fsky,
            integration_method=cfg['PyCCL']['cov_integration_method'],
            unique_probe_combs=unique_probe_combs_hs,
            ind_dict=ind_dict,
        )

# ! ========================== Combine covariance terms ================================
cov_hs_obj.build_covs(
    ccl_obj=ccl_obj,
    oc_obj=oc_obj,
    split_gaussian_cov=cfg['covariance']['split_gaussian_cov'],
)


if obs_space == 'real':
    print('Computing RS covariance...')
    start_rs = time.perf_counter()

    # TODO understand a bit better how to treat real-space SSC and cNG
    for _probe, _term in itertools.product(
        unique_probe_combs_rs, cov_rs_obj.terms_toloop
    ):
        print(f'\n***** working on probe {_probe} - term {_term} *****')
        cov_rs_obj.compute_realspace_cov(
            cov_hs_obj=cov_hs_obj, probe=_probe, term=_term
        )

    for term in cov_rs_obj.terms_toloop:
        cov_rs_obj.fill_remaining_probe_blocks(
            term, symm_probe_combs_rs, nonreq_probe_combs_rs
        )

    # put everything together
    cov_rs_obj.combine_terms_and_probes(
        unique_probe_combs=unique_probe_combs_rs,
        req_probe_combs_2d=req_probe_combs_rs_2d,
    )

    print(f'...done in {time.perf_counter() - start_rs:.2f} s')


if obs_space == 'harmonic':
    _cov_obj = cov_hs_obj
    _probes = unique_probe_combs_hs
elif obs_space == 'real':
    _cov_obj = cov_rs_obj
    _probes = unique_probe_combs_rs
else:
    raise ValueError(
        f'Unknown cfg["probe_selection"]["space"]: {cfg["probe_selection"]["space"]}'
    )

# ! save 2D covs (for each term) in npz archive
cov_dict_tosave_2d = {}
if cfg['covariance']['G']:
    cov_dict_tosave_2d['Gauss'] = _cov_obj.cov_3x2pt_g_2d
if cfg['covariance']['SSC']:
    cov_dict_tosave_2d['SSC'] = _cov_obj.cov_3x2pt_ssc_2d
if cfg['covariance']['cNG']:
    cov_dict_tosave_2d['cNG'] = _cov_obj.cov_3x2pt_cng_2d
if cfg['covariance']['split_gaussian_cov']:
    cov_dict_tosave_2d['SVA'] = _cov_obj.cov_3x2pt_sva_2d
    cov_dict_tosave_2d['SN'] = _cov_obj.cov_3x2pt_sn_2d
    cov_dict_tosave_2d['MIX'] = _cov_obj.cov_3x2pt_mix_2d
# the total covariance is equal to the Gaussian one if neither SSC nor cNG are computed,
# so only save it if at least one of the two is computed
if cfg['covariance']['cNG'] or cfg['covariance']['SSC']:
    cov_dict_tosave_2d['TOT'] = _cov_obj.cov_3x2pt_tot_2d

cov_filename = cfg['covariance']['cov_filename']
np.savez_compressed(f'{output_path}/{cov_filename}_2D.npz', **cov_dict_tosave_2d)

# ! save 6D covs (for each probe and term) in npz archive.
# ! note that the 6D covs are always probe-specific,
# ! i.e. there is no cov_3x2pt_{term}_6d
if cfg['covariance']['save_full_cov']:
    cov_dict_tosave_6d = {}

    for _probe in _probes:
        if obs_space == 'harmonic':
            probe_a, probe_b, probe_c, probe_d = tuple(_probe)
            probe_ixs = (
                const.HS_PROBE_NAME_TO_IX_DICT[probe_a],
                const.HS_PROBE_NAME_TO_IX_DICT[probe_b],
                const.HS_PROBE_NAME_TO_IX_DICT[probe_c],
                const.HS_PROBE_NAME_TO_IX_DICT[probe_d],
            )
            if cfg['covariance']['G']:
                cov_dict_tosave_6d[f'{_probe}_Gauss'] = _cov_obj.cov_3x2pt_g_10d[
                    *probe_ixs, ...
                ]
            if cfg['covariance']['SSC']:
                cov_dict_tosave_6d[f'{_probe}_SSC'] = _cov_obj.cov_3x2pt_ssc_10d[
                    *probe_ixs, ...
                ]
            if cfg['covariance']['cNG']:
                cov_dict_tosave_6d[f'{_probe}_cNG'] = _cov_obj.cov_3x2pt_cng_10d[
                    *probe_ixs, ...
                ]
            if cfg['covariance']['split_gaussian_cov']:
                cov_dict_tosave_6d[f'{_probe}_SVA'] = _cov_obj.cov_3x2pt_sva_10d[
                    *probe_ixs, ...
                ]
                cov_dict_tosave_6d[f'{_probe}_SN'] = _cov_obj.cov_3x2pt_sn_10d[
                    *probe_ixs, ...
                ]
                cov_dict_tosave_6d[f'{_probe}_MIX'] = _cov_obj.cov_3x2pt_mix_10d[
                    *probe_ixs, ...
                ]
            if cfg['covariance']['cNG'] or cfg['covariance']['SSC']:
                cov_dict_tosave_6d[f'{_probe}_TOT'] = _cov_obj.cov_3x2pt_tot_10d[
                    *probe_ixs, ...
                ]

        # This case is a bit different, no cov_3x2pt_10d is ever created, but I have
        # individual attributes for the 6d covs for each probe
        elif obs_space == 'real':
            if cfg['covariance']['G']:
                cov_dict_tosave_6d[f'{_probe}_Gauss'] = getattr(
                    _cov_obj, f'cov_{_probe}_g_6d'
                )
            if cfg['covariance']['SSC']:
                cov_dict_tosave_6d[f'{_probe}_SSC'] = getattr(
                    _cov_obj, f'cov_{_probe}_ssc_6d'
                )
            if cfg['covariance']['cNG']:
                cov_dict_tosave_6d[f'{_probe}_cNG'] = getattr(
                    _cov_obj, f'cov_{_probe}_cng_6d'
                )
            if cfg['covariance']['split_gaussian_cov']:
                cov_dict_tosave_6d[f'{_probe}_SVA'] = getattr(
                    _cov_obj, f'cov_{_probe}_sva_6d'
                )
                cov_dict_tosave_6d[f'{_probe}_SN'] = getattr(
                    _cov_obj, f'cov_{_probe}_sn_6d'
                )
                cov_dict_tosave_6d[f'{_probe}_MIX'] = getattr(
                    _cov_obj, f'cov_{_probe}_mix_6d'
                )
            if cfg['covariance']['cNG'] or cfg['covariance']['SSC']:
                cov_dict_tosave_6d[f'{_probe}_TOT'] = getattr(
                    _cov_obj, f'cov_{_probe}_tot_6d'
                )

    np.savez_compressed(f'{output_path}/{cov_filename}_6D.npz', **cov_dict_tosave_6d)

print(f'Covariance matrices saved in {output_path}\n')

# ! ============================ plot & tests ==========================================

with np.errstate(invalid='ignore', divide='ignore'):
    for cov_name, cov in cov_dict_tosave_2d.items():
        if not np.allclose(cov, 0, atol=0, rtol=1e-6):
            fig, ax = plt.subplots(1, 2, figsize=(10, 6))
            ax[0].matshow(np.log10(cov))
            ax[1].matshow(sl.cov2corr(cov), vmin=-1, vmax=1, cmap='RdBu_r')

            # ! add lines and labels for the different selected probes
            if (
                cfg['covariance']['covariance_ordering_2D'].startswith('probe')
                and cfg['misc']['plot_probe_names']
            ):
                if obs_space == 'harmonic':
                    unique_probe_combs = unique_probe_combs_hs
                    diag_probe_combs = const.HS_DIAG_PROBE_COMBS
                    latex_labels = const.HS_PROBE_NAME_TO_LATEX
                    scale_bins = ell_obj.nbl_3x2pt
                elif obs_space == 'real':
                    unique_probe_combs = unique_probe_combs_rs
                    diag_probe_combs = const.RS_DIAG_PROBE_COMBS
                    latex_labels = const.RS_PROBE_NAME_TO_LATEX
                    scale_bins = cov_rs_obj.nbt_coarse

                # this is to get the names and order of the *required* probes
                # along the diagonel
                req_diag_probes = list(set(unique_probe_combs) & set(diag_probe_combs))
                req_diag_probes = [p for p in diag_probe_combs if p in req_diag_probes]

                # set the boundaries
                elem_auto = zpairs_auto * scale_bins
                elem_cross = zpairs_cross * scale_bins

                lim_dict = {
                    'LL': elem_auto,
                    'GL': elem_cross,
                    'GG': elem_auto,
                    'xip': elem_auto,
                    'xim': elem_auto,
                    'gt': elem_cross,
                    'gg': elem_auto,
                }

                # draw the boundaries
                start_ab, start_cd = 0, 0
                for probe_abcd in req_diag_probes[:-1]:
                    if obs_space == 'harmonic':
                        probe_ab, probe_cd = probe_abcd[:2], probe_abcd[2:]
                    if obs_space == 'real':
                        probe_ab, probe_cd = sl.split_probe_name(probe_abcd)

                    kw = {'color': 'k', 'alpha': 0.7, 'ls': '--'}
                    ax[0].axvline(start_ab + lim_dict[probe_ab], **kw)
                    ax[0].axhline(start_ab + lim_dict[probe_ab], **kw)
                    ax[0].axvline(start_cd + lim_dict[probe_cd], **kw)
                    ax[0].axhline(start_cd + lim_dict[probe_cd], **kw)

                    ax[1].axvline(start_ab + lim_dict[probe_ab], **kw)
                    ax[1].axhline(start_ab + lim_dict[probe_ab], **kw)
                    ax[1].axvline(start_cd + lim_dict[probe_cd], **kw)
                    ax[1].axhline(start_cd + lim_dict[probe_cd], **kw)

                    start_ab += lim_dict[probe_ab]
                    start_cd += lim_dict[probe_cd]

                xticks, xlabels = [], []
                yticks, ylabels = [], []

                start_ab, start_cd = 0, 0
                for probe_abcd in req_diag_probes:
                    if obs_space == 'harmonic':
                        probe_ab, probe_cd = probe_abcd[:2], probe_abcd[2:]
                    if obs_space == 'real':
                        probe_ab, probe_cd = sl.split_probe_name(probe_abcd)

                    # x direction
                    center_ab = start_ab + lim_dict[probe_ab] / 2
                    xticks.append(center_ab)
                    xlabels.append(latex_labels[probe_ab])

                    # y direction
                    center_cd = start_cd + lim_dict[probe_cd] / 2
                    yticks.append(center_cd)
                    ylabels.append(latex_labels[probe_cd])

                    start_ab += lim_dict[probe_ab]
                    start_cd += lim_dict[probe_cd]

                for _a in ax:  # apply to both panels
                    _a.set_xticks(xticks)
                    _a.set_xticklabels(xlabels)
                    _a.set_yticks(yticks)
                    _a.set_yticklabels(ylabels)

            plt.colorbar(ax[0].images[0], ax=ax[0], shrink=0.8)
            plt.colorbar(ax[1].images[0], ax=ax[1], shrink=0.8)
            ax[0].set_title('log10 cov')
            ax[1].set_title('corr')
            fig.suptitle(f'cov {cov_name}', y=0.9)


# save cfg file
with open(f'{output_path}/run_config.yaml', 'w') as yaml_file:
    yaml.dump(cfg, yaml_file, default_flow_style=False)

# save nz
nz_header = (
    'This is the actual redshift distribution used internally in the\n'
    'code (albeit not necessarily on this z grid).\n'
    'Please beware that, depending on the settings in the config file,\n'
    'it might have been shifted/smoothed/interpolated/normalized with respect\n'
    'to the raw input one. Also, if you use it directly as input for a subsequent\n'
    'run, make sure to turn off the relevant flags in the config file (e.g. to avoid\n'
    'shifting it twice).\n\n'
)
col_width = 24
labels = ['z'] + [f'n_{i + 1}(z)' for i in range(zbins)]
additional_str = ''.join(label.ljust(col_width) for label in labels)
# additional_str = 'z\t' + '\t'.join([f'n_{zi+1}(z)' for zi in range(zbins)])
nz_header += f'{additional_str}'

np.savetxt(
    f'{output_path}/nz_pos.txt',
    np.column_stack((zgrid_nz_lns, nz_lns)),
    header=nz_header,
    fmt='%.18e',
)
np.savetxt(
    f'{output_path}/nz_shear.txt',
    np.column_stack((zgrid_nz_src, nz_src)),
    header=nz_header,
    fmt='%.18e',
)

# save cls
sl.write_cl_tab(output_path, 'cl_ll', ccl_obj.cl_ll_3d, ell_obj.ells_WL, zbins)
sl.write_cl_tab(output_path, 'cl_gl', ccl_obj.cl_gl_3d, ell_obj.ells_XC, zbins)
sl.write_cl_tab(output_path, 'cl_gg', ccl_obj.cl_gg_3d, ell_obj.ells_GC, zbins)

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
    # better to work with empty arrays than None

    if not compute_sb_ssc:
        k_grid_s2b = np.array([])
        sigma2_b = np.array([])
        dPmm_ddeltab = np.array([])
        dPgm_ddeltab = np.array([])
        dPgg_ddeltab = np.array([])
        d2CLL_dVddeltab = np.array([])
        d2CGL_dVddeltab = np.array([])
        d2CGG_dVddeltab = np.array([])

    if compute_sb_ssc and cfg['covariance']['use_KE_approximation']:
        # in this case, the k grid used is the same as the Pk one, I think
        k_grid_s2b = np.array([])

    if compute_sb_ssc:
        sigma2_b = ssc_obj.sigma2_b

    _bnt_matrix = np.array([]) if bnt_matrix is None else bnt_matrix
    _mag_bias_2d = (
        ccl_obj.mag_bias_2d if cfg['C_ell']['has_magnification_bias'] else np.array([])
    )

    _ell_dict = vars(ell_obj)
    # _ell_dict.pop('ell_cuts_dict')
    # _ell_dict.pop('idxs_to_delete_dict')

    if cfg['namaster']['use_namaster']:
        import pymaster

        # convert NmtBin objects to effective ells
        for key in _ell_dict:
            if key.startswith('nmt_bin_obj_'):
                assert isinstance(_ell_dict[key], pymaster.bins.NmtBin), (
                    f'Expected NmtBin for {key}, got {_ell_dict[key]}'
                )
                _ell_dict[key] = _ell_dict[key].get_effective_ells()

    # save metadata
    import datetime

    branch, commit = sl.get_git_info()
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'branch': branch,
        'commit': commit,
    }

    # this is no longer set manually
    # bench_filename = cfg['misc']['bench_filename'].format(
    #     g_code=cfg['covariance']['G_code'],
    #     ssc_code=cfg['covariance']['SSC_code'] if cfg['covariance']['SSC'] else 'None',
    #     cng_code=cfg['covariance']['cNG_code'] if cfg['covariance']['cNG'] else 'None',
    #     use_KE=str(cfg['covariance']['use_KE_approximation']),
    #     which_pk_responses=cfg['covariance']['which_pk_responses'],
    #     which_b1g_in_resp=cfg['covariance']['which_b1g_in_resp'],
    # )
    bench_filename = cfg['misc']['bench_filename']

    if os.path.exists(f'{bench_filename}.npz'):
        raise ValueError(
            'You are trying to overwrite the benchmark file at'
            f' {bench_filename}.npz.'
            'Please rename the new benchmark or delete the existing one.'
        )

    with open(f'{bench_filename}.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # save every array contained in _cov_obj
    covs_arrays_dict = {
        k: v for k, v in vars(_cov_obj).items() if isinstance(v, np.ndarray)
    }
    # remove the 'ind' arrays
    covs_arrays_dict.pop('ind')
    covs_arrays_dict.pop('ind_auto')
    covs_arrays_dict.pop('ind_cross')

    # make the keys consistent with the old benchmark files
    covs_arrays_dict_renamed = covs_arrays_dict.copy()
    for key, cov in covs_arrays_dict.items():
        # key_new = key.replace('_tot_', '_TOT_')
        key_new = key.replace('_2d', '_2D')
        covs_arrays_dict_renamed[key_new] = cov
        covs_arrays_dict_renamed.pop(key)

    np.savez_compressed(
        bench_filename,
        backup_cfg=cfg,
        ind=ind,
        z_grid=z_grid,
        z_grid_trisp=z_grid_trisp,
        k_grid=k_grid,
        k_grid_sigma2_b=k_grid_s2b,
        nz_src=nz_src,
        nz_lns=nz_lns,
        bnt_matrix=_bnt_matrix,
        gal_bias_2d=ccl_obj.gal_bias_2d,
        mag_bias_2d=_mag_bias_2d,
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
        **_ell_dict,
        **covs_arrays_dict_renamed,
        metadata=metadata,
    )


if (
    cfg['misc']['test_condition_number']
    or cfg['misc']['test_cholesky_decomposition']
    or cfg['misc']['test_numpy_inversion']
    or cfg['misc']['test_symmetry']
):
    key = list(cov_dict_tosave_2d.keys())[0] if len(cov_dict_tosave_2d) == 1 else 'TOT'
    cov = cov_dict_tosave_2d[key]
    print(f'Testing cov {cov_name}...\n')

    if cfg['misc']['test_condition_number']:
        cond_number = np.linalg.cond(cov)
        print(f'Condition number = {cond_number:.4e}')

    if cfg['misc']['test_cholesky_decomposition']:
        try:
            np.linalg.cholesky(cov)
            print('Cholesky decomposition successful')
        except np.linalg.LinAlgError:
            print(
                'Cholesky decomposition failed. Consider checking the condition'
                ' number or symmetry.'
            )

    if cfg['misc']['test_numpy_inversion']:
        try:
            inv_cov = np.linalg.inv(cov)
            print('Numpy inversion successful.')
            # Test correctness of inversion:
            identity_check = np.allclose(
                np.dot(cov, inv_cov), np.eye(cov.shape[0]), atol=1e-9, rtol=1e-7
            )
            if identity_check:
                print(
                    'Inverse test successfully (M @ M^{-1} is identity). '
                    'atol=1e-9, rtol=1e-7'
                )
            else:
                print(
                    f'Warning: Inverse test failed (M @ M^{-1} '
                    'deviates from identity). atol=0, rtol=1e-7'
                )
        except np.linalg.LinAlgError:
            print('Numpy inversion failed: Matrix is singular or near-singular.')

    if cfg['misc']['test_symmetry']:
        if not np.allclose(cov, cov.T, atol=0, rtol=1e-7):
            print('Warning: Matrix is not symmetric. atol=0, rtol=1e-7')
        else:
            print('Matrix is symmetric. atol=0, rtol=1e-7')

        print('')

# note that this is *not* compatible with %matplotlib inline in the interactive window!
if cfg['misc']['save_figs']:
    output_dir = f'{output_path}/figs'
    os.makedirs(output_dir, exist_ok=True)
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig.savefig(os.path.join(output_dir, f'fig_{i:03d}.png'))

import heracles

cov_hc_dict = io_handler.cov_sb_10d_to_heracles_dict(
    cov_hs_obj.cov_3x2pt_g_10d, squeeze=True
)

heracles.io.write('./stop_looking_at_your_laptops_eyes_on_me.fits', cov_hc_dict)

cov_hc_dict = heracles.io.read('./stop_looking_at_your_laptops_eyes_on_me.fits')
cov_10d = io_handler.cov_heracles_dict_to_sb_10d(
    cov_hc_dict, zbins, ell_obj.nbl_3x2pt, 2
)
cov_4d = sl.cov_3x2pt_10D_to_4D(
    cov_3x2pt_10D=cov_10d,
    probe_ordering=probe_ordering,
    nbl=ell_obj.nbl_3x2pt,
    zbins=zbins,
    ind_copy=ind.copy(),
    GL_OR_LG=GL_OR_LG,
    req_probe_combs_2d=req_probe_combs_hs_2d,
)
cov_2d = sl.cov_4D_to_2DCLOE_3x2pt_hs(cov_4d, zbins, req_probe_combs_hs_2d, 'ell')

# ! SIMPLIFIED VERSION OF THE RESHAPING ROUTINES TO PASS TO GUADA

req_probe_combs_2d = req_probe_combs_hs_2d
from spaceborne import cov_transform as ct

cov_4d = ct.cov_3x2pt_10d_to_4d()



print(f'Finished in {(time.perf_counter() - script_start_time) / 60:.2f} minutes')

# THIS CODE HAS BEEN COMMENTED OUT TO TEST AGAINST THE BENCHMARKS
"""
# BOOKMARK 2
# ! read OC files: list
oc_path = f'{output_path}/OneCovariance'

oc_output_covlist_fname = (
    f'{oc_path}/{cfg["OneCovariance"]["oc_output_filename"]}_list.dat'
)
cov_oc_dict_6d = oc_interface.process_cov_from_list_file(
    oc_output_covlist_fname,
    zbins=zbins,
    df_chunk_size=5_000_000,
)

# compare individual terms/probes
term = cov_rs_obj.terms_toloop[0]

term = 'mix'
for probe_sb in unique_probe_combs_rs:
    oc_interface.compare_sb_cov_to_oc_list(
        cov_rs_obj=cov_rs_obj,
        cov_oc_dict_6d=cov_oc_dict_6d,
        probe_sb=probe_sb,
        term=term,
        ind_auto=ind_auto,
        ind_cross=ind_cross,
        zpairs_auto=zpairs_auto,
        zpairs_cross=zpairs_cross,
        scale_bins=scale_bins,
        title=None,
    )

cov_sva_oc_3x2pt_8D = cov_oc_dict_6d['cov_sva_oc_3x2pt_8D']
cov_sn_oc_3x2pt_8D = cov_oc_dict_6d['cov_sn_oc_3x2pt_8D']
cov_mix_oc_3x2pt_8D = cov_oc_dict_6d['cov_mix_oc_3x2pt_8D']

cov_oc_list_8d = cov_sva_oc_3x2pt_8D + cov_sn_oc_3x2pt_8D + cov_mix_oc_3x2pt_8D

# TODO SB and OC MUST have same fmt so I can combine probes and terms with the same function!!!
cov_oc_dict_2d = {}
nbt = cfg['binning']['theta_bins']
for probe in const.RS_PROBE_NAME_TO_IX_DICT:
    # split_g_ix = (
    # cov_rs_obj.split_g_dict[term] if term in ['sva', 'sn', 'mix'] else 0
    # )

    # term_oc = (
    #     'gauss'
    #     if (len(cov_rs_obj.terms_toloop) > 1 or term == 'gauss_ell')
    #     else term
    # )

    probe_ab, probe_cd = sl.split_probe_name(probe)
    probe_ab_ix, probe_cd_ix = (
        const.RS_PROBE_NAME_TO_IX_DICT_SHORT[probe_ab],
        const.RS_PROBE_NAME_TO_IX_DICT_SHORT[probe_cd],
    )

    zpairs_ab = zpairs_cross if probe_ab_ix == 1 else zpairs_auto
    zpairs_cd = zpairs_cross if probe_cd_ix == 1 else zpairs_auto
    ind_ab = ind_cross if probe_ab_ix == 1 else ind_auto
    ind_cd = ind_cross if probe_cd_ix == 1 else ind_auto

    # no need to assign 6d and 4d to dedicated dictionary
    probe_oc = probe.replace('gt', 'gm')
    cov_oc_6d = cov_oc_list_8d[*cov_rs_obj.probe_idx_dict_short_oc[probe_oc], ...]

    # check theta simmetry
    if np.allclose(cov_oc_6d, cov_oc_6d.transpose(1, 0, 2, 3, 4, 5), atol=0, rtol=1e-5):
        print(f'probe {probe} is symmetric in theta_1, theta_2')

    # if probe in ['gtxip', 'gtxim']:
    #     print('I am manually transposing the OC blocks!!')
    #     warnings.warn('I am manually transposing the OC blocks!!', stacklevel=2)
    #     cov_oc_6d = cov_oc_6d.transpose(1, 0, 3, 2, 5, 4)

    cov_oc_4d = sl.cov_6D_to_4D_blocks(
        cov_oc_6d, nbt, zpairs_ab, zpairs_cd, ind_ab, ind_cd
    )

    cov_oc_dict_2d[probe] = sl.cov_4D_to_2D(
        cov_oc_4d, block_index='zpair', optimize=True
    )


cov_oc_list_2d = cov_real_space.stack_probe_blocks_deprecated(cov_oc_dict_2d)

# ! read OC files: mat
cov_oc_mat_2d = np.genfromtxt(
    oc_output_covlist_fname.replace('list.dat', 'matrix_gauss.mat')
)
cov_oc_mat_2d_2 = np.genfromtxt(
    oc_output_covlist_fname.replace('list.dat', 'matrix.mat')
)
np.testing.assert_allclose(cov_oc_mat_2d, cov_oc_mat_2d_2, atol=0, rtol=1e-5)

del cov_oc_mat_2d_2
gc.collect()

# compare OC list against mat - transposition issue is still present!
# sl.compare_2d_covs(
#     cov_oc_list_2d, cov_oc_mat_2d, 'list', 'mat', title=title, diff_threshold=1
# )

# ! compare full 2d SB against mat fmt
# I will compare SB against the mat fmt
cov_sb_2d = cov_rs_obj.cov_3x2pt_g_2d
cov_oc_2d = cov_oc_mat_2d

title = (
    f'integration {cfg["precision"]["cov_rs_int_method"]} - '
    f'ell_bins_rs {cfg["precision"]["ell_bins_rs"]} - '
    f'ell_max_rs {cfg["precision"]["ell_max_rs"]} - '
    f'theta bins fine {cfg["precision"]["theta_bins_fine"]}\n'
    f'n_sub {cfg["precision"]["n_sub"]} - '
    f'n_bisec_max {cfg["precision"]["n_bisec_max"]} - '
    f'rel_acc {cfg["precision"]["rel_acc"]}'
)


# # rearrange in OC 2D SB fmt
elem_auto = nbt * zpairs_auto
elem_cross = nbt * zpairs_cross
# these are the end indices
lim_1 = elem_auto
lim_2 = lim_1 + elem_cross
lim_3 = lim_2 + elem_auto
lim_4 = lim_3 + elem_auto

assert lim_4 == cov_oc_2d.shape[0]
assert lim_4 == cov_oc_2d.shape[1]

cov_oc_2d_dict = {
    # first OC row is gg
    'gggg': cov_oc_2d[:lim_1, :lim_1],
    'gggt': cov_oc_2d[:lim_1, lim_1:lim_2],
    'ggxip': cov_oc_2d[:lim_1, lim_2:lim_3],
    'ggxim': cov_oc_2d[:lim_1, lim_3:lim_4],
    'gtgg': cov_oc_2d[lim_1:lim_2, :lim_1],
    'gtgt': cov_oc_2d[lim_1:lim_2, lim_1:lim_2],
    'gtxip': cov_oc_2d[lim_1:lim_2, lim_2:lim_3],
    'gtxim': cov_oc_2d[lim_1:lim_2, lim_3:lim_4],
    'xipgg': cov_oc_2d[lim_2:lim_3, :lim_1],
    'xipgt': cov_oc_2d[lim_2:lim_3, lim_1:lim_2],
    'xipxip': cov_oc_2d[lim_2:lim_3, lim_2:lim_3],
    'xipxim': cov_oc_2d[lim_2:lim_3, lim_3:lim_4],
    'ximgg': cov_oc_2d[lim_3:lim_4, :lim_1],
    'ximgt': cov_oc_2d[lim_3:lim_4, lim_1:lim_2],
    'ximxip': cov_oc_2d[lim_3:lim_4, lim_2:lim_3],
    'ximxim': cov_oc_2d[lim_3:lim_4, lim_3:lim_4],
}

cov_oc_2d = cov_real_space.stack_probe_blocks_deprecated(cov_oc_2d_dict)


sl.compare_2d_covs(cov_sb_2d, cov_oc_2d, 'SB', 'OC', title=title, diff_threshold=5)

"""
