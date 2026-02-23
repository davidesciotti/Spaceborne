# ruff: noqa: E402 (ignore module import not on top of the file warnings)
import argparse
import contextlib
import os
import sys
import warnings

import yaml

# TODOS BRANCH
# - test against OC, and update the corresponding dedicated cfg
# - make sure to compute cng on a finer ell grid
# - ssc computation should not be in the main, btw, I don't think it'll be difficult to port it to the SSC class
# - finish commenting out the new code, also to tidy it up
# - try feeding OC NG covs to the simps projection
# - port to Melodie for speed?
# - fix RS shot noise
# - it would be nice to recycle the implementation for a quick and dirty RS NG simps cov projection
# - merge to develop in small chunks! After current validation might me a good idea
# - pylevin as a dependency should be taken care of in cloelib, so remove it from the env
# - should I remove the call to symmetrize_probe_cov_dict_6d bc I symmetrized in the load_list_fmt function=? for OC, of course
# - SSC and cNG can be computed for mmmm only when projecting En and Bn COSEBIs!!!
# - put markers in CPU vs time to understand portion of the code which could be parallelised


# [QUESTIONS FOR ROBERT]
# - what is the ell range used to compute the NG HS covs used for projection to COSEBIs NG?
# - from the comaprison it looks like the SSC normalization factor is 2pi*amax, but in the paper there's no amax...
# - on the other hand, the code has 1/4pi^2...


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

# JAX settings
if cfg['misc']['jax_platform'] != 'auto':
    os.environ['JAX_PLATFORMS'] = cfg['misc']['jax_platform']
if cfg['precision']['jax_enable_x64']:
    os.environ['JAX_ENABLE_X64'] = '1'
else:
    os.environ['JAX_ENABLE_X64'] = '0'
    warnings.warn(
        'JAX 64-bit precision is disabled. This may lead to '
        'noticeable differences with respect to the numpy '
        'implementation, which uses 64-bit precision by default.',
        stacklevel=2,
    )

# Import JAX after environment variables are set, then print device info
import jax

print(f'JAX devices: {jax.devices()}')
print(f'JAX backend: {jax.default_backend()}')

# if using the CPU, set the number of threads
num_threads = cfg['misc']['num_threads']

# Cap num_threads at logical CPU count to prevent oversubscription
cpu_count = os.cpu_count() or 1
if num_threads > cpu_count:
    print(f'WARNING: num_threads={num_threads} exceeds CPU count={cpu_count}')
    print(f'         Capping at {cpu_count} to prevent thread oversubscription')
    num_threads = cpu_count

os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
os.environ['XLA_FLAGS'] = (
    f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={str(num_threads)}'
)

# override in cfg as well
cfg['misc']['num_threads'] = num_threads

import itertools
import pprint
import time
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
    cov_cosebis,
    cov_harmonic_space,
    cov_real_space,
    ell_utils,
    io_handler,
    mask_utils,
    oc_interface,
    responses,
    wf_cl_lib,
)
from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import plot_lib as sb_plt
from spaceborne import sb_lib as sl

with contextlib.suppress(ImportError):
    import pyfiglet

    text = 'Spaceborne'
    ascii_art = pyfiglet.figlet_format(text=text, font='slant')
    print(ascii_art)

if 'ipykernel_launcher.py' not in sys.argv[0] and '--show-plots' not in sys.argv:
    matplotlib.use('Agg')


YELLOW = '\033[33m'
RESET = '\033[0m'


def _clean_warning(message, category, filename, lineno, file=None, line=None):
    return f'\n{YELLOW}{category.__name__}: {message}  \n{filename}:{lineno}{RESET}\n\n'


warnings.formatwarning = _clean_warning


warnings.filterwarnings(
    'ignore',
    message='.*FigureCanvasAgg is non-interactive, and thus cannot be shown.*',
    category=UserWarning,
)

pp = pprint.PrettyPrinter(indent=4)
script_start_time = time.perf_counter()


# UNCOMMENT TO MONITOR CPU COUNT USAGE
import threading
import time
import psutil
import pandas as pd

cpu_data = []


def monitor_cpu(interval=0.5):
    """Monitor CPU usage per core"""
    print('Starting CPU monitor...')
    while not stop_event.is_set():
        timestamp = time.time()
        per_core = psutil.cpu_percent(percpu=True, interval=interval)
        cpu_data.append(
            {
                'time': timestamp,
                'cores_used': sum(1 for x in per_core if x > 10),  # cores > 10% usage
                'per_core': per_core,
            }
        )
    print('CPU monitoring stopped')


stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_cpu, args=(0.5,))
monitor_thread.start()


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

# ! ======================== START HARDCODED OPTIONS/PARAMETERS ========================
use_h_units = False  # whether or not to normalize Megaparsecs by little h

# for the Gaussian covariance computation
k_steps_sigma2_simps = 20_000
shift_nz_interpolation_kind = 'linear'


# these are configs which should not be visible to the user
cfg['covariance']['n_probes'] = 2


# ===================================== pylevin ======================================
# Precision settings for pylevin. See the official documentation for more details:
# https://levin-bessel.readthedocs.io/en/latest/index.html
# https://github.com/rreischke/levin_bessel/blob/main/tutorial/levin_tutorial.ipynb

# number of collocation points in each bisection. default: 8
cfg['precision']['n_sub'] = 50
# maximum number of bisections used. default: 32
cfg['precision']['n_bisec_max'] = 500
# relative accuracy target. default: 1.e-4
cfg['precision']['rel_acc'] = 1.0e-7
# Type: bool. Compute bessel functions with boost instead of GSL (higher accuracy at high Bessel orders)
cfg['precision']['boost_bessel'] = True
# Type: bool. Whether to display warnings
cfg['precision']['verbose'] = True


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
    cfg['OneCovariance']['oc_output_filename'] = 'cov_oc'
    cfg['OneCovariance']['compare_against_oc'] = False

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

# Psi-statistics not implemented yet
cfg['probe_selection']['Psigl'] = False
cfg['probe_selection']['Psigg'] = False

# Sigma2_b settings, common to Spaceborne and PyCCL. Can be one of:
# - full_curved_sky: Use the full- (curved-) sky expression (for Spaceborne only).
#   In this case, the output covmat
# - from_input_mask: input a mask with path specified by mask_filename
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
cfg['covariance']['probe_ordering'] = [['L', 'L'], ['G', 'L'], ['G', 'G']]

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
# ! ======================== END HARDCODED OPTIONS/PARAMETERS ==========================

# convenence settings that have been hardcoded
n_probes = cfg['covariance']['n_probes']
which_sigma2_b = cfg['covariance']['which_sigma2_b']

# ! probe selection

# * small naming guide for the confused developer:
# - unique_probe_combs: the probe combinations which are actually computed, meaning the
#                       elements of the diagonal and, if requested, the cross-terms.
# - symm_probe_combs:   the lower triangle, (or an empty list if cross terms are not
#                       required), which are the blocks filled by symmetry
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
# symm_probe_combs   = []
# nonreq_probe_combs = {'GGGG', 'GGGL', 'GGLL', 'GLGG', 'GLLL', 'LLGG', 'LLGL'}
# req_probe_combs_2d = ['LLLL', 'LLGL', 'GLLL', 'GLGL']

probe_combs_dict_hs = sl.get_probe_combs_wrapper(
    obs_space='harmonic',
    probe_selection=cfg['probe_selection'],
    cross_cov=cfg['probe_selection']['cross_cov'],
)
probe_combs_dict_rs = sl.get_probe_combs_wrapper(
    obs_space='real',
    probe_selection=cfg['probe_selection'],
    cross_cov=cfg['probe_selection']['cross_cov'],
)
probe_combs_dict_cs = sl.get_probe_combs_wrapper(
    obs_space='cosebis',
    probe_selection=cfg['probe_selection'],
    cross_cov=cfg['probe_selection']['cross_cov'],
)

# in case real space or cosebis are required, we must compute the harmonic space
# non-gaussian covariance. This only needs to be evaluated for the appropriate probes,
# following the mapping below
if obs_space == 'real':
    _cps = cfg['probe_selection']
    probe_combs_dict_hs = sl.get_probe_combs_wrapper(
        obs_space='harmonic',
        probe_selection={
            'LL': bool(_cps['xip'] or _cps['xim']),
            'GL': bool(_cps['gt']),
            'GG': bool(_cps['w']),
        },
        cross_cov=cfg['probe_selection']['cross_cov'],
    )
elif obs_space == 'cosebis':
    _cps = cfg['probe_selection']
    probe_combs_dict_hs = sl.get_probe_combs_wrapper(
        obs_space='harmonic',
        probe_selection={
            'LL': bool(_cps['En'] or _cps['Bn']),
            'GL': bool(_cps['Psigl']),
            'GG': bool(_cps['Psigg']),
        },
        cross_cov=cfg['probe_selection']['cross_cov'],
    )


unique_probe_combs_hs = probe_combs_dict_hs['unique_probe_combs']
req_probe_combs_hs_2d = probe_combs_dict_hs['req_probe_combs_2d']
nonreq_probe_combs_hs = probe_combs_dict_hs['nonreq_probe_combs']
symm_probe_combs_hs = probe_combs_dict_hs['symm_probe_combs']

unique_probe_combs_rs = probe_combs_dict_rs['unique_probe_combs']
req_probe_combs_rs_2d = probe_combs_dict_rs['req_probe_combs_2d']
nonreq_probe_combs_rs = probe_combs_dict_rs['nonreq_probe_combs']
symm_probe_combs_rs = probe_combs_dict_rs['symm_probe_combs']

unique_probe_combs_cs = probe_combs_dict_cs['unique_probe_combs']
req_probe_combs_cs_2d = probe_combs_dict_cs['req_probe_combs_2d']
nonreq_probe_combs_cs = probe_combs_dict_cs['nonreq_probe_combs']
symm_probe_combs_cs = probe_combs_dict_cs['symm_probe_combs']


if obs_space == 'harmonic':
    req_probe_combs_2d = req_probe_combs_hs_2d
    nbx = cfg['binning']['ell_bins']
elif obs_space == 'real':
    req_probe_combs_2d = req_probe_combs_rs_2d
    nbx = cfg['binning']['theta_bins']
elif obs_space == 'cosebis':
    req_probe_combs_2d = req_probe_combs_cs_2d
    nbx = cfg['binning']['n_modes_cosebis']
else:
    raise ValueError(f'Unknown observables space: {obs_space:s}')


# ! set non-gaussian cov terms to compute
cov_terms_list = []
if cfg['covariance']['G']:
    cov_terms_list.append('G')
if cfg['covariance']['SSC']:
    cov_terms_list.append('SSC')
if cfg['covariance']['cNG']:
    cov_terms_list.append('cNG')
cov_terms_str = ''.join(cov_terms_list)

# set required terms based on config. Note that the sva, sn and mix terms are always
# computed, regardless of the value of split_gaussian_cov in the config.
req_terms = []
if cfg['covariance']['G']:
    req_terms.extend(['sva', 'sn', 'mix', 'g'])
if cfg['covariance']['SSC']:
    req_terms.append('ssc')
if cfg['covariance']['cNG']:
    req_terms.append('cng')
if 'cng' in req_terms or 'ssc' in req_terms:
    req_terms.append('tot')

if req_terms == []:
    raise ValueError('No covariance terms have been selected!')

_req_probe_combs_2d = [
    sl.split_probe_name(probe, space=obs_space) for probe in req_probe_combs_2d
]
_req_probe_combs_2d.append('3x2pt')
dims = ['6d', '4d', '2d']
cov_dict = cd.create_cov_dict(req_terms, _req_probe_combs_2d, dims=dims)

# TODO I can probably delete these?
compute_oc_g, compute_oc_ssc, compute_oc_cng = False, False, False
compute_sb_ssc, compute_sb_cng = False, False
compute_ccl_ssc, compute_ccl_cng = False, False
if cfg['covariance']['G'] and cfg['covariance']['G_code'] == 'OneCovariance':
    compute_oc_g = True
if cfg['covariance']['SSC'] and cfg['covariance']['SSC_code'] == 'OneCovariance':
    compute_oc_ssc = True
if cfg['covariance']['cNG'] and cfg['covariance']['cNG_code'] == 'OneCovariance':
    compute_oc_cng = True
if cfg['covariance']['G'] and cfg['OneCovariance']['compare_against_oc']:
    compute_oc_g = True
if cfg['covariance']['SSC'] and cfg['OneCovariance']['compare_against_oc']:
    compute_oc_ssc = True
if cfg['covariance']['cNG'] and cfg['OneCovariance']['compare_against_oc']:
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

cov_terms_and_codes = {
    'G': cfg['covariance']['G_code'] if cfg['covariance']['G'] else False,
    'SSC': cfg['covariance']['SSC_code'] if cfg['covariance']['SSC'] else False,
    'cNG': cfg['covariance']['cNG_code'] if cfg['covariance']['cNG'] else False,
}

_condition = 'GLGL' in req_probe_combs_hs_2d or 'gtgt' in req_probe_combs_rs_2d
if compute_ccl_cng and _condition:
    warnings.warn(
        'There seems to be some issue with the symmetry of the GLGL '
        'block in the '
        'CCL cNG covariance, so for the moment it is disabled. '
        'The LLLL and GGGG blocks are not affected, so you can still '
        'compute the single-probe cNG covariances.',
        stacklevel=2,
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


if cfg['precision']['use_KE_approximation']:
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
    cfg['precision']['spline_params'],
    cfg['precision']['gsl_params'],
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
    cfg['precision']['z_min'],
    cfg['precision']['z_max'],
    cfg['precision']['z_steps']
)  # fmt: skip
z_grid_trisp = np.linspace(
    cfg['precision']['z_min'],
    cfg['precision']['z_max'],
    cfg['precision']['z_steps_trisp'],
)
k_grid = np.logspace(
    cfg['precision']['log10_k_min'],
    cfg['precision']['log10_k_max'],
    cfg['precision']['k_steps'],
)
# in this case we need finer k binning because of the bessel functions
k_grid_s2b = np.logspace(
    cfg['precision']['log10_k_min'],
    cfg['precision']['log10_k_max'],
    k_steps_sigma2_simps
)  # fmt: skip

if len(z_grid) < 1000:
    warnings.warn(
        'the number of steps in the redshift grid is small, '
        'you may want to consider increasing it',
        stacklevel=2,
    )

zgrid_str = (
    f'zmin{cfg["precision"]["z_min"]}_'
    f'zmax{cfg["precision"]["z_max"]}_'
    f'zsteps{cfg["precision"]["z_steps"]}'
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
ind_dict = {'LL': ind_auto, 'GL': ind_cross, 'GG': ind_auto}

# TODO block_index must only be taken from here!
cov_ordering_2d = cfg['covariance']['covariance_ordering_2D']
if cov_ordering_2d in ['probe_scale_zpair', 'scale_probe_zpair']:
    block_index = 'ell'
elif cov_ordering_2d == 'probe_zpair_scale':
    block_index = 'zpair'
else:
    raise ValueError(f'Unknown covariance_ordering_2D: {cov_ordering_2d:s}')


# private cfg dictionary. This serves a couple different purposeses:
# 1. To store and pass hardcoded parameters in a convenient way
# 2. To make the .format() more compact
pvt_cfg = {
    'zbins': zbins,
    'ind': ind,
    'ind_dict': ind_dict,
    'ind_auto': ind_auto,
    'ind_cross': ind_cross,
    'zpairs_auto': zpairs_auto,
    'zpairs_cross': zpairs_cross,
    'zpairs_3x2pt': zpairs_3x2pt,
    'block_index': block_index,
    'req_terms': req_terms,
    'n_probes': n_probes,
    'probe_ordering': probe_ordering,
    'cov_ordering_2d': cov_ordering_2d,
    'unique_probe_combs_hs': unique_probe_combs_hs,
    'req_probe_combs_hs_2d': req_probe_combs_hs_2d,
    'req_probe_combs_rs_2d': req_probe_combs_rs_2d,
    'req_probe_combs_cs_2d': req_probe_combs_cs_2d,
    'nonreq_probe_combs_hs': nonreq_probe_combs_hs,
    'nonreq_probe_combs_rs': nonreq_probe_combs_rs,
    'nonreq_probe_combs_cs': nonreq_probe_combs_cs,
    'which_ng_cov': cov_terms_str,
    'cov_terms_list': cov_terms_list,
    'GL_OR_LG': GL_OR_LG,
    'symmetrize_output_dict': const.HS_SYMMETRIZE_OUTPUT_DICT,
    'use_h_units': use_h_units,
    'z_grid': z_grid,
    'nbx': nbx,
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

if shift_nz:
    nz_src = wf_cl_lib.shift_nz(
        zgrid_nz_src,
        nz_src,
        cfg['nz']['dzWL'],
        normalize=False,
        plot_nz=True,
        interpolation_kind=shift_nz_interpolation_kind,
        bounds_error=False,
        fill_value=0,
        plt_title='$n_i(z)$ sources shifts',
    )
    nz_lns = wf_cl_lib.shift_nz(
        zgrid_nz_lns,
        nz_lns,
        cfg['nz']['dzGC'],
        normalize=False,
        plot_nz=True,
        interpolation_kind=shift_nz_interpolation_kind,
        bounds_error=False,
        fill_value=0,
        plt_title='$n_i(z)$ lenses shifts',
    )

if cfg['nz']['smooth_nz']:
    print(
        f'Smoothing n(z) with Gaussian filter of sigma = {cfg["nz"]["sigma_smoothing"]}'
    )
    for zi in range(zbins):
        nz_src[:, zi] = gaussian_filter1d(
            nz_src[:, zi], sigma=cfg['nz']['sigma_smoothing']
        )
        nz_lns[:, zi] = gaussian_filter1d(
            nz_lns[:, zi], sigma=cfg['nz']['sigma_smoothing']
        )

# check if they are normalized, and if not do so
if cfg['nz']['normalize_nz']:
    print('Checking n(z) normalization...')
    nz_lns_integral = simps(y=nz_lns, x=zgrid_nz_lns, axis=0)
    nz_src_integral = simps(y=nz_src, x=zgrid_nz_src, axis=0)

    if not np.allclose(nz_lns_integral, 1, atol=0, rtol=1e-3):
        warnings.warn(
            '\nThe lens n(z) are not normalized within atol=0, rtol=1e-3. '
            'Proceeding to normalize them',
            stacklevel=2,
        )
        nz_lns /= nz_lns_integral
    else:
        print('Lens n(z) are normalized')

    if not np.allclose(nz_src_integral, 1, atol=0, rtol=1e-3):
        warnings.warn(
            '\nThe source n(z) are not normalized within atol=0, rtol=1e-3. '
            'Proceeding to normalize them',
            stacklevel=2,
        )
        nz_src /= nz_src_integral
    else:
        print('Source n(z) are normalized')

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
    # it's safer to use a linear interpolation, in case these functions are top-hats
    # (e.g. when requiring constant bias in each bin, over its redshift support)
    ccl_obj.gal_bias_2d, ccl_obj.gal_bias_func = sl.check_interpolate_input_tab(
        input_tab=gal_bias_input,
        z_grid_out=z_grid,
        zbins=zbins,
        kind=cfg['C_ell']['gal_bias_table_interp_method'],
    )
    ccl_obj.gal_bias_tuple = (z_grid, ccl_obj.gal_bias_2d)

elif cfg['C_ell']['which_gal_bias'] == 'polynomial_fit':
    ccl_obj.set_gal_bias_tuple_spv3(
        z_grid_lns=z_grid, magcut_lens=None, poly_fit_values=galaxy_bias_fit_fiducials
    )
else:
    raise ValueError('which_gal_bias should be "from_input" or "polynomial_fit"')

# Check if the galaxy bias is the same in all bins
# Note: the [0] (inside square brackets) means "select column 0 but keep the array
# two-dimensional", for shape consistency
single_b_of_z = np.allclose(ccl_obj.gal_bias_2d, ccl_obj.gal_bias_2d[:, [0]])


# ! ============================ Magnification bias ====================================
if cfg['C_ell']['has_magnification_bias']:
    if cfg['C_ell']['which_mag_bias'] == 'from_input':
        mag_bias_input = np.genfromtxt(cfg['C_ell']['mag_bias_table_filename'])
        # it's safer to use a linear interpolation, in case these functions are top-hats
        # (e.g. when requiring constant bias in each bin, over its redshift support)
        ccl_obj.mag_bias_2d, ccl_obj.mag_bias_func = sl.check_interpolate_input_tab(
            input_tab=mag_bias_input,
            z_grid_out=z_grid,
            zbins=zbins,
            kind=cfg['C_ell']['mag_bias_table_interp_method'],
        )
        ccl_obj.mag_bias_tuple = (z_grid, ccl_obj.mag_bias_2d)
    elif cfg['C_ell']['which_mag_bias'] == 'polynomial_fit':
        ccl_obj.set_mag_bias_tuple(
            z_grid_lns=z_grid,
            has_magnification_bias=cfg['C_ell']['has_magnification_bias'],
            magcut_lens=None,
            poly_fit_values=magnification_bias_fit_fiducials,
        )
    else:
        raise ValueError('which_mag_bias should be "from_input" or "polynomial_fit"')
else:
    ccl_obj.mag_bias_tuple = None

plt.figure()
for zi in range(zbins):
    plt.plot(z_grid, ccl_obj.gal_bias_2d[:, zi], label=f'$z_{{{zi}}}$', c=clr[zi])
plt.xlabel(r'$z$')
plt.ylabel(r'$b_{g, i}(z)$')
plt.legend()


# ! ============================ Radial kernels ========================================
ccl_obj.set_kernel_obj(cfg['C_ell']['has_rsd'], cfg['precision']['n_samples_wf'])
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

for wf_idx in range(len(wf_ccl_list)):
    plt.figure()
    for zi in range(zbins):
        plt.plot(z_grid, wf_ccl_list[wf_idx][:, zi], c=clr[zi], alpha=0.6)
    plt.xlabel('$z$')
    plt.ylabel(r'$W_i^X(z)$')
    plt.suptitle(f'{wf_names_list[wf_idx]}')
    plt.tight_layout()
    plt.show()


# ! ======================================== Cls =======================================
# ! note that the function below includes the multiplicative shear bias
with sl.timer('\nComputing Cls...'):
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


if cfg['C_ell']['use_input_cls']:
    # TODO NMT here you should ask the user for unbinned cls

    # load input cls
    io_obj.get_cl_fmt()
    io_obj.load_cls()

    # check ells before spline interpolation
    io_obj.check_ells_in(ell_obj)

    print(f'Using input Cls for LL from file\n{cfg["C_ell"]["cl_LL_filename"]}')
    print(f'Using input Cls for GGL from file\n{cfg["C_ell"]["cl_GL_filename"]}')
    print(f'Using input Cls for GG from file\n{cfg["C_ell"]["cl_GG_filename"]}')

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
ccl_obj.cl_3x2pt_5d = sl.build_cl_3x2pt_5d(
    cl_ll_3d=ccl_obj.cl_ll_3d, cl_gl_3d=ccl_obj.cl_gl_3d, cl_gg_3d=ccl_obj.cl_gg_3d
)

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
    if 'OneCovariance' in cov_terms_and_codes.values():
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
#     if 'OneCovariance' in cov_terms_and_codes.values():
#         raise NotImplementedError('You should cut also the OC Cls')

# re-set cls in the ccl_obj after BNT transform and/or ell cuts
# ccl_obj.cl_ll_3d = cl_ll_3d
# ccl_obj.cl_gg_3d = cl_gg_3d
# ccl_obj.cl_3x2pt_5d = cl_3x2pt_5d

# ! =========================== Unbinned Cls for nmt/sample/HS bin avg cov =====================
if (
    cfg['covariance']['partial_sky_method'] == 'NaMaster'
    or cfg['sample_covariance']['compute_sample_cov']
    or cfg['precision']['cov_hs_g_ell_bin_average']
):
    if cfg['C_ell']['use_input_cls']:
        cl_3x2pt_unb_5d = sl.build_cl_3x2pt_5d(
            cl_ll_3d=cl_ll_3d_spline(ell_obj.ells_3x2pt_unb),
            cl_gl_3d=cl_gl_3d_spline(ell_obj.ells_3x2pt_unb),
            cl_gg_3d=cl_gg_3d_spline(ell_obj.ells_3x2pt_unb),
        )

    else:
        cl_3x2pt_unb_5d = ccl_interface.compute_cl_3x2pt_5d(
            ccl_obj,
            ells=ell_obj.ells_3x2pt_unb,
            zbins=zbins,
            mult_shear_bias=np.array(cfg['C_ell']['mult_shear_bias']),
            cl_ccl_kwargs=cl_ccl_kwargs,
            n_probes_hs=cfg['covariance']['n_probes'],
        )

if (
    cfg['covariance']['partial_sky_method'] == 'NaMaster'
    or cfg['sample_covariance']['compute_sample_cov']
):
    from spaceborne import cov_partial_sky

    # check that the input cls are computed over a fine enough grid
    if cfg['C_ell']['use_input_cls']:
        for ells_in, ells_out in zip(
            [ells_WL_in, ells_XC_in, ells_GC_in],
            [ell_obj.ells_3x2pt_unb, ell_obj.ells_3x2pt_unb, ell_obj.ells_3x2pt_unb],
            strict=True,
        ):
            io_handler.check_ells_for_spline(ells_in)
            io_handler.check_ells_for_spline(ells_out)

    # initialize cov_nmt_obj and set a couple useful attributes
    cov_nmt_obj = cov_partial_sky.NmtCov(
        cfg=cfg, pvt_cfg=pvt_cfg, ell_obj=ell_obj, mask_obj=mask_obj
    )

    # set unbinned ells in cov_nmt_obj
    cov_nmt_obj.ells_3x2pt_unb = ell_obj.ells_3x2pt_unb
    cov_nmt_obj.nbl_3x2pt_unb = ell_obj.nbl_3x2pt_unb

    cov_nmt_obj.cl_ll_unb_3d = cl_3x2pt_unb_5d[0, 0]
    cov_nmt_obj.cl_gl_unb_3d = cl_3x2pt_unb_5d[1, 0]
    cov_nmt_obj.cl_gg_unb_3d = cl_3x2pt_unb_5d[1, 1]

else:
    cov_nmt_obj = None


# ! ============================== Init real space cov object ==========================
# initialize object
cov_rs_obj = None

if obs_space == 'real':
    # initialize cov_rs_obj and set a couple useful attributes
    cov_rs_obj = cov_real_space.CovRealSpace(cfg, pvt_cfg, mask_obj)

    # set ell values used for projection
    ell_obj.compute_ells_3x2pt_proj()
    cov_rs_obj.ells_proj_g = ell_obj.ells_3x2pt_proj_g
    cov_rs_obj.nbl_proj_g = len(ell_obj.ells_3x2pt_proj_g)
    cov_rs_obj.ells_proj_ng = ell_obj.ells_3x2pt_proj_ng
    cov_rs_obj.nbl_proj_ng = ell_obj.nbl_3x2pt_proj_ng

    # set 3x2pt cls: recompute cls on the finer ell grid...
    if cfg['C_ell']['use_input_cls']:
        cl_3x2pt_5d_for_rs = sl.build_cl_3x2pt_5d(
            cl_ll_3d=cl_ll_3d_spline(cov_rs_obj.ells_proj_g),
            cl_gl_3d=cl_gl_3d_spline(cov_rs_obj.ells_proj_g),
            cl_gg_3d=cl_gg_3d_spline(cov_rs_obj.ells_proj_g),
        )

    else:
        cl_3x2pt_5d_for_rs = ccl_interface.compute_cl_3x2pt_5d(
            ccl_obj,
            ells=cov_rs_obj.ells_proj_g,
            zbins=zbins,
            mult_shear_bias=np.array(cfg['C_ell']['mult_shear_bias']),
            cl_ccl_kwargs=cl_ccl_kwargs,
            n_probes_hs=cfg['covariance']['n_probes'],
        )
    # ...and store them in the cov_rs object
    cov_rs_obj.cl_3x2pt_5d = cl_3x2pt_5d_for_rs

# TODO this could probably be done with super.__init__() where super is the
# cov projector class
cov_cs_obj = None
if obs_space == 'cosebis':
    cov_cs_obj = cov_cosebis.CovCOSEBIs(cfg, pvt_cfg, mask_obj)
    ell_obj.compute_ells_3x2pt_proj()

    # set ell values used for projection
    cov_cs_obj.ells_proj_g = ell_obj.ells_3x2pt_proj_g
    cov_cs_obj.nbl_proj_g = ell_obj.nbl_3x2pt_proj_g
    cov_cs_obj.ells_proj_ng = ell_obj.ells_3x2pt_proj_ng
    cov_cs_obj.nbl_proj_ng = ell_obj.nbl_3x2pt_proj_ng

    # compute projection kernels over ell grids used for the integrals
    # of the G and NG terms
    cov_cs_obj.w_ells_arr = cov_cs_obj.set_w_ells(cov_cs_obj.ells_proj_g)
    if cfg['covariance']['SSC'] or cfg['covariance']['cNG']:
        cov_cs_obj.w_ells_arr_ng = cov_cs_obj.set_w_ells(cov_cs_obj.ells_proj_ng)

    # set 3x2pt cls: recompute cls on the finer ells_proj_g grid...
    if cfg['C_ell']['use_input_cls']:
        cl_3x2pt_5d_for_cs = sl.build_cl_3x2pt_5d(
            cl_ll_3d=cl_ll_3d_spline(cov_cs_obj.ells_proj_g),
            cl_gl_3d=cl_gl_3d_spline(cov_cs_obj.ells_proj_g),
            cl_gg_3d=cl_gg_3d_spline(cov_cs_obj.ells_proj_g),
        )
    else:
        cl_3x2pt_5d_for_cs = ccl_interface.compute_cl_3x2pt_5d(
            ccl_obj,
            ells=cov_cs_obj.ells_proj_g,
            zbins=zbins,
            mult_shear_bias=np.array(cfg['C_ell']['mult_shear_bias']),
            cl_ccl_kwargs=cl_ccl_kwargs,
            n_probes_hs=cfg['covariance']['n_probes'],
        )
    # ...and store them in the cov_cs object
    cov_cs_obj.cl_3x2pt_5d = cl_3x2pt_5d_for_cs

# !  =============================== Build Gaussian covs ===============================
if obs_space == 'harmonic':
    cov_hs_obj = cov_harmonic_space.SpaceborneCovariance(
        cfg, pvt_cfg, ell_obj, cov_nmt_obj, bnt_matrix
    )
    cov_hs_obj.consistency_checks()

    # set unbinned cls if needed
    if cfg['precision']['cov_hs_g_ell_bin_average']:
        cov_hs_obj.cl_3x2pt_unb_5d = cl_3x2pt_unb_5d

    if 'Spaceborne' in cov_terms_and_codes.values():
        cov_hs_obj.set_gauss_cov(ccl_obj=ccl_obj)

else:
    cov_hs_obj = None

# ! =================================== OneCovariance ================================
# initialize object
cov_oc_obj = None
if (
    'OneCovariance' in cov_terms_and_codes.values()
    or cfg['OneCovariance']['compare_against_oc']
):
    if cfg['ell_cuts']['cl_ell_cuts']:
        raise NotImplementedError(
            'TODO double check inputs in this case. This case is untested'
        )

    start_time = time.perf_counter()

    # * 1. save ingredients in ascii format
    # TODO this should be moved to io_handler.py
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
    # ! old
    ell_max_max = cfg['binning']['ell_max']
    ell_min_unb_oc = 2
    ell_max_unb_oc = 5000 if ell_max_max < 5000 else ell_max_max
    nbl_3x2pt_oc = 500

    ells_3x2pt_oc = np.geomspace(
        ell_obj.ell_min_3x2pt, ell_obj.ell_max_3x2pt, nbl_3x2pt_oc
    )

    # ! new
    # nbl_3x2pt_oc = 100
    # nbl_3x2pt_oc = pvt_cfg['nbl_3x2pt']
    # ells_3x2pt_oc, _ = ell_utils.compute_ells_oc(
    #     nbl=nbl_3x2pt_oc,
    #     ell_min=float(pvt_cfg['ell_min_3x2pt']),
    #     ell_max=ell_max_max,
    #     binning_type=cfg['binning']['binning_type'],
    #     output_ell_bin_edges=False,
    # )

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
    cl_3x2pt_5d_oc = sl.build_cl_3x2pt_5d(
        cl_ll_3d=cl_ll_3d_oc, cl_gl_3d=cl_gl_3d_oc, cl_gg_3d=cl_gg_3d_oc
    )

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
    cov_oc_obj = oc_interface.OneCovarianceInterface(
        cfg, pvt_cfg, do_g=compute_oc_g, do_ssc=compute_oc_ssc, do_cng=compute_oc_cng
    )
    cov_oc_obj.oc_path = oc_path
    cov_oc_obj.z_grid_trisp_sb = z_grid_trisp
    cov_oc_obj.path_to_config_oc_ini = f'{cov_oc_obj.oc_path}/input_configs.ini'
    cov_oc_obj.ells_sb = ell_obj.ells_3x2pt
    cov_oc_obj.build_save_oc_ini(ascii_filenames_dict, h, print_ini=True)

    # compute cov
    if cfg['OneCovariance']['new_run']:
        cov_oc_obj.call_oc_from_bash()
    else:
        warnings.warn(
            f'OneCovariance will not be re-run, loading existing files at '
            f'{oc_path}/{cfg["OneCovariance"]["oc_output_filename"]}_list.dat',
            stacklevel=2,
        )

    # load output .list file (maybe the .mat format would be better, actually...)
    # and store it into a 6d dictionary
    oc_output_covlist_fname = (
        f'{oc_path}/{cfg["OneCovariance"]["oc_output_filename"]}_list.dat'
    )
    oc_interface.process_cov_from_list_file(
        cov_dict=cov_oc_obj.cov_dict,
        oc_output_covlist_fname=oc_output_covlist_fname,
        zbins=zbins,
        obs_space=obs_space,
        nbx=nbx,
        df_chunk_size=5_000_000,
    )

    # some useful vars to make the cov processing work regardless of the space
    ps = cfg['probe_selection']
    full_cov = False  # True if all probes + the cross-covariance are required
    if obs_space == 'harmonic':
        _req_probe_combs_2d = req_probe_combs_hs_2d
        # TODO I think these can be deleted!
        full_cov = (ps['LL'] + ps['GL'] + ps['GG']) == 3 and ps['cross_cov'] is True
    elif obs_space == 'real':
        _req_probe_combs_2d = req_probe_combs_rs_2d
        full_cov = (ps['xip'] + ps['xim'] + ps['gt'] + ps['w']) == 4 and ps[
            'cross_cov'
        ] is True
    elif obs_space == 'cosebis':
        _req_probe_combs_2d = req_probe_combs_cs_2d
        full_cov = (ps['En'] + ps['Bn'] + ps['Psigl'] + ps['Psigg']) == 4 and ps[
            'cross_cov'
        ] is True

    # fill the missing probe combinations (ab, cd -> cd, ab) by symmetry
    cov_oc_obj.cov_dict = sl.symmetrize_probe_cov_dict_6d(cov_dict=cov_oc_obj.cov_dict)

    cov_oc_obj.process_cov_from_mat_file()

    # compare list and mat formats
    # TODO this can probaby be deleted (I do this check at the end)
    if full_cov:
        # For this check, we need to create 3x2pt 4d and 2d. The first step is to
        # reshape the blocks to 4d and 2d. This is done here because I don't want
        # to pass
        cov_dict_6d_to_4d_and_2d_kw = {
            'obs_space': obs_space,
            'nbx': nbx,
            'ind_auto': ind_auto,
            'ind_cross': ind_cross,
            'zpairs_auto': zpairs_auto,
            'zpairs_cross': zpairs_cross,
            'block_index': block_index,
        }

        # TODO restore this
        # cov_oc_obj.output_sanity_check(
        #     req_probe_combs_2d=_req_probe_combs_2d,
        #     cov_dict_6d_to_4d_and_2d_kw=cov_dict_6d_to_4d_and_2d_kw,
        #     rtol=1e-4,
        # )

    # This is an alternative method to call OC (more convoluted but more maintanable).
    # I keep the code for optional consistency checks
    if cfg['OneCovariance']['consistency_checks']:
        # store in temp variables for later check
        check_cov_sva_oc_3x2pt_10D = cov_oc_obj.cov_3x2pt_sva_10d
        check_cov_mix_oc_3x2pt_10D = cov_oc_obj.cov_3x2pt_mix_10d
        check_cov_sn_oc_3x2pt_10D = cov_oc_obj.cov_3x2pt_sn_10d
        check_cov_ssc_oc_3x2pt_10D = cov_oc_obj.cov_3x2pt_ssc_10d
        check_cov_cng_oc_3x2pt_10D = cov_oc_obj.cov_3x2pt_cng_10d

        cov_oc_obj.call_oc_from_class()
        cov_oc_obj.process_cov_from_class()

        # a more strict relative tolerance will make this test fail,
        # the number of digits in the .dat and .mat files is lower
        np.testing.assert_allclose(
            check_cov_sva_oc_3x2pt_10D,
            cov_oc_obj.cov_sva_oc_3x2pt_10D,
            atol=0,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            check_cov_mix_oc_3x2pt_10D,
            cov_oc_obj.cov_mix_oc_3x2pt_10D,
            atol=0,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            check_cov_sn_oc_3x2pt_10D, cov_oc_obj.cov_sn_oc_3x2pt_10D, atol=0, rtol=1e-3
        )
        np.testing.assert_allclose(
            check_cov_ssc_oc_3x2pt_10D,
            cov_oc_obj.cov_ssc_oc_3x2pt_10D,
            atol=0,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            check_cov_cng_oc_3x2pt_10D,
            cov_oc_obj.cov_cng_oc_3x2pt_10D,
            atol=0,
            rtol=1e-3,
        )

    # ! reshape probe-specific 6d covs to 4d and 2d and
    # ! construct 3x2pt 2d covs (there is no 6d nor 4d 3x2pt!)
    sl.postprocess_cov_dict(
        cov_dict=cov_oc_obj.cov_dict,
        obs_space=obs_space,
        nbx=nbx,
        ind_auto=ind_auto,
        ind_cross=ind_cross,
        zpairs_auto=zpairs_auto,
        zpairs_cross=zpairs_cross,
        block_index=block_index,
        cov_ordering_2d=cov_ordering_2d,
        req_probe_combs_2d=req_probe_combs_2d,
    )

    print(f'Time taken to compute OC: {(time.perf_counter() - start_time) / 60:.2f} m')

cov_ssc_obj = None
if cov_terms_and_codes['SSC'] == 'Spaceborne':
    # TODO most of this should go in the cov_ssc class
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
    # the finer grid is needed for the non-Gaussian covariance projection
    if obs_space == 'harmonic':
        ell_grid = ell_obj.ells_3x2pt
    # in these two cases, I have to recompute the SSC over a larger and finer grid
    elif obs_space in ['real', 'cosebis']:
        ell_grid = ell_obj.ells_3x2pt_proj_ng

    kmax_limber = cosmo_lib.get_kmax_limber(
        ell_grid, z_grid, use_h_units, ccl_obj.cosmo_ccl
    )

    # ! - test k_max_limber vs k_max_dPk and adjust z_min accordingly
    k_max_resp = np.max(k_grid)
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
    mm_list = [
        dPmm_ddeltab_spline(k_limber_func(ell_val, z_grid), z_grid, grid=False)
        for ell_val in ell_grid
    ]
    dPmm_ddeltab_klimb = np.array(mm_list)

    dPgm_ddeltab_klimb = np.zeros((len(ell_grid), len(z_grid), zbins))
    for zi in range(zbins):
        dPgm_ddeltab_spline = RectBivariateSpline(
            k_grid, z_grid_trisp, dPgm_ddeltab[:, :, zi], kx=3, ky=3
        )
        gm_list = [
            dPgm_ddeltab_spline(k_limber_func(ell_val, z_grid), z_grid, grid=False)
            for ell_val in ell_grid
        ]
        dPgm_ddeltab_klimb[:, :, zi] = np.array(gm_list)

    dPgg_ddeltab_klimb = np.zeros((len(ell_grid), len(z_grid), zbins, zbins))
    for zi in range(zbins):
        for zj in range(zbins):
            dPgg_ddeltab_spline = RectBivariateSpline(
                k_grid, z_grid_trisp, dPgg_ddeltab[:, :, zi, zj], kx=3, ky=3
            )
            gg_list = [
                dPgg_ddeltab_spline(k_limber_func(ell_val, z_grid), z_grid, grid=False)
                for ell_val in ell_grid
            ]
            dPgg_ddeltab_klimb[:, :, zi, zj] = np.array(gg_list)

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

    cov_ssc_obj = cov_ssc.SpaceborneSSC(cfg, pvt_cfg, ccl_obj, z_grid)
    cov_ssc_obj.set_sigma2_b(ccl_obj, mask_obj, k_grid_s2b, which_sigma2_b)

    cov_ssc_obj.compute_ssc(
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
        for probe_2tpl in cov_ssc_obj.cov_dict['ssc']:
            for dim in cov_ssc_obj.cov_dict['ssc'][probe_2tpl]:
                cov_ssc_obj.cov_dict['ssc'][probe_2tpl][dim] /= mask_obj.fsky
    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask', 'flat_sky']:
        pass
    else:
        raise ValueError(f'which_sigma2_b = {which_sigma2_b} not recognized')


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
    
    # prepare list of NG covs to compute
    ccl_ng_cov_terms_list = []
    if compute_ccl_ssc:
        ccl_ng_cov_terms_list.append('SSC')
    if compute_ccl_cng:
        ccl_ng_cov_terms_list.append('cNG')
        
    # prepare ell grid: coarser if you want ell-space covariance, finer (and broader)
    # if it gets projected to real space or COSEBIs
    if obs_space == 'harmonic':
        ell_grid = ell_obj.ells_3x2pt
    elif obs_space in ['real', 'cosebis']:
        ell_grid = ell_obj.ells_3x2pt_proj_ng
        
    # init cov dict
    ccl_obj.set_cov_dict(pvt_cfg, ccl_ng_cov_terms_list)

    # compute covs
    for which_ng_cov in ccl_ng_cov_terms_list:
        ccl_obj.initialize_trispectrum(
            which_ng_cov, unique_probe_combs_hs, cfg['PyCCL']
        )
        ccl_obj.compute_ng_cov_3x2pt(
            which_ng_cov,
            ell_grid,
            mask_obj.fsky,
            integration_method=cfg['PyCCL']['cov_integration_method'],
            unique_probe_combs=unique_probe_combs_hs,
            nonreq_probe_combs=nonreq_probe_combs_hs,
            ind_dict=ind_dict,
        )
    
    # symemtry sanity check
    ccl_obj.check_cov_blocks_symmetry()

# ! ========================== Combine covariance terms ================================
if obs_space == 'harmonic':
    cov_hs_obj.combine_and_reshape_covs(
        ccl_obj=ccl_obj,
        cov_ssc_obj=cov_ssc_obj,
        cov_oc_obj=cov_oc_obj,
        split_gaussian_cov=cfg['covariance']['split_gaussian_cov'],
    )


if obs_space == 'real' and 'Spaceborne' in cov_terms_and_codes.values():
    print('\nComputing real-space covariance...')
    start_rs = time.perf_counter()

    # TODO understand a bit better how to treat real-space SSC and cNG
    for _probe, _term in itertools.product(
        unique_probe_combs_rs, cov_rs_obj.terms_toloop
    ):
        probe_ab, probe_cd = sl.split_probe_name(_probe, space='real')
        print(
            f'\n2PCF cov: computing probe combination {probe_ab, probe_cd}'
            f' - term {_term.upper()}'
        )

        _cov_hs_dict = cov_hs_obj.cov_dict if cov_hs_obj is not None else None
        cov_rs_obj.compute_rs_cov_term_probe_6d(
            cov_hs_dict=_cov_hs_dict, probe_abcd=_probe, term=_term
        )

    for term in cov_rs_obj.terms_toloop:
        sl.fill_remaining_probe_blocks_6d(
            cov_dict=cov_rs_obj.cov_dict,
            term=term,
            symm_probe_combs=symm_probe_combs_rs,
            nonreq_probe_combs=nonreq_probe_combs_rs,
            space='real',
            nbx=nbx,
            zbins=zbins,
        )

    # sum sva, sn and mix to get the Gaussian term
    sl.sum_split_g_terms_allprobeblocks_alldims(cov_rs_obj.cov_dict)

    # postprocess: reshape to 4d and 2d, build 3x2pt 2d
    sl.postprocess_cov_dict(
        cov_dict=cov_rs_obj.cov_dict,
        obs_space='real',
        nbx=nbx,
        ind_auto=ind_auto,
        ind_cross=ind_cross,
        zpairs_auto=zpairs_auto,
        zpairs_cross=zpairs_cross,
        block_index=block_index,
        cov_ordering_2d=cov_ordering_2d,
        req_probe_combs_2d=req_probe_combs_2d,
    )

    print(f'...done in {time.perf_counter() - start_rs:.2f} s')

# TODO this code block is almost identical to the real-space one above, probably
# TODO can be unified
if obs_space == 'cosebis' and 'Spaceborne' in cov_terms_and_codes.values():
    print('\nComputing COSEBIs covariance...')
    start_rs = time.perf_counter()

    # TODO understand a bit better how to treat real-space SSC and cNG
    for _probe, _term in itertools.product(
        unique_probe_combs_cs, cov_cs_obj.terms_toloop
    ):
        probe_ab, probe_cd = sl.split_probe_name(_probe, space='cosebis')
        print(
            f'\nCOSEBIs cov: computing probe combination {probe_ab, probe_cd}'
            f' - term {_term.upper()}'
        )

        # in case the NG terms are required, pass the corresponding dictionaries.
        # note that, since cov_hs_obj._add_cov_ng was not called, these dictionaries
        # only contain the 4d keys for the moment. Because of this, I project the 4d
        # covs directly and reshape them to 6d inside compute_cs_cov_term_probe_6d.
        cov_hs_ng_dict = {}
        if cfg['covariance']['SSC']:
            # ORIGINAL
            cov_hs_ng_dict['ssc'] = cov_ssc_obj.cov_dict['ssc']
            # plt.semilogy(np.diag(cov_hs_ng_dict['ssc']['LL', 'LL']['4d'][:, :, 0, 0]))

            # # TEMP
            # covs_6d = np.load(
            #     '/data/sciotti/DATA/Spaceborne_jobs/OC_HS_COV_TMP/covmat_6D.npz'
            # )
            # for file in covs_6d.files:
            #     probe_abcd = file.split('_')[0]
            #     probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            #     term = file.split('_')[1].lower()
            #     if term == 'ssc':
            #         cov_ssc_oc = covs_6d[file]
            #         _nbl = cov_ssc_oc.shape[0]
            #         cov_ssc_oc = sl.cov_6D_to_4D(
            #             cov_ssc_oc, _nbl, zpairs_auto, ind_auto
            #         )
            #         cov_hs_ng_dict['ssc'][probe_ab, probe_cd]['4d'] = cov_ssc_oc

            # plt.semilogy(np.diag(cov_hs_ng_dict['ssc']['LL', 'LL']['4d'][:, :, 0, 0]))

        if cfg['covariance']['cNG']:
            # ORIGINAL
            cov_hs_ng_dict['cng'] = ccl_obj.cov_dict['cng']

            # TEMP
            # covs_6d = np.load(
            #     '/data/sciotti/DATA/Spaceborne_jobs/OC_HS_COV_TMP/covmat_6D.npz'
            # )
            # for file in covs_6d.files:
            #     probe_abcd = file.split('_')[0]
            #     probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            #     term = file.split('_')[1].lower()
            #     if term == 'cng':
            #         cov_cng_oc = covs_6d[file]
            #         _nbl = cov_cng_oc.shape[0]
            #         cov_cng_oc = sl.cov_6D_to_4D(
            #             cov_cng_oc, _nbl, zpairs_auto, ind_auto
            #         )
            #         cov_hs_ng_dict['cng'][probe_ab, probe_cd]['4d'] = cov_cng_oc

        cov_cs_obj.compute_cs_cov_term_probe_6d(
            cov_hs_ng_dict=cov_hs_ng_dict, probe_abcd=_probe, term=_term
        )

    for term in cov_cs_obj.terms_toloop:
        sl.fill_remaining_probe_blocks_6d(
            cov_dict=cov_cs_obj.cov_dict,
            term=term,
            symm_probe_combs=symm_probe_combs_cs,
            nonreq_probe_combs=nonreq_probe_combs_cs,
            space='cosebis',
            nbx=nbx,
            zbins=zbins,
        )

    # sum sva, sn and mix to get the Gaussian term
    sl.sum_split_g_terms_allprobeblocks_alldims(cov_cs_obj.cov_dict)

    # postprocess: reshape to 4d and 2d, build 3x2pt 2d
    sl.postprocess_cov_dict(
        cov_dict=cov_cs_obj.cov_dict,
        obs_space='cosebis',
        nbx=nbx,
        ind_auto=ind_auto,
        ind_cross=ind_cross,
        zpairs_auto=zpairs_auto,
        zpairs_cross=zpairs_cross,
        block_index=block_index,
        cov_ordering_2d=cov_ordering_2d,
        req_probe_combs_2d=req_probe_combs_2d,
    )

    print(f'...done in {time.perf_counter() - start_rs:.2f} s')

# ! ====================================================================================
# ! =============================== END OF FUN PART ====================================
# ! ====================================================================================

if obs_space == 'harmonic':
    _cov_dict = cov_hs_obj.cov_dict
    _probes = unique_probe_combs_hs
elif obs_space == 'real':
    _cov_dict = cov_rs_obj.cov_dict
    _probes = unique_probe_combs_rs
elif obs_space == 'cosebis':
    _cov_dict = cov_cs_obj.cov_dict
    _probes = unique_probe_combs_cs
else:
    raise ValueError(
        f'Unknown cfg["probe_selection"]["space"]: {cfg["probe_selection"]["space"]}'
    )

# in the harmonic case, this is handled by the cov_harmonic_space class
# Note that I need to copy arrays at leaf level
if obs_space != 'harmonic':
    if cfg['covariance']['G_code'] == 'OneCovariance':
        for probe_pair in _cov_dict['g']:
            for dim in _cov_dict['g'][probe_pair]:
                _cov_dict['g'][probe_pair][dim] = cov_oc_obj.cov_dict['g'][probe_pair][
                    dim
                ]

        # Copy split Gaussian terms if they exist
        if cfg['covariance']['split_gaussian_cov']:
            for term in ['sva', 'sn', 'mix']:
                for probe_pair in _cov_dict[term]:
                    for dim in _cov_dict[term][probe_pair]:
                        _cov_dict[term][probe_pair][dim] = cov_oc_obj.cov_dict[term][
                            probe_pair
                        ][dim]

    if cfg['covariance']['SSC_code'] == 'OneCovariance':
        for probe_pair in _cov_dict['ssc']:
            for dim in _cov_dict['ssc'][probe_pair]:
                _cov_dict['ssc'][probe_pair][dim] = cov_oc_obj.cov_dict['ssc'][
                    probe_pair
                ][dim]

    if cfg['covariance']['cNG_code'] == 'OneCovariance':
        for probe_pair in _cov_dict['cng']:
            for dim in _cov_dict['cng'][probe_pair]:
                _cov_dict['cng'][probe_pair][dim] = cov_oc_obj.cov_dict['cng'][
                    probe_pair
                ][dim]


# ! important note: for OC RS, list fmt seems to be missing some blocks (problem common
# ! to HS, solve it)
# ! moreover, some of the sub-blocks are transposed.
if cfg['OneCovariance']['compare_against_oc']:
    if 'OneCovariance' in cov_terms_and_codes.values():
        warnings.warn('You are likely comparing OneCovariance against itself')

    # THIS CHECK FAILS FOR REAL SPACE (I think it's a OneCov issue)
    for term in _cov_dict:
        if obs_space != 'real':
            # consistency check: list and mat formats should coincide
            if term not in ['sva', 'sn', 'mix']:
                np.testing.assert_allclose(
                    cov_oc_obj.cov_dict_matfmt[term]['3x2pt']['2d'],
                    cov_oc_obj.cov_dict[term]['3x2pt']['2d'],
                    atol=0,
                    rtol=1e-4,
                )
            # for good measure, check that the sum of the split Gaussian terms
            # coincides with G
            np.testing.assert_allclose(
                cov_oc_obj.cov_dict_matfmt['g']['3x2pt']['2d'],
                cov_oc_obj.cov_dict['sva']['3x2pt']['2d']
                + cov_oc_obj.cov_dict['sn']['3x2pt']['2d']
                + cov_oc_obj.cov_dict['mix']['3x2pt']['2d'],
                atol=0,
                rtol=1e-4,
            )

        cov_a = _cov_dict[term]['3x2pt']['2d']
        cov_b = cov_oc_obj.cov_dict[term]['3x2pt']['2d']

        sl.compare_2d_covs(
            cov_a,
            cov_b,
            'SB',
            'OC',
            f'cov {term} {obs_space} space - ',
            diff_threshold=10,
            compare_cov_2d=True,
            compare_corr_2d=False,
            compare_diag=True,
            compare_flat=True,
            compare_spectrum=True,
        )

        # compare G against mat fmt of OC. For Cosebis this is not done, since the covariance
        # is not "full" (no Psi* covariance blocks)
        # if term not in ['sva', 'sn', 'mix']:
        #     cov_a = _cov_dict[term]['3x2pt']['2d']
        #     cov_b = cov_oc_obj.cov_dict_matfmt[term]['3x2pt']['2d']

        #     sl.compare_2d_covs(
        #         cov_a,
        #         cov_b,
        #         'SB',
        #         'OC mat fmt',
        #         f'cov {term} {obs_space} space nbl {ell_obj.nbl_3x2pt} -',
        #         diff_threshold=10,
        #         compare_cov_2d=True,
        #         compare_corr_2d=False,
        #         compare_diag=True,
        #         compare_flat=True,
        #         compare_spectrum=True,
        #     )


# ! save 2D covs (for each term) in npz archive
covs_3x2pt_2d_tosave_dict = {}
if cfg['covariance']['G']:
    covs_3x2pt_2d_tosave_dict['Gauss'] = _cov_dict['g']['3x2pt']['2d']
if cfg['covariance']['SSC']:
    covs_3x2pt_2d_tosave_dict['SSC'] = _cov_dict['ssc']['3x2pt']['2d']
if cfg['covariance']['cNG']:
    covs_3x2pt_2d_tosave_dict['cNG'] = _cov_dict['cng']['3x2pt']['2d']
if cfg['covariance']['split_gaussian_cov']:
    covs_3x2pt_2d_tosave_dict['SVA'] = _cov_dict['sva']['3x2pt']['2d']
    covs_3x2pt_2d_tosave_dict['SN'] = _cov_dict['sn']['3x2pt']['2d']
    covs_3x2pt_2d_tosave_dict['MIX'] = _cov_dict['mix']['3x2pt']['2d']
# the total covariance is equal to the Gaussian one if neither SSC nor cNG are computed,
# so only save it if at least one of the two is computed
if cfg['covariance']['cNG'] or cfg['covariance']['SSC']:
    covs_3x2pt_2d_tosave_dict['TOT'] = _cov_dict['tot']['3x2pt']['2d']

cov_filename = cfg['covariance']['cov_filename']
np.savez_compressed(f'{output_path}/{cov_filename}_2D.npz', **covs_3x2pt_2d_tosave_dict)

# ! save 6D covs (for each probe and term) in npz archive.
# ! note that the 6D covs are always probe-specific,
# ! i.e. there is no cov_3x2pt_{term}_6d
if cfg['covariance']['save_full_cov']:
    covs_6d_tosave_dict = {}
    _cd = _cov_dict  # just to make the code more readable
    for _probe in _probes:
        probe_ab, probe_cd = sl.split_probe_name(_probe, obs_space)
        probe_2tpl = (probe_ab, probe_cd)  # just to make the code more readable
        if cfg['covariance']['G']:
            covs_6d_tosave_dict[f'{_probe}_Gauss'] = _cd['g'][probe_2tpl]['6d']
        if cfg['covariance']['SSC']:
            covs_6d_tosave_dict[f'{_probe}_SSC'] = _cd['ssc'][probe_2tpl]['6d']
        if cfg['covariance']['cNG']:
            covs_6d_tosave_dict[f'{_probe}_cNG'] = _cd['cng'][probe_2tpl]['6d']
        if cfg['covariance']['split_gaussian_cov']:
            covs_6d_tosave_dict[f'{_probe}_SVA'] = _cd['sva'][probe_2tpl]['6d']
            covs_6d_tosave_dict[f'{_probe}_SN'] = _cd['sn'][probe_2tpl]['6d']
            covs_6d_tosave_dict[f'{_probe}_MIX'] = _cd['mix'][probe_2tpl]['6d']
        if cfg['covariance']['cNG'] or cfg['covariance']['SSC']:
            covs_6d_tosave_dict[f'{_probe}_TOT'] = _cd['tot'][probe_2tpl]['6d']

    np.savez_compressed(f'{output_path}/{cov_filename}_6D.npz', **covs_6d_tosave_dict)

if cfg['covariance']['save_cov_fits'] and obs_space == 'harmonic':
    io_obj.save_cov_euclidlib(cov_dict=_cov_dict)
if cfg['covariance']['save_cov_fits'] and obs_space != 'harmonic':
    raise ValueError(
        'Official Euclid .fits format is only supported for harmonic space '
        'for the moment'
    )

print(f'\nCovariance matrices saved in {output_path}\n')


# ! ============================ plot & tests ==========================================
with np.errstate(invalid='ignore', divide='ignore'):
    for cov_name, cov in covs_3x2pt_2d_tosave_dict.items():
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
                elif obs_space == 'real':
                    unique_probe_combs = unique_probe_combs_rs
                    diag_probe_combs = const.RS_DIAG_PROBE_COMBS
                    latex_labels = const.RS_PROBE_NAME_TO_LATEX
                elif obs_space == 'cosebis':
                    unique_probe_combs = unique_probe_combs_cs
                    diag_probe_combs = const.CS_DIAG_PROBE_COMBS
                    latex_labels = const.CS_PROBE_NAME_TO_LATEX

                # this is to get the names and order of the *required* probes
                # along the diagonel
                req_diag_probes = list(set(unique_probe_combs) & set(diag_probe_combs))
                req_diag_probes = [p for p in diag_probe_combs if p in req_diag_probes]

                # set the boundaries
                elem_auto = zpairs_auto * nbx
                elem_cross = zpairs_cross * nbx

                lim_dict = {
                    'LL': elem_auto,
                    'GL': elem_cross,
                    'GG': elem_auto,
                    'xip': elem_auto,
                    'xim': elem_auto,
                    'gt': elem_cross,
                    'w': elem_auto,
                    'En': elem_auto,
                    'Bn': elem_auto,
                }

                # draw the boundaries
                start_ab, start_cd = 0, 0
                for probe_abcd in req_diag_probes[:-1]:
                    probe_ab, probe_cd = sl.split_probe_name(
                        probe_abcd, space=obs_space
                    )

                    kw = {'color': 'k', 'alpha': 0.7, 'ls': '--'}
                    ax[0].axvline(start_ab + lim_dict[probe_ab] - 0.5, **kw)
                    ax[0].axhline(start_ab + lim_dict[probe_ab] - 0.5, **kw)
                    ax[0].axvline(start_cd + lim_dict[probe_cd] - 0.5, **kw)
                    ax[0].axhline(start_cd + lim_dict[probe_cd] - 0.5, **kw)
                    ax[1].axvline(start_ab + lim_dict[probe_ab] - 0.5, **kw)
                    ax[1].axhline(start_ab + lim_dict[probe_ab] - 0.5, **kw)
                    ax[1].axvline(start_cd + lim_dict[probe_cd] - 0.5, **kw)
                    ax[1].axhline(start_cd + lim_dict[probe_cd] - 0.5, **kw)

                    start_ab += lim_dict[probe_ab]
                    start_cd += lim_dict[probe_cd]

                xticks, xlabels = [], []
                yticks, ylabels = [], []

                start_ab, start_cd = 0, 0
                for probe_abcd in req_diag_probes:
                    probe_ab, probe_cd = sl.split_probe_name(
                        probe_abcd, space=obs_space
                    )

                    # x direction
                    center_ab = start_ab + (lim_dict[probe_ab] - 1) / 2
                    xticks.append(center_ab)
                    xlabels.append(latex_labels[probe_ab])

                    # y direction
                    center_cd = start_cd + (lim_dict[probe_cd] - 1) / 2
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
# TODO do this for theta values in the real space case
header_list = ['ell', 'delta_ell', 'ell_lower_edges', 'ell_upper_edges']

# ells_ref, probably no need to save
# ells_2d_save = np.column_stack((
#     ell_ref_nbl32,
#     delta_l_ref_nbl32,
#     ell_edges_ref_nbl32[:-1],
#     ell_edges_ref_nbl32[1:],
# ))
# sl.savetxt_aligned(f'{output_path}/ell_values_ref.txt', ells_2d_save, header_list)

# for probe in ['WL', 'GC', '3x2pt']:
for probe in ['3x2pt']:
    ells_2d_save = np.column_stack(
        (
            getattr(ell_obj, f'ells_{probe}'),
            getattr(ell_obj, f'delta_l_{probe}'),
            getattr(ell_obj, f'ell_edges_{probe}')[:-1],
            getattr(ell_obj, f'ell_edges_{probe}')[1:],
        )
    )
    sl.savetxt_aligned(f'{output_path}/ell_values.txt', ells_2d_save, header_list)

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

    if compute_sb_ssc and cfg['precision']['use_KE_approximation']:
        # in this case, the k grid used is the same as the Pk one, I think
        k_grid_s2b = np.array([])

    if compute_sb_ssc:
        sigma2_b = cov_ssc_obj.sigma2_b

    _bnt_matrix = np.array([]) if bnt_matrix is None else bnt_matrix
    _mag_bias_2d = (
        ccl_obj.mag_bias_2d if cfg['C_ell']['has_magnification_bias'] else np.array([])
    )

    _ell_dict = {k: v for k, v in vars(ell_obj).items() if isinstance(v, np.ndarray)}

    if cfg['covariance']['partial_sky_method'] == 'NaMaster':
        import pymaster

        # convert NmtBin objects to effective ells
        for key, val in _ell_dict.items():
            if key.startswith('nmt_bin_obj_'):
                assert isinstance(val, pymaster.bins.NmtBin), (
                    f'Expected NmtBin for {key}, got {val}'
                )
                _ell_dict[key] = val.get_effective_ells()

    # other stuff to save
    misc_dict = {}

    # COSEBIs W_n kernels
    # TODO bookmark check this
    if obs_space == 'cosebis' and cov_cs_obj is not None:
        if hasattr(cov_cs_obj, 'w_ells'):
            misc_dict['cosebis_w_ells'] = cov_cs_obj.w_ells
        if hasattr(cov_cs_obj, 'ells_for_w'):
            misc_dict['cosebis_ells_for_w'] = cov_cs_obj.ells_for_w

    # Mask information
    if mask_obj is not None:
        misc_dict['mask'] = mask_obj.mask
        misc_dict['mask_ell'] = mask_obj.ell_mask
        misc_dict['mask_cl'] = mask_obj.cl_mask
        misc_dict['mask_fsky'] = np.array([mask_obj.fsky])
        misc_dict['mask_survey_area_deg2'] = np.array([mask_obj.survey_area_deg2])

    # save metadata
    import datetime

    branch, commit = sl.get_git_info()
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'branch': branch,
        'commit': commit,
    }

    bench_filename = cfg['misc']['bench_filename']

    # protect against unwanted oversubscription
    if os.path.exists(f'{bench_filename}.npz'):
        raise ValueError(
            'You are trying to overwrite the benchmark file at'
            f' {bench_filename}.npz.'
            'Please rename the new benchmark or delete the existing one.'
        )

    with open(f'{bench_filename}.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # ! Save all the values in cov_*_obj.cov_dict
    covs_totest_dict = {}
    for _cov_obj, _cov_obj_name in zip(
        [cov_hs_obj, cov_rs_obj, cov_cs_obj, cov_oc_obj, cov_nmt_obj],
        ['cov_hs_obj', 'cov_rs_obj', 'cov_cs_obj', 'cov_oc_obj', 'cov_nmt_obj'],
        strict=True,
    ):
        if _cov_obj is None:
            continue

        # optional: save every array contained in _cov_obj (slight overkill)
        # (this will potentially save more than needed)
        # covs_arrays_dict.update(
        #     {k: v for k, v in vars(_cov_obj).items() if isinstance(v, np.ndarray)}
        # )

        if not hasattr(_cov_obj, 'cov_dict'):
            continue

        # save all covariance arrays in _cov_obj.cov_dict
        for term in _cov_obj.cov_dict:
            for probe_2tpl in _cov_obj.cov_dict[term]:
                for dim in _cov_obj.cov_dict[term][probe_2tpl]:
                    cov = _cov_obj.cov_dict[term][probe_2tpl][dim]
                    if cov is None:
                        continue

                    probe_abcd = ''.join(probe_2tpl)
                    # a certain covariance term might be present in multiple
                    # covariance objects
                    key_name = f'{_cov_obj_name}_cov_{term}_{probe_abcd}_{dim}'
                    covs_totest_dict[key_name] = _cov_obj.cov_dict[term][probe_2tpl][
                        dim
                    ]

    np.savez_compressed(
        bench_filename,
        ind=ind,
        backup_cfg=cfg,
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
        **covs_totest_dict,
        **misc_dict,
        metadata=metadata,
    )

if (
    cfg['misc']['test_condition_number']
    or cfg['misc']['test_cholesky_decomposition']
    or cfg['misc']['test_numpy_inversion']
    or cfg['misc']['test_symmetry']
):
    key = (
        'TOT'
        if 'SSC' in covs_3x2pt_2d_tosave_dict or 'cNG' in covs_3x2pt_2d_tosave_dict
        else 'Gauss'
    )
    cov = covs_3x2pt_2d_tosave_dict[key]

    print(
        f'Performing sanity checks on cov {key}.\n'
        'This can take some time for large matrices. '
        'Please note that your files have already been saved.\n'
    )

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
    output_path_figs = f'{output_path}/figs'
    os.makedirs(output_path_figs, exist_ok=True)
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig.savefig(
            os.path.join(output_path_figs, f'fig_{i:03d}.pdf'),
            bbox_inches='tight',
            pad_inches=0.1,
        )
    print(f'Figures saved in {output_path_figs}\n')


print(f'Finished in {(time.perf_counter() - script_start_time) / 60:.2f} minutes')

# UNCOMMENT TO MONITOR CPU COUNT USAGE
# Stop monitoring
stop_event.set()
monitor_thread.join()

# Save and plot
df = pd.DataFrame(cpu_data)
# df.to_csv('cpu_usage.csv', index=False)

import matplotlib.pyplot as plt

plt.figure()
df['time_elapsed'] = df['time'] - df['time'].min()
plt.plot(df['time_elapsed'], df['cores_used'])
plt.xlabel('Time (s)')
plt.ylabel('Number of Active Cores')
plt.title('CPU Core Usage Over Time')
plt.show()
