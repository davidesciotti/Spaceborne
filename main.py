import argparse
import os
import multiprocessing
import sys

import matplotlib
num_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['NUMBA_NUM_THREADS'] = '32'
os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '4'

# jupyter stuff:
# os.chdir('/home/cosmo/davide.sciotti/data/Spaceborne/')  # for new interactive window

from functools import partial
from collections import OrderedDict
import numpy as np
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
import gc
import yaml
import pprint
from copy import deepcopy
import numpy.testing as npt
from scipy.interpolate import interp1d, RegularGridInterpolator
# from tabulate import tabulate

import spaceborne.ell_utils as ell_utils
import spaceborne.cl_preprocessing as cl_utils
import spaceborne.compute_Sijkl as Sijkl_utils
import spaceborne.covariance as covmat_utils
import spaceborne.fisher_matrix as fm_utils
import spaceborne.my_module as mm
import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.wf_cl_lib as wf_cl_lib
import spaceborne.pyccl_interface as pyccl_interface
import spaceborne.plot_lib as plot_lib
import spaceborne.sigma2_SSC as sigma2_SSC
import spaceborne.config_checker as config_checker
import spaceborne.onecovariance_interface as oc_interface
import spaceborne.responses as responses

pp = pprint.PrettyPrinter(indent=4)
ROOT = os.getenv('ROOT')
SB_ROOT = ROOT + '/Spaceborne'
script_start_time = time.perf_counter()


def SSC_integral_julia(d2CLL_dVddeltab, d2CGL_dVddeltab, d2CGG_dVddeltab,
                       ind_auto, ind_cross, cl_integral_prefactor, sigma2, z_grid, integration_type, num_threads=16):
    """Kernel to compute the 4D integral optimized using Simpson's rule using Julia."""

    suffix = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, 'tmp')
    unique_folder_name = folder_name

    # Loop until we find a folder name that does not exist
    while os.path.exists(unique_folder_name):
        suffix += 1
        unique_folder_name = f'{folder_name}{suffix}'
    os.makedirs(unique_folder_name)
    folder_name = unique_folder_name

    np.save(f"{folder_name}/d2CLL_dVddeltab", d2CLL_dVddeltab)
    np.save(f"{folder_name}/d2CGL_dVddeltab", d2CGL_dVddeltab)
    np.save(f"{folder_name}/d2CGG_dVddeltab", d2CGG_dVddeltab)
    np.save(f"{folder_name}/ind_auto", ind_auto)
    np.save(f"{folder_name}/ind_cross", ind_cross)
    np.save(f"{folder_name}/cl_integral_prefactor", cl_integral_prefactor)
    np.save(f"{folder_name}/sigma2", sigma2)
    np.save(f"{folder_name}/z_grid", z_grid)
    os.system(
        f"julia --project=. --threads={num_threads} {SB_ROOT}/spaceborne/ssc_integral_julia.jl {folder_name} {integration_type}")

    cov_filename = "cov_SSC_spaceborne_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D.npy"

    if integration_type == 'trapz-6D':
        cov_ssc_3x2pt_dict_8D = {}  # it's 10D, actually
        for probe_a, probe_b in probe_ordering:
            for probe_c, probe_d in probe_ordering:
                if str.join('', (probe_a, probe_b, probe_c, probe_d)) not in ['GLLL', 'GGLL', 'GGGL']:
                    print(f"Loading {probe_a}{probe_b}{probe_c}{probe_d}")
                    _cov_filename = cov_filename.format(probe_a=probe_a, probe_b=probe_b,
                                                        probe_c=probe_c, probe_d=probe_d)
                    cov_ssc_3x2pt_dict_8D[(probe_a, probe_b, probe_c, probe_d)] = np.load(
                        f"{folder_name}/{_cov_filename}")

    else:
        cov_ssc_3x2pt_dict_8D = mm.load_cov_from_probe_blocks(
            path=f'{folder_name}',
            filename=cov_filename,
            probe_ordering=probe_ordering)

    os.system(f"rm -rf {folder_name}")
    return cov_ssc_3x2pt_dict_8D


# * ====================================================================================================================
# * ====================================================================================================================
# * ====================================================================================================================


# Set up argument parsing
# parser = argparse.ArgumentParser(description="Your script description here.")
# parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
# parser.add_argument('--show_plots', action='store_true', help='Show plots if specified')

# args = parser.parse_args()

# # # Load the configuration file
# with open(args.config, 'r') as f:
#     cfg = yaml.safe_load(f)
# # # this is for the test module
# with open(args.config, 'r') as f:
#     original_cfg = yaml.safe_load(f)

# if not args.show_plots:
#     matplotlib.use('Agg')

# ! uncomment this if executing from interactive window
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
# this is for the test module
with open('config.yaml', 'r') as f:
    original_cfg = yaml.safe_load(f)

# for zbins in (3, ):
#     for ep_or_ed in ('EP', ):

# add type/number-specific nuisance/hyperparameters
zbins = cfg['general_cfg']['zbins']
ep_or_ed = cfg['general_cfg']['EP_or_ED']
# do this instead if you loop
zbins = cfg['general_cfg']['zbins'] = zbins
ep_or_ed = cfg['general_cfg']['EP_or_ED'] = ep_or_ed

nuisance_folder = cfg['covariance_cfg']['nuisance_folder'].format(ROOT=ROOT)
nuisance_filename = cfg['covariance_cfg']['nuisance_filename'].format(zbins=cfg['general_cfg']['zbins'], ep_or_ed=cfg['general_cfg']['EP_or_ED'],
                                                                      zmin_nz_lens=cfg['general_cfg']['zmin_nz_lens'],
                                                                      zmax_nz=cfg['general_cfg']['zmax_nz'],
                                                                      magcut_lens=cfg['general_cfg']['magcut_lens'],
                                                                      )
nuisance_tab = np.genfromtxt(f'{nuisance_folder}/{nuisance_filename}')

zbin_centers = nuisance_tab[:, 0]
ngal_lensing = nuisance_tab[:, 1]
ngal_clustering = nuisance_tab[:, 1]
galaxy_bias_fit_fiducials = nuisance_tab[:, 2]

dzWL_fiducial = nuisance_tab[:, 4]
dzGC_fiducial = nuisance_tab[:, 4]

# insert in cfg dict:
cfg['covariance_cfg']['zbin_centers'] = zbin_centers
cfg['covariance_cfg']['ngal_lensing'] = ngal_lensing
cfg['covariance_cfg']['ngal_clustering'] = ngal_clustering

# insert the nuisance values in the ordered cfg section
# Convert FM_ordered_params to OrderedDict to maintain order
fm_ordered_params = OrderedDict(cfg['cosmology']['FM_ordered_params'])

# Find the index of 'eIA'
keys = list(fm_ordered_params.keys())
index_eIA = keys.index('eIA')

# Create a new OrderedDict to insert dzWL and dzGC after eIA
new_fm_ordered_params = OrderedDict()

for i, key in enumerate(keys):
    new_fm_ordered_params[key] = fm_ordered_params[key]
    if i == index_eIA:
        for zi in range(1, zbins + 1):
            new_fm_ordered_params[f'm{zi:02d}'] = 0

        for zi, dz in enumerate(dzWL_fiducial, start=1):
            new_fm_ordered_params[f'dzWL{zi:02d}'] = dz

cfg['cosmology']['FM_ordered_params'] = dict(new_fm_ordered_params)

# these are already in the cfg file, they do not change with zbins number/type
galaxy_bias_fit_fiducials = np.array(
    [cfg['cosmology']['FM_ordered_params'][f'bG{zi:02d}'] for zi in range(1, 5)])
magnification_bias_fit_fiducials = np.array(
    [cfg['cosmology']['FM_ordered_params'][f'bM{zi:02d}'] for zi in range(1, 5)])
# end: now the script can run as usual

general_cfg = cfg['general_cfg']
covariance_cfg = cfg['covariance_cfg']
fm_cfg = cfg['FM_cfg']
pyccl_cfg = covariance_cfg['PyCCL_cfg']

if 'logT' in cfg['cosmology']['FM_ordered_params']:
    assert cfg['cosmology']['FM_ordered_params']['logT'] == cfg['cosmology']['other_params']['camb_extra_parameters']['camb']['HMCode_logT_AGN'], (
        'Value mismatch for logT_AGN in the parameters definition')

if general_cfg['ell_cuts']:
    covariance_cfg['cov_filename'] = covariance_cfg['cov_filename'].replace('{ndim:d}D',
                                                                            '_kmaxhoverMpc{kmax_h_over_Mpc:.03f}_{ndim:d}D')

# some convenence variables, just to make things more readable
zbins = general_cfg['zbins']
ep_or_ed = general_cfg['EP_or_ED']
ell_max_WL = general_cfg['ell_max_WL']
ell_max_GC = general_cfg['ell_max_GC']
ell_max_3x2pt = general_cfg['ell_max_3x2pt']
center_or_min = general_cfg['center_or_min']
triu_tril = covariance_cfg['triu_tril']
row_col_major = covariance_cfg['row_col_major']
GL_or_LG = covariance_cfg['GL_or_LG']
n_probes = general_cfg['n_probes']
bnt_transform = general_cfg['BNT_transform']
shift_nz_interpolation_kind = covariance_cfg['shift_nz_interpolation_kind']
nz_gaussian_smoothing = covariance_cfg['nz_gaussian_smoothing']  # does not seem to have a large effect...
nz_gaussian_smoothing_sigma = covariance_cfg['nz_gaussian_smoothing_sigma']
shift_nz = covariance_cfg['shift_nz']  # ! are vincenzo's kernels shifted?? it looks like they are not
normalize_shifted_nz = covariance_cfg['normalize_shifted_nz']
# ! let's test this
compute_bnt_with_shifted_nz_for_zcuts = covariance_cfg['compute_bnt_with_shifted_nz_for_zcuts']
include_ia_in_bnt_kernel_for_zcuts = covariance_cfg['include_ia_in_bnt_kernel_for_zcuts']
nbl_WL_opt = general_cfg['nbl_WL_opt']
covariance_ordering_2D = covariance_cfg['covariance_ordering_2D']
magcut_lens = general_cfg['magcut_lens']
magcut_source = general_cfg['magcut_source']
clr = cm.rainbow(np.linspace(0, 1, zbins))
use_h_units = general_cfg['use_h_units']
covariance_cfg['probe_ordering'] = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))
probe_ordering = covariance_cfg['probe_ordering']
which_pk = general_cfg['which_pk']
general_cfg['fm_and_cov_suffix'] = general_cfg['fm_and_cov_suffix'].format(which_cls=general_cfg['which_cls'])
which_sigma2_b = covariance_cfg['which_sigma2_b']
z_steps_ssc_integrands = covariance_cfg['Spaceborne_cfg']['z_steps_ssc_integrands']

z_grid_ssc_integrands = np.linspace(covariance_cfg['Spaceborne_cfg']['z_min_ssc_integrands'],
                                    covariance_cfg['Spaceborne_cfg']['z_max_ssc_integrands'],
                                    covariance_cfg['Spaceborne_cfg']['z_steps_ssc_integrands'])
k_grid_resp = np.geomspace(covariance_cfg['PyCCL_cfg']['k_grid_tkka_min'],
                           covariance_cfg['PyCCL_cfg']['k_grid_tkka_max'],
                           covariance_cfg['PyCCL_cfg']['k_grid_tkka_steps_SSC'])

if len(z_grid_ssc_integrands) < 250:
    warnings.warn('z_grid_ssc_integrands is small, at the moment it used to compute various intermediate quantities')

# load some nuisance parameters
# note that zbin_centers is not exactly equal to the result of wf_cl_lib.get_z_mean...
zbin_centers = cfg['covariance_cfg']['zbin_centers']
ngal_lensing = cfg['covariance_cfg']['ngal_lensing']
ngal_clustering = cfg['covariance_cfg']['ngal_clustering']
galaxy_bias_fit_fiducials = np.array(
    [cfg['cosmology']['FM_ordered_params'][f'bG{zi:02d}'] for zi in range(1, 5)])
magnification_bias_fit_fiducials = np.array(
    [cfg['cosmology']['FM_ordered_params'][f'bM{zi:02d}'] for zi in range(1, 5)])
dzWL_fiducial = np.array([cfg['cosmology']['FM_ordered_params'][f'dzWL{zi:02d}'] for zi in range(1, zbins + 1)])
dzGC_fiducial = np.array([cfg['cosmology']['FM_ordered_params'][f'dzWL{zi:02d}'] for zi in range(1, zbins + 1)])
warnings.warn('dzGC_fiducial are equal to dzWL_fiducial')

which_ng_cov_suffix = 'G' + ''.join(covariance_cfg[covariance_cfg['ng_cov_code'] + '_cfg']['which_ng_cov'])
fid_pars_dict = cfg['cosmology']
flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)

h = flat_fid_pars_dict['h']

if general_cfg['use_h_units']:
    k_txt_label = "hoverMpc"
    pk_txt_label = "Mpcoverh3"
else:
    k_txt_label = "1overMpc"
    pk_txt_label = "Mpc3"

ccl_obj = pyccl_interface.PycclClass(fid_pars_dict)
a_default_grid_ccl = ccl_obj.cosmo_ccl.get_pk_spline_a()
z_default_grid_ccl = cosmo_lib.a_to_z(a_default_grid_ccl)[::-1]

# ! some checks
assert general_cfg['use_WA'] is False, 'We do not use Wadd for SPV3 at the moment'
assert general_cfg['which_cuts'] == 'Vincenzo', ('to begin with, use only Vincenzo/standard cuts. '
                                                 'For the thesis, probably use just these')
if general_cfg['ell_cuts']:
    assert bnt_transform, 'you should BNT transform if you want to apply ell cuts'
if covariance_cfg['cov_BNT_transform']:
    assert general_cfg['cl_BNT_transform'] is False, \
        'the BNT transform should be applied either to the Cls or to the covariance'
    assert fm_cfg['derivatives_BNT_transform'], 'you should BNT transform the derivatives as well'
assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
    'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'
assert general_cfg['which_forecast'] == 'SPV3', 'ISTF forecasts not supported at the moment'

if cfg['covariance_cfg']['Spaceborne_cfg']['use_KE_approximation'] and cfg['covariance_cfg']['ng_cov_code'] == 'Spaceborne':
    assert cfg['covariance_cfg']['Spaceborne_cfg']['cl_integral_convention'] == 'Euclid_KE_approximation'
    assert cfg['covariance_cfg']['Spaceborne_cfg']['integration_type'] == 'simps_KE_approximation'
    assert cfg['covariance_cfg']['which_sigma2_b'] not in [None, 'full_curved_sky'], \
        'to use the flat-sky sigma2_b, set "flat_sky" in the cfg file. Also, bear in mind that the flat-sky '\
        'approximation for sigma2_b is likely inappropriate for the large Euclid survey area'
elif not cfg['covariance_cfg']['Spaceborne_cfg']['use_KE_approximation'] and cfg['covariance_cfg']['ng_cov_code'] == 'Spaceborne':
    assert cfg['covariance_cfg']['Spaceborne_cfg']['cl_integral_convention'] in ('Euclid', 'PySSC')
    assert cfg['covariance_cfg']['Spaceborne_cfg']['integration_type'] in ('simps', 'trapz')
    assert cfg['covariance_cfg']['which_sigma2_b'] not in [None, 'flat_sky'], \
        'If you\'re not using the KE approximation, you should set "full_curved_sky", "from_input_mask or "polar_cap_on_the_fly"'

if general_cfg['is_CLOE_run']:
    assert covariance_cfg['survey_area_deg2'] == 13245, 'survey area must be 13245 deg2'
    assert covariance_cfg['PyCCL_cfg']['which_pk_for_pyccl'] == 'CLOE', 'pk should be from CLOE'
    assert covariance_cfg['which_shape_noise'] == 'per_component', 'which_shape_noise must be per_component'
    assert ell_max_WL == ell_max_3x2pt == 5000, 'all probes should be up to lmax=5000'
    assert general_cfg['which_pk'] == 'HMCodeBar', 'which_pk must be HMCodeBar'
    assert general_cfg['which_cls'] == 'CLOE', 'which_cls must be "CLOE"'
    assert general_cfg['flat_or_nonflat'] == 'Flat', 'Model must be flat'
    assert covariance_cfg['load_CLOE_benchmark_cov'] is False, 'load_CLOE_benchmark_cov must be False'
    # assert z_steps_ssc_integrands == 7000, \
    # 'for the actual run, I used z_steps_ssc_integrands == 7000'
    assert cfg['covariance_cfg']['OneCovariance_cfg']['precision_settings'] == 'high_precision'
    assert cfg['covariance_cfg']['OneCovariance_cfg']['use_OneCovariance_cNG'] is True, \
        'for the final run you should include the OC cNG term'
    assert covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == ['cNG',], \
        'which_ng_cov must be ["cNG"]'
    assert covariance_cfg['OneCovariance_cfg']['which_gauss_cov_binning'] == 'OneCovariance', \
        'which_gauss_cov_binning must be "OneCovariance" for the moment'
    assert zbins == 13, 'zbins must be 13'
    assert ep_or_ed == 'EP', 'ep_or_ed must be "EP"'

fsky_check = cosmo_lib.deg2_to_fsky(covariance_cfg['survey_area_deg2'])
assert np.abs(mm.percent_diff(covariance_cfg['fsky'], fsky_check)) < 1e-5, 'fsky does not match the survey area'

# TODO delete this arg in save_cov function
cases_tosave = '_'

# build the ind array and store it into the covariance dictionary
ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
covariance_cfg['ind'] = ind
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :].copy()
ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()
ind_dict = {('L', 'L'): ind_auto,
            ('G', 'L'): ind_cross,
            ('G', 'G'): ind_auto}
covariance_cfg['ind_dict'] = ind_dict

if not general_cfg['ell_cuts']:
    general_cfg['ell_cuts_subfolder'] = ''
    kmax_h_over_Mpc = general_cfg['kmax_h_over_Mpc_ref']
else:
    general_cfg['ell_cuts_subfolder'] = f'{general_cfg["which_cuts"]}/ell_{general_cfg["center_or_min"]}'

assert general_cfg['nbl_WL_opt'] == 32, 'this is used as the reference binning, from which the cuts are made'
assert general_cfg['ell_max_WL_opt'] == 5000, 'this is used as the reference binning, from which the cuts are made'
assert n_probes == 2, 'The code can only accept 2 probes at the moment'
cfg_check_obj = config_checker.SpaceborneConfigChecker(cfg)
cfg_check_obj.run_all_checks()

# ! 1. compute ell values, ell bins and delta ell
# compute ell and delta ell values in the reference (optimistic) case
ell_ref_nbl32, delta_l_ref_nbl32, ell_edges_ref_nbl32 = (
    ell_utils.compute_ells(general_cfg['nbl_WL_opt'], general_cfg['ell_min'], general_cfg['ell_max_WL_opt'],
                           recipe='ISTF', output_ell_bin_edges=True))

# perform the cuts (not the redshift-dependent ones!) on the ell centers and edges
ell_dict = {}
ell_dict['ell_WL'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_WL])
ell_dict['ell_GC'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_GC])
ell_dict['ell_3x2pt'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_3x2pt])
ell_dict['ell_WA'] = np.copy(ell_ref_nbl32[(ell_ref_nbl32 > ell_max_GC) & (ell_ref_nbl32 < ell_max_WL)])
ell_dict['ell_XC'] = np.copy(ell_dict['ell_3x2pt'])

# store edges *except last one for dimensional consistency* in the ell_dict
ell_dict['ell_edges_WL'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_WL])[:-1]
ell_dict['ell_edges_GC'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_GC])[:-1]
ell_dict['ell_edges_3x2pt'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_3x2pt])[:-1]
ell_dict['ell_edges_XC'] = np.copy(ell_dict['ell_edges_3x2pt'])
ell_dict['ell_edges_WA'] = np.copy(
    ell_edges_ref_nbl32[(ell_edges_ref_nbl32 > ell_max_GC) & (ell_edges_ref_nbl32 < ell_max_WL)])[:-1]

for key in ell_dict.keys():
    if ell_dict[key].size > 0:  # Check if the array is non-empty
        assert np.max(ell_dict[key]) > 15, f'ell values for key {key} must *not* be in log space'

# set the corresponding number of ell bins
nbl_WL = len(ell_dict['ell_WL'])
nbl_GC = len(ell_dict['ell_GC'])
nbl_WA = len(ell_dict['ell_WA'])
nbl_3x2pt = nbl_GC

assert len(ell_dict['ell_3x2pt']) == len(ell_dict['ell_XC']) == len(ell_dict['ell_GC']), '3x2pt, XC and GC should '\
    ' have the same number of ell bins'
assert np.all(ell_dict['ell_3x2pt'] == ell_dict['ell_XC']), '3x2pt and XC should have the same ell values'
assert np.all(ell_dict['ell_3x2pt'] == ell_dict['ell_GC']), '3x2pt and GC should have the same ell values'

# ! the main should not change the cfg...
general_cfg['nbl_WL'] = nbl_WL
general_cfg['nbl_GC'] = nbl_GC
general_cfg['nbl_3x2pt'] = nbl_3x2pt

assert nbl_WL == nbl_3x2pt == nbl_GC, 'use the same number of bins for the moment'

delta_dict = {'delta_l_WL': np.copy(delta_l_ref_nbl32[:nbl_WL]),
              'delta_l_GC': np.copy(delta_l_ref_nbl32[:nbl_GC]),
              'delta_l_WA': np.copy(delta_l_ref_nbl32[nbl_GC:nbl_WL])}

# this is just to make the .format() more compact
variable_specs = {'EP_or_ED': ep_or_ed,
                  'ep_or_ed': ep_or_ed,
                  'zbins': zbins,
                  'ell_max_WL': ell_max_WL, 'ell_max_GC': ell_max_GC, 'ell_max_3x2pt': ell_max_3x2pt,
                  'nbl_WL': nbl_WL, 'nbl_GC': nbl_GC, 'nbl_WA': nbl_WA, 'nbl_3x2pt': nbl_3x2pt,
                  'kmax_h_over_Mpc': kmax_h_over_Mpc, 'center_or_min': center_or_min,
                  'BNT_transform': bnt_transform,
                  'which_ng_cov': which_ng_cov_suffix,
                  'ng_cov_code': covariance_cfg['ng_cov_code'],
                  'magcut_lens': magcut_lens,
                  'magcut_source': magcut_source,
                  'zmin_nz_lens': general_cfg['zmin_nz_lens'],
                  'zmin_nz_source': general_cfg['zmin_nz_source'],
                  'zmax_nz': general_cfg['zmax_nz'],
                  'which_pk': which_pk,
                  'flat_or_nonflat': general_cfg['flat_or_nonflat'],
                  'flagship_version': general_cfg['flagship_version'],
                  'idIA': general_cfg['idIA'],
                  'idM': general_cfg['idM'],
                  'idB': general_cfg['idB'],
                  'idR': general_cfg['idR'],
                  'idBM': general_cfg['idBM'],
                  }
pp.pprint(variable_specs)

# ! some check on the input nuisance values
# assert np.all(np.array(covariance_cfg['ngal_lensing']) <
# 9), 'ngal_lensing values are likely < 9 *per bin*; this is just a rough check'
# assert np.all(np.array(covariance_cfg['ngal_clustering']) <
# 9), 'ngal_clustering values are likely < 9 *per bin*; this is just a rough check'
assert np.all(np.array(covariance_cfg['ngal_lensing']) > 0), 'ngal_lensing values must be positive'
assert np.all(np.array(covariance_cfg['ngal_clustering']) > 0), 'ngal_clustering values must be positive'
assert np.all(np.array(zbin_centers) > 0), 'z_center values must be positive'
assert np.all(np.array(zbin_centers) < 3), 'z_center values are likely < 3; this is just a rough check'
assert np.all(dzWL_fiducial == dzGC_fiducial), 'dzWL and dzGC shifts do not match'
assert general_cfg['magcut_source'] == 245, 'magcut_source should be 245, only magcut lens is varied'

# ! import n(z)
# n_of_z_full: nz table including a column for the z values
# n_of_z:      nz table excluding a column for the z values
nofz_folder = covariance_cfg["nofz_folder"].format(ROOT=ROOT)
nofz_filename = covariance_cfg["nofz_filename"].format(**variable_specs)
n_of_z_full = np.genfromtxt(f'{nofz_folder}/{nofz_filename}')
assert n_of_z_full.shape[1] == zbins + \
    1, 'n_of_z must have zbins + 1 columns; the first one must be for the z values'

zgrid_nz = n_of_z_full[:, 0]
n_of_z = n_of_z_full[:, 1:]
n_of_z_original = n_of_z  # it may be subjected to a shift

# ! START SCALE CUTS: for these, we need to:
# 1. Compute the BNT. This is done with the raw, or unshifted n(z), but only for the purpose of computing the
#    ell cuts - the rest of the code uses a BNT matrix from the shifted n(z) - see also comment below.
# 2. compute the kernels for the un-shifted n(z) (for consistency)
# 3. bnt-transform these kernels (for lensing, it's only the gamma kernel), and use these to:
# 4. compute the z means
# 5. compute the ell cuts

# 1. Compute BNT
assert compute_bnt_with_shifted_nz_for_zcuts is False, 'The BNT used to compute the z_means and ell cuts is just for a simple case: no IA, no dz shift'
assert shift_nz is True, 'The signal (and BNT used to transform it) is computed with a shifted n(z); You could use an un-shifted n(z) for the BNT, but' \
    'this would be slightly inconsistent (but also what I did so far).'
assert include_ia_in_bnt_kernel_for_zcuts is False, 'We compute the BNT just for a simple case: no IA, no shift. This is because we want' \
                                                    ' to compute the z means'

# * IMPORTANT NOTE: The BNT should be computed from the same n(z) (shifted or not) which is then used to compute
# * the kernels which are then used to get the z_means, and finally the ell_cuts, for consistency. In other words,
# * we cannot compute the kernels with a shifted n(z) and transform them with a BNT computed from the unshifted n(z)
# * and viceversa. If the n(z) are shifted, one of the BNT kernels will become negative, but this is just because
# * two of the original kernels get very close after the shift: the transformation is correct.
# * Having said that, I leave the code below in case we want to change this in the future
if nz_gaussian_smoothing:
    n_of_z = wf_cl_lib.gaussian_smmothing_nz(zgrid_nz, n_of_z_original, nz_gaussian_smoothing_sigma, plot=True)
if compute_bnt_with_shifted_nz_for_zcuts:
    n_of_z = wf_cl_lib.shift_nz(zgrid_nz, n_of_z_original, dzWL_fiducial, normalize=normalize_shifted_nz, plot_nz=False,
                                interpolation_kind=shift_nz_interpolation_kind)

bnt_matrix = covmat_utils.compute_BNT_matrix(
    zbins, zgrid_nz, n_of_z, cosmo_ccl=ccl_obj.cosmo_ccl, plot_nz=False)

# 2. compute the kernels for the un-shifted n(z) (for consistency)
ccl_obj.zbins = zbins
ccl_obj.set_nz(np.hstack((zgrid_nz[:, None], n_of_z)))
ccl_obj.check_nz_tuple(zbins)
ccl_obj.set_ia_bias_tuple(z_grid=z_grid_ssc_integrands, has_ia=general_cfg['has_IA'])

# set galaxy bias
if general_cfg['which_forecast'] == 'SPV3':
    ccl_obj.set_gal_bias_tuple_spv3(z_grid=z_grid_ssc_integrands,
                                    magcut_lens=magcut_lens,
                                    poly_fit_values=None)

elif general_cfg['which_forecast'] == 'ISTF':
    bias_func_str = general_cfg['bias_function']
    bias_model = general_cfg['bias_model']
    ccl_obj.set_gal_bias_tuple_istf(z_grid=z_grid_ssc_integrands,
                                    bias_function_str=bias_func_str,
                                    bias_model=bias_model)

# set magnification bias
ccl_obj.set_mag_bias_tuple(z_grid=z_grid_ssc_integrands,
                           has_magnification_bias=general_cfg['has_magnification_bias'],
                           magcut_lens=magcut_lens / 10,
                           poly_fit_values=None)

# set pk
# this is a test to use the actual P(k) from the input files, but the agreement gets much worse
if general_cfg['which_forecast'] == 'SPV3' and pyccl_cfg['which_pk_for_pyccl'] == 'CLOE':

    cloe_pk_folder = general_cfg['CLOE_pk_folder'].format(
        ROOT=ROOT,
        which_pk=general_cfg['which_pk'],
        flat_or_nonflat=general_cfg['flat_or_nonflat'])

    cloe_pk_filename = general_cfg['CLOE_pk_filename'].format(
        CLOE_pk_folder=cloe_pk_folder,
        param_name='h',
        param_value=0.67
    )

    ccl_obj.p_of_k_a = ccl_obj.pk_obj_from_file(pk_filename=cloe_pk_filename, plot_pk_z0=False)
    # TODO finish implementing this
    warnings.warn('Extrapolating the P(k) in Tk3D_SSC!')
    # raise NotImplementedError('range needs to be extended to higher redshifts to match tkka grid (probably larger k range too), \
    # some other small consistency checks needed')

elif general_cfg['which_forecast'] == 'SPV3' and pyccl_cfg['which_pk_for_pyccl'] == 'PyCCL':
    ccl_obj.p_of_k_a = 'delta_matter:delta_matter'

elif general_cfg['which_forecast'] == 'ISTF':
    ccl_obj.p_of_k_a = 'delta_matter:delta_matter'

# set kernel arrays and objects
ccl_obj.set_kernel_obj(general_cfg['has_rsd'], covariance_cfg['PyCCL_cfg']['n_samples_wf'])
ccl_obj.set_kernel_arr(z_grid_wf=z_grid_ssc_integrands,
                       has_magnification_bias=general_cfg['has_magnification_bias'])

if general_cfg['which_forecast'] == 'SPV3':
    gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias!)'
    ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_wo_gal_bias_arr

if general_cfg['which_forecast'] == 'ISTF':
    gal_kernel_plt_title = 'galaxy kernel\n(w/ gal bias)'
    ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_w_gal_bias_arr

# 3. bnt-transform these kernels (for lensing, it's only the gamma kernel, without IA)
wf_gamma_ccl_bnt = (bnt_matrix @ ccl_obj.wf_gamma_arr.T).T

# 4. compute the z means
z_means_ll = wf_cl_lib.get_z_means(z_grid_ssc_integrands, ccl_obj.wf_gamma_arr)
z_means_gg = wf_cl_lib.get_z_means(z_grid_ssc_integrands, ccl_obj.wf_galaxy_arr)
z_means_ll_bnt = wf_cl_lib.get_z_means(z_grid_ssc_integrands, wf_gamma_ccl_bnt)

plt.figure()
for zi in range(zbins):
    plt.plot(z_grid_ssc_integrands, ccl_obj.wf_gamma_arr[:, zi], ls='-', c=clr[zi],
             alpha=0.6, label='wf_gamma_ccl' if zi == 0 else None)
    plt.plot(z_grid_ssc_integrands, wf_gamma_ccl_bnt[:, zi], ls='--', c=clr[zi],
             alpha=0.6, label='wf_gamma_ccl_bnt' if zi == 0 else None)
    plt.axvline(z_means_ll_bnt[zi], ls=':', c=clr[zi])
plt.legend()
plt.xlabel('$z$')
plt.ylabel(r'$W_i^{\gamma}(z)$')

# assert np.all(np.diff(z_means_ll) > 0), 'z_means_ll should be monotonically increasing'
# assert np.all(np.diff(z_means_gg) > 0), 'z_means_gg should be monotonically increasing'
# assert np.all(np.diff(z_means_ll_bnt) > 0), ('z_means_ll_bnt should be monotonically increasing '
#                                             '(not a strict condition, valid only if we do not shift the n(z) in this part)')

# 5. compute the ell cuts
ell_cuts_dict = {}
ell_cuts_dict['LL'] = ell_utils.load_ell_cuts(
    kmax_h_over_Mpc, z_means_ll_bnt, z_means_ll_bnt, ccl_obj.cosmo_ccl, zbins, h, general_cfg)
ell_cuts_dict['GG'] = ell_utils.load_ell_cuts(
    kmax_h_over_Mpc, z_means_gg, z_means_gg, ccl_obj.cosmo_ccl, zbins, h, general_cfg)
ell_cuts_dict['GL'] = ell_utils.load_ell_cuts(
    kmax_h_over_Mpc, z_means_gg, z_means_ll_bnt, ccl_obj.cosmo_ccl, zbins, h, general_cfg)
ell_cuts_dict['LG'] = ell_utils.load_ell_cuts(
    kmax_h_over_Mpc, z_means_ll_bnt, z_means_gg, ccl_obj.cosmo_ccl, zbins, h, general_cfg)
ell_dict['ell_cuts_dict'] = ell_cuts_dict  # this is to pass the ll cuts to the covariance module
# ! END SCALE CUTS

# now compute the BNT used for the rest of the code
if shift_nz:
    n_of_z = wf_cl_lib.shift_nz(zgrid_nz, n_of_z_original, dzWL_fiducial, normalize=normalize_shifted_nz, plot_nz=False,
                                interpolation_kind=shift_nz_interpolation_kind)
    nz_tuple = (zgrid_nz, n_of_z)
    # * this is important: the BNT matrix I use for the rest of the code (so not to compute the ell cuts) is instead
    # * consistent with the shifted n(z) used to compute the kernels
    bnt_matrix = covmat_utils.compute_BNT_matrix(
        zbins, zgrid_nz, n_of_z, cosmo_ccl=ccl_obj.cosmo_ccl, plot_nz=False)

# re-set n(z) used in CCL class, then re-compute kernels
ccl_obj.set_nz(np.hstack((zgrid_nz[:, None], n_of_z)))
ccl_obj.set_kernel_obj(general_cfg['has_rsd'], covariance_cfg['PyCCL_cfg']['n_samples_wf'])
ccl_obj.set_kernel_arr(z_grid_wf=z_grid_ssc_integrands,
                       has_magnification_bias=general_cfg['has_magnification_bias'])

if general_cfg['which_forecast'] == 'SPV3':
    gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias!)'
    ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_wo_gal_bias_arr

if general_cfg['which_forecast'] == 'ISTF':
    gal_kernel_plt_title = 'galaxy kernel\n(w/ gal bias)'
    ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_w_gal_bias_arr

if general_cfg['check_wf_against_vincenzo']:

    # import
    vincenzo_wf_folder = general_cfg['wf_folder'].format(ROOT=ROOT)
    specs = general_cfg['specs'].format(**variable_specs)
    wf_filename = general_cfg['wf_filename'].format(probe='{probe:s}', specs=specs, **variable_specs)

    wf_delta_vin = np.genfromtxt(f'{vincenzo_wf_folder}/{wf_filename.format(probe="delta")}')
    wf_gamma_vin = np.genfromtxt(f'{vincenzo_wf_folder}/{wf_filename.format(probe="gamma")}')
    wf_ia_vin = np.genfromtxt(f'{vincenzo_wf_folder}/{wf_filename.format(probe="ia")}')
    wf_mu_vin = np.genfromtxt(f'{vincenzo_wf_folder}/{wf_filename.format(probe="mu")}')

    z_grid_kernels_vin = wf_delta_vin[:, 0]
    wf_delta_vin = wf_delta_vin[:, 1:]
    wf_gamma_vin = wf_gamma_vin[:, 1:]
    wf_ia_vin = wf_ia_vin[:, 1:]
    wf_mu_vin = wf_mu_vin[:, 1:]

    # construct lensing kernel
    ia_bias = wf_cl_lib.build_ia_bias_1d_arr(z_grid_out=z_grid_kernels_vin, cosmo_ccl=ccl_obj.cosmo_ccl,
                                             flat_fid_pars_dict=flat_fid_pars_dict,
                                             input_z_grid_lumin_ratio=None, input_lumin_ratio=None,
                                             output_F_IA_of_z=False)

    wf_lensing_vin = wf_gamma_vin + ia_bias[:, None] * wf_ia_vin
    wf_galaxy_vin = wf_delta_vin + wf_mu_vin

    # interpolate
    wf_names_list = ['delta', 'gamma', 'ia', 'mu', 'lensing', gal_kernel_plt_title]
    wf_ccl_list = [ccl_obj.wf_delta_arr, ccl_obj.wf_gamma_arr, ccl_obj.wf_ia_arr, ccl_obj.wf_mu_arr,
                   ccl_obj.wf_lensing_arr, ccl_obj.wf_galaxy_arr]
    wf_interp_list = []
    diff_list = []
    for idx, wf_arr in enumerate([wf_delta_vin, wf_gamma_vin, wf_ia_vin, wf_mu_vin, wf_lensing_vin, wf_galaxy_vin]):
        wf_func = interp1d(z_grid_kernels_vin, wf_arr, axis=0, kind='linear')
        wf_interp_list.append(wf_func(z_grid_ssc_integrands))
        diff_list.append(mm.percent_diff(wf_ccl_list[idx], wf_interp_list[idx]))

    # plot
    for wf_idx in range(len(wf_interp_list)):
        clr = cm.rainbow(np.linspace(0, 1, zbins))
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5), height_ratios=[2, 1])
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        for zi in range(zbins):
            ax[0].plot(z_grid_ssc_integrands, wf_interp_list[wf_idx][:, zi], ls="-", c=clr[zi], alpha=0.6)
            ax[0].plot(z_grid_ssc_integrands, wf_ccl_list[wf_idx][:, zi], ls=":", c=clr[zi], alpha=0.6)
            ax[1].plot(z_grid_ssc_integrands, diff_list[wf_idx][:, zi], c=clr[zi])

        ax[1].set_xlabel('$z$')
        ax[0].set_xlabel('wf')
        ax[1].set_ylabel('% diff')
        ax[1].set_ylim(-20, 20)
        lines = [plt.Line2D([], [], color='k', linestyle=ls) for ls in ['-', ':']]
        plt.legend(lines, ['Vincenzo', 'CCL'], loc='upper right')
        plt.suptitle(f'{wf_names_list[wf_idx]}')
        plt.show()
# ! end import vincenzo's kernels

# compute cls
ccl_obj.cl_ll_3d = ccl_obj.compute_cls(ell_dict['ell_WL'], ccl_obj.p_of_k_a,
                                       ccl_obj.wf_lensing_obj, ccl_obj.wf_lensing_obj, 'spline')
ccl_obj.cl_gl_3d = ccl_obj.compute_cls(ell_dict['ell_XC'], ccl_obj.p_of_k_a,
                                       ccl_obj.wf_galaxy_obj, ccl_obj.wf_lensing_obj, 'spline')
ccl_obj.cl_gg_3d = ccl_obj.compute_cls(ell_dict['ell_GC'], ccl_obj.p_of_k_a,
                                       ccl_obj.wf_galaxy_obj, ccl_obj.wf_galaxy_obj, 'spline')
ccl_obj.cl_wa_3d = ccl_obj.cl_ll_3d[nbl_3x2pt:nbl_WL]

ccl_obj.cl_3x2pt_5d = np.zeros((n_probes, n_probes, nbl_3x2pt, zbins, zbins))
ccl_obj.cl_3x2pt_5d[0, 0, :, :, :] = ccl_obj.cl_ll_3d[:nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[1, 0, :, :, :] = ccl_obj.cl_gl_3d[:nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[0, 1, :, :, :] = ccl_obj.cl_gl_3d[:nbl_3x2pt, :, :].transpose(0, 2, 1)
ccl_obj.cl_3x2pt_5d[1, 1, :, :, :] = ccl_obj.cl_gg_3d[:nbl_3x2pt, :, :]

if general_cfg['which_cls'] == 'Vincenzo':
    # import Vicnenzo's cls, as a quick check (no RSDs in GCph in my Cls!!)
    cl_folder = general_cfg['cl_folder'].format(which_pk=general_cfg['which_pk'], ROOT=ROOT, probe='{probe:s}')
    cl_filename = 'dv-{probe:s}-{EP_or_ED:s}{zbins:02d}-ML{magcut_lens:d}-MS{magcut_source:d}-idIA2-idB3-idM3-idR1.dat'
    cl_ll_1d = np.genfromtxt(
        f"{cl_folder.format(probe='WLO')}/{cl_filename.format(probe='WLO', **variable_specs)}")
    cl_gg_1d = np.genfromtxt(
        f"{cl_folder.format(probe='GCO')}/{cl_filename.format(probe='GCO', **variable_specs)}")
    cl_wa_1d = np.genfromtxt(
        f"{cl_folder.format(probe='WLA')}/{cl_filename.format(probe='WLA', **variable_specs)}")
    cl_3x2pt_1d = np.genfromtxt(
        f"{cl_folder.format(probe='3x2pt')}/{cl_filename.format(probe='3x2pt', **variable_specs)}")

    # ! reshape to 3d
    cl_ll_3d_vinc = cl_utils.cl_SPV3_1D_to_3D(cl_ll_1d, 'WL', nbl_WL_opt, zbins)[:nbl_WL, :, :]
    cl_gg_3d_vinc = cl_utils.cl_SPV3_1D_to_3D(cl_gg_1d, 'GC', nbl_GC, zbins)
    cl_wa_3d_vinc = np.ones_like(cl_ll_3d_vinc)[:nbl_WA, :, :]
    # cl_wa_3d_vinc = cl_utils.cl_SPV3_1D_to_3D(cl_wa_1d, 'WA', nbl_WA, zbins)
    cl_3x2pt_5d_vinc = cl_utils.cl_SPV3_1D_to_3D(cl_3x2pt_1d, '3x2pt', nbl_3x2pt, zbins)
    cl_gl_3d_vinc = deepcopy(cl_3x2pt_5d_vinc[1, 0, :, :, :])

    cl_ll_3d = cl_ll_3d_vinc
    cl_gg_3d = cl_gg_3d_vinc
    cl_wa_3d = cl_wa_3d_vinc
    cl_gl_3d = cl_gl_3d_vinc
    cl_3x2pt_5d = cl_3x2pt_5d_vinc

elif general_cfg['which_cls'] == 'CLOE':
    # import CLOE cls
    cloe_bench_path = general_cfg["CLOE_benchmarks_path"].format(ROOT=ROOT)
    cl_ll_2d = np.genfromtxt(f'{cloe_bench_path}/Cls_zNLA_ShearShear_C00.dat')
    cl_gl_2d = np.genfromtxt(f'{cloe_bench_path}/Cls_zNLA_PosShear_C00.dat')
    cl_gg_2d = np.genfromtxt(f'{cloe_bench_path}/Cls_zNLA_PosPos_C00.dat')

    # some checks on the ell values
    np.testing.assert_allclose(cl_ll_2d[:, 0], cl_gl_2d[:, 0], atol=0, rtol=1e-10)
    np.testing.assert_allclose(cl_ll_2d[:, 0], cl_gg_2d[:, 0], atol=0, rtol=1e-10)
    np.testing.assert_allclose(cl_ll_2d[:, 0], ell_ref_nbl32, atol=0, rtol=1e-10)
    np.testing.assert_allclose(cl_ll_2d[:, 0][:nbl_3x2pt], ell_dict['ell_3x2pt'], atol=0, rtol=1e-10)

    cl_ll_2d = cl_ll_2d[:, 1:]
    cl_gl_2d = cl_gl_2d[:, 1:]
    cl_gg_2d = cl_gg_2d[:, 1:]

    cl_ll_3d = mm.cl_2D_to_3D_symmetric(cl_ll_2d, nbl=nbl_WL_opt, zpairs=zpairs_auto, zbins=zbins)
    cl_gl_3d = mm.cl_2D_to_3D_asymmetric(cl_gl_2d, nbl=nbl_WL_opt, zbins=zbins, order='C')
    cl_gg_3d = mm.cl_2D_to_3D_symmetric(cl_gg_2d, nbl=nbl_WL_opt, zpairs=zpairs_auto, zbins=zbins)
    cl_wa_3d = cl_ll_3d[nbl_3x2pt:nbl_WL]

    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
    cl_gl_3d = cl_gl_3d[:nbl_3x2pt, :, :]
    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]

    cl_3x2pt_5d = np.zeros((n_probes, n_probes, nbl_3x2pt, zbins, zbins))
    cl_3x2pt_5d[0, 0, :, :, :] = cl_ll_3d[:nbl_3x2pt, :, :]
    cl_3x2pt_5d[1, 0, :, :, :] = cl_gl_3d[:nbl_3x2pt, :, :]
    cl_3x2pt_5d[0, 1, :, :, :] = cl_gl_3d[:nbl_3x2pt, :, :].transpose(0, 2, 1)
    cl_3x2pt_5d[1, 1, :, :, :] = cl_gg_3d[:nbl_3x2pt, :, :]

elif general_cfg['which_cls'] == 'CCL':
    cl_ll_3d, cl_gl_3d, cl_gg_3d, cl_wa_3d = ccl_obj.cl_ll_3d, ccl_obj.cl_gl_3d, ccl_obj.cl_gg_3d, ccl_obj.cl_wa_3d
    cl_3x2pt_5d = ccl_obj.cl_3x2pt_5d

else:
    raise ValueError(f"general_cfg['which_cls'] = must be in ['Vincenzo', 'CLOE', 'CCL']")

clr = cm.rainbow(np.linspace(0, 1, zbins))
fig, ax = plt.subplots(2, 3, sharex=True, figsize=(10, 5), height_ratios=[2, 1])
plt.tight_layout()
fig.subplots_adjust(hspace=0)

for zi in range(zbins):
    zj = zi
    ax[0, 0].loglog(ell_dict['ell_WL'], cl_ll_3d[:, zi, zj][:nbl_WL], ls="-", c=clr[zi], alpha=0.6)
    ax[0, 0].loglog(ell_dict['ell_WL'], ccl_obj.cl_ll_3d[:, zi, zj], ls=":", c=clr[zi], alpha=0.6)

    ax[0, 1].loglog(ell_dict['ell_XC'], cl_gl_3d[:, zi, zj][:nbl_3x2pt], ls="-", c=clr[zi], alpha=0.6)
    ax[0, 1].loglog(ell_dict['ell_XC'], ccl_obj.cl_gl_3d[:, zi, zj], ls=":", c=clr[zi], alpha=0.6)

    ax[0, 2].loglog(ell_dict['ell_GC'], cl_gg_3d[:, zi, zj][:nbl_GC], ls="-", c=clr[zi], alpha=0.6)
    ax[0, 2].loglog(ell_dict['ell_GC'], ccl_obj.cl_gg_3d[:, zi, zj], ls=":", c=clr[zi], alpha=0.6)

    ax[1, 0].plot(ell_dict['ell_WL'], mm.percent_diff(
        cl_ll_3d[:nbl_WL], ccl_obj.cl_ll_3d)[:, zi, zj], c=clr[zi])
    ax[1, 1].plot(ell_dict['ell_XC'], mm.percent_diff(
        cl_gl_3d[:nbl_3x2pt], ccl_obj.cl_gl_3d)[:, zi, zj], c=clr[zi])
    ax[1, 2].plot(ell_dict['ell_GC'], mm.percent_diff(
        cl_gg_3d[:nbl_GC], ccl_obj.cl_gg_3d)[:, zi, zj], c=clr[zi])

ax[1, 0].set_xlabel('$\\ell$')
ax[1, 1].set_xlabel('$\\ell$')
ax[1, 2].set_xlabel('$\\ell$')
ax[0, 0].set_ylabel('$C_{\\ell}$')
ax[1, 0].set_ylabel('% diff')
ax[1, 1].set_ylim(-20, 20)
lines = [plt.Line2D([], [], color='k', linestyle=ls) for ls in ['-', ':']]
plt.legend(lines, [general_cfg["which_cls"], 'CCL'], loc='upper right', bbox_to_anchor=(1.55, 1))
plt.show()

# matshow for GL, to make sure it's not LG
ell_idx = 10
mm.compare_arrays(cl_gl_3d[ell_idx, ...], ccl_obj.cl_gl_3d[ell_idx, ...], abs_val=True, log_array=True,
                  name_A=f'{general_cfg["which_cls"]} GL', name_B='CCL GL')


# ! ========================================== OneCovariance ===================================================
# * 1. save ingredients in ascii format
oc_path = covariance_cfg['OneCovariance_cfg']['onecovariance_folder'].format(ROOT=ROOT, **variable_specs)

if not os.path.exists(oc_path):
    os.makedirs(oc_path)

nofz_ascii_filename = nofz_filename.replace('.dat', f'_dzshifts{shift_nz}.ascii')
nofz_tosave = np.column_stack((zgrid_nz, n_of_z))
np.savetxt(f'{oc_path}/{nofz_ascii_filename}', nofz_tosave)

cl_ll_ascii_filename = f'Cell_ll_SPV3_{general_cfg["which_cls"]}_nbl{nbl_3x2pt}'
cl_gl_ascii_filename = f'Cell_gl_SPV3_{general_cfg["which_cls"]}_nbl{nbl_3x2pt}'
cl_gg_ascii_filename = f'Cell_gg_SPV3_{general_cfg["which_cls"]}_nbl{nbl_3x2pt}'
mm.write_cl_ascii(oc_path, cl_ll_ascii_filename, cl_3x2pt_5d[0, 0, ...], ell_dict['ell_3x2pt'], zbins)
mm.write_cl_ascii(oc_path, cl_gl_ascii_filename, cl_3x2pt_5d[1, 0, ...], ell_dict['ell_3x2pt'], zbins)
mm.write_cl_ascii(oc_path, cl_gg_ascii_filename, cl_3x2pt_5d[1, 1, ...], ell_dict['ell_3x2pt'], zbins)

gal_bias_ascii_filename = f'{oc_path}/gal_bias_table_{general_cfg["which_forecast"]}.ascii'
ccl_obj.save_gal_bias_table_ascii(z_grid_ssc_integrands, gal_bias_ascii_filename)

ascii_filenames_dict = {
    'cl_ll_ascii_filename': cl_ll_ascii_filename,
    'cl_gl_ascii_filename': cl_gl_ascii_filename,
    'cl_gg_ascii_filename': cl_gg_ascii_filename,
    'gal_bias_ascii_filename': gal_bias_ascii_filename,
    'nofz_ascii_filename': nofz_ascii_filename,
}

# * 2. compute cov using the onecovariance interface class
start_time = time.perf_counter()
if (covariance_cfg['ng_cov_code'] == 'OneCovariance') or \
        ((covariance_cfg['ng_cov_code'] == 'Spaceborne') and
            covariance_cfg['OneCovariance_cfg']['use_OneCovariance_cNG']):

    print('Start NG cov computation with OneCovariance...')

    # TODO this should be defined globally...
    symmetrize_output_dict = {
        ('L', 'L'): False,
        ('G', 'L'): False,
        ('L', 'G'): False,
        ('G', 'G'): False,
    }

    oc_obj = oc_interface.OneCovarianceInterface(ROOT, cfg, variable_specs)
    oc_obj.ells_sb = ell_dict['ell_3x2pt']
    oc_obj.build_save_oc_ini(ascii_filenames_dict, print_ini=True)

    if not covariance_cfg['OneCovariance_cfg']['load_precomputed_cov']:
        oc_obj.call_onecovariance()
        oc_obj.reshape_oc_output(variable_specs, ind_dict, symmetrize_output_dict)

    oc_obj.cov_g_oc_3x2pt_10D = oc_obj.oc_output_to_dict_or_array(
        'G', '10D_array', ind_dict, symmetrize_output_dict)
    oc_obj.cov_ssc_oc_3x2pt_10D = oc_obj.oc_output_to_dict_or_array(
        'SSC', '10D_array', ind_dict, symmetrize_output_dict)
    oc_obj.cov_cng_oc_3x2pt_10D = oc_obj.oc_output_to_dict_or_array(
        'cNG', '10D_array', ind_dict, symmetrize_output_dict)

    print('Time taken to compute OC: {:.2f} m'.format((time.perf_counter() - start_time) / 60))

else:
    oc_obj = None
# ! ========================================== end OneCovariance ===================================================

# ! ========================================== start Spaceborne ===================================================
_which_pk_resp = covariance_cfg['Spaceborne_cfg']['which_pk_responses']
_which_pk_resp = 'separate_universe' if _which_pk_resp.startswith('separate_universe') else _which_pk_resp
_which_pk_resp = 'halo_model' if _which_pk_resp.startswith('halo_model') else _which_pk_resp
cov_folder_sb = covariance_cfg['Spaceborne_cfg']['cov_path'].format(ROOT=ROOT,
                                                                    which_pk_responses=_which_pk_resp,
                                                                    flagship_version=general_cfg['flagship_version'],
                                                                    cov_ell_cuts=str(
                                                                        covariance_cfg['cov_ell_cuts']),
                                                                    BNT_transform=str(general_cfg['BNT_transform']))

cov_sb_suffix = covariance_cfg['Spaceborne_cfg']['cov_suffix'].format(
    z_steps_ssc_integrands=z_steps_ssc_integrands,
    k_txt_label=k_txt_label,
    cl_integral_convention=covariance_cfg['Spaceborne_cfg']['cl_integral_convention'],
    integration_type=covariance_cfg['Spaceborne_cfg']['integration_type'],
    survey_area_deg2=covariance_cfg['survey_area_deg2'],
)

variable_specs.pop('ng_cov_code')
variable_specs.pop('which_ng_cov')
cov_sb_filename = covariance_cfg['cov_filename'].format(ng_cov_code='spaceborne',
                                                        lmax_3x2pt=general_cfg['ell_max_3x2pt'],
                                                        probe='{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}',
                                                        cov_suffix=cov_sb_suffix,
                                                        which_ng_cov=which_ng_cov_suffix.replace('G', ''),
                                                        fm_and_cov_suffix=general_cfg['fm_and_cov_suffix'],
                                                        ndim=4,
                                                        **variable_specs)

variable_specs['ng_cov_code'] = covariance_cfg['ng_cov_code']
variable_specs['which_ng_cov'] = which_ng_cov_suffix

if 'cNG' in covariance_cfg['Spaceborne_cfg']['which_ng_cov']:
    raise NotImplementedError('You should review the which_ng_cov arg in the cov_filename formatting above, "SSC" is'
                              'hardcoded at the moment')

# precompute pk_mm, pk_gm and pk_mm, if you want to rescale the responses
k_array, pk_mm_2d = cosmo_lib.pk_from_ccl(k_grid_resp, z_grid_ssc_integrands, use_h_units,
                                          ccl_obj.cosmo_ccl, pk_kind='nonlinear')

# compute P_gm, P_gg
gal_bias = ccl_obj.gal_bias_2d[:, 0]

# check that it's the same in each bin
for zi in range(zbins):
    np.testing.assert_allclose(ccl_obj.gal_bias_2d[:, 0], ccl_obj.gal_bias_2d[:, zi], atol=0, rtol=1e-5)

pk_gm_2d = pk_mm_2d * gal_bias
pk_gg_2d = pk_mm_2d * gal_bias ** 2

if covariance_cfg['ng_cov_code'] == 'Spaceborne' and not covariance_cfg['Spaceborne_cfg']['load_precomputed_cov']:
    print('Start SSC computation with Spaceborne...')

    # ! 1. Get halo model responses from CCL
    if covariance_cfg['Spaceborne_cfg']['which_pk_responses'] == 'halo_model':

        ccl_obj.initialize_trispectrum(which_ng_cov='SSC', probe_ordering=probe_ordering,
                                       pyccl_cfg=pyccl_cfg)

        # k and z grids (responses will be interpolated below)
        k_grid_resp_hm = ccl_obj.responses_dict['L', 'L', 'L', 'L']['k_1overMpc']
        a_grid_resp_hm = ccl_obj.responses_dict['L', 'L', 'L', 'L']['a_arr']
        # translate a to z and cut the arrays to the maximum redshift of the SU responses (much smaller range!)
        z_grid_resp_hm = cosmo_lib.a_to_z(a_grid_resp_hm)[::-1]

        assert np.allclose(k_grid_resp_hm, k_grid_resp, atol=0, rtol=1e-2), 'CCL and SB k_grids for responses '\
            'should match'

        dPmm_ddeltab_hm = ccl_obj.responses_dict['L', 'L', 'L', 'L']['dpk12']
        dPgm_ddeltab_hm = ccl_obj.responses_dict['L', 'L', 'G', 'L']['dpk34']
        dPgg_ddeltab_hm = ccl_obj.responses_dict['G', 'G', 'G', 'G']['dpk12']

        # a is flipped w.r.t. z
        dPmm_ddeltab_hm = np.flip(dPmm_ddeltab_hm, axis=1)
        dPgm_ddeltab_hm = np.flip(dPgm_ddeltab_hm, axis=1)
        dPgg_ddeltab_hm = np.flip(dPgg_ddeltab_hm, axis=1)

        # quick sanity check
        assert np.allclose(ccl_obj.responses_dict['L', 'L', 'G', 'L']['dpk34'],
                           ccl_obj.responses_dict['G', 'L', 'G', 'G']['dpk12'], atol=0, rtol=1e-5)
        assert np.allclose(ccl_obj.responses_dict['L', 'L', 'L', 'L']['dpk34'],
                           ccl_obj.responses_dict['L', 'L', 'L', 'L']['dpk12'], atol=0, rtol=1e-5)
        assert dPmm_ddeltab_hm.shape == dPgm_ddeltab_hm.shape == dPgg_ddeltab_hm.shape, 'dPab_ddeltab_hm shape mismatch'

        dPmm_ddeltab_hm_func = interp1d(x=z_grid_resp_hm, y=dPmm_ddeltab_hm, axis=1, kind='linear')
        dPgm_ddeltab_hm_func = interp1d(x=z_grid_resp_hm, y=dPgm_ddeltab_hm, axis=1, kind='linear')
        dPgg_ddeltab_hm_func = interp1d(x=z_grid_resp_hm, y=dPgg_ddeltab_hm, axis=1, kind='linear')

        # I do not assign diretly to dPxx_ddeltab to be able to plot later if necessary
        dPmm_ddeltab_hm = dPmm_ddeltab_hm_func(z_grid_ssc_integrands)
        dPgm_ddeltab_hm = dPgm_ddeltab_hm_func(z_grid_ssc_integrands)
        dPgg_ddeltab_hm = dPgg_ddeltab_hm_func(z_grid_ssc_integrands)
        r_mm_hm = dPmm_ddeltab_hm / pk_mm_2d
        r_gm_hm = dPgm_ddeltab_hm / pk_gm_2d
        r_gg_hm = dPgg_ddeltab_hm / pk_gg_2d

        dPmm_ddeltab = dPmm_ddeltab_hm
        dPgm_ddeltab = dPgm_ddeltab_hm
        dPgg_ddeltab = dPgg_ddeltab_hm

    # ! from Vincenzo's files
    elif covariance_cfg['Spaceborne_cfg']['which_pk_responses'] == 'separate_universe_vin':

        # import the response *coefficients* (not the responses themselves)
        su_responses_folder = covariance_cfg['Spaceborne_cfg']['separate_universe_responses_folder'].format(
            which_pk=general_cfg['which_pk'], ROOT=ROOT)
        su_responses_filename = covariance_cfg['Spaceborne_cfg']['separate_universe_responses_filename'].format(
            idBM=general_cfg['idBM'])
        rAB_of_k = np.genfromtxt(f'{su_responses_folder}/{su_responses_filename}')

        log_k_arr = np.unique(rAB_of_k[:, 0])
        k_grid_resp_vin = 10 ** log_k_arr
        z_grid_resp_vin = np.unique(rAB_of_k[:, 1])

        r_mm_vin = np.reshape(rAB_of_k[:, 2], (len(k_grid_resp_vin), len(z_grid_resp_vin)))
        r_gm_vin = np.reshape(rAB_of_k[:, 3], (len(k_grid_resp_vin), len(z_grid_resp_vin)))
        r_gg_vin = np.reshape(rAB_of_k[:, 4], (len(k_grid_resp_vin), len(z_grid_resp_vin)))

        # remove z=0 and z = 0.01
        z_grid_resp_vin = z_grid_resp_vin[2:]
        r_mm_vin = r_mm_vin[:, 2:]
        r_gm_vin = r_gm_vin[:, 2:]
        r_gg_vin = r_gg_vin[:, 2:]

        r_mm_vin_func = RegularGridInterpolator(
            (k_grid_resp_vin, z_grid_resp_vin), r_mm_vin, method='linear')
        r_gm_vin_func = RegularGridInterpolator(
            (k_grid_resp_vin, z_grid_resp_vin), r_gm_vin, method='linear')
        r_gg_vin_func = RegularGridInterpolator(
            (k_grid_resp_vin, z_grid_resp_vin), r_gg_vin, method='linear')

        k_grid_resp_xx, z_grid_ssc_integrands_yy = np.meshgrid(k_grid_resp, z_grid_ssc_integrands, indexing='ij')
        r_mm_vin = r_mm_vin_func((k_grid_resp_xx, z_grid_ssc_integrands_yy))
        r_gm_vin = r_gm_vin_func((k_grid_resp_xx, z_grid_ssc_integrands_yy))
        r_gg_vin = r_gg_vin_func((k_grid_resp_xx, z_grid_ssc_integrands_yy))

        # now turn the response coefficients into responses
        dPmm_ddeltab_vin = r_mm_vin * pk_mm_2d
        dPgm_ddeltab_vin = r_gm_vin * pk_gm_2d
        dPgg_ddeltab_vin = r_gg_vin * pk_gg_2d

        dPmm_ddeltab = dPmm_ddeltab_vin
        dPgm_ddeltab = dPgm_ddeltab_vin
        dPgg_ddeltab = dPgg_ddeltab_vin

    # ! from SpaceborneResponses class
    elif covariance_cfg['Spaceborne_cfg']['which_pk_responses'] == 'separate_universe_dav':

        resp_obj = responses.SpaceborneResponses(cfg=cfg, k_grid=k_grid_resp,
                                                 z_grid=z_grid_ssc_integrands,
                                                 ccl_obj=ccl_obj)
        r_mm_sbclass = resp_obj.compute_r1_mm()
        resp_obj.get_rab_and_dpab_ddeltab(b2g_from_halomodel=True)

        if covariance_cfg['Spaceborne_cfg']['include_b2g']:
            r_gm_sbclass = resp_obj.r1_gm
            r_gg_sbclass = resp_obj.r1_gg
        else:
            r_gm_sbclass = resp_obj.r1_gm_nob2
            r_gg_sbclass = resp_obj.r1_gg_nob2

        dPmm_ddeltab = resp_obj.dPmm_ddeltab
        dPgm_ddeltab = resp_obj.dPgm_ddeltab
        dPgg_ddeltab = resp_obj.dPgg_ddeltab

        dPmm_ddeltab = resp_obj.dPmm_ddeltab
        dPgm_ddeltab = resp_obj.dPgm_ddeltab
        dPgg_ddeltab = resp_obj.dPgg_ddeltab

        b1g_hm = resp_obj.b1g_hm
        b2g_hm = resp_obj.b2g_hm

        # ! plot for a comparison (you need to remove the 2 above "ifs")
        # z_idx = 0
        # k_idx = 0
        # plt.semilogx(k_grid_resp, r_mm_sbclass[:, z_idx], label=f'r_mm_sbclass includeb2{
        #              include_b2g}', c='tab:blue', ls='-.')
        # plt.semilogx(k_grid_resp, r_mm_hm[:, z_idx], label='r_mm_hm', c='tab:blue', ls='-')
        # plt.semilogx(k_grid_resp, r_mm_vin[:, z_idx], label='r_mm vin', c='tab:blue', ls='--')

        # plt.semilogx(k_grid_resp, r_gm_sbclass[:, z_idx], c='tab:orange', ls='-.')
        # plt.semilogx(k_grid_resp, r_gm_hm[:, z_idx], c='tab:orange', ls='-')
        # plt.semilogx(k_grid_resp, r_gm_vin[:, z_idx], c='tab:orange', ls='--')

        # plt.semilogx(k_grid_resp, r_gg_sbclass[:, z_idx], c='tab:green', ls='-.')
        # plt.semilogx(k_grid_resp, r_gg_hm[:, z_idx], c='tab:green', ls='-')
        # plt.semilogx(k_grid_resp, r_gg_vin[:, z_idx], c='tab:green', ls='--')

        # plt.legend()
        # plt.xlabel(f'k {k_txt_label}')
        # plt.ylabel(r'$R_{AB}(k)$')
        # plt.ylim(-5, 5)
        # plt.title(f'z={z_grid_ssc_integrands[z_idx]}')

        # # TODO check counterterms, to be better understood - 0 for lensing, as they should be
        # bA12 = ccl_obj.responses_dict['G', 'G', 'G', 'G']['bA12']
        # bB12 = ccl_obj.responses_dict['G', 'G', 'G', 'G']['bB12']
        # bA34 = ccl_obj.responses_dict['G', 'G', 'G', 'G']['bA34']
        # bB34 = ccl_obj.responses_dict['G', 'G', 'G', 'G']['bB34']

        # # a is flipped w.r.t. z
        # bA12 = np.flip(bA12, axis=1)
        # bB12 = np.flip(bB12, axis=1)
        # bA34 = np.flip(bA34, axis=1)
        # bB34 = np.flip(bB34, axis=1)

        # assert bA12.shape == bA34.shape == bB12.shape == bB34.shape, 'counterterms shape mismatch'

        # k_grid_resp_hm = ccl_obj.responses_dict['L', 'L', 'L', 'L']['k_1overMpc']
        # a_grid_resp_hm = ccl_obj.responses_dict['L', 'L', 'L', 'L']['a_arr']
        # z_grid_resp_hm = cosmo_lib.a_to_z(a_grid_resp_hm)[::-1]

        # plt.plot(z_grid_resp_hm, bA12[0, :], label='bA12', alpha=0.6, ls='--')
        # plt.plot(z_grid_resp_hm, bB12[0, :], label='bB12', alpha=0.6, ls='--')
        # plt.plot(z_grid_resp_hm, bA34[0, :], label='bA34', alpha=0.6, ls='--')
        # plt.plot(z_grid_resp_hm, bB34[0, :], label='bB34', alpha=0.6, ls='--')
        # plt.plot(z_grid_resp, gal_bias, label='FS2_gal_bias')
        # plt.legend()

    else:
        raise ValueError(
            'which_pk_responses must be either "halo_model" or "separate_universe_vin" or "separate_universe_dav"')

    # ! 2. prepare integrands (d2CAB_dVddeltab) and volume element
    k_limber = partial(cosmo_lib.k_limber, cosmo_ccl=ccl_obj.cosmo_ccl, use_h_units=use_h_units)
    r_of_z_func = partial(cosmo_lib.ccl_comoving_distance, use_h_units=use_h_units, cosmo_ccl=ccl_obj.cosmo_ccl)

    # ! divide by r(z)**2 if cl_integral_convention == 'PySSC'
    if covariance_cfg['Spaceborne_cfg']['cl_integral_convention'] == 'PySSC':
        r_of_z_square = r_of_z_func(z_grid_ssc_integrands) ** 2

        wf_delta = ccl_obj.wf_delta_arr / r_of_z_square[:, None]
        wf_gamma = ccl_obj.wf_gamma_arr / r_of_z_square[:, None]
        wf_ia = ccl_obj.wf_ia_arr / r_of_z_square[:, None]
        wf_mu = ccl_obj.wf_mu_arr / r_of_z_square[:, None]
        wf_lensing = ccl_obj.wf_lensing_arr / r_of_z_square[:, None]

    elif covariance_cfg['Spaceborne_cfg']['cl_integral_convention'] in ('Euclid', 'Euclid_KE_approximation'):
        wf_delta = ccl_obj.wf_delta_arr
        wf_gamma = ccl_obj.wf_gamma_arr
        wf_ia = ccl_obj.wf_ia_arr
        wf_mu = ccl_obj.wf_mu_arr
        wf_lensing = ccl_obj.wf_lensing_arr

    else:
        raise ValueError('cl_integral_convention must be either "PySSC" or "Euclid" or "Euclid_KE_approximation')

    # ! compute the Pk responses(k, z) in k_limber and z_grid_ssc_integrands
    dPmm_ddeltab_interp = RegularGridInterpolator((k_grid_resp, z_grid_ssc_integrands), dPmm_ddeltab, method='linear')
    dPgm_ddeltab_interp = RegularGridInterpolator((k_grid_resp, z_grid_ssc_integrands), dPgm_ddeltab, method='linear')
    dPgg_ddeltab_interp = RegularGridInterpolator((k_grid_resp, z_grid_ssc_integrands), dPgg_ddeltab, method='linear')

    # ! test k_max_limber vs k_max_dPk and adjust z_min_ssc_integrands accordingly
    k_max_resp = np.max(k_grid_resp)
    ell_grid = ell_dict['ell_3x2pt']
    kmax_limber = cosmo_lib.get_kmax_limber(ell_grid, z_grid_ssc_integrands, use_h_units, ccl_obj.cosmo_ccl)

    z_grid_ssc_integrands_test = deepcopy(z_grid_ssc_integrands)
    while kmax_limber > k_max_resp:
        print(f'kmax_limber > k_max_dPk ({kmax_limber:.2f} {k_txt_label} > {k_max_resp:.2f} {k_txt_label}): '
              f'Increasing z_min until kmax_limber < k_max_dPk. Alternetively, increase k_max_dPk or decrease ell_max.')
        z_grid_ssc_integrands_test = z_grid_ssc_integrands_test[1:]
        kmax_limber = cosmo_lib.get_kmax_limber(
            ell_grid, z_grid_ssc_integrands_test, use_h_units, ccl_obj.cosmo_ccl)
        print(f'New z_min = {z_grid_ssc_integrands_test[0]:.3f}')

    dPmm_ddeltab_klimb = np.array(
        [dPmm_ddeltab_interp((k_limber(ell_val, z_grid_ssc_integrands), z_grid_ssc_integrands)) for ell_val in
            ell_dict['ell_WL']])
    dPgm_ddeltab_klimb = np.array(
        [dPgm_ddeltab_interp((k_limber(ell_val, z_grid_ssc_integrands), z_grid_ssc_integrands)) for ell_val in
            ell_dict['ell_XC']])
    dPgg_ddeltab_klimb = np.array(
        [dPgg_ddeltab_interp((k_limber(ell_val, z_grid_ssc_integrands), z_grid_ssc_integrands)) for ell_val in
            ell_dict['ell_GC']])

    # ! volume element
    cl_integral_prefactor = cosmo_lib.cl_integral_prefactor(z_grid_ssc_integrands,
                                                            covariance_cfg['Spaceborne_cfg']['cl_integral_convention'],
                                                            use_h_units=use_h_units,
                                                            cosmo_ccl=ccl_obj.cosmo_ccl)

    # ! observable densities
    d2CLL_dVddeltab = np.einsum('zi,zj,Lz->Lijz', wf_lensing, wf_lensing, dPmm_ddeltab_klimb)
    d2CGL_dVddeltab = \
        np.einsum('zi,zj,Lz->Lijz', wf_delta, wf_lensing, dPgm_ddeltab_klimb) + \
        np.einsum('zi,zj,Lz->Lijz', wf_mu, wf_lensing, dPmm_ddeltab_klimb)
    d2CGG_dVddeltab = \
        np.einsum('zi,zj,Lz->Lijz', wf_delta, wf_delta, dPgg_ddeltab_klimb) + \
        np.einsum('zi,zj,Lz->Lijz', wf_delta, wf_mu, dPgm_ddeltab_klimb) + \
        np.einsum('zi,zj,Lz->Lijz', wf_mu, wf_delta, dPgm_ddeltab_klimb) + \
        np.einsum('zi,zj,Lz->Lijz', wf_mu, wf_mu, dPmm_ddeltab_klimb)

    # ! 3. Compute/load/save sigma2_b
    k_grid_sigma2 = np.logspace(covariance_cfg['Spaceborne_cfg']['log10_k_min_sigma2'],
                                covariance_cfg['Spaceborne_cfg']['log10_k_max_sigma2'],
                                covariance_cfg['Spaceborne_cfg']['k_steps_sigma2'])

    # TODO find best way to handle these conditions, too much nesting
    sigma2_b_filename = covariance_cfg['Spaceborne_cfg']['sigma2_b_filename'].format(
        ROOT=ROOT,
        which_sigma2_b=str(which_sigma2_b),
        zmin=covariance_cfg['Spaceborne_cfg']['z_min_ssc_integrands'],
        zmax=covariance_cfg['Spaceborne_cfg']['z_max_ssc_integrands'],
        zsteps=z_steps_ssc_integrands,
        log10kmin=covariance_cfg['Spaceborne_cfg']['log10_k_min_sigma2'],
        log10kmax=covariance_cfg['Spaceborne_cfg']['log10_k_max_sigma2'],
        ksteps=covariance_cfg['Spaceborne_cfg']['k_steps_sigma2']
    )
    if covariance_cfg['Spaceborne_cfg']['use_KE_approximation']:

        # compute sigma2_b(z) (1 dimension) using the existing CCL implementation
        ccl_obj.set_sigma2_b(z_grid=z_grid_ssc_integrands,
                             fsky=covariance_cfg['fsky'],
                             which_sigma2_b=which_sigma2_b,
                             nside_mask=covariance_cfg['nside_mask'],
                             mask_path=covariance_cfg['mask_path'])

        _a, sigma2_b = ccl_obj.sigma2_b_tuple
        sigma2_b = sigma2_b[::-1]
        _z = cosmo_lib.a_to_z(_a)[::-1]

        if covariance_cfg['Spaceborne_cfg']['load_precomputed_sigma2']:
            raise NotImplementedError('TODO')

    else:

        if covariance_cfg['Spaceborne_cfg']['load_precomputed_sigma2']:
            # TODO define a suitable interpolator if the zgrid doesn't match
            sigma2_b_dict = np.load(sigma2_b_filename, allow_pickle=True).item()
            cfg_sigma2_b = sigma2_b_dict['cfg']  # TODO check that the cfg matches the one
            sigma2_b = sigma2_b_dict['sigma2_b']
        else:
            print('Computing sigma2_b...')
            sigma2_b = sigma2_SSC.sigma2_z1z2_wrap(
                z_grid_ssc_integrands=z_grid_ssc_integrands,
                k_grid_sigma2=k_grid_sigma2,
                cosmo_ccl=ccl_obj.cosmo_ccl,
                which_sigma2_b=which_sigma2_b,
                area_deg2_in=covariance_cfg['survey_area_deg2'],
                nside_mask=covariance_cfg['nside_mask'],
                mask_path=covariance_cfg['mask_path']
            )

            # Note: if you want to compare sigma2 with full_curved_sky against polar_cap_on_the_fly, remember to decrease
            # the former by fsky (eq. 29 of https://arxiv.org/pdf/1612.05958)
            sigma2_b_dict_tosave = {
                'cfg': cfg,
                'sigma2_b': sigma2_b,
            }
            np.save(sigma2_b_filename, sigma2_b_dict_tosave, allow_pickle=True)

    # ! 4. Perform the integration calling the Julia module
    print('Performing the SSC integral with Julia...')
    start = time.perf_counter()
    cov_ssc_3x2pt_dict_8D = SSC_integral_julia(d2CLL_dVddeltab=d2CLL_dVddeltab,
                                               d2CGL_dVddeltab=d2CGL_dVddeltab,
                                               d2CGG_dVddeltab=d2CGG_dVddeltab,
                                               ind_auto=ind_auto, ind_cross=ind_cross,
                                               cl_integral_prefactor=cl_integral_prefactor, sigma2=sigma2_b,
                                               z_grid=z_grid_ssc_integrands,
                                               integration_type=covariance_cfg['Spaceborne_cfg']['integration_type'],
                                               num_threads=general_cfg['num_threads'])
    print('SSC computed with Julia in {:.2f} s'.format(time.perf_counter() - start))

    # in the full_curved_sky case only, sigma2_b has to be divided by fsky
    if which_sigma2_b == 'full_curved_sky':
        for key in cov_ssc_3x2pt_dict_8D.keys():
            cov_ssc_3x2pt_dict_8D[key] /= covariance_cfg['fsky']
    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask', 'flat_sky']:
        pass
    else:
        raise ValueError(f'which_sigma2_b = {which_sigma2_b} not recognized')

    # save the covariance blocks
    # ! note that these files already account for the sky fraction!!
    # TODO fsky suffix in cov name should be added only in this case... or not? the other covariance files don't have this...
    for key in cov_ssc_3x2pt_dict_8D.keys():
        probe_a, probe_b, probe_c, probe_d = key
        if str.join('', (probe_a, probe_b, probe_c, probe_d)) not in ['GLLL', 'GGLL', 'GGGL']:
            _cov_sb_filename = cov_sb_filename.format(probe_a=probe_a, probe_b=probe_b,
                                                      probe_c=probe_c, probe_d=probe_d)
            _filename = f'{cov_folder_sb}/{_cov_sb_filename}'
            np.savez_compressed(_filename, cov_ssc_3x2pt_dict_8D[key])

    # ! check SSC INTEGRANDS

    # plt.figure()
    # z1_idx = 3
    # ell1_idx, ell2_idx = 28, 28
    # zpair_AB, zpair_CD = 5, 5
    # num_col = 4
    # ind_AB, ind_CD = ind_auto, ind_auto
    # zi, zj, zk, zl = ind_AB[zpair_AB, num_col - 2], ind_AB[zpair_AB, num_col - 1], \
    #     ind_CD[zpair_CD, num_col - 2], ind_CD[zpair_CD, num_col - 1]
    # # plt.plot(z_grid_ssc_integrands, integrand_ssc_spaceborne_py_LLLL[ell1_idx, ell2_idx,zpair_AB, zpair_CD, z1_idx, :], label='total integrand')
    # plt.plot(z_grid_ssc_integrands, d2CLL_dVddeltab[ell2_idx, zk, zl, :], label='d2CLL_dVddeltab')
    # plt.plot(z_grid_ssc_integrands, cl_integral_prefactor, label='cl_integral_prefactor')
    # plt.plot(z_grid_ssc_integrands, sigma2_b[z1_idx, :], label='sigma2_b')
    # plt.yscale('log')
    # plt.legend()

    # mm.matshow(integrand_ssc_spaceborne_py_LLLL[ell1_idx, ell2_idx,zpair_AB, zpair_CD], log=True, abs_val=True)
    # mm.matshow(sigma2_b, log=True, abs_val=True)
    # plt.plot(z_grid_ssc_integrands, np.diag(sigma2_b), marker='o')
    # plt.plot(z_grid_ssc_integrands, np.diag(integrand_ssc_spaceborne_py_LLLL[ell1_idx, ell2_idx,zpair_AB, zpair_CD]), marker='o')
    # plt.yscale('log')

    # plt.plot(z_grid_ssc_integrands, integrand_ssc_spaceborne_py_LLLL[ell1_idx, ell2_idx,zpair_AB, zpair_CD, z1_idx, :], marker='o')

elif covariance_cfg['ng_cov_code'] == 'Spaceborne' and \
        covariance_cfg['Spaceborne_cfg']['load_precomputed_cov']:

    try:
        cov_ssc_3x2pt_dict_8D = mm.load_cov_from_probe_blocks(
            path=cov_folder_sb,
            filename=cov_sb_filename,
            probe_ordering=probe_ordering)
    except FileNotFoundError as err:
        print(err)
        print(f'No covariance file found in {cov_folder_sb}')
        print('Retrying with ellmax=5000 and cutting the last bins')
        cov_ssc_3x2pt_dict_8D = mm.load_cov_from_probe_blocks(
            path=cov_folder_sb,
            filename=cov_sb_filename.replace('lmax3000', 'lmax5000'),
            probe_ordering=probe_ordering)

    for key in cov_ssc_3x2pt_dict_8D.keys():
        cov_ssc_3x2pt_dict_8D[key] = cov_ssc_3x2pt_dict_8D[key][:nbl_3x2pt, :nbl_3x2pt, ...]

# this is not very elegant, find a better solution
if covariance_cfg['ng_cov_code'] == 'Spaceborne':
    covariance_cfg['cov_ssc_3x2pt_dict_8D_sb'] = cov_ssc_3x2pt_dict_8D

print('SSC computed with Spaceborne')
# TODO integrate this with Spaceborne_covg

# # ! quickly check responses
# import sys
# sys.path.append('/home/davide/Documenti/Lavoro/Programmi/exact_SSC/bin')
# import ssc_integrands_SPV3 as sscint

# z_val = 0
# z_grid_dPk_su = sscint.z_grid_dPk
# z_idx_hm = np.argmin(np.abs(z_grid_dPk_hm - z_val))
# z_idx_su = np.argmin(np.abs(z_grid_dPk_su - z_val))
# z_val_hm = z_grid_dPk_hm[z_idx_hm]
# z_val_su = z_grid_dPk_su[z_idx_su]

# # dPAB/ddeltab
# plt.figure()
# # HM
# plt.plot(k_grid_dPk_hm, np.abs(dPmm_ddeltab_hm[:, z_idx_hm]), ls='-', alpha=0.5, c='tab:blue')
# plt.plot(k_grid_dPk_hm, np.abs(dPgm_ddeltab_hm[:, z_idx_hm]), ls='-', alpha=0.5, c='tab:orange')
# plt.plot(k_grid_dPk_hm, np.abs(dPgg_ddeltab_hm[:, z_idx_hm]), ls='-', alpha=0.5, c='tab:green')

# # SU
# plt.plot(sscint.k_grid_dPk, np.abs(sscint.dPmm_ddeltab[:, z_idx_su]), ls='--', alpha=0.5, c='tab:blue')
# plt.plot(sscint.k_grid_dPk, np.abs(sscint.dPgm_ddeltab[:, z_idx_su]), ls='--', alpha=0.5, c='tab:orange')
# plt.plot(sscint.k_grid_dPk, np.abs(sscint.dPgg_ddeltab[:, z_idx_su]), ls='--', alpha=0.5, c='tab:green')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('k [1/Mpc]')
# plt.ylabel(r'${\rm abs} \; \\partial P_{AB} / \\partial \delta_b$')

# colors = ['tab:blue', 'tab:orange', 'tab:green']
# labels_a = ['dPmm_ddeltab', 'dPgm_ddeltab', 'dPgg_ddeltab']
# handles_z = [plt.Line2D([0], [0], color=colors[i], lw=2, label=labels_a[i]) for i in range(3)]
# handles_ls = [plt.Line2D([0], [0], color='k', lw=2, linestyle=ls, label=label)
#               for ls, label in zip(['-', '--'], ['signal', 'error'])]
# handles = handles_z + handles_ls
# labels = labels_a + ['Halo model', 'Separate universe']
# plt.legend(handles, labels)
# plt.title(f'z_hm = {z_val_hm:.3f}, z_su = {z_val_su:.3f}')
# plt.tight_layout()
# plt.show()

# # dlogPAB/ddeltab
# plt.figure()
# # HM
# plt.plot(k_grid_dPk_hm, dPmm_ddeltab_hm[:, z_idx_hm] / pk_mm_ccl[:, z_idx_hm], ls='-', alpha=0.5, c='tab:blue')
# plt.plot(k_grid_dPk_hm, dPgm_ddeltab_hm[:, z_idx_hm] / pk_mm_ccl[:, z_idx_hm], ls='-', alpha=0.5, c='tab:orange')
# plt.plot(k_grid_dPk_hm, dPgg_ddeltab_hm[:, z_idx_hm] / pk_mm_ccl[:, z_idx_hm], ls='-', alpha=0.5, c='tab:green')
# # SU
# plt.plot(sscint.k_grid_dPk, sscint.r_mm[:, z_idx_su], ls='--', alpha=0.5, c='tab:blue')
# plt.plot(sscint.k_grid_dPk, sscint.r_gm[:, z_idx_su], ls='--', alpha=0.5, c='tab:orange')
# plt.plot(sscint.k_grid_dPk, sscint.r_gg[:, z_idx_su], ls='--', alpha=0.5, c='tab:green')

# plt.xscale('log')
# plt.xlabel('k [1/Mpc]')
# plt.ylabel(r'$\\partial {\rm log} P_{AB} / \\partial \delta_b$')

# colors = ['tab:blue', 'tab:orange', 'tab:green']
# labels_a = ['dPmm_ddeltab/Pmm', 'dPgm_ddeltab/Pgm', 'dPgg_ddeltab/Pgg']
# handles_z = [plt.Line2D([0], [0], color=colors[i], lw=2, label=labels[i]) for i in range(3)]
# handles_ls = [plt.Line2D([0], [0], color='k', lw=2, linestyle=ls, label=label)
#               for ls, label in zip(['-', '--'], ['signal', 'error'])]
# handles = handles_z + handles_ls
# labels = labels_a + ['Halo model', 'Separate universe']
# plt.legend(handles, labels)
# plt.title(f'z_hm = {z_val_hm:.3f}, z_su = {z_val_su:.3f}')
# plt.tight_layout()
# plt.show()
# ! ========================================== end Spaceborne ===================================================

# ! ========================================== start PyCCL ===================================================
if covariance_cfg['ng_cov_code'] == 'PyCCL' and not pyccl_cfg['load_precomputed_cov']:

    # Note: this z grid has to be larger than the one requested in the trispectrum (z_grid_tkka in the cfg file).
    # You can probaby use the same grid as the one used in the trispectrum, but from my tests is should be
    # zmin_s2b < zmin_s2b_tkka and zmax_s2b =< zmax_s2b_tkka.
    # if zmin=0 it looks like I can have zmin_s2b = zmin_s2b_tkka
    ccl_obj.set_sigma2_b(z_grid=z_default_grid_ccl,
                         fsky=covariance_cfg['fsky'],
                         which_sigma2_b=which_sigma2_b,
                         nside_mask=covariance_cfg['nside_mask'],
                         mask_path=covariance_cfg['mask_path'])

    for which_ng_cov in pyccl_cfg['which_ng_cov']:
        warnings.warn('HANDLE BETTER THE DIFFERENT COV TERMS')
        ccl_obj.initialize_trispectrum(which_ng_cov, probe_ordering, pyccl_cfg)

        ccl_obj.compute_ng_cov_3x2pt(which_ng_cov, ell_dict['ell_3x2pt'], covariance_cfg['fsky'],
                                     integration_method='spline',  # TODO add try block for quad
                                     probe_ordering=probe_ordering, ind_dict=ind_dict)

        covariance_cfg[f'cov_{which_ng_cov.lower()}_3x2pt_dict_8D_ccl'] = ccl_obj.cov_ng_3x2pt_dict_8D


# for key in ccl_obj.cov_ng_3x2pt_dict_8D.keys():
#     cov_2d = mm.cov_4D_to_2D(ccl_obj.cov_ng_3x2pt_dict_8D[key])
#     mm.matshow(cov_2d, log=True, abs_val=True, title=''.join(key))

# for key in [('G', 'G', 'G', 'G'), ('L', 'L', 'L', 'L'), ('G', 'L', 'G', 'L'), ]:

#     np.testing.assert_allclose(ccl_obj.cov_ng_3x2pt_dict_8D[key],
#                                np.transpose(ccl_obj.cov_ng_3x2pt_dict_8D[key], (1, 0, 2, 3)),
#                                rtol=1e-5, atol=0,
#                                err_msg=f'cov_ng_4D {key} is not symmetric in ell1, ell2')

#     cov_2d = mm.cov_4D_to_2D(ccl_obj.cov_ng_3x2pt_dict_8D[key])
#     assert np.allclose(cov_2d, cov_2d.T, atol=0, rtol=1e-5)
#     mm.matshow(mm.cov2corr(cov_2d), log=False, abs_val=False, title=''.join(key))

# key = ('G', 'G', 'G', 'G')
# cov_2d_1 = mm.cov_4D_to_2D(np.transpose(ccl_obj.cov_ng_3x2pt_dict_8D[key], (1, 0, 2, 3)))
# # cov_2d_1 = mm.cov_4D_to_2D(np.transpose(ccl_obj.cov_ng_3x2pt_dict_8D[key], (1, 0, 3, 2)))
# cov_2d_2 = mm.cov_4D_to_2D(ccl_obj.cov_ng_3x2pt_dict_8D[key])

# mm.compare_arrays(cov_2d_1, cov_2d_2, plot_diff_hist=True, plot_diff_threshold=10)

# ! ========================================== end PyCCL ===================================================

# check that cl_wa is equal to cl_ll in the last nbl_WA_opt bins
if ell_max_WL == general_cfg['ell_max_WL_opt'] and general_cfg['use_WA']:
    if not np.array_equal(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :]):
        rtol = 1e-5
        # plt.plot(ell_dict['ell_WL'], cl_ll_3d[:, 0, 0])
        # plt.plot(ell_dict['ell_WL'][nbl_GC:nbl_WL], cl_wa_3d[:, 0, 0])
        assert (np.allclose(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :], rtol=rtol, atol=0)), \
            'cl_wa_3d should be obtainable from cl_ll_3d!'
        print(f'cl_wa_3d and cl_ll_3d[nbl_GC:nbl_WL, :, :] are not exactly equal, but have a relative '
              f'difference of less than {rtol}')

# ! BNT transform the cls (and responses?) - it's more complex since I also have to transform the noise
# ! spectra, better to transform directly the covariance matrix
if general_cfg['cl_BNT_transform']:
    print('BNT-transforming the Cls...')
    assert covariance_cfg['cov_BNT_transform'] is False, \
        'the BNT transform should be applied either to the Cls or to the covariance, not both'
    cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, bnt_matrix, 'L', 'L')
    cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, bnt_matrix, 'L', 'L')
    cl_3x2pt_5d = cl_utils.cl_BNT_transform_3x2pt(cl_3x2pt_5d, bnt_matrix)
    warnings.warn('you should probably BNT-transform the responses too!')

# ! cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
if ell_max_WL == 1500:
    warnings.warn(
        'you are cutting the datavectors and responses in the pessimistic case, but is this compatible '
        'with the redshift-dependent ell cuts? Yes, this is an old warning; nonetheless, check ')
    assert False, 'you should check this'
    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
    cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
    cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

    rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
    rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
    rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
    rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

# ! Vincenzo's method for cl_ell_cuts: get the idxs to delete for the flattened 1d cls
if general_cfg['center_or_min'] == 'center':
    prefix = 'ell'
elif general_cfg['center_or_min'] == 'min':
    prefix = 'ell_edges'
else:
    raise ValueError('general_cfg["center_or_min"] should be either "center" or "min"')

ell_dict['idxs_to_delete_dict'] = {
    'LL': ell_utils.get_idxs_to_delete(ell_dict[f'{prefix}_WL'], ell_cuts_dict['LL'], is_auto_spectrum=True, zbins=zbins),
    'GG': ell_utils.get_idxs_to_delete(ell_dict[f'{prefix}_GC'], ell_cuts_dict['GG'], is_auto_spectrum=True, zbins=zbins),
    'WA': ell_utils.get_idxs_to_delete(ell_dict[f'{prefix}_WA'], ell_cuts_dict['LL'], is_auto_spectrum=True, zbins=zbins),
    'GL': ell_utils.get_idxs_to_delete(ell_dict[f'{prefix}_XC'], ell_cuts_dict['GL'], is_auto_spectrum=False, zbins=zbins),
    'LG': ell_utils.get_idxs_to_delete(ell_dict[f'{prefix}_XC'], ell_cuts_dict['LG'], is_auto_spectrum=False, zbins=zbins),
    '3x2pt': ell_utils.get_idxs_to_delete_3x2pt(ell_dict[f'{prefix}_3x2pt'], ell_cuts_dict, zbins, covariance_cfg)
}

# ! 3d cl ell cuts (*after* BNT!!)
cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d = ell_utils.cl_ell_cut_wrap(
    ell_dict, cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d, ell_cuts_dict, general_cfg)
# TODO here you could implement 1d cl ell cuts (but we are cutting at the covariance and derivatives level)

# TODO delete this
rl_ll_3d, rl_gg_3d, rl_wa_3d, rl_3x2pt_5d = np.ones_like(cl_ll_3d), np.ones_like(cl_gg_3d), np.ones_like(cl_wa_3d), \
    np.ones_like(cl_3x2pt_5d)
# store cls and responses in a dictionary
cl_dict_3D = {
    'cl_LL_3D': cl_ll_3d,
    'cl_GG_3D': cl_gg_3d,
    'cl_WA_3D': cl_wa_3d,
    'cl_3x2pt_5D': cl_3x2pt_5d}

rl_dict_3D = {
    'rl_LL_3D': rl_ll_3d,
    'rl_GG_3D': rl_gg_3d,
    'rl_WA_3D': rl_wa_3d,
    'rl_3x2pt_5D': rl_3x2pt_5d}

# this is again to test against ccl cls
general_cfg['cl_ll_3d'] = cl_ll_3d
general_cfg['cl_gl_3d'] = cl_gl_3d
general_cfg['cl_gg_3d'] = cl_gg_3d

if covariance_cfg['compute_SSC'] and covariance_cfg['ng_cov_code'] == 'PySSC':

    transp_stacked_wf = np.vstack((wf_lensing.T, wf_galaxy.T))
    # ! compute or load Sijkl
    nz = z_arr.shape[0]  # get number of z points in nz to name the Sijkl file
    Sijkl_folder = Sijkl_cfg['Sijkl_folder']
    assert general_cfg[
        'cl_BNT_transform'] is False, 'for SSC, at the moment the BNT transform should not be ' \
        'applied to the cls, but to the covariance matrix (how ' \
        'should we deal with the responses in the former case?)'
    Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(
        flagship_version=general_cfg['flagship_version'],
        nz=nz, IA_flag=Sijkl_cfg['has_IA'],
        **variable_specs)

    # if Sijkl exists, load it; otherwise, compute it and save it
    if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):
        print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
        Sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')
    else:
        Sijkl = Sijkl_utils.compute_Sijkl(cosmo_lib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                          Sijkl_cfg['wf_normalization'])
        np.save(f'{Sijkl_folder}/{Sijkl_filename}', Sijkl)

else:
    warnings.warn('Sijkl is not computed, but set to identity')
    Sijkl = np.ones((n_probes * zbins, n_probes * zbins, n_probes * zbins, n_probes * zbins))

# ! compute covariance matrix
# TODO: if already existing, don't compute the covmat, like done above for Sijkl
cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                    ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, bnt_matrix, oc_obj)

# ! save for CLOE runs
if covariance_cfg['save_CLOE_benchmark_cov']:

    assert general_cfg['ell_max_3x2pt'] == 5000, 'ell_max_3x2pt should be == 5000 for the CLOE bench case'

    # reshape cov in CLOE format
    mm.matshow(cov_dict['cov_3x2pt_GO_2D'], log=True, abs_val=False, title='cov_GO_3x2pt pre-reshape')
    mm.matshow(cov_dict['cov_3x2pt_GS_2D'], log=True, abs_val=False, title='cov_GS_3x2pt pre-reshape')

    cov_3x2pt_GO_2DCLOE = mm.cov_2d_dav_to_cloe(cov_dict['cov_3x2pt_GO_2D'], nbl_3x2pt, zbins, 'ell', 'ell')
    cov_3x2pt_GS_2DCLOE = mm.cov_2d_dav_to_cloe(cov_dict['cov_3x2pt_GS_2D'], nbl_3x2pt, zbins, 'ell', 'ell')

    mm.matshow(cov_3x2pt_GO_2DCLOE, log=True, abs_val=False, title='cov_GO_3x2pt post-reshape')
    mm.matshow(cov_3x2pt_GS_2DCLOE, log=True, abs_val=False, title='cov_GS_3x2pt post-reshape')

    np.save(f'{cloe_bench_path}/CovMat-3x2pt-Gauss-32Bins-v2.npy', cov_3x2pt_GO_2DCLOE)
    np.save(f'{cloe_bench_path}/CovMat-3x2pt-GaussSSC-32Bins-v2.npy', cov_3x2pt_GS_2DCLOE)

    # compare against the saved ones - this requires executing the bit of code below this block,
    # or it will give error
    mm.compare_arrays(cov_3x2pt_GO_2DCLOE, cov_3x2pt_g_nbl32_2dcloe, log_diff=True)
    mm.compare_arrays(cov_3x2pt_GS_2DCLOE, cov_3x2pt_gs_nbl32_2dcloe, log_diff=True)

    plt.plot(np.diag(cov_3x2pt_g_nbl32_2dcloe), label='cov_3x2pt_g_nbl32_2dcloe')
    plt.plot(np.diag(cov_3x2pt_GO_2DCLOE), label='cov_3x2pt_GO_2DCLOE')
    plt.legend()
    plt.yscale('log')

    plt.plot(np.diag(cov_3x2pt_gs_nbl32_2dcloe), label='cov_3x2pt_gs_nbl32_2dcloe')
    plt.plot(np.diag(cov_3x2pt_GS_2DCLOE), label='cov_3x2pt_GS_2DCLOE')
    plt.legend()
    plt.yscale('log')

if covariance_cfg['load_CLOE_benchmark_cov']:
    warnings.warn('OVERWRITING cov_dict WITH CLOE BENCHMARKS')

    cloe_bench_path = general_cfg['CLOE_benchmarks_path'].format(ROOT=ROOT)
    area_deg2 = covariance_cfg['survey_area_deg2']
    cov_3x2pt_g_nbl32_2dcloe = np.load(f'{cloe_bench_path}/CovMat-3x2pt-Gauss-32Bins-{area_deg2:d}deg2.npy')
    cov_3x2pt_gs_nbl32_2dcloe = np.load(f'{cloe_bench_path}/CovMat-3x2pt-GaussSSC-32Bins-{area_deg2:d}deg2.npy')

    num_elem_auto_nbl32 = zpairs_auto * nbl_WL_opt
    num_elem_cross_nbl32 = zpairs_cross * nbl_WL_opt

    # Cut the probe blocks: once I do this, I'm back in the ell-zpair (dav) ordering!
    cov_wl_g_nbl32_2ddav = deepcopy(cov_3x2pt_g_nbl32_2dcloe)[:num_elem_auto_nbl32, :num_elem_auto_nbl32]
    cov_xc_g_nbl32_2ddav = deepcopy(cov_3x2pt_g_nbl32_2dcloe)[num_elem_auto_nbl32:num_elem_auto_nbl32 + num_elem_cross_nbl32,
                                                              num_elem_auto_nbl32:num_elem_auto_nbl32 + num_elem_cross_nbl32]
    cov_gc_g_nbl32_2ddav = deepcopy(cov_3x2pt_g_nbl32_2dcloe)[num_elem_auto_nbl32 + num_elem_cross_nbl32:,
                                                              num_elem_auto_nbl32 + num_elem_cross_nbl32:]

    cov_wl_gs_nbl32_2ddav = deepcopy(cov_3x2pt_gs_nbl32_2dcloe)[:num_elem_auto_nbl32, :num_elem_auto_nbl32]
    cov_xc_gs_nbl32_2ddav = deepcopy(cov_3x2pt_gs_nbl32_2dcloe)[num_elem_auto_nbl32:num_elem_auto_nbl32 + num_elem_cross_nbl32,
                                                                num_elem_auto_nbl32:num_elem_auto_nbl32 + num_elem_cross_nbl32]
    cov_gc_gs_nbl32_2ddav = deepcopy(cov_3x2pt_gs_nbl32_2dcloe)[num_elem_auto_nbl32 + num_elem_cross_nbl32:,
                                                                num_elem_auto_nbl32 + num_elem_cross_nbl32:]

    # now cut to 29 bins
    num_elem_auto_nbl29 = zpairs_auto * nbl_3x2pt
    num_elem_cross_nbl29 = zpairs_cross * nbl_3x2pt
    cov_wl_g_nbl29_2ddav = cov_wl_g_nbl32_2ddav[:num_elem_auto_nbl29, :num_elem_auto_nbl29]
    cov_xc_g_nbl29_2ddav = cov_xc_g_nbl32_2ddav[:num_elem_cross_nbl29, :num_elem_cross_nbl29]
    cov_gc_g_nbl29_2ddav = cov_gc_g_nbl32_2ddav[:num_elem_auto_nbl29, :num_elem_auto_nbl29]

    cov_wl_gs_nbl29_2ddav = cov_wl_gs_nbl32_2ddav[:num_elem_auto_nbl29, :num_elem_auto_nbl29]
    cov_xc_gs_nbl29_2ddav = cov_xc_gs_nbl32_2ddav[:num_elem_cross_nbl29, :num_elem_cross_nbl29]
    cov_gc_gs_nbl29_2ddav = cov_gc_gs_nbl32_2ddav[:num_elem_auto_nbl29, :num_elem_auto_nbl29]

    # reshape to dav
    cov_3x2pt_g_nbl32_2ddav = mm.cov_2d_cloe_to_dav(cov_3x2pt_g_nbl32_2dcloe, nbl_WL_opt, zbins, 'ell', 'ell')
    cov_3x2pt_gs_nbl32_2ddav = mm.cov_2d_cloe_to_dav(cov_3x2pt_gs_nbl32_2dcloe, nbl_WL_opt, zbins, 'ell', 'ell')

    # cut last 3 ell bins
    num_elem_tot_nbl29 = (zpairs_auto * n_probes + zpairs_cross) * nbl_3x2pt
    cov_3x2pt_g_nbl29_2ddav = cov_3x2pt_g_nbl32_2ddav[:num_elem_tot_nbl29, :num_elem_tot_nbl29]
    cov_3x2pt_gs_nbl29_2ddav = cov_3x2pt_gs_nbl32_2ddav[:num_elem_tot_nbl29, :num_elem_tot_nbl29]

    cov_dict['cov_WL_GO_2D'] = cov_wl_g_nbl29_2ddav
    cov_dict['cov_XC_GO_2D'] = cov_xc_g_nbl29_2ddav
    cov_dict['cov_GC_GO_2D'] = cov_gc_g_nbl29_2ddav
    cov_dict['cov_3x2pt_GO_2D'] = cov_3x2pt_g_nbl29_2ddav

    cov_dict['cov_WL_GS_2D'] = cov_wl_gs_nbl29_2ddav
    cov_dict['cov_XC_GS_2D'] = cov_xc_gs_nbl29_2ddav
    cov_dict['cov_GC_GS_2D'] = cov_gc_gs_nbl29_2ddav
    cov_dict['cov_3x2pt_GS_2D'] = cov_3x2pt_gs_nbl29_2ddav

    del cov_wl_g_nbl29_2ddav, cov_xc_g_nbl29_2ddav, cov_gc_g_nbl29_2ddav, cov_3x2pt_g_nbl29_2ddav
    del cov_wl_gs_nbl29_2ddav, cov_xc_gs_nbl29_2ddav, cov_gc_gs_nbl29_2ddav, cov_3x2pt_gs_nbl29_2ddav
    gc.collect()

# covmat_utils.save_cov(cov_folder, covariance_cfg, cov_dict, cases_tosave, **variable_specs)

# save in .dat for Vincenzo, only in the optimistic case and in 2D
if covariance_cfg['save_cov_2D_dat']:
    var_specs_here = deepcopy(variable_specs)
    var_specs_here.pop('which_ng_cov', None)
    cov_filename_vin = covariance_cfg['cov_filename'].format(**var_specs_here, probe='{probe:s}',
                                                             lmax_3x2pt=ell_max_3x2pt, ndim=2,
                                                             cov_suffix='',
                                                             which_ng_cov='{which_ng_cov:s}',
                                                             fm_and_cov_suffix=general_cfg['fm_and_cov_suffix'])
    cov_filename_vin = cov_filename_vin.replace('.npz', '')
    cov_filename_vin = cov_filename_vin.replace(
        f'_pk{which_pk}', f'_pk{which_pk}_{covariance_cfg["survey_area_deg2"]}deg2')

    var_specs_here = deepcopy(variable_specs)
    var_specs_here.pop('BNT_transform', None)
    cov_folder_vin = covariance_cfg['cov_folder'].format(ROOT=ROOT,
                                                         cov_ell_cuts=str(covariance_cfg['cov_ell_cuts']),
                                                         BNT_transform=str(general_cfg['BNT_transform']),
                                                         **var_specs_here)

    np.savetxt(f'{cov_folder_vin}/{cov_filename_vin.format(which_ng_cov=which_ng_cov_suffix, probe='3x2pt')}.dat',
               cov_dict[f'cov_3x2pt_GS_2D'], fmt='%.7e')

# load benchmark cov and check that it matches the one computed here; I am not actually using it
if (
    ep_or_ed == 'EP' and
    covariance_cfg['ng_cov_code'] == 'Spaceborne' and
    covariance_cfg['test_against_CLOE_benchmarks'] and
    not general_cfg['ell_cuts'] and
    which_pk == 'HMCodeBar' and
    not general_cfg['BNT_transform']
):

    # load CLOE benchmarks
    cov_cloe_bench_2d_G = np.load(f'{ROOT}/my_cloe_data/CovMat-3x2pt-Gauss-{nbl_WL_opt}Bins.npy')
    cov_cloe_bench_2dcloe_GSSC = np.load(f'{ROOT}/my_cloe_data/CovMat-3x2pt-GaussSSC-{nbl_WL_opt}Bins.npy')

    # reshape it in dav format
    cov_bench_2ddav_G = mm.cov_2d_cloe_to_dav(cov_cloe_bench_2d_G, nbl_WL_opt, zbins, 'ell', 'ell')
    cov_bench_2ddav_GSSC = mm.cov_2d_cloe_to_dav(
        cov_cloe_bench_2dcloe_GSSC, nbl_WL_opt, zbins, 'ell', 'ell')

    # ell cut, if needed
    assert cov_dict['cov_3x2pt_GO_2D'].shape == cov_dict['cov_3x2pt_GS_2D'].shape, \
        'cov_3x2pt_GO_2D and cov_3x2pt_GS_2D should have the same shape'
    n_cov_elements = cov_dict['cov_3x2pt_GO_2D'].shape[0]
    cov_bench_2ddav_G_lmax3000 = cov_bench_2ddav_G[:n_cov_elements, :n_cov_elements]
    cov_bench_2ddav_GSSC_lmax3000 = cov_bench_2ddav_GSSC[:n_cov_elements, :n_cov_elements]

    mm.compare_arrays(cov_dict['cov_3x2pt_GO_2D'], cov_bench_2ddav_G_lmax3000,
                      "cov_dict['cov_3x2pt_GO_2D']", "cov_bench_2ddav_G_lmax3000",
                      log_array=True, log_diff=False, abs_val=False, plot_diff_threshold=5)
    mm.compare_arrays(cov_dict['cov_3x2pt_GS_2D'], cov_bench_2ddav_GSSC_lmax3000,
                      "cov_dict['cov_3x2pt_GS_2D']", "cov_bench_2ddav_GSSC_lmax3000",
                      log_array=True, log_diff=False, abs_val=False, plot_diff_threshold=5)

    del cov_bench_2ddav_G_lmax3000, cov_bench_2ddav_GSSC_lmax3000
    gc.collect()

if covariance_cfg['compute_GSSC_condition_number']:

    cond_number = np.linalg.cond(cov_dict['cov_3x2pt_GS_2D'])
    NUMPY_PRECISION = np.finfo(float).eps
    precision = cond_number * NUMPY_PRECISION
    print(f'kmax = {kmax_h_over_Mpc}, precision in the inversion of GS covariance = '
          f'{precision:.2e}, cond number = {cond_number:.2e}')

if covariance_cfg['test_against_vincenzo'] and (not bnt_transform) and (general_cfg['which_cls'] == 'Vincenzo'):
    cov_vinc_filename = covariance_cfg['cov_vinc_filename'].format(**variable_specs, probe='3x2pt')
    cov_vinc_g = np.load(f'{covariance_cfg["cov_vinc_folder"]}/{cov_vinc_filename}')['arr_0']
    num_elements_nbl29 = cov_dict['cov_3x2pt_GO_2D'].shape[0]
    npt.assert_allclose(cov_dict['cov_3x2pt_GO_2D'], cov_vinc_g[:num_elements_nbl29, :num_elements_nbl29],
                        rtol=1e-3, atol=0, err_msg='cov_dict["cov_3x2pt_GO_2D"] does not match with Vincenzo\'s')
    print('covariance matrix matches with Vincenzo\'s ✅')

# !============================= derivatives ===================================
# this guard is just to avoid indenting the whole code below
if not fm_cfg['compute_FM']:
    del cov_dict
    gc.collect()
    # continue
    # raise KeyboardInterrupt('skipping FM computation, the script will exit now')

# list_params_to_vary = list(fid_pars_dict['FM_ordered_params'].keys())
list_params_to_vary = [param for param in fid_pars_dict['FM_ordered_params'].keys() if param != 'ODE']
# list_params_to_vary = ['h', 'wa', 'dzWL01', 'm06', 'bG02', 'bM02']
# list_params_to_vary = ['bM02', ]

if fm_cfg['which_derivatives'] == 'Spaceborne':

    if fm_cfg['load_preprocess_derivatives']:
        # a better name should be dict_4D...? anyway, not so important
        dC_dict_LL_3D = np.load(f'/home/davide/Scrivania/test_ders/dcl_LL.npy', allow_pickle=True).item()
        dC_dict_GL_3D = np.load(f'/home/davide/Scrivania/test_ders/dcl_GL.npy', allow_pickle=True).item()
        dC_dict_GG_3D = np.load(f'/home/davide/Scrivania/test_ders/dcl_GG.npy', allow_pickle=True).item()

    elif not fm_cfg['load_preprocess_derivatives']:
        start_time = time.perf_counter()
        cl_LL, cl_GL, cl_GG, dC_dict_LL_3D, dC_dict_GL_3D, dC_dict_GG_3D = wf_cl_lib.compute_cls_derivatives(
            cfg, list_params_to_vary, zbins, (n_of_z_full[:, 0], n_of_z_full[:, 1:]),
            ell_dict['ell_WL'], ell_dict['ell_XC'], ell_dict['ell_GC'], use_only_flat_models=True)
        print('derivatives computation time: {:.2f} s'.format(time.perf_counter() - start_time))

        # np.save(f'/home/davide/Scrivania/test_ders/dcl_LL.npy', dcl_LL, allow_pickle=True)
        # np.save(f'/home/davide/Scrivania/test_ders/dcl_GL.npy', dcl_GL, allow_pickle=True)
        # np.save(f'/home/davide/Scrivania/test_ders/dcl_GG.npy', dcl_GG, allow_pickle=True)

    # reshape to 4D array (instead of dictionaries)
    dC_LL_4D = np.zeros((nbl_3x2pt, zbins, zbins, len(list_params_to_vary)))
    dC_WA_4D = np.zeros((nbl_WA, zbins, zbins, len(list_params_to_vary)))
    dC_GL_4D = np.zeros((nbl_3x2pt, zbins, zbins, len(list_params_to_vary)))
    dC_GG_4D = np.zeros((nbl_3x2pt, zbins, zbins, len(list_params_to_vary)))
    dC_3x2pt_6D = np.zeros((2, 2, nbl_3x2pt, zbins, zbins, len(list_params_to_vary)))

    for par_idx, par_name in enumerate(list_params_to_vary):
        dC_LL_4D[:, :, :, par_idx] = dC_dict_LL_3D[par_name]
        dC_GL_4D[:, :, :, par_idx] = dC_dict_GL_3D[par_name]
        dC_GG_4D[:, :, :, par_idx] = dC_dict_GG_3D[par_name]
        dC_3x2pt_6D[0, 0, :, :, :, par_idx] = dC_dict_LL_3D[par_name][:nbl_3x2pt, :, :]
        dC_3x2pt_6D[1, 0, :, :, :, par_idx] = dC_dict_GL_3D[par_name][:nbl_3x2pt, :, :]
        dC_3x2pt_6D[1, 1, :, :, :, par_idx] = dC_dict_GG_3D[par_name][:nbl_3x2pt, :, :]

elif fm_cfg['which_derivatives'] == 'Vincenzo':
    # Vincenzo's derivatives
    der_prefix = fm_cfg['derivatives_prefix']
    derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs, ROOT=ROOT)
    # ! get vincenzo's derivatives' parameters, to check that they match with the yaml file
    # check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
    vinc_filenames = mm.get_filenames_in_folder(derivatives_folder)
    vinc_filenames = [vinc_filename for vinc_filename in vinc_filenames if
                      vinc_filename.startswith(der_prefix)]

    # keep only the files corresponding to the correct magcut_lens, magcut_source and zbins
    vinc_filenames = [filename for filename in vinc_filenames if
                      all(x in filename for x in
                          [f'ML{magcut_lens}', f'MS{magcut_source}', f'{ep_or_ed}{zbins:02d}'])]
    vinc_filenames = [filename.replace('.dat', '') for filename in vinc_filenames]

    vinc_trimmed_filenames = [vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames]
    vinc_trimmed_filenames = [
        vinc_trimmed_filename[len(der_prefix):] if vinc_trimmed_filename.startswith(
            der_prefix) else vinc_trimmed_filename
        for vinc_trimmed_filename in vinc_trimmed_filenames]
    vinc_param_names = list(set(vinc_trimmed_filenames))
    vinc_param_names.sort()

    # ! get fiducials names and values from the yaml file
    # remove ODE if I'm studying only flat models
    if general_cfg['flat_or_nonflat'] == 'Flat' and 'ODE' in fid_pars_dict['FM_ordered_params']:
        fid_pars_dict['FM_ordered_params'].pop('ODE')
    fm_fid_dict = fid_pars_dict['FM_ordered_params']
    param_names_3x2pt = list(fm_fid_dict.keys())
    fm_cfg['param_names_3x2pt'] = param_names_3x2pt
    fm_cfg['nparams_tot'] = len(param_names_3x2pt)

    # sort them to compare with vincenzo's param names
    my_sorted_param_names = param_names_3x2pt.copy()
    my_sorted_param_names.sort()

    for dzgc_param_name in [f'dzGC{zi:02d}' for zi in range(1, zbins + 1)]:
        if dzgc_param_name in vinc_param_names:  # ! added this if statement, not very elegant
            vinc_param_names.remove(dzgc_param_name)

    # check whether the 2 lists match and print the elements that are in one list but not in the other
    param_names_not_in_my_list = [vinc_param_name for vinc_param_name in vinc_param_names if
                                  vinc_param_name not in my_sorted_param_names]
    param_names_not_in_vinc_list = [my_sorted_param_name for my_sorted_param_name in my_sorted_param_names
                                    if
                                    my_sorted_param_name not in vinc_param_names]

    # Check if the parameter names match
    if not np.all(vinc_param_names == my_sorted_param_names):
        # Print the mismatching parameters
        print(f'Params present in input folder but not in the cfg file: {param_names_not_in_my_list}')
        print(f'Params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}')

    # ! preprocess derivatives (or load the alreay preprocessed ones)
    if fm_cfg['load_preprocess_derivatives']:
        warnings.warn(
            'loading preprocessed derivatives is faster but a bit more dangerous, make sure all the specs are taken into account')
        dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
        dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
        dC_WA_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_WA_4D.npy')
        dC_3x2pt_6D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy')

    elif not fm_cfg['load_preprocess_derivatives']:
        der_prefix = fm_cfg['derivatives_prefix']
        dC_dict_1D = dict(mm.get_kv_pairs_v2(derivatives_folder, "dat"))
        # check if dictionary is empty
        if not dC_dict_1D:
            raise ValueError(f'No derivatives found in folder {derivatives_folder}')

        # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
        dC_dict_LL_3D = {}
        dC_dict_GG_3D = {}
        dC_dict_WA_3D = {}
        dC_dict_3x2pt_5D = {}

        for key in vinc_filenames:  # loop over these, I already selected ML, MS and so on
            if not key.startswith('dDVddzGC'):
                if 'WLO' in key:
                    dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(
                        dC_dict_1D[key], 'WL', nbl_WL_opt, zbins)[:nbl_WL, :, :]
                elif 'GCO' in key:
                    dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'GC', nbl_GC, zbins)
                elif 'WLA' in key:
                    dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WA', nbl_WA, zbins)
                elif '3x2pt' in key:
                    dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(
                        dC_dict_1D[key], '3x2pt', nbl_3x2pt, zbins)

        # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
        dC_LL_4D_vin = fm_utils.dC_dict_to_4D_array(dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix)
        dC_GG_4D_vin = fm_utils.dC_dict_to_4D_array(dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix)
        # dC_WA_4D_vin = fm_utils.dC_dict_to_4D_array(dC_dict_WA_3D, param_names_3x2pt, nbl_WA, zbins, der_prefix)
        dC_WA_4D_vin = np.ones((nbl_WA, zbins, zbins, dC_LL_4D_vin.shape[-1]))
        dC_3x2pt_6D_vin = fm_utils.dC_dict_to_4D_array(
            dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins, der_prefix, is_3x2pt=True)

        # free up memory
        del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_WA_3D, dC_dict_3x2pt_5D
        gc.collect()

        # save these so they can simply be imported!
        if not os.path.exists(f'{derivatives_folder}/reshaped_into_np_arrays'):
            os.makedirs(f'{derivatives_folder}/reshaped_into_np_arrays')
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_LL_4D.npy', dC_LL_4D_vin)
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_GG_4D.npy', dC_GG_4D_vin)
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_WA_4D.npy', dC_WA_4D_vin)
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_3x2pt_6D.npy', dC_3x2pt_6D_vin)

else:
    raise ValueError('which_derivatives must be either "Spaceborne" or "Vincenzo"')

# ! compare derivatives
"""
param = list_params_to_vary[0]
for param in list_params_to_vary:
    dcl_ll_3d_vinc = dC_dict_LL_3D[f'dDVd{param}-WLO-ML{magcut_lens}-MS{magcut_source}-{ep_or_ed}{zbins}']
    dcl_gl_3d_vinc = dC_dict_3x2pt_5D[f'dDVd{param}-3x2pt-ML{magcut_lens}-MS{magcut_source}-{ep_or_ed}{zbins}'][1, 0, ...]
    dcl_gg_3d_vinc = dC_dict_GG_3D[f'dDVd{param}-GCO-ML{magcut_lens}-MS{magcut_source}-{ep_or_ed}{zbins}']

    clr = cm.rainbow(np.linspace(0, 1, zbins))
    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(10, 5), height_ratios=[2, 1])
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)

for zi in range(zbins):
    zj = zi
    ax[0, 0].loglog(ell_dict['ell_WL'], np.abs(dC_dict_LL_3D[param][:, zi, zj]), ls="-", c=clr[zi], alpha=0.6)
    ax[0, 0].loglog(ell_dict['ell_WL'], np.abs(dcl_ll_3d_vinc[:, zi, zj]), ls=":", c=clr[zi], alpha=0.6)

    ax[0, 1].loglog(ell_dict['ell_XC'], np.abs(dC_dict_GL_3D[param][:, zi, zj]), ls="-", c=clr[zi], alpha=0.6)
    ax[0, 1].loglog(ell_dict['ell_XC'], np.abs(dcl_gl_3d_vinc[:, zi, zj]), ls=":", c=clr[zi], alpha=0.6)

    ax[0, 2].loglog(ell_dict['ell_GC'], np.abs(dC_dict_GG_3D[param][:, zi, zj]), ls="-", c=clr[zi], alpha=0.6)
    ax[0, 2].loglog(ell_dict['ell_GC'], np.abs(dcl_gg_3d_vinc[:, zi, zj]), ls=":", c=clr[zi], alpha=0.6)

    ax[1, 0].plot(ell_dict['ell_WL'], mm.percent_diff(dC_dict_LL_3D[param], dcl_ll_3d_vinc)[:, zi, zj], c=clr[zi])
    ax[1, 1].plot(ell_dict['ell_XC'], mm.percent_diff(dC_dict_GL_3D[param], dcl_gl_3d_vinc)[:, zi, zj], c=clr[zi])
    ax[1, 2].plot(ell_dict['ell_GC'], mm.percent_diff(dC_dict_GG_3D[param], dcl_gg_3d_vinc)[:, zi, zj], c=clr[zi])

    ax[1, 0].set_ylim(-20, 20)
    ax[1, 1].set_ylim(-20, 20)
    ax[1, 2].set_ylim(-20, 20)

    ax[1, 0].fill_between(ell_dict['ell_WL'], -5, 5, color='grey', alpha=0.3)
    ax[1, 1].fill_between(ell_dict['ell_XC'], -5, 5, color='grey', alpha=0.3)
    ax[1, 2].fill_between(ell_dict['ell_GC'], -5, 5, color='grey', alpha=0.3)

    ax[1, 0].set_xlabel('$\\ell$')
    ax[1, 1].set_xlabel('$\\ell$')
    ax[1, 2].set_xlabel('$\\ell$')
    ax[0, 0].set_ylabel('$\\partial C_{\\ell}/ \\partial \\theta$')
    ax[1, 0].set_ylabel('% diff')
    lines = [plt.Line2D([], [], color='k', linestyle=ls) for ls in ['-', ':']]
    fig.suptitle(param)
    plt.legend(lines, ['davide', 'vincenzo'], loc='upper right', bbox_to_anchor=(1.55, 1))
    plt.show()

ell_low, ell_up = 0, 1
mm.compare_arrays(mm.block_diag(dC_dict_LL_3D[param][ell_low: ell_up]), mm.block_diag(
    dcl_ll_3d_vinc[ell_low: ell_up]), 'davide, LL', 'vincenzo', abs_val=True, plot_diff=False)
mm.compare_arrays(mm.block_diag(dC_dict_GL_3D[param][ell_low: ell_up]), mm.block_diag(
    dcl_gl_3d_vinc[ell_low: ell_up]), 'davide, GL', 'vincenzo', abs_val=True, plot_diff=False)
mm.compare_arrays(mm.block_diag(dC_dict_GG_3D[param][ell_low: ell_up]), mm.block_diag(
    dcl_gg_3d_vinc[ell_low: ell_up]), 'davide, GG', 'vincenzo', abs_val=True, plot_diff=False)


# ! compare saved cls from fiducial value (percentages = 0 case)
cl_LL_3d_fid_bench = np.load(f'/home/davide/Scrivania/test_ders/cl_LL_h.npy')
cl_GL_3d_fid_bench = np.load(f'/home/davide/Scrivania/test_ders/cl_GL_h.npy')
cl_GG_3d_fid_bench = np.load(f'/home/davide/Scrivania/test_ders/cl_GG_h.npy')

for param in list_params_to_vary:

    # in the derivatives computation, the cls computed for the fiducial prediction must match
    cl_LL_3d_fid = np.load(f'/home/davide/Scrivania/test_ders/cl_LL_{param}.npy')
    cl_GL_3d_fid = np.load(f'/home/davide/Scrivania/test_ders/cl_GL_{param}.npy')
    cl_GG_3d_fid = np.load(f'/home/davide/Scrivania/test_ders/cl_GG_{param}.npy')

    # lower tolerance for Vincenzo's cls, GG and GL are tricier to compare in this way
    np.testing.assert_allclose(cl_LL_3d_fid, cl_ll_3d_vinc, atol=0, rtol=2e-2)

    np.testing.assert_allclose(cl_LL_3d_fid, cl_LL_3d_fid_bench, atol=0, rtol=1e-5)
    np.testing.assert_allclose(cl_GL_3d_fid, cl_GL_3d_fid_bench, atol=0, rtol=1e-5)
    np.testing.assert_allclose(cl_GG_3d_fid, cl_GG_3d_fid_bench, atol=0, rtol=1e-5)

    clr = cm.rainbow(np.linspace(0, 1, zbins))
    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(10, 5), height_ratios=[2, 1])
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)

    for zi in range(zbins):
        zj = zi
        ax[0, 0].loglog(ell_dict['ell_WL'], cl_LL_3d_fid[:, zi, zj], ls="-", c=clr[zi], alpha=0.6)
        ax[0, 0].loglog(ell_dict['ell_WL'], cl_ll_3d_vinc[:, zi, zj], ls=":", c=clr[zi], alpha=0.6)

        ax[0, 1].loglog(ell_dict['ell_XC'], np.abs(cl_GL_3d_fid[:, zi, zj]), ls="-", c=clr[zi], alpha=0.6)
        ax[0, 1].loglog(ell_dict['ell_XC'][:29], np.abs(cl_gl_3d_vinc[:, zi, zj]), ls=":", c=clr[zi], alpha=0.6)

        ax[0, 2].loglog(ell_dict['ell_GC'], cl_GG_3d_fid[:, zi, zj], ls="-", c=clr[zi], alpha=0.6)
        ax[0, 2].loglog(ell_dict['ell_GC'][:29], cl_gg_3d_vinc[:, zi, zj], ls=":", c=clr[zi], alpha=0.6)

        ax[1, 0].plot(ell_dict['ell_WL'], mm.percent_diff(cl_LL_3d_fid, cl_ll_3d_vinc)[:, zi, zj], c=clr[zi])
        ax[1, 1].plot(ell_dict['ell_XC'][:29], mm.percent_diff(cl_GL_3d_fid[:29], cl_gl_3d_vinc)[:, zi, zj], c=clr[zi])
        ax[1, 2].plot(ell_dict['ell_GC'][:29], mm.percent_diff(cl_GG_3d_fid[:29], cl_gg_3d_vinc)[:, zi, zj], c=clr[zi])

    ax[1, 0].set_xlabel('$\\ell$')
    ax[1, 1].set_xlabel('$\\ell$')
    ax[1, 2].set_xlabel('$\\ell$')
    ax[0, 0].set_ylabel('$C_{\ell}$')
    ax[1, 0].set_ylabel('% diff')
    ax[1, 1].set_ylim(-20, 20)
    lines = [plt.Line2D([], [], color='k', linestyle=ls) for ls in ['-', ':']]
    plt.legend(lines, ['davide', 'vincenzo'], loc='upper center', bbox_to_anchor=(1.55, 1))
    plt.show()
"""

# import and store derivative in one big dictionary

# store the derivatives arrays in a dictionary
# deriv_dict_dav = {'dC_LL_4D': dC_LL_4D,
#                   'dC_WA_4D': dC_WA_4D,
#                   'dC_GG_4D': dC_GG_4D,
#                   'dC_3x2pt_6D': dC_3x2pt_6D}

deriv_dict_vin = {'dC_LL_4D': dC_LL_4D_vin,
                  'dC_WA_4D': dC_WA_4D_vin,
                  'dC_GG_4D': dC_GG_4D_vin,
                  'dC_3x2pt_6D': dC_3x2pt_6D_vin}

# ! ==================================== compute and save fisher matrix ================================================
fm_dict_vin = fm_utils.compute_FM(cfg, ell_dict, cov_dict, deriv_dict_vin, bnt_matrix)

# TODO finish testing derivatives
# fm_dict_dav = fm_utils.compute_FM(cfg, ell_dict, cov_dict, deriv_dict_dav, bnt_matrix)
# fm_dict_vin_modified = {key + '_vin': value for key, value in fm_dict_vin.items()}
# del fm_dict_vin_modified['fiducial_values_dict_vin']
# fm_dict = {**fm_dict_dav, **fm_dict_vin_modified}

fm_dict = fm_dict_vin

# ordered fiducial parameters entering the FM
fm_dict['fiducial_values_dict'] = cfg['cosmology']['FM_ordered_params']

fm_folder = fm_cfg['fm_folder'].format(ROOT=ROOT,
                                       ell_cuts=str(general_cfg['ell_cuts']),
                                       which_cuts=general_cfg['which_cuts'],
                                       flagship_version=general_cfg['flagship_version'],
                                       BNT_transform=str(bnt_transform),
                                       center_or_min=general_cfg['center_or_min'],)

if not general_cfg['ell_cuts']:
    # not very nice, i defined the ell_cuts_subfolder above...
    fm_folder = fm_folder.replace(f'/{general_cfg["which_cuts"]}/ell_{center_or_min}', '')

if fm_cfg['save_FM_dict']:

    if (not general_cfg['has_IA']) and (not general_cfg['has_magnification_bias']):
        simpkern = str(True)
    else:
        simpkern = str(False)

    fm_dict_filename = fm_cfg['fm_dict_filename'].format(
        **variable_specs, fm_and_cov_suffix=general_cfg['fm_and_cov_suffix'],
        lmax=ell_max_3x2pt, survey_area_deg2=covariance_cfg['survey_area_deg2'],
        cl_integral_convention=covariance_cfg['Spaceborne_cfg']['cl_integral_convention'],
        simpkern=simpkern,
        which_sigma2_b=str(which_sigma2_b))

    if covariance_cfg['ng_cov_code'] == 'Spaceborne':
        resp_filename = covariance_cfg['Spaceborne_cfg']['which_pk_responses']

        if resp_filename == 'halo_model':
            resp_filename = 'HM'
        elif resp_filename.startswith('separate_universe'):
            resp_filename = resp_filename.replace('separate_universe', 'SU')

        fm_dict_filename = fm_dict_filename.replace(
            '.pickle',
            f'_{resp_filename}.pickle'
        )

    mm.save_pickle(f'{fm_folder}/{fm_dict_filename}', fm_dict)

if fm_cfg['test_against_benchmarks']:
    saved_fm_path = f'{fm_folder}/{fm_dict_filename}'
    benchmark_path = f'{fm_folder}/benchmarks/{fm_dict_filename}'
    mm.compare_param_cov_from_fm_pickles(saved_fm_path, benchmark_path,
                                         compare_fms=True, compare_param_covs=True)

if fm_cfg['test_against_vincenzo'] and bnt_transform == False:
    fm_vinc_folder = fm_cfg["fm_vinc_folder"].format(**variable_specs, go_gs_vinc='GaussOnly')

    # for probe_vinc in ('WLO', 'GCO', '3x2pt'):
    for probe_vinc in ('3x2pt',):
        probe_dav = probe_vinc.replace('O', '')
        fm_vinc_filename = fm_cfg['fm_vinc_filename'].format(**variable_specs, probe=probe_vinc)
        fm_vinc_g = np.genfromtxt(f'{fm_vinc_folder}/{fm_vinc_filename}')

        diff = mm.percent_diff(fm_dict[f'FM_{probe_dav}_G'], fm_vinc_g)
        xticks = param_names_3x2pt
        plt.matshow(np.log10(np.abs(diff)))
        plt.colorbar()
        plt.xticks(np.arange(len(xticks)), xticks, rotation=90)

        mm.compare_arrays(fm_dict[f'FM_{probe_dav}_G'], fm_vinc_g, log_array=True, log_diff=False,
                          abs_val=False, plot_diff_threshold=5)

        npt.assert_allclose(fm_dict[f'FM_{probe_dav}_G'], fm_vinc_g, rtol=1e-3, atol=0)

# ! save/load outputs for tests
if isinstance(general_cfg['save_outputs_as_test_benchmarks_path'], str):

    print('Saving outputs as test benchmarks in folder ', general_cfg['save_outputs_as_test_benchmarks_path'])

    output_dict = {
        'original_cfg': original_cfg,
        'z_grid_ssc_integrands': z_grid_ssc_integrands,
        'delta': wf_delta,
        'gamma': wf_gamma,
        'ia': wf_ia,
        'mu': wf_mu,
        'lensing': wf_lensing,

        'ell_dict': ell_dict,
        'cl_ll_3d': cl_ll_3d,
        'cl_gl_3d': cl_gl_3d,
        'cl_gg_3d': cl_gg_3d,

        'sigma2_b': sigma2_b,
        'cov_dict': cov_dict,
        'fm_dict': fm_dict,

    }

    np.save(general_cfg['save_outputs_as_test_benchmarks_path'], output_dict, allow_pickle=True)

elif general_cfg['save_outputs_as_test_benchmarks_path'] is False:
    pass


# ! plot the results directly, as a quick check
nparams_toplot = 7
names_params_to_fix = []
divide_fom_by_10 = True
include_fom = True
which_uncertainty = 'conditional'

fix_dz = False
fix_shear_bias = False
fix_gal_bias = False
fix_mag_bias = False
shear_bias_prior = 5e-4
dz_prior = np.array(2 * 1e-3 * (1 + np.array(cfg['covariance_cfg']['zbin_centers'])))

probes = ['WL', 'GC', 'XC', '3x2pt']
dz_param_names = [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]
shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(4)]
mag_bias_param_names = [f'bM{(zi + 1):02d}' for zi in range(4)]
param_names_list = list(fid_pars_dict['FM_ordered_params'].keys())

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
fm_dict_toplot = deepcopy(fm_dict)
del fm_dict_toplot['fiducial_values_dict']
for key in list(fm_dict_toplot.keys()):
    if key != 'fiducial_values_dict' and '_WA_' not in key and '_2x2pt_' not in key:

        print(key)

        fm = deepcopy(fm_dict_toplot[key])

        masked_fm_dict[key], masked_fid_pars_dict[key] = mm.mask_fm_v2(fm, fid_pars_dict['FM_ordered_params'],
                                                                       names_params_to_fix=names_params_to_fix,
                                                                       remove_null_rows_cols=True)

        if not fix_shear_bias and any(item in key for item in ['WL', 'XC', '3x2pt']):
            print(f'adding shear bias Gaussian prior to {key}')
            shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
            masked_fm_dict[key] = mm.add_prior_to_fm(masked_fm_dict[key], masked_fid_pars_dict[key],
                                                     shear_bias_param_names, shear_bias_prior_values)

        if not fix_dz:
            print(f'adding dz Gaussian prior to {key}')
            masked_fm_dict[key] = mm.add_prior_to_fm(masked_fm_dict[key], masked_fid_pars_dict[key],
                                                     dz_param_names, dz_prior)

        uncert_dict[key] = mm.uncertainties_fm_v2(masked_fm_dict[key], masked_fid_pars_dict[key],
                                                  which_uncertainty=which_uncertainty,
                                                  normalize=True,
                                                  percent_units=True)[:nparams_toplot]

        param_names = list(masked_fid_pars_dict[key].keys())
        cosmo_param_names = list(masked_fid_pars_dict[key].keys())[:nparams_toplot]

        w0wa_idxs = param_names.index('wz'), param_names.index('wa')
        fom_dict[key] = mm.compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)

# compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
for probe in probes:

    key_a = f'FM_{probe}_G'
    key_b = f'FM_{probe}_GSSC'

    uncert_dict[f'perc_diff_{probe}_G'] = mm.percent_diff(uncert_dict[key_b], uncert_dict[key_a])
    fom_dict[f'perc_diff_{probe}_G'] = np.abs(mm.percent_diff(fom_dict[key_b], fom_dict[key_a]))

    nparams_toplot = 7
    divide_fom_by_10_plt = False if probe in ('WL' 'XC') else divide_fom_by_10

    cases_to_plot = [
        f'FM_{probe}_G',
        f'FM_{probe}_GSSC',
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

    # label and title stuff
    fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
    param_names_label = param_names_list[:nparams_toplot] + [fom_label] if include_fom else param_names_list[
        :nparams_toplot]
    lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_3x2pt']
    title = '%s %s, $\\ell_{\\rm max} = %i$, zbins %s%i, zsteps_ssc_integral %i $\\sigma_\\epsilon$ %s\n%s uncertainties' % (
        covariance_cfg['ng_cov_code'], probe,
        lmax, ep_or_ed, zbins, len(z_grid_ssc_integrands), covariance_cfg['which_shape_noise'], which_uncertainty)

    # bar plot
    if include_fom:
        nparams_toplot = 8

    for i, case in enumerate(cases_to_plot):

        cases_to_plot[i] = case
        if 'OneCovariance' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace('OneCovariance', 'OneCov')
        if f'PySSC_{probe}_G' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace(f'PySSC_{probe}_G', f'{probe}_G')

        cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
        cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')
        cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')

    plot_lib.bar_plot(uncert_array[:, :nparams_toplot], title, cases_to_plot, nparams=nparams_toplot,
                      param_names_label=param_names_label, bar_width=0.13, include_fom=include_fom, divide_fom_by_10_plt=divide_fom_by_10_plt)

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
path = '/home/davide/Documenti/Lavoro/Programmi/common_data/Spaceborne/jobs/SPV3/output/Flagship_2/FM/BNT_False/ell_cuts_False'
common_str = '_zbinsEP03_ML245_ZL02_MS245_ZS02_idIA2_idB3_idM3_idR1_pkHMCodeBar_13245deg2'


fm_dict_of_dicts = {
    # 'SB_su_fullsky': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_separateuniverse_fullcurvedsky.pickle'),
    # 'OC_hp': mm.load_pickle(f'{path}/FM_GSSC_OneCovariance{common_str}_highprecision.pickle'),
    # 'CCL': mm.load_pickle(f'{path}/FM_GSSC_PyCCL{common_str}.pickle'),
    # 'OC_hp': mm.load_pickle(f'{path}/FM_GSSC_OneCovariance{common_str}_highprecision.pickle'),
    # 'SB_hm': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_halo_model.pickle'),
    # 'OC_def': mm.load_pickle(f'{path}/FM_GSSC_OneCovariance{common_str}_defaultprecision.pickle'),
    # 'SB_su_maskotf': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_separateuniverse_polarcaponthefly.pickle'),
    # 'SB_suVin': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_vinSU_separateuniverse.pickle'),
    # 'SB_suDav_b2ghm': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_davSU_b2ghm_separateuniverse.pickle'),
    # 'SB_KEapp_su': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_Euclid_KE_approximation_separateuniverse.pickle'),
    # 'SB_KEapp_hm': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_Euclid_KE_approximation_halomodel.pickle'),
    'SB_hm_simpker': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_Euclid_simpkernTrue_sigma2bpolar_cap_on_the_fly_HM.pickle'),
    'SB_KEapp_hm_simpker': mm.load_pickle(f'{path}/FM_GSSC_Spaceborne{common_str}_Euclid_KE_approximation_simpkernTrue_sigma2bpolar_cap_on_the_fly_HM.pickle'),
    # 'OC_simpker': mm.load_pickle(f'{path}/FM_GSSC_OneCovariance{common_str}_Euclid_KE_approximation_simpkernTrue_sigma2bpolar_cap_on_the_fly.pickle'),
    'current': fm_dict
}


labels = list(fm_dict_of_dicts.keys())
fm_dict_list = list(fm_dict_of_dicts.values())
keys_toplot_in = ['FM_WL_GSSC', 'FM_GC_GSSC', 'FM_XC_GSSC', 'FM_3x2pt_GSSC']
# keys_toplot = 'all'
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:cyan', 'tab:grey', 'tab:olive', 'tab:purple']

reference = 'first_key'
nparams_toplot_in = 8
normalize_by_gauss = True

mm.compare_fm_constraints(*fm_dict_list, labels=labels,
                          keys_toplot_in=keys_toplot_in,
                          normalize_by_gauss=True,
                          which_uncertainty='marginal',
                          reference=reference,
                          colors=colors,
                          abs_FoM=True,
                          save_fig=False,
                          fig_path='/home/davide/Scrivania/')

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
param_names_list = list(fm_dict_of_dicts['SB_hm_simpker']['fiducial_values_dict'].keys())
param_names_labels_toplot = param_names_list[:8]
plot_lib.triangle_plot_new(fisher_matrices, fiducials, title, labels, param_names_list, param_names_labels_toplot, 
                  param_names_labels_tex=None, rotate_param_labels=False, contour_colors=None, line_colors=None)





print('Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60))
