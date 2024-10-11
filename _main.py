import os
import multiprocessing
import sys
num_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['NUMBA_NUM_THREADS'] = '32'
os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '4'

from tqdm import tqdm
from functools import partial
from collections import OrderedDict
import numpy as np
import time
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import warnings
import gc
import yaml
import argparse
import pprint
from copy import deepcopy
import numpy.testing as npt
from scipy.interpolate import interp1d, RegularGridInterpolator

import spaceborne.ell_utils as ell_utils
import spaceborne.cl_preprocessing as cl_utils
import spaceborne.covariance as covmat_utils
import spaceborne.fisher_matrix as fm_utils
import spaceborne.my_module as mm
import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.wf_cl_lib as wf_cl_lib
import spaceborne.pyccl_interface as pyccl_interface
import spaceborne.sigma2_SSC as sigma2_SSC
import spaceborne.onecovariance_interface as oc_interface
import spaceborne.config_checker as config_checker
import spaceborne.responses as responses

ROOT = os.getenv('ROOT')
script_start_time = time.perf_counter()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the Spaceborne analysis script.")
parser.add_argument('--nofigs', action='store_true', help="Use non-interactive backend for matplotlib to prevent figures from displaying")
args, unknown = parser.parse_known_args()

# Use 'Agg' backend if --nofigs flag is set
if args.nofigs:
    matplotlib.use('Agg')


def SSC_integral_julia(d2CLL_dVddeltab, d2CGL_dVddeltab, d2CGG_dVddeltab,
                       ind_auto, ind_cross, cl_integral_prefactor, sigma2, z_grid, integration_type, num_threads=16):
    """Kernel to compute the 4D integral optimized using Simpson's rule using Julia."""

    suffix = 0
    folder_name = 'tmp'
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
        f"julia --project=. --threads={num_threads} spaceborne/ssc_integral_julia.jl {folder_name} {integration_type}")

    cov_filename = "cov_SSC_spaceborne_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D.npy"

    if integration_type == 'trapz-6D':
        cov_ssc_3x2pt_dict_8D = {}  # it's 10D, actually
        for probe_a, probe_b in probe_ordering:
            for probe_c, probe_d in probe_ordering:
                if str.join('', (probe_a, probe_b, probe_c, probe_d)) not in ['GLLL', 'GGLL', 'GGGL']:
                    print(f"Loading {probe_a}{probe_b}{probe_c}{probe_d}")
                    cov_ssc_3x2pt_dict_8D[(probe_a, probe_b, probe_c, probe_d)] = np.load(
                        f"{folder_name}/{cov_filename.format(probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d)}")

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

# cfg = yaml.load(sys.stdin, Loader=yaml.FullLoader)

# if you want to run without arguments, uncomment the following lines
with open('example_cfg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

general_cfg = cfg['general_cfg']
covariance_cfg = cfg['covariance_cfg']
fm_cfg = cfg['FM_cfg']
pyccl_cfg = covariance_cfg['PyCCL_cfg']

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
shift_nz = covariance_cfg['shift_nz']  # ! vincenzo's kernels are shifted!
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

z_grid_ssc_integrands = np.linspace(covariance_cfg['Spaceborne_cfg']['z_min_ssc_integrands'],
                                    covariance_cfg['Spaceborne_cfg']['z_max_ssc_integrands'],
                                    covariance_cfg['Spaceborne_cfg']['z_steps_ssc_integrands'])
if len(z_grid_ssc_integrands) < 250:
    warnings.warn('z_grid_ssc_integrands is small, at the moment it used to compute various intermediate quantities')

which_ng_cov_suffix = 'G' + ''.join(covariance_cfg[covariance_cfg['ng_cov_code'] + '_cfg']['which_ng_cov'])
fid_pars_dict = cfg['cosmology']
flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)
h = flat_fid_pars_dict['h']

# load some nuisance parameters
# note that zbin_centers is not exactly equal to the result of wf_cl_lib.get_z_mean...
zbin_centers = cfg['covariance_cfg']['zbin_centers']
ngal_lensing = cfg['covariance_cfg']['ngal_lensing']
ngal_clustering = cfg['covariance_cfg']['ngal_clustering']
galaxy_bias_fit_fiducials = np.array([fid_pars_dict['FM_ordered_params'][f'bG{zi:02d}'] for zi in range(1, 5)])
magnification_bias_fit_fiducials = np.array([fid_pars_dict['FM_ordered_params'][f'bM{zi:02d}'] for zi in range(1, 5)])
dzWL_fiducial = np.array([fid_pars_dict['FM_ordered_params'][f'dzWL{zi:02d}'] for zi in range(1, zbins + 1)])
dzGC_fiducial = np.array([fid_pars_dict['FM_ordered_params'][f'dzWL{zi:02d}'] for zi in range(1, zbins + 1)])
warnings.warn('dzGC_fiducial are equal to dzWL_fiducial')

# some checks
config_checker = config_checker.SpaceborneConfigChecker(cfg)
k_txt_label, pk_txt_label = config_checker.run_all_checks()

# instantiate CCL object
ccl_obj = pyccl_interface.PycclClass(fid_pars_dict)


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
print('variable_specs:\n')
print(variable_specs)

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
ccl_obj.set_ia_bias_tuple(z_grid=z_grid_ssc_integrands)

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

# set Pk
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
plt.close()

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


wf_names_list = ['delta', 'gamma', 'ia', 'mu', 'lensing', gal_kernel_plt_title]
wf_ccl_list = [ccl_obj.wf_delta_arr, ccl_obj.wf_gamma_arr, ccl_obj.wf_ia_arr, ccl_obj.wf_mu_arr,
               ccl_obj.wf_lensing_arr, ccl_obj.wf_galaxy_arr]

# plot
for wf_idx in range(len(wf_ccl_list)):
    clr = cm.rainbow(np.linspace(0, 1, zbins))

    plt.figure()
    for zi in range(zbins):
        plt.plot(z_grid_ssc_integrands, wf_ccl_list[wf_idx][:, zi], ls="-", c=clr[zi], label=zi)

    plt.tight_layout()
    plt.xlabel('$z$')
    plt.ylabel('$W_i(z)$')
    plt.legend()
    plt.title(f'{wf_names_list[wf_idx]}')
    plt.show()
    plt.close()
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

cl_ll_3d, cl_gl_3d, cl_gg_3d, cl_wa_3d = ccl_obj.cl_ll_3d, ccl_obj.cl_gl_3d, ccl_obj.cl_gg_3d, ccl_obj.cl_wa_3d
cl_3x2pt_5d = ccl_obj.cl_3x2pt_5d


clr = cm.rainbow(np.linspace(0, 1, zbins))
fig, ax = plt.subplots(1, 3, figsize=(13, 5))
plt.tight_layout()

for zi in range(zbins):
    zj = zi
    ax[0].loglog(ell_dict['ell_WL'], cl_ll_3d[:, zi, zj], c=clr[zi])
    ax[1].loglog(ell_dict['ell_XC'], cl_gl_3d[:, zi, zj], c=clr[zi])
    ax[2].loglog(ell_dict['ell_GC'], cl_gg_3d[:, zi, zj], c=clr[zi])

ax[1].set_xlabel(r'$\ell$')
ax[0].set_ylabel(r'$C^{ii}_{\ell}$')
ax[0].set_title('LL')
ax[1].set_title('GL')
ax[2].set_title('GG')
plt.show()


# ! ========================================== SSC ============================================================


# ! ========================================== start Spaceborne ===================================================
cov_folder_sb = covariance_cfg['Spaceborne_cfg']['cov_path']
cov_sb_filename = covariance_cfg['Spaceborne_cfg']['cov_filename']
variable_specs['ng_cov_code'] = covariance_cfg['ng_cov_code']
variable_specs['which_ng_cov'] = which_ng_cov_suffix

if 'cNG' in covariance_cfg['Spaceborne_cfg']['which_ng_cov']:
    raise NotImplementedError('You should review the which_ng_cov arg in the cov_filename formatting above, "SSC" is'
                              'hardcoded at the moment')

if covariance_cfg['ng_cov_code'] == 'Spaceborne' and not covariance_cfg['Spaceborne_cfg']['load_precomputed_cov']:
    print('Start SSC computation with Spaceborne...')

    if covariance_cfg['Spaceborne_cfg']['which_pk_responses'] == 'halo_model':

        # ! 1. Get halo model responses from CCL
        ccl_obj.initialize_trispectrum(which_ng_cov='SSC', probe_ordering=probe_ordering,
                                       pyccl_cfg=pyccl_cfg, which_pk='_')

        # k and z grids (responses will be interpolated below)
        k_grid_resp = ccl_obj.responses_dict['L', 'L', 'L', 'L']['k_1overMpc']
        a_grid_resp = ccl_obj.responses_dict['L', 'L', 'L', 'L']['a_arr']
        # translate a to z and cut the arrays to the maximum redshift of the SU responses (much smaller range!)
        z_grid_resp = cosmo_lib.a_to_z(a_grid_resp)[::-1]

        dPmm_ddeltab = ccl_obj.responses_dict['L', 'L', 'L', 'L']['dpk12']
        dPgm_ddeltab = ccl_obj.responses_dict['L', 'L', 'G', 'L']['dpk34']
        dPgg_ddeltab = ccl_obj.responses_dict['G', 'G', 'G', 'G']['dpk12']

        # a is flipped w.r.t. z
        dPmm_ddeltab_hm = np.flip(dPmm_ddeltab, axis=1)
        dPgm_ddeltab_hm = np.flip(dPgm_ddeltab, axis=1)
        dPgg_ddeltab_hm = np.flip(dPgg_ddeltab, axis=1)

        # quick sanity check
        assert np.allclose(ccl_obj.responses_dict['L', 'L', 'G', 'L']['dpk34'],
                           ccl_obj.responses_dict['G', 'L', 'G', 'G']['dpk12'], atol=0, rtol=1e-5)
        assert np.allclose(ccl_obj.responses_dict['L', 'L', 'L', 'L']['dpk34'],
                           ccl_obj.responses_dict['L', 'L', 'L', 'L']['dpk12'], atol=0, rtol=1e-5)
        assert dPmm_ddeltab.shape == dPgm_ddeltab.shape == dPgg_ddeltab.shape, 'dPab_ddeltab_hm shape mismatch'

        dPmm_ddeltab_hm_func = RegularGridInterpolator((k_grid_resp, z_grid_resp), dPmm_ddeltab_hm, method='linear')
        dPgm_ddeltab_hm_func = RegularGridInterpolator((k_grid_resp, z_grid_resp), dPgm_ddeltab_hm, method='linear')
        dPgg_ddeltab_hm_func = RegularGridInterpolator((k_grid_resp, z_grid_resp), dPgg_ddeltab_hm, method='linear')
    # elif covariance_cfg['Spaceborne_cfg']['which_pk_responses'] == 'separate_universe':

        # import the response *coefficients* (not the responses themselves)
        su_responses_folder = covariance_cfg['Spaceborne_cfg']['separate_universe_responses_folder'].format(
            which_pk=general_cfg['which_pk'], ROOT=ROOT)
        su_responses_filename = covariance_cfg['Spaceborne_cfg']['separate_universe_responses_filename'].format(
            idBM=general_cfg['idBM'])
        rAB_of_k = np.genfromtxt(f'{su_responses_folder}/{su_responses_filename}')

        log_k_arr = np.unique(rAB_of_k[:, 0])
        k_grid_resp = 10 ** log_k_arr
        z_grid_resp = np.unique(rAB_of_k[:, 1])

        r_mm = np.reshape(rAB_of_k[:, 2], (len(k_grid_resp), len(z_grid_resp)))
        r_gm = np.reshape(rAB_of_k[:, 3], (len(k_grid_resp), len(z_grid_resp)))
        r_gg = np.reshape(rAB_of_k[:, 4], (len(k_grid_resp), len(z_grid_resp)))

        # remove z=0 and z = 0.01
        z_grid_resp = z_grid_resp[2:]
        r_mm = r_mm[:, 2:]
        r_gm = r_gm[:, 2:]
        r_gg = r_gg[:, 2:]

        # compute pk_mm on the responses' k, z grid to rescale them
        k_array, pk_mm_2d = cosmo_lib.pk_from_ccl(k_grid_resp, z_grid_resp, use_h_units,
                                                  ccl_obj.cosmo_ccl, pk_kind='nonlinear')

        # compute P_gm, P_gg
        gal_bias = ccl_obj.gal_bias_2d[:, 0]

        # check that it's the same in each bin
        for zi in range(zbins):
            np.testing.assert_allclose(ccl_obj.gal_bias_2d[:, 0], ccl_obj.gal_bias_2d[:, zi], atol=0, rtol=1e-5)

        gal_bias_func = interp1d(z_grid_ssc_integrands, gal_bias, kind='linear')
        gal_bias = gal_bias_func(z_grid_resp)

        pk_gm_2d = pk_mm_2d * gal_bias
        pk_gg_2d = pk_mm_2d * gal_bias ** 2

        # now turn the response coefficients into responses
        dPmm_ddeltab = r_mm * pk_mm_2d
        dPgm_ddeltab = r_gm * pk_gm_2d
        dPgg_ddeltab = r_gg * pk_gg_2d

        # from the exactSSC script
        folder = '/home/davide/Scrivania/check_responses_arfly'
        k_grid_sbload = np.load(f'{folder}/k_grid.npy')
        z_grid_sbload = np.load(f'{folder}/z_grid.npy')
        r_mm_sbload = np.load(f'{folder}/r1_mm.npy')
        r_gm_sbload = np.load(f'{folder}/r1_gm.npy')
        r_gg_sbload = np.load(f'{folder}/r1_gg.npy')
        r_gm_sbload_nob2 = np.load(f'{folder}/r1_gm_nob2.npy')
        r_gg_sbload_nob2 = np.load(f'{folder}/r1_gg_nob2.npy')
        b1_sbload = np.load(f'{folder}/b1_arr.npy')
        b2_sbload = np.load(f'{folder}/b2_arr.npy')
        pk_mm_2d_sbload = np.load(f'{folder}/pk_mm.npy')
        
        include_b2 = True
        if not covariance_cfg['Spaceborne_cfg']['include_b2']:
            r_gm_sbload = r_gm_sbload_nob2
            r_gg_sbload = r_gg_sbload_nob2

        # interpolate everything
        r1_mm_sbload_func = RegularGridInterpolator((k_grid_sbload, z_grid_sbload), r_mm_sbload, method='linear')
        r1_gm_sbload_func = RegularGridInterpolator((k_grid_sbload, z_grid_sbload), r_gm_sbload, method='linear')
        r1_gg_sbload_func = RegularGridInterpolator((k_grid_sbload, z_grid_sbload), r_gg_sbload, method='linear')
        pk_mm_sbload_func = RegularGridInterpolator((k_grid_sbload, z_grid_sbload), pk_mm_2d_sbload, method='linear')

        k_grid_resp_xx, z_grid_resp_yy = np.meshgrid(k_grid_resp, z_grid_resp, indexing='ij')
        r1_mm_sbload_interp = r1_mm_sbload_func((k_grid_resp_xx, z_grid_resp_yy))
        r1_gm_sbload_interp = r1_gm_sbload_func((k_grid_resp_xx, z_grid_resp_yy))
        r1_gg_sbload_interp = r1_gg_sbload_func((k_grid_resp_xx, z_grid_resp_yy))
        pk_mm_sbload_interp = pk_mm_sbload_func((k_grid_resp_xx, z_grid_resp_yy))
        
        # interpolate HM
        dPmm_ddeltab_hm_interp = dPmm_ddeltab_hm_func((k_grid_resp_xx, z_grid_resp_yy))
        dPgm_ddeltab_hm_interp = dPgm_ddeltab_hm_func((k_grid_resp_xx, z_grid_resp_yy))
        dPgg_ddeltab_hm_interp = dPgg_ddeltab_hm_func((k_grid_resp_xx, z_grid_resp_yy))
        r_mm_hm = dPmm_ddeltab_hm_interp / pk_mm_2d
        r_gm_hm = dPgm_ddeltab_hm_interp / pk_gm_2d
        r_gg_hm = dPgg_ddeltab_hm_interp / pk_gg_2d
        
        # further check: resp dav from new class
        resp_obj = responses.SpaceborneResponses(cfg=cfg, k_grid=k_grid_resp, 
                                                 z_grid=z_grid_resp, 
                                                 cosmo_ccl=ccl_obj.cosmo_ccl, 
                                                 b1_func=ccl_obj.gal_bias_func_ofz)
        r_mm_sbclass = resp_obj.compute_r1_mm()
        resp_obj.get_rab_and_dpab_ddeltab()
        
        r_gm_sbclass = resp_obj.r1_gm
        r_gg_sbclass = resp_obj.r1_gg
        if not covariance_cfg['Spaceborne_cfg']['include_b2']:
            r_gm_sbclass = resp_obj.r1_gm_nob2
            r_gg_sbclass = resp_obj.r1_gg_nob2
        

        z_idx = 0
        k_idx = 0
        # plt.semilogx(k_grid_resp, r1_mm_sbload_interp[:, z_idx], label=f'r1_mm_sbload_interp nob2{nob2}', c='tab:blue', ls=':)
        plt.semilogx(k_grid_resp, r_mm_sbclass[:, z_idx], label=f'r_mm_sbclass includeb2{include_b2}', c='tab:blue', ls='-.')
        plt.semilogx(k_grid_resp, r_mm_hm[:, z_idx], label='r_mm_hm', c='tab:blue', ls='-')
        plt.semilogx(k_grid_resp, r_mm[:, z_idx], label='r_mm vin', c='tab:blue', ls='--')

        # plt.semilogx(k_grid_resp, r1_gm_sbload_interp[:, z_idx], c='tab:orange', ls=':)
        plt.semilogx(k_grid_resp, r_gm_sbclass[:, z_idx], c='tab:orange', ls='-.')
        plt.semilogx(k_grid_resp, r_gm_hm[:, z_idx], c='tab:orange', ls='-')
        plt.semilogx(k_grid_resp, r_gm[:, z_idx], c='tab:orange', ls='--')

        # plt.semilogx(k_grid_resp, r1_gg_sbload_interp[:, z_idx], c='tab:green', ls=':)
        plt.semilogx(k_grid_resp, r_gg_sbclass[:, z_idx], c='tab:green', ls='-.')
        plt.semilogx(k_grid_resp, r_gg_hm[:, z_idx], c='tab:green', ls='-')
        plt.semilogx(k_grid_resp, r_gg[:, z_idx], c='tab:green', ls='--')
        
        # legend for the different linestyles
        # Custom legend for line styles
        plt.legend()
        plt.xlabel(f'k {k_txt_label}')
        plt.ylabel(r'$R_{AB}(k)$')
        
        plt.ylim(-5, 5)
        plt.title(f'z={z_grid_resp[z_idx]}')

        np.testing.assert_allclose(r_mm_sbclass, r1_mm_sbload_interp, atol=0, rtol=1e-8)
        np.testing.assert_allclose(r_gm_sbclass, r1_gm_sbload_interp, atol=0, rtol=1e-8)
        np.testing.assert_allclose(r_gg_sbclass, r1_gg_sbload_interp, atol=0, rtol=1e-8)
        np.testing.assert_allclose(pk_mm_2d, resp_obj.pk_mm, atol=0, rtol=1e-8)
        np.testing.assert_allclose(pk_mm_2d, pk_mm_2d_sbload, atol=0, rtol=1e-8)
        
        
        plt.plot(z_grid_sbload, b1_sbload[0, :], label='b1_dav', c='tab:blue')
        plt.plot(z_grid_resp, resp_obj.b1_arr[0, :], label='b1_dav', c='tab:blue')
        
        assert False, 'stop to check responses'

        

    else:
        raise ValueError('which_pk_responses must be either "halo_model" or "separate_universe"')

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

    # ! compute the Pk responses(k, z) in k_limber and z_grid_ssc_integrands
    dPmm_ddeltab_interp = RegularGridInterpolator((k_grid_resp, z_grid_resp), dPmm_ddeltab, method='linear')
    dPgm_ddeltab_interp = RegularGridInterpolator((k_grid_resp, z_grid_resp), dPgm_ddeltab, method='linear')
    dPgg_ddeltab_interp = RegularGridInterpolator((k_grid_resp, z_grid_resp), dPgg_ddeltab, method='linear')

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
    which_sigma2_B = covariance_cfg['Spaceborne_cfg']['which_sigma2_B']

    sigma2_b_path = covariance_cfg['Spaceborne_cfg']['sigma2_b_path']
    sigma2_b_filename = covariance_cfg['Spaceborne_cfg']['sigma2_b_filename']
    if covariance_cfg['Spaceborne_cfg']['load_precomputed_sigma2']:
        # TODO define a suitable interpolator if the zgrid doesn't match
        sigma2_b_dict = np.load(f'{sigma2_b_path}/{sigma2_b_filename}', allow_pickle=True).item()
        cfg_sigma2_b = sigma2_b_dict['cfg']  # TODO check that the cfg matches the one
        sigma2_b = sigma2_b_dict['sigma2_b']
    else:
        # TODO input ell and cl mask
        print('Computing sigma2_b...')
        sigma2_b = np.zeros((len(z_grid_ssc_integrands), len(z_grid_ssc_integrands)))
        for z2_idx, z2 in enumerate(tqdm(z_grid_ssc_integrands)):
            sigma2_b[:, z2_idx] = sigma2_SSC.sigma2_func_vectorized(
                z1_arr=z_grid_ssc_integrands,
                z2=z2, k_grid_sigma2=k_grid_sigma2,
                cosmo_ccl=ccl_obj.cosmo_ccl,
                which_sigma2_B=which_sigma2_B,
                ell_mask=None, cl_mask=None)

        sigma2_b_dict_tosave = {
            'cfg': cfg,
            'sigma2_b': sigma2_b,
        }
        np.save(f'{sigma2_b_path}/{sigma2_b_filename}', sigma2_b_dict_tosave, allow_pickle=True)

    mm.matshow(sigma2_b, log=True, abs_val=True, title=r'$\sigma^2_B(z_1, z_2)$')

    plt.figure()
    plt.semilogy(z_grid_ssc_integrands, np.diag(sigma2_b))
    plt.xlabel('$z$')
    plt.ylabel(r'$\sigma^2_B(z_1=z_2)$')
    plt.close()


    z1_idx = len(z_grid_ssc_integrands) // 2
    z1_val = z_grid_ssc_integrands[z1_idx]
    plt.figure()
    plt.plot(z_grid_ssc_integrands, sigma2_b[z1_idx, :])
    plt.xlabel('$z$')
    plt.ylabel(r'$\sigma^2_B(z_2, z1=%.3f)$' % z1_val)
    plt.close()

    # ! 4. Perform the integration calling the Julia module
    print('Performing the 2D integral in Julia...')
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

    # If the mask is not passed to sigma2_b, we need to divide by fsky
    if which_sigma2_B == 'full-curved-sky':
        for key in cov_ssc_3x2pt_dict_8D.keys():
            cov_ssc_3x2pt_dict_8D[key] /= covariance_cfg['fsky']
    elif which_sigma2_B == 'mask':
        raise NotImplementedError('Not implemented yet, but very easy to do')
    else:
        raise ValueError(f'which_sigma2_B must be either "full-curved-sky" or "mask"')

    # save the covariance blocks
    # ! note that these files already account for the sky fraction!!
    # TODO fsky suffix in cov name should be added only in this case... or not? the other covariance files don't have this...
    for key in cov_ssc_3x2pt_dict_8D.keys():
        probe_a, probe_b, probe_c, probe_d = key
        if str.join('', (probe_a, probe_b, probe_c, probe_d)) not in ['GLLL', 'GGLL', 'GGGL']:
            np.savez_compressed(
                f'{cov_folder_sb}/{cov_sb_filename.format(probe_a=probe_a,
                                                          probe_b=probe_b, probe_c=probe_c, probe_d=probe_d)}',
                cov_ssc_3x2pt_dict_8D[key])

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
        print('Changing ellmax to 5000 and cutting the last bins')
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

symmetrize_output_dict = {
    ('L', 'L'): False,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): False,
}
cov_ssc_sb_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
    covariance_cfg['cov_ssc_3x2pt_dict_8D_sb'], nbl_3x2pt, zbins, ind_dict, probe_ordering, symmetrize_output_dict)
cov_ssc_sb_3x2pt_10D = mm.cov_10D_dict_to_array(cov_ssc_sb_3x2pt_dict_10D, nbl_3x2pt, zbins, n_probes)
cov_3x2pt_SS_10D = cov_ssc_sb_3x2pt_10D

cov_3x2pt_SS_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SS_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(), GL_or_LG)

# to see the individual covariance blocks
cov_3x2pt_SS_2D = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_SS_4D, block_index='ell', zbins=zbins)

mm.matshow(cov_3x2pt_SS_2D, log=True, abs_val=True, title='SSC 2D')

assert False, 'stop here for now'


# ! ========================================== OneCovariance ===================================================

start_time = time.perf_counter()
if covariance_cfg['ng_cov_code'] == 'OneCovariance' or \
    (covariance_cfg['ng_cov_code'] == 'Spaceborne' and
        not covariance_cfg['OneCovariance_cfg']['use_OneCovariance_SSC']):

    # * 1. save ingredients in ascii format
    oc_path = covariance_cfg['OneCovariance_cfg']['onecovariance_folder'].format(ROOT=ROOT, **variable_specs)
    if not os.path.exists(oc_path):
        os.makedirs(oc_path)

    nofz_ascii_filename = nofz_filename.replace('.dat', f'_dzshifts{shift_nz}.ascii')
    nofz_tosave = np.column_stack((zgrid_nz, n_of_z))
    np.savetxt(f'{oc_path}/{nofz_ascii_filename}', nofz_tosave)

    cl_ll_ascii_filename = f'Cell_ll_SPV3_nbl{nbl_3x2pt}'
    cl_gl_ascii_filename = f'Cell_gl_SPV3_nbl{nbl_3x2pt}'
    cl_gg_ascii_filename = f'Cell_gg_SPV3_nbl{nbl_3x2pt}'
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
    print('Start NG cov computation with OneCovariance...')

    # TODO this should be defined globally...
    symmetrize_output_dict = {
        ('L', 'L'): False,
        ('G', 'L'): False,
        ('L', 'G'): False,
        ('G', 'G'): False,
    }

    oc_obj = oc_interface.OneCovarianceInterface(ROOT, cfg, variable_specs)
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
if general_cfg['cl_ell_cuts']:
    cl_ll_3d = cl_utils.cl_ell_cut(cl_ll_3d, ell_dict['ell_WL'], ell_cuts_dict['LL'])
    cl_gg_3d = cl_utils.cl_ell_cut(cl_gg_3d, ell_dict['ell_GC'], ell_cuts_dict['GG'])
    cl_3x2pt_5d = cl_utils.cl_ell_cut_3x2pt(cl_3x2pt_5d, ell_cuts_dict, ell_dict['ell_3x2pt'])

# TODO delete this

# store cls and responses in a dictionary
cl_dict_3D = {
    'cl_LL_3D': cl_ll_3d,
    'cl_GG_3D': cl_gg_3d,
    'cl_WA_3D': cl_wa_3d,
    'cl_3x2pt_5D': cl_3x2pt_5d}

rl_dict_3D = {
    'rl_LL_3D': np.ones_like(cl_ll_3d),
    'rl_GG_3D': np.ones_like(cl_gg_3d),
    'rl_WA_3D': np.ones_like(cl_wa_3d),
    'rl_3x2pt_5D': np.ones_like(cl_3x2pt_5d)}

# this is again to test against ccl cls
general_cfg['cl_ll_3d'] = cl_ll_3d
general_cfg['cl_gl_3d'] = cl_gl_3d
general_cfg['cl_gg_3d'] = cl_gg_3d


# ! compute covariance matrix
cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                    ell_dict, delta_dict, cl_dict_3D, rl_dict_3D,
                                    Sijkl=None, BNT_matrix=bnt_matrix, oc_obj=oc_obj)

if covariance_cfg['test_against_benchmarks']:
    cov_benchmark_folder = f'{cov_folder}/benchmarks'
    mm.test_folder_content(cov_folder, cov_benchmark_folder, covariance_cfg['cov_file_format'])


# !============================= derivatives ===================================
# this guard is just to avoid indenting the whole code below
if not fm_cfg['compute_FM']:
    del cov_dict
    gc.collect()
    raise KeyboardInterrupt('skipping FM computation, the script will exit now')

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


# store the derivatives arrays in a dictionary
deriv_dict = {'dC_LL_4D': dC_LL_4D,
              'dC_WA_4D': dC_WA_4D,
              'dC_GG_4D': dC_GG_4D,
              'dC_3x2pt_6D': dC_3x2pt_6D}


# ! ==================================== compute and save fisher matrix ================================================
fm_dict_vin = fm_utils.compute_FM(cfg, ell_dict, cov_dict, deriv_dict, bnt_matrix)

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
    fm_dict_filename = fm_cfg['fm_dict_filename'].format(
        **variable_specs, fm_and_cov_suffix=general_cfg['fm_and_cov_suffix'],
        lmax=ell_max_3x2pt, survey_area_deg2=covariance_cfg['survey_area_deg2'])
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

print('Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60))
