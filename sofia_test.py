import os
import multiprocessing
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
import matplotlib.pyplot as plt
import warnings
import gc
import yaml
import pprint
from copy import deepcopy
import numpy.testing as npt
from scipy.interpolate import interp1d, RegularGridInterpolator
from tabulate import tabulate

import spaceborne.ell_utils as ell_utils
import spaceborne.cl_preprocessing as cl_utils
import spaceborne.compute_Sijkl as Sijkl_utils
import spaceborne.covariance as covmat_utils
import spaceborne.fisher_matrix as fm_utils
import spaceborne.my_module as mm
import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.wf_cl_lib as wf_cl_lib
import spaceborne.pyccl_cov_class as pyccl_cov_class
import spaceborne.plot_lib as plot_lib
import spaceborne.sigma2_SSC as sigma2_SSC
import spaceborne.onecovariance_interface as oc_interface
import spaceborne.config_checker as config_checker

ROOT = '/Users/sofiachiarenza/Desktop/PhD_Stuff'
script_start_time = time.perf_counter()

#copied the same import as main.py
#Now read the same input file: 

with open('sofia_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

general_cfg = cfg['general_cfg']
covariance_cfg = cfg['covariance_cfg']
fm_cfg = cfg['FM_cfg']
pyccl_cfg = covariance_cfg['PyCCL_cfg']

# some convenience variables, just to make things more readable
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

R_grid_ssc_integrands = np.linspace(covariance_cfg['Spaceborne_cfg']['R_min_ssc_integrands'],
                                    covariance_cfg['Spaceborne_cfg']['R_max_ssc_integrands'],
                                    covariance_cfg['Spaceborne_cfg']['R_steps_ssc_integrands'])


################################ Presettings ###################################
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
ccl_obj = pyccl_cov_class.PycclClass(fid_pars_dict)


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




#now onto the SSC part with spaceborne
cov_folder_sb = covariance_cfg['Spaceborne_cfg']['cov_path']
cov_sb_filename = covariance_cfg['Spaceborne_cfg']['cov_filename']
variable_specs['ng_cov_code'] = covariance_cfg['ng_cov_code']
variable_specs['which_ng_cov'] = which_ng_cov_suffix


if covariance_cfg['ng_cov_code'] == 'Spaceborne' and not covariance_cfg['Spaceborne_cfg']['load_precomputed_cov']:
    print('Start SSC computation with Spaceborne...')

    if covariance_cfg['Spaceborne_cfg']['which_pk_responses'] == 'halo_model':

        print("Dummy, use separate_universe!")

    elif covariance_cfg['Spaceborne_cfg']['which_pk_responses'] == 'separate_universe':

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

    else:
        raise ValueError('which_pk_responses must be "separate_universe"')
    
    #Here i took out lines that prepare everything to perform the outer integration. 
    # Once you wor that out, put them back in
    # now I'm only looking at inner k integral:

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
        sigma2_b = sigma2_SSC.sigma2_sofia(
                z=z_grid_ssc_integrands,
                R=R_grid_ssc_integrands, k_grid_sigma2=k_grid_sigma2,
                cosmo_ccl=ccl_obj.cosmo_ccl,
                which_sigma2_B=which_sigma2_B,
                ell_mask=None, cl_mask=None)

        sigma2_b_dict_tosave = {
            'cfg': cfg,
            'sigma2_b': sigma2_b,
        }
        #np.save(f'tests/{sigma2_b_filename}', sigma2_b_dict_tosave, allow_pickle=True)

    print(np.shape(sigma2_b))
    #plt.imshow(np.flip(sigma2_b, axis=(0, 1)).T, aspect='equal', cmap='viridis')
    mm.matshow(np.flip(sigma2_b, axis=(0, 1)).T, log=True, abs_val=True, title='$\sigma^2_B(z, Rz)$')
    #TODO: capire come plottare bene sta roba, mi manca Julia :(((((

    sigma2_davide = np.load("davide.npy")

    plt.figure()
    plt.semilogy(z_grid_ssc_integrands, sigma2_b[:,-1], label="Me")
    plt.semilogy(z_grid_ssc_integrands, np.diag(sigma2_davide), label="Davide")
    plt.xlabel('$z$')
    plt.legend()
    plt.title(r'$\sigma^2_B(R=1)$')
    plt.show()

    plt.plot(z_grid_ssc_integrands, 100*(1 - sigma2_b[:,-1]/np.diag(sigma2_davide)))
    plt.title("Residuals")
    plt.xlabel('$z$')
    plt.show()


    plt.figure()
    plt.plot(R_grid_ssc_integrands[1:], sigma2_b[50,1:])
    plt.xlabel('$R$')
    plt.title(f'R_cut at fixed z ={z_grid_ssc_integrands[50]}')
    plt.show()

    

    




