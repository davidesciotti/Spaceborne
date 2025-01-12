import argparse
import os
import multiprocessing
from tqdm import tqdm
num_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(num_cores)
from functools import partial
import numpy as np
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
import yaml
import pprint
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from spaceborne import ell_utils
from spaceborne import cl_utils
from spaceborne import bnt
from spaceborne import sb_lib as sl
from spaceborne import cosmo_lib
from spaceborne import wf_cl_lib
from spaceborne import pyccl_interface
from spaceborne import sigma2_SSC
from spaceborne import config_checker
from spaceborne import onecovariance_interface as oc_interface
from spaceborne import responses
from spaceborne import covariance as sb_cov

pp = pprint.PrettyPrinter(indent=4)
script_start_time = time.perf_counter()

# ! Set up argument parsing
# parser = argparse.ArgumentParser(description="Your script description here.")
# parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
# # parser.add_argument('--show_plots', action='store_true', help='Show plots if specified',  required=False)
# args = parser.parse_args()
# with open(args.config, 'r') as f:
#     cfg = yaml.safe_load(f)
# # if not args.show_plots:
# #     matplotlib.use('Agg')

# ! LOAD CONFIG
# ! uncomment this if executing from interactive window
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# some convenence variables, just to make things more readable
h = cfg['cosmology']['h']
galaxy_bias_fit_fiducials = np.array(cfg['C_ell']['galaxy_bias_fit_coeff'])
magnification_bias_fit_fiducials = np.array(cfg['C_ell']['magnification_bias_fit_coeff'])
dzWL_fiducial = cfg['nz']['dzWL']
dzGC_fiducial = cfg['nz']['dzGC']
shift_nz_interpolation_kind = cfg['nz']['shift_nz_interpolation_kind']
nz_gaussian_smoothing = cfg['nz']['nz_gaussian_smoothing']  # does not seem to have a large effect...
nz_gaussian_smoothing_sigma = cfg['nz']['nz_gaussian_smoothing_sigma']
shift_nz = cfg['nz']['shift_nz']
normalize_shifted_nz = cfg['nz']['normalize_shifted_nz']
zbins = len(cfg['nz']['ngal_lenses'])  # this has the same length as ngal_sources, as checked below
ell_max_WL = cfg['ell_binning']['ell_max_WL']
ell_max_GC = cfg['ell_binning']['ell_max_GC']
ell_max_3x2pt = cfg['ell_binning']['ell_max_3x2pt']
nbl_WL_opt = cfg['ell_binning']['nbl_WL_opt']
triu_tril = cfg['covariance']['triu_tril']
row_col_major = cfg['covariance']['row_col_major']
n_probes = cfg['covariance']['n_probes']
which_sigma2_b = cfg['covariance']['which_sigma2_b']
z_steps_ssc_integrands = cfg['covariance']['z_steps_ssc_integrands']
include_ia_in_bnt_kernel_for_zcuts = cfg['BNT']['include_ia_in_bnt_kernel_for_zcuts']
compute_bnt_with_shifted_nz_for_zcuts = cfg['BNT']['compute_bnt_with_shifted_nz_for_zcuts']
probe_ordering = cfg['covariance']['probe_ordering']
GL_OR_LG = probe_ordering[1][0] + probe_ordering[1][1]
EP_OR_ED = cfg['nz']['EP_or_ED']
output_path = cfg['misc']['output_path']
clr = cm.rainbow(np.linspace(0, 1, zbins))

if not os.path.exists(output_path):
    raise FileNotFoundError(f"Output path {output_path} does not exist. Please create it before running the script.")
if not os.path.exists(f'{output_path}/cache'):
    os.mkdir(f'{output_path}/cache')

# ! START HARDCODED OPTIONS/PARAMETERS
use_h_units = False  # whether or not to normalize Megaparsecs by little h
# number of ell bins over which to compute the Cls passed to OC for the Gaussian covariance computation
nbl_3x2pt_oc = 500

# whether or not to symmetrize the covariance probe blocks when reshaping it from 4D to 6D.
# Useful if the 6D cov elements need to be accessed directly, whereas if the cov is again reduced to 4D or 2D
# can be set to False for a significant speedup
symmetrize_output_dict = {
    ('L', 'L'): False,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): False,
}
# ! END HARDCODED OPTIONS/PARAMETERS

# ! set non-gaussian cov terms to compute
cov_terms_list = []
if cfg['covariance']['G']:
    cov_terms_list.append("G")
if cfg['covariance']['SSC']:
    cov_terms_list.append("SSC")
if cfg['covariance']['cNG']:
    cov_terms_list.append("cNG")
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
    k_txt_label = "hoverMpc"
    pk_txt_label = "Mpcoverh3"
else:
    k_txt_label = "1overMpc"
    pk_txt_label = "Mpc3"

if not cfg['ell_cuts']['apply_ell_cuts']:
    kmax_h_over_Mpc = cfg['ell_cuts']['kmax_h_over_Mpc_ref']

# ! define grids for the SSC integrands
z_grid_ssc_integrands = np.linspace(cfg['covariance']['z_min_ssc_integrands'],
                                    cfg['covariance']['z_max_ssc_integrands'],
                                    cfg['covariance']['z_steps_ssc_integrands'])
k_grid_resp = np.geomspace(cfg['PyCCL']['k_grid_tkka_min'],
                           cfg['PyCCL']['k_grid_tkka_max'],
                           cfg['PyCCL']['k_grid_tkka_steps_SSC'])


# ! sanity checks on the configs
# TODO update this when cfg are done
cfg_check_obj = config_checker.SpaceborneConfigChecker(cfg)
cfg_check_obj.run_all_checks()

if len(z_grid_ssc_integrands) < 250:
    warnings.warn('z_grid_ssc_integrands is small, at the moment it used to compute various intermediate quantities')

# ! instantiate CCL object
ccl_obj = pyccl_interface.PycclClass(cfg['cosmology'], cfg['extra_parameters'],
                                     cfg['intrinsic_alignment'], cfg['halo_model'])
ccl_obj.p_of_k_a = 'delta_matter:delta_matter'
ccl_obj.zbins = zbins
ccl_obj.which_b1g_in_resp = cfg['covariance']['which_b1g_in_resp']
a_default_grid_ccl = ccl_obj.cosmo_ccl.get_pk_spline_a()
z_default_grid_ccl = cosmo_lib.a_to_z(a_default_grid_ccl)[::-1]
# TODO class to access CCL precision parameters

# build the ind array and store it into the covariance dictionary
zpairs_auto, zpairs_cross, zpairs_3x2pt = sl.get_zpairs(zbins)
ind = sl.build_full_ind(triu_tril, row_col_major, zbins)
ind_auto = ind[:zpairs_auto, :].copy()
ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()
ind_dict = {('L', 'L'): ind_auto,
            ('G', 'L'): ind_cross,
            ('G', 'G'): ind_auto}

# ! Import redshift distributions
# The shape of these input files should be `(zpoints, zbins + 1)`, with `zpoints` the number of points over which the
# distribution is measured and zbins the number of redshift bins. The first column should contain the redshifts values.
# We also define:
# - `nz_full`: nz table including a column for the z values
# - `nz`:      nz table excluding a column for the z values
# - `nz_original`: nz table as imported (it may be subjected to shifts later on)
nz_src_tab_full = np.genfromtxt(cfg["nz"]["nz_sources_filename"])
nz_lns_tab_full = np.genfromtxt(cfg["nz"]["nz_lenses_filename"])
zgrid_nz_src = nz_src_tab_full[:, 0]
zgrid_nz_lns = nz_lns_tab_full[:, 0]
nz_src = nz_src_tab_full[:, 1:]
nz_lns = nz_lns_tab_full[:, 1:]

# nz may be subjected to a shift
nz_unshifted_src = nz_src
nz_unshifted_lns = nz_lns

wf_cl_lib.plot_nz_src_lns(zgrid_nz_src, nz_src, zgrid_nz_lns, nz_lns, colors=clr)

# ! compute ell values, ell bins and delta ell
# compute ell and delta ell values in the reference (optimistic) case
ell_ref_nbl32, delta_l_ref_nbl32, ell_edges_ref_nbl32 = (
    ell_utils.compute_ells(cfg['ell_binning']['nbl_WL_opt'], cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_WL_opt'],
                           recipe='ISTF', output_ell_bin_edges=True))

# perform the cuts (not the redshift-dependent ones!) on the ell centers and edges
ell_dict = {}
ell_dict['ell_WL'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_WL])
ell_dict['ell_GC'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_GC])
ell_dict['ell_3x2pt'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_3x2pt])
ell_dict['ell_XC'] = np.copy(ell_dict['ell_3x2pt'])

# TODO why not save all edges??
# store edges *except last one for dimensional consistency* in the ell_dict
ell_dict['ell_edges_WL'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_WL])
ell_dict['ell_edges_GC'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_GC])
ell_dict['ell_edges_3x2pt'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_3x2pt])
ell_dict['ell_edges_XC'] = np.copy(ell_dict['ell_edges_3x2pt'])

for key in ell_dict.keys():
    assert ell_dict[key].size > 0, f'ell values for key {key} must be non-empty'
    assert np.max(ell_dict[key]) > 15, f'ell values for key {key} must *not* be in log space'

# set the corresponding number of ell bins
nbl_WL = len(ell_dict['ell_WL'])
nbl_GC = len(ell_dict['ell_GC'])
nbl_3x2pt = nbl_GC

assert len(ell_dict['ell_3x2pt']) == len(ell_dict['ell_XC']) == len(ell_dict['ell_GC']), '3x2pt, XC and GC should '\
    ' have the same number of ell bins'
assert np.all(ell_dict['ell_3x2pt'] == ell_dict['ell_XC']), '3x2pt and XC should have the same ell values'
assert np.all(ell_dict['ell_3x2pt'] == ell_dict['ell_GC']), '3x2pt and GC should have the same ell values'
assert nbl_WL == nbl_3x2pt == nbl_GC, 'use the same number of bins for the moment'

# delta_ell values, needed for gaussian covariance (if binned in this way)
ell_dict['delta_l_WL'] = np.copy(delta_l_ref_nbl32[:nbl_WL])
ell_dict['delta_l_GC'] = np.copy(delta_l_ref_nbl32[:nbl_GC])

# provate cfg dictionary. This serves a couple different purposeses:
# 1. To store and pass hardcoded parameters in a convenient way
# 2. To make the .format() more compact
pvt_cfg = {
    'EP_OR_ED': EP_OR_ED,
    'zbins': zbins,
    'ind': ind,
    'probe_ordering': probe_ordering,
    'ell_min': cfg['ell_binning']['ell_min'],
    'ell_max_WL': ell_max_WL, 'ell_max_GC': ell_max_GC, 'ell_max_3x2pt': ell_max_3x2pt,
    'nbl_WL': nbl_WL, 'nbl_GC': nbl_GC, 'nbl_3x2pt': nbl_3x2pt,
    'which_ng_cov': cov_terms_str,
    'cov_terms_list': cov_terms_list,
    'GL_OR_LG': GL_OR_LG,
    'EP_OR_ED': EP_OR_ED,
    'symmetrize_output_dict': symmetrize_output_dict,
    'use_h_units': use_h_units,
    'z_grid_ssc_integrands': z_grid_ssc_integrands,
    'ells_sb': ell_dict['ell_3x2pt'],
}

# TODO delete this? maybe I still want to print some of these options...
# pp.pprint(pvt_cfg)


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
    nz_src = wf_cl_lib.gaussian_smmothing_nz(zgrid_nz_src, nz_unshifted_src, nz_gaussian_smoothing_sigma, plot=True)
    nz_lns = wf_cl_lib.gaussian_smmothing_nz(zgrid_nz_lns, nz_unshifted_lns, nz_gaussian_smoothing_sigma, plot=True)

if compute_bnt_with_shifted_nz_for_zcuts:
    nz_src = wf_cl_lib.shift_nz(zgrid_nz_src, nz_unshifted_src, dzWL_fiducial, normalize=normalize_shifted_nz,
                                plot_nz=False, interpolation_kind=shift_nz_interpolation_kind,
                                bounds_error=False, fill_value=0)
    nz_lns = wf_cl_lib.shift_nz(zgrid_nz_lns, nz_unshifted_lns, dzGC_fiducial, normalize=normalize_shifted_nz,
                                plot_nz=False, interpolation_kind=shift_nz_interpolation_kind,
                                bounds_error=False, fill_value=0)

bnt_matrix = bnt.compute_bnt_matrix(
    zbins, zgrid_nz_src, nz_src, cosmo_ccl=ccl_obj.cosmo_ccl, plot_nz=False)

# 2. compute the kernels for the un-shifted n(z) (for consistency)
ccl_obj.set_nz(nz_full_src=np.hstack((zgrid_nz_src[:, None], nz_src)),
               nz_full_lns=np.hstack((zgrid_nz_lns[:, None], nz_lns)))
ccl_obj.check_nz_tuple(zbins)
ccl_obj.set_ia_bias_tuple(z_grid_src=z_grid_ssc_integrands, has_ia=cfg['C_ell']['has_IA'])

# ! set galaxy and magnification bias
if cfg['C_ell']['which_gal_bias'] == 'from_input':
    gal_bias_tab_full = np.genfromtxt(cfg['C_ell']['gal_bias_table_filename'])
    gal_bias_tab = sl.check_interpolate_input_tab(gal_bias_tab_full, z_grid_ssc_integrands, zbins)
    ccl_obj.gal_bias_tuple = (z_grid_ssc_integrands, gal_bias_tab)
    ccl_obj.gal_bias_2d = gal_bias_tab
elif cfg['C_ell']['which_gal_bias'] == 'FS2_polynomial_fit':
    ccl_obj.set_gal_bias_tuple_spv3(z_grid_lns=z_grid_ssc_integrands,
                                    magcut_lens=None,
                                    poly_fit_values=galaxy_bias_fit_fiducials)
else:
    raise ValueError('which_gal_bias should be "from_input" or "FS2_polynomial_fit"')

if cfg['C_ell']['has_magnification_bias']:

    if cfg['C_ell']['which_mag_bias'] == 'from_input':
        mag_bias_tab_full = np.genfromtxt(cfg['C_ell']['mag_bias_table_filename'])
        mag_bias_tab = sl.check_interpolate_input_tab(mag_bias_tab_full, z_grid_ssc_integrands, zbins)
        ccl_obj.mag_bias_tuple = (z_grid_ssc_integrands, mag_bias_tab)
    elif cfg['C_ell']['which_mag_bias'] == 'FS2_polynomial_fit':
        ccl_obj.set_mag_bias_tuple(z_grid_lns=z_grid_ssc_integrands,
                                   has_magnification_bias=cfg['C_ell']['has_magnification_bias'],
                                   magcut_lens=None,
                                   poly_fit_values=magnification_bias_fit_fiducials)
    else:
        raise ValueError('which_mag_bias should be "from_input" or "FS2_polynomial_fit"')

else:
    ccl_obj.mag_bias_tuple = None


# ! set radial kernel arrays and objects
ccl_obj.set_kernel_obj(cfg['C_ell']['has_rsd'], cfg['PyCCL']['n_samples_wf'])
ccl_obj.set_kernel_arr(z_grid_wf=z_grid_ssc_integrands,
                       has_magnification_bias=cfg['C_ell']['has_magnification_bias'])

gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias!)'
ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_wo_gal_bias_arr

# 3. ! bnt-transform these kernels (for lensing, it's only the gamma kernel, without IA)
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

assert np.all(np.diff(z_means_ll) > 0), 'z_means_ll should be monotonically increasing'
assert np.all(np.diff(z_means_gg) > 0), 'z_means_gg should be monotonically increasing'
assert np.all(np.diff(z_means_ll_bnt) > 0), ('z_means_ll_bnt should be monotonically increasing '
                                             '(not a strict condition, valid only if we do not shift the n(z) in this part)')

# 5. compute the ell cuts
ell_cuts_dict = {}
ellcuts_kw = {
    'kmax_h_over_Mpc': kmax_h_over_Mpc,
    'cosmo_ccl': ccl_obj.cosmo_ccl,
    'zbins': zbins,
    'h': h,
    'kmax_h_over_Mpc_ref': cfg['ell_cuts']['kmax_h_over_Mpc_ref'],
}
ell_cuts_dict['LL'] = ell_utils.load_ell_cuts(z_values_a=z_means_ll_bnt, z_values_b=z_means_ll_bnt, **ellcuts_kw)
ell_cuts_dict['GG'] = ell_utils.load_ell_cuts(z_values_a=z_means_gg, z_values_b=z_means_gg, **ellcuts_kw)
ell_cuts_dict['GL'] = ell_utils.load_ell_cuts(z_values_a=z_means_gg, z_values_b=z_means_ll_bnt, **ellcuts_kw)
ell_cuts_dict['LG'] = ell_utils.load_ell_cuts(z_values_a=z_means_ll_bnt, z_values_b=z_means_gg, **ellcuts_kw)
ell_dict['ell_cuts_dict'] = ell_cuts_dict  # this is to pass the ell cuts to the covariance module
# ! END SCALE CUTS

# now compute the BNT used for the rest of the code
if shift_nz:
    nz_src = wf_cl_lib.shift_nz(zgrid_nz_src, nz_unshifted_src, dzWL_fiducial, normalize=normalize_shifted_nz,
                                plot_nz=False, interpolation_kind=shift_nz_interpolation_kind)
    nz_lns = wf_cl_lib.shift_nz(zgrid_nz_lns, nz_unshifted_lns, dzGC_fiducial, normalize=normalize_shifted_nz,
                                plot_nz=False, interpolation_kind=shift_nz_interpolation_kind)
    # * this is important: the BNT matrix I use for the rest of the code (so not to compute the ell cuts) is instead
    # * consistent with the shifted n(z) used to compute the kernels
    bnt_matrix = bnt.compute_bnt_matrix(
        zbins, zgrid_nz_src, nz_src, cosmo_ccl=ccl_obj.cosmo_ccl, plot_nz=False)

wf_cl_lib.plot_nz_src_lns(zgrid_nz_src, nz_src, zgrid_nz_lns, nz_lns, colors=clr)

# re-set n(z) used in CCL class, then re-compute kernels
ccl_obj.set_nz(nz_full_src=np.hstack((zgrid_nz_src[:, None], nz_src)),
               nz_full_lns=np.hstack((zgrid_nz_lns[:, None], nz_lns)))
ccl_obj.set_kernel_obj(cfg['C_ell']['has_rsd'], cfg['PyCCL']['n_samples_wf'])
ccl_obj.set_kernel_arr(z_grid_wf=z_grid_ssc_integrands,
                       has_magnification_bias=cfg['C_ell']['has_magnification_bias'])

gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias)'
ccl_obj.wf_galaxy_arr = ccl_obj.wf_galaxy_wo_gal_bias_arr


# plot
wf_names_list = ['delta', 'gamma', 'ia', 'magnification', 'lensing', gal_kernel_plt_title]
wf_ccl_list = [ccl_obj.wf_delta_arr, ccl_obj.wf_gamma_arr, ccl_obj.wf_ia_arr, ccl_obj.wf_mu_arr,
               ccl_obj.wf_lensing_arr, ccl_obj.wf_galaxy_arr]

plt.figure()
for wf_idx in range(len(wf_ccl_list)):
    for zi in range(zbins):
        plt.plot(z_grid_ssc_integrands, wf_ccl_list[wf_idx][:, zi], c=clr[zi], alpha=0.6)
    plt.xlabel('$z$')
    plt.ylabel(r'$W_i^X(z)$')
    plt.suptitle(f'{wf_names_list[wf_idx]}')
    plt.tight_layout()
    plt.show()

# compute cls
ccl_obj.cl_ll_3d = ccl_obj.compute_cls(ell_dict['ell_WL'], ccl_obj.p_of_k_a,
                                       ccl_obj.wf_lensing_obj, ccl_obj.wf_lensing_obj, 'spline')
ccl_obj.cl_gl_3d = ccl_obj.compute_cls(ell_dict['ell_XC'], ccl_obj.p_of_k_a,
                                       ccl_obj.wf_galaxy_obj, ccl_obj.wf_lensing_obj, 'spline')
ccl_obj.cl_gg_3d = ccl_obj.compute_cls(ell_dict['ell_GC'], ccl_obj.p_of_k_a,
                                       ccl_obj.wf_galaxy_obj, ccl_obj.wf_galaxy_obj, 'spline')

# oc needs finer sampling to avoid issues
ells_3x2pt_oc = np.geomspace(cfg['ell_binning']['ell_min'], cfg['ell_binning']['ell_max_3x2pt'], nbl_3x2pt_oc)
cl_ll_3d_oc = ccl_obj.compute_cls(ells_3x2pt_oc, ccl_obj.p_of_k_a,
                                  ccl_obj.wf_lensing_obj, ccl_obj.wf_lensing_obj, 'spline')
cl_gl_3d_oc = ccl_obj.compute_cls(ells_3x2pt_oc, ccl_obj.p_of_k_a,
                                  ccl_obj.wf_galaxy_obj, ccl_obj.wf_lensing_obj, 'spline')
cl_gg_3d_oc = ccl_obj.compute_cls(ells_3x2pt_oc, ccl_obj.p_of_k_a,
                                  ccl_obj.wf_galaxy_obj, ccl_obj.wf_galaxy_obj, 'spline')

# ! add multiplicative shear bias
# ! THIS SHOULD NOT BE DONE FOR THE OC Cls!! mult shear bias values are passed in the .ini file
mult_shear_bias = np.array(cfg['C_ell']['mult_shear_bias'])
assert len(mult_shear_bias) == zbins, 'mult_shear_bias should be a scalar'
if not np.all(mult_shear_bias == 0):
    print('applying multiplicative shear bias')
    print(f'mult_shear_bias = {mult_shear_bias}')
    for ell_idx, _ in enumerate(ccl_obj.cl_ll_3d.shape[0]):
        for zi in range(zbins):
            for zj in range(zbins):
                ccl_obj.cl_ll_3d[ell_idx, zi, zj] *= (1 + mult_shear_bias[zi]) * (1 + mult_shear_bias[zj])

    for ell_idx, _ in enumerate(ccl_obj.cl_gl_3d.shape[0]):
        for zi in range(zbins):
            for zj in range(zbins):
                ccl_obj.cl_gl_3d[ell_idx, zi, zj] *= (1 + mult_shear_bias[zj])

ccl_obj.cl_3x2pt_5d = np.zeros((n_probes, n_probes, nbl_3x2pt, zbins, zbins))
ccl_obj.cl_3x2pt_5d[0, 0, :, :, :] = ccl_obj.cl_ll_3d[:nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[1, 0, :, :, :] = ccl_obj.cl_gl_3d[:nbl_3x2pt, :, :]
ccl_obj.cl_3x2pt_5d[0, 1, :, :, :] = ccl_obj.cl_gl_3d[:nbl_3x2pt, :, :].transpose(0, 2, 1)
ccl_obj.cl_3x2pt_5d[1, 1, :, :, :] = ccl_obj.cl_gg_3d[:nbl_3x2pt, :, :]

cl_ll_3d, cl_gl_3d, cl_gg_3d = ccl_obj.cl_ll_3d, ccl_obj.cl_gl_3d, ccl_obj.cl_gg_3d
cl_3x2pt_5d = ccl_obj.cl_3x2pt_5d

cl_3x2pt_5d_oc = np.zeros((n_probes, n_probes, nbl_3x2pt_oc, zbins, zbins))
cl_3x2pt_5d_oc[0, 0, :, :, :] = cl_ll_3d_oc
cl_3x2pt_5d_oc[1, 0, :, :, :] = cl_gl_3d_oc
cl_3x2pt_5d_oc[0, 1, :, :, :] = cl_gl_3d_oc.transpose(0, 2, 1)
cl_3x2pt_5d_oc[1, 1, :, :, :] = cl_gg_3d_oc

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


# ! BNT transform the cls (and responses?) - it's more complex since I also have to transform the noise
# ! spectra, better to transform directly the covariance matrix
if cfg['BNT']['cl_BNT_transform']:
    print('BNT-transforming the Cls...')
    assert cfg['BNT']['cov_BNT_transform'] is False, \
        'the BNT transform should be applied either to the Cls or to the covariance, not both'
    cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, bnt_matrix, 'L', 'L')
    cl_3x2pt_5d = cl_utils.cl_BNT_transform_3x2pt(cl_3x2pt_5d, bnt_matrix)
    warnings.warn('you should probably BNT-transform the responses too!')
    if compute_oc_g or compute_oc_ssc or compute_oc_cng:
        raise NotImplementedError('You should cut also the OC Cls')


# ! cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
if ell_max_WL == 1500:
    warnings.warn(
        'you are cutting the datavectors and responses in the pessimistic case, but is this compatible '
        'with the redshift-dependent ell cuts? Yes, this is an old warning; nonetheless, check ')
    assert False, 'you should check this'
    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
    cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

# ! Vincenzo's method for cl_ell_cuts: get the idxs to delete for the flattened 1d cls
if cfg['ell_cuts']['center_or_min'] == 'center':
    ell_prefix = 'ell'
elif cfg['ell_cuts']['center_or_min'] == 'min':
    ell_prefix = 'ell_edges'
else:
    raise ValueError('cfg["ell_cuts"]["center_or_min"] should be either "center" or "min"')

ell_dict['idxs_to_delete_dict'] = {
    'LL': ell_utils.get_idxs_to_delete(ell_dict[f'{ell_prefix}_WL'], ell_cuts_dict['LL'], is_auto_spectrum=True, zbins=zbins),
    'GG': ell_utils.get_idxs_to_delete(ell_dict[f'{ell_prefix}_GC'], ell_cuts_dict['GG'], is_auto_spectrum=True, zbins=zbins),
    'GL': ell_utils.get_idxs_to_delete(ell_dict[f'{ell_prefix}_XC'], ell_cuts_dict['GL'], is_auto_spectrum=False, zbins=zbins),
    'LG': ell_utils.get_idxs_to_delete(ell_dict[f'{ell_prefix}_XC'], ell_cuts_dict['LG'], is_auto_spectrum=False, zbins=zbins),
    '3x2pt': ell_utils.get_idxs_to_delete_3x2pt(ell_dict[f'{ell_prefix}_3x2pt'], ell_cuts_dict, zbins, cfg['covariance'])
}

# ! 3d cl ell cuts (*after* BNT!!)
# TODO here you could implement 1d cl ell cuts (but we are cutting at the covariance and derivatives level)
if cfg['ell_cuts']['cl_ell_cuts']:
    cl_ll_3d = cl_utils.cl_ell_cut(cl_ll_3d, ell_dict['ell_WL'], ell_cuts_dict['LL'])
    cl_gg_3d = cl_utils.cl_ell_cut(cl_gg_3d, ell_dict['ell_GC'], ell_cuts_dict['GG'])
    cl_3x2pt_5d = cl_utils.cl_ell_cut_3x2pt(cl_3x2pt_5d, ell_cuts_dict, ell_dict['ell_3x2pt'])
    if compute_oc_g or compute_oc_ssc or compute_oc_cng:
        raise NotImplementedError('You should cut also the OC Cls')

# re-set cls in the ccl_obj after BNT transform and/or ell cuts
ccl_obj.cl_ll_3d = cl_ll_3d
ccl_obj.cl_gg_3d = cl_gg_3d
ccl_obj.cl_3x2pt_5d = cl_3x2pt_5d

# ! build covariance matrices
cov_obj = sb_cov.SpaceborneCovariance(cfg, pvt_cfg, ell_dict, bnt_matrix)
cov_obj.set_ind_and_zpairs(ind, zbins)
cov_obj.symmetrize_output_dict = symmetrize_output_dict
cov_obj.consistency_checks()
cov_obj.set_gauss_cov(ccl_obj=ccl_obj, split_gaussian_cov=cfg['covariance']['split_gaussian_cov'])

# ! ========================================== OneCovariance ===================================================
if compute_oc_g or compute_oc_ssc or compute_oc_cng:

    if cfg['ell_cuts']['cl_ell_cuts']:
        raise NotImplementedError('TODO double check inputs in this case. This case is untested')

    start_time = time.perf_counter()

    # * 1. save ingredients in ascii format
    oc_path = f'{output_path}/OneCovariance'
    if not os.path.exists(oc_path):
        os.makedirs(oc_path)

    nz_src_ascii_filename = cfg['nz']['nz_sources_filename'].replace('.dat', f'_dzshifts{shift_nz}.ascii')
    nz_lns_ascii_filename = cfg['nz']['nz_lenses_filename'].replace('.dat', f'_dzshifts{shift_nz}.ascii')
    nz_src_ascii_filename = nz_src_ascii_filename.format(**pvt_cfg)
    nz_lns_ascii_filename = nz_lns_ascii_filename.format(**pvt_cfg)
    nz_src_ascii_filename = os.path.basename(nz_src_ascii_filename)
    nz_lns_ascii_filename = os.path.basename(nz_lns_ascii_filename)
    nz_src_tosave = np.column_stack((zgrid_nz_src, nz_src))
    nz_lns_tosave = np.column_stack((zgrid_nz_lns, nz_lns))
    np.savetxt(f'{oc_path}/{nz_src_ascii_filename}', nz_src_tosave)
    np.savetxt(f'{oc_path}/{nz_lns_ascii_filename}', nz_lns_tosave)

    cl_ll_ascii_filename = f'Cell_ll_nbl{nbl_3x2pt_oc}'
    cl_gl_ascii_filename = f'Cell_gl_nbl{nbl_3x2pt_oc}'
    cl_gg_ascii_filename = f'Cell_gg_nbl{nbl_3x2pt_oc}'
    sl.write_cl_ascii(oc_path, cl_ll_ascii_filename, cl_3x2pt_5d_oc[0, 0, ...], ells_3x2pt_oc, zbins)
    sl.write_cl_ascii(oc_path, cl_gl_ascii_filename, cl_3x2pt_5d_oc[1, 0, ...], ells_3x2pt_oc, zbins)
    sl.write_cl_ascii(oc_path, cl_gg_ascii_filename, cl_3x2pt_5d_oc[1, 1, ...], ells_3x2pt_oc, zbins)

    ascii_filenames_dict = {
        'cl_ll_ascii_filename': cl_ll_ascii_filename,
        'cl_gl_ascii_filename': cl_gl_ascii_filename,
        'cl_gg_ascii_filename': cl_gg_ascii_filename,
        'nz_src_ascii_filename': nz_src_ascii_filename,
        'nz_lns_ascii_filename': nz_lns_ascii_filename,
    }

    if cfg["covariance"]["which_b1g_in_resp"] == 'from_input':
        gal_bias_ascii_filename = f'{oc_path}/gal_bias_table.ascii'
        ccl_obj.save_gal_bias_table_ascii(z_grid_ssc_integrands, gal_bias_ascii_filename)
        ascii_filenames_dict['gal_bias_ascii_filename'] = gal_bias_ascii_filename
    elif cfg["covariance"]["which_b1g_in_resp"] == 'from_HOD':
        warnings.warn('OneCovariance will use the HOD-derived galaxy bias for the Cls and responses')

    # * 2. compute cov using the onecovariance interface class
    print('Start NG cov computation with OneCovariance...')
    # initialize object, build cfg file
    oc_obj = oc_interface.OneCovarianceInterface(cfg, pvt_cfg,
                                                 do_g=compute_oc_g,
                                                 do_ssc=compute_oc_ssc,
                                                 do_cng=compute_oc_cng)
    oc_obj.oc_path = oc_path
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
        np.testing.assert_allclose(check_cov_sva_oc_3x2pt_10D, oc_obj.cov_sva_oc_3x2pt_10D, atol=0, rtol=1e-3)
        np.testing.assert_allclose(check_cov_mix_oc_3x2pt_10D, oc_obj.cov_mix_oc_3x2pt_10D, atol=0, rtol=1e-3)
        np.testing.assert_allclose(check_cov_sn_oc_3x2pt_10D, oc_obj.cov_sn_oc_3x2pt_10D, atol=0, rtol=1e-3)
        np.testing.assert_allclose(check_cov_ssc_oc_3x2pt_10D, oc_obj.cov_ssc_oc_3x2pt_10D, atol=0, rtol=1e-3)
        np.testing.assert_allclose(check_cov_cng_oc_3x2pt_10D, oc_obj.cov_cng_oc_3x2pt_10D, atol=0, rtol=1e-3)

    print('Time taken to compute OC: {:.2f} m'.format((time.perf_counter() - start_time) / 60))

else:
    oc_obj = None

# ! ========================================== Spaceborne ===================================================

# precompute pk_mm, pk_gm and pk_mm, if you want to rescale the responses
k_array, pk_mm_2d = cosmo_lib.pk_from_ccl(k_grid_resp, z_grid_ssc_integrands, use_h_units,
                                          ccl_obj.cosmo_ccl, pk_kind='nonlinear')

# compute P_gm, P_gg
gal_bias = ccl_obj.gal_bias_2d[:, 0]

# check that it's the same in each bin
for zi in range(zbins):
    np.testing.assert_allclose(ccl_obj.gal_bias_2d[:, 0], ccl_obj.gal_bias_2d[:, zi], atol=0, rtol=1e-5)
# TODO case with different bias in each bin!

pk_gm_2d = pk_mm_2d * gal_bias
pk_gg_2d = pk_mm_2d * gal_bias ** 2

if compute_sb_ssc:
    print('Start SSC computation with Spaceborne...')

    # ! 1. Get halo model responses from CCL
    if cfg['covariance']['which_pk_responses'] == 'halo_model_CCL':

        ccl_obj.initialize_trispectrum(which_ng_cov='SSC', probe_ordering=probe_ordering,
                                       pyccl_cfg=cfg['PyCCL'])

        # k and z grids (responses will be interpolated below)
        k_grid_resp_hm = ccl_obj.responses_dict['L', 'L', 'L', 'L']['k_1overMpc']
        a_grid_resp_hm = ccl_obj.responses_dict['L', 'L', 'L', 'L']['a_arr']
        # translate a to z and cut the arrays to the maximum redshift of the SU responses (much smaller range!)
        z_grid_resp_hm = cosmo_lib.a_to_z(a_grid_resp_hm)[::-1]

        assert np.allclose(k_grid_resp_hm, k_grid_resp, atol=0, rtol=1e-2), \
            'CCL and SB k_grids for responses should match'

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

        dPmm_ddeltab_hm_func = CubicSpline(x=z_grid_resp_hm, y=dPmm_ddeltab_hm, axis=1)
        dPgm_ddeltab_hm_func = CubicSpline(x=z_grid_resp_hm, y=dPgm_ddeltab_hm, axis=1)
        dPgg_ddeltab_hm_func = CubicSpline(x=z_grid_resp_hm, y=dPgg_ddeltab_hm, axis=1)

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

    elif cfg['covariance']['which_pk_responses'] == 'halo_model_SB':

        which_b1g_in_resp = cfg['covariance']['which_b1g_in_resp']
        include_terasawa_terms = cfg['covariance']['include_terasawa_terms']
        resp_obj = responses.SpaceborneResponses(cfg=cfg, k_grid=k_grid_resp,
                                                 z_grid=z_grid_ssc_integrands,
                                                 ccl_obj=ccl_obj)
        resp_obj.use_h_units = use_h_units
        resp_obj.set_hm_resp(k_grid_resp, z_grid_ssc_integrands,
                             which_b1g_in_resp, gal_bias,
                             include_terasawa_terms=include_terasawa_terms)
        dPmm_ddeltab = resp_obj.dPmm_ddeltab_hm
        dPgm_ddeltab = resp_obj.dPgm_ddeltab_hm
        dPgg_ddeltab = resp_obj.dPgg_ddeltab_hm
        r_mm_hm = resp_obj.r1_mm_hm
        r_gm_hm = resp_obj.r1_gm_hm
        r_gg_hm = resp_obj.r1_gg_hm

    # ! from SpaceborneResponses class
    elif cfg['covariance']['which_pk_responses'] == 'separate_universe_SB':

        resp_obj = responses.SpaceborneResponses(cfg=cfg, k_grid=k_grid_resp,
                                                 z_grid=z_grid_ssc_integrands,
                                                 ccl_obj=ccl_obj)
        resp_obj.use_h_units = use_h_units
        resp_obj.set_g1mm_su_resp()
        r_mm_sbclass = resp_obj.compute_r1_mm()
        resp_obj.set_su_resp(b2g_from_halomodel=True,
                             include_b2g=cfg['covariance']['include_b2g'])

        r_gm_sbclass = resp_obj.r1_gm
        r_gg_sbclass = resp_obj.r1_gg

        dPmm_ddeltab = resp_obj.dPmm_ddeltab
        dPgm_ddeltab = resp_obj.dPgm_ddeltab
        dPgg_ddeltab = resp_obj.dPgg_ddeltab

        b1g_hm = resp_obj.b1g_hm
        b2g_hm = resp_obj.b2g_hm

    else:
        raise ValueError(
            'which_pk_responses must be either "halo_model" or "separate_universe_SB"')

    # ! 2. prepare integrands (d2CAB_dVddeltab) and volume element
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
              f'Increasing z_min until kmax_limber < k_max_dPk. Alternatively, increase k_max_dPk or decrease ell_max.')
        z_grid_ssc_integrands_test = z_grid_ssc_integrands_test[1:]
        kmax_limber = cosmo_lib.get_kmax_limber(
            ell_grid, z_grid_ssc_integrands_test, use_h_units, ccl_obj.cosmo_ccl)
        print(f'Retrying with z_min = {z_grid_ssc_integrands_test[0]:.3f}')

    k_limber = partial(cosmo_lib.k_limber, cosmo_ccl=ccl_obj.cosmo_ccl, use_h_units=use_h_units)

    dPmm_ddeltab_klimb = np.array(
        [dPmm_ddeltab_interp((k_limber(ell_val, z_grid_ssc_integrands), z_grid_ssc_integrands)) for ell_val in
            ell_dict['ell_WL']])
    dPgm_ddeltab_klimb = np.array(
        [dPgm_ddeltab_interp((k_limber(ell_val, z_grid_ssc_integrands), z_grid_ssc_integrands)) for ell_val in
            ell_dict['ell_XC']])
    dPgg_ddeltab_klimb = np.array(
        [dPgg_ddeltab_interp((k_limber(ell_val, z_grid_ssc_integrands), z_grid_ssc_integrands)) for ell_val in
            ell_dict['ell_GC']])

    # ! integral prefactor
    cl_integral_prefactor = cosmo_lib.cl_integral_prefactor(z_grid_ssc_integrands,
                                                            cl_integral_convention_ssc,
                                                            use_h_units=use_h_units,
                                                            cosmo_ccl=ccl_obj.cosmo_ccl)
    # ! observable densities
    wf_delta = ccl_obj.wf_delta_arr
    wf_gamma = ccl_obj.wf_gamma_arr
    wf_ia = ccl_obj.wf_ia_arr
    wf_mu = ccl_obj.wf_mu_arr
    wf_lensing = ccl_obj.wf_lensing_arr

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
    if cfg['covariance']['load_cached_sigma2_b']:
        sigma2_b = np.load(f'{output_path}/cache/sigma2_b.npy')

    else:
        print('Computing sigma2_b...')

        if cfg['covariance']['use_KE_approximation']:
            # compute sigma2_b(z) (1 dimension) using the existing CCL implementation
            ccl_obj.set_sigma2_b(z_grid=z_grid_ssc_integrands,
                                 fsky=cfg['mask']['fsky'],
                                 which_sigma2_b=which_sigma2_b,
                                 nside_mask=cfg['mask']['nside_mask'],
                                 mask_path=cfg['mask']['mask_path'])
            _a, sigma2_b = ccl_obj.sigma2_b_tuple
            # quick sanity check on the a/z grid
            sigma2_b = sigma2_b[::-1]
            _z = cosmo_lib.a_to_z(_a)[::-1]
            np.testing.assert_allclose(z_grid_ssc_integrands, _z, atol=0, rtol=1e-8)

        else:
            k_grid_sigma2 = np.logspace(cfg['covariance']['log10_k_min_sigma2'],
                                        cfg['covariance']['log10_k_max_sigma2'],
                                        cfg['covariance']['k_steps_sigma2'])
            sigma2_b = sigma2_SSC.sigma2_z1z2_wrap(
                z_grid_ssc_integrands=z_grid_ssc_integrands,
                k_grid_sigma2=k_grid_sigma2,
                cosmo_ccl=ccl_obj.cosmo_ccl,
                which_sigma2_b=which_sigma2_b,
                area_deg2_in=cfg['mask']['survey_area_deg2'],
                nside_mask=cfg['mask']['nside_mask'],
                mask_path=cfg['mask']['nside_mask']
            )
            # Note: if you want to compare sigma2 with full_curved_sky against polar_cap_on_the_fly, remember to divide
            # the former by fsky (eq. 29 of https://arxiv.org/pdf/1612.05958)

    if not cfg['covariance']['load_cached_sigma2_b']:
        np.save(f'{output_path}/cache/sigma2_b.npy', sigma2_b)
        np.save(f'{output_path}/cache/zgrid_sigma2_b.npy', z_grid_ssc_integrands)

    # ! 4. Perform the integration calling the Julia module
    print('Computing the SSC integral...')
    start = time.perf_counter()
    cov_ssc_3x2pt_dict_8D = cov_obj.ssc_integral_julia(d2CLL_dVddeltab=d2CLL_dVddeltab,
                                                       d2CGL_dVddeltab=d2CGL_dVddeltab,
                                                       d2CGG_dVddeltab=d2CGG_dVddeltab,
                                                       cl_integral_prefactor=cl_integral_prefactor,
                                                       sigma2=sigma2_b,
                                                       z_grid=z_grid_ssc_integrands,
                                                       integration_type=ssc_integration_type,
                                                       probe_ordering=probe_ordering,
                                                       num_threads=cfg['misc']['num_threads'])
    print('SSC computed in {:.2f} m'.format((time.perf_counter() - start) / 60))

    # in the full_curved_sky case only, sigma2_b has to be divided by fsky
    # TODO it would make much more sense to divide s2b directly...
    if which_sigma2_b == 'full_curved_sky':
        for key in cov_ssc_3x2pt_dict_8D.keys():
            cov_ssc_3x2pt_dict_8D[key] /= cfg['mask']['fsky']
    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask', 'flat_sky']:
        pass
    else:
        raise ValueError(f'which_sigma2_b = {which_sigma2_b} not recognized')

    cov_obj.cov_ssc_sb_3x2pt_dict_8D = cov_ssc_3x2pt_dict_8D

# TODO integrate this with Spaceborne_covg

# ! ========================================== PyCCL ===================================================
if (compute_ccl_ssc or compute_ccl_cng):

    # Note: this z grid has to be larger than the one requested in the trispectrum (z_grid_tkka in the cfg file).
    # You can probaby use the same grid as the one used in the trispectrum, but from my tests is should be
    # zmin_s2b < zmin_s2b_tkka and zmax_s2b =< zmax_s2b_tkka.
    # if zmin=0 it looks like I can have zmin_s2b = zmin_s2b_tkka
    ccl_obj.set_sigma2_b(z_grid=z_default_grid_ccl,
                         fsky=cfg['mask']['fsky'],
                         which_sigma2_b=which_sigma2_b,
                         nside_mask=cfg['mask']['nside_mask'],
                         mask_path=cfg['mask']['mask_path'])

    ccl_ng_cov_terms_list = []
    if compute_ccl_ssc:
        ccl_ng_cov_terms_list.append('SSC')
    if compute_ccl_cng:
        ccl_ng_cov_terms_list.append('cNG')

    for which_ng_cov in ccl_ng_cov_terms_list:

        ccl_obj.initialize_trispectrum(which_ng_cov, probe_ordering, cfg['PyCCL'])
        ccl_obj.compute_ng_cov_3x2pt(which_ng_cov, ell_dict['ell_3x2pt'], cfg['mask']['fsky'],
                                     integration_method=cfg['PyCCL']['cov_integration_method'],
                                     probe_ordering=probe_ordering, ind_dict=ind_dict)

# ! ========================================== combine covariance terms ================================================
cov_obj.build_covs(ccl_obj=ccl_obj, oc_obj=oc_obj)
cov_dict = cov_obj.cov_dict

# ! ========================================== plot & tests ================================================
for key in cov_dict.keys():
    sl.matshow(cov_dict[key], title=key)

for key in cov_dict.keys():
    np.testing.assert_allclose(cov_dict[key], cov_dict[key].T,
                               atol=0, rtol=1e-7, err_msg=f'{key} not symmetric')

with open(f'{output_path}/run_config.yaml', 'w') as yaml_file:
    yaml.dump(cfg, yaml_file, default_flow_style=False)

if cfg['misc']['save_output_as_benchmark']:

    if (
        cfg['covariance']['SSC_code'] in ['PyCCL', 'OneCovariance'] or
        cfg['covariance']['cNG_code'] in ['PyCCL', 'OneCovariance']
    ):
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
        "timestamp": datetime.datetime.now().isoformat(),
        'branch': branch,
        'commit': commit,
    }

    bench_filename = cfg['misc']['bench_filename']
    if os.path.exists(f'{bench_filename}.npz'):
        raise ValueError('You are trying to overwrite a benchmark file. Please rename the file or delete the existing one.')

    with open(f'{bench_filename}.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    np.savez_compressed(bench_filename,
                        backup_cfg=cfg,
                        z_grid_ssc_integrands=z_grid_ssc_integrands,
                        k_grid_resp=k_grid_resp,
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
                        cov_WL_g_2D=cov_dict['cov_WL_g_2D'],
                        cov_GC_g_2D=cov_dict['cov_GC_g_2D'],
                        cov_XC_g_2D=cov_dict['cov_XC_g_2D'],
                        cov_3x2pt_g_2D=cov_dict['cov_3x2pt_g_2D'],
                        cov_WL_ssc_2D=cov_dict['cov_WL_ssc_2D'],
                        cov_GC_ssc_2D=cov_dict['cov_GC_ssc_2D'],
                        cov_XC_ssc_2D=cov_dict['cov_XC_ssc_2D'],
                        cov_3x2pt_ssc_2D=cov_dict['cov_3x2pt_ssc_2D'],
                        cov_WL_cng_2D=cov_dict['cov_WL_cng_2D'],
                        cov_GC_cng_2D=cov_dict['cov_GC_cng_2D'],
                        cov_XC_cng_2D=cov_dict['cov_XC_cng_2D'],
                        cov_3x2pt_cng_2D=cov_dict['cov_3x2pt_cng_2D'],
                        metadata=metadata,
                        )


for which_cov in cov_dict.keys():
    probe = which_cov.split('_')[1]
    which_ng_cov = which_cov.split('_')[2]
    ndim = which_cov.split('_')[3]
    cov_filename = cfg['covariance']['cov_filename'].format(which_ng_cov=which_ng_cov,
                                                            probe=probe,
                                                            ndim=ndim)

    np.savez_compressed(f'{output_path}/{cov_filename}', **cov_dict)

print(f'Covariance matrices saved in {output_path}\n')

for which_cov in cov_dict.keys():

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
                print('Cholesky decomposition failed. Consider checking the condition number or symmetry.')

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
                    rtol=1e-7
                )
                if identity_check:
                    print('Inverse verified successfully (matrix product is identity). atol=1e-9, rtol=1e-7')
                else:
                    print('Warning: Inverse verification failed (matrix product deviates from identity). atol=0, rtol=1e-7')
            except np.linalg.LinAlgError:
                print('Numpy inversion failed: Matrix is singular or near-singular.')

        if cfg['misc']['test_symmetry']:
            if not np.allclose(cov_dict[which_cov], cov_dict[which_cov].T, atol=0, rtol=1e-7):
                print('Warning: Matrix is not symmetric. atol=0, rtol=1e-7')
            else:
                print('Matrix is symmetric. atol=0, rtol=1e-7')


print('Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60))
