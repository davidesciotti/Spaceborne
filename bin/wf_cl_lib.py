import warnings
from copy import deepcopy

import scipy
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import pyccl as ccl
import yaml
from joblib import Parallel, delayed
from matplotlib import cm
from numba import njit
from scipy.integrate import quad, quad_vec, simpson, dblquad, simps
from scipy.interpolate import interp1d, interp2d
from scipy.special import erf
from functools import partial
from tqdm import tqdm

# project_path = Path.cwd().parent
project_path = '/Users/davide/Documents/Lavoro/Programmi/cl_v2'
project_path_parent = '/Users/davide/Documents/Lavoro/Programmi'

# general configurations
sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg')
import common_cfg.ISTF_fid_params as ISTF
import common_cfg.mpl_cfg as mpl_cfg
from . import my_module as mm
from . import cosmo_lib as csmlib

# config files
sys.path.append(f'{project_path}/config')
import config_wlcl as cfg


# update plot pars
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
matplotlib.use('Qt5Agg')

###############################################################################
###############################################################################
###############################################################################


fiducial_pars_dict_nested = mm.read_yaml(
    '/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_cfg/ISTF_fiducial_params.yml')
fiducial_pars_dict = mm.flatten_dict(fiducial_pars_dict_nested)

c = ISTF.constants['c']

gamma = ISTF.extensions['gamma']

z_edges = ISTF.photoz_bins['all_zbin_edges']
z_median = ISTF.photoz_bins['z_median']
zbins = ISTF.photoz_bins['zbins']
z_minus = ISTF.photoz_bins['z_minus']
z_plus = ISTF.photoz_bins['z_plus']

z_0 = z_median / np.sqrt(2)
z_min = z_edges[0]
z_max = cfg.z_max
sqrt2 = np.sqrt(2)

f_out = ISTF.photoz_pdf['f_out']
c_in, z_in, sigma_in = ISTF.photoz_pdf['c_b'], ISTF.photoz_pdf['z_b'], ISTF.photoz_pdf['sigma_b']
c_out, z_out, sigma_out = ISTF.photoz_pdf['c_o'], ISTF.photoz_pdf['z_o'], ISTF.photoz_pdf['sigma_o']

simps_z_step_size = 1e-4

n_bar = np.genfromtxt(f"{project_path}/output/n_bar.txt")
n_gal = ISTF.other_survey_specs['n_gal']

z_max_cl = cfg.z_max_cl
z_grid = np.linspace(z_min, z_max_cl, cfg.zsteps_cl)
# use_h_units = cfg.use_h_units

warnings.warn('these global variables should be deleted...')
warnings.warn('RECHECK Ox0 in cosmolib')


@njit
def pph(z_p, z):
    first_addendum = (1 - f_out) / (np.sqrt(2 * np.pi) * sigma_in * (1 + z)) * \
        np.exp(-0.5 * ((z - c_in * z_p - z_in) / (sigma_in * (1 + z))) ** 2)
    second_addendum = f_out / (np.sqrt(2 * np.pi) * sigma_out * (1 + z)) * \
        np.exp(-0.5 * ((z - c_out * z_p - z_out) / (sigma_out * (1 + z))) ** 2)
    return first_addendum + second_addendum


@njit
def n_of_z(z):
    return n_gal * (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))


################################## niz_unnorm_quad(z) ##############################################


# ! load or compute niz_unnorm_quad(z)
# TODO re-compute and check niz_unnorm_quad(z), maybe compute it with scipy.special.erf
if cfg.load_external_niz:
    niz_import = np.genfromtxt(f'{cfg.niz_path}/{cfg.niz_filename}')
    # store and remove the redshift values, ie the 1st column
    z_values_from_nz = niz_import[:, 0]
    niz_import = niz_import[:, 1:]

    assert niz_import.shape[1] == zbins, "niz_import.shape[1] should be == zbins"

    # normalization array
    n_bar = simps(niz_import, z_values_from_nz, axis=0)
    if not np.allclose(n_bar, np.ones(zbins), rtol=0.01, atol=0):
        print('It looks like the input niz_unnorm_quad(z) are not normalized (they differ from 1 by more than 1%)')


def n_i_old(z, i):
    n_i_interp = interp1d(niz_import[:, 0], niz_import[:, i + 1], kind="linear")
    result_array = n_i_interp(z)  # z is considered as an array
    result = result_array.item()  # otherwise it would be a 0d array
    return result


# zbin_idx_array = np.asarray(range(zbins))
# assert zbin_idx_array.dtype == 'int64', "zbin_idx_array.dtype should be 'int64'"
# niz_import_cpy = niz_import.copy()  # remove redshift column
# note: this is NOT an interpolation in i, which are just the bin indices and will NOT differ from the values 0, 1, 2
# ... 9. The purpose is just to have a 2D vectorized callable.
# niz = interp2d(zbin_idx_array, z_values_from_nz, niz_import_cpy, kind="linear")
# note: the normalization of n_of_z(z) should be unimportant, here I compute a ratio
# where n_of_z(z) is present both at the numerator and denominator!

def n_i(z, i):
    """with quad. normalized"""
    def integrand(z_p, z): return n_of_z(z) * pph(z_p, z)
    numerator = quad(integrand, z_minus[i], z_plus[i], args=z)[0]
    denominator = dblquad(integrand, z_min, z_max, z_minus[i], z_plus[i])[0]
    return numerator / denominator


def niz_unnormalized_quad(z, zbin_idx, pph=pph):
    """with quad - 0.620401143 s, faster than quadvec..."""
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return n_of_z(z) * quad(pph, z_minus[zbin_idx], z_plus[zbin_idx], args=(z))[0]


def niz_unnormalized_simps(z_grid, zbin_idx, pph=pph, zp_points=500):
    """numerator of Eq. (112) of ISTF, with simpson integration
    Not too fast (3.0980 s for 500 z_p points)"""

    # SIMPSON WITH DIFFERENT POSSIBLE GRIDS:
    # intantiate a grid for simpson integration which passes through all the bin edges (which are the integration limits!)
    # equal number of points per bin
    zp_points_per_bin = int(zp_points / zbins)
    zp_bin_grid = np.zeros((zbins, zp_points_per_bin))
    for i in range(zbins):
        zp_bin_grid[i, :] = np.linspace(z_edges[i], z_edges[i + 1], zp_points_per_bin)

    # more pythonic way of instantiating the same grid
    # zp_bin_grid = np.linspace(z_min, z_max, zp_points)
    # zp_bin_grid = np.append(zp_bin_grid, z_edges)  # add bin edges
    # zp_bin_grid = np.sort(zp_bin_grid)
    # zp_bin_grid = np.unique(zp_bin_grid)  # remove duplicates (first and last edges were already included)
    # zp_bin_grid = np.tile(zp_bin_grid, (zbins, 1))  # repeat the grid for each bin (in each row)
    # for i in range(zbins):  # remove all the points below the bin edge
    #     zp_bin_grid[i, :] = np.where(zp_bin_grid[i, :] > z_edges[i], zp_bin_grid[i, :], 0)

    assert type(zbin_idx) == int, 'zbin_idx must be an integer'  # TODO check if these slow down the code using scalene
    niz_unnorm_integrand = np.array([pph(zp_bin_grid[zbin_idx, :], z) for z in z_grid])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_bin_grid[zbin_idx, :], axis=1)
    niz_unnorm_integral *= n_of_z(z_grid)
    return niz_unnorm_integral


# alternative: equispaced grid with z_edges added (does *not* work well, needs a lot of samples!!)
zp_grid = np.linspace(z_min, z_max, 4000)
zp_grid = np.concatenate((z_edges, zp_grid))
zp_grid = np.unique(zp_grid)
zp_grid = np.sort(zp_grid)
# indices of z_edges in zp_grid:
z_edges_idxs = np.array([np.where(zp_grid == z_edges[i])[0][0] for i in range(z_edges.shape[0])])


def niz_unnormalized_simps_fullgrid(z_grid, zbin_idx, pph=pph):
    """numerator of Eq. (112) of ISTF, with simpson integration and "global" grid"""
    warnings.warn('this function needs very high number of samples;'
                  ' the zp_bin_grid sampling should perform better')
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    z_minus = z_edges_idxs[zbin_idx]
    z_plus = z_edges_idxs[zbin_idx + 1]
    niz_unnorm_integrand = np.array([pph(zp_grid[z_minus:z_plus], z) for z in z_grid])
    niz_unnorm_integral = simps(y=niz_unnorm_integrand, x=zp_grid[z_minus:z_plus], axis=1)
    return niz_unnorm_integral * n_of_z(z_grid)


def niz_unnormalized_quadvec(z, zbin_idx, pph=pph):
    """
    :param z: float, does not accept an array. Same as above, but with quad_vec.
    ! the difference is that the integrand can be a vector-valued function (in this case in z_p),
    so it's supposedly faster? -> no, it's slower - 5.5253 s
    """
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    niz_unnorm = quad_vec(quad_integrand, z_minus[zbin_idx], z_plus[zbin_idx], args=(z, pph))[0]
    return niz_unnorm


def niz_normalization_quad(niz_unnormalized_func, zbin_idx, pph=pph):
    assert type(zbin_idx) == int, 'zbin_idx must be an integer'
    return quad(niz_unnormalized_func, z_min, z_max, args=(zbin_idx, pph))[0]


def normalize_niz_simps(niz_unnorm_arr, z_grid):
    """ much more convenient; uses simps, and accepts as input an array of shape (zbins, z_points)"""
    norm_factor = simps(niz_unnorm_arr, z_grid)
    niz_norm = (niz_unnorm_arr.T / norm_factor).T
    return niz_norm


def niz_normalized(z, zbin_idx):
    """this is a wrapper function which normalizes the result.
    The if-else is needed not to compute the normalization for each z, but only once for each zbin_idx
    Note that the niz_unnormalized_quadvec function is not vectorized in z (its 1st argument)
    """
    warnings.warn("this function should be deprecated")
    warnings.warn('or add possibility to choose pph')
    if type(z) == float or type(z) == int:
        return niz_unnormalized_quadvec(z, zbin_idx) / niz_normalization_quad(zbin_idx, niz_unnormalized_quadvec)

    elif type(z) == np.ndarray:
        niz_unnormalized_arr = np.asarray([niz_unnormalized_quadvec(z_value, zbin_idx) for z_value in z])
        return niz_unnormalized_arr / niz_normalization_quad(zbin_idx, niz_unnormalized_quadvec)

    else:
        raise TypeError('z must be a float, an int or a numpy array')


def niz_unnormalized_analytical(z, zbin_idx, z_edges=z_edges):
    """the one used by Stefano in the PyCCL notebook
    by far the fastest, 0.009592 s"""

    assert zbin_idx < 10, 'this is the analytical function used in ISTF, it does not work for zbins != 10'
    addendum_1 = erf((z - z_out - c_out * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_out))
    addendum_2 = erf((z - z_out - c_out * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_out))
    addendum_3 = erf((z - z_in - c_in * z_edges[zbin_idx]) / (sqrt2 * (1 + z) * sigma_in))
    addendum_4 = erf((z - z_in - c_in * z_edges[zbin_idx + 1]) / (sqrt2 * (1 + z) * sigma_in))

    result = n_of_z(z) / (2 * c_out * c_in) * \
        (c_in * f_out * (addendum_1 - addendum_2) + c_out * (1 - f_out) * (addendum_3 - addendum_4))
    return result


################################## end niz ##############################################


# @njit
def wil_tilde_integrand_old(z_prime, z, i):
    return n_i_old(z_prime, i) * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


def wil_tilde_old(z, i):
    # integrate in z_prime, it must be the first argument
    result = quad(wil_tilde_integrand_old, z, z_max, args=(z, i))
    return result[0]


def wil_tilde_integrand_vec(z_prime, z):
    """
    vectorized version of wil_tilde_integrand, useful to fill up the computation of the integrand array for the simpson
    integration
    """

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_prime, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_prime)

    # return niz(zbin_idx_array, z_prime).T * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))  # old, with interpolator
    return niz_normalized_arr * (1 - csmlib.r_tilde(z) / csmlib.r_tilde(z_prime))


def wil_tilde_new(z):
    # version with quad vec, very slow, I don't know why.
    # It is the zbin_idx_array that is vectorized, because z_prime is integrated over
    return quad_vec(wil_tilde_integrand_vec, z, z_max, args=(z, zbin_idx_array))[0]


def wil_noIA_IST(z, wil_tilde_array):
    return ((3 / 2) * (H0 / c) * Om0 * (1 + z) * csmlib.r_tilde(z) * wil_tilde_array.T).T


# IA
# @njit
def W_IA(z_grid):
    warnings.warn("what about the normalization?")
    warnings.warn("different niz for sources and lenses?")

    # redshift distribution
    niz_unnormalized = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = normalize_niz_simps(niz_unnormalized, z_grid)

    # return (H0 / c) * niz(zbin_idx_array, z_grid).T * csmlib.E(z_grid)  # ! old, with interpolator
    return (H0 / c) * niz_normalized_arr * csmlib.E(z_grid)


# @njit
def F_IA(z, eta_IA, beta_IA, lumin_ratio_func):
    result = (1 + z) ** eta_IA * (lumin_ratio_func(z)) ** beta_IA
    return result


# use formula 23 of ISTF paper for Om(z)
# @njit
def Om(z, Om0, cosmo_astropy):
    return Om0 * (1 + z) ** 3 / csmlib.E(z, cosmo_astropy) ** 2


# @njit
def growth_factor_integrand(x, gamma, Om0, cosmo_astropy):
    return Om(x, Om0, cosmo_astropy) ** gamma / (1 + x)


def growth_factor(z, gamma, Om0, cosmo_astropy):
    integral = quad(growth_factor_integrand, 0, z, args=(gamma, Om0, cosmo_astropy))[0]
    return np.exp(-integral)


# @njit
# def IA_term_old(z, i):
#     return (A_IA * C_IA * Om0 * F_IA(z)) / growth_factor(z) * W_IA(z, i)

# @njit
def IA_term(z_grid, growth_factor_arr, A_IA, C_IA, Om0):
    """new version, vectorized"""
    return ((A_IA * C_IA * Om0 * F_IA(z_grid)) / growth_factor_arr * W_IA(z_grid)).T


# @njit
def wil_IA_IST(z_grid, wil_tilde_array, growth_factor_arr):
    return wil_noIA_IST(z_grid, wil_tilde_array) - IA_term(z_grid, growth_factor_arr)


def wil_final(z_grid, which_wf):
    # precompute growth factor
    growth_factor_arr = np.asarray([growth_factor(z) for z in z_grid])

    # fill simpson integrand
    zpoints_simps = 700
    z_prime_array = np.linspace(z_min, z_max, zpoints_simps)
    integrand = np.zeros((z_prime_array.size, z_grid.size, zbins))
    for z_idx, z_val in enumerate(z_grid):
        # output order of wil_tilde_integrand_vec is: z_prime, i
        integrand[:, z_idx, :] = wil_tilde_integrand_vec(z_prime_array, z_val).T

    # integrate with simpson to obtain wil_tilde
    wil_tilde_array = np.zeros((z_grid.size, zbins))
    for z_idx, z_val in enumerate(z_grid):
        # take the closest value to the desired z - less than 0.1% difference with the desired z
        z_prime_idx = np.argmin(np.abs(z_prime_array - z_val))
        wil_tilde_array[z_idx, :] = simpson(integrand[z_prime_idx:, z_idx, :], z_prime_array[z_prime_idx:], axis=0)

    if which_wf == 'with_IA':
        return wil_IA_IST(z_grid, wil_tilde_array, growth_factor_arr)
    elif which_wf == 'without_IA':
        return wil_noIA_IST(z_grid, wil_tilde_array)
    elif which_wf == 'IA_only':
        return W_IA(z_grid).T
    else:
        raise ValueError('which_wf must be "with_IA", "without_IA" or "IA_only"')


###################### wig ###########################

def b_of_z_analytical(z):
    """simple analytical prescription for the linear galaxy bias:
    b(z) = sqrt(1 + z)
    """
    return np.sqrt(1 + z)


def b_of_z_fs1_leporifit(z):
    """fit to the linear galaxy bias measured from FS1. This is the fit used in Vincenzo's sscresponses paper,
    I think... Not super sure which one I should use"""
    return 0.5125 + 1.377 * z + 0.222 * z ** 2 - 0.249 * z ** 3


def b_of_z_fs1_pocinofit(z):
    """fit to the linear galaxy bias measured from FS1. This is the fit that should be used , at least for
    the responses"""
    a, b, c = 0.81, 2.80, 1.02
    return a * z ** b / (1 + z) + c


def b_of_z_fs2_fit(z, maglim, poly_fit_values=None):
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit
    if maglim == 24.5:
        b0_gal, b1_gal, b2_gal, b3_gal = 1.33291, -0.72414, 1.0183, -0.14913
    elif maglim == 23:
        b0_gal, b1_gal, b2_gal, b3_gal = 1.88571, -2.73925, 3.24688, -0.73496
    else:
        raise ValueError('maglim, i.e. the limiting magnitude of the GCph sample, must be 23 or 24.5')

    if poly_fit_values is not None:
        assert len(poly_fit_values) == 4, 'a list of 4 best-fit values must be passed'
        np.testing.assert_allclose(np.array(poly_fit_values), np.array((b0_gal, b1_gal, b2_gal, b3_gal)), atol=0,
                                   rtol=1e-5)

    return b0_gal + (b1_gal * z) + (b2_gal * z ** 2) + (b3_gal * z ** 3)


def magbias_of_z_fs2_fit(z, maglim, poly_fit_values=None):
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit

    if maglim == 24.5:
        b0_mag, b1_mag, b2_mag, b3_mag = -1.50685, 1.35034, 0.08321, 0.04279
    elif maglim == 23:
        b0_mag, b1_mag, b2_mag, b3_mag = -2.34493, 3.73098, 0.12500, -0.01788
    else:
        raise ValueError('maglim, i.e. the limiting magnitude of the GCph sample, must be 23 or 24.5')

    if poly_fit_values is not None:
        assert len(poly_fit_values) == 4, 'a list of 4 best-fit values must be passed'
        np.testing.assert_allclose(np.array(poly_fit_values), np.array((b0_mag, b1_mag, b2_mag, b3_mag)), atol=0,
                                   rtol=1e-5)

    return b0_mag + (b1_mag * z) + (b2_mag * z ** 2) + (b3_mag * z ** 3)


def s_of_z_fs2_fit(z, maglim, poly_fit_values=None):
    """ wrapper function to output the magnification bias as needed in ccl; function written by Marco """
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit
    return (magbias_of_z_fs2_fit(z, maglim, poly_fit_values=poly_fit_values) + 2) / 5


def stepwise_bias(z, gal_bias_vs_zmean, z_edges):
    """
    Returns the bias value for a given redshift, based on stepwise bias values per redshift bin.

    Parameters:
    z (float): The redshift value.
    gal_bias_vs_zmean (list or array): Array containing one bias value per redshift bin.
    z_minus (list or array): Array containing the lower edge of each redshift bin.
    z_plus (list or array): Array containing the upper edge of each redshift bin.

    Returns:
    float: Bias value corresponding to the given redshift.
    """

    assert np.all(np.diff(z_edges) > 0), 'z_edges must be sorted in ascending order'

    # Edge cases for z outside the defined bins
    if z < z_minus[0]:
        return gal_bias_vs_zmean[0]
    if z >= z_plus[-1]:
        return gal_bias_vs_zmean[-1]

    # Find and return the corresponding bias value for z
    for zbin_idx in range(len(z_minus)):
        if z_minus[zbin_idx] <= z < z_plus[zbin_idx]:
            return gal_bias_vs_zmean[zbin_idx]


def build_galaxy_bias_2d_arr(gal_bias_vs_zmean, zmeans, z_edges, zbins, z_grid, bias_model, plot_bias=False):
    """
    Builds a 2d array of shape (len(z_grid), zbins) containing the bias values for each redshift bin. The bias values
    can be given as a function of z, or as a constant value for each redshift bin. Each weight funcion will

    :param gal_bias_vs_zmean: the values of the bias computed in each bin (usually, in the mean).
    :param zmeans: array of z values for which the bias is given.
    :param zbins: number of redshift bins.
    :param z_grid: the redshift grid on which the bias is evaluated. In general, it does need to be very fine.
    :param bias_model: 'unbiased', 'linint', 'constant' or 'step-wise'.
    :param plot_bias: whether to plot the bias values for the different redshift bins.
    :return: gal_bias_vs_zmean: array of shape (len(z_grid), zbins) containing the bias values for each redshift bin.
    """

    assert len(gal_bias_vs_zmean) == zbins, 'gal_bias_vs_zmean must be an array of length zbins'
    assert len(zmeans) == zbins, 'zmeans must be an array of length zbins'

    if bias_model == 'unbiased':
        gal_bias_2d_arr = np.ones((len(z_grid), zbins))
    elif bias_model == 'linint':
        # linear interpolation
        galaxy_bias_func = scipy.interpolate.interp1d(zmeans, gal_bias_vs_zmean, kind='linear',
                                                      fill_value=(gal_bias_vs_zmean[0], gal_bias_vs_zmean[-1]),
                                                      bounds_error=False)
        gal_bias_1d_arr = galaxy_bias_func(z_grid)
        gal_bias_2d_arr = np.repeat(gal_bias_1d_arr[:, np.newaxis], zbins, axis=1)
    elif bias_model == 'constant':
        # this is the *only* case in which the bias is different for each zbin
        gal_bias_2d_arr = np.repeat(gal_bias_vs_zmean[np.newaxis, :], len(z_grid), axis=0)
    elif bias_model == 'step-wise':
        assert z_edges is not None, 'z_edges must be provided for step-wise bias'
        assert len(z_edges) == zbins + 1, 'z_edges must have length zbins + 1'
        gal_bias_1d_arr = np.array([stepwise_bias(z, gal_bias_vs_zmean, z_edges) for z in z_grid])
        gal_bias_2d_arr = np.repeat(gal_bias_1d_arr[:, np.newaxis], zbins, axis=1)
    else:
        raise ValueError('bias_model must be "unbiased", "linint", "constant" or "step-wise"')

    if plot_bias:
        plt.figure()
        plt.title(f'bias_model {bias_model}')
        for zbin_idx in range(zbins):
            plt.plot(z_grid, gal_bias_2d_arr[:, zbin_idx], label=f'zbin {zbin_idx}')
            plt.scatter(zmeans[zbin_idx], gal_bias_vs_zmean[zbin_idx], marker='o', color='black')
        plt.legend()
        plt.show()
        plt.xlabel('$z$')
        plt.ylabel('$b_i(z)$')

    assert gal_bias_2d_arr.shape == (len(z_grid), zbins), 'gal_bias_2d_arr must have shape (len(z_grid), zbins)'

    return gal_bias_2d_arr


def build_ia_bias_1d_arr(z_grid_out, cosmo_ccl, flat_fid_pars_dict, input_z_grid_lumin_ratio=None,
                         input_lumin_ratio=None, output_F_IA_of_z=False):
    """
    None is the default value, in which case we use ISTF fiducial values (or the cosmo object)
    :param input_z_grid_lumin_ratio:
    :param input_lumin_ratio:
    :param z_grid_out: the redshift grid on which the IA bias is evaluated (which can be different from the one used for
    the luminosity ratio, which are stored in z_grid_lumin_ratio! Note the presence of the interpolator)
    :param cosmo:
    :param A_IA:
    :param C_IA:
    :param eta_IA:
    :param beta_IA:
    :return:
    """

    A_IA = flat_fid_pars_dict['Aia']
    eta_IA = flat_fid_pars_dict['eIA']
    beta_IA = flat_fid_pars_dict['bIA']
    C_IA = flat_fid_pars_dict['CIA']

    growth_factor = ccl.growth_factor(cosmo_ccl, a=1 / (1 + z_grid_out))

    if input_lumin_ratio is None and input_z_grid_lumin_ratio is None:
        # in this case, take the defaults
        lumin_ratio_file = np.genfromtxt(f"/Users/davide/Documents/Lavoro/Programmi/common_data/"
                                         f"luminosity_ratio/scaledmeanlum-E2Sa.dat")
        input_z_grid_lumin_ratio = lumin_ratio_file[:, 0]
        input_lumin_ratio = lumin_ratio_file[:, 1]

    if (input_lumin_ratio is None) ^ (input_z_grid_lumin_ratio is None):
        raise ValueError('both input_lumin_ratio and input_z_grid_lumin_ratio must be either None or not None')

    input_lumin_ratio_func = scipy.interpolate.interp1d(input_z_grid_lumin_ratio, input_lumin_ratio, kind='linear',
                                                        fill_value='extrapolate')

    assert len(growth_factor) == len(z_grid_out), 'growth_factor must have the same length ' \
                                                  'as z_grid (it must be computed in these ' \
                                                  'redshifts!)'

    omega_m = cosmo_ccl.cosmo.params.Omega_m
    F_IA_of_z = F_IA(z_grid_out, eta_IA, beta_IA, input_lumin_ratio_func)
    ia_bias = -1 * A_IA * C_IA * omega_m * F_IA_of_z / growth_factor

    if output_F_IA_of_z:
        return (ia_bias, F_IA_of_z)

    return ia_bias


def wig_IST(z_grid, which_wf, zbins=10, gal_bias_2d_array=None, bias_model='step-wise'):
    """
    Computes the photometri Galaxy Clustering kernel, which is equal to the Intrinsic Alignment kernel if the sources
    and lenses distributions are equal. The kernel is computed on a grid of redshifts z_grid, and is a 2d array of
    shape (len(z_grid), zbins). The kernel is computed for each redshift bin, and the bias is assumed to be constant
    :param bias_model:
    :param z_grid:
    :param which_wf:
    :param zbins:
    :param gal_bias_2d_array:
    :return:
    """

    if gal_bias_2d_array is None:
        z_values = ISTF.photoz_bins['z_mean']
        bias_values = np.asarray([b_of_z_analytical(z) for z in z_values])
        gal_bias_2d_array = build_galaxy_bias_2d_arr(bias_values, z_values, zbins, z_grid, bias_model)

    assert gal_bias_2d_array.shape == (len(z_grid), zbins), 'gal_bias_2d_array must have shape (len(z_grid), zbins)'

    # TODO There is probably room for optimization here, no need to use the callable for niz, just use the array...
    # something like this (but it's already normalized...)
    # result = (niz_analytical_arr_norm / n_bar[zbin_idx_array]).T * H0 * csmlib.E(z_grid) / c

    # result = (niz(zbin_idx_array, z_grid) / n_bar[zbin_idx_array]).T * H0 * csmlib.E(z_grid) / c
    result = W_IA(z_grid).T  # it's the same! unless the sources are different

    if which_wf == 'with_galaxy_bias':
        result *= gal_bias_2d_array
        return result
    elif which_wf == 'without_galaxy_bias':
        return result
    elif which_wf == 'galaxy_bias_only':
        return gal_bias_2d_array
    else:
        raise ValueError('which_wf must be "with_galaxy_bias", "without_galaxy_bias" or "galaxy_bias_only"')


########################################################################################################################
########################################################################################################################
########################################################################################################################


# def instantiate_ISTFfid_PyCCL_cosmo_obj():
#     Om_m0, Om_b0, Om_nu0 = ISTF.primary['Om_m0'], ISTF.primary['Om_b0'], ISTF.neutrino_params['Om_nu0']
#     Om_Lambda0 = ISTF.extensions['Om_Lambda0']
#     Om_c0 = Om_m0 - Om_b0 - Om_nu0
#     Om_k0 = csmlib.get_omega_k0(Om_m0, Om_Lambda0)
#
#     cosmo_ccl = ccl.Cosmology(Omega_c=Om_c0, Omega_b=ISTF.primary['Om_b0'], w0=ISTF.primary['w_0'],
#                               wa=ISTF.primary['w_a'], h=ISTF.primary['h_0'], sigma8=ISTF.primary['sigma_8'],
#                               n_s=ISTF.primary['n_s'], m_nu=ISTF.extensions['m_nu'], Omega_k=Om_k0)
#     return cosmo_ccl


def wf_ccl(z_grid, probe, which_wf, flat_fid_pars_dict, cosmo_ccl, dndz_tuple, ia_bias_tuple=None, gal_bias_tuple=None,
           mag_bias_tuple=None, return_ccl_obj=False, n_samples=1000):
    """
    Computes the CCL kernel for the given probe, on the given redshift grid.
    :param z_grid:
    :param probe:
    :param which_wf:
    :param flat_fid_pars_dict:
    :param cosmo_ccl:
    :param dndz_tuple:
    :param ia_bias_tuple:
    :param gal_bias_tuple:
    :param bias_model:
    :param return_ccl_obj:
    :param n_samples:
    :return:
    """
    assert len(dndz_tuple) == 2, 'dndz must be a tuple of length 2'
    assert dndz_tuple[0].shape[0] == dndz_tuple[1].shape[0], ('dndz must be a tuple of two arrays of shape len(z_grid) '
                                                              'and (len(z_grid), zbins) respectively')
    assert probe in ['lensing', 'galaxy']

    zbins = dndz_tuple[1].shape[1]

    a_arr = csmlib.z_to_a(z_grid)
    comoving_distance = ccl.comoving_radial_distance(cosmo_ccl, a_arr)

    if probe == 'lensing':

        # build intrinsic alignment bias array
        if ia_bias_tuple is None:
            ia_bias_1d = build_ia_bias_1d_arr(z_grid_out=z_grid, cosmo_ccl=cosmo_ccl,
                                              flat_fid_pars_dict=flat_fid_pars_dict,
                                              input_z_grid_lumin_ratio=None,
                                              input_lumin_ratio=None,
                                              output_F_IA_of_z=False)
            ia_bias_tuple = (z_grid, ia_bias_1d)

        assert len(ia_bias_tuple) == 2, 'ia_bias must be a tuple of length 2'
        assert ia_bias_tuple[0].shape == ia_bias_tuple[1].shape, 'ia_bias must be a tuple of two arrays of len(z_grid)'

        wf_lensing_obj = [ccl.tracers.WeakLensingTracer(cosmo=cosmo_ccl,
                                                        dndz=(dndz_tuple[0], dndz_tuple[1][:, zbin_idx]),
                                                        ia_bias=ia_bias_tuple,
                                                        use_A_ia=False,
                                                        n_samples=n_samples) for zbin_idx in range(zbins)]

        if return_ccl_obj:
            return wf_lensing_obj

        wf_lensing_arr = np.asarray([wf_lensing_obj[zbin_idx].get_kernel(comoving_distance)
                                     for zbin_idx in range(zbins)])

        if which_wf == 'with_IA':
            wf_gamma = wf_lensing_arr[:, 0, :].T
            wf_ia = wf_lensing_arr[:, 1, :].T
            result = wf_gamma + ia_bias_tuple[1][:, None] * wf_ia
            return result
        elif which_wf == 'without_IA':
            return wf_lensing_arr[:, 0, :].T
        elif which_wf == 'IA_only':
            return wf_lensing_arr[:, 1, :].T
        else:
            raise ValueError('which_wf must be "with_IA", "without_IA" or "IA_only"')

    elif probe == 'galaxy':

        assert len(gal_bias_tuple) == 2, 'gal_bias_tuple must be a tuple of length 2'
        assert gal_bias_tuple[0].shape[0] == len(z_grid), 'gal_bias_tuple[0] must have shape len(z_grid)'
        assert gal_bias_tuple[1].shape == (len(z_grid), zbins), 'gal_bias_tuple[1] must have shape (len(z_grid), zbins)'

        if mag_bias_tuple is not None:
            assert len(mag_bias_tuple) == 2, 'mag_bias_tuple must be a tuple of length 2'
            assert mag_bias_tuple[0].shape[0] == len(z_grid), 'mag_bias_tuple[0] must have shape len(z_grid)'
            assert mag_bias_tuple[1].shape == (
                len(z_grid), zbins), 'mag_bias_tuple[1] must have shape (len(z_grid), zbins)'

            def mag_bias(zbin_idx): return (mag_bias_tuple[0], mag_bias_tuple[1][:, zbin_idx])
        else:
            mag_bias = None

        wf_galaxy_obj = [ccl.tracers.NumberCountsTracer(cosmo_ccl,
                                                        has_rsd=False,
                                                        dndz=(dndz_tuple[0], dndz_tuple[1][:, zbin_idx]),
                                                        bias=(gal_bias_tuple[0], gal_bias_tuple[1][:, zbin_idx]),
                                                        mag_bias=mag_bias(
                                                            zbin_idx) if mag_bias_tuple is not None else mag_bias,
                                                        n_samples=n_samples)
                         for zbin_idx in range(zbins)]

        if return_ccl_obj:
            return wf_galaxy_obj

        wf_galaxy_arr = np.asarray([wf_galaxy_obj[zbin_idx].get_kernel(comoving_distance) for zbin_idx in range(zbins)])

        if which_wf == 'with_galaxy_bias':
            result = wf_galaxy_arr[:, 0, :] * gal_bias_tuple[1].T
            return result.T
        elif which_wf == 'without_galaxy_bias':
            return wf_galaxy_arr[:, 0, :].T
        elif which_wf == 'galaxy_bias_only':
            return gal_bias_tuple[1]
        else:
            raise ValueError('which_wf must be "with_galaxy_bias", "without_galaxy_bias" or "galaxy_bias_only"')


def wf_galaxy_ccl(z_grid, which_wf, fiducial_params, cosmo, gal_bias_2d_array=None, bias_model='step-wise',
                  return_PyCCL_object=False, dndz=None, n_samples=1000):
    zbins = dndz[1].shape[1]

    # build galaxy bias bias array
    if gal_bias_2d_array is None:
        warnings.warn('the bias implementation should be improved, it\'s not very clean. read everything'
                      ' from the fiducals dict!!')
        z_values = np.asarray([fiducial_params[f'zmean{zi:02d}_photo'] for zi in range(1, zbins + 1)])
        bias_values = np.asarray([fiducial_params[f'b{zi:02d}_photo'] for zi in range(1, zbins + 1)])
        gal_bias_2d_array = build_galaxy_bias_2d_arr(bias_values, z_values, zbins, z_grid, bias_model)
    elif np.all(gal_bias_2d_array == 1):  # i.e., no galaxy bias
        gal_bias_2d_array = np.ones((len(z_grid), zbins))

    assert gal_bias_2d_array.shape == (len(z_grid), zbins), 'gal_bias_2d_array must have shape (len(z_grid), zbins)'

    # redshift distribution
    if dndz is None:
        niz_unnormalized_arr = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
        niz_normalized_arr = normalize_niz_simps(niz_unnormalized_arr, z_grid).T  # ! unnecessary to normalize
        dndz = (z_grid, niz_normalized_arr)
        assert niz_normalized_arr.shape == (
            len(z_grid), zbins), 'niz_normalized_arr must have shape (len(z_grid), zbins)'

    assert len(dndz) == 2, 'dndz must be a tuple of length 2'
    assert dndz[0].shape[0] == dndz[1].shape[0], ('dndz must be a tuple of two arrays of shape len(z_grid) and '
                                                  '(len(z_grid), zbins) respectively')

    wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(dndz[0], dndz[1][:, zbin_idx]),
                                          bias=(z_grid, gal_bias_2d_array[:, zbin_idx]), mag_bias=None,
                                          n_samples=n_samples)
           for zbin_idx in range(zbins)]

    if return_PyCCL_object:
        return wig

    a_arr = csmlib.z_to_a(z_grid)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)
    wig_nobias_PyCCL_arr = np.asarray([wig[zbin_idx].get_kernel(chi) for zbin_idx in range(zbins)])

    if which_wf == 'with_galaxy_bias':
        result = wig_nobias_PyCCL_arr[:, 0, :] * gal_bias_2d_array.T
        return result.T
    elif which_wf == 'without_galaxy_bias':
        return wig_nobias_PyCCL_arr[:, 0, :].T
    elif which_wf == 'galaxy_bias_only':
        return gal_bias_2d_array
    else:
        raise ValueError('which_wf must be "with_galaxy_bias", "without_galaxy_bias" or "galaxy_bias_only"')


def wf_lensing_ccl(z_grid, which_wf, cosmo, fid_pars_dict, dndz=None,
                   ia_bias=None, growth_factor=None,
                   return_PyCCL_object=False, n_samples=1000):
    """ This is a wrapper function to call the kernels with PyCCL. Arguments that default to None will be set to the
    ISTF values."""

    warnings.warn('TODO remove growth factor argument')

    zbins = dndz[1].shape[1]

    A_IA = fid_pars_dict['A_IA']
    eta_IA = fid_pars_dict['eta_IA']
    beta_IA = fid_pars_dict['beta_IA']
    C_IA = fid_pars_dict['C_IA']

    # build intrinsic alignment bias array
    if ia_bias is None:
        z_grid_lumin_ratio = lumin_ratio_file[:, 0]
        lumin_ratio = lumin_ratio_file[:, 1]
        ia_bias_1d = build_ia_bias_1d_arr(z_grid_out=z_grid, cosmo_ccl=cosmo, fid_pars_dict=fid_pars_dict,
                                          input_z_grid_lumin_ratio=z_grid_lumin_ratio,
                                          input_lumin_ratio=lumin_ratio,
                                          output_F_IA_of_z=False)
        ia_bias = (z_grid, ia_bias_1d)
    else:
        ia_bias_1d = ia_bias[1]

    assert len(ia_bias) == 2, 'ia_bias must be a tuple of length 2'
    assert ia_bias[0].shape == ia_bias[1].shape, 'ia_bias must be a tuple of two arrays of the same shape'

    # redshift distribution
    if dndz is None:
        niz_unnormalized_arr = np.asarray([niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
        niz_normalized_arr = normalize_niz_simps(niz_unnormalized_arr, z_grid).T  # ! unnecessary to normalize?
        dndz = (z_grid, niz_normalized_arr)

    assert len(dndz) == 2, 'dndz must be a tuple of length 2'
    assert dndz[0].shape[0] == dndz[1].shape[0], ('dndz must be a tuple of two arrays of shape len(z_grid) and '
                                                  '(len(z_grid), zbins) respectively')

    # compute the tracer objects
    wf_lensing_obj = [
        ccl.tracers.WeakLensingTracer(cosmo, dndz=(dndz[0], dndz[1][:, zbin_idx]), ia_bias=ia_bias, use_A_ia=False,
                                      n_samples=n_samples) for zbin_idx in range(zbins)]

    if return_PyCCL_object:
        return wf_lensing_obj

    # get the radial kernels
    # comoving distance of z
    a_arr = 1 / (1 + z_grid)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)
    wf_lensing = np.asarray([wf_lensing_obj[zbin_idx].get_kernel(chi) for zbin_idx in range(zbins)])

    # these methods do not return ISTF kernels:
    # for wil, I have the two components w_gamma and w_IA separately, see below
    if which_wf == 'with_IA':
        wf_gamma = wf_lensing[:, 0, :].T
        wf_ia = wf_lensing[:, 1, :].T
        result = wf_gamma + ia_bias_1d[:, None] * wf_ia

        # growth_factor_PyCCL = ccl.growth_factor(cosmo, a=1 / (1 + z_grid))
        # result = wf_gamma - (A_IA * C_IA * cosmo.cosmo.params.Omega_m * F_IA(
        #     z_grid)) / growth_factor_PyCCL * wf_ia
        return result
    elif which_wf == 'without_IA':
        return wf_lensing[:, 0, :].T
    elif which_wf == 'IA_only':
        return wf_lensing[:, 1, :].T
    else:
        raise ValueError('which_wf must be "with_IA", "without_IA" or "IA_only"')


################################################# cl_quad computation #######################################################

# TODO these contain cosmology dependence...
# cosmo_classy = csmlib.cosmo_classy
# cosmo_astropy = csmlib.cosmo_astropy
# k_grid, pk = csmlib.calculate_power(k_grid, z_grid, cosmo_classy, use_h_units=use_h_units)

# wrapper functions, just to shorten the names
# pk_nonlin_wrap = partial(csmlib.calculate_power, cosmo_classy=cosmo_classy, use_h_units=use_h_units,
#                          Pk_kind='nonlinear')  # TODO update this
# kl_wrap = partial(csmlib.k_limber, use_h_units=use_h_units, cosmo_astropy=cosmo_astropy)


# these are no longer needed, since I can use partial
# def pk_wrap(k_ell, z, cosmo_classy=cosmo_classy, use_h_units=use_h_units, Pk_kind='nonlinear'):
#     """just a wrapper function to set some args to default values"""
#     return csmlib.calculate_power(k_ell, z, cosmo_classy, use_h_units=use_h_units, Pk_kind=Pk_kind)


# def kl_wrap(ell, z, use_h_units=use_h_units):
#     """another simple wrapper function, so as not to have to rewrite use_h_units=use_h_units"""
#     return csmlib.k_limber(ell, z, use_h_units=use_h_units)


def K_ij(z, wf_A, wf_B, i: int, j: int):
    return wf_A(z, j) * wf_B(z, i) / (csmlib.E(z) * csmlib.r(z) ** 2)


def cl_partial_integrand(z, wf_A, wf_B, i: int, j: int, ell):
    return K_ij(z, wf_A, wf_B, i, j) * pk_nonlin_wrap(kl_wrap(ell, z), z)


def cl_partial_integral(wf_A, wf_B, i: int, j: int, zbin: int, ell):
    result = c / H0 * quad(cl_partial_integrand, z_minus[zbin], z_plus[zbin], args=(wf_A, wf_B, i, j, ell))[0]
    return result


# summing the partial integrals
def sum_cl_partial_integral(wf_A, wf_B, i: int, j: int, ell):
    print('THIS BIAS IS WRONG; MOREOVER, AM I NOT INCLUDING IT IN THE KERNELS?')
    warnings.warn('in this version the bias is not included in the kernels')
    warnings.warn('understand the bias array')
    result = 0
    for zbin in range(zbins):
        result += cl_partial_integral(wf_A, wf_B, i, j, zbin, ell) * bias_array[zbin]
    return result


###### OLD BIAS ##################
def cl_integrand(z, wf_A, wf_B, zi, zj, ell):
    return ((wf_A(z)[zi] * wf_B(z)[zj]) / (csmlib.E(z) * csmlib.r(z) ** 2)) * pk_nonlin_wrap(kl_wrap(ell, z), z)


def cl_quad(wf_A, wf_B, ell, zi, zj):
    """ when used with LG or GG, this implements the "old bias"
    """
    result = c / H0 * quad(cl_integrand, z_min, z_max_cl, args=(wf_A, wf_B, zi, zj, ell))[0]
    # xxx maybe you can try with scipy.integrate.romberg?
    return result


def cl_simps(wf_A, wf_B, ell, zi, zj):
    """ when used with LG or GG, this implements the "old bias"
    """
    integrand = [cl_integrand(z, wf_A, wf_B, zi, zj, ell) for z in z_grid]
    # integrand = c/H0 * [cl_integrand(z, wf_A, wf_B, zi, zj, ell) for z in z_grid] prefactor??
    return scipy.integrate.simps(integrand, z_grid)


def get_cl_3D_array(wf_A, wf_B, ell_values):
    # TODO optimize this with triu and/or list comprehensions
    nbl = len(ell_values)
    cl_3D = np.zeros((nbl, zbins, zbins))

    is_auto_spectrum = False
    if wf_A == wf_B:
        is_auto_spectrum = True

    if is_auto_spectrum:
        for ell_idx, ell_val in enumerate(ell_values):
            for zi, zj in zip(np.triu_indices(zbins)[0], np.triu_indices(zbins)[1]):
                cl_3D[ell_idx, zi, zj] = cl_quad(wf_A, wf_B, ell_val, zi, zj)
            cl_3D[ell_idx, :, :] = mm.symmetrize_2d_array(cl_3D[ell_idx, :, :])
    elif not is_auto_spectrum:
        for ell_idx, ell_val in enumerate(ell_values):
            cl_3D[ell_idx, :, :] = np.array([[cl_quad(wf_A, wf_B, ell_val, zi, zj)
                                              for zi in range(zbins)]
                                             for zj in range(zbins)])
    else:
        raise ValueError('is_auto_spectrum must be a bool')

    return cl_3D


def cl_PyCCL(wf_A, wf_B, ell, zbins, p_of_k_a, cosmo, limber_integration_method='qag_quad'):
    # instantiate cosmology

    is_auto_spectrum = False
    if wf_A == wf_B:
        is_auto_spectrum = True

    nbl = len(ell)

    if is_auto_spectrum:
        cl_3D = np.zeros((nbl, zbins, zbins))
        for zi, zj in zip(np.triu_indices(zbins)[0], np.triu_indices(zbins)[1]):
            cl_3D[:, zi, zj] = ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=p_of_k_a,
                                              limber_integration_method=limber_integration_method)
        for ell in range(nbl):
            cl_3D[ell, :, :] = mm.symmetrize_2d_array(cl_3D[ell, :, :])

    elif not is_auto_spectrum:
        # be very careful with the order of the zi, zj loops: you have to revert them in NESTED list comprehensions to
        # have zi as first axis and zj as second axis (the code below is tested and works)
        cl_3D = np.array([[ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=p_of_k_a,
                                          limber_integration_method=limber_integration_method)
                           for zj in range(zbins)]
                          for zi in range(zbins)]
                         ).transpose(2, 0, 1)  # transpose to have ell as first axis
    else:
        raise ValueError('is_auto_spectrum must be either True or False')

    return cl_3D


def stem(cl_4d, variations_arr, zbins, nbl):
    # instantiate array of derivatives
    dcl_3d = np.zeros((nbl, zbins, zbins))

    # create copy of the "x" and "y" arrays, because their items could get popped by the stem algorithm
    cl_4d_cpy = cl_4d.copy()
    variations_arr_cpy = variations_arr.copy()

    # TODO is there a way to specify the axis along which to fit, instead of having to loop over i, j, ell?
    for zi in range(zbins):
        for zj in range(zbins):
            for ell in range(nbl):

                # perform linear fit
                angular_coefficient, intercept = np.polyfit(variations_arr_cpy, cl_4d_cpy[:, ell, zi, zj], deg=1)
                fitted_y_values = angular_coefficient * variations_arr_cpy + intercept

                # check % difference
                perc_diffs = mm.percent_diff(cl_4d_cpy[:, ell, zi, zj], fitted_y_values)

                # as long as any element has a percent deviation greater than 1%, remove first and last values
                while np.any(np.abs(perc_diffs) > 1):
                    # if the condition is satisfied, remove the first and last values
                    cl_4d_cpy = np.delete(cl_4d_cpy, [0, -1], axis=0)
                    variations_arr_cpy = np.delete(variations_arr_cpy, [0, -1])

                    # re-compute the fit on the reduced set
                    angular_coefficient, intercept = np.polyfit(variations_arr_cpy, cl_4d_cpy[:, ell, zi, zj], deg=1)
                    fitted_y_values = angular_coefficient * variations_arr_cpy + intercept

                    # test again
                    perc_diffs = mm.percent_diff(cl_4d_cpy[:, ell, zi, zj], fitted_y_values)

                    # breakpoint()
                    # plt.figure()
                    # plt.plot(variations_arr_cpy, fitted_y_values, '--', lw=2)
                    # plt.plot(variations_arr, cl_4d_cpy[:, ell, zi, zj][:, ell, zi, zj], marker='o')

                # store the value of the derivative
                dcl_3d[ell, zi, zj] = angular_coefficient

    return dcl_3d


def cls_and_derivatives(fiducial_values_dict, extra_parameters, list_params_to_vary, zbins, dndz_tuple,
                        ell_LL, ell_GL, ell_GG,
                        bias_model, pk=None, use_only_flat_models=True, compute_in_parallel=False):
    """
    Compute the derivatives of the power spectrum with respect to the free parameters.
    The function is not single-probe for efficiency reasons
    """
    # TODO cleanup the function, + make it single-probe
    # TODO implement checks on the input parameters
    # TODO input dndz galaxy and IA bias

    # sanity check: the fiducial values dict must contain all the varied parameters
    for param in list_params_to_vary:
        assert param in fiducial_values_dict.keys(), f'{param} is not in the fiducial values dict'

    nbl_WL = len(ell_LL)
    nbl_GC = len(ell_GG)
    nbl_XC = len(ell_GL)

    percentages = np.asarray((-10., -5., -3.75, -2.5, -1.875, -1.25, -0.625, 0,
                              0.625, 1.25, 1.875, 2.5, 3.75, 5., 10.)) / 100
    num_points_derivative = len(percentages)

    z_grid = np.linspace(1e-3, 3, 1000)

    # declare cl and dcl vectors
    cl_LL, cl_GL, cl_GG = {}, {}, {}
    dcl_LL, dcl_GL, dcl_GG = {}, {}, {}

    # loop over the free parameters and store the cls in a dictionary
    for param_to_vary in list_params_to_vary:

        assert param_to_vary in fiducial_values_dict.keys(), f'{param_to_vary} is not in the fiducial values dict'

        t0 = time.perf_counter()

        print(f'working on {param_to_vary}...')

        # shift the parameter
        varied_param_values = fiducial_values_dict[param_to_vary] + fiducial_values_dict[param_to_vary] * percentages
        if param_to_vary == "w_a":  # wa is 0! take directly the percentages
            varied_param_values = percentages

        # ricorda che, quando shifti OmegaM va messo OmegaCDM in modo che OmegaB + OmegaCDM dia il valore corretto di OmegaM,
        # mentre quando shifti OmegaB deve essere aggiustato sempre OmegaCDM in modo che OmegaB + OmegaCDM = 0.32; per OmegaX
        # lo shift ti darà un OmegaM + OmegaDE diverso da 1 il che corrisponde appunto ad avere modelli non piatti

        # this dictionary will contain the shifted values of the parameters; it is initialized with the fiducial values
        # for each new parameter to be varied
        # ! important note: the fiducial and varied_fiducial dict contain all cosmological parameters, not just the ones
        # ! that are varied (specified in list_params_to_vary). This is to be able to change the set of
        # ! varied parameters easily
        varied_fiducials = deepcopy(fiducial_values_dict)

        # instantiate derivatives array for the given free parameter key
        cl_LL[param_to_vary] = np.zeros((num_points_derivative, nbl_WL, zbins, zbins))
        cl_GL[param_to_vary] = np.zeros((num_points_derivative, nbl_GC, zbins, zbins))
        cl_GG[param_to_vary] = np.zeros((num_points_derivative, nbl_GC, zbins, zbins))

        if compute_in_parallel:
            results = Parallel(n_jobs=4, backend='threading')(
                delayed(cl_parallel_helper)(param_to_vary, variation_idx, varied_fiducials, cl_LL, cl_GL, cl_GG,
                                            fiducial_values_dict, extra_parameters,
                                            list_params_to_vary, zbins, dndz_tuple, ell_LL, ell_GL, ell_GG,
                                            bias_model, pk=pk, use_only_flat_models=use_only_flat_models) for
                variation_idx, varied_fiducials[param_to_vary] in tqdm(enumerate(varied_param_values)))

            # Collect the results
            for variation_idx, cl_LL_part, cl_GL_part, cl_GG_part in results:
                cl_LL[param_to_vary][variation_idx, :, :, :] = cl_LL_part
                cl_GL[param_to_vary][variation_idx, :, :, :] = cl_GL_part
                cl_GG[param_to_vary][variation_idx, :, :, :] = cl_GG_part

        else:

            for variation_idx, varied_fiducials[param_to_vary] in tqdm(enumerate(varied_param_values)):

                if use_only_flat_models:
                    # in this case I want omk = 0, so if Om_Lambda0 varies Om_m0 will have to be adjusted and vice versa
                    # (and Om_m0 is adjusted by adjusting Omega_CDM), see the else statement

                    assert fiducial_values_dict['Om_k0'] == 0, 'if use_only_flat_models is True, Om_k0 must be 0'
                    assert varied_fiducials['Om_k0'] == 0, 'if use_only_flat_models is True, Om_k0 must be 0'
                    assert 'Om_k0' not in list_params_to_vary, 'if use_only_flat_models is True, ' \
                                                               'Om_k0 must not be in list_params_to_vary'

                    # If I vary Om_Lambda0 and Om_k0 = 0, I need to adjust Om_m0
                    if param_to_vary == 'Om_Lambda0':
                        varied_fiducials['Om_m0'] = 1 - varied_fiducials['Om_Lambda0']

                else:
                    # If I vary Om_Lambda0 or Om_m0 and Om_k0 can vary, I need to adjust Om_k0
                    varied_fiducials['Om_k0'] = 1 - varied_fiducials['Om_m0'] - varied_fiducials['Om_Lambda0']
                    if np.abs(varied_fiducials['Om_k0']) < 1e-8:
                        varied_fiducials['Om_k0'] = 0

                varied_fiducials['Om_nu0'] = csmlib.get_omega_nu0(varied_fiducials['m_nu'],
                                                                  varied_fiducials['h'],
                                                                  n_eff=varied_fiducials['N_eff'])

                # print to check that the Omegas, if you need to debug
                # Omega_c = (varied_fiducials['Om_m0'] - varied_fiducials['Omega_b'] - varied_fiducials['Om_nu0'])
                # print(
                #     f'Om_m0 = {varied_fiducials["Om_m0"]:.4f}, Omega_c = {Omega_c:.4f}, Omega_b = {varied_fiducials["Omega_b"]:.4f}, '
                #     f'Om_k0 = {varied_fiducials["Om_k0"]:.4f}, Om_nu0 = {varied_fiducials["Om_nu0"]:.4f}, Om_Lambda0 = {varied_fiducials["Om_Lambda0"]:.4f}')

                cosmo_ccl = csmlib.instantiate_cosmo_ccl_obj(varied_fiducials, extra_parameters)

                assert (varied_fiducials['Om_m0'] / cosmo_ccl.cosmo.params.Omega_m - 1) < 1e-7, \
                    'varied_fiducials["Om_m0"] is not the same as cosmo_ccl.cosmo.params.Omega_m'

                # wl_kernel = wf_lensing_ccl(z_grid, 'with_IA', cosmo=cosmo_ccl, dndz=dndz_tuple,
                #                            ia_bias=None, A_IA=varied_fiducials['A_IA'],
                #                            eta_IA=varied_fiducials['eta_IA'],
                #                            beta_IA=varied_fiducials['beta_IA'],
                #                            C_IA=varied_fiducials['C_IA'],
                #                            growth_factor=None,
                #                            return_PyCCL_object=True, n_samples=1000)

                # gc_kernel = wf_galaxy_ccl(z_grid, 'with_galaxy_bias', fiducial_params=varied_fiducials, cosmo=cosmo_ccl,
                #                           gal_bias_2d_array=None, bias_model=bias_model, return_PyCCL_object=True,
                #                           dndz=dndz, n_samples=1000)
                #
                wl_kernel = wf_ccl(z_grid, probe='lensing', which_wf='with_IA', fid_pars_dict=varied_fiducials,
                                   cosmo_ccl=cosmo_ccl, dndz_tuple=dndz_tuple,
                                   ia_bias_tuple=None,
                                   gal_bias_tuple=None,
                                   bias_model=bias_model,
                                   return_ccl_obj=True, n_samples=1000)
                gc_kernel = wf_ccl(z_grid, probe='galaxy', which_wf='with_galaxy_bias', fid_pars_dict=varied_fiducials,
                                   cosmo_ccl=cosmo_ccl, dndz_tuple=dndz_tuple,
                                   ia_bias_tuple=None,
                                   gal_bias_tuple=None,
                                   bias_model=bias_model,
                                   return_ccl_obj=True, n_samples=1000)

                cl_LL[param_to_vary][variation_idx, :, :, :] = cl_PyCCL(wl_kernel, wl_kernel, ell_LL, zbins,
                                                                        p_of_k_a=pk,
                                                                        cosmo=cosmo_ccl)
                cl_GL[param_to_vary][variation_idx, :, :, :] = cl_PyCCL(gc_kernel, wl_kernel, ell_GL, zbins,
                                                                        p_of_k_a=pk,
                                                                        cosmo=cosmo_ccl)
                cl_GG[param_to_vary][variation_idx, :, :, :] = cl_PyCCL(gc_kernel, gc_kernel, ell_GG, zbins,
                                                                        p_of_k_a=pk,
                                                                        cosmo=cosmo_ccl)

            print(f'param {param_to_vary} Cls computed in {(time.perf_counter() - t0):.2f} seconds')

        # ! no longer needed?
        # once finished looping over the variations, reset the parameter to its fiducial value
        # list_params_to_vary[param_to_vary] = fiducial_values_dict[param_to_vary]

        dcl_LL[param_to_vary] = stem(cl_LL[param_to_vary], varied_param_values, zbins, nbl_WL)
        dcl_GL[param_to_vary] = stem(cl_GL[param_to_vary], varied_param_values, zbins, nbl_GC)
        dcl_GG[param_to_vary] = stem(cl_GG[param_to_vary], varied_param_values, zbins, nbl_GC)

        print(f'SteM derivative computed for {param_to_vary}')

    return cl_LL, cl_GL, cl_GG, dcl_LL, dcl_GL, dcl_GG


# ! start new parallel
def cls_and_derivatives_parallel_new(fiducial_values_dict, extra_parameters, list_params_to_vary, zbins, dndz, ell_LL,
                                     ell_GL, ell_GG, bias_model, pk=None, use_only_flat_models=True):
    """
    Compute the derivatives of the power spectrum with respect to the free parameters.
    """

    nbl_WL = len(ell_LL)
    nbl_GC = len(ell_GG)

    percentages = np.asarray((-10., -5., -3.75, -2.5, -1.875, -1.25, -0.625, 0,
                              0.625, 1.25, 1.875, 2.5, 3.75, 5., 10.)) / 100
    num_points_derivative = len(percentages)

    z_grid = np.linspace(1e-3, 3, 1000)

    cl_LL, cl_GL, cl_GG = {}, {}, {}
    dcl_LL, dcl_GL, dcl_GG = {}, {}, {}

    for param_to_vary in list_params_to_vary:
        assert param_to_vary in fiducial_values_dict.keys(), f'{param_to_vary} is not in the fiducial values dict'

        t0 = time.perf_counter()
        print(f'working on {param_to_vary}...')

        varied_param_values = fiducial_values_dict[param_to_vary] + fiducial_values_dict[param_to_vary] * percentages
        if param_to_vary == "wa":
            varied_param_values = percentages / 100

        varied_fiducials = deepcopy(fiducial_values_dict)

        cl_LL[param_to_vary] = np.zeros((num_points_derivative, nbl_WL, zbins, zbins))
        cl_GL[param_to_vary] = np.zeros((num_points_derivative, nbl_GC, zbins, zbins))
        cl_GG[param_to_vary] = np.zeros((num_points_derivative, nbl_GC, zbins, zbins))

        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(cl_parallel_helper)(param_to_vary, variation_idx, varied_fiducials, fiducial_values_dict,
                                        extra_parameters, list_params_to_vary, zbins, dndz, ell_LL, ell_GL, ell_GG,
                                        z_grid,
                                        bias_model, pk=pk, use_only_flat_models=use_only_flat_models)
            for variation_idx in range(num_points_derivative))

        for variation_idx, (cl_LL_part, cl_GL_part, cl_GG_part) in enumerate(results):
            cl_LL[param_to_vary][variation_idx, :, :, :] = cl_LL_part
            cl_GL[param_to_vary][variation_idx, :, :, :] = cl_GL_part
            cl_GG[param_to_vary][variation_idx, :, :, :] = cl_GG_part

        print(f'param {param_to_vary} Cls computed in {(time.perf_counter() - t0):.2f} seconds')

        dcl_LL[param_to_vary] = stem(cl_LL[param_to_vary], varied_param_values, zbins, nbl_WL)
        dcl_GL[param_to_vary] = stem(cl_GL[param_to_vary], varied_param_values, zbins, nbl_GC)
        dcl_GG[param_to_vary] = stem(cl_GG[param_to_vary], varied_param_values, zbins, nbl_GC)

    return cl_LL, cl_GL, cl_GG, dcl_LL, dcl_GL, dcl_GG


def cl_parallel_helper_new(param_to_vary, variation_idx, varied_fiducials, fiducial_values_dict, extra_parameters,
                           list_params_to_vary, zbins, dndz, ell_LL, ell_GL, ell_GG, z_grid,
                           bias_model, pk=None, use_only_flat_models=True):
    try:
        if use_only_flat_models:
            assert fiducial_values_dict['Om_k0'] == 0, 'if use_only_flat_models is True, Om_k0 must be 0'
            assert varied_fiducials['Om_k0'] == 0, 'if use_only_flat_models is True, Om_k0 must be 0'
            assert 'Om_k0' not in list_params_to_vary, 'if use_only_flat_models is True, Om_k0 must not be in list_params_to_vary'

            if param_to_vary == 'Om_Lambda0':
                varied_fiducials['Om_m0'] = 1 - varied_fiducials['Om_Lambda0']
        else:
            varied_fiducials['Om_k0'] = 1 - varied_fiducials['Om_m0'] - varied_fiducials['Om_Lambda0']
            if np.abs(varied_fiducials['Om_k0']) < 1e-8:
                varied_fiducials['Om_k0'] = 0

        varied_fiducials['Om_nu0'] = csmlib.get_omega_nu0(varied_fiducials['m_nu'], varied_fiducials['h'],
                                                          n_ur=None, n_eff=varied_fiducials['N_eff'],
                                                          n_ncdm=None,
                                                          neutrino_mass_fac=None, g_factor=None)

        cosmo_ccl = csmlib.instantiate_cosmo_ccl_obj(varied_fiducials, extra_parameters)
        assert (varied_fiducials[
            'Om_m0'] / cosmo_ccl.cosmo.params.Omega_m - 1) < 1e-7, 'Om_m0 is not the same as the one in the fiducial model'

        wl_kernel = wf_lensing_ccl(z_grid, 'with_IA', cosmo=cosmo_ccl, dndz=dndz,
                                   ia_bias=None, A_IA=varied_fiducials['A_IA'],
                                   eta_IA=varied_fiducials['eta_IA'],
                                   beta_IA=varied_fiducials['beta_IA'], C_IA=varied_fiducials['C_IA'],
                                   growth_factor=None,
                                   return_PyCCL_object=True, n_samples=1000)

        gc_kernel = wf_galaxy_ccl(z_grid, 'with_galaxy_bias', fiducial_params=varied_fiducials, cosmo=cosmo_ccl,
                                  gal_bias_2d_array=None, bias_model=bias_model, return_PyCCL_object=True, dndz=dndz,
                                  n_samples=1000)

        cl_LL_part = cl_PyCCL(wl_kernel, wl_kernel, ell_LL, zbins, p_of_k_a=pk, cosmo=cosmo_ccl)
        cl_GL_part = cl_PyCCL(gc_kernel, wl_kernel, ell_GL, zbins, p_of_k_a=pk, cosmo=cosmo_ccl)
        cl_GG_part = cl_PyCCL(gc_kernel, gc_kernel, ell_GG, zbins, p_of_k_a=pk, cosmo=cosmo_ccl)

        return cl_LL_part, cl_GL_part, cl_GG_part

    except Exception as e:
        # Handle exceptions and log the error
        print(f"Error in cl_parallel_helper for parameter {param_to_vary}: {e}")
        return None, None, None


# ! end new parallel


def cls_and_derivatives_parallel(fiducial_values_dict, extra_parameters, list_params_to_vary, zbins, dndz, ell_LL,
                                 ell_GL, ell_GG,
                                 bias_model, pk=None, use_only_flat_models=True):
    """
    Compute the derivatives of the power spectrum with respect to the free parameters
    """
    # TODO cleanup the function, + make it single-probe
    # TODO implement checks on the input parameters
    # TODO input dndz galaxy and IA bias

    nbl_WL = len(ell_LL)
    nbl_GC = len(ell_GG)

    percentages = np.asarray((-10., -5., -3.75, -2.5, -1.875, -1.25, -0.625, 0,
                              0.625, 1.25, 1.875, 2.5, 3.75, 5., 10.)) / 100
    num_points_derivative = len(percentages)

    z_grid = np.linspace(1e-3, 3, 1000)

    # declare cl and dcl vectors
    cl_LL, cl_GL, cl_GG = {}, {}, {}
    dcl_LL, dcl_GL, dcl_GG = {}, {}, {}

    # loop over the free parameters and store the cls in a dictionary
    for param_to_vary in list_params_to_vary:

        assert param_to_vary in fiducial_values_dict.keys(), f'{param_to_vary} is not in the fiducial values dict'

        t0 = time.perf_counter()

        print(f'working on {param_to_vary}...')

        # shift the parameter
        varied_param_values = fiducial_values_dict[param_to_vary] + fiducial_values_dict[param_to_vary] * percentages
        if param_to_vary == "wa":  # wa is 0! take directly the percentages
            varied_param_values = percentages

        # ricorda che, quando shifti OmegaM va messo OmegaCDM in modo che OmegaB + OmegaCDM dia il valore corretto di OmegaM,
        # mentre quando shifti OmegaB deve essere aggiustato sempre OmegaCDM in modo che OmegaB + OmegaCDM = 0.32; per OmegaX
        # lo shift ti darà un OmegaM + OmegaDE diverso da 1 il che corrisponde appunto ad avere modelli non piatti

        # this dictionary will contain the shifted values of the parameters; it is initialized with the fiducial values
        # for each new parameter to be varied
        # ! important note: the fiducial and varied_fiducial dict contain all cosmological parameters, not just the ones
        # ! that are varied (specified in list_params_to_vary). This is to be able to change the set of
        # ! varied parameters easily
        varied_fiducials = deepcopy(fiducial_values_dict)

        # instantiate derivatives array for the given free parameter key
        cl_LL[param_to_vary] = np.zeros((num_points_derivative, nbl_WL, zbins, zbins))
        cl_GL[param_to_vary] = np.zeros((num_points_derivative, nbl_GC, zbins, zbins))
        cl_GG[param_to_vary] = np.zeros((num_points_derivative, nbl_GC, zbins, zbins))

        # ! newwww
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(cl_parallel_helper)(param_to_vary, variation_idx, varied_fiducials, cl_LL, cl_GL, cl_GG,
                                        fiducial_values_dict, extra_parameters,
                                        list_params_to_vary, zbins, dndz, ell_LL, ell_GL, ell_GG,
                                        bias_model, pk=pk, use_only_flat_models=use_only_flat_models) for
            variation_idx, varied_fiducials[param_to_vary] in tqdm(enumerate(varied_param_values)))

        # Collect the results
        for variation_idx, cl_LL_part, cl_GL_part, cl_GG_part in results:
            cl_LL[param_to_vary][variation_idx, :, :, :] = cl_LL_part
            cl_GL[param_to_vary][variation_idx, :, :, :] = cl_GL_part
            cl_GG[param_to_vary][variation_idx, :, :, :] = cl_GG_part
        # ! and newwww

        for variation_idx, varied_fiducials[param_to_vary] in tqdm(enumerate(varied_param_values)):
            print(f'param {param_to_vary} Cls computed in {(time.perf_counter() - t0):.2f} seconds')

        # once finished looping over the variations, reset the parameter to its fiducial value
        # list_params_to_vary[param_to_vary] = fiducial_values_dict[param_to_vary]

        # save the Cls
        dcl_LL[param_to_vary] = stem(cl_LL[param_to_vary], varied_param_values, zbins, nbl_WL)
        dcl_GL[param_to_vary] = stem(cl_GL[param_to_vary], varied_param_values, zbins, nbl_GC)
        dcl_GG[param_to_vary] = stem(cl_GG[param_to_vary], varied_param_values, zbins, nbl_GC)

        print(f'SteM derivative computed for {param_to_vary}')
    return cl_LL, cl_GL, cl_GG, dcl_LL, dcl_GL, dcl_GG


def cl_parallel_helper(param_to_vary, variation_idx, varied_fiducials, cl_LL, cl_GL, cl_GG,
                       fiducial_values_dict, extra_parameters,
                       list_params_to_vary, zbins, dndz, ell_LL, ell_GL, ell_GG,
                       bias_model, pk=None, use_only_flat_models=True):
    if use_only_flat_models:
        # in this case I want omk = 0, so if Om_Lambda0 varies Om_m0 will have to be adjusted and vice versa
        # (and Om_m0 is adjusted by adjusting Omega_CDM), see the else statement

        assert fiducial_values_dict['Om_k0'] == 0, 'if use_only_flat_models is True, Om_k0 must be 0'
        assert varied_fiducials['Om_k0'] == 0, 'if use_only_flat_models is True, Om_k0 must be 0'
        assert 'Om_k0' not in list_params_to_vary, 'if use_only_flat_models is True, ' \
                                                   'Om_k0 must not be in list_params_to_vary'

        # If I vary Om_Lambda0 and Om_k0 = 0, I need to adjust Om_m0
        if param_to_vary == 'Om_Lambda0':
            varied_fiducials['Om_m0'] = 1 - varied_fiducials['Om_Lambda0']

    else:
        # If I vary Om_Lambda0 or Om_m0 and Om_k0 can vary, I need to adjust Om_k0
        varied_fiducials['Om_k0'] = 1 - varied_fiducials['Om_m0'] - varied_fiducials['Om_Lambda0']
        if np.abs(varied_fiducials['Om_k0']) < 1e-8:
            varied_fiducials['Om_k0'] = 0

    varied_fiducials['Om_nu0'] = csmlib.get_omega_nu0(varied_fiducials['m_nu'], varied_fiducials['h'],
                                                      n_ur=None, n_eff=varied_fiducials['N_eff'],
                                                      n_ncdm=None,
                                                      neutrino_mass_fac=None, g_factor=None)

    cosmo_ccl = csmlib.instantiate_cosmo_ccl_obj(varied_fiducials, extra_parameters)

    # warnings.warn('there seems to be a small discrepancy here...')
    assert (varied_fiducials[
        'Om_m0'] / cosmo_ccl.cosmo.params.Omega_m - 1) < 1e-7, 'Om_m0 is not the same as the one in the fiducial model'

    wl_kernel = wf_lensing_ccl(z_grid, 'with_IA', cosmo=cosmo_ccl, dndz=dndz,
                               ia_bias=None, A_IA=varied_fiducials['A_IA'],
                               eta_IA=varied_fiducials['eta_IA'],
                               beta_IA=varied_fiducials['beta_IA'], C_IA=varied_fiducials['C_IA'],
                               growth_factor=None,
                               return_PyCCL_object=True, n_samples=1000)

    gc_kernel = wf_galaxy_ccl(z_grid, 'with_galaxy_bias', fiducial_params=varied_fiducials, cosmo=cosmo_ccl,
                              gal_bias_2d_array=None, bias_model=bias_model, return_PyCCL_object=True, dndz=dndz,
                              n_samples=1000)

    cl_LL[param_to_vary][variation_idx, :, :, :] = cl_PyCCL(wl_kernel, wl_kernel, ell_LL, zbins, p_of_k_a=pk,
                                                            cosmo=cosmo_ccl)
    cl_GL[param_to_vary][variation_idx, :, :, :] = cl_PyCCL(gc_kernel, wl_kernel, ell_GL, zbins, p_of_k_a=pk,
                                                            cosmo=cosmo_ccl)
    cl_GG[param_to_vary][variation_idx, :, :, :] = cl_PyCCL(gc_kernel, gc_kernel, ell_GG, zbins, p_of_k_a=pk,
                                                            cosmo=cosmo_ccl)
    return cl_LL, cl_GL, cl_GG


def shift_nz(zgrid_nz, nz_original, dz_shifts, normalize, plot_nz=False, interpolation_kind='linear', clip_min=0,
             clip_max=3):
    print(f'Shifting n(z), clipping between redshifts {clip_min} and {clip_max}')

    zbins = nz_original.shape[1]
    assert len(dz_shifts) == zbins, 'dz_shifts must have the same length as the number of zbins'
    assert np.all(np.abs(dz_shifts) < 0.1), 'dz_shifts must be small (this is a rough check)'
    assert nz_original.shape[0] == len(zgrid_nz), 'nz_original must have the same length as zgrid_nz'

    colors = cm.rainbow(np.linspace(0, 1, zbins))

    n_of_z_shifted = np.zeros_like(nz_original)
    for zi in range(zbins):
        # not-very-pythonic implementation: create an interpolator for each bin
        n_of_z_func = interp1d(zgrid_nz, nz_original[:, zi], kind=interpolation_kind)
        z_grid_nz_shifted = zgrid_nz - dz_shifts[zi]
        # where < 0, set to 0; where > 3, set to 3
        z_grid_nz_shifted = np.clip(z_grid_nz_shifted, clip_min, clip_max)
        n_of_z_shifted[:, zi] = n_of_z_func(z_grid_nz_shifted)

    if plot_nz:
        plt.figure()
        for zi in range(zbins):
            plt.plot(zgrid_nz, nz_original[:, zi], ls='-', c=colors[zi])
            plt.plot(zgrid_nz, n_of_z_shifted[:, zi], ls='--', c=colors[zi])
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('n(z)')

    if normalize:
        integrals = simps(n_of_z_shifted, zgrid_nz, axis=0)
        n_of_z_shifted /= integrals[None, :]

    return n_of_z_shifted


def get_z_means(zgrid, kernel):
    """ compute the mean of the wf distribution """
    assert kernel.shape[0] == zgrid.shape[0], 'kernel and zgrid must have the same length'
    assert kernel.ndim == 2, 'kernel must be a 2d array'
    z_means = simps(y=kernel * zgrid[:, None], x=zgrid, axis=0) / simps(y=kernel, x=zgrid, axis=0)
    return z_means


def get_z_effective_isaac(zgrid_nz, n_of_z):
    """
    Calculate the effective redshift at which to evaluate the bias.

    The effective redshift is defined as the median of the redshift distribution
    considering only the part of the distribution that is at least 10% of its maximum.

    Parameters:
    z (array-like): Array of redshifts corresponding to the n(z) distribution.
    n_of_z (array-like): The n(z) redshift distribution.

    Returns:
    float: The effective redshift.
    """
    zbins = n_of_z.shape[1]
    effective_z = np.zeros(zbins)

    for zi in range(zbins):
        n_of_zi = n_of_z[:, zi]
        threshold = max(n_of_zi) * 0.1
        effective_z[zi] = np.median(zgrid_nz[n_of_zi > threshold])

    return effective_z
