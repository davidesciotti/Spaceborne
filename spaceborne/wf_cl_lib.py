import warnings
from copy import deepcopy

import scipy
import time
import os
import numpy as np
import pyccl as ccl
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
from scipy.integrate import quad
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os

from spaceborne import sb_lib as sl
from spaceborne import cosmo_lib
from spaceborne import pyccl_interface

ROOT = os.getenv('ROOT')

c = 299792.458  # km/s

dav_to_vinc_par_names = {
    'Om': 'Omega_M',
    'Ob': 'Omega_B',
    'logT': 'HMCode_logT_AGN',
    'ns': 'n_s',
    'ODE': 'Omega_DE',
    's8': 'sigma8',
    'wz': 'w0',
}


def plot_nz_src_lns(zgrid_nz_src, nz_src, zgrid_nz_lns, nz_lns, colors):

    assert nz_src.shape[1] == nz_lns.shape[1], 'number of zbins is not the same'
    zbins = nz_src.shape[1]

    _, ax = plt.subplots(2, 1, sharex=True)
    colors = cm.rainbow(np.linspace(0, 1, zbins))
    for zi in range(zbins):
        ax[0].plot(zgrid_nz_src, nz_src[:, zi], c=colors[zi], label=r'$z_{%d}$' % (zi + 1))
        # ax[0].axvline(zbin_centers_src[zi], c=colors[zi], ls='--', alpha=0.6, label=r'$z_{%d}^{eff}$' % (zi + 1))
        ax[0].fill_between(zgrid_nz_src, nz_src[:, zi], color=colors[zi], alpha=0.2)
        ax[0].set_xlabel('$z$')
        ax[0].set_ylabel(r'$n_i(z) \; {\rm sources}$')
    ax[0].legend(ncol=2)

    for zi in range(zbins):
        ax[1].plot(zgrid_nz_lns, nz_lns[:, zi], c=colors[zi], label=r'$z_{%d}$' % (zi + 1))
        # ax[1].axvline(zbin_centers_lns[zi], c=colors[zi], ls='--', alpha=0.6, label=r'$z_{%d}^{eff}$' % (zi + 1))
        ax[1].fill_between(zgrid_nz_lns, nz_lns[:, zi], color=colors[zi], alpha=0.2)
        ax[1].set_xlabel('$z$')
        ax[1].set_ylabel(r'$n_i(z) \; {\rm lenses}$')
    ax[1].legend(ncol=2)


# @njit
def pph(z_p, z, c_in, z_in, sigma_in, c_out, z_out, sigma_out, f_out):
    first_addendum = (1 - f_out) / (np.sqrt(2 * np.pi) * sigma_in * (1 + z)) * \
        np.exp(-0.5 * ((z - c_in * z_p - z_in) / (sigma_in * (1 + z))) ** 2)
    second_addendum = f_out / (np.sqrt(2 * np.pi) * sigma_out * (1 + z)) * \
        np.exp(-0.5 * ((z - c_out * z_p - z_out) / (sigma_out * (1 + z))) ** 2)
    return first_addendum + second_addendum


# @njit
def n_of_z(z, z_0, n_gal):
    return n_gal * (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))
    # return  (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))


# @njit
def F_IA(z, eta_IA, beta_IA, z_pivot_IA, lumin_ratio_func):
    result = ((1 + z) / (1 + z_pivot_IA)) ** eta_IA * (lumin_ratio_func(z)) ** beta_IA
    return result


# use formula 23 of ISTF paper for Om(z)
# @njit
def Om(z, Om0, cosmo_astropy):
    return Om0 * (1 + z) ** 3 / cosmo_lib.E(z, cosmo_astropy) ** 2


# @njit
def growth_factor_integrand(x, gamma, Om0, cosmo_astropy):
    return Om(x, Om0, cosmo_astropy) ** gamma / (1 + x)


def growth_factor(z, gamma, Om0, cosmo_astropy):
    integral = quad(growth_factor_integrand, 0, z, args=(gamma, Om0, cosmo_astropy))[0]
    return np.exp(-integral)


def b_of_z_analytical(z):
    """simple analytical prescription for the linear galaxy bias:
    b(z) = sqrt(1 + z)
    """
    return np.sqrt(1 + z)


def b_of_z_fs1_leporifit(z):
    """fit to the linear galaxy bias measured from FS1"""
    return 0.5125 + 1.377 * z + 0.222 * z ** 2 - 0.249 * z ** 3


def b_of_z_fs1_pocinofit(z):
    """fit to the linear galaxy bias measured from FS1."""
    a, b, c = 0.81, 2.80, 1.02
    return a * z ** b / (1 + z) + c


def b_of_z_fs2_fit(z, magcut_lens, poly_fit_values=None):
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit

    if poly_fit_values is not None:
        assert len(poly_fit_values) == 4, 'a list of 4 best-fit values must be passed'
        warnings.warn('overwriting default polynomial fit coefficients with user-defined ones')
        b0_gal, b1_gal, b2_gal, b3_gal = poly_fit_values

    else:
        if magcut_lens == 24.5:
            b0_gal, b1_gal, b2_gal, b3_gal = 1.33291, -0.72414, 1.0183, -0.14913
        elif magcut_lens == 23:
            b0_gal, b1_gal, b2_gal, b3_gal = 1.88571, -2.73925, 3.24688, -0.73496
        else:
            raise ValueError('magcut_lens, i.e. the limiting magnitude of the GCph sample, must be 23 or 24.5')

    return b0_gal + (b1_gal * z) + (b2_gal * z ** 2) + (b3_gal * z ** 3)


def magbias_of_z_fs2_fit(z, magcut_lens, poly_fit_values=None):
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit
    if poly_fit_values is not None:
        assert len(poly_fit_values) == 4, 'a list of 4 best-fit values must be passed'
        warnings.warn('overwriting default polynomial fit coefficients with user-defined ones')
        b0_mag, b1_mag, b2_mag, b3_mag = poly_fit_values

    else:
        if magcut_lens == 24.5:
            b0_mag, b1_mag, b2_mag, b3_mag = -1.50685, 1.35034, 0.08321, 0.04279
        elif magcut_lens == 23:
            b0_mag, b1_mag, b2_mag, b3_mag = -2.34493, 3.73098, 0.12500, -0.01788
        else:
            raise ValueError('magcut_lens, i.e. the limiting magnitude of the GCph sample, must be 23 or 24.5')

    return b0_mag + (b1_mag * z) + (b2_mag * z ** 2) + (b3_mag * z ** 3)


def s_of_z_fs2_fit(z, magcut_lens, poly_fit_values=None):
    """ wrapper function to output the magnification bias as needed in ccl; function written by Marco """
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit
    return (magbias_of_z_fs2_fit(z, magcut_lens, poly_fit_values=poly_fit_values) + 2) / 5


def b2g_fs2_fit(z):
    """This function has been fitted by Sylvain G. Beauchamps based on FS2 measurements:
    z_meas = [0.395, 0.7849999999999999, 1.1749999999999998, 1.565, 1.9549999999999998, 2.3449999999999998]
    b2_meas = [-0.25209754,  0.14240271,  0.56409318,  1.06597924,  2.84258843,  4.8300518 ]
    """

    c0, c1, c2, c3 = -0.69682803, 1.60320679, -1.31676159, 0.70271383
    b2g_ofz = c0 + c1 * z + c2 * z ** 2 + c3 * z ** 3
    return b2g_ofz


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
    z_minus = z_edges[:-1]
    z_plus = z_edges[1:]

    # Edge cases for z outside the defined bins
    if z < z_minus[0]:
        return gal_bias_vs_zmean[0]
    if z >= z_plus[-1]:
        return gal_bias_vs_zmean[-1]

    # Find and return the corresponding bias value for z
    for zbin_idx in range(len(z_minus)):
        if z_minus[zbin_idx] <= z < z_plus[zbin_idx]:
            return gal_bias_vs_zmean[zbin_idx]


def build_galaxy_bias_2d_arr(gal_bias_vs_zmean, zmeans, z_edges, zbins, z_grid, bias_model,
                             plot_bias=False, bias_fit_function=None, kwargs_bias_fit_function=None):
    """
    Builds a 2d array of shape (len(z_grid), zbins) containing the bias values for each redshift bin. The bias values
    can be given as a function of z, or as a constant value for each redshift bin. Each weight funcion will

    :param gal_bias_vs_zmean: the values of the bias computed in each bin (usually, in the mean).
    :param zmeans: array of z values for which the bias is given.
    :param zbins: number of redshift bins.
    :param z_grid: the redshift grid on which the bias is evaluated. In general, it does need to be very fine.
    :param bias_model: 'unbiased', 'linint', 'constant' or 'step-wise'.
    :param plot_bias: whether to plot the bias values for the different redshift bins.
    :return: gal_bias_2d_arr: array of shape (len(z_grid), zbins) containing the bias values for each redshift bin.
    """

    if bias_model != 'unbiased':
        # this check can skipped in the unbiased case
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

    elif bias_model == 'polynomial':
        assert bias_fit_function is not None, 'bias_fit_function must be provided for polynomial bias'
        gal_bias_1d_arr = bias_fit_function(z_grid, **kwargs_bias_fit_function)
        # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
        gal_bias_2d_arr = np.repeat(gal_bias_1d_arr.reshape(1, -1), zbins, axis=0).T

    else:
        raise ValueError('bias_model must be "unbiased", "linint", "constant", "step-wise" or "polynomial"')

    if plot_bias:
        plt.figure()
        plt.title(f'bias_model {bias_model}')
        for zbin_idx in range(zbins):
            plt.plot(z_grid, gal_bias_2d_arr[:, zbin_idx], label=f'zbin {zbin_idx + 1}')
            plt.scatter(zmeans[zbin_idx], gal_bias_vs_zmean[zbin_idx], marker='o', color='black')
        plt.legend()
        plt.xlabel('$z$')
        plt.ylabel('$b_i(z)$')
        plt.show()

    assert gal_bias_2d_arr.shape == (len(z_grid), zbins), 'gal_bias_2d_arr must have shape (len(z_grid), zbins)'

    return gal_bias_2d_arr


def build_ia_bias_1d_arr(z_grid_out, cosmo_ccl, ia_dict, lumin_ratio_2d_arr, output_F_IA_of_z=False):
    """
    Computes the intrinsic alignment (IA) bias as a function of redshift.

    This function evaluates the IA bias on a given redshift grid based on the 
    cosmology, intrinsic alignment parameters, and an optional luminosity ratio.
    If no luminosity ratio is provided, the bias assumes a constant luminosity 
    ratio (and requires `beta_IA = 0`).

    Parameters
    ----------
    z_grid_out : array_like
        The redshift grid on which the IA bias is evaluated. This grid can differ
        from the one used for the luminosity ratio, as interpolation is performed.

    cosmo_ccl : pyccl.Cosmology
        The cosmology object from `pyccl`, which provides the cosmological
        parameters and growth factor.

    ia_dict : dict
        A dictionary containing intrinsic alignment parameters. The required keys are:
        - `Aia`: Amplitude of the IA bias.
        - `eIA`: Redshift dependence of the IA bias.
        - `bIA`: Luminosity dependence of the IA bias.
        - `z_pivot_IA`: Pivot redshift for scaling the IA bias.
        - `CIA`: Normalization constant for the IA bias.

    lumin_ratio_2d_arr : array_like or None
        A 2D array of shape (N, 2) representing the luminosity ratio. The first column
        contains the redshift grid, and the second column contains the luminosity ratio.
        If `None`, the luminosity ratio is assumed to be constant (1), and `beta_IA` must be 0.

    output_F_IA_of_z : bool, optional
        If `True`, the function returns the IA bias along with the computed F_IA(z)
        function. Default is `False`.

    Returns
    -------
    ia_bias : array_like
        The intrinsic alignment bias evaluated on `z_grid_out`.

    F_IA_of_z : array_like, optional
        The computed F_IA(z) function, returned only if `output_F_IA_of_z=True`.

    Raises
    ------
    AssertionError
        If `beta_IA != 0` and no luminosity ratio is provided.
        If the growth factor length does not match the redshift grid length.


    Notes
    -----
    - The IA bias is computed as (notice the negative sign!):
      .. math::
         \text{IA Bias} = - A_\text{IA} C_\text{IA} \Omega_m \frac{F_\text{IA}(z)}{\text{Growth Factor}}
    - The growth factor is evaluated using the `pyccl.growth_factor` function.
    """

    A_IA = ia_dict['Aia']
    eta_IA = ia_dict['eIA']
    beta_IA = ia_dict['bIA']
    z_pivot_IA = ia_dict['z_pivot_IA']
    C_IA = ia_dict['CIA']

    growth_factor = ccl.growth_factor(cosmo_ccl, a=1 / (1 + z_grid_out))

    if lumin_ratio_2d_arr is None:
        assert beta_IA == 0, 'if no luminosity ratio file is given, beta_IA must be 0'

    lumin_ratio_func = get_luminosity_ratio_interpolator(lumin_ratio_2d_arr)

    assert len(growth_factor) == len(z_grid_out), 'growth_factor must have the same length ' \
                                                  'as z_grid (it must be computed in these ' \
                                                  'redshifts!)'

    omega_m = cosmo_ccl.cosmo.params.Omega_m
    F_IA_of_z = F_IA(z_grid_out, eta_IA, beta_IA, z_pivot_IA, lumin_ratio_func)
    ia_bias = -1 * A_IA * C_IA * omega_m * F_IA_of_z / growth_factor

    if output_F_IA_of_z:
        return (ia_bias, F_IA_of_z)

    return ia_bias


def get_luminosity_ratio_interpolator(lumin_ratio_2d_arr):
    """
    Returns an interpolator function for the luminosity ratio or a default constant function.
    :param lumin_ratio_2d_arr: A 2D numpy array with shape (N, 2) where column 0 is z and column 1 is the ratio.
    :return: Interpolator function for luminosity ratio.
    """
    if lumin_ratio_2d_arr is None:
        def func(z): return 1  # Default to constant luminosity ratio

    elif isinstance(lumin_ratio_2d_arr, np.ndarray) and lumin_ratio_2d_arr.shape[1] == 2:
        func = scipy.interpolate.interp1d(x=lumin_ratio_2d_arr[:, 0],
                                          y=lumin_ratio_2d_arr[:, 1],
                                          kind="linear", fill_value="extrapolate")

    else:
        raise ValueError("lumin_ratio_2d_arr must be a 2D numpy array with two columns or None.")

    return func


def wf_ccl(z_grid, probe, which_wf, flat_fid_pars_dict, cosmo_ccl, dndz_tuple, ia_bias_tuple=None, gal_bias_tuple=None,
           mag_bias_tuple=None, has_rsd=False, return_ccl_obj=False, n_samples=1000):
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

    a_arr = cosmo_lib.z_to_a(z_grid)
    comoving_distance = ccl.comoving_radial_distance(cosmo_ccl, a_arr)

    if probe == 'lensing':

        # build intrinsic alignment bias array
        if ia_bias_tuple is None:
            ia_bias_1d = build_ia_bias_1d_arr(z_grid_out=z_grid, cosmo_ccl=cosmo_ccl,
                                              flat_fid_pars_dict=flat_fid_pars_dict,
                                              lumin_ratio_2d_arr=lumin_ratio_2d_arr,
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

        wf_galaxy_obj = []
        for zbin_idx in range(zbins):

            # this is needed to be eble to pass mag_bias = None for each zbin
            if mag_bias_tuple is None:
                mag_bias_arg = mag_bias_tuple
            else:
                mag_bias_arg = (mag_bias_tuple[0], mag_bias_tuple[1][:, zbin_idx])

            wf_galaxy_obj.append(ccl.tracers.NumberCountsTracer(cosmo_ccl,
                                                                has_rsd=has_rsd,
                                                                dndz=(dndz_tuple[0], dndz_tuple[1][:, zbin_idx]),
                                                                bias=(gal_bias_tuple[0],
                                                                      gal_bias_tuple[1][:, zbin_idx]),
                                                                mag_bias=mag_bias_arg,
                                                                n_samples=n_samples))

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

    a_arr = cosmo_lib.z_to_a(z_grid)
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





def cl_PyCCL(wf_A, wf_B, ell, zbins, p_of_k_a, cosmo, cl_ccl_kwargs: dict):

    is_auto_spectrum = False
    if wf_A == wf_B:
        is_auto_spectrum = True

    nbl = len(ell)

    if p_of_k_a is None:
        p_of_k_a = 'delta_matter:delta_matter'

    if is_auto_spectrum:
        cl_3D = np.zeros((nbl, zbins, zbins))
        for zi, zj in zip(np.triu_indices(zbins)[0], np.triu_indices(zbins)[1]):
            cl_3D[:, zi, zj] = ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=p_of_k_a,
                                              **cl_ccl_kwargs)
        for ell in range(nbl):
            cl_3D[ell, :, :] = sl.symmetrize_2d_array(cl_3D[ell, :, :])

    elif not is_auto_spectrum:
        # be very careful with the order of the zi, zj loops: you have to revert them in NESTED list comprehensions to
        # have zi as first axis and zj as second axis (the code below is tested and works)
        cl_3D = np.array([[ccl.angular_cl(cosmo, wf_A[zi], wf_B[zj], ell, p_of_k_a=p_of_k_a,
                                          **cl_ccl_kwargs)
                           for zj in range(zbins)]
                          for zi in range(zbins)]
                         ).transpose(2, 0, 1)  # transpose to have ell as first axis
    else:
        raise ValueError('is_auto_spectrum must be either True or False')

    return cl_3D


def stem(cl_4d, variations_arr, zbins, nbl, percent_tolerance=1):

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
                perc_diffs = sl.percent_diff(cl_4d_cpy[:, ell, zi, zj], fitted_y_values)

                # as long as any element has a percent deviation greater than 1%, remove first and last values
                while np.any(np.abs(perc_diffs) > percent_tolerance):
                    # if the condition is satisfied, remove the first and last values
                    cl_4d_cpy = np.delete(cl_4d_cpy, [0, -1], axis=0)
                    variations_arr_cpy = np.delete(variations_arr_cpy, [0, -1])

                    # re-compute the fit on the reduced set
                    angular_coefficient, intercept = np.polyfit(variations_arr_cpy, cl_4d_cpy[:, ell, zi, zj], deg=1)
                    fitted_y_values = angular_coefficient * variations_arr_cpy + intercept

                    # test again
                    perc_diffs = sl.percent_diff(cl_4d_cpy[:, ell, zi, zj], fitted_y_values)

                    # breakpoint()
                    # plt.figure()
                    # plt.plot(variations_arr_cpy, fitted_y_values, '--', lw=2)
                    # plt.plot(variations_arr_cpy, cl_4d_cpy[:, ell, zi, zj], marker='o')
                    # plt.xlabel('$\\theta$')

                # store the value of the derivative
                dcl_3d[ell, zi, zj] = angular_coefficient

    return dcl_3d


# ! start new parallel
def cls_and_derivatives_parallel_old(fiducial_values_dict, extra_parameters, list_params_to_vary, zbins, dndz, ell_LL,
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
            delayed(cl_parallel_helper_old)(param_to_vary, variation_idx, varied_fiducials, fiducial_values_dict,
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


def cl_parallel_helper_old(param_to_vary, variation_idx, varied_fiducials, fiducial_values_dict, extra_parameters,
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

        varied_fiducials['Om_nu0'] = cosmo_lib.get_omega_nu0(varied_fiducials['m_nu'], varied_fiducials['h'],
                                                             n_ur=None, n_eff=varied_fiducials['N_eff'],
                                                             n_ncdm=None,
                                                             neutrino_mass_fac=None, g_factor=None)

        cosmo_ccl = cosmo_lib.instantiate_cosmo_ccl_obj(varied_fiducials, extra_parameters)
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


def compute_cls_derivatives(cfg, list_params_to_vary, zbins, nz_tuple,
                            ell_LL, ell_GL, ell_GG, use_only_flat_models=True):
    """
    Compute the derivatives of the power spectrum with respect to the free parameters
    """
    # TODO cleanup the function, + make it single-probe
    # TODO implement checks on the input parameters
    # TODO input dndz galaxy and IA bias

    nbl_WL = len(ell_LL)
    nbl_XC = len(ell_GL)
    nbl_GC = len(ell_GG)

    # old
    # percentages = np.asarray((-10., -5., -3.75, -2.5, -1.875, -1.25, -0.625, 0,
    #                           0.625, 1.25, 1.875, 2.5, 3.75, 5., 10.)) / 100

    # new
    percentages = np.asarray((-10, -5, -3.75, -3, -2.5, -1.875, -1.25, -1.0, -0.625, 0.0,
                              0.625, 1.0, 1.25, 1.875, 2.5, 3.0, 3.75, 5.0, 10.0)) / 100
    num_points_derivative = len(percentages)

    free_fid_pars_dict = deepcopy(cfg['cosmology']['FM_ordered_params'])

    # declare cl and dcl vectors
    cl_LL, cl_GL, cl_GG = {}, {}, {}
    dcl_LL, dcl_GL, dcl_GG = {}, {}, {}

    if use_only_flat_models:
        assert cfg['cosmology']['other_params']['Om_k0'] == 0, 'if use_only_flat_models is True, Om_k0 must be 0'
        assert 'Om_k0' not in list_params_to_vary, 'if use_only_flat_models is True, ' \
            'Om_k0 must not be in list_params_to_vary'

    # loop over the free parameters and store the cls in a dictionary
    for name_par_tovary in list_params_to_vary:

        assert name_par_tovary in free_fid_pars_dict.keys(), f'{name_par_tovary} is not in the fiducial values dict'

        t0 = time.perf_counter()
        print(f'Cl derivatives: working on {name_par_tovary}...')

        # shift the parameter
        varied_param_values = free_fid_pars_dict[name_par_tovary] + free_fid_pars_dict[name_par_tovary] * percentages
        if free_fid_pars_dict[name_par_tovary] == 0:  # wa is 0! take directly the percentages
            varied_param_values = percentages

        # ricorda che, quando shifti OmegaM va messo OmegaCDM in modo che OmegaB + OmegaCDM dia il valore corretto di OmegaM,
        # mentre quando shifti OmegaB deve essere aggiustato sempre OmegaCDM in modo che OmegaB + OmegaCDM = 0.32; per OmegaX
        # lo shift ti darà un OmegaM + OmegaDE diverso da 1 il che corrisponde appunto ad avere modelli non piatti

        # this dictionary will contain the shifted values of the parameters; it is initialized with the fiducial values
        # for each new parameter to be varied
        # ! important note: the fiducial and varied_fiducial dict contain all cosmological parameters, not just the ones
        # ! that are varied (specified in list_params_to_vary). This is to be able to change the set of
        # ! varied parameters easily
        varied_fid_pars_dict = deepcopy(free_fid_pars_dict)

        # instantiate derivatives array for the given free parameter key
        cl_LL[name_par_tovary] = np.zeros((num_points_derivative, nbl_WL, zbins, zbins))
        cl_GL[name_par_tovary] = np.zeros((num_points_derivative, nbl_XC, zbins, zbins))
        cl_GG[name_par_tovary] = np.zeros((num_points_derivative, nbl_GC, zbins, zbins))

        # ! benchmark
        results = [cl_derivatives_helper(name_par_tovary=name_par_tovary,
                                         varied_fid_pars_dict=varied_fid_pars_dict,
                                         cl_LL=cl_LL,
                                         cl_GL=cl_GL,
                                         cl_GG=cl_GG,
                                         cfg=cfg,
                                         nz_tuple=nz_tuple,
                                         list_params_to_vary=list_params_to_vary,
                                         zbins=zbins,
                                         ell_LL=ell_LL,
                                         ell_GL=ell_GL,
                                         ell_GG=ell_GG,
                                         use_only_flat_models=use_only_flat_models) for
                   varied_fid_pars_dict[name_par_tovary] in tqdm(varied_param_values)]

        # Collect the results
        for variation_idx, (cl_LL_part, cl_GL_part, cl_GG_part) in enumerate(results):
            cl_LL[name_par_tovary][variation_idx, :, :, :] = cl_LL_part
            cl_GL[name_par_tovary][variation_idx, :, :, :] = cl_GL_part
            cl_GG[name_par_tovary][variation_idx, :, :, :] = cl_GG_part

        print(f'param {name_par_tovary} Cls computed in {(time.perf_counter() - t0):.2f} seconds')

        try:
            dcl_LL[name_par_tovary] = stem(cl_LL[name_par_tovary], varied_param_values, zbins, nbl_WL, 1)
            dcl_GL[name_par_tovary] = stem(cl_GL[name_par_tovary], varied_param_values, zbins, nbl_GC, 1)
            dcl_GG[name_par_tovary] = stem(cl_GG[name_par_tovary], varied_param_values, zbins, nbl_GC, 1)
        except np.linalg.LinAlgError as e:
            print(e)
            print('SteM derivative computation failed, increasing tolerance from 1% to 6%')
            dcl_LL[name_par_tovary] = stem(cl_LL[name_par_tovary], varied_param_values, zbins, nbl_WL, 6)
            dcl_GL[name_par_tovary] = stem(cl_GL[name_par_tovary], varied_param_values, zbins, nbl_GC, 6)
            dcl_GG[name_par_tovary] = stem(cl_GG[name_par_tovary], varied_param_values, zbins, nbl_GC, 6)

        print(f'SteM derivative computed for {name_par_tovary}')

    return cl_LL, cl_GL, cl_GG, dcl_LL, dcl_GL, dcl_GG


def cl_derivatives_helper(name_par_tovary, varied_fid_pars_dict, cl_LL, cl_GL, cl_GG,
                          cfg, nz_tuple, list_params_to_vary,
                          zbins, ell_LL, ell_GL, ell_GG,
                          use_only_flat_models=True):

    general_cfg = cfg['general_cfg']
    magcut_lens = general_cfg['magcut_lens']
    fid_pars_dict = cfg['cosmology']
    z_grid_nz, n_of_z = nz_tuple
    names_cosmo_pars = ['Om', 'Ob', 'h', 'ns', 'logT', 'ODE', 's8', 'wz', 'wa']

    # ! v1
    # if use_only_flat_models:
    #     # in this case I want omk = 0, so if ODE varies Om will have to be adjusted and vice versa
    #     # (and Om is adjusted by adjusting Omega_CDM), see the else statement

    #     assert 'Om_k0' not in list_params_to_vary, 'if use_only_flat_models is True, Om_k0 must not be in list_params_to_vary'

    #     # If I vary ODE and Om_k0 = 0, I need to adjust Om
    #     if name_par_tovary == 'ODE':
    #         varied_fid_pars_dict['Om'] = 1 - varied_fid_pars_dict['ODE']

    # else:
    #     # If I vary ODE or Om and Om_k0 can vary, I need to adjust Om_k0
    #     varied_fid_pars_dict['Om_k0'] = 1 - varied_fid_pars_dict['Om'] - varied_fid_pars_dict['ODE']
    #     if np.abs(varied_fid_pars_dict['Om_k0']) < 1e-8:
    #         varied_fid_pars_dict['Om_k0'] = 0

    # ! v2
    # from CLOE_pk_for_SPV3
    if use_only_flat_models:
        # in this case I want omk = 0, so if Omega_DE varies Omega_M will have to be adjusted and vice versa
        # (and Omega_M is adjusted by adjusting Omega_CDM), see the else statement

        assert 'Om_k0' not in list_params_to_vary, 'if use_only_flat_models is True, Om_k0 must not be in list_params_to_vary'
        omk = 0
        if name_par_tovary == 'Om':
            varied_fid_pars_dict['ODE'] = 1 - varied_fid_pars_dict['Om']
        elif name_par_tovary == 'ODE':
            varied_fid_pars_dict['Om'] = 1 - varied_fid_pars_dict['ODE']

    else:

        omk = 1 - varied_fid_pars_dict['Om'] - varied_fid_pars_dict['ODE']
        if np.abs(omk) < 1e-8:
            omk = 0

        # if name_par_tovary == 'Om':
        #     Omega_CDM = varied_fid_pars_dict['Om'] - varied_fid_pars_dict['Omega_B'] - Omega_nu

    # other CAMB quantities - call them with CAMB-friendly names already
    # Omega_CDM = varied_fid_pars_dict['Om'] - varied_fid_pars_dict['Ob'] - Omega_nu
    # omch2 = Omega_CDM * varied_fid_pars_dict['h'] ** 2
    # ombh2 = varied_fid_pars_dict['Omega_B'] * varied_fid_pars_dict['h'] ** 2
    # H0 = varied_fid_pars_dict['h'] * 100

    # TODO does this change when I change h?
    # if 'm_nu' in list_params_to_vary:
    # m_nu = varied_fid_pars_dict['m_nu'] if 'm_nu' in list_params_to_vary else fid_pars_dict['other_params']['m_nu']
    # N_eff = varied_fid_pars_dict['N_eff'] if 'N_eff' in list_params_to_vary else fid_pars_dict['other_params']['N_eff']
    # varied_fid_pars_dict['Om_nu0'] = cosmo_lib.get_omega_nu0(m_nu=m_nu, h=varied_fid_pars_dict['h'],
    #                                                       n_eff=N_eff)

    # check that the other parameters are still equal to the fiducials
    for cosmo_par in varied_fid_pars_dict.keys():
        if cosmo_par != name_par_tovary:
            assert fid_pars_dict['FM_ordered_params'][cosmo_par] == varied_fid_pars_dict[
                cosmo_par], f'{cosmo_par} is not the same as in the fiducial model'

    dzWL_shifts = [varied_fid_pars_dict[f'dzWL{zi:02d}'] for zi in range(1, zbins + 1)]
    # dzGC_shifts = [varied_fid_pars_dict[f'dzGC{zi:02d}'] for zi in range(1, zbins + 1)]

    gal_bias_polyfit_values = [varied_fid_pars_dict[f'bG{zi:02d}'] for zi in range(1, 5)]
    mag_bias_polyfit_values = [varied_fid_pars_dict[f'bM{zi:02d}'] for zi in range(1, 5)]
    mult_shear_bias_values = [varied_fid_pars_dict[f'm{zi:02d}'] for zi in range(1, zbins + 1)]

    # instantiate cosmology object. camb_extra_parameters are not varied, so they can be passed from the fid_pars_dict

    other_params = deepcopy(fid_pars_dict['other_params'])
    other_params['camb_extra_parameters']['camb']['HMCode_logT_AGN'] = varied_fid_pars_dict['logT']
    full_pars_dict_for_ccl = {**varied_fid_pars_dict, 'other_params': other_params}
    ccl_obj = pyccl_interface.PycclClass(full_pars_dict_for_ccl)
    ccl_obj.zbins = zbins

    if cfg['covariance_cfg']['PyCCL_cfg']['which_pk_for_pyccl'] == 'CLOE':

        if name_par_tovary in dav_to_vinc_par_names:
            name_par_tovary_vinc = dav_to_vinc_par_names[name_par_tovary]
        else:
            name_par_tovary_vinc = name_par_tovary

        val_par_tovary = varied_fid_pars_dict[name_par_tovary]

        cloe_pk_folder = cfg['general_cfg']['CLOE_pk_folder'].format(
            SPV3_folder=cfg['general_cfg']['SPV3_folder'],
            which_pk=cfg['general_cfg']['which_pk'],
            flat_or_nonflat=cfg['general_cfg']['flat_or_nonflat'])

        if name_par_tovary in names_cosmo_pars:

            cloe_pk_filename = cfg['general_cfg']['CLOE_pk_filename'].format(
                CLOE_pk_folder=cloe_pk_folder,
                param_name=name_par_tovary_vinc,
                param_value=val_par_tovary)

        else:
            cloe_pk_filename = cfg['general_cfg']['CLOE_pk_filename'].format(
                CLOE_pk_folder=cloe_pk_folder,
                param_name='w0',
                param_value=-1)

        # ccl_obj.cosmo_ccl.p_of_k_a = ccl_obj.pk_obj_from_file(pk_filename=cloe_pk_filename, plot_pk_z0=False)
        pk = ccl_obj.pk_obj_from_file(pk_filename=cloe_pk_filename, plot_pk_z0=False)

    elif cfg['covariance_cfg']['PyCCL_cfg']['which_pk_for_pyccl'] == 'PyCCL':
        pk = None
    else:
        raise ValueError('which_pk_for_pyccl must be either "CLOE" or "PyCCL"')

    # quick check
    assert (varied_fid_pars_dict['Om'] / ccl_obj.cosmo_ccl.cosmo.params.Omega_m - 1) < 1e-7, \
        'Om_m0 is not the same as the one in the fiducial model'

    n_of_z = shift_nz(z_grid_nz, n_of_z, dzWL_shifts, normalize=False, plot_nz=False, interpolation_kind='linear')

    ccl_obj.set_nz(np.hstack((z_grid_nz[:, None], n_of_z)))
    ccl_obj.set_ia_bias_tuple(z_grid=z_grid_nz)

    # set galaxy bias
    if general_cfg['which_forecast'] == 'SPV3':
        ccl_obj.set_gal_bias_tuple_spv3(z_grid=z_grid_nz,
                                        magcut_lens=magcut_lens,
                                        poly_fit_values=gal_bias_polyfit_values)

    elif general_cfg['which_forecast'] == 'ISTF':
        bias_func_str = general_cfg['bias_function']
        bias_model = general_cfg['bias_model']
        ccl_obj.set_gal_bias_tuple_istf(z_grid=z_grid_nz,
                                        bias_function_str=bias_func_str,
                                        bias_model=bias_model)

    ccl_obj.set_mag_bias_tuple(z_grid=z_grid_nz,
                               has_magnification_bias=general_cfg['has_magnification_bias'],
                               magcut_lens=magcut_lens,
                               poly_fit_values=mag_bias_polyfit_values)

    ccl_obj.set_kernel_obj(general_cfg['has_rsd'], n_samples_wf=256)

    # TODO set pk importing the appropriate file, more cumbersome, for the time being use the cosmo obj
    cl_LL = ccl_obj.compute_cls(ell_LL, pk, ccl_obj.wf_lensing_obj, ccl_obj.wf_lensing_obj, 'spline')
    cl_GL = ccl_obj.compute_cls(ell_GL, pk, ccl_obj.wf_galaxy_obj, ccl_obj.wf_lensing_obj, 'spline')
    cl_GG = ccl_obj.compute_cls(ell_GG, pk, ccl_obj.wf_galaxy_obj, ccl_obj.wf_galaxy_obj, 'spline')

    for ell_idx, _ in enumerate(ell_LL):
        for zi in range(zbins):
            for zj in range(zbins):
                cl_LL[ell_idx, zi, zj] *= (1 + mult_shear_bias_values[zi]) * (1 + mult_shear_bias_values[zj])

    for ell_idx, _ in enumerate(ell_GL):
        for zi in range(zbins):
            for zj in range(zbins):
                cl_GL[ell_idx, zi, zj] *= (1 + mult_shear_bias_values[zj])

    if varied_fid_pars_dict[name_par_tovary] == fid_pars_dict['FM_ordered_params'][name_par_tovary]:
        print('saving fiducial spectra for comparison')
        np.save(f'/home/davide/Scrivania/test_ders/cl_LL_{name_par_tovary}.npy', cl_LL)
        np.save(f'/home/davide/Scrivania/test_ders/cl_GL_{name_par_tovary}.npy', cl_GL)
        np.save(f'/home/davide/Scrivania/test_ders/cl_GG_{name_par_tovary}.npy', cl_GG)

    return cl_LL, cl_GL, cl_GG


def gaussian_smmothing_nz(zgrid_nz, nz_original, nz_gaussian_smoothing_sigma, plot=True):

    print(f'Applying a Gaussian filter of sigma = {nz_gaussian_smoothing_sigma} to the n(z)')

    zbins = nz_original.shape[1]
    colors = cm.rainbow(np.linspace(0, 1, zbins))

    nz_smooth = gaussian_filter1d(nz_original, nz_gaussian_smoothing_sigma, axis=0)

    if plot:
        plt.figure()
        for zi in range(zbins):
            plt.plot(zgrid_nz, nz_smooth[:, zi], label=f'zbin {zi}', c=colors[zi], ls='-')
            plt.plot(zgrid_nz, nz_smooth[:, zi], c=colors[zi], ls='--')
        plt.title(f'Gaussian filter w/ sigma = {nz_gaussian_smoothing_sigma}')

    return nz_smooth


def shift_nz(zgrid_nz, nz_original, dz_shifts, normalize, plot_nz=False, interpolation_kind='linear', clip_min=0,
             clip_max=3, bounds_error=False, fill_value=0):
    print(f'Shifting n(z), clipping between redshifts {clip_min} and {clip_max}')

    zbins = nz_original.shape[1]
    assert len(dz_shifts) == zbins, 'dz_shifts must have the same length as the number of zbins'
    assert np.all(np.abs(dz_shifts) < 0.1), 'dz_shifts must be small (this is a rough check)'
    assert nz_original.shape[0] == len(zgrid_nz), 'nz_original must have the same length as zgrid_nz'

    colors = cm.rainbow(np.linspace(0, 1, zbins))

    n_of_z_shifted = np.zeros_like(nz_original)
    for zi in range(zbins):
        # not-very-pythonic implementation: create an interpolator for each bin
        n_of_z_func = interp1d(zgrid_nz, nz_original[:, zi], kind=interpolation_kind,
                               bounds_error=bounds_error, fill_value=fill_value)
        z_grid_nz_shifted = zgrid_nz - dz_shifts[zi]
        # where < 0, set to 0; where > 3, set to 3
        z_grid_nz_shifted = np.clip(z_grid_nz_shifted, clip_min, clip_max)
        n_of_z_shifted[:, zi] = n_of_z_func(z_grid_nz_shifted)

    if normalize:
        integrals = simps(y=n_of_z_shifted, x=zgrid_nz, axis=0)
        n_of_z_shifted /= integrals[None, :]

    if plot_nz:
        plt.figure()
        for zi in range(zbins):
            plt.plot(zgrid_nz, nz_original[:, zi], ls='-', c=colors[zi])
            plt.plot(zgrid_nz, n_of_z_shifted[:, zi], ls='--', c=colors[zi])

        legend_elements = [mlines.Line2D([], [], color='k', linestyle='-', label='Original'),
                           mlines.Line2D([], [], color='k', linestyle='--', label='Shifted')]
        plt.legend(handles=legend_elements)
        plt.xlabel('$z$')
        plt.ylabel('$n_i(z)$')

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
