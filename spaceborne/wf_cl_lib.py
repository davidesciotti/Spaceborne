import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import scipy
from matplotlib import cm
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d

from spaceborne import constants
from spaceborne import sb_lib as sl

c = constants.SPEED_OF_LIGHT

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
        ax[0].plot(zgrid_nz_src, nz_src[:, zi], c=colors[zi], label=f'$z_{zi + 1}$')
        # ax[0].axvline(zbin_centers_src[zi], c=colors[zi], ls='--', alpha=0.6, label=r'$z_{%d}^{eff}$' % (zi + 1))
        ax[0].fill_between(zgrid_nz_src, nz_src[:, zi], color=colors[zi], alpha=0.2)
        ax[0].set_xlabel('$z$')
        ax[0].set_ylabel(r'$n_i(z) \; {\rm sources}$')
    ax[0].legend(ncol=2)

    for zi in range(zbins):
        ax[1].plot(zgrid_nz_lns, nz_lns[:, zi], c=colors[zi], label=f'$z_{zi + 1}$')
        # ax[1].axvline(zbin_centers_lns[zi], c=colors[zi], ls='--', alpha=0.6, label=r'$z_{%d}^{eff}$' % (zi + 1))
        ax[1].fill_between(zgrid_nz_lns, nz_lns[:, zi], color=colors[zi], alpha=0.2)
        ax[1].set_xlabel('$z$')
        ax[1].set_ylabel(r'$n_i(z) \; {\rm lenses}$')
    ax[1].legend(ncol=2)


# @njit
def pph(z_p, z, c_in, z_in, sigma_in, c_out, z_out, sigma_out, f_out):
    first_addendum = (
        (1 - f_out)
        / (np.sqrt(2 * np.pi) * sigma_in * (1 + z))
        * np.exp(-0.5 * ((z - c_in * z_p - z_in) / (sigma_in * (1 + z))) ** 2)
    )
    second_addendum = (
        f_out
        / (np.sqrt(2 * np.pi) * sigma_out * (1 + z))
        * np.exp(-0.5 * ((z - c_out * z_p - z_out) / (sigma_out * (1 + z))) ** 2)
    )
    return first_addendum + second_addendum


# @njit
def n_of_z(z, z_0, n_gal):
    return n_gal * (z / z_0) ** 2 * np.exp(-((z / z_0) ** (3 / 2)))
    # return  (z / z_0) ** 2 * np.exp(-(z / z_0) ** (3 / 2))


# @njit
def f_ia(z, eta_IA, beta_IA, z_pivot_IA, lumin_ratio_func):
    result = ((1 + z) / (1 + z_pivot_IA)) ** eta_IA * (lumin_ratio_func(z)) ** beta_IA
    return result


def b_of_z_analytical(z):
    """Simple analytical prescription for the linear galaxy bias:
    b(z) = sqrt(1 + z)
    """
    return np.sqrt(1 + z)


def b_of_z_fs1_leporifit(z):
    """Fit to the linear galaxy bias measured from FS1"""
    return 0.5125 + 1.377 * z + 0.222 * z**2 - 0.249 * z**3


def b_of_z_fs1_pocinofit(z):
    """Fit to the linear galaxy bias measured from FS1."""
    a, b, c = 0.81, 2.80, 1.02
    return a * z**b / (1 + z) + c


def b_of_z_fs2_fit(z, magcut_lens, poly_fit_values=None):
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit

    if poly_fit_values is not None:
        assert len(poly_fit_values) == 4, 'a list of 4 best-fit values must be passed'
        b0_gal, b1_gal, b2_gal, b3_gal = poly_fit_values

    elif magcut_lens == 24.5:
        b0_gal, b1_gal, b2_gal, b3_gal = 1.33291, -0.72414, 1.0183, -0.14913
    elif magcut_lens == 23:
        b0_gal, b1_gal, b2_gal, b3_gal = 1.88571, -2.73925, 3.24688, -0.73496
    else:
        raise ValueError(
            'magcut_lens, i.e. the limiting magnitude of the GCph sample, must be 23 or 24.5'
        )

    return b0_gal + (b1_gal * z) + (b2_gal * z**2) + (b3_gal * z**3)


def magbias_of_z_fs2_fit(z, magcut_lens, poly_fit_values=None):
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit
    if poly_fit_values is not None:
        assert len(poly_fit_values) == 4, 'a list of 4 best-fit values must be passed'
        b0_mag, b1_mag, b2_mag, b3_mag = poly_fit_values

    elif magcut_lens == 24.5:
        b0_mag, b1_mag, b2_mag, b3_mag = -1.50685, 1.35034, 0.08321, 0.04279
    elif magcut_lens == 23:
        b0_mag, b1_mag, b2_mag, b3_mag = -2.34493, 3.73098, 0.12500, -0.01788
    else:
        raise ValueError(
            'magcut_lens, i.e. the limiting magnitude of the GCph sample, must be 23 or 24.5'
        )

    return b0_mag + (b1_mag * z) + (b2_mag * z**2) + (b3_mag * z**3)


def s_of_z_fs2_fit(z, magcut_lens, poly_fit_values=None):
    """Wrapper function to output the magnification bias as needed in ccl; function written by Marco"""
    # from the MCMC for SPV3 google doc: https://docs.google.com/document/d/1WCGhiBrlTsvl1VS-2ngpjirMnAS-ahtnoGX_7h8JoQU/edit
    return (
        magbias_of_z_fs2_fit(z, magcut_lens, poly_fit_values=poly_fit_values) + 2
    ) / 5


def b2g_fs2_fit(z):
    """This function has been fitted by Sylvain G. Beauchamps based on FS2 measurements:
    z_meas = [0.395, 0.7849999999999999, 1.1749999999999998, 1.565, 1.9549999999999998, 2.3449999999999998]
    b2_meas = [-0.25209754,  0.14240271,  0.56409318,  1.06597924,  2.84258843,  4.8300518 ]
    """
    c0, c1, c2, c3 = -0.69682803, 1.60320679, -1.31676159, 0.70271383
    b2g_ofz = c0 + c1 * z + c2 * z**2 + c3 * z**3
    return b2g_ofz


def stepwise_bias(z, gal_bias_vs_zmean, z_edges):
    """Returns the bias value for a given redshift, based on stepwise bias values per redshift bin.

    Parameters
    ----------
    z (float): The redshift value.
    gal_bias_vs_zmean (list or array): Array containing one bias value per redshift bin.
    z_minus (list or array): Array containing the lower edge of each redshift bin.
    z_plus (list or array): Array containing the upper edge of each redshift bin.

    Returns
    -------
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


def build_galaxy_bias_2d_arr(
    gal_bias_vs_zmean,
    zmeans,
    z_edges,
    zbins,
    z_grid,
    bias_model,
    plot_bias=False,
    bias_fit_function=None,
    kwargs_bias_fit_function=None,
):
    """Builds a 2d array of shape (len(z_grid), zbins) containing the bias values for each redshift bin. The bias values
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
        assert len(gal_bias_vs_zmean) == zbins, (
            'gal_bias_vs_zmean must be an array of length zbins'
        )
        assert len(zmeans) == zbins, 'zmeans must be an array of length zbins'

    if bias_model == 'unbiased':
        gal_bias_2d_arr = np.ones((len(z_grid), zbins))

    elif bias_model == 'linint':
        # linear interpolation
        galaxy_bias_func = scipy.interpolate.interp1d(
            zmeans,
            gal_bias_vs_zmean,
            kind='linear',
            fill_value=(gal_bias_vs_zmean[0], gal_bias_vs_zmean[-1]),
            bounds_error=False,
        )
        gal_bias_1d_arr = galaxy_bias_func(z_grid)
        gal_bias_2d_arr = np.repeat(gal_bias_1d_arr[:, np.newaxis], zbins, axis=1)

    elif bias_model == 'constant':
        # this is the *only* case in which the bias is different for each zbin
        gal_bias_2d_arr = np.repeat(
            gal_bias_vs_zmean[np.newaxis, :], len(z_grid), axis=0
        )

    elif bias_model == 'step-wise':
        assert z_edges is not None, 'z_edges must be provided for step-wise bias'
        assert len(z_edges) == zbins + 1, 'z_edges must have length zbins + 1'
        gal_bias_1d_arr = np.array(
            [stepwise_bias(z, gal_bias_vs_zmean, z_edges) for z in z_grid]
        )
        gal_bias_2d_arr = np.repeat(gal_bias_1d_arr[:, np.newaxis], zbins, axis=1)

    elif bias_model == 'polynomial':
        assert bias_fit_function is not None, (
            'bias_fit_function must be provided for polynomial bias'
        )
        gal_bias_1d_arr = bias_fit_function(z_grid, **kwargs_bias_fit_function)
        # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
        gal_bias_2d_arr = np.repeat(gal_bias_1d_arr.reshape(1, -1), zbins, axis=0).T

    else:
        raise ValueError(
            'bias_model must be "unbiased", "linint", "constant", "step-wise" or "polynomial"'
        )

    if plot_bias:
        plt.figure()
        plt.title(f'bias_model {bias_model}')
        for zbin_idx in range(zbins):
            plt.plot(z_grid, gal_bias_2d_arr[:, zbin_idx], label=f'zbin {zbin_idx + 1}')
            plt.scatter(
                zmeans[zbin_idx], gal_bias_vs_zmean[zbin_idx], marker='o', color='black'
            )
        plt.legend()
        plt.xlabel('$z$')
        plt.ylabel('$b_i(z)$')
        plt.show()

    assert gal_bias_2d_arr.shape == (len(z_grid), zbins), (
        'gal_bias_2d_arr must have shape (len(z_grid), zbins)'
    )

    return gal_bias_2d_arr


def build_ia_bias_1d_arr(
    z_grid_out, cosmo_ccl, ia_dict, lumin_ratio_2d_arr, output_F_IA_of_z=False
):
    """Computes the intrinsic alignment (IA) bias as a function of redshift.

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

    assert len(growth_factor) == len(z_grid_out), (
        'growth_factor must have the same length '
        'as z_grid (it must be computed in these '
        'redshifts!)'
    )

    omega_m = cosmo_ccl.cosmo.params.Omega_m
    F_IA_of_z = f_ia(z_grid_out, eta_IA, beta_IA, z_pivot_IA, lumin_ratio_func)
    ia_bias = -1 * A_IA * C_IA * omega_m * F_IA_of_z / growth_factor

    return (ia_bias, F_IA_of_z) if output_F_IA_of_z else ia_bias


def get_luminosity_ratio_interpolator(lumin_ratio_2d_arr):
    """Returns an interpolator function for the luminosity ratio or a default constant function.
    :param lumin_ratio_2d_arr: A 2D numpy array with shape (N, 2) where column 0 is z and column 1 is the ratio.
    :return: Interpolator function for luminosity ratio.
    """
    if lumin_ratio_2d_arr is None:

        def func(z):
            return 1  # Default to constant luminosity ratio

    elif (
        isinstance(lumin_ratio_2d_arr, np.ndarray) and lumin_ratio_2d_arr.shape[1] == 2
    ):
        func = scipy.interpolate.interp1d(
            x=lumin_ratio_2d_arr[:, 0],
            y=lumin_ratio_2d_arr[:, 1],
            kind='linear',
            fill_value='extrapolate',
        )

    else:
        raise ValueError(
            'lumin_ratio_2d_arr must be a 2D numpy array with two columns or None.'
        )

    return func


def cl_ccl(wf_a, wf_b, ells, zbins, p_of_k_a, cosmo, cl_ccl_kwargs: dict):
    is_auto_spectrum = wf_a == wf_b
    nbl = len(ells)

    if p_of_k_a is None:
        p_of_k_a = 'delta_matter:delta_matter'

    if is_auto_spectrum:
        cl_3d = np.zeros((nbl, zbins, zbins))
        for zi, zj in zip(np.triu_indices(zbins)[0], np.triu_indices(zbins)[1]):
            cl_3d[:, zi, zj] = ccl.angular_cl(
                cosmo, wf_a[zi], wf_b[zj], ells, p_of_k_a=p_of_k_a, **cl_ccl_kwargs
            )
        for ell_ix in range(nbl):
            cl_3d[ell_ix, :, :] = sl.symmetrize_2d_array(cl_3d[ell_ix, :, :])

    else:
        # be very careful with the order of the zi, zj loops: you have to revert them in NESTED list comprehensions to
        # have zi as first axis and zj as second axis (the code below is tested and works)
        cl_3d = np.array(
            [
                [
                    ccl.angular_cl(
                        cosmo,
                        wf_a[zi],
                        wf_b[zj],
                        ells,
                        p_of_k_a=p_of_k_a,
                        **cl_ccl_kwargs,
                    )
                    for zj in range(zbins)
                ]
                for zi in range(zbins)
            ]
        ).transpose(2, 0, 1)  # transpose to have ell as first axis
    return cl_3d


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
                angular_coefficient, intercept = np.polyfit(
                    variations_arr_cpy, cl_4d_cpy[:, ell, zi, zj], deg=1
                )
                fitted_y_values = angular_coefficient * variations_arr_cpy + intercept

                # check % difference
                perc_diffs = sl.percent_diff(cl_4d_cpy[:, ell, zi, zj], fitted_y_values)

                # as long as any element has a percent deviation greater than 1%, remove first and last values
                while np.any(np.abs(perc_diffs) > percent_tolerance):
                    # if the condition is satisfied, remove the first and last values
                    cl_4d_cpy = np.delete(cl_4d_cpy, [0, -1], axis=0)
                    variations_arr_cpy = np.delete(variations_arr_cpy, [0, -1])

                    # re-compute the fit on the reduced set
                    angular_coefficient, intercept = np.polyfit(
                        variations_arr_cpy, cl_4d_cpy[:, ell, zi, zj], deg=1
                    )
                    fitted_y_values = (
                        angular_coefficient * variations_arr_cpy + intercept
                    )

                    # test again
                    perc_diffs = sl.percent_diff(
                        cl_4d_cpy[:, ell, zi, zj], fitted_y_values
                    )

                    # breakpoint()
                    # plt.figure()
                    # plt.plot(variations_arr_cpy, fitted_y_values, '--', lw=2)
                    # plt.plot(variations_arr_cpy, cl_4d_cpy[:, ell, zi, zj], marker='o')
                    # plt.xlabel('$\\theta$')

                # store the value of the derivative
                dcl_3d[ell, zi, zj] = angular_coefficient

    return dcl_3d


def shift_nz(
    zgrid_nz,
    nz_original,
    dz_shifts,
    normalize,
    plot_nz=False,
    interpolation_kind='linear',
    clip_min=0,
    clip_max=3,
    bounds_error=False,
    fill_value=0,
    plt_title='',
):
    print(f'Shifting n(z), clipping between redshifts {clip_min} and {clip_max}')

    zbins = nz_original.shape[1]
    assert len(dz_shifts) == zbins, (
        'dz_shifts must have the same length as the number of zbins'
    )
    assert np.all(np.abs(dz_shifts) < 0.1), (
        'dz_shifts must be small (this is a rough check)'
    )
    assert nz_original.shape[0] == len(zgrid_nz), (
        'nz_original must have the same length as zgrid_nz'
    )

    colors = cm.rainbow(np.linspace(0, 1, zbins))

    n_of_z_shifted = np.zeros_like(nz_original)
    for zi in range(zbins):
        # not-very-pythonic implementation: create an interpolator for each bin
        n_of_z_func = interp1d(
            zgrid_nz,
            nz_original[:, zi],
            kind=interpolation_kind,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )
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

        legend_elements = [
            mlines.Line2D([], [], color='k', linestyle='-', label='Original'),
            mlines.Line2D([], [], color='k', linestyle='--', label='Shifted'),
        ]
        plt.legend(handles=legend_elements)
        plt.xlabel('$z$')
        plt.ylabel('$n_i(z)$')
        plt.title(plt_title)

    return n_of_z_shifted


def get_z_means(zgrid, kernel):
    """Compute the mean of the wf distribution"""
    assert kernel.shape[0] == zgrid.shape[0], (
        'kernel and zgrid must have the same length'
    )
    assert kernel.ndim == 2, 'kernel must be a 2d array'
    z_means = simps(y=kernel * zgrid[:, None], x=zgrid, axis=0) / simps(
        y=kernel, x=zgrid, axis=0
    )
    return z_means


def get_z_effective_isaac(zgrid_nz, n_of_z):
    """Calculate the effective redshift at which to evaluate the bias.

    The effective redshift is defined as the median of the redshift distribution
    considering only the part of the distribution that is at least 10% of its maximum.

    Parameters
    ----------
    z (array-like): Array of redshifts corresponding to the n(z) distribution.
    n_of_z (array-like): The n(z) redshift distribution.

    Returns
    -------
    float: The effective redshift.

    """
    zbins = n_of_z.shape[1]
    effective_z = np.zeros(zbins)

    for zi in range(zbins):
        n_of_zi = n_of_z[:, zi]
        threshold = max(n_of_zi) * 0.1
        effective_z[zi] = np.median(zgrid_nz[n_of_zi > threshold])

    return effective_z
