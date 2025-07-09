import contextlib
import inspect
import itertools
import json
import os
import pickle
import subprocess
import time
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from scipy.integrate import simpson as simps
from scipy.interpolate import CubicSpline, RectBivariateSpline, interp1d
from scipy.special import jv

symmetrize_output_dict = {
    ('L', 'L'): True,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): True,
}


# # Gaussian covariance binning
# def bin_cov_gauss(cov, ell_values, theta_edges, fsky):
#     binned_cov = np.zeros((len(theta_edges) - 1, len(theta_edges) - 1))
#     for i in range(len(theta_edges) - 1):
#         for j in range(len(theta_edges) - 1):
#             ell_min_1, ell_max_1 = theta_edges[i], theta_edges[i + 1]
#             ell_min_2, ell_max_2 = theta_edges[j], theta_edges[j + 1]
#             overlapping_ells = np.intersect1d(
#                 np.arange(ell_min_1, ell_max_1),
#                 np.arange(ell_min_2, ell_max_2)
#             )
#             if len(overlapping_ells) == 0:
#                 continue
#             spline = UnivariateSpline(ell_values, np.diag(cov), k=1, s=0)
#             binned_cov[i, j] = 1 / fsky / np.sum((2 * overlapping_ells + 1) / spline(overlapping_ells))
#     return binned_cov

# # Non-Gaussian covariance binning
# def bin_cov_nongauss(cov, ell_values, theta_edges, area):
#     binned_cov = np.zeros((len(theta_edges) - 1, len(theta_edges) - 1))
#     spline = RegularGridInterpolator((ell_values, ell_values), np.log(cov), bounds_error=False, fill_value=None)
#     for i in range(len(theta_edges) - 1):
#         ell_min_1, ell_max_1 = theta_edges[i], theta_edges[i + 1]
#         area1 = np.pi * (ell_max_1**2 - ell_min_1**2)
#         ell_grid_1 = np.geomspace(ell_min_1, ell_max_1, 10)
#         for j in range(len(theta_edges) - 1):
#             ell_min_2, ell_max_2 = theta_edges[j], theta_edges[j + 1]
#             area2 = np.pi * (ell_max_2**2 - ell_min_2**2)
#             ell_grid_2 = np.geomspace(ell_min_2, ell_max_2, 10)
#             ell1, ell2 = np.meshgrid(ell_grid_1, ell_grid_2, indexing='ij')
#             result = simps(simps(np.exp(spline((ell1, ell2)) * ell1 * ell2, ell_grid_2), ell_grid_1))
#             result /= (area1 * area2)
#             result *= 4 * np.pi**2 / area
#             binned_cov[i, j] = result
#     return binned_cov


def build_probe_list(probes, include_cross_terms=False):
    """Return the list of probe combinations to compute.

    Parameters
    ----------
    probes : list[str]
        List of individual probes to include, e.g. ['LL', 'GL', 'GG'].
    include_cross_terms : bool
        If True, include cross-combinations between different probes.

    Returns
    -------
    list[str]
        List of probe combinations, e.g. ['LLLL', 'LLGL', ...]

    """
    if not include_cross_terms:
        return [p + p for p in probes]

    # Sort to ensure consistent ordering
    # probes = sorted(probes)
    return [p1 + p2 for p1, p2 in itertools.combinations_with_replacement(probes, 2)]


def is_main_branch():
    """Check if the current Git branch is 'main'"""
    try:
        # Get the current branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        )
        current_branch = result.stdout.strip()

        # Return True if branch is 'main', False otherwise
        return current_branch == 'main'
    except subprocess.SubprocessError:
        # If there's an error (e.g., not in a git repo), default to False
        print('Warning: Could not determine Git branch. Assuming not main branch.')
        return False


@contextlib.contextmanager
def timer(msg):
    start = time.perf_counter()
    try:
        yield
    finally:
        stop = time.perf_counter()
        print(msg % (stop - start), flush=True)


def bin_2d_array(  # fmt: skip
    cov, ells_in, ells_out, ells_out_edges, weights_in, which_binning='sum',
    interpolate = True
):  # fmt: skip
    assert cov.shape[0] == cov.shape[1] == len(ells_in), (
        'ells_in must be the same length as the covariance matrix'
    )
    assert len(ells_out) == len(ells_out_edges) - 1, (
        'ells_out must be the same length as the number of edges - 1'
    )
    assert which_binning in ['sum', 'integral'], (
        'which_binning must be either "sum" or "integral"'
    )

    binned_cov = np.zeros((len(ells_out), len(ells_out)))
    cov_interp_func = RectBivariateSpline(ells_in, ells_in, cov)

    ells_edges_low = ells_out_edges[:-1]
    ells_edges_high = ells_out_edges[1:]

    if weights_in is None:
        weights_in = np.ones_like(ells_in)

    assert len(weights_in) == len(ells_in), (
        'weights_in must be the same length as ells_in'
    )

    assert type(interpolate) is bool, 'interpolate must be a boolean'

    # Loop over the output bins
    for ell1_idx, _ in enumerate(ells_out):
        for ell2_idx, _ in enumerate(ells_out):
            # Get ell min/max for the current bins
            ell1_min = ells_edges_low[ell1_idx]
            ell1_max = ells_edges_high[ell1_idx]
            ell2_min = ells_edges_low[ell2_idx]
            ell2_max = ells_edges_high[ell2_idx]

            # isolate the relevant ranges of ell values from the original ells_in grid

            ell1_in_ix = np.where((ell1_min <= ells_in) & (ells_in < ell1_max))[0]
            ell2_in_ix = np.where((ell2_min <= ells_in) & (ells_in < ell2_max))[0]
            ell1_in = ells_in[ell1_in_ix]
            ell2_in = ells_in[ell2_in_ix]

            # mask the covariance to the relevant block
            cov_masked = cov[np.ix_(ell1_in_ix, ell2_in_ix)]

            # this equals the number of ell values within a bin in the unweighted case,
            # and delta_ell in the unweighted, unbinned case
            weights1_in = weights_in[ell1_in_ix]
            weights2_in = weights_in[ell2_in_ix]

            weights1_in_xx, weights2_in_yy = np.meshgrid(
                weights1_in, weights2_in, indexing='ij'
            )
            if which_binning == 'sum':
                partial_sum = np.sum(
                    cov_masked * weights1_in_xx * weights2_in_yy, axis=1
                )
                total = np.sum(partial_sum, axis=0)
                norm1 = np.sum(weights1_in)
                norm2 = np.sum(weights2_in)

            elif which_binning == 'integral' and not interpolate:
                partial_integral = simps(
                    y=cov_masked * weights1_in_xx * weights2_in_yy, x=ell2_in, axis=1
                )
                total = simps(y=partial_integral, x=ell1_in, axis=0)
                norm1 = simps(y=weights1_in, x=ell1_in)
                norm2 = simps(y=weights2_in, x=ell2_in)

            elif which_binning == 'integral' and interpolate:
                # Interpolate the covariance matrix to a finer grid if necessary
                ell1_fine = np.linspace(ell1_min, ell1_max, num=100)
                ell2_fine = np.linspace(ell2_min, ell2_max, num=100)
                cov_interp = cov_interp_func(ell1_fine, ell2_fine)

                # Create fine grids for weights if necessary
                weights1_fine = np.interp(ell1_fine, ell1_in, weights1_in)
                weights2_fine = np.interp(ell2_fine, ell2_in, weights2_in)

                # Perform the double integral
                partial_integral = simps(
                    y=cov_interp * weights1_fine[:, None] * weights2_fine[None, :],
                    x=ell2_fine,
                    axis=1,
                )
                total = simps(y=partial_integral, x=ell1_fine, axis=0)
                norm1 = simps(y=weights1_fine, x=ell1_fine)
                norm2 = simps(y=weights2_fine, x=ell2_fine)

            binned_cov[ell1_idx, ell2_idx] = total / (norm1 * norm2)

    return binned_cov


def bin_1d_array(
    ells_in, ells_out, ells_out_edges, cls_in, weights, which_binning, ells_eff=None
):
    """Bin the input power spectrum into the output bins.

    :param ells_in: array of input ells
    :param ells_out: array of output ells
    :param cls_in: array of input power spectrum
    :param weights: array of weights for the input power spectrum
    :return: array of binned power spectrum
    """
    weights_was_none = False
    if weights is None:
        weights = np.ones_like(ells_in)
        weights_was_none = True
    if len(ells_in) != len(cls_in):
        raise ValueError('ells_in and cls_in must have the same length')
    if len(ells_in) != len(weights):
        raise ValueError('ells_in and weights must have the same length')
    if np.any(ells_out < ells_in[0]) or np.any(ells_out > ells_in[-1]):
        raise ValueError('ells_out must be within the range of ells_in')
    if np.any(ells_out[1:] < ells_out[:-1]):
        raise ValueError('ells_out must be monotonically increasing')

    assert len(cls_in) == len(ells_in), (
        'ells_in must be the same length as the covariance matrix'
    )
    assert len(ells_out) == len(ells_out_edges) - 1, (
        'ells_out must be the same length as the number of edges - 1'
    )

    binned_cls = np.zeros(len(ells_out))
    _spline = CubicSpline(ells_in, cls_in)

    ells_edges_low = ells_out_edges[:-1]
    ells_edges_high = ells_out_edges[1:]

    # Loop over the output bins
    for ell_idx in range(len(ells_out)):
        # Get ell min/max for the current bins
        ell_min = ells_edges_low[ell_idx]
        ell_max = ells_edges_high[ell_idx]

        # this mask returns a bool array True at the ells_in indices satisfying the condition
        ell_bool_mask = (ell_min <= ells_in) & (ells_in < ell_max)
        ell_masked_idxs = np.nonzero(ell_bool_mask)[0]

        # isolate the relevant ranges of ell values from the original ells_in grid, weights and cov
        ells_in_masked = ells_in[ell_masked_idxs]
        cls_masked = cls_in[ell_masked_idxs]

        if weights.shape == (len(ells_eff), len(ells_in)):
            ells_eff_idx = np.argmin(np.abs(ells_eff - ells_out[ell_idx]))
            weights_masked = weights[ells_eff_idx, ell_masked_idxs]
        elif weights.shape == (len(ells_in),):
            weights_masked = weights[ell_masked_idxs]

        # Calculate the bin widths
        if weights_was_none:
            delta_ell = ell_max - ell_min
            assert delta_ell == np.sum(weights_masked), (
                'The weights must sum to the bin width'
            )

        # Option 1: use the original grid for integration and no weights
        if which_binning == 'integral':
            integral = simps(y=cls_masked * weights_masked, x=ells_in_masked)
            binned_cls[ell_idx] = integral / np.sum(weights_masked)

        elif which_binning == 'sum':
            binned_cls[ell_idx] = np.sum(cls_masked * weights_masked) / np.sum(
                weights_masked
            )

        else:
            raise ValueError('which_binning should be "sum" or "integral"')

        # # Option 2: create fine grids for integration over the ell ranges (GIVES GOOD RESULTS ONLY FOR nsteps=delta_ell!)
        # ell_fine = np.linspace(ell_min, ell_max, 50)
        # cls_interp = spline(ell_fine)

        # # Perform simps integration over the ell ranges
        # integral = simps(y=cls_interp * ell_fine, x=ell_fine)
        # binned_cls[ell_idx] = integral / (np.sum(ell_fine))

    return binned_cls


def matshow_vcenter(matrix, vcenter=0):
    """Plots a matrix with a 0-centered, asymmetric colorbar."""
    from matplotlib.colors import TwoSlopeNorm

    plt.matshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=vcenter))
    plt.colorbar()
    plt.show()


def j0(x):
    return jv(0, x)


def j1(x):
    return jv(1, x)


def j2(x):
    return jv(2, x)


def savetxt_aligned(filename, array_2d, header_list, col_width=25, decimals=8):
    header = ''
    for i in range(len(header_list)):
        offset = 2 if i == 0 else 0
        string = f'{header_list[i]:<{col_width - offset}}'
        header += string

    # header = ''.join(
    # [f'{header_list[i]:<{col_width - 2}}' for i in range(len(header_list))]
    # )
    fmt = [f'%-{col_width}.{decimals}f'] * len(array_2d[0])
    np.savetxt(filename, array_2d, header=header, fmt=fmt, delimiter='')


def nz_fits_to_txt(fits_filename):
    """Converts the official SGS-like fits file to the usual (z, nz) format.

    Parameters
    ----------
    fits_filename : str
        The full path and filename of the fits file to be converted.

    """
    import euclidlib as el

    z, nz = el.photo.redshift_distributions(fits_filename)

    nz_arr = np.zeros((len(z), len(nz) + 1))
    nz_arr[:, 0] = z

    plt.figure()
    for bin in nz:
        plt.plot(z, nz[bin])
        nz_arr[:, bin] = nz[bin]

    return nz_arr


def compare_funcs(
    x,
    y: dict,
    logscale_y=(False, False),
    logscale_x=False,
    title=None,
    ylim_diff=None,
    plt_kw=None,
):
    plt_kw = {} if plt_kw is None else plt_kw

    names = list(y.keys())
    y_tuple = list(y.values())
    colors = plt.get_cmap('tab10').colors  # Get tab colors

    if x is None:
        x = np.arange(len(y_tuple[0]))

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[2, 1],
    )
    fig.subplots_adjust(hspace=0)

    for i, _y in enumerate(y_tuple):
        ls = '--' if i > 0 else '-'
        # alpha = 0.8 if i > 0 else 1
        ax[0].plot(x, _y, label=names[i], c=colors[i], ls=ls, **plt_kw)
    ax[0].legend()

    for i in range(1, len(y_tuple)):
        ax[1].plot(
            x, percent_diff(y_tuple[i], y_tuple[0]), c=colors[i], ls='-', **plt_kw
        )
    ax[1].set_ylabel('A/B - 1 [%]')
    ax[1].axhspan(-10, 10, alpha=0.2, color='gray')

    for i in range(2):
        if logscale_y[i]:
            ax[i].set_yscale('log')

    if logscale_x:
        for i in range(2):
            ax[i].set_xscale('log')

    if ylim_diff is not None:
        ax[1].set_ylim(ylim_diff)

    if title is not None:
        fig.suptitle(title)


def get_git_info():
    try:
        branch = (
            subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode('utf-8')
        )

        commit = (
            subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode('utf-8')
        )

        return branch, commit
    except subprocess.CalledProcessError:
        return None, None


def mirror_upper_to_lower_vectorized(A):
    # Check if A is square
    if A.shape[0] != A.shape[1]:
        raise ValueError('Input must be a square matrix')

    # Create a copy of the original matrix
    result = A.copy()

    # Use numpy's triu_indices to get the indices of the upper triangle
    triu_indices = np.triu_indices_from(A, k=1)

    # Mirror the upper triangular elements to the lower triangle
    result[(triu_indices[1], triu_indices[0])] = A[triu_indices]

    return result


def check_interpolate_input_tab(
    input_tab: np.ndarray, z_grid_out: np.ndarray, zbins: int
) -> tuple:
    """Interpolates the input table over the 0th dimension using a cubic spline
    and returns the interpolated values on the specified grid.

    Parameters
    ----------
    - input_tab (numpy.ndarray): The input table with shape (z_points, zbins + 1).
    - z_grid_out (numpy.ndarray): The output grid for interpolation.
    - zbins (int): The number of redshift bins.

    Returns
    -------
    - output_tab (numpy.ndarray): The interpolated table with shape
    (len(z_grid_out), zbins).

    """
    assert input_tab.shape[1] == zbins + 1, (
        'The input table should have shape (z_points, zbins + 1)'
        f', but has shape {input_tab.shape} and zbins = {zbins}'
    )

    # Perform cubic spline interpolation
    spline = CubicSpline(x=input_tab[:, 0], y=input_tab[:, 1:], axis=0)
    output_tab = spline(z_grid_out)

    return output_tab, spline


def interp_2d_arr(x_in, y_in, z2d_in, x_out, y_out, output_masks):
    """Interpolate a 2D array onto a new grid using bicubic spline
    interpolation.

    Parameters
    ----------
    - x_in (numpy.ndarray): The x-coordinates of the input 2D array.
    - y_in (numpy.ndarray): The y-coordinates of the input 2D array.
    - z2d_in (numpy.ndarray): The 2D input array to be interpolated.
    - x_out (numpy.ndarray): The x-coordinates of the output grid.
    - y_out (numpy.ndarray): The y-coordinates of the output grid.
    - output_masks (bool): A boolean flag indicating whether to mask the output array.

    Returns
    -------
    - x_out_masked (numpy.ndarray): The x-coordinates of the output grid,
    clipped to avoid interpolation errors.
    - y_out_masked (numpy.ndarray): The y-coordinates of the output grid,
    clipped to avoid interpolation errors.
    - z2d_interp (numpy.ndarray): The interpolated 2D array.
    - x_mask (numpy.ndarray): A boolean mask indicating which elements of the
    original x_out array were used.
    - y_mask (numpy.ndarray): A boolean mask indicating which elements of the
    original y_out array were used.

    """
    z2d_func = RectBivariateSpline(x=x_in, y=y_in, z=z2d_in)

    # clip x and y grids to avoid interpolation errors
    x_mask = np.logical_and(x_in.min() <= x_out, x_out < x_in.max())
    y_mask = np.logical_and(y_in.min() <= y_out, y_out < y_in.max())
    x_out_masked = x_out[x_mask]
    y_out_masked = y_out[y_mask]

    if len(x_out_masked) < len(x_out):
        print(
            f'x array trimmed: old range [{x_out.min():.2e}, {x_out.max():.2e}], '
            f'new range [{x_out_masked.min():.2e}, {x_out_masked.max():.2e}]'
        )
    if len(y_out_masked) < len(y_out):
        print(
            f'y array trimmed: old range [{y_out.min():.2e}, {y_out.max():.2e}], '
            f'new range [{y_out_masked.min():.2e}, {y_out_masked.max():.2e}]'
        )

    # with RegularGridInterpolator:
    # TODO untested
    # z2d_func = RegularGridInterpolator((x_in, y_in), z2d_in, method='linear')
    # xx, yy = np.meshgrid(x_out_masked, y_out_masked)
    # z2d_interp = z2d_interp((xx, yy)).T

    z2d_interp = z2d_func(x_out_masked, y_out_masked)

    if output_masks:
        return x_out_masked, y_out_masked, z2d_interp, x_mask, y_mask
    else:
        return x_out_masked, y_out_masked, z2d_interp


def regularize_covariance(cov_matrix, lambda_reg=1e-5):
    """Regularizes the covariance matrix by adding lambda * I.

    Parameters
    ----------
    - cov_matrix: Original covariance matrix (numpy.ndarray)
    - lambda_reg: Regularization parameter

    Returns
    -------
    - Regularized covariance matrix

    """
    n = cov_matrix.shape[0]
    identity_matrix = np.eye(n)
    cov_matrix_reg = cov_matrix + lambda_reg * identity_matrix
    return cov_matrix_reg


def get_simpson_weights(n):
    """Function written by Marco Bonici."""
    number_intervals = (n - 1) // 2
    weight_array = np.zeros(n)
    if n == number_intervals * 2 + 1:
        for i in range(number_intervals):
            weight_array[2 * i] += 1 / 3
            weight_array[2 * i + 1] += 4 / 3
            weight_array[2 * i + 2] += 1 / 3
    else:
        weight_array[0] += 0.5
        weight_array[1] += 0.5
        for i in range(number_intervals):
            weight_array[2 * i + 1] += 1 / 3
            weight_array[2 * i + 2] += 4 / 3
            weight_array[2 * i + 3] += 1 / 3
        weight_array[-1] += 0.5
        weight_array[-2] += 0.5
        for i in range(number_intervals):
            weight_array[2 * i] += 1 / 3
            weight_array[2 * i + 1] += 4 / 3
            weight_array[2 * i + 2] += 1 / 3
        weight_array /= 2
    return weight_array


def zpair_from_zidx(zidx, ind):
    """Return the zpair corresponding to the zidx for a given ind array.

    To be thoroughly tested, but quite straightforward
    """
    assert ind.shape[1] == 2, (
        'ind array must have shape (n, 2), maybe you are passing the full '
        'ind file instead of ind_auto/ind_cross'
    )
    return np.where((ind == [zidx, zidx]).all(axis=1))[0][0]


def plot_dominant_array_element(
    arrays_dict, tab_colors, elements_auto, elements_cross, elements_3x2pt
):
    """Plot 2D arrays from a dictionary, highlighting the dominant component in
    each element.

    Colors are assigned based on the array with the dominant component
    at each position. If no component is dominant (all are zero), the
    color will be white.
    """
    centers = [
        elements_auto // 2,
        elements_auto + elements_cross // 2,
        elements_auto + elements_cross + elements_auto // 2,
    ]
    labels = ['WL', 'GGL', 'GCph']

    # Stack arrays along a new dimension and calculate the absolute values
    stacked_abs_arrays = np.abs(np.stack(list(arrays_dict.values()), axis=-1))

    # Find indices of the dominant array at each position
    dominant_indices = np.argmax(stacked_abs_arrays, axis=-1)

    # Add an extra category for non-dominant cases (where all arrays are zero)
    non_dominant_value = (
        -1
    )  # Choose a value that doesn't conflict with existing indices
    dominant_indices[np.all(stacked_abs_arrays == 0, axis=-1)] = non_dominant_value

    # Prepare the colormap, including an extra color for non-dominant cases
    selected_colors = ['white'] + tab_colors[
        : len(arrays_dict)
    ]  # 'white' is for non-dominant cases
    cmap = ListedColormap(selected_colors)

    # Plot the dominant indices with the custom colormap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        dominant_indices, cmap=cmap, vmin=non_dominant_value, vmax=len(arrays_dict) - 1
    )

    # Create a colorbar with labels
    # Set the ticks so they are at the center of each color segment
    cbar_ticks = np.linspace(
        non_dominant_value, len(arrays_dict) - 1, len(selected_colors)
    )
    cbar_labels = ['0'] + list(
        arrays_dict.keys()
    )  # 'None' corresponds to the non-dominant case
    cbar = plt.colorbar(im, ticks=cbar_ticks)
    cbar.set_ticklabels(cbar_labels)

    lw = 2
    plt.axvline(elements_auto, c='k', lw=lw)
    plt.axvline(elements_auto + elements_cross, c='k', lw=lw)
    plt.axhline(elements_auto, c='k', lw=lw)
    plt.axhline(elements_auto + elements_cross, c='k', lw=lw)
    plt.xticks([])
    plt.yticks([])

    for idx, label in enumerate(labels):
        x = centers[idx]
        plt.text(x, -1.5, label, va='bottom', ha='center')
        plt.text(-1.5, x, label, va='center', ha='right', rotation='vertical')

    plt.show()


def cov_3x2pt_dict_8d_to_10d(
    cov_3x2pt_dict_8D,
    nbl,
    zbins,
    ind_dict,
    probe_ordering,
    symmetrize_output_dict: bool = symmetrize_output_dict,
):
    cov_3x2pt_dict_10D = {}
    for probe_A, probe_B in probe_ordering:
        for probe_C, probe_D in probe_ordering:
            cov_3x2pt_dict_10D[probe_A, probe_B, probe_C, probe_D] = (
                cov_4D_to_6D_blocks(
                    cov_3x2pt_dict_8D[probe_A, probe_B, probe_C, probe_D],
                    nbl,
                    zbins,
                    ind_dict[probe_A, probe_B],
                    ind_dict[probe_C, probe_D],
                    symmetrize_output_dict[probe_A, probe_B],
                    symmetrize_output_dict[probe_C, probe_D],
                )
            )
    return cov_3x2pt_dict_10D


def write_cl_ascii(ascii_folder, ascii_filename, cl_3d, ells, zbins):
    with open(f'{ascii_folder}/{ascii_filename}.ascii', 'w') as file:
        # Write header
        file.write(f'#ell\ttomo_i\ttomo_j\t{ascii_filename}\n')

        # Iterate over the array and write the data
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    # Format the line with appropriate spacing
                    file.write(f'{ell_val:.3f}\t{zi + 1}\t{zj + 1}\t{value:.10e}\n')


def write_cl_tab(folder, filename, cl_3d, ells, zbins):
    """Write the Cls in the SB txt format."""
    with open(f'{folder}/{filename}.txt', 'w') as file:
        file.write(f'#ell\t\tzi\tzj\t{filename}\n')
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    file.write(f'{ell_val:.3f}\t\t{zi}\t{zj}\t{value:.10e}\n')


def compare_fm_constraints(
    *fm_dict_list,
    labels,
    keys_toplot_in,
    normalize_by_gauss,
    which_uncertainty,
    reference,
    colors,
    abs_FoM,
    nparams_toplot_in=8,
    save_fig=False,
    fig_path=None,
):
    masked_fm_dict_list = []
    masked_fid_pars_dict_list = []
    uncertainties_dict = {}
    fom_dict = {}
    legend_x_anchor = 1.4

    assert keys_toplot_in == 'all' or type(keys_toplot_in) is list, (
        'keys_toplot must be a list or "all"'
    )
    assert colors is None or type(colors) is list, 'colors must be a list or "all"'

    colors = (
        plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors
    )

    # maks fm and fid pars dict
    for fm_dict in fm_dict_list:
        masked_fm_dict, masked_fid_pars_dict = {}, {}

        # define keys and remove unused ones
        keys_toplot = list(fm_dict.keys())
        if 'fiducial_values_dict' in keys_toplot:
            keys_toplot.remove('fiducial_values_dict')
        keys_toplot = [key for key in keys_toplot if not key.startswith('FM_WA_')]
        keys_toplot = [key for key in keys_toplot if not key.startswith('FM_2x2pt_')]

        for key in keys_toplot:
            masked_fm_dict[key], masked_fid_pars_dict[key] = mask_fisher(
                fm_dict[key],
                fm_dict['fiducial_values_dict'],
                names_params_to_fix=[],
                remove_null_rows_cols=True,
            )
        masked_fm_dict_list.append(masked_fm_dict)
        masked_fid_pars_dict_list.append(masked_fid_pars_dict)

    # compute reference uncertainties
    print(key, masked_fid_pars_dict_list[0].keys())
    for key in keys_toplot:
        nparams_toplot = nparams_toplot_in
        param_names = list(masked_fid_pars_dict_list[0][key].keys())[:nparams_toplot]
        uncertainties_dict[key] = np.array(
            [
                uncertainties_fm_v2(
                    masked_fm_dict[key],
                    fiducials_dict=masked_fid_pars_dict[key],
                    which_uncertainty=which_uncertainty,
                    normalize=True,
                )[:nparams_toplot]
                for masked_fm_dict, masked_fid_pars_dict in zip(
                    masked_fm_dict_list, masked_fid_pars_dict_list
                )
            ]
        )
        w0wa_idxs = (param_names.index('wz'), param_names.index('wa'))
        fom_dict[key] = np.array(
            [
                compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)
                for masked_fm_dict in masked_fm_dict_list
            ]
        )
        uncertainties_dict[key] = np.column_stack(
            (uncertainties_dict[key], fom_dict[key])
        )
    param_names.append('FoM')

    keys_toplot = keys_toplot if keys_toplot_in == 'all' else keys_toplot_in

    # plot, and if necessary normalize by the G-only uncertainty
    for key in keys_toplot:
        probe = key.split('_')[1]

        ylabel = 'rel. unc. [%]'
        if normalize_by_gauss and not key.endswith('_G'):
            probe = key.split('_')[1]
            ng_cov = key.split('_')[2]
            uncertainties_dict[key] = (
                uncertainties_dict[key] / uncertainties_dict[f'FM_{probe}_G'] - 1
            ) * 100
            ylabel = f'{ng_cov}/G - 1 [%]'

            if abs_FoM:
                uncertainties_dict[key][:, -1] = np.fabs(uncertainties_dict[key][:, -1])

        n_rows = 2 if len(fm_dict_list) > 1 else 1
        fig, ax = plt.subplots(n_rows, 1, figsize=(10, 5), sharex=True)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        ax[0].set_title(f'{which_uncertainty} uncertainties, {key}')
        for i, uncert in enumerate(uncertainties_dict[key]):
            ax[0].scatter(
                param_names,
                uncert,
                label=f'{labels[i]}',
                marker='o',
                c=colors[i],
                alpha=0.6,
            )
        ax[0].axhline(0, c='k', ls='--')
        ax[0].set_ylabel(ylabel)
        ax[0].legend(ncol=1, loc='center right', bbox_to_anchor=(legend_x_anchor, 0.0))
        ax[0].grid()

        start_idx = 0
        title_str = reference
        if reference == 'first_key':
            ref = uncertainties_dict[key][0]
            start_idx = 1
            title_str = labels[0]
        elif reference == 'median':
            ref = np.median(uncertainties_dict[key], axis=0)
        elif reference == 'mean':
            ref = np.mean(uncertainties_dict[key], axis=0)
        else:
            raise ValueError(
                'reference must be one of "first_key", "median", or "mean"'
            )

        if len(uncertainties_dict[key]) > 1:
            diffs = [
                percent_diff(uncert, ref)
                for uncert in uncertainties_dict[key][start_idx:]
            ]

            for i, diff in enumerate(diffs):
                ax[1].scatter(
                    param_names, diff, marker='o', c=colors[i + start_idx], alpha=0.6
                )
            ax[1].fill_between(
                (0, nparams_toplot), -10, 10, color='k', alpha=0.1, label='$\\pm 10\\%$'
            )

        ax[1].set_ylabel(f'% diff wrt\n{title_str}\n')
        ax[1].legend(ncol=1, loc='center right', bbox_to_anchor=(legend_x_anchor, 0.5))
        ax[1].grid()

        if save_fig:
            plt.savefig(f'{fig_path}/{key}.png', dpi=400, bbox_inches='tight')


def block_diag(array_3d):
    """Useful for visualizing nbl, zbins, zbins arrays at a glance."""
    nbl = array_3d.shape[0]
    return scipy.linalg.block_diag(*[array_3d[ell, :, :] for ell in range(nbl)])


def compare_df_keys(dataframe, key_to_compare, value_a, value_b, num_string_columns):
    """This function compares two rows of a dataframe and returns a new row
    with the percentage difference between the two :param dataframe:

    :param key_to_compare:
    :param value_a:
    :param value_b:
    :param num_string_columns: number of columns containing only strings or
    various options, such as whether to fix a certain prior or not...
    :return:
    """
    import pandas as pd

    df_A = dataframe[dataframe[key_to_compare] == value_a]
    df_B = dataframe[dataframe[key_to_compare] == value_b]
    arr_A = df_A.iloc[:, num_string_columns:].select_dtypes('number').values
    arr_B = df_B.iloc[:, num_string_columns:].select_dtypes('number').values

    if arr_A.shape[0] != arr_B.shape[0]:
        raise ValueError(
            'Cannot compare groups with different sizes: '
            f'{arr_A.shape[0]} vs {arr_B.shape[0]}'
        )

    perc_diff_df = df_A.copy()
    # ! the reference is G, this might change to G + SSC + cNG
    perc_diff_df.iloc[:, num_string_columns:] = percent_diff(arr_B, arr_A)
    perc_diff_df[key_to_compare] = f'perc_diff_{value_b}'
    perc_diff_df['FoM'] = -perc_diff_df['FoM']  # ! abs? minus??
    dataframe = pd.concat([dataframe, perc_diff_df], axis=0, ignore_index=True)

    # dataframe = dataframe.drop_duplicates()
    columns_to_consider = [
        col for col in dataframe.columns if col not in ['fm', 'fiducials_dict']
    ]
    dataframe = dataframe.drop_duplicates(subset=columns_to_consider)

    return dataframe


def contour_FoM_calculator(sample, param1, param2, sigma_level=1):
    """This function has been written by Santiago Casas.

    Computes  the FoM from getDist samples. add()sample is a getDist
    sample object, you need as well the shapely package to compute
    polygons. The function returns the 1sigma FoM, but in principle you
    could compute 2-, or 3-sigma "FoMs"
    """
    from shapely.geometry import Polygon

    contour_coords = {}
    density = sample.get2DDensityGridData(j=param1, j2=param2, num_plot_contours=3)
    contour_levels = density.contours
    contours = plt.contour(density.x, density.y, density.P, sorted(contour_levels))
    for ii, contour in enumerate(contours.collections):
        paths = contour.get_paths()
        for path in paths:
            xy = path.vertices
            x = xy[:, 0]
            y = xy[:, 1]
            contour_coords[ii] = list(zip(x, y))
    sigma_lvls = {3: 0, 2: 1, 1: 2}
    poly = Polygon(
        contour_coords[sigma_lvls[sigma_level]]
    )  # 0:3sigma, 1:2sigma, 2:1sigma
    area = poly.area
    FoM_area = (2.3 * np.pi) / area
    return FoM_area, density


def can_be_pickled(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False


def load_cov_from_probe_blocks(path, filename, probe_ordering):
    """Load the covariance matrix from the probe blocks in 4D. The blocks are
    stored in a dictionary with keys corresponding to the probes in the order
    specified in probe_ordering. The symmetrization of the blocks is done while
    loading, so only 6 blocks need to be actually stored. :param path: Path to
    the folder containing the covariance blocks. :param filename: Filename of
    the covariance blocks. The filename should contain the placeholders
    {probe_A}, {probe_B}, {probe_C}, {probe_D} which will be replaced with the
    actual probe names. :param probe_ordering: Probe ordering tuple :return:

    YOU SHOULD USE deepcopy, otherwise the different blocks become
    correlated (and you e.g. divide twice by fsky)
    """
    cov_ssc_dict_8D = {}
    for row, (probe_a, probe_b) in enumerate(probe_ordering):
        for col, (probe_c, probe_d) in enumerate(probe_ordering):
            if col >= row:  # Upper triangle and diagonal
                formatted_filename = filename.format(
                    probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d
                )
                cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = np.load(
                    f'{path}/{formatted_filename}'
                )

                if formatted_filename.endswith('.npz'):
                    cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                        cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d]['arr_0']
                    )

            else:  # Lower triangle, set using symmetry
                cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = deepcopy(
                    cov_ssc_dict_8D[probe_c, probe_d, probe_a, probe_b].transpose(
                        1, 0, 3, 2
                    )
                )

    for key, cov in cov_ssc_dict_8D.items():
        assert cov.ndim == 4, (
            f'covariance matrix {key} has ndim={cov.ndim} instead of 4'
        )

    return cov_ssc_dict_8D


def save_dict_to_file(dict_data, file_path, indent=4):
    """Save a dictionary to a text file in a nicely formatted way.

    Parameters
    ----------
    dict_data (dict): The dictionary to be saved.
    file_path (str): The path of the file where the dictionary will be saved.
    indent (int, optional): The number of spaces for indentation. Default is 4.

    """
    with open(file_path, 'w') as f:
        json.dump(dict_data, f, indent=indent)


def figure_of_correlation(correl_matrix):
    """Compute the Figure of Correlation (FoC) from the correlation matrix
    correl_matrix.

    Parameters
    ----------
    - correl_matrix (2D numpy array): The correlation matrix.

    Returns
    -------
    - FoC (float): The Figure of Correlation.

    """
    # Invert the correlation matrix
    correl_matrix_inv = np.linalg.inv(correl_matrix)
    # Compute the FoC
    foc = np.sqrt(np.linalg.det(correl_matrix_inv))

    return foc


def plot_correlation_matrix(correlation_matrix):
    plt.matshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar()


def find_inverse_from_array(input_x, input_y, desired_y, interpolation_kind='linear'):
    from pynverse import inversefunc

    input_y_func = interp1d(input_x, input_y, kind=interpolation_kind)
    desired_y = inversefunc(
        input_y_func, y_values=desired_y, domain=(input_x[0], input_x[-1])
    )
    return desired_y


def add_ls_legend(ls_dict):
    """Add a legend for line styles.

    Parameters
    ----------
    ls_dict : dict
        A dictionary mapping line styles to labels.
        E.g. {'-': 'delta', '--': 'gamma'}

    """
    handles = []
    for ls, label in ls_dict.items():
        handles.append(mlines.Line2D([], [], color='black', linestyle=ls, label=label))
    plt.legend(handles=handles, loc='best')


def save_correlation_matrix_plot(matrix1, matrix2, labels):
    """Yet to be fully tested.

    Plot the positive and negative values with different colorbars TODO
    make it only for one matrix!
    """
    _fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 5, width_ratios=[6, 0.3, 6, 0.3, 0.3])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])
    _cbar_ax_pos = plt.subplot(gs[3])
    _cbar_ax_neg = plt.subplot(gs[4])

    # Define the colormap
    cmap_pos = plt.get_cmap('Reds')
    cmap_neg = plt.get_cmap('Blues_r')

    # Divide positive and negative parts for each matrix
    matrix1_pos = np.ma.masked_less_equal(matrix1, 0)
    matrix1_neg = -np.ma.masked_greater_equal(matrix1, 0)

    _matrix2_pos = np.ma.masked_less_equal(matrix2, 0)
    _matrix2_neg = -np.ma.masked_greater_equal(matrix2, 0)

    # Plot Gaussian matrix
    ax1.imshow(matrix1_pos, cmap=cmap_pos, norm=LogNorm())
    ax1.imshow(matrix1_neg, cmap=cmap_neg, norm=LogNorm())

    ax1.set_title('Gaussian')
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)

    # Plot Gaussian + SSC matrix
    # im2a = ax2.imshow(matrix2_pos, cmap=cmap_pos, norm=LogNorm())
    # im2b = ax2.imshow(matrix2_neg, cmap=cmap_neg, norm=LogNorm())

    ax2.set_title('Gaussian + SSC')
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)

    # Add colorbars
    # cbar_pos = fig.colorbar(im2a, cax=cbar_ax_pos, orientation='vertical')
    # cbar_neg = fig.colorbar(im2b, cax=cbar_ax_neg, orientation='vertical')

    plt.tight_layout()
    plt.show()


def table_to_3d_array(file_path, is_auto_spectrum):
    """Convert CLOE .dat files format to 3d arrays, following the indexing in
    the header.
    """
    # Read the header and data from the file
    with open(file_path) as f:
        header = f.readline().strip().split('\t')

    cl_2d = np.genfromtxt(file_path)
    ells = cl_2d[:, 0]
    cl_2d = cl_2d[:, 1:]
    nbl = len(ells)

    ind = zpairs_header_to_ind_array(header[1:]) - 1  # -1 to revert to 0-based counting
    zbins = np.max(ind) + 1

    cl_3d = np.zeros((nbl, zbins, zbins))
    for ell_idx in range(nbl):
        for zpair_idx, (zi, zj) in enumerate(zip(ind[:, 0], ind[:, 1])):
            cl_3d[ell_idx, zi, zj] = cl_2d[ell_idx, zpair_idx]

    if is_auto_spectrum:
        for ell in range(nbl):
            cl_3d[ell, ...] = symmetrize_2d_array(cl_3d[ell, ...])

    return ells, cl_3d


def zpairs_header_to_ind_array(header):
    # Initialize an empty list to store the pairs of integers
    ind = []

    # Loop through each element in the list
    for zpairs_str_list in header:
        # Split the element by the '-' character
        _zpairs_str_list = zpairs_str_list.split('-')

        # Remove the 'E'mand 'P' from each part and convert to integer
        zpairs = [
            int(zpair_str.replace('E', '').replace('P', ''))
            for zpair_str in _zpairs_str_list
        ]

        # Append the pair of integers to the list
        ind.append(zpairs)

    # Convert the list of pairs into a numpy array
    ind = np.array(ind)

    return ind


def find_nearest_idx(array, value):
    idx = np.abs(array - value).argmin()
    return idx


def flatten_dict(nested_dict):
    """Flatten a nested dictionary."""
    flattened = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def get_filenames_in_folder(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames


def test_folder_content(
    output_path, benchmarks_path, extension, verbose=False, rtol=1e-10
):
    """Test if the files in the output folder are equal to the benchmark files
    and list the discrepancies.

    Parameters
    ----------
    output_path (str): The path to the folder containing the output files.
    benchmarks_path (str): The path to the folder containing the benchmark files.
    extension (str): The extension of the files to be tested.

    Returns
    -------
    dict: A dictionary containing the comparison results and discrepancies.

    """
    old_files = os.listdir(benchmarks_path)
    new_files = os.listdir(output_path)

    # Ignore hidden files and filter by extension
    old_files = [
        file
        for file in old_files
        if not file.startswith('.') and file.endswith(extension)
    ]
    new_files = [
        file
        for file in new_files
        if not file.startswith('.') and file.endswith(extension)
    ]

    discrepancies = {
        'missing_in_benchmark': set(new_files) - set(old_files),
        'missing_in_output': set(old_files) - set(new_files),
        'comparison_results': [],
    }

    max_length_benchmark = max(
        (len(file) for file in discrepancies['missing_in_benchmark']), default=0
    )
    max_length_output = max(
        (len(file) for file in discrepancies['missing_in_output']), default=0
    )

    # Use the maximum length from either set for alignment
    max_length = max(max_length_benchmark, max_length_output)

    for file in discrepancies['missing_in_benchmark']:
        print(f'{file:<{max_length}} \t output ✅ \t benchmark ❌')

    for file in discrepancies['missing_in_output']:
        print(f'{file:<{max_length}} \t output ❌ \t benchmark ✅')
        print(f'{file} \t output ❌ \t benchmark ✅')

    for file_name in set(old_files).intersection(new_files):
        old_file_path = os.path.join(benchmarks_path, file_name)
        new_file_path = os.path.join(output_path, file_name)

        try:
            if extension == 'npz':
                np.testing.assert_allclose(
                    np.load(old_file_path)['arr_0'],
                    np.load(new_file_path)['arr_0'],
                    verbose=verbose,
                    rtol=rtol,
                    atol=0,
                )
            elif extension == 'npy':
                np.testing.assert_allclose(
                    np.load(old_file_path),
                    np.load(new_file_path),
                    verbose=verbose,
                    rtol=rtol,
                    atol=0,
                )
            elif extension in ['txt', 'dat']:
                np.testing.assert_allclose(
                    np.genfromtxt(old_file_path),
                    np.genfromtxt(new_file_path),
                    verbose=verbose,
                    rtol=rtol,
                    atol=0,
                )
            else:
                raise ValueError(f'Unknown extension: {extension}')
        except AssertionError as exc:
            discrepancies['comparison_results'].append((file_name, str(exc)))
            print(f'\nFile {file_name} does not match: {exc}')
        else:
            discrepancies['comparison_results'].append((file_name, 'Match'))
            print(f'{file_name:<{max_length}} \t matches to within {rtol * 100}% ✅')

    # Provide a summary of the results
    num_comparisons = len(discrepancies['comparison_results'])
    num_matches = sum(
        1 for _, result in discrepancies['comparison_results'] if result == 'Match'
    )
    num_discrepancies = num_comparisons - num_matches

    print(
        f'\nSummary: {num_comparisons} files compared, {num_matches} '
        f'matches, {num_discrepancies} discrepancies.\n'
    )

    return discrepancies


def import_files(folder_path, extension):
    """Imports all files with a specific extension from a folder and stores
    their contents in a dictionary.

    Parameters
    ----------
    folder_path (str): The path to the folder from which to import files.
    extension (str): The file extension to look for.

    Returns
    -------
    dict: A dictionary with filenames as keys and file contents as values.

    """
    files_dict = {}
    # Get the list of relevant file names using the get_filenames_in_folder function
    filenames = get_filenames_in_folder(folder_path)

    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        # Determine how to load the file based on the extension
        if extension in ['.txt', '.csv']:
            file_content = np.genfromtxt(
                file_path, delimiter=','
            )  # Adjust delimiter if necessary
        elif extension == '.npy':
            file_content = np.load(file_path)
        elif extension == '.npz':
            file_content = np.load(file_path)['arr_0']
        else:
            raise ValueError(f'Unsupported file extension: {extension}')

        # Use the file name without an extension as the key
        key = os.path.splitext(filename)[0]
        files_dict[key] = file_content

    return files_dict


def is_increasing(arr):
    return np.all(np.diff(arr) > 0)


def save_pickle(filename, obj):
    with open(f'{filename}', 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(filename):
    with open(f'{filename}', 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def save_compressed_pickle(title, data):
    import bz2

    with bz2.BZ2File(title + '.pbz2', 'wb') as handle:
        pickle.dump(data, handle)


def load_compressed_pickle(file):
    import bz2

    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


def read_yaml(filename):
    """A function to read YAML file.

    filename must include the path and the extension
    """
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


# @njit
def percent_diff(array_1, array_2, abs_value=False):
    array_1 = np.atleast_1d(array_1)  # Ensure array-like behavior
    array_2 = np.atleast_1d(array_2)

    diff = (array_1 / array_2 - 1) * 100

    # avoid nans
    both_zeros = np.logical_and(array_1 == 0, array_2 == 0)

    diff[both_zeros] = 0

    if abs_value:
        return np.abs(diff)
    else:
        # Convert back to scalar if necessary
        return diff.item() if diff.size == 1 else diff


# @njit
def percent_diff_mean(array_1, array_2):
    """Result is in "percent" units."""
    mean = (array_1 + array_2) / 2.0
    diff = (array_1 / mean - 1) * 100
    return diff


# @njit
def _percent_diff_nan(array_1, array_2, eraseNaN=True, log=False, abs_val=False):
    if eraseNaN:
        diff = np.where(array_1 == array_2, 0, percent_diff(array_1, array_2))
    else:
        diff = percent_diff(array_1, array_2)
    if log:
        diff = np.log10(diff)
    if abs_val:
        diff = np.abs(diff)
    return diff


def percent_diff_nan(array_1, array_2, eraseNaN=True, log=False, abs_val=False):
    """Calculate the percent difference between two arrays, handling NaN
    values.
    """
    # Handle NaN values
    if eraseNaN:
        # Mask where NaN values are present
        diff = np.ma.masked_where(
            np.isnan(array_1) | np.isnan(array_2), percent_diff(array_1, array_2)
        )
    else:
        diff = percent_diff(array_1, array_2)

    # Handle log transformation
    if log:
        # Mask zero differences before taking the log
        diff = np.ma.masked_where(diff == 0, diff)
        diff = np.log10(np.ma.abs(diff))  # Masked values will be ignored in the log

    # Handle absolute values
    if abs_val:
        diff = np.ma.abs(diff)

    return diff


def diff_threshold_check(diff, threshold):
    boolean = np.any(np.abs(diff) > threshold)
    print(f'has any element of the arrays a disagreement > {threshold}%? ', boolean)


def compute_smape(vec_true, vec_test, cov_mat=None):
    """Computes the SMAPE (Symmetric Mean Absolute Percentage Error) for a
    given 1D array with weighted elements.

    Args:
        vec_true (np.array): array of true values
        vec_test (np.array): array of predicted/approximated values
        cov_mat (np.array): covariance matrix for vec_true

    Returns:
        float: SMAPE value

    """
    if isinstance(vec_true, np.ndarray) and isinstance(vec_test, np.ndarray):
        assert len(vec_true) == len(vec_test), 'Arrays must have the same length'
        assert vec_true.ndim == 1 and vec_test.ndim == 1, 'arrays must be 1D'

    if cov_mat is not None:
        assert cov_mat.shape[0] == cov_mat.shape[1] == len(vec_true), (
            'cov_mat must be a square matrix with the same length as the input vectors'
        )
        weights = vec_true / np.sqrt(np.diag(cov_mat))
    else:
        weights = np.ones_like(vec_true)  # uniform weights

    numerator = weights * np.abs(vec_true - vec_test)
    denominator = np.abs(vec_true) + np.abs(vec_test)

    return 100 * np.mean(numerator / denominator)  # the output is already a precentage


def smape(y_true, y_pred):
    """Compute the point-by-point Symmetric Mean Absolute Percentage Error (SMAPE)
    between two arrays of the same shape.

    Parameters
    ----------
    y_true : array_like
        Reference or true values.
    y_pred : array_like
        Predicted or test values.

    Returns
    -------
    smape : ndarray
        Array of SMAPE values in percentage (same shape as inputs).

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = 0.5 * (np.abs(y_true) + np.abs(y_pred))

    with np.errstate(divide='ignore', invalid='ignore'):
        smape = 100 * numerator / denominator
        smape[denominator == 0] = 0.0  # Define SMAPE as 0 if denominator is 0

    return smape


def compute_diff_sigma(vec_true, vec_test, sigma):
    """Compute the element-wise difference between two vectors vec_true and
    vec_test, and divide it by sigma.

    Args:
    - vec_true (numpy.ndarray): A numpy array representing the first vector.
    - vec_test (numpy.ndarray): A numpy array representing the second vector.
    - sigma (numpy.ndarray): A numpy array representing the
    vector of standard deviations.

    Returns:
    - A numpy array representing the element-wise difference between vec_true and
    vec_test,
      divided by sigma.

    """
    diff = np.abs(vec_true - vec_test)
    return 100 * diff / sigma


def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def compare_arrays(
    A,
    B,
    name_A='A',
    name_B='B',
    plot_diff=True,
    plot_array=True,
    log_array=True,
    log_diff=False,
    abs_val=False,
    plot_diff_threshold=None,
    white_where_zero=True,
    plot_diff_hist=False,
):
    fontsize = 25

    if np.array_equal(A, B):
        print(f'{name_A} and {name_B} are equal ✅')
        return

    for rtol in [1e-3, 1e-2, 5e-2]:  # these are NOT percent units
        if np.allclose(A, B, rtol=rtol, atol=0):
            print(
                f'{name_A} and {name_B} are close within relative tolerance '
                f'of {rtol * 100}%) ✅'
            )
            return

    diff_AB = percent_diff_nan(A, B, eraseNaN=True, abs_val=abs_val)
    higher_rtol = plot_diff_threshold or 5.0
    max_diff = np.max(diff_AB)
    result_emoji = '❌' if max_diff > higher_rtol or np.isnan(max_diff) else '✅'
    no_outliers = np.sum(diff_AB > higher_rtol)
    additional_info = (
        f'\nMax discrepancy: {max_diff:.2f}%;'
        f'\nNumber of elements with discrepancy > {higher_rtol}%: {no_outliers}'
        f'\nFraction of elements with discrepancy > {higher_rtol}%: '
        f'{no_outliers / diff_AB.size:.5f}'
    )
    print(
        f'Are {name_A} and {name_B} different by less than {higher_rtol}%? '
        f'{result_emoji} {additional_info}'
    )

    # Check that arrays are 2D if any plotting is requested.
    if plot_diff or plot_array:
        assert A.ndim == 2 and B.ndim == 2, 'Plotting is only implemented for 2D arrays'

    # Determine number of rows:
    nrows = (1 if plot_array else 0) + (1 if plot_diff else 0)
    ncols = 2  # Always show 2 panels per row

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(17, 7 * nrows), constrained_layout=True
    )

    # Ensure ax is always 2D
    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)  # Convert row array to 2D
    if ncols == 1:
        ax = np.expand_dims(ax, axis=1)  # Convert column array to 2D

    # If plotting arrays, prepare data and plot in first row.
    if plot_array:
        A_toplot, B_toplot = A.copy(), B.copy()
        if abs_val:
            A_toplot, B_toplot = np.abs(A_toplot), np.abs(B_toplot)
        if log_array:
            A_toplot, B_toplot = np.log10(A_toplot), np.log10(B_toplot)

        im = ax[0, 0].matshow(A_toplot)
        ax[0, 0].set_title(f'{name_A}', fontsize=fontsize)
        fig.colorbar(im, ax=ax[0, 0])

        im = ax[0, 1].matshow(B_toplot)
        ax[0, 1].set_title(f'{name_B}', fontsize=fontsize)
        fig.colorbar(im, ax=ax[0, 1])

    # If plotting differences, prepare diff data and plot in next row.
    if plot_diff:
        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=False, abs_val=abs_val)
        diff_BA = percent_diff_nan(B, A, eraseNaN=True, log=False, abs_val=abs_val)

        if plot_diff_threshold is not None:
            # Mask out small differences (set them to white via the colormap's
            # "bad" color)
            diff_AB = np.ma.masked_where(np.abs(diff_AB) < plot_diff_threshold, diff_AB)
            diff_BA = np.ma.masked_where(np.abs(diff_BA) < plot_diff_threshold, diff_BA)

        if log_diff:
            # Replace nonpositive with nan to avoid -inf
            diff_AB = np.log10(np.abs(diff_AB))
            diff_BA = np.log10(np.abs(diff_BA))

        im = ax[1, 0].matshow(diff_AB)
        ax[1, 0].set_title('(A/B - 1) * 100', fontsize=fontsize)
        fig.colorbar(im, ax=ax[1, 0])

        im = ax[1, 1].matshow(diff_BA)
        ax[1, 1].set_title('(B/A - 1) * 100', fontsize=fontsize)
        fig.colorbar(im, ax=ax[1, 1])

    fig.suptitle(
        f'log_array={log_array}, abs_val={abs_val}, log_diff={log_diff}\n'
        f'plot_diff_threshold={plot_diff_threshold}%',
        fontsize=fontsize,
    )
    plt.show()

    if plot_diff_hist:
        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=False, abs_val=False)
        plt.figure()
        plt.hist(diff_AB.flatten(), bins=30, log=True, density=True)
        plt.xlabel('% difference')
        plt.ylabel('frequency')
        plt.show()


def compare_folder_content(path_A: str, path_B: str, filetype: str):
    """Compare the content of 2 folders.

    The files in folder A should be a subset of the files in folder B.
    """
    dict_A = dict(get_kv_pairs(path_A, filetype))
    dict_B = dict(get_kv_pairs(path_B, filetype))

    for key in dict_A:  # noqa: PLC0206
        result_emoji = '✅' if np.array_equal(dict_A[key], dict_B[key]) else '❌'
        print(f'is {key} equal in both folders? {result_emoji}')


def namestr(obj, namespace):
    """Does not work with slices!!!

    (why?)
    """
    return [name for name in namespace if namespace[name] is obj][0]


def find_null_rows_cols_2D(array):
    """:param array:

    :return null_rows_idxs: list
        array of null rows/columns indices
    """
    assert array.ndim == 2, (
        'ndim should be <= 2; higher-dimensional case not yet implemented'
    )
    null_rows_idxs = np.where(np.all(array == 0, axis=0))[0]
    null_cols_idxs = np.where(np.all(array == 0, axis=1))[0]

    assert np.array_equal(null_rows_idxs, null_cols_idxs), (
        'null rows and columns indices should be the same for Fisher matrices'
    )

    if null_rows_idxs.shape[0] == 0:
        # print('The input array has no null rows/columns')
        return None
    else:
        # print(f'The input array had null rows and columns '
        # f'at indices {null_rows_idxs}')
        return null_rows_idxs


def remove_rows_cols_array2D(array, rows_idxs_to_remove):
    """Removes the *same* rows and columns from an array :param array:
    numpy.ndarray.

    input 2D array
    :param rows_idxs_to_remove: list. rows (and columns) to delete
    :return: array without null rows and columns
    """
    if rows_idxs_to_remove is None:
        warnings.warn('null_rows_idxs is None, returning the input array', stacklevel=2)
        return array

    if len(rows_idxs_to_remove) == 0:
        warnings.warn(
            'null_rows_idxs is empty, returning the input array', stacklevel=2
        )
        return array

    assert array.ndim == 2, (
        'ndim should be <= 2; higher-dimensional case not yet implemented'
    )
    array = np.delete(array, rows_idxs_to_remove, axis=0)
    array = np.delete(array, rows_idxs_to_remove, axis=1)
    return array


def remove_null_rows_cols_2D(array_2d: np.ndarray):
    """Remove null rows and columns from a 2D numpy array.

    Args:
        array_2d (numpy.ndarray): The 2D numpy array to remove null rows and columns
        from.

    Returns:
        numpy.ndarray: The 2D numpy array with null rows and columns removed.

    """
    assert array_2d.ndim == 2, (
        'ndim should be <= 2; higher-dimensional case not yet implemented'
    )
    array_2d = array_2d[~np.all(array_2d == 0, axis=1)]
    array_2d = array_2d[:, ~np.all(array_2d == 0, axis=0)]
    return array_2d


def mask_FM_null_rowscols(FM, params, fid):
    """Mask the Fisher matrix, fiducial values and parameter list deleting the
    null rows and columns :param FM: Fisher Matrix, 2D numpy array :return:
    masked FM, fiducial values and parameter list.
    """
    null_idx = find_null_rows_cols_2D(FM)

    if null_idx is None:
        return FM, params, fid

    FM = remove_rows_cols_array2D(FM, null_idx)
    params = np.delete(params, obj=null_idx, axis=0)
    fid = np.delete(fid, obj=null_idx, axis=0)
    assert len(fid) == len(params), (
        'the fiducial values and parameter lists should have the same length'
    )
    return FM, list(params), list(fid)


def mask_fm_null_rowscols_v2(fm, fiducials_dict):
    """Mask the Fisher matrix, fiducial values and parameter list deleting the
    null rows and columns :param FM: Fisher Matrix, 2D numpy array :return:
    masked FM, fiducial values and parameter list.
    """
    param_names = list(fiducials_dict.keys())
    param_values = list(fiducials_dict.values())

    null_idxs = find_null_rows_cols_2D(fm)

    if null_idxs is None:
        return fm, fiducials_dict

    # remove null rows and columns
    fm = remove_rows_cols_array2D(fm, null_idxs)
    # update fiducials dict
    fiducials_dict = {
        param_names[i]: param_values[i]
        for i in range(len(param_names))
        if i not in null_idxs
    }

    return fm, fiducials_dict


def mask_fisher(
    fm: np.ndarray,
    fiducials_dict: dict,
    names_params_to_fix: list,
    remove_null_rows_cols: bool,
):
    """Trim the Fisher matrix to remove null rows/columns and/or fix nuisance
    parameters.
    """
    fm = deepcopy(fm)
    fiducials_dict = deepcopy(fiducials_dict)

    assert len(list(fiducials_dict.keys())) == fm.shape[0] == fm.shape[1], (
        'Wrong shape of FM matrix!'
    )

    if names_params_to_fix is not None:
        fm = fix_params_in_fm(fm, names_params_to_fix, fiducials_dict)  # cut fm entries
        # update fiducials_dict
        fiducials_dict = {
            key: fiducials_dict[key]
            for key in fiducials_dict
            if key not in names_params_to_fix
        }

    # remove remaining null rows_cols
    if remove_null_rows_cols:
        fm, fiducials_dict = mask_fm_null_rowscols_v2(fm, fiducials_dict)

    return fm, fiducials_dict


def fix_params_in_fm(fm, names_params_to_fix, fiducials_dict):
    param_names = list(fiducials_dict.keys())
    fm = deepcopy(fm)
    fiducials_dict = deepcopy(fiducials_dict)

    # check the correctness of the parameters' names
    for param_to_fix in names_params_to_fix:
        assert param_to_fix in param_names, (
            f'Parameter {param_to_fix} not found in param_names!'
        )

    rows_idxs_to_remove = [
        param_names.index(param_to_fix) for param_to_fix in names_params_to_fix
    ]
    fm = remove_rows_cols_array2D(fm, rows_idxs_to_remove)
    # print(f'Removing rows and columns from FM:\n{rows_idxs_to_remove}')

    return fm


def add_prior_to_fm(fm, fiducials_dict, prior_param_names, prior_param_values):
    """Adds a FM of priors (with elements 1/sigma in the correct positions) to
    the input FM.
    """
    fm = deepcopy(fm)
    fiducials_dict = deepcopy(fiducials_dict)

    assert len(list(fiducials_dict.keys())) == fm.shape[0] == fm.shape[1], (
        'Wrong shape of FM matrix!'
    )
    fid_param_names = list(fiducials_dict.keys())

    for prior_param_name in prior_param_names:
        if prior_param_name not in fid_param_names:
            warnings.warn(
                f'Prior parameter {prior_param_names} not found in fiducial parameters'
                ' dict!',
                stacklevel=2,
            )
            return fm

    prior_param_idxs = [
        fid_param_names.index(prior_param_name)
        for prior_param_name in prior_param_names
    ]

    prior_fm = np.zeros(fm.shape)
    prior_fm[prior_param_idxs, prior_param_idxs] = 1 / np.array(prior_param_values)
    return fm + prior_fm


def uncertainties_fm_v2(
    fm, fiducials_dict, which_uncertainty='marginal', normalize=True, percent_units=True
):
    """Returns relative 1-sigma error."""
    param_names = list(fiducials_dict.keys())
    param_values = np.array(list(fiducials_dict.values()))

    assert len(param_names) == param_values.shape[0] == fm.shape[0] == fm.shape[1], (
        'param_names and param_values must have the same length and be equal to the '
        'number of rows and columns of fm'
    )

    if which_uncertainty == 'marginal':
        fm_inv = np.linalg.inv(fm)
        sigma_fm = np.sqrt(np.diag(fm_inv))
    elif which_uncertainty == 'conditional':
        sigma_fm = np.sqrt(1 / np.diag(fm))
    else:
        raise ValueError('which_uncertainty must be either "marginal" or "conditional"')

    if normalize:
        # if the fiducial for is 0, substitute with 1 to avoid division by zero;
        # if it's -1, take the absolute value
        param_values = np.where(param_values == 0, 1, param_values)
        param_values = np.where(param_values < 0, np.abs(param_values), param_values)
        # normalize to get the relative uncertainties
        sigma_fm /= param_values

    if percent_units:
        return sigma_fm * 100

    return sigma_fm


def build_labels_TeX(zbins):
    galaxy_bias_label = [f'$b_{{{i + 1}}}$' for i in range(zbins)]
    shear_bias_label = [f'$m_{{{i + 1}}}$' for i in range(zbins)]
    zmean_shift_label = [f'$dz_{{{i + 1}}}$' for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def build_labels(zbins):
    galaxy_bias_label = [f'b{(i + 1):02d}' for i in range(zbins)]
    shear_bias_label = [f'm{(i + 1):02d}' for i in range(zbins)]
    zmean_shift_label = [f'dz{(i + 1):02d}' for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def matshow(
    array,
    title='title',
    log=True,
    abs_val=False,
    threshold=None,
    only_show_nans=False,
    matshow_kwargs: dict = None,
):
    """:param array:
    :param title:
    :param log:
    :param abs_val:
    :param threshold: if None, do not mask the values; otherwise,
    keep only the elements above the threshold (i.e., mask the ones below the threshold)
    :return:
    """
    if matshow_kwargs is None:
        matshow_kwargs = {}
    if only_show_nans:
        warnings.warn(
            'only_show_nans is True, better switch off log and abs_val for the moment',
            stacklevel=2,
        )
        # Set non-NaN elements to 0 and NaN elements to 1
        array = np.where(np.isnan(array), 1, 0)
        title += ' (only NaNs shown)'

    # the ordering of these is important: I want the log(abs), not abs(log)
    if abs_val:  # take the absolute value
        array = np.abs(array)
        title = 'abs ' + title
    if log:  # take the log
        with np.errstate(divide='ignore', invalid='ignore'):
            array = np.log10(array)
        title = 'log10 ' + title

    if threshold is not None:
        array = np.ma.masked_where(array < threshold, array)
        title += f' \n(masked below {threshold} %)'

    plt.matshow(array, **matshow_kwargs)
    plt.colorbar()
    plt.title(title)
    plt.show()


def get_kv_pairs(path_import, extension='npy'):
    """Load txt or dat files in dictionary.

    To use it, wrap it in "dict(), e.g.:     loaded_dict =
    dict(get_kv_pairs(path_import, filetype="dat"))
    """
    if extension in ('npy', 'npz'):
        load_function = np.load
    elif extension in ('txt', 'dat'):
        load_function = np.genfromtxt
    else:
        raise NotImplementedError(
            "extension must be either 'npy', 'npz', 'txt' or 'dat'"
        )

    for path in Path(path_import).glob(f'*.{extension}'):
        yield path.stem, load_function(str(path))


def get_kv_pairs_v2(path_import, extension='npy'):
    """Load npy, npz, txt, or dat files in dictionary.

    To use it, wrap it in "dict(), e.g.:     loaded_dict =
    dict(get_kv_pairs(path_import, filetype="dat"))
    """
    if extension in ('npy', 'npz'):
        load_function = np.load
    elif extension in ('txt', 'dat'):

        def load_function(p):
            return np.genfromtxt(p, encoding='latin1')  # Handle non-UTF-8 encoding
    else:
        raise NotImplementedError(
            "extension must be either 'npy', 'npz', 'txt' or 'dat'"
        )

    for path in Path(path_import).glob(f'*.{extension}'):
        if path.is_file():  # Ensure it's a file, not a directory
            try:
                yield path.stem, load_function(str(path))
            except UnicodeDecodeError as e:
                print(f'Error decoding file {path}: {e}')
            except Exception as e:
                print(f'Error loading file {path}: {e}')


# to display the names (keys) more tidily
def show_keys(arrays_dict):
    for key in arrays_dict:
        print(key)


# @njit
def symmetrize_Cl(Cl, nbl, zbins):
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                Cl[ell, j, i] = Cl[ell, i, j]
    return Cl


def generate_ind(triu_tril_square, row_col_major, size):
    """Generates a list of indices for the upper triangular part of a matrix
    :param triu_tril_square: str.

    if 'triu', returns the indices for the upper triangular part of the
    matrix. If 'tril', returns the indices for the lower triangular part
    of the matrix If 'full_square', returns the indices for the whole
    matrix
    :param row_col_major: str. if True, the indices are returned in row-
        major order; otherwise, in column-major order
    :param size: int. size of the matrix to take the indices of
    :return: list of indices
    """
    assert row_col_major in ['row-major', 'col-major'], (
        'row_col_major must be either "row-major" or "col-major"'
    )
    assert triu_tril_square in ['triu', 'tril', 'full_square'], (
        'triu_tril_square must be either "triu", "tril" or "full_square"'
    )

    if triu_tril_square == 'triu':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i, size)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i + 1)]
    elif triu_tril_square == 'tril':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i + 1)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i, size)]
    elif triu_tril_square == 'full_square':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(size)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(size)]

    return np.asarray(ind)


def build_full_ind(triu_tril, row_col_major, size):
    """Builds the good old ind file."""
    assert triu_tril in ['triu', 'tril'], 'triu_tril must be either "triu" or "tril"'
    assert row_col_major in ['row-major', 'col-major'], (
        'row_col_major must be either "row-major" or "col-major"'
    )

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    LL_columns = np.zeros((zpairs_auto, 2))
    GL_columns = np.hstack((np.ones((zpairs_cross, 1)), np.zeros((zpairs_cross, 1))))
    GG_columns = np.ones((zpairs_auto, 2))

    LL_columns = np.hstack(
        (LL_columns, generate_ind(triu_tril, row_col_major, size))
    ).astype(int)
    GL_columns = np.hstack(
        (GL_columns, generate_ind('full_square', row_col_major, size))
    ).astype(int)
    GG_columns = np.hstack(
        (GG_columns, generate_ind(triu_tril, row_col_major, size))
    ).astype(int)

    ind = np.vstack((LL_columns, GL_columns, GG_columns))

    assert ind.shape[0] == zpairs_3x2pt, 'ind has the wrong number of rows'

    return ind


def build_ind_dict(triu_tril, row_col_major, size, GL_OR_LG):
    ind = build_full_ind(triu_tril, row_col_major, size)
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    ind_dict = {}
    ind_dict['L', 'L'] = ind[:zpairs_auto, :]
    ind_dict['G', 'G'] = ind[(zpairs_auto + zpairs_cross) :, :]

    if GL_OR_LG == 'LG':
        ind_dict['L', 'G'] = ind[zpairs_auto : (zpairs_auto + zpairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_OR_LG == 'GL':
        ind_dict['G', 'L'] = ind[zpairs_auto : (zpairs_auto + zpairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    return ind_dict


# CHECK FOR DUPLICATES
def cl_2D_to_3D_symmetric(Cl_2D, nbl, zpairs, zbins):
    """Reshape from (nbl, zpairs) to (nbl, zbins, zbins) according to upper
    traigular ordering 0-th rows filled first, then second from i to 10...
    """
    assert Cl_2D.shape == (nbl, zpairs), (
        f'cl_2d must have shape (nbl, zpairs) = ({nbl}, {zpairs})'
    )

    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for zpair_idx in range(zpairs):
            i, j = triu_idx[0][zpair_idx], triu_idx[1][zpair_idx]
            Cl_3D[ell, i, j] = Cl_2D[ell, zpair_idx]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def cl_2D_to_3D_asymmetric(Cl_2D, nbl, zbins, order):
    """Reshape from (nbl, npairs) to (nbl, zbins, zbins), rows first (valid for
    asymmetric Cij, i.e. C_XC)
    """
    assert order in ['row-major', 'col-major', 'C', 'F'], (
        'order must be either "row-major", "C" (equivalently), or'
        '"col-major", "F" (equivalently)'
    )
    if order == 'row-major':
        order = 'C'
    elif order == 'col-major':
        order = 'F'

    Cl_3D = np.zeros((nbl, zbins, zbins))
    Cl_3D = np.reshape(Cl_2D, Cl_3D.shape, order=order)
    return Cl_3D


def Cl_3D_to_2D_symmetric(Cl_3D, nbl, npairs, zbins=10):
    """Reshape from (nbl, zbins, zbins) to (nbl, npairs)  according to upper
    traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_2D = np.zeros((nbl, npairs))
    for ell in range(nbl):
        for i in range(npairs):
            Cl_2D[ell, i] = Cl_3D[ell, triu_idx[0][i], triu_idx[1][i]]
    return Cl_2D


def Cl_3D_to_2D_asymmetric(Cl_3D):
    """Reshape from (nbl, zbins, zbins) to (nbl, npairs), rows first (valid for
    asymmetric Cij, i.e. C_XC)
    """
    assert Cl_3D.ndim == 3, 'Cl_3D must be a 3D array'

    nbl = Cl_3D.shape[0]
    zbins = Cl_3D.shape[1]
    zpairs_cross = zbins**2

    Cl_2D = np.reshape(Cl_3D, (nbl, zpairs_cross))

    # Cl_2D = np.zeros((nbl, zpairs_cross))
    # for ell in range(nbl):
    #     Cl_2D[ell, :] = Cl_3D[ell, :].flatten(order='C')
    return Cl_2D


def cl_3D_to_2D_or_1D(
    cl_3D, ind, is_auto_spectrum, use_triu_row_major, convert_to_2D, block_index
):
    """Reshape from (nbl, zbins, zbins) to (nbl, zpairs), according to the
    ordering given in the ind file (valid for asymmetric Cij, i.e. C_XC)
    """
    # warnings.warn('finish this function!! (old warning, I dont remember
    # exactly what is missing...)')
    assert cl_3D.ndim == 3, 'cl_3D must be a 3D array'
    assert cl_3D.shape[1] == cl_3D.shape[2], (
        'cl_3D must be a square array of shape (nbl, zbins, zbins)'
    )

    zbins = cl_3D.shape[1]
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    if use_triu_row_major:
        ind = build_full_ind('triu', 'row-major', zbins)

    # Select appropriate indices based on whether the spectrum is auto or cross
    zpairs = zpairs_auto if is_auto_spectrum else zpairs_cross
    selected_ind = (
        ind[:zpairs, 2:]
        if is_auto_spectrum
        else ind[zpairs_auto : zpairs_auto + zpairs_cross, 2:]
    )

    # Vectorize the selection of elements based on the indices
    cl_2D = cl_3D[:, selected_ind[:, 0], selected_ind[:, 1]]

    if convert_to_2D:
        return cl_2D

    # Flatten the array based on the 'block_index' parameter
    order = 'C' if block_index == 'ell' else 'F' if block_index == 'zpair' else None
    if order is None:
        raise ValueError('block_index must be either "ell" or "zpair"')

    return cl_2D.flatten(order=order)


# @njit
def cl_1D_to_3D(cl_1d, nbl: int, zbins: int, is_symmetric: bool):
    """This is used to unpack Vincenzo's files for SPV3 Still to be thoroughly
    checked.
    """
    cl_3d = np.zeros((nbl, zbins, zbins))
    p = 0
    if is_symmetric:
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(iz, zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1

    else:  # take all elements, not just the upper triangle
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1
    return cl_3d


# @njit
def symmetrize_2d_array(array_2d):
    """Mirror the lower/upper triangle."""
    # if already symmetric, do nothing
    if check_symmetric(array_2d, exact=True):
        return array_2d

    # there is an implicit "else" here, since the function returns array_2d if
    # the array is symmetric
    assert array_2d.ndim == 2, 'array must be square'
    size = array_2d.shape[0]

    # check that either the upper or lower triangle (not including the diagonal) is null
    triu_elements = array_2d[np.triu_indices(size, k=+1)]
    tril_elements = array_2d[np.tril_indices(size, k=-1)]
    assert np.all(triu_elements) == 0 or np.all(tril_elements) == 0, (
        'neither the upper nor the lower triangle (excluding the diagonal) are null'
    )

    # if np.any(np.diag(array_2d)) != 0:
    # warnings.warn('the diagonal elements are all null', stacklevel=2)

    # symmetrize
    array_2d = np.where(array_2d, array_2d, array_2d.T)
    # check
    if not check_symmetric(array_2d, exact=False):
        warnings.warn('check failed: the array is not symmetric', stacklevel=2)

    return array_2d


def fill_3D_symmetric_array(array_3D, nbl, zbins):
    """Mirror the lower/upper triangle."""
    assert array_3D.shape == (nbl, zbins, zbins), (
        'shape of input array must be (nbl, zbins, zbins)'
    )

    array_diag_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        array_diag_3D[ell, :, :] = np.diag(np.diagonal(array_3D, 0, 1, 2)[ell, :])
    array_3D = array_3D + np.transpose(array_3D, (0, 2, 1)) - array_diag_3D
    return array_3D


def array_2D_to_1D_ind(array_2D, zpairs, ind):
    """Unpack according to "ind" ordering, same as for the Cls."""
    assert ind.shape[0] == zpairs, 'ind must have lenght zpairs'

    array_1D = np.zeros(zpairs)
    for p in range(zpairs):
        i, j = ind[p, 2], ind[p, 3]
        array_1D[p] = array_2D[i, j]
    return array_1D


def dC_4D_to_3D(dC_4D, nbl, zpairs, nparams_tot, ind):
    """Expand the zpair indices into zi, zj, according to the ind ordering as usual"""
    dC_3D = np.zeros((nbl, zpairs, nparams_tot))
    for ell in range(nbl):
        for alf in range(nparams_tot):
            dC_3D[ell, :, alf] = array_2D_to_1D_ind(dC_4D[ell, :, :, alf], zpairs, ind)
    return dC_3D


def dcl_dict_to_4D_array(
    dcl_dict_3d, param_names, nbl, zbins, derivatives_prefix, is_3x2pt=False, n_probes=2
):
    """:param param_names: filename of the parameter, e.g. 'Om'; dCldOm = d(C(l))/d(Om)
    :param dC_dict_3D:
    :param nbl:
    :param zbins:
    :param obs_name: filename of the observable, e.g. 'Cl'; dCldOm = d(C(l))/d(Om)
    :param is_3x2pt: whether to will the 5D derivatives vector
    :param n_probes:
    :return:
    """
    # param_names should be params_tot in all cases, because when the derivative dows not exist
    # in dC_dict_3D the output array will remain null
    if is_3x2pt:
        dC_4D = np.zeros((n_probes, n_probes, nbl, zbins, zbins, len(param_names)))
    else:
        dC_4D = np.zeros((nbl, zbins, zbins, len(param_names)))

    if not dcl_dict_3d:
        warnings.warn('The input dictionary is empty', stacklevel=2)

    no_derivative_counter = 0
    for idx, param_name in enumerate(param_names):
        for key, value in dcl_dict_3d.items():
            if f'{derivatives_prefix}{param_name}' in key:
                dC_4D[..., idx] = value

        # a check, if the derivative wrt the param is not in the folder at all
        if all(f'{derivatives_prefix}{param_name}' not in key for key in dcl_dict_3d):
            print(
                f'Derivative {derivatives_prefix}{param_name} not found; setting the corresponding FM entry to zero'
            )
            no_derivative_counter += 1
        if no_derivative_counter == len(param_names):
            raise ImportError(
                'No derivative found for any of the parameters in the input dictionary'
            )
    return dC_4D


def compute_FoM(FM, w0wa_idxs):
    cov_param = np.linalg.inv(FM)
    # cov_param_reduced = cov_param[start:stop, start:stop]
    cov_param_reduced = cov_param[np.ix_(w0wa_idxs, w0wa_idxs)]

    FM_reduced = np.linalg.inv(cov_param_reduced)
    FoM = np.sqrt(np.linalg.det(FM_reduced))
    return FoM


def get_zpairs(zbins):
    zpairs_auto = int((zbins * (zbins + 1)) / 2)  # = 55 for zbins = 10, cast it as int
    zpairs_cross = zbins**2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


def cov_g_terms_helper(a, b, mix: bool, prefactor, return_only_diagonal_ells):
    """Helper function to compute covariance terms."""
    if return_only_diagonal_ells:
        if mix:
            term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', a, b)
            term_2 = np.einsum('ACLik, BDLjl -> ABCDLijkl', b, a)
            term_3 = np.einsum('ADLil, BCLjk -> ABCDLijkl', a, b)
            term_4 = np.einsum('ADLil, BCLjk -> ABCDLijkl', b, a)

        else:
            term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', a, b)
            term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', a, b)
            term_3 = 0
            term_4 = 0

        cov = np.einsum(
            'ABCDLijkl, L -> ABCDLijkl', term_1 + term_2 + term_3 + term_4, prefactor
        )

    else:
        if mix:
            term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', a, b)
            term_2 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', b, a)
            term_3 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', a, b)
            term_4 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', b, a)

        else:
            term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', a, b)
            term_2 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', a, b)
            term_3 = 0
            term_4 = 0

        cov = np.einsum(
            'ABCDLMijkl, LM -> ABCDLMijkl', term_1 + term_2 + term_3 + term_4, prefactor
        )
    return cov


def covariance_einsum(
    cl_5d,
    noise_5d,
    f_sky,
    ell_values,
    delta_ell,
    split_terms,
    return_only_diagonal_ells=False,
):
    """Optimized version to avoid repeated code in the covariance
    calculation.
    """
    assert cl_5d.shape[0] in [1, 2], 'This function only works with 1 or 2 probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], (
        'cl_5d must have shape (n_probes, n_probes, nbl, zbins, zbins)'
    )
    assert cl_5d.shape[-1] == cl_5d.shape[-2], (
        'cl_5d must have shape (n_probes, n_probes, nbl, zbins, zbins)'
    )
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have the same shape as cl_5d'

    # Prefactor setup
    prefactor = 1 / ((2 * ell_values + 1) * f_sky * delta_ell)
    if not return_only_diagonal_ells:
        prefactor = np.diag(prefactor)

    clplusn_5d = cl_5d + noise_5d

    # Compute total covariance without splitting terms
    if not split_terms:
        cov = cov_g_terms_helper(
            clplusn_5d, clplusn_5d, False, prefactor, return_only_diagonal_ells
        )
        return cov

    # Split terms
    cov_sva = cov_g_terms_helper(
        cl_5d, cl_5d, False, prefactor, return_only_diagonal_ells
    )
    cov_sn = cov_g_terms_helper(
        noise_5d, noise_5d, False, prefactor, return_only_diagonal_ells
    )
    cov_mix = cov_g_terms_helper(
        cl_5d, noise_5d, True, prefactor, return_only_diagonal_ells
    )

    return cov_sva, cov_sn, cov_mix


def cov_10D_dict_to_array(cov_10D_dict, nbl, zbins, n_probes=2):
    """Transforms a dictionary of "shape" [(A, B, C, D)][ nbl, nbl, zbins,
    zbins, zbins, zbins] (where A, B, C, D is a tuple of strings, each one
    being either 'L' or 'G') to a numpy array of shape (n_probes, n_probes,
    n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)
    """
    assert n_probes == 2, (
        'if more than 2 probes are used, the probe_dict must be changed '
        '(promote it to an argument of the function!)'
    )
    cov_10D_array = np.zeros(
        (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)
    )
    probe_dict = {'L': 0, 'G': 1}
    for A, B, C, D in cov_10D_dict:
        cov_10D_array[
            probe_dict[A], probe_dict[B], probe_dict[C], probe_dict[D], ...
        ] = cov_10D_dict[A, B, C, D]

    return cov_10D_array


def cov_10D_array_to_dict(cov_10D_array, probe_ordering):
    """Transforms a dictionary of "shape" [(A, B, C, D)][ nbl, nbl, zbins,
    zbins, zbins, zbins] (where A, B, C, D is a tuple of strings, each one
    being either 'L' or 'G') to a numpy array of shape (n_probes, n_probes,
    n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)
    """
    cov_10D_dict = {}
    probe_dict = {'L': 0, 'G': 1}
    for A_str, B_str in probe_ordering:
        for C_str, D_str in probe_ordering:
            A_idx, B_idx, C_idx, D_idx = (
                probe_dict[A_str],
                probe_dict[B_str],
                probe_dict[C_str],
                probe_dict[D_str],
            )
            cov_10D_dict[A_str, B_str, C_str, D_str] = cov_10D_array[
                A_idx, B_idx, C_idx, D_idx, ...
            ]

    return cov_10D_dict


# @njit
def build_3x2pt_dict(array_3x2pt):
    dict_3x2pt = {}
    if array_3x2pt.ndim == 5:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :, :]
    elif array_3x2pt.ndim == 4:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :]
    return dict_3x2pt


def build_3x2pt_array(cl_LL_3D, cl_GG_3D, cl_GL_3D, n_probes, nbl, zbins):
    warnings.warn(
        'shape is (n_probes, n_probes, nbl, zbins, zbins), NOT '
        '(nbl, n_probes, n_probes, zbins, zbins)',
        stacklevel=2,
    )
    assert cl_LL_3D.shape == (nbl, zbins, zbins), (
        f'cl_LL_3D.shape = {cl_LL_3D.shape}, should be (nbl, zbins, zbins)'
    )
    assert cl_GL_3D.shape == (nbl, zbins, zbins), (
        f'cl_GL_3D.shape = {cl_GL_3D.shape}, should be (nbl, zbins, zbins)'
    )
    assert cl_GG_3D.shape == (nbl, zbins, zbins), (
        f'cl_GG_3D.shape = {cl_GG_3D.shape}, should be (nbl, zbins, zbins)'
    )
    cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
    cl_3x2pt_5D[0, 0, :, :, :] = cl_LL_3D
    cl_3x2pt_5D[0, 1, :, :, :] = cl_GL_3D.transpose(0, 2, 1)
    cl_3x2pt_5D[1, 0, :, :, :] = cl_GL_3D
    cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_3D
    return cl_3x2pt_5D


def cov_3x2pt_10D_to_4D(cov_3x2pt_10D, probe_ordering, nbl, zbins, ind_copy, GL_OR_LG):
    """Takes the cov_3x2pt_10D dictionary, reshapes each A, B, C, D block
    separately in 4D, then stacks the blocks in the right order to output
    cov_3x2pt_4D (which is not a dictionary but a numpy array)

    probe_ordering: e.g. ['L', 'L'], ['G', 'L'], ['G', 'G']]
    """
    # if it's an array, convert to dictionary for the function to work
    if isinstance(cov_3x2pt_10D, np.ndarray):
        cov_3x2pt_dict_10D = cov_10D_array_to_dict(cov_3x2pt_10D, probe_ordering)
    elif isinstance(cov_3x2pt_10D, dict):
        cov_3x2pt_dict_10D = cov_3x2pt_10D
    else:
        raise ValueError('cov_3x2pt_10D must be either a dictionary or an array')

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    # Check that the cross-correlation is coherent with the probe_ordering list
    # this is a weak check, since I'm assuming that GL or LG will be the second
    # element of the datavector
    if GL_OR_LG == 'GL':
        assert probe_ordering[1][0] == 'G' and probe_ordering[1][1] == 'L', (
            'probe_ordering[1] should be "GL", e.g. [LL, GL, GG]'
        )
    elif GL_OR_LG == 'LG':
        assert probe_ordering[1][0] == 'L' and probe_ordering[1][1] == 'G', (
            'probe_ordering[1] should be "LG", e.g. [LL, LG, GG]'
        )

    # get npairs
    npairs_auto, npairs_cross, npairs_3x2pt = get_zpairs(zbins)

    # construct the ind dict
    ind_dict = {}
    ind_dict['L', 'L'] = ind_copy[:npairs_auto, :]
    ind_dict['G', 'G'] = ind_copy[(npairs_auto + npairs_cross) :, :]
    if GL_OR_LG == 'LG':
        ind_dict['L', 'G'] = ind_copy[npairs_auto : (npairs_auto + npairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_OR_LG == 'GL':
        ind_dict['G', 'L'] = ind_copy[npairs_auto : (npairs_auto + npairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    # construct the npairs dict
    npairs_dict = {}
    npairs_dict['L', 'L'] = npairs_auto
    npairs_dict['L', 'G'] = npairs_cross
    npairs_dict['G', 'L'] = npairs_cross
    npairs_dict['G', 'G'] = npairs_auto

    # initialize the 4D dictionary and list of probe combinations
    cov_3x2pt_dict_4D = {}
    combinations = []

    # make each block 4D and store it with the right 'A', 'B', 'C, 'D' key
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            combinations.append([A, B, C, D])
            cov_3x2pt_dict_4D[A, B, C, D] = cov_6D_to_4D_blocks(
                cov_3x2pt_dict_10D[A, B, C, D],
                nbl,
                npairs_dict[A, B],
                npairs_dict[C, D],
                ind_dict[A, B],
                ind_dict[C, D],
            )

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = cov_3x2pt_8D_dict_to_4D(
        cov_3x2pt_dict_4D, probe_ordering, combinations
    )

    return cov_3x2pt_4D


def cov_3x2pt_8D_dict_to_4D(cov_3x2pt_8D_dict, probe_ordering, combinations=None):
    """Convert a dictionary of 4D blocks into a single 4D array.

    This is the same code as
    in the last part of the function above.
    :param cov_3x2pt_8D_dict: Dictionary of 4D covariance blocks
    :param probe_ordering: tuple of tuple probes,
    e.g., (('L', 'L'), ('G', 'L'), ('G', 'G'))
    :combinations: list of combinations to use,
    e.g., [['L', 'L', 'L', 'L'], ['L', 'L', 'L', 'G'], ...]
    :return: 4D covariance array
    """
    # if combinations is not provided, construct it
    if combinations is None:
        combinations = []
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                combinations.append([A, B, C, D])

    for key in cov_3x2pt_8D_dict:
        assert cov_3x2pt_8D_dict[key].ndim == 4, (
            f'covariance matrix {key} has ndim={cov_3x2pt_8D_dict[key].ndim} instead '
            f'of 4'
        )

    # check that the number of combinations is correct
    assert len(combinations) == len(list(cov_3x2pt_8D_dict.keys())), (
        f'number of combinations ({len(combinations)}) '
        'does not match the number of blocks in the input dictionary '
        f'({len(list(cov_3x2pt_8D_dict.keys()))})'
    )

    # check that the combinations are correct
    for combination in combinations:
        assert tuple(combination) in list(cov_3x2pt_8D_dict.keys()), (
            f'combination {combination} not found in the input dictionary'
        )

    # take the correct combinations (stored in 'combinations') and construct
    # lists which will be converted to arrays
    row_1_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[:3]]
    row_2_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[3:6]]
    row_3_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[6:9]]

    # concatenate the lists to make rows
    row_1 = np.concatenate(row_1_list, axis=3)
    row_2 = np.concatenate(row_2_list, axis=3)
    row_3 = np.concatenate(row_3_list, axis=3)

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_3x2pt_4D


def cov_3x2pt_4d_to_10d_dict(
    cov_3x2pt_4d, zbins, probe_ordering, nbl, ind_copy, optimize=False
):
    zpairs_auto, zpairs_cross, _ = get_zpairs(zbins)

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    ind_auto = ind_copy[:zpairs_auto, :]
    ind_cross = ind_copy[zpairs_auto : zpairs_cross + zpairs_auto, :]
    ind_dict = {('L', 'L'): ind_auto, ('G', 'L'): ind_cross, ('G', 'G'): ind_auto}

    assert probe_ordering == (('L', 'L'), ('G', 'L'), ('G', 'G')), (
        'more elaborate probe_ordering not implemented yet'
    )

    # slice the 4d cov to be able to use cov_4D_to_6D_blocks on the nine separate blocks
    zpairs_sum = zpairs_auto + zpairs_cross
    cov_3x2pt_8d_dict = {}
    cov_3x2pt_8d_dict['L', 'L', 'L', 'L'] = cov_3x2pt_4d[
        :, :, :zpairs_auto, :zpairs_auto
    ]
    cov_3x2pt_8d_dict['L', 'L', 'G', 'L'] = cov_3x2pt_4d[
        :, :, :zpairs_auto, zpairs_auto:zpairs_sum
    ]
    cov_3x2pt_8d_dict['L', 'L', 'G', 'G'] = cov_3x2pt_4d[
        :, :, :zpairs_auto, zpairs_sum:
    ]

    cov_3x2pt_8d_dict['G', 'L', 'L', 'L'] = cov_3x2pt_4d[
        :, :, zpairs_auto:zpairs_sum, :zpairs_auto
    ]
    cov_3x2pt_8d_dict['G', 'L', 'G', 'L'] = cov_3x2pt_4d[
        :, :, zpairs_auto:zpairs_sum, zpairs_auto:zpairs_sum
    ]
    cov_3x2pt_8d_dict['G', 'L', 'G', 'G'] = cov_3x2pt_4d[
        :, :, zpairs_auto:zpairs_sum, zpairs_sum:
    ]

    cov_3x2pt_8d_dict['G', 'G', 'L', 'L'] = cov_3x2pt_4d[
        :, :, zpairs_sum:, :zpairs_auto
    ]
    cov_3x2pt_8d_dict['G', 'G', 'G', 'L'] = cov_3x2pt_4d[
        :, :, zpairs_sum:, zpairs_auto:zpairs_sum
    ]
    cov_3x2pt_8d_dict['G', 'G', 'G', 'G'] = cov_3x2pt_4d[:, :, zpairs_sum:, zpairs_sum:]

    if optimize:
        # this version is only marginally faster, it seems
        cov_4D_to_6D_blocks_func = cov_4D_to_6D_blocks_opt
    else:
        # safer, default value
        cov_4D_to_6D_blocks_func = cov_4D_to_6D_blocks

    cov_3x2pt_10d_dict = {}
    for key, cov in cov_3x2pt_8d_dict.items():
        cov_3x2pt_10d_dict[key] = cov_4D_to_6D_blocks_func(
            cov,
            nbl,
            zbins,
            ind_dict[key[0], key[1]],
            ind_dict[key[2], key[3]],
            symmetrize_output_dict[key[0], key[1]],
            symmetrize_output_dict[key[2], key[3]],
        )

    return cov_3x2pt_10d_dict


# @njit
def cov_4D_to_6D(cov_4D, nbl, zbins, probe, ind):
    """Transform the cov from shape (nbl, nbl, npairs, npairs) to (nbl, nbl,
    zbins, zbins, zbins, zbins).

    Not valid for 3x2pt, the total shape of the matrix is (nbl, nbl,
    zbins, zbins, zbins, zbins), not big enough to store 3 probes. Use
    cov_4D functions or cov_6D as a dictionary instead,
    """
    # TODO deprecate this in favor of cov_4D_to_6D_blocks

    npairs_auto, npairs_cross, _ = get_zpairs(zbins)
    if probe in ['LL', 'GG']:
        npairs = npairs_auto
    elif probe in ['GL', 'LG']:
        npairs = npairs_cross
    else:
        raise ValueError(
            'probe must be "LL", "LG", "GL" or "GG". 3x2pt is not supported'
        )

    assert ind.shape[0] == npairs, (
        "ind.shape[0] != npairs: maybe you're passing the whole ind file "
        'instead of ind[:npairs, :] - or similia'
    )

    # TODO use jit
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ij in range(npairs):
        for kl in range(npairs):
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3]
            cov_6D[:, :, i, j, k, l] = cov_4D[:, :, ij, kl]

    # GL is not symmetric
    # ! this part makes this function very slow
    if probe in ['LL', 'GG']:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(
                            cov_6D[ell1, ell2, :, :, i, j]
                        )
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(
                            cov_6D[ell1, ell2, i, j, :, :]
                        )

    return cov_6D


# @njit
def cov_6D_to_4D(cov_6D, nbl, zpairs, ind):
    """Transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins) to
    (nbl, nbl, zpairs, zpairs)
    """
    assert ind.shape[0] == zpairs, (
        "ind.shape[0] != zpairs: maybe you're passing the whole ind file "
        'instead of ind[:zpairs, :] - or similia'
    )
    cov_4D = np.zeros((nbl, nbl, zpairs, zpairs))

    for ell2 in range(
        nbl
    ):  # added this loop in the latest version, before it was vectorized
        for ij in range(zpairs):
            for kl in range(zpairs):
                # rename for better readability
                i, j, k, l = ind[ij, -2], ind[ij, -1], ind[kl, -2], ind[kl, -1]
                cov_4D[:, ell2, ij, kl] = cov_6D[:, ell2, i, j, k, l]

    return cov_4D


# TODO finish this
def cov_6D_to_4D_optim(cov_6D, nbl, zpairs, ind):
    """Transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins) to
    (nbl, nbl, zpairs, zpairs)
    """
    assert ind.shape[0] == zpairs, (
        "ind.shape[0] != zpairs: maybe you're passing the whole ind file "
        'instead of ind[:zpairs, :] - or similia'
    )

    i_indices = ind[:, -2]
    j_indices = ind[:, -1]
    k_indices = ind[:, -2]
    l_indices = ind[:, -1]

    cov_4D = cov_6D[:, :, i_indices, j_indices, k_indices, l_indices]
    cov_4D = cov_4D.reshape((nbl, nbl, zpairs, zpairs))

    return cov_4D


# @njit
def cov_6D_to_4D_blocks(cov_6D, nbl, npairs_AB, npairs_CD, ind_AB, ind_CD):
    """Reshapes the covariance even for the non-diagonal (hence, non-square)
    blocks needed to build the 3x2pt.

    Use npairs_AB = npairs_CD and ind_AB = ind_CD for the normal routine
    (valid for auto-covariance LL-LL, GG-GG, GL-GL and LG-LG). n_columns
    is used to determine whether the ind array has 2 or 4 columns (if
    it's given in the form of a dictionary or not)
    """
    assert ind_AB.shape[0] == npairs_AB, 'ind_AB.shape[0] != npairs_AB'
    assert ind_CD.shape[0] == npairs_CD, 'ind_CD.shape[0] != npairs_CD'
    assert (cov_6D.shape[0] == cov_6D.shape[1] == nbl) or (cov_6D.shape[0] == nbl), (
        'number of angular bins does not match first two cov axes or the first axis'
    )

    # this is to ensure compatibility with both 4-columns and 2-columns ind arrays
    # (dictionary)the penultimante element is the first index, the last one the
    # second index (see s - 1, s - 2 below)
    # number of columns: this is to understand the format of the file
    n_columns_AB = ind_AB.shape[1]
    n_columns_CD = ind_CD.shape[1]

    # check
    assert n_columns_AB == n_columns_CD, (
        'ind_AB and ind_CD must have the same number of columns'
    )
    ncol = n_columns_AB  # make the name shorter

    if cov_6D.shape[0] == cov_6D.shape[1] == nbl:
        cov_out = np.zeros((nbl, nbl, npairs_AB, npairs_CD))
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for ij in range(npairs_AB):
                    for kl in range(npairs_CD):
                        zi, zj, zk, zl = (
                            ind_AB[ij, ncol - 2],
                            ind_AB[ij, ncol - 1],
                            ind_CD[kl, ncol - 2],
                            ind_CD[kl, ncol - 1],
                        )
                        cov_out[ell1, ell2, ij, kl] = cov_6D[ell1, ell2, zi, zj, zk, zl]

    elif cov_6D.shape[0] != cov_6D.shape[1]:
        cov_out = np.zeros((nbl, npairs_AB, npairs_CD))
        for ij in range(npairs_AB):
            for kl in range(npairs_CD):
                zi, zj, zk, zl = (
                    ind_AB[ij, ncol - 2],
                    ind_AB[ij, ncol - 1],
                    ind_CD[kl, ncol - 2],
                    ind_CD[kl, ncol - 1],
                )
                cov_out[:, ij, kl] = cov_6D[:, zi, zj, zk, zl]

    return cov_out


# @njit
def cov_4D_to_6D_blocks(
    cov_4D,
    nbl,
    zbins,
    ind_ab,
    ind_cd,
    symmetrize_output_ab: bool,
    symmetrize_output_cd: bool,
):
    """Reshapes the 4D covariance matrix to a 6D covariance matrix, even for
    the cross-probe (non-square) blocks needed to build the 3x2pt covariance.

    This function can be used for the normal routine (valid for auto-covariance,
    i.e., LL-LL, GG-GG, GL-GL and LG-LG)
    where `zpairs_ab = zpairs_cd` and `ind_ab = ind_cd`.

    Args:
        cov_4D (np.ndarray): The 4D covariance matrix.
        nbl (int): The number of ell bins.
        zbins (int): The number of redshift bins.
        ind_ab (np.ndarray): The indices for the first pair of redshift bins.
        ind_cd (np.ndarray): The indices for the second pair of redshift bins.
        symmetrize_output_ab (bool): Whether to symmetrize the output cov block
        for the first pair of probes.
        symmetrize_output_cd (bool): Whether to symmetrize the output cov block
        for the second pair of probes.

    Returns:
        np.ndarray: The 6D covariance matrix.

    """
    assert ind_ab.shape[1] == ind_cd.shape[1], (
        'ind_ab and ind_cd must have the same number of columns'
    )
    assert ind_ab.shape[1] == 2 or ind_ab.shape[1] == 4, (
        'ind_ab and ind_cd must have 2 or 4 columns'
    )
    ncols = ind_ab.shape[1]

    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell2 in range(nbl):
        for ij in range(zpairs_ab):
            for kl in range(zpairs_cd):
                i, j, k, l = (
                    ind_ab[ij, ncols - 2],
                    ind_ab[ij, ncols - 1],
                    ind_cd[kl, ncols - 2],
                    ind_cd[kl, ncols - 1],
                )
                cov_6D[:, ell2, i, j, k, l] = cov_4D[:, ell2, ij, kl]

    # GL blocks are not symmetric
    # ! this part makes this function quite slow
    if symmetrize_output_ab:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(
                            cov_6D[ell1, ell2, :, :, i, j]
                        )

    if symmetrize_output_cd:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(
                            cov_6D[ell1, ell2, i, j, :, :]
                        )

    return cov_6D


def cov_4D_to_6D_blocks_opt(
    cov_4D, nbl, zbins, ind_ab, ind_cd, symmetrize_output_ab, symmetrize_output_cd
):
    assert ind_ab.shape[1] == ind_cd.shape[1], (
        'ind_ab and ind_cd must have the same number of columns'
    )
    assert ind_ab.shape[1] in {2, 4}, 'ind_ab and ind_cd must have 2 or 4 columns'

    ncols = ind_ab.shape[1]
    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))

    ell2_indices, ij_indices, kl_indices = np.ogrid[:nbl, :zpairs_ab, :zpairs_cd]
    i_indices = ind_ab[ij_indices, ncols - 2]
    j_indices = ind_ab[ij_indices, ncols - 1]
    k_indices = ind_cd[kl_indices, ncols - 2]
    l_indices = ind_cd[kl_indices, ncols - 1]

    cov_6D[:, ell2_indices, i_indices, j_indices, k_indices, l_indices] = cov_4D[
        :, ell2_indices, ij_indices, kl_indices
    ]

    if symmetrize_output_ab or symmetrize_output_cd:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                if symmetrize_output_ab:
                    for i in range(zbins):
                        for j in range(zbins):
                            cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(
                                cov_6D[ell1, ell2, :, :, i, j]
                            )
                if symmetrize_output_cd:
                    for i in range(zbins):
                        for j in range(zbins):
                            cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(
                                cov_6D[ell1, ell2, i, j, :, :]
                            )

    return cov_6D


def return_combinations(A, B, C, D):
    print(
        f'C_{A}{C}, C_{B}{D}, C_{A}{D}, C_{B}{C}, N_{A}{C}, N_{B}{D}, N_{A}{D}, N_{B}{C}'
    )


###########################
# @njit
def check_symmetric(array_2d, exact, rtol=1e-05):
    """:param a: 2d array
    :param exact: bool
    :param rtol: relative tolerance
    :return: bool, whether the array is symmetric or not
    """
    # """check if the matrix is symmetric, either exactly or within a tolerance
    # """
    assert isinstance(exact, bool), 'parameter "exact" must be either True or False'
    assert array_2d.ndim == 2, 'the array is not square'
    if exact:
        return np.array_equal(array_2d, array_2d.T)
    else:
        return np.allclose(array_2d, array_2d.T, rtol=rtol, atol=0)


def slice_cov_3x2pt_2D_ell_probe_zpair(cov_2D_ell_probe_zpair, nbl, zbins, probe):
    """Slices the 2-dimensional 3x2pt covariance ordered as a block-diagonal
    matrix in ell, probe and zpair (unpacked in this order)
    """
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)
    ell_block_size = zpairs_3x2pt

    if probe == 'WL':
        probe_start = 0
        probe_stop = zpairs_auto
    elif probe == 'GC':
        probe_start = zpairs_auto + zpairs_cross
        probe_stop = zpairs_3x2pt
    elif probe == '2x2pt':
        probe_start = zpairs_auto
        probe_stop = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, GC or 2x2pt')

    cov_1D_ell_probe_zpair = [0] * nbl
    for ell_bin in range(nbl):
        # block_index * block_size + probe_starting index in each block
        start = ell_bin * ell_block_size + probe_start
        stop = start + probe_stop
        cov_1D_ell_probe_zpair[ell_bin] = cov_2D_ell_probe_zpair[start:stop, start:stop]

    cov_2D_ell_probe_zpair_sliced = scipy.linalg.block_diag(*cov_1D_ell_probe_zpair)

    return cov_2D_ell_probe_zpair_sliced


def slice_cl_3x2pt_1D_ell_probe_zpair(cl_3x2pt_1D_ell_probe_zpair, nbl, zbins, probe):
    """Slices the 2-dimensional 3x2pt covariance ordered as a block-diagonal
    matrix in ell, probe and zpair (unpacked in this order)
    """
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)
    ell_block_size = zpairs_3x2pt

    if probe == 'WL':
        probe_start = 0
        probe_stop = zpairs_auto
    elif probe == 'GC':
        probe_start = zpairs_auto + zpairs_cross
        probe_stop = zpairs_3x2pt
    elif probe == '2x2pt':
        probe_start = zpairs_auto
        probe_stop = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, GC or 2x2pt')

    cl_1D_ell_probe_zpair_list = [0] * nbl
    for ell_bin in range(nbl):
        # block_index * block_size + probe_starting index in each block
        start = ell_bin * ell_block_size + probe_start
        stop = start + probe_stop
        cl_1D_ell_probe_zpair_list[ell_bin] = cl_3x2pt_1D_ell_probe_zpair[start:stop]

    cl_1D_ell_probe_zpair = np.array(list(itertools.chain(*cl_1D_ell_probe_zpair_list)))

    return cl_1D_ell_probe_zpair


# @njit
def cov_2D_to_4D(cov_2D, nbl, block_index, optimize=True, symmetrize=False):
    """Reshapes the covariance from 2D to 4D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use zpair or ell as the outermost index (determined by the
    ordering of the for loops)
      This is going to be the index of the blocks in the 2D covariance matrix.

    Note: this reshaping does not change the number of elements, we just go
    from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in"
    the missing elements in the
    higher-dimensional array are needed.
    """
    assert block_index in ['ell', 'C-style'] + ['zpair', 'F-style'], (
        'block_index must be "ell", "C-style" or "zpair", "F-style"'
    )
    assert cov_2D.ndim == 2, 'the input covariance must be 2-dimensional'

    zpairs_AB = cov_2D.shape[0] // nbl
    zpairs_CD = cov_2D.shape[1] // nbl

    cov_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))

    if optimize:
        if block_index in ['ell', 'C-style']:
            cov_4D = cov_2D.reshape((nbl, zpairs_AB, nbl, zpairs_CD)).transpose(
                (0, 2, 1, 3)
            )
        elif block_index in ['ij', 'sylvain', 'F-style']:
            cov_4D = cov_2D.reshape((zpairs_AB, nbl, zpairs_CD, nbl)).transpose(
                (1, 3, 0, 2)
            )

    elif block_index in ['ell', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[
                            l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair
                        ]

    elif block_index in ['zpair', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[
                            ipair * nbl + l1, jpair * nbl + l2
                        ]

    # mirror the upper triangle into the lower one
    if symmetrize:
        for l1 in range(nbl):
            for l2 in range(nbl):
                cov_4D[l1, l2, :, :] = symmetrize_2d_array(cov_4D[l1, l2, :, :])

    return cov_4D


# @njit
def cov_4D_to_2D(cov_4D, block_index, optimize=True):
    """Reshapes the covariance from 4D to 2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of
    the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.

    this function can also convert to 2D non-square blocks; this is needed to build
    the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a
    zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function
     (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix
     (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from
    [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in"
    the missing elements in the
    higher-dimensional array are needed.
    """
    assert block_index in ['ell', 'C-style'] + ['zpair', 'F-style'], (
        'block_index must be "ell", "C-style" or "zpair", "F-style"'
    )

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], (
        'the first two axes of the input covariance must have the same size'
    )
    # assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    nbl = int(cov_4D.shape[0])
    zpairs_AB = int(cov_4D.shape[2])
    zpairs_CD = int(cov_4D.shape[3])

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if optimize:
        if block_index in ['ell', 'C-style']:
            cov_2D.reshape(nbl, zpairs_AB, nbl, zpairs_CD)[:, :, :, :] = (
                cov_4D.transpose(0, 2, 1, 3)
            )

        elif block_index in ['zpair', 'F-style']:
            cov_2D.reshape(zpairs_AB, nbl, zpairs_CD, nbl)[:, :, :, :] = (
                cov_4D.transpose(2, 0, 3, 1)
            )
        return cov_2D

    # I tested that the 2 methods give the same results.
    # This code is kept to remember the
    # block_index * block_size + running_index unpacking
    if block_index in ['ell', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[
                            l1, l2, ipair, jpair
                        ]

    elif block_index in ['zpair', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[
                            l1, l2, ipair, jpair
                        ]

    return cov_2D


# @njit
def cov_4D_to_2DCLOE_3x2pt(cov_4D, zbins, block_index='ell'):
    """Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE
    ordering.

    Note that this is only necessary for the 3x2pt probe. TODO the probe
    ordering (LL, LG/GL, GG) is hardcoded, this function won't work with
    other combinations (but it TODO will work both for LG and GL) !
    Important note: block_index = 'ell' means that the overall ordering
    will ! be probe_ell_zpair. ! Setting it to 'zpair' will give you the
    ordering probe_zpair_ell. ! Bottom line: the probe is the outermost
    loop in any case. ! The ordering used by CLOE v2 is probe_ell_zpair,
    so block_index = 'ell' is ! the correct choice in this case.
    """
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    lim_1 = zpairs_auto
    lim_2 = zpairs_cross + zpairs_auto
    lim_3 = zpairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], block_index, optimize=True)
    cov_LL_LG = cov_4D_to_2D(
        cov_4D[:, :, :lim_1, lim_1:lim_2], block_index, optimize=True
    )
    cov_LL_GG = cov_4D_to_2D(
        cov_4D[:, :, :lim_1, lim_2:lim_3], block_index, optimize=True
    )

    cov_LG_LL = cov_4D_to_2D(
        cov_4D[:, :, lim_1:lim_2, :lim_1], block_index, optimize=True
    )
    cov_LG_LG = cov_4D_to_2D(
        cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], block_index, optimize=True
    )
    cov_LG_GG = cov_4D_to_2D(
        cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], block_index, optimize=True
    )

    cov_GG_LL = cov_4D_to_2D(
        cov_4D[:, :, lim_2:lim_3, :lim_1], block_index, optimize=True
    )
    cov_GG_LG = cov_4D_to_2D(
        cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], block_index, optimize=True
    )
    cov_GG_GG = cov_4D_to_2D(
        cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], block_index, optimize=True
    )

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


# @njit
def cov_2DCLOE_to_4D_3x2pt(cov_2D, nbl, zbins, block_index='ell'):
    """Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE
    ordering.

    Note that this is only necessary for the 3x2pt probe. TODO the probe
    ordering (LL, LG/GL, GG) is hardcoded, this function won't work with
    other combinations (but it TODO will work both for LG and GL)
    """
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    # now I'm reshaping the full block diagonal matrix, not just the sub-blocks
    # (cov_2D_to_4D works for both cases)
    lim_1 = zpairs_auto * nbl
    lim_2 = (zpairs_cross + zpairs_auto) * nbl
    lim_3 = zpairs_3x2pt * nbl

    # note: I'm writing cov_LG, but there should be no issue with GL; after all,
    # this function is not using the ind file
    cov_LL_LL = cov_2D_to_4D(cov_2D[:lim_1, :lim_1], nbl, block_index)
    cov_LL_LG = cov_2D_to_4D(cov_2D[:lim_1, lim_1:lim_2], nbl, block_index)
    cov_LL_GG = cov_2D_to_4D(cov_2D[:lim_1, lim_2:lim_3], nbl, block_index)

    cov_LG_LL = cov_2D_to_4D(cov_2D[lim_1:lim_2, :lim_1], nbl, block_index)
    cov_LG_LG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_1:lim_2], nbl, block_index)
    cov_LG_GG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_2:lim_3], nbl, block_index)

    cov_GG_LL = cov_2D_to_4D(cov_2D[lim_2:lim_3, :lim_1], nbl, block_index)
    cov_GG_LG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_1:lim_2], nbl, block_index)
    cov_GG_GG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_2:lim_3], nbl, block_index)

    # here it is a little more difficult to visualize the stacking,
    # but the probes are concatenated
    # along the 2 zpair_3x2pt-long axes
    cov_4D = np.zeros((nbl, nbl, zpairs_3x2pt, zpairs_3x2pt))

    zlim_1 = zpairs_auto
    zlim_2 = zpairs_cross + zpairs_auto
    zlim_3 = zpairs_3x2pt

    cov_4D[:, :, :zlim_1, :zlim_1] = cov_LL_LL
    cov_4D[:, :, :zlim_1, zlim_1:zlim_2] = cov_LL_LG
    cov_4D[:, :, :zlim_1, zlim_2:zlim_3] = cov_LL_GG

    cov_4D[:, :, zlim_1:zlim_2, :zlim_1] = cov_LG_LL
    cov_4D[:, :, zlim_1:zlim_2, zlim_1:zlim_2] = cov_LG_LG
    cov_4D[:, :, zlim_1:zlim_2, zlim_2:zlim_3] = cov_LG_GG

    cov_4D[:, :, zlim_2:zlim_3, :zlim_1] = cov_GG_LL
    cov_4D[:, :, zlim_2:zlim_3, zlim_1:zlim_2] = cov_GG_LG
    cov_4D[:, :, zlim_2:zlim_3, zlim_2:zlim_3] = cov_GG_GG

    return cov_4D


def cov_2d_dav_to_cloe(cov_2d_dav, nbl, zbins, block_index_in, block_index_out):
    """Convert a 2D covariance matrix from the davide convention to the CLOE
    convention, that is, from the probe being unraveled in the first for loop
    to the probe being unraveled in the second for loop.

    example: from ell_probe_zpair (my convention) to probe_ell_zpair (CLOE).
    The zpairs <-> ell ordering is decided by 'block_idex' (setting the first,
    or outermost, of the two)
    """
    cov_4D = cov_2D_to_4D(cov_2d_dav, nbl, block_index=block_index_in, optimize=True)
    cov_2d_cloe = cov_4D_to_2DCLOE_3x2pt(
        cov_4D, zbins=zbins, block_index=block_index_out
    )
    return cov_2d_cloe


def cov_2d_cloe_to_dav(cov_2d_cloe, nbl, zbins, block_index_in, block_index_out):
    """Convert a 2D covariance matrix from the CLOE convention to the davide
    convention, that is, from the probe being unraveled in the second for loop
    to the probe being unraveled in the first for loop.

    example: from probe_ell_zpair (CLOE) to ell_probe_zpair (my convention).
    The zpairs <-> ell ordering is decided by 'block_idex' (setting the first,
    or outermost, of the two)
    """
    cov_4D = cov_2DCLOE_to_4D_3x2pt(cov_2d_cloe, nbl, zbins, block_index=block_index_in)
    cov_2d_dav = cov_4D_to_2D(cov_4D, block_index=block_index_out, optimize=True)
    return cov_2d_dav


def cov2corr(covariance):
    """Convert a covariance matrix to a correlation matrix."""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)

    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = np.divide(covariance, outer_v)
        correlation[covariance == 0] = (
            0  # Ensure zero covariance entries are explicitly zero
        )
        correlation[~np.isfinite(correlation)] = 0  # Set any NaN or inf values to 0

    # Ensure diagonal elements are exactly 1
    # np.fill_diagonal(correlation, 1)

    return correlation


def build_noise(
    zbins: int,
    n_probes: int,
    sigma_eps2: list | tuple | np.ndarray,
    ng_shear: list | tuple | np.ndarray,
    ng_clust: list | tuple | np.ndarray,
    is_noiseless: bool = False,
) -> np.ndarray:
    """Builds the noise power spectra.

    Parameters
    ----------
    zbins : int
        Number of redshift bins.
    n_probes : int
        Number of probes.
    sigma_eps2 : list | tuple | np.ndarray
        Square of the *total* ellipticity dispersion.
        sigma_eps2 = sigma_eps ** 2, with
        sigma_eps = sigma_eps_i * sqrt(2),
        sigma_eps_i being the ellipticity dispersion *per component*
    ng_shear : list | tuple | np.ndarray
        Galaxy density of sources, relevant for cosmic shear
        If a scalar, cumulative galaxy density number density, per arcmin^2.
        This will assume equipopulated bins.
        If an array, galaxy number density, per arcmin^2, per redshift bin.
        Must have length zbins.
    ng_clust : list | tuple | np.ndarray
        Galaxy density of lenses, relevant for galaxy clustering
        If a scalar, cumulative galaxy density number density, per arcmin^2.
        This will assume equipopulated bins.
        If an array, galaxy number density, per arcmin^2, per redshift bin.
        Must have length zbins.
    is_noiseless : bool, optional
        If True, returns array of zeros of the right shape.

    Returns
    -------
    noise_4d : np.ndarray
        Noise power spectra matrices of shape (n_probes, n_probes, zbins, zbins)

    Notes
    -----
    The noise N is defined as:
        N_LL = sigma_eps^2 / (2 * n_bar)
        N_GG = 1 / n_bar
        N_GL = N_LG = 0

    """
    # assert appropriate inputs are list, tuple or np.ndarray
    for var, name in zip(
        [
            ng_shear,
            ng_clust,
        ],
        [
            'ng_shear',
            'ng_clust',
        ],
    ):
        #     [ng_shear, ng_clust, sigma_eps2],
        #     ['ng_shear', 'ng_clust', 'sigma_eps2'],
        # ):
        assert isinstance(var, (list, tuple, np.ndarray)), (
            f'{name} should be a list, tuple or np.ndarray'
        )

    # convert to np arrays if needed
    if isinstance(ng_shear, (list, tuple)):
        ng_shear = np.array(ng_shear)
    if isinstance(ng_clust, (list, tuple)):
        ng_clust = np.array(ng_clust)
    # if isinstance(ng_clust, (list, tuple)):
    # sigma_eps2 = np.array(sigma_eps2)

    conversion_factor = (180 / np.pi * 60) ** 2  # deg^2 to arcmin^2

    assert np.all(ng_shear > 0), 'ng_shear should be positive'
    assert np.all(ng_clust > 0), 'ng_clust should be positive'

    # if ng is an array, n_bar == ng (this is a slight misnomer, since ng is the
    # cumulative galaxy density, while
    # n_bar the galaxy density in each bin). In this case, if the bins are
    # quipopulated, the n_bar array should
    # have all entries almost identical.

    n_bar_shear = ng_shear * conversion_factor
    n_bar_clust = ng_clust * conversion_factor

    # create and fill N
    noise_4d = np.zeros((n_probes, n_probes, zbins, zbins))

    if is_noiseless:
        return noise_4d

    np.fill_diagonal(noise_4d[0, 0, :, :], sigma_eps2 / (2 * n_bar_shear))
    np.fill_diagonal(noise_4d[1, 1, :, :], 1 / n_bar_clust)
    noise_4d[0, 1, :, :] = 0
    noise_4d[1, 0, :, :] = 0

    return noise_4d
