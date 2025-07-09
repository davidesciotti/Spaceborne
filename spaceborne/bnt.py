import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps

from spaceborne import cosmo_lib as csmlib


def compute_bnt_matrix(zbins, zgrid_n_of_z, n_of_z_arr, cosmo_ccl, plot_nz=True):
    """Computes the BNT matrix. This function has been slightly modified from
    Santiago Casas' implementation in CLOE.

    Args:
        zbins (int): Number of redshift bins.
        zgrid_n_of_z (numpy.ndarray): Grid of redshift values for the n(z) distribution.
        n_of_z_arr (numpy.ndarray): Array of n(z) distributions, with shape
        (len(zgrid_n_of_z), zbins).
        cosmo_ccl (ccl.Cosmology): Cosmology object from the CCL library.
        plot_nz (bool, optional): Whether to plot the n(z) distributions.

    Returns:
        numpy.ndarray: BNT matrix of shape (zbins, zbins).

    """
    assert n_of_z_arr.shape[0] == len(zgrid_n_of_z), (
        'n_of_z must have zgrid_n_of_z rows'
    )
    assert n_of_z_arr.shape[1] == zbins, 'n_of_z must have zbins columns'
    assert np.all(np.diff(zgrid_n_of_z) > 0), (
        'zgrid_n_of_z must be monotonically increasing'
    )

    z_grid = zgrid_n_of_z

    if z_grid[0] == 0:
        warnings.warn(
            'z_grid starts at 0, which gives a null comoving distance. '
            'Removing the first element from the grid',
            stacklevel=2,
        )
        z_grid = z_grid[1:]
        n_of_z_arr = n_of_z_arr[1:, :]

    chi = csmlib.ccl_comoving_distance(z_grid, use_h_units=False, cosmo_ccl=cosmo_ccl)

    if plot_nz:
        plt.figure()
        for zi in range(zbins):
            plt.plot(z_grid, n_of_z_arr[:, zi], label=f'zbin {zi}')
        plt.title('n(z) used for BNT computation')
        plt.grid()
        plt.legend()

    A_list = np.zeros(zbins)
    B_list = np.zeros(zbins)
    for zbin_idx in range(zbins):
        n_of_z = n_of_z_arr[:, zbin_idx]
        A_list[zbin_idx] = simps(y=n_of_z, x=z_grid)
        B_list[zbin_idx] = simps(y=n_of_z / chi, x=z_grid)

    bnt_matrix = np.eye(zbins)
    bnt_matrix[1, 0] = -1.0
    for i in range(2, zbins):
        mat = np.array([[A_list[i - 1], A_list[i - 2]], [B_list[i - 1], B_list[i - 2]]])
        A = -1.0 * np.array([A_list[i], B_list[i]])
        soln = np.dot(np.linalg.inv(mat), A)
        bnt_matrix[i, i - 1] = soln[0]
        bnt_matrix[i, i - 2] = soln[1]

    return bnt_matrix


def cl_bnt_transform(cl_3d, bnt_matrix, probe_A, probe_B):
    assert cl_3d.ndim == 3, 'cl_3d must be 3D'
    assert bnt_matrix.ndim == 2, 'bnt_matrix must be 2D'
    assert cl_3d.shape[1] == bnt_matrix.shape[0], (
        'the number of ell bins in cl_3d and bnt_matrix must be the same'
    )

    bnt_transform_dict = {
        'L': bnt_matrix,
        'G': np.eye(bnt_matrix.shape[0]),
    }

    cl_bnt_3d = np.zeros(cl_3d.shape)
    for ell_idx in range(cl_3d.shape[0]):
        cl_bnt_3d[ell_idx, :, :] = (
            bnt_transform_dict[probe_A]
            @ cl_3d[ell_idx, :, :]
            @ bnt_transform_dict[probe_B].T
        )

    return cl_bnt_3d


def cl_bnt_transform_3x2pt(cl_3x2pt_5d, bnt_matrix):
    """Wrapper function to quickly implement the cl (or derivatives) BNT transform
    for the 3x2pt datavector
    """
    cl_3x2pt_bnt_5d = np.zeros(cl_3x2pt_5d.shape)
    cl_3x2pt_bnt_5d[0, 0, :, :, :] = cl_bnt_transform(
        cl_3x2pt_5d[0, 0, :, :, :], bnt_matrix, 'L', 'L'
    )
    cl_3x2pt_bnt_5d[0, 1, :, :, :] = cl_bnt_transform(
        cl_3x2pt_5d[0, 1, :, :, :], bnt_matrix, 'L', 'G'
    )
    cl_3x2pt_bnt_5d[1, 0, :, :, :] = cl_bnt_transform(
        cl_3x2pt_5d[1, 0, :, :, :], bnt_matrix, 'G', 'L'
    )
    cl_3x2pt_bnt_5d[1, 1, :, :, :] = cl_3x2pt_5d[
        1, 1, :, :, :
    ]  # no need to transform the GG part

    return cl_3x2pt_bnt_5d


def get_ell_cuts_indices(ell_values, ell_cuts_2d_array, zbins):
    """Creates an array of lists containing the ell indices to cut (to set to 0)
    for each zi, zj)
    """
    ell_idxs_tocut = np.zeros((zbins, zbins), dtype=list)
    for zi in range(zbins):
        for zj in range(zbins):
            ell_cut = ell_cuts_2d_array[zi, zj]
            if np.any(ell_values > ell_cut):  # i.e., if you need to do a cut at all
                ell_idxs_tocut[zi, zj] = np.where(ell_values > ell_cut)[0]
            else:
                ell_idxs_tocut[zi, zj] = np.array([])

    return ell_idxs_tocut


def build_x_matrix_bnt(bnt_matrix):
    """Builds the X matrix for the BNT transform, according to eq.
    :param bnt_matrix:
    :return:
    """
    X = {}
    delta_kron = np.eye(bnt_matrix.shape[0])
    X['L', 'L'] = np.einsum('ae, bf -> aebf', bnt_matrix, bnt_matrix)
    X['G', 'G'] = np.einsum('ae, bf -> aebf', delta_kron, delta_kron)
    X['G', 'L'] = np.einsum('ae, bf -> aebf', delta_kron, bnt_matrix)
    X['L', 'G'] = np.einsum('ae, bf -> aebf', bnt_matrix, delta_kron)
    return X


def cov_bnt_transform(
    cov_nobnt_6D, X_dict, probe_A, probe_B, probe_C, probe_D, optimize=True
):
    """Same as above, but only for one probe (i.e., LL or GL: GG is not modified
    by the BNT)
    """
    cov_bnt_6D = np.einsum(
        'aebf, cgdh, LMefgh -> LMabcd',
        X_dict[probe_A, probe_B],
        X_dict[probe_C, probe_D],
        cov_nobnt_6D,
        optimize=optimize,
    )
    return cov_bnt_6D


def cov_3x2pt_bnt_transform(cov_3x2pt_dict_10D, X_dict, optimize=True):
    """In np.einsum below, L and M are the ell1, ell2 indices, which are not
    touched by the BNT transform
    """
    cov_3x2pt_bnt_dict_10D = {}

    for probe_A, probe_B, probe_C, probe_D in cov_3x2pt_dict_10D:
        cov_3x2pt_bnt_dict_10D[probe_A, probe_B, probe_C, probe_D] = cov_bnt_transform(
            cov_3x2pt_dict_10D[probe_A, probe_B, probe_C, probe_D],
            X_dict,
            probe_A,
            probe_B,
            probe_C,
            probe_D,
            optimize=optimize,
        )

    return cov_3x2pt_bnt_dict_10D
