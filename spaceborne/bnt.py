import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps

from spaceborne import cosmo_lib as csmlib
from spaceborne import sb_lib as sl


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

    bnt_transform_dict = {'L': bnt_matrix, 'G': np.eye(bnt_matrix.shape[0])}

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
    X['LL'] = np.einsum('ae, bf -> aebf', bnt_matrix, bnt_matrix)
    X['GG'] = np.einsum('ae, bf -> aebf', delta_kron, delta_kron)
    X['GL'] = np.einsum('ae, bf -> aebf', delta_kron, bnt_matrix)
    X['LG'] = np.einsum('ae, bf -> aebf', bnt_matrix, delta_kron)
    return X


def cov_bnt_transform(
    cov_nobnt_6d: np.ndarray, X_dict: dict, probe_ab: str, probe_cd: str, optimize=True
):
    """Same as above, but only for one probe (i.e., LL or GL: GG is not modified
    by the BNT)
    """
    assert cov_nobnt_6d.ndim == 6, 'The input covariance should have 6 dimensions'

    cov_bnt_6d = np.einsum(
        'aebf, cgdh, LMefgh -> LMabcd',
        X_dict[probe_ab],
        X_dict[probe_cd],
        cov_nobnt_6d,
        optimize=optimize,
    )
    return cov_bnt_6d


def bnt_transform_whole_cov_dict(
    cov_dict: dict, bnt_matrix: np.ndarray, req_probe_combs_2d: list
) -> dict:
    """Wrapper function to apply the BNT transform to all the probes and terms in 
    the cov_dict.
    
    Note: cov_dict is modified in-place, so a return is not strictly needed, but I find 
    this to be a bit clearer
    """
    # BNT-transform 6D covs (for all terms and probe combinations)
    # TODO BNT and scale cuts of G term should go in the gauss cov function!
    # ! BNT IS LINEAR, SO BNT(COV_TOT) = \SUM_i BNT(COV_i), but should check
    x_dict = build_x_matrix_bnt(bnt_matrix)
    for term in cov_dict:
        for probe_abcd in req_probe_combs_2d:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'harmonic')
            cov_dict[term][probe_ab, probe_cd]['6d'] = cov_bnt_transform(
                cov_dict[term][probe_ab, probe_cd]['6d'],
                x_dict,
                probe_ab,
                probe_cd,
                optimize=True,
            )
    return cov_dict
