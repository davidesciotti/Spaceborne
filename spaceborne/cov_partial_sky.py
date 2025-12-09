import itertools
import os
import time
import warnings
from typing import TypedDict

import healpy as hp
import numpy as np
import pymaster as nmt
from tqdm import tqdm

from spaceborne import constants, ell_utils, mask_utils
from spaceborne import sb_lib as sl
from spaceborne import cov_dict as cd
from spaceborne import constants as const

_UNSET = object()

DEG2_IN_SPHERE = constants.DEG2_IN_SPHERE
DR1_DATE = constants.DR1_DATE


# construct a TypedDcit to allow static type checkers to check packed **kwargs
class Bin2DArrayKwargs(TypedDict):
    ells_in: np.ndarray
    ells_out: np.ndarray
    ells_out_edges: np.ndarray
    weights_in: np.ndarray | None
    which_binning: str
    interpolate: bool


def couple_cov_6d(
    mcm_ab: np.ndarray, cov_abcd_6d: np.ndarray, mcm_cd: np.ndarray
) -> np.ndarray:
    if mcm_ab.shape[1] != cov_abcd_6d.shape[0]:
        raise ValueError('mcm_ab and cov_abcd_6d have incompatible dimensions')
    if mcm_cd.shape[0] != cov_abcd_6d.shape[1]:
        raise ValueError('mcm_cd and cov_abcd_6d have incompatible dimensions')

    cov_abcd_6d_coupled = np.einsum(
        'XW, WZijkl, ZY -> XYijkl', mcm_ab, cov_abcd_6d, mcm_cd
    )

    return cov_abcd_6d_coupled


def bin_mcm(mbm_unbinned: np.ndarray, nmt_bin_obj) -> np.ndarray:
    """Simple function to bin the mode coupling matrix. Adapted from
    https://namaster.readthedocs.io/en/latest/1BasicFunctionality.html
    """
    mcm_binned_rows = np.array([nmt_bin_obj.bin_cell(row) for row in mbm_unbinned])

    # nmt_bin_obj.get_nell_list() gives the list of delta_ell,
    # aka how many multipoles in each ell bin
    mcm_binned = (
        np.array([nmt_bin_obj.bin_cell(col) for col in mcm_binned_rows.T]).T
        * nmt_bin_obj.get_nell_list()
    )

    return mcm_binned


def nmt_gaussian_cov(
    cov_dict: dict,
    cl_tt: np.ndarray,
    cl_te: np.ndarray,
    cl_ee: np.ndarray,
    cl_tb: np.ndarray,
    cl_eb: np.ndarray,
    cl_bb: np.ndarray,
    zbins: int,
    nbl: int,
    cw,
    w00,
    w02,
    w22,
    unique_probe_combs: list[str],
    *,
    coupled: bool = False,
    ells_in: np.ndarray,
    ells_out: np.ndarray,
    ells_out_edges: np.ndarray,
    which_binning: str,
    weights: np.ndarray | None,
):
    """Unified function to compute Gaussian covariance using NaMaster.

    # NOTE: the order of the arguments (in particular for the cls) is the following
    # spin_a1, spin_a2, spin_b1, spin_b2,
    # cla1b1, cla1b2, cla2b1, cla2b2
    # The order of the output dimensions depends on the order of the input list:
    # [cl_te, cl_tb] - > TE=0, TB=1
    # covar_TT_TE = covar_00_02[:, 0, :, 0]x
    # covar_TT_TB = covar_00_02[:, 0, :, 1]

    Parameters
    ----------
    - cl_tt, cl_te, cl_ee, cl_tb, cl_eb, cl_bb: Input power spectra.
    - zbins: Number of redshift bins.
    - nbl: Number of bandpower bins.
    - cw: Covariance workspace.
    - w00, w02, w22: Workspaces for different spin combinations.
    - coupled: Whether to compute coupled or decoupled covariance.
    - ells_in, ells_out, ells_out_edges: Binning parameters for coupled covariance.
    - which_binning: Binning method for coupled covariance.
    - weights: Weights for binning.

    """
    cl_et = cl_te.transpose(0, 2, 1)
    cl_bt = cl_tb.transpose(0, 2, 1)
    cl_be = cl_eb.transpose(0, 2, 1)

    for cl in [cl_tt, cl_te, cl_ee, cl_tb, cl_eb, cl_bb]:
        assert cl.shape[0] == cl_tt.shape[0], (
            'input cls have different number of ell bins'
        )

    nell = cl_tt.shape[0] if coupled else nbl

    def cl_00_list(zi, zj):
        return [cl_tt[:, zi, zj]]

    def cl_02_list(zi, zj):
        return [cl_te[:, zi, zj], cl_tb[:, zi, zj]]

    def cl_20_list(zi, zj):
        return [cl_et[:, zi, zj], cl_bt[:, zi, zj]]

    def cl_22_list(zi, zj):
        return [cl_ee[:, zi, zj], cl_eb[:, zi, zj], cl_be[:, zi, zj], cl_bb[:, zi, zj]]

    z_combinations = list(itertools.product(range(zbins), repeat=4))
    for zi, zj, zk, zl in tqdm(z_combinations):
        if 'GGGG' in unique_probe_combs:
            covar_00_00 = nmt.gaussian_covariance(cw, 
                                                0, 0, 0, 0,
                                                cl_00_list(zi, zk),
                                                cl_00_list(zi, zl),
                                                cl_00_list(zj, zk),
                                                cl_00_list(zj, zl),
                                                coupled=coupled,
                                                wa=w00, wb=w00).reshape([nell, 1, nell, 1])  # fmt: skip
            covar_TT_TT = covar_00_00[:, 0, :, 0]
        else:
            covar_TT_TT = np.zeros((nell, nell))

        if 'GGGL' in unique_probe_combs or 'GLGG' in unique_probe_combs:
            covar_00_02 = nmt.gaussian_covariance(cw, 
                                                0, 0, 0, 2,
                                                cl_00_list(zi, zk),
                                                cl_02_list(zi, zl),
                                                cl_00_list(zj, zk),
                                                cl_02_list(zj, zl),
                                                coupled=coupled,
                                                wa=w00, wb=w02).reshape([nell, 1, nell, 2])  # fmt: skip
            covar_TT_TE = covar_00_02[:, 0, :, 0]
            covar_TT_TB = covar_00_02[:, 0, :, 1]
        else:
            covar_TT_TE = np.zeros((nell, nell))
            covar_TT_TB = np.zeros((nell, nell))

        if 'GGLL' in unique_probe_combs or 'LLGG' in unique_probe_combs:
            covar_00_22 = nmt.gaussian_covariance(cw, 
                                                0, 0, 2, 2,
                                                cl_02_list(zi, zk),
                                                cl_02_list(zi, zl),
                                                cl_02_list(zj, zk),
                                                cl_02_list(zj, zl),
                                                coupled=coupled,
                                                wa=w00, wb=w22).reshape([nell, 1, nell, 4])  # fmt: skip
            covar_TT_EE = covar_00_22[:, 0, :, 0]
            covar_TT_EB = covar_00_22[:, 0, :, 1]
            covar_TT_BE = covar_00_22[:, 0, :, 2]
            covar_TT_BB = covar_00_22[:, 0, :, 3]
        else:
            covar_TT_EE = np.zeros((nell, nell))
            covar_TT_EB = np.zeros((nell, nell))
            covar_TT_BE = np.zeros((nell, nell))
            covar_TT_BB = np.zeros((nell, nell))

        if 'GLGL' in unique_probe_combs:
            covar_02_02 = nmt.gaussian_covariance(cw, 
                                                0, 2, 0, 2,
                                                cl_00_list(zi, zk),
                                                cl_02_list(zi, zl),
                                                cl_20_list(zj, zk),
                                                cl_22_list(zj, zl),
                                                coupled=coupled,
                                                wa=w02, wb=w02).reshape([nell, 2, nell, 2])  # fmt: skip
            covar_TE_TE = covar_02_02[:, 0, :, 0]
            covar_TE_TB = covar_02_02[:, 0, :, 1]
            covar_TB_TE = covar_02_02[:, 1, :, 0]
            covar_TB_TB = covar_02_02[:, 1, :, 1]
        else:
            covar_TE_TE = np.zeros((nell, nell))
            covar_TE_TB = np.zeros((nell, nell))
            covar_TB_TE = np.zeros((nell, nell))
            covar_TB_TB = np.zeros((nell, nell))

        if 'GLLL' in unique_probe_combs or 'LLGL' in unique_probe_combs:
            covar_02_22 = nmt.gaussian_covariance(cw, 
                                                0, 2, 2, 2,
                                                cl_02_list(zi, zk),
                                                cl_02_list(zi, zl),
                                                cl_22_list(zj, zk),
                                                cl_22_list(zj, zl),
                                                coupled=coupled,
                                                wa=w02, wb=w22).reshape([nell, 2, nell, 4])  # fmt: skip
            covar_TE_EE = covar_02_22[:, 0, :, 0]
            covar_TE_EB = covar_02_22[:, 0, :, 1]
            covar_TE_BE = covar_02_22[:, 0, :, 2]
            covar_TE_BB = covar_02_22[:, 0, :, 3]
            covar_TB_EE = covar_02_22[:, 1, :, 0]
            covar_TB_EB = covar_02_22[:, 1, :, 1]
            covar_TB_BE = covar_02_22[:, 1, :, 2]
            covar_TB_BB = covar_02_22[:, 1, :, 3]
        else:
            covar_TE_EE = np.zeros((nell, nell))
            covar_TE_EB = np.zeros((nell, nell))
            covar_TE_BE = np.zeros((nell, nell))
            covar_TE_BB = np.zeros((nell, nell))
            covar_TB_EE = np.zeros((nell, nell))
            covar_TB_EB = np.zeros((nell, nell))
            covar_TB_BE = np.zeros((nell, nell))
            covar_TB_BB = np.zeros((nell, nell))

        if 'LLLL' in unique_probe_combs:
            covar_22_22 = nmt.gaussian_covariance(cw, 
                                                2, 2, 2, 2,
                                                cl_22_list(zi, zk),
                                                cl_22_list(zi, zl),
                                                cl_22_list(zj, zk),
                                                cl_22_list(zj, zl),
                                                coupled=coupled,
                                                wa=w22, wb=w22).reshape([nell, 4, nell, 4])  # fmt: skip

            covar_EE_EE = covar_22_22[:, 0, :, 0]
            covar_EE_EB = covar_22_22[:, 0, :, 1]
            covar_EE_BE = covar_22_22[:, 0, :, 2]
            covar_EE_BB = covar_22_22[:, 0, :, 3]
            covar_EB_EE = covar_22_22[:, 1, :, 0]
            covar_EB_EB = covar_22_22[:, 1, :, 1]
            covar_EB_BE = covar_22_22[:, 1, :, 2]
            covar_EB_BB = covar_22_22[:, 1, :, 3]
            covar_BE_EE = covar_22_22[:, 2, :, 0]
            covar_BE_EB = covar_22_22[:, 2, :, 1]
            covar_BE_BE = covar_22_22[:, 2, :, 2]
            covar_BE_BB = covar_22_22[:, 2, :, 3]
            covar_BB_EE = covar_22_22[:, 3, :, 0]
            covar_BB_EB = covar_22_22[:, 3, :, 1]
            covar_BB_BE = covar_22_22[:, 3, :, 2]
            covar_BB_BB = covar_22_22[:, 3, :, 3]
        else:
            covar_EE_EE = np.zeros((nell, nell))
            covar_EE_EB = np.zeros((nell, nell))
            covar_EE_BE = np.zeros((nell, nell))
            covar_EE_BB = np.zeros((nell, nell))
            covar_EB_EE = np.zeros((nell, nell))
            covar_EB_EB = np.zeros((nell, nell))
            covar_EB_BE = np.zeros((nell, nell))
            covar_EB_BB = np.zeros((nell, nell))
            covar_BE_EE = np.zeros((nell, nell))
            covar_BE_EB = np.zeros((nell, nell))
            covar_BE_BE = np.zeros((nell, nell))
            covar_BE_BB = np.zeros((nell, nell))
            covar_BB_EE = np.zeros((nell, nell))
            covar_BB_EB = np.zeros((nell, nell))
            covar_BB_BE = np.zeros((nell, nell))
            covar_BB_BB = np.zeros((nell, nell))

        common_kw: Bin2DArrayKwargs = {
            'ells_in': ells_in,
            'ells_out': ells_out,
            'ells_out_edges': ells_out_edges,
            'weights_in': weights,
            'which_binning': which_binning,
            'interpolate': True,
        }

        # in the coupled case, namaster returns unbinned covariance matrices
        if coupled:
            covar_EE_EE = sl.bin_2d_array_vectorized(covar_EE_EE, **common_kw)
            covar_TE_EE = sl.bin_2d_array_vectorized(covar_TE_EE, **common_kw)
            covar_TE_TE = sl.bin_2d_array_vectorized(covar_TE_TE, **common_kw)
            covar_TT_EE = sl.bin_2d_array_vectorized(covar_TT_EE, **common_kw)
            covar_TT_TE = sl.bin_2d_array_vectorized(covar_TT_TE, **common_kw)
            covar_TT_TT = sl.bin_2d_array_vectorized(covar_TT_TT, **common_kw)

        cov_dict['g']['LL', 'LL']['6d'][:, :, zi, zj, zk, zl] = covar_EE_EE
        cov_dict['g']['GL', 'LL']['6d'][:, :, zi, zj, zk, zl] = covar_TE_EE
        cov_dict['g']['GL', 'GL']['6d'][:, :, zi, zj, zk, zl] = covar_TE_TE
        cov_dict['g']['GG', 'LL']['6d'][:, :, zi, zj, zk, zl] = covar_TT_EE
        cov_dict['g']['GG', 'GL']['6d'][:, :, zi, zj, zk, zl] = covar_TT_TE
        cov_dict['g']['GG', 'GG']['6d'][:, :, zi, zj, zk, zl] = covar_TT_TT
        # the remaining blocks can be filled in by symmetry (with zi, zj <-> zk, zl)
        cov_dict['g']['LL', 'GL']['6d'][:, :, zk, zl, zi, zj] = covar_TE_EE.T
        cov_dict['g']['LL', 'GG']['6d'][:, :, zk, zl, zi, zj] = covar_TT_EE.T
        cov_dict['g']['GL', 'GG']['6d'][:, :, zk, zl, zi, zj] = covar_TT_TE.T

    return cov_dict


def nmt_gaussian_cov_spin0(
    cov_dict: dict,
    cl_tt: np.ndarray,
    cl_te: np.ndarray,
    cl_ee: np.ndarray,
    zbins: int,
    nbl: int,
    cw,
    w00,
    unique_probe_combs: list[str],
    coupled: bool,
    ells_in: np.ndarray,
    ells_out: np.ndarray,
    ells_out_edges: np.ndarray,
    which_binning: str,
    weights: np.ndarray | None,
):
    cl_et = cl_te.transpose(0, 2, 1)

    nell = cl_tt.shape[0] if coupled else nbl

    z_combinations = list(itertools.product(range(zbins), repeat=4))
    for zi, zj, zk, zl in tqdm(z_combinations):
        if 'GGGG' in unique_probe_combs:
            covar_00_00 = nmt.gaussian_covariance(cw,  
                                                0, 0, 0, 0, 
                                                [cl_tt[:, zi, zk]],  # TT
                                                [cl_tt[:, zi, zl]],  # TT
                                                [cl_tt[:, zj, zk]],  # TT
                                                [cl_tt[:, zj, zl]],  # TT
                                                coupled=coupled,
                                                wa=w00, wb=w00)  # fmt: skip
            covar_TT_TT = covar_00_00
        else:
            covar_TT_TT = np.zeros((nell, nell))

        # here and below, for the nondiag blocks, I need to include the symmetric
        # probe combinations (by default, here in nmt, I compute the lower triangle
        # of the 3x2pt probe combination matrix, rather then the upper one as done in
        # the rest of the code)
        if 'GGGL' in unique_probe_combs or 'GLGG' in unique_probe_combs:
            covar_00_02 = nmt.gaussian_covariance(cw,  
                                                0, 0, 0, 0, 
                                                [cl_tt[:, zi, zk]],  # TT
                                                [cl_te[:, zi, zl]],  # TE, TB
                                                [cl_tt[:, zj, zk]],  # TT
                                                [cl_te[:, zj, zl]],  # TE, TB
                                                coupled=coupled,
                                                wa=w00, wb=w00)  # fmt: skip
            covar_TT_TE = covar_00_02
        else:
            covar_TT_TE = np.zeros((nell, nell))

        if 'GLGL' in unique_probe_combs:
            covar_02_02 = nmt.gaussian_covariance(cw,  
                                                0, 0, 0, 0, 
                                                [cl_tt[:, zi, zk]],  # TT
                                                [cl_te[:, zi, zl]],  # TE, TB
                                                [cl_et[:, zj, zk]],  # ET, BT
                                                [cl_ee[:, zj, zl]],  # EE, EB, BE, BB
                                                coupled=coupled,
                                                wa=w00, wb=w00)  # fmt: skip
            covar_TE_TE = covar_02_02
        else:
            covar_TE_TE = np.zeros((nell, nell))

        if 'GGLL' in unique_probe_combs or 'LLGG' in unique_probe_combs:
            covar_00_22 = nmt.gaussian_covariance(cw,  
                                                0, 0, 0, 0, 
                                                [cl_te[:, zi, zk]],  # TE, TB
                                                [cl_te[:, zi, zl]],  # TE, TB
                                                [cl_te[:, zj, zk]],  # TE, TB
                                                [cl_te[:, zj, zl]],  # TE, TB
                                                coupled=coupled,
                                                wa=w00, wb=w00)  # fmt: skip
            covar_TT_EE = covar_00_22
        else:
            covar_TT_EE = np.zeros((nell, nell))

        if 'GLLL' in unique_probe_combs or 'LLGL' in unique_probe_combs:
            covar_02_22 = nmt.gaussian_covariance(cw,  
                                                0, 0, 0, 0, 
                                                [cl_te[:, zi, zk]],  # TE, TB
                                                [cl_te[:, zi, zl]],  # TE, TB
                                                [cl_ee[:, zj, zk]],
                                                [cl_ee[:, zj, zl]],
                                                coupled=coupled,
                                                wa=w00, wb=w00)  # fmt: skip
            covar_TE_EE = covar_02_22
        else:
            covar_TE_EE = np.zeros((nell, nell))

        if 'LLLL' in unique_probe_combs:
            covar_22_22 = nmt.gaussian_covariance(cw,  
                                                0, 0, 0, 0, 
                                                [cl_ee[:, zi, zk]],
                                                [cl_ee[:, zi, zl]],
                                                [cl_ee[:, zj, zk]],
                                                [cl_ee[:, zj, zl]],
                                                coupled=coupled,
                                                wa=w00, wb=w00)  # fmt: skip
            covar_EE_EE = covar_22_22
        else:
            covar_EE_EE = np.zeros((nell, nell))

        common_kw: Bin2DArrayKwargs = {
            'ells_in': ells_in,
            'ells_out': ells_out,
            'ells_out_edges': ells_out_edges,
            'weights_in': weights,
            'which_binning': which_binning,
            'interpolate': True,
        }

        if coupled:
            covar_EE_EE = sl.bin_2d_array_vectorized(covar_EE_EE, **common_kw)
            covar_TE_EE = sl.bin_2d_array_vectorized(covar_TE_EE, **common_kw)
            covar_TE_TE = sl.bin_2d_array_vectorized(covar_TE_TE, **common_kw)
            covar_TT_EE = sl.bin_2d_array_vectorized(covar_TT_EE, **common_kw)
            covar_TT_TE = sl.bin_2d_array_vectorized(covar_TT_TE, **common_kw)
            covar_TT_TT = sl.bin_2d_array_vectorized(covar_TT_TT, **common_kw)

        cov_dict['g']['LL', 'LL']['6d'][:, :, zi, zj, zk, zl] = covar_EE_EE
        cov_dict['g']['GL', 'LL']['6d'][:, :, zi, zj, zk, zl] = covar_TE_EE
        cov_dict['g']['GL', 'GL']['6d'][:, :, zi, zj, zk, zl] = covar_TE_TE
        cov_dict['g']['GG', 'LL']['6d'][:, :, zi, zj, zk, zl] = covar_TT_EE
        cov_dict['g']['GG', 'GL']['6d'][:, :, zi, zj, zk, zl] = covar_TT_TE
        cov_dict['g']['GG', 'GG']['6d'][:, :, zi, zj, zk, zl] = covar_TT_TT
        # the remaining blocks can be filled in by symmetry (with zi, zj <-> zk, zl)
        cov_dict['g']['LL', 'GL']['6d'][:, :, zk, zl, zi, zj] = covar_TE_EE.T
        cov_dict['g']['LL', 'GG']['6d'][:, :, zk, zl, zi, zj] = covar_TT_EE.T
        cov_dict['g']['GL', 'GG']['6d'][:, :, zk, zl, zi, zj] = covar_TT_TE.T

    return cov_dict


def linear_lmin_binning(NSIDE, lmin, bw):
    """Generate a linear binning scheme based on a minimum multipole 'lmin' and
    bin width 'bw'.

    Parameters
    ----------
    NSIDE : int
        The NSIDE parameter of the HEALPix grid.

    lmin : int
        The minimum multipole to start the binning.

    bw : int
        The bin width, i.e., the number of multipoles in each bin.

    Returns
    -------
    nmt_bins
        A binning scheme object defining linearly spaced bins starting from 'lmin' with
        a width of 'bw' multipoles.

    Notes
    -----
    This function generates a binning scheme for the pseudo-Cl power spectrum estimation
    using the Namaster library. It divides the multipole range from 'lmin' to 2*NSIDE
    into bins of width 'bw'.

    Example:
    --------
    # Generate a linear binning scheme for an NSIDE of 64, starting from l=10, with bin width of 20
    bin_scheme = linear_lmin_binning(NSIDE=64, lmin=10, bw=20)

    """
    lmax = 2 * NSIDE
    nbl = (lmax - lmin) // bw + 1
    elli = np.zeros(nbl, int)
    elle = np.zeros(nbl, int)

    for i in range(nbl):
        elli[i] = lmin + i * bw
        elle[i] = lmin + (i + 1) * bw

    b = nmt.NmtBin.from_edges(elli, elle)
    return b


def coupling_matrix(bin_scheme, mask, wkspce_name):
    """Compute the mixing matrix for coupling spherical harmonic modes using
    the provided binning scheme and mask.

    Parameters
    ----------
    bin_scheme : nmt_bins
        A binning scheme object defining the bins for the coupling matrix.

    mask : nmt_field
        A mask object defining the regions of the sky to include in the computation.

    wkspce_name : str
        The file name for storing or retrieving the computed workspace containing
        the coupling matrix.

    Returns
    -------
    nmt_workspace
        A workspace object containing the computed coupling matrix.

    Notes
    -----
    This function computes the coupling matrix necessary for the pseudo-Cl power
    spectrum estimation using the NmtField and NmtWorkspace objects from the
    Namaster library.

    If the workspace file specified by 'wkspce_name' exists, the function reads
    the coupling matrix from the file. Otherwise, it computes the matrix and
    writes it to the file.

    Example:
    --------
    # Generate a linear binning scheme for an NSIDE of 64, starting from l=10, with bin width of 20
    bin_scheme = linear_lmin_binning(NSIDE=64, lmin=10, bw=20)

    # Define the mask
    mask = nmt.NmtField(mask, [mask])

    # Compute the coupling matrix and store it in 'coupling_matrix.bin'
    coupling_matrix = coupling_matrix(bin_scheme, mask, 'coupling_matrix.bin')

    """
    print('Compute the mixing matrix')
    start = time.time()
    fmask = nmt.NmtField(mask, [mask])  # nmt field with only the mask
    w = nmt.NmtWorkspace()
    if os.path.isfile(wkspce_name):
        print(
            'Mixing matrix has already been calculated and is in the workspace file : ',
            f'{wkspce_name}. Read it.',
        )
        w.read_from(wkspce_name)
    else:
        print(
            f'The file : {wkspce_name}',
            ' does not exists. Calculating the mixing matrix and writing it.',
        )
        w.compute_coupling_matrix(fmask, fmask, bin_scheme)
        w.write_to(wkspce_name)
    print('Done computing the mixing matrix. It took ', time.time() - start, 's.')
    return w


def sample_covariance( # fmt: skip
    cl_GG_unbinned, cl_LL_unbinned, cl_GL_unbinned,
    cl_BB_unbinned, cl_EB_unbinned, cl_TB_unbinned,
    nbl, zbins, mask, nside, nreal, coupled_cls, which_cls, nmt_bin_obj,
    w00, w02, w22, lmax=None, n_probes=2, fix_seed=True, n_iter=None, lite=True,
):  # fmt: skip
    if lmax is None:
        lmax = 3 * nside - 1

    if fix_seed:
        SEEDVALUE = np.arange(nreal)

    # NmtField kwargs
    nmt_field_kw = {'n_iter': n_iter, 'lite': lite, 'lmax': lmax}

    # TODO use only independent z pairs
    # TODO update this too to the new dict structure
    cov_sim_10d = np.zeros(
        (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)
    )
    sim_cl_GG = np.zeros((nreal, nbl, zbins, zbins))
    sim_cl_GL = np.zeros((nreal, nbl, zbins, zbins))
    sim_cl_LL = np.zeros((nreal, nbl, zbins, zbins))

    # 1. produce correlated maps
    print(
        f'Generating {nreal} maps for nside {nside} '
        f'and computing pseudo-cls with {which_cls}...'
    )

    cl_ring_big_list = build_cl_tomo_TEB_ring_ord(
        cl_TT=cl_GG_unbinned,
        cl_EE=cl_LL_unbinned,
        cl_BB=cl_BB_unbinned,
        cl_TE=cl_GL_unbinned,
        cl_EB=cl_EB_unbinned,
        cl_TB=cl_TB_unbinned,
        zbins=zbins,
        spectra_types=['T', 'E', 'B'],
    )

    zij_combinations = list(itertools.product(range(zbins), repeat=2))
    zijkl_combinations = list(itertools.product(range(zbins), repeat=4))

    for i in tqdm(range(nreal)):
        if fix_seed:
            np.random.seed(SEEDVALUE[i])

        # * 1. produce correlated alms
        corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=lmax, new=True)
        assert len(corr_alms_tot) == zbins * 3, 'wrong number of alms'

        # extract alm for TT, EE, BB
        corr_alms = corr_alms_tot[::3]
        corr_Elms_Blms = list(zip(corr_alms_tot[1::3], corr_alms_tot[2::3]))

        # compute correlated maps
        corr_maps_gg = [hp.alm2map(alm, nside, lmax=lmax) for alm in corr_alms]
        corr_maps_ll = [
            hp.alm2map_spin(alms=[Elm, Blm], nside=nside, spin=2, lmax=lmax)
            for (Elm, Blm) in corr_Elms_Blms
        ]

        if which_cls == 'namaster':
            f0 = np.array(
                [nmt.NmtField(mask, [map_T], **nmt_field_kw) for map_T in corr_maps_gg]
            )
            f2 = np.array(
                [
                    nmt.NmtField(mask, [map_Q, map_U], **nmt_field_kw)
                    for (map_Q, map_U) in corr_maps_ll
                ]
            )
        else:
            f0, f2 = None, None

        # * 2. compute and bin simulated cls for all zbin combinations,
        # * using input correlated maps
        for zi, zj in zij_combinations:
            sim_cl_GG_ij, sim_cl_GL_ij, sim_cl_LL_ij = pcls_from_maps(
                corr_maps_gg=corr_maps_gg,
                corr_maps_ll=corr_maps_ll,
                zi=zi,
                zj=zj,
                f0=f0,
                f2=f2,
                mask=mask,
                coupled_cls=coupled_cls,
                which_cls=which_cls,
                w00=w00,
                w02=w02,
                w22=w22,
                lmax_eff=lmax,  # TODO is this the correct lmax?
            )

            assert sim_cl_GG_ij.shape == sim_cl_GL_ij.shape == sim_cl_LL_ij.shape, (
                'Simulated cls must have the same shape'
            )

            if len(sim_cl_GG_ij) != nbl:
                sim_cl_GG[i, :, zi, zj] = nmt_bin_obj.bin_cell(sim_cl_GG_ij)
                sim_cl_GL[i, :, zi, zj] = nmt_bin_obj.bin_cell(sim_cl_GL_ij)
                sim_cl_LL[i, :, zi, zj] = nmt_bin_obj.bin_cell(sim_cl_LL_ij)
            else:
                sim_cl_GG[i, :, zi, zj] = sim_cl_GG_ij
                sim_cl_GL[i, :, zi, zj] = sim_cl_GL_ij
                sim_cl_LL[i, :, zi, zj] = sim_cl_LL_ij

    # * 3. compute sample covariance
    for zi, zj, zk, zl in tqdm(zijkl_combinations):
        # ! compute the sample covariance
        # you could also cut the mixed cov terms, but for cross-redshifts it becomes a bit tricky
        kwargs = {'rowvar': False, 'bias': False}
        cov_sim_10d[0, 0, 0, 0, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_LL[:, :, zi, zj], sim_cl_LL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[0, 0, 1, 0, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_LL[:, :, zi, zj], sim_cl_GL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[0, 0, 1, 1, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_LL[:, :, zi, zj], sim_cl_GG[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[1, 0, 0, 0, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GL[:, :, zi, zj], sim_cl_LL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[1, 0, 1, 0, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GL[:, :, zi, zj], sim_cl_GL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[1, 0, 1, 1, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GL[:, :, zi, zj], sim_cl_GG[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[1, 1, 0, 0, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GG[:, :, zi, zj], sim_cl_LL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[1, 1, 1, 0, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GG[:, :, zi, zj], sim_cl_GL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_sim_10d[1, 1, 1, 1, :, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GG[:, :, zi, zj], sim_cl_GG[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]

    return cov_sim_10d, sim_cl_GG, sim_cl_GL, sim_cl_LL


def pcls_from_maps(  # fmt: skip
    corr_maps_gg, corr_maps_ll, zi, zj, f0, f2, mask, coupled_cls, which_cls,
    w00, w02, w22, lmax_eff,
):  # fmt: skip
    """Note: both healpy anafast and nmt.compute_coupled_cell return the coupled
    ("pseudo") cls. Dividing by fsky gives a rough approximation of the true Cls.

    """
    if which_cls == 'namaster':
        # pseudo-Cls. Becomes an ok estimator for the true Cls if divided by fsky
        pcl_tt = nmt.compute_coupled_cell(f0[zi], f0[zj])
        pcl_te = nmt.compute_coupled_cell(f0[zi], f2[zj])
        pcl_ee = nmt.compute_coupled_cell(f2[zi], f2[zj])

        if coupled_cls:  # return pseudo-cls
            cl_tt_out = pcl_tt[0]
            cl_te_out = pcl_te[0]
            cl_ee_out = pcl_ee[0]
        else:  # return decoupled cls. This is the MASTER estimator
            cl_tt_out = w00.decouple_cell(pcl_tt)[0, :]
            cl_te_out = w02.decouple_cell(pcl_te)[0, :]
            cl_ee_out = w22.decouple_cell(pcl_ee)[0, :]

    elif which_cls == 'healpy':
        _corr_maps_zi = list(itertools.chain([corr_maps_gg[zi]], corr_maps_ll[zi]))
        _corr_maps_zj = list(itertools.chain([corr_maps_gg[zj]], corr_maps_ll[zj]))

        # 2. remove monopole
        _corr_maps_zi = [
            hp.remove_monopole(_corr_maps_zi[spec_ix]) for spec_ix in range(3)
        ]
        _corr_maps_zj = [
            hp.remove_monopole(_corr_maps_zj[spec_ix]) for spec_ix in range(3)
        ]

        # 3. compute cls for each bin
        hp_pcl_tot = hp.anafast(
            map1=[
                _corr_maps_zi[0] * mask,
                _corr_maps_zi[1] * mask,
                _corr_maps_zi[2] * mask,
            ],
            map2=[
                _corr_maps_zj[0] * mask,
                _corr_maps_zj[1] * mask,
                _corr_maps_zj[2] * mask,
            ],
            lmax=lmax_eff,
        )

        # output is TT, EE, BB, TE, EB, TB
        # hp_pcl_GG[:, zi, zj] = hp_pcl_tot[0, :]
        # hp_pcl_LL[:, zi, zj] = hp_pcl_tot[1, :]
        # hp_pcl_GL[:, zi, zj] = hp_pcl_tot[3, :]

        # pseudo-Cls. Becomes an ok estimator for the true Cls if divided by fsky
        pcl_tt = hp_pcl_tot[0, :]
        pcl_ee = hp_pcl_tot[1, :]
        pcl_bb = hp_pcl_tot[2, :]
        pcl_te = hp_pcl_tot[3, :]
        pcl_eb = hp_pcl_tot[4, :]
        pcl_tb = hp_pcl_tot[5, :]
        pcl_be = pcl_eb  # ! warning!!

        # return pseudo-cls from healpy
        if coupled_cls:
            cl_tt_out = pcl_tt
            cl_te_out = pcl_te
            cl_ee_out = pcl_ee
        # return decoupled cls, again via namaster (the MASTER estimator uses the
        # same deconvolution!)
        else:
            stack_te = np.vstack((pcl_te, pcl_tb))
            stack_ee = np.vstack((pcl_ee, pcl_eb, pcl_be, pcl_bb))
            cl_tt_out = w00.decouple_cell(pcl_tt[None, :])[0, :]
            cl_te_out = w02.decouple_cell(stack_te)[0, :]
            cl_ee_out = w22.decouple_cell(stack_ee)[0, :]

    else:
        raise ValueError('which_cls must be namaster or healpy')

    return np.array(cl_tt_out), np.array(cl_te_out), np.array(cl_ee_out)


def build_cl_ring_ordering(cl_3d):
    zbins = cl_3d.shape[1]
    assert cl_3d.shape[1] == cl_3d.shape[2], (
        'input cls should have shape (ell_bins, zbins, zbins)'
    )
    cl_ring_list = []

    for offset in range(zbins):  # offset defines the distance from the main diagonal
        for zi in range(zbins - offset):
            zj = zi + offset
            cl_ring_list.append(cl_3d[:, zi, zj])

    return cl_ring_list


def build_cl_tomo_TEB_ring_ord(
    cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB, zbins, spectra_types=('T', 'E', 'B')
):
    assert (
        cl_TT.shape
        == cl_EE.shape
        == cl_BB.shape
        == cl_TE.shape
        == cl_EB.shape
        == cl_TB.shape
    ), 'All input arrays must have the same shape.'
    assert cl_TT.ndim == 3, 'the ell axis should be present for all input arrays'

    # Iterate over redshift bins and spectra types to construct the
    # matrix of combinations
    row_idx = 0
    matrix = []
    for zi in range(zbins):
        for s1 in spectra_types:
            row = []
            for zj in range(zbins):
                for s2 in spectra_types:
                    row.append(f'{s1}-{zi}-{s2}-{zj}')
            matrix.append(row)
            row_idx += 1

    assert len(row) == zbins * len(spectra_types), (
        'The number of elements in the row should be equal to the number of redshift bins times the number of spectra types.'
    )

    cl_ring_ord_list = []
    for offset in range(len(row)):
        for zi in range(len(row) - offset):
            zj = zi + offset

            probe_a, _zi, probe_b, zj = matrix[zi][zj].split('-')

            if probe_a == 'T' and probe_b == 'T':
                cl = cl_TT
            elif probe_a == 'E' and probe_b == 'E':
                cl = cl_EE
            elif probe_a == 'B' and probe_b == 'B':
                cl = cl_BB
            elif probe_a == 'T' and probe_b == 'E':
                cl = cl_TE
            elif probe_a == 'E' and probe_b == 'B':
                cl = cl_EB
            elif probe_a == 'T' and probe_b == 'B':
                cl = cl_TB
            elif probe_a == 'B' and probe_b == 'T':
                cl = cl_TB.transpose(0, 2, 1)
            elif probe_a == 'B' and probe_b == 'E':
                cl = cl_EB.transpose(0, 2, 1)
            elif probe_a == 'E' and probe_b == 'T':
                cl = cl_TE.transpose(0, 2, 1)
            else:
                raise ValueError(f'Invalid combination: {probe_a}-{probe_b}')

            cl_ring_ord_list.append(cl[:, int(_zi), int(zj)])

    return cl_ring_ord_list


def get_sample_field_bu(cl_TT, cl_EE, cl_BB, cl_TE, nside, mask):
    """This routine generates a spin-0 and a spin-2 Gaussian random field based
    on these power spectra.
    From https://namaster.readthedocs.io/en/latest/source/sample_covariance.html
    """
    map_t, map_q, map_u = hp.synfast([cl_TT, cl_EE, cl_BB, cl_TE], nside)
    return nmt.NmtField(mask, [map_t], lite=False), nmt.NmtField(
        mask, [map_q, map_u], lite=False
    )


def cls_to_maps(cl_TT, cl_EE, cl_BB, cl_TE, nside, lmax=None):
    """This routine generates maps for spin-0 and a spin-2 Gaussian random field based
    on the input power spectra.

    Args:
        cl_TT (numpy.ndarray): Temperature power spectrum.
        cl_EE (numpy.ndarray): E-mode polarization power spectrum.
        cl_BB (numpy.ndarray): B-mode polarization power spectrum.
        cl_TE (numpy.ndarray): Temperature-E-mode cross power spectrum.
        nside (int): HEALPix resolution parameter.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Temperature map, Q-mode polarization map, U-mode polarization map.

    """
    if lmax is None:
        # note: this seems to be causing issues for EE when lmax_eff is significantly
        # lower than 3 * nside - 1
        lmax = 3 * nside - 1

    alm, Elm, Blm = hp.synalm(
        cls=[cl_TT, cl_EE, cl_BB, cl_TE, 0 * cl_TE, 0 * cl_TE], lmax=lmax, new=True
    )
    map_Q, map_U = hp.alm2map_spin(alms=[Elm, Blm], nside=nside, spin=2, lmax=lmax)
    map_T = hp.alm2map(alms=alm, nside=nside, lmax=lmax)
    return map_T, map_Q, map_U


def masked_maps_to_nmtFields(map_T, map_Q, map_U, mask, lmax, n_iter=None, lite=True):
    """Create NmtField objects from masked maps.

    Args:
        map_T (numpy.ndarray): Temperature map.
        map_Q (numpy.ndarray): Q-mode polarization map.
        map_U (numpy.ndarray): U-mode polarization map.
        mask (numpy.ndarray): Mask to apply to the maps.

    Returns:
        nmt.NmtField, nmt.NmtField: NmtField objects for the temperature and polarization maps.

    """
    f0 = nmt.NmtField(mask, [map_T], n_iter=n_iter, lite=lite, lmax=lmax)
    f2 = nmt.NmtField(mask, [map_Q, map_U], spin=2, n_iter=n_iter, lite=lite, lmax=lmax)
    return f0, f2


def compute_master(f_a, f_b, wsp):
    """This function computes power spectra given a pair of fields and a workspace.
    From https://namaster.readthedocs.io/en/latest/source/sample_covariance.html
    NOTE THAT nmt.compute_full_master() does:
    NmtWorkspace.compute_coupling_matrix
    deprojection_bias
    compute_coupled_cell
    NmtWorkspace.decouple_cell
    and gives perfectly consistent results!
    """
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def produce_correlated_maps(
    cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB, nreal, nside, zbins, lmax
):
    print(f'Generating {nreal} maps for nside {nside}...')

    cl_ring_big_list = build_cl_tomo_TEB_ring_ord(
        cl_TT=cl_TT,
        cl_EE=cl_EE,
        cl_BB=cl_BB,
        cl_TE=cl_TE,
        cl_EB=cl_EB,
        cl_TB=cl_TB,
        zbins=zbins,
        spectra_types=['T', 'E', 'B'],
    )

    corr_maps_gg_list = []
    corr_maps_ll_list = []

    for _ in tqdm(range(nreal)):
        corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=lmax, new=True)
        assert len(corr_alms_tot) == zbins * 3, 'wrong number of alms'

        # extract alm for TT, EE, BB
        corr_alms = corr_alms_tot[::3]
        corr_Elms_Blms = list(zip(corr_alms_tot[1::3], corr_alms_tot[2::3]))

        # compute correlated maps for each bin
        corr_maps_gg = [hp.alm2map(alm, nside, lmax) for alm in corr_alms]
        corr_maps_ll = [
            hp.alm2map_spin([Elm, Blm], nside, 2, lmax) for (Elm, Blm) in corr_Elms_Blms
        ]

        corr_maps_gg_list.append(corr_maps_gg)
        corr_maps_ll_list.append(corr_maps_ll)

    return corr_maps_gg_list, corr_maps_ll_list


class NmtCov:
    def __init__(
        self,
        cfg: dict,
        pvt_cfg: dict,
        ell_obj: ell_utils.EllBinning,
        mask_obj: mask_utils.Mask,
    ):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg

        self.ell_obj = ell_obj
        self.mask_obj = mask_obj

        self.zbins = pvt_cfg['zbins']
        self.n_probes = pvt_cfg['n_probes']
        self.coupled_cov = cfg['covariance']['coupled_cov']
        self.output_path = self.cfg['misc']['output_path']

        # instantiate cov dict
        # ! note that this class only computes
        #   - g term
        #   - g all hs probe combinations (no 3x2pt!!)
        #   - 6d dim

        self.req_terms = ['g']
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_hs_2d']
        dims = ['6d']

        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='harmonic')
            for probe in const.HS_ALL_PROBE_COMBS
        ]
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

        # check on lmax and NSIDE
        for probe in ('WL', 'GC'):
            _lmax = getattr(self.ell_obj, f'ell_max_{probe}')
            if _lmax >= 3 * self.mask_obj.nside - 1:
                warnings.warn(
                    f'lmax = {_lmax} >= 3 * NSIDE = {self.mask_obj.nside} - 1'
                    'you should probably increase NSIDE or decrease lmax ',
                    stacklevel=2,
                )

        self.cl_ll_unb_3d = _UNSET
        self.cl_gl_unb_3d = _UNSET
        self.cl_gg_unb_3d = _UNSET
        self.ells_3x2pt_unb = _UNSET
        self.nbl_3x2pt_unb = _UNSET

    def build_psky_cov(self):
        # TODO again, here I'm using 3x2pt = GC
        # 1. ell binning
        # shorten names for brevity
        nmt_bin_obj = self.ell_obj.nmt_bin_obj_GC
        fsky = self.mask_obj.fsky
        nmt_cfg = self.cfg['namaster']
        unique_probe_combs = self.pvt_cfg['unique_probe_combs']

        ells_eff = self.ell_obj.ells_3x2pt
        nbl_eff = self.ell_obj.nbl_3x2pt
        ells_eff_edges = self.ell_obj.ell_edges_3x2pt
        _ell_min_eff = self.ell_obj.ell_min_3x2pt
        ell_max_eff = self.ell_obj.ell_max_3x2pt

        # notice that bin_obj.get_ell_list(nbl_eff) is out of bounds
        # ells_eff_edges = np.array([b.get_ell_list(i)[0] for i in range(nbl_eff)])
        # ells_eff_edges = np.append(
        #     ells_eff_edges, b.get_ell_list(nbl_eff - 1)[-1] + 1
        # )  # careful f the +1!
        # ell_min_eff = ells_eff_edges[0]

        ells_unb = np.arange(ell_max_eff + 1)
        nbl_unb = len(ells_unb)
        assert nbl_unb == ell_max_eff + 1, 'nbl_tot does not match lmax_eff + 1'

        # ells_bpw = ells_unb[ell_min_eff : lmax_eff + 1]
        # delta_ells_bpw = np.diff(
        # np.array([b.get_ell_list(i)[0] for i in range(nbl_eff)])
        # )
        # assert np.all(delta_ells_bpw == ells_per_band), 'delta_ell from bpw does not match ells_per_band'

        cl_gg_4covnmt = self.cl_gg_unb_3d.copy()
        cl_gl_4covnmt = self.cl_gl_unb_3d.copy()
        cl_ll_4covnmt = self.cl_ll_unb_3d.copy()

        # ! create nmt field from the mask (there will be no maps associated to the fields)
        # TODO maks=None (as in the example) or maps=[mask]? I think None
        f0_mask = nmt.NmtField(
            mask=self.mask_obj.mask, maps=None, spin=0, lite=True, lmax=ell_max_eff
        )
        f2_mask = nmt.NmtField(
            mask=self.mask_obj.mask, maps=None, spin=2, lite=True, lmax=ell_max_eff
        )

        with sl.timer('\nComputing namaster workspaces and coupling matrices...'):
            w00 = nmt.NmtWorkspace()
            w02 = nmt.NmtWorkspace()
            w22 = nmt.NmtWorkspace()
            w00.compute_coupling_matrix(f0_mask, f0_mask, nmt_bin_obj)
            w02.compute_coupling_matrix(f0_mask, f2_mask, nmt_bin_obj)
            w22.compute_coupling_matrix(f2_mask, f2_mask, nmt_bin_obj)

        # store in cache for later reuse, if required (TODO)
        os.makedirs(f'{self.output_path}/cache/nmt', exist_ok=True)
        w00.write_to(f'{self.output_path}/cache/nmt/w00_workspace.fits')
        w02.write_to(f'{self.output_path}/cache/nmt/w02_workspace.fits')
        w22.write_to(f'{self.output_path}/cache/nmt/w22_workspace.fits')

        # if the coupled covariance is required, I'll later need to convolve the
        # non-Gaussian terms. For this, I'll need the binned mode coupling matrices
        # (mcm), which I store in self
        if self.coupled_cov:
            w20 = nmt.NmtWorkspace()
            w20.compute_coupling_matrix(f2_mask, f0_mask, nmt_bin_obj)

            # extract only the relevant blocks
            mcm_tt_unb = w00.get_coupling_matrix()[:nbl_unb, :nbl_unb]
            mcm_et_unb = w20.get_coupling_matrix()[:nbl_unb, :nbl_unb]
            mcm_te_unb = w02.get_coupling_matrix()[:nbl_unb, :nbl_unb]
            mcm_ee_unb = w22.get_coupling_matrix()[:nbl_unb, :nbl_unb]

            # bin (and store in self)
            self.mcm_tt_binned = bin_mcm(mcm_tt_unb, nmt_bin_obj)
            self.mcm_et_binned = bin_mcm(mcm_et_unb, nmt_bin_obj)
            self.mcm_te_binned = bin_mcm(mcm_te_unb, nmt_bin_obj)
            self.mcm_ee_binned = bin_mcm(mcm_ee_unb, nmt_bin_obj)

        # if you want to use the iNKA, the cls to be passed are the coupled ones
        # divided by fsky
        if nmt_cfg['use_INKA']:
            z_combinations = list(itertools.product(range(self.zbins), repeat=2))
            for zi, zj in z_combinations:
                list_gg = [self.cl_gg_unb_3d[:, zi, zj]]
                list_gl = [
                    self.cl_gl_unb_3d[:, zi, zj],
                    np.zeros_like(self.cl_gl_unb_3d[:, zi, zj]),
                ]
                list_ll = [
                    self.cl_ll_unb_3d[:, zi, zj],
                    np.zeros_like(self.cl_ll_unb_3d[:, zi, zj]),
                    np.zeros_like(self.cl_ll_unb_3d[:, zi, zj]),
                    np.zeros_like(self.cl_ll_unb_3d[:, zi, zj]),
                ]
                # TODO the denominator should be the product of the masks?
                cl_gg_4covnmt[:, zi, zj] = w00.couple_cell(list_gg)[0] / fsky
                cl_gl_4covnmt[:, zi, zj] = w02.couple_cell(list_gl)[0] / fsky
                cl_ll_4covnmt[:, zi, zj] = w22.couple_cell(list_ll)[0] / fsky

        # add noise to spectra to compute NMT cov
        cl_tt_4covnmt = cl_gg_4covnmt + self.noise_3x2pt_unb_5d[1, 1, :, :, :]
        cl_te_4covnmt = cl_gl_4covnmt + self.noise_3x2pt_unb_5d[1, 0, :, :, :]
        cl_ee_4covnmt = cl_ll_4covnmt + self.noise_3x2pt_unb_5d[0, 0, :, :, :]
        cl_tb_4covnmt = np.zeros_like(cl_tt_4covnmt)
        cl_eb_4covnmt = np.zeros_like(cl_tt_4covnmt)
        cl_bb_4covnmt = np.zeros_like(cl_tt_4covnmt)

        # ! NAMASTER covariance
        with sl.timer('\nComputing cov workspace coupling coefficients...'):
            cw = nmt.NmtCovarianceWorkspace()
            cw.compute_coupling_coefficients(f0_mask, f0_mask, f0_mask, f0_mask)

        if nmt_cfg['use_namaster']:
            # in this case, since I assign particular elements of the 6D arrays, I also
            # have to allocate the shapes
            for term in self.cov_dict:
                for probe_2tpl in self.cov_dict[term]:
                    for dim in self.cov_dict[term][probe_2tpl]:
                        self.cov_dict[term][probe_2tpl][dim] = np.zeros(
                            (
                                nbl_eff,
                                nbl_eff,
                                self.zbins,
                                self.zbins,
                                self.zbins,
                                self.zbins,
                            )
                        )

            coupled_str = 'coupled' if self.coupled_cov else 'decoupled'
            spin0_str = ' spin0' if nmt_cfg['spin0'] else ''
            start_time = time.perf_counter()
            print(
                f'Computing {coupled_str}{spin0_str} partial-sky '
                'Gaussian covariance with NaMaster...'
            )

            # the nmt_gaussian_cov_spin0 and nmt_gaussian_cov functions modify
            # cov_dict in-place, so no need to capture any return value
            if nmt_cfg['spin0']:
                nmt_gaussian_cov_spin0(
                    cov_dict=self.cov_dict,
                    cl_tt=cl_tt_4covnmt,
                    cl_te=cl_te_4covnmt,
                    cl_ee=cl_ee_4covnmt,
                    zbins=self.zbins,
                    nbl=nbl_eff,
                    cw=cw,
                    w00=w00,
                    unique_probe_combs=unique_probe_combs,
                    coupled=self.coupled_cov,
                    ells_in=ells_unb,
                    ells_out=ells_eff,
                    ells_out_edges=ells_eff_edges,
                    weights=None,
                    which_binning='sum',
                )

            else:
                nmt_gaussian_cov(
                    cov_dict=self.cov_dict,
                    cl_tt=cl_tt_4covnmt,
                    cl_te=cl_te_4covnmt,
                    cl_ee=cl_ee_4covnmt,
                    cl_tb=cl_tb_4covnmt,
                    cl_eb=cl_eb_4covnmt,
                    cl_bb=cl_bb_4covnmt,
                    zbins=self.zbins,
                    nbl=nbl_eff,
                    cw=cw,
                    w00=w00,
                    w02=w02,
                    w22=w22,
                    unique_probe_combs=unique_probe_combs,
                    coupled=self.coupled_cov,
                    ells_in=ells_unb,
                    ells_out=ells_eff,
                    ells_out_edges=ells_eff_edges,
                    weights=None,
                    which_binning='sum',
                )

            print(f'...done in {(time.perf_counter() - start_time) / 60:.2f} m')

        elif self.cfg['sample_covariance']['compute_sample_cov']:
            cl_tt_4covsim = self.cl_gg_unb_3d + self.noise_3x2pt_unb_5d[1, 1, :, :, :]
            cl_te_4covsim = self.cl_gl_unb_3d + self.noise_3x2pt_unb_5d[1, 0, :, :, :]
            cl_ee_4covsim = self.cl_ll_unb_3d + self.noise_3x2pt_unb_5d[0, 0, :, :, :]
            cl_tb_4covsim = np.zeros_like(cl_tt_4covsim)
            cl_eb_4covsim = np.zeros_like(cl_tt_4covsim)
            cl_bb_4covsim = np.zeros_like(cl_tt_4covsim)

            result = sample_covariance(
                cl_GG_unbinned=cl_tt_4covsim,
                cl_LL_unbinned=cl_ee_4covsim,
                cl_GL_unbinned=cl_te_4covsim,
                cl_BB_unbinned=cl_bb_4covsim,
                cl_EB_unbinned=cl_eb_4covsim,
                cl_TB_unbinned=cl_tb_4covsim,
                nbl=nbl_eff,
                zbins=self.zbins,
                mask=self.mask_obj.mask,
                nside=self.mask_obj.nside,
                nreal=self.cfg['sample_covariance']['nreal'],
                coupled_cls=self.coupled_cov,
                which_cls=self.cfg['sample_covariance']['which_cls'],
                nmt_bin_obj=nmt_bin_obj,
                lmax=ell_max_eff,
                w00=w00,
                w02=w02,
                w22=w22,
                fix_seed=self.cfg['sample_covariance']['fix_seed'],
                n_iter=self.cfg['precision']['n_iter_nmt'],
                lite=True,
            )
            raise NotImplementedError(
                'the sample_covariance case should also return a dict!!'
            )
            cov_10d_out, self.sim_cl_GG, self.sim_cl_GL, self.sim_cl_LL = result

        return self.cov_dict
