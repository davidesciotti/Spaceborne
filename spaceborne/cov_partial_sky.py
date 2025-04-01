import numpy as np
import itertools
import os
import time
import pymaster as nmt
import healpy as hp
from tqdm import tqdm
from spaceborne import sb_lib as sl
from spaceborne import constants
from copy import deepcopy

import yaml


DEG2_IN_SPHERE = constants.DEG2_IN_SPHERE
DR1_DATE = constants.DR1_DATE


def nmt_gaussian_cov(cl_tt, cl_te, cl_ee, cl_tb, cl_eb, cl_bb, zbins, nbl, 
                     cw, w00, w02, w22,
                     coupled=False, ells_in=None, ells_out=None,
                     ells_out_edges=None, which_binning=None, weights=None):
    """
    Unified function to compute Gaussian covariance using NaMaster.
    
    
    # NOTE: the order of the arguments (in particular for the cls) is the following
    # spin_a1, spin_a2, spin_b1, spin_b2,
    # cla1b1, cla1b2, cla2b1, cla2b2
    # The order of the output dimensions depends on the order of the input list:
    # [cl_te, cl_tb] - > TE=0, TB=1
    # covar_TT_TE = covar_00_02[:, 0, :, 0]x
    # covar_TT_TB = covar_00_02[:, 0, :, 1]
    
    Parameters:
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
        assert cl.shape[0] == cl_tt.shape[0], \
        'input cls have different number of ell bins'
    
    nell = cl_tt.shape[0] if coupled else nbl

    print('Computing partial-sky Gaussian covariance with NaMaster...')
    cov_nmt_10d_arr = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))

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

        covar_00_00 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,
                                              cl_00_list(zi, zk),
                                              cl_00_list(zi, zl),
                                              cl_00_list(zj, zk),
                                              cl_00_list(zj, zl),
                                              coupled=coupled,
                                              wa=w00, wb=w00).reshape([nell, 1, nell, 1])
        covar_TT_TT = covar_00_00[:, 0, :, 0]

        covar_00_02 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 2,
                                              cl_00_list(zi, zk),
                                              cl_02_list(zi, zl),
                                              cl_00_list(zj, zk),
                                              cl_02_list(zj, zl),
                                              coupled=coupled,
                                              wa=w00, wb=w02).reshape([nell, 1, nell, 2])
        covar_TT_TE = covar_00_02[:, 0, :, 0]
        covar_TT_TB = covar_00_02[:, 0, :, 1]

        covar_00_22 = nmt.gaussian_covariance(cw,
                                              0, 0, 2, 2,
                                              cl_02_list(zi, zk),
                                              cl_02_list(zi, zl),
                                              cl_02_list(zj, zk),
                                              cl_02_list(zj, zl),
                                              coupled=coupled,
                                              wa=w00, wb=w22).reshape([nell, 1, nell, 4])
        covar_TT_EE = covar_00_22[:, 0, :, 0]
        covar_TT_EB = covar_00_22[:, 0, :, 1]
        covar_TT_BE = covar_00_22[:, 0, :, 2]
        covar_TT_BB = covar_00_22[:, 0, :, 3]

        covar_02_02 = nmt.gaussian_covariance(cw,
                                              0, 2, 0, 2,
                                              cl_00_list(zi, zk),
                                              cl_02_list(zi, zl),
                                              cl_20_list(zj, zk),
                                              cl_22_list(zj, zl),
                                              coupled=coupled,
                                              wa=w02, wb=w02).reshape([nell, 2, nell, 2])
        covar_TE_TE = covar_02_02[:, 0, :, 0]
        covar_TE_TB = covar_02_02[:, 0, :, 1]
        covar_TB_TE = covar_02_02[:, 1, :, 0]
        covar_TB_TB = covar_02_02[:, 1, :, 1]

        covar_02_22 = nmt.gaussian_covariance(cw,
                                              0, 2, 2, 2,
                                              cl_02_list(zi, zk),
                                              cl_02_list(zi, zl),
                                              cl_22_list(zj, zk),
                                              cl_22_list(zj, zl),
                                              coupled=coupled,
                                              wa=w02, wb=w22).reshape([nell, 2, nell, 4])
        covar_TE_EE = covar_02_22[:, 0, :, 0]
        covar_TE_EB = covar_02_22[:, 0, :, 1]
        covar_TE_BE = covar_02_22[:, 0, :, 2]
        covar_TE_BB = covar_02_22[:, 0, :, 3]
        covar_TB_EE = covar_02_22[:, 1, :, 0]
        covar_TB_EB = covar_02_22[:, 1, :, 1]
        covar_TB_BE = covar_02_22[:, 1, :, 2]
        covar_TB_BB = covar_02_22[:, 1, :, 3]

        covar_22_22 = nmt.gaussian_covariance(cw,
                                              2, 2, 2, 2,
                                              cl_22_list(zi, zk),
                                              cl_22_list(zi, zl),
                                              cl_22_list(zj, zk),
                                              cl_22_list(zj, zl),
                                              coupled=coupled,
                                              wa=w22, wb=w22).reshape([nell, 4, nell, 4])

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

        if coupled:
            # in this case, the nmt output is unbinned
            cov_nmt_10d_arr[0, 0, 0, 0, :, :, zi, zj, zk, zl] = \
                sl.bin_2d_array(covar_EE_EE, ells_in, ells_out, ells_out_edges, which_binning, weights)
            cov_nmt_10d_arr[1, 0, 0, 0, :, :, zi, zj, zk, zl] = \
                sl.bin_2d_array(covar_TE_EE, ells_in, ells_out, ells_out_edges, which_binning, weights)
            cov_nmt_10d_arr[1, 0, 1, 0, :, :, zi, zj, zk, zl] = \
                sl.bin_2d_array(covar_TE_TE, ells_in, ells_out, ells_out_edges, which_binning, weights)
            cov_nmt_10d_arr[1, 1, 0, 0, :, :, zi, zj, zk, zl] = \
                sl.bin_2d_array(covar_TT_EE, ells_in, ells_out, ells_out_edges, which_binning, weights)
            cov_nmt_10d_arr[1, 1, 1, 0, :, :, zi, zj, zk, zl] = \
                sl.bin_2d_array(covar_TT_TE, ells_in, ells_out, ells_out_edges, which_binning, weights)
            cov_nmt_10d_arr[1, 1, 1, 1, :, :, zi, zj, zk, zl] = \
                sl.bin_2d_array(covar_TT_TT, ells_in, ells_out, ells_out_edges, which_binning, weights)
            # the remaining blocks can be filled in by symmetry (with zi, zj <-> zk, zl)
            cov_nmt_10d_arr[0, 0, 1, 0, :, :, zk, zl, zi, zj] = \
                sl.bin_2d_array(covar_TE_EE.T, ells_in, ells_out, ells_out_edges, which_binning, weights)
            cov_nmt_10d_arr[0, 0, 1, 1, :, :, zk, zl, zi, zj] = \
                sl.bin_2d_array(covar_TT_EE.T, ells_in, ells_out, ells_out_edges, which_binning, weights)
            cov_nmt_10d_arr[1, 0, 1, 1, :, :, zk, zl, zi, zj] = \
                sl.bin_2d_array(covar_TT_TE.T, ells_in, ells_out, ells_out_edges, which_binning, weights)
        else:
            cov_nmt_10d_arr[0, 0, 0, 0, :, :, zi, zj, zk, zl] = covar_EE_EE
            cov_nmt_10d_arr[1, 0, 0, 0, :, :, zi, zj, zk, zl] = covar_TE_EE
            cov_nmt_10d_arr[1, 0, 1, 0, :, :, zi, zj, zk, zl] = covar_TE_TE
            cov_nmt_10d_arr[1, 1, 0, 0, :, :, zi, zj, zk, zl] = covar_TT_EE
            cov_nmt_10d_arr[1, 1, 1, 0, :, :, zi, zj, zk, zl] = covar_TT_TE
            cov_nmt_10d_arr[1, 1, 1, 1, :, :, zi, zj, zk, zl] = covar_TT_TT
            # the remaining blocks can be filled in by symmetry (with zi, zj <-> zk, zl)
            cov_nmt_10d_arr[0, 0, 1, 0, :, :, zk, zl, zi, zj] = covar_TE_EE.T
            cov_nmt_10d_arr[0, 0, 1, 1, :, :, zk, zl, zi, zj] = covar_TT_EE.T
            cov_nmt_10d_arr[1, 0, 1, 1, :, :, zk, zl, zi, zj] = covar_TT_TE.T

    return cov_nmt_10d_arr

def nmt_gaussian_cov_spin0(cl_tt, cl_te, cl_ee, zbins, nbl, cw, w00, coupled, ells_in, ells_out,
                             ells_out_edges, which_binning, weights):

    cl_et = cl_te.transpose(0, 2, 1)

    print('Computing spin-0 partial-sky Gaussian covariance with NaMaster...')
    cov_nmt_10d_arr = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))

    z_combinations = list(itertools.product(range(zbins), repeat=4))
    for zi, zj, zk, zl in tqdm(z_combinations):

        covar_00_00 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_tt[:, zi, zk]],  # TT
                                              [cl_tt[:, zi, zl]],  # TT
                                              [cl_tt[:, zj, zk]],  # TT
                                              [cl_tt[:, zj, zl]],  # TT
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_TT_TT = covar_00_00

        covar_00_02 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_tt[:, zi, zk]],  # TT
                                              [cl_te[:, zi, zl]],  # TE, TB
                                              [cl_tt[:, zj, zk]],  # TT
                                              [cl_te[:, zj, zl]],  # TE, TB
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_TT_TE = covar_00_02

        covar_02_00 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_tt[:, zi, zk]],  # TT
                                              [cl_tt[:, zi, zl]],  # TT
                                              [cl_et[:, zj, zk]],  # TE, TB
                                              [cl_et[:, zj, zl]],  # TE, TB
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_TE_TT = covar_02_00

        covar_02_02 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_tt[:, zi, zk]],  # TT
                                              [cl_te[:, zi, zl]],  # TE, TB
                                              [cl_et[:, zj, zk]],  # ET, BT
                                              [cl_ee[:, zj, zl]],  # EE, EB, BE, BB
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_TE_TE = covar_02_02

        covar_00_22 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_te[:, zi, zk]],  # TE, TB
                                              [cl_te[:, zi, zl]],  # TE, TB
                                              [cl_te[:, zj, zk]],  # TE, TB
                                              [cl_te[:, zj, zl]],  # TE, TB
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_TT_EE = covar_00_22

        covar_02_22 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_te[:, zi, zk]],  # TE, TB
                                              [cl_te[:, zi, zl]],  # TE, TB
                                              [cl_ee[:, zj, zk]],
                                              [cl_ee[:, zj, zl]],
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_TE_EE = covar_02_22

        covar_22_22 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_ee[:, zi, zk]],
                                              [cl_ee[:, zi, zl]],
                                              [cl_ee[:, zj, zk]],
                                              [cl_ee[:, zj, zl]],
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_EE_EE = covar_22_22

        covar_22_02 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_et[:, zi, zk]],
                                              [cl_ee[:, zi, zl]],
                                              [cl_et[:, zj, zk]],
                                              [cl_ee[:, zj, zl]],
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_EE_TE = covar_22_02

        covar_22_00 = nmt.gaussian_covariance(cw,
                                              0, 0, 0, 0,  # Spins of the 4 fields
                                              [cl_et[:, zi, zk]],
                                              [cl_et[:, zi, zl]],
                                              [cl_et[:, zj, zk]],
                                              [cl_et[:, zj, zl]],
                                              coupled=coupled,
                                              wa=w00, wb=w00)
        covar_EE_TT = covar_22_00
        
        if coupled:
            cov_nmt_10d_arr[0, 0, 0, 0, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_EE_EE, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[0, 0, 1, 0, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_EE_TE, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[0, 0, 1, 1, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_EE_TT, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[1, 0, 0, 0, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_TE_EE, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[1, 0, 1, 0, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_TE_TE, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[1, 1, 0, 0, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_TT_EE, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[1, 1, 1, 0, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_TT_TE, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[1, 0, 1, 1, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_TE_TT, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
            cov_nmt_10d_arr[1, 1, 1, 1, :, :, zi, zj, zk, zl] = \
                    sl.bin_2d_array(covar_TT_TT, ells_in, ells_out, ells_out_edges, 
                    which_binning, weights)
        
        else:
            cov_nmt_10d_arr[0, 0, 0, 0, :, :, zi, zj, zk, zl] = covar_EE_EE
            cov_nmt_10d_arr[0, 0, 1, 0, :, :, zi, zj, zk, zl] = covar_EE_TE
            cov_nmt_10d_arr[0, 0, 1, 1, :, :, zi, zj, zk, zl] = covar_EE_TT
            cov_nmt_10d_arr[1, 0, 0, 0, :, :, zi, zj, zk, zl] = covar_TE_EE
            cov_nmt_10d_arr[1, 0, 1, 0, :, :, zi, zj, zk, zl] = covar_TE_TE
            cov_nmt_10d_arr[1, 1, 0, 0, :, :, zi, zj, zk, zl] = covar_TT_EE
            cov_nmt_10d_arr[1, 1, 1, 0, :, :, zi, zj, zk, zl] = covar_TT_TE
            cov_nmt_10d_arr[1, 0, 1, 1, :, :, zi, zj, zk, zl] = covar_TE_TT
            cov_nmt_10d_arr[1, 1, 1, 1, :, :, zi, zj, zk, zl] = covar_TT_TT



    return cov_nmt_10d_arr




def linear_lmin_binning(NSIDE, lmin, bw):
    """
    Generate a linear binning scheme based on a minimum multipole 'lmin' and bin width 'bw'.

    Parameters:
    -----------
    NSIDE : int
        The NSIDE parameter of the HEALPix grid.

    lmin : int
        The minimum multipole to start the binning.

    bw : int
        The bin width, i.e., the number of multipoles in each bin.

    Returns:
    --------
    nmt_bins
        A binning scheme object defining linearly spaced bins starting from 'lmin' with
        a width of 'bw' multipoles.

    Notes:
    ------
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
    """
    Compute the mixing matrix for coupling spherical harmonic modes using
    the provided binning scheme and mask.

    Parameters:
    -----------
    bin_scheme : nmt_bins
        A binning scheme object defining the bins for the coupling matrix.

    mask : nmt_field
        A mask object defining the regions of the sky to include in the computation.

    wkspce_name : str
        The file name for storing or retrieving the computed workspace containing
        the coupling matrix.

    Returns:
    --------
    nmt_workspace
        A workspace object containing the computed coupling matrix.

    Notes:
    ------
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
        print('Mixing matrix has already been calculated and is in the workspace file : ', wkspce_name, '. Read it.')
        w.read_from(wkspce_name)
    else:
        print('The file : ', wkspce_name, ' does not exists. Calculating the mixing matrix and writing it.')
        w.compute_coupling_matrix(fmask, fmask, bin_scheme)
        w.write_to(wkspce_name)
    print('Done computing the mixing matrix. It took ', time.time() - start, 's.')
    return w






def sample_covariance( # fmt: skip
    cl_GG_unbinned, cl_LL_unbinned, cl_GL_unbinned, 
    cl_BB_unbinned, cl_EB_unbinned, cl_TB_unbinned, 
    nbl, zbins, mask, nside, nreal, coupled_cls, which_cls, lmax=None,
):  # fmt: skip
    if lmax is None:
        lmax = 3 * nside - 1

    SEEDVALUE = np.arange(nreal)

    # TODO use only independent z pairs
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

        # * 2. compute and bin simulated cls for all zbin combinations, using input correlated maps
        for zi, zj in zij_combinations:
            sim_cl_GG_ij, sim_cl_GL_ij, sim_cl_LL_ij = pcls_from_maps(
                corr_maps_gg=corr_maps_gg,
                corr_maps_ll=corr_maps_ll,
                zi=zi,
                zj=zj,
                mask=mask,
                coupled_cls=coupled_cls,
                which_cls=which_cls,
            )

            assert sim_cl_GG_ij.shape == sim_cl_GL_ij.shape == sim_cl_LL_ij.shape, (
                'Simulated cls must have the same shape'
            )

            if len(sim_cl_GG_ij) != nbl:
                sim_cl_GG[i, :, zi, zj] = bin_obj.bin_cell(sim_cl_GG_ij)
                sim_cl_GL[i, :, zi, zj] = bin_obj.bin_cell(sim_cl_GL_ij)
                sim_cl_LL[i, :, zi, zj] = bin_obj.bin_cell(sim_cl_LL_ij)
            else:
                sim_cl_GG[i, :, zi, zj] = sim_cl_GG_ij
                sim_cl_GL[i, :, zi, zj] = sim_cl_GL_ij
                sim_cl_LL[i, :, zi, zj] = sim_cl_LL_ij

    # * 3. compute sample covariance
    for zi, zj, zk, zl in tqdm(zijkl_combinations):
        # ! compute the sample covariance
        # you could also cut the mixed cov terms, but for cross-redshifts it becomes a bit tricky
        kwargs = dict(rowvar=False, bias=False)
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


def build_cl_ring_ordering(cl_3d):
    zbins = cl_3d.shape[1]
    assert cl_3d.shape[1] == cl_3d.shape[2], (
        'input cls should have shape (ell_bins, zbins, zbins)'
    )
    cl_ring_list = []

    for offset in range(0, zbins):  # offset defines the distance from the main diagonal
        for zi in range(zbins - offset):
            zj = zi + offset
            cl_ring_list.append(cl_3d[:, zi, zj])

    return cl_ring_list


def build_cl_tomo_TEB_ring_ord(
    cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB, zbins, spectra_types=['T', 'E', 'B']
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
    for zi in range(0, zbins):
        for s1 in spectra_types:
            row = []
            for zj in range(0, zbins):
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

            probe_a, zi, probe_b, zj = matrix[zi][zj].split('-')

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

            cl_ring_ord_list.append(cl[:, int(zi), int(zj)])

    return cl_ring_ord_list


def get_sample_field_bu(cl_TT, cl_EE, cl_BB, cl_TE, nside):
    """This routine generates a spin-0 and a spin-2 Gaussian random field based
    on these power spectra.
    From https://namaster.readthedocs.io/en/latest/source/sample_covariance.html
    """
    map_t, map_q, map_u = hp.synfast([cl_TT, cl_EE, cl_BB, cl_TE], nside)
    return nmt.NmtField(mask, [map_t], lite=False), nmt.NmtField(
        mask, [map_q, map_u], lite=False
    )


def cls_to_maps(cl_TT, cl_EE, cl_BB, cl_TE, nside, lmax=None):
    """
    This routine generates maps for spin-0 and a spin-2 Gaussian random field based
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


def masked_maps_to_nmtFields(map_T, map_Q, map_U, mask, lmax, n_iter=0, lite=True):
    """
    Create NmtField objects from masked maps.

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
    cl_TT, cl_EE, cl_BB, cl_TE, cl_EB, cl_TB, nreal, nside, zbins_use
):
    print(f'Generating {nreal} maps for nside {nside}...')

    cl_ring_big_list = build_cl_tomo_TEB_ring_ord(
        cl_TT=cl_TT,
        cl_EE=cl_EE,
        cl_BB=cl_BB,
        cl_TE=cl_TE,
        cl_EB=cl_EB,
        cl_TB=cl_TB,
        zbins=zbins_use,
        spectra_types=['T', 'E', 'B'],
    )

    corr_maps_gg_list = []
    corr_maps_ll_list = []

    for _ in tqdm(range(nreal)):
        corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=lmax, new=True)
        assert len(corr_alms_tot) == zbins_use * 3, 'wrong number of alms'

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


def pcls_from_maps(corr_maps_gg, corr_maps_ll, zi, zj, mask, coupled_cls, which_cls):
    # both healpy anafast and nmt.compute_coupled_cell return the coupled cls. Dividing by fsky gives a rough
    # approximation of the true Cls
    correction_factor = 1.0 if coupled_cls else fsky

    if which_cls == 'namaster':
        f0 = np.array(
            [nmt.NmtField(mask, [map_T], n_iter=3, lite=True) for map_T in corr_maps_gg]
        )
        f2 = np.array(
            [
                nmt.NmtField(mask, [map_Q, map_U], n_iter=3, lite=True)
                for (map_Q, map_U) in corr_maps_ll
            ]
        )

        if coupled_cls:  # ! TODO fix this!!
            # pseudo-Cls. Becomes an ok estimator for the true Cls if divided by fsky
            pseudo_cl_tt = (
                nmt.compute_coupled_cell(f0[zi], f0[zj])[0] / correction_factor
            )
            pseudo_cl_te = (
                nmt.compute_coupled_cell(f0[zi], f2[zj])[0] / correction_factor
            )
            pseudo_cl_ee = (
                nmt.compute_coupled_cell(f2[zi], f2[zj])[0] / correction_factor
            )
        else:
            # best estimator for the true Cls
            pseudo_cl_tt = compute_master(f0[zi], f0[zj], w00)[0, :]
            pseudo_cl_te = compute_master(f0[zi], f2[zj], w02)[0, :]
            pseudo_cl_ee = compute_master(f2[zi], f2[zj], w22)[0, :]

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
        pseudo_cl_tt = hp_pcl_tot[0, :]
        pseudo_cl_ee = hp_pcl_tot[1, :]
        pseudo_cl_bb = hp_pcl_tot[2, :]
        pseudo_cl_te = hp_pcl_tot[3, :]
        pseudo_cl_eb = hp_pcl_tot[4, :]
        pseudo_cl_tb = hp_pcl_tot[5, :]
        pseudo_cl_be = pseudo_cl_eb  # ! warning!!
        if not coupled_cls:
            pseudo_cl_tt = w00.decouple_cell(pseudo_cl_tt[None, :])[0, :]
            pseudo_cl_ee = w22.decouple_cell(
                np.vstack((pseudo_cl_ee, pseudo_cl_eb, pseudo_cl_be, pseudo_cl_bb))
            )[0, :]
            pseudo_cl_te = w02.decouple_cell(np.vstack((pseudo_cl_te, pseudo_cl_tb)))[
                0, :
            ]

    else:
        raise ValueError('which_cls must be namaster or healpy')

    return np.array(pseudo_cl_tt), np.array(pseudo_cl_te), np.array(pseudo_cl_ee)





def linear_binning(lmax, lmin, bw, w=None):
    nbl = (lmax - lmin) // bw + 1
    bins = np.linspace(lmin, lmax + 1, nbl + 1)
    ell = np.arange(lmin, lmax + 1)
    i = np.digitize(ell, bins) - 1
    b = nmt.NmtBin(bpws=i, ells=ell, weights=w, lmax=lmax)

    return b


def log_binning(lmax, lmin, nbl, w=None):
    op = np.log10

    def inv(x):
        return 10**x

    bins = inv(np.linspace(op(lmin), op(lmax + 1), nbl + 1))
    ell = np.arange(lmin, lmax + 1)
    i = np.digitize(ell, bins) - 1
    b = nmt.NmtBin(bpws=i, ells=ell, weights=w, lmax=lmax)

    return b



def get_lmid(ells, k):
    return 0.5 * (ells[k:] + ells[:-k])


cov_blocks_names_all = (  # fmt: skip
    'LLLL', 'LLGL', 'LLGG',
    'GLLL', 'GLGL', 'GLGG',
    'GGLL', 'GGGL', 'GGGG',
)  # fmt: skip

# ! settings
# import the yaml config file
# cfg = yaml.load(sys.stdin, Loader=yaml.FullLoader)
# if you want to execute without passing the path
with open(f'{ROOT}/Spaceborne_covg/config/example_config_namaster.yaml') as file:
    cfg = yaml.safe_load(file)

survey_area_deg2 = cfg['survey_area_deg2']  # deg^2
fsky = survey_area_deg2 / constants.DEG2_IN_SPHERE

zbins = cfg['zbins']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nell_bins = cfg['ell_bins']

sigma_eps = cfg['sigma_eps_i'] * np.sqrt(2)
sigma_eps2 = sigma_eps**2

EP_or_ED = cfg['EP_or_ED']
GL_or_LG = 'GL'
triu_tril = cfg['triu_tril']
row_col_major = cfg['row_col_major']
covariance_ordering_2D = cfg['covariance_ordering_2D']

part_sky = cfg['part_sky']
workspace_path = cfg['workspace_path']
mask_path = cfg['mask_path'].format(ROOT=ROOT)

output_folder = cfg['output_folder']
n_probes = 2
# ! end settings


start = time.perf_counter()
print('Computing the partial-sky covariance with NaMaster')


ells_unbinned = np.arange(5000)
ells_per_band = cfg['ells_per_band']
nside = cfg['nside']
nreal = cfg['nreal']
zbins_use = cfg['zbins_use']
coupled = cfg['coupled']
use_INKA = cfg['use_INKA']
which_cls = cfg['which_cls']
coupled_label = 'coupled' if coupled else 'decoupled'

# if use_INKA and cfg['coupled'] :
#     raise ValueError('Cannot do iNKA for coupled Cls covariance.')

# read or generate mask
if cfg['read_mask']:
    if mask_path.endswith('footprint-gal-12.fits'):
        mask = hp.read_map(mask_path)
        mask = np.where(  # fmt: skip
            np.logical_and(mask <= constants.DR1_DATE, mask >= 0.0), 1.0, 0,
        )  # fmt: skip
        # Save the actual DR1 mask to a new FITS file
        # output_path = mask_path.replace(".fits", "_DR1.fits")
        # hp.write_map(output_path, mask, dtype=np.float64, overwrite=True)
    elif mask_path.endswith('.fits'):
        mask = hp.read_map(mask_path)
    elif mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    mask = hp.ud_grade(mask, nside_out=nside)

else:
    # mask = utils.generate_polar_cap(area_deg2=survey_area_deg2, nside=cfg['nside'])
    mask = sl.generate_polar_cap(
        area_deg2=survey_area_deg2, nside=cfg['nside']
    )

fsky = np.mean(mask**2)
survey_area_deg2 = fsky * DEG2_IN_SPHERE

if fsky == 1:
    np.testing.assert_allclose(mask, np.ones_like(mask), atol=0, rtol=1e-6)

# apodize
hp.mollview(mask, title='before apodization', cmap='inferno_r')
if cfg['apodize_mask'] and int(survey_area_deg2) != int(DEG2_IN_SPHERE):
    mask = nmt.mask_apodization(mask, aposize=cfg['aposize'], apotype='Smooth')
    hp.mollview(mask, title='after apodization', cmap='inferno_r')

# recompute after apodizing
fsky = np.mean(mask**2)
survey_area_deg2 = fsky * DEG2_IN_SPHERE

npix = hp.nside2npix(nside)
pix_area = 4 * np.pi

# check fsky and nside
nside_from_mask = hp.get_nside(mask)
assert nside_from_mask == cfg['nside'], (
    'nside from mask is not consistent with the desired nside in the cfg file'
)
assert ell_max < 3 * cfg['nside'], 'nside cannot be higher than 3*nside'

# set different possible values for lmax
lmax_mask = int(np.pi / hp.pixelfunc.nside2resol(nside))
lmax_healpy = 3 * nside
# to be safe, following https://heracles.readthedocs.io/stable/examples/example.html
lmax_healpy_safe = int(1.5 * nside)  # TODO test this
lmax = lmax_healpy

# get lmin: quick and dirty (and liely too optimistic) estimate
survey_area_sterad = np.sum(mask) * hp.nside2pixarea(nside)
lmin_mask = int(np.ceil(np.pi / np.sqrt(survey_area_sterad)))

# ! Define the set of bandpowers used in the computation of the pseudo-Cl
# Initialize binning scheme with bandpowers of constant width (ells_per_band multipoles per bin)
# ell_values, delta_values, ell_bin_edges = utils.compute_ells(nbl, 0, lmax, recipe='ISTF', output_ell_bin_edges=True)

if cfg['nmt_ell_binning'] == 'linear':
    bin_obj = linear_binning(ell_max, ell_min, ells_per_band)
elif cfg['nmt_ell_binning'] == 'log':
    bin_obj = log_binning(ell_max, ell_min, nell_bins)
else:
    raise ValueError('nmt_ell_binning must be either "linear" or "log"')

ells_eff = bin_obj.get_effective_ells()  # get effective ells per bandpower
nbl_eff = len(ells_eff)

# notice that bin_obj.get_ell_list(nbl_eff) is out of bounds
ells_eff_edges = np.array([bin_obj.get_ell_list(i)[0] for i in range(nbl_eff)])
ells_eff_edges = np.append(
    ells_eff_edges, bin_obj.get_ell_list(nbl_eff - 1)[-1] + 1
)  # careful f the +1!
lmin_eff = ells_eff_edges[0]
lmax_eff = bin_obj.lmax

ells_tot = np.arange(lmax_eff + 1)
nbl_tot = len(ells_tot)
assert nbl_tot == lmax_eff + 1, 'nbl_tot does not match lmax_eff + 1'
ells_bpw = ells_tot[lmin_eff : lmax_eff + 1]
delta_ells_bpw = np.diff(
    np.array([bin_obj.get_ell_list(i)[0] for i in range(nbl_eff)])
)
# assert np.all(delta_ells_bpw == ells_per_band), 'delta_ell from bpw does not match ells_per_band'

# ! create nmt field from the mask (there will be no maps associated to the fields)
# TODO maks=None (as in the example) or maps=[mask]? I think None
start_time = time.perf_counter()
print('computing coupling coefficients...')
f0_mask = nmt.NmtField(mask=mask, maps=None, spin=0, lite=True, lmax=lmax_eff)
f2_mask = nmt.NmtField(mask=mask, maps=None, spin=2, lite=True, lmax=lmax_eff)
w00 = nmt.NmtWorkspace()
w02 = nmt.NmtWorkspace()
w22 = nmt.NmtWorkspace()
w00.compute_coupling_matrix(f0_mask, f0_mask, bin_obj)
w02.compute_coupling_matrix(f0_mask, f2_mask, bin_obj)
w22.compute_coupling_matrix(f2_mask, f2_mask, bin_obj)
print(f'...done in {(time.perf_counter() - start_time):.2f}s')



# cut and bin the theory
cl_GG_unbinned = deepcopy(cl_GG_unbinned[: lmax_eff + 1, :zbins_use, :zbins_use])
cl_GL_unbinned = deepcopy(cl_GL_unbinned[: lmax_eff + 1, :zbins_use, :zbins_use])
cl_LL_unbinned = deepcopy(cl_LL_unbinned[: lmax_eff + 1, :zbins_use, :zbins_use])
cl_BB_unbinned = np.zeros_like(cl_LL_unbinned)
cl_TB_unbinned = np.zeros_like(cl_LL_unbinned)
cl_EB_unbinned = np.zeros_like(cl_LL_unbinned)





# ! COMPUTE AND COMPARE DIFFERENT VERSIONS OF THE Cls


# ! Let's now compute the Gaussian estimate of the covariance!
start_time = time.perf_counter()
cw = nmt.NmtCovarianceWorkspace()
print('Computing cov workspace coupling coefficients...')
cw.compute_coupling_coefficients(f0_mask, f0_mask, f0_mask, f0_mask)
print(f'...one in {(time.perf_counter() - start_time):.2f} s...')

# TODO generalize to all zbin cross-correlations; z=0 for the moment
# shape: (n_cls, n_bpws, n_cls, lmax+1)
# n_cls is the number of power spectra (1, 2 or 4 for spin 0-0, spin 0-2 and spin 2-2 correlations)

# if coupled:
#     raise ValueError('coupled case not fully implemented yet')
#     print('Inputting pseudo-Cls/fsky to use INKA...')
#     nbl_4covnmt = nbl_tot
#     cl_GG_4covnmt = pcl_GG_nmt[:, zi, zj] / fsky
#     cl_GL_4covnmt = pcl_GL_nmt[:, zi, zj] / fsky
#     cl_LL_4covnmt = pcl_LL_nmt[:, zi, zj] / fsky
#     cl_GG_4covsb = pcl_GG_nmt  # or bpw_pcl_GG_nmt?
#     cl_GL_4covsb = pcl_GL_nmt  # or bpw_pcl_GL_nmt?
#     cl_LL_4covsb = pcl_LL_nmt  # or bpw_pcl_LL_nmt?
#     ells_4covsb = ells_tot
#     nbl_4covsb = len(ells_4covsb)
#     delta_ells_4covsb = np.ones(nbl_4covsb)  # since it's unbinned
# else:

nbl_4covnmt = nbl_eff
ells_4covsb = ells_tot
nbl_4covsb = len(ells_4covsb)
delta_ells_4covsb = np.ones(nbl_4covsb)  # since it's unbinned
cl_GG_4covsb = cl_GG_unbinned[:, :zbins_use, :zbins_use]
cl_GL_4covsb = cl_GL_unbinned[:, :zbins_use, :zbins_use]
cl_LL_4covsb = cl_LL_unbinned[:, :zbins_use, :zbins_use]

if use_INKA:
    cl_GG_4covnmt = np.zeros_like(cl_GG_unbinned)
    cl_GL_4covnmt = np.zeros_like(cl_GL_unbinned)
    cl_LL_4covnmt = np.zeros_like(cl_LL_unbinned)
    z_combinations = list(itertools.product(range(zbins_use), repeat=2))
    for zi, zj in z_combinations:
        cl_GG_4covnmt[:, zi, zj] = (
            w00.couple_cell([cl_GG_unbinned[:, zi, zj]])[0] / fsky
        )
        cl_GL_4covnmt[:, zi, zj] = (
            w02.couple_cell(
                [
                    cl_GL_unbinned[:, zi, zj],
                    np.zeros_like(cl_GL_unbinned[:, zi, zj]),
                ]
            )[0]
            / fsky
        )
        cl_LL_4covnmt[:, zi, zj] = (
            w22.couple_cell(
                [
                    cl_LL_unbinned[:, zi, zj],
                    np.zeros_like(cl_LL_unbinned[:, zi, zj]),
                    np.zeros_like(cl_LL_unbinned[:, zi, zj]),
                    np.zeros_like(cl_LL_unbinned[:, zi, zj]),
                ]
            )[0]
            / fsky
        )

    # TODO not super sure about this
    # cl_GG_4covsb = pcl_GG_nmt[:, :zbins_use, :zbins_use] / fsky
    # cl_GL_4covsb = pcl_GL_nmt[:, :zbins_use, :zbins_use] / fsky
    # cl_LL_4covsb = pcl_LL_nmt[:, :zbins_use, :zbins_use] / fsky
else:
    cl_GG_4covnmt = cl_GG_unbinned
    cl_GL_4covnmt = cl_GL_unbinned
    cl_LL_4covnmt = cl_LL_unbinned
    # cl_GG_4covsb = cl_GG_unbinned[:, :zbins_use, :zbins_use]
    # cl_GL_4covsb = cl_GL_unbinned[:, :zbins_use, :zbins_use]
    # cl_LL_4covsb = cl_LL_unbinned[:, :zbins_use, :zbins_use]

# the noise is needed also for the SIM and NMT covs
noise_3x2pt_4d = sl.build_noise(
    zbins_use,
    n_probes,
    sigma_eps2=sigma_eps2,
    ng_shear=n_gal_shear,
    ng_clust=n_gal_clustering,
    EP_or_ED=EP_or_ED,
)
noise_3x2pt_5d = np.zeros((n_probes, n_probes, nbl_4covsb, zbins_use, zbins_use))
for probe_A in (0, 1):
    for probe_B in (0, 1):
        for ell_idx in range(nbl_4covsb):
            noise_3x2pt_5d[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4d[
                probe_A, probe_B, ...
            ]

cl_tt_4covnmt = cl_GG_4covnmt + noise_3x2pt_5d[1, 1, :, :, :]
cl_te_4covnmt = cl_GL_4covnmt + noise_3x2pt_5d[1, 0, :, :, :]
cl_ee_4covnmt = cl_LL_4covnmt + noise_3x2pt_5d[0, 0, :, :, :]
cl_tb_4covnmt = np.zeros_like(cl_tt_4covnmt)
cl_eb_4covnmt = np.zeros_like(cl_tt_4covnmt)
cl_bb_4covnmt = np.zeros_like(cl_tt_4covnmt)

cl_tt_4covsim = cl_GG_unbinned + noise_3x2pt_5d[1, 1, :, :, :]
cl_te_4covsim = cl_GL_unbinned + noise_3x2pt_5d[1, 0, :, :, :]
cl_ee_4covsim = cl_LL_unbinned + noise_3x2pt_5d[0, 0, :, :, :]
cl_tb_4covsim = np.zeros_like(cl_tt_4covsim)
cl_eb_4covsim = np.zeros_like(cl_tt_4covsim)
cl_bb_4covsim = np.zeros_like(cl_tt_4covsim)

# ! NAMASTER covariance
if cfg['spin0']:
    cov_nmt_10d = nmt_gaussian_cov_spin0(
        cl_tt=cl_tt_4covnmt,
        cl_te=cl_te_4covnmt,
        cl_ee=cl_ee_4covnmt,
        zbins=zbins_use,
        nbl=nbl_eff,
        cw=cw,
        w00=w00,
        coupled=cfg['coupled'],
        ells_in=ells_tot,
        ells_out=ells_eff,
        ells_out_edges=ells_eff_edges,
        weights=None,
        which_binning='sum',
    )

else:
    cov_nmt_10d = nmt_gaussian_cov(
        cl_tt=cl_tt_4covnmt,
        cl_te=cl_te_4covnmt,
        cl_ee=cl_ee_4covnmt,
        cl_tb=cl_tb_4covnmt,
        cl_eb=cl_eb_4covnmt,
        cl_bb=cl_bb_4covnmt,
        zbins=zbins_use,
        nbl=nbl_eff,
        cw=cw,
        w00=w00,
        w02=w02,
        w22=w22,
        coupled=cfg['coupled'],
        ells_in=ells_tot,
        ells_out=ells_eff,
        ells_out_edges=ells_eff_edges,
        weights=None,
        which_binning='sum',
    )

np.save(f'{output_folder}/cov_Gauss_3x2pt_10D.npy', cov_nmt_10d)
