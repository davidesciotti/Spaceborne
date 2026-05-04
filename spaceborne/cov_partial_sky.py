import itertools
import os
import time
import warnings
from itertools import combinations_with_replacement
from typing import TypedDict

import healpy as hp
import numpy as np
import pymaster as nmt
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

from spaceborne import constants, ell_utils, mask_utils
from spaceborne import cov_dict as cd
from spaceborne import sb_lib as sl

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
    spin0: bool,
    cl_tt: np.ndarray,
    cl_te: np.ndarray,
    cl_ee: np.ndarray,
    cl_tb: np.ndarray,
    cl_eb: np.ndarray,
    cl_bb: np.ndarray,
    nbl: int,
    zbins: int,
    ind_dict: dict,
    cw_dict: dict,
    w00_dict: dict,
    w02_dict: dict,
    w22_dict: dict,
    unique_probe_combs: list[str],
    nonreq_probe_combs: list[str],
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

    def cl_00_list(zi, zj, spin0):
        if spin0:
            return [cl_tt[:, zi, zj]]
        else:
            return [cl_tt[:, zi, zj]]

    def cl_02_list(zi, zj, spin0):
        if spin0:
            return [cl_te[:, zi, zj]]
        else:
            return [cl_te[:, zi, zj], cl_tb[:, zi, zj]]

    def cl_20_list(zi, zj, spin0):
        if spin0:
            return [cl_et[:, zi, zj]]
        else:
            return [cl_et[:, zi, zj], cl_bt[:, zi, zj]]

    def cl_22_list(zi, zj, spin0):
        if spin0:
            return [cl_ee[:, zi, zj]]
        else:
            return [
                cl_ee[:, zi, zj],
                cl_eb[:, zi, zj],
                cl_be[:, zi, zj],
                cl_bb[:, zi, zj],
            ]

    # define some useful dictionaries
    spin_dict = {'G': 0, 'L': 2}

    cl_list_dict = {
        '00': cl_00_list,
        '02': cl_02_list,
        '20': cl_20_list,
        '22': cl_22_list,
    }

    wsp_spin2_dict = {'00': w00_dict, '02': w02_dict, '22': w22_dict}
    wsp_spin0_dict = {'00': w00_dict, '02': w00_dict, '22': w00_dict}
    wsp_dict = wsp_spin0_dict if spin0 else wsp_spin2_dict

    bin_cov_kw: Bin2DArrayKwargs = {
        'ells_in': ells_in,
        'ells_out': ells_out,
        'ells_out_edges': ells_out_edges,
        'weights_in': weights,
        'which_binning': which_binning,
        'interpolate': True,
    }

    for probe_abcd in tqdm(unique_probe_combs):
        probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
        probe_a, probe_b, probe_c, probe_d = list(probe_abcd)

        tqdm.write(
            f'NaMaster G cov: computing probe combination {(probe_ab, probe_cd)}'
        )

        s1 = spin_dict[probe_a]
        s2 = spin_dict[probe_b]
        s3 = spin_dict[probe_c]
        s4 = spin_dict[probe_d]

        # shape of the spin axis can be either 1 (spin0) or 2/4 (spin2)
        reshape_ab = s1 + s2 if s1 + s2 > 0 else 1
        reshape_cd = s3 + s4 if s3 + s4 > 0 else 1

        zpairs_ab = ind_dict[probe_ab].shape[0]
        zpairs_cd = ind_dict[probe_cd].shape[0]

        # allocate array, since I will fill it in pieces
        cov_dict['g'][probe_ab, probe_cd]['4d'] = np.zeros(
            (nbl, nbl, zpairs_ab, zpairs_cd)
        )
        cov_dict['g'][probe_cd, probe_ab]['4d'] = np.zeros(
            (nbl, nbl, zpairs_cd, zpairs_ab)
        )

        for zij in range(zpairs_ab):
            for zkl in range(zpairs_cd):
                _, _, zi, zj = ind_dict[probe_ab][zij]
                _, _, zk, zl = ind_dict[probe_cd][zkl]

                # I think this casting is required by nmt objects...
                zi, zj, zk, zl = int(zi), int(zj), int(zk), int(zl)

                cov_l1l2 = nmt.gaussian_covariance(
                    cw=cw_dict[probe_ab, probe_cd][zi, zj, zk, zl],
                    spin_a1=0 if spin0 else s1,
                    spin_a2=0 if spin0 else s2,
                    spin_b1=0 if spin0 else s3,
                    spin_b2=0 if spin0 else s4,
                    cla1b1=cl_list_dict[f'{s1}{s3}'](zi, zk, spin0),
                    cla1b2=cl_list_dict[f'{s1}{s4}'](zi, zl, spin0),
                    cla2b1=cl_list_dict[f'{s2}{s3}'](zj, zk, spin0),
                    cla2b2=cl_list_dict[f'{s2}{s4}'](zj, zl, spin0),
                    coupled=coupled,
                    wa=wsp_dict[f'{s1}{s2}'][zi, zj],
                    wb=wsp_dict[f'{s3}{s4}'][zk, zl],
                )

                if not spin0:
                    cov_l1l2 = cov_l1l2.reshape([nell, reshape_ab, nell, reshape_cd])

                    # ! important note: I always take the [:, 0, :, 0] slice because
                    # ! I'm never interested in the off-diagonal elements of the spin
                    # ! blocks, but this is not the most general case
                    cov_l1l2 = cov_l1l2[:, 0, :, 0]

                # in the coupled case, namaster returns unbinned covariance matrices
                if coupled:
                    cov_l1l2 = sl.bin_2d_array_vectorized(cov_l1l2, **bin_cov_kw)

                cov_dict['g'][probe_ab, probe_cd]['4d'][:, :, zij, zkl] = cov_l1l2

    # * symmetrize and set to 0 the remaining probe blocks
    sl.symmetrize_and_fill_probe_blocks(
        cov_term_dict=cov_dict['g'],
        dim='4d',
        unique_probe_combs=unique_probe_combs,
        nonreq_probe_combs=nonreq_probe_combs,
        obs_space='harmonic',
        nbx=nbl,
        zbins=zbins,
        ind_dict=ind_dict,
        msg='NaMaster G cov: ',
    )

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


def _weight_per_bin(weight_maps, zi):
    """Accept either one shared mask (1D array)
    # or a per-bin container (list/dict/array with zbin axis first)"""
    if isinstance(weight_maps, np.ndarray) and weight_maps.ndim == 1:
        return weight_maps
    return weight_maps[zi]


def precompute_alms_healpy(
    corr_maps_gg: list,
    corr_maps_ll: list,
    weight_maps_gg: dict | np.ndarray,
    weight_maps_ll: dict | np.ndarray,
    lmax: int,
    n_iter: int = 3,
    remove_monopole: bool = True,
) -> tuple[list, list, list]:
    """Pre-compute masked alms for all zbins in one realization.

    Replaces the per-(zi,zj) SHT inside pcls_from_maps with 3*zbins SHTs done
    once before the pair loop. Per-pair cost then becomes O(lmax) via hp.alm2cl.

    Parameters
    ----------
    corr_maps_gg : list of length zbins, T maps
    corr_maps_ll : list of length zbins, each element is (Q_map, U_map)
    weight_maps_gg : Per-bin weight maps for spin-0 (GG), indexed by z bin.
    weight_maps_ll : Per-bin weight maps for spin-2 (LL), indexed by z bin.
    lmax         : maximum multipole

    Returns
    -------
    alms_T, alms_E, alms_B : lists of length zbins, alm arrays
    """
    # Keep peak memory low: process one z-bin at a time instead of materializing
    # masked copies for all bins at once.
    alms_T, alms_E, alms_B = [], [], []
    for zi, (map_T, (map_Q, map_U)) in enumerate(
        zip(corr_maps_gg, corr_maps_ll, strict=True)
    ):
        weight_gg = _weight_per_bin(weight_maps_gg, zi)
        weight_ll = _weight_per_bin(weight_maps_ll, zi)

        masked_T = map_T * weight_gg
        masked_Q = map_Q * weight_ll
        masked_U = map_U * weight_ll

        if remove_monopole:
            masked_T = hp.remove_monopole(masked_T)
            masked_Q = hp.remove_monopole(masked_Q)
            masked_U = hp.remove_monopole(masked_U)

        alms_T.append(hp.map2alm(masked_T, lmax=lmax, iter=n_iter))
        alm_E, alm_B = hp.map2alm_spin([masked_Q, masked_U], spin=2, lmax=lmax)
        alms_E.append(alm_E)
        alms_B.append(alm_B)

    return alms_T, alms_E, alms_B


def pcls_from_maps(  # fmt: skip
    zi: int,
    zj: int,
    f0,
    f2,
    coupled_cls,
    which_cls,
    w00_dict: dict,
    w02_dict: dict,
    w22_dict: dict,
    *,
    alms_T: list | None = None,
    alms_E: list | None = None,
    alms_B: list | None = None,
):
    """Compute binned pseudo-Cls for a single (zi, zj) pair.

    Both healpy anafast and nmt.compute_coupled_cell return the coupled
    ("pseudo") cls. Dividing by fsky gives a rough approximation of the true Cls.

    Fast branch (healpy):
        Pass pre-computed `alms_T`, `alms_E`, `alms_B` (indexed by zbin) to avoid
        re-doing the SHT on every (zi, zj) call. Pre-compute them once per
        realization with `precompute_alms_healpy` before the pair loop.
        Per-pair cost reduces from O(N_pix log N_pix) → O(lmax).

    Slow/fallback branch (healpy, no pre-computed alms):
        Computes all SHTs internally, same as the original implementation.

    NaMaster branch:
        f0 / f2 NmtField arrays must already be built outside the loop.
    """
    # ! compute (coupled) Cls with NaMaster
    if which_cls == 'namaster':
        pcl_tt = nmt.compute_coupled_cell(f0[zi], f0[zj])
        pcl_te = nmt.compute_coupled_cell(f0[zi], f2[zj])
        pcl_ee = nmt.compute_coupled_cell(f2[zi], f2[zj])

        # in this case, simply return the results
        if coupled_cls:
            cl_tt_out = pcl_tt[0]
            cl_te_out = pcl_te[0]
            cl_ee_out = pcl_ee[0]
        else:
            cl_tt_out = w00_dict[zi, zj].decouple_cell(pcl_tt)[0, :]
            cl_te_out = w02_dict[zi, zj].decouple_cell(pcl_te)[0, :]
            cl_ee_out = w22_dict[zi, zj].decouple_cell(pcl_ee)[0, :]

    # ! compute (coupled) Cls with healpy
    elif which_cls == 'healpy':
        # Fast branch: use pre-computed alms to compute cls
        pcl_tt = hp.alm2cl(alms_T[zi], alms_T[zj])
        pcl_te = hp.alm2cl(alms_T[zi], alms_E[zj])
        pcl_tb = hp.alm2cl(alms_T[zi], alms_B[zj])
        pcl_ee = hp.alm2cl(alms_E[zi], alms_E[zj])
        pcl_eb = hp.alm2cl(alms_E[zi], alms_B[zj])
        pcl_be = hp.alm2cl(alms_B[zi], alms_E[zj])
        pcl_bb = hp.alm2cl(alms_B[zi], alms_B[zj])

        if coupled_cls:
            cl_tt_out = pcl_tt
            cl_te_out = pcl_te
            cl_ee_out = pcl_ee
        else:
            cl_tt_out = w00_dict[zi, zj].decouple_cell(pcl_tt[None, :])[0, :]
            cl_te_out = w02_dict[zi, zj].decouple_cell(np.vstack([pcl_te, pcl_tb]))[
                0, :
            ]
            cl_ee_out = w22_dict[zi, zj].decouple_cell(
                np.vstack([pcl_ee, pcl_eb, pcl_be, pcl_bb])
            )[0, :]

    else:
        raise ValueError('which_cls must be namaster or healpy')

    return np.array(cl_tt_out), np.array(cl_te_out), np.array(cl_ee_out)


def sample_covariance( # fmt: skip
    cov_dict,
    cl_GG_unbinned, cl_LL_unbinned, cl_GL_unbinned,
    cl_BB_unbinned, cl_EB_unbinned, cl_TB_unbinned,
    nbl, zbins, weight_maps_gg, weight_maps_ll, nside, nreal, coupled_cls, 
    which_cls, nmt_bin_obj,
    w00_dict, w02_dict, w22_dict, lmax=None, fix_seed=True, n_iter=None, lite=True,
):  # fmt: skip
    if lmax is None:
        lmax = 3 * nside - 1

    if fix_seed:
        SEEDVALUE = np.arange(nreal)

    # instantiate arrays
    for probe_2tpl in cov_dict['g']:
        cov_dict['g'][probe_2tpl]['6d'] = np.zeros(
            (nbl, nbl, zbins, zbins, zbins, zbins)
        )

    # NmtField kwargs
    nmt_field_kw = {'n_iter': n_iter, 'lite': lite, 'lmax': lmax}

    # TODO use only independent z pairs
    sim_cl_GG = np.zeros((nreal, nbl, zbins, zbins))
    sim_cl_GL = np.zeros((nreal, nbl, zbins, zbins))
    sim_cl_LL = np.zeros((nreal, nbl, zbins, zbins))

    # 1. produce correlated maps
    print(
        f'Generating {nreal} maps for nside {nside} '
        f'and computing pseudo-cls with {which_cls}...'
    )

    # prepare cls list in ring ordering
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
    for i in tqdm(range(nreal)):
        if fix_seed:
            np.random.seed(SEEDVALUE[i])

        # Generate a set of alm given cls
        corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=lmax, new=True)
        assert len(corr_alms_tot) == zbins * 3

        # slice to select T, E, B
        corr_alms = corr_alms_tot[::3]
        corr_Elms_Blms = list(
            zip(corr_alms_tot[1::3], corr_alms_tot[2::3], strict=True)
        )

        # turn alms into (correlated) maps
        corr_maps_gg = [hp.alm2map(alm, nside, lmax=lmax) for alm in corr_alms]
        corr_maps_ll = [
            hp.alm2map_spin(alms=[Elm, Blm], nside=nside, spin=2, lmax=lmax)
            for (Elm, Blm) in corr_Elms_Blms
        ]

        # now we need to measure pseudo-Cls from the generated maps;
        # this can be done with either NaMaster or healpy.
        if which_cls == 'namaster':
            # nmt ingredients
            f0 = np.array(
                [
                    nmt.NmtField(weight_maps_gg[zi], [m], **nmt_field_kw)
                    for zi, (m) in enumerate(corr_maps_gg)
                ]
            )
            f2 = np.array(
                [
                    nmt.NmtField(weight_maps_ll[zi], [Q, U], **nmt_field_kw)
                    for zi, (Q, U) in enumerate(corr_maps_ll)
                ]
            )

            # hp ingredients
            alms_T = alms_E = alms_B = None
        else:
            # nmt ingredients
            f0, f2 = None, None

            # hp ingredients
            # ! this speeds up the computation by a lot:
            # hp.anafast = hp.map2alm (slow) + hp.alm2cl (fast).
            # hp.map2alm needs to be computed for each redshift bin, not each
            # redshift bin pair!

            # The function below does 3 things:
            # 1. mask the maps
            # 2. remove monopole (not sure if this is necessary)
            # 3. compute alms for T, E, B for each zbin
            alms_T, alms_E, alms_B = precompute_alms_healpy(
                corr_maps_gg=corr_maps_gg,
                corr_maps_ll=corr_maps_ll,
                weight_maps_gg=weight_maps_gg,
                weight_maps_ll=weight_maps_ll,
                lmax=lmax,
                n_iter=3,
                remove_monopole=True,
            )

        # Now we can compute the pseudo-Cls from the simulated maps.
        # As specified above, in the healpy case, we already computed the alms,
        # so this step is just hp.alm2cl, which is very fast.
        for zi, zj in zij_combinations:
            sim_cl_GG_ij, sim_cl_GL_ij, sim_cl_LL_ij = pcls_from_maps(
                zi=zi,
                zj=zj,
                f0=f0,
                f2=f2,
                coupled_cls=coupled_cls,
                which_cls=which_cls,
                w00_dict=w00_dict,
                w02_dict=w02_dict,
                w22_dict=w22_dict,
                alms_T=alms_T,
                alms_E=alms_E,
                alms_B=alms_B,
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
    sim_cls_to_sample_cov(cov_dict, sim_cl_GG, sim_cl_GL, sim_cl_LL, nbl, zbins)

    return sim_cl_GG, sim_cl_GL, sim_cl_LL


def sample_covariance_parallel(
    cov_dict: cd.FrozenDict,
    cl_GG_unbinned: np.ndarray,
    cl_LL_unbinned: np.ndarray,
    cl_GL_unbinned: np.ndarray,
    cl_BB_unbinned: np.ndarray,
    cl_EB_unbinned: np.ndarray,
    cl_TB_unbinned: np.ndarray,
    nbl: int,
    zbins: int,
    weight_maps_gg: np.ndarray,
    weight_maps_ll: np.ndarray,
    nside: int,
    nreal: int,
    coupled_cls: bool,
    which_cls: str,
    nmt_bin_obj: nmt.NmtBin,
    wsp_path_template: str,
    lmax: int | float,
    n_jobs: int,
    fix_seed=True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the ensemble covariance from a set of simulated power spectra"""

    SEEDVALUE = np.arange(nreal) if fix_seed else [None] * nreal

    # instantiate arrays in cov_dict
    for probe_2tpl in cov_dict['g']:
        cov_dict['g'][probe_2tpl]['6d'] = np.zeros(
            (nbl, nbl, zbins, zbins, zbins, zbins)
        )

    # TODO use only independent z pairs
    sim_cl_GG = np.zeros((nreal, nbl, zbins, zbins))
    sim_cl_GL = np.zeros((nreal, nbl, zbins, zbins))
    sim_cl_LL = np.zeros((nreal, nbl, zbins, zbins))

    # * Step I: produce (correlated) maps
    print(
        f'Generating {nreal} maps for nside {nside} '
        f'and computing pseudo-cls with {which_cls}...'
    )

    # Ia: prepare Cl list in ring ordering for synalm
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

    # Extract bin edges from nmt_bin_obj (picklable arrays)
    # NmtBin objects are SWIG objects and cannot be pickled for parallel execution
    # This is the same reason why we pass workspace paths instead of objects
    n_bands = nmt_bin_obj.get_n_bands()
    ell_min_edges = np.array([nmt_bin_obj.get_ell_min(i) for i in range(n_bands)])
    ell_max_edges = np.array([nmt_bin_obj.get_ell_max(i) + 1 for i in range(n_bands)])

    # Workers are terminated when the context manager exits cleanly,
    # avoiding zombie processes
    with parallel_backend('loky'), Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(_compute_one_realization)(
                seed=SEEDVALUE[i],
                cl_ring_big_list=cl_ring_big_list,
                lmax=lmax,
                nside=nside,
                zbins=zbins,
                weight_maps_gg=weight_maps_gg,
                weight_maps_ll=weight_maps_ll,
                coupled_cls=coupled_cls,
                which_cls=which_cls,
                wsp_path_template=wsp_path_template,
                nbl=nbl,
                ell_min_edges=ell_min_edges,
                ell_max_edges=ell_max_edges,
            )
            for i in range(nreal)
        )

    sim_cl_GG = np.stack([r[0] for r in results])  # (nreal, nbl, zbins, zbins)
    sim_cl_GL = np.stack([r[1] for r in results])
    sim_cl_LL = np.stack([r[2] for r in results])

    # * Step II: compute sample covariance
    sim_cls_to_sample_cov(cov_dict, sim_cl_GG, sim_cl_GL, sim_cl_LL, nbl, zbins)

    return sim_cl_GG, sim_cl_GL, sim_cl_LL


def sim_cls_to_sample_cov(cov_dict, sim_cl_GG, sim_cl_GL, sim_cl_LL, nbl, zbins):

    zijkl_combinations = list(itertools.product(range(zbins), repeat=4))

    # TODO only loop over required probes!
    for zi, zj, zk, zl in zijkl_combinations:
        # you could also cut the mixed cov terms,
        # but for cross-redshifts it becomes a bit tricky
        kwargs = {'rowvar': False, 'bias': False}
        cov_dict['g']['LL', 'LL']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_LL[:, :, zi, zj], sim_cl_LL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['LL', 'GL']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_LL[:, :, zi, zj], sim_cl_GL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['LL', 'GG']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_LL[:, :, zi, zj], sim_cl_GG[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['GL', 'LL']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GL[:, :, zi, zj], sim_cl_LL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['GL', 'GL']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GL[:, :, zi, zj], sim_cl_GL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['GL', 'GG']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GL[:, :, zi, zj], sim_cl_GG[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['GG', 'LL']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GG[:, :, zi, zj], sim_cl_LL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['GG', 'GL']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GG[:, :, zi, zj], sim_cl_GL[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]
        cov_dict['g']['GG', 'GG']['6d'][:, :, zi, zj, zk, zl] = np.cov(
            sim_cl_GG[:, :, zi, zj], sim_cl_GG[:, :, zk, zl], **kwargs
        )[:nbl, nbl:]


def _compute_one_realization(
    seed,
    cl_ring_big_list,
    lmax,
    nside,
    zbins,
    weight_maps_gg,
    weight_maps_ll,
    coupled_cls,
    which_cls,
    wsp_path_template,
    nbl,
    ell_min_edges,
    ell_max_edges,
):
    """Worker: one realization → Cls only. Maps are discarded before return."""

    # Set seed for reproducibility (must be set inside each worker for
    # parallel execution)
    np.random.seed(seed)

    # Load workspaces inside worker (NmtWorkspace objects are not picklable)
    # [Note]: this is only necessary if we want to decouple the Cls!
    w00_dict, w02_dict, w22_dict = {}, {}, {}
    if not coupled_cls:
        for zi, zj in itertools.product(range(zbins), repeat=2):
            w00_dict[zi, zj] = nmt.NmtWorkspace()
            w02_dict[zi, zj] = nmt.NmtWorkspace()
            w22_dict[zi, zj] = nmt.NmtWorkspace()
            w00_dict[zi, zj].read_from(wsp_path_template.format(0, 0, zi=zi, zj=zj))
            w02_dict[zi, zj].read_from(wsp_path_template.format(0, 2, zi=zi, zj=zj))
            w22_dict[zi, zj].read_from(wsp_path_template.format(2, 2, zi=zi, zj=zj))

    # Reconstruct NmtBin object from edges (nmt_bin_obj objects are not picklable)
    nmt_bin_obj = nmt.NmtBin.from_edges(ell_min_edges, ell_max_edges)

    # ! 1. Generate alms from input Cls
    corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=lmax, new=True)
    corr_alms_T = corr_alms_tot[::3]
    corr_alms_E_B = list(zip(corr_alms_tot[1::3], corr_alms_tot[2::3], strict=True))

    # ! 1. Generate (correlated) maps from alms
    corr_maps_gg = [hp.alm2map(alm, nside, lmax=lmax) for alm in corr_alms_T]
    corr_maps_ll = [
        hp.alm2map_spin([E, B], nside=nside, spin=2, lmax=lmax)
        for E, B in corr_alms_E_B
    ]
    # free alms
    del corr_alms_tot, corr_alms_T, corr_alms_E_B

    if which_cls == 'namaster':
        nmt_field_kw = {'n_iter': None, 'lite': True, 'lmax': lmax}
        f0 = np.array(
            [
                nmt.NmtField(_weight_per_bin(weight_maps_gg, zi), [m], **nmt_field_kw)
                for zi, m in enumerate(corr_maps_gg)
            ]
        )
        f2 = np.array(
            [
                nmt.NmtField(
                    _weight_per_bin(weight_maps_ll, zi), [q, u], **nmt_field_kw
                )
                for zi, (q, u) in enumerate(corr_maps_ll)
            ]
        )
        alms_T = alms_E = alms_B = None
    else:
        f0, f2 = None, None
        # ! Mask each map (there are zbins of them) and compute ("masked") alms
        alms_T, alms_E, alms_B = precompute_alms_healpy(
            corr_maps_gg=corr_maps_gg,
            corr_maps_ll=corr_maps_ll,
            weight_maps_gg=weight_maps_gg,
            weight_maps_ll=weight_maps_ll,
            lmax=lmax,
        )

    # free maps
    del corr_maps_gg, corr_maps_ll

    # ! Compute Cls *for all (zi, zj) pairs*
    # [Note]: these are coupled by default (they are computed from the masked maps),
    # but they can be decoupled.
    # [Note]: the healpy branch should be much faster now that the alms have been
    # pre-computed for each bin, rather than (uselessly) re-computed for each bin
    # pair inside the loop below pcls_from_maps.
    cl_gg = np.zeros((nbl, zbins, zbins))
    cl_gl = np.zeros((nbl, zbins, zbins))
    cl_ll = np.zeros((nbl, zbins, zbins))

    for zi, zj in itertools.product(range(zbins), repeat=2):
        gg, gl, ll = pcls_from_maps(
            zi=zi,
            zj=zj,
            f0=f0,
            f2=f2,
            coupled_cls=coupled_cls,
            which_cls=which_cls,
            w00_dict=w00_dict,
            w02_dict=w02_dict,
            w22_dict=w22_dict,
            alms_T=alms_T,
            alms_E=alms_E,
            alms_B=alms_B,
        )
        if len(gg) != nbl:
            gg = nmt_bin_obj.bin_cell(gg)
            gl = nmt_bin_obj.bin_cell(gl)
            ll = nmt_bin_obj.bin_cell(ll)

        cl_gg[:, zi, zj] = gg
        cl_gl[:, zi, zj] = gl
        cl_ll[:, zi, zj] = ll

    # shape: (nbl, zbins, zbins)
    return cl_gg, cl_gl, cl_ll


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
        'The number of elements in the row should be equal to the number of redshift '
        'bins times the number of spectra types.'
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
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Temperature map, Q-mode
        polarization map, U-mode polarization map.

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
        nmt.NmtField, nmt.NmtField: NmtField objects for the temperature and
        polarization maps.

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
        corr_Elms_Blms = list(
            zip(corr_alms_tot[1::3], corr_alms_tot[2::3], strict=True)
        )

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
        mask_obj_gg: mask_utils.Mask,
        mask_obj_ll: mask_utils.Mask,
    ):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg

        self.ell_obj = ell_obj
        self.mask_obj_gg = mask_obj_gg
        self.mask_obj_ll = mask_obj_ll

        self.zbins = pvt_cfg['zbins']
        self.n_probes = pvt_cfg['n_probes']
        self.nonreq_probe_combs = pvt_cfg['nonreq_probe_combs_hs']
        self.symmetrize_output_dict = pvt_cfg['symmetrize_output_dict']
        self.ind_dict = pvt_cfg['ind_dict']
        self.coupled_cov = cfg['covariance']['cov_type'] == 'coupled'
        self.output_path = self.cfg['misc']['output_path']
        self.load_cached_wsp = self.cfg['covariance']['load_cached_nmt_workspaces']

        self.footprint_gg = self.mask_obj_gg.footprint
        self.footprint_ll = self.mask_obj_ll.footprint
        self.weight_maps_gg = self.mask_obj_gg.weight_maps
        self.weight_maps_ll = self.mask_obj_ll.weight_maps

        self.use_weight_maps_ll = self.weight_maps_ll is not None
        self.use_weight_maps_gg = self.weight_maps_gg is not None
        # just for readability
        self.use_footprint_gg = not self.use_weight_maps_gg
        self.use_footprint_ll = not self.use_weight_maps_ll

        if self.use_footprint_gg:
            self.weight_maps_gg = np.tile(self.footprint_gg[None, :], (self.zbins, 1))
        if self.use_footprint_ll:
            self.weight_maps_ll = np.tile(self.footprint_ll[None, :], (self.zbins, 1))

        self.wsp_fname = 'wsp_s{:d}s{:d}_zi{zi:d}zj{zj:d}.fits'
        self.cw_fname = 'cw_s{:d}s{:d}s{:d}s{:d}_zi{zi:d}zj{zj:d}zk{zk:d}zl{zl:d}.fits'
        self.cache_path = f'{self.output_path}/cache/nmt'

        # instantiate cov dict
        # ! note that this class only computes
        #   - g term
        #   - g all hs probe combinations (no 3x2pt!!)
        #   - 6d dim

        self.req_terms = ['g']
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_hs_2d']
        dims = ['6d', '4d']

        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='harmonic')
            for probe in pvt_cfg['req_probe_combs_hs_2d']
        ]
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

        # check on lmax and NSIDE
        _lmax = self.ell_obj.ell_max_GC
        if _lmax >= 3 * self.mask_obj_gg.nside_cfg - 1:
            warnings.warn(
                f'lmax = {_lmax} >= 3 * NSIDE - 1 = {3 * self.mask_obj_gg.nside_cfg - 1}\n'
                f'(NSIDE = {self.mask_obj_gg.nside_cfg}) for probe GC. '
                'You should probably increase NSIDE or decrease lmax ',
                stacklevel=2,
            )
        _lmax = self.ell_obj.ell_max_WL
        if _lmax >= 3 * self.mask_obj_ll.nside_cfg - 1:
            warnings.warn(
                f'lmax = {_lmax} >= 3 * NSIDE - 1 = {3 * self.mask_obj_ll.nside_cfg - 1}\n'
                f'(NSIDE = {self.mask_obj_ll.nside_cfg}) for probe WL. '
                'You should probably increase NSIDE or decrease lmax ',
                stacklevel=2,
            )

        self.cl_3x2pt_unb_5d = _UNSET
        self.ells_3x2pt_unb = _UNSET
        self.nbl_3x2pt_unb = _UNSET

    def build_fields(self, ell_max_eff):
        # TODO XXX make this also dependent on the selected probes!
        self.f0_dict, self.f2_dict = {}, {}
        print('\nComputing namaster fields...')

        # in case only the footprint is provided, the fields can be computed once
        if not self.use_weight_maps_gg:
            self.f0_ftp = nmt.NmtField(
                mask=self.footprint_gg, maps=None, spin=0, lite=True, lmax=ell_max_eff
            )
        if not self.use_weight_maps_ll:
            self.f2_ftp = nmt.NmtField(
                mask=self.footprint_ll, maps=None, spin=2, lite=True, lmax=ell_max_eff
            )

        # now, either compute per-bin fields from weight maps, or just assign the
        # same field (from the footprint) to all bins (i.e., to all keys in the dict)
        for zi in tqdm(range(self.zbins)):
            if self.use_weight_maps_gg:
                self.f0_dict[zi] = nmt.NmtField(
                    mask=self.weight_maps_gg[zi],
                    maps=None,
                    spin=0,
                    lite=True,
                    lmax=ell_max_eff,
                )
            else:
                self.f0_dict[zi] = self.f0_ftp

            if self.use_weight_maps_ll:
                self.f2_dict[zi] = nmt.NmtField(
                    mask=self.weight_maps_ll[zi],
                    maps=None,
                    spin=2,
                    lite=True,
                    lmax=ell_max_eff,
                )
            else:
                self.f2_dict[zi] = self.f2_ftp

    def build_wsp(self):
        if self.load_cached_wsp:
            print(
                '\nLoading namaster workspaces and coupling matrices from\n'
                f'{self.cache_path}...'
            )
        else:
            print('\nComputing namaster workspaces and coupling matrices...')

        self.w00_dict, self.w02_dict, self.w22_dict = {}, {}, {}

        # ! 1. If no weight maps are passed, one wsp is sufficient
        if not self.load_cached_wsp:
            if not self.use_weight_maps_gg:
                self.w00_ftp = nmt.NmtWorkspace()
                self.w00_ftp.compute_coupling_matrix(
                    self.f0_ftp, self.f0_ftp, self.nmt_bin_obj
                )
            if (not self.use_weight_maps_ll) and (not self.use_weight_maps_gg):
                self.w02_ftp = nmt.NmtWorkspace()
                self.w02_ftp.compute_coupling_matrix(
                    self.f0_ftp, self.f2_ftp, self.nmt_bin_obj
                )
            if not self.use_weight_maps_ll:
                self.w22_ftp = nmt.NmtWorkspace()
                self.w22_ftp.compute_coupling_matrix(
                    self.f2_ftp, self.f2_ftp, self.nmt_bin_obj
                )

        # ! Regardless of the presence of weight maps, build wsp dictionaries
        # ! for all bin pairs, either by computing them (if weight maps are present)
        # ! or by assigning the same wsp (if only the footprint is present),
        # ! or by loading from cache
        # TODO XXX this can be made probe-dependent as for cw, and looped only over
        # TODO XXX unique pairs
        for zi, zj in tqdm(self.zij_cross_combs):
            if self.load_cached_wsp:
                w00_name = self.wsp_fname.format(0, 0, zi=zi, zj=zj)
                w02_name = self.wsp_fname.format(0, 2, zi=zi, zj=zj)
                w22_name = self.wsp_fname.format(2, 2, zi=zi, zj=zj)
                self.w00_dict[zi, zj] = nmt.NmtWorkspace()
                self.w02_dict[zi, zj] = nmt.NmtWorkspace()
                self.w22_dict[zi, zj] = nmt.NmtWorkspace()
                self.w00_dict[zi, zj].read_from(f'{self.cache_path}/{w00_name}')
                self.w02_dict[zi, zj].read_from(f'{self.cache_path}/{w02_name}')
                self.w22_dict[zi, zj].read_from(f'{self.cache_path}/{w22_name}')

            else:
                if self.use_weight_maps_gg:
                    self.w00_dict[zi, zj] = nmt.NmtWorkspace()
                    self.w00_dict[zi, zj].compute_coupling_matrix(
                        self.f0_dict[zi], self.f0_dict[zj], self.nmt_bin_obj
                    )
                else:
                    self.w00_dict[zi, zj] = self.w00_ftp

                if self.use_weight_maps_gg or self.use_weight_maps_ll:
                    self.w02_dict[zi, zj] = nmt.NmtWorkspace()
                    self.w02_dict[zi, zj].compute_coupling_matrix(
                        self.f0_dict[zi], self.f2_dict[zj], self.nmt_bin_obj
                    )
                else:
                    self.w02_dict[zi, zj] = self.w02_ftp

                if self.use_weight_maps_ll:
                    self.w22_dict[zi, zj] = nmt.NmtWorkspace()
                    self.w22_dict[zi, zj].compute_coupling_matrix(
                        self.f2_dict[zi], self.f2_dict[zj], self.nmt_bin_obj
                    )
                else:
                    self.w22_dict[zi, zj] = self.w22_ftp

    def build_cw(self, unique_probe_combs):
        """
        Builds the covariance workspace objects for all required probe combinations and
        redshift bins. Some clarifications about this:
        * If only the footprint is provided, the coupling coefficients are the same for
        all bin combinations, so we can avoid looping over all bins
        * If only the footprint is provided and the ll and gg footprints match,
        the coupling coefficients are the same for all probe combinations, so we can
        avopid looping over all probe combinations as well

        """

        self.cw_dict = {}

        # no need for cw if we want the sample covariance
        if self.cfg['sample_covariance']['compute_sample_cov']:
            return

        if self.load_cached_wsp:
            print(
                '\nLoading cov workspace coupling coefficients from\n'
                f'{self.cache_path}...'
            )
        else:
            print(
                '\nComputing cov workspace coupling coefficients '
                '(this may take a while)...'
            )

        cw_dict_ftp = {}

        # ! Case 1: if the footprint is used for all probes, and the masks are equal
        use_footprint_allprobes = self.use_footprint_gg and self.use_footprint_ll
        footprint_is_equal = np.array_equal(self.footprint_gg, self.footprint_ll)
        if use_footprint_allprobes and footprint_is_equal and not self.load_cached_wsp:
            with sl.timer('case 1: '):
                cw_ftp = nmt.NmtCovarianceWorkspace()
                cw_ftp.compute_coupling_coefficients(
                    self.f0_ftp, self.f0_ftp, self.f0_ftp, self.f0_ftp
                )

        # ! Case 2: if the footprint is used for all probes, but the masks are not equal
        # ! in this case we have to loop over the probes, but not over the bins
        for probe_abcd in unique_probe_combs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            if (
                use_footprint_allprobes
                and not footprint_is_equal
                and not self.load_cached_wsp
            ):
                _f1 = self.f0_ftp if probe_ab[0] == 'G' else self.f2_ftp
                _f2 = self.f0_ftp if probe_ab[1] == 'G' else self.f2_ftp
                _f3 = self.f0_ftp if probe_cd[0] == 'G' else self.f2_ftp
                _f4 = self.f0_ftp if probe_cd[1] == 'G' else self.f2_ftp
                cw_dict_ftp[probe_ab, probe_cd] = nmt.NmtCovarianceWorkspace()
                cw_dict_ftp[probe_ab, probe_cd].compute_coupling_coefficients(
                    _f1, _f2, _f3, _f4
                )
            elif (
                use_footprint_allprobes
                and footprint_is_equal
                and not self.load_cached_wsp
            ):
                cw_dict_ftp[probe_ab, probe_cd] = cw_ftp

        # the last branch is accessed only when use_footprint_allprobes is True,
        # but in the particular case of auto-correlations I can look just at the
        # specific key:
        if self.use_footprint_gg and not self.load_cached_wsp:
            cw_dict_ftp['GG', 'GG'] = nmt.NmtCovarianceWorkspace()
            cw_dict_ftp['GG', 'GG'].compute_coupling_coefficients(
                self.f0_ftp, self.f0_ftp, self.f0_ftp, self.f0_ftp
            )
        if self.use_footprint_ll and not self.load_cached_wsp:
            cw_dict_ftp['LL', 'LL'] = nmt.NmtCovarianceWorkspace()
            cw_dict_ftp['LL', 'LL'].compute_coupling_coefficients(
                self.f2_ftp, self.f2_ftp, self.f2_ftp, self.f2_ftp
            )

        # ! Case 3: some probes require weight maps, and/or
        # ! the footprints are different (I'm not checking whether the weight maps
        # ! are different...)
        # TODO XXX leverage the remaining symmetry for other slow parts, e.g. cNG!!
        for probe_abcd in tqdm(unique_probe_combs):
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            self.cw_dict[probe_ab, probe_cd] = {}
            zpairs_ab = self.ind_dict[probe_ab].shape[0]
            zpairs_cd = self.ind_dict[probe_cd].shape[0]
            is_auto = probe_ab == probe_cd

            # is there at least one probe that requires weight maps? This is the
            # "general case" (e.g.: weight maps for shear, footprint for clustering:
            # in this case, cov_GGGL can't be taken from the ones precomputed above,
            # since it contains one "bin-dependent" field
            use_weight_maps = self.use_weight_maps_ll or self.use_weight_maps_gg
            # in the special auto-covariance case, I can take just the corresponding
            # boolean flag:
            if probe_abcd == 'LLLL':
                use_weight_maps = self.use_weight_maps_ll
            elif probe_abcd == 'GGGG':
                use_weight_maps = self.use_weight_maps_gg

            # symmetry note: for auto-blocks (LL, LL), (GL, GL), (GG, GG),
            # the symmetry (12) <-> (34) gives
            # cov[GL, GL][i, j, k, l] = cov[GL, GL][k, l, i, j]
            # which is different from the cross-probe blocks, in which you'r also
            # have to exchange the probes (GL <-> LL), giving
            # cov[LL, GL][i, j, k, l] = cov[GL, LL][k, l, i, j]
            # this is implemented for by only looping over the upper triangle of
            # (zij, zkl) for the auto-blocks (and filling the lower triangle
            # by reference)
            for zij in range(zpairs_ab):
                zkl_range = range(zij, zpairs_cd) if is_auto else range(zpairs_cd)
                for zkl in zkl_range:
                    _, _, zi, zj = self.ind_dict[probe_ab][zij]
                    _, _, zk, zl = self.ind_dict[probe_cd][zkl]
                    zi, zj, zk, zl = int(zi), int(zj), int(zk), int(zl)

                    # ! Case 3: compute from weight maps (probe- and bin-dependent)
                    if use_weight_maps and (not self.load_cached_wsp):
                        _f1 = (
                            self.f0_dict[zi] if probe_ab[0] == 'G' else self.f2_dict[zi]
                        )
                        _f2 = (
                            self.f0_dict[zj] if probe_ab[1] == 'G' else self.f2_dict[zj]
                        )
                        _f3 = (
                            self.f0_dict[zk] if probe_cd[0] == 'G' else self.f2_dict[zk]
                        )
                        _f4 = (
                            self.f0_dict[zl] if probe_cd[1] == 'G' else self.f2_dict[zl]
                        )
                        self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl] = (
                            nmt.NmtCovarianceWorkspace()
                        )
                        self.cw_dict[probe_ab, probe_cd][
                            zi, zj, zk, zl
                        ].compute_coupling_coefficients(_f1, _f2, _f3, _f4)

                    # ! Case 1-2: get from previous computation
                    elif (not use_weight_maps) and (not self.load_cached_wsp):
                        self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl] = cw_dict_ftp[
                            probe_ab, probe_cd
                        ]

                    # ! Case 4: load cached
                    elif self.load_cached_wsp:
                        spin_list = [0 if p == 'G' else 2 for p in probe_abcd]
                        cw_name = self.cw_fname.format(
                            *spin_list, zi=zi, zj=zj, zk=zk, zl=zl
                        )
                        self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl] = (
                            nmt.NmtCovarianceWorkspace()
                        )
                        self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl].read_from(
                            f'{self.cache_path}/{cw_name}'
                        )

                    # fill lower triangle by reference for auto-blocks
                    # (if zkl > zij, we are in the upper triangle,
                    # so swapping them gives the corresponding point in the lower
                    # triangle)
                    # TODO XXX this can be done at the nmt cov level and deleted here!
                    if is_auto and zkl > zij:
                        self.cw_dict[probe_ab, probe_cd][zk, zl, zi, zj] = self.cw_dict[
                            probe_ab, probe_cd
                        ][zi, zj, zk, zl]

    def save_to_cache(self, unique_probe_combs):

        # if workspaces are already laoded from cache, do not save them again
        if self.load_cached_wsp:
            return

        # else, create folder if absent and save everything
        os.makedirs(f'{self.cache_path}', exist_ok=True)
        print('\nSaving namaster workspaces in cache...')
        for zi, zj in tqdm(self.zij_cross_combs):
            w00_name = self.wsp_fname.format(0, 0, zi=zi, zj=zj)
            w02_name = self.wsp_fname.format(0, 2, zi=zi, zj=zj)
            w22_name = self.wsp_fname.format(2, 2, zi=zi, zj=zj)
            self.w00_dict[zi, zj].write_to(f'{self.cache_path}/{w00_name}')
            self.w02_dict[zi, zj].write_to(f'{self.cache_path}/{w02_name}')
            self.w22_dict[zi, zj].write_to(f'{self.cache_path}/{w22_name}')

        # if the sample covariance is required, no cw are computed,
        # so no need to save them
        if self.cfg['sample_covariance']['compute_sample_cov']:
            return

        print('\nSaving covariance workspaces in cache...')
        for probe_abcd in tqdm(unique_probe_combs):
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            zpairs_ab = self.ind_dict[probe_ab].shape[0]
            zpairs_cd = self.ind_dict[probe_cd].shape[0]
            is_auto = probe_ab == probe_cd
            for zij in range(zpairs_ab):
                zkl_range = range(zij, zpairs_cd) if is_auto else range(zpairs_cd)
                for zkl in zkl_range:
                    _, _, zi, zj = self.ind_dict[probe_ab][zij]
                    _, _, zk, zl = self.ind_dict[probe_cd][zkl]
                    zi, zj, zk, zl = int(zi), int(zj), int(zk), int(zl)
                    spin_list = [0 if p == 'G' else 2 for p in probe_abcd]
                    cw_name = self.cw_fname.format(
                        *spin_list, zi=zi, zj=zj, zk=zk, zl=zl
                    )
                    self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl].write_to(
                        f'{self.cache_path}/{cw_name}'
                    )

    def build_psky_cov(self):
        # TODO again, here I'm using 3x2pt = GC
        # 1. ell binning
        # shorten names for brevity
        self.nmt_bin_obj = self.ell_obj.nmt_bin_obj_GC
        fsky_ll = self.mask_obj_ll.fsky_footprint
        fsky_gg = self.mask_obj_gg.fsky_footprint
        unique_probe_combs = self.pvt_cfg['unique_probe_combs_hs']

        self.zij_auto_combs = list(combinations_with_replacement(range(self.zbins), 2))
        self.zij_cross_combs = list(itertools.product(range(self.zbins), repeat=2))
        self.zijkl_combs = list(itertools.product(range(self.zbins), repeat=4))

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

        cl_gg_4covnmt = self.cl_3x2pt_unb_5d[1, 1, :, :, :].copy()
        cl_gl_4covnmt = self.cl_3x2pt_unb_5d[1, 0, :, :, :].copy()
        cl_ll_4covnmt = self.cl_3x2pt_unb_5d[0, 0, :, :, :].copy()

        # ! 1. Create field objects
        # ! (there will be no maps associated to the fields)
        # TODO maks=None (as in the example) or maps=[mask]? I think None

        self.build_fields(ell_max_eff)
        self.build_wsp()
        self.build_cw(unique_probe_combs)

        # if the coupled covariance is required, I'll later need to convolve the
        # non-Gaussian terms. For this, I'll need the binned mode coupling matrices
        # (mcm), which I store in self
        # TODO XXX again, only loop over the unique zpairs

        # TODO XXX in general, I should definetly create some function to declutter the
        # TODO XXX "main"...
        if (
            self.coupled_cov
            and (self.cfg['covariance']['SSC'] or self.cfg['covariance']['cNG'])
        ) or self.cfg['covariance']['save_mcms']:
            print('\nComputing and binning mode coupling matrices...')
            mcm_tt_unb, mcm_te_unb, mcm_ee_unb = {}, {}, {}
            self.mcm_tt_binned, self.mcm_et_binned = {}, {}
            self.mcm_te_binned, self.mcm_ee_binned = {}, {}
            for zi, zj in tqdm(self.zij_cross_combs):
                # extract only the relevant blocks
                mcm_tt_unb[zi, zj] = self.w00_dict[zi, zj].get_coupling_matrix()[
                    :nbl_unb, :nbl_unb
                ]
                mcm_te_unb[zi, zj] = self.w02_dict[zi, zj].get_coupling_matrix()[
                    :nbl_unb, :nbl_unb
                ]
                mcm_ee_unb[zi, zj] = self.w22_dict[zi, zj].get_coupling_matrix()[
                    :nbl_unb, :nbl_unb
                ]

                # bin (and store in self)
                self.mcm_tt_binned[zi, zj] = bin_mcm(
                    mcm_tt_unb[zi, zj], self.nmt_bin_obj
                )
                self.mcm_te_binned[zi, zj] = bin_mcm(
                    mcm_te_unb[zi, zj], self.nmt_bin_obj
                )
                self.mcm_ee_binned[zi, zj] = bin_mcm(
                    mcm_ee_unb[zi, zj], self.nmt_bin_obj
                )

            if self.cfg['covariance']['save_mcms']:
                np.savez(
                    f'{self.output_path}/mode_coupling_matrices.npz',
                    mcm_gg_unbinned=mcm_tt_unb,
                    mcm_gl_unbinned=mcm_te_unb,
                    mcm_ll_unbinned=mcm_ee_unb,
                    mcm_gg_binned=self.mcm_tt_binned,
                    mcm_lg_binned=self.mcm_et_binned,
                    mcm_gl_binned=self.mcm_te_binned,
                    mcm_ll_binned=self.mcm_ee_binned,
                )
                print(f'\nMode coupling matrices saved in {self.output_path}')

        # if you want to use the iNKA, the cls to be passed are the coupled ones
        # divided by fsky
        if self.cfg['precision']['use_iNKA']:
            # TODO XXX this could be made more efficient by only looping over the auto-combs
            # TODO XXX for ll and gg
            for zi, zj in self.zij_cross_combs:
                list_gg = [self.cl_3x2pt_unb_5d[1, 1, :, zi, zj]]
                list_gl = [
                    self.cl_3x2pt_unb_5d[1, 0, :, zi, zj],
                    np.zeros_like(self.cl_3x2pt_unb_5d[1, 0, :, zi, zj]),
                ]
                list_ll = [
                    self.cl_3x2pt_unb_5d[0, 0, :, zi, zj],
                    np.zeros_like(self.cl_3x2pt_unb_5d[0, 0, :, zi, zj]),
                    np.zeros_like(self.cl_3x2pt_unb_5d[0, 0, :, zi, zj]),
                    np.zeros_like(self.cl_3x2pt_unb_5d[0, 0, :, zi, zj]),
                ]
                # TODO the denominator should be the product of the masks?
                cl_gg_4covnmt[:, zi, zj] = (
                    self.w00_dict[zi, zj].couple_cell(list_gg)[0] / fsky_gg
                )
                cl_gl_4covnmt[:, zi, zj] = (
                    self.w02_dict[zi, zj].couple_cell(list_gl)[0] / fsky_ll
                )
                cl_ll_4covnmt[:, zi, zj] = (
                    self.w22_dict[zi, zj].couple_cell(list_ll)[0] / fsky_ll
                )

        # add noise to spectra to compute NMT cov
        cl_tt_4covnmt = cl_gg_4covnmt + self.noise_3x2pt_unb_5d[1, 1, :, :, :]
        cl_te_4covnmt = cl_gl_4covnmt + self.noise_3x2pt_unb_5d[1, 0, :, :, :]
        cl_ee_4covnmt = cl_ll_4covnmt + self.noise_3x2pt_unb_5d[0, 0, :, :, :]
        cl_tb_4covnmt = np.zeros_like(cl_tt_4covnmt)
        cl_eb_4covnmt = np.zeros_like(cl_tt_4covnmt)
        cl_bb_4covnmt = np.zeros_like(cl_tt_4covnmt)

        # ! Finally, compute covariance
        if self.cfg['covariance']['partial_sky_method'] == 'NaMaster':
            coupled_str = self.cfg['covariance']['cov_type']
            spin0_str = ' spin0' if self.cfg['precision']['spin0'] else ''

            # the nmt_gaussian_cov_opt functions modifies
            # cov_dict in-place, so no need to capture any return value
            with sl.timer(
                f'\nComputing {coupled_str}{spin0_str} partial-sky '
                'Gaussian covariance...'
            ):
                nmt_gaussian_cov(
                    cov_dict=self.cov_dict,
                    spin0=self.cfg['precision']['spin0'],
                    cl_tt=cl_tt_4covnmt,
                    cl_te=cl_te_4covnmt,
                    cl_ee=cl_ee_4covnmt,
                    cl_tb=cl_tb_4covnmt,
                    cl_eb=cl_eb_4covnmt,
                    cl_bb=cl_bb_4covnmt,
                    nbl=nbl_eff,
                    zbins=self.zbins,
                    ind_dict=self.ind_dict,
                    cw_dict=self.cw_dict,
                    w00_dict=self.w00_dict,
                    w02_dict=self.w02_dict,
                    w22_dict=self.w22_dict,
                    unique_probe_combs=unique_probe_combs,
                    nonreq_probe_combs=self.nonreq_probe_combs,
                    coupled=self.coupled_cov,
                    ells_in=ells_unb,
                    ells_out=ells_eff,
                    ells_out_edges=ells_eff_edges,
                    weights=None,
                    which_binning='sum',
                )

            # ! convert probe blocks from 4d to 6d and remove the 4d ones
            # ! to ensure compatibility with the code downstream
            # ! (the harmonic-space Gaussian covariance is computed in 6d!)
            for probe_2tpl in self.cov_dict['g']:
                probe_ab, probe_cd = probe_2tpl

                # sanity check: no 6d covs should be assigned yet
                assert self.cov_dict['g'][probe_2tpl]['6d'] is None, (
                    f'self.cov_dict[g][{probe_2tpl}][6d] is not None before assignment!'
                )

                self.cov_dict['g'][probe_2tpl]['6d'] = sl.cov_4D_to_6D_blocks(
                    cov_4D=self.cov_dict['g'][probe_2tpl]['4d'],
                    nbl=self.ell_obj.nbl_3x2pt,
                    zbins=self.zbins,
                    ind_ab=self.ind_dict[probe_ab],
                    ind_cd=self.ind_dict[probe_cd],
                    symmetrize_output_ab=self.symmetrize_output_dict[probe_ab],
                    symmetrize_output_cd=self.symmetrize_output_dict[probe_cd],
                )

            # now reset the 4d covs to avoid confusion
            for probe_2tpl in self.cov_dict['g']:
                self.cov_dict['g'][probe_2tpl]['4d'] = None

        elif self.cfg['sample_covariance']['compute_sample_cov']:
            cl_tt_4covsim = (
                self.cl_3x2pt_unb_5d[1, 1, :, :, :]
                + self.noise_3x2pt_unb_5d[1, 1, :, :, :]
            )
            cl_te_4covsim = (
                self.cl_3x2pt_unb_5d[1, 0, :, :, :]
                + self.noise_3x2pt_unb_5d[1, 0, :, :, :]
            )
            cl_ee_4covsim = (
                self.cl_3x2pt_unb_5d[0, 0, :, :, :]
                + self.noise_3x2pt_unb_5d[0, 0, :, :, :]
            )
            cl_tb_4covsim = np.zeros_like(cl_tt_4covsim)
            cl_eb_4covsim = np.zeros_like(cl_tt_4covsim)
            cl_bb_4covsim = np.zeros_like(cl_tt_4covsim)

            # ! note that self.cov_dict is mutated in-place, no need to return it
            start = time.perf_counter()
            result = sample_covariance_parallel(
                cov_dict=self.cov_dict,
                cl_GG_unbinned=cl_tt_4covsim,
                cl_LL_unbinned=cl_ee_4covsim,
                cl_GL_unbinned=cl_te_4covsim,
                cl_BB_unbinned=cl_bb_4covsim,
                cl_EB_unbinned=cl_eb_4covsim,
                cl_TB_unbinned=cl_tb_4covsim,
                nbl=nbl_eff,
                zbins=self.zbins,
                weight_maps_gg=self.weight_maps_gg,
                weight_maps_ll=self.weight_maps_ll,
                nside=self.mask_obj_ll.nside_cfg,
                nreal=self.cfg['sample_covariance']['nreal'],
                coupled_cls=self.coupled_cov,
                which_cls=self.cfg['sample_covariance']['which_cls'],
                nmt_bin_obj=self.nmt_bin_obj,
                lmax=ell_max_eff,
                wsp_path_template=self.cache_path + '/' + self.wsp_fname,
                fix_seed=self.cfg['sample_covariance']['fix_seed'],
                n_jobs=self.cfg['misc']['num_threads'],
            )
            self.sim_cl_GG, self.sim_cl_GL, self.sim_cl_LL = result
            print(f'sample covariance computed in {time.perf_counter() - start:.2f} s.')

            if self.cfg['sample_covariance']['save_sim_cls']:
                np.savez_compressed(
                    f'{self.output_path}/sample_cov_sim_cls.npz',
                    sim_cl_LL=self.sim_cl_LL,
                    sim_cl_GL=self.sim_cl_GL,
                    sim_cl_GG=self.sim_cl_GG,
                )

        return self.cov_dict
