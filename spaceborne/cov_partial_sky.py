import itertools
import os
import time
import warnings
from itertools import combinations_with_replacement

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


def couple_cov_6d_tomo(
    mcm_ab: np.ndarray, cov_abcd_6d: np.ndarray, mcm_cd: np.ndarray
) -> np.ndarray:
    """Couple a 6D (nbl, nbl, zbins, zbins, zbins, zbins) covariance with
    *per-redshift-bin* mode coupling matrices.

    When weight maps are used, the MCM is bin-pair dependent, so each block must
    be coupled with its own matrices::

        cov_coupled[:, :, zi, zj, zk, zl] =
            M_ab[zi, zj] @ cov[:, :, zi, zj, zk, zl] @ M_cd[zk, zl].T

    This generalises :func:`couple_cov_6d` (which assumes a single MCM shared by
    all bins). ``mcm_ab`` / ``mcm_cd`` are (nbl, nbl, zbins, zbins) arrays indexed
    by ``[:, :, zi, zj]`` (the ``mcm_*_binned`` arrays built in
    ``CovNaMaster.compute_and_save_mcms``).
    """
    # cov_coupled[X, Y, i, j, k, l] =
    #   sum_{W, Z} M_ab[X, W, i, j] cov[W, Z, i, j, k, l] M_cd[Y, Z, k, l]
    # (the M_cd[Y, Z, k, l] index order applies M_cd[:, :, zk, zl].T on the second
    # axis, matching the mcm_cd.T convention of couple_cov_6d)
    cov_abcd_6d_coupled = np.einsum(
        'XWij, WZijkl, YZkl -> XYijkl', mcm_ab, cov_abcd_6d, mcm_cd
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
    wgg_dict: dict,
    wgl_dict: dict,
    wll_dict: dict,
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

    def cl_gg_list(zi, zj, spin0):
        # gg is always spin0!
        return [cl_tt[:, zi, zj]]

    def cl_gl_list(zi, zj, spin0):
        if spin0:
            return [cl_te[:, zi, zj]]
        else:
            return [cl_te[:, zi, zj], cl_tb[:, zi, zj]]

    def cl_lg_list(zi, zj, spin0):
        if spin0:
            return [cl_et[:, zi, zj]]
        else:
            return [cl_et[:, zi, zj], cl_bt[:, zi, zj]]

    def cl_ll_list(zi, zj, spin0):
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
        'GG': cl_gg_list,
        'GL': cl_gl_list,
        'LG': cl_lg_list,
        'LL': cl_ll_list,
    }

    wsp_dict = {'GG': wgg_dict, 'GL': wgl_dict, 'LL': wll_dict}

    bin_cov_kw = {
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
                    cla1b1=cl_list_dict[f'{probe_a}{probe_c}'](zi, zk, spin0),
                    cla1b2=cl_list_dict[f'{probe_a}{probe_d}'](zi, zl, spin0),
                    cla2b1=cl_list_dict[f'{probe_b}{probe_c}'](zj, zk, spin0),
                    cla2b2=cl_list_dict[f'{probe_b}{probe_d}'](zj, zl, spin0),
                    coupled=coupled,
                    wa=wsp_dict[f'{probe_a}{probe_b}'][zi, zj],
                    wb=wsp_dict[f'{probe_c}{probe_d}'][zk, zl],
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


def _weight_per_bin(weight_maps, zi):
    """Returns the weight map for the given z bin index zi,
    whether we are using a footprint (1D array of shape (N_pix,))
    or weight maps (2D array of shape (zbins, N_pix))"""
    if isinstance(weight_maps, np.ndarray) and weight_maps.ndim == 1:
        return weight_maps
    return weight_maps[zi]


def mask_maps_and_compute_alms(
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


def pcls_from_maps(
    zi: int,
    zj: int,
    coupled_cls: bool,
    spin0: bool,
    wgg_dict: dict,
    wgl_dict: dict,
    wll_dict: dict,
    *,
    alms_T: list | None = None,
    alms_E: list | None = None,
    alms_B: list | None = None,
):
    """Compute binned pseudo-Cls for a single (zi, zj) pair.

    Healpy anafast returns the coupled
    ("pseudo") cls. Dividing by fsky gives a rough approximation of the true Cls.

    Fast branch (healpy):
        Pass pre-computed `alms_T`, `alms_E`, `alms_B` (indexed by zbin) to avoid
        re-doing the SHT on every (zi, zj) call. Pre-compute them once per
        realization with `precompute_alms_healpy` before the pair loop.
        Per-pair cost reduces from O(N_pix log N_pix) → O(lmax).

    Slow/fallback branch (healpy, no pre-computed alms):
        Computes all SHTs internally, same as the original implementation.

    """

    # ! compute (coupled) Cls with healpy
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
        if spin0:
            pcl_te_list = [pcl_te]
            pcl_ee_list = [pcl_ee]

        else:
            pcl_te_list = np.vstack([pcl_te, pcl_tb])
            pcl_ee_list = np.vstack([pcl_ee, pcl_eb, pcl_be, pcl_bb])

        cl_tt_out = wgg_dict[zi, zj].decouple_cell(pcl_tt[None, :])[0, :]
        cl_te_out = wgl_dict[zi, zj].decouple_cell(pcl_te_list)[0, :]
        cl_ee_out = wll_dict[zi, zj].decouple_cell(pcl_ee_list)[0, :]

    return np.array(cl_tt_out), np.array(cl_te_out), np.array(cl_ee_out)


def compute_ensemble_covariance_parallel(
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
    nl_gg_diag: np.ndarray,
    nl_ll_diag: np.ndarray,
    nside: int,
    nreal: int,
    coupled_cls: bool,
    spin0: bool,
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
        f'and computing pseudo-cls with healpy...'
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
    # Limit worker-internal BLAS/OpenMP pools only for this joblib section to
    # avoid n_jobs x OMP_NUM_THREADS oversubscription.
    with (
        parallel_backend('loky', inner_max_num_threads=1, verbose=1),
        Parallel(n_jobs=n_jobs, return_as='generator') as parallel,
    ):
        results_iter = parallel(
            delayed(_compute_one_realization)(
                seed=SEEDVALUE[i],
                cl_ring_big_list=cl_ring_big_list,
                lmax=lmax,
                nside=nside,
                zbins=zbins,
                weight_maps_gg=weight_maps_gg,
                weight_maps_ll=weight_maps_ll,
                nl_gg_diag=nl_gg_diag,
                nl_ll_diag=nl_ll_diag,
                coupled_cls=coupled_cls,
                spin0=spin0,
                wsp_path_template=wsp_path_template,
                nbl=nbl,
                ell_min_edges=ell_min_edges,
                ell_max_edges=ell_max_edges,
            )
            for i in range(nreal)
        )
        for i, (cl_gg_i, cl_gl_i, cl_ll_i) in enumerate(
            tqdm(results_iter, total=nreal)
        ):
            sim_cl_GG[i] = cl_gg_i
            sim_cl_GL[i] = cl_gl_i
            sim_cl_LL[i] = cl_ll_i

    # * Step II: compute ensemble covariance
    sim_cls_to_ensemble_cov(cov_dict, sim_cl_GG, sim_cl_GL, sim_cl_LL, nbl, zbins)

    return sim_cl_GG, sim_cl_GL, sim_cl_LL


def sim_cls_to_ensemble_cov(cov_dict, sim_cl_GG, sim_cl_GL, sim_cl_LL, nbl, zbins):

    zijkl_combinations = list(itertools.product(range(zbins), repeat=4))
    sim_cl_map = {'LL': sim_cl_LL, 'GL': sim_cl_GL, 'GG': sim_cl_GG}
    kwargs = {'rowvar': False, 'bias': False}

    for probe_ab, probe_cd in cov_dict['g']:
        cl_ab, cl_cd = sim_cl_map[probe_ab], sim_cl_map[probe_cd]
        for zi, zj, zk, zl in zijkl_combinations:
            cov_dict['g'][probe_ab, probe_cd]['6d'][:, :, zi, zj, zk, zl] = np.cov(
                cl_ab[:, :, zi, zj], cl_cd[:, :, zk, zl], **kwargs
            )[:nbl, nbl:]


def _compute_one_realization(
    seed,
    cl_ring_big_list: list,
    lmax: int,
    nside: int,
    zbins: int,
    weight_maps_gg: np.ndarray,
    weight_maps_ll: np.ndarray,
    nl_gg_diag,
    nl_ll_diag,
    coupled_cls: bool,
    spin0: bool,
    wsp_path_template: str,
    nbl: int,
    ell_min_edges: np.ndarray,
    ell_max_edges: np.ndarray,
):
    """Worker: one realization → Cls only. Maps are discarded before return."""

    # Set seed for reproducibility (must be set inside each worker for
    # parallel execution)
    np.random.seed(seed)

    # Load workspaces inside worker (NmtWorkspace objects are not picklable)
    # [Note]: this is only necessary if we want to decouple the Cls!
    wgg_dict, wgl_dict, wll_dict = {}, {}, {}
    if not coupled_cls:
        for zi, zj in itertools.product(range(zbins), repeat=2):
            wgg_dict[zi, zj] = nmt.NmtWorkspace()
            wgl_dict[zi, zj] = nmt.NmtWorkspace()
            wll_dict[zi, zj] = nmt.NmtWorkspace()
            wgg_dict[zi, zj].read_from(wsp_path_template.format('g', 'g', zi=zi, zj=zj))
            wgl_dict[zi, zj].read_from(wsp_path_template.format('g', 'l', zi=zi, zj=zj))
            wll_dict[zi, zj].read_from(wsp_path_template.format('l', 'l', zi=zi, zj=zj))

    # Reconstruct NmtBin object from edges (nmt_bin_obj objects are not picklable)
    nmt_bin_obj = nmt.NmtBin.from_edges(ell_min_edges, ell_max_edges)

    # ! 1. Generate alms from input Cls
    corr_alms_tot = hp.synalm(cl_ring_big_list, lmax=lmax, new=True)
    corr_alms_T = corr_alms_tot[::3]
    corr_alms_E_B = list(zip(corr_alms_tot[1::3], corr_alms_tot[2::3], strict=True))

    # ! 2. Generate (correlated) maps from alms
    corr_maps_gg = [hp.alm2map(alm, nside, lmax=lmax) for alm in corr_alms_T]
    corr_maps_ll = [
        hp.alm2map_spin([E, B], nside=nside, spin=2, lmax=lmax)
        for E, B in corr_alms_E_B
    ]
    # free alms
    del corr_alms_tot, corr_alms_T, corr_alms_E_B

    # ! 3. Inject noise at the map level directly to preserve "full resolution"
    # Before I was cutting the noise at ell_max_eff, but the mask-induced mode coupling
    # leaks power from high-ell into the "nmt band" [ell_min, ell_max_eff]. This is
    # important for a white spectrum, which has non-negligible power at high-ell.
    # Note: per-pixel variance is sigma^2 = N_ell / Omega_pix.

    npix = hp.nside2npix(nside)
    omega_pix = 4.0 * np.pi / npix  # pixel solid angle (in steradians)
    for zi, m in enumerate(corr_maps_gg):
        sigma_pix = np.sqrt(nl_gg_diag[zi] / omega_pix)
        m += np.random.randn(npix) * sigma_pix
    for zi, qu in enumerate(corr_maps_ll):
        sigma_pix = np.sqrt(nl_ll_diag[zi] / omega_pix)
        qu[0] += np.random.randn(npix) * sigma_pix
        qu[1] += np.random.randn(npix) * sigma_pix

    # ! 4. Mask each map (there are zbins of them) and compute ("masked") alms
    alms_T, alms_E, alms_B = mask_maps_and_compute_alms(
        corr_maps_gg=corr_maps_gg,
        corr_maps_ll=corr_maps_ll,
        weight_maps_gg=weight_maps_gg,
        weight_maps_ll=weight_maps_ll,
        lmax=lmax,
    )

    # free maps
    del corr_maps_gg, corr_maps_ll

    # ! 5. Compute Cls *for all (zi, zj) pairs*
    # [Note]: these are coupled by default (they are computed from the masked maps),
    # but they can be decoupled.
    # [Note]: the healpy branch should be much faster now that the alms have been
    # pre-computed for each bin, rather than (uselessly) re-computed for each bin
    # pair inside the loop below pcls_from_maps.
    cl_gg_3d = np.zeros((nbl, zbins, zbins))
    cl_gl_3d = np.zeros((nbl, zbins, zbins))
    cl_ll_3d = np.zeros((nbl, zbins, zbins))

    for zi, zj in itertools.product(range(zbins), repeat=2):
        cl_gg_1d, cl_gl_1d, cl_ll_1d = pcls_from_maps(
            zi=zi,
            zj=zj,
            coupled_cls=coupled_cls,
            spin0=spin0,
            wgg_dict=wgg_dict,
            wgl_dict=wgl_dict,
            wll_dict=wll_dict,
            alms_T=alms_T,
            alms_E=alms_E,
            alms_B=alms_B,
        )
        if len(cl_gg_1d) != nbl:
            cl_gg_1d = nmt_bin_obj.bin_cell(cl_gg_1d)
            cl_gl_1d = nmt_bin_obj.bin_cell(cl_gl_1d)
            cl_ll_1d = nmt_bin_obj.bin_cell(cl_ll_1d)

        cl_gg_3d[:, zi, zj] = cl_gg_1d
        cl_gl_3d[:, zi, zj] = cl_gl_1d
        cl_ll_3d[:, zi, zj] = cl_ll_1d

    # shape: (nbl, zbins, zbins)
    return cl_gg_3d, cl_gl_3d, cl_ll_3d


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


class CovNaMaster:
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
        self.nside = pvt_cfg['nside']
        self.coupled_cov = cfg['covariance']['cov_type'] == 'coupled'
        self.spin0 = cfg['precision']['spin0']
        self.output_path = self.cfg['misc']['output_path']
        self.load_cached_wsp = self.cfg['covariance']['load_cached_nmt_workspaces']
        self.save_wsp_to_cache = self.cfg['covariance']['save_nmt_wsp_to_cache']

        # just for readability
        self.footprint_gg = self.mask_obj_gg.footprint
        self.footprint_ll = self.mask_obj_ll.footprint
        self.weight_maps_gg = self.mask_obj_gg.weight_maps
        self.weight_maps_ll = self.mask_obj_ll.weight_maps

        # also just for readability (double negatives are ugly)
        self.use_weight_maps_ll = self.weight_maps_ll is not None
        self.use_weight_maps_gg = self.weight_maps_gg is not None
        self.use_footprint_gg = not self.use_weight_maps_gg
        self.use_footprint_ll = not self.use_weight_maps_ll

        if self.use_footprint_gg:
            self.weight_maps_gg = self.footprint_gg
        if self.use_footprint_ll:
            self.weight_maps_ll = self.footprint_ll

        self.wsp_fname = f'wsp_spin0{self.spin0}_' + '{:s}{:s}_zi{zi:d}zj{zj:d}.fits'
        self.cw_fname = (
            f'cw_spin0{self.spin0}_'
            + '{:s}{:s}{:s}{:s}_zi{zi:d}zj{zj:d}zk{zk:d}zl{zl:d}.fits'
        )
        self.cache_path = f'{self.output_path}/cache/nmt'

        # instantiate cov dict
        # ! note that this class only computes
        #   - only g term
        #   - all HS probe combinations (no 3x2pt!!)
        #   - only 4d and 6d dim

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
        if _lmax >= 3 * self.nside - 1:
            warnings.warn(
                f'lmax = {_lmax} >= 3 * NSIDE - 1 = {3 * self.nside - 1}\n'
                f'(NSIDE = {self.nside}) for probe GC. '
                'You should probably increase NSIDE or decrease lmax ',
                stacklevel=2,
            )
        _lmax = self.ell_obj.ell_max_WL
        if _lmax >= 3 * self.nside - 1:
            warnings.warn(
                f'lmax = {_lmax} >= 3 * NSIDE - 1 = {3 * self.nside - 1}\n'
                f'(NSIDE = {self.nside}) for probe WL. '
                'You should probably increase NSIDE or decrease lmax ',
                stacklevel=2,
            )

        self.cl_3x2pt_unb_5d = _UNSET
        self.ells_3x2pt_unb = _UNSET
        self.nbl_3x2pt_unb = _UNSET
        self.fsky_ab_dict = _UNSET

    def build_fields(self, lmax: int, spin0: bool):
        # TODO XXX make this also dependent on the selected probes!
        self.fg_dict, self.fl_dict = {}, {}
        print('\nComputing namaster fields...')

        # in case only the footprint is provided, the fields can be computed once
        if not self.use_weight_maps_gg:
            self.fg_ftp = nmt.NmtField(
                mask=self.footprint_gg, maps=None, spin=0, lite=True, lmax=lmax
            )
        if not self.use_weight_maps_ll:
            self.fl_ftp = nmt.NmtField(
                mask=self.footprint_ll,
                maps=None,
                spin=0 if spin0 else 2,
                lite=True,
                lmax=lmax,
            )

        # now, either compute per-bin fields from weight maps, or just assign the
        # same field (from the footprint) to all bins (i.e., to all keys in the dict)
        for zi in tqdm(range(self.zbins)):
            if self.use_weight_maps_gg:
                self.fg_dict[zi] = nmt.NmtField(
                    mask=_weight_per_bin(self.weight_maps_gg, zi),
                    maps=None,
                    spin=0,
                    lite=True,
                    lmax=lmax,
                )
            else:
                self.fg_dict[zi] = self.fg_ftp

            if self.use_weight_maps_ll:
                self.fl_dict[zi] = nmt.NmtField(
                    mask=_weight_per_bin(self.weight_maps_ll, zi),
                    maps=None,
                    spin=0 if spin0 else 2,
                    lite=True,
                    lmax=lmax,
                )
            else:
                self.fl_dict[zi] = self.fl_ftp

    def build_wsp(self):
        if self.load_cached_wsp:
            print(
                '\nLoading namaster workspaces and coupling matrices from\n'
                f'{self.cache_path}...'
            )
            warnings.warn(
                'You are loading files from the cache. Please make '
                'sure that,masks and cosmology are consistent with the current run',
                stacklevel=2,
            )
        else:
            print('\nComputing namaster workspaces and coupling matrices...')

        self.wgg_dict, self.wgl_dict, self.wll_dict = {}, {}, {}

        # ! 1. If no weight maps are passed, one wsp is sufficient
        if not self.load_cached_wsp:
            if not self.use_weight_maps_gg:
                self.wgg_ftp = nmt.NmtWorkspace()
                self.wgg_ftp.compute_coupling_matrix(
                    self.fg_ftp, self.fg_ftp, self.nmt_bin_obj
                )
            if (not self.use_weight_maps_ll) and (not self.use_weight_maps_gg):
                self.wgl_ftp = nmt.NmtWorkspace()
                self.wgl_ftp.compute_coupling_matrix(
                    self.fg_ftp, self.fl_ftp, self.nmt_bin_obj
                )
            if not self.use_weight_maps_ll:
                self.wll_ftp = nmt.NmtWorkspace()
                self.wll_ftp.compute_coupling_matrix(
                    self.fl_ftp, self.fl_ftp, self.nmt_bin_obj
                )

        # ! Regardless of the presence of weight maps, build wsp dictionaries
        # ! for all bin pairs, either by computing them (if weight maps are present)
        # ! or by assigning the same wsp (if only the footprint is present),
        # ! or by loading from cache
        # TODO XXX this can be made probe-dependent as for cw, and looped only over
        # TODO XXX unique pairs
        for zi, zj in tqdm(self.zij_cross_combs):
            if self.load_cached_wsp:
                wgg_name = self.wsp_fname.format('g', 'g', zi=zi, zj=zj)
                wgl_name = self.wsp_fname.format('g', 'l', zi=zi, zj=zj)
                wll_name = self.wsp_fname.format('l', 'l', zi=zi, zj=zj)
                self.wgg_dict[zi, zj] = nmt.NmtWorkspace()
                self.wgl_dict[zi, zj] = nmt.NmtWorkspace()
                self.wll_dict[zi, zj] = nmt.NmtWorkspace()
                self.wgg_dict[zi, zj].read_from(f'{self.cache_path}/{wgg_name}')
                self.wgl_dict[zi, zj].read_from(f'{self.cache_path}/{wgl_name}')
                self.wll_dict[zi, zj].read_from(f'{self.cache_path}/{wll_name}')

            else:
                if self.use_weight_maps_gg:
                    self.wgg_dict[zi, zj] = nmt.NmtWorkspace()
                    self.wgg_dict[zi, zj].compute_coupling_matrix(
                        self.fg_dict[zi], self.fg_dict[zj], self.nmt_bin_obj
                    )
                else:
                    self.wgg_dict[zi, zj] = self.wgg_ftp

                if self.use_weight_maps_gg or self.use_weight_maps_ll:
                    self.wgl_dict[zi, zj] = nmt.NmtWorkspace()
                    self.wgl_dict[zi, zj].compute_coupling_matrix(
                        self.fg_dict[zi], self.fl_dict[zj], self.nmt_bin_obj
                    )
                else:
                    self.wgl_dict[zi, zj] = self.wgl_ftp

                if self.use_weight_maps_ll:
                    self.wll_dict[zi, zj] = nmt.NmtWorkspace()
                    self.wll_dict[zi, zj].compute_coupling_matrix(
                        self.fl_dict[zi], self.fl_dict[zj], self.nmt_bin_obj
                    )
                else:
                    self.wll_dict[zi, zj] = self.wll_ftp

    def build_cw(self, unique_probe_combs, spin0: bool):
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

        # no need for cw if we want the ensemble covariance
        if self.cfg['covariance']['partial_sky_method'] == 'ensemble':
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
            cw_ftp = nmt.NmtCovarianceWorkspace()
            cw_ftp.compute_coupling_coefficients(
                self.fg_ftp, self.fg_ftp, self.fg_ftp, self.fg_ftp, spin0_only=spin0
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
                _f1 = self.fg_ftp if probe_ab[0] == 'G' else self.fl_ftp
                _f2 = self.fg_ftp if probe_ab[1] == 'G' else self.fl_ftp
                _f3 = self.fg_ftp if probe_cd[0] == 'G' else self.fl_ftp
                _f4 = self.fg_ftp if probe_cd[1] == 'G' else self.fl_ftp
                cw_dict_ftp[probe_ab, probe_cd] = nmt.NmtCovarianceWorkspace()
                cw_dict_ftp[probe_ab, probe_cd].compute_coupling_coefficients(
                    _f1, _f2, _f3, _f4, spin0_only=spin0
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
                self.fg_ftp, self.fg_ftp, self.fg_ftp, self.fg_ftp, spin0_only=spin0
            )
        if self.use_footprint_ll and not self.load_cached_wsp:
            cw_dict_ftp['LL', 'LL'] = nmt.NmtCovarianceWorkspace()
            cw_dict_ftp['LL', 'LL'].compute_coupling_coefficients(
                self.fl_ftp, self.fl_ftp, self.fl_ftp, self.fl_ftp, spin0_only=spin0
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
                            self.fg_dict[zi] if probe_ab[0] == 'G' else self.fl_dict[zi]
                        )
                        _f2 = (
                            self.fg_dict[zj] if probe_ab[1] == 'G' else self.fl_dict[zj]
                        )
                        _f3 = (
                            self.fg_dict[zk] if probe_cd[0] == 'G' else self.fl_dict[zk]
                        )
                        _f4 = (
                            self.fg_dict[zl] if probe_cd[1] == 'G' else self.fl_dict[zl]
                        )
                        self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl] = (
                            nmt.NmtCovarianceWorkspace()
                        )
                        self.cw_dict[probe_ab, probe_cd][
                            zi, zj, zk, zl
                        ].compute_coupling_coefficients(
                            _f1, _f2, _f3, _f4, spin0_only=spin0
                        )

                    # ! Case 1-2: get from previous computation
                    elif (not use_weight_maps) and (not self.load_cached_wsp):
                        self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl] = cw_dict_ftp[
                            probe_ab, probe_cd
                        ]

                    # ! Case 4: load cached
                    elif self.load_cached_wsp:
                        probe_list = list(probe_abcd)
                        cw_name = self.cw_fname.format(
                            *probe_list, zi=zi, zj=zj, zk=zk, zl=zl
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
        """Saves to cache the workspeces and covariance workspaces."""

        # if workspaces are already laoded from cache, do not save them again
        if self.load_cached_wsp:
            return

        # the decoupled ensemble cov workers read the workspaces from disk
        # (NmtWorkspace objects are not picklable), so in that case saving
        # is mandatory, regardless of save_nmt_wsp_to_cache
        ensemble_decoupled = (
            self.cfg['covariance']['partial_sky_method'] == 'ensemble'
            and not self.coupled_cov
        )
        #  not (A or B) ≡ (not A) and (not B)
        if not (self.save_wsp_to_cache or ensemble_decoupled):
            return

        # else, create folder if absent and save everything
        os.makedirs(f'{self.cache_path}', exist_ok=True)
        print('\nSaving namaster workspaces in cache...')
        for zi, zj in tqdm(self.zij_cross_combs):
            wgg_name = self.wsp_fname.format('g', 'g', zi=zi, zj=zj)
            wgl_name = self.wsp_fname.format('g', 'l', zi=zi, zj=zj)
            wll_name = self.wsp_fname.format('l', 'l', zi=zi, zj=zj)
            self.wgg_dict[zi, zj].write_to(f'{self.cache_path}/{wgg_name}')
            self.wgl_dict[zi, zj].write_to(f'{self.cache_path}/{wgl_name}')
            self.wll_dict[zi, zj].write_to(f'{self.cache_path}/{wll_name}')

        # if the ensemble covariance is required, no cw are computed,
        # so no need to save them
        if self.cfg['covariance']['partial_sky_method'] == 'ensemble':
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
                    probe_list = list(probe_abcd)
                    cw_name = self.cw_fname.format(
                        *probe_list, zi=zi, zj=zj, zk=zk, zl=zl
                    )
                    self.cw_dict[probe_ab, probe_cd][zi, zj, zk, zl].write_to(
                        f'{self.cache_path}/{cw_name}'
                    )

    def compute_and_save_mcms(self, nbl_unb: int, spin0: bool):
        """Explicitly computes and bins the mode coupling matrices.
        The guards below ensure this function is called only in the following cases:

        1. The user wants to save the MCMs,
        and/or
        2. The user wants to compute a coupled non-Gaussian covariance (SSC and/or cNG)

        In the second case, the MCMs are needed to "manually" couple the SSC and cNG
        terms, since these are not handled by nmt.

        Args:
            nbl_unb (int): number of unbinned multipoles (i.e., lmax_eff + 1)
        """

        # guards
        ssc_or_cng = self.cfg['covariance']['SSC'] or self.cfg['covariance']['cNG']
        save_mcms = self.cfg['covariance']['save_mcms']

        if save_mcms:
            pass
        elif not self.coupled_cov or not ssc_or_cng:
            return

        # indices needed for MCM slicing, depending on the spin of the fields
        ix_s0s0 = 1
        ix_s0s2 = ix_s0s0 if spin0 else 2
        ix_s2s2 = ix_s0s0 if spin0 else 4

        print('\nComputing and binning mode coupling matrices...')
        # The MCMs are stored as (nbl, nbl, zbins, zbins) arrays indexed by
        # [:, :, zi, zj]: with weight maps the MCM is bin-pair dependent, and the
        # (zi, zj) grid is dense (full product), so an ndarray is the natural
        # container (nbl-first axes, consistent with the rest of the code).
        nbl = self.nmt_bin_obj.get_n_bands()
        mcm_shape = (nbl, nbl, self.zbins, self.zbins)
        mcm_unb_shape = (nbl_unb, nbl_unb, self.zbins, self.zbins)
        mcm_tt_unb = np.zeros(mcm_unb_shape)
        mcm_te_unb = np.zeros(mcm_unb_shape)
        mcm_ee_unb = np.zeros(mcm_unb_shape)
        self.mcm_tt_binned = np.zeros(mcm_shape)
        self.mcm_te_binned = np.zeros(mcm_shape)
        self.mcm_ee_binned = np.zeros(mcm_shape)

        for zi, zj in tqdm(self.zij_cross_combs):
            # from nmt docs:
            # Mode-coupling matrix. The matrix will have shape (nrows,nrows),
            # with nrows = n_cls * n_ells, where n_cls is the number of power spectra
            # (1, 2 or 4 for spin 0-0, spin 0-2 and spin 2-2 correlations), and
            # n_ells = lmax + 1, [...]. The L-th element of the i-th power spectrum
            # is stored with index L * n_cls + i.
            mcm_tt_unb[:, :, zi, zj] = self.wgg_dict[zi, zj].get_coupling_matrix()[
                0::ix_s0s0, 0::ix_s0s0
            ]
            mcm_te_unb[:, :, zi, zj] = self.wgl_dict[zi, zj].get_coupling_matrix()[
                0::ix_s0s2, 0::ix_s0s2
            ]
            mcm_ee_unb[:, :, zi, zj] = self.wll_dict[zi, zj].get_coupling_matrix()[
                0::ix_s2s2, 0::ix_s2s2
            ]

            # bin (and store in self)
            self.mcm_tt_binned[:, :, zi, zj] = bin_mcm(
                mcm_tt_unb[:, :, zi, zj], self.nmt_bin_obj
            )
            self.mcm_te_binned[:, :, zi, zj] = bin_mcm(
                mcm_te_unb[:, :, zi, zj], self.nmt_bin_obj
            )
            self.mcm_ee_binned[:, :, zi, zj] = bin_mcm(
                mcm_ee_unb[:, :, zi, zj], self.nmt_bin_obj
            )

        if save_mcms:
            np.savez(
                f'{self.output_path}/mode_coupling_matrices.npz',
                mcm_gg_unbinned=mcm_tt_unb,
                mcm_gl_unbinned=mcm_te_unb,
                mcm_ll_unbinned=mcm_ee_unb,
                mcm_gg_binned=self.mcm_tt_binned,
                mcm_gl_binned=self.mcm_te_binned,
                mcm_ll_binned=self.mcm_ee_binned,
            )
            print(f'\nMode coupling matrices saved in {self.output_path}')

    def build_psky_cov(self):
        # TODO again, here I'm using 3x2pt = GC
        # 1. ell binning
        # shorten names for brevity

        assert np.array_equal(self.ell_obj.ell_edges_WL, self.ell_obj.ell_edges_GC), (
            'The NaMaster partial-sky covariance assumes identical WL and GC binning, '
            'but ell_edges_WL != ell_edges_GC. Per-probe NaMaster binning is not yet '
            'implemented.'
        )
        self.nmt_bin_obj = self.ell_obj.nmt_bin_obj_GC
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

        # ! 1. Create field objects
        # ! (there will be no maps associated to the fields)
        # TODO maks=None (as in the example) or maps=[mask]? I think None

        self.build_fields(ell_max_eff, spin0=self.spin0)
        self.build_wsp()
        self.build_cw(unique_probe_combs, spin0=self.spin0)
        self.save_to_cache(unique_probe_combs)
        self.compute_and_save_mcms(nbl_unb=nbl_unb, spin0=self.spin0)

        # if the coupled covariance is required, I'll later need to convolve the
        # non-Gaussian terms. For this, I'll need the binned mode coupling matrices
        # (mcm), which I store in self
        # TODO XXX again, only loop over the unique zpairs
        # TODO XXX in general, I should definetly create some function to declutter the
        # TODO XXX "main"...

        # if you want to use the iNKA, the cls to be passed are the coupled ones
        # divided by fsky

        # note: the .copy() is needed, keep it!
        cl_gg_4covnmt = self.cl_3x2pt_unb_5d[1, 1, :, :, :].copy()
        cl_gl_4covnmt = self.cl_3x2pt_unb_5d[1, 0, :, :, :].copy()
        cl_ll_4covnmt = self.cl_3x2pt_unb_5d[0, 0, :, :, :].copy()
        if self.cfg['precision']['iNKA']:
            # TODO XXX this could be made more efficient by only looping over the auto-combs
            # TODO XXX for ll and gg
            # The iNKA approximates the true Cl as couple_cell(Cl) / fsky. The fsky
            # here must be the *effective* sky fraction of the mask pair that actually
            # produced the coupling, i.e. mean(w_a * w_b) of the masks used to build
            # the workspaces. For a binary footprint w^2 = w, so this equals mean(w)
            # (= fsky_ab_dict); but for fractional weight maps mean(w^2) != mean(w)!
            for zi, zj in self.zij_cross_combs:
                # get fskys
                w_gg_zi = _weight_per_bin(self.weight_maps_gg, zi)
                w_gg_zj = _weight_per_bin(self.weight_maps_gg, zj)
                w_ll_zi = _weight_per_bin(self.weight_maps_ll, zi)
                w_ll_zj = _weight_per_bin(self.weight_maps_ll, zj)
                fsky_gg_zij = float(np.mean(w_gg_zi * w_gg_zj))
                fsky_gl_zij = float(np.mean(w_gg_zi * w_ll_zj))
                fsky_ll_zij = float(np.mean(w_ll_zi * w_ll_zj))

                # prepare inputs for couple_cell function
                list_gg = [self.cl_3x2pt_unb_5d[1, 1, :, zi, zj]]

                list_gl = [self.cl_3x2pt_unb_5d[1, 0, :, zi, zj]]
                if not self.spin0:
                    list_gl.extend(
                        [np.zeros_like(self.cl_3x2pt_unb_5d[1, 0, :, zi, zj])]
                    )
                list_ll = [self.cl_3x2pt_unb_5d[0, 0, :, zi, zj]]
                if not self.spin0:
                    list_ll.extend(
                        [
                            np.zeros_like(self.cl_3x2pt_unb_5d[0, 0, :, zi, zj]),
                            np.zeros_like(self.cl_3x2pt_unb_5d[0, 0, :, zi, zj]),
                            np.zeros_like(self.cl_3x2pt_unb_5d[0, 0, :, zi, zj]),
                        ]
                    )
                cl_gg_4covnmt[:, zi, zj] = (
                    self.wgg_dict[zi, zj].couple_cell(list_gg)[0] / fsky_gg_zij
                )
                cl_gl_4covnmt[:, zi, zj] = (
                    self.wgl_dict[zi, zj].couple_cell(list_gl)[0] / fsky_gl_zij
                )
                cl_ll_4covnmt[:, zi, zj] = (
                    self.wll_dict[zi, zj].couple_cell(list_ll)[0] / fsky_ll_zij
                )

        nl_gg_4covnmt = self.nl_3x2pt_unb_5d[1, 1, :, :, :].copy()
        nl_gl_4covnmt = self.nl_3x2pt_unb_5d[1, 0, :, :, :].copy()  # this is 0
        nl_ll_4covnmt = self.nl_3x2pt_unb_5d[0, 0, :, :, :].copy()
        nl_ll_4covnmt[:2] = 0  # a spin-2 field has no monopole or dipole!

        # add noise to spectra to compute NMT cov
        cl_tt_4covnmt = cl_gg_4covnmt + nl_gg_4covnmt
        cl_te_4covnmt = cl_gl_4covnmt + nl_gl_4covnmt
        cl_ee_4covnmt = cl_ll_4covnmt + nl_ll_4covnmt
        cl_tb_4covnmt = np.zeros_like(cl_tt_4covnmt)
        cl_eb_4covnmt = np.zeros_like(cl_tt_4covnmt)
        cl_bb_4covnmt = nl_ll_4covnmt

        # ! Finally, compute covariance
        if self.cfg['covariance']['partial_sky_method'] == 'NaMaster':
            coupled_str = self.cfg['covariance']['cov_type']
            spin0_str = ' spin0' if self.spin0 else ''

            # the nmt_gaussian_cov_opt functions modifies
            # cov_dict in-place, so no need to capture any return value
            with sl.timer(
                f'\nComputing {coupled_str}{spin0_str} partial-sky '
                'Gaussian covariance...'
            ):
                nmt_gaussian_cov(
                    cov_dict=self.cov_dict,
                    spin0=self.spin0,
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
                    wgg_dict=self.wgg_dict,
                    wgl_dict=self.wgl_dict,
                    wll_dict=self.wll_dict,
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

        elif self.cfg['covariance']['partial_sky_method'] == 'ensemble':
            len_dv = (
                self.zbins**2 + self.zbins * (self.zbins + 1) // 2 * 2
            ) * self.ell_obj.nbl_3x2pt
            print(
                '\nComputing ensemble covariance from '
                f'{self.cfg["ensemble_covariance"]["nreal"]} healpy Gaussian '
                f'realizations. The datevector length is {len_dv}'
            )

            # ! signal-only Cls for synalm. Noise is NOT included in here: doing so
            # would band-limit it at ell_max_eff, but real shape noise is white to the
            # pixel scale (~3*nside). I instead inject it as full-band per-pixel white
            # noise inside each realization (see _compute_one_realization). The mask
            # then couples its high-ell power down into the science band, as in the data.
            cl_tt_4covens = self.cl_3x2pt_unb_5d[1, 1, :, :, :]
            cl_te_4covens = self.cl_3x2pt_unb_5d[1, 0, :, :, :]
            cl_ee_4covens = self.cl_3x2pt_unb_5d[0, 0, :, :, :]
            cl_tb_4covens = np.zeros_like(cl_tt_4covens)
            cl_eb_4covens = np.zeros_like(cl_tt_4covens)
            cl_bb_4covens = np.zeros_like(cl_tt_4covens)

            # check that the noise spectra are white
            for nl, name in [(nl_gg_4covnmt, 'GG'), (nl_ll_4covnmt, 'LL')]:
                if not np.allclose(nl[2:], nl[2], rtol=1e-5, atol=0.0):
                    raise ValueError(
                        f'The {name} noise spectra are not white; the ensemble '
                        'covariance assumes white noise.'
                    )

            # extract the zi-zj diagonal of noise arrays
            nl_gg_4covens = np.diagonal(nl_gg_4covnmt[0]).copy()
            nl_ll_4covens = np.diagonal(nl_ll_4covnmt[2]).copy()

            # ! note that self.cov_dict is mutated in-place, no need to return it
            start = time.perf_counter()
            result = compute_ensemble_covariance_parallel(
                cov_dict=self.cov_dict,
                cl_GG_unbinned=cl_tt_4covens,
                cl_LL_unbinned=cl_ee_4covens,
                cl_GL_unbinned=cl_te_4covens,
                cl_BB_unbinned=cl_bb_4covens,
                cl_EB_unbinned=cl_eb_4covens,
                cl_TB_unbinned=cl_tb_4covens,
                nbl=nbl_eff,
                zbins=self.zbins,
                weight_maps_gg=self.weight_maps_gg,
                weight_maps_ll=self.weight_maps_ll,
                nl_gg_diag=nl_gg_4covens,
                nl_ll_diag=nl_ll_4covens,
                nside=self.nside,
                nreal=self.cfg['ensemble_covariance']['nreal'],
                coupled_cls=self.coupled_cov,
                spin0=self.spin0,
                nmt_bin_obj=self.nmt_bin_obj,
                lmax=ell_max_eff,
                wsp_path_template=self.cache_path + '/' + self.wsp_fname,
                fix_seed=self.cfg['ensemble_covariance']['fix_seed'],
                n_jobs=self.cfg['misc']['num_threads'],
            )
            self.sim_cl_GG, self.sim_cl_GL, self.sim_cl_LL = result
            print(
                f'ensemble covariance computed in '
                f'{(time.perf_counter() - start) / 60:.2f} m.'
            )

            if self.cfg['ensemble_covariance']['save_sim_cls']:
                np.savez_compressed(
                    f'{self.output_path}/ensemble_cov_sim_cls.npz',
                    sim_cl_LL=self.sim_cl_LL,
                    sim_cl_GL=self.sim_cl_GL,
                    sim_cl_GG=self.sim_cl_GG,
                )

        return self.cov_dict
