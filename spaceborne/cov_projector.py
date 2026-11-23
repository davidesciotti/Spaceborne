"""
Base class for projected covariance computations.

This module provides the shared infrastructure for computing covariances
of projected statistics (2PCF, COSEBIs, etc.) from harmonic-space Cl.

The key insight is that different statistics share the same integrand building
(SVA, MIX terms) but differ in how they project from Cl to the observable space.
"""

import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
from scipy.integrate import quad_vec
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d, make_interp_spline

from spaceborne import constants as const


# ! ============================= pair counts ===============================
# Analytical
def get_npair(theta_1_u, theta_1_l, survey_area_sr, n_eff_i, n_eff_j):
    r"""Compute total (ideal) number of pairs in a theta bin, i.e., N(theta).
    N(θ) = π (θ_u^2 - θ_l^2) * A * n_i * n_j
         = \int_{θ_l}^{θ_u} dθ (dN(θ)/dθ)
    """
    n_eff_i_sr = n_eff_i * const.SR_TO_ARCMIN2
    n_eff_j_sr = n_eff_j * const.SR_TO_ARCMIN2
    return (
        np.pi * (theta_1_u**2 - theta_1_l**2) * survey_area_sr * n_eff_i_sr * n_eff_j_sr
    )


def get_dnpair(theta, survey_area_sr, n_eff_i, n_eff_j):
    r"""Compute differential (ideal) number of pairs, i.e. dN(theta)/dtheta.
    dN(θ)/dθ = 2π θ * A * n_i * n_j
    """
    n_eff_i_sr = n_eff_i * const.SR_TO_ARCMIN2
    n_eff_j_sr = n_eff_j * const.SR_TO_ARCMIN2
    return 2 * np.pi * theta * survey_area_sr * n_eff_i_sr * n_eff_j_sr


# Measured on the data. Conventions:
# - the counts are the weighted *effective* counts (sum w_a w_b)^2 / sum (w_a w_b)^2,
# - auto bins hold the *ordered* count (each pair counted twice, as get_npair), not
#   the unique-pair count of a single-catalogue treecorr run;
# - arrays have shape (n_theta, zbins, zbins), GL is (theta, lens bin, source bin);
# - they are indexed by the harmonic-space probe they refer to: 'LL', 'GL', 'GG'.
def load_pair_counts(filename: str, zbins: int) -> tuple[np.ndarray, dict]:
    """Read the measured pair counts from a .npz file.

    The file contains 'theta_edges_arcmin', shape (n_theta + 1,), and any of
    'npair_LL', 'npair_GL', 'npair_GG', each of shape (n_theta, zbins, zbins).

    Returns
    -------
    theta_edges_rad : np.ndarray, shape (n_theta + 1,)
    npair_dict : dict
        {'LL' | 'GL' | 'GG': array of shape (n_theta, zbins, zbins)}, only for the
        probes present in the file.
    """

    # load data into npair_dict and into theta_edges_arcmin
    with np.load(filename) as data:
        if 'theta_edges_arcmin' not in data:
            raise KeyError(f"'theta_edges_arcmin' not found in {filename}")
        theta_edges_arcmin = np.asarray(data['theta_edges_arcmin'], dtype=float)
        npair_dict = {
            key: np.asarray(data[f'npair_{key}'], dtype=float)
            for key in const.HS_DIAG_PROBES
            if f'npair_{key}' in data
        }

    # various sanity checks
    if (
        theta_edges_arcmin.ndim != 1
        or not np.all(np.isfinite(theta_edges_arcmin))
        or theta_edges_arcmin[0] <= 0
        or np.any(np.diff(theta_edges_arcmin) <= 0)
    ):
        raise ValueError(
            f'theta_edges_arcmin in {filename} must be 1D, finite, positive and '
            'increasing'
        )
    if not npair_dict:
        raise ValueError(f'No npair_LL, npair_GL or npair_GG array found in {filename}')

    expected_shape = (len(theta_edges_arcmin) - 1, zbins, zbins)
    for key, npair in npair_dict.items():
        if npair.shape != expected_shape:
            raise ValueError(
                f'npair_{key} in {filename} has shape {npair.shape}, expected '
                f'(n_theta, zbins, zbins) = {expected_shape}'
            )

        if np.any(npair <= 0) or not np.all(np.isfinite(npair)):
            raise ValueError(f'npair_{key} in {filename} must be finite and positive')
        if key in ('LL', 'GG') and not np.allclose(npair, npair.transpose(0, 2, 1)):
            raise ValueError(
                f'npair_{key} in {filename} must be symmetric in (zi, zj): please '
                'provide both triangles'
            )

    theta_edges_rad = np.deg2rad(theta_edges_arcmin / 60)
    return theta_edges_rad, npair_dict


def _check_theta_range(theta_min, theta_max, theta_edges_in, what):
    # small tolerance, for edges computed in different ways (e.g. treecorr vs numpy)
    if theta_min < theta_edges_in[0] * (1 - 1e-8) or (
        theta_max > theta_edges_in[-1] * (1 + 1e-8)
    ):
        raise ValueError(
            f'The pair counts cover theta = [{theta_edges_in[0]:.6e}, '
            f'{theta_edges_in[-1]:.6e}] rad, but the {what} needs '
            f'[{theta_min:.6e}, {theta_max:.6e}] rad: please provide pair counts over '
            'the full range'
        )


def rebin_pair_counts(
    npair: np.ndarray, theta_edges_in: np.ndarray, theta_edges_out: np.ndarray
) -> np.ndarray:
    """Pair counts in the (coarse) bins defined by theta_edges_out, from the (fine)
    ones defined by theta_edges_in.

    The cumulative count N(<theta) is interpolated linearly in theta^2 between the
    input edges, and differenced at the output edges. Fine bins entirely within a
    coarse bin are therefore just summed, while a fine bin cut by a coarse edge is
    split proportionally to the ring area on each side, as for a uniform field
    (dN/dtheta ∝ theta). This is exact for the ideal counts, and for any counts if
    the output edges are a subset of the input ones.
    """
    _check_theta_range(
        theta_edges_out[0], theta_edges_out[-1], theta_edges_in, 'theta binning'
    )
    cumsum = np.concatenate(
        [np.zeros((1, *npair.shape[1:])), np.cumsum(npair, axis=0)], axis=0
    )
    # clip: the range check above allows a tiny tolerance beyond the input edges
    theta2_out = np.clip(
        theta_edges_out**2, theta_edges_in[0] ** 2, theta_edges_in[-1] ** 2
    )
    cumsum_out = interp1d(theta_edges_in**2, cumsum, axis=0, kind='linear')(theta2_out)
    return np.diff(cumsum_out, axis=0)


def dnpair_from_pair_counts(
    npair: np.ndarray, theta_edges_in: np.ndarray, theta_out: np.ndarray
) -> np.ndarray:
    """dN/dtheta on theta_out from (fine) pair counts.

    dN/dtheta is count / bin width, at the arithmetic bin centres (where it is exact
    for the ideal counts), interpolated linearly in log-log, and extrapolated within
    the outermost half bins. Interpolating in log-log keeps dN/dtheta positive (also
    in the extrapolated half bins, where a linear extrapolation can turn negative),
    and is still exact for the ideal dN/dtheta ∝ theta.

    Returns
    -------
    np.ndarray, shape (len(theta_out), zbins, zbins)
    """
    _check_theta_range(
        np.min(theta_out), np.max(theta_out), theta_edges_in, 'COSEBIs theta grid'
    )
    if np.any(npair <= 0) or not np.all(np.isfinite(npair)):
        raise ValueError('The pair counts must be finite and positive')
    widths = np.diff(theta_edges_in)
    centres = 0.5 * (theta_edges_in[1:] + theta_edges_in[:-1])
    log_dnpair = np.log(npair / widths[:, None, None])
    interp = interp1d(
        np.log(centres), log_dnpair, axis=0, kind='linear', fill_value='extrapolate'
    )
    return np.exp(interp(np.log(theta_out)))


def warn_if_pair_counts_off(
    npair_meas: np.ndarray, npair_ideal: np.ndarray, probe_key: str, rtol: float = 0.25
) -> None:
    """At the smallest theta, the measured counts should be close to the ideal ones
    (the survey edges and holes matter little). A ratio of ~1/2 in the auto bins
    signals unique-pair counts, instead of the expected ordered ones.
    """
    ratio = npair_meas[0] / npair_ideal[0]
    if np.any(np.abs(ratio - 1) > rtol):
        warnings.warn(
            f'The measured {probe_key} pair counts differ by more than {rtol:.0%} from '
            'the ideal ones at the smallest theta (measured/ideal ratio in '
            f'[{ratio.min():.3f}, {ratio.max():.3f}]). Please check n_eff, the survey '
            'area and the pair counts. Note that the auto bins must contain the '
            'ordered counts (each pair counted twice), i.e. twice the output of a '
            'single-catalogue treecorr run.',
            stacklevel=2,
        )


def get_delta_tomo(probe_a_ix: int, probe_b_ix: int, zbins: int) -> np.ndarray:
    return np.eye(zbins) if probe_a_ix == probe_b_ix else np.zeros((zbins, zbins))


# ! ==================== build Cl integrand for SVA and MIX covs =====================
def build_cl_integrand_5d_sva(cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
    """
    Build the SVA (sample variance) integrand in harmonic space.

    This integrand is UNIVERSAL - it's the same for all projection methods
    (real space, COSEBIs, band powers, etc.). The formula comes from the
    Gaussian covariance of Cl:

    Cov[C_ab, C_cd] ∝ C_ac * C_bd + C_ad * C_bc

    Parameters
    ----------
    cl_5d : np.ndarray
        Power spectra array with shape (n_probes, n_probes, n_ell, zbins, zbins)
    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix : int
        Probe indices (0 for shear, 1 for galaxy clustering)

    Returns
    -------
    integrand_5d : np.ndarray
        Shape (n_ell, zbins, zbins, zbins, zbins)
        The harmonic-space integrand before projection
    """
    a = np.einsum(
        'Lik,Ljl->Lijkl', cl_5d[probe_a_ix, probe_c_ix], cl_5d[probe_b_ix, probe_d_ix]
    )
    b = np.einsum(
        'Lil,Ljk->Lijkl', cl_5d[probe_a_ix, probe_d_ix], cl_5d[probe_b_ix, probe_c_ix]
    )
    return a + b


def build_cl_integrand_5d_mix(
    cl_5d, nl_4d, probe_a_ix: int, probe_b_ix: int, probe_c_ix: int, probe_d_ix: int
) -> np.ndarray:
    """Build the MIX-term integrand in harmonic space, shape (nbl, z, z, z, z).

    This is the MIX analogue of ``build_cl_integrand_5d_sva``: everything in
    the integrand except the ``ell * K_mu * K_nu`` projection weight. Extracted so
    that the FFTLog/vectorized paths can all share it.
    """

    a = np.einsum(
        'jl,Lik->Lijkl', nl_4d[probe_b_ix, probe_d_ix], cl_5d[probe_a_ix, probe_c_ix]
    )
    b = np.einsum(
        'ik,Ljl->Lijkl', nl_4d[probe_a_ix, probe_c_ix], cl_5d[probe_b_ix, probe_d_ix]
    )
    c = np.einsum(
        'jk,Lil->Lijkl', nl_4d[probe_b_ix, probe_c_ix], cl_5d[probe_a_ix, probe_d_ix]
    )
    d = np.einsum(
        'il,Ljk->Lijkl', nl_4d[probe_a_ix, probe_d_ix], cl_5d[probe_b_ix, probe_c_ix]
    )
    return a + b + c + d


def proj_cov_2d(
    ells_proj: np.ndarray,
    cov_hs_ng_4d: np.ndarray,
    kernel_1_func_of_ell: Callable[[np.ndarray], np.ndarray],
    kernel_2_func_of_ell: Callable[[np.ndarray], np.ndarray],
    integration_method: str,
) -> np.ndarray:
    r"""Projects a 2D array (a non-Gaussian covariance in harmonic space) using the
    equation
        \int d\ell_1 d\ell_2 k_1(\ell_1) k_2(\ell_1) cov(\ell_1, \ell_2)
    The input array should be 4D, with shape
    (nbl_proj_ng, nbl_proj_ng, zpairs_ab, zpairs_cd).

    Notes
    -----
    Here we're only parallelizing over s1 and s2 from outside this function,
    as opposed to the 1d case (see sva and mix) terms, where we also parallelize over
    zpairs_ab and zpairs_cd. Here we vectorize instead.

    The quad_vec branch does *not* interpolate the challenging part of the integrand,
    i.e. the kernels! This means that it only assumes the 2D input covariance to be
    a smooth function of ell1, ell2

    Also, the fact that I'm not passing workers to quad_vec is intentional, I can't
    quite get it to work even when turning off the other layers of parallelism
    (I think I should move ell*_integrand_func outside the function)
    """

    # inputs sanity checks
    if cov_hs_ng_4d.ndim != 4:
        raise ValueError('cov_hs_ng_4d must be 4D')

    if cov_hs_ng_4d.shape[0] != cov_hs_ng_4d.shape[1]:
        raise ValueError(
            f'First two axes of cov_hs_ng_4d must match, got {cov_hs_ng_4d.shape[:2]}'
        )

    nbl = len(ells_proj)
    if cov_hs_ng_4d.shape[0] != nbl:
        raise ValueError(
            f'cov_hs_ng_4d.shape={cov_hs_ng_4d.shape} inconsistent with '
            f'len(ells_proj)={nbl}'
        )

    if nbl < 2:
        raise ValueError('Need at least 2 ell points for interpolation/integration')

    if not np.all(np.diff(ells_proj) > 0):
        raise ValueError('ells_proj must be strictly increasing')

    if integration_method not in ['quad', 'simps']:
        raise ValueError(
            f'integration_method {integration_method} not recognized! '
            'Must be either "quad" or "simps"'
        )

    if integration_method == 'quad':
        if ells_proj[0] <= 0:
            raise ValueError(
                'ells_proj must be strictly positive for quad: the covariance is '
                'splined in log(ell)'
            )
        ell_min = ells_proj[0]
        ell_max = ells_proj[-1]

        zpairs_ab = cov_hs_ng_4d.shape[2]
        zpairs_cd = cov_hs_ng_4d.shape[3]
        zpairs_flat = zpairs_ab * zpairs_cd

        # flatten covariance to shape (nbl, nbl, zpairs_flat) for easier
        # interpolation and integration
        cov_ell1ell2 = cov_hs_ng_4d.reshape(nbl, nbl, zpairs_flat)
        ell2_integral = np.zeros((nbl, zpairs_flat))

        # the covariance is splined in log(ell), where it is smoother; the
        # integration variable is still ell
        log_ells_proj = np.log(ells_proj)

        # Inner integral (d\ell2)
        for i in range(nbl):
            # for each ell1, interpolate along ell2
            cov_ell1ell2_interp_func = make_interp_spline(
                log_ells_proj, cov_ell1ell2[i], k=3, axis=0
            )

            # define callable of ell2 as required by quad. note that the kernel,
            # which is the oscillatory part, is not interpolated!
            def ell2_integrand_func(ells_2, _interp=cov_ell1ell2_interp_func):
                return ells_2 * kernel_2_func_of_ell(ells_2) * _interp(np.log(ells_2))

            # evaluate integral
            ell2_integral[i], _ = quad_vec(ell2_integrand_func, ell_min, ell_max)

        # now do the same along ell1: interpolate the result...
        ell2_integral_interp_func = make_interp_spline(
            log_ells_proj, ell2_integral, k=3, axis=0
        )

        # define callable of ell1
        def ell1_integrand_func(ells_1, _interp=ell2_integral_interp_func):
            return ells_1 * kernel_1_func_of_ell(ells_1) * _interp(np.log(ells_1))

        # integrate again
        outer_result, _ = quad_vec(ell1_integrand_func, ell_min, ell_max)

        # finally, reshape the result
        cov_out = np.asarray(outer_result).reshape(zpairs_ab, zpairs_cd)

    elif integration_method == 'simps':
        # just to keep track, the two grids are the same
        ells_1 = ells_proj
        ells_2 = ells_proj

        # Evaluate projection kernels
        kernel_1 = kernel_1_func_of_ell(ells_1)
        kernel_2 = kernel_2_func_of_ell(ells_2)

        # construct integrand, reshaping the ell and kernel arrays
        integrand = (
            ells_1[:, None, None, None]
            * ells_2[None, :, None, None]
            * kernel_1[:, None, None, None]
            * kernel_2[None, :, None, None]
            * cov_hs_ng_4d
        )

        # integrate along ells_1 and ells_2
        part_integral = simps(y=integrand, x=ells_1, axis=0)
        cov_out = simps(y=part_integral, x=ells_2, axis=0)

    return cov_out


def quad_projection_weights(
    ells_proj: np.ndarray,
    kernel_funcs: list[Callable[[float], float]],
    epsrel: float = 1e-10,
) -> np.ndarray:
    r"""Weights that project a function sampled on ``ells_proj`` with quad_vec:

        w[s, j] = \int d\ell \ell k(\ell; s) B_j(\ln \ell)

    where B_j is the cubic spline (``make_interp_spline``) through the unit vector
    e_j on ``log(ells_proj)``: the function is interpolated in log(ell), where it is
    smoother (and where a geomspace grid is uniform), while the integral is still in
    ell. Since the spline is linear in the data, f(\ell) = \sum_j B_j(\ln \ell) f_j,
    and so

        \int d\ell \ell k(\ell; s) f(\ell) = \sum_j w[s, j] f_j

    for *any* f sampled on ``ells_proj``. The weights only depend on the kernels and
    on the ell grid, so the expensive (oscillatory) integration is done once and
    reused for every probe, zpair and covariance term.

    Parameters
    ----------
    ells_proj : np.ndarray, shape (nbl,)
        Multipoles at which the function to project is sampled; strictly increasing.
    kernel_funcs : list of callables, length nbs
        Projection kernels as functions of (scalar) ell, one per scale bin.
    epsrel : float
        Relative tolerance of quad_vec, which applies to the norm of the whole weight
        vector. Tighter than the 1e-8 default so that the small weights get a good
        relative accuracy too.

    Returns
    -------
    np.ndarray, shape (nbs, nbl)
    """
    nbl = len(ells_proj)
    if ells_proj[0] <= 0:
        raise ValueError(
            'ells_proj must be strictly positive: the spline basis is in log(ell)'
        )
    # evaluated at a single log(ell), returns (B_0, ..., B_{nbl-1})
    spline_basis = make_interp_spline(np.log(ells_proj), np.eye(nbl), k=3, axis=0)

    weights = np.zeros((len(kernel_funcs), nbl))
    for s, kernel in enumerate(kernel_funcs):

        def integrand(ell, kernel=kernel):
            return ell * kernel(ell) * spline_basis(np.log(ell))

        weights[s], _ = quad_vec(integrand, ells_proj[0], ells_proj[-1], epsrel=epsrel)

    return weights


def proj_cov_2d_with_weights(
    cov_hs_ng_4d: np.ndarray, weights_1: np.ndarray, weights_2: np.ndarray
) -> np.ndarray:
    r"""Project a harmonic-space covariance with the weights from
    ``quad_projection_weights``:

        cov[s1, s2] = \sum_ij w_1[s1, i] w_2[s2, j] cov_hs(\ell_i, \ell_j)

    Parameters
    ----------
    cov_hs_ng_4d : np.ndarray, shape (nbl, nbl, zpairs_ab, zpairs_cd)
    weights_1, weights_2 : np.ndarray, shape (nbs, nbl)

    Returns
    -------
    np.ndarray, shape (nbs, nbs, zpairs_ab, zpairs_cd)
    """
    return np.einsum(
        'ai,bj,ijcd->abcd', weights_1, weights_2, cov_hs_ng_4d, optimize=True
    )


def proj_cov_2d_quad_all_scales(
    ells_proj: np.ndarray,
    cov_hs_ng_4d: np.ndarray,
    kernel_1_funcs: list[Callable[[float], float]],
    kernel_2_funcs: list[Callable[[float], float]],
) -> np.ndarray:
    r"""Same integral as ``proj_cov_2d(..., 'quad')``, but for *all* the scale-bin
    pairs (s1, s2) at once:

        cov[s1, s2] = \int d\ell_1 \ell_1 k_1(\ell_1; s1)
                      \int d\ell_2 \ell_2 k_2(\ell_2; s2) cov_hs(\ell_1, \ell_2)

    ``proj_cov_2d`` interpolates cov_hs with a cubic spline in log(ell_2) and then the
    inner integral in log(ell_1). Both splines are linear in the data, so the double
    integral is exactly ``W_1 @ cov_hs @ W_2.T`` (for each zpair), with W the weights
    from ``quad_projection_weights``.

    Parameters
    ----------
    ells_proj : np.ndarray, shape (nbl,)
        Multipoles at which cov_hs_ng_4d is sampled; strictly increasing.
    cov_hs_ng_4d : np.ndarray, shape (nbl, nbl, zpairs_ab, zpairs_cd)
        Harmonic-space non-Gaussian covariance.
    kernel_1_funcs, kernel_2_funcs : list of callables, length nbs
        Projection kernels as functions of (scalar) ell, one per scale bin.

    Returns
    -------
    np.ndarray, shape (nbs, nbs, zpairs_ab, zpairs_cd)
    """
    nbl = len(ells_proj)
    if cov_hs_ng_4d.ndim != 4:
        raise ValueError('cov_hs_ng_4d must be 4D')
    if cov_hs_ng_4d.shape[0] != nbl or cov_hs_ng_4d.shape[1] != nbl:
        raise ValueError(
            f'cov_hs_ng_4d.shape={cov_hs_ng_4d.shape} inconsistent with nbl={nbl}'
        )
    if len(kernel_1_funcs) != len(kernel_2_funcs):
        raise ValueError('kernel_1_funcs and kernel_2_funcs must have the same length')

    weights_1 = quad_projection_weights(ells_proj, kernel_1_funcs)
    weights_2 = quad_projection_weights(ells_proj, kernel_2_funcs)
    return proj_cov_2d_with_weights(cov_hs_ng_4d, weights_1, weights_2)


class CovarianceProjector:
    r"""
    Base class for all projected covariance computations.

    This class provides:
    - Shared setup (survey info, galaxy densities, etc.)
    - The harmonic-space inputs of every projection: Cl and Nl, and the \ell grids
      they are projected over
    - The projection of the Gaussian (SVA, MIX) and non-Gaussian integrands, given
      the kernels of the observable

    Subclasses (CovRealSpace, CovCOSEBIs) implement:
    - Specific projection kernels (k_mu, W_n, etc.)
    - Statistic-specific infrastructure (theta bins, modes, etc.)
    """

    # name of the observable space, set by each subclass ('real', 'cosebis')
    obs_space: str

    def __init__(
        self,
        cfg: dict,
        pvt_cfg: dict,
        cl_3x2pt_5d: np.ndarray,
        nl_3x2pt_4d: np.ndarray,
        ells_proj_g: np.ndarray,
        ells_proj_ng: np.ndarray,
    ):
        """
        Initialize shared infrastructure.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary
        pvt_cfg : dict
            Private configuration with derived quantities
        cl_3x2pt_5d : np.ndarray, shape (n_probes, n_probes, nbl_g, zbins, zbins)
            3x2pt angular power spectra, sampled at ``ells_proj_g``
        nl_3x2pt_4d : np.ndarray, shape (n_probes, n_probes, zbins, zbins)
            Noise power spectra (ell-independent)
        ells_proj_g : np.ndarray
            ell grid over which the Gaussian terms are projected
        ells_proj_ng : np.ndarray
            ell grid over which the non-Gaussian terms are projected
        """
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg

        # harmonic-space inputs
        self.cl_3x2pt_5d = cl_3x2pt_5d
        self.nl_3x2pt_4d = nl_3x2pt_4d
        self.ells_proj_g = ells_proj_g
        self.ells_proj_ng = ells_proj_ng

        # Shared setup
        self.zbins = pvt_cfg['zbins']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.ind_dict = pvt_cfg['ind_dict']
        self.nbs = pvt_cfg['nbs']  # "nbs" = number of scale (theta or n) bins
        self.n_jobs = cfg['misc']['num_threads']
        self.n_probes_hs = 2

        # quad NG projection weights, one (nbs, nbl) array per Bessel order mu;
        # filled on first use by proj_ng_quad
        self.quad_weights = {}

        # both real space and COSEBIs covariances are becessarily split into
        # sva, sn and mix
        base_terms = pvt_cfg['req_terms']
        prepend = [t for t in ['sva', 'sn', 'mix'] if t not in base_terms]
        self.req_terms = prepend + list(base_terms)

        self._set_terms_toloop()
        self._set_neff_and_sigma_eps()
        self._set_pair_counts()

        self.cov_shape_6d = (
            self.nbs,
            self.nbs,
            self.zbins,
            self.zbins,
            self.zbins,
            self.zbins,
        )

    def _set_terms_toloop(self):
        self.terms_toloop = []
        if self.cfg['covariance']['G']:
            self.terms_toloop.extend(('sva', 'sn', 'mix'))
        if self.cfg['covariance']['SSC']:
            self.terms_toloop.append('ssc')
        if self.cfg['covariance']['cNG']:
            self.terms_toloop.append('cng')

    def _set_neff_and_sigma_eps(self):
        self.n_eff_lns = self.cfg['nz']['ngal_lenses']  # clustering
        self.n_eff_src = self.cfg['nz']['ngal_sources']  # lensing
        self.n_eff_2d = np.vstack((self.n_eff_src, self.n_eff_lns))
        self.sigma_eps_i = np.array(self.cfg['covariance']['sigma_eps_i'])
        # the MIX term reads the noise from nl_3x2pt_4d (already zeroed if the noise is
        # excluded), while the SN term is built analytically from sigma_eps_i and n_eff:
        # it needs this flag to be switched off consistently
        self.include_sampling_noise = self.cfg['covariance']['include_sampling_noise']

    def _set_pair_counts(self):
        """Pair counts for the SN term: 'analytic' (ideal, from n_eff and the survey
        area) or 'from_input' (measured, read from file)."""
        self.pair_counts = self.cfg['covariance']['pair_counts']
        self.pc_theta_edges_rad, self.pc_npair_dict = None, None
        if self.pair_counts == 'from_input':
            self.pc_theta_edges_rad, self.pc_npair_dict = load_pair_counts(
                self.cfg['covariance']['pair_counts_filename'], self.zbins
            )

    def _get_measured_pair_counts(self, probe_hs: str) -> np.ndarray:
        """Measured pair counts for the harmonic-space probe ('LL', 'GL', 'GG'), on
        the file's theta grid."""
        if probe_hs not in self.pc_npair_dict:
            raise KeyError(
                f'npair_{probe_hs} is needed for the SN term, but it is not in '
                f'{self.cfg["covariance"]["pair_counts_filename"]}'
            )
        return self.pc_npair_dict[probe_hs]

    def proj_mix_sva_simps_vectorized(
        self,
        cl_integrand_5d: np.ndarray,
        amax_abcd: float,
        mu: int | None = None,
        nu: int | None = None,
        kernel_func_kw: dict | None = None,
    ) -> np.ndarray:
        r"""
        Computes, for every scale-bin pair (s1, s2) and tomographic quadruplet at once,
        the integral of the SVA or MIX covariance projection, using Simpson's rule:

            cov[s1, s2, zi, zj, zk, zl] =
                1/(2 pi A_max) \int d\ell
                \ell K_mu(\ell, s1) K_nu(\ell, s2) f(\ell, zi, zj, zk, zl)

        The trick applied here is to replace the call to simps with a matrix
        multiplication. This is because Simpson is linear in y, so it's a weighted sum
        whose weights depend only on x, i.e., in general:

        \int d\ell f(\ell) = simps(y=f, x=ells)
        -> weights = simps(y=np.eye(nbl), x=ells)
        \int d\ell f(\ell) = weights * f
            (if f and weights are 1D), otherwise weights @ f

        In this case, the projection becomes one (nbs^2, nbl) @ (nbl, zbins^4)
        matmul.

        The kernels depend only on (ell, scale bin, order), so they are built once
        per scale bin rather than once per tomographic quadruplet.

        ``integrand_5d`` is the tomographic part of the integrand, shape
        (nbl, zbins, zbins, zbins, zbins): ``build_cl_integrand_5d_sva`` for SVA,
        ``build_cl_integrand_5d_mix`` for MIX. ``mu``/``nu`` are the
        Bessel orders (real space; unused for COSEBIs, where the kernel comes from
        ``kernel_func_kw['w_ells_arr']``).
        """
        ells = self.ells_proj_g
        nbl = len(ells)
        if cl_integrand_5d.shape[0] != nbl:
            raise ValueError(
                f'cl_integrand_5d has {cl_integrand_5d.shape[0]} ell samples, '
                f'expected {nbl}'
            )

        def build_kernel_array(mu: int | None) -> np.ndarray:
            """
            Builds an array of shape (nbs, nbl) containing the projection kernel for
            the given Bessel order mu.
            The projection kernel does not depend on the tomographic indices, so it
            can be computed only once per scale bin."""
            kernel_list = [
                self.get_projection_kernel_func_of_ell(
                    scale_ix=s,
                    obs_space=self.obs_space,
                    mu=mu,
                    kernel_func_kw=kernel_func_kw or {},
                )(ells)
                for s in range(self.nbs)
            ]
            return np.array(kernel_list)

        # shape: (nbs, nbl)
        k1 = build_kernel_array(mu)
        k2 = k1 if nu == mu else build_kernel_array(nu)

        # simpson simps_weights do not depend on the integrand! I can simply compute
        # them in this way and use them below:
        # simps_weights @ (integrand) and simps(integrand, x) are the same
        simps_weights = simps(y=np.eye(nbl), x=ells, axis=0)  # shape: (nbl,)

        kernel_prod = k1[:, None, :] * k2[None, :, :]  # shape: (nbs, nbs, nbl)
        weights = kernel_prod * simps_weights * ells  # shape: (nbs, nbs, nbl)

        # the matmul is the sum, weighted via the simpson weights
        # I want to sum over the ells, which need to be made the inner axes, with
        # everything else flattened:
        # (nbs * nbs, nbl) @ (nbl, zbins**4) -> (nbs * nbs, zbins**4)
        cov_out_flat = weights.reshape(-1, nbl) @ cl_integrand_5d.reshape(nbl, -1)

        # final reshape to get (nbs, nbs, zbins, zbins, zbins, zbins)
        cov_out_6d = cov_out_flat.reshape(self.cov_shape_6d)

        # apply the 1/(2 pi A_max) prefactor
        cov_out_6d /= 2.0 * np.pi * amax_abcd

        return cov_out_6d

    def get_projection_kernel_func_of_ell(
        self,
        scale_ix: int,
        obs_space: str,
        kernel_func_kw: dict,
        mu: int | None = None,
        arb_kernel_func: Callable | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Based on the scale index (theta for 2PCF, n for COSEBIs) and the observables
        space, construct the projection kernel as a function of ell.

        Parameters
        ----------
        scale_ix : int
            Scale index (theta bin for 2PCF, n for COSEBIs)
        obs_space : str
            Observable space ('real', 'cosebis' or 'arbitrary')
        mu : int | None
            Order of the bessel function for the real space case
        kernel_func_kw : dict
            Keyword arguments for the kernel function
        arb_kernel_func : callable, optional
            Arbitrary kernel function for 'arbitrary' observable space

        Returns
        -------
        kernel_func_of_ell : callable
            Partial kernel functions with signature: kernel(ell)
        """

        if obs_space == 'real':
            # in this case the kernel function is also probe-dependent (through mu)
            theta_l = self.theta_edges[scale_ix]
            theta_u = self.theta_edges[scale_ix + 1]
            kernel_func_of_ell = partial(
                self.k_mu, thetal=theta_l, thetau=theta_u, mu=mu
            )

        elif obs_space == 'cosebis':
            # in this case the kernel function neither probe nor ell-dependent, so I
            # define a simple function of ell that just returns the precomputed array.
            # ! this is only valid if ell is the grid w_ells_arr was computed on
            w_ells_arr = kernel_func_kw['w_ells_arr']

            def kernel_func_of_ell(ell):
                if np.shape(ell) != w_ells_arr.shape[1:]:
                    raise ValueError(
                        'COSEBIs W_n(ell) kernels are precomputed on a fixed ell '
                        f'grid of length {w_ells_arr.shape[1]}, cannot evaluate '
                        f'them at ell of shape {np.shape(ell)}.'
                    )
                return w_ells_arr[scale_ix]

        elif obs_space == 'arbitrary':
            # general case. the arbitrary kernel function must have signature
            # arb_kernel_func(ell, *, scale_ix, **kernel_func_kw)
            kernel_func_of_ell = partial(
                arb_kernel_func, scale_ix=scale_ix, **kernel_func_kw
            )

        else:
            raise ValueError(f'Observable space {obs_space} not recognized!')

        return kernel_func_of_ell

    def proj_ng_quad(
        self, cov_hs_ng_4d: np.ndarray, mu: int | None, nu: int | None
    ) -> np.ndarray:
        """Project a harmonic-space non-Gaussian covariance block with quad_vec.

        Same result as ``proj_cov_2d_quad_all_scales``, but the projection weights
        (the expensive part) are computed once per Bessel order and reused for all
        the probe blocks and covariance terms.

        Parameters
        ----------
        cov_hs_ng_4d : np.ndarray, shape (nbl, nbl, zpairs_ab, zpairs_cd)
            Harmonic-space covariance, sampled at ``self.ells_proj_ng``.
        mu, nu : int or None
            Bessel orders of the two probes (real space only).

        Returns
        -------
        np.ndarray, shape (nbs, nbs, zpairs_ab, zpairs_cd)
        """
        # quad_vec evaluates the kernels at arbitrary ell, while the COSEBIs W_n(ell)
        # kernels are only available on the (fixed) ells_proj_ng grid
        if self.obs_space != 'real':
            raise NotImplementedError(
                f'quad NG projection is only implemented for real space, '
                f'got obs_space={self.obs_space!r}. Use simps instead.'
            )

        for order in (mu, nu):
            if order not in self.quad_weights:
                kernel_funcs = [
                    self.get_projection_kernel_func_of_ell(
                        scale_ix=s,
                        obs_space=self.obs_space,
                        mu=order,
                        kernel_func_kw={},
                    )
                    for s in range(self.nbs)
                ]
                self.quad_weights[order] = quad_projection_weights(
                    self.ells_proj_ng, kernel_funcs
                )

        return proj_cov_2d_with_weights(
            cov_hs_ng_4d, self.quad_weights[mu], self.quad_weights[nu]
        )
