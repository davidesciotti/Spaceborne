"""
Base class for projected covariance computations.

This module provides the shared infrastructure for computing covariances
of projected statistics (2PCF, COSEBIs, etc.) from harmonic-space C_ℓ.

The key insight is that different statistics share the same integrand building
(SVA, MIX terms) but differ in how they project from C_ℓ to the observable space.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
from scipy.integrate import quad_vec
from scipy.integrate import simpson as simps
from scipy.interpolate import make_interp_spline

from spaceborne import constants as const


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


def get_delta_tomo(probe_a_ix: int, probe_b_ix: int, zbins: int) -> np.ndarray:
    return np.eye(zbins) if probe_a_ix == probe_b_ix else np.zeros((zbins, zbins))


# ! ==================== build Cl integrand for SVA and MIX covs =====================
def build_cl_integrand_5d_sva(cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
    """
    Build the SVA (sample variance) integrand in harmonic space.

    This integrand is UNIVERSAL - it's the same for all projection methods
    (real space, COSEBIs, band powers, etc.). The formula comes from the
    Gaussian covariance of C_ℓ:

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
        ell_min = ells_proj[0]
        ell_max = ells_proj[-1]

        zpairs_ab = cov_hs_ng_4d.shape[2]
        zpairs_cd = cov_hs_ng_4d.shape[3]
        zpairs_flat = zpairs_ab * zpairs_cd

        # flatten covariance to shape (nbl, nbl, zpairs_flat) for easier
        # interpolation and integration
        cov_ell1ell2 = cov_hs_ng_4d.reshape(nbl, nbl, zpairs_flat)
        ell2_integral = np.zeros((nbl, zpairs_flat))

        # Inner integral (d\ell2)
        for i in range(nbl):
            # for each ell1, interpolate along ell2
            cov_ell1ell2_interp_func = make_interp_spline(
                ells_proj, cov_ell1ell2[i], k=3, axis=0
            )

            # define callable of ell2 as required by quad. note that the kernel,
            # which is the oscillatory part, is not interpolated!
            def ell2_integrand_func(ells_2, _interp=cov_ell1ell2_interp_func):
                return ells_2 * kernel_2_func_of_ell(ells_2) * _interp(ells_2)

            # evaluate integral
            ell2_integral[i], _ = quad_vec(ell2_integrand_func, ell_min, ell_max)

        # now do the same along ell1: interpolate the result...
        ell2_integral_interp_func = make_interp_spline(
            ells_proj, ell2_integral, k=3, axis=0
        )

        # define callable of ell1
        def ell1_integrand_func(ells_1, _interp=ell2_integral_interp_func):
            return ells_1 * kernel_1_func_of_ell(ells_1) * _interp(ells_1)

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
                      └──────────────── inner(\ell_1; s2) ────────────────┘

    Calling ``proj_cov_2d`` once per (s1, s2) costs nbs^2 * (nbl + 1) adaptive
    integrations. Here it costs nbs + 1, thanks to three observations:

    1. the inner integral depends on s2 but not on s1, so it is computed once per s2
       rather than once per (s1, s2);
    2. quad_vec integrates vector-valued functions, so the loop over ell_1 in the
       inner integral becomes a single integration of a vector with one entry per
       (ell_1, zpair);
    3. for the same reason, the outer integral is a single integration of a vector
       with one entry per (s1, s2, zpair).

    The (zpair_ab, zpair_cd) axes are flattened into a single zpair axis throughout.

    Notes
    -----
    Batching changes where quad_vec subdivides the integration range, since it
    refines on the norm of the whole output vector: the results agree with the
    one-(s1, s2)-at-a-time version to within the quad_vec tolerance, but are not
    bitwise identical.

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

    zpairs_ab, zpairs_cd = cov_hs_ng_4d.shape[2], cov_hs_ng_4d.shape[3]
    nbs = len(kernel_1_funcs)

    # flatten the two zpair axes: (nbl, nbl, zpairs_ab, zpairs_cd) -> (nbl, nbl, zpairs)
    cov_ell1ell2 = cov_hs_ng_4d.reshape(nbl, nbl, zpairs_ab * zpairs_cd)

    # inner integral, over ell2. Shape (nbs, nbl, zpairs) = (s2, ell1, zpair)
    inner = _quad_inner_integral_over_ell2(ells_proj, cov_ell1ell2, kernel_2_funcs)

    # outer integral, over ell1. Shape (nbs, nbs, zpairs) = (s1, s2, zpair)
    cov_out = _quad_outer_integral_over_ell1(ells_proj, inner, kernel_1_funcs)

    return cov_out.reshape(nbs, nbs, zpairs_ab, zpairs_cd)


def _quad_inner_integral_over_ell2(
    ells_proj: np.ndarray,
    cov_ell1ell2: np.ndarray,
    kernel_2_funcs: list[Callable[[float], float]],
) -> np.ndarray:
    r"""For each scale bin s2, compute

        inner(\ell_1; s2) = \int d\ell_2 \ell_2 k_2(\ell_2; s2) cov(\ell_1, \ell_2)

    for all ell_1 and zpairs at once.

    Parameters
    ----------
    ells_proj : np.ndarray, shape (nbl,)
    cov_ell1ell2 : np.ndarray, shape (nbl, nbl, zpairs), axes (ell1, ell2, zpair)
    kernel_2_funcs : list of callables, length nbs

    Returns
    -------
    np.ndarray, shape (nbs, nbl, zpairs), axes (s2, ell1, zpair)
    """
    ell_min, ell_max = ells_proj[0], ells_proj[-1]

    # Interpolate the covariance along ell2 (axis 1). Evaluated at a single ell2, the
    # spline returns the whole (ell1, zpair) slice, shape (nbl, zpairs): this is what
    # lets quad_vec integrate over ell2 for all ell1 at once.
    # Only the covariance is interpolated: the oscillatory kernel is evaluated
    # exactly at every point quad_vec asks for.
    cov_spline_in_ell2 = make_interp_spline(ells_proj, cov_ell1ell2, k=3, axis=1)

    # The loop over s2 is deliberately not batched like the outer integral: the
    # vector here is already nbl times larger, and quad_vec keeps one copy of it per
    # subinterval, so adding an s2 axis would multiply the memory by nbs.
    inner = []
    for kernel_2 in kernel_2_funcs:

        def integrand(ell2, kernel_2=kernel_2):
            return ell2 * kernel_2(ell2) * cov_spline_in_ell2(ell2)  # (nbl, zpairs)

        integral, _ = quad_vec(integrand, ell_min, ell_max)
        inner.append(integral)

    return np.stack(inner, axis=0)


def _quad_outer_integral_over_ell1(
    ells_proj: np.ndarray,
    inner: np.ndarray,
    kernel_1_funcs: list[Callable[[float], float]],
    epsrel: float = 1e-10,
) -> np.ndarray:
    r"""For all scale-bin pairs (s1, s2) at once, compute

        cov(s1, s2) = \int d\ell_1 \ell_1 k_1(\ell_1; s1) inner(\ell_1; s2)

    Parameters
    ----------
    ells_proj : np.ndarray, shape (nbl,)
    inner : np.ndarray, shape (nbs, nbl, zpairs), axes (s2, ell1, zpair)
        Output of ``_quad_inner_integral_over_ell2``.
    kernel_1_funcs : list of callables, length nbs
    epsrel : float
        Relative tolerance of quad_vec. Tighter than its 1e-8 default because it
        applies to the norm of the whole (s1, s2, zpair) vector, so the entries much
        smaller than the largest one (e.g. the large-theta bins, down to ~1e-6 of the
        maximum) would otherwise get a correspondingly poorer relative accuracy.
        With 1e-10 they are at least as accurate as with one integration per (s1, s2)
        at the default tolerance, for ~25% more integrand evaluations.

    Returns
    -------
    np.ndarray, shape (nbs, nbs, zpairs), axes (s1, s2, zpair)
    """
    ell_min, ell_max = ells_proj[0], ells_proj[-1]

    # Interpolate the inner integral along ell1 (axis 1). Evaluated at a single ell1,
    # the spline returns shape (nbs, zpairs), i.e. every (s2, zpair).
    inner_spline_in_ell1 = make_interp_spline(ells_proj, inner, k=3, axis=1)

    def integrand(ell1):
        # all the s1 kernels at this ell1, shape (nbs,)
        kernels_1 = np.array([kernel_1(ell1) for kernel_1 in kernel_1_funcs])
        # inner integral at this ell1, shape (nbs, zpairs)
        inner_at_ell1 = inner_spline_in_ell1(ell1)

        # broadcast to (s1, s2, zpair)
        return ell1 * kernels_1[:, None, None] * inner_at_ell1[None, :, :]

    cov_out, _ = quad_vec(integrand, ell_min, ell_max, epsrel=epsrel)
    return cov_out


class CovarianceProjector:
    """
    Base class for all projected covariance computations.

    This class provides:
    - Shared setup (survey info, galaxy densities, etc.)
    - The harmonic-space inputs of every projection: C_ℓ and N_ℓ, and the ℓ grids
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

        # both real space and COSEBIs covariances are becessarily split into
        # sva, sn and mix
        base_terms = pvt_cfg['req_terms']
        prepend = [t for t in ['sva', 'sn', 'mix'] if t not in base_terms]
        self.req_terms = prepend + list(base_terms)

        self._set_terms_toloop()
        self._set_neff_and_sigma_eps()

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
            # define a simple function of ell that just returns the precomputed array
            w_ells_arr = kernel_func_kw['w_ells_arr']

            def kernel_func_of_ell(ell):
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

        Builds the projection kernels for every scale bin and calls
        ``proj_cov_2d_quad_all_scales``.

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

        def kernels_for_all_scale_bins(order):
            return [
                self.get_projection_kernel_func_of_ell(
                    scale_ix=s, obs_space=self.obs_space, mu=order, kernel_func_kw={}
                )
                for s in range(self.nbs)
            ]

        return proj_cov_2d_quad_all_scales(
            ells_proj=self.ells_proj_ng,
            cov_hs_ng_4d=cov_hs_ng_4d,
            kernel_1_funcs=kernels_for_all_scale_bins(mu),
            kernel_2_funcs=kernels_for_all_scale_bins(nu),
        )
