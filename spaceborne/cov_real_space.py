"""
This module contains functions to compute the covariance matrix in real space.
Nomenclature of the functions/variables:
hs = harmonic space
rs = real space
sva = sample variance
sn = sampling noise
mix = mixed term
"""

# TODO the NG cov has not been re-tested against OC
# TODO the NG cov needs a smaller number of ell bins for the simpson integration! It's
# TODO unpractical to compute it in 1000 ell values

import itertools
import warnings
from functools import partial

import numpy as np
import pyccl as ccl
import pylevin as levin
from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import cov_projector as cp
from spaceborne import sb_lib as sl
from spaceborne.cov_projector import CovarianceProjector
from spaceborne.twobessel_fang import TwoBessel

warnings.filterwarnings(
    'ignore', message=r'.*invalid escape sequence.*', category=SyntaxWarning
)

warnings.filterwarnings(
    'ignore',
    message=r'.*invalid value encountered in divide.*',
    category=RuntimeWarning,
)

_UNSET = object()


def b_mu(x, mu):
    r"""Implements the piecewise definition of the bracketed term b_mu(x)
    from Eq. (E.2) in Joachimi et al. (2008).
    These are just the results of
    \int_{\theta_l}^{\theta_u} d\theta \theta J_\mu(\ell \theta)
    """
    if mu == 0:
        return x * sl.j1(x)
    elif mu == 2:
        return -x * sl.j1(x) - 2.0 * sl.j0(x)
    elif mu == 4:
        # be careful with x=0!
        return (x - 8.0 / x) * sl.j1(x) - 8.0 * sl.j2(x)
    else:
        raise ValueError('mu must be one of {0,2,4}.')


def b_mu_nobessel(x, mu):
    """same as b_mu but returning the *order* of the Bessel functions"""
    if mu == 0:
        return [(x, 1)]
    elif mu == 2:
        return [(-x, 1), (-2.0, 0)]
    elif mu == 4:
        # be careful with x=0!
        return [((x - 8.0 / x), 1), (-8.0, 2)]
    else:
        raise ValueError('mu must be one of {0,2,4}.')


def k_mu(ell, *, thetal, thetau, mu):
    r"""Computes the kernel K_mu(ell * theta_i) in Eq. (E.2):

    K_mu(l * theta_i) = 2 / [ (theta_u^2 - theta_l^2) * l^2 ]
                        * [ b_mu(l * theta_u) - b_mu(l * theta_l) ].
    """
    prefactor = 2.0 / ((thetau**2 - thetal**2) * (ell**2))
    return prefactor * (b_mu(ell * thetau, mu) - b_mu(ell * thetal, mu))


def k_mu_nobessel(ell, *, thetal, thetau, mu):
    """
    Generates a list of decomposed terms for the kernel K_mu.

    Returns: List of tuples (const_coeff, bessel_order, theta)
    """
    prefactor = 2.0 / ((thetau**2 - thetal**2) * ell**2)

    terms_u = b_mu_nobessel(ell * thetau, mu)
    terms_l = b_mu_nobessel(ell * thetal, mu)

    all_terms = []
    # Add terms for theta_u
    for const_coeff, bessel_order in terms_u:
        all_terms.append((prefactor * const_coeff, bessel_order, thetau))

    # Add terms for theta_l (with a minus sign)
    for const_coeff, bessel_order in terms_l:
        all_terms.append((-prefactor * const_coeff, bessel_order, thetal))

    return all_terms


def kmuknu_nobessel(k_mu_terms, k_nu_terms):
    """
    Computes the product of two expanded kernels K_mu and K_nu.

    Returns: List of tuples (final_coeff, n1, theta1, n2, theta2)
    """
    product_terms = []
    for term1 in k_mu_terms:
        for term2 in k_nu_terms:
            c1, n1, t1 = term1
            c2, n2, t2 = term2

            # Product of constants
            final_coeff = c1 * c2

            product_terms.append((final_coeff, n1, t1, n2, t2))

    return product_terms


# ! ====================== COV RS W/ SIMPSON INTEGRATION ===============================


def t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i):
    """
    Returns tau^{sn}_{(ij)(mn)} as a (zbins, zbins) array over (i,j) of the FIRST pair (ij),
    consistent with Eq. (65).
    Assumes sigma_eps_i is sigma_{epsilon1,i} (std); if it is already variance, set sig2=sigma_eps_i.
    """
    sig2 = (
        sigma_eps_i**2
    )  # change to: sig2 = sigma_eps_i  if sigma_eps_i is already variance

    # all-source case (e.g. xip/xip or xim/xim)
    if probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 0:
        # tau(i,j) = 2 * sig2[i] * sig2[j]
        return 2.0 * sig2[:, None] * sig2[None, :]

    # all-lens case (e.g. gg/gg)
    if probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 1:
        return np.ones((zbins, zbins), dtype=float)

    # mixed case: each pair contains one lens and one source (e.g. gt/gt)
    if {probe_a_ix, probe_b_ix} == {0, 1} and {probe_c_ix, probe_d_ix} == {0, 1}:
        # Eq. (65) says tau = sigma^2_{epsilon1, source_index_in_(ij)}.
        if probe_a_ix == 0:  # (ij) = (source, lens) -> source index is i
            return sig2[:, None] * np.ones((1, zbins))
        else:  # (ij) = (lens, source) -> source index is j
            return np.ones((zbins, 1)) * sig2[None, :]

    return np.zeros((zbins, zbins), dtype=float)


def _t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i):
    # TODO move from probe indices to probe names!
    t_munu = np.zeros((zbins, zbins))

    for zi in range(zbins):
        for zj in range(zbins):
            # xipxip or ximxim
            if probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 0:
                t_munu[zi, zj] = 2 * sigma_eps_i[zi] ** 2 * sigma_eps_i[zj] ** 2

            # gggg
            elif probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 1:
                t_munu[zi, zj] = 1

            elif (
                (probe_a_ix == 0 and probe_b_ix == 1)
                or (probe_b_ix == 0 and probe_a_ix == 1)
            ) and (
                (probe_c_ix == 0 and probe_d_ix == 1)
                or (probe_d_ix == 0 and probe_c_ix == 1)
            ):
                t_munu[zi, zi] = sigma_eps_i[zi] ** 2

            else:
                t_munu[zi, zj] = 0

    return t_munu


def t_mix(probe_a_ix, zbins, sigma_eps_i):
    t_munu = np.zeros(zbins)

    # xipxip or ximxim
    if probe_a_ix == 0:
        t_munu = sigma_eps_i**2

    # gggg
    elif probe_a_ix == 1:
        t_munu = np.ones(zbins)

    return t_munu


def split_probe_ix(probe_ix):
    if probe_ix in (0, 1):
        return 0, 0
    elif probe_ix == 2:
        return 1, 0
    elif probe_ix == 3:
        return 1, 1
    else:
        raise ValueError(f'Invalid probe index: {probe_ix}. Expected 0, 1, 2, or 3.')


def integrate_bessel_single_wrapper(
    cov_2d, mu, ell, theta_centers, n_jobs,
    logx, logy, n_sub, diagonal, n_bisec_max, rel_acc, boost_bessel, verbose,
):  # fmt: skip
    assert cov_2d.ndim == 2, 'the input integrand must be 2D'

    integral_type = 1  # single cilyndrical bessel
    nbt = len(theta_centers)

    integrand = cov_2d

    # Constructor of the class
    lp = levin.pylevin(
        type=integral_type,
        x=ell,
        integrand=integrand,
        logx=logx,
        logy=logy,
        nthread=n_jobs,
        diagonal=diagonal,
    )

    lp.set_levin(
        n_col_in=n_sub,
        maximum_number_bisections_in=n_bisec_max,
        relative_accuracy_in=rel_acc,
        super_accurate=boost_bessel,
        verbose=verbose,
    )

    # N is the number of integrals to be computed
    # M is the number of arguments at which the integrals are evaluated
    N = integrand.shape[1]
    M = nbt
    result_levin = np.zeros((M, N))  # allocate the result

    lp.levin_integrate_bessel_single(
        x_min=ell[0] * np.ones(nbt),
        x_max=ell[-1] * np.ones(nbt),
        k=theta_centers,
        ell=(mu * np.ones(nbt)).astype(int),
        result=result_levin,
    )

    return result_levin


def dl1dl2_binavg_bessel_wrapper(
    cov_hs: np.ndarray,
    mu: int,
    nu: int,
    ells: np.ndarray,
    theta_edges: np.ndarray,
    n_jobs: int,
    levin_prec_kw: dict,
):
    r"""
    Wrapper function to compute the bin-averaged double Bessel integral:

    Cov(theta_p, theta_q) =
        \int d\ell_1 \ell_1 K_mu(\ell_1, theta_p) *
        \int d\ell_2 \ell_2 K_nu(\ell_2, theta_q) * C(\ell_1, \ell_2)

    where K_mu is the analytic bin-averaging kernel (Eq. E.2), decomposed via
    k_mu_nobessel into a sum of weighted Bessel terms at the bin edges.

    Parameters
    ----------
    cov_hs: np.ndarray
        Harmonic-space covariance, shape (nbl, nbl, ...).
    mu, nu: int
        Bessel orders for the two projections.
    ells: np.ndarray
        ell grid, shape (nbl,).
    theta_edges: np.ndarray
        Bin edges in radians, shape (nbt + 1,).
    n_jobs: int
        Number of parallel threads for Levin integration.
    """
    nbl = len(ells)
    nbt = len(theta_edges) - 1

    assert cov_hs.shape[0] == cov_hs.shape[1] == nbl, (
        'cov_hs shape must be (ell_bins, ell_bins, ...)'
    )
    original_shape_no_scale = cov_hs.shape[2:]
    flattened_size = (
        int(np.prod(original_shape_no_scale)) if original_shape_no_scale else 1
    )

    # Inner integral: for each fixed ell1, integrate over ell2 using ell2*K_nu.
    # K_nu decomposes as a sum of weighted Bessel terms evaluated at the bin edges.
    partial_results = np.zeros((nbl, nbt, flattened_size))
    for ell1_ix in tqdm(range(nbl), desc='ell'):
        base = cov_hs[ell1_ix, ...].reshape(nbl, -1)  # (nbl, N)
        for q in range(nbt):
            for coeff, ord_bes, theta in k_mu_nobessel(
                ells, thetal=theta_edges[q], thetau=theta_edges[q + 1], mu=nu
            ):
                result = integrate_bessel_single_wrapper(
                    base * (ells * coeff)[:, None],
                    ord_bes,
                    ells,
                    np.array([theta]),
                    n_jobs,
                    **levin_prec_kw,
                )  # (1, N)
                partial_results[ell1_ix, q] += result[0]

    # Outer integral: for each fixed theta_q, integrate over ell1 using ell1*K_mu.
    final_result = np.zeros((nbt, nbt, flattened_size))
    for q in tqdm(range(nbt), desc='theta'):
        base_second = partial_results[:, q, :]  # (nbl, N)
        for p in range(nbt):
            for coeff, ord_bes, theta in k_mu_nobessel(
                ells, thetal=theta_edges[p], thetau=theta_edges[p + 1], mu=mu
            ):
                result = integrate_bessel_single_wrapper(
                    base_second * (ells * coeff)[:, None],
                    ord_bes,
                    ells,
                    np.array([theta]),
                    n_jobs,
                    **levin_prec_kw,
                )  # (1, N)
                final_result[p, q] += result[0]

    return final_result.reshape(nbt, nbt, *original_shape_no_scale)


def dl1dl2_nobinavg_bessel_wrapper(
    cov_hs: np.ndarray,
    mu: int,
    nu: int,
    ells: np.ndarray,
    thetas: np.ndarray,
    n_jobs: int,
    levin_prec_kw: dict,
):
    r"""
    Wrapper function to compute the double Bessel integral of the form
    \int d\ell_1 * \ell_1 * J_mu(\theta_1 \ell_1) *
    \int d\ell_2 * \ell_2 * J_nu(\theta_2 \ell_2) *
    integrand(\ell_1, \ell_2)

    Note that the multiplication by \ell_1, \ell_2 is done inside this function.

    Parameters
    ----------
    cov_hs: np.ndarray
        The input covariance matrix in harmonic space. The first two dimensions
        correspond to the ell bins for the two integrations, ie (nbl, nbl, ...)
    mu: int
        The order of the Bessel function for the first integration.
    nu: int
        The order of the Bessel function for the second integration.
    ells: np.ndarray
        The array of ell values corresponding to the harmonic space covariance.
    thetas: np.ndarray of shape (theta_bins)
        The array of theta values (in radians) for the real-space covariance.
    n_jobs: int
        The number of parallel jobs to use for the Bessel integration.

    Returns
    -------
    cov_rs_6d: np.ndarray
        The projected covariance matrix in real space. The first two dimensions
        correspond to the theta bins, and the remaining dimensions correspond to
        the tomographic bin indices.

    """
    nbl = len(ells)

    assert cov_hs.shape[0] == cov_hs.shape[1] == nbl, (
        'cov_hs shape must be (ell_bins, ell_bins, ...)'
    )
    original_shape_no_scale = cov_hs.shape[2:]

    # First integration: for each fixed ell1, integrate over ell2.
    partial_results = []
    for ell1_ix in tqdm(range(nbl), desc='ell'):
        # Extract the 2D slice for fixed ell1.
        integrand = cov_hs[ell1_ix, ...].reshape(nbl, -1) * ells[:, None]
        partial_int = integrate_bessel_single_wrapper(
            integrand, nu, ells, thetas, n_jobs, **levin_prec_kw
        )
        partial_results.append(partial_int)

    # Stack partial results along the ell1 direction.
    partial_results = np.stack(partial_results, axis=0)

    # Second integration: integrate over ell1.
    nbt = partial_results.shape[1]
    flattened_size = partial_results.shape[2]
    final_result = np.zeros((nbt, nbt, flattened_size))

    for theta_idx in tqdm(range(nbt), desc='theta'):
        # For fixed theta from the first integration, extract the integrand:
        integrand_second = partial_results[:, theta_idx, :] * ells[:, None]
        final_int = integrate_bessel_single_wrapper(
            integrand_second, mu, ells, thetas, n_jobs, **levin_prec_kw
        )
        final_result[:, theta_idx, :] = final_int

    cov_rs_out = final_result.reshape(nbt, nbt, *original_shape_no_scale)

    return cov_rs_out


def levin_integrate_bessel_double_wrapper(
    integrand, x_values, bessel_args, bessel_type, ell_1, ell_2, n_jobs,
    logx, logy, n_sub, diagonal, n_bisec_max, rel_acc, boost_bessel, verbose,
):  # fmt: skip
    assert integrand.ndim == 2, 'the integrand must be 2D'
    assert integrand.shape[0] == len(x_values), (
        'integrand and x_values must have the same first dimension'
    )
    # number of integrals to perform
    N = integrand.shape[-1]
    # number of arguments at which the integrals are evaluated
    # TODO this might change in the future?
    M = len(bessel_args) ** 2

    # Constructor of the class
    lp = levin.pylevin(
        type=bessel_type,
        x=x_values,
        integrand=integrand,
        logx=logx,
        logy=logy,
        nthread=n_jobs,
        diagonal=diagonal,
    )

    lp.set_levin(
        n_col_in=n_sub,
        maximum_number_bisections_in=n_bisec_max,
        relative_accuracy_in=rel_acc,
        super_accurate=boost_bessel,
        verbose=verbose,
    )

    result_levin = np.zeros((M, N))  # allocate the result
    X, Y = np.meshgrid(bessel_args, bessel_args, indexing='ij')
    theta1_flat = X.reshape(M)
    theta2_flat = Y.reshape(M)

    lp.levin_integrate_bessel_double(
        x_min=x_values[0] * np.ones(M),
        x_max=x_values[-1] * np.ones(M),
        k_1=theta1_flat,
        k_2=theta2_flat,
        ell_1=(ell_1 * np.ones(M)).astype(int),
        ell_2=(ell_2 * np.ones(M)).astype(int),
        result=result_levin,
    )

    return result_levin


def twopcf_wrapper(
    cosmo, zi, zj, ell_grid, theta_grid, cl_3D, correlation_type, method
):
    return ccl.correlation(
        cosmo=cosmo,
        ell=ell_grid,
        C_ell=cl_3D[:, zi, zj],
        theta=theta_grid,
        method=method,
        type=correlation_type,
    )


def regularize_by_eigenvalue_cutoff(cov, threshold=1e-14):
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(cov)

    # Invert only the eigenvalues above the threshold
    eigvals_inv = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)

    # Reconstruct the inverse covariance matrix
    cov_inv = (eigvecs * eigvals_inv) @ eigvecs.T

    return cov_inv


def integrate_single_bessel_pair(
    integrand, x_values, ord_bes_1, theta1, ord_bes_2, theta2,
    bessel_type, n_jobs, logx, logy, n_sub, n_bisec_max,
    rel_acc, boost_bessel, verbose, diagonal
):  # fmt: skip
    """
    A simplified wrapper to integrate f(x) * J_n1(theta1*x) * J_n2(theta2*x).
    It computes the integral for a single pair of Bessel functions.
    """
    assert integrand.ndim == 2, 'The integrand must be 2D [x_values, combinations]'
    N = integrand.shape[1]  # Number of parallel integrals (e.g., for different z pairs)

    # Constructor of the class
    lp = levin.pylevin(
        type=bessel_type,
        x=x_values,
        integrand=integrand,
        logx=logx,
        logy=logy,
        nthread=n_jobs,
        diagonal=diagonal,  # We are always off-diagonal here
    )

    lp.set_levin(
        n_col_in=n_sub,
        maximum_number_bisections_in=n_bisec_max,
        relative_accuracy_in=rel_acc,
        super_accurate=boost_bessel,
        verbose=verbose,
    )

    # The result will have shape (1, N) because we compute for one (k1, k2) pair
    result_levin = np.zeros((1, N))

    lp.levin_integrate_bessel_double(
        x_min=np.array([x_values[0]]),
        x_max=np.array([x_values[-1]]),
        k_1=np.array([theta1]),
        k_2=np.array([theta2]),
        ell_1=np.array([ord_bes_1], dtype=int),
        ell_2=np.array([ord_bes_2], dtype=int),
        result=result_levin,
    )

    # Return the flat 1D array of results
    return result_levin[0]


def proj_cov_2d_fftlog(
    cov_hs_ell1ell2_in,
    ells_proj,
    theta_edges,
    theta_centers,
    mu,
    nu,
    nu1=1.01,  # for accuracy issues, play witin ~ [0.5, 1.5]
    nu2=1.01,  # for accuracy issues, play witin ~ [0.5, 1.5]
    N_extrap_low=0,  # number of extrapolation points at high ell (default 0, no extrapolation)
    N_extrap_high=0,  # number of extrapolation points at low ell (default 0, no extrapolation)
    c_window_width=0.25,
    N_pad=0,  # pads the input with 0s (less precise than extrapolation, but faster)
):
    r"""
    Project the NG covariance from harmonic to real space using the FFTLog-based
    double Hankel transform (TwoBessel, Fang et al. 2020).

    Computes the bin-averaged result:

        C_RS(θ_p, θ_q) = ∫ dell1 ell1 K_mu(ell1, θ_p)
        * ∫ dell2 ell2 K_nu(ell2, θ_q) C_HS(ell1, ell2)

    where K_mu(ell, θ_p) = 2/(θ_u² - θ_l²) ∫_{θ_l}^{θ_u} θ' J_mu(ellθ') dθ' is the
    bin-averaging kernel (Joachimi et al. 2008). The analytic bin-averaging is handled
    by ``TwoBessel.two_Bessel_binave`` via the ``g_l_smooth`` kernel.

    Parameters
    ----------
    cov_hs_ell1ell2_in : np.ndarray, shape (nbl, nbl, tomo_shape)
    ells : np.ndarray, shape (nbl,), log-spaced ell grid
    theta_edges : np.ndarray, shape (nbt+1,), log-spaced bin edges in radians
    theta_centers : np.ndarray, shape (nbt,), bin centres in radians
    mu, nu : int, cylindrical Bessel orders
    nu1, nu2 : float, FFTLog power-law bias parameters (default 1.01)
    c_window_width : float, smoothing fraction for FFT coefficients (default 0.25)
    N_pad : int, zero-padding length (default 0)

    Returns
    -------
    np.ndarray, shape (nbt, nbt, zpairs_ab, zpairs_cd)
    """

    # preparations and some sanity checks
    nbl = cov_hs_ell1ell2_in.shape[0]
    assert cov_hs_ell1ell2_in.shape[1] == nbl, (
        'cov_hs_ell1ell2_in must have shape (nbl, nbl, ...)'
    )
    assert len(ells_proj) == nbl, (
        'ells_proj length must match cov_hs_ell1ell2_in shape, '
        f'got {len(ells_proj)} vs {nbl}.'
    )

    nbt = len(theta_centers)

    if nbl % 2 != 0:
        raise ValueError(
            f'ells must have even length for FFTLog{nbl}. '
            'Set ell_bins_proj_nongauss to an even number.'
        )

    dlnells = np.diff(np.log(ells_proj))
    if not np.allclose(dlnells, dlnells[0], rtol=1e-8, atol=0.0):
        raise ValueError(
            'FFTLog requires a log-spaced ell grid. '
            'Ensure ells_proj_ng is generated with np.geomspace.'
        )

    dln_theta_edges = np.diff(np.log(theta_edges))
    if not np.allclose(dln_theta_edges, dln_theta_edges[0], rtol=1e-8, atol=0.0):
        raise ValueError(
            "integration_method='FFTLog' requires log-spaced theta bins."
        )

    # Constant log bin width (requires log-spaced theta bins)
    dlntheta = np.log(theta_edges[1] / theta_edges[0])

    # Warn if theta_centers fall outside the natural FFTLog output range
    theta_out_min = 1.0 / ells_proj[-1]
    theta_out_max = 1.0 / ells_proj[0]

    if theta_centers[0] < theta_out_min or theta_centers[-1] > theta_out_max:
        warnings.warn(
            f'theta_centers [{theta_centers[0]}, {theta_centers[-1]}] rad '
            f'extends outside FFTLog output range '
            f'[{theta_out_min}, {theta_out_max}] rad. '
            'Consider widening ell_min_proj / ell_max_proj.',
            stacklevel=2,
        )

    tomo_shape = cov_hs_ell1ell2_in.shape[2:]
    tomo_elements = int(np.prod(tomo_shape)) if tomo_shape else 1

    cov_hs_3d = cov_hs_ell1ell2_in.reshape(nbl, nbl, tomo_elements)

    result_3d = np.zeros((nbt, nbt, tomo_elements))

    for tomo_ix in range(tomo_elements):
        # Build integrand f(ell1,ell2) = ell1² ell2² C_HS so that
        # TwoBessel gives ∫dell1 ell1 J_mu ∫dell2 ell2 J_nu · C_HS
        fx1x2 = (
            cov_hs_3d[:, :, tomo_ix] * ells_proj[:, None] ** 2 * ells_proj[None, :] ** 2
        )
        tb = TwoBessel(
            x1=ells_proj,
            x2=ells_proj,
            fx1x2=fx1x2,
            nu1=nu1,
            nu2=nu2,
            N_extrap_low=N_extrap_low,
            N_extrap_high=N_extrap_high,
            c_window_width=c_window_width,
            N_pad=N_pad,
        )

        theta1_out, theta2_out, integral = tb.two_Bessel_binave(
            mu, nu, dlntheta, dlntheta
        )

        # Interpolate onto the desired theta grid (log-log 2D spline)
        interp = RectBivariateSpline(
            np.log(theta1_out), np.log(theta2_out), integral, kx=3, ky=3
        )
        result_3d[:, :, tomo_ix] = interp(np.log(theta_centers), np.log(theta_centers))

    return result_3d.reshape(nbt, nbt, *tomo_shape)


def proj_cov_2d_parallel_helper(
    s1: int,
    s2: int,
    theta_edges_fine: np.ndarray,
    mu: int,
    nu: int,
    integration_method: str,
    ells_proj_ng: np.ndarray,
    cov_hs_ng_4d: np.ndarray,
):
    # TODO make kernel agnostic using kernel builder
    # TODO move to covariance_projector.py
    kernel_1 = partial(
        k_mu, thetal=theta_edges_fine[s1], thetau=theta_edges_fine[s1 + 1], mu=mu
    )
    kernel_2 = partial(
        k_mu, thetal=theta_edges_fine[s2], thetau=theta_edges_fine[s2 + 1], mu=nu
    )

    block = cp.proj_cov_2d(
        ells_proj=ells_proj_ng,
        cov_hs_ng_4d=cov_hs_ng_4d,
        kernel_1_func_of_ell=kernel_1,
        kernel_2_func_of_ell=kernel_2,
        integration_method=integration_method,
    )

    return s1, s2, block


# ! ====================================================================================
# ! ====================================================================================
# ! ====================================================================================


class CovRealSpace(CovarianceProjector):
    def __init__(self, cfg, pvt_cfg, mask_obj):
        super().__init__(cfg, pvt_cfg, mask_obj)

        self.obs_space = 'real'

        # ! instantiate cov_dict
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_rs_2d']
        dims = ['6d', '4d', '2d']
        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='real')
            for probe in self.req_probe_combs_2d
        ]
        _req_probe_combs_2d.append('3x2pt')
        # note: self.req_terms is instantiated in the parent class
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

        self.symmetrize_output_dict = pvt_cfg['symmetrize_output_dict']

        # setters
        self._set_theta_binning()
        self._set_levin_bessel_precision()

        # other miscellaneous settings
        self.proj_g_int_method = self.cfg['precision']['proj_gauss_integration_method']
        self.proj_ng_int_method = self.cfg['precision'][
            'proj_nongauss_integration_method'
        ]
        self.levin_bin_avg = self.cfg['precision']['levin_bin_avg']

        assert self.proj_g_int_method in ['simps', 'levin', 'FFTLog'], (
            "integration method not implemented; choose 'simps', 'levin', or "
            "'FFTLog'"
        )
        assert self.proj_ng_int_method in ['simps', 'levin', 'quad', 'FFTLog'], (
            "integration method not implemented; choose 'simps', 'levin', 'quad', "
            "or 'FFTLog'"
        )

        # attributes set at runtime
        self.cl_3x2pt_5d = _UNSET
        self.ells_proj_g = _UNSET
        self.nbl_proj_g = _UNSET
        self.ells_proj_ng = _UNSET
        self.nbl_proj_ng = _UNSET

    def _set_theta_binning(self):
        self.theta_min_arcmin = self.cfg['binning']['theta_min_arcmin']
        self.theta_max_arcmin = self.cfg['binning']['theta_max_arcmin']
        self.nbt_coarse = self.cfg['binning']['theta_bins']
        self.nbt_fine = self.nbt_coarse

        # TODO this should probably go in the ell_binning class (which should be
        # TODO renamed)
        if self.cfg['binning']['binning_type'] == 'log':
            _binning_func = np.geomspace
        elif self.cfg['binning']['binning_type'] == 'lin':
            _binning_func = np.linspace
        else:
            raise ValueError(
                f'Binning type: {self.cfg["binning"]["binning_type"]} '
                'not supported for real-space covariance'
            )

        # Use a loop to set up fine and coarse theta binning
        for bin_type in ['fine', 'coarse']:
            nbt = getattr(self, f'nbt_{bin_type}')
            theta_edges_deg = _binning_func(
                self.theta_min_arcmin / 60, self.theta_max_arcmin / 60, nbt + 1
            )
            theta_edges = np.deg2rad(theta_edges_deg)  # in radians

            if self.cfg['binning']['binning_type'] == 'log':
                theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])
            elif self.cfg['binning']['binning_type'] == 'lin':
                theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0

            # ! the theta values used throughout the code are in radians!
            setattr(self, f'theta_edges_{bin_type}', theta_edges)
            setattr(self, f'theta_centers_{bin_type}', theta_centers)

            assert len(theta_centers) == nbt, 'theta_centers length mismatch'

    def _set_levin_bessel_precision(self):
        self.levin_prec_kw = {
            # hardcoded
            'verbose': self.cfg['precision']['verbose'],
            'logx': True,
            'logy': True,
            'diagonal': False,
            # from the cfg file
            'n_sub': self.cfg['precision']['n_sub'],
            'n_bisec_max': self.cfg['precision']['n_bisec_max'],
            'rel_acc': self.cfg['precision']['rel_acc'],
            'boost_bessel': self.cfg['precision']['boost_bessel'],
        }

    def cov_sn_rs(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu):
        npair_arr = np.zeros((self.nbt_fine, self.zbins, self.zbins))
        for theta_ix in range(self.nbt_fine):
            theta_l = self.theta_edges_fine[theta_ix]
            theta_u = self.theta_edges_fine[theta_ix + 1]
            for zi in range(self.zbins):
                for zj in range(self.zbins):
                    npair_arr[theta_ix, zi, zj] = cp.get_npair(
                        theta_u,
                        theta_l,
                        self.survey_area_sr,
                        self.n_eff_2d[probe_a_ix, zi],
                        self.n_eff_2d[probe_b_ix, zj],
                    )

        delta_mu_nu = 1.0 if (mu == nu) else 0.0
        delta_theta = np.eye(self.nbt_fine)

        t_arr = t_sn(
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, self.zbins, self.sigma_eps_i
        )

        term = (
            cp.get_delta_tomo(probe_a_ix, probe_c_ix, self.zbins)[
                None, None, :, None, :, None
            ]
            * cp.get_delta_tomo(probe_b_ix, probe_d_ix, self.zbins)[
                None, None, None, :, None, :
            ]
            + cp.get_delta_tomo(probe_a_ix, probe_d_ix, self.zbins)[
                None, None, :, None, None, :
            ]
            * cp.get_delta_tomo(probe_b_ix, probe_c_ix, self.zbins)[
                None, None, None, :, :, None
            ]
        )

        cov_sn_rs_6d = (
            delta_mu_nu
            * delta_theta[:, :, None, None, None, None]
            * term
            * t_arr[None, None, :, :, None, None]
            / npair_arr[:, None, :, :, None, None]
        )
        return cov_sn_rs_6d

    def _cov_sn_rs(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu):
        npair_arr = np.zeros((self.nbt_fine, self.zbins, self.zbins))
        for theta_ix in range(self.nbt_fine):
            for zi in range(self.zbins):
                for zj in range(self.zbins):
                    theta_1_l = self.theta_edges_fine[theta_ix]
                    theta_1_u = self.theta_edges_fine[theta_ix + 1]
                    npair_arr[theta_ix, zi, zj] = cp.get_npair(
                        theta_1_u,
                        theta_1_l,
                        self.survey_area_sr,
                        self.n_eff_2d[probe_a_ix, zi],
                        self.n_eff_2d[probe_b_ix, zj],
                    )

        delta_mu_nu = 1.0 if (mu == nu) else 0.0
        delta_theta = np.eye(self.nbt_fine)
        t_arr = t_sn(
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, self.zbins, self.sigma_eps_i
        )

        cov_sn_rs_6d = (
            delta_mu_nu
            * delta_theta[:, :, None, None, None, None]
            * (
                cp.get_delta_tomo(probe_a_ix, probe_c_ix, self.zbins)[
                    None, None, :, None, :, None
                ]
                * cp.get_delta_tomo(probe_b_ix, probe_d_ix, self.zbins)[
                    None, None, None, :, None, :
                ]
                + cp.get_delta_tomo(probe_a_ix, probe_d_ix, self.zbins)[
                    None, None, :, None, None, :
                ]
                * cp.get_delta_tomo(probe_b_ix, probe_c_ix, self.zbins)[
                    None, None, None, :, :, None
                ]
            )
            # * t_arr[None, None, :, None, :, None]
            * t_arr[None, None, :, :, None, None]
            / npair_arr[None, :, :, :, None, None]
        )

        return cov_sn_rs_6d

    def proj_sva_levin_fftlog(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):  # fmt: skip
        # Use parent method to build the universal SVA integrand
        integrand_5d = cp.build_cov_sva_integrand_5d(
            cl_5d=self.cl_3x2pt_5d,
            probe_a_ix=probe_a_ix,
            probe_b_ix=probe_b_ix,
            probe_c_ix=probe_c_ix,
            probe_d_ix=probe_d_ix,
        )

        # Child-specific: project with Levin + Bessel kernels

        if self.proj_g_int_method == 'levin':
            cov_sva_rs_6d = self.proj_levin_wrapper(
                integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
            )
        elif self.proj_g_int_method == 'FFTLog':
            cov_sva_rs_6d = self.proj_sva_mix_fftlog_wrapper(integrand_5d, mu, nu)

        return cov_sva_rs_6d

    def proj_sva_mix_fftlog_wrapper(self, integrand_5d, mu, nu):
        # expand the first 2 axis and create an ell1, ell2 diagonal matrix
        # (as needed by the twobessel module)
        # Also, in the G case, the delta function collapses two integrals into one,
        # but TwoBessel still sees a 2D array and applies both ell-measures — leaving
        # one ell too many that I must cancel manually by dividing by ell when
        # building the diagonal.
        nbl = integrand_5d.shape[0]
        integrand_6d = np.zeros((nbl, nbl) + integrand_5d.shape[1:])
        for i in range(nbl):
            integrand_6d[i, i, ...] = integrand_5d[i, ...] / self.ells_proj_g[i]

        # prefactors
        integrand_6d /= 2.0 * np.pi * self.amax

        # integrate
        integral_6d = proj_cov_2d_fftlog(
            cov_hs_ell1ell2_in=integrand_6d,
            ells_proj=self.ells_proj_g,
            theta_edges=self.theta_edges_fine,
            theta_centers=self.theta_centers_fine,
            mu=mu,
            nu=nu,
            # c_window_width=.5,
            # N_pad=200,
            # N_extrap_low=200,
            # N_extrap_high=200
        )

        return integral_6d

    def proj_mix_levin_fftlog(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):  # fmt: skip
        def _get_mix_prefac(probe_b_ix, probe_d_ix, zj, zl):
            prefac = (
                cp.get_delta_tomo(probe_b_ix, probe_d_ix, self.zbins)[zj, zl]
                * t_mix(probe_b_ix, self.zbins, self.sigma_eps_i)[zj]
                / (self.n_eff_2d[probe_b_ix, zj] * self.srtoarcmin2)
            )
            return prefac

        prefac = np.zeros((self.n_probes_hs, self.n_probes_hs, self.zbins, self.zbins))
        for _probe_a_ix in range(self.n_probes_hs):
            for _probe_b_ix in range(self.n_probes_hs):
                for _zi in range(self.zbins):
                    for _zj in range(self.zbins):
                        prefac[_probe_a_ix, _probe_b_ix, _zi, _zj] = _get_mix_prefac(
                            _probe_a_ix, _probe_b_ix, _zi, _zj
                        )

        a = np.einsum(
            'jl,Lik->Lijkl',
            prefac[probe_b_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_a_ix, probe_c_ix],
        )
        b = np.einsum(
            'ik,Ljl->Lijkl',
            prefac[probe_a_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_d_ix],
        )
        c = np.einsum(
            'jk,Lil->Lijkl',
            prefac[probe_b_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_a_ix, probe_d_ix],
        )
        d = np.einsum(
            'il,Ljk->Lijkl',
            prefac[probe_a_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_c_ix],
        )
        integrand_5d = a + b + c + d

        # compress integrand selecting only unique zpairs
        assert ind_ab.shape[1] == 2, (
            "ind_ab must have two columns, maybe you didn't cut it"
        )
        assert ind_cd.shape[1] == 2, (
            "ind_cd must have two columns, maybe you didn't cut it"
        )

        if self.proj_g_int_method == 'levin':
            cov_mix_rs_6d = self.proj_levin_wrapper(
                integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
            )
        elif self.proj_g_int_method == 'FFTLog':
            cov_mix_rs_6d = self.proj_sva_mix_fftlog_wrapper(integrand_5d, mu, nu)

        return cov_mix_rs_6d

    def cov_mix_fftlog(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):  # fmt: skip
        def _get_mix_prefac(probe_b_ix, probe_d_ix, zj, zl):
            prefac = (
                cp.get_delta_tomo(probe_b_ix, probe_d_ix, self.zbins)[zj, zl]
                * t_mix(probe_b_ix, self.zbins, self.sigma_eps_i)[zj]
                / (self.n_eff_2d[probe_b_ix, zj] * self.srtoarcmin2)
            )
            return prefac

        prefac = np.zeros((self.n_probes_hs, self.n_probes_hs, self.zbins, self.zbins))
        for _probe_a_ix in range(self.n_probes_hs):
            for _probe_b_ix in range(self.n_probes_hs):
                for _zi in range(self.zbins):
                    for _zj in range(self.zbins):
                        prefac[_probe_a_ix, _probe_b_ix, _zi, _zj] = _get_mix_prefac(
                            _probe_a_ix, _probe_b_ix, _zi, _zj
                        )

        a = np.einsum(
            'jl,Lik->Lijkl',
            prefac[probe_b_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_a_ix, probe_c_ix],
        )
        b = np.einsum(
            'ik,Ljl->Lijkl',
            prefac[probe_a_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_d_ix],
        )
        c = np.einsum(
            'jk,Lil->Lijkl',
            prefac[probe_b_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_a_ix, probe_d_ix],
        )
        d = np.einsum(
            'il,Ljk->Lijkl',
            prefac[probe_a_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_c_ix],
        )
        integrand_5d = a + b + c + d

        # compress integrand selecting only unique zpairs
        assert ind_ab.shape[1] == 2, (
            "ind_ab must have two columns, maybe you didn't cut it"
        )
        assert ind_cd.shape[1] == 2, (
            "ind_cd must have two columns, maybe you didn't cut it"
        )

        cov_mix_rs_6d = self.proj_levin_wrapper(
            integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
        )

        return cov_mix_rs_6d

    def proj_levin_wrapper(
        self, integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):
        """This function abstracts the reshaping of the integral before and after the
        integration, as well as encapsulating the two different functions to call
        depending on the levin_bin_avg value"""
        integrand_3d = sl.cov_6D_to_4D_blocks(
            cov_6D=integrand_5d,
            nbl=self.nbl_proj_g,
            npairs_AB=zpairs_ab,
            npairs_CD=zpairs_cd,
            ind_AB=ind_ab,
            ind_CD=ind_cd,
        )
        assert integrand_3d.shape[1:] == (zpairs_ab, zpairs_cd), 'shape mismatch'

        integrand_2d = integrand_3d.reshape(self.nbl_proj_g, -1)
        integrand_2d *= self.ells_proj_g[:, None]
        integrand_2d /= 2.0 * np.pi * self.amax

        if self.levin_bin_avg:
            cov_rs_4d = self.levin_binavg_helper(
                integrand_2d, mu, nu, zpairs_ab, zpairs_cd
            )
        else:
            result_levin = levin_integrate_bessel_double_wrapper(
                integrand_2d,
                x_values=self.ells_proj_g,
                bessel_args=self.theta_centers_fine,
                bessel_type=3,
                ell_1=mu,
                ell_2=nu,
                n_jobs=self.n_jobs,
                **self.levin_prec_kw,
            )

            cov_rs_4d = result_levin.reshape(
                self.nbt_fine, self.nbt_fine, zpairs_ab, zpairs_cd
            )

        cov_rs_6d = sl.cov_4D_to_6D_blocks(
            cov_rs_4d,
            nbl=self.nbt_fine,
            zbins=self.zbins,
            ind_ab=ind_ab,
            ind_cd=ind_cd,
            symmetrize_output_ab=False,
            symmetrize_output_cd=False,
        )

        return cov_rs_6d

    def levin_binavg_helper(self, integrand_2d, mu, nu, zpairs_ab, zpairs_cd):
        """Takes care of looping over and assembling the different terms needed for
        the bin-averaged Levin integral. This is used both in the SVA and MIX terms
        """
        result_shape = (zpairs_ab, zpairs_cd)
        cov_rs_4d = np.zeros((self.nbt_fine, self.nbt_fine, *result_shape))

        for p in tqdm(range(self.nbt_fine), desc='theta'):
            for q in range(self.nbt_fine):
                theta_p_lower = self.theta_edges_fine[p]
                theta_p_upper = self.theta_edges_fine[p + 1]
                theta_q_lower = self.theta_edges_fine[q]
                theta_q_upper = self.theta_edges_fine[q + 1]

                k_mu_terms = k_mu_nobessel(
                    self.ells_proj_g, thetal=theta_p_lower, thetau=theta_p_upper, mu=mu
                )
                k_nu_terms = k_mu_nobessel(
                    self.ells_proj_g, thetal=theta_q_lower, thetau=theta_q_upper, mu=nu
                )
                product_expansion = kmuknu_nobessel(k_mu_terms, k_nu_terms)

                cov_pq_element = np.zeros(result_shape)

                # Loop over each term in the kernel expansion
                for term in product_expansion:
                    const_coeff, n1, theta1, n2, theta2 = term

                    # Apply the constant coefficient from the kernel expansion
                    term_integrand_for_bessel = integrand_2d * const_coeff[:, None]

                    # Integrate this term using the new single bessel pair function
                    result_levin_1d = integrate_single_bessel_pair(
                        term_integrand_for_bessel,
                        x_values=self.ells_proj_g,
                        ord_bes_1=n1,
                        theta1=theta1,
                        ord_bes_2=n2,
                        theta2=theta2,
                        bessel_type=3,
                        n_jobs=self.n_jobs,
                        **self.levin_prec_kw,
                    )

                    cov_pq_element += result_levin_1d.reshape(result_shape)

                cov_rs_4d[p, q] = cov_pq_element

        return cov_rs_4d

    def compute_rs_cov_term_probe_6d(
        self, cov_hs_ng_dict: dict | None, probe_abcd: str, term: str
    ) -> None:
        """
        Computes the real space covariance matrix for the specified term
        and probe combination, in 6d
        """

        if term not in const.ALL_COV_TERMS:
            raise ValueError(f'Covariance term {term} not recognized!')

        probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'real')
        probe_2tpl = (probe_ab, probe_cd)

        mu, nu = const.MU_DICT[probe_ab], const.MU_DICT[probe_cd]
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = const.RS_PROBE_NAME_TO_IX_DICT[
            probe_abcd
        ]

        ind_ab = (
            self.ind_auto[:, 2:] if probe_a_ix == probe_b_ix
            else self.ind_cross[:, 2:]
        )  # fmt: skip
        ind_cd = (
            self.ind_auto[:, 2:] if probe_c_ix == probe_d_ix
            else self.ind_cross[:, 2:]
        )  # fmt: skip

        zpairs_ab = self.zpairs_auto if probe_a_ix == probe_b_ix else self.zpairs_cross
        zpairs_cd = self.zpairs_auto if probe_c_ix == probe_d_ix else self.zpairs_cross

        # just a sanity check
        assert zpairs_ab == ind_ab.shape[0], 'zpairs-ind inconsistency'
        assert zpairs_cd == ind_cd.shape[0], 'zpairs-ind inconsistency'

        # arguments for the covariance projector functions
        cov_simps_func_kw = {
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
        }

        # arguments for the covariance projector kernel functions
        kernel_builder_func_kw = {
            'mu': mu,
            'nu': nu,
            'kernel_1_func': k_mu,
            'kernel_2_func': k_mu,
        }

        # Compute covariance:
        if term == 'sva':
            if self.proj_g_int_method == 'simps':
                cov_out_6d = self.proj_cov_simps_parallel_helper_wrapper(
                    zpairs_ab=zpairs_ab,
                    zpairs_cd=zpairs_cd,
                    ind_ab=ind_ab,
                    ind_cd=ind_cd,
                    cov_simps_func=self.proj_cov_sva_simps,
                    cov_simps_func_kw=cov_simps_func_kw,
                    kernel_builder_func_kw=kernel_builder_func_kw,
                )
            elif self.proj_g_int_method in ['levin', 'FFTLog']:
                cov_out_6d = self.proj_sva_levin_fftlog(
                    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                    zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
                )  # fmt: skip

        elif term == 'mix' and probe_abcd not in ['ggxim', 'ggxip']:
            if self.proj_g_int_method == 'simps':
                cov_out_6d = self.proj_cov_simps_parallel_helper_wrapper(
                    zpairs_ab=zpairs_ab,
                    zpairs_cd=zpairs_cd,
                    ind_ab=ind_ab,
                    ind_cd=ind_cd,
                    cov_simps_func=self.proj_cov_mix_simps,
                    cov_simps_func_kw=cov_simps_func_kw,
                    kernel_builder_func_kw=kernel_builder_func_kw,
                )
            elif self.proj_g_int_method in ['levin', 'FFTLog']:
                cov_out_6d = self.proj_mix_levin_fftlog(
                    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                    zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
                )  # fmt: skip

        elif term == 'mix' and probe_abcd in ['ggxim', 'ggxip']:
            cov_out_6d = np.zeros(self.cov_shape_6d)

        elif term == 'sn':
            # this is 0 for
            # ['xipxim', 'gtxim', 'gtxip', 'ggxim', 'gggt', 'ggxip']
            # but is very fast to compute so I don't skip these terms
            cov_out_6d = self.cov_sn_rs(
                probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu
            )

        elif term in ['ssc', 'cng']:
            if cov_hs_ng_dict is None:
                raise ValueError(
                    f'Non-Gaussian covariance term {term} requested, '
                    'but no harmonic-space non-Gaussian covariance dictionary provided.'
                )

            # recover corresponding harmonic-space probe names
            probe_abcd_hs = (
                const.HS_PROBE_IX_TO_NAME_DICT[probe_a_ix]
                + const.HS_PROBE_IX_TO_NAME_DICT[probe_b_ix]
                + const.HS_PROBE_IX_TO_NAME_DICT[probe_c_ix]
                + const.HS_PROBE_IX_TO_NAME_DICT[probe_d_ix]
            )
            probe_ab_hs, probe_cd_hs = sl.split_probe_name(probe_abcd_hs, 'harmonic')

            # project hs non-gaussian cov to real space
            cov_hs_ng_4d = cov_hs_ng_dict[term][probe_ab_hs, probe_cd_hs]['4d']

            if self.proj_ng_int_method in ['simps', 'quad']:
                cov_rs_ng_4d = np.zeros((self.nbx, self.nbx, zpairs_ab, zpairs_cd))

                # to parallelize over the scale (theta, in this case) indices s1 and s2,
                # rely on proj_cov_2d_parallel_helper
                # TODO I could only loop over the upper triangle of s1, s2 and then
                # symmetrize...
                results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(proj_cov_2d_parallel_helper)(
                        s1=s1,
                        s2=s2,
                        theta_edges_fine=self.theta_edges_fine,
                        mu=mu,
                        nu=nu,
                        integration_method=self.proj_ng_int_method,
                        ells_proj_ng=self.ells_proj_ng,
                        cov_hs_ng_4d=cov_hs_ng_4d,
                    )
                    for s1 in range(self.nbx)
                    for s2 in range(self.nbx)
                )

                for s1, s2, block in results:
                    cov_rs_ng_4d[s1, s2] = block

            elif self.proj_ng_int_method == 'levin':
                cov_rs_ng_4d = dl1dl2_binavg_bessel_wrapper(
                    cov_hs=cov_hs_ng_4d,
                    mu=mu,
                    nu=nu,
                    ells=self.ells_proj_ng,
                    theta_edges=self.theta_edges_fine,
                    n_jobs=self.n_jobs,
                    levin_prec_kw=self.levin_prec_kw,
                )
            elif self.proj_ng_int_method == 'FFTLog':
                if self.cfg['binning']['binning_type'] != 'log':
                    raise ValueError(
                        "integration_method='FFTLog' requires log-spaced theta bins "
                        "(binning_type: 'log')."
                    )
                cov_rs_ng_4d = proj_cov_2d_fftlog(
                    cov_hs_ell1ell2_in=cov_hs_ng_4d,
                    ells_proj=self.ells_proj_ng,
                    theta_edges=self.theta_edges_fine,
                    theta_centers=self.theta_centers_fine,
                    mu=mu,
                    nu=nu,
                )

            # reshape to 6d and symmetrize if needed
            cov_rs_ng_6d = sl.cov_4D_to_6D_blocks(
                cov_4D=cov_rs_ng_4d,
                nbl=self.nbx,
                zbins=self.zbins,
                ind_ab=ind_ab,
                ind_cd=ind_cd,
                symmetrize_output_ab=self.symmetrize_output_dict[probe_ab_hs],
                symmetrize_output_cd=self.symmetrize_output_dict[probe_cd_hs],
            )

            # normalize
            norm = 4 * np.pi**2
            cov_rs_ng_6d /= norm

            cov_out_6d = cov_rs_ng_6d

        # ! bin sb cov 2d
        if self.nbt_coarse != self.nbt_fine:
            print(
                f'Re-binning real space covariance from {self.nbt_fine} to '
                f'{self.nbt_coarse} theta bins'
            )

            cov_rs_6d_unbinned = getattr(self, f'cov_{term}_rs_6d')
            cov_rs_6d_binned = np.zeros(
                (self.nbt_coarse, self.nbt_coarse,
                 self.zbins, self.zbins, self.zbins, self.zbins)
            )  # fmt: skip

            # cast to list to avoid problems due do "recycling" the generator
            zijkl_comb = list(itertools.product(range(self.zbins), repeat=4))

            bin_2d_array_kw = {
                'ells_in': self.theta_centers_fine,
                'ells_out': self.theta_centers_coarse,
                'ells_out_edges': self.theta_edges_coarse,
                'weights_in': None,
                'which_binning': 'sum',
                'interpolate': True,
            }

            results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(sl.bin_2d_array)(
                    cov_rs_6d_unbinned[:, :, zi, zj, zk, zl], **bin_2d_array_kw
                )
                for zi, zj, zk, zl in zijkl_comb
            )
            for (zi, zj, zk, zl), cov in zip(zijkl_comb, results, strict=True):
                cov_rs_6d_binned[:, :, zi, zj, zk, zl] = cov

            cov_out_6d = cov_rs_6d_binned

        # finally, assign the newly computed 6D cov to the appropriate key in cov_dict
        self.cov_dict[term][probe_2tpl]['6d'] = cov_out_6d

    def k_mu(self, ell, *, thetal, thetau, mu):
        """Thin wrapper around k_mu, just to make it a class method"""
        return k_mu(ell, thetal=thetal, thetau=thetau, mu=mu)
