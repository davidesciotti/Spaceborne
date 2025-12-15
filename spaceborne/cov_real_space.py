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

# import pylevin as levin
from joblib import Parallel, delayed
from scipy.integrate import simpson as simps
from tqdm import tqdm

from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import sb_lib as sl

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


def k_mu(ell, thetal, thetau, mu):
    r"""Computes the kernel K_mu(ell * theta_i) in Eq. (E.2):

    K_mu(l * theta_i) = 2 / [ (theta_u^2 - theta_l^2) * l^2 ]
                        * [ b_mu(l * theta_u) - b_mu(l * theta_l) ].
    """
    prefactor = 2.0 / ((thetau**2 - thetal**2) * (ell**2))
    return prefactor * (b_mu(ell * thetau, mu) - b_mu(ell * thetal, mu))


def k_mu_nobessel(ell, thetal, thetau, mu):
    """
    Generates a list of decomposed terms for the kernel K_μ.

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
    Computes the product of two expanded kernels K_μ and K_ν.

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


# ! __ = 'no longer used'


# ! ====================== COV RS W/ SIMPSON INTEGRATION ===============================
def cov_sva_simps(
    theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu,
    zi, zj, zk, zl, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
    cl_5d, Amax, ell_values
):  # fmt: skip
    """Computes a single entry of the real-space Gaussian SVA (sample variance)
    part of the covariance matrix.
    """
    c_ik = cl_5d[probe_a_ix, probe_c_ix, :, zi, zk]
    c_jl = cl_5d[probe_b_ix, probe_d_ix, :, zj, zl]
    c_il = cl_5d[probe_a_ix, probe_d_ix, :, zi, zl]
    c_jk = cl_5d[probe_b_ix, probe_c_ix, :, zj, zk]

    def integrand_func(ell):
        kmu = k_mu(ell, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell, theta_2_l, theta_2_u, nu)
        return ell * kmu * knu * (c_ik * c_jl + c_il * c_jk)

    integrand = integrand_func(ell_values)
    integral = simps(y=integrand, x=ell_values)

    # integrate with quad and compare
    # integral = quad_vec(integrand_func, ell_values[0], ell_values[-1])[0]

    # Finally multiply the prefactor
    cov_elem = integral / (2.0 * np.pi * Amax)
    return cov_elem


def cov_mix_simps(
    self, theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu,
    ell_values, cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
    zi, zj, zk, zl, Amax
):  # fmt: skip
    """This function accepts self as an argument, but it's not a class method"""

    def integrand_func(ell, inner_integrand):
        kmu = k_mu(ell, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell, theta_2_l, theta_2_u, nu)
        return (1 / (2 * np.pi * Amax)) * ell * kmu * knu * inner_integrand

    def get_prefac(probe_a_ix, probe_b_ix, zi, zj):
        prefac = (
            self.get_delta_tomo(probe_a_ix, probe_b_ix)[zi, zj]
            * t_mix(probe_a_ix, self.zbins, self.sigma_eps_i)[zi]
            / (self.n_eff_2d[probe_a_ix, zi] * self.srtoarcmin2)
        )
        return prefac

    # TODO generalize to different survey areas (max(Aij, Akl))
    # TODO sigma_eps_i should be a vector of length zbins

    # permutations should be performed as done in the SVA function
    integrand = integrand_func(
        ell_values,
        cl_5d[probe_a_ix, probe_c_ix, :, zi, zk]
        * get_prefac(probe_b_ix, probe_d_ix, zj, zl)
        + cl_5d[probe_b_ix, probe_d_ix, :, zj, zl]
        * get_prefac(probe_a_ix, probe_c_ix, zi, zk)
        + cl_5d[probe_a_ix, probe_d_ix, :, zi, zl]
        * get_prefac(probe_b_ix, probe_c_ix, zj, zk)
        + cl_5d[probe_b_ix, probe_c_ix, :, zj, zk]
        * get_prefac(probe_a_ix, probe_d_ix, zi, zl),
    )

    integral = simps(y=integrand, x=ell_values)

    # elif integration_method == 'quad':

    #     integral_1 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
    #                           args=(cl_5d[probe_a_ix, probe_c_ix, :, zi, zk],))[0]
    #     integral_2 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
    #                           args=(cl_5d[probe_b_ix, probe_d_ix, :, zj, zl],))[0]
    #     integral_3 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
    #                           args=(cl_5d[probe_a_ix, probe_d_ix, :, zi, zl],))[0]
    #     integral_4 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
    #                           args=(cl_5d[probe_b_ix, probe_c_ix, :, zj, zk],))[0]

    # else:
    # raise ValueError(f'integration_method {integration_method} '
    # 'not recognized.')

    return integral


def _get_t_munu(mu, nu, sigma_eps_tot):
    if mu == nu == 0 or mu == nu == 4:
        return sigma_eps_tot**4
    elif mu == nu == 2:
        return sigma_eps_tot**2 / 2
    elif mu == nu == 0:
        return 1
    elif mu != nu:
        return 0
    else:
        raise ValueError('mu and nu must be either 0, 2, or 4.')


def t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i):
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


def get_npair(theta_1_u, theta_1_l, survey_area_sr, n_eff_i, n_eff_j):
    n_eff_i *= const.SR_TO_ARCMIN2
    n_eff_j *= const.SR_TO_ARCMIN2
    return np.pi * (theta_1_u**2 - theta_1_l**2) * survey_area_sr * n_eff_i * n_eff_j


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


def dl1dl2_bessel_wrapper(
    cov_hs: np.ndarray,
    mu: int,
    nu: int,
    ells: np.ndarray,
    thetas: np.ndarray,
    zbins: int,
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

    # First integration: for each fixed ell1, integrate over ell2.
    partial_results = []
    for ell1_ix in tqdm(range(nbl)):
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

    for theta_idx in tqdm(range(nbt)):
        # For fixed theta from the first integration, extract the integrand:
        integrand_second = partial_results[:, theta_idx, :] * ells[:, None]
        final_int = integrate_bessel_single_wrapper(
            integrand_second, mu, ells, thetas, n_jobs, **levin_prec_kw
        )
        final_result[:, theta_idx, :] = final_int

    cov_rs_6d = final_result.reshape(nbt, nbt, zbins, zbins, zbins, zbins)

    return cov_rs_6d


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
    # tODO this might change in the future?
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


# ! ====================================================================================
# ! ====================================================================================
# ! ====================================================================================


class CovRealSpace:
    def __init__(self, cfg, pvt_cfg, mask_obj):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg
        self.mask_obj = mask_obj
        self.zbins = self.pvt_cfg['zbins']

        # ordering-related stuff
        self.cov_ordering_2d = self.cfg['covariance']['covariance_ordering_2D']
        self.ind_dict = pvt_cfg['ind_dict']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']

        # instantiate cov dict with the required terms and probe combinations
        self.req_terms = pvt_cfg['req_terms']
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_rs_2d']
        dims = ['6d', '4d', '2d']

        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='real')
            for probe in self.req_probe_combs_2d
        ]
        _req_probe_combs_2d.append('3x2pt')
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

        # setters
        self._set_survey_info()
        self._set_theta_binning()
        self._set_neff_and_sigma_eps()
        self._set_levin_bessel_precision()
        self._set_probe_names_idxs()
        self._set_terms_toloop()

        # other miscellaneous settings
        self.n_jobs = self.cfg['misc']['num_threads']
        self.integration_method = self.cfg['precision']['cov_rs_int_method']
        self.levin_bin_avg = self.cfg['precision']['levin_bin_avg']

        assert self.integration_method in ['simps', 'levin'], (
            'integration method not implemented'
        )

        self.cov_rs_6d_shape = (
            self.nbt_fine, self.nbt_fine, self.zbins, self.zbins, self.zbins, self.zbins
            )  # fmt: skip

        # attributes set at runtime
        self.cl_3x2pt_5d = _UNSET
        self.ells = _UNSET
        self.nbl = _UNSET

    def set_cov_2d_ordering(self):
        # settings for 2D covariance ordering
        if self.cov_ordering_2d == 'probe_scale_zpair':
            self.block_index = 'ell'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt_rs
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'zbins': self.zbins,
                'req_probe_combs_2d': self.req_probe_combs_2d,
            }
        elif self.cov_ordering_2d == 'probe_zpair_scale':
            self.block_index = 'zpair'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt_rs
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'zbins': self.zbins,
                'req_probe_combs_2d': self.req_probe_combs_2d,
            }
        elif self.cov_ordering_2d == 'scale_probe_zpair':
            self.block_index = 'ell'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2D
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'optimize': True,
            }
        elif self.cov_ordering_2d == 'zpair_probe_scale':
            self.block_index = 'zpair'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2D
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'optimize': True,
            }
        else:
            raise ValueError(f'Unknown 2D cov ordering: {self.cov_ordering_2d}')

    def _set_survey_info(self):
        self.survey_area_deg2 = self.mask_obj.survey_area_deg2
        self.survey_area_sr = self.mask_obj.survey_area_sr
        self.fsky = self.mask_obj.fsky
        self.srtoarcmin2 = const.SR_TO_ARCMIN2
        # maximum survey area in sr
        # TODO generalise to multiple survey areas
        self.amax = max((self.survey_area_sr, self.survey_area_sr))

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
            theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
            setattr(self, f'theta_edges_{bin_type}', theta_edges)
            setattr(self, f'theta_centers_{bin_type}', theta_centers)
            assert len(theta_centers) == nbt, 'theta_centers length mismatch'

    def _set_neff_and_sigma_eps(self):
        self.n_eff_lens = self.cfg['nz']['ngal_lenses']
        self.n_eff_src = self.cfg['nz']['ngal_sources']
        # in this way the indices correspond to xip, xim, g
        self.n_eff_2d = np.row_stack((self.n_eff_lens, self.n_eff_lens, self.n_eff_src))

        self.sigma_eps_i = np.array(self.cfg['covariance']['sigma_eps_i'])
        self.sigma_eps_tot = self.sigma_eps_i * np.sqrt(2)

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

    def _set_probe_names_idxs(self):
        self.munu_vals = (0, 2, 4)
        self.n_probes_rs = 4  # real space
        self.n_probes_hs = 2  # harmonic space
        self.n_split_terms = 3

        # this is only needed to be able to construct the full Gauss cov from the sum
        # of the
        # SVA, SN and MIX covs. No particular reason behind the choice of the indices.
        self.split_g_dict = {'sva': 0, 'sn': 1, 'mix': 2}

        # for validation purposes
        self.probe_idx_dict_short_oc = {}
        for key in const.RS_PROBE_NAME_TO_IX_DICT:
            probe_ab_str, probe_cd_str = sl.split_probe_name(key, 'real')
            probe_ab_str_oc = 'gm' if probe_ab_str == 'gt' else probe_ab_str
            probe_cd_str_oc = 'gm' if probe_cd_str == 'gt' else probe_cd_str
            self.probe_idx_dict_short_oc[probe_ab_str_oc + probe_cd_str_oc] = (
                const.RS_PROBE_NAME_TO_IX_DICT_SHORT[probe_ab_str],
                const.RS_PROBE_NAME_TO_IX_DICT_SHORT[probe_cd_str],
            )

    def _set_terms_toloop(self):
        self.terms_toloop = []
        if self.cfg['covariance']['G']:
            self.terms_toloop.extend(('sva', 'sn', 'mix'))
        if self.cfg['covariance']['SSC']:
            self.terms_toloop.append('ssc')
        if self.cfg['covariance']['cNG']:
            self.terms_toloop.append('cng')

    def cov_sn_rs(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu):
        # TODO generalize to different n(z)
        npair_arr = np.zeros((self.nbt_fine, self.zbins, self.zbins))
        for theta_ix in range(self.nbt_fine):
            for zi in range(self.zbins):
                for zj in range(self.zbins):
                    theta_1_l = self.theta_edges_fine[theta_ix]
                    theta_1_u = self.theta_edges_fine[theta_ix + 1]
                    npair_arr[theta_ix, zi, zj] = get_npair(
                        theta_1_u,
                        theta_1_l,
                        self.survey_area_sr,
                        self.n_eff_lens[zi],
                        self.n_eff_lens[zj],
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
                self.get_delta_tomo(probe_a_ix, probe_c_ix)[
                    None, None, :, None, :, None
                ]
                * self.get_delta_tomo(probe_b_ix, probe_d_ix)[
                    None, None, None, :, None, :
                ]
                + self.get_delta_tomo(probe_a_ix, probe_d_ix)[
                    None, None, :, None, None, :
                ]
                * self.get_delta_tomo(probe_b_ix, probe_c_ix)[
                    None, None, None, :, :, None
                ]
            )
            * t_arr[None, None, :, None, :, None]
            / npair_arr[None, :, :, :, None, None]
        )

        return cov_sn_rs_6d

    def get_delta_tomo(self, probe_a_ix, probe_b_ix):
        if probe_a_ix == probe_b_ix:
            return np.eye(self.zbins)
        else:
            return np.zeros((self.zbins, self.zbins))

    def cov_simps_wrapper(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu, func
    ):  # fmt: skip
        """Helper to parallelize the cov_sva_simps and cov_mix_simps functions"""
        cov_rs_6d = np.zeros(self.cov_rs_6d_shape)

        kwargs = {
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
            'cl_5d': self.cl_3x2pt_5d,
            'ell_values': self.ells,
            'Amax': self.amax,
        }
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.cov_parallel_helper)(
                theta_1_ix=theta_1_ix, theta_2_ix=theta_2_ix, mu=mu, nu=nu,
                zij=zij, zkl=zkl, ind_ab=ind_ab, ind_cd=ind_cd,
                func=func,
                **kwargs,
            )
            for theta_1_ix in tqdm(range(self.nbt_fine))
            for theta_2_ix in range(self.nbt_fine)
            for zij in range(zpairs_ab)
            for zkl in range(zpairs_cd)
        )  # fmt: skip

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_rs_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value

        return cov_rs_6d

    def cov_sva_levin(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):  # fmt: skip
        a = np.einsum(
            'Lik,Ljl->Lijkl',
            self.cl_3x2pt_5d[probe_a_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_d_ix],
        )
        b = np.einsum(
            'Lil,Ljk->Lijkl',
            self.cl_3x2pt_5d[probe_a_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_c_ix],
        )
        integrand_5d = a + b

        cov_sva_rs_6d = self.cov_levin_wrapper(
            integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
        )

        return cov_sva_rs_6d

    def cov_mix_levin(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):  # fmt: skip
        def _get_mix_prefac(probe_b_ix, probe_d_ix, zj, zl):
            prefac = (
                self.get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zl]
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

        cov_mix_rs_6d = self.cov_levin_wrapper(
            integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
        )

        return cov_mix_rs_6d

    def cov_levin_wrapper(
        self, integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):
        """This function abstracts the reshaping of the integral before and after the
        integration, as well as encapsulating the two different functions to call
        depending on the levin_bin_avg value"""
        integrand_3d = sl.cov_6D_to_4D_blocks(
            cov_6D=integrand_5d,
            nbl=self.nbl,
            npairs_AB=zpairs_ab,
            npairs_CD=zpairs_cd,
            ind_AB=ind_ab,
            ind_CD=ind_cd,
        )
        assert integrand_3d.shape[1:] == (zpairs_ab, zpairs_cd), 'shape mismatch'

        integrand_2d = integrand_3d.reshape(self.nbl, -1)
        integrand_2d *= self.ells[:, None]
        integrand_2d /= 2.0 * np.pi * self.amax

        if self.levin_bin_avg:
            cov_rs_4d = self.levin_binavg_helper(
                integrand_2d, mu, nu, zpairs_ab, zpairs_cd
            )
        else:
            result_levin = levin_integrate_bessel_double_wrapper(
                integrand_2d,
                x_values=self.ells,
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

        for p in tqdm(range(self.nbt_fine)):
            for q in range(self.nbt_fine):
                theta_p_lower = self.theta_edges_fine[p]
                theta_p_upper = self.theta_edges_fine[p + 1]
                theta_q_lower = self.theta_edges_fine[q]
                theta_q_upper = self.theta_edges_fine[q + 1]

                k_mu_terms = k_mu_nobessel(self.ells, theta_p_lower, theta_p_upper, mu)
                k_nu_terms = k_mu_nobessel(self.ells, theta_q_lower, theta_q_upper, nu)
                product_expansion = kmuknu_nobessel(k_mu_terms, k_nu_terms)

                cov_pq_element = np.zeros(result_shape)

                # Loop over each term in the kernel expansion
                for term in product_expansion:
                    const_coeff, n1, theta1, n2, theta2 = term

                    # Apply the constant coefficient from the kernel expansion
                    # Apply the constant coefficient from the kernel expansion
                    term_integrand_for_bessel = integrand_2d * const_coeff[:, None]

                    # Integrate this term using the new single bessel pair function
                    result_levin_1d = integrate_single_bessel_pair(
                        term_integrand_for_bessel,
                        x_values=self.ells,
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

    def cov_parallel_helper(
        self, theta_1_ix, theta_2_ix, mu, nu, zij, zkl, ind_ab, ind_cd, func, **kwargs
    ):
        theta_1_l = self.theta_edges_fine[theta_1_ix]
        theta_1_u = self.theta_edges_fine[theta_1_ix + 1]
        theta_2_l = self.theta_edges_fine[theta_2_ix]
        theta_2_u = self.theta_edges_fine[theta_2_ix + 1]

        zi, zj = ind_ab[zij, :]
        zk, zl = ind_cd[zkl, :]

        return (
            theta_1_ix, theta_2_ix, zi, zj, zk, zl, func(
                theta_1_l=theta_1_l, theta_1_u=theta_1_u, mu=mu,
                theta_2_l=theta_2_l, theta_2_u=theta_2_u, nu=nu,
                zi=zi, zj=zj, zk=zk, zl=zl,
                **kwargs,
            ),
        )  # fmt: skip

    def _sum_split_g_terms_allprobeblocks_alldims(self) -> None:
        # small sanity check probe combinations must match for terms (sva, sn, mix)
        if not (
            self.cov_dict['sva'].keys()
            == self.cov_dict['sn'].keys()
            == self.cov_dict['mix'].keys()
        ):
            raise ValueError(
                'The probe combinations keys in the SVA, SN and MIX covariance '
                'dictionaries do not match!'
            )

        # sanity check: all the probes must match
        probes_sva = set(self.cov_dict['sva'].keys())
        probes_sn = set(self.cov_dict['sn'].keys())
        probes_mix = set(self.cov_dict['mix'].keys())
        if not (probes_sva == probes_sn == probes_mix):
            raise ValueError(
                'The probe combinations in the SVA, SN and MIX covariance '
                'dictionaries do not match!'
            )

        # now sum the terms to get the Gaussian, for all probe combinations and
        # dimensions
        for probe_2tpl in self.cov_dict['sva']:
            if probe_2tpl == '3x2pt':
                continue  # skip 3x2pt, built later

            # sanity check: all the dimensions must match
            dims_sva = set(self.cov_dict['sva'][probe_2tpl].keys())
            dims_sn = set(self.cov_dict['sn'][probe_2tpl].keys())
            dims_mix = set(self.cov_dict['mix'][probe_2tpl].keys())
            if not (dims_sva == dims_sn == dims_mix):
                raise ValueError(
                    'The probe combinations in the SVA, SN and MIX covariance '
                    'dictionaries do not match!'
                )

            # for each dim, perform the sum
            for dim in ['2d', '4d', '6d']:
                self.cov_dict['g'][probe_2tpl][dim] = (
                    self.cov_dict['sva'][probe_2tpl][dim]
                    + self.cov_dict['sn'][probe_2tpl][dim]
                    + self.cov_dict['mix'][probe_2tpl][dim]
                )

    def _build_cov_3x2pt_4d_and_2d(self) -> None:
        """For each covariance term, constructs the 4d and 2d 3x2pt covs from
        the 6d probe-specific ones.

        Note: remember that there is no 6d 3x2pt 6d or 10d cov!

        Note: This exact same function is also defined in cov_harmonic_space.py
        """

        # TODO deprecate this func

        for term in self.cov_dict:
            if term == 'tot':
                continue  # tot is built at the end, skip it

            self.cov_dict[term]['3x2pt']['4d'] = (
                sl.cov_dict_4d_probeblocks_to_3x2pt_4d_array(
                    self.cov_dict[term], obs_space='real'
                )
            )
            self.cov_dict[term]['3x2pt']['2d'] = self.cov_4D_to_2D_3x2pt_func(
                self.cov_dict[term]['3x2pt']['4d'], **self.cov_4D_to_2D_3x2pt_func_kw
            )

        # this function modifies the cov_dict in place, no need to reassign the result
        # to self.cov_dict
        sl.set_cov_tot_2d_and_6d(
            cov_dict=self.cov_dict,
            req_probe_combs_2d=self.req_probe_combs_2d,
            space='real',
        )

    def combine_terms_and_probes(self, unique_probe_combs):
        """For all the required terms, constructs the 3x2pt
        (or nx2pt, depending on the n required probes) 2D cov,
        taking into account the required probe combinations
        (this is taken care of by cov_4D_to_2DCLOE_3x2pt_rs).
        sack (join) probes into a single 2D cov (for each term) and store it in the
        object"""

        # ! construct 3x2pt 2D cov for each term and store them in the object
        for term in self.terms_toloop:
            # first construct the dict
            cov_term_3x2pt_4d_dict = self.build_cov_3x2pt_8d_dict(
                self.req_probe_combs_2d, term
            )
            # then turn to 4D array
            cov_term_3x2pt_4d_arr = sl.cov_3x2pt_8D_dict_to_4D(
                cov_term_3x2pt_4d_dict, self.req_probe_combs_2d, space='real'
            )
            # then to 2D array
            cov_term_3x2pt_2d_arr = self.cov_4D_to_2D_3x2pt_func(
                cov_term_3x2pt_4d_arr, **self.cov_4D_to_2D_3x2pt_func_kw
            )
            # set attribute
            setattr(self, f'cov_3x2pt_{term}_2d', cov_term_3x2pt_2d_arr)

        # ! sum terms to get G and TOT 2D 3x2pt covs and store them in the object
        self.cov_3x2pt_g_2d = sum(
            getattr(self, f'cov_3x2pt_{term}_2d') for term in ['sva', 'sn', 'mix']
        )
        self.cov_3x2pt_tot_2d = sum(
            getattr(self, f'cov_3x2pt_{term}_2d') for term in self.terms_toloop
        )

        for probe in unique_probe_combs:
            # ! sum to get G and TOT 2D probe-specific covs and store them in the object
            # ! (not needed in this new "approach" to the files I wish to save)
            # cov_probe_g_2d = sum(
            #     getattr(self, f'cov_{probe}_{term}_2d') for term in ['sva', 'sn', 'mix']
            # )
            # cov_probe_tot_2d = sum(
            #     getattr(self, f'cov_{probe}_{term}_2d') for term in self.terms_toloop
            # )
            # setattr(self, f'cov_{probe}_g_2d', cov_probe_g_2d)
            # setattr(self, f'cov_{probe}_tot_2d', cov_probe_tot_2d)

            # ! sum terms to get, G, TOT 6D probe-specific covs
            # ! and store them in the object (required if save_full_cov is True).
            # ! note that the 6D covs are already computed and stored in the object
            # ! in the compute_realspace_cov function
            cov_probe_g_6d = sum(
                getattr(self, f'cov_{probe}_{term}_6d') for term in ['sva', 'sn', 'mix']
            )
            cov_probe_tot_6d = sum(
                getattr(self, f'cov_{probe}_{term}_6d') for term in self.terms_toloop
            )
            setattr(self, f'cov_{probe}_g_6d', cov_probe_g_6d)
            setattr(self, f'cov_{probe}_tot_6d', cov_probe_tot_6d)

    def compute_rs_cov_term_probe_6d(self, cov_hs_obj, probe_abcd, term):
        """
        Computes the real space covariance matrix for the specified term
        and probe combination, in 6d
        """

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

        # Compute covariance:
        if term == 'sva':
            if self.integration_method in ['simps', 'quad']:
                cov_out_6d = self.cov_simps_wrapper(
                    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                    zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu,
                    func=cov_sva_simps
                )  # fmt: skip

            elif self.integration_method == 'levin':
                cov_out_6d = self.cov_sva_levin(
                    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                    zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
                )  # fmt: skip

        elif term == 'mix' and probe_abcd not in ['ggxim', 'ggxip']:
            if self.integration_method == 'simps':
                # cov_mix_simps also needs self, I pass it here directly by creating a
                # partial function
                cov_out_6d = self.cov_simps_wrapper(
                    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                    zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu,
                    func=partial(cov_mix_simps, self=self)
                )  # fmt: skip

            elif self.integration_method == 'levin':
                cov_out_6d = self.cov_mix_levin(
                    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                    zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
                )  # fmt: skip

        elif term == 'mix' and probe_abcd in ['ggxim', 'ggxip']:
            cov_out_6d = np.zeros(
                (self.nbt_fine, self.nbt_fine,
                 self.zbins, self.zbins, self.zbins, self.zbins)
            )  # fmt: skip

        elif term == 'sn':
            # this is 0 for
            # ['xipxim', 'gtxim', 'gtxip', 'ggxim', 'gggt', 'ggxip']
            # but is very fast to compute so I don't skip these terms
            cov_out_6d = self.cov_sn_rs(
                probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu
            )

        elif term == 'gauss_ell':
            print('Projecting ell-space Gaussian covariance...')

            # ! Compute HS G SVA, MIX and SN (not used), then project them to RS
            # build noise vector
            noise_3x2pt_4D = sl.build_noise(
                self.zbins,
                n_probes=self.n_probes_hs,
                sigma_eps2=(self.sigma_eps_i * np.sqrt(2)) ** 2,
                ng_shear=self.n_eff_src,
                ng_clust=self.n_eff_lens,
            )

            # expand the noise array along the ell axis
            noise_5d = np.repeat(noise_3x2pt_4D[:, :, None, :, :], self.nbl, axis=2)

            # ! no delta_ell!!
            delta_ell = np.ones_like(self.ells + 1)

            cov_sva_sb_hs_10d, _cov_sn_sb_hs_10d, cov_mix_sb_hs_10d = sl.compute_g_cov(
                self.cl_3x2pt_5d,
                noise_5d,
                self.fsky,
                self.ells,
                delta_ell,
                split_terms=True,
                return_only_diagonal_ells=True,
            )

            # sum sva and mix in harmonic space ("svapmix" = "sva plus mix")
            cov_svapmix_hs_6d = (
                cov_sva_sb_hs_10d[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix]
                + cov_mix_sb_hs_10d[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix]
            )

            self.cov_svapmix_rs_6d = levin_integrate_bessel_double_wrapper(
                integrand=cov_svapmix_hs_6d.reshape(self.nbl, -1)
                * self.ells[:, None]
                * self.ells[:, None],
                x_values=self.ells,
                bessel_args=self.theta_centers_fine,
                bessel_type=3,
                ell_1=mu,
                ell_2=nu,
                n_jobs=self.n_jobs,
                **self.levin_prec_kw,
            )
            self.cov_svapmix_rs_6d = self.cov_svapmix_rs_6d.reshape(
                self.nbt_fine, self.nbt_fine,
                self.zbins, self.zbins, self.zbins, self.zbins,
            )  # fmt: skip

            norm = 4 * np.pi**2
            self.cov_svapmix_rs_6d /= norm

            # add sn - projection is numerically unstable,
            # I compute it in real space directly
            self.cov_sn_rs_6d = self.cov_sn_rs(
                probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu
            )
            # diagonal is noise-dominated, you won't see much of a diff
            cov_gauss_ell_rs_6d = self.cov_svapmix_rs_6d + self.cov_sn_rs_6d

            cov_out_6d = cov_gauss_ell_rs_6d

        # elif term in ['ssc', 'cng']:
        # warnings.warn('HS covs loaded from file', stacklevel=2)
        # get OC SSC in ell space
        # covs_oc_hs = oc_cov_list_to_array(f'{covs_path}/{cov_hs_list_name}.dat')
        # (
        #     cov_sva_oc_3x2pt_10d,
        #     cov_mix_oc_3x2pt_10d,
        #     cov_sn_oc_3x2pt_10d,
        #     cov_g_oc_3x2pt_10d,
        #     cov_ssc_oc_3x2pt_10d,
        # ) = covs_oc_hs

        # np.savez(
        #     f'{covs_path}/covs_oc_10d.npz',
        #     cov_sva_oc_3x2pt_10d=cov_sva_oc_3x2pt_10d,
        #     cov_mix_oc_3x2pt_10d=cov_mix_oc_3x2pt_10d,
        #     cov_sn_oc_3x2pt_10d=cov_sn_oc_3x2pt_10d,
        #     cov_g_oc_3x2pt_10d=cov_g_oc_3x2pt_10d,
        #     cov_ssc_oc_3x2pt_10d=cov_ssc_oc_3x2pt_10d,
        # )

        # covs_oc_hs_npz = np.load(f'{covs_oc_path}/covs_oc_10d.npz')
        # cov_ssc_oc_3x2pt_10d = covs_oc_hs_npz['cov_ssc_oc_3x2pt_10d']
        # cov_cng_oc_3x2pt_10d = covs_oc_hs_npz['cov_ng_oc_3x2pt_10d']

        elif term in ['ssc', 'cng']:
            # TODO this is yet to be checked
            # TODO this has to be computed on a sufficiently fine ell grid, may pose
            # TODO memory issues?

            # set normalization depending on the term
            norm = 4 * np.pi**2
            if term == 'cng':
                norm *= self.amax

            cov_ng_hs_10d = getattr(cov_hs_obj, f'cov_3x2pt_{term}_10d')

            # project hs nf cov to real space using pylevin
            cov_ng_hs_6d = cov_ng_hs_10d[
                probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...
            ]

            cov_ng_rs_6d = dl1dl2_bessel_wrapper(
                cov_hs=cov_ng_hs_6d,
                mu=mu,
                nu=nu,
                ells=self.ells,
                thetas=self.theta_centers_fine,
                zbins=self.zbins,
                n_jobs=self.n_jobs,
                levin_prec_kw=self.levin_prec_kw,
            )
            cov_ng_rs_6d /= norm

            cov_out_6d = cov_ng_rs_6d

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
            for (zi, zj, zk, zl), cov in zip(zijkl_comb, results):
                cov_rs_6d_binned[:, :, zi, zj, zk, zl] = cov

            cov_out_6d = cov_rs_6d_binned

        # ! reshape 6D to 2D and store this as well in the object

        # setattr(self, f'cov_{probe_abcd}_{term}_6d', cov_out_6d)
        # these are needed for 3x2pt 2D
        # setattr(self, f'cov_{probe_abcd}_{term}_4d', cov_out_4d)
        # This is not necessary since I don't save the probe-specific 2D covs anymore
        # (see comment in combine_terms_and_probes)
        # setattr(self, f'cov_{probe}_{term}_2d', cov_out_2d)

        self.cov_dict[term][probe_2tpl]['6d'] = cov_out_6d

    def fill_remaining_probe_blocks_6d(
        self, term, symm_probe_combs, nonreq_probe_combs
    ):
        """Fill the remaining probe combinations by symmetry or
        set them to 0 if not required."""

        # * fill the symmetric counterparts of the required blocks
        # * (excluding diagonal blocks)
        for probe_abcd in symm_probe_combs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'real')
            print(f'RS cov: filling probe combination {probe_ab, probe_cd} by symmetry')

            cov_cdab = self.cov_dict[term][probe_cd, probe_ab]['6d']
            cov = (cov_cdab.transpose(1, 0, 4, 5, 2, 3)).copy()
            self.cov_dict[term][probe_ab, probe_cd]['6d'] = cov

        # # * if block is not required, set it to 0
        for probe_abcd in nonreq_probe_combs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='real')
            probe_2tpl = (probe_ab, probe_cd)

            self.cov_dict[term][probe_2tpl]['6d'] = np.zeros(
                (
                    self.nbt_coarse,
                    self.nbt_coarse,
                    self.zbins,
                    self.zbins,
                    self.zbins,
                    self.zbins,
                )
            )

    def _cov_probeblocks_6d_to_4d_and_2d(self, term):
        """
        For the input term, transforms all 6d probe-blocks into 4d and 2d.
        Note: this does not apply to 3x2pt!
        """

        for probe_2tpl in self.cov_dict[term]:
            if probe_2tpl == '3x2pt':
                continue  # skip 3x2pt, handled elsewhere

            probe_abcd = probe_2tpl[0] + probe_2tpl[1]

            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = (
                const.RS_PROBE_NAME_TO_IX_DICT[probe_abcd]
            )

            ind_ab = (
                self.ind_auto[:, 2:] if probe_a_ix == probe_b_ix
                else self.ind_cross[:, 2:]
            )  # fmt: skip
            ind_cd = (
                self.ind_auto[:, 2:] if probe_c_ix == probe_d_ix
                else self.ind_cross[:, 2:]
            )  # fmt: skip

            zpairs_ab = (
                self.zpairs_auto if probe_a_ix == probe_b_ix else self.zpairs_cross
            )
            zpairs_cd = (
                self.zpairs_auto if probe_c_ix == probe_d_ix else self.zpairs_cross
            )

            # just a sanity check
            assert zpairs_ab == ind_ab.shape[0], 'zpairs-ind inconsistency'
            assert zpairs_cd == ind_cd.shape[0], 'zpairs-ind inconsistency'

            cov_6d = self.cov_dict[term][probe_2tpl]['6d']
            cov_4d = sl.cov_6D_to_4D_blocks(
                cov_6d, self.nbt_coarse, zpairs_ab, zpairs_cd, ind_ab, ind_cd
            )
            cov_2d = sl.cov_4D_to_2D(
                cov_4d, block_index=self.block_index, optimize=True
            )
            self.cov_dict[term][probe_2tpl]['4d'] = cov_4d
            self.cov_dict[term][probe_2tpl]['2d'] = cov_2d

    def fill_remaining_probe_blocks_4d(
        self, term, symm_probe_combs, nonreq_probe_combs
    ):
        """Fill the remaining probe combinations by symmetry or
        set them to 0 if not required."""

        # * fill the symmetric counterparts of the required blocks
        # * (excluding diagonal blocks)
        for probe_abcd in symm_probe_combs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'real')
            print(f'RS cov: filling probe combination {probe_ab, probe_cd} by symmetry')

            cov_cdab = self.cov_dict[term][probe_cd, probe_ab]['4d']
            cov = (cov_cdab.transpose(1, 0, 3, 2)).copy()
            self.cov_dict[term][probe_ab, probe_cd]['4d'] = cov

        # TODO verify that commenting this doesn't break anything
        # * if block is not required, set it to 0
        # for probe_abcd in nonreq_probe_combs:
        #     probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'real')
        #     probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = (
        #         const.RS_PROBE_NAME_TO_IX_DICT[probe_abcd]
        #     )
        #     zpairs_ab = (
        #         self.zpairs_auto if probe_a_ix == probe_b_ix else self.zpairs_cross
        #     )
        #     zpairs_cd = (
        #         self.zpairs_auto if probe_c_ix == probe_d_ix else self.zpairs_cross
        #     )
        #     cov = np.zeros((self.nbt_coarse, self.nbt_coarse, zpairs_ab, zpairs_cd))
        #     self.cov_dict[term][probe_ab, probe_cd]['4d'] = cov
        # setattr(self, f'cov_{probe_abcd}_{term}_4d', cov)
