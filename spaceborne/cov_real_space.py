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
from collections.abc import Callable
from functools import partial

import numpy as np
import pyccl as ccl

# import pylevin as levin
from joblib import Parallel, delayed
from scipy.integrate import simpson as simps
from tqdm import tqdm

from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import cov_projector as cp
from spaceborne import sb_lib as sl
from spaceborne.cov_projector import CovarianceProjector

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


# ! ====================== COV RS W/ SIMPSON INTEGRATION ===============================


def t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i):
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


# ! ====================================================================================
# ! ====================================================================================
# ! ====================================================================================


class CovRealSpace(CovarianceProjector):
    def __init__(self, cfg, pvt_cfg, mask_obj):
        super().__init__(cfg, pvt_cfg, mask_obj)

        self.obs_space = 'real'

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
        self._set_theta_binning()
        self._set_levin_bessel_precision()

        # other miscellaneous settings
        self.integration_method = self.cfg['precision']['cov_rs_int_method']
        self.levin_bin_avg = self.cfg['precision']['levin_bin_avg']

        assert self.integration_method in ['simps', 'levin'], (
            'integration method not implemented'
        )

        # attributes set at runtime
        self.cl_3x2pt_5d = _UNSET
        self.ells = _UNSET
        self.nbl = _UNSET

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

            # [BOOKMARK 9 dec] finish checking this
            if self.cfg['binning']['binning_type'] == 'log':
                theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])
            elif self.cfg['binning']['binning_type'] == 'lin':
                theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0

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
        # TODO generalize to different n(z)
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

        import ipdb

        ipdb.set_trace()
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(self.theta_centers_fine, npair_arr[:, 0, 0])
        # plt.savefig('debug_sb.png')

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
            * t_arr[None, None, :, None, :, None]
            / npair_arr[None, :, :, :, None, None]
        )

        return cov_sn_rs_6d

    def cov_sva_levin(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
    ):  # fmt: skip
        # Use parent method to build the universal SVA integrand
        integrand_5d = self.build_cov_sva_integrand_5d(
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix
        )

        # Child-specific: project with Levin + Bessel kernels
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

    def compute_rs_cov_term_probe_6d(self, cov_hs_obj, probe_abcd, term):
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
            if self.integration_method in ['simps', 'quad']:
                cov_out_6d = self.cov_simps_wrapper(
                    zpairs_ab=zpairs_ab,
                    zpairs_cd=zpairs_cd,
                    ind_ab=ind_ab,
                    ind_cd=ind_cd,
                    cov_simps_func=self.cov_sva_simps,
                    cov_simps_func_kw=cov_simps_func_kw,
                    kernel_builder_func_kw=kernel_builder_func_kw,
                )

            elif self.integration_method == 'levin':
                cov_out_6d = self.cov_sva_levin(
                    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                    zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
                )  # fmt: skip

        elif term == 'mix' and probe_abcd not in ['ggxim', 'ggxip']:
            if self.integration_method == 'simps':
                cov_out_6d = self.cov_simps_wrapper(
                    zpairs_ab=zpairs_ab,
                    zpairs_cd=zpairs_cd,
                    ind_ab=ind_ab,
                    ind_cd=ind_cd,
                    cov_simps_func=self.cov_mix_simps,
                    cov_simps_func_kw=cov_simps_func_kw,
                    kernel_builder_func_kw=kernel_builder_func_kw,
                )

            elif self.integration_method == 'levin':
                cov_out_6d = self.cov_mix_levin(
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

        elif term == 'gauss_ell':
            print('Projecting ell-space Gaussian covariance...')

            # ! Compute HS G SVA, MIX and SN (not used), then project them to RS
            # build noise vector
            noise_3x2pt_4D = sl.build_noise(
                self.zbins,
                n_probes=self.n_probes_hs,
                sigma_eps2=(self.sigma_eps_i * np.sqrt(2)) ** 2,
                ng_shear=self.n_eff_src,
                ng_clust=self.n_eff_lns,
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

            # project hs non-gaussian cov to real space using pylevin
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
            for (zi, zj, zk, zl), cov in zip(zijkl_comb, results, strict=True):
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

    def k_mu(self, ell, *, thetal, thetau, mu):
        """Thin wrapper around k_mu, just to make it a method of the class"""
        return k_mu(ell, thetal=thetal, thetau=thetau, mu=mu)
