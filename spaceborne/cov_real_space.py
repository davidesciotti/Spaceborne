import itertools
import re
import time
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyccl as ccl
import pylevin as levin
from joblib import Parallel, delayed
from scipy.integrate import quad_vec
from scipy.integrate import simpson as simps
from scipy.stats import chi2
from tqdm import tqdm

from spaceborne import sb_lib as sl

warnings.filterwarnings(
    'ignore', message=r'.*invalid escape sequence.*', category=SyntaxWarning
)

warnings.filterwarnings(
    'ignore',
    message=r'.*invalid value encountered in divide.*',
    category=RuntimeWarning,
)


def b_mu(x, mu):
    """Implements the piecewise definition of the bracketed term b_mu(x)
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


# def kmu_nobessel(ell, thetal, thetau, mu):
#     """Returns the prefactors and corresponding Bessel orders for given mu and nu,
#     including the subtraction of bin edges (b_mu(ell * theta_u) - b_mu(ell * theta_l)).

#     Parameters
#     ----------
#     mu, nu : int
#         Orders of the kernel.
#     ell : float or array
#         Multipole values.
#     theta_l, theta_u : float
#         Lower and upper bin edges.

#     Returns
#     -------
#     list of tuples
#         Each tuple contains (prefactor, bessel_order_mu, bessel_order_nu).

#     """

#     def b_mu_nobessel(x, mu):
#         """Implements the piecewise definition of the bracketed term b_mu(x)
#         from Eq. (E.2) in Joachimi et al. (2008).
#         These are just the results of
#         \int_{\theta_l}^{\theta_u} d\theta \theta J_\mu(\ell \theta)
#         """
#         if mu == 0:
#             return x * sl.j1(x)
#         elif mu == 2:
#             return -x * sl.j1(x) - 2.0 * sl.j0(x)
#         elif mu == 4:
#             # be careful with x=0!
#             return (x - 8.0 / x) * sl.j1(x) - 8.0 * sl.j2(x)
#         else:
#             raise ValueError('mu must be one of {0,2,4}.')

#     def b_mu_nobessel(mu):
#         """Returns (prefactor, bessel_order) pairs for a given mu."""
#         if mu == 0:
#             return [(1.0, 1)]  # x J1(x)
#         elif mu == 2:
#             return [(-1.0, 1), (-2.0, 0)]  # -x J1(x) - 2 J0(x)
#         elif mu == 4:
#             return [(1.0, 1), (-8.0, 2)]  # (x - 8/x) J1(x) - 8 J2(x)
#         else:
#             raise ValueError('mu must be one of {0,2,4}.')

#     # Compute b_mu at both bin edges
#     b_mu = b_mu(ell * theta, mu)
#     b_nu = b_mu(ell * theta, nu)

#     # Compute all possible combinations of terms
#     combined_terms = []
#     for pref_mu, j_mu in b_mu_nobessel(mu):
#         for pref_nu, j_nu in b_mu_nobessel(nu):
#             prefactor = (
#                 (pref_mu * pref_nu)
#                 * (b_mu_upper - b_mu_lower)
#                 * (b_nu_upper - b_nu_lower)
#             )
#             combined_terms.append((prefactor, j_mu, j_nu))

#     return combined_terms


def k_mu(ell, thetal, thetau, mu):
    """Computes the kernel K_mu(ell * theta_i) in Eq. (E.2):

    K_mu(l * theta_i) = 2 / [ (theta_u^2 - theta_l^2) * l^2 ]
                        * [ b_mu(l * theta_u) - b_mu(l * theta_l) ].
    """
    prefactor = 2.0 / ((thetau**2 - thetal**2) * (ell**2))
    return prefactor * (b_mu(ell * thetau, mu) - b_mu(ell * thetal, mu))


def project_ellspace_cov_vec_2d(  # fmt: skip
    theta_1_l, theta_1_u, mu,
    theta_2_l, theta_2_u, nu,
    Amax, ell2_values, ell1_values, cov_ell
):  # fmt: skip
    """This version is fully vectorized"""

    def integrand_func(ell1, ell2, cov_ell):
        kmu = k_mu(ell1, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell2, theta_2_l, theta_2_u, nu)

        # Broadcast to handle all combinations of ell1 and ell2
        ell1_matrix, ell2_matrix = np.meshgrid(ell1, ell2)
        kmu_matrix, knu_matrix = np.meshgrid(kmu, knu)

        # Compute the integrand
        part_product = ell1_matrix * ell2_matrix * kmu_matrix * knu_matrix
        integrand = part_product[:, :, None, None, None, None] * cov_ell

        return integrand

    # Compute the integrand for all combinations of ell1 and ell2
    integrand = integrand_func(ell1_values, ell2_values, cov_ell)

    part_integral = simps(y=integrand, x=ell1_values, axis=0)
    integral = simps(y=part_integral, x=ell2_values, axis=0)  # axis=1?

    # Finally multiply the prefactor
    cov_elem = integral / (4.0 * np.pi**2 * Amax)
    return cov_elem


def project_ellspace_cov_vec_1d(  # fmt: skip
    theta_1_l, theta_1_u, mu,
    theta_2_l, theta_2_u, nu,
    Amax, ell1_values, ell2_values, cov_ell_diag,
):  # fmt: skip
    """This version is vectorized anly along ell1"""

    def integrand_func(ell1, ell2, cov_ell_diag):
        # Vectorized computation of k_mu and k_nu
        kmu = k_mu(ell1, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell2, theta_2_l, theta_2_u, nu)

        # Compute the integrand
        part_product = ell1 * ell2 * kmu * knu
        integrand = part_product[:, None, None, None, None] * cov_ell_diag
        return integrand

    # Compute the integrand for all combinations of ell1 and ell2
    integrand = integrand_func(ell1_values, ell2_values, cov_ell_diag)

    integral = simps(y=integrand, x=ell1_values, axis=0)  # axis=1?

    # Finally multiply the prefactor
    cov_elem = integral / (4.0 * np.pi**2 * Amax)
    return cov_elem


def project_hs_cov_simps(  # fmt: skip
    theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu,
    zi, zj, zk, zl,
    Amax, ell1_values, ell2_values, cov_ell,
):  # fmt: skip
    # def integrand_func(ell1, ell2, cov_ell):
    #     kmu = k_mu(ell1, theta_1_l, theta_1_u, mu)
    #     knu = k_mu(ell2, theta_2_l, theta_2_u, nu)

    #     integrand = np.zeros_like(cov_ell)
    #     for ell2_ix, ell2 in enumerate(ell2_values):
    #         integrand[:, ell2_ix, zi, zj, zk, zl] = ell1 * kmu * ell2 * knu
    # * cov_ell[:, ell2_ix, zi, zk, zk, zl]

    #     return integrand

    # # Compute the integrand for all combinations of ell1 and ell2
    # integrand = integrand_func(ell1_values, ell2_values, cov_ell)

    # part_integral = simps(y=integrand, x=ell1_values, axis=0)
    # integral = simps(y=part_integral, x=ell2_values, axis=0)  # axis=1?

    # old school:
    inner_integral = np.zeros(len(ell1_values))

    for ell1_ix, _ in enumerate(ell1_values):
        inner_integrand = (
            ell2_values
            * k_mu(ell2_values, theta_2_l, theta_2_u, nu)
            * cov_ell[ell1_ix, :, zi, zj, zk, zl]
        )
        inner_integral[ell1_ix] = simps(y=inner_integrand, x=ell2_values)

    outer_integrand = (
        ell1_values**2 * k_mu(ell1_values, theta_1_l, theta_1_u, mu) * inner_integral
    )
    outer_integral = simps(y=outer_integrand, x=ell1_values)

    # Finally multiply the prefactor
    cov_elem = outer_integral / (4.0 * np.pi**2 * Amax)
    return cov_elem


def project_ellspace_cov_helper(    # fmt: skip
    theta_1_ix, theta_2_ix, mu, nu,
    zij, zkl, ind_ab, ind_cd,
    Amax, ell1_values, ell2_values, cov_ell,
):  # fmt: skip
    # TODO unify helper funcs

    theta_1_l = theta_edges[theta_1_ix]
    theta_1_u = theta_edges[theta_1_ix + 1]
    theta_2_l = theta_edges[theta_2_ix]
    theta_2_u = theta_edges[theta_2_ix + 1]

    zi, zj = ind_ab[zij, :]
    zk, zl = ind_cd[zkl, :]

    return (theta_1_ix, theta_2_ix, zi, zj, zk, zl,  # fmt: skip
        project_hs_cov_simps(
            theta_1_l, theta_1_u, mu,
            theta_2_l, theta_2_u, nu,
            zi, zj, zk, zl,
            Amax, ell1_values, ell2_values, cov_ell,
        ),
    )  # fmt: skip


def project_ellspace_cov_vec_helper(
    theta_1_ix, theta_2_ix, mu, nu, Amax, ell1_values, ell2_values, cov_ell
):
    theta_1_l = theta_edges[theta_1_ix]
    theta_1_u = theta_edges[theta_1_ix + 1]
    theta_2_l = theta_edges[theta_2_ix]
    theta_2_u = theta_edges[theta_2_ix + 1]

    return (theta_1_ix, theta_2_ix,  # fmt: skip
        project_ellspace_cov_vec_1d(
            theta_1_l, theta_1_u, mu,
            theta_2_l, theta_2_u, nu,
            Amax, ell1_values, ell2_values, cov_ell,
        ),
    )  # fmt: skip


def cov_sn_rs(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu):
    # TODO generalize to different n(z)
    npair_arr = np.zeros((nbt, zbins, zbins))
    for theta_ix in range(nbt):
        for zi in range(zbins):
            for zj in range(zbins):
                theta_1_l = theta_edges[theta_ix]
                theta_1_u = theta_edges[theta_ix + 1]
                npair_arr[theta_ix, zi, zj] = get_npair(
                    theta_1_u,
                    theta_1_l,
                    survey_area_sr,
                    n_eff_lens[zi],
                    n_eff_lens[zj],
                )

    delta_mu_nu = 1.0 if (mu == nu) else 0.0
    delta_theta = np.eye(nbt)
    t_arr = t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i)

    cov_sn_sb_6d = (
        delta_mu_nu
        * delta_theta[:, :, None, None, None, None]
        * (
            get_delta_tomo(probe_a_ix, probe_c_ix)[None, None, :, None, :, None]
            * get_delta_tomo(probe_b_ix, probe_d_ix)[None, None, None, :, None, :]
            + get_delta_tomo(probe_a_ix, probe_d_ix)[None, None, :, None, None, :]
            * get_delta_tomo(probe_b_ix, probe_c_ix)[None, None, None, :, :, None]
        )
        * t_arr[None, None, :, None, :, None]
        / npair_arr[None, :, :, :, None, None]
    )
    return cov_sn_sb_6d


def cov_parallel_helper(
    theta_1_ix, theta_2_ix, mu, nu, zij, zkl, ind_ab, ind_cd, func, **kwargs
):
    theta_1_l = theta_edges[theta_1_ix]
    theta_1_u = theta_edges[theta_1_ix + 1]
    theta_2_l = theta_edges[theta_2_ix]
    theta_2_u = theta_edges[theta_2_ix + 1]

    zi, zj = ind_ab[zij, :]
    zk, zl = ind_cd[zkl, :]

    return (  # fmt: skip
        theta_1_ix, theta_2_ix, zi, zj, zk, zl, func(
            theta_1_l=theta_1_l, theta_1_u=theta_1_u, mu=mu,
            theta_2_l=theta_2_l, theta_2_u=theta_2_u, nu=nu,
            zi=zi, zj=zj, zk=zk, zl=zl,
            **kwargs,
        ),
    )  # fmt: skip


def cov_sva_rs(  # fmt: skip
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


def cov_mix_rs(   # fmt: skip
    theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu,
    ell_values, cl_5d,
    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zi, zj, zk, zl,
    Amax, integration_method='simps',
):  # fmt: skip
    def integrand_func(ell, cl_ij):
        kmu = k_mu(ell, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell, theta_2_l, theta_2_u, nu)
        return ell * kmu * knu * cl_ij

    def integrand_scalar(ell, cl_ij):
        # Interpolate the value of cl_ij at the current ell value:
        cl_val = np.interp(ell, ell_values, cl_ij)
        kmu = k_mu(ell, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell, theta_2_l, theta_2_u, nu)
        return ell * kmu * knu * cl_val

    # I write the indices as in Robert's paper
    def get_prefac(probe_b_ix, probe_d_ix, zj, zn):
        prefac = (
            get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zn]
            * t_mix(probe_b_ix, zbins, sigma_eps_i)[zj]
            / (2 * np.pi * n_eff_2d[probe_b_ix, zj] * srtoarcmin2 * Amax)
        )
        return prefac

    # TODO generalize to different survey areas (max(Aij, Akl))
    # TODO sigma_eps_i should be a vector of length zbins
    # permutations should be performed as done in the SVA function
    prefac_1 = get_prefac(probe_a_ix, probe_c_ix, zi, zk)
    prefac_2 = get_prefac(probe_b_ix, probe_d_ix, zj, zl)
    prefac_3 = get_prefac(probe_a_ix, probe_d_ix, zi, zl)
    prefac_4 = get_prefac(probe_b_ix, probe_c_ix, zj, zk)

    if integration_method == 'simps':
        # as done in the SVA function
        integrand_1 = integrand_func(
            ell_values, cl_5d[probe_a_ix, probe_c_ix, :, zi, zk]
        )
        integrand_2 = integrand_func(
            ell_values, cl_5d[probe_b_ix, probe_d_ix, :, zj, zl]
        )
        integrand_3 = integrand_func(
            ell_values, cl_5d[probe_a_ix, probe_d_ix, :, zi, zl]
        )
        integrand_4 = integrand_func(
            ell_values, cl_5d[probe_b_ix, probe_c_ix, :, zj, zk]
        )

        integral_1 = simps(y=integrand_1, x=ell_values)
        integral_2 = simps(y=integrand_2, x=ell_values)
        integral_3 = simps(y=integrand_3, x=ell_values)
        integral_4 = simps(y=integrand_4, x=ell_values)

    elif integration_method == 'quad':
        integral_1 = quad_vec(
            integrand_scalar,
            ell_values[0],
            ell_values[-1],
            args=(cl_5d[probe_a_ix, probe_c_ix, :, zi, zk],),
        )[0]
        integral_2 = quad_vec(
            integrand_scalar,
            ell_values[0],
            ell_values[-1],
            args=(cl_5d[probe_b_ix, probe_d_ix, :, zj, zl],),
        )[0]
        integral_3 = quad_vec(
            integrand_scalar,
            ell_values[0],
            ell_values[-1],
            args=(cl_5d[probe_a_ix, probe_d_ix, :, zi, zl],),
        )[0]
        integral_4 = quad_vec(
            integrand_scalar,
            ell_values[0],
            ell_values[-1],
            args=(cl_5d[probe_b_ix, probe_c_ix, :, zj, zk],),
        )[0]

    else:
        raise ValueError(f'integration_method {integration_method} not recognized.')

    return (
        prefac_1 * integral_1
        + prefac_2 * integral_2
        + prefac_3 * integral_3
        + prefac_4 * integral_4
    )


def cov_g_mix_real_new(  # fmt: skip
    theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu,
    ell_values, cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
    zi, zj, zk, zl, Amax,
    integration_method='simps',
):  # fmt: skip
    def integrand_func(ell, inner_integrand):
        kmu = k_mu(ell, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell, theta_2_l, theta_2_u, nu)
        return (1 / (2 * np.pi * Amax)) * ell * kmu * knu * inner_integrand

    def get_prefac(probe_a_ix, probe_b_ix, zi, zj):
        prefac = (
            get_delta_tomo(probe_a_ix, probe_b_ix)[zi, zj]
            * t_mix(probe_a_ix, zbins, sigma_eps_i)[zi]
            / (n_eff_2d[probe_a_ix, zi] * srtoarcmin2)
        )
        return prefac

    # TODO generalize to different survey areas (max(Aij, Akl))
    # TODO sigma_eps_i should be a vector of length zbins

    # permutations should be performed as done in the SVA function
    if integration_method == 'simps':
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

    else:
        raise ValueError(f'integration_method {integration_method} not recognized.')

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
    n_eff_i *= srtoarcmin2
    n_eff_j *= srtoarcmin2
    return np.pi * (theta_1_u**2 - theta_1_l**2) * survey_area_sr * n_eff_i * n_eff_j


def get_delta_tomo(probe_a_ix, probe_b_ix):
    if probe_a_ix == probe_b_ix:
        return np.eye(zbins)
    else:
        return np.zeros((zbins, zbins))


def split_probe_name(full_probe_name):
    """Splits a full probe name (e.g., 'gmxim') into two component probes.

    Possible probe names are 'xip', 'xim', 'gg', 'gm'.

    Args:
        full_probe_name (str): A string containing two probe types concatenated.

    Returns:
        tuple: A tuple of two strings representing the split probes.

    Raises:
        ValueError: If the input string does not contain exactly two valid probes.

    """
    valid_probes = {'xip', 'xim', 'gg', 'gm'}

    # Try splitting at each possible position
    for i in range(2, len(full_probe_name)):
        first, second = full_probe_name[:i], full_probe_name[i:]
        if first in valid_probes and second in valid_probes:
            return first, second

    raise ValueError(
        f'Invalid probe name: {full_probe_name}. '
        f'Expected two of {valid_probes} concatenated.'
    )


def split_probe_ix(probe_ix):
    if probe_ix in (0, 1):
        return 0, 0
    elif probe_ix == 2:
        return 1, 0
    elif probe_ix == 3:
        return 1, 1
    else:
        raise ValueError(f'Invalid probe index: {probe_ix}. Expected 0, 1, 2, or 3.')


def integrate_bessel_single_wrapper(cov_2d, mu, ell, theta_centers, n_jobs):
    """cov_2d must have the first axis corresponding to the ell values),
    the second to the flattened remaining dimensions
    """
    assert cov_2d.ndim == 2, 'the input integrand must be 2D'

    integral_type = 1  # single cilyndrical bessel

    integrand = cov_2d

    # Constructor of the class
    lp = levin.pylevin(
        type=integral_type,
        x=ell,
        integrand=integrand,
        logx=logx,
        logy=logy,
        nthread=n_jobs,
        diagonal=False,
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


def oc_cov_list_to_array(oc_output_path):
    probe_idx_dict_ell = {
        'm': 0,
        'g': 1,
    }

    # set df column names
    with open(oc_output_path) as file:
        header = (
            file.readline().strip()
        )  # Read the first line and strip newline characters
    header_list = re.split(
        '\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t')
    )
    column_names = header_list

    print('Loading OC ell and z values...')
    data = pd.read_csv(oc_output_path, usecols=['ell1', 'tomoi'], sep='\s+')
    ells_oc_load = data['ell1'].unique()
    tomoi_oc_load = data['tomoi'].unique()

    cov_ell_indices = {ell_out: idx for idx, ell_out in enumerate(ells_oc_load)}
    # SB tomographic indices start from 0
    subtract_one = False
    if min(tomoi_oc_load) == 1:
        subtract_one = True

    # import .list covariance file
    shape = (n_probes_hs, n_probes_hs, n_probes_hs, n_probes_hs, # fmt: skip
            nbl, nbl, zbins, zbins, zbins, zbins)  # fmt: skip
    cov_g_oc_3x2pt_10D = np.zeros(shape)
    cov_sva_oc_3x2pt_10D = np.zeros(shape)
    cov_mix_oc_3x2pt_10D = np.zeros(shape)
    cov_sn_oc_3x2pt_10D = np.zeros(shape)
    cov_ssc_oc_3x2pt_10D = np.zeros(shape)
    # cov_cng_oc_3x2pt_10D = np.zeros(shape)
    # cov_tot_oc_3x2pt_10D = np.zeros(shape)

    print(f'Loading OneCovariance output from \n{oc_output_path}')
    for df_chunk in pd.read_csv(
        oc_output_path,
        sep='\s+',
        names=column_names,
        skiprows=1,
        chunksize=df_chunk_size,
    ):
        probe_idx_a = df_chunk['#obs'].str[0].map(probe_idx_dict_ell).values
        probe_idx_b = df_chunk['#obs'].str[1].map(probe_idx_dict_ell).values
        probe_idx_c = df_chunk['#obs'].str[2].map(probe_idx_dict_ell).values
        probe_idx_d = df_chunk['#obs'].str[3].map(probe_idx_dict_ell).values

        # Map 'ell' values to their corresponding indices
        ell1_idx = df_chunk['ell1'].map(cov_ell_indices).values
        ell2_idx = df_chunk['ell2'].map(cov_ell_indices).values

        # Compute z indices
        if subtract_one:
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values
        else:
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values

        # Vectorized assignment to the arrays
        index_tuple = (  # fmt: skip
            probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
            ell1_idx, ell2_idx,
            z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3],
        )  # fmt: skip

        cov_sva_oc_3x2pt_10D[index_tuple] = df_chunk['covg sva'].values
        cov_mix_oc_3x2pt_10D[index_tuple] = df_chunk['covg mix'].values
        cov_sn_oc_3x2pt_10D[index_tuple] = df_chunk['covg sn'].values
        cov_g_oc_3x2pt_10D[index_tuple] = (
            df_chunk['covg sva'].values
            + df_chunk['covg mix'].values
            + df_chunk['covg sn'].values
        )
        cov_ssc_oc_3x2pt_10D[index_tuple] = df_chunk['covssc'].values
        # cov_cng_oc_3x2pt_10D[index_tuple] = df_chunk['covng'].values
        # cov_tot_oc_3x2pt_10D[index_tuple] = df_chunk['cov'].values

    covs_10d = [
        cov_sva_oc_3x2pt_10D,
        cov_mix_oc_3x2pt_10D,
        cov_sn_oc_3x2pt_10D,
        cov_g_oc_3x2pt_10D,
        cov_ssc_oc_3x2pt_10D,
        # cov_cng_oc_3x2pt_10D,
        # cov_tot_oc_3x2pt_10D
    ]

    for cov_10d in covs_10d:
        cov_10d[0, 0, 1, 1] = deepcopy(
            np.transpose(cov_10d[1, 1, 0, 0], (1, 0, 4, 5, 2, 3))
        )
        cov_10d[1, 0, 0, 0] = deepcopy(
            np.transpose(cov_10d[0, 0, 1, 0], (1, 0, 4, 5, 2, 3))
        )
        cov_10d[1, 0, 1, 1] = deepcopy(
            np.transpose(cov_10d[1, 1, 1, 0], (1, 0, 4, 5, 2, 3))
        )

    return covs_10d


def dl1dl2_bessel_wrapper(
    cov_hs: np.ndarray,
    mu: int,
    nu: int,
    ells: np.ndarray,
    thetas: np.ndarray,
    n_jobs: int,
):
    """Wrapper function to compute the double Bessel integral of the form
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
            integrand, nu, ells, thetas, n_jobs
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
            integrand_second, mu, ells, thetas, n_jobs
        )
        final_result[:, theta_idx, :] = final_int

    cov_rs_6d = final_result.reshape(nbt, nbt, zbins, zbins, zbins, zbins)

    return cov_rs_6d


def integrate_bessel_double_wrapper(
    integrand, x_values, bessel_args, bessel_type, ell_1, ell_2
):
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
        nthread=n_jobs_lv,
        diagonal=False,
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


def stack_cov_blocks(cov_2d_dict):
    row_1 = np.hstack(
        (
            cov_2d_dict['gggg'],
            cov_2d_dict['gggm'],
            cov_2d_dict['ggxip'],
            cov_2d_dict['ggxim'],
        )
    )
    row_2 = np.hstack(
        (
            cov_2d_dict['gggm'].T,
            cov_2d_dict['gmgm'],
            cov_2d_dict['gmxip'],
            cov_2d_dict['gmxim'],
        )
    )
    row_3 = np.hstack(
        (
            cov_2d_dict['ggxip'].T,
            cov_2d_dict['gmxip'].T,
            cov_2d_dict['xipxip'],
            cov_2d_dict['xipxim'],
        )
    )
    row_4 = np.hstack(
        (
            cov_2d_dict['ggxim'].T,
            cov_2d_dict['gmxim'].T,
            cov_2d_dict['xipxim'].T,
            cov_2d_dict['ximxim'],
        )
    )

    return np.vstack((row_1, row_2, row_3, row_4))


def twopcf_wrapper(zi, zj, ell_grid, theta_grid, cl_3D, correlation_type, method):
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


# import matplotlib
# matplotlib.use('Agg')

# ! ====================================================================================
# ! ====================================================================================
# ! ====================================================================================

# levin bessel settings
logx = True
logy = True
diagonal = False

# accuracy settings
n_sub = 8  # number of collocation points in each bisection  # default 8
n_bisec_max = 80  # maximum number of bisections used  # default 32
rel_acc = 5e-4  # relative accuracy target  # default 1e-4
# should the bessel functions be calculated with boost instead of GSL,
# higher accuracy at high Bessel orders
boost_bessel = True
verbose = True  # should the code talk to you?

zbins = 3
survey_area_deg2 = 2500
deg2torad2 = (180 / np.pi) ** 2
srtoarcmin2 = (180 / np.pi * 60) ** 2
survey_area_sr = survey_area_deg2 / deg2torad2
fsky = survey_area_sr / (4 * np.pi)
Amax = max((survey_area_sr, survey_area_sr))
covs_oc_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test'

ell_min = 1
ell_max = 100_000
nbl = 200
theta_min_arcmin = 50
theta_max_arcmin = 300
n_theta_edges = 51
n_theta_edges_coarse = 21
df_chunk_size = 50000
cov_list_name = 'covariance_list_3x2_rcf_v2'
cov_hs_list_name = 'covariance_list_3x2_cl'
triu_tril = 'triu'
row_col_major = 'row-major'  # unit: is gal/arcmin^2
n_jobs = -1  # leave one thread free?
n_jobs_lv = 16  # might cause memory issues if too high
tpcf_ingr_method = 'fftlog'

theta_edges_coarse = np.linspace(
    theta_min_arcmin / 60, theta_max_arcmin / 60, n_theta_edges_coarse
)
theta_edges_coarse = np.deg2rad(theta_edges_coarse)
theta_centers_coarse = (theta_edges_coarse[:-1] + theta_edges_coarse[1:]) / 2.0
nbt_coarse = len(theta_centers_coarse)  # nbt = number theta bins

n_eff_lens = np.array([8.09216, 8.09215, 8.09215])
n_eff_src = np.array([8.09216, 8.09215, 8.09215])

n_eff_2d = np.row_stack(
    (n_eff_lens, n_eff_lens, n_eff_src)
)  # in this way the indices correspond to xip, xim, g
sigma_eps_i = np.array([0.26, 0.26, 0.26])
sigma_eps_tot = sigma_eps_i * np.sqrt(2)
munu_vals = (0, 2, 4)
n_probes_rs = 4  # real space
n_probes_hs = 2  # harmonic space
n_split_terms = 3
cov_sb_dict_8d = np.zeros((n_split_terms, n_probes_rs, n_probes_rs,  # fmt: skip
                       n_theta_edges_coarse-1, n_theta_edges_coarse-1,  # fmt: skip
                       zbins, zbins, zbins, zbins))  # fmt: skip

mu_dict = {'gg': 0, 'gm': 2, 'xip': 0, 'xim': 4}

# ! careful: in this representation, xipxip and ximxim (eg) have the same indices!!
probe_idx_dict = {
    'xipxip': (0, 0, 0, 0),
    'xipxim': (0, 0, 0, 0),
    'ximxim': (0, 0, 0, 0),
    'gmgm': (1, 0, 1, 0),
    'gmxim': (1, 0, 0, 0),
    'gmxip': (1, 0, 0, 0),
    'gggg': (1, 1, 1, 1),
    'ggxim': (1, 1, 0, 0),
    'gggm': (1, 1, 1, 0),
    'ggxip': (1, 1, 0, 0),
}


probe_idx_dict_short = {
    'gg': 0,  # w
    'gm': 1,  # \gamma_t
    'xip': 2,
    'xim': 3,
}

# this is only needed to be able to construct the full Gauss cov from the sum of the
# SVA, SN and MIX covs. No particular reason behind the choice of the indices.
split_g_dict = {
    'sva': 0,
    'sn': 1,
    'mix': 2,
}


probe_idx_dict_short_oc = {}
for key in probe_idx_dict:
    probe_a_str, probe_b_str = split_probe_name(key)
    probe_idx_dict_short_oc[probe_a_str + probe_b_str] = (
        probe_idx_dict_short[probe_a_str],
        probe_idx_dict_short[probe_b_str],
    )

terms_toloop = ['sva', 'sn', 'mix']
terms_toloop = ['sva']
integration_method = 'levin'
probes_toloop = probe_idx_dict
probes_toloop = ['gggm']

assert integration_method in ['simps', 'levin'], 'integration method not implemented'

theta_edges = np.linspace(theta_min_arcmin / 60, theta_max_arcmin / 60, n_theta_edges)
theta_edges = np.deg2rad(theta_edges)
# TODO in principle this could be changed
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
nbt = len(theta_centers)  # nbt = number theta bins

zpairs_auto, zpairs_cross, zpairs_3x2pt = sl.get_zpairs(zbins)
ind = sl.build_full_ind(triu_tril, row_col_major, zbins)
ind_auto = ind[:zpairs_auto, :].copy()
ind_cross = ind[zpairs_auto : zpairs_cross + zpairs_auto, :].copy()
ind_dict = {('L', 'L'): ind_auto, ('G', 'L'): ind_cross, ('G', 'G'): ind_auto}

# * basically no difference between the two recipes below!
# * (The one above is obviously much slower)
# ell_values = np.arange(ell_min, ell_max)
ell_values = np.geomspace(ell_min, ell_max, nbl)

# quick and dirty cls computation
cosmo = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.05,
    h=0.67,
    sigma8=0.816,
    n_s=0.966,
    m_nu=0.06,
    w0=-1.0,
    Neff=3.046,
    extra_parameters={
        'camb': {'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.75}
    },
)

# bias_values = [1.1440270903053593, 1.209969007589984, 1.3354449071064036,
#                1.4219803534945, 1.5275589801638865, 1.9149796097338934]
# # create an array with the bias values in each column, and the first
# bias_2d = np.tile(bias_values, reps=(len(z_nz_lenses), 1))
# bias_2d = np.column_stack((z_nz_lenses, bias_2d))
oc_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/output/OneCovariance'
nz_lenses = np.genfromtxt(
    f'{oc_path}/nzTab-EP03-zedMin02-zedMax25-mag245_dzshiftsTrue.ascii'
)
nz_sources = np.genfromtxt(
    f'{oc_path}/nzTab-EP03-zedMin02-zedMax25-mag245_dzshiftsTrue.ascii'
)
bias_2d = np.genfromtxt(f'{oc_path}/gal_bias_table.ascii')
z_nz_lenses = nz_lenses[:, 0]
z_nz_sources = nz_sources[:, 0]

wl_ker = [
    ccl.WeakLensingTracer(  # fmt: skip
        cosmo=cosmo, dndz=(nz_sources[:, 0], nz_sources[:, zi + 1]), ia_bias=None
    )  # fmt: skip
    for zi in range(zbins)
]
gc_ker = [
    ccl.NumberCountsTracer(
        cosmo=cosmo,
        has_rsd=False,
        dndz=(nz_lenses[:, 0], nz_lenses[:, zi + 1]),
        bias=(bias_2d[:, 0], bias_2d[:, zi + 1]),
    )
    for zi in range(zbins)
]

# plot as a function of comoving distance (just because it's quicker)
# for zi in range(zbins):
#     plt.plot(gc_ker[zi].get_kernel()[1][0], gc_ker[zi].get_kernel()[0][0])
# for zi in range(zbins):
#     plt.plot(wl_ker[zi].get_kernel()[1][0], wl_ker[zi].get_kernel()[0][0])

cl_gg_3d = np.zeros((nbl, zbins, zbins))
cl_gl_3d = np.zeros((nbl, zbins, zbins))
cl_ll_3d = np.zeros((nbl, zbins, zbins))
print('Computing Cls...')
for zi in tqdm(range(zbins)):
    for zj in range(zbins):
        cl_gg_3d[:, zi, zj] = ccl.angular_cl(  # fmt: skip
            cosmo, gc_ker[zi], gc_ker[zj], ell_values,
            limber_integration_method='spline'
        )  # fmt: skip
        cl_gl_3d[:, zi, zj] = ccl.angular_cl(  # fmt: skip
            cosmo, gc_ker[zi], wl_ker[zj], ell_values,
            limber_integration_method='spline'
        )  # fmt: skip
        cl_ll_3d[:, zi, zj] = ccl.angular_cl(  # fmt: skip
            cosmo, wl_ker[zi], wl_ker[zj], ell_values,
            limber_integration_method='spline'
        )  # fmt: skip

cl_5d = np.zeros((n_probes_hs, n_probes_hs, len(ell_values), zbins, zbins))
cl_5d[0, 0, ...] = cl_ll_3d
cl_5d[1, 0, ...] = cl_gl_3d
cl_5d[0, 1, ...] = cl_gl_3d.transpose(0, 2, 1)
cl_5d[1, 1, ...] = cl_gg_3d

for probe, term in itertools.product(probes_toloop, terms_toloop):
    print(
        f'\n***** probe {probe} - term {term} - '
        f'integration {integration_method} - theta bins {nbt} *****\n'
    )

    # TODO check I'm not messing up anything here...
    split_g_ix = split_g_dict[term] if term in ['sva', 'sn', 'mix'] else 0

    twoprobe_ab_str, twoprobe_cd_str = split_probe_name(probe)
    twoprobe_ab_ix, twoprobe_cd_ix = (
        probe_idx_dict_short[twoprobe_ab_str],
        probe_idx_dict_short[twoprobe_cd_str],
    )

    mu, nu = mu_dict[twoprobe_ab_str], mu_dict[twoprobe_cd_str]
    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = probe_idx_dict[probe]

    # TODO test this better, especially for cross-terms
    # TODO off-diagonal zij blocks still don't match, I think it's just a
    ind_ab = ind_auto[:, 2:] if probe_a_ix == probe_b_ix else ind_cross[:, 2:]
    ind_cd = ind_auto[:, 2:] if probe_c_ix == probe_d_ix else ind_cross[:, 2:]

    zpairs_ab = zpairs_auto if probe_a_ix == probe_b_ix else zpairs_cross
    zpairs_cd = zpairs_auto if probe_c_ix == probe_d_ix else zpairs_cross

    # jsut a sanity check
    assert zpairs_ab == ind_ab.shape[0], 'zpairs-ind inconsistency'
    assert zpairs_cd == ind_cd.shape[0], 'zpairs-ind inconsistency'

    # Compute covariance:
    cov_sva_sb_6d = np.zeros((nbt, nbt, zbins, zbins, zbins, zbins))
    cov_sn_sb_6d = np.zeros((nbt, nbt, zbins, zbins, zbins, zbins))
    cov_mix_sb_6d = np.zeros((nbt, nbt, zbins, zbins, zbins, zbins))
    cov_gfromsva_sb_6d = np.zeros((nbt, nbt, zbins, zbins, zbins, zbins))

    # ! LEVIN SVA, to be tidied up

    if term == 'sva' and integration_method in ['simps', 'quad']:
        print('Computing real-space Gaussian SVA covariance...')
        start = time.time()

        kwargs = {
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
            'cl_5d': cl_5d,
            'ell_values': ell_values,
            'Amax': Amax,
        }
        results = Parallel(n_jobs=n_jobs)(  # fmt: skip
            delayed(cov_parallel_helper)(
                theta_1_ix=theta_1_ix, theta_2_ix=theta_2_ix, mu=mu, nu=nu,
                zij=zij, zkl=zkl, ind_ab=ind_ab, ind_cd=ind_cd,
                func=cov_sva_rs,
                **kwargs,
            )
            for theta_1_ix in tqdm(range(nbt))
            for theta_2_ix in range(nbt)
            for zij in range(zpairs_ab)
            for zkl in range(zpairs_cd)
        )  # fmt: skip

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_sva_sb_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value

        print(f'... Done in: {(time.time() - start):.2f} s')

    elif term == 'sva' and integration_method == 'levin':
        start = time.perf_counter()

        a = np.einsum(
            'Lik,Ljl->Lijkl',
            cl_5d[probe_a_ix, probe_c_ix],
            cl_5d[probe_b_ix, probe_d_ix],
        )
        b = np.einsum(
            'Lil,Ljk->Lijkl',
            cl_5d[probe_a_ix, probe_d_ix],
            cl_5d[probe_b_ix, probe_c_ix],
        )
        integrand = a + b

        # remove repeated zi, zj combinations
        integrand = sl.cov_6D_to_4D_blocks(
            cov_6D=integrand,
            nbl=nbl,
            npairs_AB=zpairs_ab,
            npairs_CD=zpairs_cd,
            ind_AB=ind_ab,
            ind_CD=ind_cd,
        )

        # flatten the integrand to [ells, whatever]
        integrand = integrand.reshape(nbl, -1)
        integrand *= ell_values[:, None]
        integrand /= 2.0 * np.pi * Amax

        result_levin = integrate_bessel_double_wrapper(
            integrand,
            x_values=ell_values,
            bessel_args=theta_centers,
            bessel_type=3,
            ell_1=mu,
            ell_2=nu,
        )

        print(f'... Done in: {(time.perf_counter() - start):.2f} s')

        cov_sva_sb_4d = result_levin.reshape(nbt, nbt, zpairs_ab, zpairs_cd)
        cov_sva_sb_6d = sl.cov_4D_to_6D_blocks(
            cov_sva_sb_4d,
            nbl=nbt,
            zbins=zbins,
            ind_ab=ind_ab,
            ind_cd=ind_cd,
            symmetrize_output_ab=False,
            symmetrize_output_cd=False,
        )

    elif term == 'sn':
        print('Computing real-space Gaussian SN covariance...')
        start = time.time()

        cov_sn_sb_6d = cov_sn_rs(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu)
        print(f'... Done in: {(time.time() - start):.2f} s')

    elif term == 'mix' and integration_method == 'simps':
        start = time.time()

        kwargs = {
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
            'cl_5d': cl_5d,
            'ell_values': ell_values,
            'Amax': Amax,
            'integration_method': integration_method,
        }
        results = Parallel(n_jobs=n_jobs)(  # fmt: skip
            delayed(cov_parallel_helper)(
                theta_1_ix=theta_1_ix, theta_2_ix=theta_2_ix, mu=mu, nu=nu,
                zij=zij, zkl=zkl, ind_ab=ind_ab, ind_cd=ind_cd, func=cov_g_mix_real_new,
                **kwargs,
            )
            for theta_1_ix in tqdm(range(nbt))
            for theta_2_ix in range(nbt)
            for zij in range(zpairs_ab)
            for zkl in range(zpairs_cd)
        )  # fmt: skip

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_mix_sb_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value
        print(f'... Done in: {(time.time() - start):.2f} s')

    elif term == 'mix' and integration_method == 'levin':
        start = time.time()

        def _get_mix_prefac(probe_b_ix, probe_d_ix, zj, zl):
            prefac = (
                get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zl]
                * t_mix(probe_b_ix, zbins, sigma_eps_i)[zj]
                / (n_eff_2d[probe_b_ix, zj] * srtoarcmin2)
            )
            return prefac

        prefac = np.zeros((n_probes_hs, n_probes_hs, zbins, zbins))
        for _probe_a_ix in range(n_probes_hs):
            for _probe_b_ix in range(n_probes_hs):
                for _zi in range(zbins):
                    for _zj in range(zbins):
                        prefac[_probe_a_ix, _probe_b_ix, _zi, _zj] = _get_mix_prefac(
                            _probe_a_ix, _probe_b_ix, _zi, _zj
                        )

        a = np.einsum(
            'jl,Lik->Lijkl',
            prefac[probe_b_ix, probe_d_ix],
            cl_5d[probe_a_ix, probe_c_ix],
        )
        b = np.einsum(
            'ik,Ljl->Lijkl',
            prefac[probe_a_ix, probe_c_ix],
            cl_5d[probe_b_ix, probe_d_ix],
        )
        c = np.einsum(
            'jk,Lil->Lijkl',
            prefac[probe_b_ix, probe_c_ix],
            cl_5d[probe_a_ix, probe_d_ix],
        )
        d = np.einsum(
            'il,Ljk->Lijkl',
            prefac[probe_a_ix, probe_d_ix],
            cl_5d[probe_b_ix, probe_c_ix],
        )
        integrand_5d = a + b + c + d

        # compress integrand selecting only unique zpairs
        assert ind_ab.shape[1] == 2, (
            "ind_ab must have two columns, maybe you didn't cut it"
        )
        assert ind_cd.shape[1] == 2, (
            "ind_cd must have two columns, maybe you didn't cut it"
        )

        integrand_3d = sl.cov_6D_to_4D_blocks(
            cov_6D=integrand_5d,
            nbl=nbl,
            npairs_AB=zpairs_ab,
            npairs_CD=zpairs_cd,
            ind_AB=ind_ab,
            ind_CD=ind_cd,
        )

        integrand_2d = integrand_3d.reshape(nbl, -1)
        integrand_2d *= ell_values[:, None]
        integrand_2d /= 2 * np.pi * Amax

        result_levin = integrate_bessel_double_wrapper(
            integrand_2d,
            x_values=ell_values,
            bessel_args=theta_centers,
            bessel_type=3,
            ell_1=mu,
            ell_2=nu,
        )
        print(f'... Done in: {(time.time() - start):.2f} s')

        cov_mix_sb_4d = result_levin.reshape(nbt, nbt, zpairs_ab, zpairs_cd)
        cov_mix_sb_6d = sl.cov_4D_to_6D_blocks(
            cov_mix_sb_4d,
            nbl=nbt,
            zbins=zbins,
            ind_ab=ind_ab,
            ind_cd=ind_cd,
            symmetrize_output_ab=False,
            symmetrize_output_cd=False,
        )

    elif term == 'gauss_ell':
        print('Projecting ell-space Gaussian covariance...')
        start = time.time()

        # compute ell-space G cov al volo
        # build noise vector
        noise_3x2pt_4d = sl.build_noise(
            zbins,
            n_probes=n_probes_hs,
            sigma_eps2=(sigma_eps_i * np.sqrt(2)) ** 2,
            ng_shear=n_eff_src,
            ng_clust=n_eff_lens,
        )

        # expand the noise array along the ell axis
        noise_5d = np.repeat(noise_3x2pt_4d[:, :, None, :, :], nbl, axis=2)

        # ! no delta_ell!!
        delta_ell = np.ones_like(ell_values + 1)

        # with sl.timer('covariance_einsum_v3 %.3f s '):
        cov_sva_sb_hs_10d, cov_sn_sb_hs_10d, cov_mix_sb_hs_10d = sl.covariance_einsum(
            cl_5d,
            noise_5d,
            fsky,
            ell_values,
            delta_ell,
            split_terms=True,
            return_only_diagonal_ells=True,
        )

        covs_oc_hs_npz = np.load(f'{covs_oc_path}/covs_oc_10D.npz')
        cov_sva_oc_hs_10d = covs_oc_hs_npz['cov_sva_oc_3x2pt_10D']
        cov_mix_oc_hs_10d = covs_oc_hs_npz['cov_mix_oc_3x2pt_10D']
        cov_sn_oc_hs_10d = covs_oc_hs_npz['cov_sn_oc_3x2pt_10D']
        cov_g_oc_hs_10d = covs_oc_hs_npz['cov_g_oc_3x2pt_10D']

        cov_sn_sb_6d = cov_sn_rs(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu)

        cov_g_hs_6d = (
            cov_sva_sb_hs_10d[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix]
            + cov_mix_sb_hs_10d[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix]
        )

        cov_g_sb_6d = integrate_bessel_double_wrapper(
            integrand=cov_g_hs_6d.reshape(nbl, -1)
            * ell_values[:, None]
            * ell_values[:, None],
            x_values=ell_values,
            bessel_args=theta_centers,
            bessel_type=3,
            ell_1=mu,
            ell_2=nu,
        )
        cov_g_sb_6d = cov_g_sb_6d.reshape(nbt, nbt, zbins, zbins, zbins, zbins)

        norm = 4 * np.pi**2
        cov_g_sb_6d /= norm
        cov_g_sb_6d += (
            cov_sn_sb_6d  # diagonal is noise-dominated, you won't see much of a diff
        )

    elif term in ['ssc', 'cng']:
        warnings.warn('HS covs loaded from file', stacklevel=2)
        # get OC SSC in ell space
        # covs_oc_hs = oc_cov_list_to_array(f'{covs_path}/{cov_hs_list_name}.dat')
        # (
        #     cov_sva_oc_3x2pt_10D,
        #     cov_mix_oc_3x2pt_10D,
        #     cov_sn_oc_3x2pt_10D,
        #     cov_g_oc_3x2pt_10D,
        #     cov_ssc_oc_3x2pt_10D,
        # ) = covs_oc_hs

        # np.savez(
        #     f'{covs_path}/covs_oc_10D.npz',
        #     cov_sva_oc_3x2pt_10D=cov_sva_oc_3x2pt_10D,
        #     cov_mix_oc_3x2pt_10D=cov_mix_oc_3x2pt_10D,
        #     cov_sn_oc_3x2pt_10D=cov_sn_oc_3x2pt_10D,
        #     cov_g_oc_3x2pt_10D=cov_g_oc_3x2pt_10D,
        #     cov_ssc_oc_3x2pt_10D=cov_ssc_oc_3x2pt_10D,
        # )

        covs_oc_hs_npz = np.load(f'{covs_oc_path}/covs_oc_10D.npz')
        cov_ssc_oc_3x2pt_10d = covs_oc_hs_npz['cov_ssc_oc_3x2pt_10D']
        cov_cng_oc_3x2pt_10d = covs_oc_hs_npz['cov_ng_oc_3x2pt_10D']

        if term == 'ssc':
            cov_ng_oc_3x2pt_10d = cov_ssc_oc_3x2pt_10d
            norm = 4 * np.pi**2
        elif term == 'cng':
            cov_ng_oc_3x2pt_10d = cov_cng_oc_3x2pt_10d
            norm = 4 * np.pi**2 * Amax

        # project it to real space using Levin
        cov_ng_oc_hs_6d = cov_ng_oc_3x2pt_10d[
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...
        ]

        cov_ng_sb_6d = dl1dl2_bessel_wrapper(
            cov_hs=cov_ng_oc_hs_6d,
            mu=mu,
            nu=nu,
            ells=ell_values,
            thetas=theta_centers,
            n_jobs=n_jobs_lv,
        )
        cov_ng_sb_6d /= norm  # TODO Amax still missing

    # ! ======================================= ONECOVARIANCE ==========================
    oc_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test'
    cl_ll_ascii_filename = 'Cell_ll_realsp'
    cl_gl_ascii_filename = 'Cell_gl_realsp'
    cl_gg_ascii_filename = 'Cell_gg_realsp'
    sl.write_cl_ascii(oc_path, cl_ll_ascii_filename, cl_ll_3d, ell_values, zbins)
    sl.write_cl_ascii(oc_path, cl_gl_ascii_filename, cl_gl_3d, ell_values, zbins)
    sl.write_cl_ascii(oc_path, cl_gg_ascii_filename, cl_gg_3d, ell_values, zbins)

    # set df column names
    with open(f'{oc_path}/{cov_list_name}.dat') as file:
        header = (
            file.readline().strip()
        )  # Read the first line and strip newline characters
    header_list = re.split(
        '\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t')
    )
    column_names = header_list

    data = pd.read_csv(
        f'{oc_path}/{cov_list_name}.dat', usecols=['theta1', 'tomoi'], sep='\s+'
    )

    thetas_oc_load = data['theta1'].unique()
    thetas_oc_load_rad = np.deg2rad(thetas_oc_load / 60)
    cov_theta_indices = {theta_out: idx for idx, theta_out in enumerate(thetas_oc_load)}
    nbt_oc = len(thetas_oc_load)

    # sl.compare_funcs(
    #     None,
    #     {'thetas_oc': thetas_oc_load_rad,
    #      'thetas_sb': theta_centers},
    #     plt_kw={'marker': '.'},
    # )

    # SB tomographic indices start from 0
    tomoi_oc_load = data['tomoi'].unique()
    subtract_one = False
    if min(tomoi_oc_load) == 1:
        subtract_one = True

    # ! import .list covariance file
    shape = (n_probes_rs, n_probes_rs, nbt_oc, nbt_oc, zbins, zbins, zbins, zbins)
    cov_g_oc_3x2pt_8d = np.zeros(shape)
    cov_sva_oc_3x2pt_8d = np.zeros(shape)
    cov_mix_oc_3x2pt_8d = np.zeros(shape)
    cov_sn_oc_3x2pt_8d = np.zeros(shape)
    cov_ssc_oc_3x2pt_8d = np.zeros(shape)
    cov_cng_oc_3x2pt_8d = np.zeros(shape)
    # cov_tot_oc_3x2pt_8D = np.zeros(shape)

    print(f'Loading OneCovariance output from {cov_list_name}.dat file...')
    start = time.perf_counter()
    for df_chunk in pd.read_csv(
        f'{oc_path}/{cov_list_name}.dat',
        sep='\s+',
        names=column_names,
        skiprows=1,
        chunksize=df_chunk_size,
    ):
        # Vectorize the extraction of probe indices
        probe_idx = df_chunk['#obs'].str[:].map(probe_idx_dict_short_oc).values
        probe_idx_arr = np.array(probe_idx.tolist())  # now shape is (N, 4)

        # Map 'ell' values to their corresponding indices
        theta1_idx = df_chunk['theta1'].map(cov_theta_indices).values
        theta2_idx = df_chunk['theta2'].map(cov_theta_indices).values

        # Compute z indices
        if subtract_one:
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values
        else:
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values

        # Vectorized assignment to the arrays
        index_tuple = (  # fmt: skip
            probe_idx_arr[:, 0], probe_idx_arr[:, 1], theta1_idx, theta2_idx,
            z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3],
        )  # fmt: skip

        cov_sva_oc_3x2pt_8d[index_tuple] = df_chunk['covg sva'].values
        cov_mix_oc_3x2pt_8d[index_tuple] = df_chunk['covg mix'].values
        cov_sn_oc_3x2pt_8d[index_tuple] = df_chunk['covg sn'].values
        cov_g_oc_3x2pt_8d[index_tuple] = (
            df_chunk['covg sva'].values
            + df_chunk['covg mix'].values
            + df_chunk['covg sn'].values
        )
        cov_ssc_oc_3x2pt_8d[index_tuple] = df_chunk['covssc'].values
        cov_cng_oc_3x2pt_8d[index_tuple] = df_chunk['covng'].values
        # cov_tot_oc_3x2pt_8D[index_tuple] = df_chunk['cov'].values

    covs_8d = [
        cov_sva_oc_3x2pt_8d,
        cov_mix_oc_3x2pt_8d,
        cov_sn_oc_3x2pt_8d,
        cov_g_oc_3x2pt_8d,
        cov_ssc_oc_3x2pt_8d,
        cov_cng_oc_3x2pt_8d,
        # cov_tot_oc_3x2pt_8D
    ]

    # for cov_8d in covs_8d:
    #     cov_8d[0, 0, 1, 1] = deepcopy(
    #         np.transpose(cov_8d[1, 1, 0, 0], (1, 0, 4, 5, 2, 3))
    #     )
    #     cov_8d[1, 0, 0, 0] = deepcopy(
    #         np.transpose(cov_8d[0, 0, 1, 0], (1, 0, 4, 5, 2, 3))
    #     )
    #     cov_8d[1, 0, 1, 1] = deepcopy(
    #         np.transpose(cov_8d[1, 1, 1, 0], (1, 0, 4, 5, 2, 3))
    #     )

    # ! ================================================================================

    if term == 'sva':
        cov_oc_6d = cov_sva_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_sva_sb_6d
    elif term == 'sn':
        cov_oc_6d = cov_sn_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_sn_sb_6d
    elif term == 'mix':
        cov_oc_6d = cov_mix_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_mix_sb_6d
    elif term == 'gauss_ell':
        cov_oc_6d = cov_g_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_g_sb_6d
    elif term == 'ssc':
        cov_oc_6d = cov_ssc_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_ng_sb_6d
    elif term == 'ssc':
        cov_oc_6d = cov_cng_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        cov_cng_6d = cov_ng_sb_6d

    # for probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix in itertools.product(
    #     range(n_probes), repeat=4
    # ):
    #     if np.allclose(
    #         cov_mix_oc_3x2pt_10D[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...],
    #         0,
    #         atol=1e-20,
    #         rtol=1e-10,
    #     ):
    #         print(
    #             f'block {probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix} cov_oc_6d is zero'
    #         )
    #     else:
    #         print(
    #             f'block {probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix}  cov_oc_6d is not zero'
    #         )

    # ! bin sb cov 2d
    if nbt_coarse != nbt:
        cov_sb_6d_binned = np.zeros(
            (nbt_coarse, nbt_coarse, zbins, zbins, zbins, zbins)
        )
        zijkl_comb = itertools.product(range(zbins), repeat=4)
        for zi, zj, zk, zl in zijkl_comb:
            cov_sb_6d_binned[:, :, zi, zj, zk, zl] = sl.bin_2d_array(
                cov_sb_6d[:, :, zi, zj, zk, zl],
                theta_centers,
                theta_centers_coarse,
                theta_edges_coarse,
                weights_in=None,
                which_binning='integral',
                interpolate=True,
            )

        cov_sb_6d = cov_sb_6d_binned

    cov_sb_dict_8d[split_g_ix, twoprobe_ab_ix, twoprobe_cd_ix, ...] = cov_sb_6d

    cov_oc_4d = sl.cov_6D_to_4D_blocks(
        cov_oc_6d, nbt_oc, zpairs_ab, zpairs_cd, ind_ab, ind_cd
    )
    cov_sb_4d = sl.cov_6D_to_4D_blocks(
        cov_sb_6d, nbt_coarse, zpairs_ab, zpairs_cd, ind_ab, ind_cd
    )
    # cov_sb_vec_4d = sl.cov_6D_to_4D(cov_sb_vec_6d, theta_bins, zpairs_auto, ind_auto)
    # cov_sb_gfromsva_4d = sl.cov_6D_to_4D(cov_sb_gfromsva_6d,
    # theta_bins, zpairs_auto, ind_auto)

    cov_oc_2d = sl.cov_4D_to_2D(cov_oc_4d, block_index='zpair', optimize=True)
    cov_sb_2d = sl.cov_4D_to_2D(cov_sb_4d, block_index='zpair', optimize=True)
    # cov_sb_vec_2d = sl.cov_4D_to_2D(cov_sb_vec_4d, block_index='zpair')
    # cov_sb_gfromsva_2d = sl.cov_4D_to_2D(cov_sb_gfromsva_4d, block_index='zpair')

    if probe in ['gmxip', 'gmxim']:
        warnings.warn('!!! TRANSPOSING OC COV!!!!!', stacklevel=2)
        cov_oc_2d = cov_oc_2d.T

    sl.compare_arrays(
        cov_sb_2d,
        cov_oc_2d,
        'cov_sb_2d',
        'cov_oc_2d',
        log_diff=True,
        abs_val=True,
        plot_diff_threshold=10,
        plot_diff_hist=False,
    )

    fine_bin_str = 'coarse' if nbt_coarse == nbt else 'fine'
    common_title = f'{term}, {probe}, {integration_method} theta_bins {nbt}'

    # compare total diag
    # if cov_oc_2d.shape[0] == cov_oc_2d.shape[1]:
    sl.compare_funcs(
        None,
        {
            'OC': np.abs(np.diag(cov_oc_2d)),
            'SB': np.abs(np.diag(cov_sb_2d)),
            # 'SB/OC': np.abs(np.diag(cov_sb_2d / cov_oc_2d)),
            #  'SB_split_sum': np.abs(np.diag(cov_sb_vec_2d)),  # TODO
            #  'SB_fromsva': np.abs(np.diag(cov_sb_gfromsva_2d)),
            #  'OC_SUM': np.abs(np.diag(cov_oc_sum_2d)),
        },
        logscale_y=[True, False],
        ylim_diff=[-110, 110],
        title=f'{common_title}, total cov diag',
    )
    # plt.savefig(f'{common_title}, total cov diag.png')

    # compare flattened matrix
    sl.compare_funcs(
        None,
        {
            'OC': np.abs(cov_oc_2d.flatten()),
            'SB': np.abs(cov_sb_2d.flatten()),
            # 'SB/OC': np.abs(cov_sb_2d.flatten()) / np.abs(cov_oc_2d.flatten()),
            #  'SB_VEC': np.abs(cov_sb_vec_2d.flatten()),
            #  'SB_fromsva': np.abs(cov_sb_gfromsva_2d.flatten()),
        },
        logscale_y=[True, False],
        title=f'{common_title}, total cov flat',
        ylim_diff=[-110, 110],
    )
    # plt.savefig(f'{common_title}, total cov flat.png')

    zi, zj, zk, zl = 0, 0, 0, 0
    theta_2_ix = 17
    sl.compare_funcs(
        None,
        {
            'OC': np.abs(cov_oc_6d[:, theta_2_ix, zi, zj, zk, zl]),
            'SB': np.abs(cov_sb_6d[:, theta_2_ix, zi, zj, zk, zl]),
            #  'SB_VEC': np.abs(cov_sb_vec_2d.flatten()),
            #  'SB_fromsva': np.abs(cov_sb_gfromsva_2d.flatten()),
        },
        logscale_y=[False, False],
        title=f'{term}, {probe}, {integration_method}, cov_6d[:, {zi, zj, zk, zl}]',
        ylim_diff=[-110, 110],
    )
    # plt.savefig(f'{term}_{probe}_total_cov_flat.png')

    # plt.figure()
    # plt.plot(
    #     theta_centers, np.diag(cov_sb_6d[:, :, zi, zj, zk, zl]), marker='.',
    #     label='sb'
    # )
    # plt.plot(
    #     thetas_oc_load_rad,
    #     np.diag(cov_oc_sva_6d[:, :, zi, zj, zk, zl]),
    #     marker='.',
    #     label='oc',
    # )
    # plt.xlabel(r'$\theta$ [rad]')
    # plt.ylabel(f'diag cov {probe}')
    # plt.legend()

    # TODO double check ngal, it's totally random at the moment; same for sigma_eps
    # TODO other probes
    # TODO probably ell range as well
    # TODO integration? quad?

# ! construct full 2D cov and compare correlation matrix
cov_sb_2d_dict = {}
cov_oc_2d_dict = {}
cov_sb_full_2d = []
for term in terms_toloop:
    for probe in probe_idx_dict:
        split_g_ix = split_g_dict[term] if term in ['sva', 'sn', 'mix'] else 0

        term_oc = 'gauss' if (len(terms_toloop) > 1 or term == 'gauss_ell') else term

        twoprobe_ab_str, twoprobe_cd_str = split_probe_name(probe)
        twoprobe_ab_ix, twoprobe_cd_ix = (
            probe_idx_dict_short[twoprobe_ab_str],
            probe_idx_dict_short[twoprobe_cd_str],
        )

        zpairs_ab = zpairs_cross if twoprobe_ab_ix == 1 else zpairs_auto
        zpairs_cd = zpairs_cross if twoprobe_cd_ix == 1 else zpairs_auto
        ind_ab = ind_cross if twoprobe_ab_ix == 1 else ind_auto
        ind_cd = ind_cross if twoprobe_cd_ix == 1 else ind_auto

        if term_oc == 'sva':
            cov_oc_6d = cov_sva_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        elif term_oc == 'sn':
            cov_oc_6d = cov_sn_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        elif term_oc == 'mix':
            cov_oc_6d = cov_mix_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        elif term_oc == 'gauss':
            cov_oc_6d = cov_g_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        elif term_oc == 'ssc':
            cov_oc_6d = cov_ssc_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        elif term_oc == 'ssc':
            cov_oc_6d = cov_cng_oc_3x2pt_8d[*probe_idx_dict_short_oc[probe], ...]
        else:
            raise ValueError(f'Unknown term {term_oc}')

        warnings.warn('I am manually transposing the OC blocks!!', stacklevel=2)
        if probe in ['gmxip', 'gmxim']:
            cov_oc_6d = cov_oc_6d.transpose(1, 0, 3, 2, 5, 4)

        cov_sb_4d = sl.cov_6D_to_4D_blocks(
            cov_sb_dict_8d[split_g_ix, twoprobe_ab_ix, twoprobe_cd_ix],
            nbt_coarse,
            zpairs_ab,
            zpairs_cd,
            ind_ab,
            ind_cd,
        )
        cov_oc_4d = sl.cov_6D_to_4D_blocks(
            cov_oc_6d,
            nbt_coarse,
            zpairs_ab,
            zpairs_cd,
            ind_ab,
            ind_cd,
        )

        cov_sb_2d_dict[probe] = sl.cov_4D_to_2D(
            cov_sb_4d, block_index='zpair', optimize=True
        )
        cov_oc_2d_dict[probe] = sl.cov_4D_to_2D(
            cov_oc_4d, block_index='zpair', optimize=True
        )

    cov_sb_full_2d.append(stack_cov_blocks(cov_sb_2d_dict))

cov_sb_full_2d = np.sum(np.array(cov_sb_full_2d), axis=0)
cov_oc_list_2d = stack_cov_blocks(cov_oc_2d_dict)

corr_sb_full_2d = sl.cov2corr(cov_sb_full_2d)
corr_oc_full_2d = sl.cov2corr(cov_oc_list_2d)

# sl.plot_correlation_matrix(sl.cov2corr(corr_sb_full_2d))
# sl.plot_correlation_matrix(sl.cov2corr(corr_oc_full_2d))
sl.compare_arrays(
    cov_sb_full_2d,
    cov_oc_list_2d,
    'cov SB',
    'cov OC',
    log_diff=True,
    plot_diff_threshold=10,
)

# ! this file has been overwritten with the ellspace cov
# compare G tot against OC
if term == 'ssc':
    string = 'SSC'
elif term == 'cng':
    string = 'NG'
elif term in ['gauss_ell', 'sva', 'mix']:
    string = 'gauss'
    warnings.warn(
        'Comparin against the whole Gauss .mat file. You requested', stacklevel=2
    )

cov_oc_mat = np.genfromtxt(
    '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests'
    f'/realspace_test/covariance_matrix_3x2_rcf_v2_{string}.mat'
)

sl.compare_arrays(
    cov_sb_full_2d, cov_oc_mat, 'SB', 'OC mat', log_diff=False, plot_diff_threshold=10
)


sl.compare_funcs(
    None,
    {'cov_sb_full_2d': np.diag(cov_sb_full_2d), 'cov_oc_mat': np.diag(cov_oc_mat)},
    logscale_y=[True, False],
)
plt.suptitle(f'{term}, {integration_method} - total cov diag')
plt.xlabel('cov idx')
# plt.ylabel('diag cov')
# TODO study funcs below and adapt to real space,
# current solution (see above) is a bit messy
# cov_3x2pt_10D_to_4D cov_3x2pt_8D_dict_to_4D


eig_sb = np.linalg.eigvals(cov_sb_full_2d)
eig_oc_list = np.linalg.eigvals(cov_oc_list_2d)
eig_oc_mat = np.linalg.eigvals(cov_oc_mat)

eig_threshold = 4e-15
regularise = True


plt.figure()
plt.semilogy(eig_sb, label='SB')
# plt.semilogy(eig_oc_list, label='OC list')
plt.semilogy(eig_oc_mat, label='OC mat')
plt.axhline(eig_threshold, c='k', ls='--', label='threshold')
plt.legend()
plt.xlabel('eigenvalue index')
plt.title('eigenvalues')

# perform a chi2 test
print('Computing 2PCF...')

xip_3d = np.zeros((nbt, zbins, zbins))
xim_3d = np.zeros((nbt, zbins, zbins))
w_3d = np.zeros((nbt, zbins, zbins))
gammat_3d = np.zeros((nbt, zbins, zbins))

zij_comb = itertools.product(range(zbins), repeat=2)
for zi, zj in zij_comb:
    xip_3d[:, zi, zj] = twopcf_wrapper(
        zi, zj, ell_values, theta_centers, cl_ll_3d, 'GG+', tpcf_ingr_method
    )
    xim_3d[:, zi, zj] = twopcf_wrapper(
        zi, zj, ell_values, theta_centers, cl_ll_3d, 'GG-', tpcf_ingr_method
    )
    w_3d[:, zi, zj] = twopcf_wrapper(
        zi, zj, ell_values, theta_centers, cl_gg_3d, 'NN', tpcf_ingr_method
    )
    gammat_3d[:, zi, zj] = twopcf_wrapper(
        zi, zj, ell_values, theta_centers, cl_gl_3d, 'NG', tpcf_ingr_method
    )

# flatten to construct datavector
xip_2d = sl.Cl_3D_to_2D_symmetric(xip_3d, nbt, zpairs_auto, zbins)
xim_2d = sl.Cl_3D_to_2D_symmetric(xim_3d, nbt, zpairs_auto, zbins)
w_2d = sl.Cl_3D_to_2D_symmetric(w_3d, nbt, zpairs_auto, zbins)
gammat_2d = sl.Cl_3D_to_2D_asymmetric(gammat_3d)

# the order we are using is zpair_ell, so I need to transpose
xip_1d = xip_2d.T.flatten()
xim_1d = xim_2d.T.flatten()
w_1d = w_2d.T.flatten()
gammat_1d = gammat_2d.T.flatten()

# invert covs
cov_sb_inv = np.linalg.inv(cov_sb_full_2d)
# cov_oc_list_inv = np.linalg.inv(cov_oc_list_2d)
cov_oc_mat_inv = np.linalg.inv(cov_oc_mat)

if regularise:
    # ! regularize opt. A: add small value to the diagonal of the cov
    # reg = 0.09e-14
    # cov_sb_reg_inv = np.linalg.inv(cov_sb_full_2d + reg * np.eye(cov_sb_full_2d.shape[0]))
    # cov_oc_mat_inv = np.linalg.inv(cov_oc_mat + reg * np.eye(cov_oc_mat.shape[0]))

    # ! regularize opt. B: eigenvalue cutoff
    cov_sb_inv = regularize_by_eigenvalue_cutoff(cov_sb_full_2d, eig_threshold)
    cov_oc_mat_inv = regularize_by_eigenvalue_cutoff(cov_oc_mat, eig_threshold)
    # cov_oc_mat = np.linalg.inv(cov_oc_mat_inv)

# generate samples
nreal = 5_000
dv_fid = np.hstack([w_1d, gammat_1d, xip_1d, xim_1d])
dv_sampled = np.random.multivariate_normal(dv_fid, cov_oc_mat, size=nreal)
delta_dv = dv_sampled - dv_fid  # Shape: (nreal, data_dim)

# check that the dv and cov follow the same ordering
# plt.figure()
# plt.semilogy(dv_fid, label='dv_fid')
# plt.semilogy(dv_sampled[0], label='dv_sampled', ls='--')
# plt.semilogy(np.diag(cov_oc_mat), label='cov diagonal')
# plt.semilogy(np.var(dv_sampled, axis=0), label='sample variance', ls='--')
# plt.legend()
# plt.show()

# this is another check: re-compute covariance from the samples and plot corr matrix
# cov_sampled = np.cov(dv_sampled, rowvar=False)
# sl.compare_arrays(
#     cov_sampled,
#     cov_oc_mat,
#     'cov sampled (ordering check)',
#     'cov oc mat',
#     plot_diff=False,
# )


# Check the effective dof again
dof_sb = np.trace(cov_sb_full_2d @ cov_sb_inv)
dof_oc = np.trace(cov_oc_mat @ cov_oc_mat_inv)
# dof_oc = dv_fid.shape[0]
print(f'Effective dof (SB, reg): {dof_sb}, Effective dof (OC, reg): {dof_oc}')


# compute the chi2
chi2_sb = np.einsum('ij,jk,ik->i', delta_dv, cov_sb_inv, delta_dv)
# chi2_oc_list = np.einsum('ij,jk,ik->i', delta_dv, cov_oc_list_inv, delta_dv)
chi2_oc_mat = np.einsum('ij,jk,ik->i', delta_dv, cov_oc_mat_inv, delta_dv)

# theoretical chi2 distribution
# Define the range of chi-squared values for the theoretical curve

plt.figure()
# plt.hist(chi2_oc_list, bins=20, density=True, histtype='step', label='oc_list cov')
plt.hist(chi2_oc_mat, bins=30, density=True, histtype='step', label='oc_mat cov')
plt.hist(chi2_sb, bins=30, density=True, histtype='step', label='sb cov')

x = np.linspace(dof_oc - dof_oc * 0.4, dof_oc + dof_oc * 0.4, 1000)

chi2_dist = chi2.pdf(x, df=dof_oc)
plt.plot(
    x,
    chi2_dist,
    label=f'th $\chi^2$ (eff. dof={dof_oc:.2f})',
    linestyle='--',
    c='k',
)
plt.legend()
plt.xlabel('$\chi^2$')
plt.ylabel('$p(\chi^2)$')
plt.title('Gaussian cov, regularised covs')


print('Done.')
