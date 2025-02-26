import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyccl as ccl
from joblib import Parallel, delayed
from scipy.integrate import quad_vec
from scipy.integrate import simpson as simps
from scipy.interpolate import CubicSpline
from scipy.special import jv
from tqdm import tqdm

from spaceborne import sb_lib as sl

# To run onecov to test this script, do
# conda activate spaceborne-dav
# cd OneCovariance
# python covariance.py /home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test/config_3x2pt_rcf.ini


warnings.filterwarnings(
    "ignore",
    message=r".*invalid escape sequence*",
    category=SyntaxWarning
)

warnings.filterwarnings(
    "ignore",
    message=r".*invalid value encountered in divide*",
    category=RuntimeWarning
)


def j0(x):
    return jv(0, x)


def j1(x):
    return jv(1, x)


def j2(x):
    return jv(2, x)


def b_mu(x, mu):
    """
    Implements the piecewise definition of the bracketed term b_mu(x)
    from Eq. (E.2) in Joachimi et al. (2008).
    These are just the results of \int_{\theta_l}^{\theta_u} d\theta \theta J_\mu(\ell \theta)
    """
    if mu == 0:
        return x * j1(x)
    elif mu == 2:
        return -x * j1(x) - 2.0 * j0(x)
    elif mu == 4:
        # be careful with x=0!
        return (x - 8.0 / x) * j1(x) - 8.0 * j2(x)
    else:
        raise ValueError("mu must be one of {0,2,4}.")


def k_mu(ell, thetal, thetau, mu):
    """
    Computes the kernel K_mu(ell * theta_i) in Eq. (E.2):

        K_mu(l * theta_i) = 2 / [ (theta_u^2 - theta_l^2) * l^2 ]
                            * [ b_mu(l * theta_u) - b_mu(l * theta_l) ].
    """
    prefactor = 2.0 / ((thetau**2 - thetal**2) * (ell**2))
    return prefactor * (b_mu(ell * thetau, mu) - b_mu(ell * thetal, mu))


def project_ellspace_cov_vec_2d(theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu, Amax,
                                ell1_values, ell2_values, cov_ell):
    """this version is fully vectorized"""

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


def project_ellspace_cov_vec_1d(theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu, Amax,
                                ell1_values, ell2_values, cov_ell_diag):
    """this version is vectorized anly along ell1"""

    def integrand_func(ell1, ell2, cov_ell_diag):
        # Vectorized computation of k_mu and k_nu
        kmu = k_mu(ell1, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell1, theta_2_l, theta_2_u, nu)

        # Compute the integrand
        part_product = ell1**2 * kmu * knu
        integrand = part_product[:, None, None, None, None] * cov_ell_diag
        return integrand

    # Compute the integrand for all combinations of ell1 and ell2
    integrand = integrand_func(ell1_values, ell2_values, cov_ell_diag)

    integral = simps(y=integrand, x=ell1_values, axis=0)  # axis=1?

    # Finally multiply the prefactor
    cov_elem = integral / (4.0 * np.pi ** 2 * Amax)
    return cov_elem


def project_ellspace_cov(theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu, zi, zj, zk, zl,
                         Amax,
                         ell1_values, ell2_values, cov_ell):

    # def integrand_func(ell1, ell2, cov_ell):
    #     kmu = k_mu(ell1, theta_1_l, theta_1_u, mu)
    #     knu = k_mu(ell2, theta_2_l, theta_2_u, nu)

    #     integrand = np.zeros_like(cov_ell)
    #     for ell2_ix, ell2 in enumerate(ell2_values):
    #         integrand[:, ell2_ix, zi, zj, zk, zl] = ell1 * kmu * ell2 * knu * cov_ell[:, ell2_ix, zi, zk, zk, zl]

    #     return integrand

    # # Compute the integrand for all combinations of ell1 and ell2
    # integrand = integrand_func(ell1_values, ell2_values, cov_ell)

    # part_integral = simps(y=integrand, x=ell1_values, axis=0)
    # integral = simps(y=part_integral, x=ell2_values, axis=0)  # axis=1?

    # old school:
    inner_integral = np.zeros(len(ell1_values))

    for ell1_ix, _ in enumerate(ell1_values):

        inner_integrand = ell2_values * k_mu(ell2_values, theta_2_l, theta_2_u, nu) * \
            cov_ell[ell1_ix, :, zi, zj, zk, zl]
        inner_integral[ell1_ix] = simps(y=inner_integrand, x=ell2_values)

    outer_integrand = ell1_values**2 * k_mu(ell1_values, theta_1_l, theta_1_u, mu) * inner_integral
    outer_integral = simps(y=outer_integrand, x=ell1_values)

    # Finally multiply the prefactor
    cov_elem = outer_integral / (4.0 * np.pi**2 * Amax)
    return cov_elem


def project_ellspace_cov_helper(theta_1_ix, theta_2_ix, mu, nu, zij, zkl, ind_ab, ind_cd, Amax,
                                ell1_values, ell2_values, cov_ell):

    # TODO unify helper funcs

    theta_1_l = theta_edges[theta_1_ix]
    theta_1_u = theta_edges[theta_1_ix + 1]
    theta_2_l = theta_edges[theta_2_ix]
    theta_2_u = theta_edges[theta_2_ix + 1]

    zi, zj = ind_ab[zij, :]
    zk, zl = ind_cd[zkl, :]

    return theta_1_ix, theta_2_ix, zi, zj, zk, zl, project_ellspace_cov(theta_1_l, theta_1_u, mu,
                                                                        theta_2_l, theta_2_u, nu,
                                                                        zi, zj, zk, zl,
                                                                        Amax,
                                                                        ell1_values, ell2_values, cov_ell)


def project_ellspace_cov_vec_helper(theta_1_ix, theta_2_ix, mu, nu, Amax,
                                    ell1_values, ell2_values, cov_ell):

    theta_1_l = theta_edges[theta_1_ix]
    theta_1_u = theta_edges[theta_1_ix + 1]
    theta_2_l = theta_edges[theta_2_ix]
    theta_2_u = theta_edges[theta_2_ix + 1]

    return theta_1_ix, theta_2_ix, project_ellspace_cov_vec_1d(theta_1_l, theta_1_u, mu,
                                                               theta_2_l, theta_2_u, nu,
                                                               Amax,
                                                               ell1_values, ell2_values, cov_ell)


def cov_parallel_helper(theta_1_ix, theta_2_ix, mu, nu, zij, zkl, ind_ab, ind_cd,
                        func, **kwargs):

    theta_1_l = theta_edges[theta_1_ix]
    theta_1_u = theta_edges[theta_1_ix + 1]
    theta_2_l = theta_edges[theta_2_ix]
    theta_2_u = theta_edges[theta_2_ix + 1]

    zi, zj = ind_ab[zij, :]
    zk, zl = ind_cd[zkl, :]

    return theta_1_ix, theta_2_ix, zi, zj, zk, zl, func(theta_1_l=theta_1_l,
                                                        theta_1_u=theta_1_u,
                                                        mu=mu,
                                                        theta_2_l=theta_2_l,
                                                        theta_2_u=theta_2_u,
                                                        nu=nu,
                                                        zi=zi, zj=zj, zk=zk, zl=zl,
                                                        **kwargs)


def cov_g_sva_real(theta_1_l, theta_1_u, mu,
                   theta_2_l, theta_2_u, nu,
                   zi, zj, zk, zl,
                   probe_a_ix, probe_b_ix,
                   probe_c_ix, probe_d_ix,
                   cl_5d,
                   Amax, ell_values,
                   ):
    """
    Computes a single entry of the real-space Gaussian SVA (sample variance) part of the covariance matrix.
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


def cov_g_mix_real(theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu, ell_values,
                   cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zi, zj, zk, zl,
                   Amax, integration_method='simps'):

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
        prefac = get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zn] * \
            t_mix(probe_b_ix, zbins, sigma_eps_i)[zj] /\
            (2 * np.pi * n_eff_2d[probe_b_ix, zj] * srtoarcmin2 *
             Amax)
        return prefac

    # TODO generalize to different survey areas (max(Aij, Akl))
    # TODO sigma_eps_i should be a vector of length zbins
    # permutations should be performed as done in the SVA function
    prefac_1 = get_prefac(probe_a_ix, probe_c_ix, zi, zk)
    prefac_2 = get_prefac(probe_b_ix, probe_d_ix, zj, zl)
    prefac_3 = get_prefac(probe_a_ix, probe_d_ix, zi, zl)
    prefac_4 = get_prefac(probe_b_ix, probe_c_ix, zj, zk)

    if integration_method in ['simps', 'fft']:

        # as done in the SVA function
        integrand_1 = integrand_func(ell_values, cl_5d[probe_a_ix, probe_c_ix, :, zi, zk])
        integrand_2 = integrand_func(ell_values, cl_5d[probe_b_ix, probe_d_ix, :, zj, zl])
        integrand_3 = integrand_func(ell_values, cl_5d[probe_a_ix, probe_d_ix, :, zi, zl])
        integrand_4 = integrand_func(ell_values, cl_5d[probe_b_ix, probe_c_ix, :, zj, zk])

        if integration_method == 'simps':
            integral_1 = simps(y=integrand_1, x=ell_values)
            integral_2 = simps(y=integrand_2, x=ell_values)
            integral_3 = simps(y=integrand_3, x=ell_values)
            integral_4 = simps(y=integrand_4, x=ell_values)

    elif integration_method == 'quad':

        integral_1 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
                              args=(cl_5d[probe_a_ix, probe_c_ix, :, zi, zk],))[0]
        integral_2 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
                              args=(cl_5d[probe_b_ix, probe_d_ix, :, zj, zl],))[0]
        integral_3 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
                              args=(cl_5d[probe_a_ix, probe_d_ix, :, zi, zl],))[0]
        integral_4 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
                              args=(cl_5d[probe_b_ix, probe_c_ix, :, zj, zk],))[0]

    else:
        raise ValueError(f'integration_method {integration_method} not recognized.')

    # TODO leverage simmetry to optimize the computation?
    # if zi == zj == zk == zl and probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix:
    #     return 4 * integral_1 * prefac_1

    return prefac_1 * integral_1 + prefac_2 * integral_2 + prefac_3 * integral_3 + prefac_4 * integral_4


def cov_g_mix_real_new(theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu, ell_values,
                       cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zi, zj, zk, zl, Amax,
                       integration_method='simps'):

    def integrand_func(ell, inner_integrand):
        kmu = k_mu(ell, theta_1_l, theta_1_u, mu)
        knu = k_mu(ell, theta_2_l, theta_2_u, nu)
        return (1 /( 2 * np.pi * Amax)) * ell * kmu * knu * inner_integrand

    def get_prefac(probe_b_ix, probe_d_ix, zj, zl):
        prefac = get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zl] * \
            t_mix(probe_b_ix, zbins, sigma_eps_i)[zj] /\
            (n_eff_2d[probe_b_ix, zj] * srtoarcmin2)
        return prefac

    # TODO generalize to different survey areas (max(Aij, Akl))
    # TODO sigma_eps_i should be a vector of length zbins
    # permutations should be performed as done in the SVA function

    if integration_method == 'simps':
        # integrand = integrand_func(ell_values,
        #                            cl_5d[probe_a_ix, probe_c_ix, :, zi, zk] *
        #                            get_prefac(probe_b_ix, probe_d_ix, zj, zl) +
                                   
        #                            cl_5d[probe_b_ix, probe_d_ix, :, zj, zl] *
        #                            get_prefac(probe_a_ix, probe_c_ix, zi, zk) +
                                   
        #                            cl_5d[probe_a_ix, probe_d_ix, :, zi, zl] *
        #                            get_prefac(probe_b_ix, probe_c_ix, zj, zk) + 
                                   
        #                            cl_5d[probe_b_ix, probe_c_ix, :, zj, zk] *
        #                            get_prefac(probe_a_ix, probe_d_ix, zi, zl) 
        #                            )
        
        integrand = integrand_func(ell_values,
                                   cl_5d[probe_a_ix, probe_c_ix, :, zi, zk] *
                                   get_prefac(probe_b_ix, probe_d_ix, zj, zl) +
                                   
                                   cl_5d[probe_b_ix, probe_d_ix, :, zj, zl] *
                                   get_prefac(probe_a_ix, probe_c_ix, zi, zk) +
                                   
                                   cl_5d[probe_a_ix, probe_d_ix, :, zi, zl] *
                                   get_prefac(probe_b_ix, probe_c_ix, zj, zk) +
                                   
                                   cl_5d[probe_b_ix, probe_c_ix, :, zj, zk] *
                                   get_prefac(probe_a_ix, probe_d_ix, zi, zl)
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

    # TODO leverage simmetry to optimize the computation?
    # if zi == zj == zk == zl and probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix:
    #     return 4 * integral_1 * prefac_1

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
        raise ValueError("mu and nu must be either 0, 2, or 4.")


def t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i):

    t_munu = np.zeros((zbins, zbins))

    for zi in range(zbins):
        for zj in range(zbins):

            # xipxip or ximxim
            if (probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 0 or
                    probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 1):
                t_munu[zi, zj] = 2 * sigma_eps_i[zi]**2 * sigma_eps_i[zj]**2

            elif probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 2:  # gggg
                t_munu[zi, zj] = 1

            elif (((probe_a_ix in [0, 1] and probe_b_ix == 2) or
                   (probe_b_ix in [0, 1] and probe_a_ix == 2)) and
                  ((probe_c_ix in [0, 1] and probe_d_ix == 2) or
                   (probe_d_ix in [0, 1] and probe_c_ix == 2))
                  ):
                t_munu[zi, zi] = sigma_eps_i[zi]**2

            else:
                t_munu[zi, zj] = 0

    return t_munu


def t_mix(probe_a_ix, zbins, sigma_eps_i):

    t_munu = np.zeros(zbins)

    # xipxip or ximxim
    if probe_a_ix == 0 or probe_a_ix == 1:
        t_munu = sigma_eps_i**2

    # gggg
    elif probe_a_ix == 2:
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
    """
    Splits a full probe name (e.g., 'gmxim') into two component probes.

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

    raise ValueError(f"Invalid probe name: {full_probe_name}. Expected two of {valid_probes} concatenated.")


def split_probe_ix(probe_ix):
    if probe_ix == 0:
        return 0, 0
    elif probe_ix == 1:
        return 0, 0
    elif probe_ix == 2:
        return 1, 0
    elif probe_ix == 3:
        return 1, 1
    else:
        raise ValueError(f"Invalid probe index: {probe_ix}. Expected 0, 1, 2, or 3.")

# ! =======================================================================================================
# ! =======================================================================================================
# ! =======================================================================================================


zbins = 3
survey_area_deg2 = 2500
deg2torad2 = (180 / np.pi)**2
srtoarcmin2 = (180 / np.pi * 60)**2
survey_area_sr = survey_area_deg2 / deg2torad2
fsky = 4 * np.pi / survey_area_sr
Amax = max((survey_area_sr, survey_area_sr))

ell_min = 1
ell_max = 100_000
nbl = 500
theta_min_arcmin = 50
theta_max_arcmin = 300
n_theta_edges = 21
n_probes = 4
df_chunk_size = 50000
cov_list_name = 'covariance_list_3x2_rcf'
triu_tril = 'triu'
row_col_major = 'row-major'  # unit: is gal/arcmin^2
n_jobs = -1  # leave one thread free?

n_eff_lens = np.array([0.6, 0.6, 0.6])
n_eff_src = np.array([0.6, 0.6, 0.6])
# TODO rerun OC with more realistic values, i.e.
# n_eff_lens = np.array([8.09216, 8.09215, 8.09215])
# n_eff_src = np.array([8.09216, 8.09215, 8.09215])
n_eff_2d = np.row_stack((n_eff_lens, n_eff_lens, n_eff_src))  # in this way the indices correspond to xip, xim, g
sigma_eps_i = np.array([0.26, 0.26, 0.26])
sigma_eps_tot = sigma_eps_i * np.sqrt(2)
munu_vals = (0, 2, 4)

term = 'sva'
probe = 'gggg'
integration_method = 'simps'


probe_idx_dict = {
    'xipxip': (0, 0, 0, 0),  # * SVA 1% ok; SN 0.1% ok; MIX ok for auto-pairs
    'xipxim': (0, 0, 1, 1),  # not SVA very good in lower left corner of 2d plot, possibly not worrysome; SN ok (0)
    'ximxim': (1, 1, 1, 1),  # * SVA 5% ok; SN 0.1% ok
    'gmgm': (2, 0, 2, 0),  # * SVA 1% ok; SN 0.1% ok
    'gmxim': (2, 0, 1, 1),  # ! SVA ok, but only if I transpse my cov; SN ok (0);
    'gmxip': (2, 0, 0, 0),  # ! SVA ok, but only if I transpse my cov; SN ok (0);
    'gggg': (2, 2, 2, 2),  # * SVA mostly ok; SN ok;
    'ggxim': (2, 2, 1, 1),  # not SVA very good in lower left corner of 2d plot, possibly not worrysome; SN ok (0);
    'gggm': (2, 2, 2, 0),  # * SVA mostly ok; SN ok (0);
    'ggxip': (2, 2, 0, 0),  # * SVA mostly ok; SN ok (0);
}

# ! THE PROBLEM IS FOR SURE IN HOW THE PROBE IDXS ARE "COMRPESSED" 
probe_idx_dict_short = {
    'xip': 0,
    'xim': 1,
    'gg': 2,  # w
    'gm': 3,  # \gamma_t
}  # TODO should I invert the indices for gg and gm? is there a smarter mapping? probably not...

mu_dict = {
    'gg': 0,
    'gm': 2,
    'xip': 0,
    'xim': 4,
}

for probe in probe_idx_dict.keys():
# for probe in (probe,):

    twoprobe_a_str, twoprobe_b_str = split_probe_name(probe)
    twoprobe_a_ix, twoprobe_b_ix = probe_idx_dict_short[twoprobe_a_str],  probe_idx_dict_short[twoprobe_b_str]
    
    mu, nu = mu_dict[twoprobe_a_str], mu_dict[twoprobe_b_str]
    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = probe_idx_dict[probe]

    theta_edges = np.linspace(theta_min_arcmin / 60, theta_max_arcmin / 60, n_theta_edges)
    theta_edges = np.deg2rad(theta_edges)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0  # TODO in principle this could be changed
    theta_bins = len(theta_centers)

    zpairs_auto, zpairs_cross, zpairs_3x2pt = sl.get_zpairs(zbins)
    ind = sl.build_full_ind(triu_tril, row_col_major, zbins)
    ind_auto = ind[:zpairs_auto, :].copy()
    ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()
    ind_dict = {('L', 'L'): ind_auto,
                ('G', 'L'): ind_cross,
                ('G', 'G'): ind_auto}

    # * basically no difference between the two recipes below! (The one above is obviously much slower)
    # ell_values = np.arange(ell_min, ell_max)
    ell_values = np.geomspace(ell_min, ell_max, nbl)

    # quick and dirty cls computation
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.67,
                          A_s=2.1e-9, n_s=0.966, m_nu=0.06, w0=-1.0, Neff=3.046,
                          extra_parameters={"camb": {"halofit_version": "mead2020_feedback",
                                                     "HMCode_logT_AGN": 7.75}})

    # bias_values = [1.1440270903053593, 1.209969007589984, 1.3354449071064036,
    #                1.4219803534945, 1.5275589801638865, 1.9149796097338934]
    # # create an array with the bias values in each column, and the first
    # bias_2d = np.tile(bias_values, reps=(len(z_nz_lenses), 1))
    # bias_2d = np.column_stack((z_nz_lenses, bias_2d))

    nz_lenses = np.genfromtxt(
        '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/output/OneCovariance/nzTab-EP03-zedMin02-zedMax25-mag245_dzshiftsTrue.ascii')
    nz_sources = np.genfromtxt(
        '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/output/OneCovariance/nzTab-EP03-zedMin02-zedMax25-mag245_dzshiftsTrue.ascii')
    bias_2d = np.genfromtxt(
        '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/output/OneCovariance/gal_bias_table.ascii')
    z_nz_lenses = nz_lenses[:, 0]
    z_nz_sources = nz_sources[:, 0]

    wl_ker = [ccl.WeakLensingTracer(cosmo=cosmo,
                                    dndz=(nz_sources[:, 0], nz_sources[:, zi + 1]),
                                    ia_bias=None,
                                    )
              for zi in range(zbins)]
    gc_ker = [ccl.NumberCountsTracer(cosmo=cosmo,
                                     has_rsd=False,
                                     dndz=(nz_lenses[:, 0], nz_lenses[:, zi + 1]),
                                     bias=(bias_2d[:, 0], bias_2d[:, zi + 1]))
              for zi in range(zbins)]

    # plot as a function of comoving distance (just because it's faster)
    # for zi in range(zbins):
    #     plt.plot(gc_ker[zi].get_kernel()[1][0], gc_ker[zi].get_kernel()[0][0])
    # for zi in range(zbins):
    #     plt.plot(wl_ker[zi].get_kernel()[1][0], wl_ker[zi].get_kernel()[0][0])

    cl_gg_3d = np.zeros((len(ell_values), zbins, zbins))
    cl_gl_3d = np.zeros((len(ell_values), zbins, zbins))
    cl_ll_3d = np.zeros((len(ell_values), zbins, zbins))
    print('Computing Cls...')
    for zi in tqdm(range(zbins)):
        for zj in range(zbins):
            cl_gg_3d[:, zi, zj] = ccl.angular_cl(cosmo, gc_ker[zi], gc_ker[zj], ell_values,
                                                 limber_integration_method='spline')
            cl_gl_3d[:, zi, zj] = ccl.angular_cl(cosmo, gc_ker[zi], wl_ker[zj], ell_values,
                                                 limber_integration_method='spline')
            cl_ll_3d[:, zi, zj] = ccl.angular_cl(cosmo, wl_ker[zi], wl_ker[zj], ell_values,
                                                 limber_integration_method='spline')

    cl_5d = np.zeros((3, 3, len(ell_values), zbins, zbins))
    cl_5d[0, 0, ...] = cl_ll_3d
    cl_5d[0, 1, ...] = cl_ll_3d
    cl_5d[0, 2, ...] = cl_gl_3d.transpose(0, 2, 1)

    cl_5d[1, 0, ...] = cl_ll_3d
    cl_5d[1, 1, ...] = cl_ll_3d
    cl_5d[1, 2, ...] = cl_gl_3d.transpose(0, 2, 1)

    cl_5d[2, 0, ...] = cl_gl_3d
    cl_5d[2, 1, ...] = cl_gl_3d
    cl_5d[2, 2, ...] = cl_gg_3d

    # TODO test this better, especially for cross-terms
    # TODO off-diagonal zij blocks still don't match, I think it's just a
    if probe_a_ix == probe_b_ix:
        ind_ab = ind_auto[:, 2:]
    else:
        ind_ab = ind_cross[:, 2:]

    if probe_c_ix == probe_d_ix:
        ind_cd = ind_auto[:, 2:]
    else:
        ind_cd = ind_cross[:, 2:]

    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    # Compute covariance:
    cov_sb_sva_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_sb_sn_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_sb_mix_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_sb_g_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_sb_g_vec_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_sb_gfromsva_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))

    if term == 'sva':

        print('Computing real-space Gaussian SVA covariance...')
        start = time.time()

        kwargs = {
            'func': cov_g_sva_real,
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
            'cl_5d': cl_5d,
            'ell_values': ell_values,
            'Amax': Amax,
        }
        results = Parallel(n_jobs=n_jobs)(delayed(cov_parallel_helper)(theta_1_ix=theta_1_ix,
                                                                       theta_2_ix=theta_2_ix,
                                                                       mu=mu, nu=nu,
                                                                       zij=zij, zkl=zkl,
                                                                       ind_ab=ind_ab,
                                                                       ind_cd=ind_cd,
                                                                       **kwargs
                                                                       )
                                          for theta_1_ix in tqdm(range(theta_bins))
                                          for theta_2_ix in range(theta_bins)
                                          for zij in range(zpairs_ab)
                                          for zkl in range(zpairs_cd)
                                          )

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_sb_sva_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value

        print(f'... Done in: {(time.time() - start):.2f} s')

    elif term == 'sn':
        print('Computing real-space Gaussian SN covariance...')
        start = time.time()

        # TODO generalize to different n(z)
        npair_arr = np.zeros((theta_bins, zbins, zbins))
        for theta_ix in range(theta_bins):
            for zi in range(zbins):
                for zj in range(zbins):
                    theta_1_l = theta_edges[theta_ix]
                    theta_1_u = theta_edges[theta_ix + 1]
                    npair_arr[theta_ix, zi, zj] = get_npair(theta_1_u, theta_1_l,
                                                            survey_area_sr,
                                                            n_eff_lens[zi], n_eff_lens[zj])

        delta_theta = np.eye(theta_bins)
        t_arr = t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i)

        cov_sb_sn_6d = \
            delta_theta[:, :, None, None, None, None] * \
            (get_delta_tomo(probe_a_ix, probe_c_ix)[None, None, :, None, :, None] *
             get_delta_tomo(probe_b_ix, probe_d_ix)[None, None, None, :, None, :] +
             get_delta_tomo(probe_a_ix, probe_d_ix)[None, None, :, None, None, :] *
             get_delta_tomo(probe_b_ix, probe_c_ix)[None, None, None, :, :, None]) * \
            t_arr[None, None, :, None, :, None] / \
            npair_arr[None, :, :, :, None, None]
        print(f'... Done in: {(time.time() - start):.2f} s')

    elif term == 'mix':
        print('Computing real-space Gaussian MIX covariance...')
        start = time.time()

        kwargs = {
            'func': cov_g_mix_real_new,
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
            'cl_5d': cl_5d,
            'ell_values': ell_values,
            'Amax': Amax,
            'integration_method': integration_method,
        }
        results = Parallel(n_jobs=n_jobs)(delayed(cov_parallel_helper)(theta_1_ix=theta_1_ix,
                                                                       theta_2_ix=theta_2_ix,
                                                                       mu=mu, nu=nu,
                                                                       zij=zij, zkl=zkl,
                                                                       ind_ab=ind_ab,
                                                                       ind_cd=ind_cd,
                                                                       **kwargs
                                                                       )
                                          for theta_1_ix in tqdm(range(theta_bins))
                                          for theta_2_ix in range(theta_bins)
                                          for zij in range(zpairs_ab)
                                          for zkl in range(zpairs_cd)
                                          )

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_sb_mix_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value
        print(f'... Done in: {(time.time() - start):.2f} s')

    elif term == 'gauss_ell':

        print('Projecting ell-space Gaussian covariance...')
        start = time.time()

        # compute ell-space G cov al volo
        cl_5d_2probes = np.zeros((2, 2, len(ell_values), zbins, zbins))
        cl_5d_2probes[0, 0, ...] = cl_ll_3d
        cl_5d_2probes[0, 1, ...] = cl_gl_3d.transpose(0, 2, 1)
        cl_5d_2probes[1, 0, ...] = cl_gl_3d
        cl_5d_2probes[1, 1, ...] = cl_ll_3d

        # build noise vector
        noise_3x2pt_4D = sl.build_noise(zbins, n_probes=2,
                                        sigma_eps2=(sigma_eps_i * np.sqrt(2))**2,
                                        ng_shear=n_eff_src,
                                        ng_clust=n_eff_lens)

        # create dummy ell axis, the array is just repeated along it
        noise_5d_2dprobes = np.zeros((2, 2, nbl, zbins, zbins))
        for probe_A in (0, 1):
            for probe_B in (0, 1):
                for ell_idx in range(nbl):
                    noise_5d_2dprobes[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

        noise_5d = np.zeros((2, 2, len(ell_values), zbins, zbins))
        noise_5d[0, 0, ...] = noise_5d_2dprobes[0, 0, ...]
        noise_5d[0, 1, ...] = noise_5d_2dprobes[0, 0, ...]
        noise_5d[0, 2, ...] = noise_5d_2dprobes[0, 1, ...]
        noise_5d[1, 0, ...] = noise_5d_2dprobes[0, 0, ...]
        noise_5d[1, 1, ...] = noise_5d_2dprobes[0, 0, ...]
        noise_5d[1, 2, ...] = noise_5d_2dprobes[0, 1, ...]
        noise_5d[2, 0, ...] = noise_5d_2dprobes[1, 0, ...]
        noise_5d[2, 1, ...] = noise_5d_2dprobes[1, 0, ...]
        noise_5d[2, 2, ...] = noise_5d_2dprobes[1, 1, ...]

        # ! choose between this
        delta_ell = np.diff(ell_values)
        delta_ell = np.concatenate(((delta_ell[0],), delta_ell))
        # _fsky = fsky

        # ! or this
        _fsky = 1
        delta_ell = np.ones_like(delta_ell)

        cov_ell = sl.covariance_einsum(
            cl_5d_2probes, noise_5d_2dprobes, _fsky, ell_values, delta_ell)
        cov_ell_diag = sl.covariance_einsum(
            cl_5d_2probes, noise_5d_2dprobes, _fsky, ell_values, delta_ell,
            return_only_diagonal_ells=True)

        if probe == 'gggg':
            probe_tuple_old = (1, 1, 1, 1)
        elif probe == 'xipxip':
            probe_tuple_old = (0, 0, 0, 0)
        else:
            raise ValueError('not implemented yet')

        cov_ell = cov_ell[probe_tuple_old]
        cov_ell_diag = cov_ell_diag[probe_tuple_old]

        # remove ell-dep prefactor from Gaussian cov
        cov_ell_diag *= (2 * ell_values + 1)[:, None, None, None, None]
        for ell1_ix, ell1 in enumerate(ell_values):
            cov_ell[ell1_ix, ell1_ix, ...] *= (2 * ell1 + 1)

        # TODO use just one helper function, except for project_ellspace_cov_vec_helper
        kwargs = {
            'func': project_ellspace_cov,
            'Amax': Amax,
            'ell1_values': ell_values,
            'ell2_values': ell_values,
            'cov_ell': cov_ell,
        }
        results = Parallel(n_jobs=n_jobs)(delayed(cov_parallel_helper)(theta_1_ix=theta_1_ix,
                                                                       theta_2_ix=theta_2_ix,
                                                                       mu=mu, nu=nu,
                                                                       zij=zij, zkl=zkl,
                                                                       ind_ab=ind_ab,
                                                                       ind_cd=ind_cd,
                                                                       **kwargs
                                                                       )
                                          for theta_1_ix in tqdm(range(theta_bins))
                                          for theta_2_ix in range(theta_bins)
                                          for zij in range(zpairs_ab)
                                          for zkl in range(zpairs_cd)
                                          )

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_sb_g_6d[theta_1, theta_2, :, :, :, :] = cov_value

        results = Parallel(n_jobs=n_jobs)(delayed(project_ellspace_cov_vec_helper)(theta_1, theta_2,
                                                                                   mu, nu,
                                                                                   Amax,
                                                                                   ell_values, ell_values,
                                                                                   cov_ell_diag)
                                          for theta_1 in tqdm(range(theta_bins))
                                          for theta_2 in range(theta_bins)
                                          )

        for theta_1, theta_2, cov_value in results:
            cov_sb_g_vec_6d[theta_1, theta_2, :, :, :, :] = cov_value

        # results = Parallel(n_jobs=n_jobs)(delayed(cov_g_sva_real_helper)(theta_1, theta_2, zi, zj, zk, zl,
        #                                                                  mu, nu, cl_5d + noise_5d,
        #                                                                  *probe_tuple_old,
        #                                                                  )
        #                                   for theta_1 in tqdm(range(theta_bins))
        #                                   for theta_2 in range(theta_bins)
        #                                   for zi in range(zbins)
        #                                   for zj in range(zbins)
        #                                   for zk in range(zbins)
        #                                   for zl in range(zbins)
        #                                   )

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_sb_gfromsva_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value

        print(f'... Done in: {(time.time() - start):.2f} s')

    # ! ======================================= ONECOVARIANCE ==================================================
    oc_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test'
    cl_ll_ascii_filename = 'Cell_ll_realsp'
    cl_gl_ascii_filename = 'Cell_gl_realsp'
    cl_gg_ascii_filename = 'Cell_gg_realsp'
    sl.write_cl_ascii(oc_path, cl_ll_ascii_filename, cl_ll_3d, ell_values, zbins)
    sl.write_cl_ascii(oc_path, cl_gl_ascii_filename, cl_gl_3d, ell_values, zbins)
    sl.write_cl_ascii(oc_path, cl_gg_ascii_filename, cl_gg_3d, ell_values, zbins)

    # set df column names
    with open(f'{oc_path}/{cov_list_name}.dat', 'r') as file:
        header = file.readline().strip()  # Read the first line and strip newline characters
    header_list = re.split('\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t'))
    column_names = header_list

    # ell values actually used in OC; save in self to be able to compare to the SB ell values
    # note use delim_whitespace=True instead of sep='\s+' if this gives compatibility issues
    thetas_oc_load = pd.read_csv(f'{oc_path}/{cov_list_name}.dat',
                                 usecols=['theta1'], sep='\s+')['theta1'].unique()
    thetas_oc_load_rad = np.deg2rad(thetas_oc_load / 60)

    # check if the saved ells are within 1% of the required ones; I think the saved values are truncated to only
    # 2 decimals, so this is a rough comparison (rtol is 1%)
    # try:
    #     np.testing.assert_allclose(theta_centers*60, thetas_oc_load, atol=0, rtol=1e-2)
    # except AssertionError as err:
    #     print('ell values computed vs loaded for OC are not the same')
    #     print(err)

    cov_theta_indices = {ell_out: idx for idx, ell_out in enumerate(thetas_oc_load)}

    # ! import .list covariance file
    cov_g_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
                                   theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_sva_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
                                    theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_mix_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
                                    theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_sn_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
                                    theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    cov_ssc_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
                                    theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    # cov_cng_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
    #                                  theta_bins, theta_bins, zbins, zbins, zbins, zbins))
    # cov_tot_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
    #                                  theta_bins, theta_bins, zbins, zbins, zbins, zbins))

    print(f'Loading OneCovariance output from {cov_list_name}.dat file...')
    start = time.perf_counter()
    for df_chunk in pd.read_csv(f'{oc_path}/{cov_list_name}.dat', sep='\s+', names=column_names, skiprows=1, chunksize=df_chunk_size):

        # Vectorize the extraction of probe indices
        probe_idx = df_chunk['#obs'].str[:].map(probe_idx_dict).values
        probe_idx_arr = np.array(probe_idx.tolist())  # now shape is (N, 4)

        # Map 'ell' values to their corresponding indices
        theta1_idx = df_chunk['theta1'].map(cov_theta_indices).values
        theta2_idx = df_chunk['theta2'].map(cov_theta_indices).values

        # Compute z indices
        if np.min(df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values) == 1:
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values
        else:
            warnings.warn('tomo indices seem to start from 0...')
            z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].values

        # Vectorized assignment to the arrays
        index_tuple = (probe_idx_arr[:, 0], probe_idx_arr[:, 1], probe_idx_arr[:, 2], probe_idx_arr[:, 3],
                       theta1_idx, theta2_idx,
                       z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3])

        cov_sva_oc_3x2pt_10D[index_tuple] = df_chunk['covg sva'].values
        cov_mix_oc_3x2pt_10D[index_tuple] = df_chunk['covg mix'].values
        cov_sn_oc_3x2pt_10D[index_tuple] = df_chunk['covg sn'].values
        cov_g_oc_3x2pt_10D[index_tuple] = df_chunk['covg sva'].values + \
            df_chunk['covg mix'].values + df_chunk['covg sn'].values
        cov_ssc_oc_3x2pt_10D[index_tuple] = df_chunk['covssc'].values
        # cov_cng_oc_3x2pt_10D[index_tuple] = df_chunk['covng'].values
        # cov_tot_oc_3x2pt_10D[index_tuple] = df_chunk['cov'].values

    covs_10d = [cov_sva_oc_3x2pt_10D, cov_mix_oc_3x2pt_10D, cov_sn_oc_3x2pt_10D,
                cov_g_oc_3x2pt_10D, cov_ssc_oc_3x2pt_10D,
                # cov_cng_oc_3x2pt_10D, cov_tot_oc_3x2pt_10D
                ]

    # for cov_10d in covs_10d:
    #     cov_10d[0, 0, 1, 1] = deepcopy(np.transpose(cov_10d[1, 1, 0, 0], (1, 0, 4, 5, 2, 3)))
    #     cov_10d[1, 0, 0, 0] = deepcopy(np.transpose(cov_10d[0, 0, 1, 0], (1, 0, 4, 5, 2, 3)))
    #     cov_10d[1, 0, 1, 1] = deepcopy(np.transpose(cov_10d[1, 1, 1, 0], (1, 0, 4, 5, 2, 3)))

    # ! =============================================================================================

    if term == 'sva':
        cov_oc_6d = cov_sva_oc_3x2pt_10D[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...]
        cov_sb_6d = cov_sb_sva_6d
    elif term == 'sn':
        cov_oc_6d = cov_sn_oc_3x2pt_10D[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...]
        cov_sb_6d = cov_sb_sn_6d
    elif term == 'mix':
        cov_oc_6d = cov_mix_oc_3x2pt_10D[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...]
        cov_sb_6d = cov_sb_mix_6d
    elif term == 'gauss_ell':
        cov_oc_6d = cov_g_oc_3x2pt_10D[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...]
        cov_sb_6d = cov_sb_g_6d
        cov_sb_vec_6d = cov_sb_g_vec_6d

    # for probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix in itertools.product(range(n_probes), repeat=4):
    #     if np.allclose(cov_mix_oc_3x2pt_10D[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...], 0, atol=1e-20, rtol=1e-10):
    #         print(f'block {probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix} cov_oc_6d is zero')
    #     else:
    #         print(f'block {probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix}  cov_oc_6d is not zero')

    if np.allclose(cov_sb_6d, 0, atol=1e-20, rtol=1e-10):
        print('cov_sb_6d is zero')

    cov_oc_4d = sl.cov_6D_to_4D(cov_oc_6d, theta_bins, zpairs_auto, ind_auto)
    cov_sb_4d = sl.cov_6D_to_4D(cov_sb_6d, theta_bins, zpairs_auto, ind_auto)
    # cov_sb_vec_4d = sl.cov_6D_to_4D(cov_sb_vec_6d, theta_bins, zpairs_auto, ind_auto)
    # cov_sb_gfromsva_4d = sl.cov_6D_to_4D(cov_sb_gfromsva_6d, theta_bins, zpairs_auto, ind_auto)

    cov_oc_2d = sl.cov_4D_to_2D(cov_oc_4d, block_index='zpair')
    cov_sb_2d = sl.cov_4D_to_2D(cov_sb_4d, block_index='zpair')
    # cov_sb_vec_2d = sl.cov_4D_to_2D(cov_sb_vec_4d, block_index='zpair')
    # cov_sb_gfromsva_2d = sl.cov_4D_to_2D(cov_sb_gfromsva_4d, block_index='zpair')

    sl.compare_arrays(cov_sb_2d, cov_oc_2d,
                      'cov_sb_2d', 'cov_oc_2d',
                      log_diff=True,
                      abs_val=True,
                      plot_diff_threshold=10,
                      plot_diff_hist=False)
    # plt.savefig(f'{term}_{probe}_total_covs.png')

    # sl.compare_arrays(cov_sb_2d, cov_sb_vec_2d,
    #                   'cov_sb_2d', 'cov_sb_vec_6d',
    #                   abs_val=True, plot_diff_threshold=10,
    #                   plot_diff_hist=False)
    # sl.compare_arrays(cov_sb_vec_2d, cov_oc_2d,
    #                   'cov_sb_vec_2d', 'cov_oc_2d',
    #                   abs_val=True, plot_diff_threshold=10,
    #                   plot_diff_hist=False)

    zi, zj, zk, zl = 0, 0, 0, 0

    cov_oc_spline = CubicSpline(thetas_oc_load_rad, np.diag(cov_oc_6d[:, :, zi, zj, zk, zl]))

    # comapre total diag
    sl.compare_funcs(None,
                     {
                         'OC': np.abs(np.diag(cov_oc_2d)),
                         'SB': np.abs(np.diag(cov_sb_2d)),
                         #  'SB_VEC': np.abs(np.diag(cov_sb_vec_2d)),
                         #  'SB_split_sum': np.abs(np.diag(cov_sb_vec_2d)),  # TODO
                         #  'SB_fromsva': np.abs(np.diag(cov_sb_gfromsva_2d)),
                         #  'OC_SUM': np.abs(np.diag(cov_oc_sum_2d)),
                     },
                     logscale_y=[True, False],
                     #  ylim_diff=[-20, 20],
                     title=f'{term}, {probe}, total cov diag')
    # plt.savefig(f'{term}_{probe}_total_cov_diag.png')

    # compare flattened matrix
    sl.compare_funcs(None,
                     {
                         'OC': np.abs(cov_oc_2d.flatten()),
                         'SB': np.abs(cov_sb_2d.flatten()),
                         #  'SB_VEC': np.abs(cov_sb_vec_2d.flatten()),
                         #  'SB_fromsva': np.abs(cov_sb_gfromsva_2d.flatten()),
                     },
                     logscale_y=[True, False],
                     title=f'{term}, {probe}, total cov flat',
                    #  ylim_diff=[-110, 110])
    )
    # plt.savefig(f'{term}_{probe}_total_cov_flat.png')

    # plt.figure()
    # plt.plot(theta_centers, np.diag(cov_sb_6d[:, :, zi, zj, zk, zl]), marker='.', label='sb')
    # plt.plot(thetas_oc_load_rad, np.diag(cov_oc_sva_6d[:, :, zi, zj, zk, zl]), marker='.', label='oc')
    # plt.xlabel(r'$\theta$ [rad]')
    # plt.ylabel(f'diag cov {probe}')
    # plt.legend()

    # TODO double check ngal, it's totally random at the moment; same for sigma_eps
    # TODO other probes
    # TODO probably ell range as well
    # TODO integration? quad?

print("Done.")
