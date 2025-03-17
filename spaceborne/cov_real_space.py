from copy import deepcopy
import re
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import quad_vec
from scipy.integrate import simpson as simps
from scipy.interpolate import CubicSpline
from tqdm import tqdm

import pyccl as ccl
from spaceborne import sb_lib as sl

# To run onecov to test this script, do
# conda activate spaceborne-dav
# cd OneCovariance
# python covariance.py /home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/\
# realspace_test/config_3x2pt_rcf.ini

warnings.filterwarnings(
    'ignore', message=r'.*invalid escape sequence.*', category=SyntaxWarning
)

warnings.filterwarnings(
    'ignore',
    message=r'.*invalid value encountered in divide.*',
    category=RuntimeWarning,
)


def b_mu(x, mu):
    """
    Implements the piecewise definition of the bracketed term b_mu(x)
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


def k_mu(ell, thetal, thetau, mu):
    """
    Computes the kernel K_mu(ell * theta_i) in Eq. (E.2):

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


def project_ellspace_cov_vec_1d(  # fmt: skip
    theta_1_l, theta_1_u, mu,                               
    theta_2_l, theta_2_u, nu,                               
    Amax, ell1_values, ell2_values, cov_ell_diag,           
):  # fmt: skip
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
    cov_elem = integral / (4.0 * np.pi**2 * Amax)
    return cov_elem


def project_ellspace_cov(  # fmt: skip
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
        project_ellspace_cov( 
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


def cov_g_sva_real(  # fmt: skip
    theta_1_l, theta_1_u, mu, theta_2_l, theta_2_u, nu,  
    zi, zj, zk, zl, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,  
    cl_5d, Amax, ell_values  
):  # fmt: skip
    """
    Computes a single entry of the real-space Gaussian SVA (sample variance)
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


def cov_g_mix_real(   # fmt: skip
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

    if integration_method in ['simps', 'fft']:
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

        if integration_method == 'simps':
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

    # TODO leverage simmetry to optimize the computation?
    # if zi == zj == zk == zl and probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix:
    #     return 4 * integral_1 * prefac_1

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

    def get_prefac(probe_b_ix, probe_d_ix, zj, zl):
        prefac = (
            get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zl]
            * t_mix(probe_b_ix, zbins, sigma_eps_i)[zj]
            / (n_eff_2d[probe_b_ix, zj] * srtoarcmin2)
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

    raise ValueError(
        f'Invalid probe name: {full_probe_name}. '
        f'Expected two of {valid_probes} concatenated.'
    )


def split_probe_ix(probe_ix):
    if probe_ix == 0 or probe_ix == 1:
        return 0, 0
    elif probe_ix == 2:
        return 1, 0
    elif probe_ix == 3:
        return 1, 1
    else:
        raise ValueError(f'Invalid probe index: {probe_ix}. Expected 0, 1, 2, or 3.')


def integrate_cov_levin(cov_2d, mu, ell, theta_centers, n_jobs):
    """
    cov_2d must have the first axis corresponding to the ell values),
    the second to the flattened remaining dimensions"""
    import pylevin as levin

    integral_type = 1  # single cilyndrical bessel

    integrand = ell[:, None] * cov_2d

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


def integrate_double_bessel(cov_hs, mu, nu, ells, thetas, n_jobs):
    # First integration: for each fixed ell1, integrate over ell2.
    partial_results = []
    for ell1_ix in tqdm(range(nbl)):
        # Extract the 2D slice for fixed ell1.
        cov_hs_2d = cov_hs[ell1_ix, ...].reshape(nbl, -1)
        partial_int = integrate_cov_levin(cov_hs_2d, nu, ells, thetas, n_jobs)
        partial_results.append(partial_int)

    # Stack partial results along the ell1 direction.
    partial_results = np.stack(partial_results, axis=0)

    # Second integration: integrate over ell1.
    nbt = partial_results.shape[1]
    flattened_size = partial_results.shape[2]
    final_result = np.zeros((nbt, nbt, flattened_size))

    for theta_idx in tqdm(range(nbt)):
        # For fixed theta from the first integration, extract the integrand:
        integrand_second = partial_results[:, theta_idx, :]
        final_int = integrate_cov_levin(integrand_second, mu, ells, thetas, n_jobs)
        final_result[:, theta_idx, :] = final_int

    cov_rs_6d = final_result.reshape(nbt, nbt, zbins, zbins, zbins, zbins)
    cov_rs_6d *= 1 / (4 * np.pi**2)  # TODO Amax still missing

    return cov_rs_6d


# ! ====================================================================================
# ! ====================================================================================
# ! ====================================================================================

# levin bessel settings
logx = True
logy = True
diagonal = False

# accuracy settings
n_sub = 10  # number of collocation points in each bisection
n_bisec_max = 32  # maximum number of bisections used
rel_acc = 1e-4  # relative accuracy target  # TODO decrease to 1e-5
# should the bessel functions be calculated with boost instead of GSL,
# higher accuracy at high Bessel orders
boost_bessel = True
verbose = False  # should the code talk to you?


zbins = 3
survey_area_deg2 = 2500
deg2torad2 = (180 / np.pi) ** 2
srtoarcmin2 = (180 / np.pi * 60) ** 2
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
cov_hs_list_name = 'covariance_list_3x2_cl'
triu_tril = 'triu'
row_col_major = 'row-major'  # unit: is gal/arcmin^2
n_jobs = -1  # leave one thread free?
n_jobs_lv = 8  # might cause memory issues if too high

n_eff_lens = np.array([0.6, 0.6, 0.6])
n_eff_src = np.array([0.6, 0.6, 0.6])
# TODO rerun OC with more realistic values, i.e.
# n_eff_lens = np.array([8.09216, 8.09215, 8.09215])
# n_eff_src = np.array([8.09216, 8.09215, 8.09215])

n_eff_2d = np.row_stack(
    (n_eff_lens, n_eff_lens, n_eff_src)
)  # in this way the indices correspond to xip, xim, g
sigma_eps_i = np.array([0.26, 0.26, 0.26])
sigma_eps_tot = sigma_eps_i * np.sqrt(2)
munu_vals = (0, 2, 4)
n_probes_rs = 4  # real space
n_probes_hs = 2  # harmonic space

term = 'sva'
_probe = 'xipxip'
integration_method = 'levin'

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

# TODO Should I invert the indices for gg and gm?
# TODO Is there a smarter mapping? probably not...
probe_idx_dict_short = {
    'xip': 0,
    'xim': 1,
    'gg': 2,  # w
    'gm': 3,  # \gamma_t
}


probe_idx_dict_short_oc = {}
for key in probe_idx_dict:
    probe_a_str, probe_b_str = split_probe_name(key)
    probe_idx_dict_short_oc[probe_a_str + probe_b_str] = (
        probe_idx_dict_short[probe_a_str],
        probe_idx_dict_short[probe_b_str],
    )


# for probe in probe_idx_dict:
for probe in (_probe,):
    twoprobe_a_str, twoprobe_b_str = split_probe_name(probe)
    twoprobe_a_ix, twoprobe_b_ix = (
        probe_idx_dict_short[twoprobe_a_str],
        probe_idx_dict_short[twoprobe_b_str],
    )

    mu, nu = mu_dict[twoprobe_a_str], mu_dict[twoprobe_b_str]
    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = probe_idx_dict[probe]

    theta_edges = np.linspace(
        theta_min_arcmin / 60, theta_max_arcmin / 60, n_theta_edges
    )
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
        A_s=2.1e-9,
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
    cov_g_sb_6d = np.zeros((nbt, nbt, zbins, zbins, zbins, zbins))
    cov_g_vec_sb_6d = np.zeros((nbt, nbt, zbins, zbins, zbins, zbins))
    cov_gfromsva_sb_6d = np.zeros((nbt, nbt, zbins, zbins, zbins, zbins))

    # ! LEVIN SVA, to be tidied up
    import pylevin as levin

    if term == 'sva' and integration_method == 'simps':
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
                func=cov_g_sva_real,  
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
        a = np.einsum(
            'Lik,Mjl->LMijkl',
            cl_5d[probe_a_ix, probe_c_ix],
            cl_5d[probe_b_ix, probe_d_ix],
        )
        b = np.einsum(
            'Lil,Mjk->LMijkl',
            cl_5d[probe_a_ix, probe_d_ix],
            cl_5d[probe_b_ix, probe_c_ix],
        )
        integrand = a + b
        integrand = np.diagonal(integrand, axis1=0, axis2=1).transpose(4, 0, 1, 2, 3)
        integrand = integrand.reshape(nbl, -1)

        integrand = integrand * ell_values[:, None]
        integrand /= 2.0 * np.pi * Amax

        # Constructor of the class
        lp = levin.pylevin(
            type=3,
            x=ell_values,
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

        M = nbt**2  # number of arguments at which the integrals are evaluated
        N = integrand.shape[-1]
        result_levin = np.zeros((M, N))  # allocate the result
        # result_levin = np.zeros(N,)  # allocate the result
        
        t0 = time.time()
        lp.levin_integrate_bessel_double(
            x_min=ell_values[0] * np.ones(nbt),
            x_max=ell_values[-1] * np.ones(nbt),
            k_1=theta_centers,
            k_2=theta_centers,
            ell_1=(mu * np.ones(nbt)).astype(int),  # !mu or nu, careful
            ell_2=(nu * np.ones(nbt)).astype(int),  # !mu or nu, careful
            result=result_levin,
        )

        cov_sva_sb_6d = result_levin.reshape(nbt, nbt, zbins, zbins, zbins, zbins)

        print('Levin took', time.time() - t0, 's')




def sigma2_b_levin_batched(
    z_grid: np.ndarray,
    k_grid: np.ndarray,
    cosmo_ccl: ccl.Cosmology,
    which_sigma2_b: str,
    ell_mask: np.ndarray,
    cl_mask: np.ndarray,
    fsky_mask: float,
    n_jobs: int,
    batch_size: int,
) -> np.ndarray:
    """
    Compute sigma2_b using the Levin integration method. The computation leverages the
    symmetry in z1, z2 to reduce the number of integrals
    (only the upper triangle of the z1, z2 matrix is actually computed).

    Parameters
    ----------
    z_grid : np.ndarray
        Array of redshifts.
    k_grid : np.ndarray
        Array of wavenumbers [1/Mpc].
    cosmo_ccl : ccl.Cosmology
        Cosmological parameters.
    which_sigma2_b : str
        Type of sigma2_b to compute.
    ell_mask : np.ndarray
        Array of multipoles at which the mask is evaluated.
    cl_mask : np.ndarray
        Array containing the angular power spectrum of the mask.
    fsky_mask : float
        Fraction of sky covered by the mask.
    n_jobs : int
        Number of threads to use for the computation in parallel.
    batch_size : int, optional
        Batch size for the computation. Default is 100_000.

    Returns
    -------
    np.ndarray
        2D array of sigma2_b values, of shape (len(z_grid), len(z_grid)).
    """

    import pylevin as levin

    a_arr = cosmo_lib.z_to_a(z_grid)
    r_arr = ccl.comoving_radial_distance(cosmo_ccl, a_arr)
    growth_factor_arr = ccl.growth_factor(cosmo_ccl, a_arr)
    plin = ccl.linear_matter_power(cosmo_ccl, k=k_grid, a=1.0)

    integral_type = 2  # double spherical
    N_thread = n_jobs  # Number of threads used for hyperthreading

    zsteps = len(r_arr)
    triu_ix = np.triu_indices(zsteps)
    n_upper = len(triu_ix[0])  # number of unique integrals to compute

    result_flat = np.zeros(n_upper)

    for i in tqdm(range(0, n_upper, batch_size), desc='Batches'):
        batch_indices = slice(i, i + batch_size)
        r1_batch = r_arr[triu_ix[0][batch_indices]]
        r2_batch = r_arr[triu_ix[1][batch_indices]]
        integrand_batch = (
            k_grid[:, None] ** 2
            * plin[:, None]
            * growth_factor_arr[None, triu_ix[0][batch_indices]]
            * growth_factor_arr[None, triu_ix[1][batch_indices]]
        )

        lp = levin.pylevin(
            integral_type, k_grid, integrand_batch, logx, logy, N_thread, True
        )
        lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)

        lower_limit = k_grid[0] * np.ones(len(r1_batch))
        upper_limit = k_grid[-1] * np.ones(len(r1_batch))
        ell = np.zeros(len(r1_batch), dtype=int)

        lp.levin_integrate_bessel_double(
            lower_limit,
            upper_limit,
            r1_batch,
            r2_batch,
            ell,
            ell,
            result_flat[batch_indices],
        )        

    # Assemble symmetric result matrix
    result_matrix = np.zeros((zsteps, zsteps))
    result_matrix[triu_ix] = result_flat
    result_matrix = result_matrix + result_matrix.T - np.diag(np.diag(result_matrix))

    if which_sigma2_b == 'full_curved_sky':
        result = 1 / (2 * np.pi**2) * result_matrix

    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
        partial_summand = (2 * ell_mask + 1) * cl_mask * 2 / np.pi
        partial_summand = result_matrix[:, :, None] * partial_summand[None, None, :]
        result = np.sum(partial_summand, axis=-1)
        one_over_omega_s_squared = 1 / (4 * np.pi * fsky_mask) ** 2
        result *= one_over_omega_s_squared
    else:
        raise ValueError(
            'which_sigma2_b must be either "full_curved_sky" or '
            '"polar_cap_on_the_fly" or "from_input_mask"'
        )

    return result



















    elif term == 'sn':
        print('Computing real-space Gaussian SN covariance...')
        start = time.time()

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

        delta_theta = np.eye(nbt)
        t_arr = t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i)

        cov_sn_sb_6d = (
            delta_theta[:, :, None, None, None, None]
            * (
                get_delta_tomo(probe_a_ix, probe_c_ix)[None, None, :, None, :, None]
                * get_delta_tomo(probe_b_ix, probe_d_ix)[None, None, None, :, None, :]
                + get_delta_tomo(probe_a_ix, probe_d_ix)[None, None, :, None, None, :]
                * get_delta_tomo(probe_b_ix, probe_c_ix)[None, None, None, :, :, None]
            )
            * t_arr[None, None, :, None, :, None]
            / npair_arr[None, :, :, :, None, None]
        )
        print(f'... Done in: {(time.time() - start):.2f} s')

    elif term == 'mix':
        print('Computing real-space Gaussian MIX covariance...')
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

    elif term == 'gauss_ell':
        print('Projecting ell-space Gaussian covariance...')
        start = time.time()

        # compute ell-space G cov al volo
        # build noise vector
        noise_3x2pt_4D = sl.build_noise(
            zbins,
            n_probes=n_probes_hs,
            sigma_eps2=(sigma_eps_i * np.sqrt(2)) ** 2,
            ng_shear=n_eff_src,
            ng_clust=n_eff_lens,
        )

        # create dummy ell axis, the array is just repeated along it
        noise_5d = np.zeros((n_probes_hs, n_probes_hs, nbl, zbins, zbins))
        for probe_A in (0, 1):
            for probe_B in (0, 1):
                for ell_idx in range(nbl):
                    noise_5d[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[
                        probe_A, probe_B, ...
                    ]

        # ! choose between this
        delta_ell = np.diff(ell_values)
        delta_ell = np.concatenate(((delta_ell[0],), delta_ell))
        # _fsky = fsky

        # ! or this
        _fsky = 1
        delta_ell = np.ones_like(delta_ell)

        cov_ell = sl.covariance_einsum(cl_5d, noise_5d, _fsky, ell_values, delta_ell)
        cov_ell_diag = sl.covariance_einsum(
            cl_5d,
            noise_5d,
            _fsky,
            ell_values,
            delta_ell,
            return_only_diagonal_ells=True,
        )

        cov_ell = cov_ell[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix]
        cov_ell_diag = cov_ell_diag[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix]

        # remove ell-dep prefactor from Gaussian cov
        # cov_ell_diag *= (2 * ell_values + 1)[:, None, None, None, None]
        # for ell1_ix, ell1 in enumerate(ell_values):
        #     cov_ell[ell1_ix, ell1_ix, ...] *= 2 * ell1 + 1

        # TODO use just one helper function, except for project_ellspace_cov_vec_helper
        """
        kwargs = {
            'func': project_ellspace_cov,
            'Amax': Amax,
            'ell1_values': ell_values,
            'ell2_values': ell_values,
            'cov_ell': cov_ell,
        }
        results = Parallel(n_jobs=n_jobs)(  # fmt: skip
            delayed(cov_parallel_helper)(
                theta_1_ix=theta_1_ix, theta_2_ix=theta_2_ix, mu=mu, nu=nu,
                zij=zij, zkl=zkl, ind_ab=ind_ab, ind_cd=ind_cd,
                **kwargs,
            )
            for theta_1_ix in tqdm(range(nbt))
            for theta_2_ix in range(nbt)
            for zij in range(zpairs_ab)
            for zkl in range(zpairs_cd)
        )  # fmt: skip

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_sb_g_6d[theta_1, theta_2, :, :, :, :] = cov_value

        results = Parallel(n_jobs=n_jobs)(
            delayed(project_ellspace_cov_vec_helper)(
                theta_1, theta_2, mu, nu, Amax, ell_values, ell_values, cov_ell_diag
            )
            for theta_1 in tqdm(range(nbt))
            for theta_2 in range(nbt)
        )

        for theta_1, theta_2, cov_value in results:
            cov_sb_g_vec_6d[theta_1, theta_2, :, :, :, :] = cov_value
    
        """

        # results = Parallel(n_jobs=n_jobs)(
        #     delayed(cov_g_sva_real_helper)(
        #         theta_1,
        #         theta_2,
        #         zi,
        #         zj,
        #         zk,
        #         zl,
        #         mu,
        #         nu,
        #         cl_5d + noise_5d,
        #         *probe_tuple_old,
        #     )
        #     for theta_1 in tqdm(range(theta_bins))
        #     for theta_2 in range(theta_bins)
        #     for zi in range(zbins)
        #     for zj in range(zbins)
        #     for zk in range(zbins)
        #     for zl in range(zbins)
        # )

        # for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
        #     cov_sb_gfromsva_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value

        # result_levin = np.zeros((nbt, nbt, zbins**4))
        # partial_integral = np.zeros((nbl, nbt, zbins**4))

        # n_jobs = 1
        # for ell1_ix, _ in enumerate(ell_values):
        #     cov_ell_2d = cov_ell[ell1_ix, ...].reshape(nbl, -1)
        #      # integrate in ell2
        #     partial_integral[ell1_ix, ...] = integrate_cov_levin(cov_ell_2d, nu, ell_values, n_jobs)
        #     integral = integrate_cov_levin(partial_integral[ell1_ix, ...], mu, ell_values, n_jobs)

    elif term == 'ssc':
        covs_oc_path = (
            '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test'
        )
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
        cov_sva_oc_3x2pt_10D = covs_oc_hs_npz['cov_sva_oc_3x2pt_10D']
        cov_mix_oc_3x2pt_10D = covs_oc_hs_npz['cov_mix_oc_3x2pt_10D']
        cov_sn_oc_3x2pt_10D = covs_oc_hs_npz['cov_sn_oc_3x2pt_10D']
        cov_g_oc_3x2pt_10D = covs_oc_hs_npz['cov_g_oc_3x2pt_10D']
        cov_ssc_oc_3x2pt_10D = covs_oc_hs_npz['cov_ssc_oc_3x2pt_10D']

        # project it to real space using Levin
        cov_ssc_oc_hs_6d = cov_ssc_oc_3x2pt_10D[
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...
        ]

        cov_ssc_sb_6d = integrate_double_bessel(
            cov_hs=cov_ssc_oc_hs_6d,
            mu=mu,
            nu=nu,
            ells=ell_values,
            thetas=theta_centers,
            n_jobs=n_jobs_lv,
        )

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
    cov_theta_indices = {ell_out: idx for idx, ell_out in enumerate(thetas_oc_load)}

    # SB tomographic indices start from 0
    tomoi_oc_load = data['tomoi'].unique()
    subtract_one = False
    if min(tomoi_oc_load) == 1:
        subtract_one = True

    # ! import .list covariance file
    shape = (n_probes_rs, n_probes_rs, nbt, nbt, zbins, zbins, zbins, zbins)
    cov_g_oc_3x2pt_8D = np.zeros(shape)
    cov_sva_oc_3x2pt_8D = np.zeros(shape)
    cov_mix_oc_3x2pt_8D = np.zeros(shape)
    cov_sn_oc_3x2pt_8D = np.zeros(shape)
    cov_ssc_oc_3x2pt_8D = np.zeros(shape)
    # cov_cng_oc_3x2pt_8D = np.zeros(shape)
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

        cov_sva_oc_3x2pt_8D[index_tuple] = df_chunk['covg sva'].values
        cov_mix_oc_3x2pt_8D[index_tuple] = df_chunk['covg mix'].values
        cov_sn_oc_3x2pt_8D[index_tuple] = df_chunk['covg sn'].values
        cov_g_oc_3x2pt_8D[index_tuple] = (
            df_chunk['covg sva'].values
            + df_chunk['covg mix'].values
            + df_chunk['covg sn'].values
        )
        cov_ssc_oc_3x2pt_8D[index_tuple] = df_chunk['covssc'].values
        # cov_cng_oc_3x2pt_8D[index_tuple] = df_chunk['covng'].values
        # cov_tot_oc_3x2pt_8D[index_tuple] = df_chunk['cov'].values

    covs_8d = [
        cov_sva_oc_3x2pt_8D,
        cov_mix_oc_3x2pt_8D,
        cov_sn_oc_3x2pt_8D,
        cov_g_oc_3x2pt_8D,
        cov_ssc_oc_3x2pt_8D,
        # cov_cng_oc_3x2pt_8D, cov_tot_oc_3x2pt_8D
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
        cov_oc_6d = cov_sva_oc_3x2pt_8D[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_sva_sb_6d
    elif term == 'sn':
        cov_oc_6d = cov_sn_oc_3x2pt_8D[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_sn_sb_6d
    elif term == 'mix':
        cov_oc_6d = cov_mix_oc_3x2pt_8D[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_mix_sb_6d
    elif term == 'gauss_ell':
        cov_oc_6d = cov_g_oc_3x2pt_8D[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_g_sb_6d
        cov_sb_vec_6d = cov_g_vec_sb_6d
    elif term == 'ssc':
        cov_oc_6d = cov_ssc_oc_3x2pt_8D[*probe_idx_dict_short_oc[probe], ...]
        cov_sb_6d = cov_ssc_sb_6d

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

    if np.allclose(cov_sb_6d, 0, atol=1e-20, rtol=1e-10):
        print('cov_sb_6d is zero')

    # cov_oc_4d = sl.cov_6D_to_4D(cov_oc_6d, theta_bins, zpairs_cross, ind_cross)
    # cov_sb_4d = sl.cov_6D_to_4D(cov_sb_6d, theta_bins, zpairs_cross, ind_cross)
    cov_oc_4d = sl.cov_6D_to_4D_blocks(
        cov_oc_6d, nbt, zpairs_ab, zpairs_cd, ind_ab, ind_cd
    )
    cov_sb_4d = sl.cov_6D_to_4D_blocks(
        cov_sb_6d, nbt, zpairs_ab, zpairs_cd, ind_ab, ind_cd
    )
    # cov_sb_vec_4d = sl.cov_6D_to_4D(cov_sb_vec_6d, theta_bins, zpairs_auto, ind_auto)
    # cov_sb_gfromsva_4d = sl.cov_6D_to_4D(cov_sb_gfromsva_6d,
    # theta_bins, zpairs_auto, ind_auto)

    cov_oc_2d = sl.cov_4D_to_2D(cov_oc_4d, block_index='zpair', optimize=True)
    cov_sb_2d = sl.cov_4D_to_2D(cov_sb_4d, block_index='zpair', optimize=True)
    # cov_sb_vec_2d = sl.cov_4D_to_2D(cov_sb_vec_4d, block_index='zpair')
    # cov_sb_gfromsva_2d = sl.cov_4D_to_2D(cov_sb_gfromsva_4d, block_index='zpair')

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

    cov_oc_spline = CubicSpline(
        thetas_oc_load_rad, np.diag(cov_oc_6d[:, :, zi, zj, zk, zl])
    )

    # compare total diag
    if cov_oc_2d.shape[0] == cov_oc_2d.shape[1]:
        sl.compare_funcs(
            None,
            {
                'OC': np.abs(np.diag(cov_oc_2d)),
                'SB': np.abs(np.diag(cov_sb_2d)),
                #  'SB_VEC': np.abs(np.diag(cov_sb_vec_2d)),
                #  'SB_split_sum': np.abs(np.diag(cov_sb_vec_2d)),  # TODO
                #  'SB_fromsva': np.abs(np.diag(cov_sb_gfromsva_2d)),
                #  'OC_SUM': np.abs(np.diag(cov_oc_sum_2d)),
            },
            logscale_y=[True, False],
            ylim_diff=[-110, 110],
            title=f'{term}, {probe}, total cov diag',
        )
        # plt.savefig(f'{term}_{probe}_total_cov_diag.png')

    # compare flattened matrix
    sl.compare_funcs(
        None,
        {
            'OC': np.abs(cov_oc_2d.flatten()),
            'SB': np.abs(cov_sb_2d.flatten()),
            #  'SB_VEC': np.abs(cov_sb_vec_2d.flatten()),
            #  'SB_fromsva': np.abs(cov_sb_gfromsva_2d.flatten()),
        },
        logscale_y=[True, False],
        title=f'{term}, {probe}, total cov flat',
        ylim_diff=[-110, 110],
    )

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
        title=f'{term}, {probe}, cov_6d[:, 0, 0, 0, 0, 0]',
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

print('Done.')
