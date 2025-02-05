from copy import deepcopy
import re
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from scipy.integrate import quad, quad_vec
from scipy.special import jv
import time
import pyccl as ccl
import matplotlib.pyplot as plt
from tqdm import tqdm
from spaceborne import sb_lib as sl
from joblib import Parallel, delayed


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


def cov_g_sva_real(thetal_i, thetau_i, mu, thetal_j, thetau_j, nu, Amax, ell_values,
                   c_ik, c_jl, c_il, c_jk):
    """
    Computes a single entry of the real-space Gaussian SVA (sample variance) part of the covariance matrix.
    """

    assert c_ik.shape == c_jl.shape == c_il.shape == c_jk.shape
    assert c_ik.shape == (len(ell_values),), f'Input cls must have shape {len(ell_values)}'

    def integrand_func(ell):
        kmu = k_mu(ell, thetal_i, thetau_i, mu)
        knu = k_mu(ell, thetal_j, thetau_j, nu)
        return ell * kmu * knu * (c_ik * c_jl + c_il * c_jk)

    integrand = integrand_func(ell_values)  # integrand is very oscillatory in ell space...
    integral = simps(y=integrand, x=ell_values)

    # integrate with quad and compare
    # integral = quad_vec(integrand_func, ell_values[0], ell_values[-1])[0]

    # Finally multiply the prefactor
    cov_elem = integral / (2.0 * np.pi * Amax)
    return cov_elem


def cov_g_sva_real_helper(theta_i_ix, theta_j_ix, zi, zj, zk, zl, mu, nu, cl_5d,
                          probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
    """
    Note: do not confuse the i, j in theta_i, theta_j with zi, zj; they are completely independent indices
    """

    thetal_i = theta_edges[theta_i_ix]
    thetau_i = theta_edges[theta_i_ix + 1]
    thetal_j = theta_edges[theta_j_ix]
    thetau_j = theta_edges[theta_j_ix + 1]

    return theta_i_ix, theta_j_ix, zi, zj, zk, zl, cov_g_sva_real(thetal_i, thetau_i, mu,
                                                                  thetal_j, thetau_j, nu,
                                                                  survey_area_sr, ell_values,
                                                                  cl_5d[probe_a_ix, probe_c_ix, :, zi, zk],
                                                                  cl_5d[probe_b_ix, probe_d_ix, :, zj, zl],
                                                                  cl_5d[probe_a_ix, probe_d_ix, :, zi, zl],
                                                                  cl_5d[probe_b_ix, probe_c_ix, :, zj, zk],
                                                                  )

def cov_g_mix_real(thetal_i, thetau_i, mu, thetal_j, thetau_j, nu, ell_values,
                   cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zi, zj, zk, zl):


    def integrand_func(ell, cl_ij):
        kmu = k_mu(ell, thetal_i, thetau_i, mu)
        knu = k_mu(ell, thetal_j, thetau_j, nu)
        return ell * kmu * knu * cl_ij
    
    
    # TODO generalize to different survey areas (max(Aij, Akl))
    prefac = \
        get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zl] * \
        t_mix(probe_b_ix, zbins, sigma_eps_i)[zj] /\
        (2 * np.pi * n_eff_2d[probe_b_ix, zj] * deg2toarcmin2 *
         np.max((survey_area_sr, survey_area_sr)))
    integrand = integrand_func(ell_values, cl_5d[probe_a_ix, probe_c_ix, :, zi, zk])
    integral = simps(y=integrand, x=ell_values)
    addendum_1 = prefac * integral

    # *

    prefac = \
        get_delta_tomo(probe_c_ix, probe_a_ix)[zk, zi] * \
        t_mix(probe_c_ix, zbins, sigma_eps_i)[zk] /\
        (2 * np.pi * n_eff_2d[probe_c_ix, zk] * deg2toarcmin2 *
         np.max((survey_area_sr, survey_area_sr)))
    integrand = integrand_func(ell_values, cl_5d[probe_b_ix, probe_d_ix, :, zj, zl])
    integral = simps(y=integrand, x=ell_values)
    addendum_2 = prefac * integral

    # *

    prefac = \
        get_delta_tomo(probe_d_ix, probe_b_ix)[zl, zj] * \
        t_mix(probe_d_ix, zbins, sigma_eps_i)[zl] /\
        (2 * np.pi * n_eff_2d[probe_d_ix, zl] * deg2toarcmin2 *
         np.max((survey_area_sr, survey_area_sr)))
    integrand = integrand_func(ell_values, cl_5d[probe_c_ix, probe_a_ix, :, zk, zi])
    integral = simps(y=integrand, x=ell_values)
    addendum_3 = prefac * integral

    # *

    prefac = \
        get_delta_tomo(probe_a_ix, probe_c_ix)[zi, zk] * \
        t_mix(probe_a_ix, zbins, sigma_eps_i)[zi] /\
        (2 * np.pi * n_eff_2d[probe_a_ix, zi] * deg2toarcmin2 *
         np.max((survey_area_sr, survey_area_sr)))
    integrand = integrand_func(ell_values, cl_5d[probe_d_ix, probe_b_ix, :, zl, zj])
    integral = simps(y=integrand, x=ell_values)
    addendum_4 = prefac * integral

    return addendum_1 + addendum_2 + addendum_3 + addendum_4


def cov_g_mix_real_helper(theta_i_ix, theta_j_ix, zi, zj, zk, zl, mu, nu, cl_5d,
                          probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
    """
    Note: do not confuse the i, j in theta_i, theta_j with zi, zj; they are completely independent indices
    """

    thetal_i = theta_edges[theta_i_ix]
    thetau_i = theta_edges[theta_i_ix + 1]
    thetal_j = theta_edges[theta_j_ix]
    thetau_j = theta_edges[theta_j_ix + 1]
    
    return theta_i_ix, theta_j_ix, zi, zj, zk, zl, cov_g_mix_real(thetal_i, thetau_i, mu,
                                                                  thetal_j, thetau_j, nu,
                                                                  ell_values,
                                                                  cl_5d, 
                                                                  probe_a_ix, probe_b_ix, 
                                                                  probe_c_ix, probe_d_ix, 
                                                                  zi, zj, zk, zl
                                                                  )


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


def get_npair(thetau_i, thetal_i, survey_area_sr, n_eff_i, n_eff_j):
    n_eff_i *= deg2toarcmin2
    n_eff_j *= deg2toarcmin2
    return np.pi * (thetau_i**2 - thetal_i**2) * survey_area_sr * n_eff_i * n_eff_j


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
deg2toarcmin2 = (180 / np.pi * 60)**2
survey_area_sr = survey_area_deg2 / deg2torad2

ell_min = 2
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

n_eff_lens = np.array([0.6, 0.6, 0.6])
n_eff_src = np.array([0.6, 0.6, 0.6])
n_eff_2d = np.row_stack((n_eff_lens, n_eff_lens, n_eff_src))  # in this way the indices correspond to xip, xim, g
sigma_eps_i = np.array([0.26, 0.26, 0.26])
sigma_eps_tot = sigma_eps_i * np.sqrt(2)
munu_vals = (0, 2, 4)

probe = 'xipxip'
probe_a_str, probe_b_str = split_probe_name(probe)


probe_idx_dict = {
    'xipxip': (0, 0, 0, 0),  # * SVA 1% ok; SN 0.1% ok
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

_probe_idx_dict_short = {
    'xip': 0,
    'xim': 1,
    'g': 2,
}

mu_dict = {
    'gg': 0,
    'gm': 2,
    'xip': 0,
    'xim': 4,
}


theta_edges = np.linspace(theta_min_arcmin / 60, theta_max_arcmin / 60, n_theta_edges)
theta_edges = np.deg2rad(theta_edges)
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
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
                      A_s=2.1e-9, n_s=0.96, m_nu=0.06, w0=-1.0, Neff=3.046,
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
z_nz_lenses = nz_lenses[:, 0]
z_nz_sources = nz_sources[:, 0]
bias_2d = np.genfromtxt(
    '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/output/OneCovariance/gal_bias_table.ascii')

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
for zi in range(zbins):
    plt.plot(gc_ker[zi].get_kernel()[1][0], gc_ker[zi].get_kernel()[0][0])
for zi in range(zbins):
    plt.plot(wl_ker[zi].get_kernel()[1][0], wl_ker[zi].get_kernel()[0][0])

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


cl_5d = np.zeros((n_probes, n_probes, len(ell_values), zbins, zbins))
cl_5d[0, 0, ...] = cl_ll_3d
cl_5d[0, 1, ...] = cl_ll_3d
cl_5d[0, 2, ...] = cl_gl_3d.transpose(0, 2, 1)

cl_5d[1, 0, ...] = cl_ll_3d
cl_5d[1, 1, ...] = cl_ll_3d
cl_5d[1, 2, ...] = cl_gl_3d.transpose(0, 2, 1)

cl_5d[2, 0, ...] = cl_gl_3d
cl_5d[2, 1, ...] = cl_gl_3d
cl_5d[2, 2, ...] = cl_gg_3d


mu, nu = mu_dict[probe_a_str], mu_dict[probe_b_str]
probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = probe_idx_dict[probe]

# Compute covariance:
cov_sb_sva_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
cov_sb_sn_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
cov_sb_mix_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))

print('Computing real-space Gaussian SVA covariance...')
start = time.time()
results = Parallel(n_jobs=-1)(delayed(cov_g_sva_real_helper)(theta_1, theta_2, zi, zj, zk, zl,
                                                             mu, nu, cl_5d,
                                                             probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                                                             )
                              for theta_1 in tqdm(range(theta_bins))
                              for theta_2 in range(theta_bins)
                              for zi in range(zbins)
                              for zj in range(zbins)
                              for zk in range(zbins)
                              for zl in range(zbins)
                              )

for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
    cov_sb_sva_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value
print(f'... Done in: {(time.time() - start):.2f} s')


print('Computing real-space Gaussian SN covariance...')
start = time.time()

# TODO generalize to different n(z)
npair_arr = np.zeros((theta_bins, zbins, zbins))
for theta_ix in range(theta_bins):
    for zi in range(zbins):
        for zj in range(zbins):
            thetal_i = theta_edges[theta_ix]
            thetau_i = theta_edges[theta_ix + 1]
            npair_arr[theta_ix, zi, zj] = get_npair(thetau_i, thetal_i,
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


print('Computing real-space Gaussian MIX covariance...')
start = time.time()

# TODO max between different effective areas
# prefac = get_delta_tomo(probe_b_ix, probe_d_ix)[None, None, None, :, None, :] *\
#     t_mix(probe_b_ix, zbins, sigma_eps_i)[None, None, None, :, None, None] /\
#     (2 * np.pi * n_eff_lens[None, None, None, :, None, None] * deg2toarcmin2 *
#         np.max((survey_area_sr, survey_area_sr)))

    
results = Parallel(n_jobs=-1)(delayed(cov_g_mix_real_helper)(theta_1, theta_2, zi, zj, zk, zl,
                                                               mu, nu, cl_5d,
                                                               probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                                                               )
                              for theta_1 in tqdm(range(theta_bins))
                              for theta_2 in range(theta_bins)
                              for zi in range(zbins)
                              for zj in range(zbins)
                              for zk in range(zbins)
                              for zl in range(zbins)
                              )

cov_g_mix_real_helper(theta_1, theta_2, zi, zj, zk, zl,
                                                               mu, nu, cl_5d,
                                                               probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                                                               )

for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
    cov_sb_mix_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value
print(f'... Done in: {(time.time() - start):.2f} s')



# ! ======================================= ONECOVARIANCE ==================================================
# test agains OC: save cls
oc_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test'
cl_ll_ascii_filename = f'Cell_ll_realsp'
cl_gl_ascii_filename = f'Cell_gl_realsp'
cl_gg_ascii_filename = f'Cell_gg_realsp'
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
# cov_ssc_oc_3x2pt_10D = np.zeros((n_probes, n_probes, n_probes, n_probes,
#                                  theta_bins, theta_bins, zbins, zbins, zbins, zbins))
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
    # cov_ssc_oc_3x2pt_10D[index_tuple] = df_chunk['covssc'].values
    # cov_cng_oc_3x2pt_10D[index_tuple] = df_chunk['covng'].values
    # cov_tot_oc_3x2pt_10D[index_tuple] = df_chunk['cov'].values

covs_10d = [cov_sva_oc_3x2pt_10D, cov_mix_oc_3x2pt_10D, cov_sn_oc_3x2pt_10D,
            cov_g_oc_3x2pt_10D,
            # cov_ssc_oc_3x2pt_10D, cov_cng_oc_3x2pt_10D, cov_tot_oc_3x2pt_10D
            ]

# for cov_10d in covs_10d:
#     cov_10d[0, 0, 1, 1] = deepcopy(np.transpose(cov_10d[1, 1, 0, 0], (1, 0, 4, 5, 2, 3)))
#     cov_10d[1, 0, 0, 0] = deepcopy(np.transpose(cov_10d[0, 0, 1, 0], (1, 0, 4, 5, 2, 3)))
#     cov_10d[1, 0, 1, 1] = deepcopy(np.transpose(cov_10d[1, 1, 1, 0], (1, 0, 4, 5, 2, 3)))

print(f"OneCovariance output loaded in {time.perf_counter() - start:.2f} seconds")
# ! =============================================================================================


cov_oc_6d = cov_mix_oc_3x2pt_10D[probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, ...]
cov_sb_6d = cov_sb_mix_6d

cov_oc_4d = sl.cov_6D_to_4D(cov_oc_6d, theta_bins, zpairs_auto, ind_auto)
cov_sb_4d = sl.cov_6D_to_4D(cov_sb_6d, theta_bins, zpairs_auto, ind_auto)
cov_oc_2d = sl.cov_4D_to_2D(cov_oc_4d, block_index='ell')
cov_sb_2d = sl.cov_4D_to_2D(cov_sb_4d, block_index='ell')

sl.compare_arrays(cov_sb_2d, cov_oc_2d,
                  'cov_sb_2d', 'cov_oc_2d',
                  abs_val=True, plot_diff_threshold=5, plot_diff_hist=True)

zi, zj, zk, zl = 0, 0, 0, 0

from scipy.interpolate import CubicSpline
cov_oc_spline = CubicSpline(thetas_oc_load_rad, np.diag(cov_oc_6d[:, :, zi, zj, zk, zl]))

sl.compare_funcs(theta_centers,
                 np.abs(np.diag(cov_sb_6d[:, :, zi, zj, zk, zl])),
                 np.abs(cov_oc_spline(theta_centers)),
                 name_a='SB',
                 name_b='OC',
                 logscale_y=[True, False],
                 title=f'{probe}, block cov diag, zijkl = {zi} {zj} {zk} {zl}')
sl.compare_funcs(None,
                 np.abs(np.diag(cov_sb_2d)),
                 np.abs(np.diag(cov_oc_2d)),
                 name_a='SB',
                 name_b='OC',
                 logscale_y=[True, False],
                 title=f'{probe}, total cov diag')
sl.compare_funcs(None,
                 np.abs(cov_sb_2d.flatten()),
                 np.abs(cov_oc_2d.flatten()),
                 name_a='SB',
                 name_b='OC',
                 logscale_y=[True, False],
                 title=f'{probe}, total cov flat')

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

print(f"Done.")
