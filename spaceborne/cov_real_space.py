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
        raise ValueError("mu must be one of {0,2,4} in this simplified example.")


def k_mu(ell, thetal, thetau, mu):
    """
    Computes the kernel K_mu(ell * theta_i) in Eq. (E.2):

        K_mu(l * theta_i) = 2 / [ (theta_u^2 - theta_l^2) * l^2 ]
                            * [ b_mu(l * theta_u) - b_mu(l * theta_l) ].
    """
    prefactor = 2.0 / ((thetau**2 - thetal**2) * (ell**2))
    return prefactor * (b_mu(ell * thetau, mu) - b_mu(ell * thetal, mu))


def C_ab(lval, a, b):
    """
    Mock-up for the angular power spectrum C^(a,b)(ell).
    In a real application, this would interpolate a theoretical or measured Cl.

    Here, we just provide something simple (e.g. a power law) as a placeholder:
        C^(a,b)(l) ~ (1/l^(2)) for demonstration.
    You would replace this with your own input data or theory code.
    """
    # For instance, you could imagine each (a,b) having a slightly different amplitude:
    amp = 1.0 + 0.1 * (a + b)
    return amp / (1.0 + lval**2)


def cov_g_sva_real(
    thetal_i, thetau_i, mu, thetal_j, thetau_j, nu,
    zi, zj, zk, zl,  # tomography / field indices
    Amax,  # mock "A_{max, mu, nu}" factor
    ell_values, c_ik, c_jl, c_il, c_jk
):
    """
    Computes the Gaussian real-space covariance term, Eq. (E.1),

       Cov_{G,sva}[ Xi^{(ij)}_mu(theta_i),  Xi^{(kl)}_nu(theta_j) ] =
         1 / (2pi * Amax) * \int_{0}^{∞} [ dℓ * ℓ ] 
         * K_mu(ℓ, theta_i) * K_nu(ℓ, theta_j) 
         * [ C^(ik)(ℓ) C^(jl)(ℓ) + C^(il)(ℓ) C^(jk)(ℓ) ].

    """

    assert c_ik.shape == c_jl.shape == c_il.shape == c_jk.shape
    assert c_ik.shape == (len(ell_values),), f'Input cls must have shape {len(ell_values)}'

    def integrand_func(ell):
        kmu = k_mu(ell, thetal_i, thetau_i, mu)
        knu = k_mu(ell, thetal_j, thetau_j, nu)
        # Evaluate the needed power spectra
        return ell * kmu * knu * (c_ik * c_jl + c_il * c_jk)

    # start = time.perf_counter()
    integrand = integrand_func(ell_values)  # integrand is very oscillatory in ell space...
    integral = simps(y=integrand, x=ell_values)
    # print(f"simps time: {time.perf_counter() - start}")

    # integrate with quad and compare
    # start = time.perf_counter()
    # integral = quad_vec(integrand_func, ell_values[0], ell_values[-1])[0]
    # print(f"quad time: {time.perf_counter() - start}")

    # Finally multiply the prefactor
    cov_val = integral / (2.0 * np.pi * Amax)
    return cov_val


# Example usage:
# Suppose we have two angular bins: [thetal_i, thetau_i] and [thetal_j, thetau_j]
# and we want the covariance between Xi^(i,j)_mu for bin i,j and Xi^(k,l)_nu for bin k,l.
mu, nu = 0, 0   # e.g. Xi_+ (mu=0) and Xi_- (mu=2) in some notations
zbins = 3
survey_area_deg2 = 2500
deg2torad2 =  (180 / np.pi)**2 
Amax = survey_area_deg2 / deg2torad2

ell_min = 2
ell_max = 100_000
nbl = 500
theta_min_arcmin = 50
theta_max_arcmin = 300
n_theta_edges = 21
n_probes = 2
df_chunk_size = 50000
cov_list_name = 'covariance_list_3x2_rcf'

# theta_edges = np.arange(0.1, 2.5, 0.1)  # TODO in degrees; loosely based on Duret for BAO, refine!
theta_edges = np.linspace(theta_min_arcmin / 60, theta_max_arcmin / 60,
                          n_theta_edges)  # TODO this is what I do in OC at the moment
# TODO is rad correct? I think this should be the arg of the bessel functions
theta_edges = np.deg2rad(theta_edges)  # * 60 because it's in arcmin above
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
theta_bins = len(theta_centers)

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

# Compute covariance:
zi, zj, zk, zl = 0, 0, 0, 0  # dummy field indices (e.g. lens bin 0, lens bin 1, etc.)

cov_sva_real = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
c_ik = cl_gg_3d[:, zi, zk]
c_jl = cl_gg_3d[:, zj, zl]
c_il = cl_gg_3d[:, zi, zl]
c_jk = cl_gg_3d[:, zj, zk]

print('Computing real-space Gaussian SVA covariance...')


def compute_cov_element(theta_1, theta_2, zi, zj, zk, zl):
    thetal_i = theta_edges[theta_1]
    thetau_i = theta_edges[theta_1 + 1]
    thetal_j = theta_edges[theta_2]
    thetau_j = theta_edges[theta_2 + 1]
    return theta_1, theta_2, zi, zj, zk, zl, cov_g_sva_real(thetal_i, thetau_i, mu,
                                                            thetal_j, thetau_j, nu,
                                                            zi, zj, zk, zl,
                                                            Amax, ell_values,
                                                            c_ik, c_jl, c_il, c_jk)


start = time.time()
results = Parallel(n_jobs=-1)(delayed(compute_cov_element)(theta_1, theta_2, zi, zj, zk, zl)
                              for theta_1 in tqdm(range(theta_bins))
                              for theta_2 in range(theta_bins)
                              #   for zi in range(zbins)
                              #   for zj in range(zbins)
                              #   for zk in range(zbins)
                              #   for zl in range(zbins)
                              )

for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
    cov_sva_real[theta_1, theta_2, zi, zj, zk, zl] = cov_value
print(f'... Done in: {(time.time() - start):.2f} s')


# test agains OC: save cls
oc_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test'
cl_ll_ascii_filename = f'Cell_ll_realsp'
cl_gl_ascii_filename = f'Cell_gl_realsp'
cl_gg_ascii_filename = f'Cell_gg_realsp'
sl.write_cl_ascii(oc_path, cl_ll_ascii_filename, cl_ll_3d, ell_values, zbins)
sl.write_cl_ascii(oc_path, cl_gl_ascii_filename, cl_gl_3d, ell_values, zbins)
sl.write_cl_ascii(oc_path, cl_gg_ascii_filename, cl_gg_3d, ell_values, zbins)

# now load and reshape output

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

probe_idx_dict = {
    'm': 0,
    'g': 1,
}

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
    probe_idx_a = df_chunk['#obs'].str[0].map(probe_idx_dict).values
    probe_idx_b = df_chunk['#obs'].str[1].map(probe_idx_dict).values
    probe_idx_c = df_chunk['#obs'].str[2].map(probe_idx_dict).values
    probe_idx_d = df_chunk['#obs'].str[3].map(probe_idx_dict).values

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
    index_tuple = (probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d, theta1_idx, theta2_idx,
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


for cov_10d in covs_10d:

    cov_10d[0, 0, 1, 1] = deepcopy(np.transpose(cov_10d[1, 1, 0, 0], (1, 0, 4, 5, 2, 3)))
    cov_10d[1, 0, 0, 0] = deepcopy(np.transpose(cov_10d[0, 0, 1, 0], (1, 0, 4, 5, 2, 3)))
    cov_10d[1, 0, 1, 1] = deepcopy(np.transpose(cov_10d[1, 1, 1, 0], (1, 0, 4, 5, 2, 3)))

print(f"OneCovariance output loaded in {time.perf_counter() - start:.2f} seconds")


cov_oc_sva_6d = cov_sva_oc_3x2pt_10D[1, 1, 1, 1, ...]

zi, zj, zk, zl = 0, 0, 0, 0
sl.matshow(cov_sva_real[:, :, zi, zj, zk, zl])
sl.matshow(cov_oc_sva_6d[:, :, zi, zj, zk, zl])

sl.compare_arrays(cov_sva_real[:, :, zi, zj, zk, zl], cov_oc_sva_6d[:, :, zi, zj, zk, zl])

from scipy.interpolate import CubicSpline
cov_oc_spline = CubicSpline(thetas_oc_load_rad, np.diag(cov_oc_sva_6d[:, :, zi, zj, zk, zl]))

sl.compare_funcs(theta_centers,
                 np.diag(cov_sva_real[:, :, zi, zj, zk, zl]),
                 cov_oc_spline(theta_centers),
                 name_a='SB', name_b='OC', logscale_y=[False, False])

plt.figure()
plt.plot(theta_centers, np.diag(cov_sva_real[:, :, zi, zj, zk, zl]), marker='.', label='sb')
plt.plot(thetas_oc_load_rad, np.diag(cov_oc_sva_6d[:, :, zi, zj, zk, zl]), marker='.', label='oc')
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'diag cov ww')
plt.legend()

# TODO double check ngal, it's totally random at the moment; same for sigma_eps
# TODO galaxy bias in cls?
# TODO cross.z bins & other probes
# TODO probably ell range as well
# integration? quad?

print(f"Done.")
