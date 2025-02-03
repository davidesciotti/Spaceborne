import numpy as np
from scipy.integrate import simpson as simps
from scipy.integrate import quad
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

    # create integrand and perform integration with simps
    integrand = integrand_func(ell_values)  # integrand is quite ugly in ell space

    # start = time.perf_counter()
    integral = simps(y=integrand, x=ell_values)
    # print(f"simps time: {time.perf_counter() - start}")

    # integrate with quad and compare
    # start = time.perf_counter()
    # integral_quad = quad(integrand_func, ell_values[0], ell_values[-1])[0]
    # print(f"quad time: {time.perf_counter() - start}")

    # Finally multiply the prefactor
    cov_val = integral / (2.0 * np.pi * Amax)
    return cov_val


# Example usage:
# Suppose we have two angular bins: [thetal_i, thetau_i] and [thetal_j, thetau_j]
# and we want the covariance between Xi^(i,j)_mu for bin i,j and Xi^(k,l)_nu for bin k,l.
mu, nu = 0, 0   # e.g. Xi_+ (mu=0) and Xi_- (mu=2) in some notations
zbins = 3
survey_area_sq_deg = 13245
Amax = survey_area_sq_deg * (np.pi / 180)**2  # TODO deg2^ steradian? fsky??

theta_min_arcmin = 50
theta_max_arcmin = 300
n_theta_edges = 21

# theta_edges = np.arange(0.1, 2.5, 0.1)  # TODO in degrees; loosely based on Duret for BAO, refine!
theta_edges = np.linspace(theta_min_arcmin / 60, theta_max_arcmin / 60,
                          n_theta_edges)  # TODO this is what I do in OC at the moment
# TODO is rad correct? I think this should be the arg of the bessel functions
theta_edges = np.deg2rad(theta_edges)  # * 60 because it's in arcmin above
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
theta_bins = len(theta_centers)

ell_values = np.arange(1, 100_000)

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
cov_sva_real_2 = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
c_ik = cl_gg_3d[:, zi, zk]
c_jl = cl_gg_3d[:, zj, zl]
c_il = cl_gg_3d[:, zi, zl]
c_jk = cl_gg_3d[:, zj, zk]

print('Computing real-space Gaussian SVA covariance...')


def compute_cov_element(theta_1, theta_2):
    thetal_i = theta_edges[theta_1]
    thetau_i = theta_edges[theta_1 + 1]
    thetal_j = theta_edges[theta_2]
    thetau_j = theta_edges[theta_2 + 1]

    return theta_1, theta_2, cov_g_sva_real(thetal_i, thetau_i, mu,
                                            thetal_j, thetau_j, nu,
                                            zi, zj, zk, zl,
                                            Amax, ell_values,
                                            c_ik, c_jl, c_il, c_jk)


# TODO zi, zj, zk, zl for loops...
results = Parallel(n_jobs=-1)(delayed(compute_cov_element)(theta_1, theta_2)
                              for theta_1 in range(theta_bins)
                              for theta_2 in range(theta_bins))

for theta_1, theta_2, cov_value in results:
    cov_sva_real[theta_1, theta_2, zi, zj, zk, zl] = cov_value


sl.matshow(cov_sva_real[:, :, zi, zj, zk, zl])

plt.figure()
plt.plot(theta_centers, np.diag(cov_sva_real[:, :, zi, zj, zk, zl]), marker='.')
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'diag cov ww')

# test agains OC: save cls
oc_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/realspace_test'
cl_ll_ascii_filename = f'Cell_ll_realsp'
cl_gl_ascii_filename = f'Cell_gl_realsp'
cl_gg_ascii_filename = f'Cell_gg_realsp'
sl.write_cl_ascii(oc_path, cl_ll_ascii_filename, cl_ll_3d, ell_values, zbins)
sl.write_cl_ascii(oc_path, cl_gl_ascii_filename, cl_gl_3d, ell_values, zbins)
sl.write_cl_ascii(oc_path, cl_gg_ascii_filename, cl_gg_3d, ell_values, zbins)

print(f"Done.")
