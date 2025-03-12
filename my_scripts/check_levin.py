import pyccl as ccl
import pylevin as levin
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.965)

N_z = 300
z = np.geomspace(0.01, 1, N_z)


chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + z))
kmin, kmax, nk = 1e-4, 1e1, 500
k = np.geomspace((kmin), (kmax), nk)  # Wavenumber
pk = []
for zet in z:
    pk.append(k * ccl.linear_matter_power(cosmo, k, 1 / (1 + zet)))
pk = np.array(pk).T
pk_zz = np.sqrt(pk[:, :, None] * pk[:, None, :])

integral_type = 2
N_thread = 4  # Number of threads used for hyperthreading
logx = True  # Tells the code to create a logarithmic spline in x for f(x)
logy = True  # Tells the code to create a logarithmic spline in y for y = f(x)
n_sub = 6  # number of collocation points in each bisection
n_bisec_max = 32  # maximum number of bisections used
rel_acc = 1e-5  # relative accuracy target
boost_bessel = True  # should the bessel functions be calculated with boost instead of GSL, higher accuracy at high Bessel orders
verbose = False  # should the code talk to you?

# We have Nz integrands and Nz arguments passed to the integral.
# However, we only want those integrals where the arguments match the corresponding integral,
# They sit on the diagonal of the large Nz, Nz matrix, so the result is also only Nz
# and diagonals_only is True
result_levin = np.zeros(N_z)
lower_limit = k[0] * np.ones(N_z)
upper_limit = k[-1] * np.ones(N_z)
ell = np.zeros(N_z).astype(int)
diagonals_only = True

result = np.zeros((N_z, N_z))

t0 = time.time()
for i_chi_1, chi_1 in enumerate(chi):
    lp = levin.pylevin(
        integral_type,
        k,
        pk_zz[:, i_chi_1, i_chi_1:],
        logx,
        logy,
        N_thread,
        diagonals_only,
    )
    lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
    lp.levin_integrate_bessel_double(
        lower_limit[i_chi_1:],
        upper_limit[i_chi_1:],
        chi_1 * np.ones_like(chi[i_chi_1:]),
        chi[i_chi_1:],
        ell[i_chi_1:],
        ell[i_chi_1:],
        result_levin[i_chi_1:],
    )
    result[i_chi_1, i_chi_1:] = result_levin[i_chi_1:]
    result[i_chi_1:, i_chi_1] = result_levin[i_chi_1:]
print('Levin took', time.time() - t0, 's')


pk_zz_flat = pk_zz.reshape((nk, N_z * N_z))


result_levin_flat = np.zeros(N_z * N_z)
lower_limit = k[0] * np.ones(N_z * N_z)
upper_limit = k[-1] * np.ones(N_z * N_z)
ell = np.zeros(N_z * N_z).astype(int)
diagonals_only = True

# result = np.zeros((N_z, N_z))

X, Y = np.meshgrid(chi, chi, indexing='ij')
chi1_flat = X.reshape(N_z * N_z)
chi2_flat = Y.reshape(N_z * N_z)

lp_flat = levin.pylevin(integral_type, k, pk_zz_flat, logx, logy, N_thread, True)
lp_flat.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
t0 = time.time()
lp_flat.levin_integrate_bessel_double(
    lower_limit, upper_limit, chi1_flat, chi2_flat, ell, ell, result_levin_flat
)
print('Levin took', time.time() - t0, 's')

result_levin = result_levin_flat.reshape((N_z, N_z))
plt.imshow(result_levin)
plt.show()

# --- Efficient broadcasting: only compute the upper triangle ---
# Get indices for the upper triangle (including diagonal)
triu_indices = np.triu_indices(N_z)
n_upper = len(triu_indices[0])  # Number of unique integrals to compute

# Extract the corresponding integrand slices: shape becomes (nk, n_upper)
pk_zz_upper = pk_zz[:, triu_indices[0], triu_indices[1]]

# Build evaluation arrays for chi:
chi1_flat = chi[triu_indices[0]]  # shape (n_upper,)
chi2_flat = chi[triu_indices[1]]  # shape (n_upper,)


# Define integration limits and orders for each integral
lower_limit = k[0] * np.ones(n_upper)
upper_limit = k[-1] * np.ones(n_upper)
ell = np.zeros(n_upper, dtype=int)

lower_limit = np.array(lower_limit, dtype=np.float64).reshape(-1, 1)
upper_limit = np.array(upper_limit, dtype=np.float64).reshape(-1, 1)
chi1_flat = np.array(chi1_flat, dtype=np.float64).reshape(-1, 1)
chi2_flat = np.array(chi2_flat, dtype=np.float64).reshape(-1, 1)
ell = np.array(ell, dtype=np.int32).reshape(-1, 1)
result_flat = np.zeros((n_upper), dtype=np.float64)


# Create the pylevin instance with the flattened integrand
lp = levin.pylevin(integral_type, k, pk_zz_upper, logx, logy, N_thread, True)
lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)

t0 = time.time()
lp.levin_integrate_bessel_double(lower_limit, upper_limit, chi1_flat, chi2_flat,
                                 ell, ell, result_flat)
print('Levin took', time.time() - t0, 's')

# Assemble the full symmetric matrix from the upper-triangular results
result_matrix = np.zeros((N_z, N_z))
result_matrix[triu_indices] = result_flat
# Fill in the lower triangle by symmetry
result_matrix = result_matrix + result_matrix.T - np.diag(np.diag(result_matrix))

plt.imshow(result_matrix)
plt.colorbar()
plt.title("Integrated Result Matrix")
plt.show()

np.testing.assert_allclose(result, result_levin, atol=0, rtol=1e-6)
np.testing.assert_allclose(result, result_matrix, atol=0, rtol=1e-6)
np.testing.assert_allclose(result, result.T, atol=0, rtol=1e-6)
np.testing.assert_allclose(result_levin, result_levin.T, atol=0, rtol=1e-6)
np.testing.assert_allclose(result_matrix, result_matrix.T, atol=0, rtol=1e-6)