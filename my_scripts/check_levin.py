import pyccl as ccl
import pylevin as levin
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.965)

N_z = 100
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
        integral_type, k, pk_zz[:, i_chi_1, i_chi_1:], logx, logy, N_thread
    )
    lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
    lp.levin_integrate_bessel_double(
        lower_limit[i_chi_1:],
        upper_limit[i_chi_1:],
        chi_1 * np.ones_like(chi[i_chi_1:]),
        chi[i_chi_1:],
        ell[i_chi_1:],
        ell[i_chi_1:],
        diagonals_only,
        result_levin[i_chi_1:],
    )
    result[i_chi_1, i_chi_1:] = result_levin[i_chi_1:]
    result[i_chi_1:, i_chi_1] = result_levin[i_chi_1:]
print('Levin took', time.time() - t0, 's')

plt.matshow(np.log10(np.abs(result)))
