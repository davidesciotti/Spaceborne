import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn, loggamma
from scipy.interpolate import interp1d
from tqdm import tqdm

import pyccl as ccl
from spaceborne import cosmo_lib
from spaceborne import sb_lib as sl
from scipy.fft import rfft, irfft, fft


# * pylevin hyperparameters
n_sub = 12  # number of collocation points in each bisection
n_bisec_max = 100  # maximum number of bisections used
rel_acc = 1e-5  # relative accuracy target
boost_bessel = True  # should the bessel functions be calculated with boost
# instead of GSL, higher accuracy at high Bessel orders
verbose = True  # should the code talk to you?
logx = True  # Tells the code to create a logarithmic spline in x for f(x)
logy = True  # Tells the code to create a logarithmic spline in y for y = f(x)


# This is defined globally because of parallelization issues
COSMO_CCL = None


def init_cosmo(cosmo):
    global COSMO_CCL
    COSMO_CCL = cosmo


def sigma2_z1z2_wrap_parallel(
    z_grid: np.ndarray,
    k_grid_sigma2: np.ndarray,
    cosmo_ccl: ccl.Cosmology,
    which_sigma2_b: str,
    mask_obj,
    n_jobs: int,
    integration_scheme: str,
    batch_size: int,
    parallel: bool = True,
    h=None,
) -> np.ndarray:
    """
    Parallelized version of sigma2_z1z2_wrap using joblib.
    """

    print('Computing sigma^2_b(z_1, z_2). This may take a while...')
    start = time.perf_counter()

    if integration_scheme == 'simps':
        if parallel:
            from pathos.multiprocessing import ProcessingPool as Pool

            # Create a list of arguments—one per z2 value in z_grid
            # Build the argument list without cosmo_ccl:
            arg_list = [
                (z2, z_grid, k_grid_sigma2, which_sigma2_b, 
                mask_obj.ell_mask, mask_obj.cl_mask, mask_obj.fsky)
                for z2 in z_grid
            ]  # fmt: skip

            # Create a Pathos ProcessingPool and initialize each worker:
            start = time.perf_counter()
            pool = Pool(n_jobs, initializer=init_cosmo, initargs=(cosmo_ccl,))
            sigma2_b_list = pool.map(pool_compute_sigma2_b, arg_list)

            # Convert the list of results to a numpy array and transpose
            sigma2_b = np.array(sigma2_b_list).T

        else:
            sigma2_b = np.zeros((len(z_grid), len(z_grid)))
            for z2_idx, z2 in enumerate(tqdm(z_grid)):
                sigma2_b[:, z2_idx] = sigma2_z2_func_vectorized(
                    z1_arr=z_grid,
                    z2=z2,
                    k_grid_sigma2=k_grid_sigma2,
                    cosmo_ccl=cosmo_ccl,
                    which_sigma2_b=which_sigma2_b,
                    ell_mask=mask_obj.ell_mask,
                    cl_mask=mask_obj.cl_mask,
                    fsky_mask=mask_obj.fsky,
                    integration_scheme=integration_scheme,
                    n_jobs=n_jobs,
                    h=h,
                )

    elif integration_scheme == 'fft':
        sigma2_b = sigma2_z1z2_func_vectorized_fft(
            z1_arr=z_grid,
            z2_arr=z_grid,
            k_grid_sigma2=k_grid_sigma2,
            cosmo_ccl=cosmo_ccl,
            which_sigma2_b=which_sigma2_b,
            ell_mask=mask_obj.ell_mask,
            cl_mask=mask_obj.cl_mask,
            fsky_mask=mask_obj.fsky,
            # integration_scheme=integration_scheme,
        )

    elif integration_scheme == 'levin':
        sigma2_b = sigma2_b_levin_batched(
            z_grid=z_grid,
            k_grid=k_grid_sigma2,
            cosmo_ccl=cosmo_ccl,
            which_sigma2_b=which_sigma2_b,
            ell_mask=mask_obj.ell_mask,
            cl_mask=mask_obj.cl_mask,
            fsky_mask=mask_obj.fsky,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )

    else:
        raise ValueError(
            '"sigma2_b_integration_scheme" must be "simps", "fft" or "levin"'
        )

    print(f'done in {time.perf_counter() - start:.2f} s')

    return sigma2_b


def compute_sigma2_b(
    z2, z_grid, k_grid_sigma2, which_sigma2_b, ell_mask, cl_mask, fsky_in
):
    """
    Wrapper for sigma2_z2_func_vectorized without the unpickleable cosmo_ccl argument.
    """
    return sigma2_z2_func_vectorized(
        z1_arr=z_grid,
        z2=z2,
        k_grid_sigma2=k_grid_sigma2,
        cosmo_ccl=COSMO_CCL,
        which_sigma2_b=which_sigma2_b,
        ell_mask=ell_mask,
        cl_mask=cl_mask,
        fsky_mask=fsky_in,
    )


def pool_compute_sigma2_b(args):
    """
    Helper function to be used with pathos processing pool
    """
    return compute_sigma2_b(*args)


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


def sigma2_z1z2_func_vectorized_fft(
    z1_arr: np.ndarray,
    z2_arr: np.ndarray,
    k_grid_sigma2: np.ndarray,
    cosmo_ccl: ccl.Cosmology,
    which_sigma2_b: str,
    ell_mask: np.ndarray,
    cl_mask: np.ndarray,
    fsky_mask: float,
    *,
    nk_fft: int = 2**21,
):
    # sanity check for z1_arr and z2_arr
    z1_arr = np.atleast_1d(z1_arr)
    z2_arr = np.atleast_1d(z2_arr)
    np.testing.assert_equal(z1_arr, z2_arr)

    a1 = cosmo_lib.z_to_a(z1_arr)
    a2 = cosmo_lib.z_to_a(z2_arr)
    chi1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
    chi2 = ccl.comoving_radial_distance(cosmo_ccl, a2)
    g1 = ccl.growth_factor(cosmo_ccl, a1)
    g2 = ccl.growth_factor(cosmo_ccl, a2)

    k_min = k_grid_sigma2.min()
    k_max = k_grid_sigma2.max()

    k_grid = np.linspace(k_min, k_max, nk_fft)
    dk = k_grid[1] - k_grid[0]
    Pk0 = ccl.linear_matter_power(cosmo_ccl, k=k_grid, a=1.0)

    # real FFT → cosine coefficients on linear grid
    fft_coeffs = rfft(Pk0) * dk  # \sum f(k) cos → Re{FFT} * dk
    r_grid = np.arange(fft_coeffs.size) * 2 * np.pi / (k_max - k_min)
    c_r = fft_coeffs.real

    # interpolate C(r)
    c_0 = simps(Pk0, k_grid)
    c_func = interp1d(
        r_grid,
        c_r,
        kind='cubic',
        bounds_error=False,
        fill_value=(c_0, 0.0),
        assume_sorted=True,
    )

    # ---------- assemble integral (broadcast over z₁, z₂) ----------
    chi1_mat, chi2_mat = chi1[:, None], chi2[None, :]
    r_plus = chi1_mat + chi2_mat
    r_minus = np.abs(chi1_mat - chi2_mat)
    integral = 0.5 / (chi1_mat * chi2_mat) * (c_func(r_minus) - c_func(r_plus))

    if which_sigma2_b == 'full_curved_sky':
        return (g1[:, None] * g2[None, :]) * integral / (2.0 * np.pi**2)

    elif which_sigma2_b in {'polar_cap_on_the_fly', 'from_input_mask'}:
        part_result = np.sum((2 * ell_mask + 1) * cl_mask) * 2.0 / np.pi
        return (part_result * g1[:, None] * g2[None, :] * integral) / (
            4.0 * np.pi * fsky_mask
        ) ** 2

    raise ValueError('Invalid which_sigma2_b option.')


def sigma2_z2_func_vectorized(
    z1_arr,
    z2,
    k_grid_sigma2,
    cosmo_ccl,
    which_sigma2_b,
    ell_mask,
    cl_mask,
    fsky_mask,
    integration_scheme='simps',
    n_jobs=1,
    h=None,
):
    """
    Vectorized version of sigma2_func in z1. Implements the formula
       \sigma^2_{\rm b, \, fullsky}(z_{1}, z_{2}) = \frac{1}{2 \pi^{2}} \int_0^{\infty}
       \diff k \, k^{2} \,
       {\rm j}_{0}(k \chi_1)\,
       {\rm j}_{0}(k \chi_2) \,
       P_{\rm L}(k \, | \, z_1, z_2)
    """

    a1_arr = cosmo_lib.z_to_a(z1_arr)
    a2 = cosmo_lib.z_to_a(z2)

    r1_arr = ccl.comoving_radial_distance(cosmo_ccl, a1_arr)
    r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    growth_factor_z1_arr = ccl.growth_factor(cosmo_ccl, a1_arr)
    growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)

    # Define the integrand as a function of k
    def integrand(k):
        return (
            k**2
            * ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)
            * spherical_jn(0, k * r1_arr[:, None])
            * spherical_jn(0, k * r2)
        )

    if integration_scheme == 'simps':
        integral_result = simps(y=integrand(k_grid_sigma2), x=k_grid_sigma2, axis=1)
    elif integration_scheme == 'levin':
        raise NotImplementedError(
            'This implementation works, but is not vectorized;'
            ' you should use sigma2_b_levin_batched instead'
        )
        # integrand shape must be (len(x), N). N is the number of integrals (2)
        integrand = k_grid_sigma2**2 * ccl.linear_matter_power(
            cosmo_ccl, k=k_grid_sigma2, a=1.0
        )
        integrand = integrand[:, None]
        integral_result = integrate_levin(r1_arr, r2, integrand, k_grid_sigma2, n_jobs)
        integral_result = integral_result[:, 0]

    if which_sigma2_b == 'full_curved_sky':
        result = (
            1
            / (2 * np.pi**2)
            * growth_factor_z1_arr
            * growth_factor_z2
            * integral_result
        )

    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
        partial_summand = np.zeros((len(z1_arr), len(ell_mask)))
        # NOTE: you should include a 2/np.pi factor, see Eq. (26)
        # of https://arxiv.org/pdf/1612.05958, or Champaghe et al 2017
        partial_summand = (
            (2 * ell_mask + 1)
            * cl_mask
            * 2
            / np.pi
            * growth_factor_z1_arr[:, None]
            * growth_factor_z2
        )
        partial_summand *= integral_result[:, None]
        result = np.sum(partial_summand, axis=1)
        one_over_omega_s_squared = 1 / (4 * np.pi * fsky_mask) ** 2
        result *= one_over_omega_s_squared

        # F. Lacasa:
        # np.sum((2*ell+1)*cl_mask*Cl_XY[ipair,jpair,:])/(4*pi*fsky)**2
    else:
        raise ValueError(
            'which_sigma2_b must be either "full_curved_sky" or '
            '"polar_cap_on_the_fly" or "from_input_mask"'
        )

    return result


def integrate_levin(r1_arr, r2, integrand, k_grid_sigma2, n_jobs):
    """This can probably be further optimized by not instantiating
    the class at evey value of r2"""
    import pylevin as levin

    # Constructor of the class
    integral_type = 2  # double spherical
    N_thread = n_jobs  # Number of threads used for hyperthreading
    diagonal = False
    lp_double = levin.pylevin(
        type=integral_type,
        x=k_grid_sigma2,
        integrand=integrand,
        logx=logx,
        logy=logy,
        nthread=N_thread,
        diagonal=diagonal,
    )

    # accuracy settings
    lp_double.set_levin(
        n_col_in=n_sub,
        maximum_number_bisections_in=n_bisec_max,
        relative_accuracy_in=rel_acc,
        super_accurate=boost_bessel,
        verbose=verbose,
    )

    M = len(r1_arr)  # number of arguments at which the integrals are evaluated
    N = 1
    result_levin = np.zeros((M, N))  # allocate the result

    lp_double.levin_integrate_bessel_double(
        x_min=k_grid_sigma2[0] * np.ones(M),
        x_max=k_grid_sigma2[-1] * np.ones(M),
        k_1=r1_arr,
        k_2=r2 * np.ones(M),
        ell_1=(0 * np.ones(M)).astype(int),
        ell_2=(0 * np.ones(M)).astype(int),
        result=result_levin,
    )

    return result_levin


def plot_sigma2(sigma2_arr, z_grid_sigma2):
    font_size = 28
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['legend.fontsize'] = font_size

    plt.figure()
    pad = 0.4  # I don't want to plot sigma at the edges of the grid, it's too noisy
    for z_test in np.linspace(z_grid_sigma2.min() + pad, z_grid_sigma2.max() - pad, 4):
        z1_idx = np.argmin(np.abs(z_grid_sigma2 - z_test))
        z_1 = z_grid_sigma2[z1_idx]

        plt.plot(z_grid_sigma2, sigma2_arr[z1_idx, :], label=f'$z_1={z_1:.2f}$ ')
        plt.axvline(z_1, color='k', ls='--', label='$z_1$')
    plt.xlabel('$z_2$')
    plt.ylabel('$\\sigma^2(z_1, z_2)$')  # sigma2 is dimensionless!
    plt.legend()
    plt.show()

    font_size = 18
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['legend.fontsize'] = font_size
    sl.matshow(sigma2_arr, log=True, abs_val=True, title='$\\sigma^2(z_1, z_2)$')
