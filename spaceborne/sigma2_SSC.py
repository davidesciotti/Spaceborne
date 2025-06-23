import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn
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

# TODO finish implementing this function and test if if needed
# def sigma2_flatsky(z1, z2, k_perp_grid, k_par_grid, cosmo_ccl, Omega_S, theta_S):
#     """Compute the flatsky variance between two redshifts z1 and z2 for a cosmology given by cosmo_ccl."""

#     # Compute the comoving distance at the given redshifts
#     from scipy.special import j1 as J1

#     a1 = 1 / (1 + z1)
#     a2 = 1 / (1 + z2)
#     r1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
#     r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

#     # Compute the growth factors at the given redshifts
#     growth_factor_z1 = ccl.growth_factor(cosmo_ccl, a1)
#     growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)

#     # Compute the integrand over k_perp and k_par grids
#     def integrand(k_perp, k_par, r1, r2, theta_S):
#         k = np.sqrt(k_par**2 + k_perp**2)
#         bessel_term = (
#             J1(k_perp * theta_S * r1)
#             * J1(k_perp * theta_S * r2)
#             / (k_perp * theta_S * r1 * k_perp * theta_S * r2)
#         )
#         power_spectrum = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)
#         return k_perp * bessel_term * np.cos(k_par * (r1 - r2)) * power_spectrum

#     # Perform the double integral using Simpson's rule
#     integral_result_k_perp = np.array(
#         [
#             simps(integrand(k_perp, k_par_grid, r1, r2, theta_S), k_par_grid)
#             for k_perp in k_perp_grid
#         ]
#     )
#     integral_result = simps(integral_result_k_perp, k_perp_grid)

#     # Compute the final result
#     sigma2 = (
#         1
#         / (2 * np.pi**2)
#         * growth_factor_z1
#         * growth_factor_z2
#         * integral_result
#         / Omega_S**2
#     )

#     return sigma2


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

    if parallel and integration_scheme == 'simps':
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

    elif not parallel and integration_scheme == 'fft':
        sigma2_b = np.zeros((len(z_grid), len(z_grid)))
        for z2_idx, z2 in enumerate(tqdm(z_grid)):
            sigma2_b[:, z2_idx] = sigma2_z2_func_vectorized_fft_linear(
                z1_arr=z_grid,
                z2=z2,
                k_grid_sigma2=k_grid_sigma2,
                cosmo_ccl=cosmo_ccl,
                which_sigma2_b=which_sigma2_b,
                ell_mask=mask_obj.ell_mask,
                cl_mask=mask_obj.cl_mask,
                fsky_mask=mask_obj.fsky,
                # integration_scheme=integration_scheme,
                n_jobs=n_jobs,
            )
    elif parallel and integration_scheme == "fft":
        sigma2_b = sigma2_z1z2_func_vectorized_fft_linear(
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
    elif not parallel and integration_scheme in ['simps', 'fft_pyssc']:
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
        raise ValueError('Invalid combination of "parallel" and "integration_scheme". ')

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


from scipy.interpolate import interp1d


def __cosine_transform_fftlog(k: np.ndarray, f_k: np.ndarray, q: float = 0.0):
    """Cosine‑transform *f(k)* on a log grid via FFTLog.

    Returns ``r, C(r)`` with *r* strictly increasing (duplicates removed).
    Implements the basic prescription of Hamilton (2000).
    """
    print('in _cosine_transform_fftlog')
    ln_k = np.log(k)
    dlnk = ln_k[1] - ln_k[0]
    n = k.size

    # Bias and even‑extension
    f_bias = k**q * f_k
    f_even = np.concatenate((f_bias[::-1], f_bias))

    # FFT of the even array (real)
    f_hat = rfft(f_even)

    # Dual r‑grid (prevent overflow)
    x = np.arange(n)
    arg = np.pi * x / (n * dlnk)
    mask = arg < 690  # exp(690) ~ 1e300, safely within float64 range
    r = 1.0 / k[0] * np.exp(arg[mask])

    # Inverse FFT → cosine coefficients
    c_even = irfft(f_hat)[:n][mask]

    # CORRECTED LINE: Removed the (2.0 * np.pi) factor
    C_r = r ** (-q - 1.0) * c_even

    return r, C_r


# -----------------------------------------------------------------------------
#               2. σ_b²(z₁,z₂) via analytic FFTLog reduction
# -----------------------------------------------------------------------------


def _cosine_transform_fftlog(k: np.ndarray, f_k: np.ndarray, q: float = 0.0):
    """Cosine-transform f(k) on a log grid via FFTLog."""
    ln_k = np.log(k)
    dlnk = ln_k[1] - ln_k[0]
    n = k.size

    # Bias and even-extension
    f_bias = k**q * f_k
    f_even = np.concatenate((f_bias[::-1], f_bias))

    # FFT of the even array (real)
    f_hat = rfft(f_even)

    # Dual r-grid
    x = np.arange(n)
    arg = np.pi * x / (n * dlnk)
    mask = arg < 690
    r = 1.0 / k[0] * np.exp(arg[mask])

    # Inverse FFT to get cosine coefficients
    c_even = irfft(f_hat)[:n][mask]

    # This is the correct final line
    C_r = (2.0 * np.pi) * r ** (-q - 1.0) * c_even

    return r, C_r


def _cosine_transform_fftlog_corrected(k: np.ndarray, f_k: np.ndarray, q: float = 0.0):
    """
    Simplified and robust cosine-transform f(k) on a log grid via FFTLog.

    Based on the standard FFTLog algorithm for cosine transforms.
    """
    n = len(k)

    # Logarithmic spacing
    ln_k = np.log(k)
    dlnk = ln_k[1] - ln_k[0]

    # Apply bias to input function
    f_biased = (k**q) * f_k

    # Simple real FFT approach for cosine transform
    # Pad with zeros to avoid edge effects
    f_padded = np.zeros(2 * n)
    f_padded[:n] = f_biased

    # Take real FFT
    f_hat = rfft(f_padded)

    # Construct r-grid (conjugate to k-grid)
    m = np.arange(n)
    ln_r = -ln_k[0] - (np.pi * m) / (n * dlnk)
    r = np.exp(ln_r)

    # Extract cosine coefficients and apply normalization
    # Take only the real part of the first n coefficients
    cosine_coeffs = np.real(f_hat[:n])

    # Apply proper normalization for cosine transform
    # This includes the dlnk factor and the inverse bias
    C_r = (np.pi * dlnk) * (r ** (-q)) * cosine_coeffs

    # Ensure r is sorted in ascending order
    if not np.all(np.diff(r) >= 0):
        sort_idx = np.argsort(r)
        r = r[sort_idx]
        C_r = C_r[sort_idx]

    # Remove any invalid values
    valid_mask = np.isfinite(r) & np.isfinite(C_r) & (r > 0)
    r = r[valid_mask]
    C_r = C_r[valid_mask]

    return r, C_r


def _cosine_transform_fftlog_robust(k: np.ndarray, f_k: np.ndarray, q: float = 0.0):
    """
    Robust FFTLog cosine transform based on Hamilton (2000).

    Computes: ∫₀^∞ f(k) cos(kr) k dk
    """
    n = len(k)

    # Ensure logarithmic spacing
    ln_k = np.log(k)
    dlnk = ln_k[1] - ln_k[0]

    # Central point for FFTLog
    ln_k_c = 0.5 * (ln_k[0] + ln_k[-1])
    k_c = np.exp(ln_k_c)

    # Apply bias
    f_biased = (k**q) * f_k

    # FFTLog frequency array
    m = np.arange(n) - n // 2
    eta_m = 2j * np.pi * m / (n * dlnk)

    # Apply FFTLog kernel
    kernel = np.exp(eta_m * (ln_k - ln_k_c))
    f_kernel = f_biased * kernel

    # Pad to avoid wraparound
    f_padded = np.zeros(2 * n, dtype=complex)
    f_padded[:n] = f_kernel

    # FFT
    f_hat = fft(f_padded)[:n]

    # Conjugate r coordinate
    ln_r = -ln_k_c - eta_m.real * dlnk
    r = np.exp(ln_r)

    # Gamma function ratio for cosine transform
    # For cosine transform with bias q, we need Γ((1+q+iη)/2) / Γ((2+q+iη)/2)

    s_plus = (1 + q + 1j * eta_m.imag) / 2
    s_minus = (2 + q + 1j * eta_m.imag) / 2

    # Use log-gamma for numerical stability
    from scipy.special import loggamma

    log_gamma_ratio = loggamma(s_plus) - loggamma(s_minus)
    gamma_ratio = np.exp(log_gamma_ratio)

    # Final result
    prefactor = np.sqrt(np.pi) * dlnk * k_c ** (1 + q)
    C_r = prefactor * (r ** (-1 - q)) * gamma_ratio * f_hat

    # Take real part (cosine transform should be real)
    C_r = np.real(C_r)

    # Sort by r and remove invalid values
    valid_mask = np.isfinite(r) & np.isfinite(C_r) & (r > 0)
    r_valid = r[valid_mask]
    C_valid = C_r[valid_mask]

    sort_idx = np.argsort(r_valid)
    return r_valid[sort_idx], C_valid[sort_idx]


def _cosine_transform_mcfit_style(k: np.ndarray, f_k: np.ndarray, q: float = 0.0):
    """
    Alternative implementation following mcfit library approach.
    """
    n = len(k)
    ln_k = np.log(k)
    dlnk = ln_k[1] - ln_k[0]

    # Bias the function
    f_biased = k**q * f_k

    # Use real FFT for efficiency
    f_hat = rfft(f_biased, n=2 * n)

    # Frequency array for the transform
    m = np.arange(len(f_hat))
    omega = np.pi * m / (n * dlnk)

    # r-coordinate
    r = np.exp(-ln_k[0] - omega * dlnk)

    # Hankel transform kernel for j0 (cosine transform)
    # For j0 transform: H_ν = √(π/2) * Γ((μ+ν)/2) / Γ((μ-ν+1)/2)
    # where μ = 1+q, ν = 0 for j0
    from scipy.special import gamma

    mu = 1 + q
    nu = 0

    try:
        H = np.sqrt(np.pi / 2) * gamma((mu + nu) / 2) / gamma((mu - nu + 1) / 2)
        # For complex arguments, handle carefully
        if np.any(np.imag(omega) != 0):
            H = (
                np.sqrt(np.pi / 2)
                * gamma((mu + 1j * omega) / 2)
                / gamma((mu + 1 - 1j * omega) / 2)
            )
        else:
            H = np.sqrt(np.pi / 2) * gamma((mu) / 2) / gamma((mu + 1) / 2)
    except:
        # Fallback to simple normalization
        H = np.sqrt(np.pi / 2) * np.ones_like(omega)

    # Apply the transform
    C_r = dlnk * r ** (-q) * H * f_hat

    # Take real part and filter
    C_r = np.real(C_r)
    valid_mask = np.isfinite(r) & np.isfinite(C_r) & (r > 0)

    return r[valid_mask], C_r[valid_mask]


def _cosine_transform_simple_stable(k: np.ndarray, f_k: np.ndarray, q: float = 0.0):
    """
    Simplified but stable cosine transform.
    """
    n = len(k)
    ln_k = np.log(k)
    dlnk = ln_k[1] - ln_k[0]

    # Apply taper to reduce ringing
    taper = np.exp(-(((ln_k - ln_k.mean()) / 4) ** 2))  # Gaussian taper
    f_tapered = (k**q) * f_k * taper

    # Simple real FFT
    f_hat = rfft(f_tapered, n=4 * n)  # Zero-pad for smoother result

    # r-grid
    m = np.arange(len(f_hat))
    r = np.exp(-ln_k[0] - np.pi * m / (2 * n * dlnk))

    # Simple normalization
    C_r = (2 * dlnk) * (r ** (-q)) * np.real(f_hat)

    # Filter and sort
    valid = (r > 0) & np.isfinite(C_r) & (np.abs(C_r) > 1e-30)
    return r[valid], C_r[valid]


def _cosine_transform_fftlog_v2(k: np.ndarray, f_k: np.ndarray, q: float = 0.0):
    """
    Alternative FFTLog cosine transform implementation.
    Based on Hamilton (2000) FFTLog algorithm.
    """
    n = len(k)
    ln_k = np.log(k)
    dlnk = ln_k[1] - ln_k[0]
    ln_k_c = ln_k[0] + (n - 1) * dlnk / 2.0  # Central k

    # Bias the input function
    f_biased = k**q * f_k

    # Phase factors for FFTLog
    m = np.arange(-n // 2, n // 2)
    arg = 2j * np.pi * m / (2 * n)

    # Extend and pad for proper FFT
    f_pad = np.zeros(2 * n, dtype=complex)
    f_pad[:n] = f_biased

    # Apply FFTLog phases
    f_pad *= np.exp(arg[None] * (ln_k[:, None] - ln_k_c))

    # Take FFT
    f_hat = fft(f_pad, axis=0)

    # Extract result and apply phases
    ln_r = -ln_k_c - arg.real * dlnk
    r = np.exp(ln_r[:n])

    # Final result with proper normalization
    result = (2 * np.pi) * dlnk * r ** (-q) * np.real(f_hat[:n])

    # Sort by r
    sort_idx = np.argsort(r)
    return r[sort_idx], result[sort_idx]


def sigma2_z2_func_vectorized_fft(
    z1_arr: np.ndarray,
    z2: float,
    k_grid_sigma2: np.ndarray,
    cosmo_ccl,  # ccl.Cosmology
    which_sigma2_b: str,
    ell_mask: np.ndarray,
    cl_mask: np.ndarray,
    fsky_mask: float,
    n_jobs=None,
    *,
    k_min: float | None = None,
    k_max: float | None = None,
    n_k_log: int = 16384,
    q_bias: float = 1.0,  # Changed default from 1.5 to 0.0
):
    """
    Corrected sigma2b calculation using FFTLog.
    """
    print('in corrected fft')

    # ---------------- 1. Distances & growth ----------------
    # Assuming you have a cosmo_lib module with z_to_a function
    # If not, use: a = 1.0 / (1.0 + z)
    a1 = 1.0 / (1.0 + z1_arr)  # cosmo_lib.z_to_a(z1_arr)
    a2 = 1.0 / (1.0 + z2)  # cosmo_lib.z_to_a(z2)

    # Import ccl if not already available
    import pyccl as ccl

    chi1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
    chi2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    g1 = ccl.growth_factor(cosmo_ccl, a1)
    g2 = ccl.growth_factor(cosmo_ccl, a2)

    # ---------------- 2. k‑grid & P(k,0) -------------------
    k_min = k_min or max(1e-6, k_grid_sigma2.min() / 50)
    k_max = k_max or k_grid_sigma2.max() * 50

    k = np.logspace(np.log10(k_min), np.log10(k_max), n_k_log)
    Pk0 = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)

    # ---------------- 3. Try multiple FFTLog implementations  -----------------
    # Try different implementations in order of sophistication
    transform_success = False

    for method_name, transform_func in [
        ('simple_stable', _cosine_transform_simple_stable),
        ('mcfit_style', _cosine_transform_mcfit_style),
        ('robust', _cosine_transform_fftlog_robust),
    ]:
        try:
            print(f'Trying FFTLog method: {method_name}')
            r_grid, C_r = transform_func(k, Pk0, q=q_bias)
            if len(r_grid) > 10 and np.all(np.isfinite(C_r)):
                print(f'Success with method: {method_name}')
                transform_success = True
                break
        except Exception as e:
            print(f'Method {method_name} failed: {e}')
            continue

    if not transform_success:
        raise RuntimeError('All FFTLog methods failed!')

    # Add r=0 analytically (for χ1=χ2 case) - but check for duplicates
    C0 = np.trapz(Pk0, k)

    # Check if r=0 already exists in r_grid
    if len(r_grid) > 0 and np.min(r_grid) < 1e-10:
        # r=0 or very close to 0 already exists, replace it
        idx_zero = np.argmin(r_grid)
        r_all = r_grid.copy()
        C_all = C_r.copy()
        r_all[idx_zero] = 0.0
        C_all[idx_zero] = C0
    else:
        # r=0 doesn't exist, add it
        r_all = np.concatenate([[0.0], r_grid])
        C_all = np.concatenate([[C0], C_r])

    # Remove any duplicate r values and sort
    # Use a small tolerance for "duplicate" detection
    r_rounded = np.round(r_all / (1e-12 * np.max(r_all))) * (1e-12 * np.max(r_all))
    unique_indices = np.unique(r_rounded, return_index=True)[1]
    r_unique = r_all[unique_indices]
    C_unique = C_all[unique_indices]

    # Sort by r values
    sort_idx = np.argsort(r_unique)
    r_sorted = r_unique[sort_idx]
    C_sorted = C_unique[sort_idx]

    # Debug: print grid info
    print(
        f'r_grid range: [{np.min(r_grid):.2e}, {np.max(r_grid):.2e}], n={len(r_grid)}'
    )
    print(
        f'r_sorted range: [{np.min(r_sorted):.2e}, {np.max(r_sorted):.2e}], n={len(r_sorted)}'
    )
    print(f'C_sorted range: [{np.min(C_sorted):.2e}, {np.max(C_sorted):.2e}]')

    # Create interpolator with better extrapolation
    C = interp1d(
        r_sorted,
        C_sorted,
        kind='linear',  # Use linear for robustness
        bounds_error=False,
        fill_value=(C0, 0.0),
        assume_sorted=True,
    )

    # ---------------- 4. Assemble integral -----------------
    r_plus = chi1 + chi2
    r_minus = np.abs(chi1 - chi2)

    # Key correction: check for proper normalization
    # The factor should account for the j0(kr1)j0(kr2) -> cosine conversion
    integral = 0.5 / (chi1 * chi2) * (C(r_minus) - C(r_plus))

    # ---------------- 5. Geometry options ------------------
    if which_sigma2_b == 'full_curved_sky':
        # Make sure the normalization is consistent with Simpson's rule
        result = (g1 * g2) * integral / (2.0 * np.pi**2)

    elif which_sigma2_b in {'polar_cap_on_the_fly', 'from_input_mask'}:
        summand = (2 * ell_mask + 1) * cl_mask * 2.0 / np.pi * g1[:, None] * g2
        summand *= integral[:, None]
        result = summand.sum(axis=1) / (4.0 * np.pi * fsky_mask) ** 2

    else:
        raise ValueError(
            'which_sigma2_b must be "full_curved_sky", '
            '"polar_cap_on_the_fly", or "from_input_mask".'
        )

    return result


import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy import special
import warnings


import numpy as np
from scipy.fft import rfft
from scipy.interpolate import interp1d
# You will need cosmo_lib for z_to_a from your original code
# import cosmo_lib
# You will need ccl for cosmology functions
# import pyccl as ccl


def sigma2_z2_func_vectorized_fft_linear(
    z1_arr: np.ndarray,
    z2: float,
    k_grid_sigma2: np.ndarray,
    cosmo_ccl: ccl.Cosmology,
    which_sigma2_b: str,
    ell_mask: np.ndarray,
    cl_mask: np.ndarray,
    fsky_mask: float,
    n_jobs=None,
    *,
    nk_fft: int = 2**18, # A higher number of points may be needed for precision
):
    """
    Computes sigma_b^2(z1,z2) using a direct FFT on a linear k-grid.
    This logic is based on your working sigma2_fullsky function.
    """
    # ---------------- 1. Distances & growth ----------------
    # This part is the same as your original function
    a1 = cosmo_lib.z_to_a(z1_arr)
    a2 = cosmo_lib.z_to_a(z2)

    chi1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
    chi2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    g1 = ccl.growth_factor(cosmo_ccl, a1)
    g2 = ccl.growth_factor(cosmo_ccl, a2)

    # ---------------- 2. k-grid & Power Spectrum -------------------
    # This logic is taken directly from your working sigma2_fullsky function
    k_min = k_grid_sigma2.min()
    k_max = k_grid_sigma2.max()
    
    # Create a LINEARLY spaced k-grid for the FFT
    k = np.linspace(k_min, k_max, nk_fft)
    dk = k[1] - k[0] # The linear step size, dk
    
    # Get the power spectrum on this linear grid
    Pk0 = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)

    # ---------------- 3. Cosine Transform via direct FFT -----------------
    # This is the core logic from your working function.
    
    # The r-grid in the dual space of the FFT
    delta_k_range = k_max - k_min
    r_grid = np.linspace(0, nk_fft // 2, nk_fft // 2 + 1) * 2 * np.pi / delta_k_range

    # Perform the FFT and multiply by the integration step `dk` to approximate the integral
    fft_coeffs = rfft(Pk0) * dk
    
    # The cosine transform is the real part of the FFT result
    C_r = fft_coeffs.real
    
    # Create an interpolator for C(r).
    # We calculate C(r=0) accurately with a separate integral for the boundary condition.
    C0 = np.trapz(Pk0, k)
    C_func = interp1d(r_grid, C_r, kind='cubic', bounds_error=False, fill_value=(C0, 0.0), assume_sorted=True)

    # ---------------- 4. Assemble integral -----------------
    # This part is the same as your original function
    r_plus = chi1 + chi2
    r_minus = np.abs(chi1 - chi2)

    integral = 0.5 / (chi1 * chi2) * (C_func(r_minus) - C_func(r_plus))

    # ---------------- 5. Geometry options ------------------
    # This part is the same as your original function
    if which_sigma2_b == 'full_curved_sky':
        result = (g1 * g2) * integral / (2.0 * np.pi**2)

    elif which_sigma2_b in {'polar_cap_on_the_fly', 'from_input_mask'}:
        summand = (2 * ell_mask + 1) * cl_mask * 2.0 / np.pi * g1[:, None] * g2
        summand *= integral[:, None]
        result = summand.sum(axis=1) / (4.0 * np.pi * fsky_mask) ** 2

    else:
        raise ValueError(
            'which_sigma2_b must be "full_curved_sky", '
            '"polar_cap_on_the_fly", or "from_input_mask".'
        )

    return result


def sigma2_z1z2_func_vectorized_fft_linear(
    z1_arr: np.ndarray,
    z2_arr: np.ndarray,
    k_grid_sigma2: np.ndarray,
    cosmo_ccl: ccl.Cosmology,
    which_sigma2_b: str,
    ell_mask: np.ndarray,
    cl_mask: np.ndarray,
    fsky_mask: float,
    *,
    nk_fft: int = 2 ** 18,
):
    """Fast—but lower‑accuracy—evaluation of σ²_b on a *linear* k grid.

    The routine is *fully* vectorised in both z₁ and z₂, returning an array of
    shape ``(len(z1_arr), len(z2_arr))``.  Use the high‑precision log‑k version
    for production results; this linear‑k path is mainly for quick checks.
    """

    z1_arr = np.atleast_1d(z1_arr)
    z2_arr = np.atleast_1d(z2_arr)

    # ---------- comoving distances & growth ----------
    a1, a2 = cosmo_lib.z_to_a(z1_arr), cosmo_lib.z_to_a(z2_arr)
    chi1 = ccl.comoving_radial_distance(cosmo_ccl, a1)          # (N1,)
    chi2 = ccl.comoving_radial_distance(cosmo_ccl, a2)          # (N2,)
    g1  = ccl.growth_factor(cosmo_ccl, a1)                      # (N1,)
    g2  = ccl.growth_factor(cosmo_ccl, a2)                      # (N2,)

    # ---------- linear k grid ----------
    k_min, k_max = k_grid_sigma2.min(), k_grid_sigma2.max()
    k = np.linspace(k_min, k_max, nk_fft)
    dk = k[1] - k[0]
    Pk0 = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)

    # real FFT → cosine coefficients on linear grid
    fft_coeffs = rfft(Pk0) * dk  # ∑ f(k) cos → Re{FFT} * dk
    r_grid = np.arange(fft_coeffs.size) * 2 * np.pi / (k_max - k_min)
    C_r = fft_coeffs.real

    # interpolate C(r)
    C0 = np.trapz(Pk0, k)
    C = interp1d(
        r_grid,
        C_r,
        kind="cubic",
        bounds_error=False,
        fill_value=(C0, 0.0),
        assume_sorted=True,
    )

    # ---------- assemble integral (broadcast over z₁, z₂) ----------
    chi1_mat, chi2_mat = chi1[:, None], chi2[None, :]
    r_plus   = chi1_mat + chi2_mat
    r_minus  = np.abs(chi1_mat - chi2_mat)
    integral = 0.5 / (chi1_mat * chi2_mat) * (C(r_minus) - C(r_plus))

    # ---------- geometry options ----------
    if which_sigma2_b == "full_curved_sky":
        return (g1[:, None] * g2[None, :]) * integral / (2.0 * np.pi ** 2)

    # if which_sigma2_b in {"polar_cap_on_the_fly", "from_input_mask"}:
    #     # Expand masks over (N1,N2)
    #     summand = (
    #         (2 * ell_mask + 1)[:, None] * cl_mask[:, None] * 2.0 / np.pi
    #     )  # (ℓ,1)
    #     summand = g1[:, None, None] * g2[None, None, :] * summand  # (N1,ℓ,N2)
    #     try:
    #         summand *= integral[:, None, :]  # broadcast integral along ℓ
    #     except MemoryError:
    #         for ell_ix in range(ell_mask.size):
    #             summand[:, ell_ix, :] *= integral[:, ell_ix, :]  # broadcast integral along ℓ
        
    #     return summand.sum(axis=1) / (4.0 * np.pi * fsky_mask) ** 2
            
    elif which_sigma2_b in {"polar_cap_on_the_fly", "from_input_mask"}:
        S = np.sum((2 * ell_mask + 1) * cl_mask) * 2.0 / np.pi
        result = (
            S * g1[:, None] * g2[None, :] * integral
        ) / (4.0 * np.pi * fsky_mask) ** 2
        return result
        

    raise ValueError("Invalid which_sigma2_b option.")



def sigma2_fullsky(z_grid, cosmo_ccl, h, which_sigma2_b, ell_mask, cl_mask, fsky):
    # Find number of redshifts
    nz = z_grid.size
    a_grid = 1.0 / (1.0 + z_grid)
    # Get cosmology, comoving distances etc from dedicated auxiliary routine
    # cosmo, h, comov_dist, dcomov_dist, growth = get_cosmo(
    #     z_grid, cosmo_params=cosmo_params, cosmo_Class=cosmo_Class
    # )
    comov_dist = ccl.comoving_radial_distance(cosmo_ccl, a_grid)
    growth = ccl.growth_factor(cosmo_ccl, a_grid)

    keq = 0.02 / h  # Equality matter radiation in 1/Mpc (more or less)
    klogwidth = 10  # Factor of width of the integration range.
    # 10 seems ok ; going higher needs to increase nk_fft to reach convergence (fine cancellation issue noted in Lacasa & Grain)
    kmin = min(keq, 1.0 / comov_dist.max()) / klogwidth
    kmax = max(keq, 1.0 / comov_dist.min()) * klogwidth
    nk_fft = (
        2**13
    )  # seems to be enough. Increase to test precision, reduce to speed up.
    k_4fft = np.linspace(
        kmin, kmax, nk_fft
    )  # linear grid on k, as we need to use an FFT
    Deltak = kmax - kmin
    Dk = Deltak / nk_fft
    Pk_4fft = np.zeros(nk_fft)
    for ik in range(nk_fft):
        # Pk_4fft[ik] = cosmo.pk(k_4fft[ik], 0.0)  # In Mpc^3
        Pk_4fft[ik] = ccl.linear_matter_power(cosmo_ccl, k=k_4fft[ik], a=1.0)

    dr_fft = np.linspace(0, nk_fft // 2, nk_fft // 2 + 1) * 2 * np.pi / Deltak

    # Compute necessary FFTs and make interpolation functions
    fft0 = np.fft.rfft(Pk_4fft) * Dk
    dct0 = fft0.real
    dst0 = -fft0.imag
    Pk_dct = interp1d(dr_fft, dct0, kind='cubic')

    # Compute sigma^2(z1,z2)
    sigma2 = np.zeros((nz, nz))
    result = np.zeros((nz, nz))
    # First without growth functions (i.e. with P(k,z=0)) and z1<=z2
    for iz in range(nz):
        r1 = comov_dist[iz]
        for jz in range(iz, nz):
            r2 = comov_dist[jz]
            rsum = r1 + r2
            rdiff = abs(r1 - r2)
            Icp0 = Pk_dct(rsum)
            Icm0 = Pk_dct(rdiff)
            sigma2[iz, jz] = (Icm0 - Icp0) / (4 * np.pi**2 * r1 * r2)

    # fill lower triangular part
    for iz in range(nz):
        for jz in range(iz, nz):
            sigma2[iz, jz] = sigma2[jz, iz]

    if which_sigma2_b == 'full_curved_sky':
        for iz in range(nz):
            for jz in range(nz):
                result = (
                    1
                    / (2 * np.pi**2)
                    * growth[iz]
                    * growth[jz]
                    * sigma2[iz, jz]
                )

    elif which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
        partial_summand = np.zeros((len(z_grid), len(ell_mask)))
        for iz in range(nz):
            for jz in range(nz):
                # NOTE: you should include a 2/np.pi factor, see Eq. (26)
                # of https://arxiv.org/pdf/1612.05958, or Champaghe et al 2017
                partial_summand[iz, jz] = (
                    (2 * ell_mask + 1)
                    * cl_mask
                    * 2
                    / np.pi
                    * growth[iz]
                    * growth[jz]
                )
                partial_summand *= sigma2[iz, jz]
                result[iz, jz] = np.sum(partial_summand, axis=1)
                one_over_omega_s_squared = 1 / (4 * np.pi * fsky) ** 2
                result[iz, jz] *= one_over_omega_s_squared


    return sigma2


def _spherical_bessel_transform_optimized(
    k: np.ndarray, P_k: np.ndarray, q: float = None
):
    """
    High-precision FFTLog implementation for ∫ k² P(k) j₀(kr) dk
    with automatic bias selection and improved numerical stability.
    """
    # Compute optimal bias if not specified
    if q is None:
        logk = np.log(k)
        logP = np.log(P_k)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            slope = np.gradient(logP, logk)
        avg_slope = np.nanmean(slope[-10:])  # Focus on high-k where precision matters
        q = max(0.7, min(1.3, -avg_slope / 2 + 0.5))

    n = len(k)
    lnk = np.log(k)
    dlnk = lnk[1] - lnk[0]
    k_c = np.exp(lnk.mean())

    # Apply bias and low-pass filter
    taper = np.exp(-(((lnk - lnk.mean()) / (3 * dlnk)) ** 4))  # Quartic taper
    f_biased = (k**q) * P_k * k**0.5 * taper  # Prepare for ν=0.5 transform

    # Zero-padding for smoother results
    f_padded = np.zeros(2 * n)
    f_padded[:n] = f_biased * k ** (-0.5)  # Final adjustment

    # Real FFT
    f_hat = rfft(f_padded)
    m = np.arange(len(f_hat))

    # Gamma kernel for ν=0.5 (spherical Bessel)
    eta = 2j * np.pi * m / (2 * n * dlnk)
    s = (3 / 2 + q + eta) / 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gammas = special.gamma(s) / special.gamma(3 / 2 - s)

    # r-space grid and result
    r = np.exp(-lnk[0] - np.pi * m / (n * dlnk))
    F_r = f_hat * gammas * (2 * np.pi) ** 0.5 * r ** (-3 - q)

    # Apply high-k damping
    decay = np.exp(-((r * k[-1] / 3) ** 2))
    result = np.real(F_r) * decay

    # Filter valid range
    valid = (r > 1e-4 / k[-1]) & (r < 1e2 / k[0]) & np.isfinite(result)
    return r[valid], result[valid]


def sigma2_z2_func_high_precision(
    z1_arr: np.ndarray,
    z2: float,
    k_grid_sigma2: np.ndarray,
    cosmo_ccl,
    which_sigma2_b: str,
    ell_mask: np.ndarray = None,
    cl_mask: np.ndarray = None,
    fsky_mask: float = None,
    n_jobs=None,
    *,
    k_min: float = None,
    k_max: float = None,
    n_k_log: int = 4096,
    q_bias: float = None,  # None for auto
):
    """
    High-precision sigma2(z1,z2) calculation with optimized FFTLog.
    """
    # ================= 1. Cosmology Setup =================
    a1 = 1.0 / (1.0 + z1_arr)
    a2 = 1.0 / (1.0 + z1_arr)

    chi1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
    chi2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    g1 = ccl.growth_factor(cosmo_ccl, a1)
    g2 = ccl.growth_factor(cosmo_ccl, a2)

    # ================= 2. Optimized k-grid =================
    k_min = k_min or max(1e-5, 0.1 * k_grid_sigma2.min())
    k_max = k_max or min(1e3, 10 * k_grid_sigma2.max())

    k = np.geomspace(k_min, k_max, n_k_log * 2)  # Higher resolution
    Pk0 = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)

    # ================= 3. High-precision Transform =================
    r_grid, C_r = _spherical_bessel_transform_optimized(k, Pk0, q=q_bias)

    # Add r=0 point analytically
    C0 = np.trapz(Pk0, k)
    if len(r_grid) == 0 or r_grid[0] > 1e-6:
        r_grid = np.concatenate([[0.0], r_grid])
        C_r = np.concatenate([[C0], C_r])
    else:
        C_r[0] = C0  # Replace near-zero point

    # ================= 4. Enhanced Interpolation =================
    # Sort and remove duplicates
    sort_idx = np.argsort(r_grid)
    r_sorted = r_grid[sort_idx]
    C_sorted = C_r[sort_idx]

    # PCHIP interpolation for monotonicity
    C_interp = PchipInterpolator(r_sorted, C_sorted, extrapolate=False)

    def safe_C(r):
        """Handle extrapolation and edge cases"""
        r_clip = np.clip(r, r_sorted[0], r_sorted[-1])
        result = C_interp(r_clip)
        result[r < r_sorted[0]] = C0
        result[r > r_sorted[-1]] = 0.0
        return result

    # ================= 5. Precision Integral Calculation =================
    r_minus = np.abs(chi1 - chi2)
    r_plus = chi1 + chi2

    # Avoid singularities
    epsilon = 1e-10
    r_minus = np.maximum(r_minus, epsilon)
    r_plus = np.maximum(r_plus, epsilon)

    # Compute integral with special handling near diagonal
    integral = np.zeros_like(r_minus)
    for i in range(len(r_minus)):
        if abs(r_minus[i] - r_plus[i]) < 1e-2 * (r_minus[i] + r_plus[i]):
            # Series expansion for χ1 ≈ χ2
            r_avg = 0.5 * (r_minus[i] + r_plus[i])
            dr = r_plus[i] - r_minus[i]
            h = 1e-4 * r_avg
            dCdr = (safe_C(r_avg + h) - safe_C(r_avg - h)) / (2 * h)
            integral[i] = -dr * dCdr / (chi1[i] * chi2[i])
        else:
            # Standard calculation
            integral[i] = (
                0.5 / (chi1[i] * chi2[i]) * (safe_C(r_minus[i]) - safe_C(r_plus[i]))
            )

    # ================= 6. Geometry Options =================
    if which_sigma2_b == 'full_curved_sky':
        result = (g1 * g2) * integral / (2.0 * np.pi**2)
    elif which_sigma2_b in {'polar_cap_on_the_fly', 'from_input_mask'}:
        summand = (2 * ell_mask + 1) * cl_mask * 2.0 / np.pi * g1[:, None] * g2
        summand *= integral[:, None]
        result = summand.sum(axis=1) / (4.0 * np.pi * fsky_mask) ** 2
    else:
        raise ValueError('Invalid which_sigma2_b option')

    return result


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
