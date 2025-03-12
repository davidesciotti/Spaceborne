import time

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn
from tqdm import tqdm

import pyccl as ccl
from spaceborne import cosmo_lib, mask_utils
from spaceborne import sb_lib as sl
import pylevin as levin


# * pylevin hyperparameters
n_sub = 12  # number of collocation points in each bisection
n_bisec_max = 32  # maximum number of bisections used
rel_acc = 1e-5  # relative accuracy target
boost_bessel = True  # should the bessel functions be calculated with boost instead of GSL, higher accuracy at high Bessel orders
verbose = False  # should the code talk to you?
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


# This is defined globally since
COSMO_CCL = None


def init_cosmo(cosmo):
    global COSMO_CCL
    COSMO_CCL = cosmo


def sigma2_z1z2_wrap(
    z_grid,
    k_grid_sigma2,
    cosmo_ccl,
    which_sigma2_b,
    area_deg2_in,
    nside_mask,
    mask_path,
):
    fsky_in = cosmo_lib.deg2_to_fsky(area_deg2_in)
    if which_sigma2_b == 'full_curved_sky':
        ell_mask = None
        cl_mask = None
        # fsky_mask is not needed in this case, the whole covariance
        # is normalized at the end of the computation
        fsky_mask = None

    elif which_sigma2_b == 'polar_cap_on_the_fly':
        mask = mask_utils.generate_polar_cap(area_deg2_in, nside_mask)

    elif which_sigma2_b == 'from_input_mask':
        mask = hp.read_map(mask_path)

    if which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
        hp.mollview(
            mask,
            coord=['C', 'E'],
            title='polar cap generated on-the fly',
            cmap='inferno_r',
        )
        cl_mask = hp.anafast(mask)
        ell_mask = np.arange(len(cl_mask))
        # quick check
        fsky_mask = np.sqrt(cl_mask[0] / (4 * np.pi))
        print(f'fsky from mask: {fsky_mask:.4f}')
        assert np.abs(fsky_mask / fsky_in) < 1.01, (
            'fsky_in is not the same as the fsky of the mask'
        )

    start = time.perf_counter()
    sigma2_b = np.zeros((len(z_grid), len(z_grid)))
    for z2_idx, z2 in enumerate(tqdm(z_grid)):
        sigma2_b[:, z2_idx] = sigma2_z2_func_vectorized(
            z1_arr=z_grid,
            z2=z2,
            k_grid_sigma2=k_grid_sigma2,
            cosmo_ccl=cosmo_ccl,
            which_sigma2_b=which_sigma2_b,
            ell_mask=ell_mask,
            cl_mask=cl_mask,
            fsky_mask=fsky_in,
        )
    print(f'done in {time.perf_counter() - start} s')

    return sigma2_b


def sigma2_z1z2_wrap_parallel(
    z_grid: np.ndarray,
    k_grid_sigma2: np.ndarray,
    cosmo_ccl: ccl.Cosmology,
    which_sigma2_b: str,
    area_deg2_in: float | int,
    nside_mask: int,
    mask_path: str,
    n_jobs: int,
    parallel: bool = True,
    integration_scheme: str = 'simps',
) -> np.ndarray:
    """
    Parallelized version of sigma2_z1z2_wrap using joblib.
    """
    fsky_in = cosmo_lib.deg2_to_fsky(area_deg2_in)

    # Handle mask-related computations
    if which_sigma2_b == 'full_curved_sky':
        ell_mask = None
        cl_mask = None
        fsky_mask = None  # Not needed in this case

    elif which_sigma2_b == 'polar_cap_on_the_fly':
        mask = mask_utils.generate_polar_cap(area_deg2_in, nside_mask)

    elif which_sigma2_b == 'from_input_mask':
        mask = hp.read_map(mask_path)

    if which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
        hp.mollview(
            mask,
            coord=['C', 'E'],
            title='polar cap generated on-the fly',
            cmap='inferno_r',
        )
        cl_mask = hp.anafast(mask)
        ell_mask = np.arange(len(cl_mask))
        # Quick check
        fsky_mask = np.sqrt(cl_mask[0] / (4 * np.pi))
        print(f'fsky from mask: {fsky_mask:.4f}')
        assert np.abs(fsky_mask / fsky_in) < 1.01, (
            'fsky_in is not the same as the fsky of the mask'
        )

    print('Computing sigma^2_b(z_1, z_2). This may take a while...')
    start = time.perf_counter()

    if parallel and integration_scheme == 'simps':
        from pathos.multiprocessing import ProcessingPool as Pool

        # Create a list of arguments—one per z2 value in z_grid
        # Build the argument list without cosmo_ccl:
        arg_list = [
            (z2, z_grid, k_grid_sigma2, which_sigma2_b, ell_mask, cl_mask, fsky_in)
            for z2 in z_grid
        ]

        # Create a Pathos ProcessingPool and initialize each worker:
        start = time.perf_counter()
        pool = Pool(n_jobs, initializer=init_cosmo, initargs=(cosmo_ccl,))
        sigma2_b_list = pool.map(pool_compute_sigma2_b, arg_list)

        # Convert the list of results to a numpy array and transpose
        sigma2_b = np.array(sigma2_b_list).T

    # davide's implementation
    elif parallel and integration_scheme == 'levin':
        sigma2_b = np.zeros((len(z_grid), len(z_grid)))
        for z2_idx, z2 in enumerate(tqdm(z_grid)):
            sigma2_b[:, z2_idx] = sigma2_z2_func_vectorized(
                z1_arr=z_grid,
                z2=z2,
                k_grid_sigma2=k_grid_sigma2,
                cosmo_ccl=cosmo_ccl,
                which_sigma2_b=which_sigma2_b,
                ell_mask=ell_mask,
                cl_mask=cl_mask,
                fsky_mask=fsky_in,
                integration_scheme=integration_scheme,
                n_jobs=n_jobs,
            )

    # robert's implementation
    # vectorized - parallel must be False, no for loop - still to be tested
    elif not parallel and integration_scheme == 'levin':
        # sigma2_b = sigma2_b_levin(
        #     z_grid,
        #     z_grid,
        #     k_grid_sigma2,
        #     cosmo_ccl,
        #     which_sigma2_b,
        #     ell_mask,
        #     cl_mask,
        #     fsky_mask,
        # )
        sigma2_b = sigma2_b_levin_batched(
            z_grid=z_grid,
            k_grid=k_grid_sigma2,
            cosmo_ccl=cosmo_ccl,
            which_sigma2_b=which_sigma2_b,
            ell_mask=ell_mask,
            cl_mask=cl_mask,
            fsky_mask=fsky_mask,
            batch_size=100_000,
        )

    else:
        raise ValueError('Invalid combination of "parallel" and "integration_scheme". ')

    print(f'done in {time.perf_counter() - start} s')

    return sigma2_b


def compute_sigma2_b(
    z2, z_grid, k_grid_sigma2, which_sigma2_b, ell_mask, cl_mask, fsky_in
):
    """
    Wrapper for sigma2_z2_func_vectorized without the cosmo_ccl argument.
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


def sigma2_b_levin(  # fmt: skip
    z1_arr, z2_arr, k_grid_sigma2, cosmo_ccl, which_sigma2_b, 
    ell_mask, cl_mask, fsky_mask,
):  # fmt: skip
    """
    In this version, I try to vectorize in zi, z2. It still doesn't work, though.
    """

    if not np.all(z1_arr == z2_arr):
        raise ValueError('z1_arr and z2_arr must be the same')

    a1_arr = cosmo_lib.z_to_a(z1_arr)
    a2_arr = cosmo_lib.z_to_a(z2_arr)

    r1_arr = ccl.comoving_radial_distance(cosmo_ccl, a1_arr)
    r2_arr = ccl.comoving_radial_distance(cosmo_ccl, a2_arr)

    growth_factor_z1_arr = ccl.growth_factor(cosmo_ccl, a1_arr)
    growth_factor_z2_arr = ccl.growth_factor(cosmo_ccl, a2_arr)

    plin = ccl.linear_matter_power(cosmo_ccl, k=k_grid_sigma2, a=1.0)

    integrand = (
        k_grid_sigma2[:, None, None] ** 2
        * plin[:, None, None]
        * growth_factor_z1_arr[None, :, None]
        * growth_factor_z2_arr[None, None, :]
    )

    import pylevin as levin

    integral_type = 2  # double spherical
    N_thread = 16  # Number of threads used for hyperthreading  # TODO increase

    # My implementation
    """
    # Constructor of the class
    lp_double = levin.pylevin(
        type=integral_type,
        x=k_grid_sigma2,
        integrand=integrand,
        logx=logx,
        logy=logy,
        nthread=N_thread,
    )

    # accuracy settings
    # should the bessel functions be calculated with boost instead of GSL,
    # higher accuracy at high Bessel orders
    lp_double.set_levin(
        n_col_in=n_sub,
        maximum_number_bisections_in=n_bisec_max,
        relative_accuracy_in=rel_acc,
        super_accurate=boost_bessel,
        verbose=verbose,
    )

    M = len(r1_arr)  # number of arguments at which the integrals are evaluated
    N = len(r2_arr)
    N = 1  # number of integrals to be calculated
    diagonal = False  # If diagonal == True it must have shape (N).

    result_levin = np.zeros((N,)) if diagonal else np.zeros((M, N))

    lp_double.levin_integrate_bessel_double(
        x_min=k_grid_sigma2[0] * np.ones(M),
        x_max=k_grid_sigma2[-1] * np.ones(M),
        k_1=r1_arr,
        k_2=r2_arr,
        ell_1=(0 * np.ones(M)).astype(int),
        ell_2=(0 * np.ones(M)).astype(int),
        diagonal=diagonal,
        result=result_levin,
    )
    """

    # Robert's implementation
    """
    N_z = len(r1_arr)
    result_levin = np.zeros(N_z)
    lower_limit = k_grid_sigma2[0] * np.ones(N_z)
    upper_limit = k_grid_sigma2[-1] * np.ones(N_z)
    ell = np.zeros(N_z).astype(int)
    diagonals_only = True

    result = np.zeros((N_z, N_z))

    t0 = time.time()
    for i_chi_1, chi_1 in tqdm(enumerate(r1_arr)):
        lp = levin.pylevin(
            integral_type,
            k_grid_sigma2,
            integrand[:, i_chi_1, i_chi_1:],
            logx,
            logy,
            N_thread,
        )
        lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)
        lp.levin_integrate_bessel_double(
            lower_limit[i_chi_1:],
            upper_limit[i_chi_1:],
            chi_1 * np.ones_like(r1_arr[i_chi_1:]),
            r2_arr[i_chi_1:],
            ell[i_chi_1:],
            ell[i_chi_1:],
            diagonals_only,
            result_levin[i_chi_1:],
        )
        result[i_chi_1, i_chi_1:] = result_levin[i_chi_1:]
        result[i_chi_1:, i_chi_1] = result_levin[i_chi_1:]
    print('Levin took', time.time() - t0, 's')
    """

    # Robert's implementation, double vec

    N_z = len(r1_arr)
    triu_indices = np.triu_indices(N_z)
    n_upper = len(triu_indices[0])  # Number of unique integrals to compute

    # Extract the corresponding integrand slices: shape becomes (nk, n_upper)
    triu_integrand = integrand[:, triu_indices[0], triu_indices[1]]

    # Build evaluation arrays for chi:
    chi1_flat = r1_arr[triu_indices[0]]  # shape (n_upper,)
    chi2_flat = r2_arr[triu_indices[1]]  # shape (n_upper,)

    # Define integration limits and orders for each integral
    lower_limit = k_grid_sigma2[0] * np.ones(n_upper)
    upper_limit = k_grid_sigma2[-1] * np.ones(n_upper)
    ell = np.zeros(n_upper, dtype=int)

    lower_limit = np.array(lower_limit).reshape(-1, 1)
    upper_limit = np.array(upper_limit).reshape(-1, 1)
    chi1_flat = np.array(chi1_flat).reshape(-1, 1)
    chi2_flat = np.array(chi2_flat).reshape(-1, 1)
    ell = np.array(ell, dtype=np.int32).reshape(-1, 1)
    result_flat = np.zeros(n_upper)

    # Create the pylevin instance with the flattened integrand
    lp = levin.pylevin(
        integral_type, k_grid_sigma2, triu_integrand, logx, logy, N_thread, True
    )
    lp.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)

    t0 = time.time()
    lp.levin_integrate_bessel_double(
        lower_limit, upper_limit, chi1_flat, chi2_flat, ell, ell, result_flat
    )
    print('Levin took', time.time() - t0, 's')

    # Assemble the full symmetric matrix from the upper-triangular results
    result_matrix = np.zeros((N_z, N_z))
    result_matrix[triu_indices] = result_flat
    # Fill in the lower triangle by symmetry
    result_matrix = result_matrix + result_matrix.T - np.diag(np.diag(result_matrix))

    integral_result = result_matrix

    if which_sigma2_b == 'full_curved_sky':
        result = 1 / (2 * np.pi**2) * integral_result

    elif (
        which_sigma2_b == 'polar_cap_on_the_fly' or which_sigma2_b == 'from_input_mask'
    ):
        # partial_summand = np.zeros((len(z1_arr), len(z1_arr), len(ell_mask)))
        # NOTE: you should include a 2/np.pi factor, see Eq. (26)
        # of https://arxiv.org/pdf/1612.05958, or Champaghe et al 2017
        partial_summand = (2 * ell_mask + 1) * cl_mask * 2 / np.pi
        partial_summand = integral_result[:, :, None] * partial_summand[None, None, :]
        result = np.sum(partial_summand, axis=-1)
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


def sigma2_b_levin_batched(
    z_grid: np.ndarray,
    k_grid: np.ndarray,
    cosmo_ccl: ccl.Cosmology,
    which_sigma2_b: str,
    ell_mask: np.ndarray,
    cl_mask: np.ndarray,
    fsky_mask: float,
    batch_size: int = 100_000,
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
    batch_size : int, optional
        Batch size for the computation. Default is 100_000.

    Returns
    -------
    np.ndarray
        Array of sigma2_b values.
    """

    a_arr = cosmo_lib.z_to_a(z_grid)
    r_arr = ccl.comoving_radial_distance(cosmo_ccl, a_arr)
    growth_factor_arr = ccl.growth_factor(cosmo_ccl, a_arr)
    plin = ccl.linear_matter_power(cosmo_ccl, k=k_grid, a=1.0)

    integral_type = 2  # double spherical
    N_thread = 16  # Number of threads used for hyperthreading

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

    elif (
        which_sigma2_b == 'polar_cap_on_the_fly' or which_sigma2_b == 'from_input_mask'
    ):
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
    N_thread = n_jobs  # Number of threads used for hyperthreading  # TODO increase
    logx = True  # Tells the code to create a logarithmic spline in x for f(x)
    logy = True  # Tells the code to create a logarithmic spline in y for y = f(x)
    diagonal = False
    lp_double = levin.pylevin(
        type=integral_type,
        x=k_grid_sigma2,
        integrand=integrand,
        logx=logx,  # TODO do I need to set these to True?
        logy=logy,  # TODO do I need to set these to True?
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

        plt.plot(
            z_grid_sigma2, sigma2_arr[z1_idx, :], label='$z_1={:.2f}$ '.format(z_1)
        )
        plt.axvline(z_1, color='k', ls='--', label='$z_1$')
    plt.xlabel('$z_2$')
    plt.ylabel('$\\sigma^2(z_1, z_2)$')  # sigma2 is dimensionless!
    plt.legend()
    plt.show()

    font_size = 18
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['legend.fontsize'] = font_size
    sl.matshow(sigma2_arr, log=True, abs_val=True, title='$\\sigma^2(z_1, z_2)$')


def sigma2_pyssc(z_arr, classy_cosmo_param_dict):
    """
    Compute sigma2 with PySSC. This is just for comparison, it is not used in the code.
    Note that zmin=1e-3 as zmin gives errors in classy, probably need to increse pk_max
    """
    import PySSC

    return PySSC.sigma2_fullsky(
        z_arr, cosmo_params=classy_cosmo_param_dict, cosmo_Class=None
    )
