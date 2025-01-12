import logging
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson as simps
from scipy.special import spherical_jn
import pyccl as ccl
from tqdm import tqdm
import healpy as hp


import os
ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import spaceborne.sb_lib as sl
import spaceborne.cosmo_lib as csmlib
import spaceborne.mask_utils

start_time = time.perf_counter()


# TODO finish implementing this function and test if if needed
# def sigma2_flatsky(z1, z2, k_perp_grid, k_par_grid, cosmo_ccl, Omega_S, theta_S):
#     """Compute the flatsky variance between two redshifts z1 and z2 for a cosmology given by cosmo_ccl."""

#     # Compute the comoving distance at the given redshifts
# from scipy.special import j1 as J1
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
#         bessel_term = J1(k_perp * theta_S * r1) * J1(k_perp * theta_S * r2) / (k_perp * theta_S * r1 * k_perp * theta_S * r2)
#         power_spectrum = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)
#         return k_perp * bessel_term * np.cos(k_par * (r1 - r2)) * power_spectrum

#     # Perform the double integral using Simpson's rule
#     integral_result_k_perp = np.array([
#         simps(integrand(k_perp, k_par_grid, r1, r2, theta_S), k_par_grid)
#         for k_perp in k_perp_grid
#     ])
#     integral_result = simps(integral_result_k_perp, k_perp_grid)

#     # Compute the final result
#     sigma2 = 1 / (2 * np.pi**2) * growth_factor_z1 * growth_factor_z2 * integral_result / Omega_S**2

#     return sigma2


def sigma2_func(z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_b, ell_mask=None, cl_mask=None):
    """ Computes the integral in k. The rest is in another function, to vectorize the call to the growth_factor.
    Note that the 1/Omega_S^2 factors are missing in this function!! This is consistent with the definitio given in
    mine and Fabien's paper."""

    # compute the comoving distance at the given redshifts
    a1 = csmlib.z_to_a(z1)
    a2 = csmlib.z_to_a(z2)

    # in Mpc
    r1 = ccl.comoving_radial_distance(cosmo_ccl, a1)
    r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    # compute the growth factors at the given redshifts
    growth_factor_z1 = ccl.growth_factor(cosmo_ccl, a1)
    growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)

    # define the integrand as a function of k
    def integrand(k): return k ** 2 * ccl.linear_matter_power(cosmo_ccl, k=k, a=1.) * \
        spherical_jn(0, k * r1) * spherical_jn(0, k * r2)

    integral_result = simps(y=integrand(k_grid_sigma2), x=k_grid_sigma2)

    # different integration methods; simps seems to be the best
    # if integrating_funct == 'simps':
    #     integral_result = simps(y=integrand(k_grid_sigma2), x=k_grid_sigma2)
    # elif integrating_funct == 'quad':
    #     integral_result = quad(integrand, k_grid_sigma2[0], k_grid_sigma2[-1])[0]
    # elif integrating_funct == 'quad_vec':
    #     integral_result = quad_vec(integrand, k_grid_sigma2[0], k_grid_sigma2[-1])[0]
    # else:
    #     raise ValueError('sigma2_integrating_function must be either "simps" or "quad" or "quad_vec"')

    if which_sigma2_b == 'full_curved_sky':
        result = 1 / (2 * np.pi ** 2) * growth_factor_z1 * growth_factor_z2 * integral_result
    elif which_sigma2_b == 'mask':
        fsky = np.sqrt(cl_mask[0] / (4 * np.pi))
        result = 1 / (4 * np.pi * fsky)**2 * np.sum((2 * ell_mask + 1) * cl_mask * 2 /
                                                    np.pi * growth_factor_z1 * growth_factor_z2 * integral_result)
    else:
        raise ValueError('which_sigma2_b must be either "full_curved_sky" or "mask"')

    return result


def sigma2_z1z2_wrap(z_grid_ssc_integrands, k_grid_sigma2, cosmo_ccl, which_sigma2_b,
                     area_deg2_in, nside_mask, mask_path):

    fsky_in = csmlib.deg2_to_fsky(area_deg2_in)
    if which_sigma2_b == 'full_curved_sky':
        ell_mask = None
        cl_mask = None
        fsky_mask = None  # not needed in this case, the whole covariance is normalized at the end of the computation

    elif which_sigma2_b == 'polar_cap_on_the_fly':
        mask = mask_utils.generate_polar_cap(area_deg2_in, nside_mask)

    elif which_sigma2_b == 'from_input_mask':
        mask = hp.read_map(mask_path)

    if which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
        hp.mollview(mask, coord=['C', 'E'], title='polar cap generated on-the fly', cmap='inferno_r')
        cl_mask = hp.anafast(mask)
        ell_mask = np.arange(len(cl_mask))
        # quick check
        fsky_mask = np.sqrt(cl_mask[0] / (4 * np.pi))
        print(f'fsky from mask: {fsky_mask:.4f}')
        assert np.abs(fsky_mask / fsky_in) < 1.01, 'fsky_in is not the same as the fsky of the mask'

    sigma2_b = np.zeros((len(z_grid_ssc_integrands), len(z_grid_ssc_integrands)))
    for z2_idx, z2 in enumerate(tqdm(z_grid_ssc_integrands)):
        sigma2_b[:, z2_idx] = sigma2_z2_func_vectorized(
            z1_arr=z_grid_ssc_integrands,
            z2=z2,
            k_grid_sigma2=k_grid_sigma2,
            cosmo_ccl=cosmo_ccl,
            which_sigma2_b=which_sigma2_b,
            ell_mask=ell_mask,
            cl_mask=cl_mask,
            fsky_mask=fsky_in
        )

    return sigma2_b


def sigma2_z2_func_vectorized(z1_arr, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_b, ell_mask, cl_mask, fsky_mask):
    """
    Vectorized version of sigma2_func in z1.
    """

    a1_arr = csmlib.z_to_a(z1_arr)
    a2 = csmlib.z_to_a(z2)

    r1_arr = ccl.comoving_radial_distance(cosmo_ccl, a1_arr)
    r2 = ccl.comoving_radial_distance(cosmo_ccl, a2)

    growth_factor_z1_arr = ccl.growth_factor(cosmo_ccl, a1_arr)
    growth_factor_z2 = ccl.growth_factor(cosmo_ccl, a2)

    # Define the integrand as a function of k
    def integrand(k):
        return k ** 2 * ccl.linear_matter_power(cosmo_ccl, k=k, a=1.) * \
            spherical_jn(0, k * r1_arr[:, None]) * spherical_jn(0, k * r2)

    integral_result = simps(y=integrand(k_grid_sigma2), x=k_grid_sigma2, axis=1)

    if which_sigma2_b == 'full_curved_sky':
        result = 1 / (2 * np.pi ** 2) * growth_factor_z1_arr * growth_factor_z2 * integral_result

    elif which_sigma2_b == 'polar_cap_on_the_fly' or which_sigma2_b == 'from_input_mask':

        partial_summand = np.zeros((len(z1_arr), len(ell_mask)))
        # NOTE: you should include a 2/np.pi factor, see Eq. (26) of https://arxiv.org/pdf/1612.05958, or Champaghe et al 2017
        partial_summand = (2 * ell_mask + 1) * cl_mask * 2 / np.pi * growth_factor_z1_arr[:, None] * growth_factor_z2
        partial_summand *= integral_result[:, None]
        result = np.sum(partial_summand, axis=1)
        one_over_omega_s_squared = 1 / (4 * np.pi * fsky_mask)**2
        result *= one_over_omega_s_squared

        # F. Lacasa:
        # np.sum((2*ell+1)*cl_mask*Cl_XY[ipair,jpair,:])/(4*pi*fsky)**2
    else:
        raise ValueError(
            'which_sigma2_b must be either "full_curved_sky" or "polar_cap_on_the_fly" or "from_input_mask"')

    return result


def plot_sigma2(sigma2_arr, z_grid_sigma2):
    font_size = 28
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["legend.fontsize"] = font_size

    plt.figure()
    pad = 0.4  # I don't want to plot sigma at the edges of the grid, it's too noisy
    for z_test in np.linspace(z_grid_sigma2.min() + pad, z_grid_sigma2.max() - pad, 4):
        z1_idx = np.argmin(np.abs(z_grid_sigma2 - z_test))
        z_1 = z_grid_sigma2[z1_idx]

        plt.plot(z_grid_sigma2, sigma2_arr[z1_idx, :], label=f'$z_1=%.2f$ ' % z_1)
        plt.axvline(z_1, color='k', ls='--', label='$z_1$')
    plt.xlabel('$z_2$')
    plt.ylabel('$\\sigma^2(z_1, z_2)$')  # sigma2 is dimensionless!
    plt.legend()
    plt.show()

    font_size = 18
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["legend.fontsize"] = font_size
    sl.matshow(sigma2_arr, log=True, abs_val=True, title='$\\sigma^2(z_1, z_2)$')

    # z_steps_sigma2 = len(z_grid_sigma2)
    # plt.savefig(f'../output/plots/sigma2_spikes_zsteps{z_steps_sigma2}.pdf', dpi=500, bbox_inches='tight')
    # plt.savefig(f'../output/plots/sigma2_matshow_zsteps{z_steps_sigma2}.pdf', dpi=500, bbox_inches='tight')


def compute_sigma2(z_grid_sigma2, k_grid_sigma2, which_sigma2_b, cosmo_ccl, parallel=True, vectorize=False):
    print(f'computing sigma^2(z_1, z_2) for SSC...')

    if parallel:
        # ! parallelize with ray
        # start_time = time.perf_counter()
        # sigma2_func_remote = ray.remote(sigma2_func)
        # remote_calls = []
        # for z1 in tqdm(z_grid_sigma2):
        #     for z2 in z_grid_sigma2:
        #         remote_calls.append(sigma2_func_remote.remote(z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_b))
        # # Get the results from the remote function calls
        # sigma2_arr = ray.get(remote_calls)

        # ! with joblib
        # sigma2_arr = Parallel(n_jobs=-1, backend='loky')(delayed(sigma2_func)(
        # z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_b) for z1 in tqdm(z_grid_sigma2) for z2 in z_grid_sigma2)

        # ! with pool.map
        # Share cosmo_ccl object using custom SharedCosmology
        # shm_cosmo_ccl = SharedCosmology([cosmo_ccl])

        # with mp.Pool() as pool:
        #     sigma2_arr = pool.starmap(
        #         partial(sigma2_func, k_grid_sigma2=k_grid_sigma2, cosmo_ccl=shm_cosmo_ccl[0], which_sigma2_b=which_sigma2_b),
        #         [(z1, z2) for z1 in z_grid_sigma2 for z2 in z_grid_sigma2]
        #     )

        # reshape result
        sigma2_arr = np.array(sigma2_arr).reshape((len(z_grid_sigma2), len(z_grid_sigma2)))
        print(f'sigma2 computed in: {(time.perf_counter() - start_time):.2f} s')

    # ! serial version
    else:
        sigma2_arr = np.zeros((len(z_grid_sigma2), len(z_grid_sigma2)))

        if not vectorize:
            for z1_idx, z1 in enumerate(tqdm(z_grid_sigma2)):
                for z2_idx, z2 in enumerate(z_grid_sigma2):
                    sigma2_arr[z1_idx, z2_idx] = sigma2_func(z1, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_b)
        elif vectorize:
            for z2_idx, z2 in enumerate(tqdm(z_grid_sigma2)):
                sigma2_arr[:, z2_idx] = sigma2_z2_func_vectorized(
                    z_grid_sigma2, z2, k_grid_sigma2, cosmo_ccl, which_sigma2_b, None, None)

    return sigma2_arr


def sigma2_pyssc(z_arr, classy_cosmo_params):
    import PySSC
    """ Compute sigma2 with PySSC. This is just for comparison, it is not used in the code."""
    if classy_cosmo_params is None:
        logging.info('Using default classy cosmo params from cosmo_lib')
        classy_cosmo_params = csmlib.cosmo_par_dict_classy
    if z_arr is None:
        # ! 1e-3 as zmin gives errors in classy, probably need to increse pk_max
        z_arr = np.linspace(1e-2, 3, 300)
    return PySSC.sigma2_fullsky(z_arr, cosmo_params=classy_cosmo_params, cosmo_Class=None)
