import logging
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
from scipy.special import spherical_jn
import ray
import pyccl as ccl
from tqdm import tqdm
import PySSC

sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_lib_and_cfg/common_config')
import mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

ray.shutdown()
ray.init()


# TODO maybe re-check that the various cosmo_ccl objects are equivalent...
# TODO play around with the k_grid_sigma2 and z_grid_sigma2
# TODO should we add more points at low z for sigma2? I get a strange behavior...
# TODO test these interpolations
# TODO most likely you can easily vectorize sigma squared in one of the redshifts
# TODO range(ell1_idx, nbl) to avoid computing symmetric ell elements (check first)


# * logbook
# - the vectorization is quite messy; the quad version accepts z_1 or z_2 as vector, but only
# setting sigma2_integrating_function=quad_vec
# - the simpson version is not vectorized in z1/z2, but it is much faster than the quad version (and much noisier!!)
# - try to finish building the cov_SSC function, which is commented below.
# - find the optimal k_grid_sigma2 and just fix it


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# ! inputs needed by the code:
# * primary:
# - cosmo params
# - fsky
# - zbins
# - ell_min, ell_max, nbl (or ell_grid)

# * secondary:
# - growth_factor


def sigma2_func(z1, z2, k_grid_sigma2, cosmo_ccl):
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
    integrand = lambda k: k ** 2 * ccl.linear_matter_power(cosmo_ccl, k=k, a=1.) * \
                          spherical_jn(0, k * r1) * spherical_jn(0, k * r2)

    integral_result = simps(integrand(k_grid_sigma2), k_grid_sigma2)

    # different integration methods; simps seems to be the best
    # if integrating_funct == 'simps':
    #     integral_result = simps(integrand(k_grid_sigma2), k_grid_sigma2)
    # elif integrating_funct == 'quad':
    #     integral_result = quad(integrand, k_grid_sigma2[0], k_grid_sigma2[-1])[0]
    # elif integrating_funct == 'quad_vec':
    #     integral_result = quad_vec(integrand, k_grid_sigma2[0], k_grid_sigma2[-1])[0]
    # else:
    #     raise ValueError('sigma2_integrating_function must be either "simps" or "quad" or "quad_vec"')

    return 1 / (2 * np.pi ** 2) * growth_factor_z1 * growth_factor_z2 * integral_result


def plot_sigma2(sigma2_arr, z_grid_sigma2):
    font_size = 28
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["legend.fontsize"] = font_size

    z_steps_sigma2 = len(z_grid_sigma2)

    plt.figure()
    pad = 0.4  # I don't want to plot sigma at the edges of the grid, it's too noisy
    for z_test in np.linspace(z_grid_sigma2.min() + pad, z_grid_sigma2.max() - pad, 4):
        z1_idx = np.argmin(np.abs(z_grid_sigma2 - z_test))
        z_1 = z_grid_sigma2[z1_idx]

        plt.plot(z_grid_sigma2, sigma2_arr[z1_idx, :], label=f'$z_1=%.2f$ ' % z_1)
        # plt.axvline(z_1, color='k', ls='--', label='$z_1$')
    plt.xlabel('$z_2$')
    plt.ylabel('$\sigma^2(z_1, z_2)$')  # sigma2 is dimensionless!
    plt.legend()
    plt.show()

    font_size = 18
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams["legend.fontsize"] = font_size

    mm.matshow(sigma2_arr, log=True, abs_val=True, title='$\sigma^2(z_1, z_2)$')

    # plt.savefig(f'../output/plots/sigma2_spikes_zsteps{z_steps_sigma2}.pdf', dpi=500, bbox_inches='tight')
    # plt.savefig(f'../output/plots/sigma2_matshow_zsteps{z_steps_sigma2}.pdf', dpi=500, bbox_inches='tight')


def compute_sigma2(sigma2_cfg, ficualial_pars_dict):
    print(f'computing sigma^2(z_1, z_2) for SSC...')
    # instantiate cosmo_ccl
    cosmo_ccl = csmlib.istantiate_cosmo_ccl_obj(ficualial_pars_dict)

    z_grid_sigma2 = np.linspace(sigma2_cfg['z_min_sigma2'], sigma2_cfg['z_max_sigma2'], sigma2_cfg['z_steps_sigma2'])
    k_grid_sigma2 = np.logspace(sigma2_cfg['log10_k_min_sigma2'], sigma2_cfg['log10_k_max_sigma2'],
                                sigma2_cfg['k_steps_sigma2'])

    # ! parallelize with ray
    start_time = time.perf_counter()
    sigma2_func_remote = ray.remote(sigma2_func)
    remote_calls = []
    for z1 in tqdm(z_grid_sigma2):
        for z2 in z_grid_sigma2:
            remote_calls.append(sigma2_func_remote.remote(z1, z2, k_grid_sigma2, cosmo_ccl))
    # Get the results from the remote function calls
    sigma2_arr = ray.get(remote_calls)

    # with joblib (doesn't seem to work anymore, I still don't know why)
    # sigma2_arr = Parallel(n_jobs=-1, backend='threading')(delayed(sigma2_func)(
    #     z1, z2, k_grid_sigma2, cosmo_ccl) for z1 in tqdm(z_grid_sigma2) for z2 in z_grid_sigma2)

    # reshape result
    sigma2_arr = np.array(sigma2_arr).reshape((len(z_grid_sigma2), len(z_grid_sigma2)))
    print(f'sigma2 computed in: {(time.perf_counter() - start_time):.2f} s')

    return sigma2_arr, z_grid_sigma2


def interpolate_sigma2_arr(sigma2_arr, z_grid_original, z_grid_new):
    """ Interpolate sigma2_arr from z_grid_original to z_grid_new. This is needed because the covmat is computed
    in a different z_grid than the one used to compute sigma2."""

    # TODO test this!

    sigma2_interp_func = RegularGridInterpolator((z_grid_original, z_grid_original), sigma2_arr, method='linear')

    z_grid_new_xx, z_grid_new_yy = np.meshgrid(z_grid_new, z_grid_new)
    sigma2_arr_interpolated = sigma2_interp_func((z_grid_new_xx, z_grid_new_yy)).T
    return sigma2_arr_interpolated


logging.basicConfig(level=logging.INFO)

def sigma2_pyssc(z_arr, classy_cosmo_params):
    """ Compute sigma2 with PySSC. This is just for comparison, it is not used in the code."""
    if classy_cosmo_params is None:
        logging.info('Using default classy cosmo params from cosmo_lib')
        classy_cosmo_params = csmlib.cosmo_par_dict_classy
    if z_arr is None:
        # ! 1e-3 as zmin gives errors in classy, probably need to increse pk_max
        z_arr = np.linspace(1e-2, 3, 300)
    return PySSC.sigma2_fullsky(z_arr, cosmo_params=classy_cosmo_params, cosmo_Class=None)



sigma2_pyssc_arr = sigma2_pyssc(z_arr_pyssc, None)




def compare_sigma2_sb_vs_pyssc(z_arr_pyssc, sigma2_pyssc_arr, z_1_idx=100):
    
    path = '/Users/davide/Documents/Lavoro/Programmi/exact_SSC/output/integrand_arrays/sigma2'
    sigma2_sb = np.load(f'{path}/sigma2_zsteps3000_simps.npy')
    z_arr_sb = np.load(f'{path}/z_grid_sigma2_zsteps3000.npy')

    sigma2_sb_interp = RegularGridInterpolator((z_arr_sb, z_arr_sb), sigma2_sb, method='linear')
    z_arr_xx, z_arr_yy = np.meshgrid(z_arr_pyssc, z_arr_pyssc)
    sigma2_sb = sigma2_sb_interp((z_arr_xx, z_arr_yy)).T

    plt.figure()
    plt.plot(z_arr_pyssc, sigma2_pyssc_arr[z_1_idx, :], label='PySSC', marker='.')
    plt.plot(z_arr_pyssc, sigma2_sb[z_1_idx, :], label='Spaceborne', ls='--', marker='.')
    plt.axvline(z_arr_pyssc[z_1_idx], color='k', ls='--', label='$z_1$')
    plt.xlim(0, 2.5)
    plt.xlabel('$z_2$')
    plt.ylabel('$\sigma^2(z_1, z_2)$')
    plt.legend()

# TODO compute sigma_b with PyCCL for a rought comparison
# fsky = csmlib.deg2_to_fsky(cfg['sky_area_deg2'])
# sigma2_ccl = ccl.sigma2_B_disc(cosmo=cosmo_ccl, a=csmlib.z_to_a(z_grid_sigma2)[::-1], fsky=fsky, p_of_k_a=None)
# plt.plot(z_grid_sigma2, sigma2_ccl, label='PyCCL')
