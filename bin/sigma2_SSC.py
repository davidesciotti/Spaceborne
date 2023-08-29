import pdb
import sys
import time
import warnings
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import Parallel, delayed
import concurrent.futures
from numba import njit
from scipy.integrate import quad, quad_vec, simps
from scipy.interpolate import RegularGridInterpolator
from scipy.special import spherical_jn
import ray
import pyccl as ccl
from tqdm import tqdm

sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'/Users/davide/Documents/Lavoro/Programmi/common_data/common_config')
import mpl_cfg

sys.path.append(f'/')
import wf_cl_lib

sys.path.append(f'/')
import ell_values as ell_utils

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
    Note that the 1/Omega_S^2 factors are missing in this function!! This is consistent withthe definitio given in
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

    # instantiate cosmo_ccl
    cosmo_ccl = csmlib.istantiate_cosmo_ccl_obj(ficualial_pars_dict)

    z_grid_sigma2 = np.linspace(sigma2_cfg['z_min_sigma2'], sigma2_cfg['z_max_sigma2'], sigma2_cfg['z_steps_sigma2'])
    k_grid_sigma2 = np.logspace(sigma2_cfg['log10_k_min_sigma2'], sigma2_cfg['log10_k_max_sigma2'],
                                sigma2_cfg['k_steps_sigma2'])

    # ! parallelize with ray
    start_time = time.perf_counter()
    sigma2_func_remote = ray.remote(sigma2_func)
    remote_calls = []
    for z1 in z_grid_sigma2:
        for z2 in tqdm(z_grid_sigma2):
            remote_calls.append(sigma2_func_remote.remote(z1, z2, k_grid_sigma2, cosmo_ccl))
    # Get the results from the remote function calls
    sigma2_arr = ray.get(remote_calls)

    # with joblib (doesn't seem to work anymore, I still don't know why)
    # sigma2_arr = Parallel(n_jobs=-1, backend='threading')(delayed(sigma2_func)(
    #     z1, z2, k_grid_sigma2, cosmo_ccl) for z1 in tqdm(z_grid_sigma2) for z2 in z_grid_sigma2)

    # reshape result
    sigma2_arr = np.array(sigma2_arr).reshape((len(z_grid_sigma2), len(z_grid_sigma2)))
    print(f'sigma2 computed in: {(time.perf_counter() - start_time):.2f} s')

    return sigma2_arr

# TODO compute sigma_b with PyCCL for a rought comparison
# fsky = csmlib.deg2_to_fsky(cfg['sky_area_deg2'])
# sigma2_ccl = ccl.sigma2_B_disc(cosmo=cosmo_ccl, a=csmlib.z_to_a(z_grid_sigma2)[::-1], fsky=fsky, p_of_k_a=None)
# plt.plot(z_grid_sigma2, sigma2_ccl, label='PyCCL')
