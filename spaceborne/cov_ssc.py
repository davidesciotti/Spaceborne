import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from jax import jit
from scipy.fft import rfft
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d

from spaceborne import cosmo_lib
from spaceborne import cov_dict as cd
from spaceborne import sb_lib as sl


def sigma2_z1z2_fft(
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

    # real FFT -> cosine coefficients on linear grid
    fft_coeffs = rfft(Pk0) * dk  # \sum f(k) cos -> Re{FFT} * dk
    r_grid = np.arange(fft_coeffs.size) * 2 * np.pi / (k_max - k_min)
    c_r = fft_coeffs.real

    # interpolate C(r)
    c_0 = simps(y=Pk0, x=k_grid)
    c_func = interp1d(
        r_grid,
        c_r,
        kind='cubic',
        bounds_error=False,
        fill_value=(c_0, 0.0),
        assume_sorted=True,
    )

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


@jit
def ssc_integral_4D_simps_jax(
    d2ClAB_dVddeltab: jnp.ndarray,
    d2ClCD_dVddeltab: jnp.ndarray,
    cl_integral_prefactor: jnp.ndarray,
    sigma2: jnp.ndarray,
    delta_z: float,
    simpson_weights: jnp.ndarray,
):
    """
    JAX version of the Simpson's rule 2D integral.
    Expects d2Cl arrays to be pre-shaped to 3D: (nbl, zpairs, z_steps)
    """

    # Pre-compute combined weights
    # Shape: (z_steps, z_steps)
    prefactor_grid = jnp.outer(cl_integral_prefactor, cl_integral_prefactor)
    weight_grid = jnp.outer(simpson_weights, simpson_weights)
    combined_weights = prefactor_grid * weight_grid * sigma2

    # Compute all combinations with einsum
    # Shape: (nbl, nbl, zpairs_AB, zpairs_CD)
    result = jnp.einsum(
        'Liz,Mjw,zw->LMij', d2ClAB_dVddeltab, d2ClCD_dVddeltab, combined_weights
    )

    # multiply by step size
    return result * (delta_z**2)


@jit
def ssc_integral_4D_simps_jax_ke_approx(
    d2ClAB_dVddeltab: jnp.ndarray,
    d2ClCD_dVddeltab: jnp.ndarray,
    cl_integral_prefactor: jnp.ndarray,
    sigma2: jnp.ndarray,
    delta_z: float,
    simpson_weights: jnp.ndarray,
):
    """
    JAX version of the Simpson's rule 1D integral.
    Expects d2Cl arrays to be pre-shaped to 3D: (nbl, zpairs, z_steps)
    """

    # Pre-compute combined weights
    # Shape: (z_steps,)
    combined_weights = cl_integral_prefactor * simpson_weights * sigma2

    # Compute all combinations with einsum
    # Shape: (nbl, nbl, zpairs_AB, zpairs_CD)
    result = jnp.einsum(
        'Liz,Mjz,z->LMij', d2ClAB_dVddeltab, d2ClCD_dVddeltab, combined_weights
    )

    # multiply by step size
    return result * delta_z


class SpaceborneSSC:
    def __init__(self, cfg, pvt_cfg, ccl_obj, z_grid):
        self.use_ke_approx = cfg['covariance']['use_KE_approximation']
        self.z_grid = z_grid
        self.ccl_obj = ccl_obj

        # Enable 64-bit precision if required
        jax.config.update('jax_enable_x64', cfg['misc']['jax_enable_x64'])
        print('JAX devices:', jax.devices())

        # set some useful attributes
        if self.use_ke_approx:
            self.ssc_func = ssc_integral_4D_simps_jax_ke_approx
            self.cl_integral_convention_ssc = 'Euclid_KE_approximation'
        else:
            self.ssc_func = ssc_integral_4D_simps_jax
            self.cl_integral_convention_ssc = 'Euclid'

        self.ind_dict = pvt_cfg['ind_dict']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']

        self.zbins = pvt_cfg['zbins']
        self.use_h_units = pvt_cfg['use_h_units']

        assert self.zpairs_auto == self.ind_auto.shape[0]
        assert self.zpairs_cross == self.ind_cross.shape[0]

        req_terms = ['ssc']
        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='harmonic')
            for probe in pvt_cfg['req_probe_combs_hs_2d']
        ]  # SSC computes probe blocks only, not full 3x2pt
        dims = ['4d']
        self.cov_dict = cd.create_cov_dict(req_terms, _req_probe_combs_2d, dims=dims)

    def set_sigma2_b(self, ccl_obj, mask_obj, k_grid_s2b, which_sigma2_b):
        """Wrapper function for setting sigma2_b in 1 or 2 dimensions (depending on
        whether the KE approximation is used or not).
        """
        if self.use_ke_approx:
            # compute sigma2_b(z) (1 dimension) using the existing CCL implementation
            ccl_obj.set_sigma2_b(
                z_grid=self.z_grid, which_sigma2_b=which_sigma2_b, mask_obj=mask_obj
            )
            _a, sigma2_b = ccl_obj.sigma2_b_tuple

            # quick sanity check on the a/z grid
            sigma2_b = sigma2_b[::-1]
            _z = cosmo_lib.a_to_z(_a)[::-1]
            np.testing.assert_allclose(self.z_grid, _z, atol=0, rtol=1e-8)

        else:
            sigma2_b = sigma2_z1z2_fft(
                z1_arr=self.z_grid,
                z2_arr=self.z_grid,
                k_grid_sigma2=k_grid_s2b,
                cosmo_ccl=ccl_obj.cosmo_ccl,
                which_sigma2_b=which_sigma2_b,
                ell_mask=mask_obj.ell_mask,
                cl_mask=mask_obj.cl_mask,
                fsky_mask=mask_obj.fsky,
                nk_fft=2**21,
            )

        self.sigma2_b = sigma2_b

    def set_ssc_integral_prefactor(self):
        self.cl_integral_prefactor = cosmo_lib.cl_integral_prefactor(
            self.z_grid,
            self.cl_integral_convention_ssc,
            use_h_units=self.use_h_units,
            cosmo_ccl=self.ccl_obj.cosmo_ccl,
        )

    def compute_ssc(
        self,
        d2CLL_dVddeltab_4d,
        d2CGL_dVddeltab_4d,
        d2CGG_dVddeltab_4d,
        unique_probe_combs_hs,
        symm_probe_combs_hs,
        nonreq_probe_combs_hs,
    ):
        z_steps = len(self.z_grid)

        # ! sanity checks
        # check that nbl is the same
        assert (
            d2CLL_dVddeltab_4d.shape[0]
            == d2CGL_dVddeltab_4d.shape[0]
            == d2CGG_dVddeltab_4d.shape[0]
        ), (
            'd2CLL_dVddeltab_4d, d2CGL_dVddeltab_4d and d2CGG_dVddeltab_4d must have '
            'the same number of elements along the first axis'
        )

        # check that z_steps is the same
        assert (
            d2CLL_dVddeltab_4d.shape[-1]
            == d2CGL_dVddeltab_4d.shape[-1]
            == d2CGG_dVddeltab_4d.shape[-1]
            == z_steps
        ), (
            'd2CLL_dVddeltab_4d, d2CGL_dVddeltab_4d and d2CGG_dVddeltab_4d must have '
            'the same number of elements along the first axis'
        )

        # contract zi, zj -> zij
        nbl = d2CLL_dVddeltab_4d.shape[0]
        d2CLL_dVddeltab_3d = np.zeros((nbl, self.zpairs_auto, z_steps))
        d2CGL_dVddeltab_3d = np.zeros((nbl, self.zpairs_cross, z_steps))
        d2CGG_dVddeltab_3d = np.zeros((nbl, self.zpairs_auto, z_steps))

        for zij in range(self.zpairs_auto):
            zi, zj = self.ind_auto[zij, 2], self.ind_auto[zij, 3]
            d2CLL_dVddeltab_3d[:, zij, :] = d2CLL_dVddeltab_4d[:, zi, zj, :]
            d2CGG_dVddeltab_3d[:, zij, :] = d2CGG_dVddeltab_4d[:, zi, zj, :]
        for zij in range(self.zpairs_cross):
            zi, zj = self.ind_cross[zij, 2], self.ind_cross[zij, 3]
            d2CGL_dVddeltab_3d[:, zij, :] = d2CGL_dVddeltab_4d[:, zi, zj, :]

        d2CAB_dVddeltab_dict_3d = {
            ('LL'): d2CLL_dVddeltab_3d,
            ('GL'): d2CGL_dVddeltab_3d,
            ('GG'): d2CGG_dVddeltab_3d,
        }

        # ! necessary ingredients for the integration:
        # ! integral prefactor, simpson weights, delta_z
        self.set_ssc_integral_prefactor()
        simpson_weights = sl.get_simpson_weights(z_steps)
        delta_z = np.diff(self.z_grid)[0]

        # sanity check
        np.testing.assert_allclose(
            np.diff(self.z_grid),
            np.diff(self.z_grid)[0],
            atol=0,
            rtol=1e-8,
            err_msg='z_grid must be uniformly sampled',
        )

        # ! start the actual computation
        start = time.perf_counter()
        print('\nComputing Spaceborne SSC...')

        # * compute required blocks
        for probe_abcd in unique_probe_combs_hs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'harmonic')

            print(f'SSC: computing probe combination {probe_ab, probe_cd}')
            d2CABdVddeltab_3d = d2CAB_dVddeltab_dict_3d[(probe_ab)]
            d2CCDdVddeltab_3d = d2CAB_dVddeltab_dict_3d[(probe_cd)]

            result = self.ssc_func(
                jnp.array(d2CABdVddeltab_3d),
                jnp.array(d2CCDdVddeltab_3d),
                jnp.array(self.cl_integral_prefactor),
                jnp.array(self.sigma2_b),
                delta_z,
                jnp.array(simpson_weights),
            )

            self.cov_dict['ssc'][probe_ab, probe_cd]['4d'] = np.array(result)

        # * symmetrize and set to 0 the remaning probe blocks
        sl.symmetrize_and_fill_probe_blocks(
            cov_term_dict=self.cov_dict['ssc'],
            dim='4d',
            unique_probe_combs=unique_probe_combs_hs,
            nonreq_probe_combs=nonreq_probe_combs_hs,
            obs_space='harmonic',
            nbx=nbl,
            zbins=None,
            ind_dict=self.ind_dict,
            msg='',
        )

        print(f'...done in {(time.perf_counter() - start):.2f} s')

        return self.cov_dict
