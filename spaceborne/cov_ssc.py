import time

import jax.numpy as jnp
import numpy as np
import pyccl as ccl
from jax import jit
from scipy.fft import rfft
from scipy.integrate import simpson as simps
from scipy.interpolate import RectBivariateSpline

from spaceborne import cosmo_lib
from spaceborne import cov_dict as cd
from spaceborne import sb_lib as sl


def linear_pk_transforms(
    cosmo_ccl: ccl.Cosmology,
    k_min: float,
    k_max: float,
    r_max: float,
    nk_fft: int = 2**21,
    fft_pad: int = 8,
    dr_min: float = 2.5e-3,
) -> tuple:
    r"""Transforms of the z=0 linear power spectrum, sampled on a uniform r grid.

    Returns ``(dr, cos_transform, xi)``, with
    ``cos_transform(r) = \int dk P(k) cos(kr)`` and
    ``xi(r) = 1/(2 pi^2) \int dk k^2 P(k) j_0(kr)``, both over ``[k_min, k_max]``,
    tabulated up to (at least) ``r_max``.

    - The k sampling sets the largest separation the FFT can represent, pi / dk: at
      least ``nk_fft`` points are used, more if needed to cover 1.5 * ``r_max``.
    - The FFT is zero-padded by ``fft_pad`` to oversample r: the sharp cut at
      ``k_max`` makes both transforms ring with period 2 pi / k_max, which is also
      the unpadded r spacing, so without padding the ringing is aliased. The padding
      is reduced when dr would fall well below ``dr_min`` [Mpc], to bound memory
      for large k_max, where P(k_max), and hence the ringing, is negligible.
    """
    nk_range = 1.5 * r_max * (k_max - k_min) / np.pi
    nk = max(nk_fft, 2 ** int(np.ceil(np.log2(nk_range))))
    k = np.linspace(k_min, k_max, nk)
    dk = k[1] - k[0]
    pk = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)

    n_fft_dr_min = 2 ** int(np.ceil(np.log2(2 * np.pi / (dk * dr_min))))
    n_fft = max(2 * nk, min(fft_pad * nk, n_fft_dr_min))
    dr = 2 * np.pi / (n_fft * dk)
    n_r = int(r_max / dr) + 2
    r = np.arange(n_r) * dr

    # rfft assumes a grid starting at k=0; restore the phase exp(-i r k_min)
    phase = np.exp(-1j * r * k_min)
    cos_transform = (phase * rfft(pk, n=n_fft)[:n_r]).real * dk
    sin_transform = -(phase * rfft(k * pk, n=n_fft)[:n_r]).imag * dk  # of k P(k)

    xi = np.empty(n_r)
    xi[0] = simps(y=k**2 * pk, x=k)
    xi[1:] = sin_transform[1:] / r[1:]
    xi /= 2 * np.pi**2

    return dr, cos_transform, xi


def interp_uniform(x: np.ndarray, dx: float, y: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``y``, sampled at ``0, dx, 2 dx, ...``, at ``x >= 0``.

    Much faster than ``np.interp`` on large tables, since no bisection is needed.
    """
    pos = x / dx
    if pos.max() > y.size - 1:
        raise ValueError(
            f'interpolation point {pos.max() * dx:.6g} beyond the table range '
            f'{(y.size - 1) * dx:.6g}'
        )
    idx = np.minimum(pos.astype(np.intp), y.size - 2)
    frac = pos - idx
    return y[idx] * (1 - frac) + y[idx + 1] * frac


def sigma2_z1z2(
    z_grid: np.ndarray,
    k_min: float,
    k_max: float,
    cosmo_ccl: ccl.Cosmology,
    cl_footp_norm: np.ndarray,
    *,
    nk_fft: int = 2**21,
    fft_pad: int = 8,
    n_z_coarse: int = 600,
) -> np.ndarray:
    r"""Variance of the background density mode in the survey, sigma^2_b(z1, z2).

    For a mask with normalised spectrum
    ``w_L = (2L+1) C_L^{W_AB W_CD} / ((4 pi)^2 f_AB f_CD)`` (``cl_footp_norm``),
    Lacasa, Lima & Aguena (2016, arXiv:1612.05958) give

        sigma2 = D1 D2 \sum_L w_L 2/pi \int dk k^2 P(k) j_L(k chi1) j_L(k chi2).

    The addition theorem turns the sum over L into an angular integral,

        sigma2 = 2 pi D1 D2 \int_{-1}^{1} dmu W(mu) xi(r),
        W(mu) = \sum_L w_L P_L(mu),  r^2 = chi1^2 + chi2^2 - 2 chi1 chi2 mu,

    which is evaluated by splitting ``W(mu) = W(1) + [W(mu) - W(1)]``:

    - the monopole term ``W(1)`` integrates analytically in mu to a cosine transform
      of P(k); it holds the sharp structure around z1 = z2 and is evaluated on
      the full ``z_grid``;
    - the mask-shape term has an integrand vanishing at mu = 1, so it is smooth in
      (z1, z2): it is evaluated on ``n_z_coarse`` log-spaced redshifts and
      spline-interpolated onto ``z_grid``.
    """
    z_grid = np.atleast_1d(z_grid)
    chi = ccl.comoving_radial_distance(cosmo_ccl, cosmo_lib.z_to_a(z_grid))
    growth = ccl.growth_factor(cosmo_ccl, cosmo_lib.z_to_a(z_grid))
    assert np.all(chi > 0), 'sigma2_b requires z > 0'

    dr, cos_transform, xi = linear_pk_transforms(
        cosmo_ccl, k_min, k_max, r_max=2 * chi.max(), nk_fft=nk_fft, fft_pad=fft_pad
    )

    # * monopole term, without growth factors, using
    # * \int dk k^2 P j0(k chi1) j0(k chi2)
    # *   = [C(|chi1 - chi2|) - C(chi1 + chi2)] / (2 chi1 chi2)
    # * (upper triangle, filled row by row to limit memory usage)
    w_mu_1 = np.sum(cl_footp_norm)
    sigma2 = np.zeros((chi.size, chi.size))
    for i, chi_i in enumerate(chi):
        chi_j = chi[i:]
        sigma2[i, i:] = interp_uniform(np.abs(chi_j - chi_i), dr, cos_transform)
        sigma2[i, i:] -= interp_uniform(chi_i + chi_j, dr, cos_transform)
        sigma2[i, i:] *= w_mu_1 / (np.pi * chi_i * chi_j)
    sigma2 += np.triu(sigma2, 1).T

    # * mask-shape term
    if z_grid.size <= n_z_coarse:
        z_coarse, chi_coarse = z_grid, chi
    else:
        z_coarse = np.geomspace(z_grid[0], z_grid[-1], n_z_coarse)
        z_coarse[[0, -1]] = z_grid[[0, -1]]
        chi_coarse = ccl.comoving_radial_distance(cosmo_ccl, cosmo_lib.z_to_a(z_coarse))

    # angular trapezoid quadrature: a uniform grid resolving the oscillations of W(mu)
    # up to the mask ell_max, refined logarithmically at small angles, where
    # xi(r) varies on scales r ~ chi * theta
    theta = np.unique(
        np.concatenate(
            [
                np.geomspace(1e-6, 1e-1, 1000),
                np.linspace(0, np.pi, 8 * max(cl_footp_norm.size, 512) + 1),
            ]
        )
    )
    cos_theta = np.cos(theta)
    w_mu = np.polynomial.legendre.legval(cos_theta, cl_footp_norm)
    theta_weights = np.zeros(theta.size)
    theta_weights[1:] += np.diff(theta) / 2
    theta_weights[:-1] += np.diff(theta) / 2
    theta_weights *= (w_mu - w_mu_1) * np.sin(theta)

    shape_term = np.zeros((z_coarse.size, z_coarse.size))
    for i, chi_i in enumerate(chi_coarse):
        chi_j = chi_coarse[i:, None]
        r = np.sqrt(np.maximum(chi_i**2 + chi_j**2 - 2 * chi_i * chi_j * cos_theta, 0))
        shape_term[i, i:] = interp_uniform(r, dr, xi) @ theta_weights
    shape_term += np.triu(shape_term, 1).T
    shape_term *= 2 * np.pi

    if z_coarse is not z_grid:
        shape_term = RectBivariateSpline(z_coarse, z_coarse, shape_term)(z_grid, z_grid)

    sigma2 += shape_term
    sigma2 *= np.outer(growth, growth)
    return sigma2


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
        self.use_ke_approx = cfg['precision']['use_KE_approximation']
        self.z_grid = z_grid
        self.ccl_obj = ccl_obj
        self.k_min = 10 ** cfg['precision']['log10_k_min']
        self.k_max = 10 ** cfg['precision']['log10_k_max']

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

    def sigma2_b_func(self, cl_footp_norm_abcd: np.ndarray) -> np.ndarray:
        """sigma2_b(z) with the KE approximation (from CCL), sigma2_b(z1, z2)
        otherwise.
        """
        if self.use_ke_approx:
            a_grid, sigma2_b = self.ccl_obj.sigma2_b_func(
                z_grid=self.z_grid, cl_footp_norm_abcd=cl_footp_norm_abcd
            )
            # CCL works with increasing a, i.e. decreasing z
            np.testing.assert_allclose(
                self.z_grid, cosmo_lib.a_to_z(a_grid)[::-1], atol=0, rtol=1e-8
            )
            return sigma2_b[::-1]

        return sigma2_z1z2(
            z_grid=self.z_grid,
            k_min=self.k_min,
            k_max=self.k_max,
            cosmo_ccl=self.ccl_obj.cosmo_ccl,
            cl_footp_norm=cl_footp_norm_abcd,
        )

    def set_ssc_integral_prefactor(self):
        self.cl_integral_prefactor = cosmo_lib.cl_integral_prefactor(
            self.z_grid,
            self.cl_integral_convention_ssc,
            use_h_units=self.use_h_units,
            cosmo_ccl=self.ccl_obj.cosmo_ccl,
        )

    def compute_ssc(
        self,
        d2CLL_dVddeltab_4d: np.ndarray,
        d2CGL_dVddeltab_4d: np.ndarray,
        d2CGG_dVddeltab_4d: np.ndarray,
        sigma2_b_dict: dict,
        unique_probe_combs_hs: list,
        nonreq_probe_combs_hs: list,
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
        print('\nComputing SSC...')

        # * compute required blocks
        for probe_abcd in unique_probe_combs_hs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'harmonic')

            print(f'SSC cov: computing probe combination {(probe_ab, probe_cd)}')
            d2CABdVddeltab_3d = d2CAB_dVddeltab_dict_3d[(probe_ab)]
            d2CCDdVddeltab_3d = d2CAB_dVddeltab_dict_3d[(probe_cd)]
            sigma2_b_ABCD = sigma2_b_dict[probe_ab, probe_cd]

            result = self.ssc_func(
                jnp.array(d2CABdVddeltab_3d),
                jnp.array(d2CCDdVddeltab_3d),
                jnp.array(self.cl_integral_prefactor),
                jnp.array(sigma2_b_ABCD),
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
            nbs=nbl,
            zbins=None,
            ind_dict=self.ind_dict,
            msg='SSC cov: ',
        )

        print(f'...done in {(time.perf_counter() - start):.2f} s')

        return self.cov_dict
