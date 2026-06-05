"""Unit tests for the spaceborne.cov_ssc module (super-sample covariance).

The SSC machinery has two well-isolated, pure numerical pieces that we can pin
down without standing up the full pipeline:

* ``sigma2_z1z2_fft`` -- the sample-variance kernel sigma^2(z1, z2), built from
  an FFT of the linear matter power spectrum. We assert its symmetry, shape, the
  validation of its inputs, and an exact algebraic cross-check between the
  full-sky and masked branches (they differ only by a z-independent constant).

* ``ssc_integral_4D_simps_jax`` / ``..._ke_approx`` -- the JAX Simpson-rule
  contractions that assemble the 4D covariance. We check them against an explicit
  brute-force ``for``-loop reference (non-circular), plus the symmetry and
  linearity that follow from the einsum structure.

The ``nk_fft`` of ``sigma2_z1z2_fft`` is deliberately set far below the
production default (2**21) so the FFT is cheap; the invariants under test
(symmetry, scaling) hold at any resolution.
"""

import numpy as np
import jax.numpy as jnp
import pyccl as ccl
import pytest

from spaceborne import cov_ssc


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=1234)


@pytest.fixture(scope='module')
def cosmo():
    """A cheap-to-evaluate vanilla LCDM cosmology.

    ``eisenstein_hu`` + ``halofit`` avoid a Boltzmann-solver call, so the linear
    power spectrum used inside ``sigma2_z1z2_fft`` is fast to obtain.
    """
    return ccl.CosmologyVanillaLCDM(
        transfer_function='eisenstein_hu', matter_power_spectrum='halofit'
    )


# A small, fast FFT length. The default is 2**21; the symmetry / scaling
# invariants we test are independent of this resolution.
NK_FFT_TEST = 2**13


# ----------------------------------------------------------------------------- #
# sigma2_z1z2_fft
# ----------------------------------------------------------------------------- #
class TestSigma2Z1Z2Fft:
    """Tests for the sample-variance kernel sigma^2(z1, z2)."""

    @pytest.fixture
    def z_grid(self):
        return np.linspace(0.1, 1.5, 6)

    @pytest.fixture
    def k_grid(self):
        return np.geomspace(1e-4, 10.0, 100)

    def test_shape(self, cosmo, z_grid, k_grid):
        """Output is a (len(z1), len(z2)) matrix."""
        out = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'full_curved_sky',
            None, None, None, nk_fft=NK_FFT_TEST,
        )
        assert out.shape == (z_grid.size, z_grid.size)

    def test_symmetric_full_sky(self, cosmo, z_grid, k_grid):
        """sigma^2(z1, z2) == sigma^2(z2, z1): the kernel is symmetric in z."""
        out = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'full_curved_sky',
            None, None, None, nk_fft=NK_FFT_TEST,
        )
        np.testing.assert_allclose(out, out.T, rtol=0, atol=0)

    def test_symmetric_masked(self, cosmo, z_grid, k_grid):
        """The masked branch is symmetric in z as well."""
        ell_mask = np.arange(50.0)
        cl_mask = np.full(50, 1e-3)
        out = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'from_input_mask',
            ell_mask, cl_mask, 0.3, nk_fft=NK_FFT_TEST,
        )
        np.testing.assert_allclose(out, out.T)

    def test_diagonal_positive(self, cosmo, z_grid, k_grid):
        """The variance sigma^2(z, z) on the diagonal must be positive."""
        out = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'full_curved_sky',
            None, None, None, nk_fft=NK_FFT_TEST,
        )
        assert np.all(np.diag(out) > 0)

    def test_masked_is_full_sky_times_constant(self, cosmo, z_grid, k_grid):
        """The masked kernel equals the full-sky one times a z-independent factor.

        From the source, the only difference between the two branches is the
        prefactor ``part_result / (4 pi fsky)**2`` versus ``1 / (2 pi**2)``;
        both multiply the same g1 g2 * integral. This pins that algebra exactly.
        """
        ell_mask = np.arange(50.0)
        cl_mask = np.full(50, 1e-3)
        fsky = 0.3

        full = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'full_curved_sky',
            None, None, None, nk_fft=NK_FFT_TEST,
        )
        masked = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'from_input_mask',
            ell_mask, cl_mask, fsky, nk_fft=NK_FFT_TEST,
        )

        part_result = np.sum((2 * ell_mask + 1) * cl_mask) * 2.0 / np.pi
        const = part_result * (2.0 * np.pi**2) / (4.0 * np.pi * fsky) ** 2

        np.testing.assert_allclose(masked, full * const)

    def test_polar_cap_matches_input_mask(self, cosmo, z_grid, k_grid):
        """'polar_cap_on_the_fly' and 'from_input_mask' share the same branch."""
        ell_mask = np.arange(50.0)
        cl_mask = np.full(50, 1e-3)
        kwargs = dict(
            ell_mask=ell_mask, cl_mask=cl_mask, fsky_mask=0.3, nk_fft=NK_FFT_TEST
        )
        polar = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'polar_cap_on_the_fly', **kwargs
        )
        from_mask = cov_ssc.sigma2_z1z2_fft(
            z_grid, z_grid, k_grid, cosmo, 'from_input_mask', **kwargs
        )
        np.testing.assert_allclose(polar, from_mask)

    def test_invalid_option_raises(self, cosmo, z_grid, k_grid):
        """An unknown which_sigma2_b option is rejected."""
        with pytest.raises(ValueError, match='Invalid which_sigma2_b'):
            cov_ssc.sigma2_z1z2_fft(
                z_grid, z_grid, k_grid, cosmo, 'not_a_real_option',
                None, None, None, nk_fft=NK_FFT_TEST,
            )

    def test_mismatched_z_arrays_raise(self, cosmo, z_grid, k_grid):
        """z1_arr and z2_arr must be equal (the code asserts this)."""
        with pytest.raises(AssertionError):
            cov_ssc.sigma2_z1z2_fft(
                z_grid, z_grid + 0.01, k_grid, cosmo, 'full_curved_sky',
                None, None, None, nk_fft=NK_FFT_TEST,
            )

    def test_scalar_z_is_promoted(self, cosmo, k_grid):
        """A scalar z is promoted to 1d, giving a (1, 1) result."""
        out = cov_ssc.sigma2_z1z2_fft(
            0.5, 0.5, k_grid, cosmo, 'full_curved_sky',
            None, None, None, nk_fft=NK_FFT_TEST,
        )
        assert out.shape == (1, 1)


# ----------------------------------------------------------------------------- #
# Helpers for the einsum integral tests
# ----------------------------------------------------------------------------- #
def _brute_force_ssc_2d(d_ab, d_cd, pref, sigma2, dz, weights):
    """Explicit loop reference for ssc_integral_4D_simps_jax (2D sigma2)."""
    nbl, zp_ab, _ = d_ab.shape
    zp_cd = d_cd.shape[1]
    zsteps = pref.size
    out = np.zeros((nbl, nbl, zp_ab, zp_cd))
    for ll in range(nbl):
        for mm in range(nbl):
            for i in range(zp_ab):
                for j in range(zp_cd):
                    acc = 0.0
                    for z in range(zsteps):
                        for w in range(zsteps):
                            acc += (
                                d_ab[ll, i, z] * d_cd[mm, j, w]
                                * pref[z] * pref[w]
                                * weights[z] * weights[w]
                                * sigma2[z, w]
                            )
                    out[ll, mm, i, j] = acc * dz**2
    return out


def _brute_force_ssc_ke(d_ab, d_cd, pref, sigma2, dz, weights):
    """Explicit loop reference for the KE-approximation (1D sigma2)."""
    nbl, zp_ab, _ = d_ab.shape
    zp_cd = d_cd.shape[1]
    zsteps = pref.size
    out = np.zeros((nbl, nbl, zp_ab, zp_cd))
    for ll in range(nbl):
        for mm in range(nbl):
            for i in range(zp_ab):
                for j in range(zp_cd):
                    acc = 0.0
                    for z in range(zsteps):
                        acc += (
                            d_ab[ll, i, z] * d_cd[mm, j, z]
                            * pref[z] * weights[z] * sigma2[z]
                        )
                    out[ll, mm, i, j] = acc * dz
    return out


@pytest.fixture
def ssc_inputs(rng):
    """A small set of random inputs for the einsum integrals."""
    nbl, zp_ab, zp_cd, zsteps = 2, 3, 4, 9
    d_ab = rng.standard_normal((nbl, zp_ab, zsteps))
    d_cd = rng.standard_normal((nbl, zp_cd, zsteps))
    pref = rng.standard_normal(zsteps)
    weights = rng.standard_normal(zsteps)
    dz = 0.1
    sigma2_2d = rng.standard_normal((zsteps, zsteps))
    sigma2_2d = 0.5 * (sigma2_2d + sigma2_2d.T)  # symmetric, like the real kernel
    sigma2_1d = rng.standard_normal(zsteps)
    return dict(
        d_ab=d_ab, d_cd=d_cd, pref=pref, weights=weights, dz=dz,
        sigma2_2d=sigma2_2d, sigma2_1d=sigma2_1d,
        nbl=nbl, zp_ab=zp_ab, zp_cd=zp_cd, zsteps=zsteps,
    )


# ----------------------------------------------------------------------------- #
# ssc_integral_4D_simps_jax  (2D sigma2)
# ----------------------------------------------------------------------------- #
class TestSscIntegral2D:
    """Tests for the full (non-KE) Simpson-rule SSC integral."""

    def _call(self, p):
        return np.array(
            cov_ssc.ssc_integral_4D_simps_jax(
                jnp.array(p['d_ab']), jnp.array(p['d_cd']),
                jnp.array(p['pref']), jnp.array(p['sigma2_2d']),
                p['dz'], jnp.array(p['weights']),
            )
        )

    def test_shape(self, ssc_inputs):
        """Result has shape (nbl, nbl, zpairs_AB, zpairs_CD)."""
        out = self._call(ssc_inputs)
        assert out.shape == (
            ssc_inputs['nbl'], ssc_inputs['nbl'],
            ssc_inputs['zp_ab'], ssc_inputs['zp_cd'],
        )

    def test_matches_brute_force(self, ssc_inputs):
        """The jitted einsum equals an explicit nested-loop reference."""
        out = self._call(ssc_inputs)
        ref = _brute_force_ssc_2d(
            ssc_inputs['d_ab'], ssc_inputs['d_cd'], ssc_inputs['pref'],
            ssc_inputs['sigma2_2d'], ssc_inputs['dz'], ssc_inputs['weights'],
        )
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_block_symmetry(self, rng):
        """With the same array for AB and CD and a symmetric sigma2,
        result[L, M, i, j] == result[M, L, j, i]."""
        nbl, zp, zsteps = 3, 4, 7
        d = rng.standard_normal((nbl, zp, zsteps))
        pref = rng.standard_normal(zsteps)
        weights = rng.standard_normal(zsteps)
        sig = rng.standard_normal((zsteps, zsteps))
        sig = 0.5 * (sig + sig.T)

        out = np.array(
            cov_ssc.ssc_integral_4D_simps_jax(
                jnp.array(d), jnp.array(d), jnp.array(pref),
                jnp.array(sig), 0.1, jnp.array(weights),
            )
        )
        np.testing.assert_allclose(out, out.transpose(1, 0, 3, 2), rtol=1e-5)

    def test_linear_in_dab(self, ssc_inputs):
        """Scaling d_ab by alpha scales the result by alpha."""
        p = ssc_inputs
        base = self._call(p)
        scaled = np.array(
            cov_ssc.ssc_integral_4D_simps_jax(
                jnp.array(3.0 * p['d_ab']), jnp.array(p['d_cd']),
                jnp.array(p['pref']), jnp.array(p['sigma2_2d']),
                p['dz'], jnp.array(p['weights']),
            )
        )
        np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-4)


# ----------------------------------------------------------------------------- #
# ssc_integral_4D_simps_jax_ke_approx  (1D sigma2)
# ----------------------------------------------------------------------------- #
class TestSscIntegralKE:
    """Tests for the KE-approximation Simpson-rule SSC integral."""

    def _call(self, p):
        return np.array(
            cov_ssc.ssc_integral_4D_simps_jax_ke_approx(
                jnp.array(p['d_ab']), jnp.array(p['d_cd']),
                jnp.array(p['pref']), jnp.array(p['sigma2_1d']),
                p['dz'], jnp.array(p['weights']),
            )
        )

    def test_shape(self, ssc_inputs):
        """Result has shape (nbl, nbl, zpairs_AB, zpairs_CD)."""
        out = self._call(ssc_inputs)
        assert out.shape == (
            ssc_inputs['nbl'], ssc_inputs['nbl'],
            ssc_inputs['zp_ab'], ssc_inputs['zp_cd'],
        )

    def test_matches_brute_force(self, ssc_inputs):
        """The jitted einsum equals an explicit nested-loop reference."""
        out = self._call(ssc_inputs)
        ref = _brute_force_ssc_ke(
            ssc_inputs['d_ab'], ssc_inputs['d_cd'], ssc_inputs['pref'],
            ssc_inputs['sigma2_1d'], ssc_inputs['dz'], ssc_inputs['weights'],
        )
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_block_symmetry(self, rng):
        """With the same array for AB and CD,
        result[L, M, i, j] == result[M, L, j, i]."""
        nbl, zp, zsteps = 3, 4, 7
        d = rng.standard_normal((nbl, zp, zsteps))
        pref = rng.standard_normal(zsteps)
        weights = rng.standard_normal(zsteps)
        sig = rng.standard_normal(zsteps)

        out = np.array(
            cov_ssc.ssc_integral_4D_simps_jax_ke_approx(
                jnp.array(d), jnp.array(d), jnp.array(pref),
                jnp.array(sig), 0.1, jnp.array(weights),
            )
        )
        np.testing.assert_allclose(out, out.transpose(1, 0, 3, 2), rtol=1e-5)

    def test_linear_in_dab(self, ssc_inputs):
        """Scaling d_ab by alpha scales the result by alpha."""
        p = ssc_inputs
        base = self._call(p)
        scaled = np.array(
            cov_ssc.ssc_integral_4D_simps_jax_ke_approx(
                jnp.array(-2.0 * p['d_ab']), jnp.array(p['d_cd']),
                jnp.array(p['pref']), jnp.array(p['sigma2_1d']),
                p['dz'], jnp.array(p['weights']),
            )
        )
        np.testing.assert_allclose(scaled, -2.0 * base, rtol=1e-5)
