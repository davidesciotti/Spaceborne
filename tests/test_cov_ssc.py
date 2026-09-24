"""Unit tests for the spaceborne.cov_ssc module (super-sample covariance).

The SSC machinery has two well-isolated, pure numerical pieces that we can pin
down without standing up the full pipeline:

* ``sigma2_z1z2`` -- the sample-variance kernel sigma^2(z1, z2). It splits
  ``W(mu) = sum_L w_L P_L(mu)`` into a monopole term (evaluated via an FFT
  cosine-transform of the power spectrum) plus a mask-shape correction (evaluated
  on a coarse z grid and spline-interpolated). We cross-check it against an
  *independent* brute-force sum over Legendre multipoles, done directly with
  ``scipy.special.spherical_jn`` and ``scipy.integrate.simpson`` on a fine linear
  k grid -- this shares no code with ``sigma2_z1z2`` and is the audit's
  reproduction target (a mask-shape-blind formula must fail it).

* ``interp_uniform`` / ``linear_pk_transforms`` -- the small numerical building
  blocks ``sigma2_z1z2`` is built from.

* ``ssc_integral_4D_simps_jax`` / ``..._ke_approx`` -- the JAX Simpson-rule
  contractions that assemble the 4D covariance. We check them against an explicit
  brute-force ``for``-loop reference (non-circular), plus the symmetry and
  linearity that follow from the einsum structure.

The ``nk_fft`` of ``sigma2_z1z2`` is deliberately set far below the production
default (2**21) so the FFT is cheap; this is verified empirically below to still
be accurate to << the tolerances used in the tests.
"""

import jax.numpy as jnp
import numpy as np
import pyccl as ccl
import pytest
from scipy.integrate import simpson
from scipy.special import spherical_jn

from spaceborne import cosmo_lib, cov_ssc


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=1234)


@pytest.fixture(scope='module')
def cosmo():
    """A cheap-to-evaluate vanilla LCDM cosmology.

    ``eisenstein_hu`` + ``halofit`` avoid a Boltzmann-solver call. ``sigma2_z1z2``
    only ever calls ``ccl.linear_matter_power``, which returns the same, purely
    linear spectrum regardless of the ``matter_power_spectrum`` (nonlinear)
    setting, so the halofit choice here has no effect on the tests below.
    """
    return ccl.CosmologyVanillaLCDM(
        transfer_function='eisenstein_hu', matter_power_spectrum='halofit'
    )


# A small, fast FFT length. The default is 2**21; empirically it agrees with
# the independent Bessel reference to better than 1e-4 (scaled, see
# `_max_scaled_abs_diff`) at this resolution.
NK_FFT_TEST = 2**16

K_MIN, K_MAX = 1e-4, 1.0
Z_ARR = np.array([0.1, 0.3, 1.0])


def _bessel_sigma2_ref(
    cosmo_ccl, w_l, z_arr=Z_ARR, k_min=K_MIN, k_max=K_MAX, nk=80_000
):
    r"""Independent brute-force reference for sigma2_z1z2.

    Directly sums, on a fine *linear* k grid,

        sigma2(z1, z2) = D1 D2 sum_L w_L (2/pi) int dk k^2 P(k) j_L(k chi1) j_L(k chi2)

    using ``scipy.special.spherical_jn`` and ``scipy.integrate.simpson``. Shares
    no code with ``sigma2_z1z2`` (FFT + Legendre addition theorem), so it is a
    genuine cross-check.

    Convergence was checked by hand while writing these tests: for the
    monopole-only case (k_max=1) doubling ``nk`` from 60_000 to 120_000 changes
    the result by <= 1.3e-7 relative, so nk=80_000 (default here) is amply
    converged relative to the tolerances used below.
    """
    z_arr = np.atleast_1d(z_arr)
    a_arr = cosmo_lib.z_to_a(z_arr)
    chi = ccl.comoving_radial_distance(cosmo_ccl, a_arr)
    growth = ccl.growth_factor(cosmo_ccl, a_arr)

    k = np.linspace(k_min, k_max, nk)
    pk = ccl.linear_matter_power(cosmo_ccl, k=k, a=1.0)

    n = z_arr.size
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            kernel = np.zeros(nk)
            for ell, w_ell in enumerate(w_l):
                if w_ell == 0:
                    continue
                kernel += (
                    w_ell
                    * spherical_jn(ell, k * chi[i])
                    * spherical_jn(ell, k * chi[j])
                )
            val = simpson(k**2 * pk * kernel, x=k) * 2.0 / np.pi
            out[i, j] = out[j, i] = val * growth[i] * growth[j]
    return out


def _max_scaled_abs_diff(a, b):
    """max_ij |a_ij - b_ij| / sqrt(diag_i(b) * diag_j(b)).

    Off-diagonal sigma2 entries are near-cancelling differences of much larger
    terms (the kernel correlates z1, z2 weakly unless z1 ~ z2), so their
    *relative* error is not a meaningful comparison -- a tiny absolute
    difference can be a huge fraction of a near-zero entry. Scaling instead by
    the geometric mean of the diagonal entries (which set the overall scale of
    the matrix) gives a single, dimensionally-consistent error measure that is
    well-behaved for the whole matrix, diagonal and off-diagonal alike.
    """
    diag = np.diag(b)
    scale = np.sqrt(np.abs(np.outer(diag, diag)))
    return np.max(np.abs(a - b) / scale)


# ----------------------------------------------------------------------------- #
# sigma2_z1z2
# ----------------------------------------------------------------------------- #
class TestSigma2Z1Z2:
    """Tests for the sample-variance kernel sigma^2(z1, z2)."""

    @pytest.fixture
    def z_grid(self):
        return np.linspace(0.1, 1.5, 6)

    def test_shape(self, cosmo, z_grid):
        """Output is a (len(z), len(z)) matrix."""
        w = np.array([1 / (4 * np.pi)])
        out = cov_ssc.sigma2_z1z2(z_grid, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST)
        assert out.shape == (z_grid.size, z_grid.size)

    def test_symmetric(self, cosmo, z_grid):
        """sigma^2(z1, z2) == sigma^2(z2, z1)."""
        rng = np.random.default_rng(0)
        w = rng.uniform(0, 0.3, size=11) / (4 * np.pi)
        w[0] = 1 / (4 * np.pi)
        out = cov_ssc.sigma2_z1z2(z_grid, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST)
        np.testing.assert_allclose(out, out.T, rtol=0, atol=0)

    def test_scalar_z_is_promoted(self, cosmo):
        """A scalar z is promoted to 1d (via np.atleast_1d), giving (1, 1)."""
        w = np.array([1 / (4 * np.pi)])
        out = cov_ssc.sigma2_z1z2(0.5, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST)
        assert out.shape == (1, 1)

    def test_matches_bessel_reference_full_sky(self, cosmo):
        """Monopole-only (full-sky) window matches the independent reference,
        off-diagonals included."""
        w = np.array([1 / (4 * np.pi)])
        out = cov_ssc.sigma2_z1z2(Z_ARR, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST)
        ref = _bessel_sigma2_ref(cosmo, w)
        # empirically ~1e-6; 1e-3 leaves ample margin
        assert _max_scaled_abs_diff(out, ref) < 1e-3

    def test_matches_bessel_reference_masked(self, cosmo):
        """A synthetic mask spectrum with power at several L <= 10 matches the
        independent reference, off-diagonals included."""
        rng = np.random.default_rng(123)
        l_max = 10
        w = np.zeros(l_max + 1)
        w[0] = 1 / (4 * np.pi)
        w[1:] = rng.uniform(0, 0.3, size=l_max) / (4 * np.pi)

        out = cov_ssc.sigma2_z1z2(Z_ARR, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST)
        ref = _bessel_sigma2_ref(cosmo, w)
        # empirically ~2e-5; 1e-3 leaves ample margin
        assert _max_scaled_abs_diff(out, ref) < 1e-3

    def test_mask_shape_sensitivity(self, cosmo):
        """Audit N01 reproduction: two windows with the *same monopole* but
        extra power at different L must give *different* sigma2, and each must
        still match its own independent reference.

        The pre-fix formula multiplied the monopole-only integral by
        ``sum_L w_L`` -- which is identical for the two windows below -- so it
        would predict *exactly equal* sigma2 for both, failing this test by
        ~40x the tolerance used here.
        """
        l_max = 10
        w_l1 = np.zeros(l_max + 1)
        w_l1[0] = 1 / (4 * np.pi)
        w_l1[1] = 0.3 / (4 * np.pi)

        w_l10 = np.zeros(l_max + 1)
        w_l10[0] = 1 / (4 * np.pi)
        w_l10[10] = 0.3 / (4 * np.pi)

        assert np.sum(w_l1) == pytest.approx(np.sum(w_l10))  # same monopole

        out_l1 = cov_ssc.sigma2_z1z2(
            Z_ARR, K_MIN, K_MAX, cosmo, w_l1, nk_fft=NK_FFT_TEST
        )
        out_l10 = cov_ssc.sigma2_z1z2(
            Z_ARR, K_MIN, K_MAX, cosmo, w_l10, nk_fft=NK_FFT_TEST
        )

        ref_l1 = _bessel_sigma2_ref(cosmo, w_l1)
        ref_l10 = _bessel_sigma2_ref(cosmo, w_l10)

        assert _max_scaled_abs_diff(out_l1, ref_l1) < 1e-3
        assert _max_scaled_abs_diff(out_l10, ref_l10) < 1e-3
        # the mask-shape difference (~4e-2 scaled, empirically) is far above the
        # ~1e-3 noise floor of the code-vs-reference comparisons above
        assert _max_scaled_abs_diff(out_l1, out_l10) > 1e-2

    def test_linear_in_w(self, cosmo):
        """sigma2_z1z2 is linear in cl_footp_norm: scaling w by a constant
        scales the output by the same constant."""
        rng = np.random.default_rng(321)
        w = rng.uniform(0, 0.3, size=8) / (4 * np.pi)
        w[0] = 1 / (4 * np.pi)

        base = cov_ssc.sigma2_z1z2(Z_ARR, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST)
        scaled = cov_ssc.sigma2_z1z2(
            Z_ARR, K_MIN, K_MAX, cosmo, 2.5 * w, nk_fft=NK_FFT_TEST
        )
        np.testing.assert_allclose(scaled, 2.5 * base, rtol=1e-8)

    def test_hybrid_interpolation_matches_direct(self, cosmo):
        """For a moderately sized z grid, the coarse-grid + spline-interpolation
        path (n_z_coarse=40) agrees with the direct, no-interpolation path
        (n_z_coarse=10**6, i.e. always above z_grid.size)."""
        rng = np.random.default_rng(7)
        l_max = 20
        w = np.zeros(l_max + 1)
        w[0] = 1 / (4 * np.pi)
        w[1:] = rng.uniform(0, 0.2, size=l_max) / (4 * np.pi)

        z_grid = np.linspace(0.05, 2, 120)
        direct = cov_ssc.sigma2_z1z2(
            z_grid, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST, n_z_coarse=10**6
        )
        interp = cov_ssc.sigma2_z1z2(
            z_grid, K_MIN, K_MAX, cosmo, w, nk_fft=NK_FFT_TEST, n_z_coarse=40
        )
        # empirically ~1e-2; 2e-2 leaves a ~2x margin
        assert _max_scaled_abs_diff(interp, direct) < 2e-2


# ----------------------------------------------------------------------------- #
# interp_uniform
# ----------------------------------------------------------------------------- #
class TestInterpUniform:
    """Tests for the fast uniform-grid linear interpolator."""

    def test_reproduces_linear_function_exactly(self):
        dx = 0.05
        grid = np.arange(200) * dx
        y = 3.0 * grid + 1.5
        rng = np.random.default_rng(0)
        x = rng.uniform(0, grid[-1], size=1000)
        out = cov_ssc.interp_uniform(x, dx, y)
        np.testing.assert_allclose(out, 3.0 * x + 1.5, rtol=0, atol=1e-11)

    def test_matches_np_interp_on_smooth_function(self):
        dx = 0.05
        grid = np.arange(200) * dx
        y = np.sin(grid) + 0.1 * grid**2
        rng = np.random.default_rng(1)
        x = rng.uniform(0, grid[-1], size=1000)
        out = cov_ssc.interp_uniform(x, dx, y)
        ref = np.interp(x, grid, y)
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-11)

    def test_raises_beyond_table(self):
        """Points beyond the tabulated range raise instead of extrapolating."""
        y = np.arange(10.0)
        with pytest.raises(ValueError, match='beyond the table range'):
            cov_ssc.interp_uniform(np.array([9.5]), 1.0, y)


# ----------------------------------------------------------------------------- #
# linear_pk_transforms
# ----------------------------------------------------------------------------- #
class TestLinearPkTransforms:
    """Tests for the FFT-based transforms of the z=0 linear power spectrum."""

    def test_k_sampling_covers_large_separations(self, cosmo):
        """A too-coarse nk_fft is refined so that pi/dk covers the separations up to
        2 chi(z_max): the result must not depend on the requested nk_fft. Without
        the refinement, nk_fft=2**8 only reaches r ~ 800 Mpc."""
        z_grid = np.array([0.5, 1.5, 3.0])
        w = np.array([1 / (4 * np.pi)])
        coarse = cov_ssc.sigma2_z1z2(z_grid, K_MIN, K_MAX, cosmo, w, nk_fft=2**8)
        fine = cov_ssc.sigma2_z1z2(z_grid, K_MIN, K_MAX, cosmo, w, nk_fft=2**16)
        # empirically ~1e-5 (the unrefined grid gives O(1) errors)
        assert _max_scaled_abs_diff(coarse, fine) < 1e-4

    def test_xi_matches_direct_quadrature(self, cosmo):
        """xi(r) at a few r agrees with a direct quadrature of
        1/(2 pi^2) int k^2 P(k) j0(k r) dk over the same k range."""
        dr, _cos_transform, xi = cov_ssc.linear_pk_transforms(
            cosmo, K_MIN, K_MAX, r_max=200.0, nk_fft=2**18, fft_pad=8
        )

        r_test = np.array([1.0, 5.0, 20.0, 80.0])
        k = np.linspace(K_MIN, K_MAX, 500_000)
        pk = ccl.linear_matter_power(cosmo, k=k, a=1.0)
        xi_direct = np.array(
            [
                simpson(k**2 * pk * spherical_jn(0, k * r), x=k) / (2 * np.pi**2)
                for r in r_test
            ]
        )
        xi_code = cov_ssc.interp_uniform(r_test, dr, xi)
        # empirically <= ~1.2%; 3% leaves a margin
        np.testing.assert_allclose(xi_code, xi_direct, rtol=3e-2)


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
        # The swap symmetry is mathematically exact, but the einsum contracts in
        # a fixed order that is not symmetric under the index swap, so it only
        # holds to floating-point precision (rtol 1e-5 is too tight on some BLAS).
        np.testing.assert_allclose(out, out.transpose(1, 0, 3, 2), rtol=1e-4)

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
        # Exact swap symmetry only holds to floating-point precision because the
        # einsum reduction order is not symmetric under the index swap.
        np.testing.assert_allclose(out, out.transpose(1, 0, 3, 2), rtol=1e-4)

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
