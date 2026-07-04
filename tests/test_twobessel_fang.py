"""Unit tests for spaceborne.twobessel_fang (FFTLog double-Bessel transforms).

The FFTLog machinery is hard to validate end-to-end at unit-test cost (it needs
large, finely log-spaced grids to be numerically accurate), so this module pins
down the pure numerical building blocks instead:

* ``log_extrap`` -- log-linear extrapolation of a log-spaced grid; the original
  segment must be preserved exactly and the extension must keep the same
  log-spacing.
* ``c_window`` -- the raised-cosine tapering window used to smooth the FFT
  coefficients; it is exactly 1 in the untouched region and exactly 0 at the
  outermost points.
* ``g_m_vals`` / ``g_l`` -- the (complex) gamma-function ratios that encode the
  Bessel-function FFTLog kernel. Both the "normal" branch (``|Im(q)| <= 200``)
  and the asymptotic Stirling-series branch (``|Im(q)| > 200``) are checked
  against a ``scipy.special.loggamma``-based reference, which is numerically
  stable for large imaginary arguments where ``scipy.special.gamma`` itself
  would overflow.
* ``bilinear_extra_P`` -- 2D bilinear log-space extrapolation of a strictly
  positive matrix; the original block must be preserved and the extrapolated
  values must stay positive.

``TwoSphBessel`` / ``TwoBessel`` are only smoke-tested: instantiate on a small
log-spaced grid with a power-law input and check the output shapes and
finiteness, following the same nu1/nu2 defaults used in
``cov_real_space.proj_cov_2d_fftlog``.
"""

import numpy as np
import pytest
from scipy.special import loggamma

from spaceborne.twobessel_fang import (
    TwoBessel,
    TwoSphBessel,
    bilinear_extra_P,
    c_window,
    g_l,
    g_m_vals,
    log_extrap,
)


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=42)


# ----------------------------------------------------------------------------- #
# log_extrap
# ----------------------------------------------------------------------------- #
class TestLogExtrap:
    """Tests for the log-linear grid extrapolation."""

    def test_preserves_original_segment(self):
        """The original x values must appear unchanged in the output."""
        x = np.geomspace(1.0, 100.0, 10)
        x_ext = log_extrap(x, 3, 4)

        assert x_ext.size == x.size + 3 + 4
        np.testing.assert_allclose(x_ext[3:13], x)

    def test_log_linear_extension(self):
        """The extended points keep the same log-spacing as the original grid."""
        x = np.geomspace(1.0, 100.0, 10)
        x_ext = log_extrap(x, 3, 4)

        ratios = x_ext[1:] / x_ext[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)

    def test_no_extrapolation_is_identity(self):
        """With zero extrapolation points on both sides, x is unchanged."""
        x = np.geomspace(1.0, 100.0, 10)
        x_ext = log_extrap(x, 0, 0)
        np.testing.assert_allclose(x_ext, x)

    def test_asymmetric_extrapolation(self):
        """Extrapolating on one side only leaves the other side untouched."""
        x = np.geomspace(1.0, 100.0, 10)
        x_ext = log_extrap(x, 5, 0)

        assert x_ext.size == x.size + 5
        np.testing.assert_allclose(x_ext[5:], x)


# ----------------------------------------------------------------------------- #
# c_window
# ----------------------------------------------------------------------------- #
class TestCWindow:
    """Tests for the raised-cosine FFT-coefficient window."""

    def test_flat_region_is_unity(self):
        """Away from the edges (beyond n_cut), the window equals 1 exactly."""
        n = np.arange(-10, 11)
        n_cut = 4
        w = c_window(n, n_cut)

        flat_mask = (n > n[0] + n_cut) & (n < n[-1] - n_cut)
        np.testing.assert_array_equal(w[flat_mask], 1.0)

    def test_edges_taper_to_zero(self):
        """The window vanishes exactly at the outermost grid points."""
        n = np.arange(-10, 11)
        n_cut = 4
        w = c_window(n, n_cut)

        assert w[0] == 0.0
        assert w[-1] == 0.0

    def test_window_bounded(self):
        """The window stays within [0, 1] everywhere."""
        n = np.arange(-20, 21)
        n_cut = 8
        w = c_window(n, n_cut)

        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_symmetric_window(self):
        """The window is symmetric for a symmetric n array and cut."""
        n = np.arange(-15, 16)
        n_cut = 5
        w = c_window(n, n_cut)
        np.testing.assert_allclose(w, w[::-1])


# ----------------------------------------------------------------------------- #
# g_m_vals / g_l
# ----------------------------------------------------------------------------- #
def _analytic_g_m(mu, q):
    """Direct gamma-ratio reference, stable for large imaginary q via loggamma."""
    alpha_plus = (mu + 1 + q) / 2.0
    alpha_minus = (mu + 1 - q) / 2.0
    return np.exp(loggamma(alpha_plus) - loggamma(alpha_minus))


def _analytic_g_l(ell, z):
    """gl = 2**z * gamma((l+z)/2) / gamma((3+l-z)/2), per the g_l docstring."""
    log_num = loggamma((ell + z) / 2.0)
    log_den = loggamma((3.0 + ell - z) / 2.0)
    return 2.0**z * np.exp(log_num - log_den)


class TestGMVals:
    """Tests for the gamma-ratio helper g_m_vals."""

    def test_matches_analytic_normal_branch(self):
        """|Im(q)| <= 200 uses the direct gamma ratio."""
        mu = 1.5
        q = np.array([0.5 + 0.5j, -0.3 - 2.0j, 1.2 + 10.0j, 0.0 + 0.0j])
        out = g_m_vals(mu, q)
        ref = _analytic_g_m(mu, q)
        np.testing.assert_allclose(out, ref, rtol=1e-10)

    def test_matches_analytic_asymptotic_branch(self):
        """|Im(q)| > 200 uses the Stirling-series asymptotic expansion."""
        mu = 1.5
        q = np.array([0.5 + 300.0j, -0.3 - 250.0j])
        out = g_m_vals(mu, q)
        ref = _analytic_g_m(mu, q)
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_pole_gives_zero(self):
        """q == mu + 1 hits a gamma pole in the denominator; the code special-cases
        this to exactly 0."""
        mu = 1.5
        q = np.array([mu + 1 + 0.0j])
        out = g_m_vals(mu, q)
        np.testing.assert_array_equal(out, np.array([0.0 + 0.0j]))


class TestGL:
    """Tests for g_l against the analytic gamma-ratio formula in its docstring."""

    @pytest.mark.parametrize('ell', [0.5, 2.5, 4.5])
    def test_matches_analytic_formula(self, ell):
        z = np.array([1.0 + 0.5j, 1.2 - 2.0j, 0.8 + 10.0j, 2.0 + 0.0j])
        out = g_l(ell, z)
        ref = _analytic_g_l(ell, z)
        np.testing.assert_allclose(out, ref, rtol=1e-10)

    def test_finite_for_real_z(self):
        """g_l should be finite for a real-only z array (eta_m = 0 case)."""
        z = np.array([1.01, 1.01, 1.01])
        out = g_l(2.0, z)
        assert np.all(np.isfinite(out))


# ----------------------------------------------------------------------------- #
# bilinear_extra_P
# ----------------------------------------------------------------------------- #
class TestBilinearExtraP:
    """Tests for the 2D bilinear log-space matrix extrapolation."""

    def test_preserves_original_block(self):
        """The original matrix must reappear unchanged in the padded output."""
        mat = np.outer(np.geomspace(1.0, 10.0, 5), np.geomspace(1.0, 10.0, 6))
        out = bilinear_extra_P(mat, 2, 3)

        assert out.shape == (5 + 2 + 3, 6 + 2 + 3)
        np.testing.assert_allclose(out[2:7, 2:8], mat)

    def test_output_positive(self):
        """Log-space extrapolation of a positive matrix stays positive."""
        mat = np.outer(np.geomspace(1.0, 10.0, 5), np.geomspace(1.0, 10.0, 6))
        out = bilinear_extra_P(mat, 3, 3)
        assert np.all(out > 0)

    def test_no_padding_is_identity(self):
        """Zero extrapolation on both sides leaves the matrix unchanged."""
        mat = np.outer(np.geomspace(1.0, 10.0, 4), np.geomspace(1.0, 10.0, 4))
        out = bilinear_extra_P(mat, 0, 0)
        np.testing.assert_allclose(out, mat)

    def test_rejects_nonpositive_input(self):
        """The function requires strictly positive values for the log step."""
        mat = np.ones((4, 4))
        mat[0, 0] = 0.0
        with pytest.raises(ValueError, match='positive values'):
            bilinear_extra_P(mat, 1, 1)


# ----------------------------------------------------------------------------- #
# TwoSphBessel / TwoBessel -- smoke tests only
# ----------------------------------------------------------------------------- #
class TestTwoBesselSmoke:
    """Cheap smoke tests: shapes and finiteness on a small log-spaced grid.

    A full analytic check of the FFTLog double-Bessel transform needs finely
    resolved, wide-range grids to converge, which is too expensive for a unit
    test; we only check that the machinery runs and produces sane output.
    """

    @pytest.fixture
    def power_law_grid(self):
        n = 64
        x1 = np.geomspace(1.0, 1000.0, n)
        x2 = np.geomspace(1.0, 1000.0, n)
        fx1x2 = np.outer(x1**-1.5, x2**-1.5)
        return x1, x2, fx1x2

    def test_two_bessel_binave_shape_and_finite(self, power_law_grid):
        x1, x2, fx1x2 = power_law_grid
        tb = TwoBessel(x1, x2, fx1x2, nu1=1.01, nu2=1.01)

        dln1 = np.log(x1[1] / x1[0])
        dln2 = np.log(x2[1] / x2[0])
        y1, y2, out = tb.two_Bessel_binave(0, 0, dln1, dln2)

        assert y1.shape == (x1.size,)
        assert y2.shape == (x2.size,)
        assert out.shape == (x1.size, x2.size)
        assert np.all(np.isfinite(out))

    def test_two_sph_bessel_shape_and_finite(self, power_law_grid):
        x1, x2, fx1x2 = power_law_grid
        tsb = TwoSphBessel(x1, x2, fx1x2, nu1=1.01, nu2=1.01)

        y1, y2, out = tsb.two_sph_bessel(0, 0)

        assert y1.shape == (x1.size,)
        assert y2.shape == (x2.size,)
        assert out.shape == (x1.size, x2.size)
        assert np.all(np.isfinite(out))

    def test_extrapolation_and_padding_run(self, power_law_grid):
        """N_extrap_* and N_pad paths should also run and stay finite."""
        x1, x2, fx1x2 = power_law_grid
        tb = TwoBessel(
            x1, x2, fx1x2, nu1=1.01, nu2=1.01, N_extrap_low=8, N_extrap_high=8, N_pad=8
        )

        dln1 = np.log(x1[1] / x1[0])
        dln2 = np.log(x2[1] / x2[0])
        y1, y2, out = tb.two_Bessel_binave(2, 2, dln1, dln2)

        assert out.shape == (x1.size, x2.size)
        assert np.all(np.isfinite(out))

    def test_odd_size_raises(self, power_law_grid):
        """The constructor requires an even total array size."""
        x1, x2, fx1x2 = power_law_grid
        x1_odd = x1[:-1]
        fx1x2_odd = fx1x2[:-1, :]
        with pytest.raises(ValueError, match='must be even'):
            TwoBessel(x1_odd, x2, fx1x2_odd, nu1=1.01, nu2=1.01)
