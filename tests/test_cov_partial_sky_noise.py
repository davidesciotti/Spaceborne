"""Unit tests for the noise helpers in spaceborne.cov_partial_sky."""

import healpy as hp
import numpy as np
import pytest

from spaceborne import cov_partial_sky


class TestCoupledNoiseFactor:
    """Tests for coupled_noise_factor."""

    def test_binary_footprint_is_one(self):
        footprint = np.zeros(hp.nside2npix(16))
        footprint[:500] = 1.0
        factor = cov_partial_sky.coupled_noise_factor(
            footprint, fsky=np.mean(footprint), zbins=3
        )
        assert factor.shape == (1, 3, 1)
        np.testing.assert_allclose(factor, 1.0, rtol=1e-14)

    def test_fractional_coverage(self):
        # footprint-only run with a fractional coverage V: <V> / <V^2>
        rng = np.random.default_rng(0)
        cov_map = np.zeros(hp.nside2npix(16))
        cov_map[:500] = rng.uniform(0.2, 1.0, 500)
        factor = cov_partial_sky.coupled_noise_factor(
            cov_map, fsky=np.mean(cov_map), zbins=2
        )
        np.testing.assert_allclose(
            factor, np.mean(cov_map) / np.mean(cov_map**2), rtol=1e-14
        )

    def test_independent_of_weight_normalisation(self):
        rng = np.random.default_rng(1)
        npix = hp.nside2npix(16)
        weights = np.zeros((2, npix))
        weights[:, :500] = rng.uniform(0.5, 2.0, (2, 500))
        fsky = 500 / npix
        factor = cov_partial_sky.coupled_noise_factor(weights, fsky, zbins=2)
        factor_scaled = cov_partial_sky.coupled_noise_factor(
            weights * [[7.0], [0.2]], fsky, zbins=2
        )
        np.testing.assert_allclose(factor, factor_scaled, rtol=1e-14)


class TestPixelWindows:
    """Tests for pixel_windows and _safe_inverse."""

    def test_matches_healpy_pixwin(self):
        nside, lmax = 64, 150
        pw_t, pw_e = cov_partial_sky.pixel_windows(nside, lmax)
        ref_t, ref_e = hp.pixwin(nside, pol=True, lmax=lmax)
        assert pw_t.shape == pw_e.shape == (lmax + 1,)
        np.testing.assert_array_equal(pw_t, ref_t)
        np.testing.assert_array_equal(pw_e, ref_e)
        np.testing.assert_array_equal(pw_e[:2], 0.0)

    def test_beyond_tabulated_range_raises(self):
        # healpy silently truncates the window at 4 * nside
        with pytest.raises(ValueError):
            cov_partial_sky.pixel_windows(16, 65)

    def test_safe_inverse(self):
        _, pw_e = cov_partial_sky.pixel_windows(32, 60)
        inv = cov_partial_sky._safe_inverse(pw_e**2)
        np.testing.assert_array_equal(inv[:2], 0.0)
        np.testing.assert_allclose(inv[2:] * pw_e[2:] ** 2, 1.0, rtol=1e-14)
        # the deconvolved noise grows with ell
        assert np.all(np.diff(inv[2:]) > 0)
