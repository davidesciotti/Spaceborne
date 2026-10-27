"""Unit tests for the catalogue-sum noise model of the ensemble covariance."""

import healpy as hp
import numpy as np
import pytest

from spaceborne import cov_partial_sky


def _fractional_weights(nside, zbins, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros((zbins, hp.nside2npix(nside)))
    w[:, : w.shape[1] // 3] = rng.uniform(0.2, 1.0, (zbins, w.shape[1] // 3))
    return w


class TestCatalogueNoiseAmplitude:
    """Tests for catalogue_noise_amplitude."""

    def test_coupled_level_equals_analytic(self):
        nside, zbins = 32, 2
        w = _fractional_weights(nside, zbins)
        nl = np.array([2e-8, 5e-8])
        amp = cov_partial_sky.catalogue_noise_amplitude(w, nl, nside, zbins)
        omega_pix = 4 * np.pi / hp.nside2npix(nside)
        np.testing.assert_allclose(
            amp**2 * omega_pix * w.mean(1), nl * (w**2).mean(1), rtol=1e-14
        )

    def test_monte_carlo_pseudo_cl(self):
        # noise with per-pixel variance A^2 w: the pseudo-Cl of the map is white with
        # level A^2 Omega_pix <w> = N <w^2>, the coupled level of the analytic model
        nside, zbins, nreal = 64, 1, 20
        w = _fractional_weights(nside, zbins, seed=1)
        nl = np.array([3e-8])
        amp = cov_partial_sky.catalogue_noise_amplitude(w, nl, nside, zbins)
        rng = np.random.default_rng(2)
        lmax = 2 * nside
        cl = np.zeros(lmax + 1)
        for _ in range(nreal):
            m = rng.standard_normal(w.shape[1]) * amp[0] * np.sqrt(w[0])
            cl += hp.anafast(m, lmax=lmax) / nreal
        np.testing.assert_allclose(cl[2:].mean(), nl[0] * (w[0] ** 2).mean(), rtol=0.01)

    def test_single_footprint_for_all_bins(self):
        nside, zbins = 16, 3
        footprint = _fractional_weights(nside, 1)[0]
        nl = np.array([1e-8, 2e-8, 4e-8])
        amp = cov_partial_sky.catalogue_noise_amplitude(footprint, nl, nside, zbins)
        np.testing.assert_allclose(amp**2 / amp[0] ** 2, [1, 2, 4], rtol=1e-14)

    def test_negative_weights_raise(self):
        nside = 16
        w = _fractional_weights(nside, 1)
        w[0, 0] = -0.1
        with pytest.raises(AssertionError):
            cov_partial_sky.catalogue_noise_amplitude(w, np.array([1e-8]), nside, 1)
