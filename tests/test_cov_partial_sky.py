"""Unit tests for the noise helpers in spaceborne.cov_partial_sky."""

import healpy as hp
import numpy as np

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


class TestNoisePixwinDeconvFactor:
    """Tests for noise_pixwin_deconv_factor."""

    def test_matches_healpy_pixwin(self):
        nside, lmax = 64, 150
        inv_pw2_gg, inv_pw2_ll = cov_partial_sky.noise_pixwin_deconv_factor(
            nside, lmax
        )
        pw_t, pw_p = hp.pixwin(nside, pol=True, lmax=lmax)

        assert inv_pw2_gg.shape == inv_pw2_ll.shape == (lmax + 1, 1, 1)
        np.testing.assert_allclose(inv_pw2_gg[:, 0, 0] * pw_t**2, 1.0, rtol=1e-14)
        np.testing.assert_allclose(
            inv_pw2_ll[2:, 0, 0] * pw_p[2:] ** 2, 1.0, rtol=1e-14
        )
        # no spin-2 monopole/dipole
        np.testing.assert_array_equal(inv_pw2_ll[:2], 0.0)
        # the deconvolved noise grows with ell
        assert np.all(np.diff(inv_pw2_gg[:, 0, 0]) > 0)

    def test_broadcasts_on_noise_array(self):
        nside, lmax, zbins = 32, 60, 3
        nl = np.zeros((lmax + 1, zbins, zbins))
        nl[:, range(zbins), range(zbins)] = [1.0, 2.0, 3.0]
        inv_pw2_gg, _ = cov_partial_sky.noise_pixwin_deconv_factor(nside, lmax)
        nl_pw = nl * inv_pw2_gg
        # off-diagonals stay zero, each diagonal is scaled by the same 1 / p_ell^2
        np.testing.assert_array_equal(nl_pw[:, 0, 1], 0.0)
        np.testing.assert_allclose(nl_pw[:, 2, 2], 3.0 * inv_pw2_gg[:, 0, 0])
