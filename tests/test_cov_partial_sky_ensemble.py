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


class TestNbarFromData:
    """Tests for the footprint-mean normalisation in mask_maps_and_compute_alms."""

    def test_constant_overdensity_is_absorbed(self):
        # a constant delta is fully absorbed by the measured mean density
        nside = 16
        w = _fractional_weights(nside, 1)[0]
        npix = w.size
        alms_t, _, _, eps = cov_partial_sky.mask_maps_and_compute_alms(
            corr_maps_gg=[np.full(npix, 0.3)],
            corr_maps_ll=[(np.zeros(npix), np.zeros(npix))],
            weight_maps_gg=w,
            weight_maps_ll=w,
            lmax=2 * nside,
            nbar_from_data=True,
        )
        np.testing.assert_allclose(eps, [0.3], rtol=1e-12)
        np.testing.assert_allclose(np.abs(alms_t[0]), 0.0, atol=1e-12)

    def test_off_by_default(self):
        nside = 16
        w = _fractional_weights(nside, 1)[0]
        npix = w.size
        alms_t, _, _, eps = cov_partial_sky.mask_maps_and_compute_alms(
            corr_maps_gg=[np.full(npix, 0.3)],
            corr_maps_ll=[(np.zeros(npix), np.zeros(npix))],
            weight_maps_gg=w,
            weight_maps_ll=w,
            lmax=2 * nside,
        )
        np.testing.assert_array_equal(eps, [0.0])
        assert np.abs(alms_t[0]).max() > 0


class TestPixwinEmulation:
    """The emulated pixel-window deconvolution turns white pixel noise into N / p^2."""

    def _noise_only_pcl(self, nside, pw, seed):
        lmax, lmax_sim = 2 * nside - 1, 3 * nside - 1
        zero = np.zeros((lmax_sim + 1, 1, 1))
        cl_ring = cov_partial_sky.build_cl_tomo_TEB_ring_ord(
            cl_TT=zero, cl_EE=zero, cl_BB=zero, cl_TE=zero, cl_EB=zero, cl_TB=zero,
            zbins=1, spectra_types=['T', 'E', 'B'],
        )
        w = np.ones(hp.nside2npix(nside))
        pw_t, pw_e = pw if pw is not None else (None, None)
        cl_gg, _, cl_ll = cov_partial_sky._compute_one_realization(
            seed=seed, cl_ring_big_list=cl_ring, lmax=lmax, lmax_sim=lmax_sim,
            noise_amp_gg=None, noise_amp_ll=None, nside=nside, zbins=1,
            weight_maps_gg=w, weight_maps_ll=w, nl_gg_diag=np.array([2e-8]),
            nl_ll_diag=np.array([2e-8]), coupled_cls=True, spin0=False,
            wsp_path_template='', nbl=1, ell_min_edges=np.array([10]),
            ell_max_edges=np.array([2 * nside]), pw_t=pw_t, pw_e=pw_e,
        )
        return cl_gg[0, 0, 0], cl_ll[0, 0, 0]

    def test_noise_level(self):
        nside = 64
        lmax = 3 * nside - 1
        pw_t, pw_e = cov_partial_sky.pixel_windows(nside, lmax)
        band = slice(10, 2 * nside)
        expected_t = 2e-8 * np.mean(1 / pw_t[band] ** 2)
        expected_e = 2e-8 * np.mean(1 / pw_e[band] ** 2)
        got = np.mean(
            [self._noise_only_pcl(nside, (pw_t, pw_e), s) for s in range(4)], axis=0
        )
        np.testing.assert_allclose(got, [expected_t, expected_e], rtol=0.03)
        got_white = np.mean([self._noise_only_pcl(nside, None, s) for s in range(4)], 0)
        np.testing.assert_allclose(got_white, [2e-8, 2e-8], rtol=0.03)
