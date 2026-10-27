"""Unit tests for the automatic ell_max buffer of the NaMaster covariance."""

import types

import healpy as hp
import numpy as np
import pymaster as nmt

from spaceborne import cov_partial_sky


def _holey_mask(nside, seed=0):
    """Fractional cap with pixel-scale holes: a broad coupling kernel."""
    rng = np.random.default_rng(seed)
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    mask = (theta < 1.0) * rng.uniform(0.3, 1.0, theta.size)
    mask[rng.random(theta.size) < 0.05] = 0.0
    return mask


class TestMcmRowsSpin0:
    """Tests for mcm_rows_spin0."""

    def test_matches_namaster(self):
        nside = 32
        lmax = 3 * nside - 1
        mask = _holey_mask(nside)
        field = nmt.NmtField(mask, None, spin=0, lmax=lmax, lite=True)
        edges = np.append(np.arange(2, lmax, 8), lmax + 1)
        wsp = nmt.NmtWorkspace.from_fields(
            field, field, nmt.NmtBin.from_edges(edges[:-1], edges[1:])
        )
        mcm = wsp.get_coupling_matrix()
        # NaMaster's mask power spectrum uses 3 Jacobi iterations
        mask_cl = hp.anafast(mask, lmax=lmax, iter=3)
        ells = [5, 40, 90]
        rows = cov_partial_sky.mcm_rows_spin0(ells, mask_cl, lmax)
        np.testing.assert_allclose(rows, mcm[ells], rtol=0, atol=1e-12 * mcm.max())

    def test_full_sky_is_identity(self):
        lmax = 47
        mask_cl = np.zeros(lmax + 1)
        mask_cl[0] = 4 * np.pi  # power spectrum of w = 1
        rows = cov_partial_sky.mcm_rows_spin0([3, 20], mask_cl, lmax)
        expected = np.zeros_like(rows)
        expected[0, 3] = expected[1, 20] = 1.0
        np.testing.assert_allclose(rows, expected, atol=1e-10)


class TestCoupledLeakageAbove:
    """Tests for coupled_leakage_above."""

    def test_monotonic_and_bounded(self):
        rows = np.abs(np.random.default_rng(1).normal(size=(3, 50)))
        cl = 1 / (1 + np.arange(50.0))
        leak = cov_partial_sky.coupled_leakage_above(rows, cl)
        assert np.all(np.diff(leak) <= 0)
        assert abs(leak[-1]) < 1e-14
        assert 0 < leak[0] < 1


def _stub_cov_nmt(mask, ell_max, nside, cfg_precision):
    """A CovNaMaster with only the attributes auto_ell_max_buffer uses."""
    obj = object.__new__(cov_partial_sky.CovNaMaster)
    edges = np.linspace(10, ell_max + 1, 6).astype(int)
    obj.ell_obj = types.SimpleNamespace(
        ell_max_3x2pt=ell_max,
        nmt_bin_obj_GC=nmt.NmtBin.from_edges(edges[:-1], edges[1:]),
    )
    obj.cfg = {
        'precision': cfg_precision,
        'probe_selection': {'LL': True, 'GL': True, 'GG': True},
    }
    obj.weight_maps_gg = obj.weight_maps_ll = mask
    obj.zbins, obj.nside = 2, nside
    return obj


class TestAutoEllMaxBuffer:
    """Tests for CovNaMaster.auto_ell_max_buffer."""

    nside, ell_max = 32, 50

    def _cls(self, cap):
        ells = np.arange(cap + 1)
        cl = np.zeros((2, 2, cap + 1, 2, 2))
        cl[:, :, 2:] = (1 / ells[2:] ** 1.5)[None, None, :, None, None]
        return cl

    def test_full_sky_needs_no_buffer(self):
        cap = 3 * self.nside - 1
        obj = _stub_cov_nmt(np.ones(hp.nside2npix(self.nside)), self.ell_max,
                            self.nside, {})
        buffer, report = obj.auto_ell_max_buffer(self._cls(cap), cap, 1e-3)
        assert buffer == 0
        # only the healpy quadrature error of the constant mask's spectrum
        assert all(leak < 1e-4 for *_, leak, _ in report)

    def test_holey_mask_tolerance(self):
        cap = 3 * self.nside - 1
        obj = _stub_cov_nmt(_holey_mask(self.nside), self.ell_max, self.nside, {})
        b_loose, report = obj.auto_ell_max_buffer(self._cls(cap), cap, 5e-2)
        b_tight, _ = obj.auto_ell_max_buffer(self._cls(cap), cap, 1e-3)
        assert 0 < b_loose <= b_tight <= cap - self.ell_max
        # one entry per probe and bin, with the leakage at no buffer above the tol
        assert len(report) == 4
        assert all(leak > 1e-3 for *_, leak, _ in report)

    def test_pixel_window_reduces_the_buffer(self):
        cap = 3 * self.nside - 1
        mask = _holey_mask(self.nside)
        obj = _stub_cov_nmt(mask, self.ell_max, self.nside, {})
        obj_pw = _stub_cov_nmt(mask, self.ell_max, self.nside, {'pixwin_nside': 32})
        b, _ = obj.auto_ell_max_buffer(self._cls(cap), cap, 1e-2)
        b_pw, _ = obj_pw.auto_ell_max_buffer(self._cls(cap), cap, 1e-2)
        assert b_pw <= b
