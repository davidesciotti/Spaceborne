"""Unit tests for spaceborne.cl_utils module.

Covers the reshaping helpers that turn 1D SPV3-format Cl datavectors into the
3D/5D arrays used throughout the pipeline, and the helper that assembles the
(2, 2, nbl, zbins, zbins) 3x2pt datavector from its LL/GL/GG pieces.
"""

import numpy as np
import pytest

from spaceborne import cl_utils
from spaceborne import sb_lib as sl


def _zpairs(zbins):
    zpairs_auto = zbins * (zbins + 1) // 2
    zpairs_cross = zbins**2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


def _fill_symmetric_from_1d(cl_1d, nbl, zbins, zpairs_auto):
    """Reference (explicit-loop) reimplementation of the symmetric
    1D -> 3D unpacking used for WL/GC/LL/GG blocks: fill the upper triangle
    row-major (iz, jz) with jz >= iz, then mirror to the lower triangle."""
    cl_3d = np.zeros((nbl, zbins, zbins))
    p = 0
    for ell in range(nbl):
        for iz in range(zbins):
            for jz in range(iz, zbins):
                cl_3d[ell, iz, jz] = cl_1d[p]
                p += 1
    assert p == nbl * zpairs_auto
    for ell in range(nbl):
        for iz in range(zbins):
            for jz in range(zbins):
                cl_3d[ell, jz, iz] = cl_3d[ell, iz, jz]
    return cl_3d


def _fill_asymmetric_from_1d(cl_1d, nbl, zbins, zpairs_cross):
    """Reference (explicit-loop) reimplementation of the asymmetric
    1D -> 3D unpacking used for the XC/GL block: fill all (iz, jz) row-major,
    no symmetrization."""
    cl_3d = np.zeros((nbl, zbins, zbins))
    p = 0
    for ell in range(nbl):
        for iz in range(zbins):
            for jz in range(zbins):
                cl_3d[ell, iz, jz] = cl_1d[p]
                p += 1
    assert p == nbl * zpairs_cross
    return cl_3d


# ----------------------------------------------------------------------------- #
# build_3x2pt_datavector_5D
# ----------------------------------------------------------------------------- #
class TestBuild3x2ptDatavector5D:
    """Tests for assembling the (n_probes, n_probes, nbl, zbins, zbins) 3x2pt
    datavector from its LL, GL and GG pieces."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(seed=42)

    @pytest.fixture
    def pieces(self, rng):
        nbl, zbins = 4, 3
        dv_ll = rng.normal(size=(nbl, zbins, zbins))
        dv_gl = rng.normal(size=(nbl, zbins, zbins))
        dv_gg = rng.normal(size=(nbl, zbins, zbins))
        return dv_ll, dv_gl, dv_gg, nbl, zbins

    def test_shape(self, pieces):
        dv_ll, dv_gl, dv_gg, nbl, zbins = pieces
        dv_5d = cl_utils.build_3x2pt_datavector_5D(dv_ll, dv_gl, dv_gg, nbl, zbins)
        assert dv_5d.shape == (2, 2, nbl, zbins, zbins)

    def test_ll_and_gg_blocks_are_copied_as_is(self, pieces):
        dv_ll, dv_gl, dv_gg, nbl, zbins = pieces
        dv_5d = cl_utils.build_3x2pt_datavector_5D(dv_ll, dv_gl, dv_gg, nbl, zbins)
        np.testing.assert_array_equal(dv_5d[0, 0], dv_ll)
        np.testing.assert_array_equal(dv_5d[1, 1], dv_gg)

    def test_gl_block_is_copied_as_is(self, pieces):
        dv_ll, dv_gl, dv_gg, nbl, zbins = pieces
        dv_5d = cl_utils.build_3x2pt_datavector_5D(dv_ll, dv_gl, dv_gg, nbl, zbins)
        np.testing.assert_array_equal(dv_5d[1, 0], dv_gl)

    def test_lg_block_is_transpose_of_gl_explicit_loop(self, pieces):
        """dv_3x2pt_5D[0, 1] must equal dv_GL with the two z-axes swapped, per
        ell bin -- verified element by element, not via another transpose."""
        dv_ll, dv_gl, dv_gg, nbl, zbins = pieces
        dv_5d = cl_utils.build_3x2pt_datavector_5D(dv_ll, dv_gl, dv_gg, nbl, zbins)
        for ell in range(nbl):
            for i in range(zbins):
                for j in range(zbins):
                    assert dv_5d[0, 1, ell, i, j] == dv_gl[ell, j, i]


# ----------------------------------------------------------------------------- #
# cl_SPV3_1D_to_3D
# ----------------------------------------------------------------------------- #
class TestClSPV31DTo3D:
    """Tests for unpacking SPV3-format 1D Cl datavectors into 3D (or 5D, for
    the 3x2pt case) arrays."""

    @pytest.fixture
    def small_case(self):
        return {'nbl': 3, 'zbins': 3}

    @pytest.mark.parametrize('probe', ['WL', 'GC'])
    def test_auto_probe_shape(self, small_case, probe):
        nbl, zbins = small_case['nbl'], small_case['zbins']
        zpairs_auto, _, _ = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_auto, dtype=float)

        cl_3d = cl_utils.cl_SPV3_1D_to_3D(cl_1d, probe, nbl, zbins)

        assert cl_3d.shape == (nbl, zbins, zbins)

    @pytest.mark.parametrize('probe', ['WL', 'GC'])
    def test_auto_probe_is_symmetric(self, small_case, probe):
        nbl, zbins = small_case['nbl'], small_case['zbins']
        zpairs_auto, _, _ = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_auto, dtype=float)

        cl_3d = cl_utils.cl_SPV3_1D_to_3D(cl_1d, probe, nbl, zbins)

        for ell in range(nbl):
            np.testing.assert_array_equal(cl_3d[ell], cl_3d[ell].T)

    @pytest.mark.parametrize('probe', ['WL', 'GC'])
    def test_auto_probe_matches_explicit_loop(self, small_case, probe):
        nbl, zbins = small_case['nbl'], small_case['zbins']
        zpairs_auto, _, _ = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_auto, dtype=float)

        cl_3d = cl_utils.cl_SPV3_1D_to_3D(cl_1d, probe, nbl, zbins)
        expected = _fill_symmetric_from_1d(cl_1d, nbl, zbins, zpairs_auto)

        np.testing.assert_array_equal(cl_3d, expected)

    def test_xc_probe_is_not_symmetrized(self, small_case):
        """XC (cross) block must not be mirrored: only the raw row-major fill."""
        nbl, zbins = small_case['nbl'], small_case['zbins']
        _, zpairs_cross, _ = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_cross, dtype=float)

        cl_3d = cl_utils.cl_SPV3_1D_to_3D(cl_1d, 'XC', nbl, zbins)
        expected = _fill_asymmetric_from_1d(cl_1d, nbl, zbins, zpairs_cross)

        assert cl_3d.shape == (nbl, zbins, zbins)
        np.testing.assert_array_equal(cl_3d, expected)

    def test_invalid_probe_raises(self, small_case):
        nbl, zbins = small_case['nbl'], small_case['zbins']
        with pytest.raises(ValueError, match='probe must be'):
            cl_utils.cl_SPV3_1D_to_3D(np.zeros(10), 'XX', nbl, zbins)

    def test_length_mismatch_raises(self, small_case):
        """A 1D datavector whose length is incompatible with nbl/zbins for the
        requested probe must raise (rather than silently misreshape)."""
        nbl, zbins = small_case['nbl'], small_case['zbins']
        zpairs_auto, _, _ = _zpairs(zbins)
        bad_cl_1d = np.arange(nbl * zpairs_auto - 1, dtype=float)
        with pytest.raises(AssertionError):
            cl_utils.cl_SPV3_1D_to_3D(bad_cl_1d, 'WL', nbl, zbins)

    def test_3x2pt_shape(self, small_case):
        nbl, zbins = small_case['nbl'], small_case['zbins']
        _, _, zpairs_3x2pt = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_3x2pt, dtype=float)

        cl_3x2pt = cl_utils.cl_SPV3_1D_to_3D(cl_1d, '3x2pt', nbl, zbins)

        assert cl_3x2pt.shape == (2, 2, nbl, zbins, zbins)

    def test_3x2pt_matches_explicit_loop(self, small_case):
        """Build the LL/GL/GG blocks with an independent explicit-loop
        reimplementation, and check the full 5D output against them,
        including the LG = GL^T relation."""
        nbl, zbins = small_case['nbl'], small_case['zbins']
        zpairs_auto, zpairs_cross, zpairs_3x2pt = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_3x2pt, dtype=float)

        cl_3x2pt = cl_utils.cl_SPV3_1D_to_3D(cl_1d, '3x2pt', nbl, zbins)

        # split cl_1d exactly as cl_SPV3_1D_to_3D does internally: reshape to
        # (nbl, zpairs_3x2pt) row-major, then slice into LL | GL | GG columns
        cl_2d = cl_1d.reshape(nbl, zpairs_3x2pt)
        cl_ll_2d = cl_2d[:, :zpairs_auto]
        cl_gl_2d = cl_2d[:, zpairs_auto : zpairs_auto + zpairs_cross]
        cl_gg_2d = cl_2d[:, zpairs_auto + zpairs_cross :]

        expected_ll = _fill_symmetric_from_1d(
            cl_ll_2d.flatten(), nbl, zbins, zpairs_auto
        )
        expected_gg = _fill_symmetric_from_1d(
            cl_gg_2d.flatten(), nbl, zbins, zpairs_auto
        )
        expected_gl = _fill_asymmetric_from_1d(
            cl_gl_2d.flatten(), nbl, zbins, zpairs_cross
        )

        np.testing.assert_array_equal(cl_3x2pt[0, 0], expected_ll)
        np.testing.assert_array_equal(cl_3x2pt[1, 1], expected_gg)
        np.testing.assert_array_equal(cl_3x2pt[1, 0], expected_gl)
        for ell in range(nbl):
            np.testing.assert_array_equal(cl_3x2pt[0, 1, ell], expected_gl[ell].T)

    def test_3x2pt_ll_and_gg_blocks_are_symmetric(self, small_case):
        nbl, zbins = small_case['nbl'], small_case['zbins']
        _, _, zpairs_3x2pt = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_3x2pt, dtype=float)

        cl_3x2pt = cl_utils.cl_SPV3_1D_to_3D(cl_1d, '3x2pt', nbl, zbins)

        for ell in range(nbl):
            np.testing.assert_array_equal(cl_3x2pt[0, 0, ell], cl_3x2pt[0, 0, ell].T)
            np.testing.assert_array_equal(cl_3x2pt[1, 1, ell], cl_3x2pt[1, 1, ell].T)


# ----------------------------------------------------------------------------- #
# cross-check against sb_lib's own building blocks
# ----------------------------------------------------------------------------- #
class TestClSPV3ConsistentWithSbLib:
    """cl_SPV3_1D_to_3D is a thin wrapper around a handful of sb_lib
    reshaping primitives; sanity-check it stays consistent with them
    directly (guards against the wrapper and the primitives drifting apart).
    """

    def test_wl_matches_sb_lib_primitives_directly(self):
        nbl, zbins = 4, 3
        zpairs_auto, _, _ = _zpairs(zbins)
        cl_1d = np.arange(nbl * zpairs_auto, dtype=float)

        cl_3d = cl_utils.cl_SPV3_1D_to_3D(cl_1d, 'WL', nbl, zbins)

        expected = sl.fill_3D_symmetric_array(
            sl.cl_1D_to_3D(cl_1d, nbl, zbins, is_symmetric=True), nbl, zbins
        )
        np.testing.assert_array_equal(cl_3d, expected)
