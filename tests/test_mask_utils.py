"""Unit tests for spaceborne.mask_utils module."""

import os
import tempfile
import types

import healpy as hp
import numpy as np
import pytest

from spaceborne import constants, io_handler, mask_utils


def _make_mask_cfg(
    probe='LL',
    *,
    geometry='polar_cap',
    footprint_filename='',
    weight_maps_filename=None,
    nside=32,
    survey_area_deg2=1000.0,
):
    """Build a ``mask`` config dict in the new per-probe layout expected by Mask."""
    return {
        probe: {
            'geometry': geometry,
            'footprint_filename': footprint_filename,
            'weight_maps_filename': weight_maps_filename,
        },
        'nside': nside,
        'survey_area_deg2': survey_area_deg2,
    }


class TestGetMapsCl:
    """Tests for get_maps_cl function (cross/auto spectrum of two maps)."""

    def test_basic_functionality(self):
        """Test basic map power spectrum computation."""
        nside = 32
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)

        ells, cl = mask_utils.get_maps_cl(mask, mask)

        assert len(ells) == len(cl)
        assert ells[0] == 0
        assert np.all(np.diff(ells) == 1)

    def test_output_shapes(self):
        """Test that output arrays have expected shapes."""
        nside = 16
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)

        ells, cl = mask_utils.get_maps_cl(mask, mask)

        # For nside=16, lmax should be 3*nside-1 = 47
        expected_lmax = 3 * nside - 1
        assert len(ells) == expected_lmax + 1
        assert len(cl) == expected_lmax + 1

    def test_cross_spectrum_runs(self):
        """Test that the cross spectrum of two different maps is finite."""
        nside = 16
        npix = hp.nside2npix(nside)
        map1 = np.ones(npix)
        map2 = np.zeros(npix)
        map2[: npix // 2] = 1.0

        ells, cl = mask_utils.get_maps_cl(map1, map2)

        assert len(ells) == len(cl)
        assert np.all(np.isfinite(cl))


class TestCombinedFsky:
    """Tests for combined_fsky (mean of the product of two masks)."""

    def test_full_sky(self):
        nside = 32
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)
        assert mask_utils.combined_fsky(mask, mask) == pytest.approx(1.0)

    def test_half_sky_binary(self):
        nside = 32
        npix = hp.nside2npix(nside)
        mask = np.zeros(npix)
        mask[: npix // 2] = 1.0
        # mean(mask * mask) = 0.5 for a half-sky binary mask
        assert mask_utils.combined_fsky(mask, mask) == pytest.approx(0.5, abs=0.01)

    def test_uniform_partial_mask(self):
        nside = 32
        npix = hp.nside2npix(nside)
        coverage = 0.7
        mask = np.full(npix, coverage)
        # mean(mask * mask) = coverage**2 for a uniform mask
        assert mask_utils.combined_fsky(mask, mask) == pytest.approx(
            coverage**2, rel=1e-6
        )

    def test_returns_python_float(self):
        nside = 16
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)
        assert isinstance(mask_utils.combined_fsky(mask, mask), float)


class TestFootprintFskyAb:
    """Tests for footprint_fsky_ab (probe-pair footprints and effective fskys)."""

    def test_probe_pair_products_and_fskys(self):
        nside = 16
        npix = hp.nside2npix(nside)
        m_ll = np.full(npix, 0.5)
        m_gg = np.full(npix, 0.8)

        mask_obj_ll = types.SimpleNamespace(footprint=m_ll)
        mask_obj_gg = types.SimpleNamespace(footprint=m_gg)

        footp_ab_dict, fsky_ab_dict = mask_utils.footprint_fsky_ab(
            mask_obj_ll, mask_obj_gg
        )

        # AB footprints are the product of the two single-probe masks
        np.testing.assert_allclose(footp_ab_dict['LL'], m_ll * m_ll)
        np.testing.assert_allclose(footp_ab_dict['GL'], m_ll * m_gg)
        np.testing.assert_allclose(footp_ab_dict['GG'], m_gg * m_gg)

        # fsky is the mean of the product of the *single* masks
        assert fsky_ab_dict['LL'] == pytest.approx(0.5 * 0.5)
        assert fsky_ab_dict['GL'] == pytest.approx(0.5 * 0.8)
        assert fsky_ab_dict['GG'] == pytest.approx(0.8 * 0.8)


class TestGeneratePolarCapFunc:
    """Tests for generate_polar_cap_func function."""

    def test_basic_polar_cap_generation(self):
        """Test polar cap mask generation."""
        area_deg2 = 1000.0
        nside = 64

        mask = mask_utils.generate_polar_cap_func(area_deg2, nside)

        npix = hp.nside2npix(nside)
        assert len(mask) == npix
        assert np.all((mask == 0) | (mask == 1))

        # Check that measured area is close to expected
        measured_area = np.sum(mask) * hp.nside2pixarea(nside, degrees=True)
        assert measured_area == pytest.approx(area_deg2, rel=0.05)

    def test_small_area(self):
        """Test polar cap with small area."""
        area_deg2 = 100.0
        nside = 128

        mask = mask_utils.generate_polar_cap_func(area_deg2, nside)

        measured_area = np.sum(mask) * hp.nside2pixarea(nside, degrees=True)
        assert measured_area == pytest.approx(area_deg2, rel=0.1)

    def test_large_area(self):
        """Test polar cap with large area (most of sky)."""
        area_deg2 = 30000.0
        nside = 32

        mask = mask_utils.generate_polar_cap_func(area_deg2, nside)

        measured_area = np.sum(mask) * hp.nside2pixarea(nside, degrees=True)
        assert measured_area == pytest.approx(area_deg2, rel=0.05)
        # Should have most pixels = 1
        assert np.sum(mask) > 0.5 * len(mask)

    def test_full_sky_limit(self):
        """Test that nearly full-sky area works."""
        area_deg2 = constants.DEG2_IN_SPHERE * 0.99
        nside = 16

        mask = mask_utils.generate_polar_cap_func(area_deg2, nside)

        # Should have nearly all pixels = 1
        assert np.sum(mask) > 0.95 * len(mask)


class TestUpDowngradeMap:
    """Tests for up_downgrade_map helper."""

    def test_downgrade(self):
        nside_in, nside_out = 64, 32
        m = np.ones(hp.nside2npix(nside_in))
        out = mask_utils.up_downgrade_map(m, nside_out)
        assert len(out) == hp.nside2npix(nside_out)

    def test_no_change_when_equal(self):
        nside = 32
        m = np.ones(hp.nside2npix(nside))
        out = mask_utils.up_downgrade_map(m, nside)
        assert len(out) == hp.nside2npix(nside)

    def test_no_change_when_none(self):
        nside = 32
        m = np.ones(hp.nside2npix(nside))
        out = mask_utils.up_downgrade_map(m, None)
        assert len(out) == hp.nside2npix(nside)


class TestReadMaskingMap:
    """Tests for io_handler._read_masking_map (moved out of mask_utils)."""

    def test_nside_downgrade(self):
        """Test reading and downgrading a FITS map."""
        # Create a temporary FITS file with a simple mask
        nside_high = 64
        nside_low = 32
        npix = hp.nside2npix(nside_high)

        # Only test this if fitsio is available
        try:
            import fitsio
        except ImportError:
            pytest.skip('fitsio not available')

        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            # Create a simple partial sky map
            pixels = np.arange(npix // 2)  # First half of pixels
            weights = np.ones(npix // 2)

            data = np.zeros(len(pixels), dtype=[('PIXEL', 'i8'), ('WEIGHT', 'f8')])
            data['PIXEL'] = pixels
            data['WEIGHT'] = weights

            header = {'NSIDE': nside_high, 'ORDERING': 'NESTED'}
            fitsio.write(tmp.name, data, header=header, clobber=True)

            try:
                mask = io_handler._read_masking_map(tmp.name, nside_low, nest=False)

                assert len(mask) == hp.nside2npix(nside_low)
                # Should have some non-zero values
                assert np.sum(mask > 0) > 0
            finally:
                os.unlink(tmp.name)

    def test_invalid_nside(self):
        """Test that requesting too high nside raises error."""
        nside_in = 32
        nside_out = 64
        npix = hp.nside2npix(nside_in)

        try:
            import fitsio
        except ImportError:
            pytest.skip('fitsio not available')

        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            pixels = np.arange(npix)
            weights = np.ones(npix)

            data = np.zeros(len(pixels), dtype=[('PIXEL', 'i8'), ('WEIGHT', 'f8')])
            data['PIXEL'] = pixels
            data['WEIGHT'] = weights

            header = {'NSIDE': nside_in, 'ORDERING': 'NESTED'}
            fitsio.write(tmp.name, data, header=header, clobber=True)

            try:
                with pytest.raises(ValueError, match='greater than map NSIDE'):
                    io_handler._read_masking_map(tmp.name, nside_out)
            finally:
                os.unlink(tmp.name)


class TestMask:
    """Tests for Mask class."""

    @pytest.fixture
    def polar_cap_cfg(self):
        """Config for on-the-fly polar cap generation."""
        return _make_mask_cfg(probe='LL', geometry='polar_cap', nside=32)

    @pytest.fixture
    def footprint_npy_cfg(self, tmp_path):
        """Config for loading a footprint from a .npy file (full sky)."""
        nside = 32
        npix = hp.nside2npix(nside)
        footprint = np.ones(npix)
        footprint_filename = tmp_path / 'test_footprint.npy'
        np.save(footprint_filename, footprint)

        return _make_mask_cfg(
            probe='LL',
            geometry='footprint_file',
            footprint_filename=str(footprint_filename),
            nside=nside,
        )

    def test_initialization(self, polar_cap_cfg):
        """Test Mask class initialization."""
        mask_obj = mask_utils.Mask(polar_cap_cfg, probe='LL')

        assert mask_obj.probe == 'LL'
        assert mask_obj.geometry == 'polar_cap'
        assert mask_obj.use_weight_maps is False
        assert mask_obj.nside_cfg == 32
        assert mask_obj.desired_survey_area_deg2 == 1000.0

    def test_generate_polar_cap(self, polar_cap_cfg):
        """Test polar cap mask generation through Mask class."""
        mask_obj = mask_utils.Mask(polar_cap_cfg, probe='LL')
        mask_obj.process()

        assert hasattr(mask_obj, 'footprint')
        assert len(mask_obj.footprint) == hp.nside2npix(32)
        assert hasattr(mask_obj, 'fsky_footprint')
        assert hasattr(mask_obj, 'survey_area_deg2')
        assert hasattr(mask_obj, 'survey_area_sr')
        assert mask_obj.fsky_footprint > 0
        assert mask_obj.fsky_footprint < 1

    def test_footprint_npy(self, footprint_npy_cfg):
        """Test loading a footprint from a .npy file."""
        mask_obj = mask_utils.Mask(footprint_npy_cfg, probe='LL')
        mask_obj.process()
        assert hasattr(mask_obj, 'footprint')
        assert len(mask_obj.footprint) == hp.nside2npix(32)
        # Full sky footprint should give fsky ~ 1
        assert mask_obj.fsky_footprint == pytest.approx(1.0, rel=0.01)

    def test_footprint_fits(self, tmp_path):
        """Test loading a footprint from a .fits file."""
        nside = 32
        npix = hp.nside2npix(nside)
        footprint = np.ones(npix)
        footprint_filename = tmp_path / 'test_footprint.fits'
        hp.write_map(
            str(footprint_filename), footprint, overwrite=True, dtype=np.float64
        )

        cfg = _make_mask_cfg(
            probe='GG',
            geometry='footprint_file',
            footprint_filename=str(footprint_filename),
            nside=nside,
        )

        mask_obj = mask_utils.Mask(cfg, probe='GG')
        mask_obj.process()

        assert hasattr(mask_obj, 'footprint')
        assert len(mask_obj.footprint) == npix

    def test_footprint_file_not_found(self):
        """Test that FileNotFoundError is raised for a missing footprint file."""
        cfg = _make_mask_cfg(
            probe='LL',
            geometry='footprint_file',
            footprint_filename='/nonexistent/path/footprint.npy',
        )

        mask_obj = mask_utils.Mask(cfg, probe='LL')

        with pytest.raises(FileNotFoundError):
            mask_obj.process()

    def test_unsupported_file_format(self, tmp_path):
        """Test that an unsupported footprint file format raises ValueError."""
        footprint_filename = tmp_path / 'test_footprint.txt'
        footprint_filename.write_text('dummy data')

        cfg = _make_mask_cfg(
            probe='LL',
            geometry='footprint_file',
            footprint_filename=str(footprint_filename),
        )

        mask_obj = mask_utils.Mask(cfg, probe='LL')

        with pytest.raises(ValueError, match='Unsupported file format'):
            mask_obj.process()

    def test_invalid_geometry(self):
        """Test that an unsupported geometry raises ValueError."""
        cfg = _make_mask_cfg(probe='LL', geometry='not_a_geometry')

        mask_obj = mask_utils.Mask(cfg, probe='LL')

        with pytest.raises(ValueError, match='Unsupported geometry type'):
            mask_obj.process()

    def test_footprint_upgrade_downgrade(self, tmp_path):
        """Test changing footprint resolution."""
        nside_original = 64
        nside_target = 32
        npix = hp.nside2npix(nside_original)
        footprint = np.ones(npix)
        footprint_filename = tmp_path / 'test_footprint.npy'
        np.save(footprint_filename, footprint)

        cfg = _make_mask_cfg(
            probe='LL',
            geometry='footprint_file',
            footprint_filename=str(footprint_filename),
            nside=nside_target,
        )

        mask_obj = mask_utils.Mask(cfg, probe='LL')
        mask_obj.process()

        # Footprint should be downgraded to the target nside
        assert len(mask_obj.footprint) == hp.nside2npix(nside_target)

    def test_footprint_spectrum_attributes(self, polar_cap_cfg):
        """Test that the footprint spectrum and normalization are computed."""
        mask_obj = mask_utils.Mask(polar_cap_cfg, probe='LL')
        mask_obj.process()

        assert hasattr(mask_obj, 'ells_footprint')
        assert hasattr(mask_obj, 'cl_footprint')
        assert hasattr(mask_obj, 'cl_footprint_norm')
        assert len(mask_obj.ells_footprint) == len(mask_obj.cl_footprint)
        assert len(mask_obj.cl_footprint) == len(mask_obj.cl_footprint_norm)

    def test_survey_area_computation(self, polar_cap_cfg):
        """Test that survey area is correctly computed from fsky."""
        mask_obj = mask_utils.Mask(polar_cap_cfg, probe='LL')
        mask_obj.process()

        # Check consistency between fsky and survey areas
        expected_area_deg2 = mask_obj.fsky_footprint * constants.DEG2_IN_SPHERE
        assert mask_obj.survey_area_deg2 == pytest.approx(expected_area_deg2, rel=1e-10)

        expected_area_sr = mask_obj.survey_area_deg2 * constants.DEG2_TO_SR
        assert mask_obj.survey_area_sr == pytest.approx(expected_area_sr, rel=1e-10)


class TestMaskWeightMaps:
    """Tests for per-bin weight map loading in the Mask class."""

    @staticmethod
    def _write_weight_maps(path, nside, zbins=3):
        """Write a (zbins, npix) non-negative weight-map FITS file."""
        npix = hp.nside2npix(nside)
        weight_maps = np.abs(
            np.random.default_rng(0).normal(1.0, 0.1, size=(zbins, npix))
        )
        hp.write_map(str(path), list(weight_maps), overwrite=True, dtype=np.float64)
        return weight_maps

    def test_weight_maps_loaded(self, tmp_path):
        """Weight maps are loaded as a (zbins, npix) array at the config nside."""
        nside = 32
        zbins = 3
        wmap_filename = tmp_path / 'weight_maps.fits'
        self._write_weight_maps(wmap_filename, nside, zbins=zbins)

        cfg = _make_mask_cfg(
            probe='LL',
            geometry='polar_cap',
            weight_maps_filename=str(wmap_filename),
            nside=nside,
        )

        mask_obj = mask_utils.Mask(cfg, probe='LL')
        mask_obj.process()

        assert mask_obj.use_weight_maps is True
        assert mask_obj.weight_maps.shape == (zbins, hp.nside2npix(nside))

    def test_weight_maps_downgrade(self, tmp_path):
        """Weight maps stored at a higher nside are regraded to the config nside.

        Guards against the in-place row-assignment broadcast error: ud_grade
        changes the pixel count, so the array must be rebuilt.
        """
        nside_stored = 32
        nside_target = 16
        zbins = 3
        wmap_filename = tmp_path / 'weight_maps.fits'
        self._write_weight_maps(wmap_filename, nside_stored, zbins=zbins)

        cfg = _make_mask_cfg(
            probe='LL',
            geometry='polar_cap',
            weight_maps_filename=str(wmap_filename),
            nside=nside_target,
        )

        mask_obj = mask_utils.Mask(cfg, probe='LL')
        mask_obj.process()

        assert mask_obj.weight_maps.shape == (zbins, hp.nside2npix(nside_target))
