"""Unit tests for spaceborne.mask_utils module."""

import os
import tempfile

import healpy as hp
import numpy as np
import pytest

from spaceborne import mask_utils
from spaceborne import constants


class TestGetMaskCl:
    """Tests for get_mask_cl function."""

    def test_basic_functionality(self):
        """Test basic mask power spectrum computation."""
        nside = 32
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)

        ell_mask, cl_mask, fsky_mask = mask_utils.get_mask_cl(mask)

        assert len(ell_mask) == len(cl_mask)
        assert ell_mask[0] == 0
        assert np.all(np.diff(ell_mask) == 1)
        assert fsky_mask == pytest.approx(1.0)

    def test_half_sky_mask(self):
        """Test fsky computation for half-sky mask."""
        nside = 32
        npix = hp.nside2npix(nside)
        mask = np.zeros(npix)
        mask[: npix // 2] = 1.0

        ell_mask, cl_mask, fsky_mask = mask_utils.get_mask_cl(mask)

        # fsky = mean(mask**2) = (npix/2 * 1.0**2) / npix = 0.5 for half-sky binary mask
        assert fsky_mask == pytest.approx(0.5, abs=0.01)

    def test_uniform_partial_mask(self):
        """Test fsky for uniform partial coverage."""
        nside = 32
        npix = hp.nside2npix(nside)
        coverage = 0.7
        mask = np.full(npix, coverage)

        ell_mask, cl_mask, fsky_mask = mask_utils.get_mask_cl(mask)

        # fsky = mean(mask**2) for uniform mask
        assert fsky_mask == pytest.approx(coverage**2, rel=1e-6)

    def test_output_shapes(self):
        """Test that output arrays have expected shapes."""
        nside = 16
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)

        ell_mask, cl_mask, fsky_mask = mask_utils.get_mask_cl(mask)

        # For nside=16, lmax should be 3*nside-1 = 47
        expected_lmax = 3 * nside - 1
        assert len(ell_mask) == expected_lmax + 1
        assert len(cl_mask) == expected_lmax + 1
        assert isinstance(fsky_mask, (float, np.floating))


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


class TestReadMaskingMap:
    """Tests for _read_masking_map function."""

    def test_nside_downgrade(self):
        """Test reading and downgrading a FITS map."""
        # Create a temporary FITS file with a simple mask
        nside_high = 64
        nside_low = 32
        npix = hp.nside2npix(nside_high)
        mask_data = np.ones(npix)

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
                mask = mask_utils._read_masking_map(tmp.name, nside_low, nest=False)

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
                    mask_utils._read_masking_map(tmp.name, nside_out)
            finally:
                os.unlink(tmp.name)


class TestMask:
    """Tests for Mask class."""

    @pytest.fixture
    def basic_mask_config_generate(self):
        """Basic configuration for mask generation."""
        return {
            'load_mask': False,
            'mask_filename': '',
            'nside': 32,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': True,
        }

    @pytest.fixture
    def basic_mask_config_load(self, tmp_path):
        """Basic configuration for mask loading."""
        # Create a temporary mask file
        nside = 32
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)
        mask_filename = tmp_path / 'test_mask.npy'
        np.save(mask_filename, mask)

        return {
            'load_mask': True,
            'mask_filename': str(mask_filename),
            'nside': nside,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': False,
        }

    def test_initialization(self, basic_mask_config_generate):
        """Test Mask class initialization."""
        mask_obj = mask_utils.Mask(basic_mask_config_generate)

        assert mask_obj.load_mask is False
        assert mask_obj.generate_polar_cap is True
        assert mask_obj.nside == 32
        assert mask_obj.desired_survey_area_deg2 == 1000.0
        assert mask_obj.apodize is False
        assert mask_obj.aposize == 1.0

    def test_generate_polar_cap(self, basic_mask_config_generate):
        """Test polar cap mask generation through Mask class."""
        mask_obj = mask_utils.Mask(basic_mask_config_generate)
        mask_obj.process()

        assert hasattr(mask_obj, 'mask')
        assert len(mask_obj.mask) == hp.nside2npix(32)
        assert hasattr(mask_obj, 'fsky')
        assert hasattr(mask_obj, 'survey_area_deg2')
        assert hasattr(mask_obj, 'survey_area_sr')
        assert mask_obj.fsky > 0
        assert mask_obj.fsky < 1

    def test_load_mask_npy(self, basic_mask_config_load):
        """Test loading mask from .npy file."""
        mask_obj = mask_utils.Mask(basic_mask_config_load)
        mask_obj.process()

        assert hasattr(mask_obj, 'mask')
        assert len(mask_obj.mask) == hp.nside2npix(32)
        # Full sky mask should give fsky ~ 1
        assert mask_obj.fsky == pytest.approx(1.0, rel=0.01)

    def test_load_mask_fits(self, tmp_path):
        """Test loading mask from .fits file."""
        nside = 32
        npix = hp.nside2npix(nside)
        mask = np.ones(npix)
        mask_filename = tmp_path / 'test_mask.fits'
        hp.write_map(str(mask_filename), mask, overwrite=True, dtype=np.float64)

        config = {
            'load_mask': True,
            'mask_filename': str(mask_filename),
            'nside': nside,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': False,
        }

        mask_obj = mask_utils.Mask(config)
        mask_obj.process()

        assert hasattr(mask_obj, 'mask')
        assert len(mask_obj.mask) == npix

    def test_mask_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        config = {
            'load_mask': True,
            'mask_filename': '/nonexistent/path/mask.npy',
            'nside': 32,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': False,
        }

        mask_obj = mask_utils.Mask(config)

        with pytest.raises(FileNotFoundError):
            mask_obj.process()

    def test_unsupported_file_format(self, tmp_path):
        """Test that unsupported file format raises ValueError."""
        mask_filename = tmp_path / 'test_mask.txt'
        mask_filename.write_text('dummy data')

        config = {
            'load_mask': True,
            'mask_filename': str(mask_filename),
            'nside': 32,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': False,
        }

        mask_obj = mask_utils.Mask(config)

        with pytest.raises(ValueError, match='Unsupported file format'):
            mask_obj.process()

    def test_both_load_and_generate_fails(self):
        """Test that both load_mask and generate_polar_cap True raises error."""
        config = {
            'load_mask': True,
            'mask_filename': 'some_path.npy',
            'nside': 32,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': True,
        }

        mask_obj = mask_utils.Mask(config)

        with pytest.raises(AssertionError, match='choose whether to load OR generate'):
            mask_obj.process()

    def test_neither_load_nor_generate_fails(self):
        """Test that neither load_mask nor generate_polar_cap True raises error."""
        config = {
            'load_mask': False,
            'mask_filename': '',
            'nside': 32,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': False,
        }

        mask_obj = mask_utils.Mask(config)

        with pytest.raises(AssertionError, match='choose whether to load OR generate'):
            mask_obj.process()

    def test_mask_upgrade_downgrade(self, tmp_path):
        """Test changing mask resolution."""
        nside_original = 64
        nside_target = 32
        npix = hp.nside2npix(nside_original)
        mask = np.ones(npix)
        mask_filename = tmp_path / 'test_mask.npy'
        np.save(mask_filename, mask)

        config = {
            'load_mask': True,
            'mask_filename': str(mask_filename),
            'nside': nside_target,
            'survey_area_deg2': 1000.0,
            'apodize': False,
            'aposize': 1.0,
            'generate_polar_cap': False,
        }

        mask_obj = mask_utils.Mask(config)
        mask_obj.process()

        # Mask should be downgraded to target nside
        assert len(mask_obj.mask) == hp.nside2npix(nside_target)

    def test_apodization(self, basic_mask_config_generate):
        """Test mask apodization."""
        # Enable apodization
        basic_mask_config_generate['apodize'] = True
        basic_mask_config_generate['aposize'] = 2.0

        mask_obj = mask_utils.Mask(basic_mask_config_generate)
        mask_obj.process()

        # After apodization, mask should have non-binary values
        unique_values = np.unique(mask_obj.mask)
        assert len(unique_values) > 2  # More than just 0 and 1

    def test_mask_spectrum_attributes(self, basic_mask_config_generate):
        """Test that mask spectrum and normalization are computed."""
        mask_obj = mask_utils.Mask(basic_mask_config_generate)
        mask_obj.process()

        assert hasattr(mask_obj, 'ell_mask')
        assert hasattr(mask_obj, 'cl_mask')
        assert hasattr(mask_obj, 'cl_mask_norm')
        assert len(mask_obj.ell_mask) == len(mask_obj.cl_mask)
        assert len(mask_obj.cl_mask) == len(mask_obj.cl_mask_norm)

    def test_survey_area_computation(self, basic_mask_config_generate):
        """Test that survey area is correctly computed from fsky."""
        mask_obj = mask_utils.Mask(basic_mask_config_generate)
        mask_obj.process()

        # Check consistency between fsky and survey areas
        expected_area_deg2 = mask_obj.fsky * constants.DEG2_IN_SPHERE
        assert mask_obj.survey_area_deg2 == pytest.approx(expected_area_deg2, rel=1e-10)

        expected_area_sr = mask_obj.survey_area_deg2 * constants.DEG2_TO_SR
        assert mask_obj.survey_area_sr == pytest.approx(expected_area_sr, rel=1e-10)

    @pytest.mark.skip(reason='pymaster apodization causes segfault on some systems')
    def test_mask_dtype_for_apodization(self, basic_mask_config_generate):
        """Test that mask is converted to float64 for apodization."""
        pytest.importorskip('pymaster')

        basic_mask_config_generate['apodize'] = True
        basic_mask_config_generate['aposize'] = 1.0  # Smaller aposize

        mask_obj = mask_utils.Mask(basic_mask_config_generate)
        mask_obj.process()

        # Mask should be float64 after apodization
        assert mask_obj.mask.dtype == np.float64
