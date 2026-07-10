"""Unit tests for spaceborne.io_handler module (pure/file-based helpers only;
the IOHandler class itself needs a full pipeline config and is not covered here).
"""

import numpy as np
import pytest

from spaceborne import io_handler

healpy = pytest.importorskip('healpy')


class TestImportClTab:
    """Tests for import_cl_tab (long-format table -> 3D array)."""

    def test_basic_construction(self):
        """ell, zi, zj, Cl(ell) columns get reshaped into (nbl, zbins, zbins)."""
        ells = [10.0, 20.0]
        zbins = 2
        rows = [
            [ell, zi, zj, ell * 100 + zi * 10 + zj]
            for ell in ells
            for zi in range(zbins)
            for zj in range(zbins)
        ]
        cl_tab = np.array(rows)

        ell_values, cl_3d = io_handler.import_cl_tab(cl_tab)

        np.testing.assert_allclose(ell_values, ells)
        assert cl_3d.shape == (2, zbins, zbins)
        assert cl_3d[0, 1, 0] == pytest.approx(10 * 100 + 1 * 10 + 0)
        assert cl_3d[1, 0, 1] == pytest.approx(20 * 100 + 0 * 10 + 1)

    def test_wrong_number_of_columns_raises(self):
        cl_tab = np.zeros((5, 3))
        with pytest.raises(AssertionError, match='4 columns'):
            io_handler.import_cl_tab(cl_tab)

    def test_zi_not_starting_from_zero_raises(self):
        cl_tab = np.array([[10.0, 1, 0, 1.0], [10.0, 2, 0, 2.0]])
        with pytest.raises(AssertionError, match='start from 0'):
            io_handler.import_cl_tab(cl_tab)

    def test_mismatched_max_zi_zj_raises(self):
        cl_tab = np.array([[10.0, 0, 0, 1.0], [10.0, 1, 0, 2.0]])
        with pytest.raises(AssertionError, match='should be'):
            io_handler.import_cl_tab(cl_tab)


class TestCheckClSymm:
    """Tests for check_cl_symm."""

    def test_symmetric_passes(self):
        cl_3d = np.array([[[1.0, 2.0], [2.0, 3.0]]])
        io_handler.check_cl_symm(cl_3d)  # should not raise

    def test_asymmetric_raises(self):
        cl_3d = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        with pytest.raises(AssertionError):
            io_handler.check_cl_symm(cl_3d)


class TestFirstElementOfLeadingAxes:
    """Tests for first_element_of_leading_axes."""

    def test_2d_array_returned_as_is(self):
        arr = np.arange(6).reshape(2, 3)
        out = io_handler.first_element_of_leading_axes(arr)
        np.testing.assert_array_equal(out, arr)

    def test_4d_array_takes_first_of_leading_axes(self):
        arr = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        out = io_handler.first_element_of_leading_axes(arr)
        np.testing.assert_array_equal(out, arr[0, 0, :, :])
        assert out.shape == (4, 5)

    def test_1d_array_raises(self):
        arr = np.arange(5)
        with pytest.raises(ValueError, match='at least two dimensions'):
            io_handler.first_element_of_leading_axes(arr)


class TestCheckEllsForSpline:
    """Tests for check_ells_for_spline."""

    def test_sorted_unique_passes(self):
        io_handler.check_ells_for_spline(np.array([1.0, 2.0, 3.0]))  # no raise

    def test_unsorted_raises(self):
        with pytest.raises(AssertionError, match='not sorted'):
            io_handler.check_ells_for_spline(np.array([1.0, 3.0, 2.0]))

    def test_duplicates_raise(self):
        with pytest.raises(AssertionError):
            io_handler.check_ells_for_spline(np.array([1.0, 2.0, 2.0, 3.0]))


class TestLoadFootprint:
    """Tests for load_footprint."""

    def test_npy_format(self, tmp_path):
        nside = 8
        npix = healpy.nside2npix(nside)
        footprint = np.ones(npix)
        path = tmp_path / 'footprint.npy'
        np.save(path, footprint)

        out = io_handler.load_footprint(str(path), nside)

        np.testing.assert_array_equal(out, footprint)

    def test_fits_format_native_resolution(self, tmp_path):
        """A plain full-sky healpy FITS map (no PIXEL/WEIGHT columns) is read
        back at its native resolution via the hp.read_map fallback: it is
        NOT downgraded to the requested nside by load_footprint itself
        (that's the mask_utils.Mask/up_downgrade_map's job)."""
        nside = 16
        npix = healpy.nside2npix(nside)
        footprint = np.ones(npix)
        path = tmp_path / 'footprint.fits'
        healpy.write_map(str(path), footprint, overwrite=True, dtype=np.float64)

        out = io_handler.load_footprint(str(path), nside // 2)

        assert len(out) == npix

    def test_partial_masking_map_is_downgraded(self, tmp_path):
        """The "masking map" .fits format (PIXEL/WEIGHT columns, as produced by
        VMPZ-style partial-sky files) IS downgraded on the fly to the
        requested nside, via _read_masking_map."""
        fitsio = pytest.importorskip('fitsio')

        nside_in = 32
        nside_out = 16
        npix_in = healpy.nside2npix(nside_in)

        pixels = np.arange(npix_in)
        weights = np.ones(npix_in)
        data = np.zeros(len(pixels), dtype=[('PIXEL', 'i8'), ('WEIGHT', 'f8')])
        data['PIXEL'] = pixels
        data['WEIGHT'] = weights
        header = {'NSIDE': nside_in, 'ORDERING': 'NESTED'}
        path = tmp_path / 'partial_mask.fits'
        fitsio.write(str(path), data, header=header, clobber=True)

        out = io_handler.load_footprint(str(path), nside_out)

        assert len(out) == healpy.nside2npix(nside_out)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            io_handler.load_footprint('/no/such/footprint.fits', 8)

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / 'footprint.txt'
        path.write_text('dummy')
        with pytest.raises(ValueError, match='Unsupported file format'):
            io_handler.load_footprint(str(path), 8)


class TestLoadWeightMapFits:
    """Tests for load_weight_map_fits."""

    def test_valid_weight_map(self, tmp_path):
        nside = 8
        zbins = 3
        npix = healpy.nside2npix(nside)
        weight_maps = np.abs(
            np.random.default_rng(0).normal(1.0, 0.1, size=(zbins, npix))
        )
        path = tmp_path / 'weight_maps.fits'
        healpy.write_map(str(path), list(weight_maps), overwrite=True, dtype=np.float64)

        out = io_handler.load_weight_map_fits(str(path))

        assert out.shape == (zbins, npix)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            io_handler.load_weight_map_fits('/no/such/weight_map.fits')

    def test_non_fits_extension_raises(self, tmp_path):
        path = tmp_path / 'weight_maps.npy'
        np.save(path, np.ones((3, 10)))
        with pytest.raises(ValueError, match='must be a .fits file'):
            io_handler.load_weight_map_fits(str(path))

    def test_negative_values_raise(self, tmp_path):
        nside = 8
        zbins = 2
        npix = healpy.nside2npix(nside)
        weight_maps = np.ones((zbins, npix))
        weight_maps[0, 0] = -1.0
        path = tmp_path / 'weight_maps.fits'
        healpy.write_map(str(path), list(weight_maps), overwrite=True, dtype=np.float64)

        with pytest.raises(ValueError, match='negative values'):
            io_handler.load_weight_map_fits(str(path))

    def test_1d_map_raises_wrong_ndim(self, tmp_path):
        """A single-map (1D, zbins=1) FITS file should be rejected: the
        function requires a 2D (zbins, npix) array."""
        nside = 8
        npix = healpy.nside2npix(nside)
        weight_map = np.ones(npix)
        path = tmp_path / 'weight_map_1d.fits'
        healpy.write_map(str(path), weight_map, overwrite=True, dtype=np.float64)

        with pytest.raises(ValueError, match='2D array'):
            io_handler.load_weight_map_fits(str(path))


class TestLoadNzEuclidlib:
    """Tests for load_nz_euclidlib, using euclidlib's fixed PHZ FITS format:
    a binary table with 'bin_id' and 'n_z' (shape (3000,)) columns, on the
    fixed z grid np.linspace(0, 6, 3001).
    """

    def test_basic_round_trip(self, tmp_path):
        fitsio = pytest.importorskip('fitsio')
        pytest.importorskip('euclidlib')

        z = np.linspace(0.0, 6.0, 3001)
        n_hist = z.size - 1
        zbins = 2

        dtype = [('bin_id', 'i4'), ('n_z', 'f8', (n_hist,))]
        data = np.zeros(zbins, dtype=dtype)
        for i in range(zbins):
            data['bin_id'][i] = i + 1
            hist = np.zeros(n_hist)
            hist[100:200] = 1.0
            data['n_z'][i] = hist

        path = tmp_path / 'nz.fits'
        fitsio.write(str(path), data, clobber=True)

        z_out, nztab = io_handler.load_nz_euclidlib(str(path))

        np.testing.assert_allclose(z_out, z)
        assert nztab.shape == (len(z), zbins)


class TestLoadClEuclidlib:
    """Tests for load_cl_euclidlib, built via euclidlib's own
    `angular_power_spectra.write` helper to fabricate a valid results FITS file.
    """

    def test_basic_auto_spectrum_round_trip(self, tmp_path):
        pkwl = pytest.importorskip('euclidlib.le3.pk_wl')

        nbl = 5
        zbins = 2
        ells = np.arange(2, 2 + nbl, dtype=float)

        rng = np.random.default_rng(0)
        results = {}
        arrays = {}
        for zi in range(1, zbins + 1):
            for zj in range(zi, zbins + 1):
                arr = rng.normal(size=nbl)
                arrays[zi, zj] = arr
                results[('POS', 'POS', zi, zj)] = pkwl.AngularPowerSpectrum(
                    array=arr, ell=ells
                )

        path = tmp_path / 'cls.fits'
        pkwl.angular_power_spectra.write(str(path), results)

        ells_out, cl_3d = io_handler.load_cl_euclidlib(str(path), 'POS', 'POS')

        np.testing.assert_allclose(ells_out, ells)
        assert cl_3d.shape == (nbl, zbins, zbins)
        for (zi, zj), arr in arrays.items():
            np.testing.assert_allclose(cl_3d[:, zi - 1, zj - 1], arr)
            # auto-spectrum: lower triangle mirrors the upper one
            np.testing.assert_allclose(cl_3d[:, zj - 1, zi - 1], arr)

    def test_bad_filename_extension_raises(self):
        pytest.importorskip('euclidlib.le3.pk_wl')
        with pytest.raises(AssertionError, match=r'\.fits'):
            io_handler.load_cl_euclidlib('cls.txt', 'POS', 'POS')

    def test_invalid_key_raises(self):
        pytest.importorskip('euclidlib.le3.pk_wl')
        with pytest.raises(AssertionError, match='SHE.*POS'):
            io_handler.load_cl_euclidlib('cls.fits', 'INVALID', 'POS')
