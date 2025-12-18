"""Unit tests for spaceborne.ell_utils module."""

import numpy as np
import pytest

from spaceborne import ell_utils


class TestComputeElls:
    """Tests for compute_ells function."""

    def test_istf_recipe_basic(self):
        """Test ISTF (log) binning produces expected structure."""
        nbl = 10
        ell_min = 10
        ell_max = 1000
        
        ells, deltas = ell_utils.compute_ells(nbl, ell_min, ell_max, 'ISTF')
        
        assert len(ells) == nbl
        assert len(deltas) == nbl
        assert ells[0] > ell_min
        assert ells[-1] < ell_max
        assert np.all(np.diff(ells) > 0)  # monotonically increasing

    def test_istf_with_edges(self):
        """Test ISTF recipe returns edges when requested."""
        nbl = 10
        ell_min = 10
        ell_max = 1000
        
        ells, deltas, edges = ell_utils.compute_ells(
            nbl, ell_min, ell_max, 'ISTF', output_ell_bin_edges=True
        )
        
        assert len(edges) == nbl + 1
        assert edges[0] == ell_min
        assert edges[-1] == ell_max
        # Check that ells are approximately centered
        np.testing.assert_allclose(ells, (edges[:-1] + edges[1:]) / 2.0)

    def test_istnl_recipe_basic(self):
        """Test ISTNL (linear in log space) binning."""
        nbl = 10
        ell_min = 10
        ell_max = 1000
        
        ells, deltas = ell_utils.compute_ells(nbl, ell_min, ell_max, 'ISTNL')
        
        assert len(ells) == nbl
        assert len(deltas) == nbl
        assert ells[0] > ell_min
        assert ells[-1] < ell_max
        assert np.all(np.diff(ells) > 0)

    def test_lin_recipe_basic(self):
        """Test linear binning."""
        nbl = 10
        ell_min = 10
        ell_max = 100
        
        ells, deltas = ell_utils.compute_ells(nbl, ell_min, ell_max, 'lin')
        
        assert len(ells) == nbl
        assert len(deltas) == nbl
        # Linear spacing should have equal deltas
        np.testing.assert_allclose(deltas, deltas[0])
        # Check spacing
        expected_delta = (ell_max - ell_min) / nbl
        np.testing.assert_allclose(deltas[0], expected_delta)

    def test_invalid_recipe(self):
        """Test that invalid recipe raises ValueError."""
        with pytest.raises(ValueError, match='recipe must be either'):
            ell_utils.compute_ells(10, 10, 1000, 'invalid_recipe')

    def test_consistency_between_outputs(self):
        """Test that deltas match bin edges for linear binning."""
        nbl = 5
        ell_min = 10
        ell_max = 100
        
        # Only test for linear binning where deltas = diff(edges)
        ells, deltas, edges = ell_utils.compute_ells(
            nbl, ell_min, ell_max, 'lin', output_ell_bin_edges=True
        )
        np.testing.assert_allclose(deltas, np.diff(edges))


class TestGetLmid:
    """Tests for get_lmid function."""

    def test_basic_diagonal_1(self):
        """Test getting midpoints for 1st diagonal."""
        ells = np.array([10.0, 20.0, 30.0, 40.0])
        k = 1
        
        lmid = ell_utils.get_lmid(ells, k)
        
        expected = np.array([15.0, 25.0, 35.0])
        np.testing.assert_array_equal(lmid, expected)

    def test_basic_diagonal_2(self):
        """Test getting midpoints for 2nd diagonal."""
        ells = np.array([10.0, 20.0, 30.0, 40.0])
        k = 2
        
        lmid = ell_utils.get_lmid(ells, k)
        
        expected = np.array([20.0, 30.0])
        np.testing.assert_array_equal(lmid, expected)

    def test_diagonal_0(self):
        """Test k=0 edge case."""
        ells = np.array([10.0, 20.0, 30.0, 40.0])
        k = 0
        
        # When k=0, ells[:-0] produces an empty array, so this is an edge case
        # The function may not be designed for k=0, so we test k>=1 only
        # Skip this test or test that it raises an error
        with pytest.raises((ValueError, IndexError)):
            ell_utils.get_lmid(ells, k)


class TestEllBinning:
    """Tests for EllBinning class."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            'binning': {
                'binning_type': 'log',
                'ell_min': 10,
                'ell_max': 1000,
                'ell_bins': 10,
            },
            'namaster': {'use_namaster': False},
            'sample_covariance': {'compute_sample_cov': False},
        }

    def test_initialization(self, basic_config):
        """Test basic initialization."""
        ell_obj = ell_utils.EllBinning(basic_config)
        
        assert ell_obj.binning_type == 'log'
        assert ell_obj.use_namaster is False
        assert ell_obj.do_sample_cov is False

    def test_log_binning(self, basic_config):
        """Test logarithmic binning."""
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        
        assert len(ell_obj.ells_WL) == 10
        assert len(ell_obj.ells_GC) == 10
        assert len(ell_obj.delta_l_WL) == 10
        assert len(ell_obj.delta_l_GC) == 10
        assert ell_obj.nbl_WL == 10
        assert ell_obj.nbl_GC == 10

    def test_linear_binning(self, basic_config):
        """Test linear binning."""
        basic_config['binning']['binning_type'] = 'lin'
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        
        # Linear spacing should have approximately equal deltas
        np.testing.assert_allclose(
            ell_obj.delta_l_WL, ell_obj.delta_l_WL[0], rtol=1e-10
        )

    def test_unbinned(self, basic_config):
        """Test unbinned mode."""
        basic_config['binning']['binning_type'] = 'unbinned'
        basic_config['binning']['ell_max'] = 20  # Small range for testing
        
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        
        # Unbinned should have unit bin widths
        np.testing.assert_array_equal(ell_obj.delta_l_WL, np.ones(11))
        # Should be consecutive integers
        expected = np.arange(10, 21)
        np.testing.assert_array_equal(ell_obj.ells_WL, expected)

    def test_3x2pt_follows_gc(self, basic_config):
        """Test that 3x2pt binning follows GC."""
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        
        np.testing.assert_array_equal(ell_obj.ells_3x2pt, ell_obj.ells_GC)
        np.testing.assert_array_equal(ell_obj.delta_l_3x2pt, ell_obj.delta_l_GC)
        assert ell_obj.nbl_3x2pt == ell_obj.nbl_GC

    def test_xc_follows_gc(self, basic_config):
        """Test that XC binning follows GC."""
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        
        np.testing.assert_array_equal(ell_obj.ells_XC, ell_obj.ells_GC)
        np.testing.assert_array_equal(ell_obj.delta_l_XC, ell_obj.delta_l_GC)
        assert ell_obj.nbl_XC == ell_obj.nbl_GC

    def test_invalid_binning_type(self, basic_config):
        """Test that invalid binning type raises error."""
        basic_config['binning']['binning_type'] = 'invalid'
        ell_obj = ell_utils.EllBinning(basic_config)
        
        with pytest.raises(ValueError, match='binning_type .* not recognized'):
            ell_obj.build_ell_bins()

    def test_validate_bins(self, basic_config):
        """Test bin validation."""
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        
        # Should not raise any errors
        ell_obj._validate_bins()

    def test_compute_ells_3x2pt_unbinned(self, basic_config):
        """Test unbinned 3x2pt ell computation."""
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        ell_obj.compute_ells_3x2pt_unbinned()
        
        assert ell_obj.nbl_3x2pt_unb == ell_obj.ell_max_3x2pt + 1
        assert len(ell_obj.ells_3x2pt_unb) == ell_obj.nbl_3x2pt_unb
        assert ell_obj.ells_3x2pt_unb[-1] == ell_obj.ell_max_3x2pt

    def test_compute_ells_3x2pt_rs(self, basic_config):
        """Test real space 3x2pt ell computation."""
        basic_config['precision'] = {
            'ell_min_rs': 1,
            'ell_max_rs': 10000,
            'ell_bins_rs': 100,
        }
        
        ell_obj = ell_utils.EllBinning(basic_config)
        ell_obj.build_ell_bins()
        ell_obj.compute_ells_3x2pt_rs()
        
        assert len(ell_obj.ells_3x2pt_rs) == 100
        assert ell_obj.nbl_3x2pt_rs == 100
        assert ell_obj.ell_max_3x2pt_rs == 10000


class TestLoadEllCuts:
    """Tests for load_ell_cuts function."""

    def test_basic_functionality(self):
        """Test basic ell cuts computation."""
        # Mock simple cosmology - just test the structure
        kmax_h_over_Mpc = 1.0
        zbins = 2
        z_values_a = np.array([0.5, 1.0])
        z_values_b = np.array([0.5, 1.0])
        h = 0.7
        kmax_h_over_Mpc_ref = 1.0
        
        # Mock cosmo_ccl that returns simple values
        class MockCosmo:
            pass
        
        cosmo_ccl = MockCosmo()
        
        # Temporarily replace the function
        original_func = ell_utils.cosmo_lib.ccl_comoving_distance
        
        def mock_distance(z, use_h_units, cosmo_ccl):
            return 1000.0 * z  # Simple linear relation
        
        ell_utils.cosmo_lib.ccl_comoving_distance = mock_distance
        
        try:
            ell_cuts = ell_utils.load_ell_cuts(
                kmax_h_over_Mpc, z_values_a, z_values_b, 
                cosmo_ccl, zbins, h, kmax_h_over_Mpc_ref
            )
            
            assert ell_cuts.shape == (2, 2)
            assert np.all(ell_cuts > 0)
            # ell_cut takes the minimum of the two, so symmetric matrix
            assert ell_cuts[0, 1] == ell_cuts[1, 0]
        finally:
            # Restore original function
            ell_utils.cosmo_lib.ccl_comoving_distance = original_func

    def test_uses_default_kmax(self):
        """Test that default kmax is used when None provided."""
        kmax_ref = 1.5
        zbins = 1
        z_values_a = np.array([0.5])
        z_values_b = np.array([0.5])
        h = 0.7
        
        class MockCosmo:
            pass
        
        cosmo_ccl = MockCosmo()
        
        original_func = ell_utils.cosmo_lib.ccl_comoving_distance
        
        def mock_distance(z, use_h_units, cosmo_ccl):
            return 1000.0 * z
        
        ell_utils.cosmo_lib.ccl_comoving_distance = mock_distance
        
        try:
            # When kmax is None, should use kmax_ref
            ell_cuts = ell_utils.load_ell_cuts(
                None, z_values_a, z_values_b, 
                cosmo_ccl, zbins, h, kmax_ref
            )
            
            # Should have computed something
            assert ell_cuts.shape == (1, 1)
            assert ell_cuts[0, 0] > 0
        finally:
            ell_utils.cosmo_lib.ccl_comoving_distance = original_func
