"""Unit tests for the ell_max_nmt of the NaMaster covariance (precision.ell_max_nmt)."""

import warnings

import pytest

from spaceborne import cov_partial_sky


class TestResolveEllMaxNmt:
    """Tests for resolve_ell_max_nmt."""

    def test_default_is_one_and_a_half_ell_max(self):
        assert cov_partial_sky.resolve_ell_max_nmt(None, 1499, 3071, True) == 2249

    def test_explicit_value(self):
        assert cov_partial_sky.resolve_ell_max_nmt(2000, 1499, 3071, True) == 2000

    def test_equal_to_ell_max_is_no_buffer(self):
        assert cov_partial_sky.resolve_ell_max_nmt(1499, 1499, 3071, True) == 1499

    def test_capped_with_warning(self):
        with pytest.warns(UserWarning, match='truncated to 1535'):
            out = cov_partial_sky.resolve_ell_max_nmt(None, 1499, 1535, True)
        assert out == 1535

    def test_cap_never_below_ell_max(self):
        with pytest.warns(UserWarning, match='truncated'):
            out = cov_partial_sky.resolve_ell_max_nmt(None, 1499, 1200, True)
        assert out == 1499

    def test_below_ell_max_raises(self):
        with pytest.raises(ValueError, match='below ell_max'):
            cov_partial_sky.resolve_ell_max_nmt(1000, 1499, 3071, True)

    def test_decoupled_uses_ell_max(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert cov_partial_sky.resolve_ell_max_nmt(None, 1499, 3071, False) == 1499
        with pytest.warns(UserWarning, match='only used for the coupled'):
            out = cov_partial_sky.resolve_ell_max_nmt(2000, 1499, 3071, False)
        assert out == 1499
