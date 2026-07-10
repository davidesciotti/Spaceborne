"""Unit tests for spaceborne.cov_dict.

Covers the "reshuffling" scaffolding used to build the nested, key-locked
covariance dictionary: FrozenDict (a dict wrapper that locks its keys and,
optionally, validates leaf array dimensions) and create_cov_dict (the
3-level term -> probe_pair -> dim factory built on top of it).
"""

import numpy as np
import pytest

from spaceborne import cov_dict as cd


# ----------------------------------------------------------------------------- #
# cov_dict.FrozenDict
# ----------------------------------------------------------------------------- #
class TestFrozenDict:
    """Tests for the key-locking / dimension-validating dict wrapper."""

    def test_existing_key_is_mutable(self):
        """Values at pre-existing keys can be freely overwritten."""
        fd = cd.FrozenDict({'a': None}, validate_dims=False)
        fd['a'] = 1
        fd['a'] = 2
        assert fd['a'] == 2

    def test_new_key_raises(self):
        """Adding a key that was not present at construction time raises."""
        fd = cd.FrozenDict({'a': None}, validate_dims=False)
        with pytest.raises(KeyError):
            fd['b'] = 1

    def test_protect_structure_blocks_overwriting_nested_frozendict(self):
        """A nested FrozenDict cannot be replaced by a plain value."""
        inner = cd.FrozenDict({'4d': None}, validate_dims=False)
        outer = cd.FrozenDict({'block': inner}, protect_structure=True)
        with pytest.raises(TypeError):
            outer['block'] = {}

    def test_protect_structure_allows_nested_frozendict_replacement(self):
        """Replacing a nested FrozenDict with *another* FrozenDict is allowed by
        protect_structure itself (validate_dims is disabled here to isolate
        this behavior, since with validate_dims=True a FrozenDict value would
        also be rejected for not being None/ndarray -- see create_cov_dict,
        where every level effectively has validate_dims=True)."""
        inner = cd.FrozenDict({'4d': None}, validate_dims=False)
        outer = cd.FrozenDict(
            {'block': inner}, protect_structure=True, validate_dims=False
        )
        new_inner = cd.FrozenDict({'4d': None}, validate_dims=False)
        outer['block'] = new_inner
        assert outer['block'] is new_inner

    def test_validate_dims_rejects_non_array_non_none(self):
        """Leaf values must be None or a numpy array when validate_dims=True."""
        fd = cd.FrozenDict({'4d': None}, validate_dims=True)
        with pytest.raises(TypeError):
            fd['4d'] = [1, 2, 3]

    def test_validate_dims_accepts_none(self):
        """None is always an acceptable value, even with validate_dims=True."""
        fd = cd.FrozenDict({'4d': np.zeros((1, 1, 1, 1))}, validate_dims=True)
        fd['4d'] = None
        assert fd['4d'] is None

    def test_validate_dims_checks_ndim_from_key_name(self):
        """A key like '4d' requires an array with exactly 4 dimensions."""
        fd = cd.FrozenDict({'4d': None}, validate_dims=True)
        with pytest.raises(ValueError):
            fd['4d'] = np.zeros((2, 2, 2))  # only 3 dims

    def test_validate_dims_accepts_matching_ndim(self):
        """A '6d' key accepts a 6-dimensional array."""
        fd = cd.FrozenDict({'6d': None}, validate_dims=True)
        arr = np.zeros((1, 1, 1, 1, 1, 1))
        fd['6d'] = arr
        assert fd['6d'] is arr

    @pytest.mark.parametrize('method,args', [('pop', ()), ('clear', ())])
    def test_pop_and_clear_raise(self, method, args):
        """Deleting keys/values is always disallowed, regardless of freezing flags."""
        fd = cd.FrozenDict({'a': 1}, validate_dims=False)
        with pytest.raises(TypeError):
            getattr(fd, method)(*args)

    def test_popitem_raises(self):
        fd = cd.FrozenDict({'a': 1}, validate_dims=False)
        with pytest.raises(TypeError):
            fd.popitem()


# ----------------------------------------------------------------------------- #
# cov_dict.create_cov_dict
# ----------------------------------------------------------------------------- #
class TestCreateCovDict:
    """Tests for the 3-level (term -> probe_pair -> dim) frozen structure."""

    @pytest.fixture
    def small_cov_dict(self):
        terms = ['g', 'ssc']
        probe_pairs = [('LL', 'LL'), ('GL', 'GL'), ('GG', 'GG')]
        dims = ['2d', '4d', '6d']
        return cd.create_cov_dict(terms, probe_pairs, dims), terms, probe_pairs, dims

    def test_top_level_keys(self, small_cov_dict):
        """Only the requested terms exist at the top level."""
        cov_dict, terms, _, _ = small_cov_dict
        assert set(cov_dict.keys()) == set(terms)

    def test_probe_level_keys(self, small_cov_dict):
        """Each term contains exactly the requested probe pairs."""
        cov_dict, terms, probe_pairs, _ = small_cov_dict
        for term in terms:
            assert set(cov_dict[term].keys()) == set(probe_pairs)

    def test_dim_level_keys_and_initial_values(self, small_cov_dict):
        """Each probe pair contains exactly the requested dims, all None initially."""
        cov_dict, terms, probe_pairs, dims = small_cov_dict
        for term in terms:
            for probe_pair in probe_pairs:
                leaf = cov_dict[term][probe_pair]
                assert set(leaf.keys()) == set(dims)
                assert all(leaf[dim] is None for dim in dims)

    def test_only_requested_probe_blocks_allocated(self):
        """Requesting a single probe pair must not allocate the other blocks
        (important: single-probe runs must not crash by hardcoding all 9 combos)."""
        cov_dict = cd.create_cov_dict(['g'], [('LL', 'LL')], ['4d'])
        assert set(cov_dict['g'].keys()) == {('LL', 'LL')}

    def test_setting_array_with_correct_shape_works(self, small_cov_dict):
        """Setting a properly-shaped array at a leaf works end to end."""
        cov_dict, _, _, _ = small_cov_dict
        nbl, zpair = 4, 6
        arr = np.arange(nbl * nbl * zpair * zpair, dtype=float).reshape(
            nbl, nbl, zpair, zpair
        )
        cov_dict['g'][('LL', 'LL')]['4d'] = arr
        np.testing.assert_array_equal(cov_dict['g'][('LL', 'LL')]['4d'], arr)

    def test_setting_wrong_ndim_array_raises(self, small_cov_dict):
        """Setting a 3D array into the '4d' slot must raise."""
        cov_dict, _, _, _ = small_cov_dict
        with pytest.raises(ValueError):
            cov_dict['g'][('LL', 'LL')]['4d'] = np.zeros((2, 2, 2))

    def test_setting_unrequested_term_raises(self, small_cov_dict):
        cov_dict, _, _, _ = small_cov_dict
        with pytest.raises(KeyError):
            cov_dict['cng'] = {}

    def test_setting_unrequested_probe_pair_raises(self, small_cov_dict):
        cov_dict, _, _, _ = small_cov_dict
        with pytest.raises(KeyError):
            cov_dict['g'][('XX', 'XX')] = {}

    def test_setting_unrequested_dim_raises(self, small_cov_dict):
        cov_dict, _, _, _ = small_cov_dict
        with pytest.raises(KeyError):
            cov_dict['g'][('LL', 'LL')]['8d'] = np.zeros((1,) * 8)

    def test_overwriting_probe_pair_structure_raises(self, small_cov_dict):
        """Cannot replace the FrozenDict at a probe-pair slot with a plain dict."""
        cov_dict, _, _, _ = small_cov_dict
        with pytest.raises(TypeError):
            cov_dict['g'][('LL', 'LL')] = {}
