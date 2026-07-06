"""Unit tests for the pure-numpy (non-I/O) reshuffling helpers in
spaceborne.oc_interface: reorder_block_cov and cov_ggglll_to_llglgg, which
translate OneCovariance's 2D matrices into Spaceborne's probe ordering.
"""

import numpy as np
import pytest

from spaceborne import oc_interface as oc


class TestReorderBlockCov:
    """Tests for the generic block-permutation helper used to translate between
    OneCovariance's and Spaceborne's probe orderings."""

    def test_basic_reordering(self):
        block_sizes = {'a': 2, 'b': 3, 'c': 1}
        from_order = ['a', 'b', 'c']
        to_order = ['c', 'a', 'b']
        n = sum(block_sizes.values())
        cov = np.arange(n * n, dtype=float).reshape(n, n)

        reordered = oc.reorder_block_cov(cov, block_sizes, from_order, to_order)

        # build the expected result by explicit index permutation using the
        # cumulative offsets implied by block_sizes and from_order
        offsets = {'a': (0, 2), 'b': (2, 5), 'c': (5, 6)}
        expected_idx = np.concatenate([np.arange(*offsets[lab]) for lab in to_order])
        expected = cov[np.ix_(expected_idx, expected_idx)]

        np.testing.assert_array_equal(reordered, expected)

    def test_subset_to_order_is_allowed(self):
        """to_order may be a strict subset of from_order (selecting blocks)."""
        block_sizes = {'a': 2, 'b': 2}
        cov = np.arange(16, dtype=float).reshape(4, 4)
        reordered = oc.reorder_block_cov(cov, block_sizes, ['a', 'b'], ['b'])
        np.testing.assert_array_equal(reordered, cov[2:4, 2:4])

    def test_non_square_input_raises(self):
        with pytest.raises(ValueError, match='square'):
            oc.reorder_block_cov(np.zeros((3, 4)), {'a': 3}, ['a'], ['a'])

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError, match='does not match'):
            oc.reorder_block_cov(np.zeros((4, 4)), {'a': 3}, ['a'], ['a'])

    def test_missing_block_size_raises(self):
        with pytest.raises(ValueError, match='Missing block size'):
            oc.reorder_block_cov(np.zeros((4, 4)), {'a': 4}, ['a', 'b'], ['a'])


class TestCovGggllToLlglgg:
    """cov_ggglll_to_llglgg reorders a covariance whose blocks follow
    OneCovariance's (gg, gl, ll) ordering into Spaceborne's (ll, gl, gg).
    Since this is a pure block permutation, we check it against an explicit
    index permutation built from the block offsets."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(seed=1234)

    def test_matches_explicit_permutation_full_3x2pt(self, rng):
        elem_auto, elem_cross = 3, 2
        n = 2 * elem_auto + elem_cross
        m = rng.normal(size=(n, n))
        cov_ggglll = m + m.T  # symmetric, like a real covariance

        # input layout: gg [0, a), gl [a, a+c), ll [a+c, 2a+c)
        gg_idx = np.arange(0, elem_auto)
        gl_idx = np.arange(elem_auto, elem_auto + elem_cross)
        ll_idx = np.arange(elem_auto + elem_cross, n)
        perm = np.concatenate([ll_idx, gl_idx, gg_idx])
        expected = cov_ggglll[np.ix_(perm, perm)]

        actual = oc.cov_ggglll_to_llglgg(
            cov_ggglll,
            elem_auto,
            elem_cross,
            'harmonic',
            probe_hs_list=['ll', 'gl', 'gg'],
            probe_rs_list=[],
            probe_cs_list=[],
        )

        np.testing.assert_array_equal(actual, expected)

    def test_partial_probe_selection(self, rng):
        """With only 'll' and 'gg' requested, the function should just swap
        those two blocks (no 'gl' block present in the input at all)."""
        elem_auto = 3
        n = 2 * elem_auto
        m = rng.normal(size=(n, n))
        cov_gg_ll = m + m.T

        actual = oc.cov_ggglll_to_llglgg(
            cov_gg_ll,
            elem_auto,
            elem_cross=0,
            obs_space='harmonic',
            probe_hs_list=['ll', 'gg'],
            probe_rs_list=[],
            probe_cs_list=[],
        )

        expected = np.block(
            [
                [cov_gg_ll[elem_auto:, elem_auto:], cov_gg_ll[elem_auto:, :elem_auto]],
                [cov_gg_ll[:elem_auto, elem_auto:], cov_gg_ll[:elem_auto, :elem_auto]],
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_obs_space_raises(self):
        with pytest.raises(ValueError, match='obs_space'):
            oc.cov_ggglll_to_llglgg(np.zeros((2, 2)), 1, 1, 'invalid_space', [], [], [])
