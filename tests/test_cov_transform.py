"""Unit tests for spaceborne.cov_dict, spaceborne.cov_transform and the pure
(non-I/O) helpers in spaceborne.oc_interface.

These modules implement the "reshuffling" machinery of the covariance pipeline:
building the nested, key-locked covariance dictionary (cov_dict.py), converting
between 6D/8D/10D block representations and the final 4D array used for output
(cov_transform.py), and reordering OneCovariance's 2D matrices into Spaceborne's
probe ordering (oc_interface.py, pure-numpy functions only).
"""

import numpy as np
import pytest

from spaceborne import cov_dict as cd
from spaceborne import cov_transform as ct
from spaceborne import oc_interface as oc


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


# ----------------------------------------------------------------------------- #
# cov_transform: shared small-case fixtures
# ----------------------------------------------------------------------------- #
def _ind_auto(zbins):
    """Same construction as used internally by cov_3x2pt_10d_to_4d: upper
    triangular, row-major (i.e. np.triu_indices(zbins) ordering)."""
    return np.array([(i, j) for i in range(zbins) for j in range(i, zbins)])


def _ind_cross(zbins):
    """Same construction as used internally by cov_3x2pt_10d_to_4d: full
    row-major square ordering."""
    return np.array([(i, j) for i in range(zbins) for j in range(zbins)])


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1234)


# ----------------------------------------------------------------------------- #
# cov_transform.split_probe_name
# ----------------------------------------------------------------------------- #
class TestSplitProbeName:
    """Tests for splitting a concatenated probe-pair name into its two probes."""

    @pytest.mark.parametrize(
        'full_name,expected',
        [
            ('LLLL', ('LL', 'LL')),
            ('LLGL', ('LL', 'GL')),
            ('GLLL', ('GL', 'LL')),
            ('GLGG', ('GL', 'GG')),
            ('GGGG', ('GG', 'GG')),
        ],
    )
    def test_harmonic_space_splits(self, full_name, expected):
        assert ct.split_probe_name(full_name, 'harmonic') == expected

    @pytest.mark.parametrize(
        'full_name,expected',
        [
            ('xipxip', ('xip', 'xip')),
            ('ximgt', ('xim', 'gt')),
            ('gtgg', ('gt', 'gg')),
            ('gggg', ('gg', 'gg')),
        ],
    )
    def test_real_space_splits(self, full_name, expected):
        assert ct.split_probe_name(full_name, 'real') == expected

    def test_invalid_probe_name_raises(self):
        with pytest.raises(ValueError, match='Invalid probe name'):
            ct.split_probe_name('XXXX', 'harmonic')

    def test_invalid_space_raises(self):
        with pytest.raises(ValueError, match='space'):
            ct.split_probe_name('LLLL', 'not_a_space')

    def test_custom_valid_probes(self):
        """A custom valid_probes list overrides the harmonic/real defaults."""
        result = ct.split_probe_name('foobar', space=None, valid_probes=['foo', 'bar'])
        assert result == ('foo', 'bar')


# ----------------------------------------------------------------------------- #
# cov_transform.cov_6d_to_4d_blocks
# ----------------------------------------------------------------------------- #
class TestCov6dTo4dBlocks:
    """Element-by-element checks of the 6D -> 4D block reshaping, verified with
    explicit loops against the real ind_auto/ind_cross convention (not a
    reinvented one) also used by cov_3x2pt_10d_to_4d."""

    def test_auto_block_matches_explicit_loop(self, rng):
        """LL-LL-like (auto, auto) block: same ind array on both sides."""
        zbins, nbl = 3, 4
        ind = _ind_auto(zbins)
        zpairs = ind.shape[0]
        cov_6d = rng.normal(size=(nbl, nbl, zbins, zbins, zbins, zbins))

        cov_4d = ct.cov_6d_to_4d_blocks(cov_6d, nbl, ind, ind)

        assert cov_4d.shape == (nbl, nbl, zpairs, zpairs)
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for zij in range(zpairs):
                    zi, zj = ind[zij]
                    for zkl in range(zpairs):
                        zk, zl = ind[zkl]
                        assert (
                            cov_4d[ell1, ell2, zij, zkl]
                            == cov_6d[ell1, ell2, zi, zj, zk, zl]
                        )

    def test_cross_block_matches_explicit_loop(self, rng):
        """GL-LL-like (cross, auto) block: different ind arrays, non-square
        zpair axes."""
        zbins, nbl = 3, 3
        ind_ab = _ind_cross(zbins)  # e.g. GL: zpairs_cross = zbins**2
        ind_cd = _ind_auto(zbins)  # e.g. LL: zpairs_auto = zbins*(zbins+1)//2
        zpairs_ab, zpairs_cd = ind_ab.shape[0], ind_cd.shape[0]
        cov_6d = rng.normal(size=(nbl, nbl, zbins, zbins, zbins, zbins))

        cov_4d = ct.cov_6d_to_4d_blocks(cov_6d, nbl, ind_ab, ind_cd)

        assert cov_4d.shape == (nbl, nbl, zpairs_ab, zpairs_cd)
        assert zpairs_ab != zpairs_cd  # sanity: this is genuinely non-square
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for zij in range(zpairs_ab):
                    zi, zj = ind_ab[zij]
                    for zkl in range(zpairs_cd):
                        zk, zl = ind_cd[zkl]
                        assert (
                            cov_4d[ell1, ell2, zij, zkl]
                            == cov_6d[ell1, ell2, zi, zj, zk, zl]
                        )

    def test_zpair_counts_match_formula(self):
        """zpairs_auto = zbins*(zbins+1)/2, zpairs_cross = zbins**2."""
        zbins = 3
        assert _ind_auto(zbins).shape[0] == zbins * (zbins + 1) // 2
        assert _ind_cross(zbins).shape[0] == zbins**2


# ----------------------------------------------------------------------------- #
# cov_transform.cov_3x2pt_8d_dict_to_4d
# ----------------------------------------------------------------------------- #
class TestCov3x2pt8dDictTo4d:
    """Checks that the per-block 4D dict is stacked into the final 4D array at
    the right offsets (LL rows, then GL rows, then GG rows; and within each
    row LL columns, then GL columns, then GG columns)."""

    @pytest.fixture
    def blocks_9(self):
        """A full 3x2pt set of 9 (a,b,c,d) blocks, each filled with a distinct
        constant value so we can identify misplaced blocks unambiguously."""
        zbins, nbl = 2, 3
        zpairs_auto = zbins * (zbins + 1) // 2
        zpairs_cross = zbins**2

        row_probes = [('L', 'L'), ('G', 'L'), ('G', 'G')]
        zpairs_of = {
            ('L', 'L'): zpairs_auto,
            ('G', 'L'): zpairs_cross,
            ('G', 'G'): zpairs_auto,
        }

        req_probe_combs_2d = []
        cov_dict_4d = {}
        value = 0
        for a, b in row_probes:
            for c, d in row_probes:
                req_probe_combs_2d.append((a, b, c, d))
                shape = (nbl, nbl, zpairs_of[a, b], zpairs_of[c, d])
                cov_dict_4d[a, b, c, d] = np.full(shape, value, dtype=float)
                value += 1

        return cov_dict_4d, req_probe_combs_2d, zpairs_of, nbl

    def test_blocks_are_placed_at_correct_offsets(self, blocks_9):
        cov_dict_4d, req_probe_combs_2d, zpairs_of, nbl = blocks_9

        cov_4d = ct.cov_3x2pt_8d_dict_to_4d(cov_dict_4d, req_probe_combs_2d)

        row_probes = [('L', 'L'), ('G', 'L'), ('G', 'G')]
        row_offset = 0
        for a, b in row_probes:
            col_offset = 0
            for c, d in row_probes:
                block = cov_dict_4d[a, b, c, d]
                n_ab, n_cd = zpairs_of[a, b], zpairs_of[c, d]
                sub = cov_4d[
                    :, :, row_offset : row_offset + n_ab, col_offset : col_offset + n_cd
                ]
                np.testing.assert_array_equal(sub, block)
                col_offset += n_cd
            row_offset += n_ab

    def test_missing_probe_combination_raises(self, blocks_9):
        cov_dict_4d, req_probe_combs_2d, _, _ = blocks_9
        incomplete = list(req_probe_combs_2d) + [('X', 'X', 'X', 'X')]
        with pytest.raises(AssertionError):
            ct.cov_3x2pt_8d_dict_to_4d(cov_dict_4d, incomplete)

    def test_wrong_ndim_block_raises(self, blocks_9):
        cov_dict_4d, req_probe_combs_2d, _, _ = blocks_9
        cov_dict_4d['L', 'L', 'L', 'L'] = cov_dict_4d['L', 'L', 'L', 'L'][:, :, :, 0]
        with pytest.raises(AssertionError):
            ct.cov_3x2pt_8d_dict_to_4d(cov_dict_4d, req_probe_combs_2d)


# ----------------------------------------------------------------------------- #
# cov_transform.cov_10d_array_to_dict / cov_3x2pt_10d_to_4d
# ----------------------------------------------------------------------------- #
class TestCov10dArrayToDict:
    """cov_10d_array_to_dict references a module-level ``PROBE_ORDERING`` name
    that no longer exists in cov_transform.py (it was removed from the module
    in a refactor -- see git log "[IMP] remove probe_ordering and GL_OR_LG" --
    but this one usage inside cov_10d_array_to_dict was left behind). Calling
    the function therefore always raises NameError. This looks like a genuine
    bug/dead code path (the function is not called anywhere else in the
    codebase); marking xfail rather than silently skipping so it is easy to
    find once PROBE_ORDERING is restored (e.g. as
    ``[('L', 'L'), ('G', 'L'), ('G', 'G')]``)."""

    @pytest.mark.xfail(
        reason=(
            'cov_10d_array_to_dict references the module-level name '
            'PROBE_ORDERING, which was removed from cov_transform.py '
            "(see git log: '[IMP] remove probe_ordering and GL_OR_LG'). "
            'Calling this function always raises NameError.'
        ),
        raises=NameError,
        strict=True,
    )
    def test_roundtrip_matches_input_blocks(self, rng):
        zbins, nbl = 2, 3
        cov_10d = rng.normal(size=(2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))

        cov_10d_dict = ct.cov_10d_array_to_dict(cov_10d)

        # expected mapping: probe index 0 <-> 'L', 1 <-> 'G'
        np.testing.assert_array_equal(
            cov_10d_dict['L', 'L', 'L', 'L'], cov_10d[0, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            cov_10d_dict['G', 'L', 'G', 'G'], cov_10d[1, 0, 1, 1]
        )


class TestCov3x2pt10dTo4d:
    """cov_3x2pt_10d_to_4d delegates to cov_10d_array_to_dict internally, so it
    inherits the same NameError (see TestCov10dArrayToDict)."""

    @pytest.mark.xfail(
        reason=(
            'cov_3x2pt_10d_to_4d calls cov_10d_array_to_dict, which raises '
            'NameError due to the missing PROBE_ORDERING module-level name.'
        ),
        raises=NameError,
        strict=True,
    )
    def test_end_to_end_small_case(self, rng):
        zbins, nbl = 2, 3
        cov_10d = rng.normal(size=(2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))
        req_probe_combs_2d = [
            ('L', 'L', 'L', 'L'),
            ('G', 'L', 'L', 'L'),
            ('G', 'L', 'G', 'L'),
            ('G', 'G', 'G', 'G'),
        ]

        cov_4d = ct.cov_3x2pt_10d_to_4d(cov_10d, zbins, nbl, req_probe_combs_2d)

        assert cov_4d.shape[0] == nbl
        assert cov_4d.shape[1] == nbl


# ----------------------------------------------------------------------------- #
# oc_interface: pure-numpy reshuffling helpers (no external binary involved)
# ----------------------------------------------------------------------------- #
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
    """Consistency check: the generic, probe-list-driven cov_ggglll_to_llglgg
    must reproduce the old hardcoded _cov_ggglll_to_llglgg reference
    implementation (kept in the module for exactly this purpose) when given
    the canonical full probe_hs_list=['ll', 'gl', 'gg']."""

    def test_matches_reference_implementation_full_3x2pt(self, rng):
        elem_auto, elem_cross = 3, 2
        n = 2 * elem_auto + elem_cross
        m = rng.normal(size=(n, n))
        cov_ggglll = m + m.T  # symmetric, like a real covariance

        expected = oc._cov_ggglll_to_llglgg(cov_ggglll, elem_auto, elem_cross)
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
