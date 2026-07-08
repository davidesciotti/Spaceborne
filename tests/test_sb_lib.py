"""Unit tests for the (pure, non-CCL) helper functions in spaceborne.sb_lib.

sb_lib.py is a large grab-bag utility module; this file only targets the
functions that are pure numpy/scipy and don't need a Boltzmann/CCL cosmology
or the full pipeline to exercise. Where the source has a genuine bug we pin
it down with ``pytest.mark.xfail`` instead of touching ``spaceborne/``.
"""

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import jv

from spaceborne import constants as const
from spaceborne import sb_lib as sl
from spaceborne.cov_dict import create_cov_dict


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=1234)


# ----------------------------------------------------------------------------- #
# get_zsteps
# ----------------------------------------------------------------------------- #
class TestGetZsteps:
    """Tests for get_zsteps (grid-point count for a target linspace step)."""

    def test_exact_divisor(self):
        """z range that divides evenly into delta_z gives the expected count."""
        n = sl.get_zsteps(z_min=0.0, z_max=1.0, delta_z=0.1)
        assert n == 11
        grid = np.linspace(0.0, 1.0, n)
        np.testing.assert_allclose(np.diff(grid), 0.1)

    def test_actual_spacing_not_larger_than_requested(self):
        """The resulting linspace must never have a coarser step than delta_z."""
        z_min, z_max, delta_z = 0.2, 2.7, 0.13
        n = sl.get_zsteps(z_min, z_max, delta_z)
        grid = np.linspace(z_min, z_max, n)
        assert np.all(np.diff(grid) <= delta_z + 1e-12)

    def test_delta_z_not_positive_raises(self):
        with pytest.raises(ValueError, match='delta_z must be positive'):
            sl.get_zsteps(0.0, 1.0, 0.0)
        with pytest.raises(ValueError, match='delta_z must be positive'):
            sl.get_zsteps(0.0, 1.0, -0.1)

    def test_z_max_not_greater_than_z_min_raises(self):
        with pytest.raises(ValueError, match='z_max must be greater than z_min'):
            sl.get_zsteps(1.0, 1.0, 0.1)
        with pytest.raises(ValueError, match='z_max must be greater than z_min'):
            sl.get_zsteps(1.0, 0.5, 0.1)


# ----------------------------------------------------------------------------- #
# hartlap_factor / percival_factor
# ----------------------------------------------------------------------------- #
class TestHartlapFactor:
    """Tests for the Hartlap (2007) precision-matrix correction factor."""

    def test_known_value(self):
        """hartlap = (n_sim - n_data - 2) / (n_sim - 1)."""
        n_sim, n_data = 100, 10
        expected = (n_sim - n_data - 2) / (n_sim - 1)
        assert sl.hartlap_factor(n_sim, n_data) == pytest.approx(expected)

    def test_large_n_sim_approaches_one(self):
        """As n_sim -> infinity (n_data fixed), the correction vanishes (-> 1)."""
        assert sl.hartlap_factor(n_sim=10**7, n_data=50) == pytest.approx(1.0, abs=1e-4)

    def test_n_sim_leq_one_raises(self):
        with pytest.raises(ValueError, match='n_sim must be > 1'):
            sl.hartlap_factor(n_sim=1, n_data=5)
        with pytest.raises(ValueError, match='n_sim must be > 1'):
            sl.hartlap_factor(n_sim=0, n_data=5)

    def test_non_positive_regime_warns(self):
        """n_sim <= n_data + 2 gives a non-positive correction and should warn."""
        with pytest.warns(UserWarning, match='non-positive'):
            value = sl.hartlap_factor(n_sim=10, n_data=20)
        assert value <= 0


class TestPercivalFactor:
    """Tests for the Percival et al. (2014) precision-matrix correction factor."""

    def test_large_n_sim_approaches_one(self):
        """As n_sim -> infinity, beta -> 1 regardless of n_data, n_param."""
        beta = sl.percival_factor(n_sim=10**7, n_data=50, n_param=8)
        assert beta == pytest.approx(1.0, abs=1e-6)

    def test_n_param_equals_n_data_special_case(self):
        """When n_param == n_data, m1 == 1 and beta == 1 / (1 + A) exactly."""
        n_sim, n_data = 500, 30
        beta = sl.percival_factor(n_sim, n_data, n_param=n_data)
        a = 2 / (n_sim - n_data - 1) / (n_sim - n_data - 4)
        assert beta * (1 + a) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- #
# build_probe_list / get_probe_combs / get_probe_combs_wrapper / split_probe_name
# ----------------------------------------------------------------------------- #
class TestBuildProbeList:
    """Tests for build_probe_list."""

    def test_no_cross_terms(self):
        out = sl.build_probe_list(['LL', 'GL', 'GG'], include_cross_terms=False)
        assert out == ['LLLL', 'GLGL', 'GGGG']

    def test_with_cross_terms(self):
        out = sl.build_probe_list(['LL', 'GG'], include_cross_terms=True)
        assert out == ['LLLL', 'LLGG', 'GGGG']

    def test_with_cross_terms_three_probes(self):
        out = sl.build_probe_list(['LL', 'GL', 'GG'], include_cross_terms=True)
        # combinations_with_replacement(3, 2) -> 6 combinations
        assert out == ['LLLL', 'LLGL', 'LLGG', 'GLGL', 'GLGG', 'GGGG']

    def test_single_probe(self):
        assert sl.build_probe_list(['LL'], include_cross_terms=False) == ['LLLL']
        assert sl.build_probe_list(['LL'], include_cross_terms=True) == ['LLLL']


class TestSplitProbeName:
    """Tests for split_probe_name."""

    def test_harmonic_space(self):
        assert sl.split_probe_name('LLGG', 'harmonic') == ('LL', 'GG')
        assert sl.split_probe_name('GLGL', 'harmonic') == ('GL', 'GL')

    def test_real_space_variable_length(self):
        """Real-space probe names have variable length (xip/xim/gt/w)."""
        assert sl.split_probe_name('gtxim', 'real') == ('gt', 'xim')
        assert sl.split_probe_name('xipxip', 'real') == ('xip', 'xip')
        assert sl.split_probe_name('wgt', 'real') == ('w', 'gt')

    def test_invalid_space_raises(self):
        with pytest.raises(ValueError, match='space'):
            sl.split_probe_name('LLGG', 'not_a_space')

    def test_invalid_probe_name_raises(self):
        with pytest.raises(ValueError, match='Invalid probe name'):
            sl.split_probe_name('XXYY', 'harmonic')

    def test_custom_valid_probes(self):
        out = sl.split_probe_name('AB', 'harmonic', valid_probes=['A', 'B'])
        assert out == ('A', 'B')


class TestGetProbeCombs:
    """Tests for get_probe_combs (symmetric-fill / non-required probe combos)."""

    def test_cross_probe_yields_symmetric_partner(self):
        symm, nonreq = sl.get_probe_combs(['LLGG'], space='harmonic')
        assert symm == ['GGLL']
        assert 'LLGG' not in nonreq
        assert 'GGLL' not in nonreq
        # everything else in HS_ALL_PROBE_COMBS is non-required
        assert nonreq == set(const.HS_ALL_PROBE_COMBS) - {'LLGG', 'GGLL'}

    def test_diag_only_has_no_symmetric_partner(self):
        symm, nonreq = sl.get_probe_combs(['LLLL', 'GGGG'], space='harmonic')
        assert symm == []
        assert nonreq == set(const.HS_ALL_PROBE_COMBS) - {'LLLL', 'GGGG'}

    def test_invalid_space_raises(self):
        with pytest.raises(AssertionError):
            sl.get_probe_combs(['LLLL'], space='bogus')

    def test_unknown_probe_raises(self):
        with pytest.raises(ValueError, match='not found'):
            sl.get_probe_combs(['ZZZZ'], space='harmonic')


class TestGetProbeCombsWrapper:
    """Tests for the composite get_probe_combs_wrapper."""

    def test_two_probes_with_cross_cov(self):
        probe_selection = {'LL': True, 'GL': False, 'GG': True}
        out = sl.get_probe_combs_wrapper('harmonic', probe_selection, cross_cov=True)

        assert out['unique_probe_combs'] == ['LLLL', 'LLGG', 'GGGG']
        assert out['symm_probe_combs'] == ['GGLL']
        assert out['req_probe_combs_2d'] == ['LLLL', 'LLGG', 'GGLL', 'GGGG']
        assert out['nonreq_probe_combs'] == []

    def test_two_probes_without_cross_cov(self):
        """cross_cov=False drops LLGG/GGLL from the *computed* blocks, but they
        still show up in req_probe_combs_2d (needed for 2D assembly) and thus
        in nonreq_probe_combs (computed elsewhere or filled by symmetry)."""
        probe_selection = {'LL': True, 'GL': False, 'GG': True}
        out = sl.get_probe_combs_wrapper('harmonic', probe_selection, cross_cov=False)

        assert out['unique_probe_combs'] == ['LLLL', 'GGGG']
        assert out['symm_probe_combs'] == []
        assert out['req_probe_combs_2d'] == ['LLLL', 'LLGG', 'GGLL', 'GGGG']
        assert set(out['nonreq_probe_combs']) == {'LLGG', 'GGLL'}

    def test_unknown_obs_space_raises(self):
        with pytest.raises(ValueError, match='Unknown observables space'):
            sl.get_probe_combs_wrapper('bogus', {'LL': True}, cross_cov=True)


# ----------------------------------------------------------------------------- #
# copy_dict_leaf_level / build_cl_3x2pt_5d
# ----------------------------------------------------------------------------- #
class TestCopyDictLeafLevel:
    """Tests for copy_dict_leaf_level (deep-copies leaf arrays into new_dict)."""

    def test_deep_copies_arrays(self, rng):
        original = create_cov_dict(['g'], [('LL', 'LL')], ['6d'])
        original['g'][('LL', 'LL')]['6d'] = rng.standard_normal((2, 2, 3, 3, 3, 3))
        new = create_cov_dict(['g'], [('LL', 'LL')], ['6d'])

        out = sl.copy_dict_leaf_level(original, new)

        arr_orig = original['g'][('LL', 'LL')]['6d']
        arr_new = out['g'][('LL', 'LL')]['6d']
        np.testing.assert_array_equal(arr_orig, arr_new)
        assert not np.shares_memory(arr_orig, arr_new)

    def test_none_leaves_stay_none(self):
        original = create_cov_dict(['g'], [('LL', 'LL')], ['6d'])
        new = create_cov_dict(['g'], [('LL', 'LL')], ['6d'])

        out = sl.copy_dict_leaf_level(original, new)

        assert out['g'][('LL', 'LL')]['6d'] is None


class TestBuildCl3x2pt5d:
    """Tests for build_cl_3x2pt_5d (probe indices [0,0]=LL, [1,1]=GG, [1,0]=GL)."""

    def test_probe_block_placement(self, rng):
        nbl, zbins = 4, 3
        cl_ll = rng.standard_normal((nbl, zbins, zbins))
        cl_gl = rng.standard_normal((nbl, zbins, zbins))
        cl_gg = rng.standard_normal((nbl, zbins, zbins))

        cl_5d = sl.build_cl_3x2pt_5d(cl_ll, cl_gl, cl_gg)

        assert cl_5d.shape == (2, 2, nbl, zbins, zbins)
        np.testing.assert_array_equal(cl_5d[0, 0], cl_ll)
        np.testing.assert_array_equal(cl_5d[1, 1], cl_gg)
        np.testing.assert_array_equal(cl_5d[1, 0], cl_gl)
        np.testing.assert_array_equal(cl_5d[0, 1], cl_gl.transpose(0, 2, 1))

    def test_mismatched_shapes_raise(self, rng):
        cl_ll = rng.standard_normal((4, 3, 3))
        cl_gl = rng.standard_normal((4, 3, 3))
        cl_gg = rng.standard_normal((5, 3, 3))
        with pytest.raises(AssertionError):
            sl.build_cl_3x2pt_5d(cl_ll, cl_gl, cl_gg)

    def test_non_3d_input_raises(self, rng):
        cl_2d = rng.standard_normal((4, 3))
        with pytest.raises(AssertionError):
            sl.build_cl_3x2pt_5d(cl_2d, cl_2d, cl_2d)


# ----------------------------------------------------------------------------- #
# get_simpson_weights
# ----------------------------------------------------------------------------- #
class TestGetSimpsonWeights:
    """Tests for get_simpson_weights (Simpson quadrature weights for unit dz)."""

    @pytest.mark.parametrize('n', [5, 7, 9, 11])
    def test_matches_scipy_simpson_odd_n(self, n):
        """For an odd number of points, this is standard composite Simpson,
        which is exact (and matches scipy) for cubics."""
        x = np.linspace(0.0, 1.0, n)
        dz = x[1] - x[0]
        y = 3 * x**3 - 2 * x**2 + x + 1
        weights = sl.get_simpson_weights(n)
        estimate = np.sum(weights * y) * dz
        reference = simpson(y=y, x=x)
        assert estimate == pytest.approx(reference, rel=1e-10)

    @pytest.mark.parametrize('n', [5, 6, 7, 8, 9, 10, 11, 12])
    def test_weights_sum_to_n_minus_1(self, n):
        """Integral of the constant function 1 over a unit-spaced grid of n
        points must equal n - 1 (the total width in index units)."""
        weights = sl.get_simpson_weights(n)
        assert np.sum(weights) == pytest.approx(n - 1)

    @pytest.mark.parametrize('n', [5, 6, 7, 8, 9, 10])
    def test_weights_are_palindromic(self, n):
        weights = sl.get_simpson_weights(n)
        np.testing.assert_allclose(weights, weights[::-1])

    @pytest.mark.parametrize('n', [6, 8, 10])
    def test_exact_for_linear_even_n(self, n):
        """Even n uses a custom (non-scipy) composite rule; it should still be
        exact for a linear function, an independent property of any consistent
        (symmetric, correctly-normalized) quadrature rule."""
        x = np.linspace(0.0, 1.0, n)
        dz = x[1] - x[0]
        y = 2.0 * x + 5.0
        weights = sl.get_simpson_weights(n)
        estimate = np.sum(weights * y) * dz
        reference = simpson(y=y, x=x)
        assert estimate == pytest.approx(reference, rel=1e-8)


class TestZpairFromZidx:
    """Tests for zpair_from_zidx (locate the (i, i) auto-pair row in `ind`)."""

    @pytest.fixture
    def ind_auto(self):
        zbins = 4
        return np.array([(i, j) for i in range(zbins) for j in range(i, zbins)])

    def test_locates_diagonal_entries(self, ind_auto):
        # zbins=4 auto pairs: (0,0)(0,1)(0,2)(0,3)(1,1)(1,2)(1,3)(2,2)(2,3)(3,3)
        assert sl.zpair_from_zidx(0, ind_auto) == 0
        assert sl.zpair_from_zidx(1, ind_auto) == 4
        assert sl.zpair_from_zidx(2, ind_auto) == 7
        assert sl.zpair_from_zidx(3, ind_auto) == 9

    def test_wrong_ind_shape_raises(self, ind_auto):
        ind_bad = np.zeros((5, 3), dtype=int)
        with pytest.raises(AssertionError):
            sl.zpair_from_zidx(0, ind_bad)


# ----------------------------------------------------------------------------- #
# regularize_covariance
# ----------------------------------------------------------------------------- #
class TestRegularizeCovariance:
    """Tests for regularize_covariance (cov + lambda_reg * I)."""

    def test_eigenvalues_shifted_by_lambda(self, rng):
        n = 8
        a = rng.standard_normal((n, n))
        cov = 0.5 * (a + a.T)  # symmetric, possibly indefinite
        lambda_reg = 0.37

        reg = sl.regularize_covariance(cov, lambda_reg=lambda_reg)

        eig_orig = np.sort(np.linalg.eigvalsh(cov))
        eig_reg = np.sort(np.linalg.eigvalsh(reg))
        np.testing.assert_allclose(eig_reg, eig_orig + lambda_reg)

    def test_symmetric_input_stays_symmetric(self, rng):
        n = 5
        a = rng.standard_normal((n, n))
        cov = a + a.T
        reg = sl.regularize_covariance(cov, lambda_reg=1e-3)
        np.testing.assert_allclose(reg, reg.T)

    def test_can_fix_indefinite_matrix(self, rng):
        """A large enough lambda_reg makes an indefinite matrix SPD."""
        n = 6
        a = rng.standard_normal((n, n))
        cov = 0.5 * (a + a.T)
        min_eig = np.linalg.eigvalsh(cov).min()
        assert min_eig < 0  # sanity check that the test matrix is indefinite

        lambda_reg = -min_eig + 1.0
        reg = sl.regularize_covariance(cov, lambda_reg=lambda_reg)
        assert np.all(np.linalg.eigvalsh(reg) > 0)

    def test_default_lambda(self, rng):
        n = 4
        cov = np.eye(n)
        reg = sl.regularize_covariance(cov)
        np.testing.assert_allclose(reg, np.eye(n) * (1 + 1e-5))


# ----------------------------------------------------------------------------- #
# j0, j1, j2 (ordinary, integer-order Bessel functions J_0, J_1, J_2)
# ----------------------------------------------------------------------------- #
class TestBesselFunctions:
    """Tests for j0/j1/j2, which the source defines as jv(0/1/2, x)."""

    @pytest.fixture
    def x(self):
        return np.linspace(0.1, 20.0, 50)

    def test_known_values_at_zero(self):
        assert sl.j0(0.0) == pytest.approx(1.0)
        assert sl.j1(0.0) == pytest.approx(0.0, abs=1e-12)
        assert sl.j2(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_matches_scipy_jv(self, x):
        """Pin the exact Bessel order/kind used (ordinary J_n, not spherical j_n)."""
        np.testing.assert_allclose(sl.j0(x), jv(0, x))
        np.testing.assert_allclose(sl.j1(x), jv(1, x))
        np.testing.assert_allclose(sl.j2(x), jv(2, x))

    def test_recurrence_relation(self, x):
        """J_{n-1}(x) + J_{n+1}(x) = (2n / x) J_n(x), an independent identity
        of the ordinary (non-spherical) Bessel functions, for n = 1."""
        lhs = sl.j0(x) + sl.j2(x)
        rhs = (2.0 / x) * sl.j1(x)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# ----------------------------------------------------------------------------- #
# bin_1d_array
# ----------------------------------------------------------------------------- #
class TestBin1dArray:
    """Tests for bin_1d_array.

    Two bugs were originally pinned here with xfail and have since been
    fixed in the source: (1) leaving ``ells_eff`` at its default (None)
    crashed with ``len(None)`` in the weights-shape dispatch; (2) the
    ``'integral'`` branch normalized by ``np.sum(weights)`` instead of the
    integral ``simps(weights, x=...)`` used by ``bin_2d_array``. The tests
    below now assert the corrected behavior.
    """

    def test_default_ells_eff_works(self):
        """Calling without ells_eff (its default) must work for 1D weights."""
        ells_in = np.arange(0.0, 20.0)
        cls_in = np.full_like(ells_in, 7.0)
        ells_out = np.array([5.0, 15.0])
        ells_out_edges = np.array([0.0, 10.0, 20.0])
        out = sl.bin_1d_array(ells_in, ells_out, ells_out_edges, cls_in, None, 'sum')
        np.testing.assert_allclose(out, [7.0, 7.0])

    def test_bad_weights_shape_raises(self):
        """A weights array of neither accepted shape raises ValueError."""
        ells_in = np.arange(0.0, 20.0)
        cls_in = np.full_like(ells_in, 7.0)
        weights = np.ones((3, len(ells_in)))  # 2D but no ells_eff passed
        with pytest.raises(ValueError, match='same length'):
            sl.bin_1d_array(
                ells_in,
                np.array([5.0, 15.0]),
                np.array([0.0, 10.0, 20.0]),
                cls_in,
                weights,
                'sum',
            )

    def test_constant_input_sum_binning(self):
        """'sum' binning of a constant array returns the constant (this branch
        is not affected by the normalization bug above)."""
        ells_in = np.arange(0.0, 20.0)
        cls_in = np.full_like(ells_in, 7.0)
        ells_out = np.array([5.0, 15.0])
        ells_out_edges = np.array([0.0, 10.0, 20.0])

        out = sl.bin_1d_array(
            ells_in, ells_out, ells_out_edges, cls_in, None, 'sum', ells_eff=ells_out
        )
        np.testing.assert_allclose(out, [7.0, 7.0])

    def test_linear_input_sum_binning_matches_direct_mean(self):
        ells_in = np.arange(0.0, 20.0)
        cls_in = ells_in.copy()
        ells_out = np.array([5.0, 15.0])
        ells_out_edges = np.array([0.0, 10.0, 20.0])

        out = sl.bin_1d_array(
            ells_in, ells_out, ells_out_edges, cls_in, None, 'sum', ells_eff=ells_out
        )
        expected = [
            ells_in[(ells_in >= 0) & (ells_in < 10)].mean(),
            ells_in[(ells_in >= 10) & (ells_in < 20)].mean(),
        ]
        np.testing.assert_allclose(out, expected)

    def test_constant_input_integral_binning(self):
        """'integral' binning of a constant array returns the constant."""
        ells_in = np.arange(0.0, 20.0)
        cls_in = np.full_like(ells_in, 7.0)
        ells_out = np.array([5.0, 15.0])
        ells_out_edges = np.array([0.0, 10.0, 20.0])

        out = sl.bin_1d_array(
            ells_in,
            ells_out,
            ells_out_edges,
            cls_in,
            None,
            'integral',
            ells_eff=ells_out,
        )
        np.testing.assert_allclose(out, [7.0, 7.0])

    def test_mismatched_lengths_raise(self):
        ells_in = np.arange(0.0, 10.0)
        cls_in = np.arange(0.0, 9.0)  # wrong length
        with pytest.raises(ValueError, match='same length'):
            sl.bin_1d_array(
                ells_in, np.array([5.0]), np.array([0.0, 10.0]), cls_in, None, 'sum'
            )

    def test_ells_out_out_of_range_raises(self):
        ells_in = np.arange(0.0, 10.0)
        cls_in = ells_in.copy()
        with pytest.raises(ValueError, match='within the range'):
            sl.bin_1d_array(
                ells_in, np.array([100.0]), np.array([0.0, 10.0]), cls_in, None, 'sum'
            )


# ----------------------------------------------------------------------------- #
# bin_2d_array / bin_2d_array_vectorized
# ----------------------------------------------------------------------------- #
@pytest.fixture
def binning_setup(rng):
    """A shared random symmetric covariance + binning scheme."""
    n = 40
    ells_in = np.arange(n, dtype=float)
    a = rng.standard_normal((n, n))
    cov = a @ a.T  # symmetric
    ells_out_edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    ells_out = 0.5 * (ells_out_edges[:-1] + ells_out_edges[1:])
    return {
        'cov': cov,
        'ells_in': ells_in,
        'ells_out': ells_out,
        'edges': ells_out_edges,
    }


class TestBin2dArray:
    """Tests for bin_2d_array and its vectorized twin."""

    def test_constant_array_sum_binning(self, binning_setup):
        p = binning_setup
        cov_const = np.full_like(p['cov'], 3.14)
        out = sl.bin_2d_array(
            cov_const, p['ells_in'], p['ells_out'], p['edges'], None, 'sum', True
        )
        np.testing.assert_allclose(out, 3.14)

    def test_constant_array_integral_binning(self, binning_setup):
        p = binning_setup
        cov_const = np.full_like(p['cov'], 3.14)
        out = sl.bin_2d_array(
            cov_const, p['ells_in'], p['ells_out'], p['edges'], None, 'integral', True
        )
        np.testing.assert_allclose(out, 3.14)

    @pytest.mark.parametrize(
        'which_binning,interpolate',
        [('sum', True), ('integral', False), ('integral', True)],
    )
    def test_vectorized_matches_reference(
        self, binning_setup, which_binning, interpolate
    ):
        """The vectorized rewrite must reproduce the original implementation
        on random (non-circular w.r.t. either single implementation) input."""
        p = binning_setup
        out_ref = sl.bin_2d_array(
            p['cov'],
            p['ells_in'],
            p['ells_out'],
            p['edges'],
            None,
            which_binning,
            interpolate,
        )
        out_vec = sl.bin_2d_array_vectorized(
            p['cov'],
            p['ells_in'],
            p['ells_out'],
            p['edges'],
            None,
            which_binning,
            interpolate,
        )
        np.testing.assert_allclose(out_vec, out_ref, rtol=1e-8, atol=1e-10)

    def test_mismatched_ells_in_length_raises(self, binning_setup):
        p = binning_setup
        with pytest.raises(AssertionError):
            sl.bin_2d_array(
                p['cov'], p['ells_in'][:-1], p['ells_out'], p['edges'], None, 'sum'
            )

    def test_invalid_which_binning_raises(self, binning_setup):
        p = binning_setup
        with pytest.raises(AssertionError, match='which_binning'):
            sl.bin_2d_array(
                p['cov'], p['ells_in'], p['ells_out'], p['edges'], None, 'bogus'
            )


# ----------------------------------------------------------------------------- #
# check_interpolate_input_tab
# ----------------------------------------------------------------------------- #
class TestCheckInterpolateInputTab:
    """Tests for check_interpolate_input_tab."""

    def test_cubic_spline_reproduces_quadratic(self):
        zbins = 2
        z_in = np.linspace(0.0, 2.0, 20)
        vals = np.stack([z_in**2, 2 * z_in**2 - z_in + 1], axis=1)
        input_tab = np.column_stack([z_in, vals])

        z_out = np.linspace(0.2, 1.8, 15)
        out_tab, _ = sl.check_interpolate_input_tab(input_tab, z_out, zbins)

        expected = np.stack([z_out**2, 2 * z_out**2 - z_out + 1], axis=1)
        np.testing.assert_allclose(out_tab, expected, atol=1e-8)

    def test_linear_kind_reproduces_linear_function(self):
        zbins = 1
        z_in = np.linspace(0.0, 5.0, 10)
        vals = (3.0 * z_in + 2.0)[:, None]
        input_tab = np.column_stack([z_in, vals])

        z_out = np.linspace(0.5, 4.5, 7)
        out_tab, _ = sl.check_interpolate_input_tab(
            input_tab, z_out, zbins, kind='linear'
        )
        expected = (3.0 * z_out + 2.0)[:, None]
        np.testing.assert_allclose(out_tab, expected, atol=1e-10)

    def test_wrong_shape_raises(self):
        z_in = np.linspace(0.0, 2.0, 10)
        input_tab = np.column_stack([z_in, z_in])  # only 1 value column
        with pytest.raises(AssertionError):
            sl.check_interpolate_input_tab(input_tab, z_in, zbins=2)

    def test_invalid_kind_raises(self):
        z_in = np.linspace(0.0, 2.0, 10)
        input_tab = np.column_stack([z_in, z_in])
        with pytest.raises(ValueError, match='Unknown interpolation kind'):
            sl.check_interpolate_input_tab(input_tab, z_in, zbins=1, kind='bogus')


# ----------------------------------------------------------------------------- #
# interp_2d_arr
# ----------------------------------------------------------------------------- #
class TestInterp2dArr:
    """Tests for interp_2d_arr (bicubic-spline 2D interpolation with clipping)."""

    def test_roundtrip_accuracy_on_smooth_function(self):
        x_in = np.linspace(0.0, 2 * np.pi, 40)
        y_in = np.linspace(0.0, 2 * np.pi, 40)
        z2d_in = np.sin(x_in)[:, None] * np.cos(y_in)[None, :]

        x_out = np.linspace(0.5, 2 * np.pi - 0.5, 17)
        y_out = np.linspace(0.5, 2 * np.pi - 0.5, 17)

        x_masked, y_masked, z_interp = sl.interp_2d_arr(
            x_in, y_in, z2d_in, x_out, y_out, output_masks=False
        )
        expected = np.sin(x_masked)[:, None] * np.cos(y_masked)[None, :]
        np.testing.assert_allclose(z_interp, expected, atol=1e-3)

    def test_masks_flag_out_of_range_points(self):
        x_in = np.linspace(0.0, 10.0, 30)
        y_in = np.linspace(0.0, 10.0, 30)
        z2d_in = np.sin(x_in)[:, None] * np.cos(y_in)[None, :]

        # x_out extends beyond x_in's range
        x_out = np.linspace(-5.0, 15.0, 25)
        y_out = np.linspace(0.0, 9.0, 10)

        x_masked, y_masked, z_interp, x_mask, y_mask = sl.interp_2d_arr(
            x_in, y_in, z2d_in, x_out, y_out, output_masks=True
        )
        assert len(x_masked) < len(x_out)
        assert x_mask.sum() == len(x_masked)
        assert np.all((x_masked >= x_in.min()) & (x_masked < x_in.max()))
        assert z_interp.shape == (len(x_masked), len(y_masked))


# ----------------------------------------------------------------------------- #
# savetxt_aligned
# ----------------------------------------------------------------------------- #
class TestSavetxtAligned:
    """Tests for savetxt_aligned (write -> np.loadtxt roundtrip)."""

    def test_roundtrip(self, tmp_path, rng):
        array_2d = rng.standard_normal((6, 3))
        header_list = ['col_a', 'col_b', 'col_c']
        out_path = tmp_path / 'aligned.txt'

        sl.savetxt_aligned(str(out_path), array_2d, header_list)
        reloaded = np.loadtxt(out_path)

        np.testing.assert_allclose(reloaded, array_2d, atol=1e-7)

    def test_header_is_commented(self, tmp_path, rng):
        array_2d = rng.standard_normal((3, 2))
        out_path = tmp_path / 'aligned2.txt'
        sl.savetxt_aligned(str(out_path), array_2d, ['a', 'b'])

        with open(out_path) as f:
            first_line = f.readline()
        assert first_line.startswith('#')


# ----------------------------------------------------------------------------- #
# validate_cov_dict_structure / symmetrize_probe_cov_dict_6d
# ----------------------------------------------------------------------------- #
class TestValidateCovDictStructure:
    """Tests for validate_cov_dict_structure, using create_cov_dict (from
    spaceborne.cov_dict) to build minimal, cheap, valid dicts."""

    def test_valid_structure_passes(self):
        cov_dict = create_cov_dict(['g'], [('LL', 'LL')], ['6d'])
        cov_dict['g'][('LL', 'LL')]['6d'] = np.zeros((2, 2, 3, 3, 3, 3))
        sl.validate_cov_dict_structure(cov_dict, 'harmonic')  # should not raise

    def test_none_leaf_value_raises(self):
        """create_cov_dict pre-fills leaves with None; leaving one unset must
        be rejected by the validator (it requires actual ndarrays)."""
        cov_dict = create_cov_dict(['g'], [('LL', 'LL')], ['6d', '4d'])
        cov_dict['g'][('LL', 'LL')]['6d'] = np.zeros((2, 2, 3, 3, 3, 3))
        with pytest.raises(ValueError, match='must be a numpy array'):
            sl.validate_cov_dict_structure(cov_dict, 'harmonic')

    def test_unexpected_probe_raises(self):
        bad = {'g': {('XX', 'LL'): {'6d': np.zeros((2, 2, 3, 3, 3, 3))}}}
        with pytest.raises(ValueError, match='Unexpected probe_ab'):
            sl.validate_cov_dict_structure(bad, 'harmonic')

    def test_invalid_obs_space_raises(self):
        cov_dict = create_cov_dict(['g'], [('LL', 'LL')], ['6d'])
        cov_dict['g'][('LL', 'LL')]['6d'] = np.zeros((2, 2, 3, 3, 3, 3))
        with pytest.raises(ValueError, match='obs_space'):
            sl.validate_cov_dict_structure(cov_dict, 'bogus_space')

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match='must be a dictionary'):
            sl.validate_cov_dict_structure([], 'harmonic')


class TestSymmetrizeProbeCovDict6d:
    """Tests for symmetrize_probe_cov_dict_6d.

    create_cov_dict's FrozenDict forbids adding *new* probe keys (e.g. adding
    ('GG', 'LL') when only ('LL', 'GG') was pre-registered), which is exactly
    what this function needs to do -- so, per the task instructions, plain
    dicts (mirroring the same cov_dict[term][probe_ab, probe_cd]['6d'] shape)
    are used here instead of create_cov_dict.
    """

    def test_fills_symmetric_partner(self, rng):
        arr_lg = rng.standard_normal((2, 2, 3, 3, 3, 3))
        arr_gg = rng.standard_normal((2, 2, 3, 3, 3, 3))
        cov_dict = {'g': {('LL', 'GG'): {'6d': arr_lg}, ('GG', 'GG'): {'6d': arr_gg}}}

        out = sl.symmetrize_probe_cov_dict_6d(cov_dict)

        assert ('GG', 'LL') in out['g']
        expected = arr_lg.transpose(1, 0, 4, 5, 2, 3)
        np.testing.assert_allclose(out['g'][('GG', 'LL')]['6d'], expected)
        # auto-correlation block is untouched, no spurious keys added
        assert set(out['g'].keys()) == {('LL', 'GG'), ('GG', 'GG'), ('GG', 'LL')}

    def test_does_not_overwrite_existing_symmetric_partner(self, rng):
        arr_lg = rng.standard_normal((2, 2, 3, 3, 3, 3))
        arr_gl = rng.standard_normal((2, 2, 3, 3, 3, 3))
        cov_dict = {'g': {('LL', 'GG'): {'6d': arr_lg}, ('GG', 'LL'): {'6d': arr_gl}}}
        out = sl.symmetrize_probe_cov_dict_6d(cov_dict)
        np.testing.assert_array_equal(out['g'][('GG', 'LL')]['6d'], arr_gl)

    def test_ignores_3x2pt_key(self, rng):
        arr = rng.standard_normal((2, 2, 3, 3, 3, 3))
        cov_dict = {'g': {'3x2pt': {'2d': np.zeros((5, 5))}, ('LL', 'LL'): {'6d': arr}}}
        out = sl.symmetrize_probe_cov_dict_6d(cov_dict)
        assert '3x2pt' in out['g']
        assert set(out['g'].keys()) == {'3x2pt', ('LL', 'LL')}

    def test_non_tuple_key_raises(self):
        cov_dict = {'g': {'not_a_tuple': {'6d': np.zeros((2, 2, 3, 3, 3, 3))}}}
        with pytest.raises(ValueError, match='Expected 2-tuple key'):
            sl.symmetrize_probe_cov_dict_6d(cov_dict)
