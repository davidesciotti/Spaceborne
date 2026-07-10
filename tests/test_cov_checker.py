"""Unit tests for spaceborne.cov_checker.CovarianceMatrixChecker.

The checker runs a battery of structural/numerical sanity checks and stores
per-check dicts (with a ``'valid'`` bool and a human-readable ``'message'``)
in ``self.results``; ``is_valid_covariance_matrix()`` then ANDs together a
subset of "essential" checks (shape, square, symmetry, positive_definite,
finite_values).

We build small hand-crafted matrices and drive the checker through
``check_all`` (with ``verbose=False`` to keep test output quiet -- note the
per-check methods still ``print()`` progress messages regardless of
``verbose``; only the final summary table is silenced), then inspect
``self.results`` for the specific check that should catch each defect:

* a random SPD matrix passes every essential check;
* breaking symmetry is caught by the ``symmetry`` check (and propagates to
  ``is_valid_covariance_matrix() is False``);
* a symmetric but indefinite matrix is caught by ``positive_definite``;
* NaNs are caught by ``finite_values``.
"""

import numpy as np
import pytest

from spaceborne.cov_checker import CovarianceMatrixChecker


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=2024)


@pytest.fixture
def spd_matrix(rng):
    """A small, well-conditioned symmetric positive-definite matrix."""
    a = rng.standard_normal((5, 5))
    return a @ a.T + 5 * np.eye(5)


class TestValidCovariance:
    """A genuine SPD matrix should pass every essential check."""

    def test_is_valid_covariance_matrix(self, spd_matrix):
        checker = CovarianceMatrixChecker()
        checker.check_all(spd_matrix, verbose=False)
        assert checker.is_valid_covariance_matrix()

    def test_individual_checks_pass(self, spd_matrix):
        checker = CovarianceMatrixChecker()
        results = checker.check_all(spd_matrix, verbose=False)
        essential = (
            'shape', 'square', 'symmetry', 'positive_definite', 'finite_values'
        )  # fmt: skip
        for check in essential:
            assert results[check]['valid']

    def test_cholesky_and_invertible_also_pass(self, spd_matrix):
        checker = CovarianceMatrixChecker()
        results = checker.check_all(spd_matrix, verbose=False)
        assert results['cholesky']['valid']
        assert results['invertible']['valid']
        assert results['diagonal']['valid']


class TestNonSymmetric:
    """Breaking symmetry should be flagged by the symmetry check."""

    def test_symmetry_check_fails(self, spd_matrix):
        broken = spd_matrix.copy()
        broken[0, 1] += 5.0

        checker = CovarianceMatrixChecker()
        results = checker.check_all(broken, verbose=False)

        assert not results['symmetry']['valid']
        assert results['symmetry']['max_asymmetry'] > 0

    def test_overall_invalid(self, spd_matrix):
        broken = spd_matrix.copy()
        broken[0, 1] += 5.0

        checker = CovarianceMatrixChecker()
        checker.check_all(broken, verbose=False)
        assert not checker.is_valid_covariance_matrix()


class TestNonPositiveDefinite:
    """A symmetric but indefinite matrix should fail the PD check."""

    def test_positive_definite_check_fails(self):
        indefinite = np.diag([-1.0, 2.0, 3.0, 4.0])

        checker = CovarianceMatrixChecker()
        results = checker.check_all(indefinite, verbose=False)

        assert not results['positive_definite']['valid']
        assert results['positive_definite']['min_eigenvalue'] < 0

    def test_overall_invalid(self):
        indefinite = np.diag([-1.0, 2.0, 3.0, 4.0])

        checker = CovarianceMatrixChecker()
        checker.check_all(indefinite, verbose=False)
        assert not checker.is_valid_covariance_matrix()

    def test_cholesky_also_fails(self):
        indefinite = np.diag([-1.0, 2.0, 3.0, 4.0])

        checker = CovarianceMatrixChecker()
        results = checker.check_all(indefinite, verbose=False)
        assert not results['cholesky']['valid']


class TestNanValues:
    """NaN-containing matrices should be flagged by the finite-values check."""

    def test_finite_values_check_fails(self, spd_matrix):
        with_nan = spd_matrix.copy()
        with_nan[0, 0] = np.nan

        checker = CovarianceMatrixChecker()
        results = checker.check_all(with_nan, verbose=False)

        assert not results['finite_values']['valid']
        assert results['finite_values']['has_nan']
        assert not results['finite_values']['has_inf']

    def test_overall_invalid(self, spd_matrix):
        with_nan = spd_matrix.copy()
        with_nan[0, 0] = np.nan

        checker = CovarianceMatrixChecker()
        checker.check_all(with_nan, verbose=False)
        assert not checker.is_valid_covariance_matrix()


class TestNonSquareAndWrongShape:
    """Structural checks should catch non-square / non-2D inputs."""

    def test_non_square_matrix_fails_square_check(self):
        rect = np.ones((3, 4))
        checker = CovarianceMatrixChecker()
        results = checker.check_all(rect, verbose=False)
        assert not results['square']['valid']

    def test_1d_array_fails_shape_check(self):
        vec = np.ones(5)
        checker = CovarianceMatrixChecker()
        results = checker.check_all(vec, verbose=False)
        assert not results['shape']['valid']


class TestResultsResetBetweenCalls:
    """check_all should reset cached state so a second call is independent."""

    def test_second_call_with_different_matrix_updates_results(self, spd_matrix):
        checker = CovarianceMatrixChecker()
        checker.check_all(spd_matrix, verbose=False)
        assert checker.is_valid_covariance_matrix()

        indefinite = np.diag([-1.0, 2.0, 3.0])
        checker.check_all(indefinite, verbose=False)
        assert not checker.is_valid_covariance_matrix()
