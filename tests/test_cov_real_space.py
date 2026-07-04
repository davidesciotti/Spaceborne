"""Unit tests for the pure numerical helpers in spaceborne.cov_real_space.

This module only exercises functions that do not require pylevin, CCL, or the
full CovRealSpace pipeline:

* ``b_mu`` / ``b_mu_nobessel`` and ``k_mu`` / ``k_mu_nobessel`` -- the
  ``_nobessel`` variants decompose the closed-form Bessel expressions into
  ``(coefficient, bessel_order[, theta])`` terms (used to build Levin
  integrands). We reconstruct the Bessel sums from the decomposed terms with
  ``scipy.special.jv`` and check they reproduce the direct (bessel-evaluated)
  functions, for the three supported multipole orders mu in {0, 2, 4}.
* ``kmuknu_nobessel`` -- the product of two decomposed kernels; checked the
  same way, by reconstructing K_mu(ell1) * K_nu(ell2) from the product terms.
* ``t_sn`` vs ``_t_sn`` -- ``_t_sn`` is an explicit nested-loop reference
  implementation of ``t_sn`` living in the same module. They agree for the
  "pure" cases (all-source, all-lens, incompatible probe types) but *disagree*
  for the mixed source/lens case: ``t_sn`` returns an outer-product-like
  (zbins, zbins) matrix while ``_t_sn`` only fills the diagonal. This looks
  like a genuine inconsistency between the production and reference
  implementations, so the mixed-case comparison is marked ``xfail`` rather
  than silently adjusted (see ``TestTSn.test_mixed_case_matches_reference``).
* ``t_mix`` / ``split_probe_ix`` -- small standalone helpers, checked directly.
* ``regularize_by_eigenvalue_cutoff`` -- despite the name, reading the source
  shows it returns the *inverse* of ``cov`` with small eigenvalues clipped to
  zero contribution (a truncated pseudo-inverse), not a regularized version of
  ``cov`` itself. Tests pin down this actual behaviour: it agrees with
  ``np.linalg.inv`` for a well-conditioned SPD matrix, small eigenvalues get
  zero inverse-weight, and symmetric input stays symmetric.

Functions requiring pylevin (``integrate_bessel_single_wrapper``,
``dl1dl2_binavg_bessel_wrapper``, ``dl1dl2_nobinavg_bessel_wrapper``,
``levin_integrate_bessel_double_wrapper``, ``integrate_single_bessel_pair``),
CCL (``twopcf_wrapper``), or the full ``CovRealSpace``/``proj_cov_2d_fftlog``
pipeline are out of scope for this module.
"""

import itertools

import numpy as np
import pytest
from scipy.special import jv

from spaceborne.cov_real_space import (
    _t_sn,
    b_mu,
    b_mu_nobessel,
    k_mu,
    k_mu_nobessel,
    kmuknu_nobessel,
    regularize_by_eigenvalue_cutoff,
    split_probe_ix,
    t_mix,
    t_sn,
)


def _is_mixed_mixed(combo):
    """Both pairs are one source + one lens, e.g. probe combo (GL, GL)."""
    a, b, c, d = combo
    return {a, b} == {0, 1} and {c, d} == {0, 1}


_ALL_PROBE_COMBOS = list(itertools.product((0, 1), repeat=4))
_NON_MIXED_COMBOS = [c for c in _ALL_PROBE_COMBOS if not _is_mixed_mixed(c)]


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=2024)


MU_VALUES = (0, 2, 4)


# ----------------------------------------------------------------------------- #
# b_mu / b_mu_nobessel
# ----------------------------------------------------------------------------- #
class TestBMu:
    """b_mu_nobessel decomposes b_mu into explicit (coeff, bessel_order) terms."""

    @pytest.mark.parametrize('mu', MU_VALUES)
    def test_nobessel_reconstructs_b_mu(self, mu, rng):
        xs = rng.uniform(0.1, 20.0, 8)
        for x in xs:
            direct = b_mu(x, mu)
            terms = b_mu_nobessel(x, mu)
            recon = sum(coeff * jv(order, x) for coeff, order in terms)
            np.testing.assert_allclose(recon, direct, rtol=1e-10)

    def test_invalid_mu_raises(self):
        with pytest.raises(ValueError, match='mu must be one of'):
            b_mu(1.0, mu=1)

    def test_nobessel_invalid_mu_raises(self):
        with pytest.raises(ValueError, match='mu must be one of'):
            b_mu_nobessel(1.0, mu=3)


# ----------------------------------------------------------------------------- #
# k_mu / k_mu_nobessel
# ----------------------------------------------------------------------------- #
class TestKMu:
    """k_mu_nobessel decomposes k_mu into (coeff, bessel_order, theta) terms."""

    @pytest.mark.parametrize('mu', MU_VALUES)
    def test_nobessel_reconstructs_k_mu(self, mu, rng):
        ells = rng.uniform(10.0, 5000.0, 6)
        thetal_arr = rng.uniform(1e-3, 1e-2, 6)
        thetau_arr = thetal_arr + rng.uniform(1e-3, 1e-2, 6)

        for ell, thetal, thetau in zip(ells, thetal_arr, thetau_arr, strict=True):
            direct = k_mu(ell, thetal=thetal, thetau=thetau, mu=mu)
            terms = k_mu_nobessel(ell, thetal=thetal, thetau=thetau, mu=mu)
            recon = sum(coeff * jv(order, ell * theta) for coeff, order, theta in terms)
            np.testing.assert_allclose(recon, direct, rtol=1e-10)


# ----------------------------------------------------------------------------- #
# kmuknu_nobessel
# ----------------------------------------------------------------------------- #
class TestKMuKNuNobessel:
    """kmuknu_nobessel expands the product K_mu(ell1) * K_nu(ell2)."""

    @pytest.mark.parametrize('mu', MU_VALUES)
    @pytest.mark.parametrize('nu', MU_VALUES)
    def test_product_matches_direct(self, mu, nu, rng):
        ell1, ell2 = rng.uniform(10.0, 5000.0, 2)
        thetal1, thetau1 = 0.001, 0.003
        thetal2, thetau2 = 0.002, 0.005

        k_mu_terms = k_mu_nobessel(ell1, thetal=thetal1, thetau=thetau1, mu=mu)
        k_nu_terms = k_mu_nobessel(ell2, thetal=thetal2, thetau=thetau2, mu=nu)
        product_terms = kmuknu_nobessel(k_mu_terms, k_nu_terms)

        direct = k_mu(ell1, thetal=thetal1, thetau=thetau1, mu=mu) * k_mu(
            ell2, thetal=thetal2, thetau=thetau2, mu=nu
        )
        recon = sum(
            coeff * jv(n1, ell1 * t1) * jv(n2, ell2 * t2)
            for coeff, n1, t1, n2, t2 in product_terms
        )
        np.testing.assert_allclose(recon, direct, rtol=1e-10)

    def test_number_of_terms_is_product(self):
        """kmuknu_nobessel returns the cartesian product of the input term lists."""
        k_mu_terms = k_mu_nobessel(100.0, thetal=0.001, thetau=0.002, mu=4)
        k_nu_terms = k_mu_nobessel(200.0, thetal=0.001, thetau=0.002, mu=2)
        product_terms = kmuknu_nobessel(k_mu_terms, k_nu_terms)
        assert len(product_terms) == len(k_mu_terms) * len(k_nu_terms)


# ----------------------------------------------------------------------------- #
# t_sn vs _t_sn
# ----------------------------------------------------------------------------- #
class TestTSn:
    """t_sn (production) vs _t_sn (nested-loop reference), for all 16 probe
    index combinations. probe_ix 0 = source (shear), 1 = lens (clustering)."""

    @pytest.fixture
    def sigma_eps_i(self, rng):
        zbins = 4
        return rng.uniform(0.1, 0.5, zbins)

    @pytest.mark.parametrize('combo', _NON_MIXED_COMBOS)
    def test_matches_reference_non_mixed(self, combo, sigma_eps_i):
        """For all-source, all-lens, and probe-type-mismatched combos, the
        production and reference implementations agree exactly."""
        a, b, c, d = combo
        zbins = sigma_eps_i.size
        out = t_sn(a, b, c, d, zbins, sigma_eps_i)
        ref = _t_sn(a, b, c, d, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, ref)

    @pytest.mark.xfail(
        reason=(
            't_sn and _t_sn disagree in the mixed source/lens case (e.g. '
            'probe_ix combo (0,1,0,1), GL/GL): t_sn broadcasts the source '
            'variance sigma_eps_i[i]**2 across the full (zbins, zbins) '
            'output (an outer product with a ones vector), while the '
            'nested-loop reference _t_sn only ever fills the diagonal '
            't_munu[zi, zi], leaving all off-diagonal entries at 0. This '
            'looks like a genuine inconsistency between the production '
            'function and its own reference implementation.'
        ),
        strict=True,
    )
    def test_mixed_case_matches_reference(self, sigma_eps_i):
        zbins = sigma_eps_i.size
        out = t_sn(0, 1, 0, 1, zbins, sigma_eps_i)
        ref = _t_sn(0, 1, 0, 1, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, ref)

    def test_all_source_formula(self, sigma_eps_i):
        """xipxip/ximxim case: tau(i,j) = 2 * sig2[i] * sig2[j]."""
        zbins = sigma_eps_i.size
        out = t_sn(0, 0, 0, 0, zbins, sigma_eps_i)
        sig2 = sigma_eps_i**2
        expected = 2.0 * np.outer(sig2, sig2)
        np.testing.assert_allclose(out, expected)

    def test_all_lens_is_ones(self, sigma_eps_i):
        """gggg case: tau(i,j) = 1 for all i, j."""
        zbins = sigma_eps_i.size
        out = t_sn(1, 1, 1, 1, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, np.ones((zbins, zbins)))

    def test_incompatible_types_are_zero(self, sigma_eps_i):
        """A pure-source pair combined with a pure-lens pair contributes 0."""
        zbins = sigma_eps_i.size
        out = t_sn(0, 0, 1, 1, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, np.zeros((zbins, zbins)))


# ----------------------------------------------------------------------------- #
# t_mix
# ----------------------------------------------------------------------------- #
class TestTMix:
    def test_source_case(self, rng):
        zbins = 5
        sigma_eps_i = rng.uniform(0.1, 0.5, zbins)
        out = t_mix(0, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, sigma_eps_i**2)

    def test_lens_case(self, rng):
        zbins = 5
        sigma_eps_i = rng.uniform(0.1, 0.5, zbins)
        out = t_mix(1, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, np.ones(zbins))


# ----------------------------------------------------------------------------- #
# split_probe_ix
# ----------------------------------------------------------------------------- #
class TestSplitProbeIx:
    @pytest.mark.parametrize(
        'probe_ix,expected', [(0, (0, 0)), (1, (0, 0)), (2, (1, 0)), (3, (1, 1))]
    )
    def test_valid_indices(self, probe_ix, expected):
        assert split_probe_ix(probe_ix) == expected

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError, match='Invalid probe index'):
            split_probe_ix(4)


# ----------------------------------------------------------------------------- #
# regularize_by_eigenvalue_cutoff
# ----------------------------------------------------------------------------- #
class TestRegularizeByEigenvalueCutoff:
    """Despite its name, this function returns the eigenvalue-truncated
    (pseudo-)inverse of cov, not a regularized version of cov itself -- see
    the module docstring above."""

    def test_matches_inverse_for_well_conditioned_spd(self, rng):
        a = rng.standard_normal((5, 5))
        spd = a @ a.T + 5.0 * np.eye(5)  # well-conditioned, no small eigenvalues
        out = regularize_by_eigenvalue_cutoff(spd, threshold=1e-14)
        np.testing.assert_allclose(out, np.linalg.inv(spd), rtol=1e-8)

    def test_symmetric_input_stays_symmetric(self, rng):
        a = rng.standard_normal((6, 6))
        sym = 0.5 * (a + a.T)
        out = regularize_by_eigenvalue_cutoff(sym, threshold=1e-14)
        np.testing.assert_allclose(out, out.T)

    def test_small_eigenvalues_get_zero_weight(self, rng):
        """Eigen-directions with eigenvalue below the threshold contribute 0
        to the reconstructed (pseudo-)inverse, instead of blowing up."""
        q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        true_eigvals = np.array([1e-16, 1e-15, 2.0, 5.0])
        cov = q @ np.diag(true_eigvals) @ q.T

        out = regularize_by_eigenvalue_cutoff(cov, threshold=1e-14)
        expected = q @ np.diag([0.0, 0.0, 0.5, 0.2]) @ q.T
        np.testing.assert_allclose(out, expected, atol=1e-8)

    def test_no_cutoff_below_all_eigenvalues(self, rng):
        """With a threshold below every eigenvalue, this is a plain inverse."""
        a = rng.standard_normal((4, 4))
        spd = a @ a.T + 2.0 * np.eye(4)
        out = regularize_by_eigenvalue_cutoff(spd, threshold=-np.inf)
        np.testing.assert_allclose(out, np.linalg.inv(spd), rtol=1e-8)
