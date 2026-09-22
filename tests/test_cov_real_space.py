r"""Unit tests for the pure numerical helpers in spaceborne.cov_real_space.

This module only exercises functions that do not require CCL or the full
CovRealSpace pipeline:

* ``b_mu`` and ``k_mu`` -- the closed-form antiderivatives of x J_mu(x) and the
  resulting bin-averaged kernel. Checked against their defining integrals,
  computed numerically, for the three supported orders mu in {0, 2, 4}:
      b_mu(x) = \int_0^x x' J_mu(x') dx'   (up to a constant)
      K_mu(ell; theta_l, theta_u) = 2 / (theta_u^2 - theta_l^2)
                                    \int_{theta_l}^{theta_u} theta J_mu(ell theta)
* ``t_sn`` -- checked case by case against explicitly constructed expected
  matrices. In particular, the mixed source/lens case (e.g. gt/gt) is pinned
  as a regression test: the source-bin variance sigma_eps_i**2 must be
  broadcast over *all* (lens, source) tomographic pairs, not only the
  diagonal. The tomographic Kronecker deltas are applied separately in
  ``cov_sn_rs`` (via ``get_delta_tomo``), and the harmonic-space analog
  Cov_SN(C^GL_ij, C^GL_kl) = delta_ik delta_jl N^gg_i N^ee_j is nonzero for
  i != j (same convention as OneCovariance's gmgm shot-noise term). A
  diagonal-only variant (the former ``_t_sn``) was confirmed wrong and
  removed in 2026-07.

"""

import itertools

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

from spaceborne.cov_real_space import b_mu, k_mu, t_sn


def _is_mixed_mixed(combo):
    """Both pairs are one source + one lens, e.g. probe combo (GL, GL)."""
    a, b, c, d = combo
    return {a, b} == {0, 1} and {c, d} == {0, 1}


_ALL_PROBE_COMBOS = list(itertools.product((0, 1), repeat=4))
_MIXED_COMBOS = [c for c in _ALL_PROBE_COMBOS if _is_mixed_mixed(c)]
_NON_MIXED_COMBOS = [c for c in _ALL_PROBE_COMBOS if not _is_mixed_mixed(c)]


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=2024)


MU_VALUES = (0, 2, 4)


# ----------------------------------------------------------------------------- #
# b_mu
# ----------------------------------------------------------------------------- #
class TestBMu:
    """b_mu is an antiderivative of x J_mu(x)."""

    @pytest.mark.parametrize('mu', MU_VALUES)
    def test_is_antiderivative_of_x_jmu(self, mu, rng):
        x1, x2 = np.sort(rng.uniform(0.1, 20.0, 2))
        expected, _ = quad(lambda x: x * jv(mu, x), x1, x2, epsabs=0, epsrel=1e-12)
        np.testing.assert_allclose(b_mu(x2, mu) - b_mu(x1, mu), expected, rtol=1e-10)

    def test_invalid_mu_raises(self):
        with pytest.raises(ValueError, match='mu must be one of'):
            b_mu(1.0, mu=1)


# ----------------------------------------------------------------------------- #
# k_mu
# ----------------------------------------------------------------------------- #
class TestKMu:
    """k_mu is the bin average of J_mu(ell theta), with weight theta."""

    @pytest.mark.parametrize('mu', MU_VALUES)
    def test_matches_bin_averaged_bessel(self, mu, rng):
        ells = rng.uniform(10.0, 5000.0, 6)
        thetal_arr = rng.uniform(1e-3, 1e-2, 6)
        thetau_arr = thetal_arr + rng.uniform(1e-3, 1e-2, 6)

        for ell, thetal, thetau in zip(ells, thetal_arr, thetau_arr, strict=True):
            integral, _ = quad(
                lambda t, ell=ell: t * jv(mu, ell * t),
                thetal,
                thetau,
                epsabs=0,
                epsrel=1e-12,
                limit=200,
            )
            expected = 2.0 / (thetau**2 - thetal**2) * integral
            direct = k_mu(ell, thetal=thetal, thetau=thetau, mu=mu)
            np.testing.assert_allclose(direct, expected, rtol=1e-9)

    @pytest.mark.parametrize('mu', MU_VALUES)
    def test_vectorised_in_ell(self, mu, rng):
        """An array of ells gives the same values as one ell at a time."""
        ells = rng.uniform(10.0, 5000.0, 7)
        vec = k_mu(ells, thetal=0.002, thetau=0.004, mu=mu)
        loop = [k_mu(ell, thetal=0.002, thetau=0.004, mu=mu) for ell in ells]
        np.testing.assert_array_equal(vec, loop)


# ----------------------------------------------------------------------------- #
# t_sn
# ----------------------------------------------------------------------------- #
class TestTSn:
    """t_sn checked against explicitly constructed expected matrices, for all
    16 probe index combinations. probe_ix 0 = source (shear), 1 = lens
    (clustering)."""

    @pytest.fixture
    def sigma_eps_i(self, rng):
        zbins = 4
        return rng.uniform(0.1, 0.5, zbins)

    @pytest.mark.parametrize('combo', _MIXED_COMBOS)
    def test_mixed_case_broadcasts_source_variance(self, combo, sigma_eps_i):
        """Regression test (2026-07 review): for the mixed source/lens case
        (e.g. gt/gt), the source-bin variance must be broadcast over all
        (zbins, zbins) tomographic pairs of the first probe pair -- shape
        noise is present for lens != source bins too. The tomographic
        Kronecker deltas are applied separately in cov_sn_rs, so a
        diagonal-only t_sn (the former _t_sn) would wrongly zero the shape
        noise for every cross lens-source gt pair."""
        a, b, c, d = combo
        zbins = sigma_eps_i.size
        sig2 = sigma_eps_i**2
        out = t_sn(a, b, c, d, zbins, sigma_eps_i)

        # the source index within the first pair (ij) is i if a is the
        # source (a == 0), j otherwise
        if a == 0:
            expected = np.tile(sig2[:, None], (1, zbins))
        else:
            expected = np.tile(sig2[None, :], (zbins, 1))

        np.testing.assert_allclose(out, expected)
        # explicitly pin the off-diagonal behavior
        assert np.all(out != 0.0)

    @pytest.mark.parametrize('combo', [c for c in _NON_MIXED_COMBOS if len(set(c)) > 1])
    def test_probe_type_mismatch_is_zero(self, combo, sigma_eps_i):
        """Every non-mixed combo other than all-source/all-lens (e.g. one
        pure-source pair with one pure-lens pair) contributes no shot/shape
        noise."""
        a, b, c, d = combo
        zbins = sigma_eps_i.size
        out = t_sn(a, b, c, d, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, np.zeros((zbins, zbins)))

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
