"""Unit tests for the module-level pure helpers in spaceborne.cov_projector.

These are the statistic-agnostic building blocks shared by the real-space
and COSEBIs Gaussian-covariance projections:

* ``get_npair`` / ``get_dnpair`` -- the (ideal) pair-count normal is a
  textbook area integral, N(theta) = pi (theta_u^2 - theta_l^2) * A * n_i *
  n_j; we check the closed form directly and cross-check it against a
  Simpson integral of the differential dN/dtheta over the same annulus.
* ``t_mix`` -- the per-probe MIX-term variance factor (shape noise squared
  for shear, unity for clustering).
* ``get_delta_tomo`` -- Kronecker delta in tomographic bin space, identity
  for a probe with itself and zero across different probes.
* ``build_cov_sva_integrand_5d`` -- the universal Gaussian SVA integrand
  Cov[C_ab, C_cd] ~ C_ac C_bd + C_ad C_bc; checked against explicit index
  arithmetic on a small random C_ell array (using the [0,0]=LL, [1,1]=GG,
  [1,0]=GL probe-index convention from CLAUDE.md).

``proj_cov_2d`` and the ``CovarianceProjector`` class need a full pipeline
config and are intentionally not covered here.
"""

import numpy as np
import pytest
from scipy.integrate import simpson as simps

from spaceborne import cov_projector as cp


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=7)


# ----------------------------------------------------------------------------- #
# get_npair / get_dnpair
# ----------------------------------------------------------------------------- #
class TestGetNpair:
    """Tests for the ideal (analytic) pair-count in a theta annulus."""

    def test_matches_analytic_formula(self):
        theta_l, theta_u = 1.0, 2.0
        survey_area_sr = 0.5
        n_eff_i, n_eff_j = 3.0, 5.0

        out = cp.get_npair(theta_u, theta_l, survey_area_sr, n_eff_i, n_eff_j)

        from spaceborne import constants as const

        n_i_sr = n_eff_i * const.SR_TO_ARCMIN2
        n_j_sr = n_eff_j * const.SR_TO_ARCMIN2
        expected = np.pi * (theta_u**2 - theta_l**2) * survey_area_sr * n_i_sr * n_j_sr
        np.testing.assert_allclose(out, expected)

    def test_zero_width_annulus_gives_zero_pairs(self):
        out = cp.get_npair(1.0, 1.0, 0.5, 3.0, 5.0)
        assert out == 0.0

    def test_scales_linearly_with_area(self):
        base = cp.get_npair(2.0, 1.0, 1.0, 3.0, 5.0)
        scaled = cp.get_npair(2.0, 1.0, 2.5, 3.0, 5.0)
        np.testing.assert_allclose(scaled, 2.5 * base)


class TestGetDnpair:
    """Tests for the differential (ideal) pair-count dN(theta)/dtheta."""

    def test_integrates_to_get_npair(self):
        """dnpair integrated over [theta_l, theta_u] reproduces get_npair."""
        theta_l, theta_u = 1.0, 2.0
        survey_area_sr = 0.5
        n_eff_i, n_eff_j = 3.0, 5.0

        npair = cp.get_npair(theta_u, theta_l, survey_area_sr, n_eff_i, n_eff_j)

        theta = np.linspace(theta_l, theta_u, 20_000)
        dnpair = cp.get_dnpair(theta, survey_area_sr, n_eff_i, n_eff_j)
        integral = simps(y=dnpair, x=theta)

        np.testing.assert_allclose(integral, npair, rtol=1e-8)

    def test_zero_at_theta_zero(self):
        out = cp.get_dnpair(0.0, 0.5, 3.0, 5.0)
        assert out == 0.0

    def test_positive_for_positive_theta(self):
        theta = np.array([0.1, 1.0, 10.0])
        out = cp.get_dnpair(theta, 0.5, 3.0, 5.0)
        assert np.all(out > 0)


# ----------------------------------------------------------------------------- #
# t_mix
# ----------------------------------------------------------------------------- #
class TestTMix:
    """Tests for the MIX-term per-probe variance factor."""

    def test_shear_probe_uses_sigma_eps_squared(self):
        zbins = 4
        sigma_eps_i = np.array([0.1, 0.2, 0.3, 0.4])
        out = cp.t_mix(0, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, sigma_eps_i**2)

    def test_clustering_probe_is_ones(self):
        zbins = 4
        sigma_eps_i = np.array([0.1, 0.2, 0.3, 0.4])
        out = cp.t_mix(1, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, np.ones(zbins))

    def test_unknown_probe_index_is_zero(self):
        """Neither branch matches, so the pre-allocated zeros are returned."""
        zbins = 4
        sigma_eps_i = np.array([0.1, 0.2, 0.3, 0.4])
        out = cp.t_mix(2, zbins, sigma_eps_i)
        np.testing.assert_allclose(out, np.zeros(zbins))


# ----------------------------------------------------------------------------- #
# get_delta_tomo
# ----------------------------------------------------------------------------- #
class TestGetDeltaTomo:
    """Tests for the tomographic-bin Kronecker delta."""

    def test_auto_probe_is_identity(self):
        for probe_ix in (0, 1):
            out = cp.get_delta_tomo(probe_ix, probe_ix, 3)
            np.testing.assert_array_equal(out, np.eye(3))

    def test_cross_probe_is_zero(self):
        out = cp.get_delta_tomo(0, 1, 3)
        np.testing.assert_array_equal(out, np.zeros((3, 3)))

    def test_shape(self):
        out = cp.get_delta_tomo(0, 0, 5)
        assert out.shape == (5, 5)


# ----------------------------------------------------------------------------- #
# build_cov_sva_integrand_5d
# ----------------------------------------------------------------------------- #
class TestBuildCovSvaIntegrand5d:
    """Tests for the universal Gaussian SVA harmonic-space integrand."""

    @pytest.fixture
    def cl_5d(self, rng):
        """Random Cl array of shape (n_probes, n_probes, n_ell, zbins, zbins).

        Probe index convention (see CLAUDE.md): 0=shear/LL, 1=clustering/GG,
        cl_5d[1, 0]=GL. The array need not be physically symmetric for this
        purely-algebraic test.
        """
        n_probes, n_ell, zbins = 2, 3, 4
        return rng.standard_normal((n_probes, n_probes, n_ell, zbins, zbins))

    def test_shape(self, cl_5d):
        n_ell, zbins = cl_5d.shape[2], cl_5d.shape[3]
        out = cp.build_cov_sva_integrand_5d(cl_5d, 0, 0, 0, 0)
        assert out.shape == (n_ell, zbins, zbins, zbins, zbins)

    def test_matches_explicit_index_arithmetic_ll_only(self, cl_5d):
        """Pure LL (probe index 0 everywhere): Cov ~ C_ik C_jl + C_il C_jk."""
        out = cp.build_cov_sva_integrand_5d(cl_5d, 0, 0, 0, 0)
        ell, i, j, k, l = 1, 2, 0, 3, 1
        expected = (
            cl_5d[0, 0, ell, i, k] * cl_5d[0, 0, ell, j, l]
            + cl_5d[0, 0, ell, i, l] * cl_5d[0, 0, ell, j, k]
        )
        np.testing.assert_allclose(out[ell, i, j, k, l], expected)

    def test_matches_explicit_index_arithmetic_mixed_probes(self, cl_5d):
        """probe_a=LL(0), probe_b=GG(1), probe_c=GG(1), probe_d=GL(1,0)."""
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = 0, 1, 1, 0
        out = cp.build_cov_sva_integrand_5d(
            cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix
        )
        ell, i, j, k, l = 0, 1, 2, 3, 0
        expected = (
            cl_5d[probe_a_ix, probe_c_ix, ell, i, k]
            * cl_5d[probe_b_ix, probe_d_ix, ell, j, l]
            + cl_5d[probe_a_ix, probe_d_ix, ell, i, l]
            * cl_5d[probe_b_ix, probe_c_ix, ell, j, k]
        )
        np.testing.assert_allclose(out[ell, i, j, k, l], expected)

    def test_symmetric_under_ij_kl_swap_with_equal_probes(self, rng):
        """When a==b and c==d, the integrand is symmetric under (i,j)<->(j,i)
        and (k,l)<->(l,k) simultaneously, since both terms just swap."""
        n_probes, n_ell, zbins = 2, 2, 3
        cl_5d = rng.standard_normal((n_probes, n_probes, n_ell, zbins, zbins))
        out = cp.build_cov_sva_integrand_5d(cl_5d, 0, 0, 0, 0)
        np.testing.assert_allclose(out, out.transpose(0, 2, 1, 4, 3))
