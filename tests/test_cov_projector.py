"""Unit tests for the module-level pure helpers in spaceborne.cov_projector.

These are the statistic-agnostic building blocks shared by the real-space
and COSEBIs Gaussian-covariance projections:

* ``get_npair`` / ``get_dnpair`` -- the (ideal) pair-count normal is a
  textbook area integral, N(theta) = pi (theta_u^2 - theta_l^2) * A * n_i *
  n_j; we check the closed form directly and cross-check it against a
  Simpson integral of the differential dN/dtheta over the same annulus.
* ``get_delta_tomo`` -- Kronecker delta in tomographic bin space, identity
  for a probe with itself and zero across different probes.
* ``build_cl_integrand_5d_sva`` -- the universal Gaussian SVA integrand
  Cov[C_ab, C_cd] ~ C_ac C_bd + C_ad C_bc; checked against explicit index
  arithmetic on a small random C_ell array (using the [0,0]=LL, [1,1]=GG,
  [1,0]=GL probe-index convention from CLAUDE.md).
* ``build_cl_integrand_5d_mix`` -- the Gaussian MIX integrand
  C_ik N_jl + C_jl N_ik + C_il N_jk + C_jk N_il; checked element by element, for
  all 16 probe combinations, against the explicit formula of the per-element
  implementation it replaced.

* ``proj_cov_2d_quad_all_scales`` -- the batched NG quad projection, checked
  against an analytic mu=0 case and against one scale pair at a time.
* ``proj_mix_sva_simps_vectorized`` -- the vectorised Gaussian projection,
  checked against a brute-force per-element Simpson integral, including the
  COSEBIs W_n(ell) kernel path (on a minimal ``CovCOSEBIs`` stub, so no cloelib
  is needed).

``proj_cov_2d`` needs a full pipeline config and is not covered here.
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
# build_cl_integrand_5d_sva
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
        out = cp.build_cl_integrand_5d_sva(cl_5d, 0, 0, 0, 0)
        assert out.shape == (n_ell, zbins, zbins, zbins, zbins)

    def test_matches_explicit_index_arithmetic_ll_only(self, cl_5d):
        """Pure LL (probe index 0 everywhere): Cov ~ C_ik C_jl + C_il C_jk."""
        out = cp.build_cl_integrand_5d_sva(cl_5d, 0, 0, 0, 0)
        ell, i, j, k, l = 1, 2, 0, 3, 1
        expected = (
            cl_5d[0, 0, ell, i, k] * cl_5d[0, 0, ell, j, l]
            + cl_5d[0, 0, ell, i, l] * cl_5d[0, 0, ell, j, k]
        )
        np.testing.assert_allclose(out[ell, i, j, k, l], expected)

    def test_matches_explicit_index_arithmetic_mixed_probes(self, cl_5d):
        """probe_a=LL(0), probe_b=GG(1), probe_c=GG(1), probe_d=GL(1,0)."""
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = 0, 1, 1, 0
        out = cp.build_cl_integrand_5d_sva(
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
        out = cp.build_cl_integrand_5d_sva(cl_5d, 0, 0, 0, 0)
        np.testing.assert_allclose(out, out.transpose(0, 2, 1, 4, 3))


# ----------------------------------------------------------------------------- #
# build_cl_integrand_5d_mix
# ----------------------------------------------------------------------------- #
class TestBuildClIntegrand5dMix:
    """Tests for the Gaussian MIX harmonic-space integrand."""

    N_PROBES, N_ELL, ZBINS = 2, 3, 3
    PROBE_COMBOS = [
        (a, b, c, d)
        for a in range(2)
        for b in range(2)
        for c in range(2)
        for d in range(2)
    ]

    @pytest.fixture
    def cl_5d(self, rng):
        return rng.standard_normal(
            (self.N_PROBES, self.N_PROBES, self.N_ELL, self.ZBINS, self.ZBINS)
        )

    @pytest.fixture
    def nl_4d(self, rng):
        """Generic (non-diagonal) noise, so that every index placement matters."""
        return rng.standard_normal(
            (self.N_PROBES, self.N_PROBES, self.ZBINS, self.ZBINS)
        )

    @pytest.mark.parametrize('probes', PROBE_COMBOS)
    def test_matches_explicit_formula(self, cl_5d, nl_4d, probes):
        a, b, c, d = probes
        out = cp.build_cl_integrand_5d_mix(cl_5d, nl_4d, a, b, c, d)
        assert out.shape == (self.N_ELL,) + (self.ZBINS,) * 4

        for ell in range(self.N_ELL):
            for i, j, k, l in np.ndindex(*(self.ZBINS,) * 4):
                expected = (
                    cl_5d[a, c, ell, i, k] * nl_4d[b, d, j, l]
                    + cl_5d[b, d, ell, j, l] * nl_4d[a, c, i, k]
                    + cl_5d[a, d, ell, i, l] * nl_4d[b, c, j, k]
                    + cl_5d[b, c, ell, j, k] * nl_4d[a, d, i, l]
                )
                np.testing.assert_allclose(out[ell, i, j, k, l], expected, rtol=1e-14)

    def test_zero_noise_gives_zero(self, cl_5d):
        nl_4d = np.zeros((self.N_PROBES, self.N_PROBES, self.ZBINS, self.ZBINS))
        out = cp.build_cl_integrand_5d_mix(cl_5d, nl_4d, 0, 1, 0, 1)
        np.testing.assert_array_equal(out, 0.0)


# ----------------------------------------------------------------------------- #
# proj_cov_2d_quad_all_scales
# ----------------------------------------------------------------------------- #
_THETA_EDGES = np.geomspace(5, 300, 4) * np.pi / 10800  # 3 log bins, in rad


def _real_space_kernels(mu, nbs=3):
    from spaceborne.cov_real_space import k_mu

    return [
        lambda ell, p=p: k_mu(
            ell, thetal=_THETA_EDGES[p], thetau=_THETA_EDGES[p + 1], mu=mu
        )
        for p in range(nbs)
    ]


class TestProjCov2dQuadAllScales:
    """The non-Gaussian quad projection, batched over scale bins and ell_1."""

    @staticmethod
    def _binavg_j0_gaussian(s, thetal, thetau):
        r"""Exact 2/(tu^2 - tl^2) \int_tl^tu dt t \int_0^inf dl l J0(l t) e^{-l^2/2s^2},
        using \int_0^inf l J0(l t) e^{-l^2/2s^2} dl = s^2 e^{-t^2 s^2 / 2}."""
        return (
            2
            * (np.exp(-(thetal**2) * s**2 / 2) - np.exp(-(thetau**2) * s**2 / 2))
            / (thetau**2 - thetal**2)
        )

    def test_mu0_matches_analytic(self):
        """A separable Gaussian C(l1, l2) = f(l1) f(l2) projects to V_p V_q."""
        s = 30.0
        # the ell range covers the Gaussian entirely, so truncation is negligible
        ells = np.geomspace(1e-3, 400, 300)
        f = np.exp(-(ells**2) / (2 * s**2))
        cov = np.outer(f, f)[:, :, None, None]

        out = cp.proj_cov_2d_quad_all_scales(
            ells, cov, _real_space_kernels(0), _real_space_kernels(0)
        )

        v = np.array([
            self._binavg_j0_gaussian(s, _THETA_EDGES[p], _THETA_EDGES[p + 1])
            for p in range(3)
        ])  # fmt: skip
        # the residual is the cubic-spline interpolation error of f on this grid
        np.testing.assert_allclose(out[..., 0, 0], np.outer(v, v), rtol=5e-6, atol=0)

    def test_matches_one_scale_pair_at_a_time(self):
        """Batching over (s1, s2) and ell_1 reproduces proj_cov_2d(..., 'quad'),
        including for zpairs whose amplitude is 1e-4 of the largest one, and for
        non-square (zpairs_ab, zpairs_cd)."""
        nbl, nbs = 30, 2
        ells = np.geomspace(10, 3000, nbl)
        c = ells**-1.2
        u = c * np.sin(np.log(ells))
        shapes = [np.outer(c, c), np.outer(c, c) + 0.3 * np.outer(u, u), np.outer(u, c)]
        amplitudes = [1.0, 1.0, 1e-4, 2.0, 2.0, 2e-4]
        cov = np.stack(
            [a * shapes[i % 3] for i, a in enumerate(amplitudes)], axis=-1
        ).reshape(nbl, nbl, 2, 3)
        k1, k2 = _real_space_kernels(2, nbs), _real_space_kernels(4, nbs)

        batched = cp.proj_cov_2d_quad_all_scales(ells, cov, k1, k2)

        assert batched.shape == (nbs, nbs, 2, 3)
        for p in range(nbs):
            for q in range(nbs):
                one_pair = cp.proj_cov_2d(ells, cov, k1[p], k2[q], 'quad')
                np.testing.assert_allclose(batched[p, q], one_pair, rtol=1e-7, atol=0)

    def test_rejects_non_4d_input(self):
        ells = np.geomspace(10, 100, 5)
        with pytest.raises(ValueError, match='must be 4D'):
            cp.proj_cov_2d_quad_all_scales(
                ells, np.ones((5, 5, 3)), _real_space_kernels(0), _real_space_kernels(0)
            )

    def test_rejects_kernel_lists_of_different_length(self):
        ells = np.geomspace(10, 100, 5)
        with pytest.raises(ValueError, match='same length'):
            cp.proj_cov_2d_quad_all_scales(
                ells,
                np.ones((5, 5, 1, 1)),
                _real_space_kernels(0, nbs=3),
                _real_space_kernels(0, nbs=2),
            )


# ----------------------------------------------------------------------------- #
# proj_mix_sva_simps_vectorized, obs_space='cosebis'
# ----------------------------------------------------------------------------- #
class TestProjMixSvaSimpsVectorizedCosebis:
    r"""For COSEBIs the projection kernel is the precomputed W_n(ell), passed as
    ``kernel_func_kw={'w_ells_arr': ...}`` with shape (n_modes, nbl), and there
    are no Bessel orders (mu = nu = None). The one-matmul projection must equal

        1/(2 pi A_max) simps(ell W_n(ell) W_m(ell) f(ell, zi, zj, zk, zl), ell)

    for every (n, m, zi, zj, zk, zl) element. The W_n are synthetic, so neither
    cloelib nor the CovCOSEBIs __init__ are needed."""

    N_MODES, ZBINS, NBL = 4, 2, 60
    AMAX = 0.3

    @pytest.fixture
    def cov_cs(self):
        from spaceborne.cov_cosebis import CovCOSEBIs

        # bypass __init__ (which needs the full pipeline config and cloelib) and
        # set only the attributes the projection uses
        obj = CovCOSEBIs.__new__(CovCOSEBIs)
        obj.ells_proj_g = np.geomspace(2, 5000, self.NBL)
        obj.nbs = self.N_MODES
        obj.zbins = self.ZBINS
        obj.cov_shape_6d = (self.N_MODES, self.N_MODES) + (self.ZBINS,) * 4
        return obj

    @pytest.fixture
    def w_ells_arr(self, rng):
        return rng.standard_normal((self.N_MODES, self.NBL))

    def _brute_force(self, ells, w_ells_arr, integrand_5d):
        expected = np.zeros((self.N_MODES, self.N_MODES) + (self.ZBINS,) * 4)
        for n, m in np.ndindex(self.N_MODES, self.N_MODES):
            for z in np.ndindex(*(self.ZBINS,) * 4):
                y = (
                    ells
                    * w_ells_arr[n]
                    * w_ells_arr[m]
                    * integrand_5d[(slice(None), *z)]
                )
                expected[(n, m, *z)] = simps(y=y, x=ells) / (2 * np.pi * self.AMAX)
        return expected

    def test_obs_space_is_cosebis(self, cov_cs):
        assert cov_cs.obs_space == 'cosebis'

    def test_matches_element_by_element_simpson(self, cov_cs, w_ells_arr, rng):
        ells = cov_cs.ells_proj_g
        integrand_5d = rng.standard_normal((self.NBL,) + (self.ZBINS,) * 4)
        integrand_5d *= (ells**-2)[:, None, None, None, None]

        out = cov_cs.proj_mix_sva_simps_vectorized(
            cl_integrand_5d=integrand_5d,
            amax_abcd=self.AMAX,
            kernel_func_kw={'w_ells_arr': w_ells_arr},
        )

        assert out.shape == cov_cs.cov_shape_6d
        expected = self._brute_force(ells, w_ells_arr, integrand_5d)
        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=0)

    @pytest.mark.parametrize('term', ['sva', 'mix'])
    def test_compute_cs_cov_term_uses_w_ells(self, cov_cs, w_ells_arr, rng, term):
        """End to end through ``compute_cs_cov_term_probe_6d``: the EnEn Gaussian
        blocks are the brute-force projection of the SVA/MIX integrand with
        ``w_ells_arr_g``, the MIX one using the physical noise from
        ``build_noise``; the B-mode blocks vanish."""
        from spaceborne import cov_dict as cd
        from spaceborne import sb_lib as sl

        zbins, nbl = self.ZBINS, self.NBL
        ells = cov_cs.ells_proj_g
        zpairs_auto, zpairs_cross, _ = sl.get_zpairs(zbins)
        ind = sl.build_full_ind('triu', 'row-major', zbins)
        cov_cs.zpairs_auto, cov_cs.zpairs_cross = zpairs_auto, zpairs_cross
        cov_cs.ind_auto = ind[:zpairs_auto, :].copy()
        cov_cs.ind_cross = ind[zpairs_auto : zpairs_auto + zpairs_cross, :].copy()
        cov_cs.w_ells_arr_g = w_ells_arr
        cov_cs.cl_3x2pt_5d = rng.uniform(0.5, 1.5, (2, 2, nbl, zbins, zbins))
        cov_cs.cl_3x2pt_5d *= 1e-6 * (ells**-1.2)[None, None, :, None, None]
        cov_cs.nl_3x2pt_4d = sl.build_noise(
            zbins,
            2,
            sigma_eps2=(np.array([0.26, 0.37]) * np.sqrt(2)) ** 2,
            ng_shear=np.array([8.1, 11.5]),
            ng_clust=np.array([4.2, 9.3]),
        )
        cov_cs.cov_dict = cd.create_cov_dict(
            [term], [('En', 'En'), ('En', 'Bn')], dims=['6d']
        )

        for probe_abcd in ('EnEn', 'EnBn'):
            cov_cs.compute_cs_cov_term_probe_6d(
                cov_hs_ng_dict=None,
                probe_abcd=probe_abcd,
                term=term,
                amax_abcd=self.AMAX,
            )

        if term == 'sva':
            integrand_5d = cp.build_cl_integrand_5d_sva(cov_cs.cl_3x2pt_5d, 0, 0, 0, 0)
        else:
            integrand_5d = cp.build_cl_integrand_5d_mix(
                cov_cs.cl_3x2pt_5d, cov_cs.nl_3x2pt_4d, 0, 0, 0, 0
            )
        expected = self._brute_force(ells, w_ells_arr, integrand_5d)

        out = cov_cs.cov_dict[term]['En', 'En']['6d']
        assert np.abs(expected).max() > 0
        np.testing.assert_allclose(
            out, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max()
        )
        np.testing.assert_array_equal(cov_cs.cov_dict[term]['En', 'Bn']['6d'], 0.0)
