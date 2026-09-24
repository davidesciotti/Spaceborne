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

from spaceborne.cov_real_space import CovRealSpace, b_mu, k_mu, t_sn


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
                epsrel=1e-10,
                limit=500,
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


# ----------------------------------------------------------------------------- #
# proj_mix_sva_simps_vectorized
# ----------------------------------------------------------------------------- #
class TestProjMixSvaSimpsVectorized:
    """The Gaussian projection as one matmul must equal Simpson's rule applied to
    every (s1, s2, zi, zj, zk, zl) element separately."""

    NBS, ZBINS, NBL = 3, 2, 50

    @pytest.fixture
    def cov_rs(self):
        # bypass __init__ (which needs the full pipeline config) and set only the
        # attributes the projection uses
        obj = CovRealSpace.__new__(CovRealSpace)
        obj.ells_proj_g = np.geomspace(10, 3000, self.NBL)
        obj.nbs = self.NBS
        obj.theta_edges = np.geomspace(5, 300, self.NBS + 1) * np.pi / 10800
        obj.cov_shape_6d = (self.NBS, self.NBS) + (self.ZBINS,) * 4
        return obj

    @pytest.mark.parametrize(('mu', 'nu'), [(0, 0), (2, 4), (4, 0)])
    def test_matches_element_by_element_simpson(self, cov_rs, rng, mu, nu):
        from scipy.integrate import simpson

        ells, edges = cov_rs.ells_proj_g, cov_rs.theta_edges
        integrand_5d = rng.standard_normal((self.NBL,) + (self.ZBINS,) * 4)
        integrand_5d *= (ells**-2)[:, None, None, None, None]
        amax = 0.3

        out = cov_rs.proj_mix_sva_simps_vectorized(
            cl_integrand_5d=integrand_5d, amax_abcd=amax, mu=mu, nu=nu
        )

        expected = np.zeros(cov_rs.cov_shape_6d)
        for s1, s2 in itertools.product(range(self.NBS), repeat=2):
            k1 = k_mu(ells, thetal=edges[s1], thetau=edges[s1 + 1], mu=mu)
            k2 = k_mu(ells, thetal=edges[s2], thetau=edges[s2 + 1], mu=nu)
            for z in itertools.product(range(self.ZBINS), repeat=4):
                y = ells * k1 * k2 * integrand_5d[(slice(None), *z)]
                expected[(s1, s2, *z)] = simpson(y, x=ells) / (2 * np.pi * amax)

        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=0)


# ----------------------------------------------------------------------------- #
# MIX-term noise: absolute normalisation
# ----------------------------------------------------------------------------- #
def _t_mix_develop(probe_ix, zbins, sigma_eps_i):
    """Verbatim copy of the ``t_mix`` removed from cov_real_space/cov_projector
    (origin/develop), kept here as the reference normalisation."""
    t_munu = np.zeros(zbins)
    if probe_ix == 0:
        t_munu = sigma_eps_i**2
    elif probe_ix == 1:
        t_munu = np.ones(zbins)
    return t_munu


class TestMixNoiseNormalisation:
    r"""The real-space MIX term now takes its noise from ``nl_3x2pt_4d``, built in
    main.py by ``sb_lib.build_noise`` with sigma_eps2 = (sigma_eps_i sqrt(2))^2.
    On origin/develop the MIX term instead used the explicit prefactors

        LL: delta_ij sigma_{eps,i}^2 / (n_eff_src,i SR_TO_ARCMIN2)
        GG: delta_ij 1 / (n_eff_lns,i SR_TO_ARCMIN2)
        GL: 0

    (``get_delta_tomo`` * ``t_mix`` / ``n_eff_2d``). These tests pin the absolute
    normalisation of the new path against that convention, with realistic,
    bin-dependent sigma_eps_i and galaxy densities (the other tests use random Nl,
    which pins only the index structure).
    """

    ZBINS, NBS, NBL = 3, 3, 40
    AMAX = 0.4  # sr
    SIGMA_EPS_I = np.array([0.26, 0.30, 0.37])
    NGAL_SOURCES = np.array([8.1, 10.4, 11.5])  # arcmin^-2
    NGAL_LENSES = np.array([4.2, 7.9, 9.3])  # arcmin^-2

    @classmethod
    def _nl_4d(cls):
        """Noise exactly as built in main.py."""
        from spaceborne import sb_lib as sl

        return sl.build_noise(
            cls.ZBINS,
            2,
            sigma_eps2=(cls.SIGMA_EPS_I * np.sqrt(2)) ** 2,
            ng_shear=cls.NGAL_SOURCES,
            ng_clust=cls.NGAL_LENSES,
        )

    @classmethod
    def _develop_prefac(cls, probe_a_ix, probe_b_ix, zi, zj):
        """``get_prefac`` from origin/develop ``proj_cov_mix_simps``."""
        from spaceborne import constants as const
        from spaceborne import cov_projector as cp

        n_eff_2d = np.vstack((cls.NGAL_SOURCES, cls.NGAL_LENSES))
        return (
            cp.get_delta_tomo(probe_a_ix, probe_b_ix, cls.ZBINS)[zi, zj]
            * _t_mix_develop(probe_a_ix, cls.ZBINS, cls.SIGMA_EPS_I)[zi]
            / (n_eff_2d[probe_a_ix, zi] * const.SR_TO_ARCMIN2)
        )

    def test_build_noise_matches_explicit_expressions(self):
        from spaceborne import constants as const

        nl_4d = self._nl_4d()
        assert nl_4d.shape == (2, 2, self.ZBINS, self.ZBINS)

        # explicit closed forms, one block at a time
        nl_ll = np.diag(self.SIGMA_EPS_I**2 / (self.NGAL_SOURCES * const.SR_TO_ARCMIN2))
        nl_gg = np.diag(1.0 / (self.NGAL_LENSES * const.SR_TO_ARCMIN2))
        np.testing.assert_allclose(nl_4d[0, 0], nl_ll, rtol=1e-14, atol=0)
        np.testing.assert_allclose(nl_4d[1, 1], nl_gg, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(nl_4d[1, 0], 0.0)
        np.testing.assert_array_equal(nl_4d[0, 1], 0.0)

        # ...and element by element against the develop MIX prefactor
        for a, b in itertools.product((0, 1), repeat=2):
            for zi, zj in itertools.product(range(self.ZBINS), repeat=2):
                np.testing.assert_allclose(
                    nl_4d[a, b, zi, zj],
                    self._develop_prefac(a, b, zi, zj),
                    rtol=1e-14,
                    atol=0,
                )

    @pytest.fixture
    def cov_rs(self, rng):
        """A small but complete CovRealSpace, built through __init__."""
        from spaceborne import constants as const
        from spaceborne import sb_lib as sl

        zbins, nbl = self.ZBINS, self.NBL
        zpairs_auto, zpairs_cross, _ = sl.get_zpairs(zbins)
        ind = sl.build_full_ind('triu', 'row-major', zbins)
        ind_auto = ind[:zpairs_auto, :].copy()
        ind_cross = ind[zpairs_auto : zpairs_auto + zpairs_cross, :].copy()

        cfg = {
            'misc': {'num_threads': 1},
            'covariance': {
                'G': True,
                'SSC': False,
                'cNG': False,
                'sigma_eps_i': self.SIGMA_EPS_I.tolist(),
            },
            'nz': {
                'ngal_sources': self.NGAL_SOURCES.tolist(),
                'ngal_lenses': self.NGAL_LENSES.tolist(),
            },
            'binning': {
                'theta_min_arcmin': 5.0,
                'theta_max_arcmin': 300.0,
                'theta_bins': self.NBS,
                'binning_type': 'log',
            },
            'precision': {
                'proj_gauss_integration_method': 'simps',
                'proj_nongauss_integration_method': 'quad',
            },
        }
        pvt_cfg = {
            'zbins': zbins,
            'zpairs_auto': zpairs_auto,
            'zpairs_cross': zpairs_cross,
            'ind_auto': ind_auto,
            'ind_cross': ind_cross,
            'ind_dict': {'LL': ind_auto, 'GL': ind_cross, 'GG': ind_auto},
            'nbs': self.NBS,
            'req_terms': ['g'],
            'req_probe_combs_rs_2d': list(const.RS_ALL_PROBE_COMBS),
            'symmetrize_output_dict': const.HS_SYMMETRIZE_OUTPUT_DICT,
        }

        # positive, power-law Cl of realistic amplitude (comparable to the noise)
        ells = np.geomspace(10, 3000, nbl)
        cl_3x2pt_5d = rng.uniform(0.5, 1.5, (2, 2, nbl, zbins, zbins))
        cl_3x2pt_5d *= 1e-6 * (ells**-1.2)[None, None, :, None, None]

        return CovRealSpace(
            cfg=cfg,
            pvt_cfg=pvt_cfg,
            cl_3x2pt_5d=cl_3x2pt_5d,
            nl_3x2pt_4d=self._nl_4d(),
            ells_proj_g=ells,
            ells_proj_ng=ells,
        )

    @pytest.mark.parametrize(
        'probe_abcd',
        ['xipxip', 'ximxim', 'xipxim', 'gtgt', 'ww', 'xipgt', 'gtw', 'ximw'],
    )
    def test_mix_cov_matches_develop_convention(self, cov_rs, probe_abcd):
        """CovRealSpace's MIX block equals, element by element, the develop
        per-element Simpson integral built with the old t_mix prefactors."""
        from scipy.integrate import simpson

        from spaceborne import constants as const
        from spaceborne import sb_lib as sl

        cov_rs.compute_rs_cov_term_probe_6d(
            cov_hs_ng_dict=None, probe_abcd=probe_abcd, term='mix', amax_abcd=self.AMAX
        )
        probe_2tpl = sl.split_probe_name(probe_abcd, 'real')
        out = cov_rs.cov_dict['mix'][probe_2tpl]['6d']

        # independent theta binning: 3 log bins in [5, 300] arcmin
        edges = np.deg2rad(np.geomspace(5.0 / 60, 300.0 / 60, self.NBS + 1))
        np.testing.assert_allclose(cov_rs.theta_edges, edges, rtol=1e-14)

        mu = const.MU_DICT[probe_2tpl[0]]
        nu = const.MU_DICT[probe_2tpl[1]]
        a, b, c, d = const.RS_PROBE_NAME_TO_IX_DICT[probe_abcd]
        ells, cl, pf = cov_rs.ells_proj_g, cov_rs.cl_3x2pt_5d, self._develop_prefac

        expected = np.zeros(cov_rs.cov_shape_6d)
        for s1, s2 in itertools.product(range(self.NBS), repeat=2):
            k1 = k_mu(ells, thetal=edges[s1], thetau=edges[s1 + 1], mu=mu)
            k2 = k_mu(ells, thetal=edges[s2], thetau=edges[s2 + 1], mu=nu)
            for i, j, k, l in itertools.product(range(self.ZBINS), repeat=4):
                inner = (
                    cl[a, c, :, i, k] * pf(b, d, j, l)
                    + cl[b, d, :, j, l] * pf(a, c, i, k)
                    + cl[a, d, :, i, l] * pf(b, c, j, k)
                    + cl[b, c, :, j, k] * pf(a, d, i, l)
                )
                y = ells * k1 * k2 * inner / (2 * np.pi * self.AMAX)
                expected[s1, s2, i, j, k, l] = simpson(y, x=ells)

        # the reference is non-trivial for every combo but ximw (zero noise
        # overlap between a pure-source and a pure-lens pair)
        if probe_abcd != 'ximw':
            assert np.abs(expected).max() > 0
        atol = 1e-12 * np.abs(expected).max()
        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=atol)
