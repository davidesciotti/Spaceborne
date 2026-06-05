"""Unit tests for the spaceborne.cosmo_lib module.

These cover the pure helper functions (unit conversions, closed-form density and
ell prefactor formulas, dict key remapping) and the CCL-backed cosmology
quantities (comoving distance, growth, Limber-k, integral prefactor conventions,
cosmology instantiation). Expected values for the closed-form helpers are
hard-coded literals so that an accidental change to a formula is actually caught
(re-deriving with the same formula in the test would be circular).
"""

import numpy as np
import pyccl as ccl
import pytest

from spaceborne import cosmo_lib as cl


# ----------------------------------------------------------------------------- #
# Fixtures
# ----------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def cosmo():
    """Cheap background-only cosmology for distance/growth tests."""
    return ccl.CosmologyVanillaLCDM()


@pytest.fixture(scope='module')
def cosmo_analytic():
    """Cosmology with analytic transfer + halofit so pk evaluation is fast
    (no Boltzmann/CAMB call)."""
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.81,
        n_s=0.96,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='halofit',
    )


# ----------------------------------------------------------------------------- #
# z <-> a conversions
# ----------------------------------------------------------------------------- #
class TestScaleFactorConversions:
    def test_z_to_a_known_values(self):
        assert cl.z_to_a(0.0) == 1.0
        assert cl.z_to_a(1.0) == 0.5
        assert cl.z_to_a(3.0) == 0.25

    def test_a_to_z_known_values(self):
        assert cl.a_to_z(1.0) == 0.0
        assert cl.a_to_z(0.5) == 1.0

    def test_round_trip(self):
        z = np.array([0.0, 0.3, 1.0, 2.5, 5.0])
        np.testing.assert_allclose(cl.a_to_z(cl.z_to_a(z)), z)

    def test_vectorized(self):
        z = np.array([0.0, 1.0, 3.0])
        np.testing.assert_allclose(cl.z_to_a(z), [1.0, 0.5, 0.25])


# ----------------------------------------------------------------------------- #
# fsky <-> deg2 conversions
# ----------------------------------------------------------------------------- #
class TestFskyConversions:
    def test_full_sky_deg2(self):
        """The full sphere is 4*pi sr = 41252.96... deg^2 -> fsky = 1."""
        full_sky_deg2 = 4 * np.pi * (180 / np.pi) ** 2
        np.testing.assert_allclose(cl.deg2_to_fsky(full_sky_deg2), 1.0)

    def test_fsky_to_deg2_full_sky(self):
        np.testing.assert_allclose(cl.fsky_to_deg2(1.0), 41252.96125, rtol=1e-6)

    def test_round_trip(self):
        for area in [1000.0, 13245.0, 41252.96]:
            np.testing.assert_allclose(cl.fsky_to_deg2(cl.deg2_to_fsky(area)), area)

    def test_half_sky(self):
        full = cl.fsky_to_deg2(1.0)
        np.testing.assert_allclose(cl.deg2_to_fsky(full / 2), 0.5)


# ----------------------------------------------------------------------------- #
# neutrino density parameter
# ----------------------------------------------------------------------------- #
class TestOmegaNu:
    def test_omega_nu0_new_value(self):
        """Omega_nu0 = m_nu / (93.14 * h^2). Hard-coded reference value."""
        np.testing.assert_allclose(
            cl.get_omega_nu0_new(0.06, 0.6737), 0.0014193252274, rtol=1e-9
        )

    def test_omega_nu0_new_zero_mass(self):
        assert cl.get_omega_nu0_new(0.0, 0.6737) == 0.0

    def test_omega_nu0_new_scales_linearly_with_mass(self):
        a = cl.get_omega_nu0_new(0.06, 0.67)
        b = cl.get_omega_nu0_new(0.12, 0.67)
        np.testing.assert_allclose(b, 2 * a)

    def test_omega_nu0_value(self):
        """The g-factor version with default n_eff=3.046, fac=94.07."""
        np.testing.assert_allclose(
            cl.get_omega_nu0(0.06, 0.6737, n_eff=3.046), 0.0013893463627, rtol=1e-9
        )

    def test_omega_nu0_zero_mass(self):
        assert cl.get_omega_nu0(0.0, 0.6737) == 0.0


# ----------------------------------------------------------------------------- #
# Omega_k
# ----------------------------------------------------------------------------- #
class TestOmegaK:
    def test_open_universe(self):
        """Clearly non-flat: no warning, returns 1 - Om - ODE."""
        np.testing.assert_allclose(cl.get_omega_k0(0.3, 0.5), 0.2)

    def test_closed_universe(self):
        np.testing.assert_allclose(cl.get_omega_k0(0.6, 0.5), -0.1)

    def test_flat_universe_warns_and_returns_zero(self):
        """1 - 0.3 - 0.7 is ~1e-17 (not exactly 0) -> warns and is set to 0."""
        with pytest.warns(UserWarning, match='Omega_k is very small'):
            result = cl.get_omega_k0(0.3, 0.7)
        assert result == 0


# ----------------------------------------------------------------------------- #
# ell prefactors
# ----------------------------------------------------------------------------- #
class TestEllPrefactors:
    def test_mag_prefactor_formula(self):
        """ell*(ell+1)/(ell+0.5)^2."""
        ell = 100.0
        expected = 100.0 * 101.0 / 100.5**2
        np.testing.assert_allclose(cl.ell_prefactor_mag(ell), expected)

    def test_mag_prefactor_large_ell_tends_to_one(self):
        """For large ell the prefactor -> 1."""
        np.testing.assert_allclose(cl.ell_prefactor_mag(1e6), 1.0, atol=1e-9)

    def test_gamma_ia_prefactor_formula(self):
        """sqrt((ell+2)(ell+1)ell(ell-1)) / (ell+0.5)^2."""
        ell = 50.0
        expected = np.sqrt(52.0 * 51.0 * 50.0 * 49.0) / 50.5**2
        np.testing.assert_allclose(cl.ell_prefactor_gamma_and_ia(ell), expected)

    def test_gamma_ia_prefactor_large_ell_tends_to_one(self):
        np.testing.assert_allclose(cl.ell_prefactor_gamma_and_ia(1e6), 1.0, atol=1e-6)

    def test_prefactors_vectorized(self):
        ell = np.array([10.0, 100.0, 1000.0])
        assert cl.ell_prefactor_mag(ell).shape == ell.shape
        assert cl.ell_prefactor_gamma_and_ia(ell).shape == ell.shape


# ----------------------------------------------------------------------------- #
# map_keys
# ----------------------------------------------------------------------------- #
class TestMapKeys:
    def test_default_mapping_renames_known_keys(self):
        out = cl.map_keys({'Om': 0.3, 's8': 0.8, 'h': 0.67}, key_mapping=None)
        assert out['Om_m0'] == 0.3
        assert out['sigma_8'] == 0.8
        assert out['h'] == 0.67

    def test_retains_unmapped_keys(self):
        """Keys absent from the mapping are passed through unchanged."""
        out = cl.map_keys({'Om': 0.3, 'extra_param': 42}, key_mapping=None)
        assert out['extra_param'] == 42
        assert out['Om_m0'] == 0.3

    def test_missing_keys_not_added(self):
        """Mapping keys not present in the input do not appear in the output."""
        out = cl.map_keys({'Om': 0.3}, key_mapping=None)
        assert 'sigma_8' not in out
        assert 'Om_Lambda0' not in out

    def test_custom_mapping(self):
        out = cl.map_keys({'a': 1, 'b': 2}, key_mapping={'a': 'alpha'})
        assert out['alpha'] == 1
        assert out['b'] == 2  # unmapped, retained


# ----------------------------------------------------------------------------- #
# comoving distance (CCL-backed)
# ----------------------------------------------------------------------------- #
class TestComovingDistance:
    def test_zero_at_z0(self, cosmo):
        assert cl.ccl_comoving_distance(0.0, use_h_units=False, cosmo_ccl=cosmo) == 0.0

    def test_monotonically_increasing(self, cosmo):
        z = np.linspace(0.0, 3.0, 20)
        chi = cl.ccl_comoving_distance(z, use_h_units=False, cosmo_ccl=cosmo)
        assert np.all(np.diff(chi) > 0)

    def test_h_units_scaling(self, cosmo):
        """Distance in Mpc/h is the Mpc value multiplied by h."""
        z = np.array([0.5, 1.0, 2.0])
        chi_mpc = cl.ccl_comoving_distance(z, use_h_units=False, cosmo_ccl=cosmo)
        chi_mpch = cl.ccl_comoving_distance(z, use_h_units=True, cosmo_ccl=cosmo)
        h = cosmo.cosmo.params.h
        np.testing.assert_allclose(chi_mpch, chi_mpc * h)


# ----------------------------------------------------------------------------- #
# growth factor (CCL-backed)
# ----------------------------------------------------------------------------- #
class TestGrowthFactor:
    def test_normalized_to_one_today(self, cosmo):
        """CCL normalizes the growth factor to 1 at z=0."""
        np.testing.assert_allclose(cl.growth_factor_ccl(0.0, cosmo), 1.0)

    def test_decreasing_with_redshift(self, cosmo):
        z = np.linspace(0.0, 3.0, 20)
        d = cl.growth_factor_ccl(z, cosmo)
        assert np.all(np.diff(d) < 0)


# ----------------------------------------------------------------------------- #
# k_limber (CCL-backed)
# ----------------------------------------------------------------------------- #
class TestKLimber:
    def test_definition(self, cosmo):
        """k_limber = (ell + 0.5) / chi(z)."""
        ell = np.array([10.0, 100.0, 1000.0])
        z = 1.0
        chi = cl.ccl_comoving_distance(z, use_h_units=False, cosmo_ccl=cosmo)
        expected = (ell + 0.5) / chi
        np.testing.assert_allclose(
            cl.k_limber(ell, z, use_h_units=False, cosmo_ccl=cosmo), expected
        )

    def test_requires_bool_use_h_units(self, cosmo):
        with pytest.raises(AssertionError, match='use_h_units must be'):
            cl.k_limber(100.0, 1.0, use_h_units='yes', cosmo_ccl=cosmo)

    def test_get_kmax_limber_returns_max(self, cosmo):
        ell_grid = np.array([10.0, 100.0, 1000.0])
        z_grid = np.array([0.5, 1.0, 2.0])
        kmax = cl.get_kmax_limber(ell_grid, z_grid, False, cosmo)
        # the maximum is reached at max ell and min chi (lowest z)
        expected = cl.k_limber(
            ell_grid.max(), z_grid.min(), use_h_units=False, cosmo_ccl=cosmo
        )
        np.testing.assert_allclose(kmax, expected)


# ----------------------------------------------------------------------------- #
# pk_from_ccl (CCL-backed, analytic cosmo for speed)
# ----------------------------------------------------------------------------- #
class TestPkFromCcl:
    def test_shape(self, cosmo_analytic):
        k = np.array([0.01, 0.1, 1.0])
        z = np.array([0.0, 0.5, 1.0])
        _, pk = cl.pk_from_ccl(k.copy(), z, use_h_units=False, cosmo_ccl=cosmo_analytic)
        assert pk.shape == (len(k), len(z))

    def test_invalid_pk_kind_raises(self, cosmo_analytic):
        with pytest.raises(ValueError, match='linear.*nonlinear'):
            cl.pk_from_ccl(
                np.array([0.1]), np.array([0.0]), False, cosmo_analytic, pk_kind='bogus'
            )

    def test_h_units_scaling(self, cosmo_analytic):
        """In h-units, pk is multiplied by h^3 and k divided by h."""
        k = np.array([0.01, 0.1, 1.0])
        z = np.array([0.0, 1.0])
        h = cosmo_analytic.cosmo.params.h

        k_no, pk_no = cl.pk_from_ccl(
            k.copy(), z, use_h_units=False, cosmo_ccl=cosmo_analytic
        )
        k_h, pk_h = cl.pk_from_ccl(
            k.copy(), z, use_h_units=True, cosmo_ccl=cosmo_analytic
        )

        np.testing.assert_allclose(k_h, k_no / h)
        np.testing.assert_allclose(pk_h, pk_no * h**3)

    def test_linear_vs_nonlinear_differ(self, cosmo_analytic):
        """Nonlinear power exceeds linear at small scales (high k)."""
        k = np.array([1.0, 5.0])
        z = np.array([0.0])
        _, pk_lin = cl.pk_from_ccl(k.copy(), z, False, cosmo_analytic, pk_kind='linear')
        _, pk_nl = cl.pk_from_ccl(
            k.copy(), z, False, cosmo_analytic, pk_kind='nonlinear'
        )
        assert np.all(pk_nl > pk_lin)


# ----------------------------------------------------------------------------- #
# cl_integral_prefactor (CCL-backed)
# ----------------------------------------------------------------------------- #
class TestClIntegralPrefactor:
    def test_pyssc_convention_deprecated(self, cosmo):
        with pytest.raises(ValueError, match='PySSC.*deprecated'):
            cl.cl_integral_prefactor(1.0, 'PySSC', use_h_units=False, cosmo_ccl=cosmo)

    def test_invalid_convention_raises(self, cosmo):
        with pytest.raises(ValueError, match='Euclid.*PySSC'):
            cl.cl_integral_prefactor(1.0, 'bogus', use_h_units=False, cosmo_ccl=cosmo)

    def test_ke_to_euclid_ratio_is_inverse_r_squared(self, cosmo):
        """The KE-approximation prefactor differs from the Euclid one by 1/r^2."""
        z = np.array([0.5, 1.0, 2.0])
        r = cl.ccl_comoving_distance(z, use_h_units=False, cosmo_ccl=cosmo)
        euclid = cl.cl_integral_prefactor(
            z, 'Euclid', use_h_units=False, cosmo_ccl=cosmo
        )
        ke = cl.cl_integral_prefactor(
            z, 'Euclid_KE_approximation', use_h_units=False, cosmo_ccl=cosmo
        )
        np.testing.assert_allclose(ke / euclid, 1.0 / r**2, rtol=1e-10)

    def test_euclid_prefactor_positive(self, cosmo):
        z = np.array([0.3, 1.0, 2.5])
        pref = cl.cl_integral_prefactor(z, 'Euclid', use_h_units=False, cosmo_ccl=cosmo)
        assert np.all(pref > 0)


# ----------------------------------------------------------------------------- #
# instantiate_cosmo_ccl_obj
# ----------------------------------------------------------------------------- #
class TestInstantiateCosmoCcl:
    @pytest.fixture
    def fiducial(self):
        return {
            'Om_m0': 0.32,
            'Om_b0': 0.05,
            'Om_k0': 0.0,
            'w_0': -1.0,
            'w_a': 0.0,
            'h': 0.6737,
            'sigma_8': 0.816,
            'n_s': 0.966,
            'm_nu': 0.06,
            'N_eff': 3.046,
        }

    def test_returns_ccl_cosmology(self, fiducial):
        cosmo = cl.instantiate_cosmo_ccl_obj(fiducial, extra_parameters=None)
        assert isinstance(cosmo, ccl.Cosmology)

    def test_passed_parameters_match(self, fiducial):
        cosmo = cl.instantiate_cosmo_ccl_obj(fiducial, extra_parameters=None)
        p = cosmo.cosmo.params
        np.testing.assert_allclose(p.h, fiducial['h'])
        np.testing.assert_allclose(p.Omega_b, fiducial['Om_b0'])
        np.testing.assert_allclose(p.sigma8, fiducial['sigma_8'])
        np.testing.assert_allclose(p.n_s, fiducial['n_s'])

    def test_omega_c_excludes_baryons_and_neutrinos(self, fiducial):
        """Omega_c = Om_m0 - Om_b0 - Omega_nu (cold dark matter only)."""
        cosmo = cl.instantiate_cosmo_ccl_obj(fiducial, extra_parameters=None)
        omega_nu = cl.get_omega_nu0(
            fiducial['m_nu'], fiducial['h'], n_eff=fiducial['N_eff']
        )
        expected_omega_c = fiducial['Om_m0'] - fiducial['Om_b0'] - omega_nu
        np.testing.assert_allclose(cosmo.cosmo.params.Omega_c, expected_omega_c)
