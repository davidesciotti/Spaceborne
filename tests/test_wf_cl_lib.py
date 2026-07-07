"""Unit tests for the spaceborne.wf_cl_lib module (n(z), photo-z, galaxy/IA bias).

Strategy: these are mostly pure numerical functions, so we pin down analytic
invariants rather than golden numbers:

* ``normalize_nz`` -- the normalized n(z) integrates to 1 (Simpson's rule),
  both for 1D input and per-column for the 2D ``(z, zbins)`` arrays used
  everywhere else in the codebase.
* ``nz_smail`` -- positivity and the analytic peak location
  (``z_peak = z_0 * (4/3)**(2/3)``, from setting d/dz[log n(z)] = 0).
* ``p_ph`` -- for fixed ``z_p`` it is a properly normalized density in ``z``
  (mixture of two truncated-ish Gaussians with weights ``1 - f_out`` and
  ``f_out``).
* ``convolve_nz_with_p_ph`` -- shape, non-negativity, and normalization.
* Galaxy/IA bias fits -- finite-ness, shape, spot values at z=0, and the
  ``ValueError`` guard for unsupported ``magcut_lens``.
* ``stepwise_bias`` -- piecewise-constant behaviour, including edge cases.
* ``build_galaxy_bias_2d_arr`` -- shape and the "constant bias reproduces the
  constant" invariant across the ``unbiased/linint/constant/step-wise/
  polynomial`` branches.
* ``build_ia_bias_1d_arr`` -- uses the cheap eisenstein_hu + halofit
  cosmology (same pattern as ``tests/test_cov_ssc.py``) since it needs the
  CCL growth factor.
* ``shift_nz`` -- a zero shift is the identity; a shift by delta moves the
  first moment (``get_z_means``) by ~delta.

Plotting helpers, CCL-heavy ``cl_ccl``/``compute_cls_or_interpolate_input_cls``
and the STEM derivative helper are intentionally not covered here.
"""

import numpy as np
import pyccl as ccl
import pytest
from scipy.integrate import simpson as simps

from spaceborne import wf_cl_lib as wl


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope='module')
def cosmo():
    """A cheap-to-evaluate vanilla LCDM cosmology (see test_cov_ssc.py)."""
    return ccl.CosmologyVanillaLCDM(
        transfer_function='eisenstein_hu', matter_power_spectrum='halofit'
    )


# ----------------------------------------------------------------------------- #
# normalize_nz
# ----------------------------------------------------------------------------- #
class TestNormalizeNz:
    """Tests for normalize_nz."""

    def test_integrates_to_one_1d(self):
        """A 1D n(z) is rescaled to integrate to exactly 1."""
        z = np.linspace(0.001, 4, 2000)
        n_z = wl.nz_smail(z, 0.6, 1.0)
        n_z_norm = wl.normalize_nz(n_z, z)
        np.testing.assert_allclose(simps(y=n_z_norm, x=z), 1.0, rtol=1e-10)

    def test_integrates_to_one_2d_per_bin(self):
        """Each column of a 2D n(z) should integrate to 1 independently."""
        z = np.linspace(0.001, 4, 2000)
        n_z = np.column_stack([wl.nz_smail(z, 0.6, 1.0), wl.nz_smail(z, 0.9, 1.0)])
        n_z_norm = wl.normalize_nz(n_z, z)
        np.testing.assert_allclose(
            simps(y=n_z_norm, x=z, axis=0), [1.0, 1.0], rtol=1e-10
        )


# ----------------------------------------------------------------------------- #
# nz_smail
# ----------------------------------------------------------------------------- #
class TestNzSmail:
    """Tests for the Smail-type n(z) parametrization."""

    def test_positive(self):
        z = np.linspace(0.001, 4, 500)
        assert np.all(wl.nz_smail(z, 0.6, 1.0) > 0)

    def test_peaks_near_expected_z(self):
        """d/dz[2 log(z) - (z/z0)**1.5] = 0 => z_peak = z0 * (4/3)**(2/3)."""
        z0 = 0.6
        z = np.linspace(0.001, 4, 200_000)
        nz = wl.nz_smail(z, z0, 1.0)
        peak_z = z[np.argmax(nz)]
        analytic_peak = z0 * (4 / 3) ** (2 / 3)
        np.testing.assert_allclose(peak_z, analytic_peak, rtol=1e-3)

    def test_scales_with_n_gal(self):
        z = np.linspace(0.001, 4, 100)
        base = wl.nz_smail(z, 0.6, 1.0)
        scaled = wl.nz_smail(z, 0.6, 3.0)
        np.testing.assert_allclose(scaled, 3.0 * base)


# ----------------------------------------------------------------------------- #
# p_ph
# ----------------------------------------------------------------------------- #
class TestPPh:
    """Tests for the photo-z scatter kernel p_ph(z_p, z)."""

    def test_normalized_in_z(self):
        """For fixed z_p, p_ph integrates to ~1 over z (it's a density in z)."""
        zp = 0.5
        kw = {
            'c_in': 1.0, 'z_in': 0.0, 'sigma_in': 0.05,
            'c_out': 1.0, 'z_out': 0.1, 'sigma_out': 0.1, 'f_out': 0.1,
        }  # fmt: skip
        z = np.linspace(0.0, 5.0, 20_000)
        pph = wl.p_ph(zp, z, **kw)
        np.testing.assert_allclose(simps(y=pph, x=z), 1.0, rtol=5e-3)

    def test_non_negative(self):
        zp = 0.5
        kw = {
            'c_in': 1.0, 'z_in': 0.0, 'sigma_in': 0.05,
            'c_out': 1.0, 'z_out': 0.1, 'sigma_out': 0.1, 'f_out': 0.1,
        }  # fmt: skip
        z = np.linspace(0.0, 5.0, 1000)
        assert np.all(wl.p_ph(zp, z, **kw) >= 0)

    def test_f_out_zero_is_single_gaussian(self):
        """With f_out=0, p_ph reduces exactly to the first Gaussian term."""
        zp = 0.7
        z = np.linspace(0.0, 5.0, 100)
        kw = {
            'c_in': 1.0, 'z_in': 0.0, 'sigma_in': 0.05,
            'c_out': 1.0, 'z_out': 0.1, 'sigma_out': 0.1, 'f_out': 0.0,
        }  # fmt: skip
        out = wl.p_ph(zp, z, **kw)
        expected = (
            1
            / (np.sqrt(2 * np.pi) * kw['sigma_in'] * (1 + z))
            * np.exp(
                -0.5
                * ((z - kw['c_in'] * zp - kw['z_in']) / (kw['sigma_in'] * (1 + z))) ** 2
            )
        )
        np.testing.assert_allclose(out, expected)


# ----------------------------------------------------------------------------- #
# convolve_nz_with_p_ph
# ----------------------------------------------------------------------------- #
class TestConvolveNzWithPPh:
    """Tests for the n(z) -> photo-z-convolved n(z) pipeline."""

    @pytest.fixture
    def z_grid(self):
        return np.linspace(0.001, 4, 1000)

    @pytest.fixture
    def p_ph_kw(self):
        return {
            'c_in': 1.0, 'z_in': 0.0, 'sigma_in': 0.05,
            'c_out': 1.0, 'z_out': 0.1, 'sigma_out': 0.1, 'f_out': 0.1,
        }  # fmt: skip

    def test_shape(self, z_grid, p_ph_kw):
        nz = wl.nz_smail(z_grid, 0.6, 1.0)
        out = wl.convolve_nz_with_p_ph(nz, 0.2, 1.0, z_grid, p_ph_kw, zp_steps=200)
        assert out.shape == z_grid.shape

    def test_non_negative(self, z_grid, p_ph_kw):
        nz = wl.nz_smail(z_grid, 0.6, 1.0)
        out = wl.convolve_nz_with_p_ph(nz, 0.2, 1.0, z_grid, p_ph_kw, zp_steps=200)
        assert np.all(out >= -1e-12)

    def test_normalize_true_integrates_to_one(self, z_grid, p_ph_kw):
        nz = wl.nz_smail(z_grid, 0.6, 1.0)
        out = wl.convolve_nz_with_p_ph(
            nz, 0.2, 1.0, z_grid, p_ph_kw, zp_steps=200, normalize=True
        )
        np.testing.assert_allclose(simps(y=out, x=z_grid), 1.0, rtol=1e-8)

    def test_f_out_zero_is_still_non_negative_and_finite(self, z_grid):
        """With no outlier fraction, the convolution is a plain Gaussian smoothing."""
        p_ph_kw = {
            'c_in': 1.0, 'z_in': 0.0, 'sigma_in': 0.05,
            'c_out': 1.0, 'z_out': 0.1, 'sigma_out': 0.1, 'f_out': 0.0,
        }  # fmt: skip
        nz = wl.nz_smail(z_grid, 0.6, 1.0)
        out = wl.convolve_nz_with_p_ph(nz, 0.2, 1.0, z_grid, p_ph_kw, zp_steps=200)
        assert np.all(np.isfinite(out))
        assert np.all(out >= -1e-12)


# ----------------------------------------------------------------------------- #
# f_ia
# ----------------------------------------------------------------------------- #
class TestFIa:
    """Tests for the intrinsic alignment redshift/luminosity scaling f_ia."""

    def test_beta_zero_drops_luminosity_term(self):
        """With beta_IA=0, f_ia == ((1+z)/(1+z_pivot))**eta_IA regardless of func."""

        def lum_func(z):
            return 2.0

        z = np.array([0.0, 1.0, 2.0])
        out = wl.f_ia(
            z, eta_IA=1.5, beta_IA=0.0, z_pivot_IA=0.5, lumin_ratio_func=lum_func
        )
        expected = ((1 + z) / 1.5) ** 1.5
        np.testing.assert_allclose(out, expected)

    def test_eta_zero_drops_redshift_term(self):
        """With eta_IA=0, f_ia == lumin_ratio_func(z)**beta_IA."""

        def lum_func(z):
            return z + 2.0

        z = np.array([0.0, 1.0, 2.0])
        out = wl.f_ia(
            z, eta_IA=0.0, beta_IA=2.0, z_pivot_IA=0.5, lumin_ratio_func=lum_func
        )
        expected = (z + 2.0) ** 2.0
        np.testing.assert_allclose(out, expected)


# ----------------------------------------------------------------------------- #
# Galaxy bias fits
# ----------------------------------------------------------------------------- #
class TestBOfZFits:
    """Spot-checks and monotonicity for the various b(z) parametrizations."""

    def test_b_of_z_analytical(self):
        z = np.linspace(0, 3, 50)
        b = wl.b_of_z_analytical(z)
        np.testing.assert_allclose(b, np.sqrt(1 + z))
        assert np.all(np.diff(b) > 0)  # monotonically increasing

    def test_b_of_z_fs1_leporifit_at_zero(self):
        assert wl.b_of_z_fs1_leporifit(0.0) == pytest.approx(0.5125)

    def test_b_of_z_fs1_pocinofit_at_zero(self):
        assert wl.b_of_z_fs1_pocinofit(0.0) == pytest.approx(1.02)

    @pytest.mark.parametrize('magcut,expected_b0', [(24.5, 1.33291), (23, 1.88571)])
    def test_b_of_z_fs2_fit_at_zero(self, magcut, expected_b0):
        assert wl.b_of_z_fs2_fit(0.0, magcut) == pytest.approx(expected_b0)

    def test_b_of_z_fs2_fit_invalid_magcut_raises(self):
        with pytest.raises(ValueError, match='magcut_lens'):
            wl.b_of_z_fs2_fit(0.0, magcut_lens=20)

    def test_b_of_z_fs2_fit_poly_fit_values_override(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        z = np.array([0.0, 1.0])
        out = wl.b_of_z_fs2_fit(z, magcut_lens=24.5, poly_fit_values=vals)
        np.testing.assert_allclose(out, [1.0, 1 + 2 + 3 + 4])

    @pytest.mark.parametrize('magcut,expected_b0', [(24.5, -1.50685), (23, -2.34493)])
    def test_magbias_of_z_fs2_fit_at_zero(self, magcut, expected_b0):
        assert wl.magbias_of_z_fs2_fit(0.0, magcut) == pytest.approx(expected_b0)

    def test_s_of_z_fs2_fit_matches_definition(self):
        z = np.linspace(0, 2, 10)
        mb = wl.magbias_of_z_fs2_fit(z, 24.5)
        s = wl.s_of_z_fs2_fit(z, 24.5)
        np.testing.assert_allclose(s, (mb + 2) / 5)

    def test_b2g_fs2_fit_finite_and_close_to_measured_points(self):
        """Spot-check against the measured (z, b2) pairs quoted in the docstring.

        This is a polynomial *fit*, not an interpolation, so we only require
        it to be reasonably close (the fit residuals are up to ~0.2 for the
        quoted points).
        """
        z_meas = np.array([0.395, 0.785, 1.175, 1.565, 1.955, 2.345])
        b2_meas = np.array(
            [-0.25209754, 0.14240271, 0.56409318, 1.06597924, 2.84258843, 4.8300518]
        )
        out = wl.b2g_fs2_fit(z_meas)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, b2_meas, atol=0.3)


# ----------------------------------------------------------------------------- #
# stepwise_bias
# ----------------------------------------------------------------------------- #
class TestStepwiseBias:
    """Tests for the piecewise-constant bias helper."""

    @pytest.fixture
    def z_edges(self):
        return np.array([0.0, 0.5, 1.0, 1.5])

    @pytest.fixture
    def gal_bias(self):
        return np.array([1.0, 2.0, 3.0])

    def test_value_inside_each_bin(self, z_edges, gal_bias):
        assert wl.stepwise_bias(0.25, gal_bias, z_edges) == 1.0
        assert wl.stepwise_bias(0.75, gal_bias, z_edges) == 2.0
        assert wl.stepwise_bias(1.25, gal_bias, z_edges) == 3.0

    def test_below_first_edge_uses_first_bin(self, z_edges, gal_bias):
        assert wl.stepwise_bias(-1.0, gal_bias, z_edges) == gal_bias[0]

    def test_at_or_above_last_edge_uses_last_bin(self, z_edges, gal_bias):
        assert wl.stepwise_bias(1.5, gal_bias, z_edges) == gal_bias[-1]
        assert wl.stepwise_bias(5.0, gal_bias, z_edges) == gal_bias[-1]

    def test_bin_boundary_is_half_open_on_the_right(self, z_edges, gal_bias):
        """z_minus[i] <= z < z_plus[i], so z=0.5 belongs to the *second* bin."""
        assert wl.stepwise_bias(0.5, gal_bias, z_edges) == gal_bias[1]

    def test_unsorted_edges_raise(self, gal_bias):
        with pytest.raises(AssertionError):
            wl.stepwise_bias(0.5, gal_bias, np.array([0.0, 1.0, 0.5, 1.5]))


# ----------------------------------------------------------------------------- #
# build_galaxy_bias_2d_arr
# ----------------------------------------------------------------------------- #
class TestBuildGalaxyBias2dArr:
    """Tests for the 2D (z_grid, zbins) galaxy bias array builder."""

    @pytest.fixture
    def z_grid(self):
        return np.linspace(0, 2, 50)

    @pytest.fixture
    def zbins(self):
        return 3

    @pytest.fixture
    def zmeans(self):
        return np.array([0.3, 0.7, 1.1])

    def test_unbiased_is_all_ones(self, z_grid, zbins):
        out = wl.build_galaxy_bias_2d_arr(
            None, None, None, zbins, z_grid, bias_model='unbiased'
        )
        assert out.shape == (len(z_grid), zbins)
        assert np.all(out == 1.0)

    def test_constant_reproduces_per_bin_constants(self, z_grid, zbins, zmeans):
        gal_bias_vs_zmean = np.array([1.2, 1.5, 1.8])
        out = wl.build_galaxy_bias_2d_arr(
            gal_bias_vs_zmean, zmeans, None, zbins, z_grid, bias_model='constant'
        )
        assert out.shape == (len(z_grid), zbins)
        np.testing.assert_allclose(out, np.tile(gal_bias_vs_zmean, (len(z_grid), 1)))

    def test_linint_with_constant_input_reproduces_constant(
        self, z_grid, zbins, zmeans
    ):
        """If the per-bin bias values are all equal, the interpolant is flat."""
        const_bias = np.array([2.0, 2.0, 2.0])
        out = wl.build_galaxy_bias_2d_arr(
            const_bias, zmeans, None, zbins, z_grid, bias_model='linint'
        )
        assert out.shape == (len(z_grid), zbins)
        np.testing.assert_allclose(out, 2.0)
        # linint gives the *same* curve to every zbin column
        np.testing.assert_allclose(out[:, 0], out[:, 1])

    def test_step_wise_matches_stepwise_bias(self, z_grid, zbins, zmeans):
        gal_bias_vs_zmean = np.array([1.2, 1.5, 1.8])
        z_edges = np.array([0.0, 0.7, 1.4, 2.1])
        out = wl.build_galaxy_bias_2d_arr(
            gal_bias_vs_zmean, zmeans, z_edges, zbins, z_grid, bias_model='step-wise'
        )
        expected = np.array(
            [wl.stepwise_bias(z, gal_bias_vs_zmean, z_edges) for z in z_grid]
        )
        np.testing.assert_allclose(out[:, 0], expected)
        # same curve broadcast to every zbin column
        np.testing.assert_allclose(out[:, 0], out[:, -1])

    def test_polynomial_uses_fit_function(self, z_grid, zbins, zmeans):
        gal_bias_vs_zmean = np.array([1.2, 1.5, 1.8])
        out = wl.build_galaxy_bias_2d_arr(
            gal_bias_vs_zmean,
            zmeans,
            None,
            zbins,
            z_grid,
            bias_model='polynomial',
            bias_fit_function=wl.b_of_z_analytical,
            kwargs_bias_fit_function={},
        )
        expected = wl.b_of_z_analytical(z_grid)
        np.testing.assert_allclose(out[:, 0], expected)
        np.testing.assert_allclose(out[:, 0], out[:, -1])

    def test_invalid_bias_model_raises(self, z_grid, zbins, zmeans):
        gal_bias_vs_zmean = np.array([1.2, 1.5, 1.8])
        with pytest.raises(ValueError, match='bias_model'):
            wl.build_galaxy_bias_2d_arr(
                gal_bias_vs_zmean, zmeans, None, zbins, z_grid, bias_model='bogus'
            )


# ----------------------------------------------------------------------------- #
# build_ia_bias_1d_arr
# ----------------------------------------------------------------------------- #
class TestBuildIaBias1dArr:
    """Tests for the intrinsic alignment bias, using a cheap CCL cosmology."""

    @pytest.fixture
    def z_grid(self):
        return np.linspace(0.01, 2.5, 50)

    @pytest.fixture
    def ia_dict(self):
        return {'Aia': 1.0, 'eIA': 0.0, 'bIA': 0.0, 'z_pivot_IA': 0.5, 'CIA': 0.0134}

    def test_shape_and_finite(self, cosmo, z_grid, ia_dict):
        out = wl.build_ia_bias_1d_arr(z_grid, cosmo, ia_dict, lumin_ratio_2d_arr=None)
        assert out.shape == z_grid.shape
        assert np.all(np.isfinite(out))

    def test_sign_convention(self, cosmo, z_grid, ia_dict):
        """With positive Aia, CIA, Omega_m, the IA bias is negative (see docstring)."""
        out = wl.build_ia_bias_1d_arr(z_grid, cosmo, ia_dict, lumin_ratio_2d_arr=None)
        assert np.all(out < 0)

    def test_output_f_ia_of_z_flag(self, cosmo, z_grid, ia_dict):
        ia_bias, f_ia_of_z = wl.build_ia_bias_1d_arr(
            z_grid, cosmo, ia_dict, lumin_ratio_2d_arr=None, output_F_IA_of_z=True
        )
        assert ia_bias.shape == z_grid.shape
        assert f_ia_of_z.shape == z_grid.shape

    def test_beta_ia_nonzero_without_lumin_ratio_raises(self, cosmo, z_grid, ia_dict):
        ia_dict = {**ia_dict, 'bIA': 1.0}
        with pytest.raises(AssertionError):
            wl.build_ia_bias_1d_arr(z_grid, cosmo, ia_dict, lumin_ratio_2d_arr=None)


# ----------------------------------------------------------------------------- #
# get_luminosity_ratio_interpolator
# ----------------------------------------------------------------------------- #
class TestGetLuminosityRatioInterpolator:
    """Tests for the luminosity-ratio interpolator factory."""

    def test_none_gives_constant_one(self):
        f = wl.get_luminosity_ratio_interpolator(None)
        assert f(0.5) == 1
        assert f(np.array([0.1, 5.0])) == 1

    def test_array_input_interpolates(self):
        arr = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        f = wl.get_luminosity_ratio_interpolator(arr)
        np.testing.assert_allclose(f(0.5), 1.5)
        np.testing.assert_allclose(f(1.5), 2.5)


# ----------------------------------------------------------------------------- #
# get_z_means / get_z_effective_isaac
# ----------------------------------------------------------------------------- #
class TestGetZMeans:
    """Tests for the first-moment (mean redshift) helper."""

    def test_uniform_kernel_gives_midpoint(self):
        zgrid = np.linspace(0, 10, 5000)
        kernel = np.zeros((len(zgrid), 2))
        mask = (zgrid >= 2) & (zgrid <= 8)
        kernel[mask, :] = 1.0
        z_means = wl.get_z_means(zgrid, kernel)
        np.testing.assert_allclose(z_means, [5.0, 5.0], atol=1e-2)

    def test_requires_2d_kernel(self):
        zgrid = np.linspace(0, 10, 10)
        with pytest.raises(AssertionError):
            wl.get_z_means(zgrid, np.ones(10))


class TestGetZEffectiveIsaac:
    """Tests for the 10%-of-peak-threshold median redshift helper."""

    def test_narrow_gaussian_recovers_center(self):
        zgrid = np.linspace(0, 10, 5000)
        g = np.exp(-0.5 * ((zgrid - 3.0) / 0.3) ** 2)
        n_of_z = np.column_stack([g, g])
        z_eff = wl.get_z_effective_isaac(zgrid, n_of_z)
        np.testing.assert_allclose(z_eff, [3.0, 3.0], atol=0.05)


# ----------------------------------------------------------------------------- #
# shift_nz
# ----------------------------------------------------------------------------- #
class TestShiftNz:
    """Tests for the n(z) redshift-shifting helper."""

    @pytest.fixture
    def z_grid(self):
        return np.linspace(0, 5, 2000)

    def test_zero_shift_is_identity(self, z_grid):
        nz = wl.nz_smail(z_grid, 0.6, 1.0)[:, None]
        shifted = wl.shift_nz(z_grid, nz, dz_shifts=[0.0], normalize=False)
        np.testing.assert_allclose(shifted[:, 0], nz[:, 0])

    def test_shift_moves_mean_by_delta(self, z_grid):
        nz = wl.nz_smail(z_grid, 0.6, 1.0)[:, None]
        dz = 0.05
        shifted = wl.shift_nz(z_grid, nz, dz_shifts=[dz], normalize=False)
        mean_before = wl.get_z_means(z_grid, nz)
        mean_after = wl.get_z_means(z_grid, shifted)
        np.testing.assert_allclose(mean_after - mean_before, dz, atol=1e-3)

    def test_wrong_length_dz_shifts_raises(self, z_grid):
        nz = wl.nz_smail(z_grid, 0.6, 1.0)[:, None]
        with pytest.raises(AssertionError):
            wl.shift_nz(z_grid, nz, dz_shifts=[0.0, 0.1], normalize=False)

    def test_large_shift_warns(self, z_grid):
        nz = wl.nz_smail(z_grid, 0.6, 1.0)[:, None]
        with pytest.warns(UserWarning, match='dz_shifts are quite large'):
            wl.shift_nz(z_grid, nz, dz_shifts=[0.5], normalize=False)
