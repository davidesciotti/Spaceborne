"""Unit tests for spaceborne.config_checker module.

These tests exercise ``SpaceborneConfigChecker`` against the repo's own
``config.yaml`` (the reference config, documented inline in the file itself)
plus a set of targeted invalid mutations.
"""

import copy
import os

import pytest
import yaml

from spaceborne import config_checker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, 'config.yaml')


def _load_raw_config():
    """Load config.yaml exactly as main.py's ``load_config`` does."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _apply_main_hardcoded_overrides(cfg):
    """Mirror the "hardcoded options" block in main.py (see the section between
    '! START HARDCODED OPTIONS/PARAMETERS' and '! END HARDCODED OPTIONS/PARAMETERS',
    roughly lines 296-388) that main.py injects into cfg *before* instantiating
    SpaceborneConfigChecker. These keys are intentionally absent from config.yaml,
    since the comment there states they "should not be visible to the user".
    """
    cfg['covariance']['n_probes'] = 2
    cfg['precision']['n_sub'] = 16
    cfg['precision']['n_bisec_max'] = 128
    cfg['precision']['rel_acc'] = 1.0e-4
    cfg['precision']['boost_bessel'] = True
    cfg['precision']['verbose'] = True
    cfg['covariance'].setdefault('G_code', 'Spaceborne')
    cfg['covariance'].setdefault('SSC_code', 'Spaceborne')
    cfg['covariance'].setdefault('cNG_code', 'PyCCL')
    cfg['probe_selection']['Psigl'] = False
    cfg['probe_selection']['Psigg'] = False
    cfg['covariance']['which_sigma2_b'] = 'from_input_mask'
    cfg['covariance']['sigma2_b_int_method'] = 'fft'
    return cfg


@pytest.fixture
def valid_cfg(tmp_path):
    """A fully valid config, built from config.yaml plus main.py's runtime
    overrides. The GG footprint file is redirected to a small dummy file in
    tmp_path, since the path in config.yaml points outside the repo (to a data
    directory that may not exist on every machine/CI runner) and
    ``check_mask`` only requires ``os.path.isfile`` to succeed.
    """
    cfg = copy.deepcopy(_load_raw_config())
    cfg = _apply_main_hardcoded_overrides(cfg)

    # The 'OneCovariance' section is optional in config.yaml; ensure it exists
    # (with valid defaults) so the OneCovariance-specific tests can toggle its
    # keys regardless of whether the reference config ships the section.
    oc_cfg = cfg.setdefault('OneCovariance', {})
    oc_cfg.setdefault('path_to_oc_executable', '../OneCovariance/covariance.py')
    oc_cfg.setdefault('consistency_checks', False)
    oc_cfg.setdefault('oc_output_filename', 'cov_oc')
    oc_cfg.setdefault('compare_against_oc', False)

    fake_footprint = tmp_path / 'footprint.fits'
    fake_footprint.write_text('dummy')
    cfg['mask']['GG']['footprint_filename'] = str(fake_footprint)

    return cfg


def _zbins(cfg):
    return len(cfg['nz']['ngal_lenses'])


class TestRepoConfigYaml:
    """Tests using the repo's own config.yaml as the reference config."""

    def test_bare_config_yaml_is_missing_runtime_only_keys(self):
        """config.yaml alone (without main.py's runtime overrides) is missing
        keys such as 'G_code' that are only injected by main.py at runtime.
        This documents the current, intentional main.py/config.yaml split
        rather than being a bug: check_types should fail with a KeyError
        before even reaching an assertion.
        """
        cfg = copy.deepcopy(_load_raw_config())
        checker = config_checker.SpaceborneConfigChecker(cfg, _zbins(cfg))
        with pytest.raises(KeyError):
            checker.check_types()

    def test_repo_config_with_main_overrides_passes_all_checks(self, valid_cfg):
        """The full, effective config (as main.py would build it) should pass
        every check with no exceptions."""
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        checker.run_all_checks()

    def test_check_types_passes(self, valid_cfg):
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        checker.check_types()


class TestCheckBNTTransform:
    """Tests for check_BNT_transform."""

    def test_bnt_in_real_space_raises(self, valid_cfg):
        valid_cfg['covariance']['BNT_transform'] = True
        valid_cfg['probe_selection']['space'] = 'real'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_BNT_transform()

    def test_bnt_in_harmonic_space_ok(self, valid_cfg):
        valid_cfg['covariance']['BNT_transform'] = True
        valid_cfg['probe_selection']['space'] = 'harmonic'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        checker.check_BNT_transform()


class TestCheckKEApproximation:
    """Tests for check_KE_approximation."""

    def test_ke_approx_with_disallowed_sigma2_b_raises(self, valid_cfg):
        valid_cfg['precision']['use_KE_approximation'] = True
        valid_cfg['covariance']['SSC_code'] = 'Spaceborne'
        valid_cfg['covariance']['which_sigma2_b'] = 'full_curved_sky'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_KE_approximation()

    def test_no_ke_approx_with_disallowed_sigma2_b_raises(self, valid_cfg):
        valid_cfg['precision']['use_KE_approximation'] = False
        valid_cfg['covariance']['SSC_code'] = 'Spaceborne'
        valid_cfg['covariance']['which_sigma2_b'] = 'flat_sky'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_KE_approximation()


class TestCheckProbeSelection:
    """Tests for check_probe_selection."""

    def test_invalid_key_raises(self, valid_cfg):
        valid_cfg['probe_selection']['not_a_real_key'] = True
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_probe_selection()

    def test_invalid_space_raises(self, valid_cfg):
        valid_cfg['probe_selection']['space'] = 'not_a_space'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(ValueError, match='must be either'):
            checker.check_probe_selection()

    def test_real_space_needs_at_least_one_probe(self, valid_cfg):
        valid_cfg['probe_selection']['space'] = 'real'
        valid_cfg['probe_selection']['xip'] = False
        valid_cfg['probe_selection']['xim'] = False
        valid_cfg['probe_selection']['gt'] = False
        valid_cfg['probe_selection']['w'] = False
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_probe_selection()


class TestCheckCov:
    """Tests for check_cov."""

    def test_invalid_triu_tril_raises(self, valid_cfg):
        valid_cfg['covariance']['triu_tril'] = 'not_triu_or_tril'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_cov()

    def test_invalid_row_col_major_raises(self, valid_cfg):
        valid_cfg['covariance']['row_col_major'] = 'not_a_valid_order'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_cov()

    def test_namaster_in_real_space_raises(self, valid_cfg):
        valid_cfg['covariance']['partial_sky_method'] = 'NaMaster'
        valid_cfg['probe_selection']['space'] = 'real'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_cov()


class TestCheckMask:
    """Tests for check_mask."""

    def test_invalid_geometry_raises(self, valid_cfg):
        valid_cfg['mask']['GG']['geometry'] = 'not_a_geometry'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_mask()

    def test_missing_footprint_file_raises(self, valid_cfg):
        """The file-existence check only runs for geometry='footprint_file',
        so set it explicitly rather than relying on config.yaml's choice."""
        valid_cfg['mask']['GG']['geometry'] = 'footprint_file'
        valid_cfg['mask']['GG']['footprint_filename'] = '/no/such/file.fits'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError, match='not found'):
            checker.check_mask()


class TestCheckNmt:
    """Tests for check_nmt."""

    def test_coupled_gaussian_requires_namaster_or_ensemble(self, valid_cfg):
        valid_cfg['covariance']['cov_type'] = 'coupled'
        valid_cfg['covariance']['G'] = True
        valid_cfg['covariance']['partial_sky_method'] = 'Knox'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_nmt()

    def test_namaster_incompatible_with_ref_cut_binning(self, valid_cfg):
        valid_cfg['covariance']['partial_sky_method'] = 'NaMaster'
        valid_cfg['covariance']['G_code'] = 'Spaceborne'
        valid_cfg['binning']['binning_type'] = 'ref_cut'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_nmt()

    def test_namaster_requires_spaceborne_g_code(self, valid_cfg):
        valid_cfg['covariance']['partial_sky_method'] = 'NaMaster'
        valid_cfg['covariance']['G_code'] = 'PyCCL'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(ValueError):
            checker.check_nmt()


class TestCheckOnecov:
    """Tests for check_onecov."""

    def test_magnification_bias_with_oc_raises(self, valid_cfg):
        valid_cfg['C_ell']['has_magnification_bias'] = True
        valid_cfg['OneCovariance']['compare_against_oc'] = True
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(ValueError):
            checker.check_onecov()

    def test_ssc_without_ke_and_oc_raises(self, valid_cfg):
        valid_cfg['covariance']['SSC'] = True
        valid_cfg['precision']['use_KE_approximation'] = False
        valid_cfg['covariance']['SSC_code'] = 'OneCovariance'
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(ValueError):
            checker.check_onecov()


class TestCheckMisc:
    """Tests for check_misc."""

    def test_wrong_n_probes_raises(self, valid_cfg):
        valid_cfg['covariance']['n_probes'] = 3
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_misc()


class TestCheckNz:
    """Tests for check_nz."""

    def test_negative_ngal_sources_raises(self, valid_cfg):
        valid_cfg['nz']['ngal_sources'] = [-1.0, 2.0, 3.0]
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_nz()

    def test_mismatched_dz_shifts_raise(self, valid_cfg):
        valid_cfg['nz']['shift_nz'] = True
        valid_cfg['nz']['dzWL'] = [0.1, 0.0, 0.0]
        valid_cfg['nz']['dzGC'] = [0.2, 0.0, 0.0]
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_nz()


class TestCheckLists:
    """Tests for check_lists."""

    def test_wrong_length_galaxy_bias_fit_coeff_raises(self, valid_cfg):
        valid_cfg['C_ell']['galaxy_bias_fit_coeff'] = [1.0, 2.0, 3.0]
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_lists()

    def test_wrong_length_mult_shear_bias_raises(self, valid_cfg):
        valid_cfg['C_ell']['mult_shear_bias'] = [0.0, 0.0]
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_lists()


class TestCheckTypesMutations:
    """Tests for check_types with deliberately wrong types."""

    def test_non_float_cosmology_param_raises(self, valid_cfg):
        valid_cfg['cosmology']['Om'] = 1  # int instead of float
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_types()

    def test_non_bool_covariance_g_raises(self, valid_cfg):
        valid_cfg['covariance']['G'] = 'True'  # str instead of bool
        checker = config_checker.SpaceborneConfigChecker(valid_cfg, _zbins(valid_cfg))
        with pytest.raises(AssertionError):
            checker.check_types()
