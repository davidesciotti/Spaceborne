"""Unit tests for spaceborne.batch_run_utils module.

run_spaceborne is not covered here since it launches subprocesses.
"""

import subprocess

import pytest
import yaml

from spaceborne import batch_run_utils


class TestCreatePaths:
    """Tests for create_paths."""

    def test_builds_expected_paths(self):
        paths = batch_run_utils.create_paths('/root', 'my_job')

        assert paths['sb_root_path'] == '/root/Spaceborne'
        assert paths['io_path'] == '/root/DATA/Spaceborne_jobs_IO/my_job'
        assert paths['configs_path'] == (
            '/root/DATA/Spaceborne_jobs_IO/my_job/generated_configs'
        )

    def test_returns_only_expected_keys(self):
        paths = batch_run_utils.create_paths('/root', 'job')
        assert set(paths) == {'sb_root_path', 'configs_path', 'io_path'}


class TestGenerateZippedConfigs:
    """Tests for generate_zipped_configs (zipped override semantics)."""

    @pytest.fixture
    def base_config(self):
        return {
            'covariance': {
                'partial_sky_method': 'Knox',
                'G': True,
                'n_probes': 2
            },
            'precision': {'spin0': False, 'iNKA': True},
            'misc': {'output_path': '/base/output'}
        }  # fmt: skip

    def test_n_changes_produces_n_configs(self, base_config):
        changes_list = [
            {'covariance': {'partial_sky_method': 'Knox'}},
            {'covariance': {'partial_sky_method': 'NaMaster'}},
            {'covariance': {'partial_sky_method': 'ensemble'}}
        ]  # fmt: skip

        configs = batch_run_utils.generate_zipped_configs(base_config, changes_list)

        assert len(configs) == 3

    def test_nested_key_is_overridden(self, base_config):
        changes_list = [{'precision': {'spin0': True}}]

        configs = batch_run_utils.generate_zipped_configs(base_config, changes_list)

        assert configs[0]['precision']['spin0'] is True
        # sibling key untouched
        assert configs[0]['precision']['iNKA'] is True

    def test_base_config_is_preserved_across_configs(self, base_config):
        """Each output config is an independent deepcopy: mutating one config's
        nested dict must not affect the base config or the other configs."""
        changes_list = [
            {'covariance': {'partial_sky_method': 'NaMaster'}},
            {'covariance': {'partial_sky_method': 'ensemble'}}
        ]  # fmt: skip

        configs = batch_run_utils.generate_zipped_configs(base_config, changes_list)

        assert base_config['covariance']['partial_sky_method'] == 'Knox'
        assert configs[0]['covariance']['partial_sky_method'] == 'NaMaster'
        assert configs[1]['covariance']['partial_sky_method'] == 'ensemble'

        # mutating one resulting config must not leak into the others/base
        configs[0]['covariance']['G'] = False
        assert configs[1]['covariance']['G'] is True
        assert base_config['covariance']['G'] is True

    def test_unrelated_keys_are_unaffected(self, base_config):
        changes_list = [{'covariance': {'partial_sky_method': 'NaMaster'}}]

        configs = batch_run_utils.generate_zipped_configs(base_config, changes_list)

        assert configs[0]['misc']['output_path'] == '/base/output'
        assert configs[0]['covariance']['n_probes'] == 2

    def test_multiple_keys_in_single_change_set(self, base_config):
        changes_list = [
            {
                'covariance': {'partial_sky_method': 'NaMaster'},
                'precision': {'spin0': True}
            }
        ]  # fmt: skip

        configs = batch_run_utils.generate_zipped_configs(base_config, changes_list)

        assert configs[0]['covariance']['partial_sky_method'] == 'NaMaster'
        assert configs[0]['precision']['spin0'] is True

    def test_new_top_level_key_is_created(self, base_config):
        changes_list = [{'new_section': {'new_key': 42}}]

        configs = batch_run_utils.generate_zipped_configs(base_config, changes_list)

        assert configs[0]['new_section']['new_key'] == 42

    def test_overwriting_non_dict_value_with_dict(self, base_config, capsys):
        """If a change tries to set a nested dict where the base config
        currently has a scalar, the scalar is overwritten wholesale (with a
        warning printed to stdout)."""
        changes_list = [{'misc': {'output_path': {'nested': 'value'}}}]

        configs = batch_run_utils.generate_zipped_configs(base_config, changes_list)

        assert configs[0]['misc']['output_path'] == {'nested': 'value'}
        assert 'Warning' in capsys.readouterr().out

    def test_empty_changes_list_produces_no_configs(self, base_config):
        configs = batch_run_utils.generate_zipped_configs(base_config, [])
        assert configs == []


class TestSaveConfigsToYaml:
    """Tests for save_configs_to_yaml."""

    def test_writes_and_reloads_configs(self, tmp_path):
        configs = [
            {'a': 1, 'b': {'c': 2.5, 'd': 'text'}},
            {'a': 2, 'b': {'c': 3.5, 'd': 'other'}}
        ]  # fmt: skip
        filenames = [
            str(tmp_path / 'cfg_0.yaml'),
            str(tmp_path / 'cfg_1.yaml')
        ]  # fmt: skip

        batch_run_utils.save_configs_to_yaml(configs, filenames)

        for config, filename in zip(configs, filenames, strict=True):
            with open(filename) as f:
                reloaded = yaml.safe_load(f)
            assert reloaded == config

    def test_creates_missing_parent_directories(self, tmp_path):
        nested_path = tmp_path / 'nested' / 'dirs' / 'cfg.yaml'

        batch_run_utils.save_configs_to_yaml([{'a': 1}], [str(nested_path)])

        assert nested_path.is_file()

    def test_mismatched_lengths_raise(self, tmp_path):
        configs = [{'a': 1}, {'a': 2}]
        filenames = [str(tmp_path / 'cfg_0.yaml')]

        with pytest.raises(AssertionError, match='must match'):
            batch_run_utils.save_configs_to_yaml(configs, filenames)


class TestAssertRepoBranch:
    """Tests for assert_repo_branch, using disposable tmp_path git repos so the
    tests don't depend on the checked-out branch/HEAD state of the actual
    Spaceborne repo (which may be a detached HEAD in CI)."""

    @staticmethod
    def _make_repo(path, branch):
        subprocess.run(['git', 'init', '-q', '-b', branch, str(path)], check=True)
        subprocess.run(
            ['git', '-C', str(path), 'config', 'user.email', 'test@test.com'],
            check=True
        )  # fmt: skip
        subprocess.run(
            ['git', '-C', str(path), 'config', 'user.name', 'test'], check=True
        )
        (path / 'f.txt').write_text('hello')
        subprocess.run(['git', '-C', str(path), 'add', 'f.txt'], check=True)
        subprocess.run(
            ['git', '-C', str(path), 'commit', '-q', '-m', 'init'], check=True
        )

    def test_matching_branch_does_not_raise(self, tmp_path):
        self._make_repo(tmp_path, 'my-branch')
        batch_run_utils.assert_repo_branch(str(tmp_path), 'my-branch')

    def test_mismatched_branch_raises(self, tmp_path):
        self._make_repo(tmp_path, 'my-branch')
        with pytest.raises(RuntimeError, match='expected'):
            batch_run_utils.assert_repo_branch(str(tmp_path), 'other-branch')

    def test_non_git_directory_raises_called_process_error(self, tmp_path):
        with pytest.raises(subprocess.CalledProcessError):
            batch_run_utils.assert_repo_branch(str(tmp_path), 'any-branch')
