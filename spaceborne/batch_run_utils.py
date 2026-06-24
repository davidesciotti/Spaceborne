import os
import subprocess
from copy import deepcopy

import yaml


def create_paths(ROOT, job_name):
    sb_root_path = f'{ROOT}/Spaceborne'
    io_path = f'{ROOT}/DATA/Spaceborne_jobs_IO/{job_name}'
    configs_path = f'{io_path}/generated_configs'
    return {
        'sb_root_path': sb_root_path,
        'configs_path': configs_path,
        'io_path': io_path,
    }


def assert_repo_branch(repo_path: str, expected_branch: str) -> None:
    is_git_repo = subprocess.run(
        ['git', '-C', repo_path, 'rev-parse', '--is-inside-work-tree'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if is_git_repo != 'true':
        raise RuntimeError(f'{repo_path} is not a git repository')

    current_branch = subprocess.run(
        ['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    if current_branch != expected_branch:
        raise RuntimeError(
            f'Repo at {repo_path} is on branch "{current_branch}", expected "{expected_branch}".'
        )


def generate_zipped_configs(base_config: dict, changes_list: list[dict]) -> list:
    """Apply changes to a base config and return the list of resulting configs."""
    configs = []

    def apply_changes(target_dict, changes_dict):
        for key, value in changes_dict.items():
            if isinstance(value, dict):
                if key not in target_dict or not isinstance(target_dict[key], dict):
                    if key in target_dict:
                        print(f'Warning: Overwriting non-dict value at key "{key}"')
                    target_dict[key] = {}
                apply_changes(target_dict[key], value)
            else:
                target_dict[key] = value

    for change_set in changes_list:
        config = deepcopy(base_config)

        apply_changes(config, change_set)
        configs.append(config)

    return configs


def save_configs_to_yaml(configs: list, filenames: list) -> None:
    """Save each config to a YAML file with the provided filenames."""
    assert len(configs) == len(filenames), (
        'Number of configs must match number of filenames'
    )

    for config, path in zip(configs, filenames, strict=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


def run_spaceborne(
    yaml_files: list[str], sb_root_path: str, continue_on_error: bool
) -> None:
    """Run Spaceborne for a list of YAML config paths.

    If continue_on_error is True, a failing job won't stop the run; the
    failed configs are collected and printed together at the end.
    """
    original_dir = os.getcwd()
    os.chdir(sb_root_path)
    failed: list[str] = []

    print(f'🚜🚜🚜 Starting Spaceborne jobs for {len(yaml_files)} configs 🚜🚜🚜')

    try:
        for path in yaml_files:
            print(f'\n🚜 Running job with config: 🚜\n{path}')
            try:
                subprocess.run(['python', 'main.py', '--config', path], check=True)
            except subprocess.CalledProcessError as exc:
                if not continue_on_error:
                    raise
                print(
                    f'\n❌ Job failed (exit code {exc.returncode}) for config:\n{path}'
                )
                failed.append(path)
    finally:
        os.chdir(original_dir)

    if failed:
        print(f'\n❌❌❌ {len(failed)} config(s) failed:')
        for path in failed:
            print(f'  - {path}')
