import os
import subprocess
from copy import deepcopy

import yaml


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

    try:
        for path in yaml_files:
            print(f'\n🧮🧮🧮 Running job with config:\n{path}')
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
