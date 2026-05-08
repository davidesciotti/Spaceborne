import os
import subprocess
import time
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


def run_spaceborne(yaml_files: list[str], sb_root_path: str) -> None:
    """Run Spaceborne for a list of YAML config paths."""
    original_dir = os.getcwd()
    os.chdir(sb_root_path)
    try:
        for path in yaml_files:
            print(f'\n🧮🧮🧮 Running job with config:\n{path}')
            subprocess.run(['python', 'main.py', '--config', path], check=True)
    finally:
        os.chdir(original_dir)


# ! SETTINGS START
ROOT = '/home/sciotti/code'
DATA_ROOT = '/data/sciotti'
sb_root_path = f'{ROOT}/Spaceborne'
base_cfg_path = '/data/sciotti/DATA/Spaceborne_jobs/cov_validation_2026/cov_FS2_jose/run_config_live_cone_vis24.5_h22.yaml'
create_output_folders = False
# ! SETTINGS END

start_time = time.time()

with open(base_cfg_path) as f:
    base_cfg = yaml.safe_load(f)


configs_to_run = []
for partial_sky_method, sample_cov, output_folder in zip(
    ['NaMaster', 'Knox'],
    [False, True],
    ['output_cone_vis24.5_h22_nmt', 'output_cone_vis24.5_h22_sample'],
    strict=True,
):
    out_path = (
        f'/data/sciotti/DATA/Spaceborne_jobs/cov_validation_2026/'
        f'cov_FS2_jose/{output_folder}'
    )
    configs_to_run.append(
        {
            'covariance': {'partial_sky_method': partial_sky_method},
            'misc': {'output_path': out_path},
            'sample_covariance': {'compute_sample_cov': sample_cov},
        }
    )
    if create_output_folders:
        os.makedirs(out_path, exist_ok=True)


# assign yaml filenames based on output path
yaml_filenames = [
    f'{sb_root_path}/{cfg["misc"]["output_path"].split("/")[-1]}.yaml'
    for cfg in configs_to_run
]

# === Apply changes to base config ===
configs = generate_zipped_configs(base_cfg, configs_to_run)

# === Save configs ===
save_configs_to_yaml(configs, yaml_filenames)

# === Run all ===
run_spaceborne(yaml_filenames, sb_root_path)

time_hours = (time.time() - start_time) / 3600
print(f'\n✅ All Spaceborne runs finished in {time_hours:.2f} hours!')
