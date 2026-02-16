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

    for config, path in zip(configs, filenames):
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
ROOT = '/home/cosmo/davide.sciotti/data'
sb_root_path = f'{ROOT}/Spaceborne'
base_cfg_path = f'{sb_root_path}/config.yaml'
create_output_folders = False
# ! SETTINGS END

with open(base_cfg_path) as f:
    base_cfg = yaml.safe_load(f)

configs_to_run = []
for nongauss in (False, True):
    # this runs Knox decoupled, NaMaster decoupled, NaMaster coupled
    for partial_sky_method, coupled_cov in zip(
        ['Knox', 'NaMaster', 'NaMaster'], [False, False, True]
    ):
        out_path = (
            f'{ROOT}/DATA/Spaceborne_jobs/TR1_cov/'
            f'psky{partial_sky_method}_nongauss{nongauss}_coupled{coupled_cov}'
        )
        configs_to_run.append(
            {
                'covariance': {
                    'partial_sky_method': partial_sky_method,
                    'coupled_cov': coupled_cov,
                    'SSC': nongauss,
                    'cNG': nongauss,
                },
                'misc': {'output_path': out_path},
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

print('\n✅ All Spaceborne runs finished!')
