import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path

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
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def run_spaceborne(
    yaml_files: list[str],
    output_paths: list[str],
    sb_root_path: str,
    fail_fast: bool = False,
) -> list[dict]:
    """Run Spaceborne jobs and collect results with one log per output directory."""
    assert len(yaml_files) == len(output_paths), (
        'Number of YAML files must match number of output paths'
    )
    original_dir = os.getcwd()
    os.chdir(sb_root_path)
    results = []
    try:
        for i, (path, output_path) in enumerate(
            zip(yaml_files, output_paths, strict=True), start=1
        ):
            job_name = Path(path).stem
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / f'{i:02d}_{job_name}.log'
            run_start = time.time()

            print(f'\n🧮 Running job {i}/{len(yaml_files)} with config:\n{path}')
            with open(log_path, 'w') as logf:
                proc = subprocess.run(
                    ['python', 'main.py', '--config', path],
                    check=False,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            run_time_s = time.time() - run_start
            ok = proc.returncode == 0
            results.append(
                {
                    'config': path,
                    'ok': ok,
                    'returncode': proc.returncode,
                    'log': str(log_path),
                    'duration_s': run_time_s,
                }
            )
            status = 'OK' if ok else f'FAIL ({proc.returncode})'
            print(f'   -> {status} | {run_time_s:.1f}s | log: {log_path}')

            if fail_fast and not ok:
                break
    finally:
        os.chdir(original_dir)
    return results


# ! SETTINGS START
ROOT = '/home/sciotti/code'
DATA_ROOT = '/data/sciotti'
sb_root_path = f'{ROOT}/Spaceborne'
base_cfg_path = '/data/sciotti/DATA/Spaceborne_jobs/DR1_area_selection/configs/config_easy_3zbins.yaml'
create_output_folders = True
fail_fast = False  # If True, stop at first error
# ! SETTINGS END

start_time = time.time()

with open(base_cfg_path) as f:
    base_cfg = yaml.safe_load(f)


# get footprint names
footprint_filenames = sorted(
    Path('/data/sciotti/DATA/Spaceborne_jobs/DR1_area_selection/input/footprints').glob(
        '*.fits'
    )
)


configs_to_run = []
for partial_sky_method in ['NaMaster', 'ensemble']:
    for footprint_path in footprint_filenames:
        # for apodize in [False, True]:

        footprint_name = Path(footprint_path).stem

        # ! remember to update this, and careful of create_output_folders
        out_path = (
            '/data/sciotti/DATA/Spaceborne_jobs/DR1_area_selection'
            f'/output/{footprint_name}_psky{partial_sky_method}'
        )
        if create_output_folders:
            os.makedirs(out_path, exist_ok=True)

        configs_to_run.append(
            {
                'mask': {
                    'apodize': False,
                    'LL': {'footprint_filename': str(footprint_path)},
                    'GG': {'footprint_filename': str(footprint_path)},
                },
                'misc': {'output_path': out_path},
                'covariance': {'partial_sky_method': partial_sky_method},
            }
        )


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
output_paths = [cfg['misc']['output_path'] for cfg in configs_to_run]
results = run_spaceborne(
    yaml_filenames, output_paths, sb_root_path, fail_fast=fail_fast
)

time_hours = (time.time() - start_time) / 3600
failed = [res for res in results if not res['ok']]

print('\n=== Run summary ===')
for res in results:
    status = 'OK' if res['ok'] else f'FAIL ({res["returncode"]})'
    print(f'{status:12} {res["config"]} | {res["duration_s"]:.1f}s | {res["log"]}')

if failed:
    print(f'\n❌ {len(failed)}/{len(results)} jobs failed in {time_hours:.2f} hours.')
    raise SystemExit(1)

print(f'\n✅ All {len(results)} Spaceborne runs finished in {time_hours:.2f} hours.')
