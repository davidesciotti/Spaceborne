import os
import time

import yaml

from spaceborne import batch_run_utils

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

# Apply changes to base config
configs = batch_run_utils.generate_zipped_configs(base_cfg, configs_to_run)

# Save configs
batch_run_utils.save_configs_to_yaml(configs, yaml_filenames)

# Run all
batch_run_utils.run_spaceborne(yaml_filenames, sb_root_path)

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
