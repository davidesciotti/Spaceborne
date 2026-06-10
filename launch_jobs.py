import os
import time

import yaml

from spaceborne import batch_run_utils

# ! SETTINGS START
ROOT = '/home/sciotti/code'
DATA_ROOT = '/data/sciotti'
sb_root_path = f'{ROOT}/Spaceborne'
base_cfg_path = f'{sb_root_path}/config.yaml'
create_output_folders = True
# ! SETTINGS END

start_time = time.time()

with open(base_cfg_path) as f:
    base_cfg = yaml.safe_load(f)


configs_to_run = []
for ell_min in [10, 20, 40, 100]:
    for cov_type in ['decoupled', 'coupled']:
        for partial_sky_method, sample_cov in zip(
            ['NaMaster', 'Knox'], [False, True], strict=True
        ):
            out_path = (
                f'{DATA_ROOT}/DATA/Spaceborne_jobs/cov_validation_2026/'
                f'v2_psky{partial_sky_method}_sample{sample_cov}_{cov_type}_ellmin{ell_min}'
            )
            configs_to_run.append(
                {
                    'binning': {'ell_min': ell_min},
                    'covariance': {
                        'partial_sky_method': partial_sky_method,
                        'cov_type': cov_type,
                    },
                    'misc': {'output_path': out_path},
                    'sample_covariance': {
                        'compute_sample_cov': sample_cov,
                        'which_cls': 'healpy',
                        'nreal': 20000,
                        'fix_seed': True,
                    },
                }
            )
            if create_output_folders:
                os.makedirs(out_path, exist_ok=True)


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
print(f'\n✅ All Spaceborne runs finished in {time_hours:.2f} hours!')
