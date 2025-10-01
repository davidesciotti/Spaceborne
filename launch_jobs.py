import os
import subprocess
from copy import deepcopy

import yaml


def generate_zipped_configs(base_config: dict, changes_list: list[dict]) -> list:
    """Apply changes to a base config and return the list of resulting configs."""
    configs = []

    for change_set in changes_list:
        config = deepcopy(base_config)

        def apply_changes(target_dict, changes_dict):
            for key, value in changes_dict.items():
                if isinstance(value, dict):
                    if key not in target_dict or not isinstance(target_dict[key], dict):
                        target_dict[key] = {}
                    apply_changes(target_dict[key], value)
                else:
                    target_dict[key] = value

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


def run_benchmarks(yaml_files: list[str], sb_root_path: str) -> None:
    """Run Spaceborne for a list of YAML config paths."""
    original_dir = os.getcwd()
    os.chdir(sb_root_path)
    try:
        for path in yaml_files:
            print(f'\n 🛰️ Running: {path}')
            subprocess.run(['python', 'main.py', '--config', path], check=True)
    finally:
        os.chdir(original_dir)


# CONFIGURE BASE
ROOT = '/home/cosmo/davide.sciotti/data'
sb_root_path = f'{ROOT}/Spaceborne'

with open(f'{sb_root_path}/config.yaml') as f:
    base_cfg = yaml.safe_load(f)

neff_src_rr2 = [2.5202727, 5.1424327, 7.8001622, 8.5047784, 6.6663741, 3.8253194]
neff_lns_rr2 = [2.3604811, 1.7091029, 1.6091857, 1.1883694, 0.3216123, 0.0785469]


# DEFINE PRODUCTION CONFIGURATIONS
configs_to_run = [
    # ! nzRR2_EP06_nbl32_ellmax1500
    # {
    #     'nz': {
    #         'nz_sources_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_SHE_tombins_unitweights_nz_SOMbin_C2020z.fits',
    #         'nz_lenses_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_POS_tombins_unitweights_nz_SOMbin_C2020z.fits',
    #         'ngal_sources': neff_src_rr2,
    #         'ngal_lenses': neff_lns_rr2,
    #         'smooth_nz': True,
    #     },  # fmt: skip
    #     'ell_binning': {'ell_max_WL': 1500, 'ell_max_GC': 1500, 'ell_max_ref': 1500},
    #     'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': False},
    #     'misc': {
    #         'output_path': f'{ROOT}/common_data/Spaceborne_jobs/vincenzo_2025_08/nzRR2_EP06_nbl32_ellmax1500'
    #     },
    # },
    # ! nzRR2_EP06_nbl32_ellmax5000
    # {
    #     'nz': {
    #         'nz_sources_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_SHE_tombins_unitweights_nz_SOMbin_C2020z.fits',
    #         'nz_lenses_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_POS_tombins_unitweights_nz_SOMbin_C2020z.fits',
    #         'ngal_sources': neff_src_rr2,
    #         'ngal_lenses': neff_lns_rr2,
    #         'smooth_nz': True,
    #     },
    #     'ell_binning': {'ell_max_WL': 5000, 'ell_max_GC': 5000, 'ell_max_ref': 5000},
    #     'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': False},
    #     'misc': {
    #         'output_path': f'{ROOT}/common_data/Spaceborne_jobs/vincenzo_2025_08/nzRR2_EP06_nbl32_ellmax5000'
    #     },
    # },
    # ! nzRR2_EP06_nbl32_ellmax1500_BNT
    {
        'nz': {
            'nz_sources_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_SHE_tombins_unitweights_nz_SOMbin_C2020z.fits',
            'nz_lenses_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_POS_tombins_unitweights_nz_SOMbin_C2020z.fits',
            'ngal_sources': neff_src_rr2,
            'ngal_lenses': neff_lns_rr2,
            'smooth_nz': True,
        },
        'ell_binning': {'ell_max_WL': 1500, 'ell_max_GC': 1500, 'ell_max_ref': 1500},
        'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': True},
        'misc': {
            'output_path': f'{ROOT}/common_data/Spaceborne_jobs/vincenzo_2025_08/nzRR2_EP06_nbl32_ellmax1500_BNT'
        },
    },
    # ! nzRR2_EP06_nbl32_ellmax5000_BNT
    # {
    #     'nz': {
    #         'nz_sources_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_SHE_tombins_unitweights_nz_SOMbin_C2020z.fits',
    #         'nz_lenses_filename': f'{ROOT}/common_data/RR2/Davide/Reg2_POS_tombins_unitweights_nz_SOMbin_C2020z.fits',
    #         'ngal_sources': neff_src_rr2,
    #         'ngal_lenses': neff_lns_rr2,
    #         'smooth_nz': True,
    #     },
    #     'ell_binning': {'ell_max_WL': 5000, 'ell_max_GC': 5000, 'ell_max_ref': 5000},
    #     'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': True},
    #     'misc': {
    #         'output_path': f'{ROOT}/common_data/Spaceborne_jobs/vincenzo_2025_08/nzRR2_EP06_nbl32_ellmax5000_BNT'
    #     },
    # },
    # ! nzSPV3_EP06_nbl32_ellmax5000 - "DR1" forecast
    # ! nz from https://drive.google.com/drive/u/2/folders/1oh9tdoE10kE-2CQfPhyIpdylRfpwoarx
    # {
    #     'nz': {
    #         # /home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/DR1/NzTab/nzLenses-EP06-zedMin02-zedMax25-IE235.dat
    #         'nz_sources_filename': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/DR1/NzTab/nzSources-EP06-zedMin02-zedMax25-SN05.dat',
    #         'nz_lenses_filename': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/DR1/NzTab/nzLenses-EP06-zedMin02-zedMax25-IE235.dat',
    #         # /home/cosmo/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/DR1/NzPar/ngbsLenses-EP06-zedMin02-zedMax25-IE235.dat
    #         'ngal_sources': [5.98069, 5.98077, 5.98074, 5.98072, 5.98074, 5.98073], # fmt: skip
    #         'ngal_lenses': [1.68850, 1.68847, 1.68853, 1.68850, 1.68850, 1.68850], # fmt: skip
    #         'smooth_nz': False,
    #     },
    #     'ell_binning': {'ell_max_WL': 5000, 'ell_max_GC': 5000, 'ell_max_ref': 5000},
    #     'covariance': {'sigma_eps_i': [0.26] * 6},
    #     'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': True},
    #     'misc': {
    #         'output_path': f'{ROOT}/common_data/Spaceborne_jobs/vincenzo_2025_08/nzSPV3_EP06_nbl32_ellmax5000'
    #     },
    # },
    # # ! nzSPV3_EP13_nbl32_ellmax5000
    # {
    #     'nz': {
    #         'nz_sources_filename': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/NzTab/nzTab-EP13-zedMin02-zedMax25-mag230.dat',
    #         'nz_lenses_filename': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/NzTab/nzTab-EP13-zedMin02-zedMax25-mag230.dat',
    #         # sftp://davide.sciotti@melodie.phys.uniroma1.it/export/NAS/cosmo/users/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/NzPar/from ngbsTab-EP13-zedMin02-zedMax25-mag230
    #         'ngal_sources': [0.51609, 0.51609, 0.51610, 0.5160, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51612, 0.5160], # fmt: skip
    #         'ngal_lenses': [0.51609, 0.51609, 0.51610, 0.5160, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51612, 0.5160], # fmt: skip
    #         'smooth_nz': False,
    #     },
    #     'ell_binning': {'ell_max_WL': 5000, 'ell_max_GC': 5000, 'ell_max_ref': 5000},
    #     'C_ell': {'mult_shear_bias': [0.]*13},
    #     'covariance': {'sigma_eps_i': [0.26] * 13},
    #     'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': True},
    #     'misc': {
    #         'output_path': f'{ROOT}/common_data/Spaceborne_jobs/vincenzo_2025_08/nzSPV3_EP13_nbl32_ellmax5000'
    #     },
    # },
    # # ! nzSPV3_EP13_nbl32_ellmax5000_BNT
    # {
    #     'nz': {
    #         'nz_sources_filename': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/NzTab/nzTab-EP13-zedMin02-zedMax25-mag230.dat',
    #         'nz_lenses_filename': f'{ROOT}/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/NzTab/nzTab-EP13-zedMin02-zedMax25-mag230.dat',
    #         # sftp://davide.sciotti@melodie.phys.uniroma1.it/export/NAS/cosmo/users/davide.sciotti/data/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/NzPar/from ngbsTab-EP13-zedMin02-zedMax25-mag230
    #         'ngal_sources': [0.51609, 0.51609, 0.51610, 0.5160, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51612, 0.5160], # fmt: skip
    #         'ngal_lenses': [0.51609, 0.51609, 0.51610, 0.5160, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51609, 0.51612, 0.5160], # fmt: skip
    #         'smooth_nz': False,
    #     },
    #     'ell_binning': {'ell_max_WL': 5000, 'ell_max_GC': 5000, 'ell_max_ref': 5000},
    #     'C_ell': {'mult_shear_bias': [0.]*13},
    #     'covariance': {'sigma_eps_i': [0.26] * 13},
    #     'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': True},
    #     'misc': {
    #         'output_path': f'{ROOT}/common_data/Spaceborne_jobs/vincenzo_2025_08/nzSPV3_EP13_nbl32_ellmax5000_BNT'
    #     },
    # },
]


# assign yaml filenames based on output path
yaml_filenames = [
    f'{sb_root_path}/{cfg["misc"]["output_path"].split("/")[-1]}.yaml'
    for cfg in configs_to_run
]


# Apply changes to base config
configs = generate_zipped_configs(base_cfg, configs_to_run)

# Save configs
save_configs_to_yaml(configs, yaml_filenames)

# Run all
run_benchmarks(yaml_filenames, sb_root_path)

print('\n✅ All Spaceborne runs finished!')
