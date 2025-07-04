"""
This script performs the following operations:
1. Imports the cfg yaml file in the Spaceborne root directory (as a baseline cfg)
2. It changes some settings (for example, to speed up the code), and stores this updated
   baseline config in base_cfg
1. Defines combinations of parameters to test with lists of dictionaries,
   allowing for "zipped" iteration through sets of changes.
2. Saves the set of cfg dicts to yaml files. in the folder
    /home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench/bench_set_cfg
3. Runs SB with these yaml files, generating a set of benchmarks to use as
   an exhaustive reference to test the code against. The benchmarks are stored in
   /home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench/bench_set_output
   [NOTE] the code will raise an error if the benchmark files are already
   present. If this is the case, delete the existing ones or change the filenames for
   the new bench files
   [NOTE] the SB output is in
   /home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench/bench_set_output/_sb_output,
   but you don't need to care about this
"""

import gc
import json
import os
import subprocess
from copy import deepcopy
from datetime import datetime

import yaml


def generate_zipped_configs(base_config, changes_list, output_dir):
    """
    Generate configurations by applying a predefined list of changes
    to the base configuration. Each item in changes_list is a dictionary
    representing a set of specific updates to apply.

    Args:
        base_config (dict): The initial base configuration.
        changes_list (list): A list of dictionaries, where each dictionary
                             specifies a set of changes to apply to the base config.
        output_dir (str): Directory to save the generated configuration YAML files.

    Returns:
        list: A list of full configuration dictionaries.
    """
    os.makedirs(output_dir, exist_ok=True)

    configs = []

    for change_set in changes_list:
        config = deepcopy(base_config)

        # Function to recursively update the dictionary
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


def save_configs_to_yaml(configs, bench_set_path_cfg, output_path):
    """
    Save each configuration to a separate YAML file with a descriptive name.

    Args:
        configs (list): List of configuration dictionaries
        output_dir (str): Directory to save the YAML files

    Returns:
        list: List of paths to the generated YAML files
    """
    yaml_files = []

    for i, config in enumerate(configs):
        # Create a descriptive filename based on key parameters
        # You can customize this to include specific parameters that are most relevant
        filename = f'config_{i:04d}'

        # Set the output path and bench filename in the configuration
        config['misc']['output_path'] = output_path
        config['misc']['bench_filename'] = f'{bench_set_path_results}/{filename}'

        yaml_path = os.path.join(bench_set_path_cfg, f'{filename}.yaml')

        # Save the configuration to a YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        yaml_files.append(yaml_path)

    return yaml_files


def run_benchmarks(yaml_files, sb_root_path, output_dir):
    """
    Run the benchmarks for each configuration file.

    Args:
        yaml_files (list): List of paths to YAML configuration files
        sb_root_path (str): Path to the root directory of the Spaceborne project
        output_dir (str): Directory to save the benchmark results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the current working directory to restore it later
    original_dir = os.getcwd()

    # Convert sb_root_path to absolute path if it's relative
    if not os.path.isabs(sb_root_path):
        sb_root_path = os.path.abspath(os.path.join(original_dir, sb_root_path))

    results = {}
    try:
        # Change to the root directory
        os.chdir(sb_root_path)
        print(f'Changed working directory to: {sb_root_path}')

        for yaml_file in yaml_files:
            config_name = os.path.basename(yaml_file)

            # Convert yaml_file to absolute path if needed
            if not os.path.isabs(yaml_file):
                # Make the path relative to the original directory, not the new working directory
                yaml_file = os.path.abspath(os.path.join(original_dir, yaml_file))

            print(f'Running benchmark with config: {yaml_file}')

            # Run the main script with the current configuration
            start_time = datetime.now()
            result = subprocess.run(
                ['python', 'main.py', '--config', yaml_file],
                capture_output=False,
                # text=True,
                check=True,
            )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

            end_time = datetime.now()

            # Store the results
            if output_dir:
                result_file = os.path.join(
                    output_dir, f'{os.path.splitext(config_name)[0]}_result.json'
                )
            else:
                result_file = None

            results[config_name] = {
                'exit_code': exit_code,
                'stdout': stdout,
                'stderr': stderr,
                'duration': (end_time - start_time).total_seconds(),
                'result_file': result_file
                if result_file and os.path.exists(result_file)
                else None,
            }

        gc.collect()

    finally:
        # Always restore the original working directory
        os.chdir(original_dir)
        print(f'Restored working directory to: {original_dir}')

    # Save the summary of all benchmark runs
    if output_dir:
        with open(
            os.path.join(output_dir, 'benchmark_summary.json'), 'w', encoding='utf-8'
        ) as f:
            json.dump(results, f, indent=2)

    return results


# Example usage
ROOT = '/home/davide/Documenti/Lavoro/Programmi'
bench_set_path = f'{ROOT}/Spaceborne_bench'
bench_set_path_cfg = f'{bench_set_path}/bench_set_cfg'
bench_set_path_results = f'{bench_set_path}/bench_set_output'
output_path = f'{bench_set_path_results}/_sb_output'
sb_root_path = f'{ROOT}/Spaceborne'

# start by importing a cfg file
with open(f'{sb_root_path}/config.yaml', 'r', encoding='utf-8') as f:
    base_cfg = yaml.safe_load(f)

# Base configuration (common parameters) - these will be applied first
base_cfg['covariance']['z_steps'] = 20
base_cfg['covariance']['z_steps_trisp'] = 10
base_cfg['covariance']['k_steps'] = 50
# disable runtime tests for speed
base_cfg['misc']['test_numpy_inversion'] = False
base_cfg['misc']['test_condition_number'] = False
base_cfg['misc']['test_cholesky_decomposition'] = False
base_cfg['misc']['test_symmetry'] = False
base_cfg['misc']['save_output_as_benchmark'] = True
# the base cfg has systematics on
base_cfg['C_ell']['has_IA'] = True
base_cfg['C_ell']['has_rsd'] = True
base_cfg['C_ell']['has_magnification_bias'] = True
base_cfg['nz']['shift_nz'] = True

base_cfg['ell_binning']['binning_type'] = 'ref_cut'
base_cfg['ell_binning']['ell_max_WL'] = 1500
base_cfg['ell_binning']['ell_max_GC'] = 1500
base_cfg['ell_binning']['ell_max_3x2pt'] = 1500
base_cfg['ell_binning']['ell_bins_ref'] = 20
# base_cfg['C_ell']['cl_LL_path'] = f'{ROOT}/Spaceborne_jobs/RR2_cov/input/cl_ll.txt'
# base_cfg['C_ell']['cl_GL_path'] = f'{ROOT}/Spaceborne_jobs/RR2_cov/input/cl_gl.txt'
# base_cfg['C_ell']['cl_GG_path'] = f'{ROOT}/Spaceborne_jobs/RR2_cov/input/cl_gg.txt'
base_cfg['C_ell']['which_gal_bias'] = 'FS2_polynomial_fit'
base_cfg['C_ell']['which_mag_bias'] = 'FS2_polynomial_fit'

# Define your "zipped" sets of changes as a list of dictionaries
# Each dictionary represents one configuration to test
test_g_space_zipped = [
    # Configuration 1:
    {
        'covariance': {
            'G': True,
            'SSC': True,
            'cNG': False,
            'no_sampling_noise': True,
            'use_KE_approximation': True,
        },
    },
    # Configuration 2: use input files [TODO]
    {
        # 'C_ell': {
        # 'use_input_cls': 'from_input',
        # },
        'covariance': {
            'G': True,
            'SSC': False,
            'cNG': False,
            'no_sampling_noise': False,
            'use_KE_approximation': False,
        },
    },
    # Configuration 3: no systematics
    {
        'C_ell': {
            'has_IA': False,
            'has_rsd': False,
            'has_magnification_bias': False,
        },
        'nz': {'shift_nz': False},
        'covariance': {
            'G': True,
            'SSC': False,
            'cNG': False,
            'no_sampling_noise': True,
        },
    },
]

test_ssc_space_zipped = [
    {
        'covariance': {
            'G': False,
            'SSC': True,
            'cNG': False,
            'which_pk_responses': 'halo_model',
            'include_b2g': True,
            'which_sigma2_b': 'full_curved_sky',
            'include_terasawa_terms': True,
            'use_KE_approximation': True,
        }
    },
    {
        'covariance': {
            'G': False,
            'SSC': True,
            'cNG': False,
            'which_pk_responses': 'separate_universe',
            'include_b2g': False,
            'which_sigma2_b': 'polar_cap_on_the_fly',
            'include_terasawa_terms': False,
            'use_KE_approximation': False,
        }
    },
]

test_cng_space_zipped = [
    {
        'covariance': {
            'G': True,
            'SSC': False,
            'cNG': True,
            'z_steps_trisp': 20,
        }
    },
    # Add other cNG-specific configurations here if needed
    {
        'covariance': {
            'G': True,
            'SSC': False,
            'cNG': True,
            'z_steps_trisp': 30,  # Example of another cNG config
        }
    },
]


# Choose which parameter space to use for zipped iteration
param_space_to_use = test_g_space_zipped


# Generate configurations using the new function
configs = generate_zipped_configs(base_cfg, param_space_to_use, bench_set_path_cfg)
print(f'Generated {len(configs)} configurations')

# Save configurations to YAML files
yaml_files = save_configs_to_yaml(configs, bench_set_path_cfg, output_path)

# Optionally run benchmarks
run_benchmarks(yaml_files, sb_root_path=sb_root_path, output_dir=bench_set_path_results)

# Or you can manually run specific configurations
for yaml_file in yaml_files[:1]:  # Run only the first config as an example
    print(f'To run a specific config: python main.py --config {yaml_file}')
