"""
This script performs the following operations:
1. Defines combinations of parameters to test with tuples and automatically generates
the different possible combinations
2. saves the set of cfg dicts to yaml files
3. runs SB with these yaml files, generating a large set of benchmarks to use as
an exhaustive reference to test the code against.
"""

import yaml
import itertools
import gc
import os
from copy import deepcopy
import subprocess
import json
from datetime import datetime


def generate_configs(base_config, param_space, output_dir):
    """
    Generate configs by:
    1. First updating base_config with non-iterable params from param_space
    2. Then varying only the parameters marked with tuples/lists in param_space
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Update base_config with non-iterable params from param_space
    # (This ensures e.g., `cNG: False` in test_g_space overrides `cNG: True` in base_config)
    def update_config(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                update_config(target[key], value)
            elif not isinstance(
                value, (tuple, list)
            ):  # Only update non-iterable values
                target[key] = value

    updated_base_config = deepcopy(base_config)
    update_config(updated_base_config, param_space)

    # Step 2: Extract parameters to vary (only those with tuples/lists)
    flat_param_names = []
    flat_param_values = []

    for key, value in param_space.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (tuple, list)):
                    flat_param_names.append((key, subkey))
                    flat_param_values.append(subvalue)
        elif isinstance(value, (tuple, list)):
            flat_param_names.append(key)
            flat_param_values.append(value)

    # Generate all combinations
    configs = []
    for combination in itertools.product(*flat_param_values):
        config = deepcopy(updated_base_config)  # Start with updated base

        # Apply the varying parameters
        for param_name, param_value in zip(flat_param_names, combination):
            if isinstance(param_name, tuple):
                key, subkey = param_name
                config[key][subkey] = param_value
            else:
                config[param_name] = param_value

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
            try:
                result = subprocess.run(
                    ['python', 'main.py', '--config', yaml_file],
                    capture_output=False,
                    text=True,
                )
                stdout = result.stdout
                stderr = result.stderr
                exit_code = result.returncode
            except Exception as e:
                stdout = ''
                stderr = str(e)
                exit_code = -1

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
        with open(os.path.join(output_dir, 'benchmark_summary.json'), 'w') as f:
            json.dump(results, f, indent=2)

    return results


# Example usage
bench_set_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/tests/bench_set'
bench_set_path_cfg = f'{bench_set_path}/cfg'
bench_set_path_results = f'{bench_set_path}/results'
output_path = f'{bench_set_path}/_outputs'
sb_root_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne'

# start by importing a cfg file
with open(f'{sb_root_path}/config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Base configuration (common parameters)
base_config['covariance']['z_steps'] = 20
base_config['covariance']['z_steps_trisp'] = 10
base_config['covariance']['k_steps'] = 20
base_config['misc']['test_numpy_inversion'] = False
base_config['misc']['test_condition_number'] = False
base_config['misc']['test_cholesky_decomposition'] = False
base_config['misc']['test_symmetry'] = False
base_config['misc']['save_output_as_benchmark'] = True
base_config['ell_binning']['binning_type'] = 'log'
base_config['ell_binning']['ell_max_WL'] = 1500
base_config['ell_binning']['ell_max_GC'] = 1500
base_config['ell_binning']['ell_max_3x2pt'] = 1500


test_g_space = {'C_ell': {}, 'nz': {}, 'ell_binning': {}, 'covariance': {}, 'misc': {}}
test_g_space['C_ell']['which_gal_bias'] = ('FS2_polynomial_fit', 'from_input')
test_g_space['C_ell']['which_mag_bias'] = ('FS2_polynomial_fit', 'from_input')
test_g_space['C_ell']['has_IA'] = (True, False)
test_g_space['C_ell']['has_magnification_bias'] = (True, False)
test_g_space['nz']['shift_nz'] = (True, False)
test_g_space['covariance']['G'] = True
test_g_space['covariance']['SSC'] = False
test_g_space['covariance']['cNG'] = False
test_g_space['covariance']['noiseless_spectra'] = (True, False)


test_ssc_space = {
    'G': True,
    'SSC': True,
    'cNG': False,
    'which_pk_responses': ('halo_model', 'separate_universe'),
    'include_b2g': (True, False),
    'which_sigma2_b': ('full_curved_sky', 'polar_cap_on_the_fly', 'flat_sky'),
    'include_terasawa_terms': (True, False),
    'use_KE_approximation': (True, False),
}

test_cng_space = {
    'G': True,
    'SSC': False,
    'cNG': True,
    'z_steps_trisp': 20,
}

# Choose which parameter space to use
param_space = test_g_space

# Generate configurations
configs = generate_configs(base_config, param_space, bench_set_path_cfg)
print(f'Generated {len(configs)} configurations')

# Save configurations to YAML files
yaml_files = save_configs_to_yaml(configs, bench_set_path_cfg, output_path)

# Optionally run benchmarks
run_benchmarks(yaml_files, sb_root_path=sb_root_path, output_dir=bench_set_path_results)

# Or you can manually run specific configurations
for yaml_file in yaml_files[:3]:  # Run only the first 3 configs as an example
    print(f'To run a specific config: python main.py --config {yaml_file}')
