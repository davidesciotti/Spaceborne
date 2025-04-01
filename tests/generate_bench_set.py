# import glob
# import os
# import subprocess

# import numpy as np
# import yaml

# # start by importing a cfg file
# with open('../config.yaml', 'r') as f:
#     cfg = yaml.safe_load(f)
    
# # these are fixed for all test cases
# z_steps = 50
# z_steps_trisp = 15
# ksteps = 50
    
# test_g = {
#     'G': True,
#     'SSC': False,
#     'cNG': False,
#     'use_input_cls': (True, False),
#     'cl_LL_path': ...,
#     'cl_GL_path': ...,
#     'cl_GG_path': ...,
#     'which_gal_bias': ('FS2_polynomial_fit', 'from_input'),
#     'which_mag_bias': ('FS2_polynomial_fit', 'from_input'),
#     'has_IA': (True, False),
#     'has_magnification_bias': (True, False),
#     'shift_nz': (True, False),
#     'binning_type': ('log', 'unbinned'),
#     'ell_max_WL': (3000, 1500),
#     'ell_max_GC': (3000, 1500),
#     'ell_max_3x2pt': (3000, 1500),
#     # 'split_gaussian_cov': (True, False),  # TODO
#     'noiseless_spectra': (True, False),
    
# }
# test_ssc = {
#     'G': True,
#     'SSC': True,
#     'cNG': False,
#     'which_pk_responses': ('halo_model', 'separate_universe'),
#     'include_b2g': (True, False),
#     'which_sigma2_b': ('full_curved_sky', 'polar_cap_on_the_fly', 'flat_sky'),
#     'include_terasawa_terms': (True, False),
#     'use_KE_approximation': (True, False),
    
# }

# test_cng = {
#     'G': True,
#     'SSC': False,
#     'cNG': True,
#     'z_steps_trisp': 20,
    
    
# }
import yaml
import itertools
import os
from copy import deepcopy
import subprocess
import json
from datetime import datetime

def generate_configs(base_config, param_space, output_dir="benchmark_configs"):
    """
    Generate all possible configurations from a parameter space.
    
    Args:
        base_config (dict): Base configuration that will be updated with specific params
        param_space (dict): Dictionary of parameters to test and their possible values
        output_dir (str): Directory to save the generated YAML files
    
    Returns:
        list: List of generated configuration dictionaries
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameter names and their possible values
    param_names = []
    param_values = []
    for param, values in param_space.items():
        if isinstance(values, (tuple, list)):
            param_names.append(param)
            param_values.append(values)
    
    # Generate all combinations of parameter values
    configs = []
    for combination in itertools.product(*param_values):
        # Create a new config by updating the base config
        config = deepcopy(base_config)
        
        # Update config with the current combination of parameters
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        
        configs.append(config)
    
    return configs

def save_configs_to_yaml(configs, output_dir="benchmark_configs"):
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
        filename = f"config_{i:04d}"
        if 'G' in config:
            filename += f"_G{config['G']}"
        if 'SSC' in config:
            filename += f"_SSC{config['SSC']}"
        if 'cNG' in config:
            filename += f"_cNG{config['cNG']}"
        
        yaml_path = os.path.join(output_dir, f"{filename}.yaml")
        
        # Save the configuration to a YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        yaml_files.append(yaml_path)
    
    return yaml_files

def run_benchmarks(yaml_files, script_path="main.py", output_dir="benchmark_results"):
    """
    Run the benchmarks for each configuration file.
    
    Args:
        yaml_files (list): List of paths to YAML configuration files
        script_path (str): Path to the main script that will process the config
        output_dir (str): Directory to save the benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for yaml_file in yaml_files:
        config_name = os.path.basename(yaml_file)
        result_file = os.path.join(output_dir, f"{os.path.splitext(config_name)[0]}_result.json")
        
        print(f"Running benchmark with config: {yaml_file}")
        
        # Run the main script with the current configuration
        start_time = datetime.now()
        result = subprocess.run(
            ["python", script_path, "--config", yaml_file, "--output", result_file],
            capture_output=True,
            text=True
        )
        end_time = datetime.now()
        
        # Store the results
        results[config_name] = {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": (end_time - start_time).total_seconds(),
            "result_file": result_file if os.path.exists(result_file) else None
        }
    
    # Save the summary of all benchmark runs
    with open(os.path.join(output_dir, "benchmark_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)

# Example usage

# Base configuration (common parameters)
base_config = {
    'G': True,
    # Add other common parameters here
}

# Define parameter spaces for different test scenarios
test_g_space = {
    'G': True,
    'SSC': False,
    'cNG': False,
    'use_input_cls': (True, False),
    'which_gal_bias': ('FS2_polynomial_fit', 'from_input'),
    'which_mag_bias': ('FS2_polynomial_fit', 'from_input'),
    'has_IA': (True, False),
    'has_magnification_bias': (True, False),
    'shift_nz': (True, False),
    'binning_type': ('log', 'unbinned'),
    'ell_max_WL': (3000, 1500),
    'ell_max_GC': (3000, 1500),
    'ell_max_3x2pt': (3000, 1500),
    'noiseless_spectra': (True, False),
}

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
configs = generate_configs(base_config, param_space)
print(f"Generated {len(configs)} configurations")

# Save configurations to YAML files
yaml_files = save_configs_to_yaml(configs)

# Optionally run benchmarks
# run_benchmarks(yaml_files)

# Or you can manually run specific configurations
for yaml_file in yaml_files[:3]:  # Run only the first 3 configs as an example
    print(f"To run a specific config: python main.py --config {yaml_file}")