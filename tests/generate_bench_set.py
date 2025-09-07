"""This script performs the following operations.

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
   but you don't need to care about this.

NOTES

1. The code will raise an error if the benchmark files are already present.
   If you want to overwrite them, delete the existing ones (e.g.):
   {ROOT}/Spaceborne_bench/bench_set_output/config_0005.yaml
"""

import gc
import json
import os
import subprocess
from copy import deepcopy
from datetime import datetime

import yaml


def generate_zipped_configs(base_config, changes_list, output_dir):
    """Generate configurations by applying a predefined list of changes
    to the base configuration.

    Each item in changes_list is a dictionary
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


def save_configs_to_yaml(configs, bench_set_path_cfg, output_path, start_ix=0):
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
        filename = f'config_{(i + start_ix):04d}'

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
    """Run the benchmarks for each configuration file.

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
                _yaml_file = os.path.abspath(os.path.join(original_dir, yaml_file))
            else:
                _yaml_file = yaml_file

            print('\n')
            print('*********************************************************')
            print(f'🧪🧪🧪 Running benchmark with config: {yaml_file} 🧪🧪🧪')
            print('*********************************************************')
            print('\n')

            # Run the main script with the current configuration
            start_time = datetime.now()
            result = subprocess.run(
                ['python', 'main.py', '--config', _yaml_file],
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

# ! DEFINE A BASIC CFG FILE TO START FROM
base_cfg = {
    'cosmology': {
        'Om': 0.32,
        'Ob': 0.05,
        'wz': -1.0,
        'wa': 0.0,
        'h': 0.6737,
        'ns': 0.966,
        's8': 0.816,
        'ODE': 0.68,
        'm_nu': 0.06,
        'N_eff': 3.046,
        'Om_k0': 0,
    },
    'intrinsic_alignment': {
        'Aia': 0.16,
        'eIA': 1.66,
        'bIA': 0.0,
        'CIA': 0.0134,
        'z_pivot_IA': 0,
        'lumin_ratio_filename': None,
    },
    'extra_parameters': {
        'camb': {
            'halofit_version': 'mead2020_feedback',
            'kmax': 100,
            'HMCode_logT_AGN': 7.75,
            'num_massive_neutrinos': 1,
            'dark_energy_model': 'ppf',
        }
    },
    'halo_model': {
        'mass_def': 'MassDef200m',
        'concentration': 'ConcentrationDuffy08',
        'mass_function': 'MassFuncTinker10',
        'halo_bias': 'HaloBiasTinker10',
        'halo_profile_dm': 'HaloProfileNFW',
        'halo_profile_hod': 'HaloProfileHOD',
    },
    'probe_selection': {'LL': True, 'GL': True, 'GG': True, 'cross_cov': True},
    'C_ell': {
        'use_input_cls': False,
        'cl_LL_path': f'{ROOT}/common_data/Spaceborne_jobs/RR2_cov/input/cl_ll.txt',
        'cl_GL_path': f'{ROOT}/common_data/Spaceborne_jobs/RR2_cov/input/cl_gl.txt',
        'cl_GG_path': f'{ROOT}/common_data/Spaceborne_jobs/RR2_cov/input/cl_gg.txt',
        'which_gal_bias': 'FS2_polynomial_fit',
        'which_mag_bias': 'FS2_polynomial_fit',
        'galaxy_bias_fit_coeff': [1.33291, -0.72414, 1.0183, -0.14913],
        'magnification_bias_fit_coeff': [-1.50685, 1.35034, 0.08321, 0.04279],
        'gal_bias_table_filename': f'{ROOT}/common_data/Spaceborne_jobs/RR2_cov/input/gal_bias.txt',
        'mag_bias_table_filename': f'{ROOT}/common_data/Spaceborne_jobs/RR2_cov/input/mag_bias.txt',
        'mult_shear_bias': [0.0, 0.0, 0.0],
        'has_rsd': False,
        'has_IA': False,
        'has_magnification_bias': False,
        'cl_CCL_kwargs': {
            'l_limber': -1,
            'limber_integration_method': 'spline',
            'non_limber_integration_method': 'FKEM',
        },
    },
    'nz': {
        'nz_sources_filename': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/nzTab-EP03-zedMin02-zedMax25-mag245.dat',
        'nz_lenses_filename': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/nzTab-EP03-zedMin02-zedMax25-mag245.dat',
        'ngal_sources': [8.09216, 8.09215, 8.09215],
        'ngal_lenses': [8.09216, 8.09215, 8.09215],
        'shift_nz': False,
        'dzWL': [-0.008848, 0.051368, 0.059484],
        'dzGC': [-0.008848, 0.051368, 0.059484],
        'normalize_shifted_nz': True,
        'clip_zmin': 0,
        'clip_zmax': 3,
        'smooth_nz': False,
        'sigma_smoothing': 10,
    },
    'mask': {
        'load_mask': False,
        'mask_path': '../input/mask.fits',
        'generate_polar_cap': True,
        'nside': 1024,
        'survey_area_deg2': 13245,
        'apodize': False,
        'aposize': 0.1,
    },
    'ell_binning': {
        'binning_type': 'ref_cut',
        'ell_min_WL': 10,
        'ell_max_WL': 3000,
        'ell_bins_WL': 15,
        'ell_min_GC': 10,
        'ell_max_GC': 3000,
        'ell_bins_GC': 15,
        'ell_min_ref': 10,
        'ell_max_ref': 3000,
        'ell_bins_ref': 15,
    },
    'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': False},
    'covariance': {
        'G': True,
        'SSC': False,
        'cNG': False,
        'coupled_cov': False,
        'triu_tril': 'triu',
        'row_col_major': 'row-major',
        'covariance_ordering_2D': 'probe_ell_zpair',
        'save_full_cov': True,
        'split_gaussian_cov': False,
        'sigma_eps_i': [0.26, 0.26, 0.26],
        'no_sampling_noise': False,
        'which_pk_responses': 'halo_model',
        'which_b1g_in_resp': 'from_input',
        'include_b2g': True,
        'include_terasawa_terms': False,
        'log10_k_min': -5,
        'log10_k_max': 2,
        'k_steps': 20,
        'z_min': 0.02,
        'z_max': 3.0,
        'z_steps': 100,
        'z_steps_trisp': 10,
        'use_KE_approximation': False,
        'cov_filename': 'cov_{which_ng_cov:s}_{probe:s}_{ndim}.npz',
    },
    # Base configuration (common parameters) - these will be applied first
    'namaster': {
        'use_namaster': False,
        'spin0': False,
        'use_INKA': True,
        'workspace_path': None,
    },
    'sample_covariance': {
        'compute_sample_cov': False,
        'which_cls': 'namaster',
        'nreal': 5000,
        'fix_seed': True,
    },
    'PyCCL': {
        'cov_integration_method': 'spline',
        'load_cached_tkka': False,
        'use_default_k_a_grids': False,
        'n_samples_wf': 1000,
        'spline_params': {'A_SPLINE_NA_PK': 240, 'K_MAX_SPLINE': 300},
        'gsl_params': None,
    },
    'precision': {'n_iter_nmt': None},
    'misc': {
        'num_threads': 40,
        'test_numpy_inversion': True,
        'test_condition_number': True,
        'test_cholesky_decomposition': True,
        'test_symmetry': True,
        'cl_triangle_plot': False,
        'save_figs': False,
        'output_path': './output',
        'save_output_as_benchmark': True,
    },
}


# Define your "zipped" sets of changes as a list of dictionaries
# Each dictionary represents one configuration to test
configs_to_test = [
    # G with split
    {'covariance': {'G': True, 'split_gaussian_cov': False}},
    #  G without split
    {'covariance': {'G': True, 'split_gaussian_cov': True}},
    # G nmt spin0, log ell binning
    # SSC KE
    {'covariance': {'G': True, 'SSC': True, 'cNG': False}},
    # SSC LR
    {'covariance': {'G': True, 'SSC': True, 'cNG': False}},
    # cNG
    {'covariance': {'G': True, 'SSC': False, 'cNG': True}},
    
    # === namaster runs, quite slow ===
    {
        'covariance': {'G': True},
        'namaster': {'use_namaster': True, 'spin0': True},
        'ell_binning': {'binning_type': 'log'},
    },
    # G spin0, lin ell binning
    # ==============================
    {
        'covariance': {'G': True},
        'namaster': {'use_namaster': True, 'spin0': True},
        'ell_binning': {'binning_type': 'lin'},
    },
    # G spin2, lin ell binning
    {
        'covariance': {'G': True},
        'namaster': {'use_namaster': True, 'spin0': False},
        'ell_binning': {'binning_type': 'lin'},
    },
]


# Generate configurations
configs = generate_zipped_configs(base_cfg, configs_to_test, bench_set_path_cfg)
print(f'Generated {len(configs)} configurations')

# Save configurations to YAML files
yaml_files = save_configs_to_yaml(configs, bench_set_path_cfg, output_path, start_ix=14)

# Run benchmarks
run_benchmarks(yaml_files, sb_root_path=sb_root_path, output_dir=bench_set_path_results)

# To manually run specific configurations:
# To run a specific config:
#   python main.py --config {yaml_file}

print('\nAll benchmarks saved!🎉')
