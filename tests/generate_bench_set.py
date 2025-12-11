"""
LAST UPDATE: 2025-09-16

Script to produce a set of benchmarks to test the Spaceborne code. More in detail, it
performs the following operations:

1. Manually define a `base_cfg` dict (this is safer than importing the standard
   Spaceborne/config.yaml file) with fast runtime (e.g. setting a low number of points
   for the z and k grids).
2. Change some settings to test the different parts of the code (and, to ensure fast
   execution), thereby producing a list of cfg dicts to test (`configs_to_test`).
3. Save the list of cfg dicts to yaml files in the folder
   {ROOT}/Spaceborne_bench/bench_set_cfg.
4. Run SB with these yaml files, generating a set of benchmarks (npz archives) to use as
   an exhaustive reference to test the code against. The benchmarks are stored in
   {ROOT}/Spaceborne_bench/bench_set_output


NOTES

-  The code will raise an error if a benchmark file already exists.
   If you want to overwrite them, delete the existing ones (e.g.):
   {ROOT}/Spaceborne_bench/bench_set_output/config_0005.yaml
   or change the benchmark filename.

-  The SB output produced at runtime during the production of these benchmarks
   is in
   {ROOT}/Spaceborne_bench/bench_set_output/_sb_output,
   but you don't need to care about this.
"""

import gc
import json
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from itertools import product

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
        start_ix (int): Starting index for

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
        config['misc']['bench_filename'] = f'{bench_set_output_path}/{filename}'

        yaml_path = os.path.join(bench_set_path_cfg, f'{filename}.yaml')

        # Save the configuration to a YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        yaml_files.append(yaml_path)

    return yaml_files


def run_benchmarks(yaml_files, sb_root_path, output_dir, skip_existing: bool = False):
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
                # Make the path relative to the original directory, not the new working
                # directory
                _yaml_filename = os.path.abspath(os.path.join(original_dir, yaml_file))
            else:
                _yaml_filename = yaml_file

            # e.g. config_0000 from config_0000.yaml
            _filename = os.path.splitext(config_name)[0]
            bench_output_filename = f'{bench_set_output_path}/{_filename}.npz'

            if os.path.exists(bench_output_filename) and skip_existing:
                print(
                    f'Benchmark file {bench_output_filename} '
                    'already exists. Skipping it...'
                )
                continue

            print('\n')
            print('*********************************************************')
            print(f'🧪🧪🧪 Running benchmark with config: {yaml_file} 🧪🧪🧪')
            print('*********************************************************')
            print('\n')

            # Run the main script with the current configuration
            start_time = datetime.now()
            result = subprocess.run(
                ['python', 'main.py', '--config', _yaml_filename],
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
ROOT = '/u/dsciotti/code'
bench_set_path = f'{ROOT}/Spaceborne_bench'
bench_set_cfg_path = f'{bench_set_path}/bench_set_cfg'
bench_set_output_path = f'{bench_set_path}/bench_set_output'
output_path = f'{bench_set_output_path}/_sb_output'
sb_root_path = f'{ROOT}/Spaceborne'

skip_existing = True  # Skip benchmarks that already exist

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
        'Om_k0': 0.0,
    },
    'intrinsic_alignment': {
        'Aia': 0.16,
        'eIA': 1.66,
        'bIA': 0.0,
        'CIA': 0.0134,
        'z_pivot_IA': 0.0,
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
    'probe_selection': {
        'space': 'harmonic',
        'LL': True,
        'GL': True,
        'GG': True,
        'xip': True,
        'xim': True,
        'gt': True,
        'w': True,
        'cross_cov': True,
    },
    'nz': {
        'nz_sources_filename': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/nzTab-EP03-zedMin02-zedMax25-mag245.dat',
        'nz_lenses_filename': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/nzTab-EP03-zedMin02-zedMax25-mag245.dat',
        'normalize_nz': True,
        'ngal_sources': [8.09216, 8.09215, 8.09215],
        'ngal_lenses': [8.09216, 8.09215, 8.09215],
        'shift_nz': False,
        'dzWL': [0.0, 0.0, 0.0],
        'dzGC': [0.0, 0.0, 0.0],
        'smooth_nz': False,
        'sigma_smoothing': 10,
    },
    'binning': {
        'binning_type': 'log',
        'ell_min': 10,
        'ell_max': 3000,
        'ell_bins': 10,  # TODO change to 5
        'ell_bins_filename': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/ell_values_3x2pt.txt',
        'theta_min_arcmin': 50,
        'theta_max_arcmin': 300,
        'theta_bins': 5,
    },
    'C_ell': {
        'use_input_cls': False,
        'cl_LL_path': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/cl_ll.txt',
        'cl_GL_path': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/cl_gl.txt',
        'cl_GG_path': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/cl_gg.txt',
        'which_gal_bias': 'FS2_polynomial_fit',
        'which_mag_bias': 'FS2_polynomial_fit',
        'galaxy_bias_fit_coeff': [1.33291, -0.72414, 1.0183, -0.14913],
        'magnification_bias_fit_coeff': [-1.50685, 1.35034, 0.08321, 0.04279],
        'gal_bias_table_filename': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/gal_bias_table.txt',
        'mag_bias_table_filename': f'{ROOT}/common_data/Spaceborne_jobs/develop/input/mag_bias_table.txt',
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
    'mask': {
        'load_mask': False,
        'mask_path': f'{ROOT}/common_data/RR2/Davide/EUC_LE3_COVERAGE_RR2-R1-TEST_20250519T100352.127658Z_00.00_NSIDE1024.fits',
        'generate_polar_cap': True,
        'nside': 1024,
        'survey_area_deg2': 13245,
        'apodize': False,
        'aposize': 0.1,
    },
    'BNT': {'cl_BNT_transform': False, 'cov_BNT_transform': False},
    'covariance': {
        'G': True,
        'SSC': True,
        'cNG': False,
        'coupled_cov': False,
        'triu_tril': 'triu',
        'row_col_major': 'row-major',
        'covariance_ordering_2D': 'probe_scale_zpair',
        'save_full_cov': True,
        'split_gaussian_cov': False,
        'sigma_eps_i': [0.26, 0.26, 0.26],
        'no_sampling_noise': False,
        'which_pk_responses': 'halo_model',
        'which_b1g_in_resp': 'from_input',
        'include_b2g': True,
        'include_terasawa_terms': False,
        'log10_k_min': -5.0,
        'log10_k_max': 2.0,
        'k_steps': 20,
        'z_min': 0.02,
        'z_max': 3.0,
        'z_steps': 500,
        'z_steps_trisp': 10,
        'use_KE_approximation': False,
        'cov_filename': 'cov_{which_ng_cov:s}_{probe:s}_{ndim}.npz',
        'save_cov_fits': False,
    },
    'PyCCL': {
        'cov_integration_method': 'spline',
        'load_cached_tkka': False,
        'use_default_k_a_grids': False,
        'n_samples_wf': 1000,
        'spline_params': {'A_SPLINE_NA_PK': 240, 'K_MAX_SPLINE': 300},
        'gsl_params': None,
    },
    'namaster': {
        'use_namaster': False,
        'spin0': False,
        'use_INKA': True,
        'workspace_path': '...',
    },
    'sample_covariance': {
        'compute_sample_cov': False,
        'which_cls': 'namaster',
        'nreal': 5000,
        'fix_seed': True,
    },
    'precision': {
        'n_iter_nmt': None,
        'n_sub': 20,
        'n_bisec_max': 500,
        'rel_acc': 1.0e-7,
        'boost_bessel': True,
        'verbose': True,
        'ell_min_rs': 2,
        'ell_max_rs': 100000,
        'ell_bins_rs': 50,
        'ell_bins_rs_nongauss': 50,
    },
    'misc': {
        'num_threads': 72,
        'jax_platform': 'auto',
        'jax_enable_x64': True,
        'test_numpy_inversion': False,
        'test_condition_number': False,
        'test_cholesky_decomposition': False,
        'test_symmetry': False,
        'cl_triangle_plot': False,
        'plot_probe_names': True,
        'output_path': './output',
        'save_output_as_benchmark': True,
        'save_figs': False,
    },
}


# Define your "zipped" sets of changes as a list of dictionaries
# Each dictionary represents one configuration to test
configs_to_test = []


# ! Bias models
for which_gal_bias in ['from_input', 'FS2_polynomial_fit']:
    # for which_mag_bias in ['from_input', 'FS2_polynomial_fit']:
    for which_b1g_in_resp in ['from_HOD', 'from_input']:
        configs_to_test.append(
            {
                'C_ell': {
                    'which_gal_bias': which_gal_bias
                    # 'which_mag_bias': which_mag_bias,
                },
                'covariance': {'which_b1g_in_resp': which_b1g_in_resp},
            }
        )

# ! Power spectrum responses
# for which_pk_responses in ['halo_model', 'separate_universe']:
#     configs_to_test.append({'covariance': {'which_pk_responses': which_pk_responses}})

# ! RSD and magnification bias
for has_IA in [True, False]:
    for has_rsd in [True, False]:
        for has_magnification_bias in [True, False]:
            configs_to_test.append(
                {
                    'C_ell': {
                        'has_IA': has_IA,
                        'has_rsd': has_rsd,
                        'has_magnification_bias': has_magnification_bias,
                    }
                }
            )


# ! Intrinsic Alignment parameters (only when IA is enabled)
for Aia in [0.16, 0.5, 1.0]:
    for eIA in [1.66, 0.0]:
        configs_to_test.append(
            {'C_ell': {'has_IA': True}, 'intrinsic_alignment': {'Aia': Aia, 'eIA': eIA}}
        )

# ! Multiplicative shear bias
for mult_shear_bias in [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]]:
    configs_to_test.append({'C_ell': {'mult_shear_bias': mult_shear_bias}})

# ! Input Cls vs computed
for use_input_cls in [True, False]:
    configs_to_test.append({'C_ell': {'use_input_cls': use_input_cls}})

# ! No sampling noise
for no_sampling_noise in [True, False]:
    configs_to_test.append({'covariance': {'no_sampling_noise': no_sampling_noise}})


# ! covariance ordering
for ordering in ['probe_scale_zpair', 'probe_zpair_scale', 'scale_probe_zpair']:
    for triu_tril in ['triu', 'tril']:
        for row_col in ['row-major', 'col-major']:
            configs_to_test.append(
                {
                    'covariance': {
                        'covariance_ordering_2D': ordering,
                        'triu_tril': triu_tril,
                        'row_col_major': row_col,
                    }
                }
            )

# ! SSC  variations
for ke_approx in [True, False]:
    for include_b2g in [True, False]:
        for include_terasawa in [True, False]:
            configs_to_test.append(
                {
                    'covariance': {
                        'use_KE_approximation': ke_approx,
                        'include_b2g': include_b2g,
                        'include_terasawa_terms': include_terasawa,
                    }
                }
            )

# ! cNG
# for cng in [True, False]:
#     configs_to_test.append({'covariance': {'cNG': cng}})

# ! other codes [TODO add OneCov]
for ssc_code in ['Spaceborne', 'PyCCL']:
    configs_to_test.append({'covariance': {'SSC_code': ssc_code}})

# ! HS probe combinations
for LL, GL, GC in product([True, False], repeat=3):
    for split_gaussian_cov in [True, False]:
        for cross_cov in [True, False]:
            if not any([LL, GL, GC]):
                continue
            configs_to_test.append(
                {
                    'probe_selection': {
                        'LL': LL,
                        'GL': GL,
                        'GG': GC,
                        'cross_cov': cross_cov,
                        'space': 'harmonic',
                    },
                    'covariance': {'split_gaussian_cov': split_gaussian_cov},
                }
            )

# ! RS probe combinations
for xip, xim, gt, w in product([True, False], repeat=4):
    for split_gaussian_cov in [True, False]:
        for cross_cov in [True, False]:
            if not any([xip, xim, gt, w]):
                continue
            configs_to_test.append(
                {
                    'probe_selection': {
                        'xip': xip,
                        'xim': xim,
                        'gt': gt,
                        'w': w,
                        'cross_cov': cross_cov,
                        'space': 'real',
                    },
                    'covariance': {
                        'split_gaussian_cov': split_gaussian_cov,
                        'SSC': False,
                    },
                }
            )

# ! nz variations
for shift_nz in [True, False]:
    for smooth_nz in [True, False]:
        configs_to_test.append({'nz': {'shift_nz': shift_nz, 'smooth_nz': smooth_nz}})

# ! Mask variations
for load_input_mask, generate_polar_cap in zip([True, False], [False, True]):
    for nside in [512, 1024]:
        configs_to_test.append(
            {
                'mask': {
                    'generate_polar_cap': generate_polar_cap,
                    'load_mask': load_input_mask,
                    'nside': nside,
                }
            }
        )

# ! NAMASTER (test only G cov)
for coupled_cov in [True, False]:
    for spin0 in [True, False]:
        for use_INKA in [True, False]:
            for binning_type in ['log', 'lin', 'from_input']:
                configs_to_test.append(
                    {
                        'covariance': {'SSC': False, 'coupled_cov': coupled_cov},
                        'namaster': {
                            'use_namaster': True,
                            'spin0': spin0,
                            'use_INKA': use_INKA,
                        },
                        'binning': {'binning_type': binning_type},
                    }
                )


# ! BNT transform
for cl_BNT_transform, cov_BNT_transform in zip([True, False], [False, True], strict=True):
        configs_to_test.append(
            {
                'BNT': {
                    'cl_BNT_transform': cl_BNT_transform,
                    'cov_BNT_transform': cov_BNT_transform,
                }
            }
        )


# Generate configurations
configs = generate_zipped_configs(base_cfg, configs_to_test, bench_set_cfg_path)
print(f'Generated {len(configs)} configurations')


# Save configurations to YAML files
yaml_files = save_configs_to_yaml(configs, bench_set_cfg_path, output_path, start_ix=0)

# Run benchmarks
start = time.perf_counter()
run_benchmarks(
    yaml_files,
    sb_root_path=sb_root_path,
    output_dir=bench_set_output_path,
    skip_existing=skip_existing,
)

# To manually run specific configurations:
# To run a specific config:
#   python main.py --config {yaml_file}

print(f'All Benchmarks generated in {(time.perf_counter() - start):.2f} s')
print('\nAll benchmarks saved!🎉')
