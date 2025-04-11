"""
To run these tests: 
1.  Decide on a branch/commit/version you wish to use as benchmark. 
    Then, set `save_output_as_benchmark` to `True` in the config file and choose a 
    unique benchmark filename. Note that these options are in main.py, as of now. 
    Also, pay attention to all of the hardcoded configs in main.py, they need to match 
    between the different versions you're testing.
2.  Make sure there's no FM-related section at the end of main.py, the code has to finish 
    without errors.
3.  Run the code to generate the benchmark file and the associate yaml cfg file.
4.  Switch branch (for example) and make sure the hardcoded options in main.py are 
    consistent with the benchmark version.
5.  Open this script and make sure you indicate the relevant benchmark file name 
    in the `bench_names` list, then run it.
6.  If some configs are missing, check the benchmark .yaml file and manually paste them
    there, rather than adding hardcoded options in main.py.
    
Note:   if all checks are run, the content of the tmp folder is deleted, preventing you 
        to inspect the output files in more detail. In this case, simply stop the script
        at the end of test_main_script func, eg with 
        `assert False, 'stop here'`
"""

import glob
import os
import subprocess

import numpy as np
import yaml


def test_main_script(test_cfg_path):
    # Run the main script with the test config
    subprocess.run(['python', main_script_path, '--config', test_cfg_path], check=True)

    # Load the benchmark output
    bench_data = np.load(f'{bench_path}/{bench_name}.npz', allow_pickle=True)

    # Load the test output
    test_data = np.load(f'{temp_output_filename}.npz', allow_pickle=True)

    keys_test = test_data.files
    keys_bench = bench_data.files
    common_keys = list(set(keys_test) & set(keys_bench))
    common_keys.sort()
    
    print(f'Keys not in common: {set(keys_test) ^ set(keys_bench)}')


    # Compare outputs
    for key in common_keys:
        if key in excluded_keys:
            continue

        bench_arr = np.asarray(bench_data[key])
        test_arr = np.asarray(test_data[key])

        try:
            # Direct comparison (handles empty arrays automatically)
            np.testing.assert_allclose(
                bench_arr,
                test_arr,
                atol=0,
                rtol=1e-5,
                err_msg=f"{key} doesn't match the benchmark ❌",
            )
            print(f'{key} matches the benchmark ✅')
        except ValueError as e:
            # Catch shape mismatches (e.g., one empty, one non-empty)
            print(f"Shape mismatch for '{key}': {e}")
        except (TypeError, AssertionError) as e:
            # Catch other errors (dtype mismatches, numerical differences)
            print(f'Comparison failed for {key}: {e}')

    # check that cov TOT = G + SSC + cNG
    for probe in ['WL', 'GC', '3x2pt']:
        for _dict in [bench_data, test_data]:
            try:
                # Direct comparison (handles empty arrays automatically)
                np.testing.assert_allclose(
                    _dict[f'cov_{probe}_tot_2D'],
                    _dict[f'cov_{probe}_g_2D']
                    + _dict[f'cov_{probe}_ssc_2D']
                    + _dict[f'cov_{probe}_cng_2D'],
                    atol=0,
                    rtol=1e-5,
                    err_msg=f'cov {probe} tot != G + SSC + cNG ❌',
                )
                print(f'cov {probe} tot = G + SSC + cNG ✅')
            except ValueError as e:
                # Catch shape mismatches (e.g., one empty, one non-empty)
                print(f"Shape mismatch for '{key}': {e}")
            except (TypeError, AssertionError) as e:
                # Catch other errors (dtype mismatches, numerical differences)
                print(f'Comparison failed for {key}: {e}')
                
    # example of the Note above
    # assert False, 'stop here'
    # sl.compare_arrays(bench_data['cov_3x2pt_tot_2D'], test_data['cov_3x2pt_tot_2D'], plot_diff_threshold=1, plot_diff_hist=True)


# Path
ROOT = '/home/davide/Documenti/Lavoro/Programmi'
bench_path = f'{ROOT}/Spaceborne_bench'
# run all tests...
bench_names = glob.glob(f'{bench_path}/*.npz')
bench_names = [os.path.basename(file) for file in bench_names]
bench_names = [bench_name.replace('.npz', '') for bench_name in bench_names]
# ... or run specific tests
bench_names = [
    'output_GSpaceborne_SSCSpaceborne_cNGPyCCL_KEFalse_resphalo_model_b1gfrom_HOD_devmerge',
]

main_script_path = f'{ROOT}/Spaceborne/main.py'
temp_output_filename = f'{ROOT}/Spaceborne_bench/tmp/test_file'
temp_output_folder = os.path.dirname(temp_output_filename)
excluded_keys = ['backup_cfg', 'metadata']

# set the working directory to the main script path
# %cd main_script_path.rstrip('/main.py')
os.chdir(os.path.dirname(main_script_path))

if os.path.exists(f'{temp_output_filename}.npz'):
    message = f'{temp_output_filename}.npz already exists, most likely '
    'from a previous failed test. Do you want to overwrite it? y/n: '
    if input(message) != 'y':
        print('Exiting...')
        exit()
    else:
        os.remove(f'{temp_output_filename}.npz')

for bench_name in bench_names:
    print(f'Testing {bench_name}...')

    # ! update the cfg file to avoid overwriting the benchmarks
    # Load the benchmark config
    with open(f'{bench_path}/{bench_name}.yaml') as f:
        cfg = yaml.safe_load(f)

    # Update config for the test run
    cfg['misc']['save_output_as_benchmark'] = True
    cfg['misc']['bench_filename'] = temp_output_filename
    # just to make sure I don't overwrite any output files
    cfg['misc']['output_path'] = temp_output_folder

    # Save the updated test config
    test_cfg_path = f'{bench_path}/tmp/test_config.yaml'
    with open(test_cfg_path, 'w') as f:
        yaml.dump(cfg, f)

    # ! run the actual test
    test_main_script(test_cfg_path)

    # delete the output test files in tmp folder
    for file_path in glob.glob(f'{temp_output_folder}/*'):
        if os.path.isfile(file_path):
            os.remove(file_path)
