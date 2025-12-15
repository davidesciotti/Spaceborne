"""
LAST UPDATE: 2025-09-16

Script to run Spaceborne and compare the output against a specified set of benchmarks
More in detail, it performs the following operations:

1. Set a list of benchmark filenames (`bench_yaml_names`) to test the current
   branch/version of the code against.
2. For each benchmark:
   2a. Run Spaceborne using the corresponding .yaml file as cfg
   2b. Save the output as benchmark (just for the sake of comparing against the "real"
   benchmarks in {ROOT}/Spaceborne_bench/bench_set_output/*.npz)
   2c. Compare the output (`test`) against the benchmark



# OLD INSTRUCTIONS
1.  Decide on a branch/commit/version you wish to use as benchmark.
    Then, set `save_output_as_benchmark` to `True` in the config file and choose a
    unique benchmark filename. *Note that these options are in main.py, as of now*.
    Also, pay attention to all of the hardcoded configs in main.py, they need to match
    between the different versions you're testing.
2.  Make sure there's no FM-related section at the end of main.py, the code has to finish
    without errors.
3.  Run the code to generate the benchmark file and the associated yaml cfg file.
----------------------------------------------------------------------------------------
4.  Switch branch (for example) and make sure the hardcoded options in main.py are
    consistent with the benchmark version.
    4.1  In particular, in main.py, comment out the lines:
           # cfg['misc']['save_output_as_benchmark'] = ...
           # cfg['misc']['bench_filename'] = ...
    4.2  If you're testing the main branch, don't worry about
         config/example_config separation (UPDATE 15/05/2025: I removed example_config)
5.  Open this script and make sure you indicate the relevant benchmark file name
    in the `bench_names` list, then run it.
6.  If some configs are missing, check the benchmark .yaml file and manually paste them
    there, rather than adding hardcoded options in main.py.
# END OLD INSTRUCTIONS

NOTES
-  If all checks are run, the content of the tmp folder is deleted, preventing you
   to inspect the output files in more detail. In this case, simply stop the script
   at the end of test_main_script func, eg with
   `assert False, 'stop here'
-  You will likely have to manually edit the .yaml config files, i.e.
   {ROOT}/Spaceborne_bench/bench_set_output/*.yaml
   when you update the structure of the config file in the branch to be tested.
   BE CAREFUL NO TO INTRODUCE INCONSISTENCIES WITHT THE CORRESPONDING .npz
   (e.g., if config_0000.yaml has `SSC: False` you cannot change it to True, otherwise
   the code will prouce a `test` inconsistent with the config_0000.npz)
"""

import glob
import os
import subprocess
import sys

import numpy as np
import yaml

# # get working directory with os
# main_script_path = os.path.abspath(__file__)
# main_script_dir = os.path.dirname(main_script_path)


def test_main_script(test_cfg_path):
    # Run the main script with the test config
    subprocess.run(['python', main_script_path, '--config', test_cfg_path], check=True)

    # Load the benchmark output
    bench_data = np.load(f'{bench_path}/{bench_name}.npz', allow_pickle=True)

    # Load the test output
    test_data = np.load(f'{temp_output_filename}.npz', allow_pickle=True)

    keys_test = test_data.files
    keys_bench = bench_data.files

    # print keys not in common
    uncommon_keys = list(set(keys_test) ^ set(keys_bench))
    for key in uncommon_keys:
        key_in_test = '✅' if key in keys_test else '✖️'
        key_in_bench = '✅' if key in keys_bench else '✖️'
        print(f'{key:<15} \t\t in test: {key_in_test} \t in bench: {key_in_bench}')

    # test keys in common
    common_keys = list(set(keys_test) & set(keys_bench))
    common_keys.sort()
    print('\n')
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
                    _dict[f'cov_{probe}_tot_2d'],
                    _dict[f'cov_{probe}_g_2d']
                    + _dict[f'cov_{probe}_ssc_2d']
                    + _dict[f'cov_{probe}_cng_2d'],
                    atol=0,
                    rtol=1e-5,
                    err_msg=f'cov {probe} tot != G + SSC + cNG ❌',
                )
                print(f'cov {probe} tot = G + SSC + cNG ✅')
            except ValueError as e:
                # Catch shape mismatches (e.g., one empty, one non-empty)
                print(f"Shape mismatch for '{probe}': {e}")
            except (TypeError, AssertionError) as e:
                # Catch other errors (dtype mismatches, numerical differences)
                print(f'Comparison failed for {probe}: {e}')
            except KeyError as e:
                # Catch missing keys
                print(
                    f'It looks like cov_{probe}_tot_2D or one of the other '
                    'covariances is missing. This may be because of the probes '
                    'selected in the config, and is not necessarily an error.'
                )
                print(f'Error: \n{e}\n')

    # example of the Note above
    # assert False, 'stop here'
    # sl.compare_arrays(bench_data['cov_3x2pt_tot_2D'], test_data['cov_3x2pt_tot_2D'], plot_diff_threshold=1, plot_diff_hist=True)


# Path
ROOT = '/Users/davidesciotti/Documents/Work/Code'
bench_path = f'{ROOT}/Spaceborne_bench/bench_set_output'

# run this to also save output of this script to a file
# python test_outputs.py 2>&1 | tee test_outputs_log.txt

# run all tests...
bench_yaml_names = glob.glob(f'{bench_path}/*.npz')
bench_yaml_names = [os.path.basename(file) for file in bench_yaml_names]
bench_yaml_names = [bench_name.replace('.npz', '') for bench_name in bench_yaml_names]
bench_yaml_names.sort()

# slow_benchs = [
#     'config_0004',
#     'config_0005',
#     'config_0008',
#     'config_0009',
#     'config_0010',
#     'config_0013',
#     'config_0018',
# ]

# remove slow_benchs from bench_yaml_names
# for bench_name in slow_benchs:
#     if bench_name in bench_yaml_names:
#         bench_yaml_names.remove(bench_name)

# ... or run specific tests
# bench_yaml_names = [
#     'config_0006',
#     'config_0007',
#     'config_0008',
#     'config_0009',
#     'config_0010',
#     'config_0011',
#     'config_0012',
#     'config_0013',
# ]

main_script_path = f'{ROOT}/Spaceborne/main.py'
temp_output_filename = f'{ROOT}/Spaceborne_bench/tmp/test_file'
temp_output_folder = os.path.dirname(temp_output_filename)
excluded_keys = ['backup_cfg', 'metadata']

# set the working directory to the main script path
# %cd main_script_path.rstrip('/main.py')
os.chdir(os.path.dirname(main_script_path))

if os.path.exists(f'{temp_output_filename}.npz'):
    message = (
        f'{temp_output_filename}.npz already exists, most likely '
        'from a previous failed test. Do you want to overwrite it? y/n: '
    )
    if input(message) != 'y':
        print('Exiting...')
        sys.exit()
    else:
        os.remove(f'{temp_output_filename}.npz')

for bench_name in bench_yaml_names:
    print(f'\n\n🧪🧪🧪 Testing {bench_name} 🧪🧪🧪...\n')

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
    test_cfg_path = f'{bench_path}/_tmp/test_config.yaml'
    with open(test_cfg_path, 'w') as f:
        yaml.dump(cfg, f)

    # ! run the actual test
    test_main_script(test_cfg_path)

    # delete the output test files in tmp folder
    for file_path in glob.glob(f'{temp_output_folder}/*'):
        if os.path.isfile(file_path):
            os.remove(file_path)


print('Done.')
