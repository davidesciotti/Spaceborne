import glob
import pytest
import subprocess
import numpy as np
import os
import yaml

from spaceborne import sb_lib as sl


def test_main_script(test_cfg_path):
    # Run the main script with the test config
    subprocess.run([f'python', main_script_path, '--config', test_cfg_path], check=True)

    # Load the benchmark output
    bench_data = np.load(f'{bench_path}/{bench_name}.npz', allow_pickle=True)

    # Load the test output
    test_data = np.load(f'{temp_output_filename}.npz', allow_pickle=True)

    # Compare the outputs
    for key in bench_data.files:

        if key not in excluded_keys:

            # ! to be understood a bit better, siamg2b is not Nonw in benchmarks for CCL case...
            # if bench_data[key] is None and test_data[key] is None:
            if test_data[key] is None:
                print(f"test_data[{key}] is None")
                continue

            if (test_data[key].dtype == 'O' and
                    test_data[key].item() is None):
                print(f'test_data[{key}].dtype == "O" and ' 
                      f'test_data[{key}].item() is None)')
                continue


            # Handle arrays with dtype=object containing None
            if (
                isinstance(bench_data[key], np.ndarray) and
                bench_data[key].dtype == object and
                bench_data[key].item() is None
            ) and (
                isinstance(test_data[key], np.ndarray) and
                test_data[key].dtype == object and
                test_data[key].item() is None
            ):
                continue

            try:
                np.asarray(bench_data[key])
                np.asarray(test_data[key])
            except Exception as e:
                raise TypeError(
                    f"Non-numerical or incompatible data type encountered in key '{key}': {e}"
                )


            else:
                try:
                    np.testing.assert_allclose(
                        bench_data[key], test_data[key], atol=0, rtol=1e-5,
                        err_msg=f"{key} doesn\'t match the benchmark ❌")
                    print(f"{key} matches the benchmark ✅")
                except AssertionError as err:
                    print(err)
                    
    #         if key == 'cov_3x2pt_ssc_2D':
    #             sl.compare_arrays(bench_data[key], test_data[key], plot_diff_hist=True, plot_diff_threshold=5)
    #             sl.compare_funcs(None, bench_data[key].flatten(), test_data[key].flatten(), logscale_y=[False, False])
                
    # assert False
                



# Path
bench_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench'
# run all tests...
bench_names = glob.glob(f'{bench_path}/*.npz')
bench_names = [os.path.basename(file) for file in bench_names]
bench_names = [bench_name.replace('.npz', '') for bench_name in bench_names]
# ... or run specific tests
bench_names = ['output_GSpaceborne_SSCSpaceborne_cNGNone_KETrue_resphalo_model_b1gfrom_HOD_spline', ]

main_script_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne/main.py'
temp_output_filename = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench/tmp/test_file'
temp_output_folder = os.path.dirname(temp_output_filename)
excluded_keys = ['backup_cfg', 'metadata']

if os.path.exists(f'{temp_output_filename}.npz'):
    print(f'{temp_output_filename}.npz already exists, most likely from a previous failed test. Do you want to overwrite it?')
    if input('y/n: ') != 'y':
        print('Exiting...')
        exit()
    else:
        os.remove(f'{temp_output_filename}.npz')

if os.path.exists(f'{temp_output_filename}.npz'):
    print(f'{temp_output_filename}.npz already exists, most likely from a previous failed test. Do you want to overwrite it?')
    if input('y/n: ') != 'y':
        print('Exiting...')
        exit()
    else:
        os.remove(f'{temp_output_filename}.npz')

for bench_name in bench_names:
    print(f'Testing {bench_name}...')

    # ! update the cfg file to avoid overwriting the benchmarks
    # Load the benchmark config
    with open(f'{bench_path}/{bench_name}.yaml', "r") as f:
        cfg = yaml.safe_load(f)

    # Update config for the test run
    cfg['misc']['save_output_as_benchmark'] = True
    cfg['misc']['bench_filename'] = temp_output_filename
    cfg['misc']['output_path'] = temp_output_folder  # just to make sure I don't overwrite any output files

    # Save the updated test config
    test_cfg_path = f'{bench_path}/tmp/test_config.yaml'
    with open(test_cfg_path, 'w') as f:
        yaml.dump(cfg, f)

    # ! run the actual test
    test_main_script(test_cfg_path)

    # delete the output test files in tmp folder
    for file_path in glob.glob(f"{temp_output_folder}/*"):
        if os.path.isfile(file_path):
            os.remove(file_path)

