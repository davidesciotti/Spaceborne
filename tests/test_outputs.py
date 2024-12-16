import pytest
import subprocess
import numpy as np
import os
import yaml

def test_main_script(test_cfg_path):
    # Run the main script with the test config
    subprocess.run(['python', 'main.py', '--config', test_cfg_path], check=True)

    # Load the benchmark output
    bench_data = np.load(f'{bench_path}/{bench_name}.npz', allow_pickle=True)

    # Load the test output
    test_data = np.load(f'{temp_output_filename}', allow_pickle=True)

    # Compare the outputs
    for key in bench_data.files:
        if key not in excluded_keys:
            print(f"Comparing {key}...")
            np.testing.assert_allclose(
                bench_data[key], test_data[key], atol=0, rtol=1e-5,
            err_msg=f"Mismatch in {key}")

    print("All outputs match the benchmarks ✅")
    
# Paths
bench_name = 'output_SB_LG' # ! THIS IS THE ONLY THING TO CHANGE

bench_path = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench'
temp_output_filename = '/home/davide/Documenti/Lavoro/Programmi/Spaceborne_bench/tmp/test_file.npz'
temp_output_folder = os.path.dirname(temp_output_filename)
excluded_keys = ['backup_cfg', 'metadata']

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
