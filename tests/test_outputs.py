import pytest
import subprocess
import numpy as np
import os

import yaml


@pytest.fixture
def load_benchmarks(benchmarks_path):
    """Load the benchmark output dictionary including cfg."""
    bench_output_dict = np.load(benchmarks_path, allow_pickle=True).item()
    return bench_output_dict['original_cfg'], bench_output_dict


# @pytest.fixture
# def generate_output_from_cfg(tmp_path, load_benchmarks):
#     """Run the script with the loaded cfg and save the results to a temp folder, then delete the
#     loaded cfg file"""
#     cfg, _ = load_benchmarks

#     # Set the temporary output folder
#     tmp_dir = tmp_path / "outputs"
#     tmp_dir.mkdir()

#     # Update the path in cfg for saving outputs
#     cfg['general_cfg']['save_outputs_as_test_benchmarks_path'] = str(tmp_dir)

#     # save cfg as yaml file
#     with open('../test_cfg.yaml', 'w') as f:
#         f.write(yaml.dump(cfg))

#     # Run your script with the given cfg, using subprocess and saving outputs in temp
#     result = subprocess.run(
#         ['python', '../main.py', f'< ./test_cfg.yaml'],
#         capture_output=True,
#         text=True
#     )

#     # delete test_cfg.yaml
#     os.remove('test_cfg.yaml')

#     # Ensure the script ran successfully
#     assert result.returncode == 0, f"Script failed with error: {result.stderr}"

#     # Load the generated outputs from the temp directory
#     generated_output_dict = np.load(tmp_dir / 'output_dict.npy', allow_pickle=True).item()

#     return generated_output_dict

@pytest.fixture
def generate_output_from_cfg(tmp_path, load_benchmarks):
    """Run the script with the loaded cfg and save the results to a temp folder, then delete the 
    loaded cfg file."""
    cfg, _ = load_benchmarks

    # Update the path in cfg for saving outputs in the temp folder
    cfg['general_cfg']['save_outputs_as_test_benchmarks_path'] = str(tmp_path / 'output_dict.npy')

    # Specify the full path for the test_cfg.yaml file
    cfg_file_path = tmp_path / 'test_cfg.yaml'  # Save in the temp directory

    # Save cfg as yaml file
    with open(cfg_file_path, 'w') as f:
        f.write(yaml.dump(cfg))

    # Run your script with the given cfg, using subprocess and saving outputs in temp
    result = subprocess.run(
        ['python', '../main.py', '--config', str(cfg_file_path)],  # Pass the config file as an argument without quotes
        capture_output=False,
        text=True
    )

    print('finished execution of main.py in test_outputs.py')

    # Ensure the script ran successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Load the generated outputs from the temp directory
    generated_output_dict = np.load(tmp_path / 'output_dict.npy', allow_pickle=True).item()

    return generated_output_dict


@pytest.mark.parametrize("benchmarks_path", [
    './benchmarks/output_dict.npy',
    # You can add more benchmark files if needed
    # './benchmarks/output_dict_1.npy',
    # './benchmarks/output_dict_2.npy'
])
def test_outputs_match_benchmarks(load_benchmarks, generate_output_from_cfg):
    """Compare the outputs from the script with the pre-saved benchmarks."""
    _, bench_output_dict = load_benchmarks
    generated_output_dict = generate_output_from_cfg

    array_names = ['delta', 'gamma', 'ia', 'mu', 'lensing', 'cl_ll_3d', 'cl_gl_3d', 'cl_gg_3d', 'sigma2_b',
                   'z_grid_ssc_integrands']

    for name in array_names:
        np.testing.assert_allclose(
            generated_output_dict[name], bench_output_dict[name], rtol=1e-5, atol=0)

    for key in generated_output_dict['cov_dict'].keys():
        np.testing.assert_allclose(
            generated_output_dict['cov_dict'][key], bench_output_dict['cov_dict'][key], rtol=1e-5, atol=0)

    # for key in generated_output_dict['ell_dict'].keys():
    #     np.testing.assert_allclose(
    #         generated_output_dict['ell_dict'][key], bench_output_dict['ell_dict'][key], rtol=1e-5, atol=0)

    for key in generated_output_dict['fm_dict'].keys():
        np.testing.assert_allclose(
            generated_output_dict['fm_dict'][key], bench_output_dict['fm_dict'][key], rtol=1e-5, atol=0)
