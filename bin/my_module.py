import bz2
from copy import deepcopy
import json
import sys
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import numpy as np
import yaml
from numba import njit
from pynverse import inversefunc
from scipy.interpolate import interp1d
import scipy
import pickle
import itertools
import os
import inspect
import datetime
from tqdm import tqdm
import pandas as pd

# from ..common_cfg import ISTF_fid_params as ISTF_fid


###############################################################################


def write_cl_ascii(ascii_folder, ascii_filename, cl_3d, ells, zbins):

    with open(f'{ascii_folder}/{ascii_filename}.ascii', 'w') as file:
        # Write header
        file.write(f'#ell\ttomo_i\ttomo_j\t{ascii_filename}\n')

        # Iterate over the array and write the data
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    # Format the line with appropriate spacing
                    file.write(f"{ell_val:.3f}\t{zi + 1}\t{zj + 1}\t{value:.10e}\n")

    print(f"Data has been written to {ascii_folder}/{ascii_filename}")


def compare_param_cov_from_fm_pickles(fm_pickle_path_a, fm_pickle_path_b, compare_fms=True, compare_param_covs=True, plot=True, n_params_toplot=10):

    fm_dict_a = load_pickle(fm_pickle_path_a)
    fm_dict_b = load_pickle(fm_pickle_path_b)

    # check that the keys match
    assert fm_dict_a.keys() == fm_dict_b.keys()

    # check if the dictionaries contained in the key 'fiducial_values_dict' match
    assert fm_dict_a['fiducial_values_dict'] == fm_dict_b['fiducial_values_dict'], 'fiducial values do not match!'

    # check that the values match
    for key in fm_dict_a.keys():
        if key != 'fiducial_values_dict' and 'WA' not in key:
            print('Comparing ', key)
            fm_dict_a[key] = remove_null_rows_cols_2D_copilot(fm_dict_a[key])
            fm_dict_b[key] = remove_null_rows_cols_2D_copilot(fm_dict_b[key])

            cov_a = np.linalg.inv(fm_dict_a[key])
            cov_b = np.linalg.inv(fm_dict_b[key])
            
            if compare_fms:
                compare_arrays(fm_dict_a[key], fm_dict_b[key], 'FM_A', 'FM_B', plot_diff_threshold=5)

            if compare_param_covs:

                compare_arrays(cov_a, cov_b, 'cov_A', 'cov_B', plot_diff_threshold=5)
                
            if plot:
                param_names = list(fm_dict_a['fiducial_values_dict'].keys())[:n_params_toplot]
                fiducials_a = list(fm_dict_a['fiducial_values_dict'].values())[:n_params_toplot]
                fiducials_b = list(fm_dict_b['fiducial_values_dict'].values())[:n_params_toplot]
                uncert_a = uncertainties_FM(fm_dict_a[key], n_params_toplot, fiducials=fiducials_a, which_uncertainty='marginal', normalize=True)
                uncert_b = uncertainties_FM(fm_dict_b[key], n_params_toplot, fiducials=fiducials_b, which_uncertainty='marginal', normalize=True)
                diff = percent_diff(uncert_a, uncert_b)
                
                
                plt.figure()
                plt.title(f'Marginalised uncertainties, {key}')
                plt.plot(param_names, uncert_a, label='FM_A')
                plt.plot(param_names, uncert_b, ls='--', label='FM_B')
                plt.plot(param_names, diff, label='percent diff')
                plt.legend()
                


def is_file_created_in_last_x_hours(file_path, hours):
    """
    Check if the specified file was created in the last 24 hours.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    bool: True if the file was created in the last 24 hours, False otherwise.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    # Get the current time
    now = datetime.datetime.now()

    # Get the file creation time
    creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))

    # Calculate the time difference
    time_diff = now - creation_time

    # Check if the difference is less than or equal to 24 hours
    return time_diff <= datetime.timedelta(hours=hours)


def block_diag(array_3d):
    """
    Useful for visualizing nbl, zbins, zbins arrays at a glance
    """
    nbl = array_3d.shape[0]
    return scipy.linalg.block_diag(*[array_3d[ell, :, :] for ell in range(nbl)])


def compare_df_keys(dataframe, key_to_compare, value_a, value_b, num_string_colums):
    """
    This function compares two rows of a dataframe and returns a new row with the percentage difference between the two
    :param dataframe:
    :param key_to_compare:
    :param value_a:
    :param value_b:
    :param num_string_colums: number of columns containing only strings or various options, such as whether to fix a certain prior or not...
    :return:
    """
    df_A = dataframe[dataframe[key_to_compare] == value_a]
    df_B = dataframe[dataframe[key_to_compare] == value_b]
    arr_A = df_A.iloc[:, num_string_colums:].select_dtypes('number').values
    arr_B = df_B.iloc[:, num_string_colums:].select_dtypes('number').values
    perc_diff_df = df_A.copy()
    # ! the reference is G, this might change to G + SSC + cNG
    perc_diff_df.iloc[:, num_string_colums:] = percent_diff(arr_B, arr_A)
    perc_diff_df[key_to_compare] = f'perc_diff_{value_b}'
    perc_diff_df['FoM'] = -perc_diff_df['FoM']  # ! abs? minus??
    dataframe = pd.concat([dataframe, perc_diff_df], axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates()
    return dataframe


def contour_FoM_calculator(sample, param1, param2, sigma_level=1):
    """ Santiago's function to compute the FoM from getDist samples.
    add()sample is a getDist sample object, you need as well the shapely package to compute polygons. The function returns the 1sigma FoM, but in principle you could compute 2-, or 3-sigma "FoMs"
    """
    from shapely.geometry import Polygon
    contour_coords = {}
    density = sample.get2DDensityGridData(j=param1, j2=param2, num_plot_contours=3)
    contour_levels = density.contours
    contours = plt.contour(density.x, density.y, density.P, sorted(contour_levels))
    for ii, contour in enumerate(contours.collections):
        paths = contour.get_paths()
        for path in paths:
            xy = path.vertices
            x = xy[:, 0]
            y = xy[:, 1]
            contour_coords[ii] = list(zip(x, y))
    sigma_lvls = {3: 0, 2: 1, 1: 2}
    poly = Polygon(contour_coords[sigma_lvls[sigma_level]])  # 0:3sigma, 1:2sigma, 2:1sigma
    area = poly.area
    FoM_area = (2.3 * np.pi) / area
    return FoM_area, density


def can_be_pickled(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False


def load_cov_from_probe_blocks(path, filename, probe_ordering):
    """
    Load the covariance matrix from the probe blocks in 4D. The blocks are stored in a dictionary with keys
    corresponding to the probes in the order specified in probe_ordering. The symmetrization of the blocks is done
    while loading, so only 6 blocks need to be actually stored.
    :param path: Path to the folder containing the covariance blocks.
    :param filename: Filename of the covariance blocks. The filename should contain the placeholders {probe_A},
    {probe_B}, {probe_C}, {probe_D} which will be replaced with the actual probe names.
    :param probe_ordering: Probe ordering tuple
    :return:

    YOU SHOULD USE deepcopy, otherwise the different blocks become correlated
    """
    cov_ssc_dict_8D = {}
    for row, (probe_a, probe_b) in enumerate(probe_ordering):
        for col, (probe_c, probe_d) in enumerate(probe_ordering):
            if col >= row:  # Upper triangle and diagonal
                formatted_filename = filename.format(probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d)
                cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = np.load(f"{path}/{formatted_filename}")

                if formatted_filename.endswith('.npz'):
                    cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = cov_ssc_dict_8D[
                        probe_a, probe_b, probe_c, probe_d]['arr_0']

            else:  # Lower triangle, set using symmetry
                cov_ssc_dict_8D[probe_a, probe_b, probe_c, probe_d] = deepcopy(cov_ssc_dict_8D[
                    probe_c, probe_d, probe_a, probe_b].transpose(1, 0, 3, 2))

    for key in cov_ssc_dict_8D.keys():
        assert cov_ssc_dict_8D[key].ndim == 4, (f'covariance matrix {key} has ndim={cov_ssc_dict_8D[key].ndim} instead '
                                                f'of 4')

    return cov_ssc_dict_8D


def save_dict_to_file(dict_data, file_path, indent=4):
    """
    Save a dictionary to a text file in a nicely formatted way.

    Parameters:
    dict_data (dict): The dictionary to be saved.
    file_path (str): The path of the file where the dictionary will be saved.
    indent (int, optional): The number of spaces for indentation. Default is 4.
    """
    with open(file_path, 'w') as f:
        json.dump(dict_data, f, indent=indent)


def figure_of_correlation(correl_matrix):
    """
    Compute the Figure of Correlation (FoC) from the correlation matrix correl_matrix.

    Parameters:
    - correl_matrix (2D numpy array): The correlation matrix.

    Returns:
    - FoC (float): The Figure of Correlation.
    """
    # Invert the correlation matrix
    correl_matrix_inv = np.linalg.inv(correl_matrix)
    # Compute the FoC
    foc = np.sqrt(np.linalg.det(correl_matrix_inv))

    return foc


def plot_correlation_matrix(correlation_matrix, labels, title):
    """
    Plots a heatmap of the given correlation matrix with provided labels.

    Parameters:
    - correlation_matrix (2D numpy array): The correlation matrix to be plotted.
    - labels (list): List of parameter names for labeling the axes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Using the RdBu_r colormap for the heatmap
    # cax = ax.matshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    cax = ax.matshow(correlation_matrix, cmap='viridis')

    # Display color bar
    cbar = fig.colorbar(cax)

    # Set labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate x-axis labels for better clarity
    plt.xticks()

    # Set the title
    ax.set_title(title, pad=20, loc='center', rotation=90)

    # Display the plot
    plt.show()


# Example usage
# Assuming you have a correlation matrix named "correlation_matrix" and labels
# correlation_matrix = ...


def find_inverse_from_array(input_x, input_y, desired_y, interpolation_kind='linear'):
    input_y_func = interp1d(input_x, input_y, kind=interpolation_kind)
    desired_y = inversefunc(input_y_func, y_values=desired_y, domain=(input_x[0], input_x[-1]))
    return desired_y


def plot_bnt_matrix(bnt_matrix, zbins):
    plt.figure()
    plt.matshow(bnt_matrix)
    plt.title('BNT matrix')
    plt.colorbar()
    plt.xlabel('$z_{\\rm bin}$')
    plt.ylabel('$z_{\\rm bin}$')

    # Set tick locations and labels to go from 1 to zbins+1
    tick_positions = np.arange(0, zbins)  # Tick positions are still zero-based
    tick_labels = np.arange(1, zbins + 1).astype(str)  # Tick labels go from 1 to zbins+1

    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)


def add_ls_legend(ls_dict):
    """
    Add a legend for line styles.

    Parameters
    ----------
    ls_dict : dict
        A dictionary mapping line styles to labels.
        E.g. {'-': 'delta', '--': 'gamma'}
    """
    handles = []
    for ls, label in ls_dict.items():
        handles.append(mlines.Line2D([], [], color='black', linestyle=ls, label=label))
    plt.legend(handles=handles, loc='best')


def save_correlation_matrix_plot(matrix1, matrix2, labels):
    """Yet to be fully tested. Plot the positive and negative values with different colorbars
    TODO make it only for one matrix!"""

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 5, width_ratios=[6, 0.3, 6, 0.3, 0.3])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])
    cbar_ax_pos = plt.subplot(gs[3])
    cbar_ax_neg = plt.subplot(gs[4])

    # Define the colormap
    cmap_pos = plt.get_cmap('Reds')
    cmap_neg = plt.get_cmap('Blues_r')

    # Divide positive and negative parts for each matrix
    matrix1_pos = np.ma.masked_less_equal(matrix1, 0)
    matrix1_neg = -np.ma.masked_greater_equal(matrix1, 0)

    matrix2_pos = np.ma.masked_less_equal(matrix2, 0)
    matrix2_neg = -np.ma.masked_greater_equal(matrix2, 0)

    # Plot Gaussian matrix
    ax1.imshow(matrix1_pos, cmap=cmap_pos, norm=LogNorm())
    ax1.imshow(matrix1_neg, cmap=cmap_neg, norm=LogNorm())

    ax1.set_title("Gaussian")
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)

    # Plot Gaussian + SSC matrix
    im2a = ax2.imshow(matrix2_pos, cmap=cmap_pos, norm=LogNorm())
    im2b = ax2.imshow(matrix2_neg, cmap=cmap_neg, norm=LogNorm())

    ax2.set_title("Gaussian + SSC")
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)

    # Add colorbars
    cbar_pos = fig.colorbar(im2a, cax=cbar_ax_pos, orientation='vertical')
    cbar_neg = fig.colorbar(im2b, cax=cbar_ax_neg, orientation='vertical')

    plt.tight_layout()
    plt.show()


def table_to_3d_array(file_path, is_auto_spectrum):
    """convert CLOE .dat files format to 3d arrays, following the indexing in the header
    """

    # Read the header and data from the file
    with open(file_path, 'r') as f:
        header = f.readline().strip().split('\t')

    cl_2d = np.genfromtxt(file_path)
    ells = cl_2d[:, 0]
    cl_2d = cl_2d[:, 1:]
    nbl = len(ells)

    ind = zpairs_header_to_ind_array(header[1:]) - 1  # -1 to revert to 0-based counting
    zbins = np.max(ind) + 1

    cl_3d = np.zeros((nbl, zbins, zbins))
    for ell_idx in range(nbl):
        for zpair_idx, (zi, zj) in enumerate(zip(ind[:, 0], ind[:, 1])):
            cl_3d[ell_idx, zi, zj] = cl_2d[ell_idx, zpair_idx]

    if is_auto_spectrum:
        for ell in range(nbl):
            cl_3d[ell, ...] = symmetrize_2d_array(cl_3d[ell, ...])

    return ells, cl_3d


def zpairs_header_to_ind_array(header):
    # Initialize an empty list to store the pairs of integers
    ind = []

    # Loop through each element in the list
    for zpairs_str_list in header:
        # Split the element by the '-' character
        zpairs_str_list = zpairs_str_list.split('-')

        # Remove the 'E'mand 'P' from each part and convert to integer
        zpairs = [int(zpair_str.replace('E', '').replace('P', '')) for zpair_str in zpairs_str_list]

        # Append the pair of integers to the list
        ind.append(zpairs)

    # Convert the list of pairs into a numpy array
    ind = np.array(ind)

    return ind


def say(msg="Finish", voice="Samantha"):
    os.system(f'say -v {voice} {msg}')


def say_beep():
    sys.stdout.write('\a')
    sys.stdout.flush()


def find_nearest_idx(array, value):
    idx = np.abs(array - value).argmin()
    return idx


def flatten_dict(nested_dict):
    """Flatten a nested dictionary."""
    flattened = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def get_filenames_in_folder(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames


def test_folder_content(output_path, benchmarks_path, extension, verbose=False, rtol=1e-10):
    """Test if the files in the output folder are equal to the benchmark files and list the discrepancies.

    Parameters:
    output_path (str): The path to the folder containing the output files.
    benchmarks_path (str): The path to the folder containing the benchmark files.
    extension (str): The extension of the files to be tested.

    Returns:
    dict: A dictionary containing the comparison results and discrepancies.
    """
    old_files = os.listdir(benchmarks_path)
    new_files = os.listdir(output_path)

    # Ignore hidden files and filter by extension
    old_files = [file for file in old_files if not file.startswith('.') and file.endswith(extension)]
    new_files = [file for file in new_files if not file.startswith('.') and file.endswith(extension)]

    discrepancies = {
        'missing_in_benchmark': set(new_files) - set(old_files),
        'missing_in_output': set(old_files) - set(new_files),
        'comparison_results': []
    }

    max_length_benchmark = max((len(file) for file in discrepancies['missing_in_benchmark']), default=0)
    max_length_output = max((len(file) for file in discrepancies['missing_in_output']), default=0)

    # Use the maximum length from either set for alignment
    max_length = max(max_length_benchmark, max_length_output)

    for file in discrepancies['missing_in_benchmark']:
        print(f"{file:<{max_length}} \t output ✅ \t benchmark ❌")

    for file in discrepancies['missing_in_output']:
        print(f"{file:<{max_length}} \t output ❌ \t benchmark ✅")
        print(f"{file} \t output ❌ \t benchmark ✅")

    for file_name in set(old_files).intersection(new_files):
        old_file_path = os.path.join(benchmarks_path, file_name)
        new_file_path = os.path.join(output_path, file_name)

        try:
            if extension == 'npz':
                np.testing.assert_allclose(np.load(old_file_path)['arr_0'], np.load(new_file_path)['arr_0'],
                                           verbose=verbose, rtol=rtol, atol=0)
            elif extension == 'npy':
                np.testing.assert_allclose(np.load(old_file_path), np.load(new_file_path), verbose=verbose,
                                           rtol=rtol, atol=0)
            elif extension == 'txt' or extension == 'dat':
                np.testing.assert_allclose(np.genfromtxt(old_file_path), np.genfromtxt(new_file_path),
                                           verbose=verbose, rtol=rtol, atol=0)
            else:
                raise ValueError(f"Unknown extension: {extension}")
        except AssertionError as exc:
            discrepancies['comparison_results'].append((file_name, str(exc)))
            print(f'\nFile {file_name} does not match: {exc}')
        else:
            discrepancies['comparison_results'].append((file_name, 'Match'))
            print(f"{file_name:<{max_length}} \t matches to within {rtol*100}% ✅")

    # Provide a summary of the results
    num_comparisons = len(discrepancies['comparison_results'])
    num_matches = sum(1 for _, result in discrepancies['comparison_results'] if result == 'Match')
    num_discrepancies = num_comparisons - num_matches

    print(f"\nSummary: {num_comparisons} files compared, {num_matches} matches, {num_discrepancies} discrepancies.\n")

    return discrepancies


def import_files(folder_path, extension):
    """
    Imports all files with a specific extension from a folder and stores their contents in a dictionary.

    Parameters:
    folder_path (str): The path to the folder from which to import files.
    extension (str): The file extension to look for.

    Returns:
    dict: A dictionary with filenames as keys and file contents as values.
    """
    files_dict = {}
    # Get the list of relevant file names using the get_filenames_in_folder function
    filenames = get_filenames_in_folder(folder_path)

    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        # Determine how to load the file based on the extension
        if extension in ['.txt', '.csv']:
            file_content = np.genfromtxt(file_path, delimiter=',')  # Adjust delimiter if necessary
        elif extension == '.npy':
            file_content = np.load(file_path)
        elif extension == '.npz':
            file_content = np.load(file_path)['arr_0']
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        # Use the file name without an extension as the key
        key = os.path.splitext(filename)[0]
        files_dict[key] = file_content

    return files_dict


def is_increasing(arr):
    return np.all(np.diff(arr) > 0)


def save_pickle(filename, obj):
    with open(f'{filename}', 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(filename):
    with open(f'{filename}', 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def save_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'wb') as handle:
        pickle.dump(data, handle)


def load_compressed_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


def read_yaml(filename):
    """ A function to read YAML file. filename must include the path and the extension"""
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config


@njit
def percent_diff(array_1, array_2, abs_value=False):
    diff = (array_1 / array_2 - 1) * 100
    if abs_value:
        return np.abs(diff)
    else:
        return diff


@njit
def percent_diff_mean(array_1, array_2):
    """
    result is in "percent" units
    """
    mean = (array_1 + array_2) / 2.0
    diff = (array_1 / mean - 1) * 100
    return diff


@njit
def percent_diff_nan(array_1, array_2, eraseNaN=True, log=False, abs_val=False):
    if eraseNaN:
        diff = np.where(array_1 == array_2, 0, percent_diff(array_1, array_2))
    else:
        diff = percent_diff(array_1, array_2)
    if log:
        diff = np.log10(diff)
    if abs_val:
        diff = np.abs(diff)
    return diff


def diff_threshold_check(diff, threshold):
    boolean = np.any(np.abs(diff) > threshold)
    print(f"has any element of the arrays a disagreement > {threshold}%? ", boolean)


def compute_smape(vec_true, vec_test, cov_mat=None):
    """
    Computes the SMAPE (Symmetric Mean Absolute Percentage Error) for a given 1D array with weighted elements

    Args:
        vec_true (np.array): array of true values
        vec_test (np.array): array of predicted/approximated values
        cov_mat (np.array): covariance matrix for vec_true

    Returns:
        float: SMAPE value
    """
    if type(vec_true) == np.ndarray and type(vec_test) == np.ndarray:
        assert len(vec_true) == len(vec_test), "Arrays must have the same length"
        assert vec_true.ndim == 1 and vec_test.ndim == 1, 'arrays must be 1D'

    if cov_mat is not None:
        assert cov_mat.shape[0] == cov_mat.shape[1] == len(vec_true), 'cov_mat must be a square matrix with the same ' \
                                                                      'length as the input vectors'
        weights = vec_true / np.sqrt(np.diag(cov_mat))
    else:
        weights = np.ones_like(vec_true)  # uniform weights

    numerator = weights * np.abs(vec_true - vec_test)
    denominator = np.abs(vec_true) + np.abs(vec_test)

    return 100 * np.mean(numerator / denominator)  # the output is already a precentage


def compute_diff_sigma(vec_true, vec_test, sigma):
    """
    Compute the element-wise difference between two vectors vec_true and vec_test,
    and divide it by sigma.

    Args:
    - vec_true (numpy.ndarray): A numpy array representing the first vector.
    - vec_test (numpy.ndarray): A numpy array representing the second vector.
    - sigma (numpy.ndarray): A numpy array representing the vector of standard deviations.

    Returns:
    - A numpy array representing the element-wise difference between vec_true and vec_test,
      divided by sigma.
    """
    diff = np.abs(vec_true - vec_test)
    return 100 * diff / sigma


def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def compare_arrays_v0(A, B, name_A='A', name_B='B', plot_diff=True, plot_array=True, log_array=True, log_diff=False,
                      abs_val=False, plot_diff_threshold=None, white_where_zero=True):
    if plot_diff or plot_array:
        assert A.ndim == 2 and B.ndim == 2, 'plotting is only implemented for 2D arrays'

    # white = to_rgb('white')
    # cmap = ListedColormap([white] + plt.cm.viridis(np.arange(plt.cm.viridis.N)))
    # # set the color for 0 values as white and all other values to the standard colormap
    # cmap = plt.cm.viridis
    # cmap.set_bad(color=white)

    if plot_diff:

        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=log_diff, abs_val=abs_val)
        diff_BA = percent_diff_nan(B, A, eraseNaN=True, log=log_diff, abs_val=abs_val)

        if not np.allclose(diff_AB, diff_BA, rtol=1e-3, atol=0):
            print('diff_AB and diff_BA have a relative difference of more than 1%')

        if plot_diff_threshold is not None:
            # take the log of the threshold if using the log of the precent difference
            if log_diff:
                plot_diff_threshold = np.log10(plot_diff_threshold)

            print(f'plotting the *absolute value* of the difference only where it is below the given threshold '
                  f'({plot_diff_threshold}%)')
            diff_AB = np.ma.masked_where(np.abs(diff_AB) < plot_diff_threshold, np.abs(diff_AB))
            diff_BA = np.ma.masked_where(np.abs(diff_BA) < plot_diff_threshold, np.abs(diff_BA))

        fig, ax = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
        im = ax[0].matshow(diff_AB)
        ax[0].set_title(f'(A/B - 1) * 100')
        fig.colorbar(im, ax=ax[0])

        im = ax[1].matshow(diff_BA)
        ax[1].set_title(f'(B/A - 1) * 100')
        fig.colorbar(im, ax=ax[1])

        fig.suptitle(f'log={log_diff}, abs={abs_val}')
        plt.show()

    if plot_array:
        A_toplot, B_toplot = A, B

        if abs_val:
            A_toplot, B_toplot = np.abs(A), np.abs(B)
        if log_array:
            A_toplot, B_toplot = np.log10(A), np.log10(B)

        fig, ax = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
        im = ax[0].matshow(A_toplot)
        ax[0].set_title(f'{name_A}')
        fig.colorbar(im, ax=ax[0])

        im = ax[1].matshow(B_toplot)
        ax[1].set_title(f'{name_B}')
        fig.colorbar(im, ax=ax[1])
        fig.suptitle(f'log={log_array}, abs={abs_val}')
        plt.show()

    if np.array_equal(A, B):
        print('A and B are equal ✅')
        return

    for rtol in [1e-5, 1e-3, 1e-2, 5e-2, 1e-1]:  # these are NOT percent units, see print below
        if np.allclose(A, B, rtol=rtol, atol=0):
            print(f'{name_A} and {name_B} are close within relative tolerance of {rtol * 100}%) ✅')
            return

    diff_AB = percent_diff_nan(A, B, eraseNaN=True, abs_val=True)
    higher_rtol = plot_diff_threshold  # in "percent" units
    if higher_rtol is None:
        higher_rtol = 5.0
    result_emoji = '❌'
    no_outliers = np.where(diff_AB > higher_rtol)[0].shape[0]
    additional_info = f'\nMax discrepancy: {np.max(diff_AB):.2f}%;' \
                      f'\nNumber of elements with discrepancy > {higher_rtol}%: {no_outliers}' \
                      f'\nFraction of elements with discrepancy > {higher_rtol}%: {no_outliers / diff_AB.size:.5f}'
    print(f'Are {name_A} and {name_B} different by less than {higher_rtol}%? {result_emoji} {additional_info}')


def compare_arrays(A, B, name_A='A', name_B='B', plot_diff=True, plot_array=True, log_array=True, log_diff=False,
                   abs_val=False, plot_diff_threshold=None, white_where_zero=True):

    if np.array_equal(A, B):
        print(f'{name_A} and {name_B} are equal ✅')
        return
    else:
        for rtol in [1e-3, 1e-2, 5e-2]:  # these are NOT percent units
            if np.allclose(A, B, rtol=rtol, atol=0):
                print(f'{name_A} and {name_B} are close within relative tolerance of {rtol * 100}%) ✅')
                return

        diff_AB = percent_diff_nan(A, B, eraseNaN=True, abs_val=True)
        higher_rtol = plot_diff_threshold or 5.0
        result_emoji = '❌'
        no_outliers = np.sum(diff_AB > higher_rtol)
        additional_info = f'\nMax discrepancy: {np.max(diff_AB):.2f}%;' \
                          f'\nNumber of elements with discrepancy > {higher_rtol}%: {no_outliers}' \
                          f'\nFraction of elements with discrepancy > {higher_rtol}%: {no_outliers / diff_AB.size:.5f}'
        print(f'Are {name_A} and {name_B} different by less than {higher_rtol}%? {result_emoji} {additional_info}')

        if plot_diff or plot_array:
            assert A.ndim == 2 and B.ndim == 2, 'plotting is only implemented for 2D arrays'

        if plot_diff:
            diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=log_diff, abs_val=abs_val)

            if plot_diff_threshold is not None:
                # take the log of the threshold if using the log of the precent difference
                if log_diff:
                    plot_diff_threshold = np.log10(plot_diff_threshold)

                diff_AB = np.ma.masked_where(np.abs(diff_AB) < plot_diff_threshold, np.abs(diff_AB))

            fig, ax = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
            im = ax[0].matshow(diff_AB)
            ax[0].set_title(f'(A/B - 1) * 100')
            fig.colorbar(im, ax=ax[0])

            im = ax[1].matshow(diff_AB)
            ax[1].set_title(f'(A/B - 1) * 100')
            fig.colorbar(im, ax=ax[1])

            fig.suptitle(f'log={log_diff}, abs={abs_val}')
            plt.show()

        if plot_array:
            A_toplot, B_toplot = A, B

            if abs_val:
                A_toplot, B_toplot = np.abs(A), np.abs(B)
            if log_array:
                A_toplot, B_toplot = np.log10(A), np.log10(B)

            fig, ax = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
            im = ax[0].matshow(A_toplot)
            ax[0].set_title(f'{name_A}')
            fig.colorbar(im, ax=ax[0])

            im = ax[1].matshow(B_toplot)
            ax[1].set_title(f'{name_B}')
            fig.colorbar(im, ax=ax[1])
            fig.suptitle(f'log={log_array}, abs={abs_val}')
            plt.show()


def compare_folder_content(path_A: str, path_B: str, filetype: str):
    """
    Compare the content of 2 folders. The files in folder A should be a subset of the files in folder B.
    """
    dict_A = dict(get_kv_pairs(path_A, filetype))
    dict_B = dict(get_kv_pairs(path_B, filetype))

    for key in dict_A.keys():
        if np.array_equal(dict_A[key], dict_B[key]):
            result_emoji = '✅'
        else:
            result_emoji = '❌'
        print(f'is {key} equal in both folders? {result_emoji}')


def namestr(obj, namespace):
    """ does not work with slices!!! (why?)"""
    return [name for name in namespace if namespace[name] is obj][0]


def plot_FM(array, style=".-"):
    name = namestr(array, globals())
    plt.plot(range(7), array, style, label=name)


################################################ Fisher Matrix utilities ################################################
def find_null_rows_cols_2D(array):
    """
    :param array:

    :return null_rows_idxs: list
        array of null rows/columns indices
    """
    assert array.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    null_rows_idxs = np.where(np.all(array == 0, axis=0))[0]
    null_cols_idxs = np.where(np.all(array == 0, axis=1))[0]

    assert np.array_equal(null_rows_idxs, null_cols_idxs), \
        'null rows and columns indices should be the same for Fisher matrices'

    if null_rows_idxs.shape[0] == 0:
        # print('The input array has no null rows/columns')
        return None
    else:
        # print(f'The input array had null rows and columns at indices {null_rows_idxs}')
        return null_rows_idxs


def remove_rows_cols_array2D(array, rows_idxs_to_remove):
    """
    Removes the *same* rows and columns from an array
    :param array: numpy.ndarray. input 2D array
    :param rows_idxs_to_remove: list. rows (and columns) to delete
    :return: array without null rows and columns
    """
    if rows_idxs_to_remove is None:
        warnings.warn('null_rows_idxs is None, returning the input array')
        return array

    if len(rows_idxs_to_remove) == 0:
        warnings.warn('null_rows_idxs is empty, returning the input array')
        return array

    assert array.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    array = np.delete(array, rows_idxs_to_remove, axis=0)
    array = np.delete(array, rows_idxs_to_remove, axis=1)
    return array


def remove_null_rows_cols_2D_copilot(array_2d):
    """
    Remove null rows and columns from a 2D array - version by GitHub Copilot
    """

    assert array_2d.ndim == 2, 'ndim should be <= 2; higher-dimensional case not yet implemented'
    array_2d = array_2d[~np.all(array_2d == 0, axis=1)]
    array_2d = array_2d[:, ~np.all(array_2d == 0, axis=0)]
    return array_2d


def mask_FM_null_rowscols(FM, params, fid):
    """
    Mask the Fisher matrix, fiducial values and parameter list deleting the null rows and columns
    :param FM: Fisher Matrix, 2D numpy array
    :return: masked FM, fiducial values and parameter list
    """
    null_idx = find_null_rows_cols_2D(FM)

    if null_idx is None:
        return FM, params, fid

    FM = remove_rows_cols_array2D(FM, null_idx)
    params = np.delete(params, obj=null_idx, axis=0)
    fid = np.delete(fid, obj=null_idx, axis=0)
    assert len(fid) == len(params), 'the fiducial values and parameter lists should have the same length'
    return FM, list(params), list(fid)


def mask_fm_null_rowscols_v2(fm, fiducials_dict):
    """
    Mask the Fisher matrix, fiducial values and parameter list deleting the null rows and columns
    :param FM: Fisher Matrix, 2D numpy array
    :return: masked FM, fiducial values and parameter list
    """
    param_names = list(fiducials_dict.keys())
    param_values = list(fiducials_dict.values())

    null_idxs = find_null_rows_cols_2D(fm)

    if null_idxs is None:
        return fm, fiducials_dict

    # remove null rows and columns
    fm = remove_rows_cols_array2D(fm, null_idxs)
    # update fiducials dict
    fiducials_dict = {param_names[i]: param_values[i] for i in range(len(param_names)) if i not in null_idxs}

    return fm, fiducials_dict


def mask_FM(FM, param_names_dict, fiducials_dict, params_tofix_dict, remove_null_rows_cols=True):
    """
    Trim the Fisher matrix to remove null rows/columns and/or fix nuisance parameters
    :param FM:
    :param remaining_param_names_list:
    :param fid:
    :param n_cosmo_params:
    :param kwargs:
    :return:
    """

    if type(param_names_dict) == list or type(param_names_dict) == np.ndarray:
        param_names_dict = {'_': param_names_dict}
    if type(fiducials_dict) == list or type(fiducials_dict) == np.ndarray:
        fiducials_dict = {'_': fiducials_dict}
    if type(params_tofix_dict) == list or type(params_tofix_dict) == np.ndarray:
        params_tofix_dict = {'_': params_tofix_dict}

    # join param_names_dict.values() into single list
    all_param_names_list = list(itertools.chain(*list(param_names_dict.values())))
    all_fiducials_list = list(itertools.chain(*list(fiducials_dict.values())))

    # TODO - add option to fix specific parameter
    # TODO  - test this!!
    idx_todelete = []
    for key in params_tofix_dict.keys():
        if params_tofix_dict[key]:
            _param_names_list = param_names_dict[key]
            param_idxs = [all_param_names_list.index(param_name) for param_name in _param_names_list]
            idx_todelete.append(param_idxs)

    # make a continuous list
    # idx_todelete = np.flatten(idx_todelete.flatten())
    idx_todelete = list(itertools.chain(*idx_todelete))

    if idx_todelete:
        FM = np.delete(FM, idx_todelete, axis=0)
        FM = np.delete(FM, idx_todelete, axis=1)
        remaining_param_names_list = np.delete(all_param_names_list, idx_todelete)
        remaining_fiducials_list = np.delete(all_fiducials_list, idx_todelete)
    else:
        remaining_param_names_list = all_param_names_list
        remaining_fiducials_list = all_fiducials_list

    # remove remaining null rows_cols
    if remove_null_rows_cols:
        FM, remaining_param_names_list, remaining_fiducials_list = mask_FM_null_rowscols(FM, remaining_param_names_list,
                                                                                         remaining_fiducials_list)

    return FM, list(remaining_param_names_list), list(remaining_fiducials_list)


def mask_fm_v2(fm: np.ndarray, fiducials_dict: dict, names_params_to_fix: list, remove_null_rows_cols: bool):
    """
    Trim the Fisher matrix to remove null rows/columns and/or fix nuisance parameters
    """

    if names_params_to_fix is not None:
        fm = fix_params_in_fm(fm, names_params_to_fix, fiducials_dict)  # cut fm entries
        # update fiducials_dict
        fiducials_dict = {key: fiducials_dict[key] for key in fiducials_dict.keys() if key not in names_params_to_fix}

    # remove remaining null rows_cols
    if remove_null_rows_cols:
        fm, fiducials_dict = mask_fm_null_rowscols_v2(fm, fiducials_dict)

    return fm, fiducials_dict


def fix_params_in_fm(fm, names_params_to_fix, fiducials_dict):
    param_names = list(fiducials_dict.keys())

    # check the correctness of the parameters' names
    for param_to_fix in names_params_to_fix:
        assert param_to_fix in param_names, f'Parameter {param_to_fix} not found in param_names!'

    rows_idxs_to_remove = [param_names.index(param_to_fix) for param_to_fix in names_params_to_fix]
    fm = remove_rows_cols_array2D(fm, rows_idxs_to_remove)
    # print(f'Removing rows and columns from FM:\n{rows_idxs_to_remove}')

    return fm


def add_prior_to_fm(fm, fiducials_dict, prior_param_names, prior_param_values):
    """ adds a FM of priors (with elements 1/sigma in the correct positions) to the input FM"""

    assert len(list(fiducials_dict.keys())) == fm.shape[0] == fm.shape[1], 'Wrong shape of FM matrix!'
    fid_param_names = list(fiducials_dict.keys())

    for prior_param_name in prior_param_names:
        if prior_param_name not in fid_param_names:
            warnings.warn(f'Prior parameter {prior_param_names} not found in fiducial parameters dict!')
            return fm

    prior_param_idxs = [fid_param_names.index(prior_param_name) for prior_param_name in prior_param_names]

    prior_fm = np.zeros(fm.shape)
    prior_fm[prior_param_idxs, prior_param_idxs] = 1 / np.array(prior_param_values)
    return fm + prior_fm


def uncertainties_FM(FM, nparams, fiducials=None, which_uncertainty='marginal', normalize=True):
    """
    returns relative *percentage!* error
    """

    if which_uncertainty == 'marginal':
        FM_inv = np.linalg.inv(FM)
        sigma_FM = np.sqrt(np.diag(FM_inv))[:nparams] * 100
    elif which_uncertainty == 'conditional':
        sigma_FM = np.sqrt(1 / np.diag(FM))[:nparams] * 100
    else:
        raise ValueError('which_uncertainty must be either "marginal" or "conditional"')

    if normalize:
        fiducials = np.asarray(fiducials)  # turn list into array to make np.where work

        assert fiducials.shape[0] == nparams, 'the fiducial must have the same length as the number of parameters'

        if fiducials is None:
            assert False, 'you should definetly provide fiducial values!'
            print('No fiducial values provided, using the ISTF values (for flat w0waCDM cosmology and no extensions)')
            fiducials = np.asarray(list(ISTF_fid.primary.values())[:7])

        # if the fiducial for is 0, substitute with 1 to avoid division by zero; if it's -1, take the absolute value
        fiducials = np.where(fiducials == 0, 1, fiducials)
        fiducials = np.where(fiducials == -1, 1, fiducials)
        sigma_FM /= fiducials

    return sigma_FM


def uncertainties_fm_v2(fm, fiducials_dict, which_uncertainty='marginal', normalize=True, percent_units=True):
    """
    returns relative 1-sigma error
    """

    param_names = list(fiducials_dict.keys())
    param_values = np.array(list(fiducials_dict.values()))

    # pdb.set_trace()

    assert len(param_names) == param_values.shape[0] == fm.shape[0] == fm.shape[1], \
        'param_names and param_values must have the same length and be equal to the number of rows and columns of fm'

    if which_uncertainty == 'marginal':
        fm_inv = np.linalg.inv(fm)
        sigma_fm = np.sqrt(np.diag(fm_inv))
    elif which_uncertainty == 'conditional':
        sigma_fm = np.sqrt(1 / np.diag(fm))
    else:
        raise ValueError('which_uncertainty must be either "marginal" or "conditional"')

    if normalize:
        # if the fiducial for is 0, substitute with 1 to avoid division by zero; if it's -1, take the absolute value
        param_values = np.where(param_values == 0, 1, param_values)
        param_values = np.where(param_values == -1, 1, param_values)
        # normalize to get the relative uncertainties
        sigma_fm /= param_values

    if percent_units:
        return sigma_fm * 100

    return sigma_fm


def build_labels_TeX(zbins):
    galaxy_bias_label = ['$b_{%i}$' % (i + 1) for i in range(zbins)]
    shear_bias_label = ['$m_{%i}$' % (i + 1) for i in range(zbins)]
    zmean_shift_label = ['$dz_{%i}$' % (i + 1) for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def build_labels(zbins):
    galaxy_bias_label = [f'b{(i + 1):02d}' for i in range(zbins)]
    shear_bias_label = [f'm{(i + 1):02d}' for i in range(zbins)]
    zmean_shift_label = [f'dz{(i + 1):02d}' for i in range(zbins)]
    return [galaxy_bias_label, shear_bias_label, zmean_shift_label]


def matshow(array, title="title", log=False, abs_val=False, threshold=None, only_show_nans=False):
    """
    :param array:
    :param title:
    :param log:
    :param abs_val:
    :param threshold: if None, do not mask the values; otherwise, keep only the elements above the threshold
    (i.e., mask the ones below the threshold)
    :return:
    """

    if only_show_nans:
        warnings.warn('only_show_nans is True, better switch off log and abs_val for the moment')
        # Set non-NaN elements to 0 and NaN elements to 1
        array = np.where(np.isnan(array), 1, 0)
        title += ' (only NaNs shown)'

    # the ordering of these is important: I want the log(abs), not abs(log)
    if abs_val:  # take the absolute value
        array = np.abs(array)
        title = 'abs ' + title
    if log:  # take the log
        array = np.log10(array)
        title = 'log10 ' + title

    if threshold is not None:
        array = np.ma.masked_where(array < threshold, array)
        title += f" \n(masked below {threshold} \%)"

    plt.matshow(array)
    plt.colorbar()
    plt.title(title)
    plt.show()


def get_kv_pairs(path_import, extension='npy'):
    """
    Load txt or dat files in dictionary.
    To use it, wrap it in "dict(), e.g.:
        loaded_dict = dict(get_kv_pairs(path_import, filetype="dat"))
    """
    if extension == 'npy' or extension == 'npz':
        load_function = np.load
    elif extension == 'txt' or extension == 'dat':
        load_function = np.genfromtxt
    else:
        raise NotImplementedError("extension must be either 'npy', 'npz', 'txt' or 'dat'")

    for path in Path(path_import).glob(f"*.{extension}"):
        yield path.stem, load_function(str(path))


# to display the names (keys) more tidily
def show_keys(arrays_dict):
    for key in arrays_dict:
        print(key)


def cl_interpolator(cl_2D, zpairs, new_ell_values, nbl, kind='linear'):
    original_ell_values = cl_2D[:, 0]

    # switch to linear scale, the "15" is arbitrary
    if original_ell_values.max() < 15:
        original_ell_values = 10 ** original_ell_values
    if new_ell_values.max() < 15:
        new_ell_values = 10 ** new_ell_values

    cl_interpolated = np.zeros((nbl, zpairs))
    for zpair_idx in range(zpairs):
        f = interp1d(original_ell_values, cl_2D[:, zpair_idx + 1], kind=kind)
        cl_interpolated[:, zpair_idx] = f(new_ell_values)
    return cl_interpolated


# def cl_interpolator_no_1st_column(npairs, cl_2D, original_ell_values, new_ell_values, nbl):
#     Cl_interpolated = np.zeros((nbl, npairs))
#     for j in range(npairs):
#         f = interp1d(original_ell_values, cl_2D[:, j], kind='linear')
#         Cl_interpolated[:, j] = f(new_ell_values)
#     return Cl_interpolated


@njit
def symmetrize_Cl(Cl, nbl, zbins):
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                Cl[ell, j, i] = Cl[ell, i, j]
    return Cl


def generate_ind(triu_tril_square, row_col_major, size):
    """
    Generates a list of indices for the upper triangular part of a matrix
    :param triu_tril_square: str. if 'triu', returns the indices for the upper triangular part of the matrix.
    If 'tril', returns the indices for the lower triangular part of the matrix
    If 'full_square', returns the indices for the whole matrix
    :param row_col_major: str. if True, the indices are returned in row-major order; otherwise, in column-major order
    :param size: int. size of the matrix to take the indices of
    :return: list of indices
    """
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'
    assert triu_tril_square in ['triu', 'tril', 'full_square'], 'triu_tril_square must be either "triu", "tril" or ' \
                                                                '"full_square"'

    if triu_tril_square == 'triu':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i, size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i + 1)]
    elif triu_tril_square == 'tril':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i + 1)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i, size)]
    elif triu_tril_square == 'full_square':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(size)]

    return np.asarray(ind)


def build_full_ind(triu_tril, row_col_major, size):
    """
    Builds the good old ind file
    """

    assert triu_tril in ['triu', 'tril'], 'triu_tril must be either "triu" or "tril"'
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    LL_columns = np.zeros((zpairs_auto, 2))
    GL_columns = np.hstack((np.ones((zpairs_cross, 1)), np.zeros((zpairs_cross, 1))))
    GG_columns = np.ones((zpairs_auto, 2))

    LL_columns = np.hstack((LL_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)
    GL_columns = np.hstack((GL_columns, generate_ind('full_square', row_col_major, size))).astype(int)
    GG_columns = np.hstack((GG_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)

    ind = np.vstack((LL_columns, GL_columns, GG_columns))

    assert ind.shape[0] == zpairs_3x2pt, 'ind has the wrong number of rows'

    return ind


# CHECK FOR DUPLICATES
def cl_2D_to_3D_symmetric(Cl_2D, nbl, zpairs, zbins):
    """ reshape from (nbl, zpairs) to (nbl, zbins, zbins) according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    assert Cl_2D.shape == (nbl, zpairs), f'cl_2d must have shape (nbl, zpairs) = ({nbl}, {zpairs})'

    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for zpair_idx in range(zpairs):
            i, j = triu_idx[0][zpair_idx], triu_idx[1][zpair_idx]
            Cl_3D[ell, i, j] = Cl_2D[ell, zpair_idx]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def cl_2D_to_3D_symmetric_bu(Cl_2D, nbl, zpairs, zbins):
    """ reshape from (nbl, zpairs) to (nbl, zbins, zbins) according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for zpair_idx in range(zpairs):
            i, j = triu_idx[0][zpair_idx], triu_idx[1][zpair_idx]
            Cl_3D[ell, i, j] = Cl_2D[ell, zpair_idx]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def cl_2D_to_3D_asymmetric(Cl_2D, nbl, zbins, order):
    """ reshape from (nbl, npairs) to (nbl, zbins, zbins), rows first
    (valid for asymmetric Cij, i.e. C_XC)
    """
    assert order in ['row-major', 'col-major', 'C', 'F'], 'order must be either "row-major", "C" (equivalently), or' \
                                                          '"col-major", "F" (equivalently)'
    if order == 'row-major':
        order = 'C'
    elif order == 'col-major':
        order = 'F'

    Cl_3D = np.zeros((nbl, zbins, zbins))
    Cl_3D = np.reshape(Cl_2D, Cl_3D.shape, order=order)
    return Cl_3D


def Cl_3D_to_2D_symmetric(Cl_3D, nbl, npairs, zbins=10):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs)  according to
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_2D = np.zeros((nbl, npairs))
    for ell in range(nbl):
        for i in range(npairs):
            Cl_2D[ell, i] = Cl_3D[ell, triu_idx[0][i], triu_idx[1][i]]
    return Cl_2D


def Cl_3D_to_2D_asymmetric(Cl_3D):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs), rows first 
    (valid for asymmetric Cij, i.e. C_XC)
    """
    assert Cl_3D.ndim == 3, 'Cl_3D must be a 3D array'

    nbl = Cl_3D.shape[0]
    zbins = Cl_3D.shape[1]
    zpairs_cross = zbins ** 2

    Cl_2D = np.reshape(Cl_3D, (nbl, zpairs_cross))

    # Cl_2D = np.zeros((nbl, zpairs_cross))
    # for ell in range(nbl):
    #     Cl_2D[ell, :] = Cl_3D[ell, :].flatten(order='C')
    return Cl_2D


def cl_3D_to_2D_or_1D(cl_3D, ind, is_auto_spectrum, use_triu_row_major, convert_to_2D, block_index):
    """ reshape from (nbl, zbins, zbins) to (nbl, zpairs), according to the ordering given in the ind file
    (valid for asymmetric Cij, i.e. C_XC)
    """

    warnings.warn('finish this function!!')
    assert cl_3D.ndim == 3, 'cl_3D must be a 3D array'
    assert cl_3D.shape[1] == cl_3D.shape[2], 'cl_3D must be a square array of shape (nbl, zbins, zbins)'

    nbl = cl_3D.shape[0]
    zbins = cl_3D.shape[1]
    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    if use_triu_row_major:
        ind = build_full_ind('triu', 'row-major', zbins)

    if is_auto_spectrum:
        ind = ind[:zpairs_auto, :]
        zpairs = zpairs_auto
    elif not is_auto_spectrum:
        ind = ind[zpairs_auto:zpairs_cross, :]
        zpairs = zpairs_cross
    else:
        raise ValueError('is_auto_spectrum must be either True or False')

    cl_2D = np.zeros((nbl, zpairs))
    for ell in range(nbl):
        zpair = 0
        for zi, zj in ind[:, 2:]:
            cl_2D[ell, zpair] = cl_3D[ell, zi, zj]
            zpair += 1

    if convert_to_2D:
        return cl_2D

    if block_index == 'ell' or block_index == 'vincenzo':
        cl_1D = cl_2D.flatten(order='C')
    elif block_index == 'ij' or block_index == 'sylvain':
        cl_1D = cl_2D.flatten(order='F')
    else:
        raise ValueError('block_index must be either "ij" or "ell"')

    return cl_1D


@njit
def cl_1D_to_3D(cl_1d, nbl: int, zbins: int, is_symmetric: bool):
    """ This is used to unpack Vincenzo's files for SPV3
    Still to be thoroughly checked."""

    cl_3d = np.zeros((nbl, zbins, zbins))
    p = 0
    if is_symmetric:
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(iz, zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1

    else:  # take all elements, not just the upper triangle
        for ell in range(nbl):
            for iz in range(zbins):
                for jz in range(zbins):
                    cl_3d[ell, iz, jz] = cl_1d[p]
                    p += 1
    return cl_3d


# XXX NEW AND CORRECTED FUNCTIONS TO MAKE THE Cl 3D
###############################################################################
def array_2D_to_3D_ind(array_2D, nbl, zbins, ind, start, stop):
    # ! is this to be deprecated??
    """ unpack according to "ind" ordering the same as the Cl!! """
    print('attention, assuming npairs = 55 (that is, zbins = 10)!')
    array_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for k, p in enumerate(range(start, stop)):
            array_3D[ell, ind[p, 2], ind[p, 3]] = array_2D[ell, k]
            # enumerate is in case p deosn't start from p, that is, for LG
    return array_3D


# @njit
def symmetrize_2d_array(array_2d):
    """ mirror the lower/upper triangle """

    # if already symmetric, do nothing
    if check_symmetric(array_2d, exact=True):
        return array_2d

    # there is an implicit "else" here, since the function returns array_2d if the array is symmetric
    assert array_2d.ndim == 2, 'array must be square'
    size = array_2d.shape[0]

    # check that either the upper or lower triangle (not including the diagonal) is null
    triu_elements = array_2d[np.triu_indices(size, k=+1)]
    tril_elements = array_2d[np.tril_indices(size, k=-1)]
    assert np.all(triu_elements) == 0 or np.all(tril_elements) == 0, 'neither the upper nor the lower triangle ' \
                                                                     '(excluding the diagonal) are null'

    assert np.any(np.diag(array_2d)) != 0, 'the diagonal elements are all null. ' \
                                           'This is not necessarily an error, but is suspect'

    # symmetrize
    array_2d = np.where(array_2d, array_2d, array_2d.T)
    # check
    if not check_symmetric(array_2d, exact=False):
        warnings.warn('check failed: the array is not symmetric')

    return array_2d


def fill_3D_symmetric_array(array_3D, nbl, zbins):
    """ mirror the lower/upper triangle """
    assert array_3D.shape == (nbl, zbins, zbins), 'shape of input array must be (nbl, zbins, zbins)'

    array_diag_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        array_diag_3D[ell, :, :] = np.diag(np.diagonal(array_3D, 0, 1, 2)[ell, :])
    array_3D = array_3D + np.transpose(array_3D, (0, 2, 1)) - array_diag_3D
    return array_3D


def array_2D_to_1D_ind(array_2D, zpairs, ind):
    """ unpack according to "ind" ordering, same as for the Cls """
    assert ind.shape[0] == zpairs, 'ind must have lenght zpairs'

    array_1D = np.zeros(zpairs)
    for p in range(zpairs):
        i, j = ind[p, 2], ind[p, 3]
        array_1D[p] = array_2D[i, j]
    return array_1D


###############################################################################


############### FISHER MATRIX ################################


# interpolator for FM
# XXXX todo
def interpolator(dC_interpolated_dict, dC_dict, obs_name, params_names, nbl, zpairs, ell_values, suffix):
    print('deprecated?')
    for param_name in params_names:  # loop for each parameter
        # pick array to interpolate
        dC_to_interpolate = dC_dict[f"dCij{obs_name}d{param_name}-{suffix}"]
        dC_interpolated = np.zeros((nbl, zpairs))  # initialize interpolated array

        # now interpolate
        original_ell_values = dC_to_interpolate[:, 0]  # first column is ell
        dC_to_interpolate = dC_to_interpolate[:, 1:]  # remove ell column
        for zpair_idx in range(zpairs):
            f = interp1d(original_ell_values, dC_to_interpolate[:, zpair_idx], kind='linear')
            dC_interpolated[:, zpair_idx] = f(ell_values)  # fill zpair_idx-th column
            dC_interpolated_dict[f"dCij{obs_name}d{param_name}-{suffix}"] = dC_interpolated  # store array in the dict

    return dC_interpolated_dict


# @njit
def fill_dC_array(params_names, dC_interpolated_dict, probe_code, dC, suffix):
    for (counter, param) in enumerate(params_names):
        dC[:, :, counter] = dC_interpolated_dict[f"dCij{probe_code}d{param}-{suffix}"]
    return dC


def fill_datavector_4D(nParams, nbl, npairs, zbins, ind, dC_4D):
    # XXX pairs_tot
    D_4D = np.zeros((nbl, zbins, zbins, nParams))

    for alf in range(nParams):
        for elle in range(nbl):
            for p in range(npairs):
                if ind[p, 0] == 0 and ind[p, 1] == 0:
                    D_4D[elle, ind[p, 2], ind[p, 3], alf] = dC_4D[elle, ind[p, 2], ind[p, 3], alf]
    return D_4D


@njit
def datavector_3D_to_2D(D_3D, nParams, nbl, npairs):
    D_2D = np.zeros((npairs * nbl, nParams))
    for alf in range(nParams):
        count = 0
        for elle in range(nbl):
            for p in range(npairs):
                D_2D[count, alf] = D_3D[elle, p, alf]
                count = count + 1
    return D_2D


# @njit
def compute_FM_3D(nbl, npairs, nParams, cov_inv, D_3D):
    """ Compute FM using 3D datavector - 2D + the cosmological parameter axis - and 3D covariance matrix (working but
    deprecated in favor of compute_FM_2D)"""
    b = np.zeros((nbl, npairs, nParams))
    FM = np.zeros((nParams, nParams))
    for alf in range(nParams):
        for bet in range(nParams):
            for elle in range(nbl):
                b[elle, :, bet] = cov_inv[elle, :, :] @ D_3D[elle, :, bet]
                FM[alf, bet] = FM[alf, bet] + (D_3D[elle, :, alf] @ b[elle, :, bet])
    return FM


# @njit
def compute_FM_2D(nbl, npairs, nparams_tot, cov_2D_inv, D_2D):
    """ Compute FM using 2D datavector - 1D + the cosmological parameter axis - and 2D covariance matrix"""
    b = np.zeros((nbl * npairs, nparams_tot))
    FM = np.zeros((nparams_tot, nparams_tot))
    for alf in range(nparams_tot):
        for bet in range(nparams_tot):
            b[:, bet] = cov_2D_inv[:, :] @ D_2D[:, bet]
            FM[alf, bet] = D_2D[:, alf] @ b[:, bet]
    return FM


def compute_FM_2D_optimized(nbl, npairs, nparams_tot, cov_2D_inv, D_2D):
    """ Compute FM using 2D datavector - 1D + the cosmological parameter axis - and 2D covariance matrix"""
    warnings.warn('deprecate this?')
    b = np.zeros((nbl * npairs, nparams_tot))
    FM = np.zeros((nparams_tot, nparams_tot))
    for alf in range(nparams_tot):
        for bet in range(nparams_tot):
            b[:, bet] = cov_2D_inv[:, :] @ D_2D[:, bet]
            FM[alf, bet] = D_2D[:, alf] @ b[:, bet]

    # do it with np.einsum in one line
    # FM = np.einsum('ij,ik,jk->ij', D_2D, b, cov_2D_inv)
    b = np.einsum('ij,jk->ik', cov_2D_inv, D_2D)
    FM = np.einsum('ij,jk->ik', D_2D, b)
    return FM


def compute_FoM(FM, w0wa_idxs):
    start = w0wa_idxs[0]
    stop = w0wa_idxs[1] + 1
    cov_param = np.linalg.inv(FM)
    cov_param_reduced = cov_param[start:stop, start:stop]
    FM_reduced = np.linalg.inv(cov_param_reduced)
    FoM = np.sqrt(np.linalg.det(FM_reduced))
    return FoM


def get_ind_file(path, ind_ordering, which_forecast):
    if ind_ordering == 'vincenzo' or which_forecast == 'sylvain':
        ind = np.genfromtxt(path.parent / "common_data/indici.dat").astype(int)
        ind = ind - 1
    elif ind_ordering == 'CLOE':
        ind = np.genfromtxt(path.parent / "common_data/indici_luca/indici_cloe_like.dat").astype(int)
        ind = ind - 1
    elif ind_ordering == 'SEYFERT':
        ind = np.genfromtxt(path.parent / "common_data/indici_luca/indici_seyfert_like.dat").astype(int)
        ind = ind - 1
    else:
        raise ValueError('ind_ordering must be vincenzo, sylvain, CLOE or SEYFERT')
    return ind


def get_output_folder(ind_ordering, which_forecast):
    if which_forecast == 'IST':
        if ind_ordering == 'vincenzo':
            output_folder = 'ISTspecs_indVincenzo'
        elif ind_ordering == 'CLOE':
            output_folder = 'ISTspecs_indCLOE'
        elif ind_ordering == 'SEYFERT':
            output_folder = 'ISTspecs_indSEYFERT'
    elif which_forecast == 'sylvain':
        output_folder = 'common_ell_and_deltas'
    return output_folder


def get_zpairs(zbins):
    zpairs_auto = int((zbins * (zbins + 1)) / 2)  # = 55 for zbins = 10, cast it as int
    zpairs_cross = zbins ** 2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


##############################################################################
#################### COVARIANCE MATRIX COMPUTATION ############################
###############################################################################
# TODO unify these 3 into a single function
# TODO workaround for start_index, stop_index (super easy)

@njit
def covariance(nbl, npairs, start_index, stop_index, Cij, noise, l_lin, delta_l, fsky, ind):
    # create covariance array
    covariance = np.zeros((nbl, nbl, npairs, npairs))
    # compute cov(ell, p, q)
    for ell in range(nbl):
        for p in range(start_index, stop_index):
            for q in range(start_index, stop_index):
                covariance[ell, ell, p - start_index, q - start_index] = \
                    ((Cij[ell, ind[p, 2], ind[q, 2]] + noise[ind[p, 0], ind[q, 0], ind[p, 2], ind[q, 2]]) *
                     (Cij[ell, ind[p, 3], ind[q, 3]] + noise[ind[p, 1], ind[q, 1], ind[p, 3], ind[q, 3]]) +
                     (Cij[ell, ind[p, 2], ind[q, 3]] + noise[ind[p, 0], ind[q, 1], ind[p, 2], ind[q, 3]]) *
                     (Cij[ell, ind[p, 3], ind[q, 2]] + noise[ind[p, 1], ind[q, 0], ind[p, 3], ind[q, 2]])) / \
                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return covariance


def covariance_einsum(cl_5d, noise_5d, f_sky, ell_values, delta_ell, return_only_diagonal_ells=False):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as noise_5d

    :param cl_5d:
    :param noise_5d:
    :param f_sky:
    :param ell_values:
    :param delta_ell:
    :param return_only_diagonal_ells:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.

    example code to compute auto probe data and spectra, if needed
    cl_LL_5D = cl_LL_3D[np.newaxis, np.newaxis, ...]
    noise_LL_5D = noise_3x2pt_5D[0, 0, ...][np.newaxis, np.newaxis, ...]
    cov_WL_6D = mm.covariance_einsum(cl_LL_5D, noise_LL_5D, fsky, ell_values, delta_ell)[0, 0, 0, 0, ...]
    """

    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert noise_5d.shape == cl_5d.shape, 'noise_5d must have shape the same shape as cl_5d, although there ' \
                                          'is no ell dependence'

    nbl = cl_5d.shape[2]

    prefactor = 1 / ((2 * ell_values + 1) * f_sky * delta_ell)

    # considering ells off-diagonal (wrong for Gauss: I am not implementing the delta)
    # term_1 = np.einsum('ACLik, BDMjl -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # term_2 = np.einsum('ADLil, BCMjk -> ABCDLMijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    # cov_10d = np.einsum('ABCDLMijkl, L -> ABCDLMijkl', term_1 + term_2, prefactor)

    # considering only ell diagonal
    term_1 = np.einsum('ACLik, BDLjl -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    term_2 = np.einsum('ADLil, BCLjk -> ABCDLijkl', cl_5d + noise_5d, cl_5d + noise_5d)
    cov_9d = np.einsum('ABCDLijkl, L -> ABCDLijkl', term_1 + term_2, prefactor)

    if return_only_diagonal_ells:
        warnings.warn('return_only_diagonal_ells is True, the array will be 9-dimensional, potentially causing '
                      'problems when reshaping or summing to cov_SSC arrays')
        return cov_9d

    n_probes = cov_9d.shape[0]
    zbins = cov_9d.shape[-1]
    cov_10d = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    cov_10d[:, :, :, :, np.arange(nbl), np.arange(nbl), ...] = cov_9d[:, :, :, :, np.arange(nbl), ...]

    return cov_10d


def expand_dims_sijkl(sijkl, zbins):
    n_probes = 2
    s_ABCD_ijkl = np.zeros((n_probes, n_probes, n_probes, n_probes, zbins, zbins, zbins, zbins))
    assert sijkl.shape == (2 * zbins, 2 * zbins, 2 * zbins, 2 * zbins), 'sijkl must have shape ' \
                                                                        '(2 * zbins, 2 * zbins, 2 * zbins, 2 * zbins)'
    s_ABCD_ijkl[0, 0, 0, 0, ...] = sijkl[:zbins, :zbins, :zbins, :zbins]
    s_ABCD_ijkl[0, 0, 0, 1, ...] = sijkl[:zbins, :zbins, :zbins, zbins:]
    s_ABCD_ijkl[0, 0, 1, 0, ...] = sijkl[:zbins, :zbins, zbins:, :zbins]
    s_ABCD_ijkl[0, 0, 1, 1, ...] = sijkl[:zbins, :zbins, zbins:, zbins:]
    s_ABCD_ijkl[0, 1, 0, 0, ...] = sijkl[:zbins, zbins:, :zbins, :zbins]
    s_ABCD_ijkl[0, 1, 0, 1, ...] = sijkl[:zbins, zbins:, :zbins, zbins:]
    s_ABCD_ijkl[0, 1, 1, 0, ...] = sijkl[:zbins, zbins:, zbins:, :zbins]
    s_ABCD_ijkl[0, 1, 1, 1, ...] = sijkl[:zbins, zbins:, zbins:, zbins:]
    s_ABCD_ijkl[1, 0, 0, 0, ...] = sijkl[zbins:, :zbins, :zbins, :zbins]
    s_ABCD_ijkl[1, 0, 0, 1, ...] = sijkl[zbins:, :zbins, :zbins, zbins:]
    s_ABCD_ijkl[1, 0, 1, 0, ...] = sijkl[zbins:, :zbins, zbins:, :zbins]
    s_ABCD_ijkl[1, 0, 1, 1, ...] = sijkl[zbins:, :zbins, zbins:, zbins:]
    s_ABCD_ijkl[1, 1, 0, 0, ...] = sijkl[zbins:, zbins:, :zbins, :zbins]
    s_ABCD_ijkl[1, 1, 0, 1, ...] = sijkl[zbins:, zbins:, :zbins, zbins:]
    s_ABCD_ijkl[1, 1, 1, 0, ...] = sijkl[zbins:, zbins:, zbins:, :zbins]
    s_ABCD_ijkl[1, 1, 1, 1, ...] = sijkl[zbins:, zbins:, zbins:, zbins:]

    return s_ABCD_ijkl


def expand_dims_sijkl_generalized(sijkl, zbins):
    n_probes = 3
    s_shape = (n_probes, n_probes, n_probes, n_probes, zbins, zbins, zbins, zbins)
    s_ABCD_ijkl = np.zeros(s_shape)

    assert sijkl.shape == (
        n_probes * zbins, n_probes * zbins, n_probes * zbins,
        n_probes * zbins), 'sijkl must have shape (2 * zbins, 2 * zbins, 2 * zbins, 2 * zbins)'

    for i in range(n_probes):
        for j in range(n_probes):
            for k in range(n_probes):
                for l in range(n_probes):
                    s_ABCD_ijkl[i, j, k, l, ...] = sijkl[i * zbins:(i + 1) * zbins, j * zbins:(j + 1) * zbins,
                                                         k * zbins:(k + 1) * zbins, l * zbins:(l + 1) * zbins]

    return s_ABCD_ijkl


def covariance_SSC_einsum(cl_5d, rl_5d, s_ABCD_ijkl, fsky, optimize='greedy'):
    """
    computes the 10-dimensional covariance matrix, of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins). The 5-th axis is added only if
    return_only_diagonal_ells is True. *for the single-probe case, n_probes = 1*

    In np.einsum, the indices have the following meaning:
        A, B, C, D = probe identifier. 0 for WL, 1 for GCph
        L, M = ell, ell_prime
        i, j, k, l = redshift bin indices

    cl_5d must have shape = (n_probes, n_probes, nbl, zbins, zbins) = (A, B, L, i, j), same as rl_5d

    :param cl_5d:
    :param rl_5d:
    :param noise_5d:
    :param fsky:
    :return: 10-dimensional numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, (nbl), zbins, zbins, zbins, zbins), containing the covariance.
    """

    assert cl_5d.shape[0] == 1 or cl_5d.shape[0] == 2, 'This funcion only works with 1 or two probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape[-1] == cl_5d.shape[-2], 'cl_5d must be an array of shape (n_probes, n_probes, nbl, zbins, zbins)'
    assert cl_5d.shape == rl_5d.shape, 'cl_5d and rl_5d must have the same shape'

    zbins = cl_5d.shape[-1]

    assert s_ABCD_ijkl.shape == (
        cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], zbins, zbins, zbins, zbins), \
        's_ABCD_ijkl must have shape (cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], zbins, zbins, zbins, zbins) = ' \
        f'{(cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], cl_5d.shape[0], zbins, zbins, zbins, zbins)}'

    cov_SSC_10d = np.einsum('ABLij, ABLij, CDMkl, CDMkl, ABCDijkl -> ABCDLMijkl', rl_5d, cl_5d, rl_5d, cl_5d,
                            s_ABCD_ijkl, optimize=optimize)

    return cov_SSC_10d / fsky


def cov_10D_dict_to_array(cov_10D_dict, nbl, zbins, n_probes=2):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""

    assert n_probes == 2, 'if more than 2 probes are used, the probe_dict must be changed ' \
                          '(promote it to an argument of the function!)'
    cov_10D_array = np.zeros((n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins))
    probe_dict = {'L': 0, 'G': 1}
    for A, B, C, D in cov_10D_dict.keys():
        cov_10D_array[probe_dict[A], probe_dict[B], probe_dict[C], probe_dict[D], ...] = cov_10D_dict[A, B, C, D]

    return cov_10D_array


def cov_10D_array_to_dict(cov_10D_array, probe_ordering):
    """ transforms a dictionary of "shape" [(A, B, C, D)][nbl, nbl, zbins, zbins, zbins, zbins] (where A, B, C, D is a
    tuple of strings, each one being either 'L' or 'G') to a numpy array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)"""

    cov_10D_dict = {}
    probe_dict = {'L': 0, 'G': 1}
    for A_str, B_str in probe_ordering:
        for C_str, D_str in probe_ordering:
            A_idx, B_idx, C_idx, D_idx = probe_dict[A_str], probe_dict[B_str], probe_dict[C_str], probe_dict[D_str]
            cov_10D_dict[A_str, B_str, C_str, D_str] = cov_10D_array[A_idx, B_idx, C_idx, D_idx, ...]

    return cov_10D_dict


# @njit
def covariance_WA(nbl, npairs, start_index, stop_index, Cij, noise, l_lin, delta_l, fsky, ind, ell_WA):
    covariance = np.zeros((nbl, nbl, npairs, npairs))

    for ell in range(nbl):
        for p in range(start_index, stop_index):
            for q in range(start_index, stop_index):

                if ell_WA.size == 1:  # in the case of just one bin it would give error
                    denominator = ((2 * l_lin + 1) * fsky * delta_l)
                else:
                    denominator = ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])

                covariance[ell, ell, p - start_index, q - start_index] = \
                    ((Cij[ell, ind[p, 2], ind[q, 2]] + noise[ind[p, 0], ind[q, 0], ind[p, 2], ind[q, 2]]) * (
                        Cij[ell, ind[p, 3], ind[q, 3]] + noise[ind[p, 1], ind[q, 1], ind[p, 3], ind[q, 3]]) + (
                        Cij[ell, ind[p, 2], ind[q, 3]] + noise[ind[p, 0], ind[q, 1], ind[p, 2], ind[q, 3]]) * (
                        Cij[ell, ind[p, 3], ind[q, 2]] + noise[ind[p, 1], ind[q, 0], ind[p, 3], ind[q, 2]])) \
                    / denominator
    return covariance


# covariance matrix for ALL
@njit
def covariance_ALL(nbl, npairs, Cij, noise, l_lin, delta_l, fsky, ind):
    # assert Cij.shape == (2, 2, nbl, zbins, zbins), "Cij has wrong shape"
    # create covariance array
    cov_GO = np.zeros((nbl, nbl, npairs, npairs))
    # compute cov(ell, p, q)
    for ell in range(nbl):
        for p in range(npairs):
            for q in range(npairs):
                # ind carries info about both the probes and the z indices!
                A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]
                i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                cov_GO[ell, ell, p, q] = \
                    ((Cij[A, C, ell, i, k] + noise[A, C, i, k]) * (Cij[B, D, ell, j, l] + noise[B, D, j, l]) +
                     (Cij[A, D, ell, i, l] + noise[A, D, i, l]) * (Cij[B, C, ell, j, k] + noise[B, C, j, k])) / \
                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return cov_GO


@njit
def cov_SSC(nbl, zpairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    if probe in ["WL", "WA"]:
        shift = 0
    elif probe == "GC":
        shift = zbins
    else:
        raise ValueError('probe must be "WL", "WA" or "GC"')

    cov_SSC = np.zeros((nbl, nbl, zpairs, zpairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(zpairs):
                for q in range(zpairs):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_SSC[ell1, ell2, p, q] = (Rl[ell1, i, j] * Rl[ell2, k, l] *
                                                 Cij[ell1, i, j] * Cij[ell2, k, l] *
                                                 Sijkl[i + shift, j + shift, k + shift, l + shift])
    cov_SSC /= fsky
    return cov_SSC


@njit
def build_Sijkl_dict(Sijkl, zbins):
    # build probe lookup dictionary, to set the right start and stop values
    probe_lookup = {
        'L': {
            'start': 0,
            'stop': zbins
        },
        'G': {
            'start': zbins,
            'stop': 2 * zbins
        }
    }

    # fill Sijkl dictionary
    Sijkl_dict = {}
    for probe_A in ['L', 'G']:
        for probe_B in ['L', 'G']:
            for probe_C in ['L', 'G']:
                for probe_D in ['L', 'G']:
                    Sijkl_dict[probe_A, probe_B, probe_C, probe_D] = \
                        Sijkl[probe_lookup[probe_A]['start']:probe_lookup[probe_A]['stop'],
                              probe_lookup[probe_B]['start']:probe_lookup[probe_B]['stop'],
                              probe_lookup[probe_C]['start']:probe_lookup[probe_C]['stop'],
                              probe_lookup[probe_D]['start']:probe_lookup[probe_D]['stop']]

    return Sijkl_dict


@njit
def build_3x2pt_dict(array_3x2pt):
    dict_3x2pt = {}
    if array_3x2pt.ndim == 5:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :, :]
    elif array_3x2pt.ndim == 4:
        dict_3x2pt['L', 'L'] = array_3x2pt[0, 0, :, :]
        dict_3x2pt['L', 'G'] = array_3x2pt[0, 1, :, :]
        dict_3x2pt['G', 'L'] = array_3x2pt[1, 0, :, :]
        dict_3x2pt['G', 'G'] = array_3x2pt[1, 1, :, :]
    return dict_3x2pt


def build_3x2pt_array(cl_LL_3D, cl_GG_3D, cl_GL_3D, n_probes, nbl, zbins):
    warnings.warn("shape is (n_probes, n_probes, nbl, zbins, zbins), NOT (nbl, n_probes, n_probes, zbins, zbins)")
    assert cl_LL_3D.shape == (nbl, zbins, zbins), f'cl_LL_3D.shape = {cl_LL_3D.shape}, should be (nbl, zbins, zbins)'
    assert cl_GL_3D.shape == (nbl, zbins, zbins), f'cl_GL_3D.shape = {cl_GL_3D.shape}, should be (nbl, zbins, zbins)'
    assert cl_GG_3D.shape == (nbl, zbins, zbins), f'cl_GG_3D.shape = {cl_GG_3D.shape}, should be (nbl, zbins, zbins)'
    cl_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
    cl_3x2pt_5D[0, 0, :, :, :] = cl_LL_3D
    cl_3x2pt_5D[0, 1, :, :, :] = cl_GL_3D.transpose(0, 2, 1)
    cl_3x2pt_5D[1, 0, :, :, :] = cl_GL_3D
    cl_3x2pt_5D[1, 1, :, :, :] = cl_GG_3D
    return cl_3x2pt_5D


@njit
def cov_SSC_ALL(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """The fastest routine to compute the SSC covariance matrix.
    """

    cov_ALL_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]
                    A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]

                    # the shift is implemented by multiplying A, B, C, D by zbins: if lensing, probe == 0 and shift = 0
                    # if probe is GC, probe == 1 and shift = zbins. this does not hold if you switch probe indices!
                    cov_ALL_SSC[ell1, ell2, p, q] = (Rl[A, B, ell1, i, j] *
                                                     Rl[C, D, ell2, k, l] *
                                                     D_3x2pt[A, B, ell1, i, j] *
                                                     D_3x2pt[C, D, ell2, k, l] *
                                                     Sijkl[i + A * zbins, j + B * zbins, k + C * zbins, l + D * zbins])

    cov_ALL_SSC /= fsky
    return cov_ALL_SSC


# ! to be deprecated
@njit
def cov_SSC_ALL_dict(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """Buil the 3x2pt covariance matrix using a dict for Sijkl. slightly slower (because of the use of dicts, I think)
    but cleaner (no need for multiple if statements, except to set the correct probes).
    Note that the ell1, ell2 slicing does not work! You can substitute only one of the for loops (in this case the one over ell1).
    A_str = probe A as string (e.g. 'L' for lensing)
    A_num = probe A as number (e.g. 0 for lensing)
    """

    Sijkl_dict = build_Sijkl_dict(Sijkl, zbins)
    print('xxxxx x xx  x x x x  x x x x  x x xvariable response not implemented!')

    cov_3x2pt_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):

                    # TODO do this with a dictionary!!
                    if ind[p, 0] == 0:
                        A_str = 'L'
                    elif ind[p, 0] == 1:
                        A_str = 'G'
                    if ind[p, 1] == 0:
                        B_str = 'L'
                    elif ind[p, 1] == 1:
                        B_str = 'G'
                    if ind[q, 0] == 0:
                        C_str = 'L'
                    elif ind[q, 0] == 1:
                        C_str = 'G'
                    if ind[q, 1] == 0:
                        D_str = 'L'
                    elif ind[q, 1] == 1:
                        D_str = 'G'

                    A_num, B_num, C_num, D_num = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_3x2pt_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                       D_3x2pt[A_num, B_num, ell1, i, j] *
                                                       D_3x2pt[C_num, D_num, ell2, k, l] *
                                                       Sijkl_dict[A_str, B_str, C_str, D_str][i, j, k, l])
    return cov_3x2pt_SSC / fsky


def cov_G_10D_dict(cl_dict, noise_dict, nbl, zbins, l_lin, delta_l, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically. 
    This one works with dictionaries, in particular for the cls and noise arrays. 
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.

    This version is faster, it is a wrapper function for covariance_6D_blocks,
    which makes use of jit
    """

    cov_10D_dict = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_10D_dict[A, B, C, D] = cov_GO_6D_blocks(
                cl_dict[A, C], cl_dict[B, D], cl_dict[A, D], cl_dict[B, C],
                noise_dict[A, C], noise_dict[B, D], noise_dict[A, D], noise_dict[B, C],
                nbl, zbins, l_lin, delta_l, fsky)
    return cov_10D_dict


def cov_SS_10D_dict(Cl_dict, Rl_dict, Sijkl_dict, nbl, zbins, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically.
    This one works with dictionaries, in particular for the cls and noise arrays.
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.

    This version is faster, it is a wrapper function for covariance_6D_blocks,
    which makes use of jit
    """

    cov_SS_10D_dict = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_SS_10D_dict[A, B, C, D] = cov_SS_6D_blocks(Rl_dict[A, B], Cl_dict[A, B], Rl_dict[C, D], Cl_dict[C, D],
                                                           Sijkl_dict[A, B, C, D], nbl, zbins, fsky)

    return cov_SS_10D_dict


# This function does mix the indices, but not automatically: it only indicates which ones to use and where
# It can be used for the individual blocks of the 3x2pt (unlike the one above),
# but it has to be called once for each block combination (see cov_blocks_LG_4D
# and cov_blocks_GL_4D)
# best used in combination with cov_10D_dictionary
@njit
def cov_GO_6D_blocks(C_AC, C_BD, C_AD, C_BC, N_AC, N_BD, N_AD, N_BC, nbl, zbins, l_lin, delta_l, fsky):
    cov_GO_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                for k in range(zbins):
                    for l in range(zbins):
                        cov_GO_6D[ell, ell, i, j, k, l] = \
                            ((C_AC[ell, i, k] + N_AC[i, k]) *
                             (C_BD[ell, j, l] + N_BD[j, l]) +
                             (C_AD[ell, i, l] + N_AD[i, l]) *
                             (C_BC[ell, j, k] + N_BC[j, k])) / \
                            ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return cov_GO_6D


@njit
def cov_SS_6D_blocks(Rl_AB, Cl_AB, Rl_CD, Cl_CD, Sijkl_ABCD, nbl, zbins, fsky):
    """ experimental"""
    cov_SS_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for i in range(zbins):
                for j in range(zbins):
                    for k in range(zbins):
                        for l in range(zbins):
                            cov_SS_6D[ell1, ell2, i, j, k, l] = \
                                (Rl_AB[ell1, i, j] *
                                 Cl_AB[ell1, i, j] *
                                 Rl_CD[ell2, k, l] *
                                 Cl_CD[ell2, k, l] *
                                 Sijkl_ABCD[i, j, k, l])
    cov_SS_6D /= fsky
    return cov_SS_6D


def cov_3x2pt_10D_to_4D(cov_3x2pt_10D, probe_ordering, nbl, zbins, ind_copy, GL_or_LG):
    """
    Takes the cov_3x2pt_10D dictionary, reshapes each A, B, C, D block separately
    in 4D, then stacks the blocks in the right order to output cov_3x2pt_4D
    (which is not a dictionary but a numpy array)

    probe_ordering: e.g. ['L', 'L'], ['G', 'L'], ['G', 'G']]
    """

    # if it's an array, convert to dictionary for the function to work
    if type(cov_3x2pt_10D) == np.ndarray:
        cov_3x2pt_dict_10D = cov_10D_array_to_dict(cov_3x2pt_10D, probe_ordering)
    elif type(cov_3x2pt_10D) == dict:
        cov_3x2pt_dict_10D = cov_3x2pt_10D
    else:
        raise ValueError('cov_3x2pt_10D must be either a dictionary or an array')

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    # Check that the cross-correlation is coherent with the probe_ordering list
    # this is a weak check, since I'm assuming that GL or LG will be the second
    # element of the datavector
    if GL_or_LG == 'GL':
        assert probe_ordering[1][0] == 'G' and probe_ordering[1][1] == 'L', \
            'probe_ordering[1] should be "GL", e.g. [LL, GL, GG]'
    elif GL_or_LG == 'LG':
        assert probe_ordering[1][0] == 'L' and probe_ordering[1][1] == 'G', \
            'probe_ordering[1] should be "LG", e.g. [LL, LG, GG]'

    # get npairs
    npairs_auto, npairs_cross, npairs_3x2pt = get_zpairs(zbins)

    # construct the ind dict
    ind_dict = {}
    ind_dict['L', 'L'] = ind_copy[:npairs_auto, :]
    ind_dict['G', 'G'] = ind_copy[(npairs_auto + npairs_cross):, :]
    if GL_or_LG == 'LG':
        ind_dict['L', 'G'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_or_LG == 'GL':
        ind_dict['G', 'L'] = ind_copy[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    # construct the npairs dict
    npairs_dict = {}
    npairs_dict['L', 'L'] = npairs_auto
    npairs_dict['L', 'G'] = npairs_cross
    npairs_dict['G', 'L'] = npairs_cross
    npairs_dict['G', 'G'] = npairs_auto

    # initialize the 4D dictionary and list of probe combinations
    cov_3x2pt_dict_4D = {}
    combinations = []

    # make each block 4D and store it with the right 'A', 'B', 'C, 'D' key
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            combinations.append([A, B, C, D])
            cov_3x2pt_dict_4D[A, B, C, D] = cov_6D_to_4D_blocks(cov_3x2pt_dict_10D[A, B, C, D], nbl, npairs_dict[A, B],
                                                                npairs_dict[C, D], ind_dict[A, B], ind_dict[C, D])

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = cov_3x2pt_8D_dict_to_4D(cov_3x2pt_dict_4D, probe_ordering, combinations)

    return cov_3x2pt_4D


def cov_3x2pt_8D_dict_to_4D(cov_3x2pt_8D_dict, probe_ordering, combinations=None):
    """
    Convert a dictionary of 4D blocks into a single 4D array. This is the same code as in the last part of the function above.
    :param cov_3x2pt_8D_dict: Dictionary of 4D covariance blocks
    :param probe_ordering: tuple of tuple probes, e.g., (('L', 'L'), ('G', 'L'), ('G', 'G'))
    :combinations: list of combinations to use, e.g., [['L', 'L', 'L', 'L'], ['L', 'L', 'L', 'G'], ...]
    :return: 4D covariance array
    """

    # if combinations is not provided, construct it
    if combinations is None:
        combinations = []
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                combinations.append([A, B, C, D])

    for key in cov_3x2pt_8D_dict.keys():
        assert cov_3x2pt_8D_dict[key].ndim == 4, (
            f'covariance matrix {key} has ndim={cov_3x2pt_8D_dict[key].ndim} instead '
            f'of 4')

    # check that the number of combinations is correct
    assert len(combinations) == len(list(cov_3x2pt_8D_dict.keys())), \
        f'number of combinations ({len(combinations)}) does not match the number of blocks in the input dictionary ' \
        f'({len(list(cov_3x2pt_8D_dict.keys()))})'

    # check that the combinations are correct
    for i, combination in enumerate(combinations):
        assert tuple(combination) in list(
            cov_3x2pt_8D_dict.keys()), f'combination {combination} not found in the input dictionary'

    # take the correct combinations (stored in 'combinations') and construct
    # lists which will be converted to arrays
    row_1_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[:3]]
    row_2_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[3:6]]
    row_3_list = [cov_3x2pt_8D_dict[A, B, C, D] for A, B, C, D in combinations[6:9]]

    # concatenate the lists to make rows
    row_1 = np.concatenate(row_1_list, axis=3)
    row_2 = np.concatenate(row_2_list, axis=3)
    row_3 = np.concatenate(row_3_list, axis=3)

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_3x2pt_4D


def cov_3x2pt_4d_to_10d_dict(cov_3x2pt_4d, zbins, probe_ordering, nbl, ind_copy):

    zpairs_auto, zpairs_cross, _ = get_zpairs(zbins)

    ind_copy = ind_copy.copy()  # just to ensure the input ind file is not changed

    ind_auto = ind_copy[:zpairs_auto, :]
    ind_cross = ind_copy[zpairs_auto:zpairs_cross + zpairs_auto, :]
    ind_dict = {('L', 'L'): ind_auto,
                ('G', 'L'): ind_cross,
                ('G', 'G'): ind_auto}

    assert probe_ordering == (('L', 'L'), ('G', 'L'), ('G', 'G')), 'more elaborate probe_ordering not implemented yet'

    # slice the 4d cov to be able to use cov_4D_to_6D_blocks on the nine separate blocks
    zpairs_sum = zpairs_auto + zpairs_cross
    cov_vinc_no_bnt_8d_dict = {}
    cov_vinc_no_bnt_8d_dict['L', 'L', 'L', 'L'] = cov_3x2pt_4d[:, :, :zpairs_auto, :zpairs_auto]
    cov_vinc_no_bnt_8d_dict['L', 'L', 'G', 'L'] = cov_3x2pt_4d[:, :, :zpairs_auto, zpairs_auto:zpairs_sum]
    cov_vinc_no_bnt_8d_dict['L', 'L', 'G', 'G'] = cov_3x2pt_4d[:, :, :zpairs_auto, zpairs_sum:]

    cov_vinc_no_bnt_8d_dict['G', 'L', 'L', 'L'] = cov_3x2pt_4d[:, :, zpairs_auto:zpairs_sum, :zpairs_auto]
    cov_vinc_no_bnt_8d_dict['G', 'L', 'G', 'L'] = cov_3x2pt_4d[:, :, zpairs_auto:zpairs_sum, zpairs_auto:zpairs_sum]
    cov_vinc_no_bnt_8d_dict['G', 'L', 'G', 'G'] = cov_3x2pt_4d[:, :, zpairs_auto:zpairs_sum, zpairs_sum:]

    cov_vinc_no_bnt_8d_dict['G', 'G', 'L', 'L'] = cov_3x2pt_4d[:, :, zpairs_sum:, :zpairs_auto]
    cov_vinc_no_bnt_8d_dict['G', 'G', 'G', 'L'] = cov_3x2pt_4d[:, :, zpairs_sum:, zpairs_auto:zpairs_sum]
    cov_vinc_no_bnt_8d_dict['G', 'G', 'G', 'G'] = cov_3x2pt_4d[:, :, zpairs_sum:, zpairs_sum:]

    cov_vinc_no_bnt_10d_dict = {}
    for key in cov_vinc_no_bnt_8d_dict.keys():
        cov_vinc_no_bnt_10d_dict[key] = cov_4D_to_6D_blocks(
            cov_vinc_no_bnt_8d_dict[key], nbl, zbins, ind_dict[key[0], key[1]], ind_dict[key[2], key[3]])

    return cov_vinc_no_bnt_10d_dict


# ! to be deprecated
@njit
def symmetrize_ij(cov_6D, zbins):
    warnings.warn('THIS FUNCTION ONLY WORKS IF THE MATRIX TO SYMMETRIZE IS UPPER *OR* LOWER TRIANGULAR, NOT BOTH')
    # TODO thorough check?
    for i in range(zbins):
        for j in range(zbins):
            cov_6D[:, :, i, j, :, :] = cov_6D[:, :, j, i, :, :]
            cov_6D[:, :, :, :, i, j] = cov_6D[:, :, :, :, j, i]
    return cov_6D


# @njit
# ! this function is new - still to be thouroughly tested
def cov_4D_to_6D(cov_4D, nbl, zbins, probe, ind):
    """transform the cov from shape (nbl, nbl, npairs, npairs) 
    to (nbl, nbl, zbins, zbins, zbins, zbins). Not valid for 3x2pt, the total
    shape of the matrix is (nbl, nbl, zbins, zbins, zbins, zbins), not big 
    enough to store 3 probes. Use cov_4D functions or cov_6D as a dictionary
    instead,
    """

    npairs_auto, npairs_cross, npairs_tot = get_zpairs(zbins)
    if probe in ['LL', 'GG']:
        npairs = npairs_auto
    elif probe in ['GL', 'LG']:
        npairs = npairs_cross
    else:
        raise ValueError('probe must be "LL", "LG", "GL" or "GG". 3x2pt is not supported')

    assert ind.shape[0] == npairs, 'ind.shape[0] != npairs: maybe you\'re passing the whole ind file ' \
                                   'instead of ind[:npairs, :] - or similia'

    # TODO use jit
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ij in range(npairs):
        for kl in range(npairs):
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3]
            cov_6D[:, :, i, j, k, l] = cov_4D[:, :, ij, kl]

    # GL is not symmetric
    # ! this part makes this function very slow
    if probe in ['LL', 'GG']:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(cov_6D[ell1, ell2, :, :, i, j])
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(cov_6D[ell1, ell2, i, j, :, :])

    return cov_6D


@njit
def cov_6D_to_4D(cov_6D, nbl, zpairs, ind):
    """transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins)
    to (nbl, nbl, zpairs, zpairs)"""
    assert ind.shape[0] == zpairs, "ind.shape[0] != zpairs: maybe you're passing the whole ind file " \
                                   "instead of ind[:zpairs, :] - or similia"
    cov_4D = np.zeros((nbl, nbl, zpairs, zpairs))

    for ell2 in range(nbl):  # added this loop in the latest version, before it was vectorized
        for ij in range(zpairs):
            for kl in range(zpairs):
                # rename for better readability
                i, j, k, l = ind[ij, -2], ind[ij, -1], ind[kl, -2], ind[kl, -1]
                cov_4D[:, ell2, ij, kl] = cov_6D[:, ell2, i, j, k, l]

    return cov_4D


# TODO finish this
def cov_6D_to_4D_optim(cov_6D, nbl, zpairs, ind):
    """transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins)
    to (nbl, nbl, zpairs, zpairs)"""
    assert ind.shape[0] == zpairs, "ind.shape[0] != zpairs: maybe you're passing the whole ind file " \
                                   "instead of ind[:zpairs, :] - or similia"

    i_indices = ind[:, -2]
    j_indices = ind[:, -1]
    k_indices = ind[:, -2]
    l_indices = ind[:, -1]

    cov_4D = cov_6D[:, :, i_indices, j_indices, k_indices, l_indices]
    cov_4D = cov_4D.reshape((nbl, nbl, zpairs, zpairs))

    return cov_4D


@njit
def cov_6D_to_4D_blocks(cov_6D, nbl, npairs_AB, npairs_CD, ind_AB, ind_CD):
    """ reshapes the covariance even for the non-diagonal (hence, non-square) blocks needed to build the 3x2pt.
    use npairs_AB = npairs_CD and ind_AB = ind_CD for the normal routine (valid for auto-covariance 
    LL-LL, GG-GG, GL-GL and LG-LG). n_columns is used to determine whether the ind array has 2 or 4 columns
    (if it's given in the form of a dictionary or not)
    """
    assert ind_AB.shape[0] == npairs_AB, 'ind_AB.shape[0] != npairs_AB'
    assert ind_CD.shape[0] == npairs_CD, 'ind_CD.shape[0] != npairs_CD'

    # this is to ensure compatibility with both 4-columns and 2-columns ind arrays (dictionary)
    # the penultimante element is the first index, the last one the second index (see s - 1, s - 2 below)
    n_columns_AB = ind_AB.shape[1]  # of columns: this is to understand the format of the file
    n_columns_CD = ind_CD.shape[1]

    # check
    assert n_columns_AB == n_columns_CD, 'ind_AB and ind_CD must have the same number of columns'
    nc = n_columns_AB  # make the name shorter

    cov_4D = np.zeros((nbl, nbl, npairs_AB, npairs_CD))
    for ell2 in range(nbl):  # ! this is an untested addition; but did it work for ssc too without this additional loop?
        for ij in range(npairs_AB):
            for kl in range(npairs_CD):
                i, j, k, l = ind_AB[ij, nc - 2], ind_AB[ij, nc - 1], ind_CD[kl, nc - 2], ind_CD[kl, nc - 1]
                cov_4D[:, ell2, ij, kl] = cov_6D[:, ell2, i, j, k, l]
    return cov_4D


# @njit
def cov_4D_to_6D_blocks(cov_4D, nbl, zbins, ind_ab, ind_cd):
    """ reshapes the covariance even for the non-diagonal (hence, non-square) blocks needed to build the 3x2pt.
    use zpairs_ab = zpairs_cd and ind_ab = ind_cd for the normal routine (valid for auto-covariance
    LL-LL, GG-GG, GL-GL and LG-LG).
    """
    warnings.warn(
        'This function does not symmetrize the output covariance block, but it works if you want to re-reduce '
        'to the 4d or 2d covariance')

    assert ind_ab.shape[1] == ind_cd.shape[1], 'ind_ab and ind_cd must have the same number of columns'
    assert ind_ab.shape[1] == 2 or ind_ab.shape[1] == 4, 'ind_ab and ind_cd must have 2 or 4 columns'
    ncols = ind_ab.shape[1]

    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell2 in range(nbl):
        for ij in range(zpairs_ab):
            for kl in range(zpairs_cd):
                i, j, k, l = ind_ab[ij, ncols - 2], ind_ab[ij, ncols - 1], ind_cd[kl, ncols - 2], ind_cd[kl, ncols - 1]
                cov_6D[:, ell2, i, j, k, l] = cov_4D[:, ell2, ij, kl]

    return cov_6D


def return_combinations(A, B, C, D):
    print(f'C_{A}{C}, C_{B}{D}, C_{A}{D}, C_{B}{C}, N_{A}{C}, N_{B}{D}, N_{A}{D}, N_{B}{C}')


###########################
# @njit
def check_symmetric(array_2d, exact, rtol=1e-05):
    """
    :param a: 2d array
    :param exact: bool
    :param rtol: relative tolerance
    :return: bool, whether the array is symmetric or not
    """
    # """check if the matrix is symmetric, either exactly or within a tolerance
    # """
    assert type(exact) == bool, 'parameter "exact" must be either True or False'
    assert array_2d.ndim == 2, 'the array is not square'
    if exact:
        return np.array_equal(array_2d, array_2d.T)
    else:
        return np.allclose(array_2d, array_2d.T, rtol=rtol, atol=0)


@njit
# reshape from 3 to 4 dimensions
def array_3D_to_4D(cov_3D, nbl, npairs):
    print('XXX THIS FUNCTION ONLY WORKS FOR GAUSS-ONLY COVARIANCE')
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ell in range(nbl):
        for p in range(npairs):
            for q in range(npairs):
                cov_4D[ell, ell, p, q] = cov_3D[ell, p, q]
    return cov_4D


def slice_cov_3x2pt_2D_ell_probe_zpair(cov_2D_ell_probe_zpair, nbl, zbins, probe):
    """ Slices the 2-dimensional 3x2pt covariance ordered as a block-diagonal matrix in ell, probe and zpair
    (unpacked in this order)"""

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)
    ell_block_size = zpairs_3x2pt

    if probe == 'WL':
        probe_start = 0
        probe_stop = zpairs_auto
    elif probe == 'GC':
        probe_start = zpairs_auto + zpairs_cross
        probe_stop = zpairs_3x2pt
    elif probe == '2x2pt':
        probe_start = zpairs_auto
        probe_stop = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, GC or 2x2pt')

    cov_1D_ell_probe_zpair = [0] * nbl
    for ell_bin in range(nbl):
        # block_index * block_size + probe_starting index in each block
        start = ell_bin * ell_block_size + probe_start
        stop = start + probe_stop
        cov_1D_ell_probe_zpair[ell_bin] = cov_2D_ell_probe_zpair[start:stop, start:stop]

    cov_2D_ell_probe_zpair_sliced = scipy.linalg.block_diag(*cov_1D_ell_probe_zpair)

    return cov_2D_ell_probe_zpair_sliced


def slice_cl_3x2pt_1D_ell_probe_zpair(cl_3x2pt_1D_ell_probe_zpair, nbl, zbins, probe):
    """ Slices the 2-dimensional 3x2pt covariance ordered as a block-diagonal matrix in ell, probe and zpair
    (unpacked in this order)"""

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)
    ell_block_size = zpairs_3x2pt

    if probe == 'WL':
        probe_start = 0
        probe_stop = zpairs_auto
    elif probe == 'GC':
        probe_start = zpairs_auto + zpairs_cross
        probe_stop = zpairs_3x2pt
    elif probe == '2x2pt':
        probe_start = zpairs_auto
        probe_stop = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, GC or 2x2pt')

    cl_1D_ell_probe_zpair_list = [0] * nbl
    for ell_bin in range(nbl):
        # block_index * block_size + probe_starting index in each block
        start = ell_bin * ell_block_size + probe_start
        stop = start + probe_stop
        cl_1D_ell_probe_zpair_list[ell_bin] = cl_3x2pt_1D_ell_probe_zpair[start:stop]

    cl_1D_ell_probe_zpair = np.array(list(itertools.chain(*cl_1D_ell_probe_zpair_list)))

    return cl_1D_ell_probe_zpair


# @njit
def cov_2D_to_4D(cov_2D, nbl, block_index='vincenzo', optimize=True):
    """ new (more elegant) version of cov_2D_to_4D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops)
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'zpair_wise', me and Vincenzo block_index == 'ell':
    I add this distinction in the "if" to make it clearer.

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'
    assert cov_2D.ndim == 2, 'the input covariance must be 2-dimensional'

    zpairs_AB = cov_2D.shape[0] // nbl
    zpairs_CD = cov_2D.shape[1] // nbl

    cov_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))

    if optimize:
        if block_index in ['ell', 'vincenzo', 'C-style']:
            cov_4D = cov_2D.reshape((nbl, zpairs_AB, nbl, zpairs_CD)).transpose((0, 2, 1, 3))
        elif block_index in ['ij', 'sylvain', 'F-style']:
            cov_4D = cov_2D.reshape((zpairs_AB, nbl, nbl, zpairs_CD)).transpose((1, 2, 0, 3))
        return cov_4D

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]
    return cov_4D


@njit
def cov_4D_to_2D(cov_4D, block_index='vincenzo', optimize=True):
    """ new (more elegant) version of cov_4D_to_2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], 'the first two axes of the input covariance must have the same size'
    # assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    nbl = int(cov_4D.shape[0])
    zpairs_AB = int(cov_4D.shape[2])
    zpairs_CD = int(cov_4D.shape[3])

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if optimize:
        if block_index in ['ell', 'vincenzo', 'C-style']:
            cov_2D.reshape(nbl, zpairs_AB, nbl, zpairs_CD)[:, :, :, :] = cov_4D.transpose(0, 2, 1, 3)

        elif block_index in ['ij', 'sylvain', 'F-style']:
            cov_2D.reshape(zpairs_AB, nbl, zpairs_CD, nbl)[:, :, :, :] = cov_4D.transpose(2, 0, 3, 1)
        return cov_2D

    # I tested that the 2 methods give the same results. This code is kept to remember the
    # block_index * block_size + running_index unpacking
    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]

    return cov_2D


@njit
def cov_4D_to_2D_v0(cov_4D, nbl, zpairs_AB, zpairs_CD=None, block_index='vincenzo'):
    """ new (more elegant) version of cov_4D_to_2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], 'the first two axes of the input covariance must have the same size'
    assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    # if not passed, zpairs_CD must be equal to zpairs_AB
    if zpairs_CD is None:
        zpairs_CD = zpairs_AB

    if zpairs_AB != zpairs_CD:
        print('warning: zpairs_AB != zpairs_CD, the output covariance will be non-square')

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]
    return cov_2D


# @njit
def cov_4D_to_2DCLOE_3x2pt(cov_4D, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    ! important note: block_index = 'vincenzo' means that the overall ordering will be probe_ell_zpair. Setting it to 'zpair'
    ! will give you the ordering probe_zpair_ell. Bottom line: the probe is the outermost loop in any case.
    ! The ordering used by CLOE is probe_ell_zpair, so block_index = 'vincenzo' is the correct choice.
    """

    warnings.warn(
        "the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
        " will work both for LG and GL) ")
    warnings.warn('did you remove the nbl argument?')

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    lim_1 = zpairs_auto
    lim_2 = zpairs_cross + zpairs_auto
    lim_3 = zpairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], block_index)
    cov_LL_LG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_1:lim_2], block_index)
    cov_LL_GG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_2:lim_3], block_index)

    cov_LG_LL = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, :lim_1], block_index)
    cov_LG_LG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], block_index)
    cov_LG_GG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], block_index)

    cov_GG_LL = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, :lim_1], block_index)
    cov_GG_LG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], block_index)
    cov_GG_GG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], block_index)

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


# @njit
def cov_2DCLOE_to_4D_3x2pt(cov_2D, nbl, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    warnings.warn(
        "the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
        " will work both for LG and GL) ")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    # now I'm reshaping the full block diagonal matrix, not just the sub-blocks (cov_2D_to_4D works for both cases)
    lim_1 = zpairs_auto * nbl
    lim_2 = (zpairs_cross + zpairs_auto) * nbl
    lim_3 = zpairs_3x2pt * nbl

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_2D_to_4D(cov_2D[:lim_1, :lim_1], nbl, block_index)
    cov_LL_LG = cov_2D_to_4D(cov_2D[:lim_1, lim_1:lim_2], nbl, block_index)
    cov_LL_GG = cov_2D_to_4D(cov_2D[:lim_1, lim_2:lim_3], nbl, block_index)

    cov_LG_LL = cov_2D_to_4D(cov_2D[lim_1:lim_2, :lim_1], nbl, block_index)
    cov_LG_LG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_1:lim_2], nbl, block_index)
    cov_LG_GG = cov_2D_to_4D(cov_2D[lim_1:lim_2, lim_2:lim_3], nbl, block_index)

    cov_GG_LL = cov_2D_to_4D(cov_2D[lim_2:lim_3, :lim_1], nbl, block_index)
    cov_GG_LG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_1:lim_2], nbl, block_index)
    cov_GG_GG = cov_2D_to_4D(cov_2D[lim_2:lim_3, lim_2:lim_3], nbl, block_index)

    # here it is a little more difficult to visualize the stacking, but the probes are concatenated
    # along the 2 zpair_3x2pt-long axes
    cov_4D = np.zeros((nbl, nbl, zpairs_3x2pt, zpairs_3x2pt))

    zlim_1 = zpairs_auto
    zlim_2 = zpairs_cross + zpairs_auto
    zlim_3 = zpairs_3x2pt

    cov_4D[:, :, :zlim_1, :zlim_1] = cov_LL_LL
    cov_4D[:, :, :zlim_1, zlim_1:zlim_2] = cov_LL_LG
    cov_4D[:, :, :zlim_1, zlim_2:zlim_3] = cov_LL_GG

    cov_4D[:, :, zlim_1:zlim_2, :zlim_1] = cov_LG_LL
    cov_4D[:, :, zlim_1:zlim_2, zlim_1:zlim_2] = cov_LG_LG
    cov_4D[:, :, zlim_1:zlim_2, zlim_2:zlim_3] = cov_LG_GG

    cov_4D[:, :, zlim_2:zlim_3, :zlim_1] = cov_GG_LL
    cov_4D[:, :, zlim_2:zlim_3, zlim_1:zlim_2] = cov_GG_LG
    cov_4D[:, :, zlim_2:zlim_3, zlim_2:zlim_3] = cov_GG_GG

    return cov_4D


def cov_2d_dav_to_cloe(cov_2d_dav, nbl, zbins, block_index_in, block_index_out):
    """convert a 2D covariance matrix from the davide convention to the CLOE convention, that is, from the probe being
    unraveled in the first for loop to the probe being unraveled in the second for loop.
    example: from ell_probe_zpair (my convention) to probe_ell_zpair (CLOE).
    The zpairs <-> ell ordering is decided by 'block_idex' (setting the first, or outermost, of the two)"""
    cov_4D = cov_2D_to_4D(cov_2d_dav, nbl, block_index=block_index_in, optimize=True)
    cov_2d_cloe = cov_4D_to_2DCLOE_3x2pt(cov_4D, zbins=zbins, block_index=block_index_out)
    return cov_2d_cloe


def cov_2d_cloe_to_dav(cov_2d_cloe, nbl, zbins, block_index_in, block_index_out):
    """convert a 2D covariance matrix from the CLOE convention to the davide convention, that is, from the probe being
    unraveled in the second for loop to the probe being unraveled in the first for loop.
    example: from probe_ell_zpair (CLOE) to ell_probe_zpair (my convention).
    The zpairs <-> ell ordering is decided by 'block_idex' (setting the first, or outermost, of the two)"""
    cov_4D = cov_2DCLOE_to_4D_3x2pt(cov_2d_cloe, nbl, zbins, block_index=block_index_in)
    cov_2d_dav = cov_4D_to_2D(cov_4D, block_index=block_index_out, optimize=True)
    return cov_2d_dav


def cov_4D_to_2DCLOE_3x2pt_bu(cov_4D, nbl, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    warnings.warn(
        "the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
        " will work both for LG and GL) ")

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    lim_1 = zpairs_auto
    lim_2 = zpairs_cross + zpairs_auto
    lim_3 = zpairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], nbl, zpairs_auto, zpairs_auto, block_index)
    cov_LL_LG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_1:lim_2], nbl, zpairs_auto, zpairs_cross, block_index)
    cov_LL_GG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_2:lim_3], nbl, zpairs_auto, zpairs_auto, block_index)

    cov_LG_LL = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, :lim_1], nbl, zpairs_cross, zpairs_auto, block_index)
    cov_LG_LG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], nbl, zpairs_cross, zpairs_cross, block_index)
    cov_LG_GG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], nbl, zpairs_cross, zpairs_auto, block_index)

    cov_GG_LL = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, :lim_1], nbl, zpairs_auto, zpairs_auto, block_index)
    cov_GG_LG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], nbl, zpairs_auto, zpairs_cross, block_index)
    cov_GG_GG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], nbl, zpairs_auto, zpairs_auto, block_index)

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


def correlation_from_covariance(covariance):
    """ not thoroughly tested. Taken from 
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    does NOT work with 3x2pt
    """
    if covariance.shape[0] > 2000:
        print("this function doesn't work for 3x2pt")

    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# compute Sylvain's deltas
def delta_l_Sylvain(nbl, ell):
    delta_l = np.zeros(nbl)
    for l in range(1, nbl):
        delta_l[l] = ell[l] - ell[l - 1]
    delta_l[0] = delta_l[1]
    return delta_l


def Recast_Sijkl_1xauto(Sijkl, zbins):
    npairs_auto = (zbins * (zbins + 1)) // 2
    pairs_auto = np.zeros((2, npairs_auto), dtype=int)
    count = 0
    for ibin in range(zbins):
        for jbin in range(ibin, zbins):
            pairs_auto[0, count] = ibin
            pairs_auto[1, count] = jbin
            count += 1
    Sijkl_recast = np.zeros((npairs_auto, npairs_auto))
    for ipair in range(npairs_auto):
        ibin = pairs_auto[0, ipair]
        jbin = pairs_auto[1, ipair]
        for jpair in range(npairs_auto):
            kbin = pairs_auto[0, jpair]
            lbin = pairs_auto[1, jpair]
            Sijkl_recast[ipair, jpair] = Sijkl[ibin, jbin, kbin, lbin]
    return [Sijkl_recast, npairs_auto, pairs_auto]


def Recast_Sijkl_3x2pt(Sijkl, nzbins):
    npairs_auto = (nzbins * (nzbins + 1)) // 2
    npairs_full = nzbins * nzbins + 2 * npairs_auto
    pairs_full = np.zeros((2, npairs_full), dtype=int)
    count = 0
    for ibin in range(nzbins):
        for jbin in range(ibin, nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    for ibin in range(nzbins, nzbins * 2):
        for jbin in range(nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    for ibin in range(nzbins, 2 * nzbins):
        for jbin in range(ibin, 2 * nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    Sijkl_recast = np.zeros((npairs_full, npairs_full))
    for ipair in range(npairs_full):
        ibin = pairs_full[0, ipair]
        jbin = pairs_full[1, ipair]
        for jpair in range(npairs_full):
            kbin = pairs_full[0, jpair]
            lbin = pairs_full[1, jpair]
            Sijkl_recast[ipair, jpair] = Sijkl[ibin, jbin, kbin, lbin]
    return [Sijkl_recast, npairs_full, pairs_full]


## build the noise matrices ##
def build_noise(zbins, nProbes, sigma_eps2, ng, EP_or_ED='EP'):
    """
    function to build the noise power spectra.
    ng = number of galaxies per arcmin^2 (constant, = 30 in IST:F 2020)
    n_bar = # of gal per bin
    """
    conversion_factor = 11818102.860035626  # deg to arcmin^2

    # if ng is a number, n_bar will be ng/zbins and the bins have to be equipopulated
    if type(ng) == int or type(ng) == float:
        assert ng > 0, 'ng should be positive'
        assert EP_or_ED == 'EP', 'if ng is a scalar (not a vector), the bins should be equipopulated'
        if ng < 20:
            warnings.warn(
                'ng should roughly be > 20 (this check is meant to make sure that ng is the cumulative galaxy '
                'density, not the galaxy density in each bin)')
        n_bar = ng / zbins * conversion_factor

    # if ng is an array, n_bar == ng (this is a slight minomer, since ng is the cumulative galaxy density, while
    # n_bar the galaxy density in each bin). In this case, if the bins are quipopulated, the n_bar array should
    # have all entries almost identical.
    elif type(ng) == np.ndarray:
        assert np.all(ng > 0), 'ng should be positive'
        assert np.sum(ng) > 20, 'ng should roughly be > 20'
        if EP_or_ED == 'EP':
            assert np.allclose(np.ones_like(ng) * ng[0], ng, rtol=0.05,
                               atol=0), 'if ng is a vector and the bins are equipopulated, ' \
                                        'the value in each bin should be the same (or very similar)'
        n_bar = ng * conversion_factor

    else:
        raise ValueError('ng must be an int, float or numpy.ndarray')

    # create and fill N
    N = np.zeros((nProbes, nProbes, zbins, zbins))
    np.fill_diagonal(N[0, 0, :, :], sigma_eps2 / n_bar)
    np.fill_diagonal(N[1, 1, :, :], 1 / n_bar)
    N[0, 1, :, :] = 0
    N[1, 0, :, :] = 0
    return N


def my_exit():
    print('\nquitting script with sys.exit()')
    sys.exit()


def pk_vinc_file_to_2d_npy(path, plot_pk_z0):
    # e.g. path = '/home/davide/Documenti/Lavoro/Programmi/CAMB_pk_baryons/output/Omega_M/PddVsZedLogK-Omega_M_3.040e-01.dat'
    warnings.warn(
        'double-check the units in the header and whether k is in log scale in the input file (this function assumes it is))')
    warnings.warn('the output ordering is [k, z], not the other way around!')

    pkfile = np.genfromtxt(path)
    z_array = np.unique(pkfile[:, 0])
    k_array = 10 ** np.unique(pkfile[:, 1])
    pk_2D = pkfile[:, 2].reshape(len(z_array), len(k_array)).T

    
    if plot_pk_z0:
        plt.figure()
        plt.plot(k_array, pk_2D[:, 0])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('k [1/Mpc]')
        plt.ylabel('P(k) Mpc^3')
        plt.show()

    return k_array, z_array, pk_2D


########################### SYLVAINS FUNCTIONS ################################
@njit
def cov_4D_to_2D_sylvains_ord(cov_4D, nbl, npairs):
    """Reshape from 2D to 4D using Sylvain's ordering"""
    cov_2D = np.zeros((nbl * npairs, nbl * npairs))
    for ipair in range(npairs):
        for jpair in range(npairs):
            for l1 in range(nbl):
                for l2 in range(nbl):
                    cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]
    return cov_2D


@njit
def cov_2D_to_4D_sylvains_ord(cov_2D, nbl, npairs):
    """Reshape from 4D to 2D using Sylvain's ordering"""
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ipair in range(npairs):
        for jpair in range(npairs):
            for l1 in range(nbl):
                for l2 in range(nbl):
                    cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]
    return cov_4D


def cl_3D_to_1D(cl_3D, ind, is_auto_spectrum, block_index):
    """This flattens the Cl_3D to 1D. Two ordeting conventions are used:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    - which ind file to use
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.
    :param is_auto_spectrum:
    """

    assert cl_3D.shape[1] == cl_3D.shape[2], 'cl_3D should be an array of shape (nbl, zbins, zbins)'

    nbl = cl_3D.shape[0]
    zbins = cl_3D.shape[1]

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(zbins)

    # 1. reshape to 2D
    if is_auto_spectrum:
        cl_2D = Cl_3D_to_2D_symmetric(cl_3D, nbl, zpairs_auto, zbins)
    elif not is_auto_spectrum:
        cl_2D = Cl_3D_to_2D_asymmetric(cl_3D)
    else:
        raise ValueError('is_auto_spectrum must be either True or False')

    # 2. flatten to 1D
    if block_index == 'ell' or block_index == 'vincenzo':
        cl_1D = cl_2D.flatten(order='C')
    elif block_index == 'ij' or block_index == 'sylvain':
        cl_1D = cl_2D.flatten(order='F')
    else:
        raise ValueError('block_index must be either "ij" or "ell"')

    return cl_1D
