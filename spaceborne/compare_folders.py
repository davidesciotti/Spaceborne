import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
ROOT = os.getenv('ROOT')
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
import common_cfg.mpl_cfg as mpl_cfg

plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

def compare_param_cov_from_fm_pickles(fm_pickle_path_a, fm_pickle_path_b, compare_fms=True, compare_param_covs=True, plot=True):

    fm_dict_a = mm.load_pickle(fm_pickle_path_a)
    fm_dict_b = mm.load_pickle(fm_pickle_path_b)

    # check that the keys match
    assert fm_dict_a.keys() == fm_dict_b.keys()

    # check if the dictionaries contained in the key 'fiducial_values_dict' match
    assert fm_dict_a['fiducial_values_dict'] == fm_dict_b['fiducial_values_dict'], 'fiducial values do not match!'

    # check that the values match
    for key in fm_dict_a.keys():
        if key != 'fiducial_values_dict' and 'WA' not in key:
            print('Comparing ', key)
            fm_dict_a[key] = mm.remove_null_rows_cols_2D_copilot(fm_dict_a[key])
            fm_dict_b[key] = mm.remove_null_rows_cols_2D_copilot(fm_dict_b[key])

            cov_a = np.linalg.inv(fm_dict_a[key])
            cov_b = np.linalg.inv(fm_dict_b[key])
            
            if compare_fms:
                mm.compare_arrays(fm_dict_a[key], fm_dict_b[key], 'FM_A', 'FM_B', plot_diff_threshold=5)

            if compare_param_covs:

                mm.compare_arrays(cov_a, cov_b, 'cov_A', 'cov_B', plot_diff_threshold=5)
                
            if plot:
                param_names = list(fm_dict_a['fiducial_values_dict'].keys())[:10]
                fiducials_a = list(fm_dict_a['fiducial_values_dict'].values())[:10]
                fiducials_b = list(fm_dict_b['fiducial_values_dict'].values())[:10]
                uncert_a = mm.uncertainties_FM(fm_dict_a[key], 10, fiducials=fiducials_a, which_uncertainty='marginal', normalize=True)
                uncert_b = mm.uncertainties_FM(fm_dict_b[key], 10, fiducials=fiducials_b, which_uncertainty='marginal', normalize=True)
                diff = mm.percent_diff(uncert_a, uncert_b)
                
                
                plt.figure()
                plt.title(f'Marginalised uncertainties, {key}')
                plt.plot(param_names, uncert_a, label='FM_A')
                plt.plot(param_names, uncert_b, ls='--', label='FM_B')
                plt.plot(param_names, diff, label='percent diff')
                plt.legend()

folder_a = '/home/cosmo/davide.sciotti/data/OneCovariance/output_ISTF'
folder_b = '/home/cosmo/davide.sciotti/data/OneCovariance/output_ISTF_v2'
extension = 'npz'

mm.test_folder_content(folder_a, folder_b, extension, verbose=True, rtol=1e-3)

for fm_name in os.listdir(folder_b):
    if fm_name.endswith('.txt'):
        fm_a = np.genfromtxt(f'{folder_a}/{fm_name}')
        fm_b = np.genfromtxt(f'{folder_b}/{fm_name}')
        
        fm_a = mm.remove_null_rows_cols_2D_copilot(fm_a)
        fm_b = mm.remove_null_rows_cols_2D_copilot(fm_b)
        
        cov_a = np.linalg.inv(fm_a)
        cov_b = np.linalg.inv(fm_b)
        
        mm.compare_arrays(cov_a, cov_b, plot_diff_threshold=5)
        
        plt.figure()        
        plt.plot((np.sqrt(np.diag(cov_a))/np.sqrt(np.diag(cov_b))-1)*100)
