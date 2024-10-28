import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
ROOT = os.getenv('ROOT')
sys.path.append(f'{ROOT}/Spaceborne')
import spaceborne.my_module as mm
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
                uncert_a = mm.uncertainties_FM(
                    fm_dict_a[key], 10, fiducials=fiducials_a, which_uncertainty='marginal', normalize=True)
                uncert_b = mm.uncertainties_FM(
                    fm_dict_b[key], 10, fiducials=fiducials_b, which_uncertainty='marginal', normalize=True)
                diff = mm.percent_diff(uncert_a, uncert_b)

                plt.figure()
                plt.title(f'Marginalised uncertainties, {key}')
                plt.plot(param_names, uncert_a, label='FM_A')
                plt.plot(param_names, uncert_b, ls='--', label='FM_B')
                plt.plot(param_names, diff, label='percent diff')
                plt.legend()


ml = 245
ms = 245
for zbins in [3, 5, 7, 9, 10, 13, 15]:
    for ep_ed in ['EP', 'ED']:

        # wf
        fire = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/Windows/WiFid'
        life = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3_may24/InputFiles/InputSSC/Windows'
        fire = np.genfromtxt(f'{fire}/widelta-{ep_ed}{zbins:02d}-ML{ml}-MS{ms}-idIA2-idB3-idM3-idR1.dat')
        life = np.genfromtxt(f'{life}/widelta-{ep_ed}{zbins:02d}-ML{ml}-MS{ms}-idIA2-idB3-idM3-idR1.dat')
        np.testing.assert_allclose(fire, life, rtol=1e-3, atol=0)
        try:
            np.testing.assert_allclose(fire, life, rtol=1e-3, atol=0)
        except AssertionError:
            print(f'wf {ep_ed}, {zbins}, {ml}, {ms} mismatch')

        # dv
        fire = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/FiRe/OutputQuantities/DataVectors/DataVecFid/SPV3/All/HMCodeBar'
        life = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3_may24/OutputFiles/DataVectors/Temp/3x2pt/HMCodeBar'
        fire = np.genfromtxt(f'{fire}/dv-3x2pt-{ep_ed}{zbins:02d}-zedMin02-zedMax25-ML{ml}-ML{ms}.dat')
        life = np.genfromtxt(f'{life}/dv-3x2pt-{ep_ed}{zbins:02d}-ML{ml}-MS{ms}-idIA2-idB3-idM3-idR1.dat')
        try:
            np.testing.assert_allclose(fire, life, rtol=1e-3, atol=0)
        except AssertionError:
            print(f'dv {ep_ed}, {zbins}, {ml}, {ms} mismatch')


# wf
fire = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/FiRe/InputQuantities/NzFiles/SPV3/Windows/WiFid'
fire1 = np.genfromtxt(f'{fire}/widelta-ED13-ML{ml}-MS{ms}-idIA2-idB3-idM3-idR1.dat')
fire2 = np.genfromtxt(f'{fire}/widelta-EP13-ML{ml}-MS{ms}-idIA2-idB3-idM3-idR1.dat')
try:
    np.testing.assert_allclose(fire1, fire2, rtol=1e-3, atol=0)
except AssertionError:
    print(f'dv {ep_ed}, {zbins}, {ml}, {ms} mismatch')

folder_a = '/home/davide/Scaricati/DataVectors-20240821T141626Z-001/DataVectors/DataVecFid/SPV3/All/HMCodeBar'
folder_b = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3_may24/OutputFiles/DataVectors/Temp/3x2pt/HMCodeBar/'
extension = 'dat'

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
        plt.plot((np.sqrt(np.diag(cov_a)) / np.sqrt(np.diag(cov_b)) - 1) * 100)
