import math
import warnings

pi = math.pi

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import matplotlib
from classy import Class
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import json
from PySSC import Sijkl
from pathlib import Path

# ! don't touch the imports and/or their ordering, otherwise I get a malloc error when compiling

# get project directory
project_path = Path.cwd().parent.parent.parent

sys.path.append(f'{project_path}/jobs/SPV3/configs')
import config_SPV3 as cfg

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTF

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm


# COSE DA CHIARIRE
# 1) i parametri che passo nei due casi sono corretti? E le estensioni?
# 2) definizione delle window functions:
# 3) va bene se CovSSC non è simmetrica?
# 4) LE INTERPOLAZIONI NON SONO BUONE PER GLI ULTIMI W
# 5) Omega_lambda non rientra nel set di parametri, lo devo includere?


def load_WF(Sijkl_cfg, zbins, EP_or_ED):
    wf_input_folder = Sijkl_cfg['wf_input_folder']
    wf_filename = Sijkl_cfg['wf_input_filename']
    WF_normalization = Sijkl_cfg['WF_normalization']
    IA_flag = Sijkl_cfg['IA_flag']

    if not IA_flag:
        raise ValueError('IA_flag must be True')

    WF_path = f'{project_path.parent}/common_data/everyones_WF_from_Gdrive'

    if wf_input_folder == 'luca':

        wil = np.load(f'{WF_path}/luca/wil_SEYFERT_IA_{WF_normalization}_nz10000.npy')
        wig = np.load(f'{WF_path}/luca/wig_SEYFERT_{WF_normalization}_nz10000.npy')
        z_arr = np.genfromtxt(f'{WF_path}/luca/z_values.txt')  # same as me: np.linspace (1e-3, 4, 1e4)

    else:

        if wf_input_folder == 'davide':
            wil = np.genfromtxt(f'{WF_path}/davide/nz10000/wil_dav_IA_{WF_normalization}_nz10000.txt')
            wig = np.genfromtxt(f'{WF_path}/davide/nz10000/wig_dav_{WF_normalization}_nz10000.txt')

        elif wf_input_folder == 'vincenzo':
            wil = np.genfromtxt(f'{WF_path}/vincenzo/wil_vinc_IA_{WF_normalization}_nz8000.txt')
            wig = np.genfromtxt(f'{WF_path}/vincenzo/wig_vinc_{WF_normalization}_nz8000.txt')

        elif wf_input_folder == 'marco':
            wil = np.load(f'{WF_path}/marco/wil_mar_bia2.17_{WF_normalization}_nz10000.npy')
            wig = np.load(f'{WF_path}/marco/wig_mar_{WF_normalization}_nz10000.npy')

        elif wf_input_folder == 'sylvain':
            wil = np.genfromtxt(f'{WF_path}/sylvain/new_WF_IA_corrected/wil_sylv_IA_{WF_normalization}_nz7000.txt')
            wig = np.genfromtxt(f'{WF_path}/sylvain/new_WF_IA_corrected/wig_sylv_{WF_normalization}_nz7000.txt')

        elif 'SPV3_07_2022/Flagship_1' in wf_input_folder:
            assert WF_normalization == 'IST', 'WF_normalization must be IST for Vincenzo SPV3_07_2022/Flagship_1 WFs'
            wil = np.genfromtxt(f'{wf_input_folder}/WiWL-{EP_or_ED}{zbins:02}.dat')
            wig = np.genfromtxt(f'{wf_input_folder}/WiGC-{EP_or_ED}{zbins:02}.dat')

        elif 'SPV3_07_2022/Flagship_2' in wf_input_folder:
            assert WF_normalization == 'IST', 'WF_normalization must be IST for Vincenzo SPV3_07_2022/Flagship_2 WFs'
            wil = np.genfromtxt(f'{wf_input_folder}/WiWL-{EP_or_ED}{zbins:02}-FS2.dat')
            wig = np.genfromtxt(f'{wf_input_folder}/WiGC-{EP_or_ED}{zbins:02}-FS2.dat')


        else:
            raise ValueError('input_WF must be either davide, sylvain, marco, vincenzo_SPV3, vincenzo or luca')

        if wil[0, 0] == 0 or wig[0, 0] == 0:
            print('Warning: the redshift array for the weight functions starts from 0, not accepted by PySSC; '
                  'removing the first row from the array')
            wil = np.delete(wil, 0, axis=0)
            wig = np.delete(wig, 0, axis=0)

        # set the redshift array, z_arr
        z_arr = wil[:, 0]

        # check that the redshift array in wil and wig is effectively the same
        assert np.array_equal(wil[:, 0], wig[:, 0]), 'the redshift array for the weight functions is not the same'

        # delete the redshift column (0-th column):
        wil = np.delete(wil, 0, axis=1)
        wig = np.delete(wig, 0, axis=1)

        # transpose
        wil = np.transpose(wil)
        wig = np.transpose(wig)

    # vertically stack the WFs (row-wise, wil first, wig second)
    windows = np.vstack((wil, wig))

    return z_arr, windows


def preprocess_wf(wf, zbins):
    """
    Preprocess the weight functions: removes and returns the first row (the redshift array) and the wf without it
    :param wf: the weight functions
    :return: the preprocessed weight functions
    """
    if wf[0, 0] == 0:
        print('Warning: the redshift array for the weight functions starts from 0, not accepted by PySSC; '
              'removing the first row from the array')
        wf = np.delete(wf, 0, axis=0)

    assert wf.shape[1] == zbins + 1, 'the number of columns in the input weight functions is not zbins + 1 (the first' \
                                     'column being the redshift array)'
    z_arr = wf[:, 0]
    wf = np.delete(wf, 0, axis=1)

    # further check
    assert wf.shape[1] == zbins, 'the number of weight functions is not correct'
    return z_arr, wf


def compute_Sijkl(cosmo_params_dict, z_arr, WF, WF_normalization, zbins, EP_or_ED, Sijkl_cfg=None):

    if WF_normalization == 'PySSC':
        convention = 0
    elif WF_normalization == 'IST':
        convention = 1
    else:
        raise ValueError('WF_normalization must be either PySSC or IST')

    if z_arr is None and WF is None:
        warnings.warn("Warning: The imports filepath should be specified outside this function/module!", DeprecationWarning)
        print('in Sijkl_utils: Warning: ensuring backwards compatibility; this part of the function should be changed!')
        z_arr, windows = load_WF(Sijkl_cfg, zbins, EP_or_ED=EP_or_ED)

    start = time.perf_counter()
    Sijkl_arr = Sijkl(z_arr=z_arr, windows=WF, cosmo_params=cosmo_params_dict, precision=10, tol=1e-3,
                      convention=convention)
    print(f'Sijkl matrix computed in {time.perf_counter() - start:.2f} s')

    return Sijkl_arr

