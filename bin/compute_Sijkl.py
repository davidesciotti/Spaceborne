import math

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
print(project_path)

sys.path.append(str(project_path / 'jobs/SPV3/configs'))
import config_SPV3 as cfg

sys.path.append(str(project_path.parent / 'common_data/common_config'))
import ISTF_fid_params as ISTF

sys.path.append(str(project_path.parent / 'common_data/common_lib'))
import my_module as mm


# COSE DA CHIARIRE
# 1) i parametri che passo nei due casi sono corretti? E le estensioni?
# 2) definizione delle window functions:
# 3) va bene se CovSSC non è simmetrica?
# 4) LE INTERPOLAZIONI NON SONO BUONE PER GLI ULTIMI W
# 5) Omega_lambda non rientra nel set di parametri, lo devo includere?
# TODO sylvain's redshift starts from 0, not accepted by Sijkl

def load_WF(Sijkl_config, zbins):
    input_WF = cfg.Sijkl_config['input_WF']
    WF_normalization = cfg.Sijkl_config['WF_normalization']
    has_IA = cfg.Sijkl_config['has_IA']

    WF_path = project_path.parent / 'common_data/everyones_WF_from_Gdrive'

    if input_WF == 'luca':

        data_luca = np.load(project_path / f"data/CLOE/seyfert_inputs_for_SSC/inputs_for_SSC.npz")

        wil_luca = data_luca['W_L_iz_over_chi2_z']
        wig_luca = data_luca['W_G_iz_over_chi2_z']

        z_arr = data_luca['z_grid']  # same as me: np.linspace (1e-3, 4, 1e4)
        z_points = z_arr.shape[0]

        windows = np.vstack((wil_luca, wig_luca))  # vertically stacking the WFs (row-wise, wil first, wig second)

    elif input_WF != 'luca':

        if input_WF == 'davide':
            wil = np.genfromtxt(f'{WF_path}/davide/nz10000/wil_dav_IA_{WF_normalization}_nz10000.txt')
            wig = np.genfromtxt(f'{WF_path}/davide/nz10000/wig_dav_{WF_normalization}_nz10000.txt')

        elif input_WF == 'vincenzo':
            wil = np.genfromtxt(f'{WF_path}/vincenzo/wil_vinc_IA_{WF_normalization}_nz8000.txt')
            wig = np.genfromtxt(f'{WF_path}/vincenzo/wig_vinc_{WF_normalization}_nz8000.txt')

        elif input_WF == 'marco':
            wil = np.load(f'{WF_path}/marco/wil_mar_bia2.17_{WF_normalization}_nz10000.npy')
            wig = np.load(f'{WF_path}/marco/wig_mar_{WF_normalization}_nz10000.npy')

        elif input_WF == 'sylvain':
            wil = np.genfromtxt(f'{WF_path}/sylvain/new_WF_IA_corrected/wil_sylv_IA_{WF_normalization}_nz7000.txt')
            wig = np.genfromtxt(f'{WF_path}/sylvain/new_WF_IA_corrected/wig_sylv_{WF_normalization}_nz7000.txt')

        elif input_WF == 'vincenzo_SPV3':
            wil = np.genfromtxt(f'{WF_path}/vincenzo/SPV3/KernelFun/WiWL-EP{zbins}.dat')
            wig = np.genfromtxt(f'{WF_path}/vincenzo/SPV3/KernelFun/WiGC-EP{zbins}.dat')

        else:
            raise ValueError('input_WF must be either davide, sylvain, marco, vincenzo_SPV3, vincenzo or luca')

        if wil[0, 0] == 0 or wig[0, 0] == 0:
            print('Warning: the redshift array for the weight functions starts from 0, not accepted by PySSC; '
                  'removing the first row from the array')

            wil = np.delete(wil, 0, axis=0)
            wig = np.delete(wig, 0, axis=0)

        z_arr = wil[:, 0]  # setting the redshift array, z_arr
        z_points = z_arr.shape[0]

        # deleting the redshift column (0-th column):
        wil = np.delete(wil, 0, axis=1)
        wig = np.delete(wig, 0, axis=1)

        # transpose
        wil = np.transpose(wil)
        wig = np.transpose(wig)

        print(wil.shape, wig.shape)

        # vertically stack the WFs (row-wise, wil first, wig second)
        windows = np.vstack((wil, wig))

    return z_arr, windows


def compute_Sijkl(cosmo_params_dict, Sijkl_config, zbins):
    WF_normalization = cfg.Sijkl_config['WF_normalization']

    if WF_normalization == 'PySSC':
        convention = 0
    elif WF_normalization == 'IST':
        convention = 1
    else:
        raise ValueError('WF_normalization must be either PySSC or IST')

    z_arr, windows = load_WF(Sijkl_config, zbins)

    start = time.perf_counter()
    Sijkl_arr = Sijkl(z_arr=z_arr, windows=windows, cosmo_params=cosmo_params_dict, precision=10, tol=1e-3,
                      convention=convention)
    print(f'Sijkl matrix computed in {time.perf_counter() - start}')

    return Sijkl_arr
