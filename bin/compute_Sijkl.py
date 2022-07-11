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
project_path = Path.cwd().parent

sys.path.append(str(project_path))
sys.path.append(str(project_path.parent / 'common_data/common_config'))
sys.path.append(str(project_path.parent / 'common_data/common_lib'))

import ISTF_fid_params as ISTF
import my_module as mm



# COSE DA CHIARIRE
# 1) i parametri che passo nei due casi sono corretti? E le estensioni?
# 2) definizione delle window functions:
# 3) va bene se CovSSC non è simmetrica?
# 4) LE INTERPOLAZIONI NON SONO BUONE PER GLI ULTIMI W
# 5) Omega_lambda non rientra nel set di parametri, lo devo includere?
# TODO sylvain's redshift starts from 0, not accepted by Sijkl


def compute_Sijkl(cosmo_params_dict, Sijkl_config):


    save_Sijkl = cfg.Sijkl_config['save_Sijkl']
    input_WF = cfg.Sijkl_config['input_WF']
    WF_normalization = cfg.Sijkl_config['WF_normalization']

    WF_path = project_path.parent / 'common_data/everyones_WF_from_Gdrive'

    if WF_normalization == 'PySSC':
        convention = 0
    elif WF_normalization == 'IST':
        convention = 1
    else:
        raise ValueError('WF_normalization must be either PySSC or IST')

    if input_WF == 'davide':

        wil = np.genfromtxt(f'{WF_path}/davide/nz10000/wil_dav_IA_{WF_normalization}_nz10000.txt')
        wig = np.genfromtxt(f'{WF_path}/davide/nz10000/wig_dav_{WF_normalization}_nz10000.txt')

        z_arr = wil[:, 0]  # setting the redshift array, z_arr
        z_points = z_arr.shape[0]

        # deleting the redshift column (0-th column):
        wil = np.delete(wil, 0, axis=1)
        wig = np.delete(wig, 0, axis=1)

        # transpose
        wil = np.transpose(wil)
        wig = np.transpose(wig)

        # vertically stack the WFs (row-wise, wil first, wig second)
        w = np.vstack((wil, wig))

    elif input_WF == 'luca':

        wig = np.load('/Users/davide/Documents/Lavoro/Programmi/common_data/everyones_WF_from_Gdrive/luca/wig_SEYFERT_IST_nz10000.npy')

        # wil = np.genfromtxt(f'{WF_path}/luca/nz10000/wil_dav_IA_{WF_normalization}_nz10000.txt")

        data_luca = np.load(project_path / f"data/CLOE/seyfert_inputs_for_SSC/inputs_for_SSC.npz")

        wil_luca = data_luca['W_L_iz_over_chi2_z']
        wig_luca = data_luca['W_G_iz_over_chi2_z']

        z_arr = data_luca['z_grid']  # same as me: np.linspace (1e-3, 4, 1e4)
        z_points = z_arr.shape[0]

        w = np.vstack((wil_luca, wig_luca))  # vertically stacking the WFs (row-wise, wil first, wig second)

    else:
        raise ValueError('input_WF must be either davide or luca')

    fileName = f"Sijkl_WF{input_WF}_nz{z_points}_IA.npy"

    # calling the routine
    # ! change convention!!!
    windows = w

    start = time.perf_counter()
    Sijkl_arr = Sijkl(z_arr, windows, cosmo_params=cosmo_params_davide, precision=10, tol=1e-3, convention=1)

    # np.save(f"{project_path}/output/Sijkl_everyonesWF/no_z_interpolation/CLOE/{fileName}", Sijkl_arr)

    print(f"Sijkl matrix computed in {time.perf_counter() - start})

    return Sijkl_arr
