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
start = time.time()

sys.path.append(str(project_path))
sys.path.append(str(project_path.parent / 'common_data/common_config'))
sys.path.append(str(project_path.parent / 'common_data/common_lib'))

import ISTF_fid_params as ISTF
import my_module as mm

c = ISTF.constants['c']

Om0 = ISTF.primary['Om_m0']
Ob0 = ISTF.primary['Om_b0']
w0 = ISTF.primary['w_0']
wa = ISTF.primary['w_a']
h = ISTF.primary['h_0']
n_s = ISTF.primary['n_s']
sigma_8 = ISTF.primary['sigma_8']

Ode0 = ISTF.extensions['Om_Lambda0']
Ok0 = ISTF.extensions['Om_k0']

Neff = ISTF.neutrino_params['N_eff']
m_nu = ISTF.extensions['m_nu']

Oc0 = Om0 - Ob0
H0 = h * 100

# ! options
WF_normalization = 'IST'
inputs = 'davide'
# ! end options

# TODO sylvain's redshift starts from 0, not accepted by Sijkl


# COSE DA CHIARIRE
# 1) i parametri che passo nei due casi sono corretti? E le estensioni?
# 2) definizione delle window functions:
# 3) va bene se CovSSC non è simmetrica?
# 4) LE INTERPOLAZIONI NON SONO BUONE PER GLI ULTIMI W 
# 5) Omega_lambda non rientra nel set di parametri, lo devo includere?


###### CHOOSING THE PARAMETERS
Omega_b = 0.05
Omega_m = 0.32
h = 0.67
n_s = 0.96
sigma8 = 0.816
m_ncdm = 0.06

Omega_ni = m_ncdm / (93.14 * h * h)
Omega_cdm = Omega_m - Omega_b - Omega_ni

N_ncdm = 1
N_ur = 2.03351
k_max = 30

# posso anche dargli
Omega_Lambda = 0.68  # che però non rientra nel set di parametri
w0_fld = -1
wa_fld = 0
# gamma non lo trovo in CLASS

# Default values for redshift bin, cosmo parameters etc
# cosmo_params_default = {'omega_b':0.022,'omega_cdm':0.12,'H0':67.,'n_s':0.96,'sigma8':0.81}

cosmo_params_davide = {  # with neutrinos
    'output': 'mPk',  # xxx uncommented on feb 22
    # 'non linear':'halofit', #xxx punto delicato, takabird?
    'Omega_b': Omega_b,
    'Omega_cdm': Omega_cdm,
    'h': h,
    'sigma8': sigma8,
    'n_s': n_s,
    'P_k_max_1/Mpc': k_max,
    #                       'z_max_pk': 2.038, zmax glielo passa lui
    'N_ncdm': N_ncdm,
    'm_ncdm': m_ncdm,
    'N_ur': N_ur,
    'Omega_Lambda': Omega_Lambda,
    'w0_fld': w0_fld,
    'wa_fld': wa_fld,
    'z_pk': '0, 0.5',  # ! I get an error without this option?
}

########## DEFINING THE FUNCTION ###########
# Routine to compute the Sij matrix with top-hat disjoint redshift window functions
# example : galaxy clustering with perfect/spectroscopic redshift determinations so that bins are sharp.
# Inputs : stakes of the redshift bins (array), cosmological parameters (dictionnary as in CLASS's wrapper classy)
# Output : Sij matrix (size: nbins x nbins)

# Routine to compute the Sijkl matrix, i.e. the most general case with cross-spectra
# Inputs : window functions, cosmological parameters, same format as Sij()
# Format for window functions : one table of redshifts with size nz, one 2D table 
# for the collection of window functions with shape (nbins,nz)
# Output : Sijkl matrix (shape: nbins x nbins x nbins x nbins)
# Equation used :  Sijkl = 1/(2*pi^2) \int kk^2 dkk P(kk) U(i,j;kk)/Inorm(i,j) U(k,l;kk)/Inorm(k,l)
# with Inorm(i,j) = int dV window(i,z) window(j,z) and U(i,j;kk) = int dV window(i,z) window(j,z) growth(z) j_0(kk*r)


zbin = 10

############################## check WFs: 
"""
dav = wig 
luca = wig_luca

diff = (luca.T/dav[:,1:]-1)*100

for z_bin in range(10):

    plt.plot(z_luca, luca[z_bin, :], label='luca')
    plt.plot(dav[:,0], dav[:,z_bin+1], label='davide')
    
    plt.plot(z_luca, np.abs(diff[:,z_bin]), label='diff')

    plt.yscale('log')
    plt.legend()
"""

######### PASSING THE WINDOW FUNCTIONS ##########



if WF_normalization == 'PySSC':
    convention = 0
elif WF_normalization == 'IST':
    convention = 1
else:
    raise ValueError('WF_normalization must be either PySSC or IST')

if inputs == 'davide':
    wil = np.genfromtxt(
        project_path.parent / f"common_data/everyones_WF_from_Gdrive/davide/nz10000/wil_dav_IA_IST_nz10000.txt")
    wig = np.genfromtxt(
        project_path.parent / f"common_data/everyones_WF_from_Gdrive/davide/nz10000/wig_dav_IST_nz10000.txt")

    z_arr = wil[:, 0]  # setting the redshift array, z_arr
    z_points = z_arr.shape[0]

    # deleting the redshift column (0-th column):
    wil = np.delete(wil, 0, axis=1)
    wig = np.delete(wig, 0, axis=1)

    # transpose
    wil = np.transpose(wil)
    wig = np.transpose(wig)

    # vertically stackthe WFs (row-wise, wil first, wig second)
    w = np.vstack((wil, wig))

elif inputs == 'luca':
    data_luca = np.load(f"{project_path}/data/CLOE/seyfert_inputs_for_SSC/inputs_for_SSC.npz")

    wil_luca = data_luca['W_L_iz_over_chi2_z']
    wig_luca = data_luca['W_G_iz_over_chi2_z']

    z_arr = data_luca['z_grid']  # same as me: np.linspace (1e-3, 4, 1e4)
    z_points = z_arr.shape[0]

    w = np.vstack((wil_luca, wig_luca))  # vertically stacking the WFs (row-wise, wil first, wig second)

else:
    print('input not recognised')

fileName = f"Sijkl_WF{inputs}_nz{z_points}_IA.npy"

# calling the routine
# ! change convention!!!
windows = w
Sijkl_arr = Sijkl(z_arr, windows, cosmo_params=cosmo_params_davide, precision=10, tol=1e-3, convention=1)

# np.save(f"{project_path}/output/Sijkl_everyonesWF/no_z_interpolation/CLOE/{fileName}", Sijkl_arr)

print("the program took %i seconds to run" % (time.time() - start))

Sijkl_dav = np.load(project_path / "config/common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy")  # davide, eNLA
plt.matshow(Sijkl_dav[0, 0, :, :])
plt.matshow(Sijkl_arr[0, 0, :, :])

diff = mm.percent_diff_nan(Sijkl_dav, Sijkl_arr)
print(np.where(diff > 3))


