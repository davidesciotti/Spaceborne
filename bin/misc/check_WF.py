import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time
from scipy.interpolate import interp1d

# get project directory
path = Path.cwd().parent.parent

# import configuration and functions modules
sys.path.append(str(path.parent / 'common_data'))
import my_config as my_config
sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

start_time = time.perf_counter()

params = {'lines.linewidth' : 2.5,
          'font.size' : 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral'
          }
plt.rcParams.update(params)
markersize = 10

c = 299792.458 # km/s 
H0 = 67 #km/(s*Mpc)

###############################################################################
###############################################################################
###############################################################################

# this is to interpolate marco's WFs in my (300) z grid
def WF_interpolator(zbins, z_import, WF_import, z_values, nz):
    WF_interpolated = np.zeros((nz, zbins))
    for j in range(zbins):
        f = interp1d(z_import, WF_import[:, j], kind='linear')
        WF_interpolated[:, j] = f(z_values)
    return WF_interpolated

# marco, plus seyfert (to have bia = 2.17)
wil_m = np.load(path.parent / 'common_data/everyones_WF_from_Gdrive/marco/WL_WeightFunction_zNLA.npy').T
wig_m = np.load(path.parent / 'common_data/everyones_WF_from_Gdrive/marco/GC_WeightFunction.npy').T
wil_m_seyf = np.load(path.parent / 'SSCcomp_prove/data/seyfert_inputs_for_SSC/inputs_for_SSC.npz')['W_L_iz'].T
wil_m_bia0 = np.load(path / 'data/Cl_e_WF_marco/WL_WeightFunction_zNLA_bia0.0.npy').T
wil_m_bia2 = np.load(path / 'data/Cl_e_WF_marco/WL_WeightFunction_zNLA_bia2.17.npy').T

# wil_m_bia2 = 
# wigl_m = 
# /home/cosmo/davide.sciotti/data/SSC_for_ISTNL/data/Cl_e_WF_marco

# pay attention to the different bia values
wil_d_bia0 = np.genfromtxt(path.parent / 'Cij_davide/output/WF/WFs_v15_zNLA_gen22/wil_davide_IA_IST_nz300_bia0.00.txt')
wil_d_bia2 = np.genfromtxt(path.parent / 'Cij_davide/output/WF/WFs_v15_zNLA_gen22/wil_davide_IA_IST_nz300_bia2.17.txt')
wil_d_old = np.genfromtxt(path.parent / 'Cij_davide/output/WF/WFs_v14_multiBinBias_renamed_nz300/wil_davide_IA_IST_nz300.txt')

wig_d = np.genfromtxt(path.parent / 'Cij_davide/output/WF/WFs_v15_zNLA_gen22/wig_davide_multiBinBias_IST_nz300.txt')

# vincenzo (wil noIA)
wil_v = np.genfromtxt(path.parent / 'common_data/everyones_WF_from_Gdrive/vincenzo/wil_vincenzo_noIA_IST_nz300.dat')
wig_v = np.genfromtxt(path.parent / 'common_data/everyones_WF_from_Gdrive/vincenzo/wig_vincenzo_IST_nz300.dat')

wil_s = np.genfromtxt(path.parent / 'common_data/everyones_WF_from_Gdrive/sylvain/new_WF_IA_corrected/wil_sylvain_IA_IST_nz7000.txt')
wig_s = np.genfromtxt(path.parent / 'common_data/everyones_WF_from_Gdrive/sylvain/new_WF_IA_corrected/wig_sylvain_IST_nz7000.txt')

z_marco = np.linspace(0.001, 4, 10_000)
z_dav = wig_d[:,0]

# interpolate marco's WF in my z values 
# wil_m_interp = WF_interpolator(zbins=10, z_import=z_marco, WF_import=wil_m, z_values=z_dav, nz=z_dav.shape[0])
# diff = mm.percent_diff(wil_m_interp, wil_d_bia0[:, 1:])
# for i in range(10):
#     plt.plot(z_dav, diff[:, i], '.-', label=f'marco {i}') # bia = 0
# plt.legend()



marco = wil_m
dav_bia0 = wil_d_bia0
dav_bia2 = wil_d_bia2
dav_old = wil_d_old
vinc = wil_v
sylv = wil_s


plt.figure()
# for i in range(10):
#     plt.plot(z_marco, marco[:, i], label='marco') # bia = 0
#     plt.plot(z_marco, wil_m_seyf[:, i], '--', label='wil_m_seyf') # bia = 2.17
    
    # plt.plot(z_dav, wil_m_interp[:, i], '.', label='wil_m_interp')
    # plt.plot(z_dav, ratio[:, i], '.', label='ratio')

    # plt.plot(dav_bia0[:, 0], dav_bia0[:, i+1], '--', label='dav_bia0')
    # plt.plot(dav_bia2[:, 0], dav_bia2[:, i+1], '--', label='dav_bia2')
    # plt.plot(dav_old[:, 0], dav_old[:, i+1], '--', label='dav_old')
    
    # plt.plot(vinc[:, 0], vinc[:, i+1], '--', label='vincenzo noIA')
    # plt.plot(sylv[:, 0], sylv[:, i+1], '--', label='sylv')
    
# plt.yscale('log')

for i in [0, 5, 9]:
    plt.plot(z_marco, wil_m_bia0[:, i], label='wil_m_bia0') 
    plt.plot(z_marco, wil_m_bia2[:, i], label='wil_m_bia2') 
    plt.plot(dav_bia0[:, 0], dav_bia0[:, i+1], '--', label='dav_bia0')
    plt.plot(dav_bia2[:, 0], dav_bia2[:, i+1], '--', label='dav_bia2')
plt.legend()
