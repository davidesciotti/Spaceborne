import warnings

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

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

###############################################################################
###############################################################################
###############################################################################

# function to reshape 2D Cls:
# fill upper triangle: LL, GG, WLonly    
zbins = 10    




# Wadd
# for ell in range(nbl_WA):
#     for i in range(npairs):
#         C_WA_3D[ell, triu_idx[0][i], triu_idx[1][i]] = C_WA_2D[ell, i]
        
# fill asymmetric




C_LL_3D_marco_bia0 = np.load(path / 'data/CosmoCentral_outputs/Cl/C_LL_3D_marco_bia0.0.npy')
C_GL_3D_marco_bia0 = np.load(path / 'data/CosmoCentral_outputs/Cl/C_GL_3D_marco_bia0.0.npy')
C_GG_3D_marco_bia0 = np.load(path / 'data/CosmoCentral_outputs/Cl/C_GG_3D_marco_bia0.0.npy')

C_LL_3D_marco_bia2 = np.load(path / 'data/CosmoCentral_outputs/Cl/C_LL_3D_marco_bia2.17.npy')
C_GL_3D_marco_bia2 = np.load(path / 'data/CosmoCentral_outputs/Cl/C_GL_3D_marco_bia2.17.npy')
C_GG_3D_marco_bia2 = np.load(path / 'data/CosmoCentral_outputs/Cl/C_GG_3D_marco_bia2.17.npy')

C_LL_2D_CLOE_bia0 = np.genfromtxt(path.parent / 'common_data/CLOE/Cl_from_simulate_data_dotpy/Cls_zNLA_ShearShear_NL_flag_2.dat')
C_GL_2D_CLOE_bia0 = np.genfromtxt(path.parent / 'common_data/CLOE/Cl_from_simulate_data_dotpy/Cls_zNLA_PosShear_NL_flag_2.dat')
C_GG_2D_CLOE_bia0 = np.genfromtxt(path.parent / 'common_data/CLOE/Cl_from_simulate_data_dotpy/Cls_zNLA_PosPos_NL_flag_2.dat')

C_LL_2D_vinc_bia2 = np.genfromtxt(path.parent / 'common_data/vincenzo/14may/CijDers/EP10/CijLL-GR-Flat-eNLA-NA.dat')
C_GL_2D_vinc_bia2 = np.genfromtxt(path.parent / 'common_data/vincenzo/14may/CijDers/EP10/CijGL-GR-Flat-eNLA-NA.dat')
C_GG_2D_vinc_bia2 = np.genfromtxt(path.parent / 'common_data/vincenzo/14may/CijDers/EP10/CijGG-GR-Flat-eNLA-NA.dat')

ell_marco = np.load(path / 'data/CosmoCentral_outputs/ell_values_marco.npy')
ell_santi = C_LL_2D_CLOE_bia0[:,0]
ell_vinc = 10**C_LL_2D_vinc_bia2[:,0]

# remove ell column
C_LL_2D_CLOE_bia0 = C_LL_2D_CLOE_bia0[:, 1:]
C_GL_2D_CLOE_bia0 = C_GL_2D_CLOE_bia0[:, 1:]
C_GG_2D_CLOE_bia0 = C_GG_2D_CLOE_bia0[:, 1:]

C_LL_2D_vinc_bia2 = C_LL_2D_vinc_bia2[:, 1:]
C_GL_2D_vinc_bia2 = C_GL_2D_vinc_bia2[:, 1:]
C_GG_2D_vinc_bia2 = C_GG_2D_vinc_bia2[:, 1:]

# reshape 
C_LL_3D_CLOE_bia0 = mm.cl_2D_to_3D_symmetric(C_LL_2D_CLOE_bia0, 20, 55)
C_LL_3D_vinc_bia2 = mm.cl_2D_to_3D_symmetric(C_LL_2D_vinc_bia2, 101, 55)
C_GL_3D_vinc_bia2 = mm.cl_2D_to_3D_asymmetric(C_GL_2D_vinc_bia2, 101, 55)
C_GL_2D_vinc_bia2_revert = mm.Cl_3D_to_2D_asymmetric(C_GL_3D_vinc_bia2, 101, 100)
warnings.warn('C_GL_2D_vinc_bia2_revert is reshaped according to ordering=F, is that right? probably yes but check')

i = 90
diff = mm.percent_diff(C_LL_3D_marco_bia0, C_LL_3D_marco_bia2)
mm.matshow(diff[i, :, :], title=f'(C_LL_bia0/C_LL_bia2.17 - 1)*100, $\\ell$ = {ell_marco[i]:.2f}')
plt.xlabel('$z_i$')
plt.ylabel('$z_j$')

plt.figure()
for i in [0, 4, 6, 9]:
    plt.plot(ell_marco, C_LL_3D_marco_bia0[:, i, i], label='marco_bia0')
    plt.plot(ell_marco, C_LL_3D_marco_bia2[:, i, i], '--', label='marco_bia2')
    # plt.plot(ell_santi, C_LL_3D_CLOE_bia0[:, i, i], '--', label='santi_bia0')
    # plt.plot(ell_vinc, C_LL_3D_vinc_bia2[:, i, i], '--', label='vinc_bia2')

plt.legend()
plt.yscale('log')


