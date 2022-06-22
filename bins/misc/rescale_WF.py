import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time
from astropy.cosmology import w0waCDM


# get project directory
path = Path.cwd().parent.parent

# import configuration and functions modules
sys.path.append(str(path.parent / 'common_data'))
import my_config as my_config
sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

start_time = time.perf_counter()

params = {'lines.linewidth' : 3.5,
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


c = 299792.458 # km/s 
H0 = 67 #km/(s*Mpc)

Om0  = 0.32
Ode0 = 0.68
Ox0  = 0



wil_m_bia0 = np.load(path / 'data/CosmoCentral_outputs/WF/WL_WeightFunction_bia0.0.npy').T
wil_m_bia2 = np.load(path / 'data/CosmoCentral_outputs/WF/WL_WeightFunction_bia2.17.npy').T
wig_m = np.load(path / 'data/CosmoCentral_outputs/WF/GC_WeightFunction.npy').T

cosmo = w0waCDM(H0, Om0, Ode0, w0=-1.0, wa=0.0)

z_arr = np.linspace(0.001, 4, 10_000)

r_astropy = cosmo.comoving_distance(z_arr).value

wil_m_bia0_PySSC = np.zeros(wil_m_bia0.shape)
wil_m_bia2_PySSC = np.zeros(wil_m_bia2.shape)
wig_m_PySSC = np.zeros(wig_m.shape)
for i in range(10):
    wil_m_bia0_PySSC[:,i] = wil_m_bia0[:,i]/r_astropy**2
    wil_m_bia2_PySSC[:,i] = wil_m_bia2[:,i]/r_astropy**2
    wig_m_PySSC[:,i] = wig_m[:,i]/r_astropy**2

    
# plt.plot(z_arr, wig_m_PySSC[:,0])
# plt.plot(z_arr, wig_m[:,0])

np.save(path / 'data/CosmoCentral_outputs/WF/WL_WeightFunction_bia0.0_PySSC.npy', wil_m_bia0_PySSC)
np.save(path / 'data/CosmoCentral_outputs/WF/WL_WeightFunction_bia2.17_PySSC.npy', wil_m_bia2_PySSC)
np.save(path / 'data/CosmoCentral_outputs/WF/GC_WeightFunction_PySSC.npy', wig_m_PySSC)

# check: import sylvain and plot
wil_s = np.genfromtxt(path.parent / 'common_data/everyones_WF_from_Gdrive/sylvain/new_WF_IA_corrected/wil_sylvain_IA_PySSC_nz7000.txt')
wig_s = np.genfromtxt(path.parent / 'common_data/everyones_WF_from_Gdrive/sylvain/new_WF_IA_corrected/wig_sylvain_PySSC_nz7000.txt')

i = 0
plt.plot(z_arr, wil_m_bia2_PySSC[:,i])
plt.plot(wil_s[:,0], wil_s[:,i+1])
plt.yscale('log') # ok


