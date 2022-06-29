import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad, nquad
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



###############################################################################



c = 299792.458 # km/s 
H0 = 67 #km/(s*Mpc)

Om0  = 0.32
Ode0 = 0.68
Ox0  = 0


####################################### function definition
def E(z):
    result = np.sqrt(Om0*(1 + z)**3 + Ode0 + Ox0*(1 + z)**2)
    return result

def inv_E(z):
    result = 1/np.sqrt(Om0*(1 + z)**3 + Ode0 + Ox0*(1 + z)**2)
    return result
    
def r_tilde(z):
    #r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
    # have r_tilde(z)
    result = quad(inv_E, 0, z)  # integrate 1/E(z) from 0 to z
    return result[0]

def r(z):
    result = c/H0 * quad(inv_E, 0, z)[0] 
    return result


cosmo = w0waCDM(H0, Om0, Ode0, w0=-1.0, wa=0.0)


z_arr = np.linspace(0.001, 4, 300)

my_r = np.asarray([r(zi) for zi in z_arr])
r_astropy = cosmo.comoving_distance(z_arr)

# plt.plot(z_arr, my_r)
# plt.plot(z_arr, r_astropy, '--')

diff = mm.percent_diff(my_r, r_astropy.value)
plt.plot(z_arr, diff)








    