import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d, interp1d
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent.parent
job_path = Path.cwd().parent.parent

sys.path.append(str(project_path / 'lib'))
import my_module as mm

matplotlib.use('Qt5Agg')

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8)
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()

###############################################################################
###############################################################################
###############################################################################
h = 0.67

G1 = np.genfromtxt(project_path / f'config/common_data/alex_response/Resp_G1_fromsims.dat')

# take k and z values (the latter from the header)
# ! k is in [h/Mpc]
k_G1 = G1[:, 0]
z_G1 = np.array((0.00, 0.50, 1.00, 2.00, 3.00))

# remove k column
G1 = G1[:, 1:]

# for z_bin in range(len(z)):
z_bin = 0
k_bin = 0
plt.plot(k_G1, G1[:, z_bin], label=f'z={z_G1[z_bin]}')
plt.legend()

# interpolate G1
G1_funct = interp2d(x=z_G1, y=k_G1, z=G1, kind='linear')

h_units = False  # ! BUG ALERT, WHICH ONE SHOULD I USE?

# vincenzo's input Pk and stefano's reshaping method
# input is in k [1/Mpc], and P(k,z) [Mpc^3]
PkFILE_vinc = np.genfromtxt(job_path / f'input/variable_response/PnlFid.dat')
k_vinc = 10 ** (np.unique(PkFILE_vinc[:, 0]))  # / h
k_points_v = len(k_vinc)
z_vinc = np.unique(PkFILE_vinc[:, 1])
z_points_v = len(z_vinc)
Pnl_v = (10 ** PkFILE_vinc[:, 2]).reshape(z_points_v, k_points_v)  # * (h ** 3)

# stefano's input Pk and reshaping method
# input is in k [h/Mpc], and P(k,z) [Mpc^3/h^3]
PkFILE_stef = np.genfromtxt(job_path / f'input/variable_response/pkz-Fiducial.txt')
z_stef = np.unique(PkFILE_stef[:, 0])
k_points_s = int(len(PkFILE_stef[:, 2]) / len(z_stef))
k_stef = PkFILE_stef[:k_points_s, 1]  # * h
z_points_s = len(z_stef)
Pnl_s = PkFILE_stef[:, 3].reshape(z_points_s, k_points_s)  # / h ** 3

if h_units:  # i.e. if you want k [h/Mpc], and P(k,z) [Mpc^3/h^3]
    k_vinc /= h
    Pnl_v *= h ** 3
else:  # i.e. if you want k [1/Mpc], and P(k,z) [Mpc^3]
    k_stef *= h
    Pnl_s /= h ** 3

# take the derivative w.r.t. k
# ! BUG ALERT, IS THIS THE CORRECT WAY TO DO IT?
dP_dk = np.gradient(Pnl_v, k_vinc, axis=1)

# check the derivative, 'by eye'
z_idx = 0
plt.figure()
plt.plot(k_vinc, Pnl_v[z_idx, :], label='vinc')
plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(k_vinc, dP_dk[z_idx, :], label='dP_dk_v')
plt.legend()
plt.xscale('log')
plt.yscale('log')




z_1 = 2.5
z_2 = 2.491722





# test interpolation: take 10% different values in k and in z
# ! why does it not give an interpolation error (I'm out of range both for z and for k...) - interp1d gives an error!
# ad in k:
k_test_arr = k_G1 + k_G1 * 0.01
k_test_arr = k_test_arr[::5]  # reduce the number of elements in the sample to better see the interpolated array
# k_test_arr[-1] = k[-1] # remove last element so as not to be above interpolation range (although it doesn't complain...)
z_test_arr = z_G1 + z_G1 * 0.01
# z_test_arr[-1] = z[-1]

# ! attention: the function is G1_funct(z, k), while the array is G1[k, z]
# notice that the call to G1_funct(float, array) is a 1-dim array, I'm already projecting the function
# when passing a number (float) instead of an array - same goes for (array, float)

# BUGFIXED:
# G1_funct(z[z_bin], k_test_arr).shape = (65, 1), while
# G1_funct(z_test_arr, k[k_bin]).shape = (5,), for some reason... It's sort of the same shape, in the end


# plot vs k: original and interpolated
plt.figure()
plt.plot(k_G1, G1[:, z_bin], label=f'original, z = {z_G1[z_bin]}')
plt.plot(k_test_arr, G1_funct(z_G1[z_bin], k_test_arr), '.', label=f'interp, z = {z_G1[z_bin]}')
plt.legend()
plt.grid()
plt.xlabel(r'$k$ [$h/Mpc$]')
plt.ylabel(r'$G_1(k,z=%.2f)$' % z_G1[z_bin])

# plot vs z: original and interpolated
plt.figure()
plt.plot(z_G1, G1[k_bin, :], label=f'original, k = {k_G1[k_bin]}')
plt.plot(z_test_arr, G1_funct(z_test_arr, k_G1[k_bin]), '.', label=f'interp, k = {k_G1[k_bin]}')
plt.legend()
plt.grid()
plt.xlabel(r'$z$')
plt.ylabel(r'$G_1(k=%.2f,z)$' % k_G1[k_bin])
