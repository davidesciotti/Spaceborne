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
B1 = 0.75

# remove k column
G1 = G1[:, 1:]

# interpolate G1
# ! attention: the function is G1_funct(z, k), while the array is G1[k, z]
G1_funct = interp2d(x=z_G1, y=k_G1, z=G1, kind='cubic')

k_max = k_G1[-1]
k_fund = 0.012  # h/Mpc


# extrapolate according to Eq. (2.7) in Alex's paper:
def G1_extrap(z, k, G1_funct):
    # ! is the reshaping correct?
    result = B1 + (G1_funct(z, k_max).reshape(z.size, 1) - B1) * (k / k_max) ** (-1 / 2)
    return result


def G1_tot_funct(z, k, G1_funct, G1_extrap):
    """
    G1 is equal to:
    * 26/21 for k < k_fund
    * G1 from Alex's table for k_fund < k < k_max
    * G1 from fEq. (2.7)  for k > k_max
    """

    # find indices for the various thresholds
    k_low_idx = np.where(k <= k_fund)[0]
    k_mid_idx = np.where((k_fund < k) & (k <= k_max))[0]
    k_high_idx = np.where(k > k_max)[0]

    # fill the 3 arrays
    low = np.zeros((z.size, k_low_idx.size))
    low.fill(26/21)
    mid = G1_funct(z, k[k_mid_idx]).T
    high = G1_extrap(z, k[k_high_idx], G1_funct)

    # concatenate the 3 arrays over the columns, i.e. the k values
    return np.concatenate((low, mid, high), axis=1)


# * options
h_units = True  # ! BUG ALERT, WHICH ONE SHOULD I USE?
which_PS = 'stefano'
# * end options

# vincenzo's input Pk and stefano's reshaping method
# input is in k [1/Mpc], and P(k,z) [Mpc^3]
PkFILE_vinc = np.genfromtxt(job_path / f'input/variable_response/PnlFid.dat')
k_vinc = 10 ** (np.unique(PkFILE_vinc[:, 0]))  # / h
k_points_v = len(k_vinc)
z_vinc = np.unique(PkFILE_vinc[:, 1])
z_points_v = len(z_vinc)
Pnl_vinc = (10 ** PkFILE_vinc[:, 2]).reshape(z_points_v, k_points_v)  # * (h ** 3)

# stefano's input Pk and reshaping method
# input is in k [h/Mpc], and P(k,z) [Mpc^3/h^3]
PkFILE_stef = np.genfromtxt(job_path / f'input/variable_response/pkz-Fiducial.txt')
z_stef = np.unique(PkFILE_stef[:, 0])
k_points_s = int(len(PkFILE_stef[:, 2]) / len(z_stef))
k_stef = PkFILE_stef[:k_points_s, 1]  # * h
z_points_s = len(z_stef)
Pnl_stef = PkFILE_stef[:, 3].reshape(z_points_s, k_points_s)  # / h ** 3

if h_units:  # i.e. if you want k [h/Mpc], and P(k,z) [Mpc^3/h^3]
    k_vinc /= h
    Pnl_vinc *= h ** 3
else:  # i.e. if you want k [1/Mpc], and P(k,z) [Mpc^3]
    k_stef *= h
    Pnl_stef /= h ** 3

if which_PS == 'vincenzo':
    Pnl = Pnl_vinc
    k = k_vinc
    z = z_vinc
elif which_PS == 'stefano':
    Pnl = Pnl_stef
    k = k_stef
    z = z_stef
else:
    raise ValueError('which_PS must be "vincenzo" or "stefano"')

# take the derivative w.r.t. k
# ! BUG ALERT, IS THIS THE CORRECT WAY TO DO IT?
dP_dk = np.gradient(Pnl, k, axis=1)

R1_mm = 1 - 1 / 3 * k / Pnl * dP_dk + G1_tot_funct(z, k, G1_funct, G1_extrap)  # incl. extrapolation
# R1_mm = 1 - 1 / 3 * k / Pnl * dP_dk + G1_funct(z, k).T  # doesn't incl. extrapolation



# reproduce Alex's plot
z_max = 1.8  # from the figure in the paper
z_max_idx = np.argmin(np.abs(z - z_max))
z_reduced = z[:z_max_idx + 1]

# from https://stackoverflow.com/questions/26545897/drawing-a-colorbar-aside-a-line-plot-using-matplotlib
# norm is a class which, when called, can normalize data into the [0.0, 1.0] interval.
norm = matplotlib.colors.Normalize(
    vmin=np.min(z_reduced),
    vmax=np.max(z_reduced))

# choose a colormap and a line width
c_m = matplotlib.cm.jet
lw = 1

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

plt.figure()
# the colors are chosen by calling the ScalarMappable that was initialised with c_m and norm
for z_idx, z_val in enumerate(z_reduced):
    plt.plot(k, R1_mm[z_idx, :], color=c_m(norm(z_val)), lw=lw)

if h_units:
    x_label = '$k \\, [h/Mpc]$'
else:
    x_label = '$k \\, [1/Mpc]$'

plt.colorbar(s_m)
plt.grid()
plt.xscale('log')
plt.xlabel(x_label)
plt.axvline(x=k_max, color='k', ls='--', lw=lw)
plt.xlim(1e-2, 1e1)
plt.ylim(0.5, 4)
plt.show()

# ! test extrapolation
tot = G1_tot_funct(z, k, G1_funct, G1_extrap)
k_low = k[np.where(k <= k_max)[0]]
k_high = k[np.where(k > k_max)[0]]
G1_low = G1_funct(z, k_low).T
G1_high = G1_extrap(z, k_high, G1_funct)

print(np.allclose(tot[:, np.where(k <= k_max)[0]], G1_low))
print(np.allclose(tot[:, np.where(k > k_max)[0]], G1_high))

z_idx = 1
plt.plot(k_low, G1_funct(z[z_idx], k_low), label='G1')
plt.plot(k_high, G1_extrap(z[z_idx], k_high, G1_funct)[0, :], label='G1_extrap')
plt.plot(k, G1_tot_funct(z[z_idx], k, G1_funct, G1_extrap)[0, :], '--', label='G1_tot')
plt.axvline(k_max, color='k', linestyle='--', label='k_max', lw=1)

plt.plot(k, (1 - 1 / 3 * k / Pnl * dP_dk)[z_idx, :], label='log derivative')

plt.legend()
plt.grid()
# ! end test extrapolation

