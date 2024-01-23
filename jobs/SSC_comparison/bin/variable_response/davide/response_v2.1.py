import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from classy import Class
import numpy as np
from scipy.interpolate import interp2d, interp1d
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent.parent.parent
job_path = Path.cwd().parent.parent.parent

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
          'figure.figsize': (12, 8)
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()


###############################################################################
###############################################################################
###############################################################################

# extrapolate according to Eq. (2.7) in Alex's paper:
def G1_extrap(z, k, G1_funct):
    # ! is the reshaping correct?
    result = B1 + (G1_funct(z, k_max_G1).reshape(z.size, 1) - B1) * (k / k_max_G1) ** (-1 / 2)
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
    k_mid_idx = np.where((k_fund < k) & (k <= k_max_G1))[0]
    k_high_idx = np.where(k > k_max_G1)[0]

    # fill the 3 arrays
    low = np.zeros((z.size, k_low_idx.size))
    low.fill(26 / 21)
    mid = G1_funct(z, k[k_mid_idx]).T
    high = G1_extrap(z, k[k_high_idx], G1_funct)

    # concatenate the 3 arrays over the columns, i.e. the k values
    return np.concatenate((low, mid, high), axis=1)


def Pk_with_classy(cosmo, h_units, z_array, k_array):
    # TODO understand units of k_array. This function as is is works with k in 1/Mpc

    # Call these for the nonlinear and linear matter power spectra
    Pnonlin = np.zeros((len(z_array), len(k_array)))
    Plin = np.zeros((len(z_array), len(k_array)))
    for z_idx, z_val in enumerate(z_array):
        # Plin[z_idx, :] = np.array([cosmo.pk_lin(ki, z_val) for ki in k_array])
        Pnonlin[z_idx, :] = np.array([cosmo.pk(ki, z_val) for ki in k_array])

    # NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
    # to use in the toolkit. To do this you would do:
    if h_units:
        k_array /= h
        Plin *= h ** 3
        Pnonlin *= h ** 3

    return k_array, Pnonlin


def Pk_sylv_original(cosmo, k, z):
    pk = np.zeros(k.size)
    for kk in range(k.size):
        pk[kk] = cosmo.pk(k[kk] * cosmo.h(), z) * cosmo.h() ** 3
    return pk


def Pk_sylv(cosmo, k, z):
    pk = np.zeros(k.size)
    for kk in range(k.size):
        pk[kk] = cosmo.pk(k[kk], z) / cosmo.h() ** 3
        pk[kk] = cosmo.pk(k[kk] * cosmo.h(), z) / cosmo.h() ** 3
    return pk


def calculate_power(cosmo, cosmo_par_dict,  k_min, k_max, z=0, num_k=500, scaled_by_h=True):
    """
    Calculate the power spectrum P(k,z) over the range k_min <= k <= k_max.
    from https://python.hotexamples.com/it/examples/classy/Class/pk/python-class-pk-method-examples.html
    """

    # if scaled_by_h:
    #     cosmo_par_dict['P_k_max_h/Mpc'] = k_max
    # else:
    #     cosmo_par_dict['P_k_max_1/Mpc'] = k_max
    #
    # # update the P_k_max... parameter
    # cosmo.set(cosmo_par_dict)
    # cosmo.compute()

    if scaled_by_h:
        k_scale = cosmo.h()
        Pk_scale = cosmo.h() ** 3
    else:
        k_scale = 1.
        Pk_scale = 1.

    result = np.empty((num_k,), dtype=[('k', float), ('Pk', float)])
    result['k'][:] = np.logspace(np.log10(k_min), np.log10(k_max), num_k)
    for i, k in enumerate(result['k']):
        result['Pk'][i] = cosmo.pk(k * k_scale, z) * Pk_scale

    # this causes an error!!!
    # cosmo.struct_cleanup()
    # cosmo.empty()

    return result

def calculate_power_2(cosmo, cosmo_par_dict, k_array, z_array, scaled_by_h=True):

    if scaled_by_h:
        k_scale = cosmo.h()
        Pk_scale = cosmo.h() ** 3
    else:
        k_scale = 1.
        Pk_scale = 1.

    num_k = k_array.size
    k_min = k_array[0]
    k_max = k_array[-1]

    Pnl = np.zeros((len(z_array), num_k))
    for zi, z in enumerate(z_array):
        for ki, k in enumerate(k_array):
            Pnl[zi, ki] = cosmo.pk(k * k_scale, z) * Pk_scale

    return Pnl


###############################################################################
###############################################################################
###############################################################################

cosmo_par_dict = {'Omega_b': 0.05,
                  'Omega_cdm': 0.27,
                  'n_s': 0.96,
                  'A_s': 2.1265e-9,
                  'h': 0.67,
                  'output': 'mPk',
                  # 'P_k_max_h/Mpc': 20,
                  'z_pk': '0, 0.5, 1, 2, 3',
                  'non linear': 'halofit'}

# instantiate and initialize Class object
cosmo = Class()
cosmo.set(cosmo_par_dict)
cosmo.compute()

h = cosmo_par_dict['h']  # for conversions

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
G1_funct = interp2d(x=z_G1, y=k_G1, z=G1, kind='linear')

# k_idx = 0
# z_test = np.array((0.02, 0.57, 1.50, 2.50, 2.99))
#
# k_test = k_G1[k_idx]
# plt.plot(z_G1, G1[k_idx, :], label=f'original')
# plt.plot(z_test, G1_funct(z_test, k_test), label=f'interpolated')

# assert 1 > 2


k_max_G1 = k_G1[-1]
k_fund = 0.012  # h/Mpc

# * options
h_units = True  # ! BUG ALERT, WHICH ONE SHOULD I USE?
which_PS = 'stefano'
# * end options

# assert h_units, 'k and Pk must be in h units (at least, to reproduce Alex\'s results)'

# vincenzo's input Pk and stefano's reshaping method
# input is in k [1/Mpc], and P(k,z) [Mpc^3]
PkFILE_vinc = np.genfromtxt(job_path / f'input/variable_response/PnlFid.dat')
k_vinc = 10 ** (np.unique(PkFILE_vinc[:, 0]))  # in [1/Mpc]
z_vinc = np.unique(PkFILE_vinc[:, 1])
k_points_v = len(k_vinc)
z_points_v = len(z_vinc)
Pnl_vinc = (10 ** PkFILE_vinc[:, 2]).reshape(z_points_v, k_points_v)  # in [Mpc^3]
Plin_vinc = (10 ** PkFILE_vinc[:, 3]).reshape(z_points_v, k_points_v)  # in [Mpc^3]

# stefano's input Pk and reshaping method
# input is in k [h/Mpc], and P(k,z) [Mpc^3/h^3]
PkFILE_stef = np.genfromtxt(job_path / f'input/variable_response/pkz-Fiducial.txt')
z_stef = np.unique(PkFILE_stef[:, 0])
k_points_s = int(len(PkFILE_stef[:, 2]) / len(z_stef))
k_stef = PkFILE_stef[:k_points_s, 1]  # * h
z_points_s = len(z_stef)
Pnl_stef = PkFILE_stef[:, 3].reshape(z_points_s, k_points_s)  # / h ** 3
Pnl_stef = np.flip(Pnl_stef, axis=0)

if h_units:  # i.e. if you want k [h/Mpc], and P(k,z) [Mpc^3/h^3]
    k_vinc /= h
    Pnl_vinc *= h ** 3
    x_label = '$k \\, [h/Mpc]$'
else:  # i.e. if you want k [1/Mpc], and P(k,z) [Mpc^3]
    k_stef *= h
    Pnl_stef /= h ** 3
    x_label = '$k \\, [1/Mpc]$'

if which_PS == 'vincenzo':
    Pnl = Pnl_vinc
    k = k_vinc
    z = z_vinc
elif which_PS == 'stefano':
    Pnl = Pnl_stef
    k = k_stef
    z = z_stef
# elif which_PS == 'CLASS':
    # TODO
    # k_array = np.arange(1e-5, 30, 0.01)  # in [1/Mpc]
    # z = z_stef
    # k, Pnl = Pk_with_classy(cosmo_par_dict, h_units, z_array=z, k_array=k_array)  # both in h units

else:
    raise ValueError('which_PS must be "vincenzo", "stefano" or "CLASS"')

k_max_classy = 10
scaled_by_h = True
if scaled_by_h or h_units:
    cosmo_par_dict['P_k_max_h/Mpc'] = k_max_classy
else:
    cosmo_par_dict['P_k_max_1/Mpc'] = k_max_classy


# 1. compute with classy from cluster_toolkit
num_k = 1000
k_classy = np.logspace(np.log10(1e-5), np.log10(k_max_classy), num_k)  # in [1/Mpc]
k_classy, Pnl_classy = Pk_with_classy(cosmo, h_units, z_array=z, k_array=k_classy)  # both in h units

# 2. compute from online example
Pk_reference = calculate_power(cosmo, cosmo_par_dict, k_min=1e-5, k_max=k_max_classy, z=0, num_k=num_k, scaled_by_h=scaled_by_h)
Pk_reference_2 = calculate_power_2(cosmo, cosmo_par_dict, k_array=k_classy, z_array=z, scaled_by_h=True)


z_val_test = 1.5
# find z value closer to the one you want
z_idx = np.argmin(np.abs(z - z_val_test))

plt.plot(k, Pnl[z_idx, :], '--', label=f'{which_PS} h_units={h_units}')  # stefano/vincenzo
plt.plot(k_G1, Pk_sylv_original(cosmo, k_G1, z_val_test), '--', label='sylvain original')  # sylvain original
plt.plot(k_classy, Pnl_classy[z_idx, :], '--', label=f'cluster toolkit, h_units={h_units}')  # cluster toolkit
plt.plot(Pk_reference['k'], Pk_reference['Pk'], '--', label=f'online example, scaled_by_h={scaled_by_h}')  # online example
plt.plot(k_classy, Pk_reference_2[z_idx, :], '.', label=f'online example modified, scaled_by_h={scaled_by_h}')  # online example 2


plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

Pnl = Pk_reference_2
k = k_classy

assert 1 > 2














# take the derivative w.r.t. k - but this k is log spaced! the derivative sucks at high k
Pnl_interp = interp2d(x=k, y=z, z=Pnl, kind='linear')
k_deriv = np.arange(k[0], k[-1], 0.01)
Pnl_toderiv = Pnl_interp(k_deriv, z)
dP_dk = np.gradient(Pnl_toderiv, k_deriv, axis=1)

# now re-interpolate the derivative to the original k values
dP_dk_interp = interp2d(x=k_deriv, y=z, z=dP_dk, kind='linear')
dP_dk = dP_dk_interp(k, z)

dlogPk_dlogk = k / Pnl * dP_dk
np.save('dlogPk_dlogk.npy', dlogPk_dlogk)
np.save('k_dav.npy', k)
np.save('z_dav.npy', z)

np.save('Pnl.npy', Pnl)

# assert 1 > 2
# # old method:
dP_dk_old = np.gradient(Pnl, k, axis=1)
#
# z_idx = 0
# plt.plot(k, dP_dk[z_idx, :], label='new')
# plt.plot(k, dP_dk_old[z_idx, :], label='old')


# compute response
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
cmap = matplotlib.cm.jet
lw = 1

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
s_m.set_array([])

plt.figure()
# the colors are chosen by calling the ScalarMappable that was initialised with c_m and norm
for z_idx, z_val in enumerate(z_reduced):
    plt.plot(k, R1_mm[z_idx, :], color=cmap(norm(z_val)), lw=lw)

plt.colorbar(s_m)
plt.xscale('log')
plt.xlabel(x_label)
plt.ylabel('$R_1^{mm}(k, z)$')
plt.axvline(x=k_max_G1, color='k', ls='--', lw=lw)
plt.xlim(1e-2, 1e1)
plt.ylim(0.5, 4)
plt.grid()
plt.show()

# https://bdiemer.bitbucket.io/colossus/_static/tutorial_cosmology.html dedrivative of matter Pk


# ! test extrapolation
# tot = G1_tot_funct(z, k, G1_funct, G1_extrap)
# k_low = k[np.where(k <= k_max_G1)[0]]
# k_high = k[np.where(k > k_max_G1)[0]]
# G1_low = G1_funct(z, k_low).T
# G1_high = G1_extrap(z, k_high, G1_funct)

# z_idx = 300
# plt.plot(k_low, G1_funct(z[z_idx], k_low), label='G1')
# plt.plot(k_high, G1_extrap(z[z_idx], k_high, G1_funct)[0, :], label='G1_extrap')
# plt.plot(k, G1_tot_funct(z[z_idx], k, G1_funct, G1_extrap)[0, :], '--', label='G1_tot')
# plt.axvline(k_max, color='k', linestyle='--', label='k_max', lw=1)

# plt.plot(k, (k / Pnl * dP_dk)[z_idx, :], label='d log Pk / d log k')
# plt.plot(k, (1 - 1 / 3 * k / Pnl * dP_dk + G1_tot_funct(z, k, G1_funct, G1_extrap))[z_idx, :],
#          label='2nd addendum + G1')

# plt.legend()
# plt.grid()
# ! end test extrapolation
