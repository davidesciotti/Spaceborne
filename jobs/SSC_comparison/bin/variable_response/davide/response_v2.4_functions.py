import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from classy import Class
import numpy as np
from scipy.interpolate import interp2d, interp1d
from astropy.cosmology import w0waCDM
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent.parent.parent
job_path = Path.cwd().parent.parent.parent

sys.path.append(str(project_path / 'lib'))
import my_module as mm

matplotlib.use('Agg')

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
def G1_extrap_original(z, k, G1_funct):
    # ! is the reshaping correct?
    # TODO I'm here, finish testing this!
    result = B1 + (G1_funct(z, k_max_G1).reshape(z.size, 1) - B1) * (k / k_max_G1) ** (- 1 / 2)
    # result = B1 + (G1_funct(z, k_max_G1) - B1) * (k / k_max_G1) ** (- 1 / 2)
    return result


def G1_extrap(z_array, k_array, G1_funct):
    # ! is the reshaping correct?
    # TODO I'm here, finish testing this!

    result = np.zeros((z_array.size, k_array.size))
    for zi, z in enumerate(z_array):
        for ki, k in enumerate(k_array):
            result[zi, ki] = B1 + (G1_funct(z, k_max_G1) - B1) * (k / k_max_G1) ** (- 1 / 2)
    # result = B1 + (G1_funct(z, k_max_G1) - B1) * (k / k_max_G1) ** (- 1 / 2)
    return result


def G1_tot_funct(z, k, G1_funct, G1_extrap):
    """
    G1 is equal to:
    * 26/21 for k < k_fund
    * G1 from Alex's table for k_fund < k < k_max
    * G1 from fEq. (2.7)  for k > k_max
    """

    # find indices for the various thresholds
    k_low_indices = np.where(k <= k_fund)[0]
    k_mid_indices = np.where((k_fund < k) & (k <= k_max_G1))[0]
    k_high_indices = np.where(k > k_max_G1)[0]

    # fill the 3 arrays
    low = np.zeros((z.size, k_low_indices.size))
    low.fill(26 / 21)
    mid = G1_funct(z, k[k_mid_indices]).T
    high = G1_extrap(z, k[k_high_indices], G1_funct)

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


def calculate_power(cosmo, cosmo_par_dict, z_array, k_array, scaled_by_h=True, Pk_kind='nonlinear'):

    if scaled_by_h:
        k_scale = cosmo.h()
        Pk_scale = cosmo.h() ** 3
    else:
        k_scale = 1.
        Pk_scale = 1.

    num_k = k_array.size

    if Pk_kind == 'nonlinear':
        classy_Pk = cosmo.pk
    elif Pk_kind == 'linear':
        classy_Pk = cosmo.pk_lin

    Pk = np.zeros((len(z_array), num_k))
    for zi, z in enumerate(z_array):
        for ki, k in enumerate(k_array):
            # the argument of classy_Pk must be in units of 1/Mpc
            Pk[zi, ki] = classy_Pk(k * k_scale, z) * Pk_scale

    return k_array, Pk


def get_external_Pk(whos_Pk='vincenzo', Pk_kind='nonlinear', scaled_by_h=True):
    if whos_Pk == 'vincenzo':
        filename = 'PnlFid.dat'
        z_column = 1
        k_column = 0  # in [1/Mpc]
        Pnl_column = 2  # in [Mpc^3]
        Plin_column = 3  # in [Mpc^3]

    elif whos_Pk == 'stefano':
        filename = 'pkz-Fiducial.txt'
        z_column = 0
        k_column = 1  # in [h/Mpc]
        Pnl_column = 3  # in [Mpc^3/h^3]
        Plin_column = 2  # in [Mpc^3/h^3]

    if Pk_kind == 'linear':
        Pk_column = Plin_column
    elif Pk_kind == 'nonlinear':
        Pk_column = Pnl_column
    else:
        raise ValueError(f'Pk_kind must be either "linear" or "nonlinear"')

    Pkfile = np.genfromtxt(job_path / f'input/variable_response/{filename}')
    z_array = np.unique(Pkfile[:, z_column])
    k_array = np.unique(Pkfile[:, k_column])
    Pk = Pkfile[:, Pk_column].reshape(z_array.size, k_array.size)  # / h ** 3

    if whos_Pk == 'vincenzo':
        k_array = 10 ** k_array
        Pk = 10 ** Pk

    # h scaling
    if scaled_by_h is True:  # i.e. if you want k [h/Mpc], and P(k,z) [Mpc^3/h^3]
        if whos_Pk == 'vincenzo':
            k_array /= h
            Pk *= h ** 3
    elif scaled_by_h is False:
        if whos_Pk == 'stefano':  # i.e. if you want k [1/Mpc], and P(k,z) [Mpc^3]
            k_array *= h
            Pk /= h ** 3

    # flip, the redshift array is ordered from high to low
    Pk = np.flip(Pk, axis=0)

    return z_array, k_array, Pk


###############################################################################
###############################################################################
###############################################################################

cosmo_par_dict = {'Omega_b': 0.05,
                  'Omega_cdm': 0.27,
                  'n_s': 0.96,
                  'A_s': 2.1265e-9,
                  'h': 0.67,
                  'output': 'mPk',
                  'z_pk': '0, 0.5, 1, 2, 3',
                  'non linear': 'halofit'}

h = cosmo_par_dict['h']  # for conversions

# get growth only values
G1 = np.genfromtxt(project_path / f'config/common_data/alex_response/Resp_G1_fromsims.dat')

# take k and z values (the latter from the header)
# these are all in [h/Mpc]
k_G1 = G1[:, 0]
k_max_G1 = k_G1[-1]
k_fund = 0.012
z_G1 = np.array((0.00, 0.50, 1.00, 2.00, 3.00))
B1 = -0.75

# options for the Pk
z_min = 0.
z_max = 2.5
z_num = 303

k_min = 1e-5
k_max = 10
k_num = 800

# remove k column
G1 = G1[:, 1:]

# interpolate G1; attention: the function is G1_funct(z, k), while the array is G1[k, z]
G1_funct = interp2d(x=z_G1, y=k_G1, z=G1, kind='linear')

# ! options
scaled_by_h = True
whos_PS = 'CLASS'
Pk_kind = 'nonlinear'

# checks
# assert scaled_by_h, 'k and Pk must be in h units (at least, to reproduce Alex\'s results)'
assert whos_PS in ('vincenzo', 'stefano', 'CLASS'), 'whos_PS must be either "vincenzo", "stefano" or "CLASS"'
assert Pk_kind == 'nonlinear', 'Pk_kind must be "nonlinear"'

# set kmax in the right units
if scaled_by_h:
    cosmo_par_dict['P_k_max_h/Mpc'] = k_max
    x_label = '$k \\, [h/Mpc]$'

else:
    cosmo_par_dict['P_k_max_1/Mpc'] = k_max
    x_label = '$k \\, [1/Mpc]$'

# instantiate and initialize Class object
cosmo = Class()
cosmo.set(cosmo_par_dict)
cosmo.compute()

# get k and P(k,z)
if whos_PS in ('vincenzo', 'stefano'):
    z_array, k_array, Pk = get_external_Pk(whos_Pk=whos_PS, Pk_kind=Pk_kind, scaled_by_h=scaled_by_h)
elif whos_PS == 'CLASS':
    z_array = np.linspace(z_min, z_max, z_num)
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), k_num)
    k_array, Pk = calculate_power(cosmo, cosmo_par_dict, z_array, k_array, scaled_by_h=scaled_by_h, Pk_kind=Pk_kind)
else:
    raise ValueError('whos_PS must be either "vincenzo", "stefano" or "CLASS"')


# take the derivative and compute log derivative
# https://bdiemer.bitbucket.io/colossus/_static/tutorial_cosmology.html dedrivative of matter Pk, just as rough reference
dP_dk = np.gradient(Pk, k_array, axis=1)
dlogPk_dlogk = k_array / Pk * dP_dk

# compute response
R1_mm = 1 - 1 / 3 * k_array / Pk * dP_dk + G1_tot_funct(z_array, k_array, G1_funct, G1_extrap)  # incl. extrapolation
# R1_mm = 1 - 1 / 3 * k / Pnl * dP_dk + G1_funct(z, k).T  # doesn't incl. extrapolation


# reproduce Alex's plot
z_max = 1.8  # from the figure in the paper
z_max_idx = np.argmin(np.abs(z_array - z_max))
z_reduced = z_array[:z_max_idx + 1]

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
    plt.plot(k_array, R1_mm[z_idx, :], color=cmap(norm(z_val)), lw=lw)

plt.colorbar(s_m)
plt.xscale('log')
plt.xlabel(x_label)
plt.ylabel('$R_1^{mm}(k, z)$')
plt.axvline(x=k_max_G1, color='k', ls='--', lw=lw)
plt.xlim(1e-2, 1e1)
plt.ylim(0.5, 4)
plt.grid()
plt.show()





# ! test extrapolation
tot = G1_tot_funct(z, k, G1_funct, G1_extrap)
k_low = k[np.where(k <= k_max_G1)[0]]
k_high = k[np.where(k > k_max_G1)[0]]
G1_low = G1_funct(z, k_low).T
G1_high = G1_extrap(z, k_high, G1_funct)

z_val = 0  # from the figure in the paper
z_idx = np.argmin(np.abs(z - z_val))

plt.plot(k_low, G1_funct(z[z_idx], k_low), label='G1_funct')
plt.plot(k_high, G1_extrap(z, k_high, G1_funct)[z_idx, :], label='G1_extrap')
plt.plot(k, G1_tot_funct(z, k, G1_funct, G1_extrap)[z_idx, :], '--', label='G1_tot_funct')
plt.plot(k, (k / k_max_G1) ** (- 1 / 2), '--', label='(k / k_max_G1) ** (- 1 / 2)')
plt.axvline(k_max_G1, color='k', linestyle='--', label='k_max_G1', lw=1)
plt.axhline(B1, color='k', linestyle='--', label='B1', lw=1)

plt.plot(k, (k / Pnl * dP_dk)[z_idx, :], label='d log Pk / d log k')
#
# plt.plot(k, (1 - 1 / 3 * k / Pnl * dP_dk + G1_tot_funct(z, k, G1_funct, G1_extrap))[z_idx, :],
#          label='2nd addendum + G1')
plt.plot(k, (1 - 1 / 3 * k / Pnl * dP_dk)[z_idx, :], label='1 - 1 / 3 * k / Pnl * dP_dk')

plt.legend()
plt.grid()
plt.xscale('log')
plt.xlim(1e-2, 1e1)
plt.ylim(0.5, 4)
# ! end test extrapolation


Oc0 = cosmo_par_dict['Omega_cdm']
Ob0 = cosmo_par_dict['Omega_b']
Om0 = Oc0 + Ob0
Ode0 = 1 - Om0

cosmo_astropy = w0waCDM(H0=h * 100, Om0=Om0, Ode0=Ode0, w0=-1.0, wa=0.0, Neff=3.04, m_nu=0.06, Ob0=Ob0)


def k_limber(z, ell, cosmo_astropy, scaled_by_h):
    # astropy gives values in Mpg, I need to convert to Mpc/h
    comoving_distance = cosmo_astropy.comoving_distance(z).value  # in Mpc

    if scaled_by_h:
        comoving_distance /= h

    k_ell = (ell + 0.5) / comoving_distance
    return k_ell


#  compute ell values
ell_min = 10
ell_max_WL = 5000
nbl = 30
ell_WL = np.logspace(np.log10(ell_min), np.log10(ell_max_WL), nbl + 1)  # WL
ell_WL = (ell_WL[1:] + ell_WL[:-1]) / 2

z_test = 0.3
ell_test = ell_WL[9]
kl_array = k_limber(z_test, ell=ell_test, cosmo_astropy=cosmo_astropy, scaled_by_h=scaled_by_h)

# populate Rmm and Pk arrays by computing in k = k_limber
Rmm_WL = np.zeros((nbl, len(z)))

assert 1 > 2

# now project the responses
zbins = 10
for i in range(zbins):
    for j in range(zbins):
        for zi, zval in enumerate(z):
            for kli, klval in enumerate(kl):  # k_limber
                integrand[i, j, zi] = W_A[i, j, zi] * W_B[i, j, zi] * R1_mm[zi, kli] * Pk[zi, kli]

# integrate over z with simpson's rule
integral = np.zeros((zbins, zbins))
for i in range(zbins):
    for j in range(zbins):
        integral[i, j] = simps(integrand[i, j, :], z)

integral = np.trapz(integrand, kl, axis=2)
