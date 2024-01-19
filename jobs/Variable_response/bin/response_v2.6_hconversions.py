import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from classy import Class
import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import interp2d, interp1d
from astropy.cosmology import w0waCDM
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
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


def G1_tot_funct(z, k_array, G1_funct, G1_extrap):
    """
    G1 is equal to:
    * 26/21 for k < k_fund
    * G1 from Alex's table for k_fund < k < k_max
    * G1 from fEq. (2.7)  for k > k_max
    """

    # find indices for the various thresholds
    k_low_indices = np.where(k_array <= k_fund)[0]
    k_mid_indices = np.where((k_fund < k_array) & (k_array <= k_max_G1))[0]
    k_high_indices = np.where(k_array > k_max_G1)[0]

    # fill the 3 arrays
    low = np.zeros((z.size, k_low_indices.size))
    low.fill(26 / 21)
    mid = G1_funct(z, k_array[k_mid_indices]).T
    high = G1_extrap(z, k_array[k_high_indices], G1_funct)

    # concatenate the 3 arrays over the columns, i.e. the k values
    return np.concatenate((low, mid, high), axis=1)


def Pk_with_classy_clustertlkt(cosmo, z_array, k_array, use_h_units, Pk_kind='nonlinear', argument_type='arrays'):
    # This function wants as input k in 1/Mpc

    if Pk_kind == 'nonlinear':
        classy_Pk = cosmo.pk
    elif Pk_kind == 'linear':
        classy_Pk = cosmo.pk_lin
    else:
        raise ValueError('Pk_kind must be either "nonlinear" or "linear"')

    if argument_type == 'scalar':
        Pk = classy_Pk(k_array, z_array)  # k_array and z_array are not arrays, but scalars!

    elif argument_type == 'arrays':
        num_k = k_array.size

        Pk = np.zeros((len(z_array), num_k))
        for z_idx, z_val in enumerate(z_array):
            Pk[z_idx, :] = np.array([classy_Pk(ki, z_val) for ki in k_array])

    # NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
    # to use in the toolkit. To do this you would do:
    if use_h_units:
        k_array /= h
        Pk *= h ** 3

    # return also k_array, to have it in the correct h scaling
    return k_array, Pk


def calculate_power(cosmo, z_array, k_array, use_h_units=True, Pk_kind='nonlinear', argument_type='arrays'):
    if use_h_units:
        k_scale = cosmo.h()
        Pk_scale = cosmo.h() ** 3
    else:
        k_scale = 1.
        Pk_scale = 1.

    if Pk_kind == 'nonlinear':
        classy_Pk = cosmo.pk
    elif Pk_kind == 'linear':
        classy_Pk = cosmo.pk_lin
    else:
        raise ValueError('Pk_kind must be either "nonlinear" or "linear"')

    # if z and k are not arrays, return scalar output
    if argument_type == 'scalar':
        Pk = classy_Pk(k_array * k_scale, z_array) * Pk_scale  # k_array and z_array are not arrays, but scalars!

    elif argument_type == 'arrays':
        num_k = k_array.size

        Pk = np.zeros((len(z_array), num_k))
        for zi, z in enumerate(z_array):
            for ki, k in enumerate(k_array):
                # the argument of classy_Pk must be in units of 1/Mpc?
                Pk[zi, ki] = classy_Pk(k * k_scale, z) * Pk_scale

    else:
        raise ValueError('argument_type must be either "scalar" or "arrays"')

    return Pk


def get_external_Pk(whos_Pk='vincenzo', Pk_kind='nonlinear', use_h_units=True):
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
    else:
        raise ValueError('whos_Pk must be either stefano or vincenzo')

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
    if use_h_units is True:  # i.e. if you want k [h/Mpc], and P(k,z) [Mpc^3/h^3]
        if whos_Pk == 'vincenzo':
            k_array /= h
            Pk *= h ** 3
    elif use_h_units is False:
        if whos_Pk == 'stefano':  # i.e. if you want k [1/Mpc], and P(k,z) [Mpc^3]
            k_array *= h
            Pk /= h ** 3

    # flip, the redshift array is ordered from high to low
    Pk = np.flip(Pk, axis=0)

    return z_array, k_array, Pk


def k_limber(z, ell, cosmo_astropy, use_h_units):
    # astropy gives values in Mpc, so I call astropy_comoving_distance to have the correct values in both cases (h units
    # or not)
    comoving_distance = astropy_comoving_distance(z, cosmo_astropy, use_h_units)
    k_ell = (ell + 0.5) / comoving_distance
    return k_ell


def astropy_comoving_distance(z, cosmo_astropy, use_h_units):
    if use_h_units:
        return cosmo_astropy.comoving_distance(z).value * h  # Mpc/h
    else:
        return cosmo_astropy.comoving_distance(z).value  # Mpc


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
c = 299792.458  # km/s
H0 = h * 100  # km/(s*Mpc)

# get growth only values - DIMENSIONLESS
G1 = np.genfromtxt(project_path / f'input/alex_response/Resp_G1_fromsims.dat')

# take k and z values (the latter from the header)
# these are all in [h/Mpc]
k_G1 = G1[:, 0]
k_max_G1 = k_G1[-1]
k_fund = 0.012
z_G1 = np.array((0.00, 0.50, 1.00, 2.00, 3.00))
B1 = -0.75

# remove k column
G1 = G1[:, 1:]

# options for the Pk
# z_max = 2.5
z_min, z_max, z_num = 1e-3, 3., 303
zbins = 10

k_min, k_max, k_num = 1e-5, 20, 800
k_max_classy = 50

# ! options
use_h_units = True
whos_PS = 'CLASS_clustertlkt'
Pk_kind = 'nonlinear'
plot_Rmm = False
PySSC_kernel_convention = True  # if True, normalize the IST WFs by r(z) ** 2
save_Pk = False
# ! options

# for whos_PS in ['stefano', 'vincenzo', 'CLASS', 'CLASS_clustertlkt']:
# for use_h_units in [True, False]:

# kmax in [h/Mpc], if I understand correctly this does not set any unit; it just tells CLASS which is the maximum k
# to compute (but not necessarily to use!), and one should of course say which units he's using
cosmo_par_dict['P_k_max_h/Mpc'] = k_max_classy

# note: 'use_h_units' means whether I want everything h units. This means that e.g. if use_h_units is True,
# what already is in h units should be left untouched, and what is not should be converted. so the scaling is not
# universal, but depends on which unit is being used for the element in question.
if use_h_units:
    x_label = '$k \\, [h/Mpc]$'
    k_scale, r_scale = h, h
else:
    x_label = '$k \\, [1/Mpc]$'
    k_scale, r_scale = 1., 1.

#
# rescale - I do not scale G1 since it is only used for the interpolation below, and in the original table is in h/Mpc
# if use_h_units is False:
#     k_max_G1 *= h
#     k_fund *= h
# now there are in 1/Mpc


# interpolate G1; attention: the function is G1_funct(z, k), while the array is G1[k, z]
G1_funct = interp2d(x=z_G1, y=k_G1, z=G1, kind='linear')

# checks
# assert use_h_units, 'k and Pk must be in h units (at least, to reproduce Alex\'s results)'
assert Pk_kind == 'nonlinear', 'Pk_kind must be "nonlinear"'

# instantiate and initialize Class object
cosmo = Class()
cosmo.set(cosmo_par_dict)
cosmo.compute()

# get k and P(k,z)
if whos_PS in ['vincenzo', 'stefano']:
    z_array, k_array, Pk = get_external_Pk(whos_Pk=whos_PS, Pk_kind=Pk_kind, use_h_units=use_h_units)

elif whos_PS == 'CLASS':
    z_array = np.linspace(z_min, z_max, z_num)
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), k_num)  # this is in h/Mpc. The calculate_power function
    # takes care of the corerct h_units when computing Pk, but only returns Pk, so k_array has to be made consistent
    # by hand
    Pk = calculate_power(cosmo, z_array, k_array, use_h_units=use_h_units, Pk_kind=Pk_kind)

elif whos_PS == 'CLASS_clustertlkt':
    z_array = np.linspace(z_min, z_max, z_num)
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), k_num)  # this is in 1/Mpc. The Pk_with_classy_clustertlkt
    # function also returns k, rescaled or not
    k_array, Pk = Pk_with_classy_clustertlkt(cosmo, z_array=z_array, k_array=k_array, use_h_units=use_h_units,
                                             Pk_kind=Pk_kind)

else:
    raise ValueError('whos_PS must be either "vincenzo", "stefano", "CLASS" or "CLASS_clustertlkt"')

if save_Pk:
    np.save(job_path / f'output/Pk/Pk_kind={Pk_kind}_hunits={use_h_units}.npy', Pk)
    np.save(job_path / f'output/Pk/k_array_hunits={use_h_units}.npy', k_array)
    np.save(job_path / f'output/Pk/z_array.npy', z_array)

# take the derivative and compute log derivative
# https://bdiemer.bitbucket.io/colossus/_static/tutorial_cosmology.html dedrivative of matter Pk, just as rough reference
dP_dk = np.gradient(Pk, k_array, axis=1)
dlogPk_dlogk = k_array / Pk * dP_dk

# compute response...
R1_mm = 1 - 1 / 3 * k_array / Pk * dP_dk + G1_tot_funct(z_array, k_array, G1_funct, G1_extrap)  # incl. extrapolation
# R1_mm = 1 - 1 / 3 * k / Pnl * dP_dk + G1_funct(z, k).T  # doesn't incl. extrapolation

# test this part
R1_mm_partial = 1 - 1 / 3 * k_array / Pk * dP_dk

z_idx = 0
plt.plot(k_array, R1_mm[z_idx, :], '.')
plt.plot(k_array, G1_tot_funct(z_array, k_array, G1_funct, G1_extrap)[z_idx, :], '.', label=f'{use_h_units}')
plt.axvline(x=k_max_G1, color='k', ls='--', lw=1)

plt.legend()
plt.grid()

np.save('/Users/davide/Desktop/pyccl_responses/alex_responses/R1_mm.npy', R1_mm)
np.save('/Users/davide/Desktop/pyccl_responses/alex_responses/k_array.npy', k_array)

# ...and plot it
if plot_Rmm:
    # reproduce Alex's plot
    z_max_plot = 1.8  # from the figure in the paper
    z_max_idx = np.argmin(np.abs(z_array - z_max_plot))
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

    # plt.figure()<<
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

# assert 1 > 2
########################################### now, project the response ##################################################

# instantiate cosmology to compute comoving distance from astropy
Oc0 = cosmo_par_dict['Omega_cdm']
Ob0 = cosmo_par_dict['Omega_b']
Om0 = Oc0 + Ob0
Ode0 = 1 - Om0

# instantiate cosmo object from astropy
cosmo_astropy = w0waCDM(H0=h * 100, Om0=Om0, Ode0=Ode0, w0=-1.0, wa=0.0, Neff=3.04, m_nu=0.06, Ob0=Ob0)

# compute ell values
ell_min = 10
ell_max_WL = 5000
nbl = 30
ell_WL = np.logspace(np.log10(ell_min), np.log10(ell_max_WL), nbl + 1)  # WL
ell_WL = (ell_WL[1:] + ell_WL[:-1]) / 2

# # fill k_limber array with meshgrid
# zz, ll = np.meshgrid(z_array, ell_WL)
# kl_array_mesh = k_limber(zz, ell=ll, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)
#

# # fill k_limber array manually
# kl_array_manual = np.zeros((nbl, z_num))
# for ell_idx, ellval in enumerate(ell_WL):
#     kl_array_manual[ell_idx, :] = k_limber(z_array, ell=ellval, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)


# at low redshift and high ells, k_limber explodes: cut the z range
z_array_limber = z_array[5:]  # ! this has been found by hand, fix this!

ell_test = ell_WL[10]
kl = k_limber(z_array_limber, ell=ell_test, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)

# compute P(k_ell, z)
Pk = calculate_power(cosmo, z_array_limber, kl, use_h_units=use_h_units, Pk_kind=Pk_kind)

# compute R1(k_ell, z)

# 1. the easy way: interpolate and fill
R1_mm_interp = interp2d(k_array, z_array, R1_mm, kind='linear')

# 2. TODO the hard way: construct R1_mm function and evaluate it in kl, z
# ...

# import WF
# ! these should be in Mpc ** -1 !! include a scaling below (after removing the z column)
W_LL = np.genfromtxt(
    '/home/cosmo/davide.sciotti/data/common_data/everyones_WF_from_Gdrive/davide/nz10000/gen2022/wil_dav_IA_IST_nz10000_bia2.17.txt')
z_WF = W_LL[:, 0]
W_LL = W_LL[:, 1:]  # resmove redshift column

# convert to h/Mpc if h_units is True, else leave it as it is
if use_h_units:
    W_LL /= h

# normalize by r(z)**2 to translate them into PySSC convention:
if PySSC_kernel_convention:
    my_r_of_z = astropy_comoving_distance(z_WF, cosmo_astropy, use_h_units=use_h_units)
    W_LL /= np.repeat(my_r_of_z[:, None], zbins, axis=1) ** 2

# interpolate WF
W_interp = interp1d(z_WF, W_LL, kind='linear', axis=0)
W_LL_array = W_interp(z_array_limber).T

# ! tests
my_r_of_z = astropy_comoving_distance(z_array_limber, cosmo_astropy, use_h_units=use_h_units)

# r(z) and dr/dz in Mpc. The scaling by h is implemented below.
zofr = cosmo.z_of_r(z_array_limber)
comov_dist = zofr[0] * r_scale  # Comoving distance r(z)
dr_dz = (1 / zofr[1]) * r_scale  # Derivative dr/dz

# it should be
dV = comov_dist ** 2 * dr_dz

# TODO check again all the r_scale, k_scale and h scalings in general, I altready found 3 mistakes
# TODO does k_limber function return kl in the correct units?
# TODO recover plateau from before?

# # ! test k limber and stuff
"""
kl_arr = np.zeros((z_array_limber.size, ell_WL.size))
R_of_ell_z = np.zeros((z_array_limber.size, ell_WL.size))

for zi, zval in enumerate(z_array_limber):
    for ell_idx, ell_val in enumerate(ell_WL):
        # evaluate kl, R1(kl, z) and P(kl, z)
        kl_arr[zi, ell_idx] = k_limber(zval, ell_val, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)
        kl_val = k_limber(zval, ell_val, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)
        P_kl_z = calculate_power(cosmo, zval, kl_val, use_h_units=use_h_units, Pk_kind=Pk_kind,
                                 argument_type='scalar')
        R_of_ell_z[zi, ell_idx] = R1_mm_interp(kl_val, zval)[0]

# plot
ell_idx = 0
z_idx = 5

# plot vs ell
# plt.figure()
plt.plot(ell_WL, kl_arr[z_idx, :], label=f'kl_arr vs ell, {use_h_units}')
plt.plot(ell_WL, R_of_ell_z[z_idx, :], label=f'R_of_ell_z vs ell, {use_h_units}')

# plot vs z
# plt.figure()
# plt.plot(z_array_limber, kl_arr[:, ell_idx], label=f'kl_val vs z, {use_h_units}')
plt.legend()
"""
# ! test k limber and stuff


# plt.plot(z_array_limber, my_r_of_z, label='my_r_of_z')
# plt.plot(z_array_limber, comov_dist, '--', label='comov_dist')
# plt.plot(z_array_limber, np.gradient(comov_dist), label='np.gradient(zofr)')
# plt.plot(z_array_limber, zofr[1], label='dz_dr?')
# plt.plot(z_array_limber, np.gradient(comov_dist) / dr_dz, label='dcomov_dist')  # equal up to a factor!!
# plt.legend()

# ! end tests

# now project the responses
integrand = np.zeros((nbl, zbins, zbins, z_array_limber.size))
cl_integrand = np.zeros((nbl, zbins, zbins, z_array_limber.size))
start_time = time.time()
# for zi, zval in enumerate(z_array_limber):
#     for ell_idx, ell_val in enumerate(ell_WL):
#
#         # evaluate kl, R1mm(kl, z) and P(kl, z)
#
#         # k_limber should already be in the correct units, from the cosmo_astropy call
#         kl = k_limber(zval, ell_val, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)
#
#         R_of_ell_z = R1_mm_interp(kl, zval)[0]
#         P_kl_z = calculate_power(cosmo, zval, kl, use_h_units=use_h_units, Pk_kind=Pk_kind,
#                                  argument_type='scalar')
#
#         for i in range(zbins):
#             for j in range(zbins):
#                 # integrand[i, j, zi, ell_idx] = W_A[i, zi] * W_B[j, zi] * R1_mm_limb[zi, kli] * Pk_limb[zi, kli]
#                 integrand[ell_idx, i, j, zi] = dV[zi] * W_LL_array[i, zi] * W_LL_array[j, zi] * R_of_ell_z * P_kl_z


# integrand = np.zeros((nbl, zbins, zbins, z_array_limber.size))
start_time = time.time()
for zi, zval in enumerate(z_array_limber):
    for ell_idx, ell_val in enumerate(ell_WL):

        # evaluate kl, R1mm(kl, z) and P(kl, z)

        # k_limber should already be in the correct units, from the cosmo_astropy call
        kl = k_limber(zval, ell_val, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)

        # Pk_with_classy_clustertlkt wants in input k in 1/Mpc; so, if I'm using h units, transform kl to 1/Mpc
        # ! this is assuming that the k_limber function returns kl in the correct units
        if use_h_units:
            kl *= h

        kl, P_kl_z = Pk_with_classy_clustertlkt(cosmo, zval, kl, use_h_units, Pk_kind='nonlinear',
                                                argument_type='scalar')

        R_of_ell_z = R1_mm_interp(kl, zval)[0]

        for i in range(zbins):
            for j in range(zbins):
                # integrand[i, j, zi, ell_idx] = W_A[i, zi] * W_B[j, zi] * R1_mm_limb[zi, kli] * Pk_limb[zi, kli]
                integrand[ell_idx, i, j, zi] = dV[zi] * W_LL_array[i, zi] * W_LL_array[j, zi] * R_of_ell_z * P_kl_z
                cl_integrand[ell_idx, i, j, zi] = dV[zi] * W_LL_array[i, zi] * W_LL_array[j, zi] * P_kl_z

# plt.plot(z_array_limber, R1_mm_interp(kl, z_array_limber).T[0, :])


# integrate over z with simpson's rule
# ! is there a c/H0 factor in the integral?
integral = simps(integrand, z_array_limber, axis=-1)
cl_integral = simps(cl_integrand, z_array_limber, axis=-1)

# import Cl
# Cl_LL = np.load(job_path / 'output/cl_3D/C_LL_WLonly_3D.npy')

# finally, divide by Cl
integral = integral / cl_integral

plt.figure()
for i in range(zbins):
    j = i
    plt.plot(ell_WL, integral[:, i, j], label='$R_\ell^{%i, %i}$' % (i, j))
    plt.legend()
plt.xlabel('$\ell$')
plt.ylabel('$R_\ell^{%i, %i}$' % (i, j))
plt.grid()

print('done')

# ! tests
"""
# test that k is indeed in the correct units  
k_array = np.logspace(np.log10(k_min), np.log10(k_max), k_num)  # this is in h/Mpc.
# the k_array defined and fed into calculate_power is in h/Mpc, so if I want to use h units, I have to rescale
if use_h_units is False:
    k_array *= h

# this function also outputs k in the correect units, use it for validation
k_array_corrunits, Pk = Pk_with_classy_clustertlkt(cosmo, z_array=z_array, k_array=k_array, h_units=use_h_units, Pk_kind=Pk_kind)

# check against the k_array produced by Pk_with_classy_clustertlkt, which is rescaled as well
print(np.array_equal(k_array, k_array_corrunits))
"""
