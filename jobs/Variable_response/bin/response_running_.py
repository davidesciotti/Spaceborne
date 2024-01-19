import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from classy import Class
import numpy as np
from matplotlib import cm
from scipy.integrate import quad, simps
from scipy.interpolate import interp2d, interp1d
from astropy.cosmology import w0waCDM
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(str(project_path.parent / 'common_data/common_lib'))
sys.path.append(str(project_path.parent / 'common_data/common_config'))
sys.path.append(str(project_path.parent / 'Spaceborne/bin'))
sys.path.append(str(job_path / 'config'))

# general libraries
import my_module as mm
import cosmo_lib as csmlib

# general configurations
import ISTF_fid_params
import mpl_cfg

# job-specific congiguration
import config_variable_response as cfg

import ell_values as ell_utils

matplotlib.use('Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
markersize = ['lines.markersize']


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


# ! quad with PySSC cl formula
def dV_func(z):
    zofr = cosmo_classy.z_of_r([z])
    comov_dist = zofr[0] * r_scale  # Comoving distance r(z)
    dr_dz = (1 / zofr[1]) * r_scale  # Derivative dr/dz
    dV = comov_dist ** 2 * dr_dz
    return dV[0]


def integrand_PySSC(z, wf_A, wf_B, i, j, ell):
    kl = kl_wrap(ell=ell, z=z)
    return dV_func(z) * wf_A(z)[i] * wf_B(z)[j] * R1_mm_interp(kl, z)[0] * Pk_wrap(kl, z=z)


def R_LL_quad(wf_A, wf_B, i, j, ell):
    return quad(integrand_PySSC, z_array_limber[0], z_array_limber[-1], args=(wf_A, wf_B, i, j, ell))[0]


# ! quad with ISTF cl formula - tested for the cls, should be fine!
def integrand_ISTF_v0(z, wf_A, wf_B, i, j, ell):
    return (wf_A(z)[i] * wf_B(z)[j]) / (csmlib.E(z) * csmlib.r(z) ** 2) * \
           R1_mm_interp(kl_wrap(ell, z), z)[0] * Pk_wrap(kl_wrap(ell, z), z)


# use_h_units version
def integrand_ISTF(z, wf_A, wf_B, i, j, ell):
    return (wf_A(z)[i] * wf_B(z)[j]) / (csmlib.E(z) / r_scale *
                                        csmlib.astropy_comoving_distance(z, use_h_units=use_h_units) ** 2) * \
           R1_mm_interp(kl_wrap(ell, z), z)[0] * Pk_wrap(kl_wrap(ell, z), z)


def cl_integral(wf_A, wf_B, i, j, ell):
    return c / H0 * quad(integrand_ISTF, z_array_limber[0], z_array_limber[-1], args=(wf_A, wf_B, i, j, ell))[0]


def R1_mm_func(k, z):
    result = 1 - 1 / 3 * k / Pk_wrap(k, z) * dP_dk_func(x=k, y=z) + G1_tot_funct_scalar(k, z)
    return result[0]


def plot_Rmm_funct():
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

    plt.figure()
    # the colors are chosen by calling the ScalarMappable that was initialised with c_m and norm
    for z_idx, zval in enumerate(z_reduced):
        plt.plot(k_array, R1_mm[z_idx, :], color=cmap(norm(zval)), lw=lw)

    plt.colorbar(s_m)
    plt.xscale('log')
    plt.xlabel(x_label)
    plt.ylabel('$R_1^{mm}(k, z)$')
    plt.axvline(x=k_max_G1, color='k', ls='--', lw=lw)
    plt.xlim(1e-2, 1e1)
    plt.ylim(0.5, 4)
    plt.grid()
    plt.show()


###############################################################################
###############################################################################
###############################################################################

# 🐛🐛🐛🐛🐛🐛🐛🐛
# i think yhe problem is in R_of_kl_z
# maybe the division by C_LL is the issue though, me and Vincenzo are not using the same?

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
# z_min, z_max, z_num = 1e-3, 3., 703
zbins = 10

k_min, k_max, k_num = 1e-5, 20, 800
k_max_classy = 70

# ! options
use_h_units = True
whos_PS = 'vincenzo'
Pk_kind = 'nonlinear'
plot_Rmm = False
save_Pk = False
quad_integration = False
cl_formula = 'PySSC'
whos_WF = 'davide'
zmin_limber = 0.03  # found by trial and error
# ! options


# for whos_PS in ['stefano', 'vincenzo', 'CLASS', 'CLASS_clustertlkt']:
# for use_h_units in [True, False]:

# kmax in [h/Mpc], if I understand correctly this does not set any unit; it just tells CLASS which is the maximum k
# to compute (but not necessarily to use!), and one should of course say which units he's using
cosmo_par_dict['P_k_max_h/Mpc'] = k_max_classy

# note: 'use_h_units' means whether I want everything h units or not. This means that e.g. if use_h_units is True,
# what already is in h units should be left untouched, and what is not should be converted. so the scaling is not
# universal, but depends on which unit is being used for the element in question.
if use_h_units:
    x_label = '$k \\, [h/Mpc]$'
    k_scale, r_scale = h, h
else:
    x_label = '$k \\, [1/Mpc]$'
    k_scale, r_scale = 1., 1.

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
cosmo_classy = Class()
cosmo_classy.set(cosmo_par_dict)
cosmo_classy.compute()

# get k and P(k,z)
if whos_PS in ['vincenzo', 'stefano']:
    z_array, k_array, Pk = csmlib.get_external_Pk(whos_Pk=whos_PS, Pk_kind=Pk_kind, use_h_units=use_h_units)

elif whos_PS == 'CLASS':
    z_array = np.linspace(z_min, z_max, z_num)
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), k_num)  # this is in h/Mpc. The calculate_power function
    # takes care of the corerct h_units when computing Pk, but only returns Pk, so k_array has to be made consistent
    # by hand
    Pk = csmlib.calculate_power(k=k_array, z=z_array, cosmo_classy=cosmo_classy, use_h_units=use_h_units,
                                Pk_kind=Pk_kind)

elif whos_PS == 'CLASS_clustertlkt':
    z_array = np.linspace(z_min, z_max, z_num)
    k_array = np.logspace(np.log10(k_min), np.log10(k_max), k_num)  # this is in 1/Mpc. The Pk_with_classy_clustertlkt
    # function also returns k, rescaled or not
    k_array, Pk = csmlib.Pk_with_classy_clustertlkt(cosmo=cosmo_classy, z_array=z_array, k_array=k_array,
                                                    use_h_units=use_h_units, Pk_kind=Pk_kind)
else:
    raise ValueError('whos_PS must be either "vincenzo", "stefano", "CLASS" or "CLASS_clustertlkt"')

if save_Pk:
    np.save(job_path / f'output/Pk/Pk_kind={Pk_kind}_hunits={use_h_units}.npy', Pk)
    np.save(job_path / f'output/Pk/k_array_hunits={use_h_units}.npy', k_array)
    np.save(job_path / f'output/Pk/z_array.npy', z_array)


def Pk_wrap(k_ell, z, cosmo_classy=cosmo_classy, use_h_units=use_h_units, Pk_kind='nonlinear', argument_type='scalar'):
    """just a wrapper function to set some args to default values"""
    return csmlib.calculate_power(k_ell, z, cosmo_classy, use_h_units=use_h_units, Pk_kind=Pk_kind)


def kl_wrap(ell, z, use_h_units=use_h_units):
    """another simpe wrapper function, so as not to have to rewrite use_h_units=use_h_units"""
    return csmlib.k_limber(ell, z, use_h_units=use_h_units)


# take the derivative and compute log derivative
# https://bdiemer.bitbucket.io/colossus/_static/tutorial_cosmology.html dedrivative of matter Pk, just as rough reference
dP_dk = np.gradient(Pk, k_array, axis=1)
dlogPk_dlogk = k_array / Pk * dP_dk

# these are needed to build R1_mm as a function!
dP_dk_func = interp2d(x=k_array, y=z_array, z=dP_dk, kind='linear')
G1_tot_funct_scalar = interp2d(x=k_array, y=z_array, z=G1_tot_funct(z_array, k_array, G1_funct, G1_extrap),
                               kind='linear')  # this is not very elegant, but the function only accepts arrays...

# compute response
R1_mm = 1 - 1 / 3 * k_array / Pk * dP_dk + G1_tot_funct(z_array, k_array, G1_funct, G1_extrap)  # incl. extrapolation
# R1_mm = 1 - 1 / 3 * k / Pnl * dP_dk + G1_funct(z, k).T  # doesn't incl. extrapolation


if plot_Rmm:
    plot_Rmm_funct()

########################################### now, project the response ##################################################

# instantiate cosmology to compute comoving distance from astropy
Oc0 = cosmo_par_dict['Omega_cdm']
Ob0 = cosmo_par_dict['Omega_b']
Om0 = Oc0 + Ob0
Ode0 = 1 - Om0

# instantiate cosmo object from astropy
cosmo_astropy = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=-1.0, wa=0.0, Neff=3.04, m_nu=0.06, Ob0=Ob0)

# set the parameters, the functions wants a dict as input
# compute ells using the function in SSC_restructured_v2
nbl = cfg.nbl
ell_LL, _ = ell_utils.compute_ells(nbl, cfg.ell_min, cfg.ell_max_WL, 'ISTF')
ell_GG, _ = ell_utils.compute_ells(nbl, cfg.ell_min, cfg.ell_max_GC, 'ISTF')
ell_LG = ell_GG.copy()

# # fill k_limber array with meshgrid
# zz, ll = np.meshgrid(z_array, ell_WL)
# kl_array_mesh = k_limber(zz, ell=ll, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)
#

# # fill k_limber array manually
# kl_array_manual = np.zeros((nbl, z_num))
# for ell_idx, ellval in enumerate(ell_WL):
#     kl_array_manual[ell_idx, :] = k_limber(z_array, ell=ellval, cosmo_astropy=cosmo_astropy, use_h_units=use_h_units)


# at low redshift and high ells, k_limber explodes: cut the z range

min_idx = np.argmin(np.abs(z_array - zmin_limber))
z_array_limber = z_array[min_idx:]  # ! this has been found by hand, fix this!
# z_array_limber = z_array[5:]  # ! this has been found by hand, fix this!

# compute R1(k_ell, z)
# 1. the easy way: interpolate and fill
R1_mm_interp = interp2d(k_array, z_array, R1_mm, kind='linear')

# import WF
# ! these should be in Mpc ** -1 !! include a scaling below (after removing the z column)
if whos_WF == 'davide':
    W_LL_ISTF = np.genfromtxt(
        project_path.parent / 'common_data/everyones_WF_from_Gdrive/davide/nz10000/gen2022/wil_dav_IA_IST_nz10000_bia2.17.txt')
elif whos_WF == 'marco':
    W_LL_ISTF = np.load(
        project_path.parent / 'common_data/everyones_WF_from_Gdrive/marco/wil_mar_bia2.17_IST_nz10000.npy')
else:
    raise ValueError('whos_WF must be either "davide" or "marco"')

z_WF = W_LL_ISTF[:, 0]
W_LL_ISTF = W_LL_ISTF[:, 1:]  # resmove redshift column

# convert to h/Mpc if h_units is True, else leave it as it is
if use_h_units:
    W_LL_ISTF /= h

# normalize by r(z)**2 to translate them into PySSC convention:
my_r_of_z = csmlib.astropy_comoving_distance(z_WF, use_h_units=use_h_units, cosmo_astropy=cosmo_astropy)
W_LL_PySSC = W_LL_ISTF / np.repeat(my_r_of_z[:, None], zbins, axis=1) ** 2

# interpolate WF
W_LL_ISTF_interp = interp1d(z_WF, W_LL_ISTF, kind='linear', axis=0)
W_LL_PySSC_interp = interp1d(z_WF, W_LL_PySSC, kind='linear', axis=0)
W_LL_ISTF_array = W_LL_ISTF_interp(z_array_limber).T
W_LL_PySSC_array = W_LL_PySSC_interp(z_array_limber).T

# r(z) and dr/dz in Mpc. The scaling by h is implemented below.
zofr = cosmo_classy.z_of_r(z_array_limber)
# * the following have been tested against csmlib.astropy_comoving_distance and
# * np.gradient(csmlib.astropy_comoving_distance, z_array_limber), respectively; the h scaling works as well
comov_dist = zofr[0] * r_scale  # Comoving distance r(z)
dr_dz = (1 / zofr[1]) * r_scale  # Derivative dr/dz

# it should be
dV = comov_dist ** 2 * dr_dz

# this is to project Rl with ISTF formula
# ! this too has to be scaled by h!
Hz_arr = csmlib.H(z_array_limber, cosmo_astropy=cosmo_astropy) / r_scale

# TODO check again all the r_scale, k_scale and h scalings in general, I altready found 3 mistakes
# TODO recover plateau from before?


if quad_integration:

    if cl_formula == 'PySSC':
        function = R_LL_quad
        wf_A = W_LL_PySSC_interp
        wf_B = W_LL_PySSC_interp

    elif cl_formula == 'ISTF':
        function = cl_integral
        wf_A = W_LL_ISTF_interp
        wf_B = W_LL_ISTF_interp
    else:
        raise ValueError('cl_formula must be either "PySSC" or "ISTF"')

    print('quad integration started')
    start = time.perf_counter()
    R_LL_quad_arr = np.zeros((nbl, zbins, zbins))
    for ell_idx, ellval in enumerate(ell_LL):
        for i in range(zbins):
            for j in range(zbins):
                R_LL_quad_arr[ell_idx, i, j] = function(wf_A, wf_B, i, j, ellval)
    print('quad integration done in ', time.perf_counter() - start, ' seconds')


# TODO interpolate n_i(z) in 1D as done here for the WF! much smarter

simps_integrand = np.zeros((nbl, zbins, zbins, z_array_limber.size))
cl_simps_integrand = np.zeros((nbl, zbins, zbins, z_array_limber.size))
for zi, zval in enumerate(z_array_limber):
    for ell_idx, ell_val in enumerate(ell_LL):

        kl = kl_wrap(ell=ell_val, z=zval)
        P_of_kl_z = Pk_wrap(k_ell=kl, z=zval)
        R_of_kl_z = R1_mm_interp(kl, zval)[0]

        if cl_formula == 'PySSC':
            for i in range(zbins):
                for j in range(zbins):
                    simps_integrand[ell_idx, i, j, zi] = dV[zi] * W_LL_PySSC_array[i, zi] * W_LL_PySSC_array[j, zi] * \
                                                         R_of_kl_z * P_of_kl_z
                    cl_simps_integrand[ell_idx, i, j, zi] = dV[zi] * W_LL_PySSC_array[i, zi] * W_LL_PySSC_array[j, zi] * \
                                                         P_of_kl_z

        # ! does not work
        elif cl_formula == 'ISTF':
            for i in range(zbins):
                for j in range(zbins):
                    simps_integrand[ell_idx, i, j, zi] = c * (W_LL_ISTF_array[i, zi] * W_LL_ISTF_array[j, zi]) / \
                                                         (Hz_arr[zi] * comov_dist[zi] ** 2) * R_of_kl_z * P_of_kl_z
        else:
            raise ValueError('cl_formula must be either PySSC or ISTF')




# integrate over z with simpson's rule
R_LL = simps(simps_integrand, z_array_limber, axis=-1)
cl = simps(cl_simps_integrand, z_array_limber, axis=-1)


# finally, divide by Cl
# Cl_LL = np.load(job_path.parent / 'SSC_comparison/output/cl_3D/C_LL_WLonly_3D.npy')
R_LL /= cl
if quad_integration:
    R_LL_quad_arr /= cl

# test
# import vincenzo
R_LL_vinc = np.load(f'{project_path}/input/vincenzo/Pk_responses_2D/R_LL_WLonly_3D.npy')

color = cm.rainbow(np.linspace(0, 1, zbins))
plt.figure()
for i in range(zbins):
    j = i
    plt.plot(ell_LL, R_LL[:, i, j], c=color[i])  # , label='$R_\ell^{%i, %i}$' % (i, j))
    plt.plot(ell_LL, R_LL_vinc[:, i, j], '--', c=color[i])  # , label='$R_\ell^{%i, %i} vinc$' % (i, j))
    if quad_integration:
        plt.plot(ell_LL, R_LL_quad_arr[:, i, j], '.', c=color[i])  # , label='$R_\ell^{%i, %i}$' % (i, j))

plt.title(f'use_h_units = {use_h_units}, cl_formula = {cl_formula}')
plt.xlabel('$\ell$')
plt.ylabel('$R_\ell^{i, i}$')
plt.xscale("log")
plt.grid()


# ! test: plot R_of_kl_z
# ell_test = 4000
# print([kl_wrap(ell=ell_test, z=z) for z in z_array_limber])
# R_of_kl_z = [R1_mm_interp(kl_wrap(ell=ell_test, z=z), z)[0] for z in z_array_limber]
#
# plt.figure()
# plt.plot(z_array_limber, R_of_kl_z, label='$R_\ell^{%i}$' % ell_test)

# ! test: plot R_of_kl_z
ell_test = 2000
z_test = 1.6
ztest_idx = 160  # random


kl_arr = [kl_wrap(ell=ell, z=z_array[ztest_idx]) for ell in ell_LL]
P_of_kl_z = [Pk_wrap(k_ell=kl, z=z_array[ztest_idx]) for kl in kl_arr]
R_of_kl_z = [R1_mm_interp(kl, z_array[ztest_idx])[0] for kl in kl_arr]

plt.figure()
plt.plot(k_array, R1_mm[ztest_idx, :], label='$R_\ell^{%i}$' % ell_test)
plt.plot(kl_arr, R_of_kl_z, '.', label='$R_\ell^{%i}$' % ell_test)
plt.xscale('log')

plt.figure()
plt.plot(k_array, Pk[ztest_idx, :], label='$R_\ell^{%i}$' % ell_test)
plt.plot(kl_arr, P_of_kl_z, '.', label='$R_\ell^{%i}$' % ell_test)
plt.xscale('log')
plt.yscale('log')



# z_array_limber = z_array[5:]
# # ! test: kl_arr from older script
# kl_arr_old = np.load(project_path / f'jobs/SSC_comparison/output/kl_arr_use_h_units_{use_h_units}.npy')
# kl_arr_new = np.zeros(kl_arr_old.shape)
# for zi, zval in enumerate(z_array_limber):
#     for ell_idx, ell_val in enumerate(ell_LL):
#         kl_arr_new[zi, ell_idx] = kl_wrap(ell=ell_val, z=zval)
#
#
# print(np.allclose(kl_arr_new, kl_arr_old))
# mm.matshow(kl_arr_new[:30, :], 'ratio')
# mm.matshow(kl_arr_old[:30, :], 'kl_arr_old')


np.save(f'{job_path}/output/ell_values.npy', ell_LL)
np.save(f'{job_path}/output/R1_mm_kz.npy', R1_mm)
np.save(f'{job_path}/output/R_LL_WLonly_3D.npy', R_LL)
if quad_integration:
    np.save(f'{job_path}/output/R_LL_WLonly_3D_quad.npy', R_LL_quad_arr)

print('done')


