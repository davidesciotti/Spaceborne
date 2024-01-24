import time
import warnings
from copy import deepcopy
from glob import glob
import camb
import os
import numpy as np
from astropy.cosmology import w0waCDM
# from classy import Class
from numba import njit
import pyccl as ccl
from scipy.integrate import simps
import sys

import os
ROOT = os.getenv('ROOT')
sys.path.append(f'{ROOT}/Spaceborne/bin')
import my_module as mm


# TODO create function to compute pk from CAMB, hoping it accepts a vectorized k or z
# TODO check that the modifications to calculate_power don't break anything, I switched the order
#  of the arguments z and k in the output numpy array


c = 299792.458  # km/s


# ! example of how to instantiate a cosmo_astropy object
# cosmo_astropy = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa, Neff=Neff, m_nu=m_nu, Ob0=Ob0)

# ! example of how to instantiate a cosmo_class object
# cosmo_par_dict_classy = {'Omega_cdm': Oc0,
#                          'Omega_b': Ob0,
#                          'w0_fld': w0,
#                          'wa_fld': wa,
#                          'h': h,
#                          'n_s': n_s,
#                          'sigma8': sigma_8,
#
#                          'm_ncdm': m_nu,
#                          'N_ncdm': ISTF.neutrino_params['N_ncdm'],
#                          'N_ur': ISTF.neutrino_params['N_ur'],
#
#                          'Omega_Lambda': ISTF.extensions['Om_Lambda0'],
#
#                          'P_k_max_1/Mpc': 1200,
#                          'output': 'mPk',
#                          'non linear': 'halofit',  # ! takabird?
#
#                          # 'z_max_pk': 2.038,
#                          'z_max_pk': 4,  # do I get an error without this key?
#                          }


# old dictionary for CLASS
# cosmo_par_dict_classy = {
#     'Omega_b': Ob0,
#     'Omega_cdm': Oc0,
#     'n_s': n_s,
#     'sigma8': sigma_8,
#     'h': h,
#     'output': 'mPk',
#     'z_pk': '0, 0.5, 1, 2, 3',
#     'P_k_max_h/Mpc': 70,
#     'non linear': 'halofit'}


# this makes the import quite slow!!
# cosmo_classy = Class()
# cosmo_classy.set(cosmo_par_dict_classy)
# cosmo_classy.compute()


def map_keys(input_dict, key_mapping):
    """
    Maps the keys of a dictionary based on a given key mapping, while retaining
    keys not specified in the mapping.

    Parameters:
    -----------
    input_dict : dict
        The dictionary whose keys are to be mapped.
    key_mapping : dict
        A dictionary containing the mapping from the old keys to the new keys.

    Returns:
    --------
    new_dict : dict
        A dictionary with keys mapped according to the key_mapping, and
        with additional keys from the original dictionary.
    """

    # Mapping between flat_fid_pars_dict keys and instantiate_cosmo_ccl_obj expected keys
    if key_mapping is None:
        key_mapping = {
            'Om': 'Om_m0',
            'ODE': 'Om_Lambda0',
            'Ob': 'Om_b0',
            'wz': 'w_0',
            'wa': 'w_a',
            'h': 'h',
            'ns': 'n_s',
            's8': 'sigma_8',
            'm_nu': 'm_nu',
            'N_eff': 'N_eff'
        }

    new_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in input_dict:
            new_dict[new_key] = input_dict[old_key]

    for key in set(input_dict.keys()) - set(key_mapping.keys()):
        new_dict[key] = input_dict[key]

    return new_dict


@njit
def inv_E(z, Om0, Ode0, Ok0):
    result = 1 / np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ok0 * (1 + z) ** 2)
    return result


def E(z, cosmo_astropy):
    return cosmo_astropy.H(z).value / cosmo_astropy.H0.value


def H(z, cosmo_astropy):
    return cosmo_astropy.H(z).value


def r(z, cosmo_astropy):
    """ in Mpc, NOT Mpc/h"""
    return cosmo_astropy.comoving_distance(z).value


def astropy_comoving_distance(z, use_h_units, cosmo_astropy):
    h = cosmo_astropy.h
    if use_h_units:
        return cosmo_astropy.comoving_distance(z).value * h  # Mpc/h
    else:
        return cosmo_astropy.comoving_distance(z).value  # Mpc


def ccl_comoving_distance(z, use_h_units, cosmo_ccl):
    a = z_to_a(z)
    if use_h_units:
        return ccl.comoving_radial_distance(cosmo_ccl, a) * cosmo_ccl.cosmo.params.h  # Mpc/h
    else:
        return ccl.comoving_radial_distance(cosmo_ccl, a)  # Mpc


def r_tilde(z, cosmo_astropy):
    return cosmo_astropy.H0.value / c * r(z, cosmo_astropy)


def ang_diam_dist(z, cosmo_astropy):
    return cosmo_astropy.angular_diameter_distance(z).value


def k_limber(ell, z, use_h_units, cosmo_ccl):
    """ this function is vectorized in ell OR z, but not both at the same time. To vectorize in both, use
    something like meshgrid:
    zz, ll = np.meshgrid(z_grid_sigma2, ell_WL)
    kl_array_mesh = k_limber(zz, ell=ll, cosmo_ccl=cosmo_ccl, use_h_units=use_h_units)

    """
    assert type(use_h_units) == bool, 'use_h_units must be True or False'

    # astropy gives values in Mpc, so I call ccl_comoving_distance to have the correct values in both cases (h_units
    # or not)
    return (ell + 0.5) / ccl_comoving_distance(z, use_h_units, cosmo_ccl)


def get_kmax_limber(ell_grid, z_grid, use_h_units, cosmo_ccl):
    """ returns the maximum k_limber value for a given ell_grid and z_grid"""
    k_limber_list = []
    for z in z_grid:
        k_limber_list.append(k_limber(ell_grid, z, use_h_units, cosmo_ccl))
    return np.max(k_limber_list)


# @njit
# def E(z):
#     result = np.sqrt(Om0 * (1 + z) ** 3 + Ode0 + Ok0 * (1 + z) ** 2)
#     return result

# old, "manual", slowwwww
# def r_tilde(z):
#     # r(z) = c/H0 * int_0*z dz/E(z); I don't include the prefactor c/H0 so as to
#     # have r_tilde(z)
#     result = quad(inv_E, 0, z)[0]  # integrate 1/E(z) from 0 to z
#     return result

#
# def r(z):
#     result = c / H0 * quad(inv_E, 0, z)[0]
#     return result


def Pk_with_classy_clustertlkt(cosmo_class, z_array, k_array, use_h_units, Pk_kind='nonlinear', argument_type='arrays'):
    print('Warning: this function takes as input k in 1/Mpc and returns it in the specified units')

    if Pk_kind == 'nonlinear':
        classy_Pk = cosmo_class.pk
    elif Pk_kind == 'linear':
        classy_Pk = cosmo_class.pk_lin
    else:
        raise ValueError('Pk_kind must be either "nonlinear" or "linear"')

    if argument_type == 'scalar':
        Pk = classy_Pk(k_array, z_array)  # k_array and z_array are not arrays, but scalars!

    elif argument_type == 'arrays':
        num_k = k_array.size

        Pk = np.zeros((len(z_array), num_k))
        for z_idx, z_val in enumerate(z_array):
            Pk[z_idx, :] = np.array([classy_Pk(ki, z_val) for ki in k_array])
    else:
        raise ValueError('argument_type must be either "scalar" or "arrays"')

    # NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
    # to use in the toolkit. To do this you would do:
    if use_h_units:
        warnings.warn('take h from cosmo_classy, very easy')
        k_array /= h
        Pk *= h ** 3

    # return also k_array, to have it in the correct h scaling
    return k_array, Pk


def pk_from_ccl(k_array, z_array, use_h_units, cosmo_ccl, pk_kind='nonlinear'):
    """ ! * the input k_array must be in 1/Mpc * .
    If use_h_units is True, the output pk_2d is in (Mpc/h)^3, and k_array is in h/Mpc.
    If use_h_units is False, the output pk_2d is in Mpc^3, and k_array is in 1/Mpc (so it is returned unchanged)"""

    if pk_kind == 'nonlinear':
        ccl_pk_func = ccl.nonlin_matter_power
    elif pk_kind == 'linear':
        ccl_pk_func = ccl.linear_matter_power
    else:
        raise ValueError('pk_kind must be either "linear" or "nonlinear"')

    z_array = np.atleast_1d(z_array)
    pk_2d = np.array([ccl_pk_func(cosmo_ccl, k_array, a=1 / (1 + zval)) for zval in z_array]).T

    h = cosmo_ccl.cosmo.params.h
    if use_h_units:
        pk_2d *= h ** 3
        k_array /= h

    return k_array, pk_2d


def calculate_power(k, z, cosmo_classy, use_h_units=True, Pk_kind='nonlinear'):
    """
    The input k is always assumed to be in 1/Mpc, as classy_Pk does. The output k is then rescaled if use_h_units is
    True, so that it is in h/Mpc. The same is done for Pk: if use_h_units is True, the output Pk is rescaled to have
    it in (Mpc/h)^3. If use_h_units is False, the output Pk is in Mpc^3.
    """

    if use_h_units:
        k_scale = cosmo_classy.h()
        Pk_scale = cosmo_classy.h() ** 3
    else:
        k_scale = 1.
        Pk_scale = 1.

    # nice way to avoid the if-elif-else statement
    classy_Pk = {
        'nonlinear': cosmo_classy.pk,
        'linear': cosmo_classy.pk_lin
    }

    if np.isscalar(k):
        k = np.array([k])
    if np.isscalar(z):
        z = np.array([z])

    # Pk = np.zeros((len(k_array), len(z_array)))
    # for ki, k in enumerate(k_array):
    #     for zi, z in enumerate(z_array):
    # the argument of classy_Pk must be in units of 1/Mpc?
    pk = np.array([[classy_Pk[Pk_kind](k_val, z_val)
                    for z_val in z]
                   for k_val in k]
                  )

    if pk.shape == (1, 1):
        pk = pk[0, 0]

    return k / k_scale, pk * Pk_scale


def get_external_Pk(h, whos_Pk='vincenzo', Pk_kind='nonlinear', use_h_units=True):
    if whos_Pk == 'vincenzo':
        z_column = 1
        k_column = 0  # in [1/Mpc]
        Pnl_column = 2  # in [Mpc^3]
        Plin_column = 3  # in [Mpc^3]
        extension = 'dat'

    elif whos_Pk == 'stefano':
        z_column = 0
        k_column = 1  # in [h/Mpc]
        Pnl_column = 3  # in [Mpc^3/h^3]
        Plin_column = 2  # in [Mpc^3/h^3]
        extension = 'txt'
    else:
        raise ValueError('whos_Pk must be either stefano or vincenzo')

    if Pk_kind == 'linear':
        Pk_column = Plin_column
    elif Pk_kind == 'nonlinear':
        Pk_column = Pnl_column
    else:
        raise ValueError(f'Pk_kind must be either "linear" or "nonlinear"')

    Pkfile = np.genfromtxt(glob(f'/home/davide/Documenti/Lavoro/Programmi/common_data/Pk/Pk_{whos_Pk}.*')[0])
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
    pk = np.flip(Pk, axis=0)

    return z_array, k_array, pk


# def pk_camb():
#     # Now get matter power spectra and sigma8 at redshift 0 and 0.8
#     pars = camb.CAMBparams()
#     pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
#     pars.InitPower.set_params(ns=0.965)
#     # Note non-linear corrections couples to smaller scales than you want
#     pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)
#
#     # Linear spectra
#     pars.NonLinear = model.NonLinear_none
#     results = camb.get_results(pars)
#     kh, z, pk = results.get_matter_power_spectrum(minkh=k_min, maxkh=1, npoints=200)
#     s8 = np.array(results.get_sigma8())
#
#     # Non-Linear spectra (Halofit)
#     pars.NonLinear = model.NonLinear_both
#     results.calc_power_spectra(pars)
#     kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=k_min, maxkh=1, npoints=200)


def z_to_a(z):
    # these lines flip the array to have it monothonically increasing, breaking the correspondance between z and a!!!
    # be careful
    # if mm.is_increasing(z):
    #     result = 1. / (1 + z)
    #     return result[::-1]
    return 1. / (1 + z)


def a_to_z(a):
    return 1 / a - 1


def growth_factor_pyccl(z, cosmo_ccl):
    return ccl.growth_factor(cosmo_ccl, z_to_a(z))


def deg2_to_fsky(survey_area_deg2):
    # deg2_in_sphere = 41252.96  # deg^2 in a spere
    # return survey_area_deg2 / deg2_in_sphere

    f_sky = survey_area_deg2 * (np.pi / 180) ** 2 / (4 * np.pi)
    return f_sky

def fsky_to_deg2(f_sky):
    return f_sky * 4 * np.pi / (np.pi / 180) ** 2


def cl_integral_prefactor(z, cl_integral_convention, use_h_units, cosmo_ccl):
    """ this is the integration "prefactor" for the cl integral, without the dz, which is "added"
    afterwards in the actual integration, for example when using simps or trapz.

    note that this is not the volume element, but the collection of prefactors:
    PySSC: Cl = \int dV W^A_pyssc W^B_pyssc Pk = \int dz * dr/dz * r(z)**2 W^A_pyssc W^B_pyssc Pk
    Euclid: Cl = \int dz/(c*H0*E(z)*r(z)**2) W^A_euc W^B_euc Pk = \int dz * dr/dz * 1/r(z)**2 W^A W^B Pk

    This function is simply returning the terms after dz and excluding the kernels and the Pk.

    Equating the two above equations gives W_pyssc = W_euc / r(z)**2, so:

    - Option 1: PySSC convention. Euclid kernels divided by r(z)**2 and integration "prefactor" = r(z)**2 * dr/dz
    - Option 2: Euclid convention. Euclid kernels and integration "prefactor" = dr/dz / r(z)**2 = c/H0 / (E(z)*r(z)**2)
    this is because dr/dz = c/H(z) = c/(H0*E(z))
    """
    r_of_z = ccl_comoving_distance(z, use_h_units=use_h_units, cosmo_ccl=cosmo_ccl)
    dr_dz = np.gradient(r_of_z, z)

    if cl_integral_convention == 'PySSC':
        cl_integral_prefactor = r_of_z ** 2 * dr_dz  # this is dV/dz
    elif cl_integral_convention == 'Euclid':
        cl_integral_prefactor = 1 / r_of_z ** 2 * dr_dz  # this is not dV/dz! that's why I don't write dV in the function name
    else:
        raise ValueError('cl_integral_convention must be either "Euclid" or "PySSC"')
    return cl_integral_prefactor


def get_omega_nu0(m_nu, h, n_eff=3.046, neutrino_mass_fac=94.07):
    """
    Calculate the neutrino mass density parameter (Omega_nu h^2).
    Look in https://arxiv.org/pdf/2207.05766.pdf for a comment on the difference between the factors 93.14eV and 94.07eV

    Parameters:
    # m_nu : float
        Total neutrino mass (sum of the masses of the neutrino species).
    h : float
        Reduced Hubble constant.
    n_eff : float, optional
        Effective number of neutrinos. Default is 3.046.
    neutrino_mass_fac : float, optional
        Neutrino mass factor for conversion to density parameter. Default is 94.07.

    Returns:
    float
        Neutrino mass density parameter.
    """
    # n_eff = n_ur + n_ncdm  # masless + massive neutrinos
    g_factor = n_eff / 3
    omega_nu0 = m_nu / (neutrino_mass_fac * g_factor ** 0.75 * h ** 2)
    return omega_nu0


def get_omega_k0(omega_m0, omega_Lambda0):
    omega_k0 = 1 - omega_m0 - omega_Lambda0
    if np.abs(omega_k0) < 1e-10:
        warnings.warn("Omega_k is very small but not exactly 0, probably due to numerical errors. Setting it to 0")
        omega_k0 = 0
    return omega_k0


def instantiate_cosmo_ccl_obj(fiducial_pars_dict, extra_parameters):
    # example extra_parameters:
    # extra_parameters = {"camb": {"halofit_version": "mead2020_feedback",
    #                              "HMCode_logT_AGN": 7.75}}

    fiducial_pars_dict = mm.flatten_dict(fiducial_pars_dict)  # flatten the dictionary if it's nested

    Omega_nu = get_omega_nu0(fiducial_pars_dict['m_nu'],
                             fiducial_pars_dict['h'],
                             n_eff=fiducial_pars_dict['N_eff'])
    Omega_c = (fiducial_pars_dict['Om_m0'] - fiducial_pars_dict['Om_b0'] - Omega_nu)

    if 'Om_k0' in fiducial_pars_dict.keys():
        Omega_k = fiducial_pars_dict['Om_k0']
    else:
        Omega_k = get_omega_k0(omega_m0=fiducial_pars_dict['Om_m0'], omega_Lambda0=fiducial_pars_dict['Om_Lambda0'])

    if extra_parameters is None:
        print('No extra parameters passed to CAMB in instantiate_cosmo_ccl_obj')

    cosmo_ccl = ccl.Cosmology(Omega_c=Omega_c,
                              Omega_b=fiducial_pars_dict['Om_b0'],
                              w0=fiducial_pars_dict['w_0'],
                              wa=fiducial_pars_dict['w_a'],
                              h=fiducial_pars_dict['h'],
                              sigma8=fiducial_pars_dict['sigma_8'],
                              n_s=fiducial_pars_dict['n_s'],
                              m_nu=fiducial_pars_dict['m_nu'],
                              Omega_k=Omega_k,
                              baryonic_effects=None,
                              mass_split='single',
                              matter_power_spectrum='camb',
                              extra_parameters=extra_parameters)

    return cosmo_ccl


def project_pk_helper(key, pab_k_z_interp_func_dict, wf_dict, z_grid, ell_grid, cl_integral_convention, use_h_units,
                      cosmo_ccl):
    print('computing cls for probe combination', key)
    probe_a = wf_dict[key[0]]
    probe_b = wf_dict[key[1]]
    return key, project_pk(pab_k_z_interp_func_dict[key], probe_a, probe_b, z_grid, ell_grid,
                           cl_integral_convention, use_h_units, cosmo_ccl)


def project_pk(pab_k_z_interp_func, kernel_a, kernel_b, z_grid, ell_grid, cl_integral_convention, use_h_units,
               cosmo_ccl):
    """
        Project the pk to get the cls, or the pk responses to get the projected responses.
        ! Remember, if you use cl_integral_convention = PySSC you must normalize the kernels properly, it is not enough to
        ! pass the approptiate cl_integral_convention to csmlib.cl_integral_prefactor!
        :param pab_k_z_interp_func:
        :param kernel_a:
        :param kernel_b:
        :return:
        """

    # Compute prefactors
    cl_integral_prefactor_arr = cl_integral_prefactor(z_grid, cl_integral_convention,
                                                      use_h_units=use_h_units,
                                                      cosmo_ccl=cosmo_ccl)

    # Generate the projection operator
    projection_operator = np.einsum('o, oi, oj-> oij', cl_integral_prefactor_arr, kernel_a, kernel_b)

    # Initialize pab_kl_z_arr
    pab_kl_z_arr = np.zeros((ell_grid.size, z_grid.size))

    # Compute pab_kl_z_arr using vectorization where possible
    start_time = time.time()
    for zgrid_idx, zgrid_val in enumerate(z_grid):
        # Vectorized k_limber over all ell for this z
        kl_values = k_limber(ell_grid, zgrid_val, use_h_units=use_h_units, cosmo_ccl=cosmo_ccl)

        # Vectorized evaluation of the pk and pk responses for this z
        pab_kl_z_arr[:, zgrid_idx] = pab_k_z_interp_func((kl_values, zgrid_val))

    # Compute cl_integrand
    cl_integrand = np.einsum('oij, Lo  -> Lijo', projection_operator, pab_kl_z_arr)

    # Integrate over z with Simpson's rule
    cl = simps(cl_integrand, z_grid, axis=-1)

    print(f'cl integral done in {(time.time() - start_time):.2f} seconds')
    return cl


def reshape_pk_vincenzo_to_2d(filename):
    # Import and reshape P(k,z) from the input used bu Vincenzo
    pk = np.genfromtxt(filename)
    z_grid_pk = np.unique(pk[:, 0])
    k_grid_pk = 10 ** np.unique(pk[:, 1])
    pk_2d = pk[:, 2].reshape(len(z_grid_pk), len(k_grid_pk)).T
    return pk_2d, k_grid_pk, z_grid_pk


def sigma8_to_As(pars, extra_args):
    params = deepcopy(pars)

    if 'As' in params:
        ini_As = params['As']

    else:
        ini_As = 2.1e-9
        params['As'] = ini_As

    final_sig8 = params['sigma8']

    params['WantTransfer'] = True
    params.update(extra_args)
    pars = camb.set_params(**{key: val for key, val in params.items() if key != 'sigma8'})
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    results = camb.get_results(pars)
    ini_sig8 = np.array(results.get_sigma8())[-1]
    final_As = ini_As * (final_sig8 / ini_sig8) ** 2.

    return final_As
