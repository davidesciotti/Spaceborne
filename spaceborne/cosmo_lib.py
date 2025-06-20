import time
import warnings
from copy import deepcopy

import numpy as np
from scipy.integrate import simpson as simps

import pyccl as ccl
from spaceborne import sb_lib as sl
from spaceborne import constants

# ! prefactor for limber and curved-sky corrections
# prefactor = np.array(
#     [np.sqrt(math.factorial(int(ell) + 2) / math.factorial(int(ell) - 2)) * \
# (2 / (2 * ell + 1)) ** 2
#      for ell in ell_grid])

# prefactor = prefactor.reshape((-1, 1, 1))

# if divide_cls_by_prefactor:
#     cl_LL_3D /= prefactor ** 2
#     cl_GL_3D /= prefactor


c = constants.SPEED_OF_LIGHT


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

    # Mapping between flat_fid_pars_dict keys and instantiate_cosmo_ccl_obj
    # expected keys
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
            'N_eff': 'N_eff',
        }

    new_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in input_dict:
            new_dict[new_key] = input_dict[old_key]

    for key in set(input_dict.keys()) - set(key_mapping.keys()):
        new_dict[key] = input_dict[key]

    return new_dict


def ccl_comoving_distance(z, use_h_units, cosmo_ccl):
    a = z_to_a(z)
    if use_h_units:
        return (
            ccl.comoving_radial_distance(cosmo_ccl, a) * cosmo_ccl.cosmo.params.h
        )  # Mpc/h
    else:
        return ccl.comoving_radial_distance(cosmo_ccl, a)  # Mpc


def k_limber(ell, z, use_h_units, cosmo_ccl):
    """this function is vectorized in ell OR z, but not both at the same time.
    To vectorize in both, use something like meshgrid:
    zz, ll = np.meshgrid(z_grid_sigma2, ell_WL)
    kl_array_mesh = k_limber(zz, ell=ll, cosmo_ccl=cosmo_ccl, use_h_units=use_h_units)

    """
    assert isinstance(use_h_units, bool), 'use_h_units must be True or False'

    # astropy gives values in Mpc, so I call ccl_comoving_distance to have the
    # correct values in both cases (h_units or not)
    return (ell + 0.5) / ccl_comoving_distance(z, use_h_units, cosmo_ccl)


def get_kmax_limber(ell_grid, z_grid, use_h_units, cosmo_ccl):
    """returns the maximum k_limber value for a given ell_grid and z_grid"""
    k_limber_list = []
    for z in z_grid:
        k_limber_list.append(k_limber(ell_grid, z, use_h_units, cosmo_ccl))
    return np.max(k_limber_list)


def pk_from_ccl(k_array, z_array, use_h_units, cosmo_ccl, pk_kind='nonlinear'):
    """! * the input k_array must be in 1/Mpc * .
    If use_h_units is True, the output pk_2d is in (Mpc/h)^3, and k_array is in h/Mpc.
    If use_h_units is False, the output pk_2d is in Mpc^3, and k_array is in 1/Mpc
    (so it is returned unchanged)"""

    if pk_kind == 'nonlinear':
        ccl_pk_func = ccl.nonlin_matter_power
    elif pk_kind == 'linear':
        ccl_pk_func = ccl.linear_matter_power
    else:
        raise ValueError('pk_kind must be either "linear" or "nonlinear"')

    z_array = np.atleast_1d(z_array)
    pk_2d = np.array(
        [ccl_pk_func(cosmo_ccl, k_array, a=1 / (1 + zval)) for zval in z_array]
    ).T

    h = cosmo_ccl.cosmo.params.h
    if use_h_units:
        pk_2d *= h**3
        k_array /= h

    return k_array, pk_2d


def z_to_a(z):
    # these lines flip the array to have it monothonically increasing, breaking the
    # correspondance between z and a!!!
    # be careful
    # if sl.is_increasing(z):
    #     result = 1. / (1 + z)
    #     return result[::-1]
    return 1.0 / (1 + z)


def a_to_z(a):
    return 1 / a - 1


def growth_factor_ccl(z, cosmo_ccl):
    return ccl.growth_factor(cosmo_ccl, z_to_a(z))


def deg2_to_fsky(survey_area_deg2):
    f_sky = survey_area_deg2 * (np.pi / 180) ** 2 / (4 * np.pi)
    return f_sky


def fsky_to_deg2(f_sky):
    return f_sky * 4 * np.pi / (np.pi / 180) ** 2


def cl_integral_prefactor(z, cl_integral_convention, use_h_units, cosmo_ccl):
    r"""this is the integration "prefactor" for the cl integral, without the dz, 
    which is "added"
    afterwards in the actual integration, for example when using simps or trapz.

    note that this is not the volume element, but the collection of prefactors:
    PySSC: Cl = \int dV W^A_pyssc W^B_pyssc Pk = \
        \int dz * dr/dz * r(z)**2 * \W^A_pyssc * W^B_pyssc * Pk  
    Euclid: Cl = \int dz * c/(H0*E(z)*r(z)**2) * W^A_euc * W^B_euc * Pk = \
        \int dz * dr/dz * 1/r(z)**2 * W^A_euc * W^B_euc * Pk

    This function is simply returning the terms after dz and excluding the kernels and 
    the Pk (which is *not* d^2Cl/dVddeltab)

    Equating the two above equations gives W_pyssc = W_euc / r(z)**2, so:

    - Option 1: PySSC convention (deprecated). Euclid kernels divided by r(z)**2 and 
    integration "prefactor" = r(z)**2 * dr/dz
    - Option 2: Euclid convention. Euclid kernels and integration 
    "prefactor" = dr/dz / r(z)**2 = c/H0 / (E(z)*r(z)**2)
    this is because dr/dz = c/H(z) = c/(H0*E(z))
    - Option 3: Euclid_KE_approximation. In this case you only need one of these 
    prefactors in the integral (since it's
    a simple integral over distance). The dr_1/r_1^2 * dr_2/r_2^2 of the previous 
    case becomes dr/r**4, in this way
    """
    r_of_z = ccl_comoving_distance(z, use_h_units=use_h_units, cosmo_ccl=cosmo_ccl)
    a = z_to_a(z)
    h_of_z = ccl.background.h_over_h0(cosmo_ccl, a) * cosmo_ccl.cosmo.params.h * 100
    dr_dz = c / h_of_z
    # dr_dz = np.gradient(r_of_z, z)

    if cl_integral_convention == 'PySSC':
        raise ValueError('PySSC integral convention is deprecated')
        cl_integral_prefactor = r_of_z**2 * dr_dz  # this is dV/dz
    elif cl_integral_convention == 'Euclid':
        cl_integral_prefactor = (
            1 / r_of_z**2 * dr_dz
        )  # this is not dV/dz! that's why I don't write dV in the function name
    elif cl_integral_convention == 'Euclid_KE_approximation':
        cl_integral_prefactor = (
            1 / r_of_z**4 * dr_dz
        )  # this is not dV/dz! that's why I don't write dV in the function name
    else:
        raise ValueError('cl_integral_convention must be either "Euclid" or "PySSC"')
    return cl_integral_prefactor


def get_omega_nu0(m_nu, h, n_eff=3.046, neutrino_mass_fac=94.07):
    """
    Calculate the neutrino mass density parameter (Omega_nu h^2).
    Look in https://arxiv.org/pdf/2207.05766.pdf for a comment on the difference
    between the factors 93.14eV and 94.07eV

    Parameters:
    m_nu : float
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
    omega_nu0 = m_nu / (neutrino_mass_fac * g_factor**0.75 * h**2)
    return omega_nu0


def get_omega_k0(omega_m0, omega_Lambda0):
    omega_k0 = 1 - omega_m0 - omega_Lambda0
    if np.abs(omega_k0) < 1e-10:
        warnings.warn(
            'Omega_k is very small but not exactly 0, probably due to numerical '
            'errors. Setting it to 0',
            stacklevel=2,
        )
        omega_k0 = 0
    return omega_k0


def instantiate_cosmo_ccl_obj(fiducial_pars_dict, extra_parameters):
    # example extra_parameters:
    # extra_parameters = {"camb": {"halofit_version": "mead2020_feedback",
    #                              "HMCode_logT_AGN": 7.75}}

    # flatten the dictionary if it's nested
    fiducial_pars_dict = sl.flatten_dict(fiducial_pars_dict)

    Omega_nu = get_omega_nu0(
        fiducial_pars_dict['m_nu'],
        fiducial_pars_dict['h'],
        n_eff=fiducial_pars_dict['N_eff'],
    )
    Omega_c = fiducial_pars_dict['Om_m0'] - fiducial_pars_dict['Om_b0'] - Omega_nu

    if 'Om_k0' in fiducial_pars_dict:
        Omega_k = fiducial_pars_dict['Om_k0']
    else:
        Omega_k = get_omega_k0(
            omega_m0=fiducial_pars_dict['Om_m0'],
            omega_Lambda0=fiducial_pars_dict['Om_Lambda0'],
        )

    if extra_parameters is None:
        try:
            extra_parameters = fiducial_pars_dict['other_params'][
                'camb_extra_parameters'
            ]
        except KeyError:
            print('No extra parameters passed to CAMB in instantiate_cosmo_ccl_obj')

    cosmo_ccl = ccl.Cosmology(
        Omega_c=Omega_c,
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
        extra_parameters=extra_parameters,
    )

    return cosmo_ccl


def project_pk(
    pab_k_z_interp_func,
    kernel_a,
    kernel_b,
    z_grid,
    ell_grid,
    cl_integral_convention,
    use_h_units,
    cosmo_ccl,
):
    """
    Project the pk to get the cls, or the pk responses to get the projected responses.
    ! Remember, if you use cl_integral_convention = PySSC you must normalize the kernels
    properly, it is not enough to
    ! pass the approptiate cl_integral_convention to csmlib.cl_integral_prefactor!
    :param pab_k_z_interp_func:
    :param kernel_a:
    :param kernel_b:
    :return:
    """

    # Compute prefactors
    cl_integral_prefactor_arr = cl_integral_prefactor(
        z_grid, cl_integral_convention, use_h_units=use_h_units, cosmo_ccl=cosmo_ccl
    )

    # Generate the projection operator
    projection_operator = np.einsum(
        'o, oi, oj-> oij', cl_integral_prefactor_arr, kernel_a, kernel_b
    )

    # Initialize pab_kl_z_arr
    pab_kl_z_arr = np.zeros((ell_grid.size, z_grid.size))

    # Compute pab_kl_z_arr using vectorization where possible
    start_time = time.time()
    for zgrid_idx, zgrid_val in enumerate(z_grid):
        # Vectorized k_limber over all ell for this z
        kl_values = k_limber(
            ell_grid, zgrid_val, use_h_units=use_h_units, cosmo_ccl=cosmo_ccl
        )

        # Vectorized evaluation of the pk and pk responses for this z
        pab_kl_z_arr[:, zgrid_idx] = pab_k_z_interp_func((kl_values, zgrid_val))

    # Compute cl_integrand
    cl_integrand = np.einsum('oij, Lo  -> Lijo', projection_operator, pab_kl_z_arr)

    # Integrate over z with Simpson's rule
    cl = simps(cl_integrand, z_grid, axis=-1)

    print(f'cl integral done in {(time.time() - start_time):.2f} seconds')
    return cl


def ell_prefactor_gamma_and_ia(ell):
    """
    From Kilbinger, M., Heymans, C., Asgari, M., et al. 2017, MNRAS, 472, 2126.
    Taken from CLOE. Formula is
    (ell + 2)!/(ell-2)! * (2/(2*ell+1))**4
    """
    result = np.sqrt((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0)) / (ell + 0.5) ** 2
    return result


def ell_prefactor_mag(ell):
    """
    Taken from CLOE.
    """
    return ell * (ell + 1) / (ell + 0.5) ** 2


def reshape_pk_vincenzo_to_2d(filename):
    # Import and reshape P(k,z) from the input used by Vincenzo
    pk = np.genfromtxt(filename)
    z_grid_pk = np.unique(pk[:, 0])
    k_grid_pk = 10 ** np.unique(pk[:, 1])
    pk_2d = pk[:, 2].reshape(len(z_grid_pk), len(k_grid_pk)).T
    return pk_2d, k_grid_pk, z_grid_pk


def sigma8_to_As(pars, extra_args):
    """This function has been written by Matteo Martinelli"""
    import camb

    params = deepcopy(pars)

    if 'As' in params:
        ini_As = params['As']

    else:
        ini_As = 2.1e-9
        params['As'] = ini_As

    final_sig8 = params['sigma8']

    params['WantTransfer'] = True
    params.update(extra_args)
    pars = camb.set_params(
        **{key: val for key, val in params.items() if key != 'sigma8'}
    )
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    results = camb.get_results(pars)
    ini_sig8 = np.array(results.get_sigma8())[-1]
    final_As = ini_As * (final_sig8 / ini_sig8) ** 2.0

    return final_As
