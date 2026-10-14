import time
import warnings
from copy import deepcopy

import numpy as np
import pyccl as ccl
from scipy.integrate import simpson as simps

from spaceborne import constants
from spaceborne import sb_lib as sl

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


def cloelib_to_spaceborne_params_mapping(
    cloelib_pars_dict: dict, neutrino_mass_conversion_factor: float = 93.14
) -> dict:
    """Maps cosmological parameters from the cloelib basis to the Spaceborne one.
    Requires the jaxcosmo package to compute sigma8 from As."""

    import jaxcosmo

    h = cloelib_pars_dict['H0'] / 100
    Omega_nu0 = cloelib_pars_dict['mnu'] / (93.14 * h**2)
    Omega_m0 = (
        cloelib_pars_dict['Omega_cdm0'] + cloelib_pars_dict['Omega_b0'] + Omega_nu0
    )
    sigma8 = jaxcosmo.As_to_sigma8_max_precision(
        cloelib_pars_dict['As'],
        Omega_m0,
        cloelib_pars_dict['Omega_b0'],
        h,
        cloelib_pars_dict['ns'],
        cloelib_pars_dict['mnu'],
        cloelib_pars_dict['w0'],
        cloelib_pars_dict['wa'],
    )

    sigma8 = float(sigma8)
    # each massive species contributes (T_ncdm/T_cmb)^4 (11/4)^(4/3) = 1.0132 to N_eff
    # while relativistic (T_ncdm = 0.71611, as in CLASS and CCL); this way CCL's
    # N_nu_rel equals N_ur
    neff_per_massive_nu = 0.71611**4 * (11 / 4) ** (4 / 3)
    N_eff = cloelib_pars_dict['N_ur'] + cloelib_pars_dict['N_mnu'] * neff_per_massive_nu

    # Set some Spaceborne configs at runtime; the rest are in pipe_cfg.yaml
    sb_pars_dict = {}
    sb_pars_dict['Omega_m0'] = Omega_m0
    sb_pars_dict['Omega_b0'] = cloelib_pars_dict['Omega_b0']
    sb_pars_dict['Omega_k0'] = float(cloelib_pars_dict['Omega_k0'])
    sb_pars_dict['w0'] = float(cloelib_pars_dict['w0'])
    sb_pars_dict['wa'] = float(cloelib_pars_dict['wa'])
    sb_pars_dict['h'] = h
    sb_pars_dict['ns'] = cloelib_pars_dict['ns']
    sb_pars_dict['sigma8'] = sigma8
    sb_pars_dict['m_nu'] = cloelib_pars_dict['mnu']
    sb_pars_dict['Neff'] = N_eff

    return sb_pars_dict


def migrate_legacy_cosmo_cfg(cosmo_cfg: dict) -> dict:
    """Renames the old cosmology config keys to the cosmology API/cloelib ones, and
    removes ODE.

    The dark energy density is no longer an input (it is set by closure); an old ODE
    is only accepted if consistent with Omega_m0 and Omega_k0, or used to derive
    Omega_k0 if the latter is missing.
    """

    # old cosmology config keys -> cloelib/cosmology API names
    LEGACY_COSMO_KEYS = {
        'Om': 'Omega_m0',
        'Ob': 'Omega_b0',
        'Om_k0': 'Omega_k0',
        'wz': 'w0',
        's8': 'sigma8',
        'N_eff': 'Neff',
    }
    cosmo_cfg = dict(cosmo_cfg)
    renamed = []
    for old, new in LEGACY_COSMO_KEYS.items():
        if old not in cosmo_cfg:
            continue
        if new in cosmo_cfg:
            raise ValueError(
                f'cosmology: both {old!r} (old name) and {new!r} are given; '
                f'please only use {new!r}'
            )
        cosmo_cfg[new] = cosmo_cfg.pop(old)
        renamed.append(f'{old} -> {new}')

    if 'ODE' in cosmo_cfg:
        omega_de0 = cosmo_cfg.pop('ODE')
        if 'Omega_k0' not in cosmo_cfg:
            cosmo_cfg['Omega_k0'] = get_omega_k0(cosmo_cfg['Omega_m0'], omega_de0)
            renamed.append('ODE -> Omega_k0 = 1 - Omega_m0 - ODE')
        else:
            closure = cosmo_cfg['Omega_m0'] + omega_de0 + cosmo_cfg['Omega_k0'] - 1
            # the tolerance accounts for the radiation density (~1e-4)
            if np.abs(closure) > 1e-3:
                raise ValueError(
                    'Inconsistent cosmology: Omega_m0 + ODE + Omega_k0 - 1 = '
                    f'{closure:.4g}. ODE is no longer an input (it is set by '
                    'closure): please remove it.'
                )
            renamed.append('ODE removed (set by closure)')

    if renamed:
        warnings.warn(
            'Old cosmology config keys found, please update your config. '
            f'Converted: {", ".join(renamed)}',
            stacklevel=2,
        )

    return cosmo_cfg


def ccl_comoving_distance(z, use_h_units, cosmo_ccl):
    a = z_to_a(z)
    if use_h_units:
        return (
            ccl.comoving_radial_distance(cosmo_ccl, a) * cosmo_ccl.cosmo.params.h
        )  # Mpc/h
    else:
        return ccl.comoving_radial_distance(cosmo_ccl, a)  # Mpc


def k_limber(ell, z, use_h_units, cosmo_ccl):
    """This function is vectorized in ell OR z, but not both at the same time.
    To vectorize in both, use something like meshgrid:
    zz, ll = np.meshgrid(z_grid_sigma2, ell_WL)
    kl_array_mesh = k_limber(zz, ell=ll, cosmo_ccl=cosmo_ccl, use_h_units=use_h_units)

    """
    assert isinstance(use_h_units, bool), 'use_h_units must be True or False'

    # astropy gives values in Mpc, so I call ccl_comoving_distance to have the
    # correct values in both cases (h_units or not)
    return (ell + 0.5) / ccl_comoving_distance(z, use_h_units, cosmo_ccl)


def get_kmax_limber(ell_grid, z_grid, use_h_units, cosmo_ccl):
    """Returns the maximum k_limber value for a given ell_grid and z_grid"""
    k_limber_list = []
    z_grid = np.atleast_1d(z_grid)
    for z in z_grid:
        k_limber_list.append(k_limber(ell_grid, z, use_h_units, cosmo_ccl))
    return float(np.max(k_limber_list))


def pk_from_ccl(k_array, z_array, use_h_units, cosmo_ccl, pk_kind='nonlinear'):
    """! * the input k_array must be in 1/Mpc * .
    If use_h_units is True, the output pk_2d is in (Mpc/h)^3, and k_array is in h/Mpc.
    If use_h_units is False, the output pk_2d is in Mpc^3, and k_array is in 1/Mpc
    (so it is returned unchanged)
    """
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
    fsky = survey_area_deg2 * (np.pi / 180) ** 2 / (4 * np.pi)
    return fsky


def fsky_to_deg2(fsky):
    return fsky * 4 * np.pi / (np.pi / 180) ** 2


def cl_integral_prefactor(z, cl_integral_convention, use_h_units, cosmo_ccl):
    r"""This is the integration "prefactor" for the cl integral, without the dz,
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
    """Calculate the neutrino mass density parameter (Omega_nu h^2).
    Look in https://arxiv.org/pdf/2207.05766.pdf for a comment on the difference
    between the factors 93.14eV and 94.07eV

    Parameters
    ----------
    m_nu : float
        Total neutrino mass (sum of the masses of the neutrino species).
    h : float
        Reduced Hubble constant.
    n_eff : float, optional
        Effective number of neutrinos. Default is 3.046.
    neutrino_mass_fac : float, optional
        Neutrino mass factor for conversion to density parameter. Default is 94.07.

    Returns
    -------
    float
        Neutrino mass density parameter.

    """
    # n_eff = n_ur + n_ncdm  # masless + massive neutrinos
    g_factor = n_eff / 3
    omega_nu0 = m_nu / (neutrino_mass_fac * g_factor**0.75 * h**2)
    return omega_nu0


def get_omega_nu0_new(m_nu, h, neutrino_mass_fac=93.14):
    """Compute present-day neutrino density parameter Ω_ν₀ (dimensionless).

    Ω_ν₀ = Σ m_ν / (93.14 eV · h²)

    See Eq. (19) of Lesgourgues & Pastor (2012, arXiv:1212.6154).
    See also https://arxiv.org/pdf/2207.05766.pdf for a comment on the difference
    between neutrino_mass_fac = 93.14eV and 94.07eV.

    Note that this function returns the present-day neutrino density parameter
    Ω_ν₀, *not* ω_ν = Ω_ν₀ · h²

    Parameters
    ----------
    m_nu : float
        Total neutrino mass in eV.
    h : float
        Dimensionless Hubble constant (H0 / 100 km/s/Mpc).
    neutrino_mass_fac : float, optional
        Conversion constant, default 93.14 eV.

    Returns
    -------
    float
        Ω_ν₀.
    """
    return m_nu / (neutrino_mass_fac * h**2)


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

    # Take Omega_nu (massive) from CCL itself, so that the CCL Omega_m matches the
    # requested Omega_m0 exactly (CCL: Omega_m = Omega_c + Omega_b + Omega_nu_mass)
    omega_nu0 = ccl.Cosmology(
        Omega_c=0.25,  # placeholder, Omega_nu_mass does not depend on it
        Omega_b=fiducial_pars_dict['Omega_b0'],
        h=fiducial_pars_dict['h'],
        sigma8=fiducial_pars_dict['sigma8'],
        n_s=fiducial_pars_dict['ns'],
        m_nu=fiducial_pars_dict['m_nu'],
        Neff=fiducial_pars_dict['Neff'],
        mass_split='single',
    )['Omega_nu_mass']

    omega_cdm0 = (
        fiducial_pars_dict['Omega_m0'] - fiducial_pars_dict['Omega_b0'] - omega_nu0
    )

    if extra_parameters is None:
        try:
            extra_parameters = fiducial_pars_dict['other_params'][
                'camb_extra_parameters'
            ]
        except KeyError:
            print('No extra parameters passed to CAMB in instantiate_cosmo_ccl_obj')

    cosmo_ccl = ccl.Cosmology(
        Omega_c=omega_cdm0,
        Omega_b=fiducial_pars_dict['Omega_b0'],
        w0=fiducial_pars_dict['w0'],
        wa=fiducial_pars_dict['wa'],
        h=fiducial_pars_dict['h'],
        sigma8=fiducial_pars_dict['sigma8'],
        n_s=fiducial_pars_dict['ns'],
        m_nu=fiducial_pars_dict['m_nu'],
        Neff=fiducial_pars_dict['Neff'],
        # the dark energy density is set by CCL by closure
        Omega_k=fiducial_pars_dict['Omega_k0'],
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
    """Project the pk to get the cls, or the pk responses to get the projected responses.
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

    print(f'cl integral done in {(time.time() - start_time):.2f} s')
    return cl


def ell_prefactor_gamma_and_ia(ell):
    """From Kilbinger, M., Heymans, C., Asgari, M., et al. 2017, MNRAS, 472, 2126.
    Taken from CLOE. Formula is
    (ell + 2)!/(ell-2)! * (2/(2*ell+1))**4
    """
    result = np.sqrt((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0)) / (ell + 0.5) ** 2
    return result


def ell_prefactor_mag(ell):
    """Taken from CLOE."""
    return ell * (ell + 1) / (ell + 0.5) ** 2


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
