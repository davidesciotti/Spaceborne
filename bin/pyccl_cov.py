import pdb
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import yaml
from joblib import Parallel, delayed
from scipy.special import erf
import ray
from tqdm import tqdm

ray.shutdown()
ray.init()

# get project directory adn import useful modules
project_path = Path.cwd().parent

sys.path.append(f'../../common_lib_and_cfg/common_lib')
import my_module as mm
import cosmo_lib

sys.path.append(f'../../common_lib_and_cfg/common_config')
import ISTF_fid_params as ISTF_fid
import mpl_cfg

sys.path.append(f'../../cl_v2/bin')
import wf_cl_lib

matplotlib.use('Qt5Agg')
start_time = time.perf_counter()
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX
# notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb
# HALO MODEL PRESCRIPTIONS:
# KiDS1000 Methodology: https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)
# Krause2017: https://arxiv.org/pdf/1601.05779.pdf

# it was p_of_k_a=Pk, but it should use the LINEAR power spectrum, so we leave it as None (see documentation:
# https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=halomod_Tk3D_SSC#pyccl.halos.halo_model.halomod_Tk3D_SSC)
# 🐛 bug fixed: normprof shoud be True
# 🐛 bug fixed?: p_of_k_a=None instead of Pk


def initialize_trispectrum(cosmo_ccl, hm_recipe, probe_ordering, use_HOD_for_GCph):
    # ! =============================================== halo model =========================================================
    # TODO we're not sure about the values of Delta and rho_type
    # mass_def = ccl.halos.massdef.MassDef(Delta='vir', rho_type='matter', c_m_relation=name)
    # from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.massdef.html?highlight=.halos.massdef.MassDef#pyccl.halos.massdef.MassDef200c

    # about the mass definition, the paper says:
    # "Throughout this paper we define halo properties using the over density ∆ = 200 ¯ρ, with ¯ρ the mean matter density"
    halomod_start_time = time.perf_counter()
    # mass definition
    if hm_recipe == 'KiDS1000':  # arXiv:2007.01844
        c_m = 'Duffy08'
        mass_def = ccl.halos.MassDef200c(c_m=c_m)  # ! testing 200c
        c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=mass_def)
    elif hm_recipe == 'Krause2017':  # arXiv:1601.05779
        c_m = 'Bhattacharya13'  # see paper, after Eq. 1
        mass_def = ccl.halos.MassDef200c(c_m=c_m)  # ! testing 200c
        c_M_relation = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=mass_def)  # above Eq. 12
    else:
        raise ValueError('Wrong choice of hm_recipe: it must be either "KiDS1000" or "Krause2017".')

    halo_mass_func = ccl.halos.hmfunc.MassFuncTinker10(cosmo_ccl, mass_def=mass_def, mass_def_strict=True)
    halo_bias_func = ccl.halos.hbias.HaloBiasTinker10(cosmo_ccl, mass_def=mass_def, mass_def_strict=True)
    hm_calculator = ccl.halos.halo_model.HMCalculator(cosmo_ccl, massfunc=halo_mass_func, hbias=halo_bias_func,
                                                      mass_def=mass_def)
    halo_profile_nfw = ccl.halos.profiles.HaloProfileNFW(c_M_relation=c_M_relation)
    halo_profile_hod = ccl.halos.profiles.HaloProfileHOD(c_M_relation=c_M_relation)

    # old, not a dict
    # tkka = ccl.halos.halo_model.halomod_Tk3D_SSC(cosmo_ccl, hm_calculator,
    #                                              prof1=halo_profile_nfw,
    #                                              prof2=None,
    #                                              prof3=None,
    #                                              prof4=None,
    #                                              prof12_2pt=None,
    #                                              prof34_2pt=None,
    #                                              normprof1=True, normprof2=True,
    #                                              normprof3=True, normprof4=True,
    #                                              p_of_k_a=None, lk_arr=None, a_arr=None, extrap_order_lok=1,
    #                                              extrap_order_hik=1, use_log=False)
    # TODO a_arr as in latest version
    # TODO pk from input files

    if use_HOD_for_GCph:
        # this is the correct way to initialize the trispectrum, but the code does not run.
        # Asked David Alonso about this.
        halo_profile_dict = {
            'L': halo_profile_nfw,
            'G': halo_profile_hod,
        }

        prof_2pt_dict = {
            ('L', 'L'): ccl.halos.Profile2pt(),
            ('G', 'L'): ccl.halos.Profile2pt(),
            # see again https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
            ('G', 'G'): ccl.halos.Profile2ptHOD(),
        }
    else:
        warnings.warn('using the same halo profile (NFW) for all probes, this is not quite correct')
        halo_profile_dict = {
            'L': halo_profile_nfw,
            'G': halo_profile_nfw,
        }

        prof_2pt_dict = {
            ('L', 'L'): None,
            ('G', 'L'): None,
            ('G', 'G'): None,
        }

    # store the trispectrum for the various probes in a dictionary
    tkka_dict = {}

    for A, B in probe_ordering:
        for C, D in probe_ordering:
            print(f'Computing tkka for {A}{B}{C}{D}')
            tkka_dict[A, B, C, D] = ccl.halos.halomod_Tk3D_SSC(cosmo=cosmo_ccl, hmc=hm_calculator,
                                                               prof1=halo_profile_dict[A],
                                                               prof2=halo_profile_dict[B],
                                                               prof3=halo_profile_dict[C],
                                                               prof4=halo_profile_dict[D],
                                                               prof12_2pt=prof_2pt_dict[A, B],
                                                               prof34_2pt=prof_2pt_dict[C, D],
                                                               normprof1=True, normprof2=True,
                                                               normprof3=True, normprof4=True,
                                                               # lk_arr=None, a_arr=a_grid_increasing_for_ttka,
                                                               lk_arr=None, a_arr=None,
                                                               p_of_k_a=None)

    # assert False, 'should I use HaloProfileHOD for number counts???'  # TODO
    print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - halomod_start_time))
    return tkka_dict


def compute_ng_cov_ccl(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                       ind_AB, ind_CD, which_ng_cov, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]
    nbl = len(ell)

    # TODO switch off the integration method and see if it crashes

    start_time = time.perf_counter()

    if which_ng_cov == 'SSC':
        cov_ng_4D = Parallel(n_jobs=-1, backend='threading')(
            delayed(ccl.covariances.angular_cl_cov_SSC)(cosmo,
                                                        cltracer1=kernel_A[ind_AB[ij, -2]],
                                                        cltracer2=kernel_B[ind_AB[ij, -1]],
                                                        ell=ell, tkka=tkka,
                                                        sigma2_B=None, fsky=f_sky,
                                                        cltracer3=kernel_C[ind_CD[kl, -2]],
                                                        cltracer4=kernel_D[ind_CD[kl, -1]],
                                                        ell2=None,
                                                        integration_method=integration_method)
            for kl in tqdm(range(zpairs_CD))  # outer loop
            for ij in range(zpairs_AB))  # inner loop; the trasponsition below fixes things

    elif which_ng_cov == 'cNG':
        cov_ng_4D = Parallel(n_jobs=-1, backend='threading')(
            delayed(ccl.covariances.angular_cl_cov_cNG)(cosmo,
                                                        cltracer1=kernel_A[ind_AB[ij, -2]],
                                                        cltracer2=kernel_B[ind_AB[ij, -1]],
                                                        ell=ell, tkka=tkka, fsky=f_sky,
                                                        cltracer3=kernel_C[ind_CD[kl, -2]],
                                                        cltracer4=kernel_D[ind_CD[kl, -1]],
                                                        ell2=None,
                                                        integration_method=integration_method)
            for kl in tqdm(range(zpairs_CD))
            for ij in range(zpairs_AB))
    else:
        raise ValueError('which_ng_cov must be either SSC or cNG')

    print(f'{which_ng_cov} computed with pyccl in {(time.perf_counter() - start_time):.2} seconds')

    cov_ng_4D = np.array(cov_ng_4D).transpose(1, 2, 0).reshape(nbl, nbl, zpairs_AB, zpairs_CD)

    return cov_ng_4D


def compute_3x2pt_PyCCL(cosmo, kernel_dict, ell, tkka_dict, f_sky, integration_method,
                        probe_ordering, ind_dict, which_ng_cov, output_4D_array):
    cov_ng_3x2pt_dict_8D = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            print('3x2pt: working on probe combination ', A, B, C, D)
            cov_ng_3x2pt_dict_8D[A, B, C, D] = compute_ng_cov_ccl(cosmo=cosmo,
                                                                  kernel_A=kernel_dict[A],
                                                                  kernel_B=kernel_dict[B],
                                                                  kernel_C=kernel_dict[C],
                                                                  kernel_D=kernel_dict[D],
                                                                  ell=ell, tkka=tkka_dict[A, B, C, D],
                                                                  f_sky=f_sky,
                                                                  ind_AB=ind_dict[A + B],
                                                                  ind_CD=ind_dict[C + D],
                                                                  which_ng_cov=which_ng_cov,
                                                                  integration_method=integration_method)

    if output_4D_array:
        return mm.cov_3x2pt_8D_dict_to_4D(cov_ng_3x2pt_dict_8D, probe_ordering)

    return cov_ng_3x2pt_dict_8D


def compute_cov_ng_with_pyccl(probe, which_ng_cov, ell_grid, z_grid_nofz, n_of_z, general_cfg, covariance_cfg):
    # ! settings
    zbins = general_cfg['zbins']
    f_sky = covariance_cfg['fsky']
    ind = covariance_cfg['ind']
    GL_or_LG = covariance_cfg['GL_or_LG']
    nbl = len(ell_grid)

    pyccl_cfg = covariance_cfg['pyccl_cfg']
    hm_recipe = pyccl_cfg['hm_recipe']
    z_grid = np.linspace(pyccl_cfg['z_grid_min'], pyccl_cfg['z_grid_max'], pyccl_cfg['z_grid_steps'])
    n_samples_wf = pyccl_cfg['n_samples_wf']
    get_3xtpt_cov_in_4D = pyccl_cfg['get_3xtpt_cov_in_4D']
    bias_model = pyccl_cfg['bias_model']
    use_HOD_for_GCph = pyccl_cfg['use_HOD_for_GCph']
    # ! settings

    # just a check on the settings
    print(f'\n****************** settings ****************'
          f'\nprobe = {probe}\nwhich_ng_cov = {which_ng_cov}'
          f'\nintegration_method = {integration_method_dict[probe][which_ng_cov]}'
          f'\nnbl = {nbl}\nhm_recipe = {hm_recipe}'
          f'\n********************************************\n')

    assert probe in ['LL', 'GG', '3x2pt'], 'probe must be either LL, GG, or 3x2pt'
    assert which_ng_cov in ['SSC', 'cNG'], 'which_ng_cov must be either SSC or cNG'
    assert GL_or_LG == 'GL', 'you should update ind_cross (used in ind_dict) for GL, but we work with GL...'

    # TODO plot kernels and cls to check that they make sense

    # get number of redshift pairs
    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
    ind_auto = ind[:zpairs_auto, :]
    ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

    # ! compute cls, just as a test

    # Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
    # functions
    # TODO this should be generalized to any set of cosmo params
    cosmo_ccl = wf_cl_lib.instantiate_ISTFfid_PyCCL_cosmo_obj()

    # TODO input n(z)
    # source redshift distribution, default ISTF values for bin edges & analytical prescription for the moment

    if z_grid_nofz is None and n_of_z is None:
        print('using default ISTF analytical n(z) values')
        niz_unnormalized_arr = np.asarray(
            [wf_cl_lib.niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
        niz_normalized_arr = wf_cl_lib.normalize_niz_simps(niz_unnormalized_arr, z_grid).T
        n_of_z = niz_normalized_arr

    assert n_of_z.shape == (len(z_grid), zbins), 'n_of_z must be a 2D array with shape (len(z_grid_nofz), zbins)'

    # galaxy bias
    galaxy_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values=None, z_values=None, zbins=zbins,
                                                              z_grid=z_grid, bias_model=bias_model,
                                                              plot_bias=False)

    # IA bias
    ia_bias_1d_array = wf_cl_lib.build_IA_bias_1d_arr(z_grid, input_lumin_ratio=None, cosmo=cosmo_ccl,
                                                      A_IA=None, eta_IA=None, beta_IA=None, C_IA=None,
                                                      growth_factor=None,
                                                      Omega_m=None)

    # # ! compute tracer objects
    wf_lensing = [ccl.tracers.WeakLensingTracer(cosmo_ccl, dndz=(z_grid, n_of_z[:, zbin_idx]),
                                                ia_bias=(z_grid, ia_bias_1d_array), use_A_ia=False,
                                                n_samples=n_samples_wf)
                  for zbin_idx in range(zbins)]

    wf_galaxy = [ccl.tracers.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z_grid, n_of_z[:, zbin_idx]),
                                                bias=(z_grid, galaxy_bias_2d_array[:, zbin_idx]),
                                                mag_bias=None, n_samples=n_samples_wf)
                 for zbin_idx in range(zbins)]

    # fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10, 4))
    # plt.title('Tracer objects')
    wf_lensing_arr = wf_cl_lib.ccl_tracer_obj_to_arr(z_grid, wf_lensing, cosmo_ccl)

    # TODO finish plotting this

    # the cls are not needed, but just in case:
    # cl_LL_3D = wf_cl_lib.cl_PyCCL(wf_lensing, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
    # cl_GL_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
    # cl_GG_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_galaxy, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)

    # covariance ordering stuff
    if probe == 'LL':
        probe_ordering = (('L', 'L'),)
    elif probe == 'GG':
        probe_ordering = (('G', 'G'),)
    elif probe == '3x2pt':

        probe_ordering = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))
        # probe_ordering = (('G', 'L'), ) for testing 3x2pt GLGL, which seems a problematic case.

    # convenience dictionaries
    ind_dict = {
        'LL': ind_auto,
        'GL': ind_cross,
        'GG': ind_auto,
    }

    kernel_dict = {
        'L': wf_lensing,
        'G': wf_galaxy
    }

    # ! =============================================== compute covs ===============================================

    tkka_dict = initialize_trispectrum(cosmo_ccl, hm_recipe, probe_ordering, use_HOD_for_GCph)

    if probe in ['LL', 'GG']:

        kernel_A = kernel_dict[probe[0]]
        kernel_B = kernel_dict[probe[1]]
        kernel_C = kernel_dict[probe[0]]
        kernel_D = kernel_dict[probe[1]]
        ind_AB = ind_dict[probe[0] + probe[1]]
        ind_CD = ind_dict[probe[0] + probe[1]]

        cov_ng_4D = compute_ng_cov_ccl(cosmo=cosmo_ccl,
                                       kernel_A=kernel_A,
                                       kernel_B=kernel_B,
                                       kernel_C=kernel_C,
                                       kernel_D=kernel_D,
                                       ell=ell_grid, tkka=tkka_dict[probe[0], probe[1], probe[0], probe[1]], f_sky=f_sky,
                                       ind_AB=ind_AB,
                                       ind_CD=ind_CD,
                                       which_ng_cov=which_ng_cov,
                                       integration_method=integration_method_dict[probe][which_ng_cov])

    elif probe == '3x2pt':
        # TODO remove this if statement and use the same code for all probes
        cov_ng_4D = compute_3x2pt_PyCCL(cosmo=cosmo_ccl,
                                        kernel_dict=kernel_dict,
                                        ell=ell_grid, tkka_dict=tkka_dict, f_sky=f_sky,
                                        probe_ordering=probe_ordering,
                                        ind_dict=ind_dict,
                                        output_4D_array=get_3xtpt_cov_in_4D,
                                        which_ng_cov=which_ng_cov,
                                        integration_method=integration_method_dict[probe][which_ng_cov])

    else:
        raise ValueError('probe must be either LL, GG, or 3x2pt')

    # test if cov is symmetric in ell1, ell2
    np.testing.assert_allclose(cov_ng_4D, np.transpose(cov_ng_4D, (1, 0, 2, 3)), rtol=1e-6, atol=0)

    return cov_ng_4D


# integration_method_dict = {
#     'LL': {
#         'SSC': 'spline',
#         'cNG': 'spline',
#     },
#     'GG': {
#         'SSC': 'qag_quad',
#         'cNG': 'qag_quad',
#     },
#     '3x2pt': {
#         'SSC': 'qag_quad',
#         'cNG': 'spline',
#     }
# }


integration_method_dict = {
    'LL': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    },
    'GG': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    },
    '3x2pt': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    }
}
