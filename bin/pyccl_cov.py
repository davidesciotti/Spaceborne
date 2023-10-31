import pdb
import pickle
import sys
import time
import warnings
from pathlib import Path
import ipdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import yaml
from joblib import Parallel, delayed
from matplotlib import cm
from scipy.special import erf
import ray
from tqdm import tqdm
from matplotlib.lines import Line2D

ray.shutdown()
ray.init()

# get project directory adn import useful modules
project_path = Path.cwd().parent

sys.path.append(f'../../common_lib_and_cfg')
import common_lib.my_module as mm
import common_lib.cosmo_lib as cosmo_lib
import common_lib.wf_cl_lib as wf_cl_lib
import common_cfg.mpl_cfg as mpl_cfg
import common_cfg.ISTF_fid_params as ISTF_fid

matplotlib.use('Qt5Agg')
start_time = time.perf_counter()
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)

""" This is run with v 2.7 of pyccl
"""


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
def initialize_trispectrum(cosmo_ccl, probe_ordering, pyccl_cfg, p_of_k_a):
    use_hod_for_gg = pyccl_cfg['use_HOD_for_GCph']
    z_grid_tkka = np.linspace(pyccl_cfg['z_grid_tkka_min'], pyccl_cfg['z_grid_tkka_max'],
                              pyccl_cfg['z_grid_tkka_steps'])
    a_grid_increasing_for_ttka = cosmo_lib.z_to_a(z_grid_tkka)[::-1]

    # from https://github.com/LSSTDESC/CCL/blob/4df2a29eca58d7cd171bc1986e059fd35f425d45/benchmarks/test_covariances.py
    # see also https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/tools/covariance_NG.py#L282
    halomod_start_time = time.perf_counter()
    mass_def = ccl.halos.MassDef200m()
    c_M_relation = ccl.halos.ConcentrationDuffy08(mass_def)
    hmf = ccl.halos.MassFuncTinker10(cosmo_ccl, mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(cosmo_ccl, mass_def=mass_def)
    hmc = ccl.halos.HMCalculator(cosmo_ccl, hmf, hbf, mass_def)
    halo_profile_nfw = ccl.halos.HaloProfileNFW(c_M_relation, fourier_analytic=True)
    halo_profile_hod = ccl.halos.HaloProfileHOD(c_M_relation=c_M_relation)

    # TODO pk from input files

    if use_hod_for_gg:
        # This is the correct way to initialize the trispectrum, but the code does not run.
        # Asked David Alonso about this.
        halo_profile_dict = {
            'L': halo_profile_nfw,
            'G': halo_profile_hod,
        }

        prof_2pt_dict = {
            # see again https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
            ('L', 'L'): ccl.halos.Profile2pt(),
            ('G', 'L'): ccl.halos.Profile2pt(),
            ('G', 'G'): ccl.halos.Profile2ptHOD(),
        }

    else:
        warnings.warn('!!! using the same halo profile (NFW) for all probes, this produces wrong results for GCph!!')
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
            tkka_dict[A, B, C, D] = ccl.halos.halomod_Tk3D_SSC(cosmo=cosmo_ccl, hmc=hmc,
                                                               prof1=halo_profile_dict[A],
                                                               prof2=halo_profile_dict[B],
                                                               prof3=halo_profile_dict[C],
                                                               prof4=halo_profile_dict[D],
                                                               prof12_2pt=prof_2pt_dict[A, B],
                                                               prof34_2pt=prof_2pt_dict[C, D],
                                                               normprof1=True, normprof2=True,
                                                               normprof3=True, normprof4=True,
                                                               lk_arr=None, a_arr=a_grid_increasing_for_ttka,
                                                               p_of_k_a=p_of_k_a)

    print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - halomod_start_time))
    return tkka_dict


def compute_ng_cov_ccl(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                       ind_AB, ind_CD, which_ng_cov, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]
    nbl = len(ell)
    zbins = len(kernel_A)

    # TODO switch off the integration method and see if it crashes

    start_time = time.perf_counter()

    # switch between the two functions, which are identical except for the sigma2_B argument
    func_map = {
        'SSC': 'angular_cl_cov_SSC',
        'cNG': 'angular_cl_cov_cNG'
    }
    if which_ng_cov not in func_map:
        raise ValueError("Invalid value for which_ng_cov. Must be 'SSC' or 'cNG'.")
    func_to_call = getattr(ccl.covariances, func_map[which_ng_cov])
    sigma2_B_arg = {'sigma2_B': None} if which_ng_cov == 'SSC' else {}

    cov_ng_4D = Parallel(n_jobs=-1, backend='threading')(
        delayed(func_to_call)(cosmo,
                              cltracer1=kernel_A[ind_AB[ij, -2]],
                              cltracer2=kernel_B[ind_AB[ij, -1]],
                              ell=ell,
                              tkka=tkka,
                              fsky=f_sky,
                              cltracer3=kernel_C[ind_CD[kl, -2]],
                              cltracer4=kernel_D[ind_CD[kl, -1]],
                              ell2=None,
                              integration_method=integration_method,
                              **sigma2_B_arg)
        for ij in tqdm(range(zpairs_AB))
        for kl in range(zpairs_CD)
    )
    # this is to move ell1, ell2 to the first axes and unpack the result in two separate dimensions
    cov_ng_4D = np.array(cov_ng_4D).transpose(1, 2, 0).reshape(nbl, nbl, zpairs_AB, zpairs_CD)

    print(f'{which_ng_cov} computed with pyccl in {(time.perf_counter() - start_time) / 60:.2} min')

    return cov_ng_4D


def compute_3x2pt_PyCCL(cosmo, kernel_dict, ell, tkka_dict, f_sky, integration_method,
                        probe_ordering, ind_dict, which_ng_cov, output_4D_array):
    cov_ng_3x2pt_dict_8D = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            # TODO optimize this by computing only the upper triangle, then understand the symmetry
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
                                                                  integration_method=integration_method,
                                                                  )

    if output_4D_array:
        return mm.cov_3x2pt_8D_dict_to_4D(cov_ng_3x2pt_dict_8D, probe_ordering)

    return cov_ng_3x2pt_dict_8D


def compute_cov_ng_with_pyccl(flat_fid_pars_dict, probe, which_ng_cov, ell_grid, general_cfg,
                              covariance_cfg):
    # ! settings
    zbins = general_cfg['zbins']
    nz_tuple = general_cfg['nz_tuple']
    f_sky = covariance_cfg['fsky']
    ind = covariance_cfg['ind']
    GL_or_LG = covariance_cfg['GL_or_LG']
    nbl = len(ell_grid)

    pyccl_cfg = covariance_cfg['PyCCL_cfg']
    z_grid = np.linspace(pyccl_cfg['z_grid_min'], pyccl_cfg['z_grid_max'], pyccl_cfg['z_grid_steps'])
    n_samples_wf = pyccl_cfg['n_samples_wf']
    get_3x2pt_cov_in_4D = pyccl_cfg['get_3x2pt_cov_in_4D']  # TODO save all blocks separately
    bias_model = pyccl_cfg['bias_model']
    # ! settings

    # just a check on the settings
    print(f'\n****************** settings ****************'
          f'\nprobe = {probe}\nwhich_ng_cov = {which_ng_cov}'
          f'\nintegration_method = {integration_method_dict[probe][which_ng_cov]}'
          f'\nnbl = {nbl}\nf_sky = {f_sky}\nzbins = {zbins}'
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
    cosmo_ccl = wf_cl_lib.instantiate_cosmo_ccl_obj(flat_fid_pars_dict)

    # TODO input n(z)
    # source redshift distribution, default ISTF values for bin edges & analytical prescription for the moment

    if nz_tuple is None:
        print('using default ISTF analytical n(z) values')
        niz_unnormalized_arr = np.asarray(
            [wf_cl_lib.niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
        niz_normalized_arr = wf_cl_lib.normalize_niz_simps(niz_unnormalized_arr, z_grid).T
        nz_tuple = niz_normalized_arr

    assert nz_tuple.shape == (len(z_grid), zbins), 'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'

    # new kernel stuff
    zgrid_nz = nz_tuple[0]

    # ! my kernels
    ia_bias = wf_cl_lib.build_IA_bias_1d_arr(zgrid_nz, input_z_grid_lumin_ratio=None,
                                             input_lumin_ratio=None,
                                             cosmo=cosmo_ccl,
                                             A_IA=flat_fid_pars_dict['Aia'],
                                             eta_IA=flat_fid_pars_dict['eIA'],
                                             beta_IA=flat_fid_pars_dict['bIA'],
                                             C_IA=None,
                                             growth_factor=None,
                                             output_F_IA_of_z=False)

    warnings.warn('Im not sure the bias is step-wise...')
    maglim = general_cfg['magcut_source'] / 10
    bias_values = wf_cl_lib.b_of_z_fs2_fit(zgrid_nz, maglim=maglim)
    galaxy_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values=bias_values, z_values=zgrid_nz, zbins=zbins,
                                                              z_grid=z_grid, bias_model='step-wise',
                                                              plot_bias=False)

    # Define the keyword arguments as a dictionary
    wil_ccl_kwargs = {
        'cosmo': cosmo_ccl,
        'dndz': nz_tuple,
        'ia_bias': ia_bias,
        'A_IA': flat_fid_pars_dict['Aia'],
        'eta_IA': flat_fid_pars_dict['eIA'],
        'beta_IA': flat_fid_pars_dict['bIA'],
        'C_IA': None,
        'growth_factor': None,
        'return_PyCCL_object': True,
        'n_samples': len(zgrid_nz)
    }
    wig_ccl_kwargs = {
        'gal_bias_2d_array': np.ones((len(zgrid_nz), zbins)),
        'fiducial_params': flat_fid_pars_dict,
        'bias_model': 'step-wise',
        'cosmo': cosmo_ccl,
        'return_PyCCL_object': True,
        'dndz': nz_tuple,
        'n_samples': len(zgrid_nz)
    }

    # Use * to unpack positional arguments and ** to unpack keyword arguments
    wf_lensing_obj = wf_cl_lib.wil_PyCCL(zgrid_nz, 'with_IA', **wil_ccl_kwargs)
    wf_lensing_arr = wf_cl_lib.wil_PyCCL(zgrid_nz, 'with_IA',
                                             **{**wil_ccl_kwargs, 'return_PyCCL_object': False})

    wf_galaxy_obj = wf_cl_lib.wig_PyCCL(zgrid_nz, 'with_galaxy_bias', **wig_ccl_kwargs)
    wf_galaxy_arr = wf_cl_lib.wig_PyCCL(zgrid_nz, 'with_galaxy_bias',
                                            **{**wig_ccl_kwargs, 'return_PyCCL_object': False})

    # end of new kernel stuff
    #
    # # galaxy bias
    # galaxy_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values=None, z_values=None, zbins=zbins,
    #                                                           z_grid=z_grid, bias_model=bias_model,
    #                                                           plot_bias=False)
    #
    # # IA bias
    # ia_bias_1d_array = wf_cl_lib.build_IA_bias_1d_arr(z_grid, input_lumin_ratio=None, cosmo=cosmo_ccl,
    #                                                   A_IA=None, eta_IA=None, beta_IA=None, C_IA=None,
    #                                                   growth_factor=None,
    #                                                   Omega_m=None)
    #
    # # # ! compute tracer objects
    # wf_lensing = [ccl.tracers.WeakLensingTracer(cosmo_ccl, dndz=(z_grid, n_of_z[:, zbin_idx]),
    #                                             ia_bias=(z_grid, ia_bias_1d_array), use_A_ia=False,
    #                                             n_samples=n_samples_wf)
    #               for zbin_idx in range(zbins)]
    #
    # wf_galaxy = [ccl.tracers.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z_grid, n_of_z[:, zbin_idx]),
    #                                             bias=(z_grid, galaxy_bias_2d_array[:, zbin_idx]),
    #                                             mag_bias=None, n_samples=n_samples_wf)
    #              for zbin_idx in range(zbins)]
    #
    # # try to create a tracer object with a tabulated kernel
    # # kernel =
    # # ccl.tracers.add_tracer(cosmo_ccl, *, kernel=None, transfer_ka=None, transfer_k=None, transfer_a=None, der_bessel=0, der_angles=0,
    # #            is_logt=False, extrap_order_lok=0, extrap_order_hik=2)
    #
    # # compare pyccl kernels with the importwd ones (used by PySSC):
    # warnings.warn('THIS MODULE NEEDS TO IMPORT A COSMOLOGY DICT, E.G. HERE THE IA VALUES ARE THE DEFAULT ONES')
    # wf_lensing_arr = wf_cl_lib.wil_PyCCL(z_grid, 'with_IA', cosmo=cosmo_ccl, dndz=(z_grid, n_of_z),
    #                                      ia_bias=(z_grid, ia_bias_1d_array),
    #                                      A_IA=None, eta_IA=None, beta_IA=None, C_IA=None,
    #                                      growth_factor=None,
    #                                      return_PyCCL_object=False,
    #                                      n_samples=n_samples_wf)
    # wf_galaxy_arr = wf_cl_lib.wig_PyCCL(z_grid, 'with_galaxy_bias', gal_bias_2d_array=galaxy_bias_2d_array,
    #                                     fiducial_params=None,
    #                                     bias_model='step-wise',
    #                                     cosmo=cosmo_ccl, return_PyCCL_object=False, dndz=(z_grid, n_of_z),
    #                                     n_samples=n_samples_wf)

    wf_lensing_import = general_cfg['wf_WL']
    wf_galaxy_import = general_cfg['wf_GC']
    z_grid_wf_import = general_cfg['z_grid_wf']

    colors = cm.rainbow(np.linspace(0, 1, zbins))

    # plot them in 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for zi in range(zbins):
        ax[0].plot(z_grid, wf_lensing_arr[:, zi], ls="-", c=colors[zi], alpha=0.7)
        ax[1].plot(z_grid, wf_galaxy_arr[:, zi], ls="-", c=colors[zi], alpha=0.7)
        ax[0].plot(z_grid_wf_import, wf_lensing_import[:, zi], ls="--", c=colors[zi], alpha=0.7)
        ax[1].plot(z_grid_wf_import, wf_galaxy_import[:, zi], ls="--", c=colors[zi], alpha=0.7)
    # set labels
    ax[0].set_title('lensing kernel')
    ax[1].set_title('galaxy kernel')
    ax[0].set_xlabel('z')
    ax[1].set_xlabel('z')
    ax[0].set_ylabel('wil')
    ax[1].set_ylabel('wig')
    # set legend to linestyles
    # Create custom legend
    custom_lines = [Line2D([0], [0], ls='-'),
                    Line2D([0], [0], ls='--')]
    ax[0].legend(custom_lines, ['pyccl'])
    ax[0].legend(custom_lines, ['import'])

    ax[1].legend(custom_lines, ['pyccl'])
    ax[1].legend(custom_lines, ['import'])
    plt.show()

    assert False, 'stop here to check the new kernel implementation with ccl'

    # the cls are not needed, but just in case:
    # cl_LL_3D = wf_cl_lib.cl_PyCCL(wf_lensing, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
    # cl_GL_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
    # cl_GG_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_galaxy, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)

    # covariance ordering stuff, also used to compute the trispectrum
    if probe == 'LL':
        probe_ordering = (('L', 'L'),)
    elif probe == 'GG':
        probe_ordering = (('G', 'G'),)
    elif probe == '3x2pt':
        probe_ordering = covariance_cfg['probe_ordering']
        # probe_ordering = (('G', 'L'), ) # for testing 3x2pt GLGL, which seems a problematic case.
    else:
        raise ValueError('probe must be either LL, GG, or 3x2pt')

    # convenience dictionaries
    ind_dict = {
        'LL': ind_auto,
        'GL': ind_cross,
        'GG': ind_auto,
    }

    kernel_dict = {
        'L': wf_lensing_obj,
        'G': wf_galaxy_obj
    }

    # ! =============================================== compute covs ===============================================

    tkka_dict = initialize_trispectrum(cosmo_ccl, probe_ordering, pyccl_cfg, p_of_k_a=None)

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
                                       ell=ell_grid, tkka=tkka_dict[probe[0], probe[1], probe[0], probe[1]],
                                       f_sky=f_sky,
                                       ind_AB=ind_AB,
                                       ind_CD=ind_CD,
                                       which_ng_cov=which_ng_cov,
                                       integration_method=integration_method_dict[probe][which_ng_cov],
                                       )

    elif probe == '3x2pt':
        # TODO remove this if statement and use the same code for all probes
        cov_ng_4D = compute_3x2pt_PyCCL(cosmo=cosmo_ccl,
                                        kernel_dict=kernel_dict,
                                        ell=ell_grid, tkka_dict=tkka_dict, f_sky=f_sky,
                                        probe_ordering=probe_ordering,
                                        ind_dict=ind_dict,
                                        output_4D_array=get_3x2pt_cov_in_4D,
                                        which_ng_cov=which_ng_cov,
                                        integration_method=integration_method_dict[probe][which_ng_cov],
                                        )

    else:
        raise ValueError('probe must be either LL, GG, or 3x2pt')

    # test if cov is symmetric in ell1, ell2
    # np.testing.assert_allclose(cov_ng_4D, np.transpose(cov_ng_4D, (1, 0, 2, 3)), rtol=1e-6, atol=0)

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
