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
def initialize_trispectrum(cosmo_ccl, which_ng_cov, probe_ordering, pyccl_cfg, p_of_k_a, which_pk):
    use_hod_for_gg = pyccl_cfg['use_HOD_for_GCph']
    z_grid_tkka = np.linspace(pyccl_cfg['z_grid_tkka_min'], pyccl_cfg['z_grid_tkka_max'],
                              pyccl_cfg['z_grid_tkka_steps'])
    a_grid_increasing_for_ttka = cosmo_lib.z_to_a(z_grid_tkka)[::-1]

    # from https://github.com/LSSTDESC/CCL/blob/4df2a29eca58d7cd171bc1986e059fd35f425d45/benchmarks/test_covariances.py
    # see also https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/tools/covariance_NG.py#L282
    halomod_start_time = time.perf_counter()

    # breakpoint()
    mass_def = ccl.halos.MassDef200m
    c_M_relation = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
    hmf = ccl.halos.MassFuncTinker10(mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mass_def)
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf, mass_def=mass_def)
    halo_profile_nfw = ccl.halos.HaloProfileNFW(mass_def=mass_def, concentration=c_M_relation)
    halo_profile_hod = ccl.halos.HaloProfileHOD(mass_def=mass_def, concentration=c_M_relation)

    # TODO pk from input files
    assert use_hod_for_gg, ('you need to use HOD for GG to get correct results for GCph! I previously got an error, '
                            'should be fixed now')

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

    if which_ng_cov == 'SSC':
        tkka_func = ccl.halos.halomod_Tk3D_SSC
    elif which_ng_cov == 'cNG':
        tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_cNG
    else:
        raise ValueError(f"Invalid value for which_ng_cov. It is {which_ng_cov}, must be 'SSC' or 'cNG'.")

    for row, (A, B) in enumerate(probe_ordering):
        for col, (C, D) in enumerate(probe_ordering):
            if col >= row:
                print(f'Computing trispectrum for {which_ng_cov}, npoints = {a_grid_increasing_for_ttka.size}, probe combination {A}{B}{C}{D}')
                tkka_dict[A, B, C, D] = tkka_func(cosmo=cosmo_ccl, hmc=hmc,
                                                  prof=halo_profile_dict[A],
                                                  prof2=halo_profile_dict[B],
                                                  prof3=halo_profile_dict[C],
                                                  prof4=halo_profile_dict[D],
                                                  prof12_2pt=prof_2pt_dict[A, B],
                                                  prof13_2pt=prof_2pt_dict[A, B],
                                                  prof14_2pt=prof_2pt_dict[C, D],
                                                  prof24_2pt=prof_2pt_dict[C, D],
                                                  prof34_2pt=None, p_of_k_a=None, lk_arr=None,
                                                  a_arr=a_grid_increasing_for_ttka,
                                                  extrap_order_lok=1, extrap_order_hik=1, use_log=False)

    print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - halomod_start_time))
    if pyccl_cfg['save_trispectrum']:
        trispectrum_filename = pyccl_cfg['trispectrum_filename'].format(which_ng_cov=which_ng_cov, which_pk=which_pk)
        mm.save_pickle(trispectrum_filename, tkka_dict)

    # TODO pass lk_arr?
    # TODO do they interpolate existing tracer arrays?
    # TODO spline for SSC...
    # TODO update to halomod_Tk3D_cNG with pyccl v3.0.0

    return tkka_dict


def compute_ng_cov_ccl(cosmo, which_ng_cov, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                       ind_AB, ind_CD, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]
    nbl = len(ell)

    start_time = time.perf_counter()
    # switch between the two functions, which are identical except for the sigma2_B argument
    func_map = {
        'SSC': 'angular_cl_cov_SSC',
        'cNG': 'angular_cl_cov_cNG'
    }
    if which_ng_cov not in func_map.keys():
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


def compute_ng_cov_3x2pt(cosmo, which_ng_cov, kernel_dict, ell, tkka_dict, f_sky, integration_method,
                         probe_ordering, ind_dict, covariance_cfg, output_4D_array):
    cov_ng_3x2pt_dict_8D = {}

    for row, (probe_a, probe_b) in enumerate(probe_ordering):
        for col, (probe_c, probe_d) in enumerate(probe_ordering):
            if col >= row:

                print('3x2pt: working on probe combination ', probe_a, probe_b, probe_c, probe_d)
                cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                    compute_ng_cov_ccl(cosmo=cosmo,
                                       kernel_A=kernel_dict[probe_a],
                                       kernel_B=kernel_dict[probe_b],
                                       kernel_C=kernel_dict[probe_c],
                                       kernel_D=kernel_dict[probe_d],
                                       ell=ell,
                                       tkka=tkka_dict[probe_a, probe_b, probe_c, probe_d],
                                       f_sky=f_sky,
                                       ind_AB=ind_dict[probe_a + probe_b],
                                       ind_CD=ind_dict[probe_c + probe_d],
                                       which_ng_cov=which_ng_cov,
                                       integration_method=integration_method,
                                       ))

                # save only the upper triangle blocks
                if covariance_cfg['PyCCL_cfg']['save_cov']:
                    cov_path = covariance_cfg['PyCCL_cfg']['cov_path']
                    cov_filename = covariance_cfg['PyCCL_cfg']['cov_filename'].format(probe_a=probe_a, probe_b=probe_b,
                                                                                      probe_c=probe_c, probe_d=probe_d)
                    np.savez_compressed(
                        f'{cov_path}/{cov_filename}', cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d])

            else:
                print('3x2pt: skipping probe combination ', probe_a, probe_b, probe_c, probe_d)
                cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                    cov_ng_3x2pt_dict_8D[probe_c, probe_d, probe_a, probe_b].transpose(1, 0, 3, 2))

    if output_4D_array:
        return mm.cov_3x2pt_8D_dict_to_4D(cov_ng_3x2pt_dict_8D, probe_ordering)

    return cov_ng_3x2pt_dict_8D


def compute_cov_ng_with_pyccl(fiducial_pars_dict, probe, which_ng_cov, ell_grid, general_cfg,
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
    print(f'\n****************** ccl settings ****************'
          f'\nprobe = {probe}\nwhich_ng_cov = {which_ng_cov}'
          f'\nintegration_method = {integration_method_dict[probe][which_ng_cov]}'
          f'\nnbl = {nbl}\nf_sky = {f_sky}\nzbins = {zbins}'
          f'\n************************************************\n')

    assert probe in ['LL', 'GG', '3x2pt'], 'probe must be either LL, GG, or 3x2pt'
    assert which_ng_cov in ['SSC', 'cNG'], 'which_ng_cov must be either SSC or cNG'
    assert GL_or_LG == 'GL', 'you should update ind_cross (used in ind_dict) for GL, but we work with GL...'

    # get number of redshift pairs
    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
    ind_auto = ind[:zpairs_auto, :]
    ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

    # Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
    # functions
    flat_fid_pars_dict = mm.flatten_dict(fiducial_pars_dict)
    cosmo_dict_ccl = cosmo_lib.map_keys(flat_fid_pars_dict, key_mapping=None)
    cosmo_ccl = cosmo_lib.instantiate_cosmo_ccl_obj(cosmo_dict_ccl,
                                                    fiducial_pars_dict['other_params']['camb_extra_parameters'])

    assert isinstance(nz_tuple, tuple), 'nz_tuple must be a tuple'
    assert nz_tuple[0].shape == z_grid.shape, 'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'
    assert nz_tuple[1].shape == (len(z_grid), zbins), 'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'

    # new kernel stuff
    zgrid_nz = nz_tuple[0]

    # ! ccl kernels
    ia_bias_1d = wf_cl_lib.build_ia_bias_1d_arr(zgrid_nz, cosmo_ccl=cosmo_ccl, flat_fid_pars_dict=flat_fid_pars_dict,
                                                input_z_grid_lumin_ratio=None,
                                                input_lumin_ratio=None, output_F_IA_of_z=False)
    ia_bias_tuple = (zgrid_nz, ia_bias_1d)

    maglim = general_cfg['magcut_source'] / 10
    gal_bias_1d = wf_cl_lib.b_of_z_fs2_fit(zgrid_nz, maglim=maglim)
    # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
    gal_bias_2d = np.repeat(gal_bias_1d.reshape(1, -1), zbins, axis=0).T
    gal_bias_tuple = (zgrid_nz, gal_bias_2d)

    # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
    mag_bias_1d = wf_cl_lib.s_of_z_fs2_fit(zgrid_nz, maglim=maglim, poly_fit_values=None)
    mag_bias_2d = np.repeat(mag_bias_1d.reshape(1, -1), zbins, axis=0).T
    mag_bias_tuple = (zgrid_nz, mag_bias_2d)

    if covariance_cfg['shift_nz']:
        warnings.warn('assuming that the shift is in the WL bins')
        dz_shifts = np.array([flat_fid_pars_dict[f'dzWL{zi:02d}'] for zi in range(1, zbins + 1)])
        n_of_z = wf_cl_lib.shift_nz(zgrid_nz, nz_tuple[1], dz_shifts, normalize=True, plot_nz=False,
                                    interpolation_kind='linear')
        nz_tuple = (zgrid_nz, n_of_z)

    wf_lensing_obj = wf_cl_lib.wf_ccl(zgrid_nz, 'lensing', 'with_IA', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                      ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                      mag_bias_tuple=mag_bias_tuple, return_ccl_obj=True, n_samples=1000)
    wf_lensing_arr = wf_cl_lib.wf_ccl(zgrid_nz, 'lensing', 'with_IA', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                      ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                      mag_bias_tuple=mag_bias_tuple, return_ccl_obj=False, n_samples=1000)
    wf_galaxy_obj = wf_cl_lib.wf_ccl(zgrid_nz, 'galaxy', 'with_galaxy_bias', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                     ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                     mag_bias_tuple=mag_bias_tuple, return_ccl_obj=True, n_samples=1000)

    # ! manually construct galaxy = delta + magnification radial kernel
    a_arr = cosmo_lib.z_to_a(z_grid)
    comoving_distance = ccl.comoving_radial_distance(cosmo_ccl, a_arr)
    wf_galaxy_tot_arr = np.asarray([wf_galaxy_obj[zbin_idx].get_kernel(comoving_distance) for zbin_idx in range(zbins)])
    wf_delta_arr = wf_galaxy_tot_arr[:, 0, :].T
    wf_mu_arr = wf_galaxy_tot_arr[:, 1, :].T
    wf_galaxy_arr = wf_delta_arr + wf_mu_arr

    # alternative way to get the magnification kernel
    wf_mu_tot_alt_arr = -2 * np.array(
        [ccl.tracers.get_lensing_kernel(cosmo=cosmo_ccl, dndz=(nz_tuple[0], nz_tuple[1][:, zi]),
                                        mag_bias=(mag_bias_tuple[0], mag_bias_tuple[1][:, zi]),
                                        n_chi=1000)
         for zi in range(zbins)])
    wf_mu_alt_arr = wf_mu_tot_alt_arr[:, 1, :].T

    # ! import Vincenzo's kernels and compare
    wf_lensing_import = general_cfg['wf_WL']
    wf_galaxy_import = general_cfg['wf_GC']
    wf_delta_import = general_cfg['wf_delta']
    wf_mu_import = general_cfg['wf_mu']
    z_grid_wf_import = general_cfg['z_grid_wf']

    colors = cm.rainbow(np.linspace(0, 1, zbins))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for zi in range(zbins):
        ax[0].plot(z_grid, wf_lensing_arr[:, zi], ls="-", c=colors[zi], alpha=0.6,
                   label='lensing ccl' if zi == 0 else None)
        ax[1].plot(z_grid, wf_galaxy_arr[:, zi], ls="-", c=colors[zi], alpha=0.6,
                   label='galaxy ccl' if zi == 0 else None)
        ax[0].plot(z_grid_wf_import, wf_lensing_import[:, zi], ls="--", c=colors[zi], alpha=0.6,
                   label='lensing vinc' if zi == 0 else None)
        ax[1].plot(z_grid_wf_import, wf_galaxy_import[:, zi], ls="--", c=colors[zi], alpha=0.6,
                   label='galaxy vinc' if zi == 0 else None)
    # set labels
    ax[0].set_title('lensing kernel')
    ax[1].set_title('galaxy kernel\nno gal bias!')
    ax[0].set_xlabel('$z$')
    ax[1].set_xlabel('$z$')
    ax[0].set_ylabel('wil')
    ax[1].set_ylabel('wig')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # the cls are not needed, but just in case:
    p_of_k_a = 'delta_matter:delta_matter'
    cl_ll_3d = wf_cl_lib.cl_PyCCL(wf_lensing_obj, wf_lensing_obj, ell_grid, zbins, p_of_k_a=p_of_k_a, cosmo=cosmo_ccl, limber_integration_method='spline')
    cl_gl_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_lensing_obj, ell_grid, zbins, p_of_k_a=p_of_k_a, cosmo=cosmo_ccl, limber_integration_method='spline')
    cl_gg_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_galaxy_obj, ell_grid, zbins, p_of_k_a=p_of_k_a, cosmo=cosmo_ccl, limber_integration_method='spline')
    cl_ll_3d_vinc = general_cfg['cl_ll_3d']
    cl_gl_3d_vinc = general_cfg['cl_gl_3d']
    cl_gg_3d_vinc = general_cfg['cl_gg_3d']

    fig, ax = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for zi in range(zbins):
        zj = zi
        ax[0].loglog(ell_grid, cl_ll_3d[:, zi, zj], ls="-", c=colors[zi], alpha=0.6,
                     label='ll' if zi == 0 else None)
        ax[0].loglog(ell_grid, cl_ll_3d_vinc[:29, zi, zj], ls="--", c=colors[zi], alpha=0.6,
                     label='ll vinc' if zi == 0 else None)
        ax[1].loglog(ell_grid, cl_gl_3d[:, zi, zj], ls="-", c=colors[zi], alpha=0.6,
                     label='gl' if zi == 0 else None)
        ax[1].loglog(ell_grid, cl_gl_3d_vinc[:29, zi, zj], ls="--", c=colors[zi], alpha=0.6,
                     label='gl vinc' if zi == 0 else None)
        ax[2].loglog(ell_grid, cl_gg_3d[:, zi, zj], ls="-", c=colors[zi], alpha=0.6,
                     label='gg' if zi == 0 else None)
        ax[2].loglog(ell_grid, cl_gg_3d_vinc[:29, zi, zj], ls="--", c=colors[zi], alpha=0.6,
                     label='gg vinc' if zi == 0 else None)
    # set labels
    ax[0].set_xlabel('$\\ell$')
    ax[1].set_xlabel('$\\ell$')
    ax[2].set_xlabel('$\\ell$')
    ax[0].set_ylabel('cl_ll')
    ax[1].set_ylabel('cl_gl')
    ax[2].set_ylabel('cl_gg')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

    # covariance ordering stuff, also used to compute the trispectrum
    if probe == 'LL':
        probe_ordering = (('L', 'L'),)
    elif probe == 'GG':
        probe_ordering = (('G', 'G'),)
    elif probe == '3x2pt':
        # probe_ordering = covariance_cfg['probe_ordering']
        warnings.warn('TESTING ONLY GLGL TO DEBUG 3X2PT cNG')
        probe_ordering = (('G', 'L'),)  # for testing 3x2pt GLGL, which seems a problematic case.
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
    which_pk = fiducial_pars_dict['other_params']['camb_extra_parameters']['camb']['halofit_version']
    tkka_dict = initialize_trispectrum(cosmo_ccl, which_ng_cov, probe_ordering, pyccl_cfg, p_of_k_a=None,
                                       which_pk=which_pk)

    if probe in ['LL', 'GG']:

        kernel_A = kernel_dict[probe[0]]
        kernel_B = kernel_dict[probe[1]]
        kernel_C = kernel_dict[probe[0]]
        kernel_D = kernel_dict[probe[1]]
        ind_AB = ind_dict[probe[0] + probe[1]]
        ind_CD = ind_dict[probe[0] + probe[1]]

        cov_ng_4D = compute_ng_cov_ccl(cosmo=cosmo_ccl,
                                       which_ng_cov=which_ng_cov,
                                       kernel_A=kernel_A,
                                       kernel_B=kernel_B,
                                       kernel_C=kernel_C,
                                       kernel_D=kernel_D,
                                       ell=ell_grid, tkka=tkka_dict[probe[0], probe[1], probe[0], probe[1]],
                                       f_sky=f_sky,
                                       ind_AB=ind_AB,
                                       ind_CD=ind_CD,
                                       integration_method=integration_method_dict[probe][which_ng_cov],
                                       )

    elif probe == '3x2pt':
        # TODO remove this if statement and use the same code for all probes
        cov_ng_4D = compute_ng_cov_3x2pt(cosmo=cosmo_ccl,
                                         which_ng_cov=which_ng_cov,
                                         kernel_dict=kernel_dict,
                                         ell=ell_grid, tkka_dict=tkka_dict, f_sky=f_sky,
                                         probe_ordering=probe_ordering,
                                         ind_dict=ind_dict,
                                         output_4D_array=get_3x2pt_cov_in_4D,
                                         covariance_cfg=covariance_cfg,
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
        'SSC': 'spline',
        'cNG': 'spline',
    }
}
