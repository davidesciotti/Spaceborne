import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import sys
from joblib import Parallel, delayed
from matplotlib import cm
from tqdm import tqdm
import ipdb

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
import bin.cosmo_lib as cosmo_lib
import bin.wf_cl_lib as wf_cl_lib
import common_cfg.mpl_cfg as mpl_cfg


start_time = time.perf_counter()
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)

""" This is run with v 3.0.1 of pyccl
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
def initialize_trispectrum(cosmo_ccl, which_ng_cov, probe_ordering, pyccl_cfg, which_pk):
    use_hod_for_gg = pyccl_cfg['use_HOD_for_GCph']
    z_grid_tkka = np.linspace(pyccl_cfg['z_grid_tkka_min'], pyccl_cfg['z_grid_tkka_max'],
                              pyccl_cfg['z_grid_tkka_steps'])
    a_grid_increasing_for_ttka = cosmo_lib.z_to_a(z_grid_tkka)[::-1]
    logn_k_grid_tkka = np.log(np.geomspace(1E-5, 1E2, 1000))

    # a_grid_increasing_for_ttka = None
    # logn_k_grid_tkka = None

    # from https://github.com/LSSTDESC/CCL/blob/4df2a29eca58d7cd171bc1986e059fd35f425d45/benchmarks/test_covariances.py
    # see also https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/tools/covariance_NG.py#L282
    halomod_start_time = time.perf_counter()

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
            ('L', 'G'): ccl.halos.Profile2pt(),
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

    for row, (A, B) in tqdm(enumerate(probe_ordering)):
        for col, (C, D) in enumerate(probe_ordering):
            if col >= row:
                print(f'Computing trispectrum for {which_ng_cov},  probe combination {A}{B}{C}{D}')
                if a_grid_increasing_for_ttka is not None and logn_k_grid_tkka is not None:
                    print(f'z points = {a_grid_increasing_for_ttka.size}, k points = {logn_k_grid_tkka.size}')

                # not very nice to put this if-else in the for loop, but A, B, C, D are referenced only here
                if which_ng_cov == 'SSC':
                    tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_SSC
                    prof_2pt_args = {}
                elif which_ng_cov == 'cNG':
                    tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_cNG
                    prof_2pt_args = {
                        'prof13_2pt': prof_2pt_dict[A, C],
                        'prof14_2pt': prof_2pt_dict[A, D],
                        'prof24_2pt': prof_2pt_dict[B, D]
                    }
                else:
                    raise ValueError(f"Invalid value for which_ng_cov. It is {which_ng_cov}, must be 'SSC' or 'cNG'.")

                tkka_dict[A, B, C, D] = tkka_func(cosmo=cosmo_ccl,
                                                  hmc=hmc,
                                                  prof=halo_profile_dict[A],
                                                  prof2=halo_profile_dict[B],
                                                  prof3=halo_profile_dict[C],
                                                  prof4=halo_profile_dict[D],
                                                  prof12_2pt=prof_2pt_dict[A, B],
                                                  prof34_2pt=prof_2pt_dict[C, D],
                                                  p_of_k_a=None, lk_arr=logn_k_grid_tkka,
                                                  a_arr=a_grid_increasing_for_ttka,
                                                  extrap_order_lok=1, extrap_order_hik=1, use_log=False,
                                                  probe_block=A + B + C + D,
                                                  **prof_2pt_args)

    print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - halomod_start_time))
    if pyccl_cfg['save_trispectrum']:
        trispectrum_filename = pyccl_cfg['trispectrum_filename'].format(which_ng_cov=which_ng_cov, which_pk=which_pk)
        mm.save_pickle(trispectrum_filename, tkka_dict)

    # TODO do they interpolate existing tracer arrays?
    # TODO spline for SSC...

    return tkka_dict


def compute_ng_cov_ccl(cosmo, which_ng_cov, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                       ind_AB, ind_CD, sigma2_B_tuple, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]
    nbl = len(ell)

    start_time = time.perf_counter()
    # switch between the two functions, which are identical except for the sigma2_B argument
    if which_ng_cov == 'SSC':
        ng_cov_func = ccl.covariances.angular_cl_cov_SSC
        sigma2_B_arg = {'sigma2_B': sigma2_B_tuple}
    elif which_ng_cov == 'cNG':
        ng_cov_func = ccl.covariances.angular_cl_cov_cNG
        sigma2_B_arg = {}
    else:
        raise ValueError("Invalid value for which_ng_cov. Must be 'SSC' or 'cNG'.")

    n_jobs = 1  # 17 min with 32 jobs, 11 min with 1 job...
    print('n_jobs = ', n_jobs)
    cov_ng_4D = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(ng_cov_func)(cosmo,
                             tracer1=kernel_A[ind_AB[ij, -2]],
                             tracer2=kernel_B[ind_AB[ij, -1]],
                             ell=ell,
                             t_of_kk_a=tkka,
                             fsky=f_sky,
                             tracer3=kernel_C[ind_CD[kl, -2]],
                             tracer4=kernel_D[ind_CD[kl, -1]],
                             ell2=None,
                             integration_method=integration_method,
                             **sigma2_B_arg)
        for ij in tqdm(range(zpairs_AB))
        for kl in range(zpairs_CD)
    )
    # this is to move ell1, ell2 to the first axes and unpack the result in two separate dimensions
    cov_ng_4D = np.array(cov_ng_4D).transpose(1, 2, 0).reshape(nbl, nbl, zpairs_AB, zpairs_CD)

    print(f'{which_ng_cov} computed with pyccl in {(time.perf_counter() - start_time) / 60:.0f} min')

    return cov_ng_4D


def compute_ng_cov_3x2pt(cosmo, which_ng_cov, kernel_dict, ell, tkka_dict, f_sky, integration_method,
                         probe_ordering, ind_dict, sigma2_B_tuple, covariance_cfg, output_4D_array):
    cov_ng_3x2pt_dict_8D = {}

    for row, (probe_a, probe_b) in enumerate(probe_ordering):
        for col, (probe_c, probe_d) in enumerate(probe_ordering):
            if col >= row:

                print('3x2pt: working on probe combination ', probe_a, probe_b, probe_c, probe_d)
                cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                    compute_ng_cov_ccl(cosmo=cosmo,
                                       which_ng_cov=which_ng_cov,
                                       kernel_A=kernel_dict[probe_a],
                                       kernel_B=kernel_dict[probe_b],
                                       kernel_C=kernel_dict[probe_c],
                                       kernel_D=kernel_dict[probe_d],
                                       ell=ell,
                                       tkka=tkka_dict[probe_a, probe_b, probe_c, probe_d],
                                       f_sky=f_sky,
                                       ind_AB=ind_dict[probe_a + probe_b],
                                       ind_CD=ind_dict[probe_c + probe_d],
                                       sigma2_B_tuple=sigma2_B_tuple,
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
    # this is needed only for a visual check of the cls, which are not used for SSC anyways
    bias_model = general_cfg['bias_model']
    has_rsd = general_cfg['has_rsd']
    has_magnification_bias = general_cfg['has_magnification_bias']
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
    assert has_rsd == False, 'RSD not implemented yet'

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

    # ! ccl kernels
    zgrid_nz = nz_tuple[0]
    assert isinstance(nz_tuple, tuple), 'nz_tuple must be a tuple'

    assert nz_tuple[0].shape == zgrid_nz.shape, 'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'
    assert nz_tuple[1].shape == (
        len(zgrid_nz), zbins), 'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'

    ia_bias_1d = wf_cl_lib.build_ia_bias_1d_arr(zgrid_nz, cosmo_ccl=cosmo_ccl, flat_fid_pars_dict=flat_fid_pars_dict,
                                                input_z_grid_lumin_ratio=None,
                                                input_lumin_ratio=None, output_F_IA_of_z=False)
    ia_bias_tuple = (zgrid_nz, ia_bias_1d)

    if bias_model == 'SPV3_bias':
        maglim = general_cfg['magcut_source'] / 10
        gal_bias_1d = wf_cl_lib.b_of_z_fs2_fit(zgrid_nz, maglim=maglim)
        # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
        gal_bias_2d = np.repeat(gal_bias_1d.reshape(1, -1), zbins, axis=0).T
    elif bias_model == 'ISTF_bias':
        z_means = np.array([flat_fid_pars_dict[f'zmean{zbin:02d}_photo'] for zbin in range(1, zbins + 1)])
        gal_bias_1d = wf_cl_lib.b_of_z_fs1_pocinofit(z_means)
        gal_bias_2d = wf_cl_lib.build_galaxy_bias_2d_arr(
            gal_bias_1d, z_means, None, zbins, zgrid_nz, bias_model='constant', plot_bias=True)
    else:
        raise ValueError('bias_model must be either SPV3_bias or ISTF_bias')

    gal_bias_tuple = (zgrid_nz, gal_bias_2d)

    if has_magnification_bias:
        # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
        mag_bias_1d = wf_cl_lib.s_of_z_fs2_fit(zgrid_nz, maglim=maglim, poly_fit_values=None)
        mag_bias_2d = np.repeat(mag_bias_1d.reshape(1, -1), zbins, axis=0).T
        mag_bias_tuple = (zgrid_nz, mag_bias_2d)
    else:
        mag_bias_tuple = None

    wf_lensing_obj = wf_cl_lib.wf_ccl(zgrid_nz, 'lensing', 'with_IA', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                      ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                      mag_bias_tuple=mag_bias_tuple, has_rsd=has_rsd, return_ccl_obj=True, n_samples=n_samples_wf)
    wf_lensing_arr = wf_cl_lib.wf_ccl(zgrid_nz, 'lensing', 'with_IA', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                      ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                      mag_bias_tuple=mag_bias_tuple, has_rsd=has_rsd, return_ccl_obj=False, n_samples=n_samples_wf)
    wf_galaxy_obj = wf_cl_lib.wf_ccl(zgrid_nz, 'galaxy', 'with_galaxy_bias', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                     ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                     mag_bias_tuple=mag_bias_tuple, has_rsd=has_rsd, return_ccl_obj=True, n_samples=n_samples_wf)
    # TODO better understand galaxy bias in the plots below and in the ITF signal......
    wf_galaxy_arr = wf_cl_lib.wf_ccl(zgrid_nz, 'galaxy', 'with_galaxy_bias', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                     ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                     mag_bias_tuple=mag_bias_tuple, has_rsd=has_rsd, return_ccl_obj=False, n_samples=n_samples_wf)

    # ! manually construct galaxy = delta + magnification radial kernel
    a_arr = cosmo_lib.z_to_a(zgrid_nz)
    comoving_distance = ccl.comoving_radial_distance(cosmo_ccl, a_arr)
    wf_galaxy_tot_arr = np.asarray([wf_galaxy_obj[zbin_idx].get_kernel(comoving_distance) for zbin_idx in range(zbins)])
    wf_delta_arr = wf_galaxy_tot_arr[:, 0, :].T
    wf_mu_arr = wf_galaxy_tot_arr[:, 1, :].T if has_magnification_bias else np.zeros_like(wf_delta_arr)
    wf_galaxy_arr = wf_delta_arr + wf_mu_arr

    # alternative way to get the magnification kernel
    # wf_mu_tot_alt_arr = -2 * np.array(
    #     [ccl.tracers.get_lensing_kernel(cosmo=cosmo_ccl, dndz=(nz_tuple[0], nz_tuple[1][:, zi]),
    #                                     mag_bias=(mag_bias_tuple[0], mag_bias_tuple[1][:, zi]),
    #                                     n_chi=n_samples_wf)
    #      for zi in range(zbins)])
    # wf_mu_alt_arr = wf_mu_tot_alt_arr[:, 1, :].T

    # ! import Vincenzo's kernels and compare
    wf_lensing_import = general_cfg['wf_WL']
    wf_galaxy_import = general_cfg['wf_GC']
    # wf_delta_import = general_cfg['wf_delta']
    # wf_mu_import = general_cfg['wf_mu']
    z_grid_wf_import = general_cfg['z_grid_wf']

    colors = cm.rainbow(np.linspace(0, 1, zbins))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for zi in range(zbins):
        ax[0].plot(zgrid_nz, wf_lensing_arr[:, zi], ls="-", c=colors[zi], alpha=0.6,
                   label='lensing ccl' if zi == 0 else None)
        ax[1].plot(zgrid_nz, wf_galaxy_arr[:, zi], ls="-", c=colors[zi], alpha=0.6,
                   label='galaxy ccl' if zi == 0 else None)
        ax[0].plot(z_grid_wf_import, wf_lensing_import[:, zi], ls="--", c=colors[zi], alpha=0.6,
                   label='lensing vinc' if zi == 0 else None)
        ax[1].plot(z_grid_wf_import, wf_galaxy_import[:, zi], ls="--", c=colors[zi], alpha=0.6,
                   label='galaxy vinc' if zi == 0 else None)
    # set labels
    ax[0].set_title('lensing kernel')
    ax[1].set_title('galaxy kernel\n(no gal bias!)')
    ax[0].set_xlabel('$z$')
    ax[1].set_xlabel('$z$')
    ax[0].set_ylabel('wil')
    ax[1].set_ylabel('wig')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # the cls are not needed, but just in case:
    p_of_k_a = 'delta_matter:delta_matter'
    # this is a test to use the actual P(k) from the input files, but the agreement gets much worse
    # pk_mm_table = np.genfromtxt(f'/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/'
    #                             'LiFEforSPV3/InputFiles/InputPS/HMCodeBar/'
    #                             'InFiles/Flat/h/PddVsZedLogK-h_6.700e-01.dat')
    # # reshape pk
    # z_grid_Pk = np.unique(pk_mm_table[:, 0])
    # k_grid_Pk = np.unique(10 ** pk_mm_table[:, 1])
    # pk_mm_2d = np.reshape(pk_mm_table[:, 2], (len(z_grid_Pk), len(k_grid_Pk))).T  # I want P(k, z), not P(z, k)
    # scale_factor_grid_pk = cosmo_lib.z_to_a(z_grid_Pk)  # flip it
    # p_of_k_a = ccl.pk2d.Pk2D(a_arr=scale_factor_grid_pk, lk_arr=np.log(k_grid_Pk),
    #                             pk_arr=pk_mm_2d.T, is_logp=False)

    cl_ll_3d = wf_cl_lib.cl_PyCCL(wf_lensing_obj, wf_lensing_obj, ell_grid, zbins,
                                  p_of_k_a=p_of_k_a, cosmo=cosmo_ccl, limber_integration_method='spline')
    cl_gl_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_lensing_obj, ell_grid, zbins,
                                  p_of_k_a=p_of_k_a, cosmo=cosmo_ccl, limber_integration_method='spline')
    cl_gg_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_galaxy_obj, ell_grid, zbins,
                                  p_of_k_a=p_of_k_a, cosmo=cosmo_ccl, limber_integration_method='spline')
    cl_ll_3d_vinc = general_cfg['cl_ll_3d']
    cl_gl_3d_vinc = general_cfg['cl_gl_3d']
    cl_gg_3d_vinc = general_cfg['cl_gg_3d']

    nbl_plt = len(ell_grid)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for zi in range(zbins):
        zj = zi
        ax[0].loglog(ell_grid, cl_ll_3d[:nbl_plt, zi, zj], ls="-", c=colors[zi], alpha=0.6,
                     label='ll' if zi == 0 else None)
        ax[0].loglog(ell_grid, cl_ll_3d_vinc[:nbl_plt, zi, zj], ls="--", c=colors[zi], alpha=0.6,
                     label='ll vinc' if zi == 0 else None)
        ax[1].loglog(ell_grid, cl_gl_3d[:nbl_plt, zi, zj], ls="-", c=colors[zi], alpha=0.6,
                     label='gl' if zi == 0 else None)
        ax[1].loglog(ell_grid, cl_gl_3d_vinc[:nbl_plt, zi, zj], ls="--", c=colors[zi], alpha=0.6,
                     label='gl vinc' if zi == 0 else None)
        ax[2].loglog(ell_grid, cl_gg_3d[:nbl_plt, zi, zj], ls="-", c=colors[zi], alpha=0.6,
                     label='gg' if zi == 0 else None)
        ax[2].loglog(ell_grid, cl_gg_3d_vinc[:nbl_plt, zi, zj], ls="--", c=colors[zi], alpha=0.6,
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

    # # ! this should be handled a bit better bruh
    # z_grid_tkka = np.linspace(pyccl_cfg['z_grid_tkka_min'], pyccl_cfg['z_grid_tkka_max'],
    #                           pyccl_cfg['z_grid_tkka_steps'])
    # a_grid_ttka = cosmo_lib.z_to_a(z_grid_tkka)
    # sigma2_B_ccl = ccl.covariances.sigma2_B_disc(
    #     cosmo_ccl, a_arr=a_grid_ttka, fsky=f_sky, p_of_k_a='delta_matter:delta_matter')

    # area_deg2 = 14700
    # nside = 2048
    # assert mm.percent_diff(f_sky, area_deg2 / 41253, abs_value=True) < 1, 'f_sky is not correct'

    # ell_mask = np.load(
    #     f'/home/davide/Documenti/Lavoro/Programmi/common_data/sylvain/mask/ell_circular_1pole_{area_deg2:d}deg2_nside{nside:d}_davide.npy')
    # cl_mask = np.load(
    #     f'/home/davide/Documenti/Lavoro/Programmi/common_data/sylvain/mask/Cell_circular_1pole_{area_deg2:d}deg2_nside{nside:d}_davide.npy')

    # mask_wl = cl_mask * (2 * ell_mask + 1) / (4 * np.pi * f_sky)**2  # ! important to normalize!

    # # this is because p_of_k_a='delta_matter:delta_matter' gives an error, probably a bug in pyccl
    # cosmo_ccl.compute_linear_power()
    # p_of_k_a = cosmo_ccl.get_linear_power()

    # sigma2_B_ccl_polar_cap = ccl.covariances.sigma2_B_from_mask(
    #     cosmo_ccl, a_arr=a_grid_ttka, mask_wl=mask_wl, p_of_k_a=p_of_k_a)

    # np.save(f'{covariance_cfg["PyCCL_cfg"]["cov_path"]}/sigma2_B_ccl.npy', sigma2_B_ccl)
    # np.save(f'{covariance_cfg["PyCCL_cfg"]["cov_path"]}/sigma2_B_ccl_polar_cap.npy', sigma2_B_ccl_polar_cap)
    # np.save(f'{covariance_cfg["PyCCL_cfg"]["cov_path"]}/z_grid_tkka.npy', z_grid_tkka)

    # ! this should be handled a bit better bruh

    # covariance ordering stuff, also used to compute the trispectrum
    if probe == 'LL':
        probe_ordering = (('L', 'L'),)
    elif probe == 'GG':
        probe_ordering = (('G', 'G'),)
    elif probe == '3x2pt':
        probe_ordering = covariance_cfg['probe_ordering']
        # warnings.warn('TESTING ONLY GLGL TO DEBUG 3X2PT cNG')
        probe_ordering = (('L', 'L'),)  # for testing 3x2pt GLGL, which seems a problematic case.
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

    if pyccl_cfg['which_sigma2_B'] == 'mask':

        print('Computing sigma2_B from mask')

        area_deg2 = pyccl_cfg['area_deg2_mask']
        nside = pyccl_cfg['nside_mask']

        assert mm.percent_diff(f_sky, cosmo_lib.deg2_to_fsky(area_deg2), abs_value=True) < 1, 'f_sky is not correct'

        ell_mask = np.load(pyccl_cfg['ell_mask_filename'].format(area_deg2=area_deg2, nside=nside))
        cl_mask = np.load(pyccl_cfg['cl_mask_filename'].format(area_deg2=area_deg2, nside=nside))

        mask_wl = cl_mask * (2 * ell_mask + 1) / (4 * np.pi * f_sky)**2  # ! important to normalize!

        # this is because p_of_k_a='delta_matter:delta_matter' gives an error, probably a bug in pyccl
        cosmo_ccl.compute_linear_power()
        p_of_k_a = cosmo_ccl.get_linear_power()

        z_grid_sigma2_B = np.linspace(pyccl_cfg['z_grid_tkka_min'], pyccl_cfg['z_grid_tkka_max'],
                                      pyccl_cfg['z_grid_tkka_steps'])
        a_grid_sigma2_B = cosmo_lib.z_to_a(z_grid_sigma2_B)

        sigma2_B = ccl.covariances.sigma2_B_from_mask(
            cosmo_ccl, a_arr=a_grid_sigma2_B, mask_wl=mask_wl, p_of_k_a=p_of_k_a)
        sigma2_B_tuple = (a_grid_sigma2_B, sigma2_B)

    elif pyccl_cfg['which_sigma2_B'] == 'file':
        a_grid_sigma2_B = np.load(pyccl_cfg['a_grid_sigma2_B_filename'])
        sigma2_B = np.load(pyccl_cfg['sigma2_B_filename'])
        sigma2_B_tuple = (a_grid_sigma2_B, sigma2_B)

    elif pyccl_cfg['which_sigma2_B'] == None:
        sigma2_B_tuple = None
    else:
        raise ValueError('which_sigma2_B must be either mask, file or None')

    if pyccl_cfg['which_sigma2_B'] == None:

        z_grid_sigma2_B = np.linspace(pyccl_cfg['z_grid_tkka_min'], pyccl_cfg['z_grid_tkka_max'],
                                      pyccl_cfg['z_grid_tkka_steps'])
        a_grid_sigma2_B = cosmo_lib.z_to_a(z_grid_sigma2_B)
        sigma2_B = ccl.covariances.sigma2_B_disc(
            cosmo_ccl, a_arr=a_grid_sigma2_B, fsky=f_sky, p_of_k_a='delta_matter:delta_matter')

    plt.figure()
    plt.plot(z_grid_sigma2_B, sigma2_B)
    plt.xlabel('z')
    plt.ylabel('sigma2_B(z)')
    plt.yscale('log')

    if pyccl_cfg['save_sigma2_B']:
        np.save(f'{pyccl_cfg["cov_path"]}/{pyccl_cfg["z_grid_sigma2_B_filename"]}.npy', z_grid_sigma2_B)
        np.save(f'{pyccl_cfg["cov_path"]}/{pyccl_cfg["sigma2_B_filename"]}.npy', sigma2_B)

    which_pk = fiducial_pars_dict['other_params']['camb_extra_parameters']['camb']['halofit_version']
    tkka_dict = initialize_trispectrum(cosmo_ccl, which_ng_cov, probe_ordering, pyccl_cfg, which_pk=which_pk)

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
                                       sigma2_B_tuple=sigma2_B_tuple,
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
