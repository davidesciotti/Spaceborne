""" This module should be run with pyccl >= v3.0.0
"""

import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import os
import sys
from matplotlib import cm
from tqdm import tqdm
from scipy.interpolate import interp1d

import os
ROOT = os.getenv('ROOT')
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
import bin.cosmo_lib as cosmo_lib
import bin.wf_cl_lib as wf_cl_lib
import bin.sigma2_SSC as sigma2_SSC
import common_cfg.mpl_cfg as mpl_cfg


start_time = time.perf_counter()
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)


# TODO do they interpolate existing tracer arrays?


# ccl.gsl_params["INTEGRATION_EPSREL"] = 1e-7  # was 1e-4
# ccl.gsl_params["N_ITERATION"] = 10000  # was 1000
ccl.gsl_params.reload()
ccl.spline_params.reload()


# ccl.spline_params['A_SPLINE_NA'] *= 10
# ccl.spline_params['A_SPLINE_NLOG'] *= 10
# ccl.spline_params['A_SPLINE_NLOG_PK'] *= 10
ccl.spline_params['A_SPLINE_NA_PK'] = 150  # gives CAMB error if too high
# ccl.spline_params['N_ELL_CORR'] *= 10

# ccl.spline_params['K_MAX_SPLINE'] = 100
# ccl.spline_params['N_K'] = 1670
# ccl.spline_params['K_MIN'] = 1e-5

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX
# notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb
# HALO MODEL PRESCRIPTIONS:
# KiDS1000 Methodology: https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)
# Krause2017: https://arxiv.org/pdf/1601.05779.pdf
def initialize_trispectrum(cosmo_ccl, which_ng_cov, probe_ordering, pyccl_cfg, which_pk, p_of_k_a):
    
    save_tkka = pyccl_cfg['save_tkka']
    comp_load_str = 'Loading' if pyccl_cfg['load_precomputed_tkka'] else 'Computing'
    
    tkka_folder = f'Tk3D_{which_ng_cov}'
    tkka_path = f'{pyccl_cfg["cov_path"]}/{tkka_folder}'
    
    k_z_str = f'zmin{pyccl_cfg["z_grid_tkka_min"]:.1f}_zmax{pyccl_cfg["z_grid_tkka_max"]:.1f}_zsteps{pyccl_cfg[f"z_grid_tkka_steps_{which_ng_cov}"]:d}_' \
              f'kmin{pyccl_cfg["k_grid_tkka_min"]:.1e}_kmax{pyccl_cfg["k_grid_tkka_max"]:.1e}_ksteps{pyccl_cfg[f"k_grid_tkka_steps_{which_ng_cov}"]:d}'

    a_grid_tkka_SSC = np.linspace(
        cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_max']),
        cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_min']),
        pyccl_cfg['z_grid_tkka_steps_SSC'])

    logn_k_grid_tkka_SSC = np.log(np.geomspace(pyccl_cfg['k_grid_tkka_min'],
                                           pyccl_cfg['k_grid_tkka_max'],
                                           pyccl_cfg['k_grid_tkka_steps_SSC']))
    

    a_grid_tkka_cNG = np.linspace(
        cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_max']),
        cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_min']),
        pyccl_cfg['z_grid_tkka_steps_cNG'])

    logn_k_grid_tkka_cNG = np.log(np.geomspace(pyccl_cfg['k_grid_tkka_min'],
                                           pyccl_cfg['k_grid_tkka_max'],
                                           pyccl_cfg['k_grid_tkka_steps_cNG']))

    # or, to set to the default:
    # a_grid_tkka = None
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
    # This is the correct way to initialize the trispectrum (I Asked David Alonso about this.)
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

    # store the trispectrum for the various probes in a dictionary

    # the default pk bust be passed to yhe Tk3D functions as None, not as 'delta_matter:delta_matter'
    p_of_k_a = None if p_of_k_a == 'delta_matter:delta_matter' else p_of_k_a

    if a_grid_tkka_SSC is not None and logn_k_grid_tkka_SSC is not None:
        print(f'SSC tkka: z points = {a_grid_tkka_SSC.size}, k points = {logn_k_grid_tkka_SSC.size}')
    if a_grid_tkka_cNG is not None and logn_k_grid_tkka_cNG is not None:
        print(f'cNG tkka: z points = {a_grid_tkka_cNG.size}, k points = {logn_k_grid_tkka_cNG.size}')

    tkka_dict = {}
    for row, (A, B) in tqdm(enumerate(probe_ordering)):
        for col, (C, D) in enumerate(probe_ordering):
            
            if col >= row: 
                print(f'{comp_load_str} trispectrum for {which_ng_cov}, probe combination {A}{B}{C}{D}')
            

            if col >= row and pyccl_cfg['load_precomputed_tkka']:
                
                save_tkka = False

                a_arr = np.load(f'{tkka_path}/a_arr_tkka.npy')
                k1_arr = np.load(f'{tkka_path}/k1_arr_tkka.npy')
                k2_arr = np.load(f'{tkka_path}/k2_arr_tkka.npy')
                pk2_arr_k1_tkka = np.load(f'{tkka_path}/pk2_arr_k1_tkka.npy')
                pk2_arr_k2_tkka = np.load(f'{tkka_path}/pk2_arr_k2_tkka.npy')
                

                assert np.array_equal(k1_arr, k2_arr), 'k1_arr and k2_arr must be equal'
                assert pk2_arr_k1_tkka[0].shape == pk2_arr_k2_tkka[1].shape, 'pk2_arr_k1_tkka and pk2_arr_k2_tkka must have the same shape'

                tkka_dict[A, B, C, D] = ccl.tk3d.Tk3D(a_arr=a_arr,
                                                      lk_arr=k1_arr,
                                                      tkk_arr=None,
                                                      pk1_arr=pk2_arr_k1_tkka,
                                                      pk2_arr=pk2_arr_k2_tkka,
                                                      is_logt=False,
                                                      extrap_order_lok=1,
                                                      extrap_order_hik=1
                                                      )

            elif col >= row and not pyccl_cfg['load_precomputed_tkka']:

                # not very nice to put this if-else in the for loop, but A, B, C, D are referenced only here
                if which_ng_cov == 'SSC':
                    tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_SSC
                    additional_args = {
                        'prof12_2pt': prof_2pt_dict[A, B],
                        'prof34_2pt': prof_2pt_dict[C, D],
                        'lk_arr':logn_k_grid_tkka_SSC,
                        'a_arr':a_grid_tkka_SSC,
                    }
                elif which_ng_cov == 'cNG':
                    tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_cNG
                    additional_args = {
                        'prof12_2pt': prof_2pt_dict[A, B],
                        'prof13_2pt': prof_2pt_dict[A, C],
                        'prof14_2pt': prof_2pt_dict[A, D],
                        'prof24_2pt': prof_2pt_dict[B, D],
                        'prof32_2pt': prof_2pt_dict[C, B],
                        'prof34_2pt': prof_2pt_dict[C, D],
                        'lk_arr': logn_k_grid_tkka_cNG,
                        'a_arr': a_grid_tkka_cNG,
                    }
                    # tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_1h
                    # additional_args = {}
                else:
                    raise ValueError(f"Invalid value for which_ng_cov. It is {which_ng_cov}, must be 'SSC' or 'cNG'.")

                tkka_dict[A, B, C, D], responses_dict = tkka_func(cosmo=cosmo_ccl,
                                                                  hmc=hmc,
                                                                  prof=halo_profile_dict[A],
                                                                  prof2=halo_profile_dict[B],
                                                                  prof3=halo_profile_dict[C],
                                                                  prof4=halo_profile_dict[D],

                                                                  extrap_order_lok=1, extrap_order_hik=1, use_log=False,
                                                                  **additional_args)

                tkka_folder = 'Tk3D_SSC' if which_ng_cov == 'SSC' else 'Tk3D_cNG'
               
                # save responses
                probe_block = A + B + C + D
                if which_ng_cov == 'SSC' and pyccl_cfg['save_hm_responses']:
                    for key, value in responses_dict.items():
                        np.save(f"{tkka_path}/{key}_{probe_block}.npy", value)

                if save_tkka:
                    (a_arr, k1_arr, k2_arr, tk3d_arr) = tkka_dict[A, B, C, D].get_spline_arrays()
                    np.save(f'{tkka_path}/a_arr_tkka_{probe_block}_{k_z_str}.npy', a_arr)
                    np.save(f'{tkka_path}/k1_arr_tkka_{probe_block}_{k_z_str}.npy', k1_arr)
                    np.save(f'{tkka_path}/k2_arr_tkka_{probe_block}_{k_z_str}.npy', k2_arr)
                    np.save(f'{tkka_path}/pk1_arr_tkka_{probe_block}_{k_z_str}.npy', tk3d_arr[0])
                    np.save(f'{tkka_path}/pk2_arr_tkka_{probe_block}_{k_z_str}.npy', tk3d_arr[1])

    print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - halomod_start_time))

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
        sigma2_B_arg = {'sigma2_B': sigma2_B_tuple,
                        }
    elif which_ng_cov == 'cNG':
        ng_cov_func = ccl.covariances.angular_cl_cov_cNG
        sigma2_B_arg = {}
    else:
        raise ValueError("Invalid value for which_ng_cov. Must be 'SSC' or 'cNG'.")

    cov_ng_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))
    for ij in tqdm(range(zpairs_AB)):
        for kl in range(zpairs_CD):
            cov_ng_4D[:, :, ij, kl] = ng_cov_func(cosmo,
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

    print(f'{which_ng_cov} computed with pyccl in {(time.perf_counter() - start_time) / 60:.0f} min')

    return cov_ng_4D


def compute_ng_cov_3x2pt(cosmo, which_ng_cov, kernel_dict, ell, tkka_dict, f_sky, integration_method,
                         probe_ordering, ind_dict, sigma2_B_tuple, covariance_cfg, cov_filename):

    pyccl_cfg = covariance_cfg['PyCCL_cfg']

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
                if pyccl_cfg['save_cov']:
                    cov_path = pyccl_cfg['cov_path']
                    cov_filename_fmt = cov_filename.format(probe_a=probe_a, probe_b=probe_b,
                                                           probe_c=probe_c, probe_d=probe_d)

                    nbl_grid_here = len(ell)
                    assert f'nbl{nbl_grid_here}' in cov_filename, \
                        f'cov_filename could be inconsistent with the actual grid used'
                    np.savez_compressed(
                        f'{cov_path}/{cov_filename_fmt}', cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d])

            else:
                print('3x2pt: skipping probe combination ', probe_a, probe_b, probe_c, probe_d)
                cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                    cov_ng_3x2pt_dict_8D[probe_c, probe_d, probe_a, probe_b].transpose(1, 0, 3, 2))

    return cov_ng_3x2pt_dict_8D


def compute_cov_ng_with_pyccl(fiducial_pars_dict, probe, which_ng_cov, ell_grid, general_cfg,
                              covariance_cfg, cov_filename):
    # ! settings
    zbins = general_cfg['zbins']
    nz_tuple = general_cfg['nz_tuple']
    f_sky = covariance_cfg['fsky']
    ind = covariance_cfg['ind']
    GL_or_LG = covariance_cfg['GL_or_LG']
    nbl = len(ell_grid)

    pyccl_cfg = covariance_cfg['PyCCL_cfg']
    n_samples_wf = pyccl_cfg['n_samples_wf']
    # this is needed only for a visual check of the cls, which are not used for SSC anyways
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
    assert has_rsd == False, 'RSD not validated yet...'

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

    if general_cfg['which_forecast'] == 'SPV3':

        maglim = general_cfg['magcut_source'] / 10
        gal_bias_1d = wf_cl_lib.b_of_z_fs2_fit(zgrid_nz, maglim=maglim)
        # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
        gal_bias_2d = np.repeat(gal_bias_1d.reshape(1, -1), zbins, axis=0).T

        # this is a test to use the actual P(k) from the input files, but the agreement gets much worse
        k_grid_Pk, z_grid_Pk, pk_mm_2d = mm.pk_vinc_file_to_2d_npy(general_cfg['CLOE_pk_filename'], plot_pk_z0=True)
        pk_flipped_in_z = np.flip(pk_mm_2d, axis=1)
        scale_factor_grid_pk = cosmo_lib.z_to_a(z_grid_Pk)[::-1]  # flip it
        p_of_k_a = ccl.pk2d.Pk2D(a_arr=scale_factor_grid_pk, lk_arr=np.log(k_grid_Pk),
                                 pk_arr=pk_flipped_in_z.T, is_logp=False)

        # TODO delete the line below to use input pk, range needs to be extended to higher redshifts to match tkka grid (probably larger k range too)
        p_of_k_a = 'delta_matter:delta_matter'

    elif general_cfg['which_forecast'] == 'ISTF':

        istf_bias_func_dict = {
            'analytical': wf_cl_lib.b_of_z_analytical,
            'leporifit': wf_cl_lib.b_of_z_fs1_leporifit,
            'pocinofit': wf_cl_lib.b_of_z_fs1_pocinofit,
        }
        istf_bias_func = istf_bias_func_dict[general_cfg['bias_function']]
        bias_model = general_cfg['bias_model']

        z_means = np.array([flat_fid_pars_dict[f'zmean{zbin:02d}_photo'] for zbin in range(1, zbins + 1)])
        z_edges = np.array([flat_fid_pars_dict[f'zedge{zbin:02d}_photo'] for zbin in range(1, zbins + 2)])

        gal_bias_1d = istf_bias_func(z_means)
        gal_bias_2d = wf_cl_lib.build_galaxy_bias_2d_arr(
            gal_bias_1d, z_means, z_edges, zbins, zgrid_nz, bias_model=bias_model, plot_bias=True)

        p_of_k_a = 'delta_matter:delta_matter'

    else:
        raise ValueError('general_cfg["which_forecast"] must be either SPV3 or ISTF')

    gal_bias_tuple = (zgrid_nz, gal_bias_2d)

    # save in ascii for OneCovariance
    gal_bias_table = np.hstack((zgrid_nz.reshape(-1, 1), gal_bias_2d))
    np.savetxt(f'{covariance_cfg["nofz_folder"]}/'
               f'gal_bias_table_{general_cfg["which_forecast"]}.ascii', gal_bias_table)

    if has_magnification_bias:
        maglim = general_cfg['magcut_source'] / 10
        # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
        mag_bias_1d = wf_cl_lib.s_of_z_fs2_fit(zgrid_nz, maglim=maglim, poly_fit_values=None)
        mag_bias_2d = np.repeat(mag_bias_1d.reshape(1, -1), zbins, axis=0).T
        mag_bias_tuple = (zgrid_nz, mag_bias_2d)
    else:
        # this is the correct way to set the magnification bias values so that the actual bias is 1, ant the corresponding
        # wf_mu is zero (which is, in theory, the case mag_bias_tuple=None, which however causes pyccl to crash!)
        # mag_bias_2d = (np.ones_like(gal_bias_2d) * + 2) / 5
        # mag_bias_tuple = (zgrid_nz, mag_bias_2d)
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

    # ! manually construct galaxy = delta + magnification radial kernel
    a_arr = cosmo_lib.z_to_a(zgrid_nz)
    comoving_distance = ccl.comoving_radial_distance(cosmo_ccl, a_arr)

    wf_galaxy_tot_arr = np.asarray([wf_galaxy_obj[zbin_idx].get_kernel(comoving_distance) for zbin_idx in range(zbins)])
    wf_delta_arr = wf_galaxy_tot_arr[:, 0, :].T
    wf_mu_arr = wf_galaxy_tot_arr[:, -1, :].T if has_magnification_bias else np.zeros_like(wf_delta_arr)

    gal_kernel_plt_title = 'galaxy kernel\n(w/o gal bias!)'
    # in the case of ISTF, the galaxt bias is bin-per-bin and is therefore included in the kernels. Add it here
    # for a fair comparison with vincenzo's kernels, in the plot.
    # * Note that the galaxy bias is included in the wf_ccl_obj in any way, both in ISTF and SPV3 cases! It must
    # * in fact be passed to the congular_cov_SSC function
    if general_cfg['which_forecast'] == 'ISTF':
        wf_delta_arr *= gal_bias_2d
        gal_kernel_plt_title = 'galaxy kernel\n(w/ gal bias)'

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
    ax[1].set_title(gal_kernel_plt_title)
    ax[0].set_xlabel('$z$')
    ax[1].set_xlabel('$z$')
    ax[0].set_ylabel('wil')
    ax[1].set_ylabel('wig')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # the cls are not needed, but just in case:
    cl_ll_3d = wf_cl_lib.cl_PyCCL(wf_lensing_obj, wf_lensing_obj, ell_grid, zbins,
                                  p_of_k_a=p_of_k_a, cosmo=cosmo_ccl)
    cl_gl_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_lensing_obj, ell_grid, zbins,
                                  p_of_k_a=p_of_k_a, cosmo=cosmo_ccl)
    cl_gg_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_galaxy_obj, ell_grid, zbins,
                                  p_of_k_a=p_of_k_a, cosmo=cosmo_ccl)

    # if you need to save finely sampled cls for OneCovariance
    # ell_grid = np.geomspace(10, 5000, 90)
    # which_pk = general_cfg['which_pk']
    # mm.write_cl_ascii(general_cfg['cl_folder'].format(which_pk=which_pk),
    #                   f'Cell_ll_SPV3_ccl', cl_ll_3d, ell_grid, zbins)
    # mm.write_cl_ascii(general_cfg['cl_folder'].format(which_pk=which_pk),
    #                   f'Cell_gl_SPV3_ccl', cl_gl_3d, ell_grid, zbins)
    # mm.write_cl_ascii(general_cfg['cl_folder'].format(which_pk=which_pk),
    #                   f'Cell_gg_SPV3_ccl', cl_gg_3d, ell_grid, zbins)

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

    # covariance ordering stuff, also used to compute the trispectrum
    if probe == 'LL':
        probe_ordering = (('L', 'L'),)
    elif probe == 'GG':
        probe_ordering = (('G', 'G'),)
    elif probe == '3x2pt':
        probe_ordering = covariance_cfg['probe_ordering']
        # warnings.warn('TESTING ONLY GLGL TO DEBUG')
        # probe_ordering = (('G', 'L'),)  # for testing 3x2pt GLGL, which seems a problematic case.
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

    a_grid_sigma2_B = np.linspace(cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_max']),
                                  cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_min']),
                                  pyccl_cfg['z_grid_tkka_steps_SSC'])

    # z_grid_sigma2_B = z_grid_tkka
    z_grid_sigma2_B = cosmo_lib.z_to_a(a_grid_sigma2_B)[::-1]

    a_grid_sigma2_B = np.linspace(
        cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_max']),
        cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_min']),
        pyccl_cfg['z_grid_tkka_steps_SSC'])

    z_grid_sigma2_B = cosmo_lib.a_to_z(a_grid_sigma2_B)[::-1]

    if pyccl_cfg['which_sigma2_B'] == 'mask':

        print('Computing sigma2_B from mask')

        area_deg2 = pyccl_cfg['area_deg2_mask']
        nside = pyccl_cfg['nside_mask']

        assert mm.percent_diff(f_sky, cosmo_lib.deg2_to_fsky(area_deg2), abs_value=True) < 1, 'f_sky is not correct'

        ell_mask = np.load(pyccl_cfg['ell_mask_filename'].format(area_deg2=area_deg2, nside=nside))
        cl_mask = np.load(pyccl_cfg['cl_mask_filename'].format(area_deg2=area_deg2, nside=nside))

        # normalization has been checked from https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/scripts/compute_SSC_mask_power.py
        # and is the same as CSST paper https://zenodo.org/records/7813033
        cl_mask_norm = cl_mask * (2 * ell_mask + 1) / (4 * np.pi * f_sky)**2

        sigma2_B = ccl.covariances.sigma2_B_from_mask(
            cosmo=cosmo_ccl, a_arr=a_grid_sigma2_B, mask_wl=cl_mask_norm, p_of_k_a='delta_matter:delta_matter')

        sigma2_B_tuple = (a_grid_sigma2_B, sigma2_B)

    elif pyccl_cfg['which_sigma2_B'] == 'spaceborne':

        print('Computing sigma2_B with Spaceborne, using input mask')

        area_deg2 = pyccl_cfg['area_deg2_mask']
        nside = pyccl_cfg['nside_mask']

        assert mm.percent_diff(f_sky, cosmo_lib.deg2_to_fsky(area_deg2), abs_value=True) < 1, 'f_sky is not correct'

        ell_mask = np.load(pyccl_cfg['ell_mask_filename'].format(area_deg2=area_deg2, nside=nside))
        cl_mask = np.load(pyccl_cfg['cl_mask_filename'].format(area_deg2=area_deg2, nside=nside))

        k_grid_tkka = np.geomspace(pyccl_cfg['k_grid_tkka_min'],
                                   pyccl_cfg['k_grid_tkka_max'],
                                   5000)

        sigma2_B = np.array([sigma2_SSC.sigma2_func(zi, zi, k_grid_tkka, cosmo_ccl, 'mask', ell_mask=ell_mask, cl_mask=cl_mask)
                             for zi in tqdm(z_grid_sigma2_B)])  # if you pass the mask, you don't need to divide by fsky
        sigma2_B_tuple = (a_grid_sigma2_B, sigma2_B[::-1])

    elif pyccl_cfg['which_sigma2_B'] == None:
        sigma2_B_tuple = None

    else:
        raise ValueError('which_sigma2_B must be either mask, spaceborne or None')

    if pyccl_cfg['which_sigma2_B'] != None:
        plt.figure()
        plt.plot(sigma2_B_tuple[0], sigma2_B_tuple[1], marker='o')
        plt.xlabel('$a$')
        plt.ylabel('$\sigma^2_B(a)$')
        plt.yscale('log')
        plt.show()

    which_pk = fiducial_pars_dict['other_params']['camb_extra_parameters']['camb']['halofit_version']
    tkka_dict = initialize_trispectrum(cosmo_ccl, which_ng_cov, probe_ordering,
                                       pyccl_cfg, which_pk=which_pk, p_of_k_a=p_of_k_a)
    cov_ng_8D_dict = {}

    if probe in ['LL', 'GG']:
        raise NotImplementedError('you should check that the dictionary cov_ng_8D_dict return works, in this case')

        kernel_A = kernel_dict[probe[0]]
        kernel_B = kernel_dict[probe[1]]
        kernel_C = kernel_dict[probe[0]]
        kernel_D = kernel_dict[probe[1]]
        ind_AB = ind_dict[probe[0] + probe[1]]
        ind_CD = ind_dict[probe[0] + probe[1]]

        cov_ng_8D_dict[probe[0], probe[1], probe[0], probe[1]] = compute_ng_cov_ccl(cosmo=cosmo_ccl,
                                                                                    which_ng_cov=which_ng_cov,
                                                                                    kernel_A=kernel_A,
                                                                                    kernel_B=kernel_B,
                                                                                    kernel_C=kernel_C,
                                                                                    kernel_D=kernel_D,
                                                                                    ell=ell_grid, tkka=tkka_dict[probe[0],
                                                                                                                 probe[1], probe[0], probe[1]],
                                                                                    f_sky=f_sky,
                                                                                    ind_AB=ind_AB,
                                                                                    ind_CD=ind_CD,
                                                                                    sigma2_B_tuple=sigma2_B_tuple,
                                                                                    integration_method=integration_method_dict[probe][which_ng_cov],
                                                                                    )

    elif probe == '3x2pt':
        # TODO remove this if statement and use the same code for all probes
        cov_ng_8D_dict = compute_ng_cov_3x2pt(cosmo=cosmo_ccl,
                                              which_ng_cov=which_ng_cov,
                                              kernel_dict=kernel_dict,
                                              ell=ell_grid, tkka_dict=tkka_dict, f_sky=f_sky,
                                              probe_ordering=probe_ordering,
                                              ind_dict=ind_dict,
                                              covariance_cfg=covariance_cfg,
                                              cov_filename=cov_filename,
                                              sigma2_B_tuple=sigma2_B_tuple,
                                              integration_method=integration_method_dict[probe][which_ng_cov],
                                              )

    else:
        raise ValueError('probe must be either LL, GG, or 3x2pt')

    # test if cov is symmetric in ell1, ell2 (only for the diagonal covariance blocks!! 
    # the noff-diagonal need *not* to be symmetrix in ell1, ell2)
    for key in cov_ng_8D_dict.keys():
        if (key == ('L', 'L', 'L', 'L')) or (key == ('G', 'L', 'G', 'L')) or (key == ('G', 'G', 'G', 'G')):
            try:
                np.testing.assert_allclose(cov_ng_8D_dict[key], np.transpose(cov_ng_8D_dict[key], (1, 0, 2, 3)), rtol=1e-6, atol=0,
                                        err_msg=f'cov_ng_4D {key} is not symmetric in ell1, ell2')
            except AssertionError as error:
                print(error)

    return cov_ng_8D_dict


integration_method_dict = {
    'LL': {
        'SSC': 'spline',
        'cNG': 'spline',
    },
    'GG': {
        'SSC': 'spline',
        'cNG': 'spline',
    },
    '3x2pt': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',  # qag_quad
    }
}
