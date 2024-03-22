from copy import deepcopy
import gc
import sys
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pprint import pprint
import warnings
import pandas as pd
from matplotlib import cm

from getdist import plots   
from getdist.gaussian_mixtures import GaussianND

import os
ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.my_module as mm
import bin.ell_values as ell_utils
import bin.cl_preprocessing as cl_utils
import bin.compute_Sijkl as Sijkl_utils
import bin.covariance as covmat_utils
import bin.fisher_matrix as FM_utils
import bin.plots_FM_running as plot_utils
import common_cfg.mpl_cfg as mpl_cfg
import common_cfg.ISTF_fid_params as ISTF_fid

# job configuration and modules
from jobs.ISTF.config import config_ISTF as cfg

# mpl.use('Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

os.environ['OMP_NUM_THREADS'] = '16'

# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks


###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg
Sijkl_cfg = cfg.Sijkl_cfg
FM_cfg = cfg.FM_cfg

probes = ['WL', 'GC', 'XC', '3x2pt', '2x2pt']


# for covariance_cfg['SSC_code'] in ['PyCCL', 'OneCovariance', 'Spaceborne', 'PySSC']:
for covariance_cfg['SSC_code'] in (covariance_cfg['SSC_code'], ):
    # for covariance_cfg['SSC_code'] in ('OneCovariance',  ):

    # check_specs.consistency_checks(general_cfg, covariance_cfg)

    # some variables used for I/O naming, just to make things more readable
    zbins = general_cfg['zbins']
    EP_or_ED = general_cfg['EP_or_ED']
    ell_min = general_cfg['ell_min']
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    ell_max_XC = general_cfg['ell_max_XC']
    triu_tril = covariance_cfg['triu_tril']
    row_col_major = covariance_cfg['row_col_major']
    n_probes = general_cfg['n_probes']
    nbl_WL = general_cfg['nbl_WL']
    nbl_GC = general_cfg['nbl_GC']
    nbl = nbl_WL
    bIA = ISTF_fid.IA_free['beta_IA']
    GL_or_LG = covariance_cfg['GL_or_LG']
    fiducials_dict = FM_cfg['fiducials_dict']
    param_names_dict = FM_cfg['param_names_dict']
    param_names_3x2pt = FM_cfg['param_names_3x2pt']
    nparams_tot = FM_cfg['nparams_tot']
    der_prefix = FM_cfg['derivatives_prefix']
    derivatives_suffix = FM_cfg['derivatives_suffix']
    ssc_code = covariance_cfg['SSC_code']

    # which cases to save: GO, GS or GO, GS and SSC
    cases_tosave = []  #
    if covariance_cfg[f'save_cov_GO']:
        cases_tosave.append('G')
    if covariance_cfg[f'save_cov_GS']:
        cases_tosave.append('GSSC')
    if covariance_cfg[f'save_cov_SSC']:
        cases_tosave.append('SSC')

    # some checks
    assert EP_or_ED == 'EP' and zbins == 10, 'ISTF uses 10 equipopulated bins'
    assert covariance_cfg['GL_or_LG'] == 'GL', 'Cij_14may uses GL, also for the probe responses'
    assert nbl_GC == nbl_WL, 'for ISTF we are using the same number of ell bins for WL and GC'
    assert general_cfg['ell_cuts'] is False, 'ell_cuts is not implemented for ISTF'
    assert general_cfg['BNT_transform'] is False, 'BNT_transform is not implemented for ISTF'

    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
    ind = mm.build_full_ind(covariance_cfg['triu_tril'], covariance_cfg['row_col_major'], zbins)
    covariance_cfg['ind'] = ind

    covariance_cfg['probe_ordering'] = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))

    # ! compute ell and delta ell values
    ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_cfg)
    nbl_WA = ell_dict['ell_WA'].shape[0]
    ell_WL, ell_GC, ell_WA = ell_dict['ell_WL'], ell_dict['ell_GC'], ell_dict['ell_WA']

    if covariance_cfg['use_sylvains_deltas']:
        delta_dict['delta_l_WL'] = mm.delta_l_Sylvain(nbl_WL, ell_dict['ell_WL'])
        delta_dict['delta_l_GC'] = mm.delta_l_Sylvain(nbl_GC, ell_dict['ell_GC'])
        delta_dict['delta_l_WA'] = mm.delta_l_Sylvain(nbl_WA, ell_dict['ell_WA'])

    variable_specs = {
        'zbins': zbins,
        'EP_or_ED': EP_or_ED,
        'triu_tril': triu_tril,
        'row_col_major': row_col_major,
        'ell_max_WL': general_cfg['ell_max_WL'],
        'ell_max_GC': general_cfg['ell_max_GC'],
        'ell_max_XC': general_cfg['ell_max_XC'],
        'nbl_WL': general_cfg['nbl_WL'],
        'nbl_GC': general_cfg['nbl_GC'],
        'nbl_WA': nbl_WA,
        'nbl_3x2pt': general_cfg['nbl_3x2pt'],
    }

    # ! import, interpolate and reshape the power spectra and probe responses
    cl_folder = general_cfg['cl_folder'].format(**variable_specs)
    cl_filename = general_cfg['cl_filename']
    cl_LL_2D = np.genfromtxt(f'{cl_folder}/{cl_filename.format(probe="LL")}')
    cl_GL_2D = np.genfromtxt(f'{cl_folder}/{cl_filename.format(probe="GL")}')
    cl_GG_2D = np.genfromtxt(f'{cl_folder}/{cl_filename.format(probe="GG")}')

    rl_folder = general_cfg['rl_folder'].format(**variable_specs)
    rl_filename = general_cfg['rl_filename']
    rl_LL_2D = np.genfromtxt(f'{rl_folder}/{rl_filename.format(probe="ll")}')
    rl_GL_2D = np.genfromtxt(f'{rl_folder}/{rl_filename.format(probe="gl")}')
    rl_GG_2D = np.genfromtxt(f'{rl_folder}/{rl_filename.format(probe="gg")}')

    # interpolate
    cl_dict_2D = {}
    cl_dict_2D['cl_LL_2D'] = mm.cl_interpolator(cl_LL_2D, zpairs_auto, ell_WL, nbl_WL)
    cl_dict_2D['cl_GG_2D'] = mm.cl_interpolator(cl_GG_2D, zpairs_auto, ell_GC, nbl_GC)
    cl_dict_2D['cl_WA_2D'] = mm.cl_interpolator(cl_LL_2D, zpairs_auto, ell_WA, nbl_WA)
    cl_dict_2D['cl_GL_2D'] = mm.cl_interpolator(cl_GL_2D, zpairs_cross, ell_GC, nbl_GC)
    cl_dict_2D['cl_LLfor3x2pt_2D'] = mm.cl_interpolator(cl_LL_2D, zpairs_auto, ell_GC, nbl_GC)

    rl_dict_2D = {}
    rl_dict_2D['rl_LL_2D'] = mm.cl_interpolator(rl_LL_2D, zpairs_auto, ell_WL, nbl_WL)
    rl_dict_2D['rl_GG_2D'] = mm.cl_interpolator(rl_GG_2D, zpairs_auto, ell_GC, nbl_GC)
    rl_dict_2D['rl_WA_2D'] = mm.cl_interpolator(rl_LL_2D, zpairs_auto, ell_WA, nbl_WA)
    rl_dict_2D['rl_GL_2D'] = mm.cl_interpolator(rl_GL_2D, zpairs_cross, ell_GC, nbl_GC)
    rl_dict_2D['rl_LLfor3x2pt_2D'] = mm.cl_interpolator(rl_LL_2D, zpairs_auto, ell_GC, nbl_GC)

    # reshape to 3D
    cl_dict_3D = {}
    cl_dict_3D['cl_LL_3D'] = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_LL_2D'], nbl_WL, zpairs_auto, zbins)
    cl_dict_3D['cl_GG_3D'] = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_GG_2D'], nbl_GC, zpairs_auto, zbins)
    cl_dict_3D['cl_WA_3D'] = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_WA_2D'], nbl_WA, zpairs_auto, zbins)

    rl_dict_3D = {}
    rl_dict_3D['rl_LL_3D'] = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_LL_2D'], nbl_WL, zpairs_auto, zbins)
    rl_dict_3D['rl_GG_3D'] = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_GG_2D'], nbl_GC, zpairs_auto, zbins)
    rl_dict_3D['rl_WA_3D'] = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_WA_2D'], nbl_WA, zpairs_auto, zbins)

    # build 3x2pt 5D datavectors; the GL and LLfor3x2pt are only needed for this!
    cl_GL_3D = mm.cl_2D_to_3D_asymmetric(cl_dict_2D['cl_GL_2D'], nbl_GC, zbins, order='C')
    rl_GL_3D = mm.cl_2D_to_3D_asymmetric(rl_dict_2D['rl_GL_2D'], nbl_GC, zbins, order='C')
    cl_LLfor3x2pt_3D = mm.cl_2D_to_3D_symmetric(cl_dict_2D['cl_LLfor3x2pt_2D'], nbl_GC, zpairs_auto, zbins)
    rl_LLfor3x2pt_3D = mm.cl_2D_to_3D_symmetric(rl_dict_2D['rl_LLfor3x2pt_2D'], nbl_GC, zpairs_auto, zbins)

    cl_dict_3D['cl_3x2pt_5D'] = cl_utils.build_3x2pt_datavector_5D(cl_LLfor3x2pt_3D,
                                                                   cl_GL_3D,
                                                                   cl_dict_3D['cl_GG_3D'],
                                                                   nbl_GC, zbins, n_probes)
    rl_dict_3D['rl_3x2pt_5D'] = cl_utils.build_3x2pt_datavector_5D(rl_LLfor3x2pt_3D,
                                                                   rl_GL_3D,
                                                                   rl_dict_3D['rl_GG_3D'],
                                                                   nbl_GC, zbins, n_probes)

    general_cfg['cl_ll_3d'] = cl_LLfor3x2pt_3D
    general_cfg['cl_gl_3d'] = cl_GL_3D
    general_cfg['cl_gg_3d'] = cl_dict_3D['cl_GG_3D']

    # reshape for OneCovariance code
    mm.write_cl_ascii(cl_folder, 'Cell_ll', cl_LLfor3x2pt_3D, ell_dict['ell_GC'], zbins)
    mm.write_cl_ascii(cl_folder, 'Cell_gl', cl_GL_3D, ell_dict['ell_GC'], zbins)
    mm.write_cl_ascii(cl_folder, 'Cell_gg', cl_dict_3D['cl_GG_3D'], ell_dict['ell_GC'], zbins)

    # ! compute covariance matrix
    if not covariance_cfg['compute_covmat']:
        raise KeyboardInterrupt('Fisher matrix computation is set to False; exiting')

    # ! load kernels
    # TODO this should not be done if Sijkl is loaded; I have a problem with nz, which is part of the file name...
    nz = Sijkl_cfg["nz"]
    wf_folder = Sijkl_cfg["wf_input_folder"].format(nz=nz)
    wil_filename = Sijkl_cfg["wf_WL_input_filename"].format(normalization=Sijkl_cfg['wf_normalization'],
                                                            has_IA=str(Sijkl_cfg['has_IA']), nz=nz, bIA=bIA)
    wig_filename = Sijkl_cfg["wf_GC_input_filename"].format(normalization=Sijkl_cfg['wf_normalization'], nz=nz)
    wil = np.genfromtxt(f'{wf_folder}/{wil_filename}')
    wig = np.genfromtxt(f'{wf_folder}/{wig_filename}')

    # preprocess (remove redshift column)
    z_arr, wil = Sijkl_utils.preprocess_wf(wil, zbins)
    z_arr_2, wig = Sijkl_utils.preprocess_wf(wig, zbins)
    assert np.array_equal(z_arr, z_arr_2), 'the redshift arrays are different for the GC and WL kernels'
    assert nz == z_arr.shape[0], 'nz is not the same as the number of redshift points in the kernels'

    nz_import = np.genfromtxt(f'{covariance_cfg["nofz_folder"]}/{covariance_cfg["nofz_filename"]}')
    np.savetxt(f'{covariance_cfg["nofz_folder"]}/{covariance_cfg["nofz_filename"].replace("dat", "ascii")}', nz_import)
    z_grid_nz = nz_import[:, 0]
    nz_import = nz_import[:, 1:]
    nz_tuple = (z_grid_nz, nz_import)

    # store them to be passed to pyccl_cov for comparison (or import)
    general_cfg['wf_WL'] = wil
    general_cfg['wf_GC'] = wig
    general_cfg['z_grid_wf'] = z_arr
    general_cfg['nz_tuple'] = nz_tuple

    # ! compute or load Sijkl
    # if Sijkl exists, load it; otherwise, compute it and save it
    Sijkl_folder = Sijkl_cfg['Sijkl_folder']
    Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(nz=Sijkl_cfg['nz'])

    if Sijkl_cfg['load_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):

        print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
        sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')

    else:
        # transpose and stack, ordering is important here!
        transp_stacked_wf = np.vstack((wil.T, wig.T))
        sijkl = Sijkl_utils.compute_Sijkl(cosmo_lib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                          Sijkl_cfg['wf_normalization'])
        if Sijkl_cfg['save_sijkl']:
            np.save(f'{Sijkl_folder}/{Sijkl_filename}', sijkl)

    # ! compute covariance matrix
    cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                        ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl, BNT_matrix=None)
    
    
    # ! start chi2 plot like Barreira 2018    
    import bin.wf_cl_lib as wf_cl_lib
    import bin.cosmo_lib as cosmo_lib
    import pyccl as ccl
    from scipy.optimize import minimize
    from tqdm import tqdm
    
    has_magnification_bias = False
    mag_bias_tuple = None
    has_rsd = False
    p_of_k_a = 'delta_matter:delta_matter'
    flat_fid_pars_dict = mm.flatten_dict(deepcopy(cfg.fid_pars_dict))
   
    def cls_with_ccl(par_tovary_value, ell_grid, par_tovary_name):

        if par_tovary_name in flat_fid_pars_dict.keys():
            flat_fid_pars_dict[par_tovary_name] = float(par_tovary_value)
        else:
            raise ValueError(f'{par_tovary_name} not found in fiducial parameters')    
        
        cosmo_dict_ccl = cosmo_lib.map_keys(flat_fid_pars_dict, key_mapping=None)
        cosmo_ccl = cosmo_lib.instantiate_cosmo_ccl_obj(cosmo_dict_ccl,
                                                        cfg.fid_pars_dict['other_params']['camb_extra_parameters'])
        
        ia_bias_1d = wf_cl_lib.build_ia_bias_1d_arr(z_grid_nz, cosmo_ccl=cosmo_ccl, flat_fid_pars_dict=flat_fid_pars_dict,
                                            input_z_grid_lumin_ratio=None,
                                            input_lumin_ratio=None, output_F_IA_of_z=False)
        ia_bias_tuple = (z_grid_nz, ia_bias_1d)


        # istf_bias_func_dict = {
        #     'analytical': wf_cl_lib.b_of_z_analytical,
        #     'leporifit': wf_cl_lib.b_of_z_fs1_leporifit,
        #     'pocinofit': wf_cl_lib.b_of_z_fs1_pocinofit,
        # }
        # istf_bias_func = istf_bias_func_dict[general_cfg['bias_function']]
        # bias_model = general_cfg['bias_model']

        # z_means = np.array([flat_fid_pars_dict[f'zmean{zbin:02d}_photo'] for zbin in range(1, zbins + 1)])
        # z_edges = np.array([flat_fid_pars_dict[f'zedge{zbin:02d}_photo'] for zbin in range(1, zbins + 2)])

        # gal_bias_1d = istf_bias_func(z_means)
        # gal_bias_2d = wf_cl_lib.build_galaxy_bias_2d_arr(
        #     gal_bias_1d, z_means, z_edges, zbins, z_grid_nz, bias_model=bias_model, plot_bias=False)


        # gal_bias_tuple = (z_grid_nz, gal_bias_2d)

        # # save in ascii for OneCovariance
        # gal_bias_table = np.hstack((z_grid_nz.reshape(-1, 1), gal_bias_2d))
        # np.savetxt(f'{covariance_cfg["nofz_folder"]}/'
        #         f'gal_bias_table_{general_cfg["which_forecast"]}.ascii', gal_bias_table)

        


        wf_lensing_obj = wf_cl_lib.wf_ccl(z_grid_nz, 'lensing', 'with_IA', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                        ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=None,
                                        mag_bias_tuple=mag_bias_tuple, has_rsd=has_rsd, return_ccl_obj=True, n_samples=256)
        # wf_galaxy_obj = wf_cl_lib.wf_ccl(z_grid_nz, 'galaxy', 'with_galaxy_bias', flat_fid_pars_dict, cosmo_ccl, nz_tuple,
                                        # ia_bias_tuple=ia_bias_tuple, gal_bias_tuple=gal_bias_tuple,
                                        # mag_bias_tuple=mag_bias_tuple, has_rsd=has_rsd, return_ccl_obj=True, n_samples=256)


        # the cls are not needed, but just in case:
        cl_ll_3d = wf_cl_lib.cl_PyCCL(wf_lensing_obj, wf_lensing_obj, ell_grid, zbins,
                                    p_of_k_a=p_of_k_a, cosmo=cosmo_ccl, limber_integration_method='spline')
        # cl_gl_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_lensing_obj, ell_grid, zbins,
                                    # p_of_k_a=p_of_k_a, cosmo=cosmo_ccl)
        # cl_gg_3d = wf_cl_lib.cl_PyCCL(wf_galaxy_obj, wf_galaxy_obj, ell_grid, zbins,
                                    # p_of_k_a=p_of_k_a, cosmo=cosmo_ccl)
        
        cl_ll_1d = mm.cl_3D_to_2D_or_1D(cl_ll_3d, ind, True, True, False, 'vincenzo')
        
        return cl_ll_1d
    
    def objective(par_tovary_value, x_data, y_data, inv_cov):
        # Define your objective function here. This is usually the sum of squared residuals.
        diff = y_data - cls_with_ccl(par_tovary_value=par_tovary_value, ell_grid=x_data, par_tovary_name=par_tovary_name)
        chi2 = diff @ inv_cov @ diff
        return chi2
    
    
    def parallel_wrapper(sample_idx, cl_wl_1d_fid_samples):
        y_data = cl_wl_1d_fid_samples[sample_idx, :]
        result_om = minimize(objective, x0=(x0, ), args=(x_data, y_data, inv_cov_wl_2d), bounds=((0.2, 0.6), ))
        return result_om
    
    # ! settings for chi2 test
    par_tovary_dict = {'Om_m0': 0.32}
    par_tovary_name, par_tovary_value = list(par_tovary_dict.items())[0]
    x0 = par_tovary_value
    x_data = ell_dict['ell_WL']
    n_samples = 5
    
    
    
    # pick a fiducial bwteen ccl and vincenzo's
    # cl_wl_1d_fid = cl_dict_2D['cl_LL_2D'].flatten()
    start = time.perf_counter()
    cl_wl_1d_fid = cls_with_ccl(par_tovary_value = par_tovary_value, ell_grid=ell_dict['ell_WL'], par_tovary_name=par_tovary_name)
    print(f'ccl took {time.perf_counter() - start:.2f} s')
    
    
    cov_wl_2d = cov_dict['cov_WL_GS_2D']
    inv_cov_wl_2d = np.linalg.inv(cov_wl_2d)  # TODO put different types of NG cov here
    cl_wl_1d_fid_samples = np.random.multivariate_normal(cl_wl_1d_fid, cov_wl_2d, n_samples)
        
    

    # parallel version
    print('starting minimization in parallel...')
    start = time.perf_counter()
    result_parallel_list = Parallel(n_jobs=-1)(delayed(parallel_wrapper)(sample_idx, cl_wl_1d_fid_samples) for sample_idx in range(n_samples))
    print(f'...done in {time.perf_counter() - start:.2f} s')
    
    best_fit_parall, chi2_bf_parall = [], []
    for sample_idx in tqdm(range(n_samples)):
        best_fit_parall.append(result_parallel_list[sample_idx].x[0])
        chi2_bf_parall.append(result_parallel_list[sample_idx].fun)
        
    print('starting minimization in serial...')
    start = time.perf_counter()
    best_fit_serial, chi2_bf_serial = [], []
    for sample_idx in tqdm(range(n_samples)):
        y_data = cl_wl_1d_fid_samples[sample_idx, :]
        result_serial = minimize(objective, x0=(x0, ), args=(x_data, y_data, inv_cov_wl_2d), bounds=((0.2, 0.6), ))
        best_fit_serial.append(result_serial.x[0])
        chi2_bf_serial.append(result_serial.fun)
    print(f'...done {time.perf_counter() - start:.2f} s')
    
    # check that // and serial results coincide
    # assert np.array_equal(np.array(best_fit_parall), np.array(best_fit_serial))
    
    # plt.plot((np.array(chi2_bf_serial)/np.array(chi2_bf_parall) - 1)*100)
    # plt.ylabel('% diff chi2')
    # plt.ylabel('sample idx')
    
    ax, fig = plt.subplots((1, 2),)
    ax[0].hist(chi2_bf_parall)
    ax[1].hist(best_fit_parall)
    
    plt.figure()
    # plt.hist(best_fit_om)
        
    # for sample_idx in [1, 100, 500, 900]:
    plt.loglog(cl_wl_1d_fid)
    # plt.loglog(dv_wl_samples[sample_idx, :], '--')
    plt.loglog(y_data, '--')
    
    assert False, 'stop here for chi2 test'



    # ! save and test against benchmarks
    cov_folder = covariance_cfg["cov_folder"].format(SSC_code=ssc_code, **variable_specs)
    covmat_utils.save_cov(cov_folder, covariance_cfg, cov_dict, cases_tosave, **variable_specs)
    if general_cfg['test_against_benchmarks']:
        mm.test_folder_content(cov_folder, cov_folder + '/benchmarks', covariance_cfg['cov_file_format'])

    # ! compute Fisher Matrix
    if not FM_cfg['compute_FM']:
        raise KeyboardInterrupt('Fisher matrix computation is set to False; exiting')

    derivatives_folder = FM_cfg['derivatives_folder'].format(**variable_specs)
    if FM_cfg['load_preprocess_derivatives']:

        print(f'Loading precomputed derivatives from folder\n{derivatives_folder}')
        dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D')
        dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D')
        dC_WA_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_WA_4D')
        dC_3x2pt_6D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D')

    else:

        dC_dict_2D = dict(mm.get_kv_pairs(derivatives_folder, "dat"))
        # check if dictionary is empty
        if not dC_dict_2D:
            raise ValueError(f'No derivatives found in folder {derivatives_folder}')

        # interpolate and separate into probe-specific dictionaries, as
        # ; then reshape from 2D to 3D
        dC_dict_LL_2D, dC_dict_LL_3D = {}, {}
        dC_dict_GG_2D, dC_dict_GG_3D = {}, {}
        dC_dict_GL_2D, dC_dict_GL_3D = {}, {}
        dC_dict_WA_2D, dC_dict_WA_3D = {}, {}
        dC_dict_LLfor3x2pt_2D, dC_dict_LLfor3x2pt_3D = {}, {}

        for key in dC_dict_2D.keys():
            if key.endswith(derivatives_suffix):

                if key.startswith(der_prefix.format(probe='LL')):
                    dC_dict_LL_2D[key] = mm.cl_interpolator(dC_dict_2D[key], zpairs_auto, ell_WL, nbl_WL)
                    dC_dict_WA_2D[key] = mm.cl_interpolator(dC_dict_2D[key], zpairs_auto, ell_WA, nbl_WA)
                    dC_dict_LLfor3x2pt_2D[key] = mm.cl_interpolator(dC_dict_2D[key], zpairs_auto, ell_GC, nbl_GC)

                    dC_dict_LL_3D[key] = mm.cl_2D_to_3D_symmetric(dC_dict_LL_2D[key], nbl_WL, zpairs_auto, zbins)
                    dC_dict_WA_3D[key] = mm.cl_2D_to_3D_symmetric(dC_dict_WA_2D[key], nbl_WA, zpairs_auto, zbins)
                    dC_dict_LLfor3x2pt_3D[key] = mm.cl_2D_to_3D_symmetric(dC_dict_LLfor3x2pt_2D[key], nbl_GC, zpairs_auto,
                                                                          zbins)

                elif key.startswith(der_prefix.format(probe='GG')):
                    dC_dict_GG_2D[key] = mm.cl_interpolator(dC_dict_2D[key], zpairs_auto, ell_GC, nbl_GC)
                    dC_dict_GG_3D[key] = mm.cl_2D_to_3D_symmetric(dC_dict_GG_2D[key], nbl_GC, zpairs_auto, zbins)

                elif key.startswith(der_prefix.format(probe=GL_or_LG)):
                    dC_dict_GL_2D[key] = mm.cl_interpolator(dC_dict_2D[key], zpairs_cross, ell_GC, nbl_GC)
                    dC_dict_GL_3D[key] = mm.cl_2D_to_3D_asymmetric(dC_dict_GL_2D[key], nbl_GC, zbins, 'row-major')

        # turn dictionary keys into entries of 4-th array axis
        # TODO the obs_name must be defined in the config file
        dC_LL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LL_3D, param_names_3x2pt, nbl, zbins,
                                                der_prefix.format(probe='LL'))
        dC_WA_4D = FM_utils.dC_dict_to_4D_array(dC_dict_WA_3D, param_names_3x2pt, nbl_WA, zbins,
                                                der_prefix.format(probe='LL'))
        dC_LLfor3x2pt_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LLfor3x2pt_3D, param_names_3x2pt, nbl, zbins,
                                                        der_prefix.format(probe='LL'))
        dC_GG_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GG_3D, param_names_3x2pt, nbl, zbins,
                                                der_prefix.format(probe='GG'))
        dC_GL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GL_3D, param_names_3x2pt, nbl, zbins,
                                                der_prefix.format(probe=GL_or_LG))

        # build 5D array of derivatives for the 3x2pt
        dC_3x2pt_6D = np.zeros((n_probes, n_probes, nbl, zbins, zbins, nparams_tot))
        dC_3x2pt_6D[0, 0, :, :, :, :] = dC_LLfor3x2pt_4D
        dC_3x2pt_6D[0, 1, :, :, :, :] = dC_GL_4D.transpose(0, 2, 1, 3)
        dC_3x2pt_6D[1, 0, :, :, :, :] = dC_GL_4D
        dC_3x2pt_6D[1, 1, :, :, :, :] = dC_GG_4D

        np.save(f'{derivatives_folder}/reshaped_into_4d_arrays', dC_LL_4D)
        np.save(f'{derivatives_folder}/reshaped_into_4d_arrays', dC_GG_4D)
        np.save(f'{derivatives_folder}/reshaped_into_4d_arrays', dC_WA_4D)
        np.save(f'{derivatives_folder}/reshaped_into_4d_arrays', dC_3x2pt_6D)

    # store the arrays of derivatives in a dictionary to pass to the Fisher Matrix function
    deriv_dict = {'dC_LL_4D': dC_LL_4D,
                  'dC_GG_4D': dC_GG_4D,
                  'dC_WA_4D': dC_WA_4D,
                  'dC_3x2pt_6D': dC_3x2pt_6D}

    fm_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict)
    fm_dict['param_names_dict'] = param_names_dict
    fm_dict['fiducial_values_dict'] = fiducials_dict

    # free memory, cov_dict is HUGE
    del cov_dict
    gc.collect()

    # ! save and test
    fm_folder = FM_cfg["fm_folder"].format(SSC_code=ssc_code)

    for probe in probes:

        lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_3x2pt']
        filename_fm_g = f'{fm_folder}/FM_{probe}_G_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'

        which_ng_cov_suffix = ''.join(covariance_cfg[covariance_cfg['SSC_code'] + '_cfg']['which_ng_cov'])
        filename_fm_from_ssc_code = filename_fm_g.replace('_G_', f'_G{which_ng_cov_suffix}_')

        np.savetxt(f'{filename_fm_g}', fm_dict[f'FM_{probe}_G'])
        np.savetxt(f'{filename_fm_from_ssc_code}', fm_dict[f'FM_{probe}_G{which_ng_cov_suffix}'])
        # np.savetxt(f'{filename_fm_from_ssc_code}', FM_dict[f'FM_{probe}_GSSCcNG'])

        # probe_ssc_code = covariance_cfg[f'{covariance_cfg["SSC_code"]}_cfg']['probe']
        # probe_ssc_code = 'WL' if probe_ssc_code == 'LL' else probe_ssc_code
        # probe_ssc_code = 'GC' if probe_ssc_code == 'GG' else probe_ssc_code

    if general_cfg['test_against_benchmarks']:
        mm.test_folder_content(fm_folder, fm_folder + '/benchmarks', FM_cfg['FM_file_format'])

################################################ ! plot ############################################################

# plot settings
nparams_toplot = 7
include_fom = True
divide_fom_by_10 = True

fm_dict_loaded = {}
for ssc_code_here in ['PySSC', 'PyCCL', 'Spaceborne', 'OneCovariance']:
    for probe in probes:

        fm_folder = FM_cfg["fm_folder"].format(SSC_code=ssc_code_here)
        # if 'jan_2024' in fm_folder:
        # fm_folder_std = fm_folder.replace("jan_2024", "standard")
        # else:
        # raise ValueError('you are not using the jan_2024 folder!')

        lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_XC']

        fm_dict_loaded[f'FM_{ssc_code_here}_{probe}_G'] = (
            np.genfromtxt(f'{fm_folder}/FM_{probe}_G_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))

        fm_dict_loaded[f'FM_{ssc_code_here}_{probe}_GSSC'] = (
            np.genfromtxt(f'{fm_folder}/FM_{probe}_GSSC_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))

        try:
            fm_dict_loaded[f'FM_{ssc_code_here}_{probe}_GSSCcNG'] = (
                np.genfromtxt(f'{fm_folder}/FM_{probe}_GSSCcNG_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))
        except FileNotFoundError:
            print(f'FM_{ssc_code_here}_{probe}_GSSCcNG not found')

        try:
            fm_dict_loaded[f'FM_{ssc_code_here}_{probe}_GcNG'] = (
                np.genfromtxt(f'{fm_folder}/FM_{probe}_GcNG_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))
        except FileNotFoundError:
            print(f'FM_{ssc_code_here}_{probe}_GcNG not found')

        # make sure that this file has been created very recently (aka, is the one just produced)
        if ssc_code_here == covariance_cfg['SSC_code']:
            assert mm.is_file_created_in_last_x_hours(f'{fm_folder}/FM_{probe}_G{which_ng_cov_suffix}_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt', 0.1), \
                'the file has not been created very recently'

        # # add the standard case
        # FM_dict_loaded[f'FM_{ssc_code_here}_{probe}_GSSC_std'] = (
        #     np.genfromtxt(f'{fm_folder_std}/FM_{probe}_GSSC_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))

# just a test: the Gaussian FMs must be equal. This is true also for OneCovariance if I do not use the OneCovariance Gaussian cov,
# of course. The baseline is PySSC, but it's an arbitrary choice.

ssc_code_here_list = ['PyCCL', 'Spaceborne']
if covariance_cfg['OneCovariance_cfg']['use_OneCovariance_Gaussian'] is False:
    ssc_code_here_list.append('OneCovariance')

for ssc_code_here in ssc_code_here_list:
    for probe in probes:
        np.testing.assert_allclose(fm_dict_loaded[f'FM_{ssc_code_here}_{probe}_G'],
                                   fm_dict_loaded[f'FM_PySSC_{probe}_G'],
                                   rtol=1e-5, atol=0,
                                   err_msg=f'Gaussian FMs are not equal for {ssc_code_here} and {probe}!')

# compute FoM
fom_dict = {}
uncert_dict = {}
masked_FM_dict = {}
for key in list(fm_dict_loaded.keys()):
    if key not in ['param_names_dict', 'fiducial_values_dict']:
        masked_FM_dict[key], param_names_list, fiducials_list = mm.mask_FM(fm_dict_loaded[key], param_names_dict,
                                                                           fiducials_dict,
                                                                           params_tofix_dict={})

        nparams = len(param_names_list)

        assert nparams == len(fiducials_list), f'number of parameters in the Fisher Matrix ({nparams}) '

        uncert_dict[key] = mm.uncertainties_FM(masked_FM_dict[key], nparams=masked_FM_dict[key].shape[0],
                                               fiducials=fiducials_list,
                                               which_uncertainty='marginal', normalize=True)[:nparams_toplot]
        fom_dict[key] = mm.compute_FoM(masked_FM_dict[key], w0wa_idxs=(2, 3))


# compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
for probe in probes:

    # for ssc_code in [ 'OneCovariance', 'PySSC', 'PyCCL', 'Spaceborne']:
    for ssc_code in ['OneCovariance', 'PyCCL']:
        cov_a = 'G'
        cov_b = 'GSSC'

        key_a = f'FM_{ssc_code}_{probe}_{cov_a}'
        key_b = f'FM_{ssc_code}_{probe}_{cov_b}'

        uncert_dict[f'perc_diff_{ssc_code}_{probe}_{cov_b}'] = mm.percent_diff(uncert_dict[key_b], uncert_dict[key_a])
        fom_dict[f'perc_diff_{ssc_code}_{probe}_{cov_b}'] = np.abs(mm.percent_diff(fom_dict[key_b], fom_dict[key_a]))

    # do the same for cNG
    # for ssc_code in ['OneCovariance', 'PyCCL']:
    for ssc_code in ['OneCovariance', ]:
        cov_a = 'G'

        for cov_b in ['GSSC', 'GcNG', 'GSSCcNG']:

            key_a = f'FM_{ssc_code}_{probe}_{cov_a}'
            key_b = f'FM_{ssc_code}_{probe}_{cov_b}'

            uncert_dict[f'perc_diff_{ssc_code}_{probe}_{cov_b}'] = mm.percent_diff(
                uncert_dict[key_b], uncert_dict[key_a])
            fom_dict[f'perc_diff_{ssc_code}_{probe}_{cov_b}'] = np.abs(
                mm.percent_diff(fom_dict[key_b], fom_dict[key_a]))

for probe in probes:
    nparams_toplot = 7
    divide_fom_by_10_plt = False if probe in ('WL' 'XC') else divide_fom_by_10

    cases_to_plot = [f'FM_PySSC_{probe}_G',
                     #  f'FM_OneCovariance_{probe}_G',

                     #  f'FM_PySSC_{probe}_GSSC',
                     # f'FM_PyCCL_{probe}_GSSC',
                     # f'FM_PyCCL_{probe}_GcNG',
                     # f'FM_PyCCL_{probe}_GSSCcNG',
                     # f'FM_Spaceborne_{probe}_GSSC',
                     f'FM_OneCovariance_{probe}_GSSC',
                     #  f'FM_OneCovariance_{probe}_GcNG',
                     f'FM_OneCovariance_{probe}_GSSCcNG',

                     #  f'perc_diff_PyCCL_{probe}_GSSC',
                     #  f'perc_diff_PyCCL_{probe}_GcNG',
                     #  f'perc_diff_PyCCL_{probe}_GSSCcNG',
                     # f'perc_diff_Spaceborne_{probe}_GSSC',
                     #  f'perc_diff_OneCovariance_{probe}_GSSC',
                     # f'perc_diff_OneCovariance_{probe}_GcNG',
                     #  f'perc_diff_OneCovariance_{probe}_GSSCcNG'
                     ]

    df = pd.DataFrame(uncert_dict)  # you should switch to using this...

    # # transform dict. into an array and add the fom
    uncert_array, fom_array = [], []

    for case in cases_to_plot:

        uncert_array.append(uncert_dict[case])
        if divide_fom_by_10 and 'FM' in case and 'WL' not in case:
            fom_dict[case] /= 10
        fom_array.append(fom_dict[case])

    uncert_array = np.asarray(uncert_array)
    fom_array = np.asarray(fom_array)

    uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

    # label and title stuff
    fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
    param_names_label = param_names_list[:nparams_toplot] + [fom_label] if include_fom else param_names_list[
        :nparams_toplot]
    lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_XC']
    title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i' % (probe, lmax, EP_or_ED, zbins)
    # bar plot
    if include_fom:
        nparams_toplot = 8

    for i, case in enumerate(cases_to_plot):

        cases_to_plot[i] = case
        if 'OneCovariance' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace('OneCovariance', 'OneCov')
        if f'PySSC_{probe}_G' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace(f'PySSC_{probe}_G', f'{probe}_G')

        cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'_GSSC', f'G+SSC')
        cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
        cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')
        cases_to_plot[i] = cases_to_plot[i].replace(f'OneCov', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'perc diff', f'% diff')

    plot_utils.bar_plot(uncert_array[:, :nparams_toplot], title, cases_to_plot, nparams=nparams_toplot,
                        param_names_label=None, bar_width=0.13,
                        include_fom=include_fom, divide_fom_by_10_plt=divide_fom_by_10_plt,
                        )

    # plt.savefig(f'/home/davide/Documenti/Science/Talks/2024_03_20 - Waterloo/{probe}_ISTF_GSSCcNG.png', bbox_inches='tight', dpi=300)


divide_fom_by_10_plt = False
divide_fom_by_10 = False
include_fom = False
nparams_toplot = 7

cases_to_plot = [
    f'perc_diff_OneCovariance_WL_GSSCcNG',
    f'perc_diff_OneCovariance_GC_GSSCcNG',
    f'perc_diff_OneCovariance_3x2pt_GSSCcNG',
    # f'perc_diff_OneCovariance_WL_GcNG',
    # f'perc_diff_OneCovariance_GC_GcNG',
    # f'perc_diff_OneCovariance_3x2pt_GcNG',
]

df = pd.DataFrame(uncert_dict)  # you should switch to using this...

# # transform dict. into an array and add the fom
uncert_array, fom_array = [], []

for case in cases_to_plot:

    uncert_array.append(uncert_dict[case])
    if divide_fom_by_10 and 'FM' in case and 'WL' not in case:
        fom_dict[case] /= 10
    fom_array.append(fom_dict[case])

uncert_array = np.asarray(uncert_array)
fom_array = np.asarray(fom_array)

uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

# label and title stuff
fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
param_names_label = param_names_list[:nparams_toplot] + [fom_label] if include_fom else param_names_list[
    :nparams_toplot]
lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_XC']
title = '$\\ell_{\\rm max} = %i$, zbins %s%i, G + SSC + cNG' % (lmax, EP_or_ED, zbins)
# bar plot
if include_fom:
    nparams_toplot = 8

for i, case in enumerate(cases_to_plot):

    cases_to_plot[i] = case
    if 'OneCovariance' in cases_to_plot[i]:
        cases_to_plot[i] = cases_to_plot[i].replace('OneCovariance', 'OneCov')
    if f'PySSC_{probe}_G' in cases_to_plot[i]:
        cases_to_plot[i] = cases_to_plot[i].replace(f'PySSC_{probe}_G', f'{probe}_G')

    cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
    cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
    cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
    cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')
    cases_to_plot[i] = cases_to_plot[i].replace(f'GcNG', f'G+cNG')
    cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')
    cases_to_plot[i] = cases_to_plot[i].replace('perc diff', '')
    cases_to_plot[i] = cases_to_plot[i].replace('OneCov', '')
    cases_to_plot[i] = cases_to_plot[i].replace('G+SSC+cNG', '')
    cases_to_plot[i] = cases_to_plot[i].replace('G+SSC', '')
    cases_to_plot[i] = cases_to_plot[i].replace('G+cNG', '')

plot_utils.bar_plot(uncert_array[:, :nparams_toplot], title, cases_to_plot, nparams=nparams_toplot,
                    param_names_label=None, bar_width=0.13, include_fom=include_fom, divide_fom_by_10_plt=divide_fom_by_10_plt,
                    ylabel='uncert. increase [%]')


# ! triangle plot
ssc_code = 'OneCovariance'
fm_triangle_list = [f'FM_{ssc_code}_3x2pt_GSSCcNG', f'FM_{ssc_code}_3x2pt_GSSC', f'FM_{ssc_code}_3x2pt_G']
samples_list = []
pars_toplot = 7
fid_pars_dict_here = deepcopy(cfg.fid_pars_dict_for_fm)
param_names_tex = mpl_cfg.general_dict['cosmo_labels_TeX']

for fm_name in fm_triangle_list:

    fm = fm_dict_loaded[fm_name]
    fm = mm.remove_null_rows_cols_2D_copilot(fm)

    if 'Om_Lambda0' in fid_pars_dict_here:
        del fid_pars_dict_here['Om_Lambda0']

    params_plt_list = list(fid_pars_dict_here.keys())[:pars_toplot]
    trimmed_fid_dict = {param: fid_pars_dict_here[param] for param in
                        params_plt_list}

    # get the covariance matrix (careful on how you cut the FM!!)
    fm_idxs_tokeep = [list(fid_pars_dict_here.keys()).index(param) for param in
                      params_plt_list]
    cov = np.linalg.inv(fm)[fm_idxs_tokeep, :][:, fm_idxs_tokeep]

    samples_list.append(GaussianND(mean=list(trimmed_fid_dict.values()), cov=cov, names=(trimmed_fid_dict.keys())))
    # samples_list.append(GaussianND(mean=list(trimmed_fid_dict.values()), cov=cov, names=(param_names_tex)))

g = plots.get_subplot_plotter()
g.settings.linewidth = 2
g.settings.legend_fontsize = 30
g.settings.linewidth_contour = 2.5
g.settings.axes_fontsize = 27
g.settings.axes_labelsize = 30
g.settings.subplot_size_ratio = 1
g.settingstight_layout = True
g.settings.solid_colors = 'tab10'
g.triangle_plot(samples_list, filled=True, contour_lws=1.4,
                legend_labels=['G+SSC+cNG', 'G+SSC', 'G'], legend_loc='upper right')
plt.suptitle('$3\\times 2{\\rm pt}$', fontsize='xx-large')


# ! silent check against IST:F (which does not exist for GC alone):
for which_probe in ['WL', '3x2pt']:
    np.set_printoptions(precision=2)
    print('\nprobe:', which_probe)
    uncert_dict['ISTF'] = ISTF_fid.forecasts[f'{which_probe}_opt_w0waCDM_flat']
    try:
        rtol = 10e-2
        assert np.allclose(uncert_dict[f'FM_PySSC_{which_probe}_G'][:nparams_toplot], uncert_dict['ISTF'], atol=0,
                           rtol=rtol)
        print(f'IST:F and GO are consistent for probe {which_probe} within {rtol * 100}% ✅')
    except AssertionError:
        print(f'IST:F and GO are not consistent for probe {which_probe} within {rtol * 100}% ❌')
        print('(remember that you are checking against the optimistic case, with lmax_WL = 5000. '
              f'\nYour lmax_WL is {ell_max_WL})')
        print('ISTF GO:\t', uncert_dict['ISTF'])
        print('Spaceborne GO:\t', uncert_dict[f'FM_PySSC_{which_probe}_G'][:nparams_toplot])
        print('percent_discrepancies (*not wrt mean!*):\n',
              mm.percent_diff(uncert_dict[f'FM_PySSC_{which_probe}_G'][:nparams_toplot],
                              uncert_dict['ISTF']))

        print('Spaceborne GS:\t', uncert_dict[f'FM_{ssc_code}_{which_probe}_GSSC'][:nparams_toplot])


print('done')

# veeeeery old FMs, to test ISTF-like forecasts I guess...
# FM_test_G = np.genfromtxt(
#     '/home/davide/Documenti/Lavoro/Programmi/!archive/SSC_restructured_v2_didntmanagetopush/jobs'
#     '/SSC_comparison/output/FM/FM_3x2pt_G_lmaxXC3000_nbl30.txt')
# FM_test_GSSC = np.genfromtxt(
#     '/home/davide/Documenti/Lavoro/Programmi/!archive/SSC_restructured_v2_didntmanagetopush/jobs'
#     '/SSC_comparison/output/FM/FM_3x2pt_GSSC_lmaxXC3000_nbl30.txt')
# uncert_FM_G_test = mm.uncertainties_FM(FM_test_G, FM_test_G.shape[0], fiducials=fiducials_list,
#                                         which_uncertainty='marginal',
#                                         normalize=True)[:nparams_toplot]
# uncert_FM_GSSC_test = mm.uncertainties_FM(FM_test_GSSC, FM_test_GSSC.shape[0], fiducials=fiducials_list,
#                                         which_uncertainty='marginal',
#                                         normalize=True)[:nparams_toplot]
