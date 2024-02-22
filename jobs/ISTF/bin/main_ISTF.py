import gc
import sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pprint import pprint
import warnings
import pandas as pd
from matplotlib import cm

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


# for covariance_cfg['SSC_code'] in ['PySSC', 'exactSSC', 'PyCCL', 'OneCovariance']:
for covariance_cfg['SSC_code'] in (covariance_cfg['SSC_code'], ):
    # check_specs.consistency_checks(general_cfg, covariance_cfg)
    # for covariance_cfg['SSC_code'] in ['PyCCL', 'exactSSC']:
    #     for covariance_cfg[covariance_cfg['SSC_code'] + '_cfg']['probe'] in ['LL', 'GG', '3x2pt']:
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
    
    
    
    # # ! check the difference in the Gaussian covariances
    
    # fsky = covariance_cfg['fsky']
    # nbl_3x2pt = nbl_GC
    # ell_3x2pt = ell_GC
    # probe_ordering = (('L', 'L'), ('G', 'L'), ('G', 'G'))
    
    
    
    #     # build noise vector
    # noise_3x2pt_4D = mm.build_noise(zbins, n_probes, sigma_eps2=covariance_cfg['sigma_eps2'], ng=covariance_cfg['ng'],
    #                                 EP_or_ED=general_cfg['EP_or_ED'])

    # # create dummy ell axis, the array is just repeated along it
    # nbl_max = np.max((nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA))
    # noise_5D = np.zeros((n_probes, n_probes, nbl_max, zbins, zbins))
    # for probe_A in (0, 1):
    #     for probe_B in (0, 1):
    #         for ell_idx in range(nbl_WL):
    #             noise_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

    # # remember, the ell axis is a dummy one for the noise, is just needs to be of the
    # # same length as the corresponding cl one
    # noise_LL_5D = noise_5D[0, 0, :nbl_WL, :, :][np.newaxis, np.newaxis, ...]
    # noise_GG_5D = noise_5D[1, 1, :nbl_GC, :, :][np.newaxis, np.newaxis, ...]
    # noise_WA_5D = noise_5D[0, 0, :nbl_WA, :, :][np.newaxis, np.newaxis, ...]
    # noise_3x2pt_5D = noise_5D[:, :, :nbl_3x2pt, :, :]


    # start = time.perf_counter()
    # cl_LL_5D = cl_dict_3D['cl_LL_3D'][np.newaxis, np.newaxis, ...]
    # cl_GG_5D = cl_dict_3D['cl_GG_3D'][np.newaxis, np.newaxis, ...]
    # cl_WA_5D = cl_dict_3D['cl_WA_3D'][np.newaxis, np.newaxis, ...]

    # # 5d versions of auto-probe spectra
    # cov_WL_GO_6D = mm.covariance_einsum(cl_LL_5D, noise_LL_5D, fsky, ell_WL, delta_dict['delta_l_WL'])[0, 0, 0, 0, ...]
    # cov_GC_GO_6D = mm.covariance_einsum(cl_GG_5D, noise_GG_5D, fsky, ell_GC, delta_dict['delta_l_GC'])[0, 0, 0, 0, ...]
    # cov_WA_GO_6D = mm.covariance_einsum(cl_WA_5D, noise_WA_5D, fsky, ell_WA, delta_dict['delta_l_WA'])[0, 0, 0, 0, ...]
    # cov_3x2pt_GO_10D = mm.covariance_einsum(cl_dict_3D['cl_3x2pt_5D'], noise_3x2pt_5D, fsky, ell_3x2pt, delta_dict['delta_l_3x2pt'])
    
    # cov_WL_SN_6D = mm.covariance_einsum(np.zeros_like(cl_LL_5D), noise_LL_5D, fsky, ell_WL, delta_dict['delta_l_WL'], prefactor=1)[0, 0, 0, 0, ...]
    # cov_GC_SN_6D = mm.covariance_einsum(np.zeros_like(cl_GG_5D), noise_GG_5D, fsky, ell_GC, delta_dict['delta_l_GC'], prefactor=1)[0, 0, 0, 0, ...]
    # cov_WA_SN_6D = mm.covariance_einsum(np.zeros_like(cl_WA_5D), noise_WA_5D, fsky, ell_WA, delta_dict['delta_l_WA'], prefactor=1)[0, 0, 0, 0, ...]
    # cov_3x2pt_SN_10D = mm.covariance_einsum(np.zeros_like(cl_dict_3D['cl_3x2pt_5D']), noise_3x2pt_5D, fsky, ell_3x2pt, delta_dict['delta_l_3x2pt'], prefactor=1)
    
    # cov_WL_SVA_6D = mm.covariance_einsum(cl_LL_5D, np.zeros_like(noise_LL_5D), fsky, ell_WL, delta_dict['delta_l_WL'])[0, 0, 0, 0, ...]
    # cov_GC_SVA_6D = mm.covariance_einsum(cl_GG_5D, np.zeros_like(noise_GG_5D), fsky, ell_GC, delta_dict['delta_l_GC'])[0, 0, 0, 0, ...]
    # cov_WA_SVA_6D = mm.covariance_einsum(cl_WA_5D, np.zeros_like(noise_WA_5D), fsky, ell_WA, delta_dict['delta_l_WA'])[0, 0, 0, 0, ...]
    # cov_3x2pt_SVA_10D = mm.covariance_einsum(cl_dict_3D['cl_3x2pt_5D'], np.zeros_like(noise_3x2pt_5D), fsky, ell_3x2pt, delta_dict['delta_l_3x2pt'])
    
    # ind_auto = ind[:zpairs_auto, :].copy()
    # ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()
    # ind_dict = {('L', 'L'): ind_auto,
    #             ('G', 'L'): ind_cross,
    #             ('G', 'G'): ind_auto}
    # covariance_cfg['ind_dict'] = ind_dict
    # from copy import deepcopy
    

    # cov_path = '/home/davide/Documenti/Lavoro/Programmi/OneCovariance/output_ISTF_v2'
    # cov_SN_filename = covariance_cfg['OneCovariance_cfg']['cov_filename'].format(
    # which_ng_cov='SN', probe_a='{probe_a:s}', probe_b='{probe_b:s}',
    # probe_c='{probe_c:s}', probe_d='{probe_d}', nbl=nbl, lmax=3000,
    # EP_or_ED=general_cfg['EP_or_ED'],
    # zbins=zbins)
    # cov_MIX_filename = cov_SN_filename.replace('_SN_', '_MIX_')
    # cov_SVA_filename = cov_SN_filename.replace('_SN_', '_SVA_')
    # cov_G_filename = cov_SN_filename.replace('_SN_', '_G_')


    # # load SSC blocks in 4D and store them into a dictionary of 8D blocks
    # cov_SN_3x2pt_dict_8D_OC = mm.load_cov_from_probe_blocks(cov_path, cov_SN_filename, probe_ordering)
    # cov_MIX_3x2pt_dict_8D_OC = mm.load_cov_from_probe_blocks(cov_path, cov_MIX_filename, probe_ordering)
    # cov_SVA_3x2pt_dict_8D_OC = mm.load_cov_from_probe_blocks(cov_path, cov_SVA_filename, probe_ordering)
    # cov_G_3x2pt_dict_8D_OC = mm.load_cov_from_probe_blocks(cov_path, cov_G_filename, probe_ordering)
    
    # # reshape the blocks in the dictionary from 4D to 6D, as needed by the BNT
    # cov_SN_3x2pt_dict_10D_OC = {}
    # cov_SVA_3x2pt_dict_10D_OC = {}
    # cov_MIX_3x2pt_dict_10D_OC = {}
    # cov_G_3x2pt_dict_10D_OC = {}
    # for probe_A, probe_B in probe_ordering:
    #     for probe_C, probe_D in probe_ordering:
    #         cov_SN_3x2pt_dict_10D_OC[probe_A, probe_B, probe_C, probe_D] = mm.cov_4D_to_6D_blocks(
    #             cov_SN_3x2pt_dict_8D_OC[probe_A, probe_B, probe_C, probe_D],
    #             nbl, zbins, ind_dict[probe_A, probe_B], ind_dict[probe_C, probe_D])
    #         cov_MIX_3x2pt_dict_10D_OC[probe_A, probe_B, probe_C, probe_D] = mm.cov_4D_to_6D_blocks(
    #             cov_MIX_3x2pt_dict_8D_OC[probe_A, probe_B, probe_C, probe_D],
    #             nbl, zbins, ind_dict[probe_A, probe_B], ind_dict[probe_C, probe_D])
    #         cov_SVA_3x2pt_dict_10D_OC[probe_A, probe_B, probe_C, probe_D] = mm.cov_4D_to_6D_blocks(
    #             cov_SVA_3x2pt_dict_8D_OC[probe_A, probe_B, probe_C, probe_D],
    #             nbl, zbins, ind_dict[probe_A, probe_B], ind_dict[probe_C, probe_D])
    #         cov_G_3x2pt_dict_10D_OC[probe_A, probe_B, probe_C, probe_D] = mm.cov_4D_to_6D_blocks(
    #             cov_G_3x2pt_dict_8D_OC[probe_A, probe_B, probe_C, probe_D],
    #             nbl, zbins, ind_dict[probe_A, probe_B], ind_dict[probe_C, probe_D])
            
        
    # cov_3x2pt_SN_4D_OC = mm.cov_3x2pt_10D_to_4D(cov_SN_3x2pt_dict_10D_OC, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
    # cov_3x2pt_SN_4D_SB = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SN_10D, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
    # cov_3x2pt_SVA_4D_OC = mm.cov_3x2pt_10D_to_4D(cov_SVA_3x2pt_dict_10D_OC, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
    # cov_3x2pt_SVA_4D_SB = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SVA_10D, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
    
    # cov_3x2pt_SN_2D_OC = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_SN_4D_OC, zbins, block_index='vincenzo')
    # cov_3x2pt_SN_2D_SB = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_SN_4D_SB, zbins, block_index='vincenzo')
    # cov_3x2pt_SVA_2D_OC = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_SVA_4D_OC, zbins, block_index='vincenzo')
    # cov_3x2pt_SVA_2D_SB = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_SVA_4D_SB, zbins, block_index='vincenzo')
    
    # cov_3x2pt_SN_diag_OC = np.diag(cov_3x2pt_SN_2D_OC)
    # cov_3x2pt_SN_diag_SB = np.diag(cov_3x2pt_SN_2D_SB)
    
    # plt.figure()
    # # plt.plot(cov_3x2pt_SN_diag_OC, label='OC')
    # # plt.plot(cov_3x2pt_SN_diag_SB, label='SB', ls='--')
    # plt.plot(cov_3x2pt_SN_diag_SB/cov_3x2pt_SN_diag_OC, label='ratio', ls='-', marker='.')
    # plt.legend()
    
    
    # A = cov_3x2pt_SVA_2D_OC
    # B = cov_3x2pt_SVA_2D_SB
    # diff = mm.percent_diff(A, B)
    # mm.matshow(diff, log=True, abs_val=True)
    
    # log_diff = False
    # abs_val = False
    # plot_diff_threshold = 5
    # mm.compare_arrays(A, B, plot_diff_threshold=plot_diff_threshold)
    
    # diff_AB = mm.percent_diff_nan(A, B, eraseNaN=True, log=log_diff, abs_val=abs_val)

    # if plot_diff_threshold is not None:
    #     # take the log of the threshold if using the log of the precent difference
    #     if log_diff:
    #         plot_diff_threshold = np.log10(plot_diff_threshold)

    #     diff_AB = np.ma.masked_where(np.abs(diff_AB) < plot_diff_threshold, np.abs(diff_AB))

    # fig, ax = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True)
    # im = ax[0].matshow(diff_AB)
    # ax[0].set_title(f'(A/B - 1) * 100')
    # fig.colorbar(im, ax=ax[0])

    # im = ax[1].matshow(diff_AB)
    # ax[1].set_title(f'(A/B - 1) * 100')
    # fig.colorbar(im, ax=ax[1])

    # fig.suptitle(f'log={log_diff}, abs={abs_val}')
    # plt.show()



    
    # assert False, 'stop here to check cov G'
    
    
    
    
    
    
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

    FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict)
    FM_dict['param_names_dict'] = param_names_dict
    FM_dict['fiducial_values_dict'] = fiducials_dict

    # free memory, cov_dict is HUGE
    del cov_dict
    gc.collect()
    

    # ! save and test
    fm_folder = FM_cfg["fm_folder"].format(SSC_code=ssc_code)

    for probe in ('WL', 'GC', 'XC', '3x2pt'):
        
        lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_3x2pt']
        filename_fm_g = f'{fm_folder}/FM_{probe}_G_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'
        
        which_ng_cov_suffix = ''.join(covariance_cfg[covariance_cfg['SSC_code'] + '_cfg']['which_ng_cov'])
        filename_fm_from_ssc_code = filename_fm_g.replace('_G_', f'_G{which_ng_cov_suffix}_')

        np.savetxt(f'{filename_fm_g}', FM_dict[f'FM_{probe}_G'])
        np.savetxt(f'{filename_fm_from_ssc_code}', FM_dict[f'FM_{probe}_G{which_ng_cov_suffix}'])
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

FM_dict_loaded = {}
for ssc_code_here in ['PySSC', 'PyCCL', 'exactSSC', 'OneCovariance']:
    for probe in ['WL', 'GC', 'XC', '3x2pt']:

        fm_folder = FM_cfg["fm_folder"].format(SSC_code=ssc_code_here)
        if 'jan_2024' in fm_folder:
            fm_folder_std = fm_folder.replace("jan_2024", "standard")
        else:
            raise ValueError('you are not using the jan_2024 folder!')

        lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_XC']
        
        FM_dict_loaded[f'FM_{ssc_code_here}_{probe}_G'] = (
            np.genfromtxt(f'{fm_folder}/FM_{probe}_G_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))

        FM_dict_loaded[f'FM_{ssc_code_here}_{probe}_GSSC'] = (
            np.genfromtxt(f'{fm_folder}/FM_{probe}_GSSC_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))
        
        try:
            FM_dict_loaded[f'FM_{ssc_code_here}_{probe}_GSSCcNG'] = (
                np.genfromtxt(f'{fm_folder}/FM_{probe}_GSSCcNG_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))
        except FileNotFoundError:
            print(f'FM_{ssc_code_here}_{probe}_GSSCcNG not found')

        # make sure that this file has been created very recently (aka, is the one just produced)
        mm.is_file_created_in_last_x_hours(f'{fm_folder}/FM_{probe}_GSSC_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt', 0.1)

        # # add the standard case
        # FM_dict_loaded[f'FM_{ssc_code_here}_{probe}_GSSC_std'] = (
        #     np.genfromtxt(f'{fm_folder_std}/FM_{probe}_GSSC_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))

# just a test: the Gaussian FMs must be equal. This is true also for OneCovariance if I do not use the OneCovariance Gaussian cov,
# of course. The baseline is PySSC, but it's an arbitrary choice.

ssc_code_here_list = ['PyCCL', 'exactSSC']
if covariance_cfg['OneCovariance_cfg']['use_OneCovariance_Gaussian'] is False:
    ssc_code_here_list.append('OneCovariance')

for ssc_code_here in ssc_code_here_list:
    for probe in ['WL', 'GC', 'XC', '3x2pt']:
        np.testing.assert_allclose(FM_dict_loaded[f'FM_{ssc_code_here}_{probe}_G'], 
                                   FM_dict_loaded[f'FM_PySSC_{probe}_G'],
                                   rtol=1e-5, atol=0,
                                   err_msg=f'Gaussian FMs are not equal for {ssc_code_here} and {probe}!')

# compute FoM
fom_dict = {}
uncert_dict = {}
masked_FM_dict = {}
for key in list(FM_dict_loaded.keys()):
    if key not in ['param_names_dict', 'fiducial_values_dict']:
        masked_FM_dict[key], param_names_list, fiducials_list = mm.mask_FM(FM_dict_loaded[key], param_names_dict,
                                                                           fiducials_dict,
                                                                           params_tofix_dict={})

        nparams = len(param_names_list)

        assert nparams == len(fiducials_list), f'number of parameters in the Fisher Matrix ({nparams}) '

        uncert_dict[key] = mm.uncertainties_FM(masked_FM_dict[key], nparams=masked_FM_dict[key].shape[0],
                                               fiducials=fiducials_list,
                                               which_uncertainty='marginal', normalize=True)[:nparams_toplot]
        fom_dict[key] = mm.compute_FoM(masked_FM_dict[key], w0wa_idxs=(2, 3))


# compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
for probe in ['WL', 'GC', 'XC', '3x2pt']:

    for ssc_code in ['PySSC', 'PyCCL', 'exactSSC', 'OneCovariance']:
        key_a = f'FM_{ssc_code}_{probe}_G'
        key_b = f'FM_{ssc_code}_{probe}_GSSC'
        
        uncert_dict[f'perc_diff_{ssc_code}_{probe}_GSSC'] = mm.percent_diff(uncert_dict[key_b], uncert_dict[key_a])
        fom_dict[f'perc_diff_{ssc_code}_{probe}_GSSC'] = np.abs(mm.percent_diff(fom_dict[key_b], fom_dict[key_a]))
        
    # do the same for cNG
    # for ssc_code in ['OneCovariance',]:
    #     key_a = f'FM_{ssc_code}_{probe}_G'
    #     key_b = f'FM_{ssc_code}_{probe}_GSSCcNG'
        
    #     uncert_dict[f'perc_diff_{ssc_code}_{probe}_GSSCcNG'] = mm.percent_diff(uncert_dict[key_b], uncert_dict[key_a])
    #     fom_dict[f'perc_diff_{ssc_code}_{probe}_GSSCcNG'] = np.abs(mm.percent_diff(fom_dict[key_b], fom_dict[key_a]))
        
for probe in ['WL', 'GC', 'XC', '3x2pt']:
    nparams_toplot = 7
    divide_fom_by_10_plt = False if probe in ('WL' 'XC') else divide_fom_by_10

    cases_to_plot = [f'FM_PySSC_{probe}_G', 
                    #  f'FM_OneCovariance_{probe}_G', 
                    
                    #  f'FM_PySSC_{probe}_GSSC', 
                     f'FM_PyCCL_{probe}_GSSC', 
                     f'FM_exactSSC_{probe}_GSSC', 
                    #  f'FM_OneCovariance_{probe}_GSSC',
                    #  f'FM_OneCovariance_{probe}_GSSCcNG',
                    
                     f'perc_diff_PyCCL_{probe}_GSSC', 
                    f'perc_diff_exactSSC_{probe}_GSSC', 
                    # f'perc_diff_OneCovariance_{probe}_GSSC', 
                    # f'perc_diff_OneCovariance_{probe}_GSSCcNG'
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
        cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
        cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')
        cases_to_plot[i] = cases_to_plot[i].replace(f'SSCcNG', f'SSC+cNG')
    
    plot_utils.bar_plot(uncert_array[:, :nparams_toplot], title, cases_to_plot, nparams=nparams_toplot,
                        param_names_label=None, bar_width=0.13, include_fom=include_fom, divide_fom_by_10_plt=divide_fom_by_10_plt)
    # plt.yscale('log')
    
    # plt.savefig(f'{fm_folder}/{title}.png', dpi=400)


# silent check against IST:F (which does not exist for GC alone):
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

