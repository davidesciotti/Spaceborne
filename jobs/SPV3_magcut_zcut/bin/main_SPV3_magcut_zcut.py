import pickle
import sys
import time
from pathlib import Path
import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import warnings
import scipy.sparse as spar

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

# general libraries
sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

# general configurations
sys.path.append(f'{project_path.parent}/common_data/common_config')
import mpl_cfg

# job configuration
sys.path.append(f'{job_path}/config')
import config_SPV3_magcut_zcut as cfg

# project libraries
sys.path.append(f'{project_path}/bin')
import ell_values_running as ell_utils
import Cl_preprocessing_running as cl_utils
import compute_Sijkl as Sijkl_utils
import covariance_running as covmat_utils
import FM_running as FM_utils
import utils_running as utils
import unit_test

matplotlib.use('Qt5Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)

start_time = time.perf_counter()

# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks
# TODO super check that things work with different # of z bins

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_cfg = cfg.general_config
covariance_cfg = cfg.covariance_config
Sijkl_cfg = cfg.Sijkl_config
FM_cfg = cfg.FM_config

which_probe_response = covariance_cfg['which_probe_response']
# set the string, just for the file names
if which_probe_response == 'constant':
    which_probe_response_str = 'const'
elif which_probe_response == 'variable':
    which_probe_response_str = 'var'
else:
    raise ValueError('which_probe_response must be either constant or variable')

    # for general_cfg['zbins'] in general_cfg['zbins_list']:
    # for (general_cfg['ell_max_WL'], general_cfg['ell_max_GC']) in ((5000, 3000),):
    # for general_cfg['EP_or_ED'] in ('ED',):
for general_cfg['magcut_lens'] in general_cfg['magcut_lens_list']:
    for general_cfg['magcut_source'] in general_cfg['magcut_source_list']:
        for general_cfg['zcut_lens'] in general_cfg['zcut_lens_list']:
            for general_cfg['zcut_source'] in general_cfg['zcut_source_list']:

                # utils.consistency_checks(general_cfg, covariance_cfg)

                # some variables used for I/O naming, just to make things more readable
                zbins = general_cfg['zbins']
                EP_or_ED = general_cfg['EP_or_ED']
                ell_max_WL = general_cfg['ell_max_WL']
                ell_max_GC = general_cfg['ell_max_GC']
                ell_max_XC = ell_max_GC
                nbl_WL_32 = general_cfg['nbl_WL_32']
                magcut_source = general_cfg['magcut_source']
                magcut_lens = general_cfg['magcut_lens']
                zcut_source = general_cfg['zcut_source']
                zcut_lens = general_cfg['zcut_lens']
                zmax = int(general_cfg['zmax'] * 10)

                assert general_cfg['flagship_version'] == 2, 'The input files used in this job for flagship version 2!'

                # import the ind files and store it into the covariance dictionary
                ind_folder = covariance_cfg['ind_folder']
                ind_filename = covariance_cfg['ind_filename'].format(ind_ordering=covariance_cfg['ind_ordering'],
                                                                     zbins=zbins)
                ind = np.genfromtxt(f'{ind_folder}/{ind_filename}', dtype=int)
                covariance_cfg['ind'] = ind

                assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
                    'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

                # compute ell and delta ell values in the reference (optimistic) case
                ell_WL_nbl32, delta_l_WL_nbl32 = ell_utils.compute_ells(general_cfg['nbl_WL_32'],
                                                                        general_cfg['ell_min'],
                                                                        general_cfg['ell_max_WL_opt'], recipe='ISTF')

                ell_WL_nbl32 = np.log10(ell_WL_nbl32)

                # perform the cuts
                ell_dict = {}
                ell_dict['ell_WL'] = np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_WL])
                ell_dict['ell_GC'] = np.copy(ell_WL_nbl32[10 ** ell_WL_nbl32 < ell_max_GC])
                ell_dict['ell_WA'] = np.copy(
                    ell_WL_nbl32[(10 ** ell_WL_nbl32 > ell_max_GC) & (10 ** ell_WL_nbl32 < ell_max_WL)])
                ell_dict['ell_XC'] = np.copy(ell_dict['ell_GC'])

                # set corresponding # of ell bins
                nbl_WL = ell_dict['ell_WL'].shape[0]
                nbl_GC = ell_dict['ell_GC'].shape[0]
                nbl_WA = ell_dict['ell_WA'].shape[0]
                nbl_3x2pt = nbl_GC
                general_cfg['nbl_WL'] = nbl_WL

                delta_dict = {}
                delta_dict['delta_l_WL'] = np.copy(delta_l_WL_nbl32[:nbl_WL])
                delta_dict['delta_l_GC'] = np.copy(delta_l_WL_nbl32[:nbl_GC])
                delta_dict['delta_l_WA'] = np.copy(delta_l_WL_nbl32[nbl_GC:])

                # set # of nbl in the opt case, import and reshape, then cut the reshaped datavectors in the pes case
                nbl_WL_opt = 32
                nbl_GC_opt = 29
                nbl_WA_opt = 3
                nbl_3x2pt_opt = 29

                if ell_max_WL == general_cfg['ell_max_WL_opt']:
                    assert (nbl_WL_opt, nbl_GC_opt, nbl_WA_opt, nbl_3x2pt_opt) == (nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt), \
                        'nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt don\'t match with the expected values for the optimistic case'

                # ! import datavectors (cl) and response functions (rl)
                # this is just to make the .format() more compact
                clrl_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins,
                              'magcut_lens': magcut_lens, 'zcut_lens': zcut_lens,
                              'magcut_source': magcut_source, 'zcut_source': zcut_source}

                cl_fld = general_cfg['cl_folder']
                cl_filename = general_cfg['cl_filename']
                cl_ll_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='WLO', **clrl_specs)}")
                cl_gg_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='GCO', **clrl_specs)}")
                cl_wa_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='WLA', **clrl_specs)}")
                cl_3x2pt_1d = np.genfromtxt(f"{cl_fld}/{cl_filename.format(probe='3x2pt', **clrl_specs)}")

                rl_fld = general_cfg['rl_folder']
                rl_filename = general_cfg['rl_filename']
                rl_ll_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLO', **clrl_specs)}")
                rl_gg_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='GCO', **clrl_specs)}")
                rl_wa_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLA', **clrl_specs)}")
                rl_3x2pt_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='3x2pt', **clrl_specs)}")

                # ! reshape to 3 dimensions
                cl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(cl_ll_1d, 'WL', nbl_WL_opt, zbins)
                cl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(cl_gg_1d, 'GC', nbl_GC_opt, zbins)
                cl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(cl_wa_1d, 'WA', nbl_WA_opt, zbins)
                cl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(cl_3x2pt_1d, '3x2pt', nbl_3x2pt_opt, zbins)

                rl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(rl_ll_1d, 'WL', nbl_WL_opt, zbins)
                rl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(rl_gg_1d, 'GC', nbl_GC_opt, zbins)
                rl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(rl_wa_1d, 'WA', nbl_WA_opt, zbins)
                rl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(rl_3x2pt_1d, '3x2pt', nbl_3x2pt_opt, zbins)

                # ! BNT transform if the corresponding flag is set to True
                if general_cfg['cl_BNT_transform']:
                    assert general_cfg['EP_or_ED'] == 'ED', 'cl_BNT_transform is only available for ED'
                    assert general_cfg['zbins'] == 13, 'cl_BNT_transform is only available for zbins=13'
                    BNT_matrix = np.genfromtxt(f'{general_cfg["BNT_matrix_path"]}/{general_cfg["BNT_matrix_filename"]}',
                                               delimiter=',')
                    cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, BNT_matrix)
                    cl_gg_3d = cl_utils.cl_BNT_transform(cl_gg_3d, BNT_matrix)
                    cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, BNT_matrix)
                    cl_3x2pt_5d = cl_utils.cl_BNT_transform(cl_3x2pt_5d, BNT_matrix)
                    print('you shuld BNT transform the responses do this with the responses too!')

                # check that cl_wa is equal to cl_ll in the last nbl_WA_opt bins
                if ell_max_WL == general_cfg['ell_max_WL_opt']:
                    if not np.array_equal(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :]):
                        rtol = 1e-5
                        # plt.plot(ell_dict['ell_WL'], cl_ll_3d[:, 0, 0])
                        # plt.plot(ell_dict['ell_WL'][nbl_GC:nbl_WL], cl_wa_3d[:, 0, 0])
                        assert (np.allclose(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :], rtol=rtol, atol=0)), \
                            'cl_wa_3d should be obtainable from cl_ll_3d!'
                        print(f'cl_wa_3d and cl_ll_3d[nbl_GC:nbl_WL, :, :] are not exactly equal, but have a relative '
                              f'difference of less than {rtol}')

                # cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
                if ell_max_WL == 1500:
                    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
                    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
                    cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
                    cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

                    rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
                    rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
                    rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
                    rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

                # store cls and responses in a dictionary
                cl_dict_3D = {
                    'C_LL_WLonly_3D': cl_ll_3d,
                    'C_GG_3D': cl_gg_3d,
                    'C_WA_3D': cl_wa_3d,
                    'C_3x2pt_5D': cl_3x2pt_5d}

                rl_dict_3D = {
                    'R_LL_WLonly_3D': rl_ll_3d,
                    'R_GG_3D': rl_gg_3d,
                    'R_WA_3D': rl_wa_3d,
                    'R_3x2pt_5D': rl_3x2pt_5d}

                # ! load kernels
                WF_fld = Sijkl_cfg["wf_input_folder"]
                WF_filename = Sijkl_cfg["wf_input_filename"]
                wf_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins,
                            'magcut_source': magcut_source, 'zcut_source': zcut_source}
                wil = np.genfromtxt(f'{WF_fld}/{WF_filename.format(which_WF="WiWL", **wf_specs)}')
                wig = np.genfromtxt(f'{WF_fld}/{WF_filename.format(which_WF="WiGC", **wf_specs)}')

                # preprocess (remove redshift column)
                z_arr, wil = Sijkl_utils.preprocess_wf(wil, zbins)
                z_arr_2, wig = Sijkl_utils.preprocess_wf(wig, zbins)
                assert np.array_equal(z_arr, z_arr_2), 'the redshift arrays are different for the GC and WL kernels'

                # transpose and stack, ordering is important here!
                transp_stacked_wf = np.vstack((wil.T, wig.T))

                # ! compute or load Sijkl
                nz = z_arr.shape[0]  # get number of z points in nz to name the Sijkl file
                Sijkl_folder = Sijkl_cfg['Sijkl_folder']
                Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(flagship_version=general_cfg['flagship_version'],
                                                                    nz=nz, EP_or_ED=EP_or_ED, zbins=zbins,
                                                                    IA_flag=Sijkl_cfg['IA_flag'],
                                                                    magcut_source=magcut_source,
                                                                    zcut_source=zcut_source)
                # if Sijkl exists, load it; otherwise, compute it and save it
                if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):
                    print(f'Sijkl matrix already exists in {Sijkl_folder}; loading it')
                    Sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')
                else:
                    Sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                                      Sijkl_cfg['WF_normalization'])
                    np.save(f'{Sijkl_folder}/{Sijkl_filename}', Sijkl)

                # ! compute covariance matrix
                if covariance_cfg['compute_covmat']:
                    ng_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins, 'zcut_source': zcut_source,
                                'zmax': zmax, 'magcut_source': magcut_source}
                    ng_filename = f'{covariance_cfg["ng_filename"].format(**ng_specs)}'
                    covariance_cfg['ng'] = np.genfromtxt(f'{covariance_cfg["ng_folder"]}/'f'{ng_filename}')[:, 1]
                    cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                                        ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl)

                # ! compute Fisher Matrix
                if FM_cfg['compute_FM']:
                    FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict)

                # ! save cls and responses:
                # this is just to set the correct probe names
                probe_dav_dict = {
                    'WL': 'LL_WLonly_3D',
                    'GC': 'GG_3D',
                    'WA': 'WA_3D',
                    '3x2pt': '3x2pt_5D'}

                # just a dict for the output file names
                clrl_dict = {
                    'cl_dict_3D': cl_dict_3D,
                    'rl_dict_3D': rl_dict_3D,
                    'cl_inputname': 'dv',
                    'rl_inputname': 'rf',
                    'cl_dict_key': 'C',
                    'rl_dict_key': 'R',
                }
                for cl_or_rl in ['cl', 'rl']:
                    if general_cfg[f'save_{cl_or_rl}s_3d']:

                        for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
                            # save cl and/or response; not very readable but it works, plus all the cases are in the for loop

                            filepath = f'{general_cfg[f"{cl_or_rl}_folder"]}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}'
                            filename = general_cfg[f'{cl_or_rl}_filename'].format(probe=probe_vinc, **clrl_specs
                                                                                  ).replace(".dat", "_3D.npy")
                            file = clrl_dict[f"{cl_or_rl}_dict_3D"][
                                f'{clrl_dict[f"{cl_or_rl}_dict_key"]}_{probe_dav_dict[probe_dav]}']
                            np.save(f'{filepath}/{filename}', file)

                            # save ells and deltas
                            if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                                filepath = f'{general_cfg[f"{cl_or_rl}_folder"]}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}'
                                ells_filename = f'ell_{probe_dav}_ellmaxWL{ell_max_WL}'
                                np.savetxt(f'{filepath}/{ells_filename}.txt', 10 ** ell_dict[f'ell_{probe_dav}'])
                                np.savetxt(f'{filepath}/delta_{ells_filename}.txt', delta_dict[f'delta_l_{probe_dav}'])

                # ! save covariance:
                if covariance_cfg['cov_file_format'] == 'npy':
                    save_funct = np.save
                    extension = 'npy'
                elif covariance_cfg['cov_file_format'] == 'npz':
                    save_funct = np.savez_compressed
                    extension = 'npz'
                else:
                    raise ValueError('cov_file_format not recognized: must be "npy" or "npz"')

                covmat_path = f'{covariance_cfg["cov_folder"]}/zbins{zbins:02}'
                for ndim in (2, 4, 6):
                    if covariance_cfg[f'save_cov_{ndim}D']:

                        # save GO, GS or GO, GS and SS
                        which_cov_list = ['GO', 'GS']
                        Rl_str_list = ['', f'_Rl{which_probe_response_str}']
                        if covariance_cfg[f'save_cov_SS']:
                            which_cov_list.append('SS')
                            Rl_str_list.append(f'_Rl{which_probe_response_str}')

                        # set probes to save; the ndim == 6 case is different
                        probe_list = ['WL', 'GC', '3x2pt', 'WA']
                        ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
                        nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]
                        # in this case, 3x2pt is saved in 10D as a dictionary
                        if ndim == 6:
                            probe_list = ['WL', 'GC', 'WA']
                            ellmax_list = [ell_max_WL, ell_max_GC, ell_max_WL]
                            nbl_list = [nbl_WL, nbl_GC, nbl_WA]

                        # save all covmats in the optimistic case
                        if ell_max_WL == 5000:

                            for which_cov, Rl_str in zip(which_cov_list, Rl_str_list):
                                for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                                    covmat_filename = f'covmat_{which_cov}_{probe}_lmax{ell_max}_nbl{nbl}_' \
                                                      f'zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_' \
                                                      f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                                      f'ZS{zcut_source:02d}{Rl_str}_{ndim}D.{extension}'
                                    # save
                                    save_funct(f'{covmat_path}/{covmat_filename}',
                                               cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'])

                                # in this case, 3x2pt is saved in 10D as a dictionary
                                if ndim == 6:
                                    covmat_filename = f'{covmat_path}/covmat_{which_cov}_3x2pt_lmax{ell_max_XC}_' \
                                                      f'nbl{nbl_3x2pt}_zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_' \
                                                      f'ZL{zcut_lens:02d}_MS{magcut_source:03d}_' \
                                                      f'ZS{zcut_source:02d}{Rl_str}_10D.pickle'
                                    with open(covmat_filename, 'wb') as handle:
                                        pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)

                        # in the pessimistic case, save only WA
                        elif ell_max_WL == 1500:
                            for which_cov, Rl_str in zip(['GO', 'GS'], ['', f'_Rl{which_probe_response_str}']):
                                covmat_filename = f'covmat_{which_cov}_WA_lmax{ell_max_WL}_nbl{nbl_WA}_zbins{EP_or_ED}{zbins:02}_ML{magcut_lens:03d}_ZL{zcut_lens:02d}_MS{magcut_source:03d}_ZS{zcut_source:02d}{Rl_str}_{ndim}D.npy'
                                np.save(f'{covmat_path}/{covmat_filename}', cov_dict[f'cov_WA_{which_cov}_{ndim}D'])

                # save in .dat for Vincenzo, only in the optimistic case and in 2D
                if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
                    raise NotImplementedError('not implemented yet')
                    path_vinc_fmt = f'{job_path}/output/covmat/vincenzos_format'
                    for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
                        for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):
                            np.savetxt(f'{path_vinc_fmt}/{GOGS_folder}/{probe_vinc}/cm-{probe_vinc}-{nbl_WL}'
                                       f'-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.dat',
                                       cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.10e')

                # check for Stefano
                if covariance_cfg['save_cov_6D']:
                    warnings.warn('old checks below, you could probably discard...')
                    npairs = (zbins * (zbins + 1)) // 2
                    cov_WL_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GO_6D'], nbl_WL, npairs, ind[:npairs, :])
                    cov_GC_GO_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GO_6D'], nbl_GC, npairs, ind[:npairs, :])
                    cov_WL_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_WL_GS_6D'], nbl_WL, npairs, ind[:npairs, :])
                    cov_GC_GS_4D = mm.cov_6D_to_4D(cov_dict[f'cov_GC_GS_6D'], nbl_GC, npairs, ind[:npairs, :])
                    assert np.array_equal(cov_WL_GO_4D, cov_dict[f'cov_WL_GO_4D'])
                    assert np.array_equal(cov_GC_GO_4D, cov_dict[f'cov_GC_GO_4D'])
                    assert np.allclose(cov_WL_GS_4D, cov_dict[f'cov_WL_GS_4D'], rtol=1e-9, atol=0)
                    assert np.allclose(cov_GC_GS_4D, cov_dict[f'cov_GC_GS_4D'], rtol=1e-9, atol=0)

                # ! save FM
                if FM_cfg['save_FM']:
                    probe_list = ['WL', 'GC', '3x2pt', 'WA']
                    ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
                    nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]

                    for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                        for which_cov in ['GO', 'GS']:
                            np.savetxt(f'{FM_cfg["FM_output_folder"]}/'
                                       f'FM_{probe}_{which_cov}_lmax{ell_max}_nbl{nbl}_zbins{EP_or_ED}{zbins:02}{Rl_str}.txt',
                                       FM_dict[f'FM_{probe}_{which_cov}'])

                if FM_cfg['save_FM_as_dict']:
                    sio.savemat(job_path / f'output/FM/FM_dict.mat', FM_dict)

print('done')
