import bz2
import pickle
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import warnings

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
import ISTF_fid_params as ISTFfid

# project modules
sys.path.append(f'{project_path}/bin')
import ell_values as ell_utils
import cl_preprocessing as cl_utils
import compute_Sijkl as Sijkl_utils
import covariance as covmat_utils
import fisher_matrix as FM_utils

# job configuration and modules
sys.path.append(f'{project_path}/jobs')
import BNT_for_ISTL.config.config_BNT_for_ISTL as cfg

mpl.use('Qt5Agg')
mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks
# TODO super check that things work with different # of z bins

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg
Sijkl_cfg = cfg.Sijkl_cfg
FM_cfg = cfg.FM_cfg

# utils.consistency_checks(general_cfg, covariance_cfg)

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
bIA = ISTFfid.IA_free['beta_IA']

# which cases to save: GO, GS or GO, GS and SSC
cases_tosave = ['GO', ]
if covariance_cfg['save_cov_GS']:
    cases_tosave.append('GS')
if covariance_cfg['save_cov_SS']:
    cases_tosave.append('SS')

variable_specs = {
    'zbins': zbins,
    'EP_or_ED': EP_or_ED,
    'triu_tril': triu_tril,
    'row_col_major': row_col_major,
}

# some checks
assert EP_or_ED == 'EP' and zbins == 10, 'ISTF uses 10 equipopulated bins'
assert covariance_cfg['GL_or_LG'] == 'GL', 'Cij_14may uses GL, also for the probe responses'
assert nbl_GC == nbl_WL, 'for ISTF we are using the same number of ell bins for WL and GC'

zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

# import the ind files and store it into the covariance dictionary
ind_folder = covariance_cfg['ind_folder'].format(**variable_specs)
ind_filename = covariance_cfg['ind_filename'].format(**variable_specs)
ind = np.genfromtxt(f'{ind_folder}/{ind_filename}', dtype=int)
covariance_cfg['ind'] = ind

# ! import CLOE's ells, delta_ell and cls
data_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/BNT_for_ISTL/data'
cl_LL_3d = np.load(f'{data_path}/Cls_zNLA_ShearShear_new.npy')
cl_GL_3d = np.load(f'{data_path}/Cls_zNLA_PosShear_new.npy')
cl_GG_3d = np.load(f'{data_path}/Cls_zNLA_PosPos_new.npy')
cl_3x2pt_5D = cl_utils.build_3x2pt_datavector_5D(cl_LL_3d, cl_GL_3d, cl_GG_3d, nbl_GC, zbins, n_probes)

# the ell values are all the same!
ells = np.load(f'{data_path}/ells.npy')
ell_bin_edges = np.load(f'{data_path}/ell_bin_edges.npy')
delta_ell = np.diff(ell_bin_edges)

ell_dict = {
    'ell_WL': ells,
    'ell_GC': ells,
    'ell_WA': ells,
}

delta_dict = {
    'delta_l_WL': delta_ell,
    'delta_l_GC': delta_ell,
    'delta_l_WA': delta_ell,
}

cl_dict_3D = {
    'cl_LL_3D': cl_LL_3d,
    'cl_GL_3D': cl_GL_3d,
    'cl_GG_3D': cl_GG_3d,
    'cl_WA_3D': cl_LL_3d,  # ! not used
    'cl_3x2pt_5D': cl_3x2pt_5D,
}

rl_dict_3D = {}

# ! compute covariance matrix
if covariance_cfg['compute_covmat']:

    # ! compute or load Sijkl
    # if Sijkl exists, load it; otherwise, compute it and save it
    Sijkl_folder = Sijkl_cfg['Sijkl_folder']
    Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(nz=Sijkl_cfg['nz'])

    if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):

        print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
        sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')

    else:

        # ! load kernels
        # TODO this should not be done if Sijkl is loaded; I have a problem with nz, which is part of the file name...
        nz = Sijkl_cfg["nz"]
        wf_folder = Sijkl_cfg["wf_input_folder"].format(nz=nz)
        wil_filename = Sijkl_cfg["wil_filename"].format(normalization=Sijkl_cfg['wf_normalization'],
                                                        has_IA=str(Sijkl_cfg['has_IA']), nz=nz, bIA=bIA)
        wig_filename = Sijkl_cfg["wig_filename"].format(normalization=Sijkl_cfg['wf_normalization'], nz=nz)
        wil = np.genfromtxt(f'{wf_folder}/{wil_filename}')
        wig = np.genfromtxt(f'{wf_folder}/{wig_filename}')

        # preprocess (remove redshift column)
        z_arr, wil = Sijkl_utils.preprocess_wf(wil, zbins)
        z_arr_2, wig = Sijkl_utils.preprocess_wf(wig, zbins)
        assert np.array_equal(z_arr, z_arr_2), 'the redshift arrays are different for the GC and WL kernels'
        assert nz == z_arr.shape[0], 'nz is not the same as the number of redshift points in the kernels'

        # transpose and stack, ordering is important here!
        transp_stacked_wf = np.vstack((wil.T, wig.T))
        sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                          Sijkl_cfg['WF_normalization'])
        np.save(f'{Sijkl_folder}/{Sijkl_filename}', sijkl)

    # ! compute covariance matrix
    cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                        ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, sijkl)

# check
cov_3x2pt_GO_2DCLOE = cov_dict['cov_3x2pt_GO_2DCLOE']
cov_3x2pt_GO_2DCLOE_test = np.load('/Users/davide/Documents/Lavoro/Programmi/likelihood-implementation/data/'
                                   'ExternalBenchmark/Photometric/data/CovMat-3x2pt-Gauss-20Bins-probe_ell_zpair.npy')

diff = mm.percent_diff(cov_3x2pt_GO_2DCLOE, cov_3x2pt_GO_2DCLOE_test)
mm.matshow(np.abs(diff), title='diff', log=False, abs_val=False, threshold=10)

mm.compare_arrays(cov_3x2pt_GO_2DCLOE, cov_3x2pt_GO_2DCLOE_test,
                  name_A='cov_3x2pt_GO_2DCLOE', name_B='cov_3x2pt_GO_2DCLOE_test', plot_diff=True, log_diff=True,
                  plot_array=True, log_array=True)

# ! ok but not perfect there is still a (small) number of outliers; maybe check:
# ! also, why do I have to recompute the covariance?
# - cls (undo the modifications to simulate_data to have the triu elements, then unpack...)
# - deltas
# - ell values?

assert 1 > 2, 'stop here'

# ! compute Fisher Matrix
if FM_cfg['compute_FM']:

    paramnames_3x2pt = FM_cfg['paramnames_3x2pt']
    nparams_total = len(paramnames_3x2pt)
    GL_or_LG = covariance_cfg['GL_or_LG']
    # these define the derivatives filenames
    der_prefix = FM_cfg['derivatives_prefix']
    derivatives_suffix = FM_cfg['derivatives_suffix']

    derivatives_folder = FM_cfg['derivatives_folder'].format(**variable_specs)
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
                dC_dict_LLfor3x2pt_3D[key] = mm.cl_2D_to_3D_symmetric(dC_dict_LL_2D[key], nbl_GC, zpairs_auto, zbins)

            elif key.startswith(der_prefix.format(probe='GG')):
                dC_dict_GG_2D[key] = mm.cl_interpolator(dC_dict_2D[key], zpairs_auto, ell_GC, nbl_GC)
                dC_dict_GG_3D[key] = mm.cl_2D_to_3D_symmetric(dC_dict_GG_2D[key], nbl_GC, zpairs_auto, zbins)

            elif key.startswith(der_prefix.format(probe=GL_or_LG)):
                dC_dict_GL_2D[key] = mm.cl_interpolator(dC_dict_2D[key], zpairs_cross, ell_GC, nbl_GC)
                dC_dict_GL_3D[key] = mm.cl_2D_to_3D_asymmetric(dC_dict_GL_2D[key], nbl_GC, zbins, 'row_major')

    # turn dictionary keys into entries of 4-th array axis
    # TODO the obs_name must be defined in the config file
    dC_LL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LL_3D, paramnames_3x2pt, nbl, zbins,
                                            der_prefix.format(probe='LL'))
    dC_WA_4D = FM_utils.dC_dict_to_4D_array(dC_dict_WA_3D, paramnames_3x2pt, nbl_WA, zbins,
                                            der_prefix.format(probe='LL'))
    dC_LLfor3x2pt_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LLfor3x2pt_3D, paramnames_3x2pt, nbl, zbins,
                                                    der_prefix.format(probe='LL'))
    dC_GG_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GG_3D, paramnames_3x2pt, nbl, zbins,
                                            der_prefix.format(probe='GG'))
    dC_GL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GL_3D, paramnames_3x2pt, nbl, zbins,
                                            der_prefix.format(probe=GL_or_LG))

    # build 5D array of derivatives for the 3x2pt
    dC_3x2pt_5D = np.zeros((nbl, n_probes, n_probes, zbins, zbins, nparams_total))
    dC_3x2pt_5D[:, 0, 0, :, :, :] = dC_LLfor3x2pt_4D
    dC_3x2pt_5D[:, 0, 1, :, :, :] = dC_GL_4D.transpose(0, 2, 1, 3)
    dC_3x2pt_5D[:, 1, 0, :, :, :] = dC_GL_4D
    dC_3x2pt_5D[:, 1, 1, :, :, :] = dC_GG_4D

    # store the arrays of derivatives in a dictionary to pass to the Fisher Matrix function
    deriv_dict = {'dC_LL_4D': dC_LL_4D,
                  'dC_GG_4D': dC_GG_4D,
                  'dC_WA_4D': dC_WA_4D,
                  'dC_3x2pt_5D': dC_3x2pt_5D}

    # finally, define the fiducial values to write them in the FM header file
    paramnames_cosmo = ["Om", "Ob", "wz", "wa", "h", "ns", "s8"]
    paramnames_IA = ["Aia", "eIA", "bIA"]
    paramnames_galbias = [f'bL{zbin_idx:02d}' for zbin_idx in range(1, zbins + 1)]

    fid_cosmo = [ISTFfid.primary['Om_m0'], ISTFfid.primary['Om_b0'], ISTFfid.primary['w_0'], ISTFfid.primary['w_a'],
                 ISTFfid.primary['h_0'], ISTFfid.primary['n_s'], ISTFfid.primary['sigma_8']]
    fid_IA = [ISTFfid.IA_free['A_IA'], ISTFfid.IA_free['eta_IA'], ISTFfid.IA_free['beta_IA']]
    fid_gal_bias = [ISTFfid.photoz_galaxy_bias[f'b{zbin:02d}_photo'] for zbin in range(1, zbins + 1)]
    fid_3x2pt = fid_cosmo + fid_IA + fid_gal_bias

    FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict)
    FM_dict['parameters_names'] = paramnames_3x2pt
    FM_dict['fiducial_values'] = fid_3x2pt

# ! save:
# this is just to set the correct probe names
probe_dav_dict = {
    'WL': 'LL_3D',
    'GC': 'GG_3D',
    'WA': 'WA_3D',
    '3x2pt': '3x2pt_5D'}

# just a dict for the output file names
clrl_dict = {
    'cl_inputname': 'dv',
    'rl_inputname': 'rf',
    'cl_dict_3D': cl_dict_3D,
    'rl_dict_3D': rl_dict_3D,
    'cl_dict_key': 'C',
    'rl_dict_key': 'R',
}
for cl_or_rl in ['cl', 'rl']:
    folder = general_cfg[f'{cl_or_rl}_folder']
    if general_cfg[f'save_{cl_or_rl}s_3d']:

        for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
            # save cl and/or response
            np.save(f'{folder}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}/'
                    f'{clrl_dict[f"{cl_or_rl}_inputname"]}-{probe_vinc}-{nbl_WL}-{general_cfg["specs"]}-{EP_or_ED}{zbins:02}.npy',
                    clrl_dict[f"{cl_or_rl}_dict_3D"][
                        f'{clrl_dict[f"{cl_or_rl}_dict_key"]}_{probe_dav_dict[probe_dav]}'])

            # save ells and deltas
            if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                np.savetxt(
                    f'{folder}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}/ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                    10 ** ell_dict[f'ell_{probe_dav}'])
                np.savetxt(
                    f'{folder}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}/delta_ell_{probe_dav}_ellmaxWL{ell_max_WL}.txt',
                    delta_dict[f'delta_l_{probe_dav}'])

# TODO save methods should probably be moved to a function
"""
cov_folder = covariance_cfg["cov_folder"].format(SSC_code=covariance_cfg['SSC_code'])
for ndim in (2, 4, 6):
    if covariance_cfg[f'save_cov_{ndim}D']:

        # save GO, GS or GO, GS and SSC
        which_cov_list = ['GO', 'GS']
        if covariance_cfg[f'save_cov_SSC']:
            which_cov_list.append('SSC')

        # set probes to save; the ndim == 6 case is different
        probe_list = ['WL', 'GC', '3x2pt', 'WA']
        ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
        nbl_list = [nbl_WL, nbl_GC, nbl_GC, nbl_WA]
        # in this case, 3x2pt is saved in 10D as a dictionary
        if ndim == 6:
            probe_list = ['WL', 'GC', 'WA']
            ellmax_list = [ell_max_WL, ell_max_GC, ell_max_WL]
            nbl_list = [nbl_WL, nbl_GC, nbl_WA]

        # save all covmats in the optimistic case
        if ell_max_WL == 5000:

            for which_cov in which_cov_list:
                for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                    np.save(f'{cov_folder}/'
                            f'covmat_{which_cov}_{probe}_lmax{ell_max}_nbl{nbl}_zbins{EP_or_ED}{zbins:02}_{ndim}D.npy',
                            cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'])

                # in this case, 3x2pt is saved in 10D as a dictionary
                if ndim == 6:
                    filename = f'{cov_folder}/covmat_{which_cov}_3x2pt_lmax{ell_max_XC}_nbl{nbl_GC}_zbins{EP_or_ED}{zbins:02}_10D.pickle'
                    with open(filename, 'wb') as handle:
                        pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)

        # in the pessimistic case, save only WA
        elif ell_max_WL == 1500:
            for which_cov in which_cov_list:
                np.save(
                    f'{cov_folder}/covmat_{which_cov}_WA_lmax{ell_max_WL}_nbl{nbl_WA}_zbins{EP_or_ED}{zbins:02}_{ndim}D.npy',
                    cov_dict[f'cov_WA_{which_cov}_{ndim}D'])
"""

cov_folder = covariance_cfg["cov_folder"].format(SSC_code=covariance_cfg['SSC_code'])
covmat_utils.save_cov(covariance_cfg, general_cfg, cov_dict, cov_folder)

if FM_cfg['compute_FM'] and FM_cfg['save_FM']:
    # saves as txt file
    probe_list = ['WL', 'GC', '3x2pt', 'WA']
    ellmax_list = [ell_max_WL, ell_max_GC, ell_max_XC, ell_max_WL]
    nbl_list = [nbl_WL, nbl_GC, nbl_GC, nbl_WA]
    which_cov_list = ['GO', 'GS']
    header = f"parameters: {paramnames_3x2pt} \nfiducials: {fid_3x2pt}"
    FM_folder = FM_cfg["FM_folder"].format(SSC_code=covariance_cfg['SSC_code'])

    for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
        for which_cov in which_cov_list:
            FM_filename = FM_cfg["FM_filename"].format(probe=probe, which_cov=which_cov, ell_max=ell_max, nbl=nbl,
                                                       **variable_specs)
            np.savetxt(f'{FM_folder}/{FM_filename}', FM_dict[f'FM_{probe}_{which_cov}'], header=header)

if FM_cfg['compute_FM'] and FM_cfg['save_FM_as_dict']:
    mm.save_pickle(f'{FM_folder}/FM_dict_{EP_or_ED}{zbins:02}.pickle', FM_dict)

ut.test_cov_FM(covariance_cfg['SSC_code'], f'{general_cfg["cfg_name"]}/covmat')
ut.test_cov_FM(covariance_cfg['SSC_code'], f'{general_cfg["cfg_name"]}/FM')

print('done')
