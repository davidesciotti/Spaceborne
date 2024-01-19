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


ROOT = '/home/davide/Documenti/Lavoro/Programmi'
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


mpl.use('Qt5Agg')
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
                dC_dict_LLfor3x2pt_3D[key] = mm.cl_2D_to_3D_symmetric(dC_dict_LL_2D[key], nbl_GC, zpairs_auto,
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
fm_folder_gssc = FM_cfg["fm_folder"].format(SSC_code=ssc_code)
fm_folder_g = FM_cfg["fm_folder"].format(SSC_code=ssc_code).replace(ssc_code, 'Gauss')

for probe in ('WL', 'GC', '3x2pt'):
    lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_3x2pt']
    filename_fm_g = f'{fm_folder_g}/FM_{probe}_G_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'
    filename_fm_from_ssc_code = f'{fm_folder_gssc}/FM_{probe}_GSSC_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'

    np.savetxt(f'{filename_fm_g}', FM_dict[f'FM_{probe}_G'])
    np.savetxt(f'{filename_fm_from_ssc_code}', FM_dict[f'FM_{probe}_GSSC'])

    # probe_ssc_code = covariance_cfg[f'{covariance_cfg["SSC_code"]}_cfg']['probe']
    # probe_ssc_code = 'WL' if probe_ssc_code == 'LL' else probe_ssc_code
    # probe_ssc_code = 'GC' if probe_ssc_code == 'GG' else probe_ssc_code

if general_cfg['test_against_benchmarks']:
    mm.test_folder_content(fm_folder_g, fm_folder_g + '/benchmarks', FM_cfg['FM_file_format'])
    mm.test_folder_content(fm_folder_gssc, fm_folder_gssc + '/benchmarks', FM_cfg['FM_file_format'])

################################################ ! plot ############################################################

# plot settings
nparams_toplot = 7
include_fom = False
divide_fom_by_10 = False

for ssc_code_here in ['PyCCL', 'PySSC', 'exactSSC']:
    for probe in ['WL', 'GC', '3x2pt']:
        fm_folder = FM_cfg["fm_folder"].format(SSC_code=ssc_code_here)
        lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_XC']
        FM_dict[f'FM_{ssc_code_here}_{probe}_GSSC'] = (
            np.genfromtxt(f'{fm_folder}/FM_{probe}_GSSC_lmax{lmax}_nbl{nbl}_zbinsEP{zbins}.txt'))


fom_dict = {}
uncert_dict = {}
masked_FM_dict = {}
for key in list(FM_dict.keys()):
    if key not in ['param_names_dict', 'fiducial_values_dict']:
        masked_FM_dict[key], param_names_list, fiducials_list = mm.mask_FM(FM_dict[key], param_names_dict,
                                                                           fiducials_dict,
                                                                           params_tofix_dict={})

        nparams = len(param_names_list)

        assert nparams == len(fiducials_list), f'number of parameters in the Fisher Matrix ({nparams}) '

        uncert_dict[key] = mm.uncertainties_FM(masked_FM_dict[key], nparams=masked_FM_dict[key].shape[0],
                                               fiducials=fiducials_list,
                                               which_uncertainty='marginal', normalize=True)[:nparams_toplot]
        fom_dict[key] = mm.compute_FoM(masked_FM_dict[key], w0wa_idxs=(2, 3))


for probe in ['WL', 'GC', '3x2pt']:
    nparams_toplot = 7
    to_compare_A = f'FM_PySSC_{probe}_GSSC'
    to_compare_B = f'FM_PyCCL_{probe}_GSSC'
    to_compare_C = f'FM_exactSSC_{probe}_GSSC'
    cases_to_plot = (f'FM_{probe}_G', to_compare_A, to_compare_B, to_compare_C)

    uncert_dict['perc_diff'] = mm.percent_diff(uncert_dict[to_compare_A], uncert_dict[to_compare_B])
    fom_dict['perc_diff'] = np.abs(mm.percent_diff(fom_dict[to_compare_A], fom_dict[to_compare_B]))

    # just a check, to be performed only if I am actually using PyCCL as well
    if 'FM_PySSC_G' in uncert_dict.keys() and 'FM_PyCCL_G' in uncert_dict.keys():
        assert np.array_equal(uncert_dict['FM_PySSC_G'], uncert_dict['FM_PyCCL_G']), \
            'the GO uncertainties must be the same, I am only changing the SSC code!'

    df = pd.DataFrame(uncert_dict)  # you should switch to using this...

    # # transform dict. into an array and add the fom
    uncert_array, fom_array = [], []
    for case in cases_to_plot:
        uncert_array.append(uncert_dict[case])
        fom_array.append(fom_dict[case])
    uncert_array = np.asarray(uncert_array)
    fom_array = np.asarray(fom_array)

    if divide_fom_by_10:
        fom_array /= 10
    uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

    # label and title stuff
    fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
    param_names_label = param_names_list[:nparams_toplot] + [fom_label] if include_fom else param_names_list[
        :nparams_toplot]
    lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_XC']
    use_hod_for_gc = 'use_HOD' + str(covariance_cfg["PyCCL_cfg"]["use_HOD_for_GCph"]) if covariance_cfg[
        "SSC_code"] == 'PyCCL' else ''
    title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i, %s' % (probe, lmax, EP_or_ED, zbins, use_hod_for_gc)
    # bar plot
    if include_fom:
        nparams_toplot = 8
    plot_utils.bar_plot(uncert_array[:, :nparams_toplot], title, cases_to_plot, nparams=nparams_toplot,
                        param_names_label=param_names_label, bar_width=0.12)
    # plt.yscale('log')


# silent check against IST:F (which does not exist for GC alone):
for which_probe in ['WL', '3x2pt']:
    np.set_printoptions(precision=2)
    print('\nprobe:', which_probe)
    uncert_dict['ISTF'] = ISTF_fid.forecasts[f'{which_probe}_opt_w0waCDM_flat']
    try:
        rtol = 10e-2
        assert np.allclose(uncert_dict[f'FM_{which_probe}_G'][:nparams_toplot], uncert_dict['ISTF'], atol=0,
                           rtol=rtol)
        print(f'IST:F and GO are consistent for probe {which_probe} within {rtol * 100}% ✅')
    except AssertionError:
        print(f'IST:F and GO are not consistent for probe {which_probe} within {rtol * 100}% ❌')
        print('(remember that you are checking against the optimistic case')
        print('ISTF GO:\t', uncert_dict['ISTF'])
        print('Spaceborne GO:\t', uncert_dict[f'FM_{which_probe}_G'][:nparams_toplot])
        print('percent_discrepancies (*not wrt mean!*):\n',
              mm.percent_diff(uncert_dict[f'FM_{which_probe}_G'][:nparams_toplot],
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


# ! save cls and responses: THIS MUST BE MOVED TO A DIFFERENT FUNCTION!
"""
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
"""
