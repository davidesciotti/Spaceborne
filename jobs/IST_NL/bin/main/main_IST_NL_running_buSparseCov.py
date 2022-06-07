import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spar
import sys
import time
from pathlib import Path

# get project directory
project_path = Path.cwd().parent.parent.parent.parent
job_path = Path.cwd().parent.parent
home_path = Path.home()

# job-specific modules and configurations
sys.path.append(str(job_path / 'configs'))
sys.path.append(str(job_path / 'bin/utils'))

# job configuration
import config_IST_NL as config

# job utils
import utils_IST_NL as utils

# lower level modules
sys.path.append(str(project_path / 'lib'))
sys.path.append(str(project_path / 'bin/1_ell_values'))
sys.path.append(str(project_path / 'bin/2_cl_preprocessing'))
sys.path.append(str(project_path / 'bin/3_covmat'))
sys.path.append(str(project_path / 'bin/4_FM'))
sys.path.append(str(project_path / 'bin/5_plots/plot_FM'))

import Cl_preprocessing_running as Cl_utils
import covariance_running as covmat_utils

start_time = time.perf_counter()

params = {'lines.linewidth' : 3.5,
          'font.size' : 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': [10, 7]
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

# import the configuration dictionaries from common_config.py
general_config = config.general_config
covariance_config = config.covariance_config
FM_config = config.FM_config
plot_config = config.plot_config

# plot settings:
params = plot_config['params']
markersize = plot_config['markersize']
dpi = plot_config['dpi']
pic_format = plot_config['pic_format']


# consistency checks:
utils.consistency_checks(general_config, covariance_config)

# for the time being, I/O is manual and from the main
# load inputs (job-specific)
ind = np.genfromtxt(job_path / "input/indici_cloe_like.dat").astype(int) - 1
covariance_config['ind'] = ind

# convention = 0 # Lacasa & Grain 2019, incorrect (using IST WFs)
convention = 1 # Euclid, correct

bia = 0.0 # the one used by CLOE at the moment
# bia = 2.17

assert bia == 0.0, 'bia must be 0!'
assert convention == 1, 'convention must be 1 for Euclid!'

# normalization = 'IST'
# normalization = 'PySSC'

# from new SSC code (need to specify the 'convention' needed)
Sijkl_marco = np.load(job_path / f"input/CosmoCentral_outputs/Sijkl/Sijkl_WFmarco_nz10000_bia{bia}_conv{convention}_gen22.npy") 

# from old PySSC code (so no 'convention' parameter needed)
# Sijkl_marco = np.load(path / f"data/CosmoCentral_outputs/Sijkl/Sijkl_WFmarco_nz10000_bia{bia}_{normalization}normaliz_oldCode.npy") 

# old Sijkl matrices
# Sijkl_sylv = np.load(path.parent / "common_data/Sijkl/sylvain_cosmologyFixe_IAFixe_Sijkl_GCph_WL_XC_3may.npy") 
# Sijkl_dav = np.load(path.parent / "common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy") 


Sijkl = Sijkl_marco

# a couple more checks
assert np.all(Sijkl == Sijkl_marco) == True, 'Sijkl should be Sijkl_marco'
assert bia == 0., 'IST_NL uses bia = 0 (for the moment)'


###############################################################################
######################### FORECAST COMPUTATION ################################
###############################################################################

for NL_flag in range(5):

    # import Cl (I've done this for NL_flag = 3)
    if NL_flag == 0: # linear case, different file names
        C_LL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data/Cls_zNLA_ShearShear.dat')
        C_GL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data/Cls_zNLA_PosShear.dat')
        C_GG_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data/Cls_zNLA_PosPos.dat')
    else: 
        C_LL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data/Cls_zNLA_ShearShear_NL_flag_{NL_flag}.dat')
        C_GL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data/Cls_zNLA_PosShear_NL_flag_{NL_flag}.dat')
        C_GG_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data/Cls_zNLA_PosPos_NL_flag_{NL_flag}.dat')

    # remove ell column
    C_LL_2D = C_LL_2D[:,1:]
    C_GL_2D = C_GL_2D[:,1:]
    C_GG_2D = C_GG_2D[:,1:]

    # set ells and deltas
    ell_bins = np.linspace(np.log(10.), np.log(5000.), 21) 
    ells = (ell_bins[:-1]+ell_bins[1:])/2.
    ells = np.exp(ells)
    deltas = np.diff(np.exp(ell_bins))

    # store everything in dicts
    ell_dict = {
        'ell_WL': ells,
        'ell_GC': ells,
        'ell_WA': ells} # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
    delta_dict = {
        'delta_l_WL': deltas,
        'delta_l_GC': deltas,
        'delta_l_WA': deltas} # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
    cl_dict_2D = {
        'C_LL_2D': C_LL_2D,
        'C_LL_WLonly_2D': C_LL_2D, # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
        'C_XC_2D': C_GL_2D,
        'C_GG_2D': C_GG_2D,
        'C_WA_2D': C_LL_2D} # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything


    ###############################################################################


    # ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_config)
    # Cl_dict = Cl_utils.generate_Cls(general_config, ell_dict, cl_dict_2D)
    
    cl_dict_3D = Cl_utils.reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D)
    cov_dict = covmat_utils.compute_cov(general_config, covariance_config, ell_dict, delta_dict, cl_dict_3D, Sijkl)

    # save in .npy
    # smarter save: loop over the names (careful of the ordering, tuples are better than lists because immutable)
    probe_CLOE_list = ('PosPos', 'ShearShear', '3x2pt')
    probe_script_list = ('GC', 'WL', '3x2pt')
    GO_or_GS_CLOE_list = ('Gauss', 'GaussSSC')
    GO_or_GS_script_list = ('GO', 'GS')

    for probe_CLOE, probe_script in zip(probe_CLOE_list, probe_script_list):
        for GO_or_GS_CLOE, GO_or_GS_script in zip(GO_or_GS_CLOE_list, GO_or_GS_script_list):
            if NL_flag == 0:  # linear case, different file names (no '-NL_flag_{NL_flag} suffix')
                # np.save(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins.npy', cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_2D'])
                spar.save_npz(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins.npz', cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_2D_sparse'])
            else:
                # np.save(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins-NL_flag_{NL_flag}.npy', cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_2D'])
                spar.save_npz(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins-NL_flag_{NL_flag}.npz', cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_2D_sparse'])

# TODO add "sparse" to saved filename
# TODO NL_flag string instead of the if save


    # spar.save_npz('CovMat-3x2pt-Gauss-20bins-NL_flag_1-Sparse.npz', sparse_cov_3x2_gauss)
    # loaded_sparse_cov = spar.load_npz('CovMat-3x2pt-Gauss-20bins-NL_flag_1-Sparse.npz')
    # sys.getsizeof(loaded_sparse_cov)
    # 48
    # cov_3x2_gauss_loaded = loaded_sparse_cov.toarray()
    # sys.getsizeof(cov_3x2_gauss_loaded)

    # compute FM and pot, just to get an idea (the derivatives are still from Vincenzo!)
    # FM_dict = FM_utils.compute_FM(general_config, covariance_config, FM_config, ell_dict, cl_dict_3D, cov_dict)
    # plt.figure()
    # plot_utils.plot_FM(general_config, covariance_config, plot_config, FM_dict)

    # mm.show_keys(cov_dict)


import my_module as mm

for NL_flag in range(5):
    for probe_CLOE, probe_script in zip(probe_CLOE_list, probe_script_list):
        for GO_or_GS_CLOE, GO_or_GS_script in zip(GO_or_GS_CLOE_list, GO_or_GS_script_list):
            if NL_flag == 0:  # linear case, different file names (no '-NL_flag_{NL_flag} suffix')
                # np.save(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins.npy', cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_2D'])
                # spar.save_npz(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins.npy', cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_2D_sparse'])
                sparse_cov = spar.load_npz(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins.npz').toarray()
                npy_cov = np.load(job_path / f'output/covmat/npy_test_agains_sparse/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins.npy')
            else:
                sparse_cov = spar.load_npz(job_path / f'output/covmat/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins-NL_flag_{NL_flag}.npz').toarray()
                npy_cov = np.load(job_path / f'output/covmat/npy_test_agains_sparse/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins-NL_flag_{NL_flag}.npy')

            print(np.all(sparse_cov == npy_cov))
    #
### test sparse cov
path_old = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/output/covmat/npy_test_agains_sparse'
path_new = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/output/covmat'

loaded_sparse_cov = spar.load_npz('CovMat-3x2pt-Gauss-20bins-NL_flag_1-Sparse.npz')

cov_3x2_gauss_loaded = loaded_sparse_cov.toarray()


# some tests
# TODO put these tests in another module
# probe = '3x2pt'

# tests

# TODO finish testing
lik_path = '/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data'
cov_dict_new = dict(mm.get_kv_pairs_npy(job_path / 'output/covmat'))
cov_dict_old = dict(mm.get_kv_pairs_npy(job_path / 'output/covmat/before_23march_update'))
cov_dict_santiago = dict(mm.get_kv_pairs_npy(job_path / 'output/covmat/covs_for_santiago/covs_for_santiago'))
cov_dict_lik = dict(mm.get_kv_pairs_npy(f'{lik_path}'))

for NL_flag in range(1, 5):
    # for probe in ['ShearShear', 'PosPos', '3x2pt']:
    for probe in ['ShearShear', '3x2pt']:
        for GO_or_GS in ['Gauss', 'GaussSSC']:
            filename = f'CovMat-{probe}-{GO_or_GS}-20bins-NL_flag_{NL_flag}'
            filename_santi = f'CovMat-{probe}-{GO_or_GS}-20bins-NL_flag_{NL_flag}_2D'
            diff = mm.percent_diff_nan(cov_dict_new[filename], cov_dict_old[filename]).mean()
            print(NL_flag, probe, GO_or_GS,  np.all(cov_dict_santiago[filename_santi] == cov_dict_old[filename]), diff)

for NL_flag in range(1, 5):
    GO_or_GS = 'GaussSSC'
    probe = '3x2pt'
    filename = f'CovMat-{probe}-{GO_or_GS}-20bins-NL_flag_{NL_flag}'
    mm.matshow(cov_dict_new[filename][:210, :210], log=True, title=f'NL_flag_{NL_flag} {GO_or_GS} {probe} NEW')
    mm.matshow(cov_dict_lik[filename][:210, :210], log=True, title=f'NL_flag_{NL_flag} {GO_or_GS} {probe} OLD')



    # cov_WL_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-ShearShear-Gauss-20Bins.npy')
    # cov_GC_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-PosPos-Gauss-20Bins.npy')
    # cov_3x2pt_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-3x2pt-Gauss-20Bins.npy')

    # cov_d_4D = cov_dict[f'cov_{probe}_G_4D']
    # cov_d_2D = cov_dict[f'cov_{probe}_G_2D']
    # # cov_d_2DCLOE = cov_dict[f'cov_{probe}_G_2DCLOE']



    # cov_d = cov_d_2D
    # cov_s = cov_3x2pt_benchmark2

    # limit = 210
    # mm.matshow(cov_d[:limit,:limit], f'Davide {probe} 1st', log=True, abs_val=True)
    # mm.matshow(cov_s[:limit,:limit], f'Santiago {probe} 1st', log=True, abs_val=True)
    # mm.matshow(cov_d[-limit:,-limit:], f'Davide {probe} last', log=True, abs_val=True)
    # mm.matshow(cov_s[-limit:,-limit:], f'Santiago {probe} last', log=True, abs_val=True)
    # mm.matshow(cov_d, f'Davide {probe}', log=True, abs_val=True)
    # mm.matshow(cov_s, f'Santiago {probe}', log=True, abs_val=True)

    # # mm.matshow(np.abs(cov_d_2DCLOE), f'Davide {probe} 2DCLOE (wrong?)', log=True)



    # diff = mm.percent_diff(cov_s, cov_d)
    # # diff = np.where(np.abs(diff) > 5, diff, 1)
    # mm.matshow(diff[:limit,:limit], 'diff 1st block', log=False, abs_val=True)
    # mm.matshow(diff[-limit:,-limit:], 'diff last block', log=False, abs_val=True)
    # mm.matshow(diff, 'diff', log=False, abs_val=True)



