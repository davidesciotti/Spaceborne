import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spar

matplotlib.use('Qt5Agg')

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

import covariance_running as covmat_utils

start_time = time.perf_counter()

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
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

# convention = 0 # Lacasa & Grain 2019, incorrect (using IST WFs)
convention = 1  # Euclid, correct

bia = 0.0  # the one used by CLOE at the moment
# bia = 2.17

assert bia == 0.0, 'bia must be 0!'
assert convention == 1, 'convention must be 1 for Euclid!'

# normalization = 'IST'
# normalization = 'PySSC'
# from new SSC code (need to specify the 'convention' needed)
Sijkl_marco = np.load(
    job_path / f"input/CosmoCentral_outputs/Sijkl/Sijkl_WFmarco_nz10000_bia{bia}_conv{convention}_gen22.npy")

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

for covariance_config['block_index'] in ['C-style', 'F-style']:
    for which_ind, which_ind_str in zip(['indici_cloe_like', 'indici_vincenzo_like', 'indici_triu_like'], ('indCLOE', 'indVincenzo', 'indTriu')):

        # for the time being, I/O is manual and from the main
        # load inputs (job-specific)
        ind = np.genfromtxt(project_path / f"config/common_data/ind/{which_ind}.dat").astype(int) - 1
        covariance_config['ind'] = ind

        # this is the string indicating the flattening, or "block", convention
        # TODO raise exception here?
        which_flattening = None  # initialize to a value
        if covariance_config['block_index'] in ['ell', 'vincenzo', 'C-style']:
            which_flattening = 'Cstyle'
        elif covariance_config['block_index'] in ['ij', 'sylvain', 'F-style']:
            which_flattening = 'Fstyle'

        # import Cl
        # C_LL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data'
        #                                     f'/Cls_zNLA_ShearShear{NL_flag_string_cl}.dat')
        # C_GL_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data'
        #                                     f'/Cls_zNLA_PosShear{NL_flag_string_cl}.dat')
        # C_GG_2D = np.genfromtxt(home_path / f'likelihood-implementation/data/ExternalBenchmark/Photometric/data'
        #                                     f'/Cls_zNLA_PosPos{NL_flag_string_cl}.dat')
        #
        # # remove ell column
        # C_LL_2D = C_LL_2D[:, 1:]
        # C_GL_2D = C_GL_2D[:, 1:]
        # C_GG_2D = C_GG_2D[:, 1:]

        # set ells and deltas
        ell_bins = np.linspace(np.log(10.), np.log(5000.), 21)
        ells = (ell_bins[:-1] + ell_bins[1:]) / 2.
        ells = np.exp(ells)
        deltas = np.diff(np.exp(ell_bins))

        # load marco's 3D cls
        path_CC_outp = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/input/CosmoCentral_outputs/Cl'
        C_LL_3D = np.load(f'{path_CC_outp}/nbl20/C_LL_3D_marco_bia0.0_nbl20.npy')
        C_GG_3D = np.load(f'{path_CC_outp}/nbl20/C_GG_3D_marco_bia0.0_nbl20.npy')
        C_GL_3D = np.load(f'{path_CC_outp}/nbl20/C_GL_3D_marco_bia0.0_nbl20.npy')

        # build 3x2pt datavector
        nbl = 20
        zbins = 10
        D_3x2pt = np.zeros((nbl, 2, 2, zbins, zbins))
        D_3x2pt[:, 0, 0, :, :] = C_LL_3D
        D_3x2pt[:, 1, 1, :, :] = C_GG_3D
        D_3x2pt[:, 0, 1, :, :] = np.transpose(C_GL_3D, (0, 2, 1))
        D_3x2pt[:, 1, 0, :, :] = C_GL_3D

        # store them into the 3D dict
        cl_dict_3D = {}
        cl_dict_3D['C_LL_WLonly_3D'] = C_LL_3D
        cl_dict_3D['C_GG_3D'] = C_GG_3D
        cl_dict_3D['C_WA_3D'] = C_LL_3D
        cl_dict_3D['D_3x2pt'] = D_3x2pt

        # store everything in dicts
        ell_dict = {
            'ell_WL': ells,
            'ell_GC': ells,
            'ell_WA': ells}  # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
        delta_dict = {
            'delta_l_WL': deltas,
            'delta_l_GC': deltas,
            'delta_l_WA': deltas}  # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
        # cl_dict_2D = {
        #     'C_LL_2D': C_LL_2D,
        #     'C_LL_WLonly_2D': C_LL_2D,
        #     # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything
        #     'C_XC_2D': C_GL_2D,
        #     'C_GG_2D': C_GG_2D,
        #     'C_WA_2D': C_LL_2D}  # ! XXX this has to be fixed, I'm assigning some values to Wadd just to be able to run everything

        ###############################################################################

        # ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_config)
        # Cl_dict = Cl_utils.generate_Cls(general_config, ell_dict, cl_dict_2D)

        # cl_dict_3D = Cl_utils.reshape_cls_2D_to_3D(general_config, ell_dict, cl_dict_2D)
        # assert 1>2
        cov_dict = covmat_utils.compute_cov(general_config, covariance_config, ell_dict, delta_dict, cl_dict_3D, Sijkl)

        # save
        # smarter save: loop over the names (careful of the ordering, tuples are better than lists because immutable)
        probe_CLOE_list = ('PosPos', 'ShearShear', '3x2pt')
        probe_script_list = ('GC', 'WL', '3x2pt')
        # ! settle this once we solve the ordering issue!! If we go for fstyle there will no longer be need for 2DCLOE
        # ndim_list = ('2D', '2D', '2DCLOE')  # in the 3x2pt case the name is '2DCLOE', not '2D'
        # no 2DCLOE
        ndim_list = ('2D', '2D', '2D')  # in the 3x2pt case the name is '2DCLOE', not '2D'

        GO_or_GS_CLOE_list = ('Gauss', 'GaussSSC')
        GO_or_GS_script_list = ('GO', 'GS')

        # probes and dimensions
        # for probe_CLOE, probe_script, ndim in zip(probe_CLOE_list, probe_script_list, ndim_list):
        #     for GO_or_GS_CLOE, GO_or_GS_script in zip(GO_or_GS_CLOE_list, GO_or_GS_script_list):
                # save sparse (.npz)
                # spar.save_npz(
                #     job_path / f'output/covmat/ordering_marco/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins-{ndim}-'
                #                f'{which_flattening}-Sparse.npz',
                #     spar.csr_matrix(cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_{ndim}']))
                #
                # # save normal (.npy)
                # np.save(
                #     job_path / f'output/covmat/ordering_marco/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins-{ndim}-'
                #                f'{which_flattening}.npy',
                #     cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_{ndim}'])
                #
                # # save 4D in npy (sparse only works for arrays with dimension <= 2)
                # np.save(
                #     job_path / f'output/covmat/ordering_marco/CovMat-{probe_CLOE}-{GO_or_GS_CLOE}-20bins_4D.npy',
                #     cov_dict[f'cov_{probe_script}_{GO_or_GS_script}_4D'])

        # more readable and more error prone?, probably...
        np.save(job_path / f'output/covmat/ordering_marco/CovMat-ShearShear-Gauss-20bins_2D{which_flattening}_{which_ind_str}.npy', cov_dict[f'cov_WL_GO_2D'])
        np.save(job_path / f'output/covmat/ordering_marco/CovMat-PosPos-Gauss-20bins_2D{which_flattening}_{which_ind_str}.npy', cov_dict[f'cov_GC_GO_2D'])
        np.save(job_path / f'output/covmat/ordering_marco/CovMat-3x2pt-Gauss-20bins_2D{which_flattening}_{which_ind_str}.npy', cov_dict[f'cov_3x2pt_GO_2D'])

        np.save(job_path / f'output/covmat/ordering_marco/CovMat-ShearShear-GaussSSC-20bins_2D{which_flattening}_{which_ind_str}.npy', cov_dict[f'cov_WL_GS_2D'])
        np.save(job_path / f'output/covmat/ordering_marco/CovMat-PosPos-GaussSSC-20bins_2D{which_flattening}_{which_ind_str}.npy', cov_dict[f'cov_GC_GS_2D'])
        np.save(job_path / f'output/covmat/ordering_marco/CovMat-3x2pt-GaussSSC-20bins_2D{which_flattening}_{which_ind_str}.npy', cov_dict[f'cov_3x2pt_GS_2D'])

        # np.save(job_path / f'output/covmat/ordering_marco/CovMat-ShearShear-SSC-20bins_4D.npy', cov_dict[f'cov_WL_SS_4D'])
        # np.save(job_path / f'output/covmat/ordering_marco/CovMat-PosPos-SSC-20bins_4D.npy', cov_dict[f'cov_GC_SS_4D'])
        # np.save(job_path / f'output/covmat/ordering_marco/CovMat-3x2pt-SSC-20bins_4D.npy', cov_dict[f'cov_3x2pt_SS_4D'])

        # compute FM and pot, just to get an idea (the derivatives are still from Vincenzo!)
        # FM_dict = FM_utils.compute_FM(general_config, covariance_config, FM_config, ell_dict, cov_dict)
        # plt.figure()
        # plot_utils.plot_FM(general_config, covariance_config, plot_config, FM_dict)

        # some tests
        # TODO put these tests in another module
        # probe = '3x2pt'

        # tests

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

stop_time = time.perf_counter() - start_time
print(f'done in {stop_time:.2f} s')
