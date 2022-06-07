import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

# get project directory
path = Path.cwd().parent.parent

# import configuration and functions modules
sys.path.append(str(path.parent / 'common_data'))
import my_config as my_config
sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

sys.path.append(str(path / 'source/1_ell_values'))
import ell_values_GUT as ell_utils

sys.path.append(str(path / 'source/2_matrici_base'))
import matrici_base_v53_SSCprove_GUT_noSEYF as Cl_utils

sys.path.append(str(path / 'source/3_covmat'))
import cov_v95_SSCprove_GUT_more_general as covmat_utils

sys.path.append(str(path / 'source/4_FM'))
import FM_v98_SSCprove_GUT_forISTNL as FM_utils

sys.path.append(str(path / 'source/5_plots/plot_FM'))
import plots_FM_v106_SSCprove_GUT_noCLOE as plot_utils


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

survey_area = 15000 # deg^2
deg2_in_sphere = 41252.96 # deg^2 in a spere
fsky_IST = survey_area/deg2_in_sphere
fsky_syvain = 0.375 

general_settings = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 5000,
    'zbins': 10,
    'nProbes': 2,
    'nbl': 20,
    'which_forecast': 'IST',
    'Cij_folder': 'Cl_CLOE'
    }

cov_settings = {
    'ind_ordering': 'CLOE',
    'GL_or_LG': 'LG',
    'save_SSC_only_covmats': False,
    'compute_covariance_in_blocks': False,
    'fsky': fsky_IST,
    'Rl': 4
    }

FM_settings = {
    'nParams': 20,
    'save_npy': False
    }

plot_settings = {
    'case': 'opt',
    'probe': 'WL',
    'SSC_flag': 'GpSSC',
    'covmat_dav_flag': 'no',
    'which_plot': 'constraints_only'
    }



# convention = 0 # Lacasa & Grain 2019, incorrect (using IST WFs)
convention = 1 # Euclid, correct

bia = 0.0
# bia = 2.17

NL_flag = 4


assert bia == 0.0, 'bia must be 0!'
assert convention == 1, 'convention must be 1 for Euclid!'

# normalization = 'IST'
# normalization = 'PySSC'

# from new SSC code ('need to specify the convention needed')
Sijkl_marco = np.load(path / f"data/CosmoCentral_outputs/Sijkl/Sijkl_WFmarco_nz10000_bia{bia}_conv{convention}_gen22.npy") 

# from old PySSC code (so no 'convention' parameter needed)
# Sijkl_marco = np.load(path / f"data/CosmoCentral_outputs/Sijkl/Sijkl_WFmarco_nz10000_bia{bia}_{normalization}normaliz_oldCode.npy") 

# Sijkl_sylv = np.load(path.parent / "common_data/Sijkl/sylvain_cosmologyFixe_IAFixe_Sijkl_GCph_WL_XC_3may.npy") 
# Sijkl_dav = np.load(path.parent / "common_data/Sijkl/Sijkl_WFdavide_nz10000_IA_3may.npy") 


Sijkl = Sijkl_marco

###############################################################################
######################### FORECAST COMPUTATION ################################
###############################################################################


# import Cl (from common_data folder!)
# C_LL_2D = np.genfromtxt(path.parent / f'common_data/CLOE/Cl_from_simulate_data_dotpy/Cls_zNLA_ShearShear_NL_flag_{NL_flag}.dat')
# C_GL_2D = np.genfromtxt(path.parent / f'common_data/CLOE/Cl_from_simulate_data_dotpy/Cls_zNLA_PosShear_NL_flag_{NL_flag}.dat')
# C_GG_2D = np.genfromtxt(path.parent / f'common_data/CLOE/Cl_from_simulate_data_dotpy/Cls_zNLA_PosPos_NL_flag_{NL_flag}.dat')

# maybe better to do (I've done this for NL_flag = 3)
C_LL_2D = np.genfromtxt(path / f'data/Cl_CLOE/Cls_zNLA_ShearShear_NL_flag_{NL_flag}.dat')
C_GL_2D = np.genfromtxt(path / f'data/Cl_CLOE/Cls_zNLA_PosShear_NL_flag_{NL_flag}.dat')
C_GG_2D = np.genfromtxt(path / f'data/Cl_CLOE/Cls_zNLA_PosPos_NL_flag_{NL_flag}.dat')


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
    'ell_GC': ells}
delta_dict = {
    'delta_l_WL': deltas,
    'delta_l_GC': deltas}
Cl_dict_2D = {
    'C_LL_2D': C_LL_2D,
    'C_GL_2D': C_GL_2D,
    'C_GG_2D': C_GG_2D}


###############################################################################


# ell_dict, delta_dict = ell_utils.generate_ell_and_deltas(general_settings)
Cl_dict = Cl_utils.generate_Cls(general_settings, ell_dict, Cl_dict_2D)
cov_dict = covmat_utils.compute_cov(general_settings, cov_settings, ell_dict, delta_dict, Cl_dict, Sijkl)


# save in npy
# GO
np.save(path / f'output/covmat/CovMat-PosPos-Gauss-20bins-NL_flag_{NL_flag}.npy', cov_dict['cov_GC_G_2D'])
np.save(path / f'output/covmat/CovMat-ShearShear-Gauss-20bins-NL_flag_{NL_flag}.npy', cov_dict['cov_WL_G_2D'])
np.save(path / f'output/covmat/CovMat-3x2pt-Gauss-20bins-NL_flag_{NL_flag}.npy', cov_dict['cov_3x2pt_G_2D'])
# GS
np.save(path / f'output/covmat/CovMat-PosPos-GaussSSC-20bins-NL_flag_{NL_flag}.npy', cov_dict['cov_GC_GpSSC_2D'])
np.save(path / f'output/covmat/CovMat-ShearShear-GaussSSC-20bins-NL_flag_{NL_flag}.npy', cov_dict['cov_WL_GpSSC_2D'])
np.save(path / f'output/covmat/CovMat-3x2pt-GaussSSC-20bins-NL_flag_{NL_flag}.npy', cov_dict['cov_3x2pt_GpSSC_2D'])

FM_dict = FM_utils.compute_FM(general_settings, cov_settings, FM_settings, ell_dict, Cl_dict, cov_dict)
# plt.figure()
plot_utils.plot_FM(general_settings, cov_settings, plot_settings, FM_dict)
# XXX TODO: group parameters in dictionaries!


# mm.show_keys(cov_dict)


# some tests
# probe = '3x2pt'

# cov_WL_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-ShearShear-Gauss-20Bins.npy')
# cov_GC_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-PosPos-Gauss-20Bins.npy')
# cov_3x2pt_benchmark = np.load('/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data/CovMat-3x2pt-Gauss-20Bins.npy')

# cov_d_4D = cov_dict[f'cov_{probe}_G_4D']
# cov_d_2D = cov_dict[f'cov_{probe}_G_2D']
# # cov_d_2DCLOE = cov_dict[f'cov_{probe}_G_2DCLOE']



# cov_d = cov_d_2D
# cov_s = cov_3x2pt_benchmark

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



