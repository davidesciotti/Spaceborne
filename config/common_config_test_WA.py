from pathlib import Path
# get project directory
path = Path.cwd().parent.parent
import numpy as np

# official Euclid survey area
survey_area = 15000 # deg^2
deg2_in_sphere = 41252.96 # deg^2 in a spere
fsky_IST = survey_area/deg2_in_sphere
fsky_syvain = 0.375 

ind =  np.genfromtxt("/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_forecast/input/vincenzo/indici_vincenzo_like.dat").astype(int); ind = ind - 1


which_forecast = 'CLOE'

if which_forecast == 'IST':
    fsky = fsky_IST
    GL_or_LG = 'GL'
    ind_ordering = 'vincenzo'
    Cij_folder = 'Cij_14may'


elif which_forecast == 'sylvain': 
    fsky = fsky_syvain
    GL_or_LG = 'GL'
    ind_ordering = 'vincenzo'
    Cij_folder = 'common_ell_and_deltas'


elif which_forecast == 'CLOE':
    fsky = fsky_IST
    GL_or_LG = 'LG'
    ind_ordering = 'CLOE'
    Cij_folder = 'Cl_CLOE'


general_config = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 5000,
    'zbins': 10,
    'nProbes': 2,
    'nbl': 30,
    'which_forecast': which_forecast, # ie choose whether to have IST's or sylvain's deltas
    'Cij_folder': Cij_folder,
    'use_WA': True 
    }

if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_ordering': ind_ordering,
    'GL_or_LG': GL_or_LG,  
    'save_SSC_only_covmats': False,
    'compute_covariance_in_blocks': False,
    'fsky': fsky,
    'Rl': 4,
    'ind': ind
    }

FM_config = {
    'nParams': 20,
    'save_FM': False
    }

plot_config = {
    'case': 'opt',
    'probe': '3x2pt',
    'SSC_flag': 'GpSSC',
    'covmat_dav_flag': 'no',
    'which_plot': 'constraints_only'
    }
