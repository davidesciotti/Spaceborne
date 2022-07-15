# official Euclid survey area
survey_area = 15000 # deg^2
deg2_in_sphere = 41252.96 # deg^2 in a spere
fsky_IST = survey_area/deg2_in_sphere
fsky_syvain = 0.375 

which_forecast = 'sylvain'
if which_forecast == 'IST' or which_forecast == 'CLOE': fsky = fsky_IST
elif which_forecast == 'sylvain': fsky = fsky_syvain


general_config = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': 10,
    'nProbes': 2,
    'nbl': 30,
    'which_forecast': which_forecast, # ie choose whether to have IST's or sylvain's deltas
    'Cij_folder': 'Cij_14may',
    'use_WA': True 
    }

if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_ordering': 'vincenzo',
    'GL_or_LG': 'GL',  
    'save_SSC_only_covmats': False,
    'compute_covariance_in_blocks': False,
    'fsky': fsky,
    'Rl': 4
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
