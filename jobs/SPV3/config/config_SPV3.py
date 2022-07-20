import numpy as np

# official Euclid survey area
survey_area = 15_000  # deg^2
survey_area_SPV3 = 14_000  # deg^2, new in 2022
deg2_in_sphere = 41252.96  # deg^2 in a spere

fsky_IST = survey_area / deg2_in_sphere
fsky_syvain = 0.375
fsky_SPV3 = survey_area_SPV3 / deg2_in_sphere


which_forecast = 'SPV3'

if which_forecast == 'IST':
    fsky = fsky_IST
    GL_or_LG = 'GL'
    ind_ordering = 'vincenzo'
    cl_folder = 'Cij_14may'

elif which_forecast == 'sylvain':
    fsky = fsky_syvain
    GL_or_LG = 'GL'
    ind_ordering = 'vincenzo'
    cl_folder = 'Cij_14may'

elif which_forecast == 'CLOE':
    fsky = fsky_IST
    GL_or_LG = 'LG'
    ind_ordering = 'CLOE'
    cl_folder = 'Cl_CLOE'

elif which_forecast == 'SPV3':
    fsky = fsky_SPV3
    GL_or_LG = 'GL'
    ind_ordering = 'triu' # ! still not super sure, but much better than vincenzo
    cl_folder = 'SPV3'

else:
    raise ValueError('which_forecast must be IST, CLOE or syvain')

general_config = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': 10,
    'nProbes': 2,
    'nbl_WL': 32,
    'which_forecast': which_forecast,  # ie choose whether to have IST's or sylvain's deltas
    'cl_folder': cl_folder,
    'use_WA': True,
    'save_cls': True
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
    'save_covariance': True,
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'block_index': 'ell',
    'which_probe_response': 'variable',
    'sigma_eps2': (0.3*np.sqrt(2))**2,
    'ng': 30,
}

Sijkl_config = {
    'save_Sijkl': True,
    'input_WF': 'vincenzo_SPV3',
    'WF_normalization': 'IST',
    'has_IA': True,  # whether or not to include IA in the WF used to compute Sijkl
    'use_precomputed_sijkl': True,
}

FM_config = {
    'nParams': 20,
    'save_FM': True,
    'save_FM_as_dict': True
}

plot_config = {
    'case': 'opt',
    'probe': 'WL',
    'GO_or_GS': 'GS',  # Gauss-only or Gauss + SSC
    'covmat_dav_flag': 'no',
    'which_plot': 'constrians_only',
    'plot_sylvain': True,
    'plot_ISTF': True,
    'custom_label': '',

    'params': {'lines.linewidth': 3.5,
               'font.size': 25,
               'axes.labelsize': 'xx-large',
               'axes.titlesize': 'xx-large',
               'xtick.labelsize': 'xx-large',
               'ytick.labelsize': 'xx-large',
               'mathtext.fontset': 'stix',
               'font.family': 'STIXGeneral',
               'figure.figsize': (16, 10)
               },

    'markersize': 10,
    'dpi': 500,
    'pic_format': 'pdf',
    'param_names_label': ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_{\\rm s}$",
                          "$\sigma_8$"]
}
