# official Euclid survey area
survey_area = 15000  # deg^2
deg2_in_sphere = 41252.96  # deg^2 in a spere
fsky_IST = survey_area / deg2_in_sphere
fsky_syvain = 0.375

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
    GL_or_LG = 'GL'  # ! Actually, it may very well be (actually, it is at 99%) ind_ordering=triu, GL_or_LG=GL!!!!
    ind_ordering = 'triu'
    Cij_folder = 'Cij_14may'
    # it was GL_or_LG = 'LG', ind_ordering = 'CLOE'

general_config = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 5000,
    'zbins': 10,
    'nProbes': 2,
    'nbl': 20,
    'which_forecast': which_forecast,  # ie choose whether to have IST's or sylvain's deltas
    'Cij_folder': Cij_folder,
    'use_WA': False
}

# if I'm using the same ell_max there's no Wadd
if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    # these 3 are the principal ordering conventions for the covariance matrix
    'ind_ordering': ind_ordering,
    'GL_or_LG': GL_or_LG,
    'block_index': 'C-style',  # Sylvain and CLOE use ij, corresponding to "F-style" flattening of the datavector.

    'save_SSC_only_covmats': False,
    'compute_covariance_in_blocks': False,
    'fsky': fsky,
    'Rl': 4,
    'save_covariance': True,
    # me and Vincenzo use instead "ell", corresponding to "C-style", flattening of the datavector.
}

FM_config = {
    'nParams': 20,
    'save_FM': False
}

plot_config = {
    'case': 'opt',
    'probe': '3x2pt',
    'GO_or_GS': 'GO',  # Gauss-only or Gauss + SSC
    'covmat_dav_flag': 'no',
    'which_plot': 'constraints_only',

    'params': {'lines.linewidth': 3.5,
               'font.size': 25,
               'axes.labelsize': 'xx-large',
               'axes.titlesize': 'xx-large',
               'xtick.labelsize': 'xx-large',
               'ytick.labelsize': 'xx-large',
               'mathtext.fontset': 'stix',
               'font.family': 'STIXGeneral',
               'figure.figsize': (14, 10)
               },

    'markersize': 200,
    'dpi': 500,
    'pic_format': 'pdf',

    'param_names_label': ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_s$",
                          "$\sigma_8$"]

}
