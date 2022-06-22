# official Euclid survey area
survey_area = 15000  # deg^2
deg2_in_sphere = 41252.96  # deg^2 in a spere
fsky_IST = survey_area / deg2_in_sphere
fsky_syvain = 0.375

general_config = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': 10,
    'nProbes': 2,
    'nbl': 30,
    'which_forecast': 'ISTF',  # ie choose whether to have IST's or sylvain's deltas
    'cl_folder': 'Cij_14may',
    'use_WA': True
}

# Wadd is defined only if ell_max_WL != ell_max_GC
if general_config['ell_max_WL'] == general_config['ell_max_GC']:
    general_config['use_WA'] = False

covariance_config = {
    'ind_ordering': 'vincenzo',
    'GL_or_LG': 'GL',
    'save_SSC_only_covmats': False,
    'compute_covariance_in_blocks': False,
    'fsky': fsky_IST,
    'Rl': 4,
    'save_covariance': True,
    'block_index': 'ij',
    'which_SSC': 'PyCCL',  # use Cosmolike or PyCCL
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'which_probe_response': 'constant',

    'PyCCL_config': {
        'hm_recipe': 'KiDS1000',
        'probe': 'GC',
        'SSC_or_cNG': 'SS'}
}

FM_config = {
    'nParams': 20,
    'save_FM': False
}

plot_config = {
    'case': 'opt',
    'probe': '3x2pt',
    'GO_or_GS': 'GS',  # Gauss-only or Gauss + SSC
    'covmat_dav_flag': 'no',
    'which_plot': 'constraints_only',
    'plot_sylvain': True,
    'plot_ISTF': True,

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

    'param_names_label': ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_{\\rm s}$",
                          "$\sigma_8$"]

}
