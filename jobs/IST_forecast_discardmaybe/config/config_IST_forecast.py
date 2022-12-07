# official Euclid survey area
survey_area = 15000  # deg^2
deg2_in_sphere = 41252.96  # deg^2 in a spere
fsky_IST = survey_area / deg2_in_sphere
fsky_syvain = 0.375

which_forecast = 'IST'

if which_forecast == 'IST':
    fsky = fsky_IST
    GL_or_LG = 'GL'
    ind_ordering = 'vincenzo'
    cl_folder = 'Cij_14may'

elif which_forecast == 'sylvain':
    fsky = fsky_syvain
    GL_or_LG = 'GL'
    ind_ordering = 'vincenzo'
    cl_folder = 'common_ell_and_deltas'

elif which_forecast == 'CLOE':
    fsky = fsky_IST
    GL_or_LG = 'LG'
    ind_ordering = 'CLOE'
    cl_folder = 'Cl_CLOE'

else:
    raise ValueError('which_forecast must be IST, CLOE or syvain')

general_cfg = {
    'ell_min': 10,
    'ell_max_WL': 5000,
    'ell_max_GC': 3000,
    'zbins': 10,
    'nbl': 30,  # equal nbl for all probes
    'nbl_WL': 30,  # different nbl for all probes, with nbl_probe computed from the ell cuts and nbl_WL
    'nbl_GC': 30,
    'which_forecast': which_forecast,  # ie choose whether to have IST's or sylvain's deltas
    'cl_folder': '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/14may/CijDers/EP10',
    'rl_folder': '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/Pk_responses_2D/EP10',
    'use_WA': True,
    'save_cls': False,
    'EP_or_ED': 'EP',
    'n_probes': 2,
}

if general_cfg['ell_max_WL'] == general_cfg['ell_max_GC']:
    general_cfg['use_WA'] = False

covariance_cfg = {
    'ind_ordering': ind_ordering,
    'GL_or_LG': GL_or_LG,
    'save_SSC_only_covmats': False,
    'compute_covariance_in_blocks': False,
    'fsky': fsky,
    'Rl': 4,
    'save_covariance': False,
    'block_index': 'ell',  # ! should probably be ell
    # this is the one used by me, Vincenzo and CLOE. The blocks in the 2D covmat will be indexed by ell1, ell2
    'which_probe_response': 'constant',
    'sigma_eps2': 0.3 ** 2,
    'ng': 30,
}

FM_cfg = {
    'nParams': 20,
    'save_FM': True,
    'save_FM_as_dict': False
}

plot_cfg = {
    'case': 'opt',
    'probe': '3x2pt',
    'GO_or_GS': 'GS',  # Gauss-only or Gauss + SSC
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
