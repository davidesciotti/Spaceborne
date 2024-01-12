"""
Some general configurations for matplotlib
"""
import sys

ROOT = '/home/davide/Documenti/Lavoro/Programmi'

sys.path.append(f'{ROOT}/Spaceborne/bin')
import my_module as mm


mpl_rcParams_dict = {'lines.linewidth': 3.5,
                     'font.size': 20,
                     'axes.labelsize': 'x-large',
                     'axes.titlesize': 'x-large',
                     'xtick.labelsize': 'x-large',
                     'ytick.labelsize': 'x-large',
                     'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral',
                     'figure.figsize': (15, 10),
                     'lines.markersize': 8,
                     # 'axes.grid': True,
                     # 'figure.constrained_layout.use': True,
                     # 'axes.axisbelow': True
                     }

general_dict = {
    'cosmo_labels_TeX': ["$\Omega_{{\\rm m},0}$", "$\Omega_{{\\rm b},0}$", "$w_0$", "$w_a$", "$h$", "$n_{\\rm s}$",
                         "$\sigma_8$"],
    'IA_labels_TeX': ['$A_{\\rm IA}$', '$\eta_{\\rm IA}$', '$\\beta_{\\rm IA}$'],
    'galaxy_bias_labels_TeX': mm.build_labels_TeX(zbins=10)[0],
    'shear_bias_labels_TeX': mm.build_labels_TeX(zbins=10)[1],
    'zmean_shift_labels_TeX': mm.build_labels_TeX(zbins=10)[2],

    'cosmo_labels': ['Om', 'Ob', 'wz', 'wa', 'h', 'ns', 's8'],
    'IA_labels': ['AIA', 'etaIA', 'betaIA'],
    'galaxy_bias_labels': mm.build_labels(zbins=10)[0],
    'shear_bias_labels': mm.build_labels(zbins=10)[1],
    'zmean_shift_labels': mm.build_labels(zbins=10)[2],

    'ylabel_perc_diff_wrt_mean': "$ \\bar{\sigma}_\\alpha^i / \\bar{\sigma}^{\\; m}_\\alpha -1 $ [%]",
    'ylabel_sigma_relative_fid': '$ \\sigma_\\alpha/ \\theta^{fid}_\\alpha $ [%]',
    'dpi': 500,
    'pic_format': 'pdf'
}

h_over_mpc_tex = '$h\,{\\rm Mpc}^{-1}$'
kmax_tex = '$k_{\\rm max}$'
kmax_star_tex = '$k_{\\rm max}^\\star$'
