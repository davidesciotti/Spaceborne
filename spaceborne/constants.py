import numpy as np

# Acronyms:
# HS = harmonic space
# RS = real space
# SB = Spaceborne
# HC = Heracles

DEG2_IN_SPHERE = 4 * np.pi * (180 / np.pi) ** 2
DEG2_TO_SR = (np.pi / 180) ** 2
SR_TO_ARCMIN2 = (180 / np.pi * 60) ** 2

DR1_DATE = 9191.0
SPEED_OF_LIGHT = 299792.458  # km/s

ALL_COV_TERMS = ['sva', 'sn', 'mix', 'g', 'ssc', 'cng', 'tot']

# admittedly, these are not physical constants ^^
HS_ALL_PROBE_COMBS = [
    'LLLL', 'LLGL', 'LLGG',
    'GLLL', 'GLGL', 'GLGG',
    'GGLL', 'GGGL', 'GGGG',
]  # fmt: skip

RS_ALL_PROBE_COMBS = [
    'xipxip', 'xipxim', 'xipgt', 'xipw',
    'ximxip', 'ximxim', 'ximgt', 'ximw',
    'gtxip',  'gtxim',  'gtgt',  'gtw',
    'wxip',  'wxim',  'wgt',  'ww',
]  # fmt: skip

CS_ALL_PROBE_COMBS = [
    'EnEn',     'EnBn',     'EnPsigl',     'EnPsigg',
    'BnEn',     'BnBn',     'BnPsigl',     'BnPsigg',
    'PsiglEn',  'PsiglBn',  'PsiglPsigl',  'PsiglPsigg',
    'PsiggEn',  'PsiggBn',  'PsiggPsigl',  'PsiggPsigg',
]  # fmt: skip


HS_DIAG_PROBES = ['LL', 'GL', 'GG']
RS_DIAG_PROBES = ['xip', 'xim', 'gt', 'w']
CS_DIAG_PROBES = ['En', 'Bn', 'Psigl', 'Psigg']
# these are just the diag probe names repeated
HS_DIAG_PROBE_COMBS = [p + p for p in HS_DIAG_PROBES]
RS_DIAG_PROBE_COMBS = [p + p for p in RS_DIAG_PROBES]
CS_DIAG_PROBE_COMBS = [p + p for p in CS_DIAG_PROBES]

# OneCov and OneCov to SB mappings
HS_DIAG_PROBES_OC = ['mm', 'gm', 'gg']
RS_DIAG_PROBES_OC = ['xip', 'xim', 'gm', 'gg']
CS_DIAG_PROBES_OC = ['Emm', 'Bmm', 'Psigm', 'Psigg']
HS_DIAG_PROBES_OC_TO_SB = {'mm': 'LL', 'gm': 'GL', 'gg': 'GG'}
RS_DIAG_PROBES_OC_TO_SB = {'xip': 'xip', 'xim': 'xim', 'gm': 'gt', 'gg': 'w'}
CS_DIAG_PROBES_OC_TO_SB = {'Emm': 'En', 'Bmm': 'Bn', 'Psigm': 'Psigl', 'Psigg': 'Psigg'}

# used for the reshaping logic
HS_AUTO_PROBES = ['LL', 'GG']
RS_AUTO_PROBES = ['xip', 'xim', 'w']
CS_AUTO_PROBES = ['En', 'Bn', 'Psigg']
    
HS_PROBE_NAME_TO_IX_DICT = {'L': 0, 'G': 1}
HS_PROBE_IX_TO_NAME_DICT = {0: 'L', 1: 'G'}

# whether or not to symmetrize the covariance probe blocks when
# reshaping it from 4D to 6D.
# Useful if the 6D cov elements need to be accessed directly, whereas if
# the cov is again reduced to 4D or 2D.
# Can be set to False for a significant speedup, but better to leave as-is for
# safety.
HS_SYMMETRIZE_OUTPUT_DICT = {'LL': True, 'GL': False, 'LG': False, 'GG': True}

# bessel functions order for the different real space probes
MU_DICT = {'w': 0, 'gt': 2, 'xip': 0, 'xim': 4}

# ! careful: in this representation, xipxip and ximxim (eg) have
# ! the same indices!!
RS_PROBE_NAME_TO_IX_DICT = {
    'xipxip': (0, 0, 0, 0),
    'xipxim': (0, 0, 0, 0),
    'xipgt':  (0, 0, 1, 0),
    'xipw':  (0, 0, 1, 1),

    'ximxip': (0, 0, 0, 0),
    'ximxim': (0, 0, 0, 0),
    'ximgt':  (0, 0, 1, 0),
    'ximw':  (0, 0, 1, 1),

    'gtxip':  (1, 0, 0, 0),
    'gtxim':  (1, 0, 0, 0),
    'gtgt':   (1, 0, 1, 0),
    'gtw':   (1, 0, 1, 1),

    'wxip':  (1, 1, 0, 0),
    'wxim':  (1, 1, 0, 0),
    'wgt':   (1, 1, 1, 0),
    'ww':   (1, 1, 1, 1),
}  # fmt: skip

# TODO delete this after you finish OC checks
# RS_PROBE_NAME_TO_IX_DICT_TRIL = {
#     'xipxip': (0, 0, 0, 0),
#     'xipxim': (0, 0, 0, 0),
#     # 'xipgt':  (0, 0, 1, 0),
#     # 'xipw':  (0, 0, 1, 1),

#     # 'ximxip': (0, 0, 0, 0),
#     'ximxim': (0, 0, 0, 0),
#     # 'ximgt':  (0, 0, 1, 0),
#     # 'ximw':  (0, 0, 1, 1),

#     'gtxip':  (1, 0, 0, 0),
#     'gtxim':  (1, 0, 0, 0),
#     'gtgt':   (1, 0, 1, 0),
#     # 'gtw':   (1, 0, 1, 1),

#     'wxip':  (1, 1, 0, 0),
#     'wxim':  (1, 1, 0, 0),
#     'wgt':   (1, 1, 1, 0),
#     'ww':   (1, 1, 1, 1),
# }  # fmt: skip

# Heracles-specific probe mappings: POS (position, spin-0), SHE (shear, spin-2)
HS_PROBE_IX_TO_NAME_DICT_HERACLES = {0: 'POS', 1: 'SHE'}
HS_PROBE_NAME_TO_IX_DICT_HERACLES = {'POS': 0, 'SHE': 1}

# this dictionary specifies the dimension of the corresponding axes in the output
# arrays. The dimensions correspond to the spin, except POS (spin-0) still needs 1
# dimension (not 0!)
HS_PROBE_DIMS_DICT_HERACLES = {'POS': 1, 'SHE': 2}

RS_PROBE_NAME_TO_IX_DICT_SHORT = {
    'w': 0,  # w
    'gt': 1,  # \gamma_t
    'xip': 2,
    'xim': 3,
}

RS_PROBE_NAME_TO_LATEX = {
    'xip': r'$\xi_{+}$',
    'xim': r'$\xi_{-}$',
    'gt': r'$\gamma_{t}$',
    'w': r'$w$',
}
HS_PROBE_NAME_TO_LATEX = {'LL': r'${\rm LL}$', 'GL': r'${\rm GL}$', 'GG': r'${\rm GG}$'}

# adapted from notebook by G. C. Herrera
labels_tex = {
    # old
    'w': 'w_0',
    'omegam': '\\Omega_{\\rm m}',
    'omegab': '\\Omega_{\\rm b}',
    'HMCode_logT_AGN': '\\log{T_{\\rm AGN}}',
    # new
    'H0': 'H_0',
    'h': 'h',
    'ns': 'n_{\\rm s}',
    'log10TAGN': '\\log{T_{\\rm AGN}}',
    'logA': '\\ln{10^{10}\\,A_{\\rm s}}',
    'As': 'A_\\mathrm{s}',
    'w0': 'w_0',
    'wa': 'w_a',
    'Omega_k0': '\\Omega_{\\rm k}',
    'Omega_m0': '\\Omega_{\\rm m}',
    'Omega_b0': '\\Omega_{\\rm b}',
    'Omega_cdm0': '\\Omega_{\\rm c}',
    'ombh2': '\\Omega_{{\\rm b}}\\,h^2',
    'omch2': '\\Omega_{{\\rm c}}\\,h^2',
    'sigma8': '\\sigma_8',
    'gamma_MG': '\\gamma_{\\rm g}',
    'S8': 'S_8',
    'AIA': '\\mathcal{A}_{\\rm IA}',
    'EIA': '\\eta_{\\rm IA}',
    'CIA': '\\mathcal{C}_{\\rm IA}',
    'mnu': '\\Sigma m_\\nu',
    'N_mnu': 'N_\\nu',
    'b1_photo_poly0': 'b_{{\\rm G},0}',
    'b1_photo_poly1': 'b_{{\\rm G},1}',
    'b1_photo_poly2': 'b_{{\\rm G},2}',
    'b1_photo_poly3': 'b_{{\\rm G},3}',
    'magnification_bias_1': 'b_{{\\rm mag},1}',
    'magnification_bias_2': 'b_{{\\rm mag},2}',
    'magnification_bias_3': 'b_{{\\rm mag},3}',
    'magnification_bias_4': 'b_{{\\rm mag},4}',
    'magnification_bias_5': 'b_{{\\rm mag},5}',
    'magnification_bias_6': 'b_{{\\rm mag},6}',
    'dz_pos_1': '\\Delta z_{\\rm G}^{(1)}',
    'dz_pos_2': '\\Delta z_{\\rm G}^{(2)}',
    'dz_pos_3': '\\Delta z_{\\rm G}^{(3)}',
    'dz_pos_4': '\\Delta z_{\\rm G}^{(4)}',
    'dz_pos_5': '\\Delta z_{\\rm G}^{(5)}',
    'dz_pos_6': '\\Delta z_{\\rm G}^{(6)}',
    'dz_pos_7': '\\Delta z_{\\rm G}^{(7)}',
    'dz_pos_8': '\\Delta z_{\\rm G}^{(8)}',
    'dz_pos_9': '\\Delta z_{\\rm G}^{(9)}',
    'dz_pos_10': '\\Delta z_{\\rm G}^{(10)}',
    'dz_pos_11': '\\Delta z_{\\rm G}^{(10)}',
    'dz_pos_12': '\\Delta z_{\\rm G}^{(10)}',
    'dz_pos_13': '\\Delta z_{\\rm G}^{(10)}',
    'dz_shear_1': '\\Delta z_{\\rm L}^{(1)}',
    'dz_shear_2': '\\Delta z_{\\rm L}^{(2)}',
    'dz_shear_3': '\\Delta z_{\\rm L}^{(3)}',
    'dz_shear_4': '\\Delta z_{\\rm L}^{(4)}',
    'dz_shear_5': '\\Delta z_{\\rm L}^{(5)}',
    'dz_shear_6': '\\Delta z_{(6)}^{\\rm L}',
    'dz_shear_7': '\\Delta z_{\\rm L}^{(7)}',
    'dz_shear_8': '\\Delta z_{\\rm L}^{(8)}',
    'dz_shear_9': '\\Delta z_{\\rm L}^{(9)}',
    'dz_shear_10': '\\Delta z_{\\rm L}^{(10)}',
    'dz_shear_11': '\\Delta z_{\\rm L}^{(11)}',
    'dz_shear_12': '\\Delta z_{\\rm L}^{(12)}',
    'dz_shear_13': '\\Delta z_{\\rm L}^{(13)}',
    'multiplicative_bias_1': 'm_{\\rm L}^{(1)}',
    'multiplicative_bias_2': 'm_{\\rm L}^{(2)}',
    'multiplicative_bias_3': 'm_{\\rm L}^{(3)}',
    'multiplicative_bias_4': 'm_{\\rm L}^{(4)}',
    'multiplicative_bias_5': 'm_{\\rm L}^{(5)}',
    'multiplicative_bias_6': 'm_{\\rm L}^{(6)}',
    'multiplicative_bias_7': 'm_{\\rm L}^{(7)}',
    'multiplicative_bias_8': 'm_{\\rm L}^{(8)}',
    'multiplicative_bias_9': 'm_{\\rm L}^{(9)}',
    'multiplicative_bias_10': 'm_{\\rm L}^{(10)}',
    'multiplicative_bias_11': 'm_{\\rm L}^{(11)}',
    'multiplicative_bias_12': 'm_{\\rm L}^{(12)}',
    'multiplicative_bias_13': 'm_{\\rm L}^{(13)}',
    'FoM': '{\\rm FoM}',
}
