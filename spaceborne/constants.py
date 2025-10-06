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

# admittedly, these are not physical constants ^^
HS_ALL_PROBE_COMBS = [
    'LLLL', 'LLGL', 'LLGG',
    'GLLL', 'GLGL', 'GLGG',
    'GGLL', 'GGGL', 'GGGG',
]  # fmt: skip

RS_ALL_PROBE_COMBS = [
    'xipxip', 'xipxim', 'xipgt', 'xipgg',
    'ximxip', 'ximxim', 'ximgt', 'ximgg',
    'gtxip',  'gtxim',  'gtgt',  'gtgg',
    'ggxip',  'ggxim',  'gggt',  'gggg',
]  # fmt: skip

HS_DIAG_PROBE_COMBS = ['LLLL', 'GLGL', 'GGGG']
RS_DIAG_PROBE_COMBS = ['xipxip', 'ximxim', 'gtgt', 'gggg']
# not used for the moment
HS_DIAG_PROBES = [ 'LL', 'GL', 'GG']
RS_DIAG_PROBES = [ 'xip', 'xim', 'gt', 'gg']
HS_DIAG_PROBES_OC = [ 'mm', 'gm', 'gg']
RS_DIAG_PROBES_OC = [ 'xip', 'xim', 'gm', 'gg']

HS_PROBE_NAME_TO_IX_DICT = {'L': 0, 'G': 1}
HS_PROBE_IX_TO_NAME_DICT = {0: 'L', 1: 'G'}

HS_SYMMETRIZE_OUTPUT_DICT = {
    ('L', 'L'): True,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): True,
}

# bessel functions order for the different real space probes
MU_DICT = {'gg': 0, 'gt': 2, 'xip': 0, 'xim': 4}

# ! careful: in this representation, xipxip and ximxim (eg) have
# ! the same indices!!
RS_PROBE_NAME_TO_IX_DICT = {
    'xipxip': (0, 0, 0, 0),
    'xipxim': (0, 0, 0, 0),
    'xipgt':  (0, 0, 1, 0),
    'xipgg':  (0, 0, 1, 1),

    'ximxip': (0, 0, 0, 0),
    'ximxim': (0, 0, 0, 0),
    'ximgt':  (0, 0, 1, 0),
    'ximgg':  (0, 0, 1, 1),

    'gtxip':  (1, 0, 0, 0),
    'gtxim':  (1, 0, 0, 0),
    'gtgt':   (1, 0, 1, 0),
    'gtgg':   (1, 0, 1, 1),

    'ggxip':  (1, 1, 0, 0),
    'ggxim':  (1, 1, 0, 0),
    'gggt':   (1, 1, 1, 0),
    'gggg':   (1, 1, 1, 1),
}  # fmt: skip

# TODO delete this after you finish OC checks
# RS_PROBE_NAME_TO_IX_DICT_TRIL = {
#     'xipxip': (0, 0, 0, 0),
#     'xipxim': (0, 0, 0, 0),
#     # 'xipgt':  (0, 0, 1, 0),
#     # 'xipgg':  (0, 0, 1, 1),

#     # 'ximxip': (0, 0, 0, 0),
#     'ximxim': (0, 0, 0, 0),
#     # 'ximgt':  (0, 0, 1, 0),
#     # 'ximgg':  (0, 0, 1, 1),

#     'gtxip':  (1, 0, 0, 0),
#     'gtxim':  (1, 0, 0, 0),
#     'gtgt':   (1, 0, 1, 0),
#     # 'gtgg':   (1, 0, 1, 1),

#     'ggxip':  (1, 1, 0, 0),
#     'ggxim':  (1, 1, 0, 0),
#     'gggt':   (1, 1, 1, 0),
#     'gggg':   (1, 1, 1, 1),
# }  # fmt: skip

# Heracles-specific probe mappings: POS (position, spin-0), SHE (shear, spin-2)
HS_PROBE_IX_TO_NAME_DICT_HERACLES = {0: 'POS', 1: 'SHE'}
HS_PROBE_NAME_TO_IX_DICT_HERACLES = {'POS': 0, 'SHE': 1}

# this dictionary specifies the dimension of the corresponding axes in the output
# arrays. The dimensions correspond to the spin, except POS (spin-0) still needs 1
# dimension (not 0!)
HS_PROBE_DIMS_DICT_HERACLES = {'POS': 1, 'SHE': 2}

RS_PROBE_NAME_TO_IX_DICT_SHORT = {
    'gg': 0,  # w
    'gt': 1,  # \gamma_t
    'xip': 2,
    'xim': 3,
}

RS_PROBE_NAME_TO_LATEX = {
    'xip': r'$\xi_{+}$',
    'xim':  r'$\xi_{-}$',
    'gt':  r'$\gamma_{t}$',
    'gg':  r'$w$',
}
HS_PROBE_NAME_TO_LATEX = {
    'LL': r'${\rm LL}$',
    'GL':  r'${\rm GL}$',
    'GG':  r'${\rm GG}$',
}