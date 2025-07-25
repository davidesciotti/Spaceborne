import numpy as np

# HS = harmonic space
# RS = real space

DEG2_IN_SPHERE = 4 * np.pi * (180 / np.pi) ** 2
DEG2_TO_SR = (180 / np.pi) ** 2
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
    'xipxip', 'xipxim', 'xipgm', 'xipgg',
    'ximxip', 'ximxim', 'ximgm', 'ximgg',
    'gmxip',  'gmxim',  'gmgm',  'gmgg',
    'ggxip',  'ggxim',  'gggm',  'gggg',
]  # fmt: skip

HS_DIAG_PROBE_COMBS = ['LLLL', 'GLGL', 'GGGG']
RS_DIAG_PROBE_COMBS = ['xipxip', 'ximxim', 'gmgm', 'gggg']

HS_PROBE_NAME_TO_IX_DICT = {'L': 0, 'G': 1}
HS_PROBE_IX_TO_NAME_DICT = {0: 'L', 1: 'G'}

HS_SYMMETRIZE_OUTPUT_DICT = {
    ('L', 'L'): True,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): True,
}

# bessel functions order for the different real space probes
MU_DICT = {'gg': 0, 'gm': 2, 'xip': 0, 'xim': 4}

# ! careful: in this representation, xipxip and ximxim (eg) have
# ! the same indices!!
RS_PROBE_NAME_TO_IX_DICT = {
    'xipxip': (0, 0, 0, 0),
    'xipxim': (0, 0, 0, 0),
    'xipgm':  (0, 0, 1, 0),
    'xipgg':  (0, 0, 1, 1),

    'ximxip': (0, 0, 0, 0),
    'ximxim': (0, 0, 0, 0),
    'ximgm':  (0, 0, 1, 0),
    'ximgg':  (0, 0, 1, 1),

    'gmxip':  (1, 0, 0, 0),
    'gmxim':  (1, 0, 0, 0),
    'gmgm':   (1, 0, 1, 0),
    'gmgg':   (1, 0, 1, 1),

    'ggxip':  (1, 1, 0, 0),
    'ggxim':  (1, 1, 0, 0),
    'gggm':   (1, 1, 1, 0),
    'gggg':   (1, 1, 1, 1),
}  # fmt: skip


RS_PROBE_NAME_TO_IX_DICT_SHORT = {
    'gg': 0,  # w
    'gm': 1,  # \gamma_t
    'xip': 2,
    'xim': 3,
}
