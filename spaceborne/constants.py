import numpy as np

DEG2_IN_SPHERE = 4 * np.pi * (180 / np.pi)**2
DEG2_TO_SR = (180 / np.pi) ** 2
SR_TO_ARCMIN2 = (180 / np.pi * 60) ** 2

DR1_DATE = 9191.0
SPEED_OF_LIGHT = 299792.458  # km/s

# admittedly, these are not physical constants ^^
ALL_PROBE_COMBS = [
    'LLLL', 'LLGL', 'LLGG',
    'GLLL', 'GLGL', 'GLGG',
    'GGLL', 'GGGL', 'GGGG',
]  # fmt: skip

DIAG_PROBE_COMBS = [
    'LLLL',
    'GLGL',
    'GGGG',
]

PROBE_DICT = {'L': 0, 'G': 1}

SYMMETRIZE_OUTPUT_DICT = {
    ('L', 'L'): True,
    ('G', 'L'): False,
    ('L', 'G'): False,
    ('G', 'G'): True,
}

PROBENAME_DICT = {0: 'L', 1: 'G'}
PROBENAME_DICT_INV = {'L': 0, 'G': 1}
