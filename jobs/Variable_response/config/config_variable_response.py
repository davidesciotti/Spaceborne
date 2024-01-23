import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.cm import get_cmap



params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8)
          # 'backend': 'Qt5Agg'
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()


nbl = 30
ell_min = 10
ell_max_WL = 5000
ell_max_GC = 3000



