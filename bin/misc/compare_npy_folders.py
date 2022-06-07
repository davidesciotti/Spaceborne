import numpy as np
import matplotlib.pyplot as plt
import sys

path_lib = '/'

sys.path.append('path_lib')
import my_module as mm

"""
This script compares all the npy files in 2 folders
"""

# change to the desired folder containing the npy files
new = dict(mm.get_kv_pairs_npy('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/jobs/IST_NL/output/covmat/new_settings'))
old = dict(mm.get_kv_pairs_npy('/jobs/IST_NL/output/covmat'))

for key in new.keys():

    if np.array_equal(old[key], new[key]):
        emoji = '✅'
    else:
        emoji = '🤔'

    print(key, emoji)

