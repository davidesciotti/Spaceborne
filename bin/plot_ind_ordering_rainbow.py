import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time

project_path = Path.cwd().parent
sys.path.append(str(project_path))
import lib.my_module as mm

start_time = time.perf_counter()

params = {'lines.linewidth' : 3.5,
          'font.size' : 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral'
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
###############################################################################
###############################################################################
"""
This code plots the indices unpacking in the desired ordeing, 
showing it on top of the matrix. The indices of the matrix are i and j, the value
is p
"""


z_bins = 10

p = 0
data = np.zeros((z_bins, z_bins))
for i in range(z_bins):
    for j in range(i, z_bins):
        data[i, j] = p
        p+=1


data = data.astype('int')
        
fig, ax = plt.subplots()
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(data, cmap='rainbow')

# my way (no zeros on lower diagonal)
for i in range(z_bins):
    for j in range(i, z_bins):
        ax.text(j, i, f'{data[i, j]}', ha='center', va='center')
plt.show()

# with zeros under the lower diagonal  
for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, f'{z}', ha='center', va='center')
plt.show()