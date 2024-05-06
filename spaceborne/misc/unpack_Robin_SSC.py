import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time
import re

# get project directory
path = Path.cwd().parent.parent

# import configuration and functions modules
sys.path.append(str(path.parent / 'my_module'))
import my_module as mm

start_time = time.perf_counter()

params = {'lines.linewidth' : 3.5,
          'font.size' : 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8)
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
###############################################################################
###############################################################################



"""
cov_ss_{probe}_zXzX_{probe}_zXzX.txt, where {probe} is "clust" for clustering, 
"shear", or "ggl" for galaxy-galaxy lensing, and X is the four redshift bins 
numbered 1-10. In the case of GGL, the first redshift bin is the position (lens) 
bin and the second is the shear (source).
"""


# /home/davide/Documenti/Lavoro/Programmi/SSC_paper_jan22/data/robin/lmax5000_noextrap

lmax = 5000
zbins = 10
extrapolation = 'noextrap'
probe = 'shear'
nbl = 30
npairs = 55
cov_R_WL_SSC_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
ind = ind = mm.get_ind_file(path, 'vincenzo', 'sylvain')




# import all files in dict
cov_R_dict = dict(
    mm.get_kv_pairs('/home/davide/Documenti/Lavoro/Programmi/SSC_paper_jan22/data/robin/lmax5000_noextrap', 'txt'))
# select only the cov_ss_shear_..._shear... strings
mystr = 'cov_ss_shear_z'
shear_keys_list = []
for key in cov_R_dict.keys():
    if key.startswith(mystr) and ('ggl' not in key) and (('clust' not in key)): shear_keys_list.append(key)

# extract the digits, store them into a list, convert as int, subtract one
z = []
for name in shear_keys_list:
    z.append(re.findall(r'\d+', name))

z = np.asarray(z).astype('int') - 1

# import the txt files fas
for i, j, k, l in zip(z[:, 0], z[:, 1], z[:, 2], z[:, 3]):
    cov_R_WL_SSC_6D[:, :, i, j, k, l] = np.genfromtxt(path /\
                        f'data/robin/lmax{lmax}_{extrapolation}/cov_ss_{probe}_z{i+1}z{j+1}_{probe}_z{k+1}z{l+1}.txt')


# symmetryze (works)
for i, j, k, l in zip(z[:, 0], z[:, 1], z[:, 2], z[:, 3]):
    cov_R_WL_SSC_6D[:, :, k, l, i, j] = cov_R_WL_SSC_6D[:, :, i, j, k, l]

                     

# TODO check if the cov reshaping works?



# now symmetrize i<->j!!! for LL and GG this should be the case? 
# TODO thorough check?
for i in range(zbins):
    for j in range(zbins):
        cov_R_WL_SSC_6D[:,:,j,i,:,:] = cov_R_WL_SSC_6D[:,:,i,j,:,:]
        cov_R_WL_SSC_6D[:,:,:,:,j,i] = cov_R_WL_SSC_6D[:,:,:,:,i,j]





# reduce dimensions
cov_R_WL_SSC_4D = mm.cov_6D_to_4D(cov_R_WL_SSC_6D, nbl, zbins, npairs, ind[:55, :])
cov_R_WL_SSC_2D = mm.array_4D_to_2D(cov_R_WL_SSC_4D, nbl, npairs)






# load my SS cov
cov_D_WL_SSC_4D = np.load(path.parent / 'SSC_paper_jan22/output/covmat/cov_WL_SSC.npy')
# reshape in 6D (XXX possible bug)
cov_D_WL_SSC_6D = mm.cov_4D_to_6D(cov_D_WL_SSC_4D, nbl, zbins, npairs, ind[:55, :]) # BUGGY




ell1 = 0
ell2 = 0
i = 5
j = 5
mm.matshow(cov_D_WL_SSC_4D[ell1, ell2, ...], 'davide')
mm.matshow(cov_R_WL_SSC_4D[ell1, ell2, ...], 'rob')

diff = mm.percent_diff(cov_D_WL_SSC_4D, cov_R_WL_SSC_4D)
mm.matshow(diff[ell1, ell2, ...], '% diff')

"""
# cov_D_WL_G_2D = np.load('/home/davide/Documenti/Lavoro/Programmi/SSCcomp_prove/output/covmat/common_ell_and_deltas/Cij_14may/covmat_G_WL_lmaxWL5000_nbl30_2D.npy')
cov_D_WL_SSC_4D = np.load('/home/davide/Documenti/Lavoro/Programmi/SSCcomp_prove/output/covmat/common_ell_and_deltas/Cij_14may/covmat_G_WL_lmaxWL5000_nbl30_4D.npy')
cov_D_WL_G_2D = mm.array_4D_to_2D(cov_D_WL_G_4D, nbl, npairs) # this is tested and works

print(np.all(cov_D_WL_G_2D == cov_D_WL_G_2D.T)) # the 2D matrix is symmetrix, as it should be


# create cov_6D
cov_D_WL_G_6D = mm.cov_4D_to_6D(cov_D_WL_G_4D, nbl, zbins, npairs, ind[:55, :])
# go back to see what happens
cov_D_WL_G_4D_test = mm.cov_6D_to_4D(cov_D_WL_G_6D, nbl, zbins, npairs, ind[:55, :])
cov_D_WL_G_2D_test = mm.array_4D_to_2D(cov_D_WL_G_4D_test, nbl, npairs)
mm.matshow(cov_D_WL_G_2D_test)


# perform some tests: symmetry (i,j) <-> (k,l)
cov_D_WL_G_6D[0, 0, 1, 1, 1, 2] == cov_D_WL_G_6D[0, 0, 1, 2, 1, 1]
# cov_D_WL_G_6D is also triangular (lower)!! I think this means that it should be symmetric and the 
# cov_D_WL_G_6D

# compare with other 2D and 4D covariance matrices (tricky, need to know which indices are which)
# probably 1st and last correspond very inuitively
print(cov_D_WL_G_2D[0, 0])
print(cov_D_WL_G_4D[0, 0, 0, 0])
"""
 


# TODO check ell values
# TODO finish testing cov_6D, although it should be fine!
# TODO finish writing routine to compute the covariance directly in 6D, then compare it with 
# the cov_4D transformed to 6D with the above routine

# DONE check: symmetry is indeed respected after files import, no couples appear more than once                                 










