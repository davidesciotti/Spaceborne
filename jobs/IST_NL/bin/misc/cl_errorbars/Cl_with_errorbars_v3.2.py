# %load_ext autoreload
# %autoreload 2
# %matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_path = Path.cwd().parent.parent
import time

sys.path.append(str(project_path.parent / 'my_module'))
import my_module as mm
from matplotlib.pyplot import cm
from matplotlib.cm import get_cmap

start_time = time.perf_counter()

# params = {'lines.linewidth' : 3.5,
#           'font.size' : 20,
#           'axes.labelsize': 'x-large',
#           'axes.titlesize':'x-large',
#           'xtick.labelsize':'x-large',
#           'ytick.labelsize':'x-large',
#           'mathtext.fontset': 'stix',
#           'font.family': 'STIXGeneral',
#           'figure.figsize': (10, 10)
#           }
# plt.rcParams.update(params)
# markersize = 10

# %matplotlib widget

## Define a color palette for the plots
name = "Dark2"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors


###############################################################################
###############################################################################
###############################################################################


def fill_upper_triangle(Cl_2D, nbl, npairs, zbins):
    # fill upper triangle: LL, GG, WLonly        
    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for i in range(npairs):
            Cl_3D[ell, triu_idx[0][i], triu_idx[1][i]] = Cl_2D[ell, i]
    return Cl_3D


###############################################################################
###############################################################################
###############################################################################

# TODO: covariance of the linear Cls 

path = '/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric'

NL_flags = 5  # 4 different recipes + 1 for the linear model
nbl = 20
zbins = 10
n_zpairs = zbins * (zbins + 1) // 2
iz_values = np.asarray([1, 3, 5, 10])

ind_CLOE = np.genfromtxt('/Users/davide/Documents/Lavoro/Programmi/common_data/ind_files/indici_cloe_like.dat').astype(int) - 1
ind_CLOE_LL = ind_CLOE[:55, 2:]


# import Cls:
ClWL_2D = np.zeros((NL_flags, nbl, n_zpairs + 1))
ClWL_2D[0, ...] = np.genfromtxt(f'{path}/CijLL-LCDM-Lin-zNLA.dat')
for NL_flag in range(1, NL_flags):
    ClWL_2D[NL_flag, ...] = np.genfromtxt(f'{path}/data/Cls_zNLA_ShearShear_NL_flag_{NL_flag}.dat')

# set ell vaues
ell_values = ClWL_2D[1, :, 0]

# just a check on the number of ell bins
nbl_fromCl = ell_values.shape[0]
assert nbl == nbl_fromCl, 'check the number of ell bins'

# delete ell column
ClWL_2D = np.delete(ClWL_2D, 0, 2)

# reshape in 3D
ClWL_3D = np.zeros((NL_flags, nbl, zbins, zbins))
for NL_flag in range(5):
    ClWL_3D[NL_flag, ...] = fill_upper_triangle(ClWL_2D[NL_flag, ...], nbl, n_zpairs, zbins)
    ClWL_3D[NL_flag, ...] = mm.fill_3D_symmetric_array(ClWL_3D[NL_flag, ...], nbl, zbins)


# covmats: import,  initialize and reshape
cov_WL_GO_2D = np.zeros((NL_flags, nbl * n_zpairs, nbl * n_zpairs))
cov_WL_GS_2D = np.copy(cov_WL_GO_2D)
cov_WL_GO_4D = np.zeros((NL_flags, nbl, nbl, n_zpairs, n_zpairs))
cov_WL_GS_4D = np.copy(cov_WL_GO_4D)
cov_WL_GO_6D = np.zeros((NL_flags, nbl, nbl, zbins, zbins, zbins, zbins))
cov_WL_GS_6D = np.copy(cov_WL_GO_6D)
for NL_flag in range(1, NL_flags):
    # import
    cov_WL_GO_2D[NL_flag, ...] = np.load(f'/Users/davide/Documents/Lavoro/Programmi/SSC_for_ISTNL/output/covmat/CovMat-ShearShear-Gauss-20bins-NL_flag_{NL_flag}.npy')
    cov_WL_GS_2D[NL_flag, ...] = np.load(f'/Users/davide/Documents/Lavoro/Programmi/SSC_for_ISTNL/output/covmat/CovMat-ShearShear-GaussSSC-20bins-NL_flag_{NL_flag}.npy')
    # reshape to 4D
    cov_WL_GO_4D[NL_flag, ...] = mm.cov_2D_to_4D_new(cov_WL_GO_2D[NL_flag, ...], nbl, n_zpairs, block_index='vincenzo')
    cov_WL_GS_4D[NL_flag, ...] = mm.cov_2D_to_4D_new(cov_WL_GS_2D[NL_flag, ...], nbl, n_zpairs, block_index='vincenzo')
    # reshape to 6D
    cov_WL_GO_6D[NL_flag, ...] = mm.cov_4D_to_6D(cov_WL_GO_4D[NL_flag, ...], nbl, zbins, probe='WL',ind=ind_CLOE[:55, :])
    cov_WL_GS_6D[NL_flag, ...] = mm.cov_4D_to_6D(cov_WL_GS_4D[NL_flag, ...], nbl, zbins, probe='WL',ind=ind_CLOE[:55, :])




# fill the sigma arrays; these are actually 3 for loops!
sigma_WL_GO = np.zeros((NL_flags, iz_values.size, nbl))
sigma_WL_GS = np.copy((sigma_WL_GO))
for NL_flag in range(1, NL_flags):
    for iz_idx, iz in enumerate(iz_values):
        sigma_WL_GO[NL_flag, iz_idx, :] = np.sqrt([cov_WL_GO_6D[NL_flag, cur_ell, cur_ell, iz-1, iz-1, iz-1, iz-1] for cur_ell, _ in enumerate(ell_values)])
        sigma_WL_GS[NL_flag, iz_idx, :] = np.sqrt([cov_WL_GS_6D[NL_flag, cur_ell, cur_ell, iz-1, iz-1, iz-1, iz-1] for cur_ell, _ in enumerate(ell_values)])


# test plot with errorbars
iz = 1
NL_flag = 1
ClWL_values = [ClWL_3D[NL_flag, cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ell_values)]


# plt.figure()
# plt.errorbar(ells, ClWL_values, sigma_WL_GS[NL_flag, iz, :], c='red',  label = 'GS')
# plt.errorbar(ells, ClWL_values, sigma_WL_GO[NL_flag, iz, :], fmt='none', alpha=0.6, c='blue', label = 'GO')
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()

# Visualize the result
# plt.plot(ells, ClWL_values, c='red',  label = 'GS')
# plt.fill_between(ells, ClWL_values - sigma_WL_GS[NL_flag, iz, :], ClWL_values + sigma_WL_GS[NL_flag, iz, :],
#                  color='gray', alpha=0.2)

# plt.yscale('log')
# plt.xscale('log')
# plt.legend()


#######################################################################################
########################## sample from multivariate Gaussian ##########################
#######################################################################################
# take only the correct elements for each iz (for the moment just one iz is implemented)
# unpack the iz indices:
# FIXME this whould be implemented in a better way, (not strictly necessary)
# for iz in iz_values:
#     column_1 = np.where(ind_CLOE_LL[:, 0] == iz-1)[0]
#     column_2 = np.where(ind_CLOE_LL[:, 1] == iz-1)[0]
#     print(iz, column_1, column_2)

# store the resulting zpairs
iz_expanded = np.asarray([0, 19, 34, 54])  

# check to make sure the zpairs correspond to the iz_values given
assert np.all(iz_values == [1, 3, 5, 10]), 'the zpairs indices corresponding to the iz chosen are hardcoded, check your iz values'


# initialize random number generator
rng = np.random.default_rng()

# TODO is "flatten" the right ordering? ✅ yes, should be 
# TODO take the correct 20 elements from the 1100 outputted by the routine below
# TODO still unsure about both the unpacking and if the elements im pulling out of the datavectors are the correct ones

n_samples = 10_000
GO_samples_thinned = np.zeros((NL_flags, iz_values.size, nbl, n_samples))
GS_samples_thinned = np.copy(GO_samples_thinned)
GO_samples_in_range = np.copy(GO_samples_thinned)
GS_samples_in_range = np.copy(GO_samples_thinned)
GO_in_conf_interval = []
GS_in_conf_interval = []

# fill the sigma arrays from the multivariate Gaussian
for NL_flag in range(1, NL_flags):

    # generate the multivariate Gaussian
    mean = ClWL_2D[NL_flag, ...].flatten() # TODO check this flatten
    GO_samples = rng.multivariate_normal(mean=mean, cov=cov_WL_GO_2D[NL_flag, ...], size=n_samples)
    GS_samples = rng.multivariate_normal(mean=mean, cov=cov_WL_GS_2D[NL_flag, ...], size=n_samples)

   
    for cur_ell in range(nbl):
        for iz_idx in range(4):
            
            # this takes the correct index (translates (iz_idx, iz_idx) into zpair)
            zpair = iz_expanded[iz_idx]
            GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] = GO_samples[:, cur_ell*n_zpairs + zpair]
            GS_samples_thinned[NL_flag, iz_idx, cur_ell, :] = GS_samples[:, cur_ell*n_zpairs + zpair]

            # fill the sigma_sampled array
            sigma_WL_GO_sampled = np.sqrt(np.var(GO_samples_thinned, axis=3))
            sigma_WL_GS_sampled = np.sqrt(np.var(GS_samples_thinned, axis=3))

            # test: the variance computed using the 2 methods must me the same (for GO)
            # print(iz_idx, cur_ell, 'sigma_chain =\t', sigma_WL_GS_sampled[NL_flag, iz_idx, cur_ell])
            # print(iz_idx, cur_ell, 'sigma_old =\t', sigma_WL_GS[NL_flag, iz_idx, cur_ell], '\n')

            # santiago's checks:
            GO_samples_in_range[NL_flag, iz_idx, cur_ell, :] =\
            np.logical_and(GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] < mean[cur_ell*n_zpairs + zpair] + sigma_WL_GO[NL_flag, iz_idx, cur_ell],\
                        GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] > mean[cur_ell*n_zpairs + zpair] - sigma_WL_GO[NL_flag, iz_idx, cur_ell])
            GS_samples_in_range[NL_flag, iz_idx, cur_ell, :] =\
            np.logical_and(GS_samples_thinned[NL_flag, iz_idx, cur_ell, :] < mean[cur_ell*n_zpairs + zpair] + sigma_WL_GS[NL_flag, iz_idx, cur_ell],\
                        GS_samples_thinned[NL_flag, iz_idx, cur_ell, :] > mean[cur_ell*n_zpairs + zpair] - sigma_WL_GS[NL_flag, iz_idx, cur_ell])
            
            n_samples_in_range_GO = np.count_nonzero(GO_samples_in_range[NL_flag, iz_idx, cur_ell, :]) / n_samples * 100
            n_samples_in_range_GS = np.count_nonzero(GS_samples_in_range[NL_flag, iz_idx, cur_ell, :]) / n_samples * 100
            
            GO_in_conf_interval.append(n_samples_in_range_GO)
            GS_in_conf_interval.append(n_samples_in_range_GS)
            
plt.figure(figsize=(10, 8))              
plt.plot(range(len(GO_in_conf_interval)), GO_in_conf_interval, label="% GO_in_conf_interval")
plt.plot(range(len(GS_in_conf_interval)), GS_in_conf_interval, label="% GS_in_conf_interval")

plt.ylabel('% elements in $\mu \pm 1\sigma$')
plt.xlabel('case')
plt.legend()
sys.exit()
   
   
cur_ell = 0
iz_idx = 0
zpair = iz_expanded[iz_idx]
# santiago's checks:
GO_samples_in_range[NL_flag, iz_idx, cur_ell, :] =\
    np.logical_and(GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] < mean[cur_ell*n_zpairs + zpair] + sigma_WL_GO[NL_flag, iz_idx, cur_ell],\
                   GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] > mean[cur_ell*n_zpairs + zpair] - sigma_WL_GO[NL_flag, iz_idx, cur_ell])
# )
#     GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] < mean[cur_ell*n_zpairs + zpair] + sigma_WL_GS[NL_flag, iz_idx, cur_ell] and \
#     GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] > mean[cur_ell*n_zpairs + zpair] - sigma_WL_GS[NL_flag, iz_idx, cur_ell]

# a = np.asarray((4, 5, 7, 9, 10, 11, 12, 13, 14, 15))
print(np.count_nonzero(GO_samples_in_range[NL_flag, iz_idx, cur_ell, :]) / n_samples * 100,\
    '% of elements fall within confidence interval')
# print(np.count_nonzero((9 < a) and (a < 14)))

            
# compare if the samples in 2, fall inside or outside the confidence bands of 1
# if everything was diagonal and perfect
# 68% of the samples should fall inside the 1-sigma confidence band
# you can do it also with 2 or 3-sigma



# test the flattening
"""NL_flag = 1
ClWL_1D_1 = ClWL_2D[NL_flag, ...].flatten()
ClWL_1D_2 = ClWL_2D[NL_flag, ...].transpose().flatten()

ClWL_1D_3 = np.zeros((nbl*n_zpairs))
p = 0
for ell in range(nbl):
    for zpair in range(n_zpairs):
        ClWL_1D_3[p] = ClWL_2D[NL_flag, ell, zpair]
        p += 1

plt.figure(figsize=(13, 10))
plt.plot(range(1100), ClWL_1D_1, label='flatten')
plt.plot(range(1100), ClWL_1D_2, label='transpose flatten')
plt.plot(range(1100), ClWL_1D_3, label='manual')
plt.legend()"""

    

# plot the % diff
for NL_flag in range(1, NL_flags):
    mm.compare_2D_arrays(sigma_WL_GS_sampled[NL_flag, ...], sigma_WL_GS[NL_flag, ...])
# plt.figure()
plt.hist(GO_samples_thinned[NL_flag, iz_idx, cur_ell, :], bins = 40)




iz_dict = {1: [0, 0], 3: [0, 1], 5: [1, 0], 10: [1, 1]}
labels = ['Linear Recipe', 'Halofit', 'Mead2020', 'Euclid Emulator', 'Bacco Recipe'] # TODO linear recipe
labels = ['Halofit', 'Mead2020', 'Euclid Emulator', 'Bacco Recipe']

fig, axs = plt.subplots(2, 2, figsize=(24, 18), dpi=200)
fig.suptitle(r'WL $C^{ij}_{\ell}$')

for iz_idx, iz in enumerate(iz_values):
    
    ax1 = axs[iz_dict[iz][0], iz_dict[iz][1]]

    for NL_flag, pl_label in zip(range(1, 5), labels):
        
        # rename for better readibility
        sigma_WL_GO_list = sigma_WL_GO[NL_flag, iz_idx, :]
        sigma_WL_GS_list = sigma_WL_GS[NL_flag, iz_idx, :]
        
        sigma_WL_GO_sampled_list = sigma_WL_GO_sampled[NL_flag, iz_idx, :]
        sigma_WL_GS_sampled_list = sigma_WL_GS_sampled[NL_flag, iz_idx, :]
    
        ClWL_3D_list = [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ell_values)]
        
        # plot, no uncertainties
        ax1.loglog(ell_values, ClWL_3D_list, color=colors[NL_flag], ls='--', label=f'{pl_label}', lw=1.2)

        # plot GO and GS uncertainties
        if NL_flag == 4:
            ax1.fill_between(ell_values, ClWL_3D_list - sigma_WL_GS_sampled_list, ClWL_3D_list + sigma_WL_GS_sampled_list, label = 'GaussSSC' if NL_flag == 4 else "", color='blue', alpha=0.2)
            ax1.fill_between(ell_values, ClWL_3D_list - sigma_WL_GO_sampled_list, ClWL_3D_list + sigma_WL_GO_sampled_list, label = 'Gauss-only'if NL_flag == 4 else "", color='red', alpha=0.3)


    """ # old code
    pl_label_0 = 'Linear Recipe'
    ax1.loglog(ells, [ClWL_0_3D[cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ells)], color=colors[1], ls='-.', label=pl_label_0, lw=1.2)
   
    pl_label_1 = 'Halofit'
    NL_flag = 1
    # ax1.loglog(ells, [ClWL_1_3D[cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ells)], color=colors[2], ls='--', label=pl_label_1, lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GS[NL_flag, iz_idx, :], color=colors[2], ls='--', label=f'{pl_label_1} GS', lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GO[NL_flag, iz_idx, :], fmt='none', color=colors[6], ls='--', label=f'{pl_label_1} GO', lw=1.2)

    pl_label_2 = 'Mead2020'
    NL_flag = 2
    # ax1.loglog(ells, [ClWL_2_3D[cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ells)], color=colors[3], ls='--', label=pl_label_2, lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GS[NL_flag, iz_idx, :], color=colors[3], ls='--', label=f'{pl_label_2} GS', lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GO[NL_flag, iz_idx, :], fmt='none', color=colors[7], ls='--', label=f'{pl_label_2} GO', lw=1.2)

    pl_label_3 = 'Euclid Emulator'
    NL_flag = 3
    # ax1.loglog(ells, [ClWL_3_3D[cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ells)], color=colors[4], ls=':', label=pl_label_3, lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GS[NL_flag, iz_idx, :], color=colors[4], ls='--', label=f'{pl_label_3} GS', lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GO[NL_flag, iz_idx, :], fmt='none', color=colors[7], ls='--', label=f'{pl_label_3} GO', lw=1.2)

    pl_label_4 = 'Bacco Recipe'
    NL_flag = 4
    # ax1.loglog(ells, [ClWL_4_3D[cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ells)], color=colors[5], ls=':', label=pl_label_4, lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GS[NL_flag, iz_idx, :], color=colors[5], ls='--', label=f'{pl_label_4} GS', lw=1.2)
    ax1.errorbar(ells, [ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ells)],
                 yerr=sigma_WL_GO[NL_flag, iz_idx, :], fmt='none', color=colors[5], ls='--', label=f'{pl_label_4} GO', lw=1.2)
    """

    ax1.set_xlabel(r'$\ell$', fontsize=7)
    ax1.set_ylabel(r'$C_\ell$ $[sr^{-1}]$', fontsize=7)
    ax1.set_title('bin i,j={:d},{:d}'.format(iz, iz))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend(prop={'size': 6})
