import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spar
import scipy.stats as stats
from matplotlib.cm import get_cmap

matplotlib.use('Qt5Agg')

project_path = Path.cwd().parent.parent.parent.parent.parent
job_path = Path.cwd().parent.parent.parent

sys.path.append(str(project_path / 'lib'))
import my_module as mm

matplotlib.use('Qt5Agg')

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8)
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()

# Define a color palette for the plots
name = "Dark2"
cmap = get_cmap(name)
# comment: type: matplotlib.colors.ListedColormap
colors = cmap.colors

# link to notebook: https://gitlab.euclid-sgs.uk/pf-ist-nonlinear/likelihood-implementation/-/blob/develop_NL_emu
# /notebooks/DEMO_ISTNL_photo.ipynb
# cells 29-30


###############################################################################
###############################################################################
###############################################################################

# ! the linear Cl errors (both analytical and sampled) are extremely large in the iz=1 case, at high ell
# ! this may not be an error in the code? the other iz look fine...

repo_path = '/Users/davide/likelihood-implementation/data/ExternalBenchmark/Photometric/data'

NL_flags = 5  # 4 different recipes + 1 for the linear model
NL_labels = ['Linear Recipe', 'Halofit', 'Mead2020', 'Euclid Emulator', 'Bacco Recipe']

nbl = 20
zbins = 10
n_zpairs = zbins * (zbins + 1) // 2
iz_values = np.asarray([1, 3, 5, 10])

ind_CLOE = np.genfromtxt('/Users/davide/Documents/Lavoro/Programmi/common_data/ind_files/indici_cloe_like.dat').astype(
    int) - 1
ind_CLOE_LL = ind_CLOE[:55, 2:]

# import Cls:
ClWL_2D = np.zeros((NL_flags, nbl, n_zpairs + 1))
for NL_flag in range(NL_flags):
    if NL_flag == 0:
        ClWL_2D[NL_flag, ...] = np.genfromtxt(f'{repo_path}/Cls_zNLA_ShearShear.dat')
    else:
        ClWL_2D[NL_flag, ...] = np.genfromtxt(f'{repo_path}/Cls_zNLA_ShearShear_NL_flag_{NL_flag}.dat')

# set ell values
ell_values = ClWL_2D[1, :, 0]

# just a check on the number of ell bins
nbl_fromCl = ell_values.shape[0]
assert nbl == nbl_fromCl, 'check the number of ell bins'

# delete ell column
ClWL_2D = np.delete(ClWL_2D, 0, 2)

# reshape in 3D
ClWL_3D = np.zeros((NL_flags, nbl, zbins, zbins))
for NL_flag in range(5):
    ClWL_3D[NL_flag, ...] = mm.cl_2D_to_3D_symmetric(ClWL_2D[NL_flag, ...], nbl, n_zpairs, zbins=10)

# covmats: import, initialize and reshape
cov_WL_GO_2D = np.zeros((NL_flags, nbl * n_zpairs, nbl * n_zpairs))
cov_WL_GS_2D = np.copy(cov_WL_GO_2D)
cov_WL_SS_2D = np.copy(cov_WL_GO_2D)
cov_WL_mix_2D = np.copy(cov_WL_GO_2D)

cov_WL_GO_4D = np.zeros((NL_flags, nbl, nbl, n_zpairs, n_zpairs))
cov_WL_GS_4D = np.copy(cov_WL_GO_4D)
cov_WL_SS_4D = np.copy(cov_WL_GO_4D)
cov_WL_mix_4D = np.copy(cov_WL_GO_4D)

cov_WL_GO_6D = np.zeros((NL_flags, nbl, nbl, zbins, zbins, zbins, zbins))
cov_WL_GS_6D = np.copy(cov_WL_GO_6D)
cov_WL_SS_6D = np.copy(cov_WL_GO_6D)
cov_WL_mix_6D = np.copy(cov_WL_GO_6D)

for NL_flag in range(NL_flags):

    if NL_flag == 0:  # linear: store in NL = 0 index
        cov_WL_GO_2D[NL_flag, ...] = spar.load_npz(
            job_path / f'output/covmat/CovMat-ShearShear-Gauss-20bins-Sparse.npz').toarray()
        cov_WL_GS_2D[NL_flag, ...] = spar.load_npz(
            job_path / f'output/covmat/CovMat-ShearShear-GaussSSC-20bins-Sparse.npz').toarray()
    else:  # nonlinear
        cov_WL_GO_2D[NL_flag, ...] = spar.load_npz(
            job_path / f'output/covmat/CovMat-ShearShear-Gauss-20bins-NL_flag_{NL_flag}-Sparse.npz').toarray()
        cov_WL_GS_2D[NL_flag, ...] = spar.load_npz(
            job_path / f'output/covmat/CovMat-ShearShear-GaussSSC-20bins-NL_flag_{NL_flag}-Sparse.npz').toarray()

    # ! create mix and SS covariance
    # SS-only covariance: GS-GO
    cov_WL_SS_2D[NL_flag, ...] = cov_WL_GS_2D[NL_flag, ...] - cov_WL_GO_2D[NL_flag, ...]

    # TODO check the reshaping for SS and mix? Not quite sure why though...

    # reshape to 4D
    cov_WL_GO_4D[NL_flag, ...] = mm.cov_2D_to_4D(cov_WL_GO_2D[NL_flag, ...], nbl, n_zpairs, block_index='vincenzo')
    cov_WL_GS_4D[NL_flag, ...] = mm.cov_2D_to_4D(cov_WL_GS_2D[NL_flag, ...], nbl, n_zpairs, block_index='vincenzo')
    cov_WL_SS_4D[NL_flag, ...] = mm.cov_2D_to_4D(cov_WL_SS_2D[NL_flag, ...], nbl, n_zpairs, block_index='vincenzo')

    # reshape to 6D
    cov_WL_GO_6D[NL_flag, ...] = mm.cov_4D_to_6D(cov_WL_GO_4D[NL_flag, ...], nbl, zbins, probe='LL',
                                                 ind=ind_CLOE[:n_zpairs, :])
    cov_WL_GS_6D[NL_flag, ...] = mm.cov_4D_to_6D(cov_WL_GS_4D[NL_flag, ...], nbl, zbins, probe='LL',
                                                 ind=ind_CLOE[:n_zpairs, :])
    cov_WL_SS_6D[NL_flag, ...] = mm.cov_4D_to_6D(cov_WL_SS_4D[NL_flag, ...], nbl, zbins, probe='LL',
                                                 ind=ind_CLOE[:n_zpairs, :])

    # SS off-diagonal, GO on-diagonal. We call it 'mix'
    # ! note that I'm using the 4D covariance to build the "mix", I put this for loop after the reshaping
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            if ell1 == ell2:
                cov_WL_mix_4D[NL_flag, ell1, ell2, :, :] = cov_WL_GO_4D[NL_flag, ell1, ell2, :, :]
            else:
                cov_WL_mix_4D[NL_flag, ell1, ell2, :, :] = cov_WL_GS_4D[NL_flag, ell1, ell2, :, :]

    # reshape mix
    cov_WL_mix_2D[NL_flag, ...] = mm.cov_4D_to_2D(cov_WL_mix_4D[NL_flag, ...], nbl, n_zpairs, block_index='vincenzo')
    cov_WL_mix_6D[NL_flag, ...] = mm.cov_4D_to_6D(cov_WL_mix_4D[NL_flag, ...], nbl, zbins,
                                                  probe='LL', ind=ind_CLOE[:n_zpairs, :])


# ✅ test passed: cov_WL_mix_2D, 4D, 6D look well-defined indeed
# 🚧 test: do the 4 above matrices qualify as covariance matrices?

# from https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


A = np.arange(1, 5, 1).reshape((2, 2))
for NL_flag in range(NL_flags):
    print(is_pos_def(cov_WL_GO_2D[NL_flag, ...]))
    print(is_pos_def(cov_WL_GS_2D[NL_flag, ...]))
    print(is_pos_def(cov_WL_SS_2D[NL_flag, ...]))
    print(is_pos_def(cov_WL_mix_2D[NL_flag, ...]))

# 🚧 end test


mm.pycharm_exit()

# compute 'analytical' sigmas
sigma_WL_GO = np.zeros((NL_flags, nbl, iz_values.size))
sigma_WL_GS = np.copy(sigma_WL_GO)
sigma_WL_SS = np.copy(sigma_WL_GO)
sigma_WL_mix = np.copy(sigma_WL_GO)

for NL_flag in range(NL_flags):
    for iz_idx, iz in enumerate(iz_values):
        for ell_idx, _ in enumerate(ell_values):
            sigma_WL_GO[NL_flag, ell_idx, iz_idx] = np.sqrt(
                cov_WL_GO_6D[NL_flag, ell_idx, ell_idx, iz - 1, iz - 1, iz - 1, iz - 1])
            sigma_WL_GS[NL_flag, ell_idx, iz_idx] = np.sqrt(
                cov_WL_GS_6D[NL_flag, ell_idx, ell_idx, iz - 1, iz - 1, iz - 1, iz - 1])
            sigma_WL_SS[NL_flag, ell_idx, iz_idx] = np.sqrt(
                cov_WL_SS_6D[NL_flag, ell_idx, ell_idx, iz - 1, iz - 1, iz - 1, iz - 1])
            sigma_WL_mix[NL_flag, ell_idx, iz_idx] = np.sqrt(
                cov_WL_mix_6D[NL_flag, ell_idx, ell_idx, iz - 1, iz - 1, iz - 1, iz - 1])

# test plot with error bars - this is one of the 4 panels shown in the multi-panel plot below
# ! careful of the difference between iz and iz_idx
"""
iz = 10
iz_idx = np.where(iz_values == iz)[0][0]
NL_flag = 0 # thiese are the uncert I show in the 4-panel plot
ClWL_values = [ClWL_3D[NL_flag, cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ell_values)]

relative_uncert_GO = sigma_WL_GO[NL_flag, :, iz_idx]/ClWL_values

# # Visualize the result
plt.figure()
plt.errorbar(ell_values, ClWL_values, sigma_WL_GS[NL_flag, :, iz_idx], c='red',  label = 'GS')
plt.errorbar(ell_values, ClWL_values, sigma_WL_GO[NL_flag, :, iz_idx], fmt='none', alpha=0.6, c='blue', label = 'GO')
# plt.fill_between(ell_values, ClWL_values - sigma_WL_GS[NL_flag, :, iz_idx], ClWL_values + sigma_WL_GS[NL_flag, :, iz_idx],
#                   color='gray', alpha=0.2)
plt.yscale('log')
plt.xscale('log')
plt.legend()
"""

#######################################################################################
########################## sample from multivariate Gaussian ##########################
#######################################################################################

# take only the correct elements for each iz (for the moment just one iz is implemented)
# unpack the iz indices:
# FIXME this would be implemented in a better way, (not strictly necessary)
# for iz in iz_values:
#     column_1 = np.where(ind_CLOE_LL[:, 0] == iz-1)[0]
#     column_2 = np.where(ind_CLOE_LL[:, 1] == iz-1)[0]
#     print(iz, column_1, column_2)

# store the resulting zpairs
iz_expanded = np.asarray([0, 19, 34, 54])

# check to make sure the zpairs correspond to the iz_values given
assert np.all(
    iz_values == [1, 3, 5, 10]), 'the zpairs indices corresponding to the iz chosen are hardcoded, check your iz values'

# initialize random number generator
rng = np.random.default_rng()

# TODO is "flatten" the right ordering? ✅ yes, should be 
# TODO take the correct 20 elements from the 1100 outputted by the routine below
# TODO still unsure about both the unpacking and if the elements im pulling out of the datavectors are the correct ones


n_samples = 5_000

if n_samples != 10_000:
    print('\nn_samples must be 10_000, just speeding up the code, FIXME')

GO_samples = np.zeros((NL_flags, n_samples, nbl * n_zpairs))
GS_samples = np.copy(GO_samples)
SS_samples = np.copy(GO_samples)
mix_samples = np.copy(GO_samples)

GO_samples_thinned_2D = np.zeros((NL_flags, nbl, iz_values.size, n_samples))
GS_samples_thinned_2D = np.copy(GO_samples_thinned_2D)
SS_samples_thinned_2D = np.copy(GO_samples_thinned_2D)
mix_samples_thinned_2D = np.copy(GO_samples_thinned_2D)

GO_samples_in_range = np.copy(GO_samples_thinned_2D)
GS_samples_in_range = np.copy(GO_samples_thinned_2D)
SS_samples_in_range = np.copy(GO_samples_thinned_2D)
mix_samples_in_range = np.copy(GO_samples_thinned_2D)

# new
sigma_WL_GO_sampled_thinned_2D = np.zeros((NL_flags, nbl, iz_values.size))
sigma_WL_GS_sampled_thinned_2D = np.copy(sigma_WL_GO_sampled_thinned_2D)
sigma_WL_SS_sampled_thinned_2D = np.copy(sigma_WL_GO_sampled_thinned_2D)
sigma_WL_mix_sampled_thinned_2D = np.copy(sigma_WL_GO_sampled_thinned_2D)

GO_samples_2D = np.zeros((NL_flags, nbl, n_zpairs, n_samples))
GS_samples_2D = np.copy(GO_samples_2D)
SS_samples_2D = np.copy(GO_samples_2D)
mix_samples_2D = np.copy(GO_samples_2D)

# store the samples in the analytical confidence interval
GO_in_conf_interval = []
GS_in_conf_interval = []

# sampled covariance, check that it's equal to the analytical covariance (suggested by Martin)
cov_sampled_GO_2D = np.zeros((NL_flags, cov_WL_GO_2D.shape[1], cov_WL_GO_2D.shape[2]))
cov_sampled_GS_2D = np.copy(cov_sampled_GO_2D)

# fill the sigma arrays from the multivariate Gaussian
for NL_flag in range(NL_flags):
    # to generate the multivariate Gaussian, flatten the "mean" (ClWL_1D)
    # the inverse transformation is just a 'reshape'
    ClWL_1D = ClWL_2D[NL_flag, ...].flatten()  # check this flatten ✅

    # draw samples from the multivariate Gaussian with the various covariance matrices
    GO_samples[NL_flag, :, :] = rng.multivariate_normal(mean=ClWL_1D, cov=cov_WL_GO_2D[NL_flag, :, :], size=n_samples)
    GS_samples[NL_flag, :, :] = rng.multivariate_normal(mean=ClWL_1D, cov=cov_WL_GS_2D[NL_flag, :, :], size=n_samples)
    SS_samples[NL_flag, :, :] = rng.multivariate_normal(mean=ClWL_1D, cov=cov_WL_SS_2D[NL_flag, :, :], size=n_samples)
    mix_samples[NL_flag, :, :] = rng.multivariate_normal(mean=ClWL_1D, cov=cov_WL_mix_2D[NL_flag, :, :], size=n_samples)

# compute the variance and covariance before reshaping, on the axis corresponding to n_samples
# nicer to do it out of the NL_flag for loop
sigma_WL_GO_sampled = np.sqrt(np.var(GO_samples, axis=1))
sigma_WL_GS_sampled = np.sqrt(np.var(GS_samples, axis=1))
sigma_WL_SS_sampled = np.sqrt(np.var(SS_samples, axis=1))
sigma_WL_mix_sampled = np.sqrt(np.var(mix_samples, axis=1))

for NL_flag in range(NL_flags):

    # further check: compute the covariance from the samples
    cov_sampled_GO_2D[NL_flag, :, :] = np.cov(GO_samples[NL_flag, :, :].T)
    cov_sampled_GS_2D[NL_flag, :, :] = np.cov(GS_samples[NL_flag, :, :].T)

    # manually reshape the samples
    p = 0
    for ell in range(nbl):
        for ij in range(n_zpairs):
            GO_samples_2D[NL_flag, ell, ij, :] = GO_samples[NL_flag, :, p]
            GS_samples_2D[NL_flag, ell, ij, :] = GS_samples[NL_flag, :, p]
            SS_samples_2D[NL_flag, ell, ij, :] = SS_samples[NL_flag, :, p]
            mix_samples_2D[NL_flag, ell, ij, :] = mix_samples[NL_flag, :, p]
            p += 1

    for ell in range(nbl):
        for iz_idx, iz in enumerate(iz_expanded):
            # take the correct z index, i.e. thin out the chains
            # this corresponds to the old reshaping
            GO_samples_thinned_2D[NL_flag, ell, iz_idx, :] = GO_samples_2D[NL_flag, ell, iz, :]
            GS_samples_thinned_2D[NL_flag, ell, iz_idx, :] = GS_samples_2D[NL_flag, ell, iz, :]
            SS_samples_thinned_2D[NL_flag, ell, iz_idx, :] = SS_samples_2D[NL_flag, ell, iz, :]
            mix_samples_thinned_2D[NL_flag, ell, iz_idx, :] = mix_samples_2D[NL_flag, ell, iz, :]

            # same thing: keep only the 4 iz indices considered
            # compute sigma directly in 2D
            # it's axis=2 instead of axis=3 because I'm slicing the array by specifying the NL_flag
            sigma_WL_GO_sampled_thinned_2D[NL_flag, :, :] = np.sqrt(np.var(GO_samples_thinned_2D[NL_flag, :, :], axis=2))
            sigma_WL_GS_sampled_thinned_2D[NL_flag, :, :] = np.sqrt(np.var(GS_samples_thinned_2D[NL_flag, :, :], axis=2))
            sigma_WL_SS_sampled_thinned_2D[NL_flag, :, :] = np.sqrt(np.var(SS_samples_thinned_2D[NL_flag, :, :], axis=2))
            sigma_WL_mix_sampled_thinned_2D[NL_flag, :, :] = np.sqrt(np.var(mix_samples_thinned_2D[NL_flag, :, :], axis=2))

            # santiago's checks:
            # GO_samples_in_range[NL_flag, iz_idx, ell, :] =\
            # np.logical_and(GO_samples_thinned[NL_flag, iz_idx, ell, :] < ClWL_2D[NL_flag, ell, iz] + sigma_WL_GO[NL_flag, iz_idx, ell],\
            #                GO_samples_thinned[NL_flag, iz_idx, ell, :] > ClWL_2D[NL_flag, ell, iz] - sigma_WL_GO[NL_flag, iz_idx, ell])
            # GS_samples_in_range[NL_flag, iz_idx, ell, :] =\
            # np.logical_and(GS_samples_thinned[NL_flag, iz_idx, ell, :] < ClWL_2D[NL_flag, ell, iz] + sigma_WL_GS[NL_flag, iz_idx, ell],\
            #                GS_samples_thinned[NL_flag, iz_idx, ell, :] > ClWL_2D[NL_flag, ell, iz] - sigma_WL_GS[NL_flag, iz_idx, ell])

            # n_samples_in_range_GO = np.count_nonzero(GO_samples_in_range[NL_flag, iz_idx, ell, :]) / n_samples * 100
            # n_samples_in_range_GS = np.count_nonzero(GS_samples_in_range[NL_flag, iz_idx, ell, :]) / n_samples * 100

            # GO_in_conf_interval.append(n_samples_in_range_GO)
            # GS_in_conf_interval.append(n_samples_in_range_GS)

            # check how many of the samples fall in the analytical confidence interval 
            # (which is a way to check if the variances are the same)

            # rename for clarity
            # mean = ClWL_2D[NL_flag, ell, iz]
            # sigma_GO = sigma_WL_GO[NL_flag, ell, iz_idx]
            # sigma_GS = sigma_WL_GS[NL_flag, ell, iz_idx]
            #
            # # store the samples in the mean \pm sigma interval to count them
            # # (analytical sigma)
            #
            # GO_samples_in_range[NL_flag, ell, iz_idx, :] = \
            #     np.logical_and(GO_samples_thinned_2D[NL_flag, ell, iz_idx, :] < mean + sigma_GO,
            #                    GO_samples_thinned_2D[NL_flag, ell, iz_idx, :] > mean - sigma_GO)
            #
            # GS_samples_in_range[NL_flag, ell, iz_idx, :] = \
            #     np.logical_and(GS_samples_thinned_2D[NL_flag, ell, iz_idx, :] < mean + sigma_GS,
            #                    GS_samples_thinned_2D[NL_flag, ell, iz_idx, :] > mean - sigma_GS)
            #
            # n_samples_in_range_GO = np.count_nonzero(GO_samples_in_range[NL_flag, ell, iz_idx, :]) / n_samples * 100
            # n_samples_in_range_GS = np.count_nonzero(GS_samples_in_range[NL_flag, ell, iz_idx, :]) / n_samples * 100
            #
            # GO_in_conf_interval.append(n_samples_in_range_GO)
            # GS_in_conf_interval.append(n_samples_in_range_GS)

# plot how many samples fall in che analytical confidence interval
"""
plt.figure(figsize=(10, 8))              
plt.plot(range(len(GO_in_conf_interval)), GO_in_conf_interval, label="% GO_in_conf_interval")
plt.plot(range(len(GS_in_conf_interval)), GS_in_conf_interval, label="% GS_in_conf_interval")

plt.ylabel('% elements in $\mu \pm 1\sigma$')
plt.xlabel('case')
plt.legend()
"""

# TODO check the sampled covariance against the analytical one


################### begin check: corner plot


# TODO marginalize this


# I cannot make a corner plot of nbl x npairs variables (each variable is an
# element of the cl vector)
NL_flag = 1
start = 0
stop = 20
step = 3
iz_idx_test = 1
ell_test = 4

"""
figure = corner.corner(GO_samples[NL_flag, :, :stop:step],
                       quantiles=[0.16, 0.5, 0.84], label='GO_samples',
                       show_titles=True, title_kwargs={"fontsize": 12})
figure = corner.corner(GS_samples[NL_flag,:, :stop:step], fig = figure,
                       quantiles=[0.16, 0.5, 0.84], label='GS_samples',
                       show_titles=False, title_kwargs={"fontsize": 12}, color = 'red')
figure = corner.corner(mix_samples[NL_flag, :, :stop:step], fig = figure,
                        quantiles=[0.16, 0.5, 0.84], label='mix_samples',
                        show_titles=False, title_kwargs={"fontsize": 12}, color = 'red')
"""

# same thing, but with the samples array reshaped to make sure to get the right elements
# I transpose because corner's first argument must be of shape (n_samples, n_dim)

# XXX this corner plot is not very convincing...
# XXX check if samples == samples_2D

# NOTE: the comparison should be between GO and mix, GS broadens the variance as well
# figure = corner.corner(GO_samples_thinned_2D[NL_flag, start:stop:step, iz_idx_test, :].transpose(),
#                        quantiles=[0.16, 0.5, 0.84], label='GO_samples',
#                        show_titles=True, title_kwargs={"fontsize": 12})
# figure = corner.corner(GS_samples_thinned_2D[NL_flag, start:stop:step, iz_idx_test, :].transpose(), fig = figure,
#                        quantiles=[NL_flag, 0.16, 0.5, 0.84], label='GS_samples',
#                        show_titles=False, title_kwargs={"fontsize": 12}, color = 'red')
# figure = corner.corner(mix_samples_thinned_2D[NL_flag, start:stop:step, iz_idx_test, :].transpose(), fig=figure,
#                        quantiles=[0.16, 0.5, 0.84], label='mix_samples',
#                        show_titles=False, title_kwargs={"fontsize": 12}, color='red')
# figure.set_size_inches(10, 10)
# figure.legend()

# ! manually plot a hist of the diagonal plots in the corner plot

# quite useless, the one below is equally informative
# plt.figure()
# plt.hist(GO_samples_thinned_2D[NL_flag, ell_test, iz_idx_test, :], alpha=0.6, bins=40)
# plt.hist(GS_samples_thinned[NL_flag, ell_test, iz_idx_test, :], alpha=0.6, bins=40)
# plt.hist(mix_samples_thinned_2D[NL_flag, ell_test, iz_idx_test, :], alpha=0.6, bins=40)


# TODO plot gaussian on top of histogram

# plot the analytical gaussian
# rename for clarity
plt.figure()
mean = ClWL_2D[NL_flag, ell_test, iz_expanded[iz_idx_test]]
sigma_GO = sigma_WL_GO[NL_flag, ell_test, iz_idx_test]
sigma_GS = sigma_WL_GS[NL_flag, ell_test, iz_idx_test]
sigma_mix = sigma_WL_mix[NL_flag, ell_test, iz_idx_test]

sigma = sigma_GS
samples_thinned = GS_samples_thinned_2D[NL_flag, ell_test, iz_idx_test, :]

# plot Gaussian
x = np.linspace(mean - 3.5 * sigma, mean + 3.5 * sigma, 100)
plt.plot(x, stats.norm.pdf(x, mean, sigma))

# (over)plot sampled data as a histogram
plt.hist(samples_thinned, alpha=0.6, bins=40, color='red', density=True)
# plt.hist(GS_samples_thinned[NL_flag, ell_test, iz_idx_test, :], alpha=0.6, bins=40)
# plt.hist(mix_samples_thinned[NL_flag, ell_test, iz_idx_test, :], alpha=0.6, bins=40)
plt.show()

# ! compare the sample and analytical sigmas
plt.figure()
plt.title(f'NL_flag = {NL_labels[NL_flag]}, (iz, jz) = ({iz_values[iz_idx_test]}, {iz_values[iz_idx_test]})')
plt.plot(range(nbl), sigma_WL_GO[NL_flag, :, iz_idx_test], label='analytical GO')
plt.plot(range(nbl), sigma_WL_GO_sampled_thinned_2D[NL_flag, :, iz_idx_test], '--', label='sampled GO')

plt.plot(range(nbl), sigma_WL_GS[NL_flag, :, iz_idx_test], label='analytical GS')
plt.plot(range(nbl), sigma_WL_GS_sampled_thinned_2D[NL_flag, :, iz_idx_test], '--', label='sampled GS')

plt.plot(range(nbl), sigma_WL_mix[NL_flag, :, iz_idx_test], label='analytical mix')
plt.plot(range(nbl), sigma_WL_mix_sampled_thinned_2D[NL_flag, :, iz_idx_test], '--', label='sampled mix')

plt.plot(range(nbl), sigma_WL_SS[NL_flag, :, iz_idx_test], label='analytical SS')
plt.plot(range(nbl), sigma_WL_SS_sampled_thinned_2D[NL_flag, :, iz_idx_test], '--', label='sampled SS')

plt.legend()
plt.xlabel('$\\ell$ bin')
plt.ylabel('sigma_WL_GO')
plt.yscale('log')
plt.grid()

# XXX working here
# mm.pycharm_exit()

########## from stackoverflow
# fig = plt.figure(figsize=(8,8))
# ax  = fig.add_subplot(111)
# # now plot
# alpha, loc, beta = 5, 100, 22
# ax.hist(GO_samples_thinned[NL_flag, ell_test, iz_idx_test, :], 100, alpha=0.6)
# x = np.linspace(mean - 3*sigma_GO, mean + 3*sigma_GO, 100)
# ax.plot(x, stats.norm.pdf(x, mean, sigma_GO), lw=2)

# # show
# plt.show()
# sys.exit()

# ! check sampled covariance
NL_flag = 1

# plot GO_sample # 0
plt.plot(GO_samples[NL_flag, 0, :], label='GO_sample')
plt.plot(GS_samples[NL_flag, 0, :], label='GS_sample')
theory = ClWL_2D[NL_flag, ...].flatten()
mean_sampled = np.mean(GS_samples[NL_flag, :, :], axis=0)
diff = mm.percent_diff(theory, mean_sampled)
plt.plot(diff, label='ratio')

mm.compare_2D_arrays(cov_WL_GO_2D[NL_flag, ...], cov_sampled_GO_2D[NL_flag, ...], 'analytical', 'sampled', True, True)
mm.compare_2D_arrays(cov_WL_GS_2D[NL_flag, ...], cov_sampled_GS_2D[NL_flag, ...], 'analytical', 'sampled', True, True)

sampled = cov_sampled_GO_2D[NL_flag, ...][-55:, -55:]
analytical = cov_WL_GO_2D[NL_flag, ...][-55:, -55:]

mm.matshow(cov_sampled_GO_2D[NL_flag, ...][:55, :55])

mm.matshow(sampled, log=True, title='sampled')
mm.matshow(analytical, log=True, title='analytical')
diff = mm.percent_diff(sampled, analytical)
plt.plot(diff, title='diff')

diff = mm.percent_diff(cov_sampled_GS_2D[NL_flag, ...], cov_WL_GS_2D[NL_flag, ...])
plt.hist(diff.flatten(), bins=40)
plt.yscale('log')

mm.matshow(diff, log=True, abs_val=True)

mm.pycharm_exit()

# check the importance of GO vs GS as a function of ell
# use HaloModel as NL_flag (=1) \
"""
# plt.figure()
for zi in range(10):
    # for zj in range(zi+1):
    zj = 2
    GO_elements = np.asarray([cov_WL_GO_4D[1, ell, ell, zi, zj] for ell in range(nbl)])
    GS_elements = np.asarray([cov_WL_GS_4D[1, ell, ell, zi, zj] for ell in range(nbl)])
    SS_elements = np.asarray([cov_WL_SS_4D[1, ell, ell, zi, zj] for ell in range(nbl)])
    
    plt.plot(ell_values, np.abs(SS_elements/GO_elements), '.-', lw=3, label=f'$(z_i, z_j)$ = ({zi}, {zj})')
    
        
plt.axhline(y=1, ls='--', label='$Cov_{SS}/Cov_{GO}=1$', lw=3)
plt.xlabel('$\\ell$')
plt.ylabel('$Cov_{SS}/Cov_{GO}$')
plt.legend()

sys.exit()
"""
############################ end this check


# end huge check

# samples within confidence interval
"""
cur_ell = 0
iz_idx = 0
zpair = iz_expanded[iz_idx]
# santiago's checks:
GO_samples_in_range[NL_flag, iz_idx, cur_ell, :] = \
    np.logical_and(GO_samples_thinned_2D[NL_flag, iz_idx, cur_ell, :] < ClWL_1D[cur_ell * n_zpairs + zpair] + sigma_WL_GO[
        NL_flag, iz_idx, cur_ell], \
                   GO_samples_thinned_2D[NL_flag, iz_idx, cur_ell, :] > ClWL_1D[cur_ell * n_zpairs + zpair] - sigma_WL_GO[
                       NL_flag, iz_idx, cur_ell])
# )
#     GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] < ClWL_1D[cur_ell*n_zpairs + zpair] + sigma_WL_GS[NL_flag, iz_idx, cur_ell] and \
#     GO_samples_thinned[NL_flag, iz_idx, cur_ell, :] > ClWL_1D[cur_ell*n_zpairs + zpair] - sigma_WL_GS[NL_flag, iz_idx, cur_ell]

# a = np.asarray((4, 5, 7, 9, 10, 11, 12, 13, 14, 15))
print(np.count_nonzero(GO_samples_in_range[NL_flag, iz_idx, cur_ell, :]) / n_samples * 100, \
      '% of elements fall within confidence interval')
# print(np.count_nonzero((9 < a) and (a < 14)))
"""

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
# for NL_flag in range(NL_flags):
#     mm.compare_2D_arrays(sigma_WL_GS_sampled[NL_flag, ...], sigma_WL_GS[NL_flag, ...])
# # plt.figure()
# plt.hist(GO_samples_thinned[NL_flag, iz_idx, cur_ell, :], bins = 40)


""" THIS IS THE ERRORBAR PLOT """

params = {'lines.linewidth': 1.,
          'font.size': 7,
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (10, 10)
          }
plt.rcParams.update(params)

# this is the actual plot with uncertainties
plot_uncertainties = True
which_uncertainties = 'analytical'
# which_uncertainties = 'sampled'
# which_uncertainties = 'analytical_relative'

# plot_linear = True
plot_linear = False
# which_plot = 'ratio_to_halofit'
which_plot = 'absolute'

# plot or not the linear recipe
if plot_linear:
    start_idx = 0
else:
    start_idx = 1

NL_values = [0, 1, 2, 3, 4]
if which_plot == 'ratio_to_halofit':
    NL_values.remove(1)
    plot_linear = False
if not plot_linear:
    NL_values.remove(0)

# labels and (zi, zi) values
iz_dict = {1: [0, 0], 3: [0, 1], 5: [1, 0], 10: [1, 1]}

# get the labels corresponding to the NL_flag values, used as indices
# source: https://www.geeksforgeeks.org/python-accessing-all-elements-at-given-list-of-indexes/
NL_labels = ['Linear Recipe', 'Halofit', 'Mead2020', 'Euclid Emulator', 'Bacco Recipe']
NL_labels = list(itemgetter(*NL_values)(NL_labels))
linewidth = 0.65

fig, axs = plt.subplots(2, 2, figsize=(20, 16), dpi=400, linewidth=linewidth)
fig.suptitle(r'WL $C^{ij}_{\ell}$, %s' % which_plot)

# note: the asymmetry of the error bands w.r.t. the Cls is simply due to the log scale on the y axis
for iz_idx, iz in enumerate(iz_values):

    # set the correct axis
    ax1 = axs[iz_dict[iz][0], iz_dict[iz][1]]

    # pre-compute halofit, to compute the ratio
    ClWL_3D_halofit_list = np.asarray([ClWL_3D[1, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ell_values)])

    for NL_flag, pl_label in zip(NL_values, NL_labels):

        # compute Cls
        ClWL_3D_list = np.asarray([ClWL_3D[NL_flag, cur_ell, iz - 1, iz - 1] for cur_ell, _ in enumerate(ell_values)])

        # rename for better readibility
        if which_uncertainties == 'analytical':
            sigma_WL_GO_list = sigma_WL_GO[NL_flag, :, iz_idx]
            sigma_WL_GS_list = sigma_WL_GS[NL_flag, :, iz_idx]
            sigma_WL_GO_halofit_list = sigma_WL_GO[1, :, iz_idx]
            sigma_WL_GS_halofit_list = sigma_WL_GS[1, :, iz_idx]

        elif which_uncertainties == 'sampled':
            sigma_WL_GO_list = sigma_WL_GO_sampled[NL_flag, :, iz_idx]
            sigma_WL_GS_list = sigma_WL_GS_sampled[NL_flag, :, iz_idx]
            sigma_WL_GO_halofit_list = sigma_WL_GO_sampled[1, :, iz_idx]
            sigma_WL_GS_halofit_list = sigma_WL_GS_sampled[1, :, iz_idx]

        elif which_uncertainties == 'analytical_relative':
            sigma_WL_GO_list = sigma_WL_GO[NL_flag, :, iz_idx] / ClWL_3D_list
            sigma_WL_GS_list = sigma_WL_GS[NL_flag, :, iz_idx] / ClWL_3D_list
            sigma_WL_GO_halofit_list = sigma_WL_GO[1, :, iz_idx] / ClWL_3D_list
            sigma_WL_GS_halofit_list = sigma_WL_GS[1, :, iz_idx] / ClWL_3D_list
            assert which_plot == 'absolute', 'plot cannot be "ratio to halofit"'

        # TODO check this
        ClWL_3D_ratio_list = ClWL_3D_list / ClWL_3D_halofit_list

        # plot, no uncertainties
        if which_plot == 'absolute':
            ax1.loglog(ell_values, ClWL_3D_list, color=colors[NL_flag], ls='--', label=f'{pl_label}', zorder=2,
                       lw=linewidth)

        # plot the ratios to halofit
        elif which_plot == 'ratio_to_halofit':
            ax1.loglog(ell_values, ClWL_3D_ratio_list, color=colors[NL_flag], ls='--', label=f'{pl_label}', zorder=2,
                       lw=linewidth)
            # compute uncertainties on the ratio
            sigma_WL_GO_list = np.abs(ClWL_3D_ratio_list) * np.sqrt(
                (sigma_WL_GO_list / ClWL_3D_list) ** 2 + (sigma_WL_GO_halofit_list / ClWL_3D_halofit_list) ** 2)
            sigma_WL_GS_list = np.abs(ClWL_3D_ratio_list) * np.sqrt(
                (sigma_WL_GS_list / ClWL_3D_list) ** 2 + (sigma_WL_GS_halofit_list / ClWL_3D_halofit_list) ** 2)

        # plot GO and GS uncertainties 
        # I plot only one of the NL uncertainties, otherwise the plots become too many
        # XXX decide which one. At the mooment is halofit
        # XXX ajust the "if" for the label?
        if (NL_flag == 4 or NL_flag == 0) and plot_uncertainties:
            if which_plot == 'absolute':
                for (sigma, label, color) in zip([sigma_WL_GS_list, sigma_WL_GO_list], ['GaussSSC', 'Gauss-only', ],
                                                 ['blue', 'red']):
                    ax1.fill_between(ell_values,
                                     ClWL_3D_list - sigma, ClWL_3D_list + sigma,
                                     label=label if NL_flag == 4 else "",
                                     color=color, alpha=0.2, zorder=1)

            # if I'm plotting the ratio, I need to change the Cls
            elif which_plot == 'ratio_to_halofit':
                for (sigma, label, color) in zip([sigma_WL_GS_list, sigma_WL_GO_list], ['GaussSSC', 'Gauss-only', ],
                                                 ['blue', 'red']):
                    ax1.fill_between(ell_values,
                                     ClWL_3D_ratio_list - sigma, ClWL_3D_ratio_list + sigma,
                                     label=label if NL_flag == 4 else "",
                                     color=color, alpha=0.2, zorder=1)

    """ # old code
    pl_label_0 = 'Linear Recipe'
    ax1.loglog(ells, [ClWL_0_3D[cur_ell, iz-1, iz-1] for cur_ell, _ in enumerate(ells)], color=colors[1], ls='-.', label=pl_label_0)
   
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

    # original
    ax1.set_xlabel(r'$\ell$', fontsize=7)
    ax1.set_ylabel(r'$C_\ell$ $[sr^{-1}]$', fontsize=7)

    ax1.set_title('bin i,j={:d},{:d}'.format(iz, iz))
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    # original
    # ax1.legend(prop={'size': 6})# spyder
    ax1.legend(prop={'size': 3})
    ax1.grid()

plt.savefig(job_path / f'output/Cl_errorbars/{which_plot}_plot_uncer_{plot_uncertainties}_{which_uncertainties}.png',
            dpi=300)

print('done')
