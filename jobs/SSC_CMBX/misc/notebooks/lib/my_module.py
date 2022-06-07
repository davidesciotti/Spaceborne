import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from scipy.interpolate import interp1d


###############################################################################
################# USEFUL FUNCTIONS ############################################
###############################################################################


def matshow(array, title="title", log=False, abs_val=False):
    # the ordering of these is important: I want the log(abs), not abs(log)
    if abs_val:  # take the absolute value
        array = np.abs(array)
        title = 'abs ' + title
    if log:  # take the log
        array = np.log10(array)
        title = 'log10 ' + title

    plt.matshow(array)
    plt.colorbar()
    plt.title(title)




###############################################################################



###############################################################################
#################### COVARIANCE MATRIX COMPUTATION ############################ 
###############################################################################
# TODO unify these 3 into a single function
# TODO workaround for start_index, stop_index (super easy)

@jit(nopython=True)
def covariance(nbl, npairs, start_index, stop_index, Cij, noise, l_lin, delta_l, fsky, ind):
    # create covariance array
    covariance = np.zeros((nbl, nbl, npairs, npairs))
    # compute cov(ell, p, q)
    for ell in range(nbl):
        for p in range(start_index, stop_index):
            for q in range(start_index, stop_index):
                covariance[ell, ell, p - start_index, q - start_index] = \
                    ((Cij[ell, ind[p, 2], ind[q, 2]] + noise[ind[p, 0], ind[q, 0], ind[p, 2], ind[q, 2]]) *
                     (Cij[ell, ind[p, 3], ind[q, 3]] + noise[ind[p, 1], ind[q, 1], ind[p, 3], ind[q, 3]]) +
                     (Cij[ell, ind[p, 2], ind[q, 3]] + noise[ind[p, 0], ind[q, 1], ind[p, 2], ind[q, 3]]) *
                     (Cij[ell, ind[p, 3], ind[q, 2]] + noise[ind[p, 1], ind[q, 0], ind[p, 3], ind[q, 2]])) / \
                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return covariance


# @jit(nopython=True)
def covariance_WA(nbl, npairs, start_index, stop_index, Cij, noise, l_lin, delta_l, fsky, ind, ell_WA):
    covariance = np.zeros((nbl, nbl, npairs, npairs))

    for ell in range(nbl):
        for p in range(start_index, stop_index):
            for q in range(start_index, stop_index):

                if ell_WA.size == 1:  # in the case of just one bin it would give error
                    denominator = ((2 * l_lin + 1) * fsky * delta_l)
                else:
                    denominator = ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])

                covariance[ell, ell, p - start_index, q - start_index] = \
                    ((Cij[ell, ind[p, 2], ind[q, 2]] + noise[ind[p, 0], ind[q, 0], ind[p, 2], ind[q, 2]]) * (
                                Cij[ell, ind[p, 3], ind[q, 3]] + noise[ind[p, 1], ind[q, 1], ind[p, 3], ind[q, 3]]) + (
                                Cij[ell, ind[p, 2], ind[q, 3]] + noise[ind[p, 0], ind[q, 1], ind[p, 2], ind[q, 3]]) * (
                                Cij[ell, ind[p, 3], ind[q, 2]] + noise[ind[p, 1], ind[q, 0], ind[p, 3], ind[q, 2]])) \
                                / denominator
    return covariance


# covariance matrix for ALL
@jit(nopython=True)
def covariance_ALL(nbl, npairs, Cij, noise, l_lin, delta_l, fsky, ind):
    # create covariance array
    cov_GO = np.zeros((nbl, nbl, npairs, npairs))
    # compute cov(ell, p, q)
    for ell in range(nbl):
        for p in range(npairs):
            for q in range(npairs):

                # ind carries info about both the probes and the z indices!
                A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]
                i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                cov_GO[ell, ell, p, q] = \
                    ((Cij[ell, A, C, i, k] + noise[A, C, i, k]) * (Cij[ell, B, D, j, l] + noise[B, D, j, l]) +
                     (Cij[ell, A, D, i, l] + noise[A, D, i, l]) * (Cij[ell, B, C, j, k] + noise[B, C, j, k])) / \
                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return cov_GO


@jit(nopython=True)
def cov_SSC(nbl, npairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    assert probe in ['WL', 'WA', 'GC'], 'probe must be "WL", "WA" or "GC"'

    if probe in ["WL", "WA"]:
        shift = 0
    elif probe == "GC":
        shift = zbins

    cov_SSC = np.zeros((nbl, nbl, npairs, npairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs):
                for q in range(npairs):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                 Cij[ell1, i, j] * Cij[ell2, k, l] *
                                                 Sijkl[i + shift, j + shift, k + shift, l + shift])
    cov_SSC /= fsky
    return cov_SSC


@jit(nopython=True)
def build_Sijkl_dict(Sijkl, zbins):
    # build probe lookup dictionary, to set the right start and stop values
    probe_lookup = {
        'L': {
            'start': 0,
            'stop': zbins
        },
        'G': {
            'start': zbins,
            'stop': 2 * zbins
        }
    }

    # fill Sijkl dictionary
    Sijkl_dict = {}
    for probe_A in ['L', 'G']:
        for probe_B in ['L', 'G']:
            for probe_C in ['L', 'G']:
                for probe_D in ['L', 'G']:
                    Sijkl_dict[probe_A, probe_B, probe_C, probe_D] = \
                        Sijkl[probe_lookup[probe_A]['start']:probe_lookup[probe_A]['stop'],
                        probe_lookup[probe_B]['start']:probe_lookup[probe_B]['stop'],
                        probe_lookup[probe_C]['start']:probe_lookup[probe_C]['stop'],
                        probe_lookup[probe_D]['start']:probe_lookup[probe_D]['stop']]

    return Sijkl_dict


@njit
def build_D_3x2pt_dict(D_3x2pt):
    D_3x2pt_dict = {}
    D_3x2pt_dict['L', 'L'] = D_3x2pt[:, 0, 0, :, :]
    D_3x2pt_dict['L', 'G'] = D_3x2pt[:, 0, 1, :, :]
    D_3x2pt_dict['G', 'L'] = D_3x2pt[:, 1, 0, :, :]
    D_3x2pt_dict['G', 'G'] = D_3x2pt[:, 1, 1, :, :]
    return D_3x2pt_dict

@njit
def cov_SSC_3x2pt_10D_dict(nbl, D_3x2pt, Sijkl, fsky, zbins, Rl, probe_ordering):
    """Buil the 3x2pt covariance matrix using a dict for Sijkl. Slightly slower (because of the use of dicts, I think)
    but cleaner (no need for multiple if statements).
    """

    # build and/or initialize the dictionaries
    Sijkl_dict = build_Sijkl_dict(Sijkl, zbins)
    D_3x2pt_dict = build_D_3x2pt_dict(D_3x2pt)
    cov_3x2pt_SSC_10D = {}

    # compute the SS cov only for the relevant probe combinations
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_3x2pt_SSC_10D[A, B, C, D] = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
            for ell1 in range(nbl):
                for ell2 in range(nbl):
                    for i in range(zbins):
                        for j in range(zbins):
                            for k in range(zbins):
                                for l in range(zbins):
                                    cov_3x2pt_SSC_10D[A, B, C, D][ell1, ell2, i, j, k, l] = (
                                            Rl * Rl *
                                            D_3x2pt_dict[A, B][ell1, i, j] *
                                            D_3x2pt_dict[C, D][ell2, k, l] *
                                            Sijkl_dict[A, B, C, D][i, j, k, l])/fsky
            print('computing SSC in blocks: working probe combination', A, B, C, D)

    return cov_3x2pt_SSC_10D



@njit
def cov_SSC_ALL(nbl, npairs_tot, ind, D_ALL, Sijkl, fsky, zbins, Rl):
    """Not the most elegant, but fast!
    """

    cov_ALL_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):

                    # LL_LL
                    if ind[p, 0] == 0 and ind[p, 1] == 0 and ind[q, 0] == 0 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]])
                    # LL_GL
                    elif ind[p, 0] == 0 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2], ind[p, 3], ind[q, 2] + zbins, ind[q, 3]])
                    # LL_GG
                    elif ind[p, 0] == 0 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 1:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2], ind[p, 3], ind[q, 2] + zbins, ind[
                                                             q, 3] + zbins])

                    # GL_LL
                    elif ind[p, 0] == 1 and ind[p, 1] == 0 and ind[q, 0] == 0 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3], ind[q, 2], ind[q, 3]])
                    # GL_GL
                    elif ind[p, 0] == 1 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3], ind[q, 2] + zbins, ind[
                                                             q, 3]])
                    # GL_GG
                    elif ind[p, 0] == 1 and ind[p, 1] == 0 and ind[q, 0] == 1 and ind[q, 1] == 1:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3], ind[q, 2] + zbins, ind[
                                                             q, 3] + zbins])

                    # GG_LL
                    elif ind[p, 0] == 1 and ind[p, 1] == 1 and ind[q, 0] == 0 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3] + zbins, ind[q, 2], ind[
                                                             q, 3]])
                    # GG_GL
                    elif ind[p, 0] == 1 and ind[p, 1] == 1 and ind[q, 0] == 1 and ind[q, 1] == 0:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3] + zbins, ind[q, 2] + zbins,
                                                               ind[q, 3]])
                    # GG_GG
                    elif ind[p, 0] == 1 and ind[p, 1] == 1 and ind[q, 0] == 1 and ind[q, 1] == 1:
                        cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                         D_ALL[ell1, ind[p, 0], ind[p, 1], ind[p, 2], ind[p, 3]] *
                                                         D_ALL[ell2, ind[q, 0], ind[q, 1], ind[q, 2], ind[q, 3]] *
                                                         Sijkl[ind[p, 2] + zbins, ind[p, 3] + zbins, ind[q, 2] + zbins,
                                                               ind[q, 3] + zbins])
    cov_ALL_SSC /= fsky
    return cov_ALL_SSC

@njit
def cov_SSC_ALL_dict(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """Buil the 3x2pt covariance matrix using a dict for Sijkl. slightly slower (because of the use of dicts, I think)
    but cleaner (no need for multiple if statements, except to set the correct probes). Note that the ell1, ell2 slicing
    does not work! You can substitute only one of the for loops (in this case the one over ell1).
    A_str = probe A as string (e.g. 'L' for lensing)
    A_num = probe A as number (e.g. 0 for lensing)
    """

    Sijkl_dict = build_Sijkl_dict(Sijkl, zbins)

    cov_3x2pt_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):

                    if ind[p, 0] == 0:
                        A_str = 'L'
                    elif ind[p, 0] == 1:
                        A_str = 'G'
                    if ind[p, 1] == 0:
                        B_str = 'L'
                    elif ind[p, 1] == 1:
                        B_str = 'G'
                    if ind[q, 0] == 0:
                        C_str = 'L'
                    elif ind[q, 0] == 1:
                        C_str = 'G'
                    if ind[q, 1] == 0:
                        D_str = 'L'
                    elif ind[q, 1] == 1:
                        D_str = 'G'

                    A_num, B_num, C_num, D_num = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_3x2pt_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                    D_3x2pt[ell1, A_num, B_num, i, j] *
                                                    D_3x2pt[ell2, C_num, D_num, k, l] *
                                                    Sijkl_dict[A_str, B_str, C_str, D_str][i, j, k, l])
    return cov_3x2pt_SSC / fsky




def cov_3x2pt_dict_10D_to_4D(cov_3x2pt_dict_10D, probe_ordering, nbl, zbins, ind, GL_or_LG):
    """
    Takes the cov_3x2pt_10D dictionary, reshapes each A, B, C, D block separately
    in 4D, then stacks the blocks in the right order to output cov_3x2pt_4D 
    (which is not a dictionary but a numpy array)
    """

    # Check that the cross-correlation is coherent with the probe_ordering list
    # this is a weak check, since I'm assuming that GL or LG will be the second 
    # element of the datavector
    if GL_or_LG == 'GL':
        assert probe_ordering[1][0] == 'G' and probe_ordering[1][1] == 'L', \
            'probe_ordering[1] should be "GL", e.g. [LL, GL, GG]'
    elif GL_or_LG == 'LG':
        assert probe_ordering[1][0] == 'L' and probe_ordering[1][1] == 'G', \
            'probe_ordering[1] should be "LG", e.g. [LL, LG, GG]'

    # get npairs
    npairs_auto, npairs_cross, npairs_3x2pt = get_pairs(zbins)

    # construct the ind dict
    ind_dict = {}
    ind_dict['L', 'L'] = ind[:npairs_auto, :]
    ind_dict['G', 'G'] = ind[(npairs_auto + npairs_cross):, :]
    if GL_or_LG == 'LG':
        ind_dict['L', 'G'] = ind[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['G', 'L'] = ind_dict['L', 'G'].copy()  # copy and switch columns
        ind_dict['G', 'L'][:, [2, 3]] = ind_dict['G', 'L'][:, [3, 2]]
    elif GL_or_LG == 'GL':
        ind_dict['G', 'L'] = ind[npairs_auto:(npairs_auto + npairs_cross), :]
        ind_dict['L', 'G'] = ind_dict['G', 'L'].copy()  # copy and switch columns
        ind_dict['L', 'G'][:, [2, 3]] = ind_dict['L', 'G'][:, [3, 2]]

    # construct the npairs dict 
    npairs_dict = {}
    npairs_dict['L', 'L'] = npairs_auto
    npairs_dict['L', 'G'] = npairs_cross
    npairs_dict['G', 'L'] = npairs_cross
    npairs_dict['G', 'G'] = npairs_auto

    # initialize the 4D dictionary and list of probe combinations
    cov_3x2pt_dict_4D = {}
    combinations = []

    # make each block 4D and store it with the right 'A', 'B', 'C, 'D' key 
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            combinations.append([A, B, C, D])
            cov_3x2pt_dict_4D[A, B, C, D] = cov_6D_to_4D_blocks(cov_3x2pt_dict_10D[A, B, C, D], nbl, npairs_dict[A, B],
                                                                npairs_dict[C, D], ind_dict[A, B], ind_dict[C, D])

    # take the correct combinations (stored in 'combinations') and construct
    # lists which will be converted to arrays
    row_1_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[:3]]
    row_2_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[3:6]]
    row_3_list = [cov_3x2pt_dict_4D[A, B, C, D] for A, B, C, D in combinations[6:9]]

    # concatenate the lists to make rows
    row_1 = np.concatenate(row_1_list, axis=3)
    row_2 = np.concatenate(row_2_list, axis=3)
    row_3 = np.concatenate(row_3_list, axis=3)

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4D = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_3x2pt_4D


# @jit(nopython=True) # XXX this function is new - still to be thouroughly tested
def cov_4D_to_6D(cov_4D, nbl, zbins, probe, ind):
    """transform the cov from shape (nbl, nbl, npairs, npairs) 
    to (nbl, nbl, zbins, zbins, zbins, zbins). Not valid for 3x2pt, the total
    shape of the matrix is (nbl, nbl, zbins, zbins, zbins, zbins), not big 
    enough to store 3 probes. Use cov_4D functions or cov_6D as a dictionary
    instead,
    """

    assert probe in ['LL', 'GG', 'LG', 'GL'], 'probe must be "LL", "LG", "GL" or "GG". 3x2pt is not supported'

    npairs_auto, npairs_cross, npairs_tot = get_pairs(zbins)
    if probe in ['LL', 'GG']:
        npairs = npairs_auto
    elif probe in ['GL', 'LG']:
        npairs = npairs_cross

    assert ind.shape[0] == npairs, 'ind.shape[0] != npairs'

    # TODO use jit 
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ij in range(npairs):
        for kl in range(npairs):
            # rename for better readability
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3]
            # reshape
            cov_6D[:, :, i, j, k, l] = cov_4D[:, :, ij, kl]

    # GL is not symmetric
    if probe == 'LL' or probe == 'GG':
        cov_6D = symmetrize_ij(cov_6D, zbins=10)

    return cov_6D


@jit(nopython=True)
def cov_6D_to_4D(cov_6D, nbl, npairs, ind):
    """transform the cov from shape (nbl, nbl, zbins, zbins, zbins, zbins) 
    to (nbl, nbl, npairs, npairs)"""
    assert ind.shape[0] == npairs, 'ind.shape[0] != npairs'
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ij in range(npairs):
        for kl in range(npairs):
            # rename for better readability
            i, j, k, l = ind[ij, 2], ind[ij, 3], ind[kl, 2], ind[kl, 3]
            cov_4D[:, :, ij, kl] = cov_6D[:, :, i, j, k, l]
    return cov_4D





@jit(nopython=True)
def cov_4D_to_2D(cov_4D, nbl, npairs_AB, npairs_CD=None, block_index='vincenzo'):
    """ new (more elegant) version of cov_4D_to_2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering; it is sufficient to pass a npairs_CD != npairs_AB value (by default npairs_CD == npairs_AB).
    """
    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    # if not passed, npairs_CD must be equal to npairs_AB
    if npairs_CD is None:
        npairs_CD = npairs_AB

    cov_2D = np.zeros((nbl * npairs_AB, nbl * npairs_CD))

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(npairs_AB):
                    for jpair in range(npairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * npairs_AB + ipair, l2 * npairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(npairs_AB):
                    for jpair in range(npairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]
    return cov_2D



def Recast_Sijkl_3x2pt(Sijkl, nzbins):
    npairs_auto = (nzbins * (nzbins + 1)) // 2
    npairs_full = nzbins * nzbins + 2 * npairs_auto
    pairs_full = np.zeros((2, npairs_full), dtype=int)
    count = 0
    for ibin in range(nzbins):
        for jbin in range(ibin, nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    for ibin in range(nzbins, nzbins * 2):
        for jbin in range(nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    for ibin in range(nzbins, 2 * nzbins):
        for jbin in range(ibin, 2 * nzbins):
            pairs_full[0, count] = ibin
            pairs_full[1, count] = jbin
            count += 1
    Sijkl_recast = np.zeros((npairs_full, npairs_full))
    for ipair in range(npairs_full):
        ibin = pairs_full[0, ipair]
        jbin = pairs_full[1, ipair]
        for jpair in range(npairs_full):
            kbin = pairs_full[0, jpair]
            lbin = pairs_full[1, jpair]
            Sijkl_recast[ipair, jpair] = Sijkl[ibin, jbin, kbin, lbin]
    return [Sijkl_recast, npairs_full, pairs_full]


