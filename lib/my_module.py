import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from scipy.interpolate import interp1d


###############################################################################
################# USEFUL FUNCTIONS ############################################
###############################################################################

def pycharm_exit():
    assert 1 > 2, 'aborting execution'


# @jit(nopython = True)
def percent_diff(array_1, array_2):
    diff = (array_1 / array_2 - 1) * 100
    return diff


def percent_diff_mean(array_1, array_2):
    """
    result is in "percent" units
    """
    mean = (array_1 + array_2) / 2.0
    diff = (array_1 / mean - 1) * 100
    return diff


def percent_diff_nan(array_1, array_2, eraseNaN=True):
    if eraseNaN:
        diff = np.where(array_1 == array_2, 0, percent_diff(array_1, array_2))
    else:
        diff = percent_diff(array_1, array_2)
    return diff


def diff_threshold_check(diff, threshold):
    boolean = np.any(np.abs(diff) > threshold)
    print(f"has any element of the arrays a disagreement > {threshold}%? ", boolean)


def compare_2D_arrays(A, B, name_A='A', name_B='B', log_arr=False, log_diff=False, abs_val=False):
    # FIXME namestr for slices!
    # matshow(A, namestr(A, globals()), log_arr, abs_val)
    # matshow(B, namestr(B, globals()), log_arr, abs_val)

    # fixme better to plot them side by side, also the diffs
    matshow(A, name_A, log_arr, abs_val)
    matshow(B, name_B, log_arr, abs_val)

    diff_AB = percent_diff(A, B)
    diff_BA = percent_diff(B, A)
    matshow(diff_AB, '(A/B - 1) * 100', log_diff, abs_val)
    matshow(diff_BA, '(B/A - 1) * 100', log_diff, abs_val)


def namestr(obj, namespace):
    """ does not work with slices!!! (why?)"""
    return [name for name in namespace if namespace[name] is obj][0]


def plot_FM(array, style=".-"):
    name = namestr(array, globals())
    plt.plot(range(7), array, style, label=name)


def uncertainties_FM(FM):
    '''
    returns relative (not percentage!) error
    '''
    fid = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.55, 1, 1)
    # fidmn = (0.32, 0.05, 1, 1, 0.67, 0.96, 0.816, 0.06, 0.55, 1) # with massive neutrinos
    FM_inv = np.linalg.inv(FM)
    sigma_FM = np.zeros(10)
    for i in range(10):
        sigma_FM[i] = np.sqrt(FM_inv[i, i]) / fid[i]
    return sigma_FM



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


# load txt or dat files in dictionary
def get_kv_pairs(path_import, filetype="dat"):
    '''
    to use it, wrap it in "dict(), e.g.:
        loaded_dict = dict(get_kv_pairs(path_import, filetype="dat"))
    '''
    for path in Path(path_import).glob(f"*.{filetype}"):
        yield path.stem, np.genfromtxt(str(path))


# load npy files in dictionary
def get_kv_pairs_npy(path_import):
    for path in Path(path_import).glob("*.npy"):
        yield path.stem, np.load(str(path))


# to display the names (keys) more tidily
def show_keys(arrays_dict):
    for key in arrays_dict:
        print(key)


def Cl_interpolator(npairs, Cl_import, ell_values, nbl):
    Cl_interpolated = np.zeros((nbl, npairs))
    for j in range(npairs):
        f = interp1d(Cl_import[:, 0], Cl_import[:, j + 1], kind='linear')
        Cl_interpolated[:, j] = f(ell_values)
    return Cl_interpolated


def Cl_interpolator_no_1st_column(npairs, Cl_import, original_ell_values, new_ell_values, nbl):
    Cl_interpolated = np.zeros((nbl, npairs))
    for j in range(npairs):
        f = interp1d(original_ell_values, Cl_import[:, j], kind='linear')
        Cl_interpolated[:, j] = f(new_ell_values)
    return Cl_interpolated


def Cl_2D_to_3D_symmetric_old(nbl, zbins, Cl_2D):
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        k = 0
        for i in range(zbins):
            for j in range(i, zbins):
                Cl_3D[ell, i, j] = Cl_2D[ell, k]
                k += 1
    return Cl_3D


@jit(nopython=True)
def symmetrize_Cl(Cl, nbl, zbins):
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                Cl[ell, j, i] = Cl[ell, i, j]
    return Cl


######## CHECK FOR DUPLICATES
def Cl_2D_to_3D_symmetric(Cl_2D, nbl, npairs, zbins=10):
    """ reshape from (nbl, npairs) to (nbl, zbins, zbins) according to 
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for i in range(npairs):
            Cl_3D[ell, triu_idx[0][i], triu_idx[1][i]] = Cl_2D[ell, i]
    # fill lower diagonal (the matrix is symmetric!)
    Cl_3D = fill_3D_symmetric_array(Cl_3D, nbl, zbins)
    return Cl_3D


def Cl_3D_to_2D_symmetric(Cl_3D, nbl, npairs, zbins=10):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs)  according to 
    upper traigular ordering 0-th rows filled first, then second from i to 10...
    """
    triu_idx = np.triu_indices(zbins)
    Cl_2D = np.zeros((nbl, npairs))
    for ell in range(nbl):
        for i in range(npairs):
            Cl_2D[ell, i] = Cl_3D[ell, triu_idx[0][i], triu_idx[1][i]]
    return Cl_2D


def Cl_2D_to_3D_asymmetric(Cl_2D, nbl, npairs, zbins=10):
    """ reshape from (nbl, npairs) to (nbl, zbins, zbins), rows first 
    (valid for asymmetric Cij, i.e. C_XC)
    """
    Cl_3D = np.zeros((nbl, zbins, zbins))
    Cl_3D = np.reshape(Cl_2D, Cl_3D.shape)
    return Cl_3D


def Cl_3D_to_2D_asymmetric(Cl_3D, nbl, npairs, zbins=10):
    """ reshape from (nbl, zbins, zbins) to (nbl, npairs), rows first 
    (valid for asymmetric Cij, i.e. C_XC)
    """
    Cl_2D = np.zeros((nbl, npairs))
    Cl_2D = np.reshape(Cl_3D, Cl_2D.shape)
    return Cl_2D


# XXX NEW AND CORRECTED FUNCTIONS TO MAKE THE Cl 3D
###############################################################################
def array_2D_to_3D_ind(array_2D, nbl, zbins, ind, start=0, stop=55):
    """ unpack according to "ind" ordering the same as the Cl!! """
    print('attention, assuming npairs = 55 (that is, zbins = 10)!')
    array_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        for k, p in enumerate(range(start, stop)):
            array_3D[ell, ind[p, 2], ind[p, 3]] = array_2D[ell, k]
            # enumerate is in case p deosn't start from p, that is, for LG
    return array_3D


def fill_3D_symmetric_array(array_3D, nbl, zbins):
    """ mirror the lower/upper triangle """
    assert array_3D.shape == (nbl, zbins, zbins), 'shape of input array must be (nbl, zbins, zbins)'

    array_diag_3D = np.zeros((nbl, zbins, zbins))
    for ell in range(nbl):
        array_diag_3D[ell, :, :] = np.diag(np.diagonal(array_3D, 0, 1, 2)[ell, :])
    array_3D = array_3D + np.transpose(array_3D, (0, 2, 1)) - array_diag_3D

    return array_3D


def array_2D_to_1D_ind(array_2D, npairs, ind):
    """ unpack according to "ind" ordering, same as for the Cls """
    assert ind.shape[0] == npairs, 'ind must have lenght npairs'

    array_1D = np.zeros(npairs)
    for p in range(npairs):
        i, j = ind[p, 2], ind[p, 3]
        array_1D[p] = array_2D[i, j]
    return array_1D


###############################################################################


############### FISHER MATRIX ################################


# interpolator for FM
# XXXX todo
def interpolator(dC_interpolated_dict, dC_dict, probe_code, params_names, nbl, npairs, ell_values, suffix):
    for param in params_names:  # loop for each parameter
        # pick array to interpolate
        dC_to_interpolate = dC_dict[f"dCij{probe_code}d{param}-{suffix}"]
        dC_interpolated = np.zeros((nbl, npairs))  # initialize interpolated array

        # now interpolate 
        for j in range(npairs):
            f = interp1d(dC_to_interpolate[:, 0], dC_to_interpolate[:, j + 1], kind='linear')
            dC_interpolated[:, j] = f(ell_values)  # fill j-th column
            dC_interpolated_dict[
                f"dCij{probe_code}d{param}-{suffix}"] = dC_interpolated  # store array in the dictionary

    return dC_interpolated_dict


# @jit(nopython=True)
def fill_dC_array(params_names, dC_interpolated_dict, probe_code, dC, suffix):
    for (counter, param) in enumerate(params_names):
        dC[:, :, counter] = dC_interpolated_dict[f"dCij{probe_code}d{param}-{suffix}"]
    return dC


def fill_datavector_4D(nParams, nbl, npairs, zbins, ind, dC_4D):
    # XXX pairs_tot
    D_4D = np.zeros((nbl, zbins, zbins, nParams))

    for alf in range(nParams):
        for elle in range(nbl):
            for p in range(npairs):
                if ind[p, 0] == 0 and ind[p, 1] == 0:
                    D_4D[elle, ind[p, 2], ind[p, 3], alf] = dC_4D[elle, ind[p, 2], ind[p, 3], alf]
    return D_4D


@jit(nopython=True)
def datavector_4D_to_3D(D_4D, ind, nParams, nbl, npairs):
    D_3D = np.zeros((nbl, npairs, nParams))
    for alf in range(nParams):
        for elle in range(nbl):
            for p in range(npairs):
                D_3D[elle, p, alf] = D_4D[elle, ind[p, 2], ind[p, 3], alf]
    return D_3D


@jit(nopython=True)
def datavector_3D_to_2D(D_3D, ind, nParams, nbl, npairs):
    D_2D = np.zeros((npairs * nbl, nParams))
    for alf in range(nParams):
        count = 0
        for elle in range(nbl):
            for p in range(npairs):
                D_2D[count, alf] = D_3D[elle, p, alf]
                count = count + 1
    return D_2D


# @jit(nopython=True)
def compute_FM_3D(nbl, npairs, nParams, cov_inv, D_3D):
    """ Compute FM using 3D datavector - 2D + the cosmological parameter axis - and 3D covariance matrix (working but
    deprecated in favor of compute_FM_2D)"""
    b = np.zeros((nbl, npairs, nParams))
    FM = np.zeros((nParams, nParams))
    for alf in range(nParams):
        for bet in range(nParams):
            for elle in range(nbl):
                b[elle, :, bet] = cov_inv[elle, :, :] @ D_3D[elle, :, bet]
                FM[alf, bet] = FM[alf, bet] + (D_3D[elle, :, alf] @ b[elle, :, bet])
    return FM


# @jit(nopython=True)
def compute_FM_2D(nbl, npairs, nParams, cov_2D_inv, D_2D):
    """ Compute FM using 2D datavector - 1D + the cosmological parameter axis - and 2D covariance matrix"""
    b = np.zeros((nbl * npairs, nParams))
    FM = np.zeros((nParams, nParams))
    for alf in range(nParams):
        for bet in range(nParams):
            b[:, bet] = cov_2D_inv[:, :] @ D_2D[:, bet]
            FM[alf, bet] = D_2D[:, alf] @ b[:, bet]
    return FM


def compute_FoM(FM):
    print('rows/cols 2 and 3 for w0, wa')
    cov_param = np.linalg.inv(FM)
    cov_param_reduced = cov_param[2:4, 2:4]
    FM_reduced = np.linalg.inv(cov_param_reduced)
    FoM = np.sqrt(np.linalg.det(FM_reduced))
    return FoM


def get_ind_file(path, ind_ordering, which_forecast):
    if ind_ordering == 'vincenzo' or which_forecast == 'sylvain':
        ind = np.genfromtxt(path.parent / "common_data/indici.dat").astype(int);
        ind = ind - 1
    elif ind_ordering == 'CLOE':
        ind = np.genfromtxt(path.parent / "common_data/indici_luca/indici_cloe_like.dat").astype(int);
        ind = ind - 1
    elif ind_ordering == 'SEYFERT':
        ind = np.genfromtxt(path.parent / "common_data/indici_luca/indici_seyfert_like.dat").astype(int);
        ind = ind - 1
    return ind


def get_output_folder(ind_ordering, which_forecast):
    if which_forecast == 'IST':
        if ind_ordering == 'vincenzo':
            output_folder = 'ISTspecs_indVincenzo'
        elif ind_ordering == 'CLOE':
            output_folder = 'ISTspecs_indCLOE'
        elif ind_ordering == 'SEYFERT':
            output_folder = 'ISTspecs_indSEYFERT'
    elif which_forecast == 'sylvain':
        output_folder = 'common_ell_and_deltas'
    return output_folder


def get_pairs(zbins):
    npairs_auto = int((zbins * (zbins + 1)) / 2)  # = 55 for zbins = 10, cast it as int
    npairs_cross = zbins ** 2
    npairs_3x2pt = 2 * npairs_auto + npairs_cross
    return npairs_auto, npairs_cross, npairs_3x2pt


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
def cov_SSC_old(nbl, npairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    if probe in ["WL", "WA"]:
        shift = 0
    elif probe == "GC":
        shift = zbins
    else:
        raise ValueError('probe must be "WL", "WA" or "GC"')

    cov_SSC = np.zeros((nbl, nbl, npairs, npairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs):
                for q in range(npairs):
                    i, j = ind[p, 2], ind[p, 3]
                    k, l = ind[q, 2], ind[q, 3]

                    cov_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                 Cij[ell1, i, j] * Cij[ell2, k, l] *
                                                 Sijkl[i + shift, j + shift, k + shift, l + shift])
    cov_SSC /= fsky
    return cov_SSC


@jit(nopython=True)
def cov_SSC(nbl, npairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    if probe in ["WL", "WA"]:
        shift = 0
    elif probe == "GC":
        shift = zbins
    else:
        raise ValueError('probe must be "WL", "WA" or "GC"')

    cov_SSC = np.zeros((nbl, nbl, npairs, npairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs):
                for q in range(npairs):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]

                    cov_SSC[ell1, ell2, p, q] = (Rl[ell1, i, j] * Rl[ell2, k, l] *
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
    but much cleaner (no need for multiple if statements).
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
                                                                                                    D_3x2pt_dict[A, B][
                                                                                                        ell1, i, j] *
                                                                                                    D_3x2pt_dict[C, D][
                                                                                                        ell2, k, l] *
                                                                                                    Sijkl_dict[
                                                                                                        A, B, C, D][
                                                                                                        i, j, k, l]) / fsky
            print('computing SSC in blocks: working probe combination', A, B, C, D)

    return cov_3x2pt_SSC_10D






@njit
def cov_SSC_ALL(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """The fastest routine to compute the SSC covariance matrix.
    """

    cov_ALL_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]
                    A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]

                    # the shift is implemented by multiplying A, B, C, D by zbins: if lensing, probe == 0 and shift = 0
                    # if probe is GC, probe == 1 and shift = zbins. this does not hold if you switch probe indices!
                    cov_ALL_SSC[ell1, ell2, p, q] = (Rl[ell1, A, B, i, j] *
                                                     Rl[ell2, C, D, k, l] *
                                                     D_3x2pt[ell1, A, B, i, j] *
                                                     D_3x2pt[ell2, C, D, k, l] *
                                                     Sijkl[i + A * zbins, j + B * zbins, k + C * zbins, l + D * zbins])

    cov_ALL_SSC /= fsky
    return cov_ALL_SSC


@njit
def cov_SSC_ALL_dict(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """Buil the 3x2pt covariance matrix using a dict for Sijkl. slightly slower (because of the use of dicts, I think)
    but cleaner (no need for multiple if statements, except to set the correct probes).
    Note that the ell1, ell2 slicing does not work! You can substitute only one of the for loops (in this case the one over ell1).
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


def covariance_10D_dict(cl_dict, noise_dict, nbl, zbins, l_lin, delta_l, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically. 
    This one works with dictionaries, in particular for the cls and noise arrays. 
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.
    
    This version is faster, it it a wrapper function for covariance_6D_blocks,
    which makes use of jit
    """

    cov_10D_dict = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            cov_10D_dict[A, B, C, D] = covariance_6D_blocks(
                cl_dict[A, C], cl_dict[B, D], cl_dict[A, D], cl_dict[B, C],
                noise_dict[A, C], noise_dict[B, D], noise_dict[A, D], noise_dict[B, C],
                nbl, zbins, l_lin, delta_l, fsky)
    return cov_10D_dict


# This function does mix the indices, but not automatically: it only indicates which ones to use and where
# It can be used for the individual blocks of the 3x2pt (unlike the one above),
# but it has to be called once for each block combination (see cov_blocks_LG_4D
# and cov_blocks_GL_4D)
# best used in combination with cov_10D_dictionary
@jit(nopython=True)
def covariance_6D_blocks(C_AC, C_BD, C_AD, C_BC, N_AC, N_BD, N_AD, N_BC, nbl, zbins, l_lin, delta_l, fsky):
    # create covariance array
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                for k in range(zbins):
                    for l in range(zbins):
                        cov_6D[ell, ell, i, j, k, l] = \
                            ((C_AC[ell, i, k] + N_AC[i, k]) *
                             (C_BD[ell, j, l] + N_BD[j, l]) +
                             (C_AD[ell, i, l] + N_AD[i, l]) *
                             (C_BC[ell, j, k] + N_BC[j, k])) / \
                            ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
    return cov_6D


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


@jit(nopython=True)
def symmetrize_ij(cov_6D, zbins=10):
    # TODO thorough check?
    for i in range(zbins):
        for j in range(zbins):
            cov_6D[:, :, i, j, :, :] = cov_6D[:, :, j, i, :, :]
            cov_6D[:, :, :, :, i, j] = cov_6D[:, :, :, :, j, i]
    return cov_6D


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
def cov_6D_to_4D_blocks(cov_6D, nbl, npairs_AB, npairs_CD, ind_AB, ind_CD):
    """ reshapes the covariance even for the non-diagonal (hence, non-square) blocks needed to build the 3x2pt.
    use npairs_AB = npairs_CD and ind_AB = ind_CD for the normal routine (valid for auto-covariance 
    LL-LL, GG-GG, GL-GL and LG-LG). n_columns is used to determine whether the ind array has 2 or 4 columns
    (if it's given in the form of a dictionary or not)
    """
    assert ind_AB.shape[0] == npairs_AB, 'ind_AB.shape[0] != npairs_AB'
    assert ind_CD.shape[0] == npairs_CD, 'ind_CD.shape[0] != npairs_CD'

    # this is to ensure compatibility with both 4-columns and 2-columns ind arrays (dictionary)
    # the penultimante element is the first index, the last one the second index (see s - 1, s - 2 below)
    n_columns_AB = ind_AB.shape[1]  # of columns: this is to understand the format of the file
    n_columns_CD = ind_CD.shape[1]

    # check
    assert n_columns_AB == n_columns_CD, 'ind_AB and ind_CD must have the same # of columns'
    # make the name shorter
    nc = n_columns_AB

    cov_4D = np.zeros((nbl, nbl, npairs_AB, npairs_CD))
    for ij in range(npairs_AB):
        for kl in range(npairs_CD):
            i, j, k, l = ind_AB[ij, nc - 2], ind_AB[ij, nc - 1], ind_CD[kl, nc - 2], ind_CD[kl, nc - 1]
            cov_4D[:, :, ij, kl] = cov_6D[:, :, i, j, k, l]
    return cov_4D


def return_combinations(A, B, C, D):
    print(f'C_{A}{C}, C_{B}{D}, C_{A}{D}, C_{B}{C}, N_{A}{C}, N_{B}{D}, N_{A}{D}, N_{B}{C}')


###########################

# check if the matrix is symmetric
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_array_equality(arr_1, arr_2):
    print(np.all(arr_1 == arr_2))


@jit(nopython=True)
# reshape from 3 to 4 dimensions
def array_3D_to_4D(cov_3D, nbl, npairs):
    print('XXX THIS FUNCTION ONLY WORKS FOR GAUSS-ONLY COVARIANCE')
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ell in range(nbl):
        for p in range(npairs):
            for q in range(npairs):
                cov_4D[ell, ell, p, q] = cov_3D[ell, p, q]
    return cov_4D


@jit(nopython=True)
def cov_2D_to_4D(cov_2D, nbl, npairs, block_index='vincenzo'):
    """ new (more elegant) version of cov_2D_to_4D. Also works for 3x2pt. The order
    of the for loops does not affect the result!
    
    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops)
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    cov_4D = np.zeros((nbl, nbl, npairs, npairs))

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(npairs):
                    for jpair in range(npairs):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[l1 * npairs + ipair, l2 * npairs + jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(npairs):
                    for jpair in range(npairs):
                        # block_index * block_size + running_index
                        cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]
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


def cov_4D_to_2DCLOE_3x2pt(cov_4D, nbl, zbins, block_index='vincenzo'):
    """
    Reshape according to the "multi-diagonal", non-square blocks 2D_CLOE ordering. Note that this is only necessary for
    the 3x2pt probe.
    TODO the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it
    TODO will work both for LG and GL)
    """

    print("the probe ordering (LL, LG/GL, GG) is hardcoded, this function won't work with other combinations (but it"
          " will work both for LG and GL) ")

    npairs_auto, npairs_cross, npairs_3x2pt = get_pairs(zbins)

    lim_1 = npairs_auto
    lim_2 = npairs_cross + npairs_auto
    lim_3 = npairs_3x2pt

    # note: I'm writing cov_LG, but there should be no issue with GL; after all, this function is not using the ind file
    cov_LL_LL = cov_4D_to_2D(cov_4D[:, :, :lim_1, :lim_1], nbl, npairs_auto, npairs_auto, block_index)
    cov_LL_LG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_1:lim_2], nbl, npairs_auto, npairs_cross, block_index)
    cov_LL_GG = cov_4D_to_2D(cov_4D[:, :, :lim_1, lim_2:lim_3], nbl, npairs_auto, npairs_auto, block_index)

    cov_LG_LL = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, :lim_1], nbl, npairs_cross, npairs_auto, block_index)
    cov_LG_LG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], nbl, npairs_cross, npairs_cross, block_index)
    cov_LG_GG = cov_4D_to_2D(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], nbl, npairs_cross, npairs_auto, block_index)

    cov_GG_LL = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, :lim_1], nbl, npairs_auto, npairs_auto, block_index)
    cov_GG_LG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], nbl, npairs_auto, npairs_cross, block_index)
    cov_GG_GG = cov_4D_to_2D(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], nbl, npairs_auto, npairs_auto, block_index)

    # make long rows and stack together
    row_1 = np.hstack((cov_LL_LL, cov_LL_LG, cov_LL_GG))
    row_2 = np.hstack((cov_LG_LL, cov_LG_LG, cov_LG_GG))
    row_3 = np.hstack((cov_GG_LL, cov_GG_LG, cov_GG_GG))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


def correlation_from_covariance(covariance):
    """ not thoroughly tested. Taken from 
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    does NOT work with 3x2pt
    """
    if covariance.shape[0] > 2000: print("this function doesn't work for 3x2pt")

    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# compute Sylvain's deltas
def delta_l_Sylvain(nbl, ell):
    delta_l = np.zeros(nbl)
    for l in range(1, nbl):
        delta_l[l] = ell[l] - ell[l - 1]
    delta_l[0] = delta_l[1]
    return delta_l


def Recast_Sijkl_1xauto(Sijkl, zbins):
    npairs_auto = (zbins * (zbins + 1)) // 2
    pairs_auto = np.zeros((2, npairs_auto), dtype=int)
    count = 0
    for ibin in range(zbins):
        for jbin in range(ibin, zbins):
            pairs_auto[0, count] = ibin
            pairs_auto[1, count] = jbin
            count += 1
    Sijkl_recast = np.zeros((npairs_auto, npairs_auto))
    for ipair in range(npairs_auto):
        ibin = pairs_auto[0, ipair]
        jbin = pairs_auto[1, ipair]
        for jpair in range(npairs_auto):
            kbin = pairs_auto[0, jpair]
            lbin = pairs_auto[1, jpair]
            Sijkl_recast[ipair, jpair] = Sijkl[ibin, jbin, kbin, lbin]
    return [Sijkl_recast, npairs_auto, pairs_auto]


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


## build the noise matrices ##
def build_noise(zbins, nProbes, sigma_eps2=0.09, ng=30):
    """
    function to build the noise power spectra.
    ng = number of galaxies per arcmin^2 (constant) 
    """
    conversion_factor = 11818102.860035626  # deg to arcmin^2
    fraction = 1 / zbins
    n_bar = ng * conversion_factor * fraction

    # create and fill N
    N = np.zeros((nProbes, nProbes, zbins, zbins))
    np.fill_diagonal(N[0, 0, :, :], sigma_eps2 / n_bar)
    np.fill_diagonal(N[1, 1, :, :], 1 / n_bar)
    N[0, 1, :, :] = 0
    N[1, 0, :, :] = 0
    return N


def my_exit():
    print('\nquitting script with sys.exit()')
    sys.exit()


########################### SYLVAINS FUNCTIONS ################################
@jit(nopython=True)
def cov_4D_to_2D_sylvains_ord(cov_4D, nbl, npairs):
    """Reshape from 2D to 4D using Sylvain's ordering"""
    cov_2D = np.zeros((nbl * npairs, nbl * npairs))
    for ipair in range(npairs):
        for jpair in range(npairs):
            for l1 in range(nbl):
                for l2 in range(nbl):
                    cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]
    return cov_2D


@jit(nopython=True)
def cov_2D_to_4D_sylvains_ord(cov_2D, nbl, npairs):
    """Reshape from 4D to 2D using Sylvain's ordering"""
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    for ipair in range(npairs):
        for jpair in range(npairs):
            for l1 in range(nbl):
                for l2 in range(nbl):
                    cov_4D[l1, l2, ipair, jpair] = cov_2D[ipair * nbl + l1, jpair * nbl + l2]
    return cov_4D


def Cl_3D_to_1D(Cl_3D, nbl, npairs, ind, block_index='ij'):
    """This flattens the Cl_3D to 1D. Two ordeting conventions are used:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    - which ind file to use
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.
    """

    Cl_1D = np.zeros((nbl * npairs))

    k = 0
    if block_index == 'ij' or block_index == 'sylvain':
        for ij in range(npairs):  # ATTENTION TO THE LOOP ORDERING!!
            for ell in range(nbl):
                Cl_1D[k] = Cl_3D[ell, ind[ij, 0], ind[ij, 1]]
                k += 1

    elif block_index == 'ell' or block_index == 'vincenzo':
        for ell in range(nbl):
            for ij in range(npairs):  # ATTENTION TO THE LOOP ORDERING!!
                Cl_1D[k] = Cl_3D[ell, ind[ij, 0], ind[ij, 1]]
                k += 1

    return Cl_1D


########################### OLD FUNCTIONS #####################################

@njit
def cov_SSC_ALL_old_improved(nbl, npairs_tot, ind, D_3x2pt, Sijkl, fsky, zbins, Rl):
    """The fastest routine to compute the SSC covariance matrix.
    Implements the new shift, which is much better (no ifs!!!)
    Superseeded by passing Rl as an array (fill with the same values, in case of a constant probe response)
    """

    cov_ALL_SSC = np.zeros((nbl, nbl, npairs_tot, npairs_tot))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs_tot):
                for q in range(npairs_tot):
                    i, j, k, l = ind[p, 2], ind[p, 3], ind[q, 2], ind[q, 3]
                    A, B, C, D = ind[p, 0], ind[p, 1], ind[q, 0], ind[q, 1]

                    # the shift is implemented by multiplying A, B, C, D by zbins: if lensing, probe == 0 and shift = 0
                    # if probe is GC, probe == 1 and shift = zbins. this does not hold if you switch probe indices!
                    cov_ALL_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                     D_3x2pt[ell1, A, B, i, j] *
                                                     D_3x2pt[ell2, C, D, k, l] *
                                                     Sijkl[i + A * zbins, j + B * zbins, k + C * zbins, l + D * zbins])

    cov_ALL_SSC /= fsky
    return cov_ALL_SSC

@njit
def cov_SSC_ALL_old(nbl, npairs_tot, ind, D_ALL, Sijkl, fsky, zbins, Rl):
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

# @jit(nopython=True) # not usable with dictionaries
def covariance_6D_dictionary_slow(cl_dict, noise_dict, nbl, zbins, l_lin, delta_l, fsky, probe_ordering):
    """
    A universal 6D covmat function, which mixes the indices automatically. 
    This one works with dictionaries, in particular for the cls and noise arrays. 
    probe_ordering = ['L', 'L'] or ['G', 'G'] for the individual probes, and
    probe_ordering = [['L', 'L'], ['L', 'G'], ['G', 'G']] (or variations)
    for the 3x2pt case.
    Note that, adding together the different datavectors, cov_3x2pt_6D needs
    probe indices, becoming 10D (maybe a (nbl, nbl, 3*zbins, 3*zbins, 3*zbins, 3*zbins))
    shape would work? Anyway, much less convenient to work with.
    
    this version is slower (but easier to read), it uses dictionaries directly 
    and cannot make use of numba jit
    """

    cov_10D_dict = {}
    cov_6D_arr = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            # this was the error: the array has to be initialized at every probe iteration!!!!
            cov_6D_arr = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
            for ell in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        for k in range(zbins):
                            for l in range(zbins):
                                cov_6D_arr[ell, ell, i, j, k, l] = \
                                    ((cl_dict[A, C][ell, i, k] + noise_dict[A, C][i, k]) *
                                     (cl_dict[B, D][ell, j, l] + noise_dict[B, D][j, l]) +
                                     (cl_dict[A, D][ell, i, l] + noise_dict[A, D][i, l]) *
                                     (cl_dict[B, C][ell, j, k] + noise_dict[B, C][j, k])) / \
                                    ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])
            cov_10D_dict[A, B, C, D] = cov_6D_arr

    return cov_10D_dict


# XXX these 2 are not fit for 3x2pt!!
# FIXME this function does not mix the A and B indices, is only fit for LL and GG
@jit(nopython=True)
def covariance_6D(nbl, zbins, npairs, Cij, noise, l_lin, delta_l, fsky, ind, probe):
    print('this function is deprecated, use covariance_6D_dictionary instead')

    # some checks
    assert probe == "LL" or probe == "GG", 'probe must be LL or GG, this function cannot compute 3x2pt at the moment'
    if probe == "LL":
        probe_A = 0;
        probe_B = 0
    elif probe == "GG":
        probe_A = 1;
        probe_B = 1

    # create covariance array
    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell in range(nbl):
        for i in range(zbins):
            for j in range(zbins):
                for k in range(zbins):
                    for l in range(zbins):
                        cov_6D[ell, ell, i, j, k, l] = \
                            ((Cij[ell, i, k] + noise[probe_A, probe_B, i, k]) *
                             (Cij[ell, j, l] + noise[probe_A, probe_B, j, l]) +
                             (Cij[ell, i, l] + noise[probe_A, probe_B, i, l]) *
                             (Cij[ell, j, k] + noise[probe_A, probe_B, j, k])) / \
                            ((2 * l_lin[ell] + 1) * fsky * delta_l[ell])

    return cov_6D


# the following 2 are deprecated in favour of and cov_3x2pt_dict_10D_to_4D
def cov_blocks_GL_4D(D_ALL, N, nbl, zbins, l_lin_XC, delta_l_XC, fsky, ind, npairs, npairs_asimm):
    """
    computes the 3x2pt covariance in 6 blocks of 6D, then reshapes each block 
    individually to 4D and stacks everything into cov_3x2pt_4D. This one is specifically 
    made for the probe ordering (LL, GL, GG)
    """

    print('this function is deprecated, use covariance_6D_dictionary and cov_3x2pt_dict_10D_to_4D instead')

    C_LL = D_ALL[:, 0, 0, :, :]
    C_GG = D_ALL[:, 1, 1, :, :]
    C_LG = D_ALL[:, 0, 1, :, :]  # I'm renaming, should be correct, XXX BUG ALERT
    C_GL = D_ALL[:, 1, 0, :, :]

    # noise
    N_LL = N[0, 0, :, :]
    N_GG = N[1, 1, :, :]
    N_LG = N[0, 1, :, :]
    N_GL = N[1, 0, :, :]

    print('attention: there may be an issue with the ind array: \n ind[55:155, :] may actually be ind_LG, not ind_GL')
    print('THIS FUNCTION HAS TO BE FINISHED')

    ind_LL = ind[:55, :]
    ind_GG = ind[:55, :]
    ind_GL = ind[55:155, :]  # ind_GL????? XXX BUG ALERT
    ind_LG = np.copy(ind_GL)
    ind_LG[:, [2, 3]] = ind_LG[:, [3, 2]]

    cov_LL_LL_6D = covariance_6D_blocks(C_LL, C_LL, C_LL, C_LL, N_LL, N_LL, N_LL, N_LL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_LL_GL_6D = covariance_6D_blocks(C_LG, C_LL, C_LL, C_LG, N_LG, N_LL, N_LL, N_LG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_LL_GG_6D = covariance_6D_blocks(C_LG, C_LG, C_LG, C_LG, N_LG, N_LG, N_LG, N_LG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)

    cov_GL_LL_6D = covariance_6D_blocks(C_GL, C_LL, C_GL, C_LL, N_GL, N_LL, N_GL, N_LL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_GL_GL_6D = covariance_6D_blocks(C_GG, C_LL, C_GL, C_LG, N_GG, N_LL, N_GL, N_LG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_GL_GG_6D = covariance_6D_blocks(C_GG, C_LG, C_GG, C_LG, N_GG, N_LG, N_GG, N_LG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)

    cov_GG_LL_6D = covariance_6D_blocks(C_GL, C_GL, C_GL, C_GL, N_GL, N_GL, N_GL, N_GL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_GG_GL_6D = covariance_6D_blocks(C_GG, C_GL, C_GL, C_GG, N_GG, N_GL, N_GL, N_GG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_GG_GG_6D = covariance_6D_blocks(C_GG, C_GG, C_GG, C_GG, N_GG, N_GG, N_GG, N_GG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)

    # 6D to 3D:
    cov_LL_LL_4D = cov_6D_to_4D_blocks(cov_LL_LL_6D, nbl, npairs, npairs, ind_LL, ind_LL)
    cov_LL_GL_4D = cov_6D_to_4D_blocks(cov_LL_GL_6D, nbl, npairs, npairs_asimm, ind_LL,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_LL_GG_4D = cov_6D_to_4D_blocks(cov_LL_GG_6D, nbl, npairs, npairs, ind_LL, ind_GG)

    cov_GL_LL_4D = cov_6D_to_4D_blocks(cov_GL_LL_6D, nbl, npairs_asimm, npairs, ind_GL, ind_LL)
    cov_GL_GL_4D = cov_6D_to_4D_blocks(cov_GL_GL_6D, nbl, npairs_asimm, npairs_asimm, ind_GL,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_GL_GG_4D = cov_6D_to_4D_blocks(cov_GL_GG_6D, nbl, npairs_asimm, npairs, ind_GL, ind_GG)

    cov_GG_LL_4D = cov_6D_to_4D_blocks(cov_GG_LL_6D, nbl, npairs, npairs, ind_GG, ind_LL)
    cov_GG_GL_4D = cov_6D_to_4D_blocks(cov_GG_GL_6D, nbl, npairs, npairs_asimm, ind_GG,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_GG_GG_4D = cov_6D_to_4D_blocks(cov_GG_GG_6D, nbl, npairs, npairs, ind_GG, ind_GG)

    # put the matrix together
    row_1 = np.concatenate((cov_LL_LL_4D, cov_LL_GL_4D, cov_LL_GG_4D), axis=3)
    row_2 = np.concatenate((cov_GL_LL_4D, cov_GL_GL_4D, cov_GL_GG_4D), axis=3)
    row_3 = np.concatenate((cov_GG_LL_4D, cov_GG_GL_4D, cov_GG_GG_4D), axis=3)
    cov_4D_GL = np.concatenate((row_1, row_2, row_3), axis=2)

    return cov_4D_GL


def cov_blocks_LG_4D(D_ALL, N, nbl, zbins, l_lin_XC, delta_l_XC, fsky, ind, npairs, npairs_asimm):
    """
    computes the 3x2pt covariance in 6 blocks of 6D, then reshapes each block 
    individually to 4D and stacks everything into cov_3x2pt_4D. This one is specifically 
    made for the probe ordering (LL, LG, GG)
    """

    print('this function is deprecated, use covariance_6D_dictionary and cov_3x2pt_dict_10D_to_4D instead')

    C_LL = D_ALL[:, 0, 0, :, :]
    C_GG = D_ALL[:, 1, 1, :, :]
    C_LG = D_ALL[:, 0, 1, :, :]  # I'm renaming, should be correct, XXX BUG ALERT
    C_GL = D_ALL[:, 1, 0, :, :]

    # noise
    N_LL = N[0, 0, :, :]
    N_GG = N[1, 1, :, :]
    N_LG = N[0, 1, :, :]
    N_GL = N[1, 0, :, :]

    print('attention: there may be an issue with the ind array: \n ind[55:155, :] may actually be ind_LG, not ind_GL')
    print('THIS FUNCTION HAS TO BE FINISHED')
    ind_LL = ind[:55, :]
    ind_GG = ind[:55, :]
    ind_GL = ind[55:155, :]  # ind_GL????? XXX BUG ALERT
    ind_LG = np.copy(ind_GL)
    ind_LG[:, [2, 3]] = ind_LG[:, [3, 2]]

    # def cov_blocks_LG_6D(D_ALL, N, nbl, zbins, l_lin, delta_l, fsky):
    cov_LL_LL_6D = covariance_6D_blocks(C_LL, C_LL, C_LL, C_LL, N_LL, N_LL, N_LL, N_LL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_LL_LG_6D = covariance_6D_blocks(C_LL, C_LG, C_LG, C_LL, N_LL, N_LG, N_LG, N_LL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_LL_GG_6D = covariance_6D_blocks(C_LG, C_LG, C_LG, C_LG, N_LG, N_LG, N_LG, N_LG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)

    cov_LG_LL_6D = covariance_6D_blocks(C_LL, C_GL, C_LL, C_GL, N_LL, N_GL, N_LL, N_GL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_LG_LG_6D = covariance_6D_blocks(C_LL, C_GG, C_LG, C_GL, N_LL, N_GG, N_LG, N_GL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_LG_GG_6D = covariance_6D_blocks(C_LG, C_GG, C_LG, C_GG, N_LG, N_GG, N_LG, N_GG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)

    cov_GG_LL_6D = covariance_6D_blocks(C_GL, C_GL, C_GL, C_GL, N_GL, N_GL, N_GL, N_GL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_GG_LG_6D = covariance_6D_blocks(C_GL, C_GG, C_GG, C_GL, N_GL, N_GG, N_GG, N_GL, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)
    cov_GG_GG_6D = covariance_6D_blocks(C_GG, C_GG, C_GG, C_GG, N_GG, N_GG, N_GG, N_GG, nbl, zbins, l_lin_XC,
                                        delta_l_XC, fsky)

    # 6D to 3D:
    cov_LL_LL_4D = cov_6D_to_4D_blocks(cov_LL_LL_6D, nbl, npairs, npairs, ind_LL, ind_LL)
    cov_LL_LG_4D = cov_6D_to_4D_blocks(cov_LL_LG_6D, nbl, npairs, npairs_asimm, ind_LL,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_LL_GG_4D = cov_6D_to_4D_blocks(cov_LL_GG_6D, nbl, npairs, npairs, ind_LL, ind_GG)

    cov_LG_LL_4D = cov_6D_to_4D_blocks(cov_LG_LL_6D, nbl, npairs_asimm, npairs, ind_GL, ind_LL)
    cov_LG_LG_4D = cov_6D_to_4D_blocks(cov_LG_LG_6D, nbl, npairs_asimm, npairs_asimm, ind_GL,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_LG_GG_4D = cov_6D_to_4D_blocks(cov_LG_GG_6D, nbl, npairs_asimm, npairs, ind_GL, ind_GG)

    cov_GG_LL_4D = cov_6D_to_4D_blocks(cov_GG_LL_6D, nbl, npairs, npairs, ind_GG, ind_LL)
    cov_GG_LG_4D = cov_6D_to_4D_blocks(cov_GG_LG_6D, nbl, npairs, npairs_asimm, ind_GG,
                                       ind_GL)  # XXXXXXXXXXXX it's ind_GL!!!!!
    cov_GG_GG_4D = cov_6D_to_4D_blocks(cov_GG_GG_6D, nbl, npairs, npairs, ind_GG, ind_GG)

    # put the matrix together
    row_1 = np.concatenate((cov_LL_LL_4D, cov_LL_LG_4D, cov_LL_GG_4D), axis=3)
    row_2 = np.concatenate((cov_LG_LL_4D, cov_LG_LG_4D, cov_LG_GG_4D), axis=3)
    row_3 = np.concatenate((cov_GG_LL_4D, cov_GG_LG_4D, cov_GG_GG_4D), axis=3)
    cov_4D_LG = np.concatenate((row_1, row_2, row_3), axis=2)
    return cov_4D_LG


@jit(nopython=True)
def cov_SSC_old(nbl, npairs, ind, Cij, Sijkl, fsky, probe, zbins, Rl):
    """old version. What changed in the new version is just the i, j, k, l 
    instead of the less readable ind[p, 2]... and so forth
    """

    assert probe in ['WL', 'WA', 'GC'], 'probe must be "WL", "WA" or "GC"'

    if probe == "WL" or probe == "WA":
        shift = 0
    elif probe == "GC":
        shift = zbins

    cov_SSC = np.zeros((nbl, nbl, npairs, npairs))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for p in range(npairs):
                for q in range(npairs):
                    cov_SSC[ell1, ell2, p, q] = (Rl * Rl *
                                                 Cij[ell1, ind[p, 2], ind[p, 3]] *
                                                 Cij[ell2, ind[q, 2], ind[q, 3]] *
                                                 Sijkl[ind[p, 2] + shift, ind[p, 3] + shift,
                                                       ind[q, 2] + shift, ind[q, 3] + shift])
    cov_SSC /= fsky
    return cov_SSC


@jit(nopython=True)
def cov_4D_to_2D_old(cov_4D, nbl, npairs):
    """reshape from 4 to 2 dimensions, deprecated. Working but quite convoluted and difficult to read. Not efficient because of
    the various if statements"""
    cov_2D = np.zeros((npairs * nbl, npairs * nbl))
    row = 0
    col = 0
    for ell1 in range(nbl):
        for p in range(npairs):
            col = 0
            if ell1 == 0 and p == 0:
                row = 0
            else:
                row = row + 1
            for ell2 in range(nbl):
                for q in range(npairs):
                    cov_2D[row, col] = cov_4D[ell1, ell2, p, q]
                    col = col + 1
    return cov_2D


@jit(nopython=True)
def cov_4D_to_2D_CLOE_old(cov_4D, nbl, p_max, q_max):
    """same as above, but able to accept different zpairs values, producting non-square 2D blocks. Originally used to
    build 3x2pt_2DCLOE covmat"""
    cov_2D = np.zeros((p_max * nbl, q_max * nbl))
    row = 0
    col = 0
    for ell1 in range(nbl):
        for p in range(p_max):
            col = 0
            if ell1 == 0 and p == 0:
                row = 0
            else:
                row = row + 1
            for ell2 in range(nbl):
                for q in range(q_max):
                    cov_2D[row, col] = cov_4D[ell1, ell2, p, q]
                    col = col + 1
    return cov_2D


def cov_4D_to_2D_3x2pt_CLOE_old(cov_4D, nbl, zbins):
    """Builds 3x2pt_2DCLOE using the old cov_4D_to_2D function, discarded.
    """
    npairs, npairs_asimm, npairs_tot = get_pairs(zbins)

    lim_1 = npairs
    lim_2 = npairs_asimm + npairs
    lim_3 = npairs_tot

    cov_L_L = cov_4D_to_2D_CLOE_old(cov_4D[:, :, :lim_1, :lim_1], nbl, npairs, npairs)
    cov_L_LG = cov_4D_to_2D_CLOE_old(cov_4D[:, :, :lim_1, lim_1:lim_2], nbl, npairs, npairs_asimm)
    cov_L_G = cov_4D_to_2D_CLOE_old(cov_4D[:, :, :lim_1, lim_2:lim_3], nbl, npairs, npairs)

    cov_LG_L = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_1:lim_2, :lim_1], nbl, npairs_asimm, npairs)
    cov_LG_LG = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_1:lim_2, lim_1:lim_2], nbl, npairs_asimm, npairs_asimm)
    cov_LG_G = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_1:lim_2, lim_2:lim_3], nbl, npairs_asimm, npairs)

    cov_G_L = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_2:lim_3, :lim_1], nbl, npairs, npairs)
    cov_G_LG = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_2:lim_3, lim_1:lim_2], nbl, npairs, npairs_asimm)
    cov_G_G = cov_4D_to_2D_CLOE_old(cov_4D[:, :, lim_2:lim_3, lim_2:lim_3], nbl, npairs, npairs)

    # make long rows and stack together
    row_1 = np.hstack((cov_L_L, cov_L_LG, cov_L_G))
    row_2 = np.hstack((cov_LG_L, cov_LG_LG, cov_LG_G))
    row_3 = np.hstack((cov_G_L, cov_G_LG, cov_G_G))

    array_2D = np.vstack((row_1, row_2, row_3))

    return array_2D


@jit(nopython=True)  # XXX this function is new - still to be thouroughly tested (I don't think it's true)
def cov_2D_to_4D_old(cov_2D, nbl, npairs):
    print('this function is deprecated, please use cov_2D_to_4D instead')
    """reshape from 2 to 4 dimensions"""
    # TODO maybe re-check the corresponding new function?
    cov_4D = np.zeros((nbl, nbl, npairs, npairs))
    row = 0
    col = 0
    for ell1 in range(nbl):
        for p in range(npairs):
            col = 0
            if ell1 == 0 and p == 0:
                row = 0
            else:
                row = row + 1
            for ell2 in range(nbl):
                for q in range(npairs):
                    cov_4D[ell1, ell2, p, q] = cov_2D[row, col]
                    col = col + 1
    return cov_4D
