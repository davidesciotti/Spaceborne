import numpy as np
import sys
from pathlib import Path

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm


###############################################################################
############# CODE TO CREATE THE ELL VALUES ###################################
###############################################################################


def generate_ell_and_deltas(general_config):
    """old function, but useful to compute ell and delta_ell for Wadd!"""
    nbl_WL = general_config['nbl_WL']
    nbl_GC = general_config['nbl_GC']
    assert nbl_WL == nbl_GC, 'nbl_WL and nbl_GC must be the same'
    nbl = nbl_WL

    ell_min = general_config['ell_min']
    ell_max_WL = general_config['ell_max_WL']
    ell_max_GC = general_config['ell_max_GC']
    use_WA = general_config['use_WA']

    ell_dict = {}
    delta_dict = {}

    # XC has the same ell values as GC
    ell_max_XC = ell_max_GC
    ell_max_WA = ell_max_XC

    # creating nbl ell values logarithmically equi-spaced between 10 and ell_max
    ell_WL = np.logspace(np.log10(ell_min), np.log10(ell_max_WL), nbl + 1)  # WL
    ell_GC = np.logspace(np.log10(ell_min), np.log10(ell_max_GC), nbl + 1)  # GC

    # central values of each bin
    l_centr_WL = (ell_WL[1:] + ell_WL[:-1]) / 2
    l_centr_GC = (ell_GC[1:] + ell_GC[:-1]) / 2

    # automatically compute ell_WA
    if use_WA:
        ell_WA = np.log10(np.asarray(l_centr_WL[np.where(l_centr_WL > ell_max_GC)]))
    # FIXME: this is a very bad way to implement use_WA = False. I'm computing it anyway
    # for some random values.
    else:
        ell_WA = np.log10(np.asarray(l_centr_WL[np.where(l_centr_WL > ell_max_GC / 2)]))
    nbl_WA = ell_WA.shape[0]

    # generate the deltas     
    delta_l_WL = np.diff(ell_WL)
    delta_l_GC = np.diff(ell_GC)
    delta_l_WA = np.diff(ell_WL)[-nbl_WA:]  # take only the last nbl_WA (e.g. 4) values

    # take the log10 of the values
    logarithm_WL = np.log10(l_centr_WL)
    logarithm_GC = np.log10(l_centr_GC)

    # update the ell_WL, ell_GC arrays with the right values
    ell_WL = logarithm_WL
    ell_GC = logarithm_GC

    # ell values in linear scale:
    l_lin_WL = 10 ** ell_WL
    l_lin_GC = 10 ** ell_GC
    l_lin_WA = 10 ** ell_WA
    l_lin_XC = l_lin_GC

    if use_WA and np.any(l_centr_WL == ell_max_GC):
        # check in the unlikely case that one element of l_centr_WL is == ell_max_GC. Anyway, the recipe
        # says (l_centr_WL > ell_max_GC, NOT >=).
        print('warning: one element of l_centr_WL is == ell_max_GC; the recipe says to take only\
        the elements >, but you may want to double check what to do in this case')

    # save the values
    ell_dict['ell_WL'] = ell_WL
    ell_dict['ell_GC'] = ell_GC
    ell_dict['ell_WA'] = ell_WA

    delta_dict['delta_l_WL'] = delta_l_WL
    delta_dict['delta_l_GC'] = delta_l_GC
    delta_dict['delta_l_WA'] = delta_l_WA

    return ell_dict, delta_dict


def compute_ells(nbl: int, ell_min: int, ell_max: int, recipe, output_ell_bin_edges: bool = False):
    """Compute the ell values and the bin widths for a given recipe.

    Parameters
    ----------
    nbl : int
        Number of ell bins.
    ell_min : int
        Minimum ell value.
    ell_max : int
        Maximum ell value.
    recipe : str
        Recipe to use. Must be either "ISTF" or "ISTNL".
    output_ell_bin_edges : bool, optional
        If True, also return the ell bin edges, by default False

    Returns
    -------
    ells : np.ndarray
        Central ell values.
    deltas : np.ndarray
        Bin widths
    ell_bin_edges : np.ndarray, optional
        ell bin edges. Returned only if output_ell_bin_edges is True.
    """
    if recipe == 'ISTF':
        ell_bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
        ells = (ell_bin_edges[1:] + ell_bin_edges[:-1]) / 2
        deltas = np.diff(ell_bin_edges)
    elif recipe == 'ISTNL':
        ell_bin_edges = np.linspace(np.log(ell_min), np.log(ell_max), nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.
        ells = np.exp(ells)
        deltas = np.diff(np.exp(ell_bin_edges))
    else:
        raise ValueError('recipe must be either "ISTF" or "ISTNL"')

    if output_ell_bin_edges:
        return ells, deltas, ell_bin_edges

    return ells, deltas
