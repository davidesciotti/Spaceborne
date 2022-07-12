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

    nbl = general_config['nbl']
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


def ISTNL_ells(general_config):
    # TODO unify these 2 functions
    """
    compute ell values as the centers of nbl+1 lin spaced bins, slightly different from the above recipe
    (different ells, same deltas)
    """
    nbl = general_config['nbl']
    ell_min = general_config['ell_min']
    ell_max = general_config['ell_max']

    ell_bins = np.linspace(np.log(ell_min), np.log(ell_max), nbl + 1)
    ells = (ell_bins[:-1] + ell_bins[1:]) / 2.
    ells = np.exp(ells)

    deltas = np.diff(np.exp(ell_bins))

    return ells, deltas


def ISTF_ells(nbl, ell_min, ell_max):
    """
    ISTF recipe, doesn't output a dictionary (i.e., is single-probe), which is also cleaner
    """
    ell_bins = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
    ells = (ell_bins[1:] + ell_bins[:-1]) / 2

    deltas = np.diff(ell_bins)

    return ells, deltas

# save as text file
# probes = ['WL', 'GC', 'WA']
# ell_maxes = [ell_max_WL, ell_max_GC, ell_max_WA] # XXX probe ordering must be the same as above!!

# for probe, ell_max in zip(probes, ell_maxes):
#     np.savetxt(path / f"output/ell_values/ell_{probe}_ellMax{probe}{ell_max}_nbl{nbl}.txt", ell_dict[f'ell_{probe}'])
#     np.savetxt(path / f"output/ell_values/delta_l_{probe}_ellMax{probe}{ell_max}_nbl{nbl}.txt", delta_dict[f'delta_l_{probe}'])
