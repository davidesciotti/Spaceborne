from copy import deepcopy
import sys
import time
from pathlib import Path
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import gc
import array_to_latex as a2l
import pdb
from tabulate import tabulate
from matplotlib import cm
from pprint import pprint

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

import os
ROOT = os.getenv('ROOT')
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
import bin.cosmo_lib as csmlib
import bin.ell_values as ell_utils
import bin.pyccl_cov_class as pyccl_cov_class
import bin.cl_preprocessing as cl_utils
import bin.compute_Sijkl as Sijkl_utils
import bin.covariance as covmat_utils
import bin.wf_cl_lib as wf_cl_lib
import bin.fisher_matrix as fm_utils
import bin.plots_FM_running as plot_lib
import common_cfg.ISTF_fid_params as ISTF_fid
import common_cfg.mpl_cfg as mpl_cfg


# job configuration
sys.path.append(f'{job_path}/config')
import config_SSCpaper_final as cfg


mpl.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()


# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks
# TODO super check that things work with different # of z bins

# TODO reorder all these cutting functions...
# TODO recompute Sijkl to be safe
# TODO redefine the last delta value
# TODO check what happens for ell_cuts_LG (instead of GL) = ell_cuts_XC file
# TODO cut if ell > ell_edge_lower (!!)
# TODO activate BNT transform (!!)
# TODO cut à la Vincenzo


###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

def load_ell_cuts(kmax_h_over_Mpc):
    """loads ell_cut valeus, rescales them and load into a dictionary"""
    if kmax_h_over_Mpc is None:
        kmax_h_over_Mpc = general_cfg['kmax_h_over_Mpc_ref']

    if general_cfg['which_cuts'] == 'Francis':

        ell_cuts_fldr = general_cfg['ell_cuts_folder']
        ell_cuts_filename = general_cfg['ell_cuts_filename']
        kmax_h_over_Mpc_ref = general_cfg['kmax_h_over_Mpc_ref']

        ell_cuts_LL = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="WL", **variable_specs)}')
        ell_cuts_GG = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="GC", **variable_specs)}')
        warnings.warn('I am not sure this ell_cut file is for GL, the filename is "XC"')
        ell_cuts_GL = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="XC", **variable_specs)}')
        ell_cuts_LG = ell_cuts_GL.T

        # ! linearly rescale ell cuts
        ell_cuts_LL *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
        ell_cuts_GG *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
        ell_cuts_GL *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
        ell_cuts_LG *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref

        ell_cuts_dict = {
            'LL': ell_cuts_LL,
            'GG': ell_cuts_GG,
            'GL': ell_cuts_GL,
            'LG': ell_cuts_LG}

    elif general_cfg['which_cuts'] == 'Vincenzo':

        h = 0.67
        ell_cuts_array = np.zeros((zbins, zbins))
        for zi, zval_i in enumerate(z_center_values):
            for zj, zval_j in enumerate(z_center_values):
                r_of_zi = cosmo_lib.astropy_comoving_distance(zval_i, use_h_units=False)
                r_of_zj = cosmo_lib.astropy_comoving_distance(zval_j, use_h_units=False)
                kmax_1_over_Mpc = kmax_h_over_Mpc * h
                ell_cut_i = kmax_1_over_Mpc * r_of_zi - 1 / 2
                ell_cut_j = kmax_1_over_Mpc * r_of_zj - 1 / 2
                ell_cuts_array[zi, zj] = np.min((ell_cut_i, ell_cut_j))

        ell_cuts_dict = {
            'LL': ell_cuts_array,
            'GG': ell_cuts_array,
            'GL': ell_cuts_array,
            'LG': ell_cuts_array}

    else:
        raise Exception('which_cuts must be either "Francis" or "Vincenzo"')

    return ell_cuts_dict


def cl_ell_cut_wrap(ell_dict, cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d, kmax_h_over_Mpc):
    """Wrapper for the ell cuts. Avoids the 'if general_cfg['cl_ell_cuts']' in the main loop
    (i.e., we use extraction)"""

    if not general_cfg['cl_ell_cuts']:
        return cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d

    raise Exception('I decided to implement the cuts in 1dim, this function should not be used')

    print('Performing the cl ell cuts...')

    cl_ll_3d = cl_utils.cl_ell_cut(cl_ll_3d, ell_dict['ell_WL'], ell_cuts_dict['WL'])
    cl_wa_3d = cl_utils.cl_ell_cut(cl_wa_3d, ell_dict['ell_WA'], ell_cuts_dict['WL'])
    cl_gg_3d = cl_utils.cl_ell_cut(cl_gg_3d, ell_dict['ell_GC'], ell_cuts_dict['GC'])
    cl_3x2pt_5d = cl_utils.cl_ell_cut_3x2pt(cl_3x2pt_5d, ell_cuts_dict, ell_dict['ell_3x2pt'])

    return cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d


def get_idxs_to_delete(ell_values, ell_cuts, is_auto_spectrum):
    """ ell_values can be the bin center or the bin lower edge; Francis suggests the second option is better"""

    if is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_idx, ell_val in enumerate(ell_values):
            for zi in range(zbins):
                for zj in range(zi, zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1

    elif not is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_idx, ell_val in enumerate(ell_values):
            for zi in range(zbins):
                for zj in range(zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1
    else:
        raise ValueError('is_auto_spectrum must be True or False')

    return idxs_to_delete


def get_idxs_to_delete_3x2pt(ell_values_3x2pt, ell_cuts_dict):
    """this tries to implement the indexing for the flattening ell_probe_zpair"""

    idxs_to_delete_3x2pt = []
    count = 0
    for ell_idx, ell_val in enumerate(ell_values_3x2pt):
        for zi in range(zbins):
            for zj in range(zi, zbins):
                if ell_val > ell_cuts_dict['LL'][zi, zj]:
                    idxs_to_delete_3x2pt.append(count)
                count += 1
        for zi in range(zbins):
            for zj in range(zbins):
                if ell_val > ell_cuts_dict['GL'][zi, zj]:
                    idxs_to_delete_3x2pt.append(count)
                count += 1
        for zi in range(zbins):
            for zj in range(zi, zbins):
                if ell_val > ell_cuts_dict['GG'][zi, zj]:
                    idxs_to_delete_3x2pt.append(count)
                count += 1

    # check if the array is monotonically increasing
    assert np.all(np.diff(idxs_to_delete_3x2pt) > 0)

    return list(idxs_to_delete_3x2pt)


def get_idxs_to_delete_3x2pt_v0(ell_values_3x2pt, ell_cuts_dict):
    """this implements the indexing for the flattening probe_ell_zpair"""
    raise Exception('Concatenation must be done *before* flattening, this function is not compatible with the '
                    '"ell-block ordering of the wf_cl_lib matrix"')
    idxs_to_delete_LL = get_idxs_to_delete(ell_values_3x2pt, ell_cuts_dict['LL'], is_auto_spectrum=True)
    idxs_to_delete_GL = get_idxs_to_delete(ell_values_3x2pt, ell_cuts_dict['GL'], is_auto_spectrum=False)
    idxs_to_delete_GG = get_idxs_to_delete(ell_values_3x2pt, ell_cuts_dict['GG'], is_auto_spectrum=True)

    # when concatenating, we need to add the offset from the stacking of the 3 datavectors
    idxs_to_delete_3x2pt = np.concatenate((
        np.array(idxs_to_delete_LL),
        nbl_3x2pt * zpairs_auto + np.array(idxs_to_delete_GL),
        nbl_3x2pt * (zpairs_auto + zpairs_cross) + np.array(idxs_to_delete_GG)))

    # check if the array is monotonically increasing
    assert np.all(np.diff(idxs_to_delete_3x2pt) > 0)

    return list(idxs_to_delete_3x2pt)


def plot_nz_tocheck_func():
    if not covariance_cfg['plot_nz_tocheck']:
        return
    plt.figure()
    for zi in range(zbins):
        plt.plot(zgrid_n_of_z, n_of_z[:, zi], label=f'zbin {zi}')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('n(z)')


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# TODO restore the for loops
# TODO iterate over the different pks
# TODO ell_cuts
# TODO BNT
# TODO SSC


general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg
Sijkl_cfg = cfg.Sijkl_cfg
fm_cfg = cfg.FM_cfg

# for kmax_h_over_Mpc in general_cfg['kmax_h_over_Mpc_list']:
# for general_cfg['which_cuts'] in ['Francis', 'Vincenzo']:
#     for general_cfg['center_or_min'] in ['center', 'min']:

warnings.warn('TODO restore the for loops!')
general_cfg['which_cuts'] = 'Vincenzo'
general_cfg['center_or_min'] = 'min'
kmax_h_over_Mpc = general_cfg['kmax_h_over_Mpc_list'][5]

# some convenence variables, just to make things more readable
zbins = general_cfg['zbins']
EP_or_ED = general_cfg['EP_or_ED']
ell_max_WL = general_cfg['ell_max_WL']
ell_max_GC = general_cfg['ell_max_GC']
ell_max_XC = general_cfg['ell_max_XC']
magcut_source = general_cfg['magcut_source']
magcut_lens = general_cfg['magcut_lens']
zcut_source = general_cfg['zcut_source']
zcut_lens = general_cfg['zcut_lens']
flat_or_nonflat = general_cfg['flat_or_nonflat']
center_or_min = general_cfg['center_or_min']
zmax = int(general_cfg['zmax'] * 10)
triu_tril = covariance_cfg['triu_tril']
row_col_major = covariance_cfg['row_col_major']
n_probes = general_cfg['n_probes']
which_pk = general_cfg['which_pk']

if (general_cfg['ell_max_WL'], general_cfg['ell_max_3x2pt'], general_cfg['ell_max_XC'], general_cfg['ell_max_GC']) == (5000, 3000, 3000, 3000):
    which_case = 'Opt'
elif (general_cfg['ell_max_WL'], general_cfg['ell_max_3x2pt'], general_cfg['ell_max_XC'], general_cfg['ell_max_GC']) == (1500, 750, 750, 750):
    which_case = 'Pes'
else:
    raise ValueError('This combination of ell_max_WL and ell_max_3x2pt is not relevant to the paper!')

# some checks
# assert general_cfg['flagship_version'] == 2, 'The input files used in this job for flagship version 2!'
assert general_cfg['use_WA'] is False, 'We do not use Wadd for SPV3 at the moment'

if covariance_cfg['cov_BNT_transform']:
    assert general_cfg[
        'cl_BNT_transform'] is False, 'the BNT transform should be applied either to the Cls ' \
        'or to the covariance'
    assert fm_cfg['derivatives_BNT_transform'], 'you should BNT transform the derivatives as well'

# which cases to save: GO, GS or GO, GS and SS
cases_tosave = ['G', 'GSSC']


flat_fid_pars_dict = {
    'Om_m0': ISTF_fid.primary['Om_m0'],
    'Om_b0': ISTF_fid.primary['Om_b0'],
    'Om_Lambda0': ISTF_fid.extensions['Om_Lambda0'],
    'w_0': ISTF_fid.primary['w_0'],
    'w_a': ISTF_fid.primary['w_a'],
    'h': ISTF_fid.primary['h_0'],
    'n_s': ISTF_fid.primary['n_s'],
    'sigma_8': ISTF_fid.primary['sigma_8'],
    'm_nu': ISTF_fid.extensions['m_nu'],
    'N_eff': ISTF_fid.neutrino_params['N_eff'],
    'sigma8': ISTF_fid.primary['sigma_8'],
    'A_IA': ISTF_fid.IA_free['A_IA'],
    'eta_IA': ISTF_fid.IA_free['eta_IA'],
    'beta_IA': ISTF_fid.IA_free['beta_IA'],
    'C_IA': ISTF_fid.IA_fixed['C_IA'],

    'other_params': {
        'camb_extra_parameters': {
            'camb': {
                'halofit_version': 'mead2020_feedback',
                'HMCode_logT_AGN': 7.75,
                'num_massive_neutrinos': 1,
                'dark_energy_model': 'ppf',
            }
        }
    },
}

ccl_obj = pyccl_cov_class.PycclClass(flat_fid_pars_dict)

# build the ind array and store it into the covariance dictionary
ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
covariance_cfg['ind'] = ind

# convenience vectors
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :].copy()
# ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()

assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
    'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

# compute ell and delta ell values in the reference (optimistic) case
ell_WL_nbl32, delta_l_WL_nbl32, ell_edges_WL_nbl32 = ell_utils.compute_ells(general_cfg['nbl_WL_opt'],
                                                                            general_cfg['ell_min'],
                                                                            general_cfg['ell_max_WL_opt'],
                                                                            recipe='ISTF',
                                                                            output_ell_bin_edges=True)

# perform the cuts (not the redshift-dependent ones!) on the ell centers and edges
ell_dict = {}
ell_dict['ell_WL'] = np.copy(ell_WL_nbl32[ell_WL_nbl32 < ell_max_WL])
ell_dict['ell_GC'] = np.copy(ell_WL_nbl32[ell_WL_nbl32 < ell_max_GC])
ell_dict['ell_WA'] = np.copy(ell_WL_nbl32[(ell_WL_nbl32 > ell_max_GC) & (ell_WL_nbl32 < ell_max_WL)])
ell_dict['ell_XC'] = np.copy(ell_dict['ell_GC'])
ell_dict['ell_3x2pt'] = np.copy(ell_dict['ell_XC'])

# store edges *except last one for dimensional consistency* in the ell_dict
ell_dict['ell_edges_WL'] = np.copy(ell_edges_WL_nbl32[ell_edges_WL_nbl32 < ell_max_WL])[:-1]
ell_dict['ell_edges_GC'] = np.copy(ell_edges_WL_nbl32[ell_edges_WL_nbl32 < ell_max_GC])[:-1]
ell_dict['ell_edges_WA'] = np.copy(
    ell_edges_WL_nbl32[(ell_edges_WL_nbl32 > ell_max_GC) & (ell_edges_WL_nbl32 < ell_max_WL)])[:-1]
ell_dict['ell_edges_XC'] = np.copy(ell_dict['ell_edges_GC'])[:-1]
ell_dict['ell_edges_3x2pt'] = np.copy(ell_dict['ell_edges_XC'])[:-1]

for key in ell_dict.keys():
    if 'WA' not in key:
        assert np.max(ell_dict[key]) > 15, 'ell values must *not* be in log space'

# set corresponding number of ell bins
nbl_WL = len(ell_dict['ell_WL'])
nbl_GC = len(ell_dict['ell_GC'])
nbl_WA = len(ell_dict['ell_WA'])
nbl_3x2pt = nbl_GC
general_cfg['nbl_WL'] = nbl_WL

delta_dict = {'delta_l_WL': np.copy(delta_l_WL_nbl32[:nbl_WL]),
              'delta_l_GC': np.copy(delta_l_WL_nbl32[:nbl_GC]),
              'delta_l_WA': np.copy(delta_l_WL_nbl32[nbl_GC:nbl_WL])}

nbl_WL_opt = general_cfg['nbl_WL_opt']
nbl_GC_opt = general_cfg['nbl_GC_opt']
nbl_WA_opt = general_cfg['nbl_WA_opt']
nbl_3x2pt_opt = general_cfg['nbl_3x2pt_opt']

# if ell_max_WL == general_cfg['ell_max_WL_opt']:
#     assert (nbl_WL_opt, nbl_GC_opt, nbl_WA_opt, nbl_3x2pt_opt) == (nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt), \
#         'nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt don\'t match with the expected values for the optimistic case'

# this is just to make the .format() more compact
variable_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins,
                  'ell_max_WL': ell_max_WL, 'ell_max_GC': ell_max_GC, 'ell_max_XC': ell_max_XC,
                  'nbl_WL': nbl_WL, 'nbl_GC': nbl_GC, 'nbl_WA': nbl_WA, 'nbl_3x2pt': nbl_3x2pt,
                  'ell_max_3x2pt': general_cfg['ell_max_3x2pt'],
                  'magcut_source': magcut_source, 'magcut_lens': magcut_lens,
                  'zcut_source': zcut_source, 'zcut_lens': zcut_lens,
                  'zmin': general_cfg['zmin'], 'zmax': zmax, 'magcut': general_cfg['magcut'],
                  'flat_or_nonflat': general_cfg['flat_or_nonflat'], 'which_pk': which_pk,
                  }


# ! import and reshape datavectors (cl) and response functions (rl)
cl_fld = general_cfg['cl_folder'].format(which_pk=which_pk, probe='{probe:s}')
cl_filename = general_cfg['cl_filename']
cl_ll_1d = np.genfromtxt(
    f"{cl_fld.format(probe='WLO', which_pk=which_pk)}/{cl_filename.format(probe='WLO', nbl=nbl_WL, **variable_specs)}")
cl_gg_1d = np.genfromtxt(
    f"{cl_fld.format(probe='GCO', which_pk=which_pk)}/{cl_filename.format(probe='GCO', nbl=nbl_WL, **variable_specs)}")
cl_3x2pt_1d = np.genfromtxt(
    f"{cl_fld.format(probe='3x2pt', which_pk=which_pk)}/{cl_filename.format(probe='3x2pt', nbl=nbl_WL, **variable_specs)}")

if general_cfg['use_WA']:
    cl_wa_1d = np.genfromtxt(
        f"{cl_fld.format(probe='WLA', which_pk=which_pk)}/{cl_filename.format(probe='WLA', nbl=nbl_WL, **variable_specs)}")
else:
    cl_wa_1d = np.ones_like(cl_ll_1d)

# reshape to 3 dimensions
try:
    cl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(cl_ll_1d, 'WL', nbl_WL, zbins)
    cl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(cl_gg_1d, 'GC', nbl_GC, zbins)
    cl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(cl_3x2pt_1d, '3x2pt', nbl_3x2pt, zbins)
except AssertionError as err:
    print(err)
    print('Importing ellmax=5000 files and cutting')
    cl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(cl_ll_1d, 'WL', nbl_WL_opt, zbins)
    cl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(cl_gg_1d, 'GC', nbl_WL_opt, zbins)
    cl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(cl_3x2pt_1d, '3x2pt', nbl_WL_opt, zbins)

    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
    cl_3x2pt_5d = cl_3x2pt_5d[:, :, :nbl_3x2pt, :, :]

cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]

# plot cls
plt.rcParams.update({'xtick.labelsize': 19})
plt.rcParams.update({'ytick.labelsize': 19})
plt.rcParams.update({'axes.labelsize': 21})
plt.rcParams.update({'axes.titlesize': 21})

fig, axs = plt.subplots(1, 3, figsize=(20, 7))
colors = cm.rainbow(np.linspace(0, 1, zbins))
for zi in range(zbins):
    zj = zi
    axs[0].loglog(ell_dict['ell_GC'], cl_3x2pt_5d[0, 0, :, zi, zj], label='$z_{\\rm bin}$ %d' % (zi + 1), c=colors[zi])
    axs[1].loglog(ell_dict['ell_GC'], cl_3x2pt_5d[1, 0, :, zi, zj], c=colors[zi])
    axs[2].loglog(ell_dict['ell_GC'], cl_3x2pt_5d[1, 1, :, zi, zj], c=colors[zi])

axs[0].set_title('WL')
axs[1].set_title('XC')
axs[2].set_title('GCph')
axs[0].set_ylabel('$C^{AB}_{ij}(\ell)$')
axs[0].set_xlabel('$\ell$')
axs[1].set_xlabel('$\ell$')
axs[2].set_xlabel('$\ell$')
fig.legend(loc='right')
# plt.savefig('/home/davide/Documenti/Lavoro/Programmi/phd_thesis_plots/plots/cls.pdf', dpi=500, bbox_inches='tight')


# assert False, 'stop here and undo the latest changes with git, they were just to produce the cls plot'

# Columns in the nuisance parameters file are as follows for each bin:
# 1. mean redshift (not used in the code)
# 2. total number density (needed for covariance estimate)
# 3. galaxy bias b_g
# 4. slope s_M of the luminosity function
# 5. shift dz in the mean redshift of the bin
# 6. variance of the photo - z error (not used in the code)
nuisance_folder = covariance_cfg["nuisance_folder"]
nuisance_filename = f'{covariance_cfg["nuisance_filename"].format(**variable_specs)}'
nuisance_tab = np.genfromtxt(f'{nuisance_folder}/'f'{nuisance_filename}')
_z_center_values_import = nuisance_tab[:, 0] # this is simply the mean, computed by Vincenzo but not used (by him)
covariance_cfg['ng'] = nuisance_tab[:, 1]
gal_bias_fid = nuisance_tab[:, 2]
dz_shifts = nuisance_tab[:, 4]

nofz_folder = covariance_cfg["nofz_folder"]
nofz_filename = f'{covariance_cfg["nofz_filename"].format(**variable_specs)}'
n_of_z = np.genfromtxt(f'{nofz_folder}/'f'{nofz_filename}')
zgrid_n_of_z = n_of_z[:, 0]
n_of_z = n_of_z[:, 1:]

plot_nz_tocheck_func()

z_center_values = wf_cl_lib.get_z_effective_isaac(zgrid_n_of_z, n_of_z)  # this is the actual values to be used for the gal bias


# some check on the input nz files
assert np.all(covariance_cfg['ng'] < 5), 'ng values are likely < 5 *per bin*; this is just a rough check'
assert np.all(covariance_cfg['ng'] > 0), 'ng values must be positive'
assert np.all(z_center_values > 0), 'z_center values must be positive'
assert np.all(z_center_values < 3), 'z_center values are likely < 3; this is just a rough check'
assert np.all(gal_bias_fid > 1), 'galaxy bias should be > 1'
assert np.all(gal_bias_fid < 3), 'galaxy bias seems a bit large; this is just a rough check'


# print('Computing BNT mrix...')
# BNT_matrix = covmat_utils.compute_BNT_matrix(zbins, zgrid_n_of_z, n_of_z, plot_nz=False)
BNT_matrix = np.eye(zbins)
if general_cfg['BNT_transform']:
    raise NotImplementedError('BNT transform not implemented yet')

rl_fld = general_cfg['rl_folder']
rl_filename = general_cfg['rl_filename']
rl_ll_1d = np.genfromtxt(
    f"{rl_fld.format(probe='WLO')}/{rl_filename.format(probe='WLO', nbl=nbl_WL_opt, **variable_specs)}")
rl_gg_1d = np.genfromtxt(
    f"{rl_fld.format(probe='GCO')}/{rl_filename.format(probe='GCO', nbl=nbl_3x2pt_opt, **variable_specs)}")
rl_3x2pt_1d = np.genfromtxt(
    f"{rl_fld.format(probe='3x2pt')}/{rl_filename.format(probe='3x2pt', nbl=nbl_3x2pt_opt, **variable_specs)}")


rl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(rl_ll_1d, 'WL', nbl_WL_opt, zbins)
rl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(rl_gg_1d, 'GC', nbl_3x2pt_opt, zbins)
rl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(rl_3x2pt_1d, '3x2pt', nbl_3x2pt_opt, zbins)

rl_ll_3d=rl_ll_3d[:nbl_WL, :, :]
rl_gg_3d=rl_gg_3d[:nbl_GC, :, :]
rl_3x2pt_5d=rl_3x2pt_5d[:, :,:nbl_3x2pt, :, :]
rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]

# check that cl_wa is equal to cl_ll in the last nbl_WA_opt bins
if ell_max_WL == general_cfg['ell_max_WL_opt']:
    if not np.array_equal(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :]):
        rtol = 1e-5
        # plt.plot(ell_dict['ell_WL'], cl_ll_3d[:, 0, 0])
        # plt.plot(ell_dict['ell_WL'][nbl_GC:nbl_WL], cl_wa_3d[:, 0, 0])
        assert (np.allclose(cl_wa_3d, cl_ll_3d[nbl_GC:nbl_WL, :, :], rtol=rtol, atol=0)), \
            'cl_wa_3d should be obtainable from cl_ll_3d!'
        print(f'cl_wa_3d and cl_ll_3d[nbl_GC:nbl_WL, :, :] are not exactly equal, but have a relative '
              f'difference of less than {rtol}')

# ! BNT transform the cls (and responses?) - it's more complex since I also have to transform the noise
# ! spectra, better to transform directly the covariance matrix
if general_cfg['cl_BNT_transform']:
    print('BNT-transforming the Cls...')
    assert covariance_cfg['cov_BNT_transform'] is False, \
        'the BNT transform should be applied either to the Cls or to the covariance, not both'
    cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, BNT_matrix, 'L', 'L')
    cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, BNT_matrix, 'L', 'L')
    cl_3x2pt_5d = cl_utils.cl_BNT_transform_3x2pt(cl_3x2pt_5d, BNT_matrix)
    warnings.warn('you should probebly BNT-transform the responses too!')

# ! cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
if ell_max_WL == 1500:
    warnings.warn(
        'you are cutting the datavectors and responses in the pessimistic case, but is this compatible '
        'with the redshift-dependent ell cuts?')
    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
    cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
    cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

    rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
    rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
    rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
    rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

# this is to pass the ll cuts to the covariance module
# ell_cuts_dict = load_ell_cuts(kmax_h_over_Mpc)
ell_cuts_dict = {}
ell_dict['ell_cuts_dict'] = ell_cuts_dict  # rename for better readability

# ! Vincenzo's method for cl_ell_cuts: get the idxs to delete for the flattened 1d cls
if general_cfg['center_or_min'] == 'center':
    prefix = 'ell'
elif general_cfg['center_or_min'] == 'min':
    prefix = 'ell_edges'
else:
    raise ValueError('general_cfg["center_or_min"] should be either "center" or "min"')

# ell_dict['idxs_to_delete_dict'] = {
#     'LL': get_idxs_to_delete(ell_dict[f'{prefix}_WL'], ell_cuts_dict['LL'], is_auto_spectrum=True),
#     'GG': get_idxs_to_delete(ell_dict[f'{prefix}_GC'], ell_cuts_dict['GG'], is_auto_spectrum=True),
#     'WA': get_idxs_to_delete(ell_dict[f'{prefix}_WA'], ell_cuts_dict['LL'], is_auto_spectrum=True),
#     'GL': get_idxs_to_delete(ell_dict[f'{prefix}_XC'], ell_cuts_dict['GL'], is_auto_spectrum=False),
#     'LG': get_idxs_to_delete(ell_dict[f'{prefix}_XC'], ell_cuts_dict['LG'], is_auto_spectrum=False),
#     '3x2pt': get_idxs_to_delete_3x2pt(ell_dict[f'{prefix}_3x2pt'], ell_cuts_dict)
# }

# ! 3d cl ell cuts (*after* BNT!!)
cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d = cl_ell_cut_wrap(
    ell_dict, cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d, kmax_h_over_Mpc)
# TODO here you could implement 1d cl ell cuts (but we are cutting at the covariance and derivatives level)

# store cls and responses in a dictionary
cl_dict_3D = {
    'cl_LL_3D': cl_ll_3d,
    'cl_GG_3D': cl_gg_3d,
    'cl_WA_3D': cl_wa_3d,
    'cl_3x2pt_5D': cl_3x2pt_5d}

rl_dict_3D = {
    'rl_LL_3D': rl_ll_3d,
    'rl_GG_3D': rl_gg_3d,
    'rl_WA_3D': rl_wa_3d,
    'rl_3x2pt_5D': rl_3x2pt_5d}

if covariance_cfg['compute_SSC']:

    # ! first line is wrong, need to subtract Om_nu0
    cosmo_par_dict_classy = {'Omega_cdm': ISTF_fid.primary['Om_m0'] - ISTF_fid.primary['Om_b0'],
                             'Omega_b': ISTF_fid.primary['Om_b0'],
                             'w0_fld': ISTF_fid.primary['w_0'],
                             'wa_fld': ISTF_fid.primary['w_a'],
                             'h': ISTF_fid.primary['h_0'],
                             'n_s': ISTF_fid.primary['n_s'],
                             'sigma8': ISTF_fid.primary['sigma_8'],

                             'm_ncdm': ISTF_fid.extensions['m_nu'],
                             'N_ncdm': ISTF_fid.neutrino_params['N_ncdm'],
                             'N_ur': ISTF_fid.neutrino_params['N_ur'],

                             'Omega_Lambda': ISTF_fid.extensions['Om_Lambda0'],

                             'P_k_max_1/Mpc': 1200,
                             'output': 'mPk',
                             'non linear': 'halofit',  # ! takabird?

                             # 'z_max_pk': 2.038,
                             'z_max_pk': 4,  # do I get an error without this key?
                             }

    # ! load kernels
    # TODO this should not be done if Sijkl is loaded; I have a problem with nz, which is part of the file name...
    wf_folder = Sijkl_cfg["wf_input_folder"].format(**variable_specs)
    wf_gamma_filename = Sijkl_cfg["wf_gamma_input_filename"]
    wf_ia_filename = Sijkl_cfg["wf_ia_input_filename"]
    wf_delta_filename = Sijkl_cfg["wf_delta_input_filename"]

    wf_gamma = np.genfromtxt(f'{wf_folder}/{wf_gamma_filename.format(**variable_specs)}')
    wf_ia = np.genfromtxt(f'{wf_folder}/{wf_ia_filename.format(**variable_specs)}')
    wf_delta = np.genfromtxt(f'{wf_folder}/{wf_delta_filename.format(**variable_specs)}')

    z_arr = wf_delta[:, 0]
    wf_delta = wf_delta[:, 1:]
    wf_gamma = wf_gamma[:, 1:]
    wf_ia = wf_ia[:, 1:]

    # construct lensing kernel
    ia_bias = wf_cl_lib.build_ia_bias_1d_arr(z_grid_out=z_arr, cosmo_ccl=ccl_obj.cosmo_ccl,
                                             flat_fid_pars_dict=flat_fid_pars_dict,
                                             input_z_grid_lumin_ratio=None, input_lumin_ratio=None,
                                             output_F_IA_of_z=False)

    wil = wf_gamma + ia_bias[:, None] * wf_ia
    wig = wf_delta

    # transpose and stack, ordering is important here!
    assert wil.shape == wig.shape, 'the GC and WL kernels have different shapes'
    assert wil.shape == (z_arr.shape[0], zbins), 'the kernels have the wrong shape'
    transp_stacked_wf = np.vstack((wil.T, wig.T))
    # ! compute or load Sijkl
    nz = z_arr.shape[0]  # get number of z points in nz to name the Sijkl file
    Sijkl_folder = Sijkl_cfg['Sijkl_folder']
    assert general_cfg[
        'cl_BNT_transform'] is False, 'for SSC, at the moment the BNT transform should not be ' \
        'applied to the cls, but to the covariance matrix (how ' \
        'should we deal with the responses in the former case?)'
    Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(flagship_version=general_cfg['flagship_version'],
                                                        nz=nz, IA_flag=Sijkl_cfg['has_IA'],
                                                        **variable_specs)

    # if Sijkl exists, load it; otherwise, compute it and save it
    if Sijkl_cfg['use_precomputed_sijkl'] and os.path.isfile(f'{Sijkl_folder}/{Sijkl_filename}'):
        print(f'Sijkl matrix already exists in folder\n{Sijkl_folder}; loading it')
        Sijkl = np.load(f'{Sijkl_folder}/{Sijkl_filename}')
    else:
        Sijkl = Sijkl_utils.compute_Sijkl(cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                          Sijkl_cfg['wf_normalization'])
        np.save(f'{Sijkl_folder}/{Sijkl_filename}', Sijkl)

else:
    warnings.warn('Sijkl is not computed, but set to identity')
    Sijkl = np.ones((n_probes * zbins, n_probes * zbins, n_probes * zbins, n_probes * zbins))

# ! compute covariance matrix
cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                    ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, BNT_matrix)

# save covariance matrix and test against benchmarks
cov_folder = covariance_cfg['cov_folder'].format(cov_ell_cuts=str(covariance_cfg['cov_ell_cuts']),
                                                 **variable_specs)
covmat_utils.save_cov(cov_folder, covariance_cfg, cov_dict, cases_tosave=['GO', 'GS'], **variable_specs)

if general_cfg['test_against_benchmarks']:
    cov_benchmark_folder = f'{cov_folder}/benchmarks'
    mm.test_folder_content(cov_folder, cov_benchmark_folder, covariance_cfg['cov_file_format'])

# ! compute Fisher matrix
if not fm_cfg['compute_FM']:
    # this guard is just to avoid indenting the whole code below
    raise KeyboardInterrupt('skipping FM computation, the script will exit now')

# set the fiducial values in a dictionary and a list
fiducials_dict = {
    'cosmo': [ISTF_fid.primary['Om_m0'], ISTF_fid.primary['Om_b0'],
              ISTF_fid.primary['w_0'], ISTF_fid.primary['w_a'],
              ISTF_fid.primary['h_0'], ISTF_fid.primary['n_s'], ISTF_fid.primary['sigma_8'], 7.75],
    'IA': [ISTF_fid.IA_free['A_IA'], ISTF_fid.IA_free['eta_IA'], ISTF_fid.IA_free['beta_IA']],
    'shear_bias': np.zeros((zbins,)),
    'dzWL': np.zeros((zbins,)),  # for the time being, equal to the GC ones
    # 'dzGC': np.zeros((zbins,)),  # for the time being, equal to the GC ones
    'galaxy_bias': gal_bias_fid,
}
fiducials_values_3x2pt = list(np.concatenate([fiducials_dict[key] for key in fiducials_dict.keys()]))

# set parameters' names, as a dict and as a list
param_names_dict = fm_cfg['param_names_dict']
param_names_3x2pt = fm_cfg['param_names_3x2pt']

assert param_names_dict.keys() == fiducials_dict.keys(), \
    'the parameter names and fiducial values dictionaries should have the same keys'

assert len(fiducials_values_3x2pt) == len(param_names_3x2pt), \
    'the fiducial values list and parameter names should have the same length'

# ! preprocess derivatives (or load the alreay preprocessed ones)
# import and store them in one big dictionary
# start_time = time.perf_counter()
# derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs)

# # check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
# der_prefix = fm_cfg['derivatives_prefix']
# vinc_filenames = mm.get_filenames_in_folder(derivatives_folder)
# vinc_filenames = [vinc_filename for vinc_filename in vinc_filenames if vinc_filename.startswith(der_prefix)]
# vinc_filenames = [
#     vinc_filename for vinc_filename in vinc_filenames if f'{EP_or_ED}{zbins:02d}-zedMin{general_cfg["zmin"]:02d}-zedMax{zmax}-mag{general_cfg["magcut"]}' in vinc_filename]

# # perform some checks on the filenames before trimming them
# for vinc_filename in vinc_filenames:
#     assert f'{EP_or_ED}{zbins}' in vinc_filename, f'{EP_or_ED}{zbins} not in filename {vinc_filename}'
#     assert f'mag{general_cfg["magcut"]}' in vinc_filename, f'mag{general_cfg["magcut"]} not in filename {vinc_filename}'
#     assert f'zedMax{zmax}' in vinc_filename, f'zedMax{zmax} not in filename {vinc_filename}'

# vinc_trimmed_filenames = [vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames]
# vinc_trimmed_filenames = [vinc_trimmed_filename[len(der_prefix):] if vinc_trimmed_filename.startswith(der_prefix) else vinc_trimmed_filename
#                           for vinc_trimmed_filename in vinc_trimmed_filenames]
# vinc_param_names = list(set(vinc_trimmed_filenames))
# vinc_param_names.sort()

# my_sorted_param_names = param_names_3x2pt.copy()
# my_sorted_param_names.sort()

# # check whether the 2 lists match and print the elements that are in one list but not in the other
# param_names_not_in_my_list = [vinc_param_name for vinc_param_name in vinc_param_names if
#                               vinc_param_name not in my_sorted_param_names]
# param_names_not_in_vinc_list = [my_sorted_param_name for my_sorted_param_name in my_sorted_param_names if
#                                 my_sorted_param_name not in vinc_param_names]
# try:
#     assert np.all(vinc_param_names == my_sorted_param_names), \
#         f'\nparams present in input folder but not in the cfg file: {param_names_not_in_my_list}\n' \
#         f'params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}'
# except AssertionError as error:
#     print(error)
#     if param_names_not_in_vinc_list == ['logT_AGN']:
#         print('the derivative w.r.t logT_AGN is missing in the input folder but '
#               'the corresponding FM is still set to 0; moving on')
#     else:
#         warnings.warn('there is something wrong with the parameter names in the derivatives folder. '
#                       'SETTING MY PARAM NAMES AS THE REFERENCE')

# if fm_cfg['load_preprocess_derivatives']:
#     dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
#     dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
#     dC_WA_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_WA_4D.npy')
#     dC_3x2pt_6D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy')

# elif not fm_cfg['load_preprocess_derivatives']:
#     der_prefix = fm_cfg['derivatives_prefix']
#     dC_dict_1D = dict(mm.get_kv_pairs(derivatives_folder, "dat"))
#     # check if dictionary is empty
#     if not dC_dict_1D:
#         raise ValueError(f'No derivatives found in folder {derivatives_folder}')
    
#     # filter the dict
#     dC_dict_1D = {key: value for key, value in dC_dict_1D.items() if any(param in key for param in my_sorted_param_names)}

#     # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
#     dC_dict_LL_3D = {}
#     dC_dict_GG_3D = {}
#     dC_dict_WA_3D = {}
#     dC_dict_3x2pt_5D = {}
    
#     try:
#         for key in dC_dict_1D.keys():
#             if 'WLO' in key:
#                 dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WL', nbl_WL, zbins)
#             elif 'GCO' in key:
#                 dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'GC', nbl_GC, zbins)
#             elif 'WLA' in key:
#                 dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WA', nbl_WA, zbins)
#             elif '3x2pt' in key:
#                 dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], '3x2pt', nbl_3x2pt, zbins)
#     except AssertionError as err:
#         print(err)
#         print('Importing ellmax=5000 files and cutting')
        
#         for key in dC_dict_1D.keys():
#             if 'WLO' in key:
#                 dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WL', nbl_WL, zbins)
#             elif 'GCO' in key:
#                 dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'GC', nbl_WL, zbins)
#                 dC_dict_GG_3D[key] = dC_dict_GG_3D[key][ :nbl_GC, :, :]
#             elif 'WLA' in key:
#                 # dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WA', nbl_WL, zbins)
#                 dC_dict_WA_3D[key] = dC_dict_LL_3D[key][nbl_GC:nbl_WL, :, :]
#             elif '3x2pt' in key:
#                 dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], '3x2pt', nbl_WL, zbins)
#                 dC_dict_3x2pt_5D[key] = dC_dict_3x2pt_5D[key][:, :, :nbl_3x2pt, :, :]


#     # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
#     dC_LL_4D = fm_utils.dC_dict_to_4D_array(dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix)
#     dC_GG_4D = fm_utils.dC_dict_to_4D_array(dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix)
#     # dC_WA_4D = FM_utils.dC_dict_to_4D_array(dC_dict_WA_3D, param_names_3x2pt, nbl_WA, zbins, der_prefix)
#     dC_WA_4D = dC_LL_4D[nbl_GC:nbl_WL, ...]
#     dC_3x2pt_6D = fm_utils.dC_dict_to_4D_array(dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins,
#                                                der_prefix, is_3x2pt=True)


param_names_wl = [param_name for param_name in param_names_3x2pt if 'bG' not in param_name]
param_names_wl = [param_name for param_name in param_names_wl if 'dzGC' not in param_name]
param_names_gc = [param_name for param_name in param_names_3x2pt if 'm0' not in param_name]
param_names_gc = [param_name for param_name in param_names_gc if 'm1' not in param_name]
param_names_gc = [param_name for param_name in param_names_gc if 'dzWL' not in param_name]
param_names_gc = [param_name for param_name in param_names_gc if not any(substring in param_name for substring in ['Aia', 'eIA', 'bIA'])]
param_names_full_dict = {
    'WL': param_names_wl,
    'GC': param_names_gc,
    '3x2pt': param_names_3x2pt,
}


param_names_3x2pt_vin = deepcopy(param_names_3x2pt)
index = param_names_3x2pt_vin.index('logT')
param_names_3x2pt_vin.insert(index, 'gamma')

param_names_wl_vin = [param_name for param_name in param_names_3x2pt_vin if 'bG' not in param_name]
param_names_wl_vin = [param_name for param_name in param_names_wl_vin if 'dzGC' not in param_name]
param_names_gc_vin = [param_name for param_name in param_names_3x2pt_vin if 'm0' not in param_name]
param_names_gc_vin = [param_name for param_name in param_names_gc_vin if 'm1' not in param_name]
param_names_gc_vin = [param_name for param_name in param_names_gc_vin if 'dzWL' not in param_name]
param_names_gc_vin = [param_name for param_name in param_names_gc_vin if not any(substring in param_name for substring in ['Aia', 'eIA', 'bIA'])]

# separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
dC_dict_LL_3D = {}
dC_dict_GG_3D = {}
dC_dict_WA_3D = {}
dC_dict_3x2pt_5D = {}
derivatives_filename = fm_cfg['derivatives_filename']
derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs, ROOT=ROOT, probe='{probe:s}')

for param in param_names_wl:
    der_name = derivatives_filename.format(probe='WLO', param_name=param, **variable_specs)
    dcl_wl_1d = np.genfromtxt(f'{derivatives_folder}/{der_name}')
    dC_dict_LL_3D[param] = cl_utils.cl_SPV3_1D_to_3D(dcl_wl_1d, 'WL', nbl_WL_opt, zbins)[:nbl_WL, :, :]
    dC_dict_WA_3D[param] = dC_dict_LL_3D[param][nbl_GC:nbl_WL]

for param in param_names_gc:
    der_name = derivatives_filename.format(probe='GCO', param_name=param, **variable_specs)
    dcl_gc_1d = np.genfromtxt(f'{derivatives_folder}/{der_name}')
    dC_dict_GG_3D[param] = cl_utils.cl_SPV3_1D_to_3D(dcl_gc_1d, 'GC', nbl_WL_opt, zbins)[:nbl_GC, :, :]
    
for param in param_names_3x2pt:
    der_name = derivatives_filename.format(probe='3x2pt', param_name=param, **variable_specs)
    dcl_3x2pt_1d = np.genfromtxt(f'{derivatives_folder}/{der_name}')
    dC_dict_3x2pt_5D[param] = cl_utils.cl_SPV3_1D_to_3D(dcl_3x2pt_1d, '3x2pt', nbl_WL_opt, zbins)[:, :, :nbl_3x2pt, :, :]


# turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
dC_LL_4D = fm_utils.dC_dict_to_4D_array(dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, derivatives_prefix='')
dC_GG_4D = fm_utils.dC_dict_to_4D_array(dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, derivatives_prefix='')
dC_WA_4D = fm_utils.dC_dict_to_4D_array(dC_dict_WA_3D, param_names_3x2pt, nbl_WA, zbins, derivatives_prefix='')
# dC_WA_4D = np.ones((nbl_WA, zbins, zbins, dC_LL_4D.shape[-1]))
dC_3x2pt_6D = fm_utils.dC_dict_to_4D_array(
    dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins, derivatives_prefix='', is_3x2pt=True)

# free up memory
del dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_WA_3D, dC_dict_3x2pt_5D
gc.collect()

print('derivatives reshaped in 4D arrays in {:.2f} seconds'.format(time.perf_counter() - start_time))

# store the derivatives arrays in a dictionary
deriv_dict = {'dC_LL_4D': dC_LL_4D,
              'dC_WA_4D': dC_WA_4D,
              'dC_GG_4D': dC_GG_4D,
              'dC_3x2pt_6D': dC_3x2pt_6D}

# ! compute and save fisher matrix
fm_dict = fm_utils.compute_FM(general_cfg, covariance_cfg, fm_cfg, ell_dict, cov_dict, deriv_dict,
                              BNT_matrix)
fm_dict['param_names_dict'] = param_names_dict
fm_dict['fiducial_values_dict'] = fiducials_dict

fm_dict['fiducial_values_dict_v2'] = {}
for idx, param_name in enumerate(param_names_dict['cosmo']):
    fm_dict['fiducial_values_dict_v2'][param_name] = fiducials_dict['cosmo'][idx]
for idx, param_name in enumerate(param_names_dict['IA']):
    fm_dict['fiducial_values_dict_v2'][param_name] = fiducials_dict['IA'][idx]
for idx, param_name in enumerate(param_names_dict['galaxy_bias']):
    fm_dict['fiducial_values_dict_v2'][param_name] = fiducials_dict['galaxy_bias'][idx]
for idx, param_name in enumerate(param_names_dict['shear_bias']):
    fm_dict['fiducial_values_dict_v2'][param_name] = fiducials_dict['shear_bias'][idx]
for idx, param_name in enumerate(param_names_dict['dzWL']):
    fm_dict['fiducial_values_dict_v2'][param_name] = fiducials_dict['dzWL'][idx]
# for idx, param_name in enumerate(param_names_dict['dzGC']):
    # fm_dict['fiducial_values_dict_v2'][param_name] = fiducials_dict['dzGC'][idx]


fm_folder = fm_cfg['fm_folder'].format(ell_cuts=str(general_cfg['ell_cuts']),
                                       which_cuts=general_cfg['which_cuts'],
                                       center_or_min=general_cfg['center_or_min'])

fm_utils.save_FM(fm_folder, fm_dict, fm_cfg, cases_tosave, save_txt=fm_cfg['save_FM_txt'], save_dict=False,
                 **variable_specs)

if fm_cfg['save_FM_dict']:
    fm_dict_filename = fm_cfg['FM_dict_filename'].format(**variable_specs, nbl=nbl_3x2pt)
    mm.save_pickle(f'{fm_folder}/{fm_dict_filename}.pickle', fm_dict)


if fm_cfg['test_against_benchmarks']:
    mm.test_folder_content(fm_folder, fm_folder + '/benchmarks', 'txt')

del cov_dict
gc.collect()

# ! ================================ vincenzo 
fm_dict_vin = {}
fm_folder = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/FiRe/OutputQuantities/FishMat/Davide/Flat/HMCodeBar'
fm_dict_vin['FM_3x2pt_G'] = np.genfromtxt(f'{fm_folder}/fmnew-3x2pt-{EP_or_ED}{zbins:02d}-zedMin{general_cfg["zmin"]:02d}-zedMax{zmax}-mag{general_cfg["magcut"]}-GO-{which_case}.dat')
fm_dict_vin['FM_3x2pt_GSSC'] = np.genfromtxt(f'{fm_folder}/fmnew-3x2pt-{EP_or_ED}{zbins:02d}-zedMin{general_cfg["zmin"]:02d}-zedMax{zmax}-mag{general_cfg["magcut"]}-GS-{which_case}.dat')

fm_dict_vin['FM_WL_G'] = np.genfromtxt(f'{fm_folder}/fmnew-WLO-{EP_or_ED}{zbins:02d}-zedMin{general_cfg["zmin"]:02d}-zedMax{zmax}-mag{general_cfg["magcut"]}-GO-{which_case}.dat')
fm_dict_vin['FM_WL_GSSC'] = np.genfromtxt(f'{fm_folder}/fmnew-WLO-{EP_or_ED}{zbins:02d}-zedMin{general_cfg["zmin"]:02d}-zedMax{zmax}-mag{general_cfg["magcut"]}-GS-{which_case}.dat')

fm_dict_vin['FM_GC_G'] = np.genfromtxt(f'{fm_folder}/fmnew-GCO-{EP_or_ED}{zbins:02d}-zedMin{general_cfg["zmin"]:02d}-zedMax{zmax}-mag{general_cfg["magcut"]}-GO-{which_case}.dat')
fm_dict_vin['FM_GC_GSSC'] = np.genfromtxt(f'{fm_folder}/fmnew-GCO-{EP_or_ED}{zbins:02d}-zedMin{general_cfg["zmin"]:02d}-zedMax{zmax}-mag{general_cfg["magcut"]}-GS-{which_case}.dat')

fm_dict_vin['fiducial_values_dict_v2_3x2pt'] = deepcopy(fm_dict['fiducial_values_dict_v2'])
# Convert the dictionary to a list of tuples
params_list = list(fm_dict_vin['fiducial_values_dict_v2_3x2pt'].items())

# Find the index of 'logT'
index = next(i for i, v in enumerate(params_list) if v[0] == 'logT')

# Insert 'gamma': 0.55 before 'logT'
params_list.insert(index, ('gamma', 0.55))

# Convert the list of tuples back to a dictionary
fm_dict_vin['fiducial_values_dict_v2_3x2pt'] = dict(params_list)
fm_dict_vin['fiducial_values_dict_v2_WL'] = {param: fm_dict_vin['fiducial_values_dict_v2_3x2pt'][param] for param in param_names_wl_vin}
fm_dict_vin['fiducial_values_dict_v2_GC'] = {param: fm_dict_vin['fiducial_values_dict_v2_3x2pt'][param] for param in param_names_gc_vin}


# ! =========================== FM settings #######################################
nparams_toplot_ref = 8
names_params_to_fix = []
divide_fom_by_10 = True
include_fom = True
which_uncertainty = 'conditional'
remove_null_rows_cols = True

fix_dz = False
fix_shear_bias = False
fix_gal_bias = False
fix_logT = False
fix_gamma = True

nsigma_logT_prior = 3
shear_bias_prior = 5e-4  # 5e-4 or None
dz_prior = np.array(2 * 1e-3 * (1 + np.array(z_center_values)))
logT_prior = (8 - 7.6)/(nsigma_logT_prior*2)  # BAHAMAS range "converted" to Gaussian prior
# shear_bias_prior = None
# dz_prior = None
# logT_prior = None
# ! =========================== FM settings end #######################################

probes = ['WL', 'GC', '3x2pt']
dzWL_param_names = [f'dzWL{(zi + 1):02d}' for zi in range(zbins)]
shear_bias_param_names = [f'm{(zi + 1):02d}' for zi in range(zbins)]
gal_bias_param_names = [f'bG{(zi + 1):02d}' for zi in range(zbins)]
param_names_list = param_names_3x2pt
param_names_list_vin = param_names_3x2pt_vin

for name in dzWL_param_names:
    assert name in param_names_list, f"{name} not found in param_names_list"
for name in shear_bias_param_names:
    assert name in param_names_list, f"{name} not found in param_names_list"
for name in gal_bias_param_names:
    assert name in param_names_list, f"{name} not found in param_names_list"

if fix_dz:
    names_params_to_fix += dzWL_param_names

if fix_shear_bias:
    names_params_to_fix += shear_bias_param_names

if fix_gal_bias:
    names_params_to_fix += gal_bias_param_names

if fix_logT:
    names_params_to_fix += ['logT']
    
if fix_gamma:
    names_params_to_fix_vin = names_params_to_fix + ['gamma']
else:
    names_params_to_fix_vin = names_params_to_fix
    
names_params_to_fix_dict_vin = {}
names_params_to_fix_dict_vin['WL'] = names_params_to_fix_vin
names_params_to_fix_dict_vin['GC'] = [name for name in names_params_to_fix_vin if 'dzWL' not in name]
names_params_to_fix_dict_vin['GC'] = [name for name in names_params_to_fix_dict_vin['GC'] if not name.startswith('m')]
names_params_to_fix_dict_vin['3x2pt'] = names_params_to_fix_vin

fom_dict = {}
uncert_dict = {}
masked_fm_dict = {}
masked_fid_pars_dict = {}
fm_dict_toplot = deepcopy(fm_dict)

masked_fm_dict_vin = {}
masked_fid_pars_dict_vin = {}
fm_dict_toplot_vin = deepcopy(fm_dict_vin)

for key in list(fm_dict_toplot.keys()):
    if '_WA_' not in key and '_2x2pt_' not in key and '_XC_' not in key and 'fiducial_values_dict' not in key and 'param_names_dict' not in key:

        nparams_toplot = nparams_toplot_ref
        print(key)
        probe = key.split('_')[-2]
        
        fm = deepcopy(fm_dict_toplot[key])
        fm_vin = deepcopy(fm_dict_toplot_vin[key])
        
        masked_fm_dict[key], masked_fid_pars_dict[key] = mm.mask_fm_v2(fm, fm_dict['fiducial_values_dict_v2'],
                                                                       names_params_to_fix=names_params_to_fix,
                                                                       remove_null_rows_cols=remove_null_rows_cols)
        masked_fm_dict_vin[key], masked_fid_pars_dict_vin[key] = mm.mask_fm_v2(fm_vin, fm_dict_vin[f'fiducial_values_dict_v2_{probe}'],
                                                                       names_params_to_fix=names_params_to_fix_dict_vin[probe],
                                                                       remove_null_rows_cols=remove_null_rows_cols)
        
        if not fix_shear_bias and any(item in key for item in ['WL', 'XC', '3x2pt', '2x2pt']) and shear_bias_prior is not None:
            print(f'adding shear bias Gaussian prior to {key}')
            shear_bias_prior_values = np.array([shear_bias_prior] * zbins)
            masked_fm_dict[key] = mm.add_prior_to_fm(masked_fm_dict[key], masked_fid_pars_dict[key],
                                                     shear_bias_param_names, shear_bias_prior_values)
            masked_fm_dict_vin[key] = mm.add_prior_to_fm(masked_fm_dict_vin[key], masked_fid_pars_dict_vin[key],
                                                     shear_bias_param_names, shear_bias_prior_values)

        if not fix_dz and dz_prior is not None and any(item in key for item in ['WL', 'XC', '3x2pt', '2x2pt']):
            print(f'adding dz Gaussian prior to {key}')
            masked_fm_dict[key] = mm.add_prior_to_fm(
                masked_fm_dict[key], masked_fid_pars_dict[key], dzWL_param_names, dz_prior)
            masked_fm_dict_vin[key] = mm.add_prior_to_fm(
                masked_fm_dict_vin[key], masked_fid_pars_dict_vin[key], dzWL_param_names, dz_prior)
            
        if not fix_logT and logT_prior is not None:
            print(f'adding logT Gaussian prior to {key}')
            masked_fm_dict[key] = mm.add_prior_to_fm(
                masked_fm_dict[key], masked_fid_pars_dict[key], ['logT'], logT_prior)
            masked_fm_dict_vin[key] = mm.add_prior_to_fm(
                masked_fm_dict_vin[key], masked_fid_pars_dict_vin[key], ['logT'], logT_prior)

        # save for sylvain/vinc
        # np.savetxt(f'{fm_folder}/{key}_{which_case}_noPriors_cut{remove_null_rows_cols}.txt', masked_fm_dict[key])
        # np.savetxt(f'{fm_folder}/fid_param_values.txt', list(fm_dict['fiducial_values_dict_v2'].values()))
        # with open(f'{fm_folder}/fid_param_names.txt', 'w') as file:
        #     for param in list(fm_dict['fiducial_values_dict_v2'].values()):
        #         file.write(f"{param}\n")
        # continue
        
        uncert_dict[key] = mm.uncertainties_fm_v2(masked_fm_dict[key], masked_fid_pars_dict[key],
                                                  which_uncertainty=which_uncertainty,
                                                  normalize=True,
                                                  percent_units=True)[:nparams_toplot]
        uncert_dict[key + '_vin'] = mm.uncertainties_fm_v2(masked_fm_dict_vin[key], masked_fid_pars_dict_vin[key],
                                                  which_uncertainty=which_uncertainty,
                                                  normalize=True,
                                                  percent_units=True)[:nparams_toplot]

        param_names = list(masked_fid_pars_dict[key].keys())
        param_names_vin = list(masked_fid_pars_dict_vin[key].keys())
        cosmo_param_names = list(masked_fid_pars_dict[key].keys())[:nparams_toplot]

        w0wa_idxs = param_names.index('wz'), param_names.index('wa')
        fom_dict[key] = mm.compute_FoM(masked_fm_dict[key], w0wa_idxs=w0wa_idxs)
        fom_dict[key + '_vin'] = mm.compute_FoM(masked_fm_dict_vin[key], w0wa_idxs=w0wa_idxs)

# compute percent diff btw Gauss and G+SSC, using the respective Gaussian covariance
for probe in probes:
    nparams_toplot = nparams_toplot_ref

    key_a = f'FM_{probe}_G'
    key_b = f'FM_{probe}_GSSC'
    key_a_vin = f'FM_{probe}_G_vin'
    key_b_vin = f'FM_{probe}_GSSC_vin'

    uncert_dict[f'perc_diff_{probe}_G'] = mm.percent_diff(uncert_dict[key_b], uncert_dict[key_a])
    uncert_dict[f'ratio_{probe}_G'] = uncert_dict[key_b]/ uncert_dict[key_a]
    fom_dict[f'perc_diff_{probe}_G'] = np.abs(mm.percent_diff(fom_dict[key_b], fom_dict[key_a]))
    fom_dict[f'ratio_{probe}_G'] = fom_dict[key_b]/ fom_dict[key_a]
    
    uncert_dict[f'perc_diff_{probe}_G_vin'] = mm.percent_diff(uncert_dict[key_b_vin], uncert_dict[key_a_vin])
    uncert_dict[f'ratio_{probe}_G_vin'] = uncert_dict[key_b_vin]/ uncert_dict[key_a_vin]
    fom_dict[f'perc_diff_{probe}_G_vin'] = np.abs(mm.percent_diff(fom_dict[key_b_vin], fom_dict[key_a_vin]))
    fom_dict[f'ratio_{probe}_G_vin'] = fom_dict[key_b_vin]/ fom_dict[key_a_vin]

    divide_fom_by_10_plt = False if probe in ('WL' 'XC') else divide_fom_by_10

    cases_to_plot = [
        f'FM_{probe}_G',
        # f'FM_{probe}_G_vin',
        f'FM_{probe}_GSSC',
        # f'FM_{probe}_GSSC_vin',
        f'perc_diff_{probe}_G',
        # f'perc_diff_{probe}_G_vin',
    ]

    # # transform dict. into an array and add the fom
    uncert_array, fom_array = [], []

    for case in cases_to_plot:

        uncert_array.append(uncert_dict[case])
        
        if divide_fom_by_10 and 'FM' in case and 'WL' not in case:
            fom_dict[case] /= 10
            
        fom_array.append(fom_dict[case])

    uncert_array = np.asarray(uncert_array)
    fom_array = np.asarray(fom_array)

    uncert_array = np.hstack((uncert_array, fom_array.reshape(-1, 1)))

    # label and title stuff
    fom_label = 'FoM/10\nperc_diff' if divide_fom_by_10 else 'FoM'
    param_names_label = param_names_list[:nparams_toplot] + [fom_label] if include_fom else param_names_list[
        :nparams_toplot]
    lmax = general_cfg[f'ell_max_{probe}'] if probe in ['WL', 'GC'] else general_cfg['ell_max_3x2pt']
    title = '%s, $\\ell_{\\rm max} = %i$, zbins %s%i, $\\sigma_\\epsilon$ %s\nfix_shear_bias %s, fix_dz %s, fix_logT %s' % (
        probe, lmax, EP_or_ED, zbins, covariance_cfg['which_shape_noise'], str(fix_shear_bias), str(fix_dz), str(fix_logT))


    # prettify legend
    for i, case in enumerate(cases_to_plot):

        cases_to_plot[i] = case
        if f'PySSC_{probe}_G' in cases_to_plot[i]:
            cases_to_plot[i] = cases_to_plot[i].replace(f'PySSC_{probe}_G', f'{probe}_G')

        cases_to_plot[i] = cases_to_plot[i].replace(f'_{probe}', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'FM_', f'')
        cases_to_plot[i] = cases_to_plot[i].replace(f'_', f' ')
        cases_to_plot[i] = cases_to_plot[i].replace(f'GSSC', f'G+SSC')

    # bar plot
    if include_fom:
        nparams_toplot += 1
    plot_lib.bar_plot(uncert_array[:, :nparams_toplot], title, cases_to_plot, nparams=nparams_toplot,
                      param_names_label=param_names_label[:nparams_toplot], bar_width=0.13, include_fom=include_fom, divide_fom_by_10_plt=divide_fom_by_10_plt)


# ! Print tables
if include_fom:
    nparams_toplot = nparams_toplot_ref + 1 
titles = param_names_list[:nparams_toplot_ref] + ['FoM']

# for uncert_dict, _, name in zip([uncert_dict, uncert_dict], [fm_dict, fm_dict_vin], ['Davide', 'Vincenzo']):
# print(f"G uncertainties {name} [%]:")
# data = []
# for probe in probes:
#     uncerts = [f'{uncert:.3f}' for uncert in uncert_dict[f'FM_{probe}_G']]
#     fom = f'{fom_dict[f"FM_{probe}_G"]:.2f}'
#     data.append([probe] + uncerts + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))

print(f"GSSC/G ratio {which_uncertainty} {which_case}:")
data = []
table = []  # tor tex
for probe in probes:
    ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'ratio_{probe}_G']]
    fom = f'{fom_dict[f"ratio_{probe}_G"]:.2f}'
    data.append([probe] + ratios + [fom])
    table.append(ratios + [fom])
print(tabulate(data, headers=titles, tablefmt="pretty"))
a2l.to_ltx(np.array(table, dtype=float), frmt = '{:6.3f}', print_out=True)

print(f"SSC % increase {name}:")
data = []
for probe in probes:
    ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'perc_diff_{probe}_G']]
    fom = f'{fom_dict[f"perc_diff_{probe}_G"]:.2f}'
    data.append([probe] + ratios + [fom])
print(tabulate(data, headers=titles, tablefmt="pretty"))

# * invert ratio to check against barreira
# print(f"G/GSSC ratio {name}:")
# data = []
# for probe in probes:
#     ratios = [f'{ratio:.3f}' for ratio in uncert_dict[f'ratio_{probe}_G']**-1]
#     fom = f'{fom_dict[f"ratio_{probe}_G"]**-1:.2f}'
#     data.append([probe] + ratios + [fom])
# print(tabulate(data, headers=titles, tablefmt="pretty"))



# * 3x2pt cosmo is ok!
# TODO logT prior??

print('done')
