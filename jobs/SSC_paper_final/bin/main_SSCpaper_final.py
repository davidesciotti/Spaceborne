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
import pdb
from matplotlib import cm

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent
home_path = Path.home()
job_name = job_path.parts[-1]

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
import bin.cosmo_lib as csmlib
import bin.ell_values as ell_utils
import bin.cl_preprocessing as cl_utils
import bin.compute_Sijkl as Sijkl_utils
import bin.covariance as covmat_utils
import bin.fisher_matrix as FM_utils
import common_cfg.ISTF_fid_params as ISTFfid
import common_cfg.mpl_cfg as mpl_cfg


# job configuration
sys.path.append(f'{job_path}/config')
import config_SSCpaper_final as cfg


matplotlib.use('Qt5Agg')
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
                    '"ell-block ordering of the covariance matrix"')
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
FM_cfg = cfg.FM_cfg

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

# some checks
# assert general_cfg['flagship_version'] == 2, 'The input files used in this job for flagship version 2!'
assert general_cfg['use_WA'] is False, 'We do not use Wadd for SPV3 at the moment'

if covariance_cfg['cov_BNT_transform']:
    assert general_cfg[
        'cl_BNT_transform'] is False, 'the BNT transform should be applied either to the Cls ' \
        'or to the covariance'
    assert FM_cfg['derivatives_BNT_transform'], 'you should BNT transform the derivatives as well'

# which cases to save: GO, GS or GO, GS and SS
cases_tosave = ['GO', ]
if covariance_cfg[f'compute_SSC']:
    cases_tosave.append('GS')
if covariance_cfg[f'save_cov_SSC']:
    cases_tosave.append('SS')

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
    assert np.max(ell_dict[key]) > 15, 'ell values must *not* be in log space'

# set corresponding number of ell bins
nbl_WL = len(ell_dict['ell_WL'])
nbl_GC = len(ell_dict['ell_GC'])
nbl_WA = len(ell_dict['ell_WA'])
nbl_3x2pt = nbl_GC
general_cfg['nbl_WL'] = nbl_WL

delta_dict = {'delta_l_WL': np.copy(delta_l_WL_nbl32[:nbl_WL]),
              'delta_l_GC': np.copy(delta_l_WL_nbl32[:nbl_GC]),
              'delta_l_WA': np.copy(delta_l_WL_nbl32[nbl_GC:])}

# set # of nbl in the opt case, import and reshape, then cut the reshaped datavectors in the pes case
assert (general_cfg['ell_max_WL_opt'],
        general_cfg['ell_max_WL'],
        general_cfg['ell_max_GC'],
        general_cfg['ell_max_XC']) == (5000, 5000, 3000, 3000), \
    'the number of bins defined in the config file is compatible with these ell_max values'

nbl_WL_opt = general_cfg['nbl_WL_opt']
nbl_GC_opt = general_cfg['nbl_GC_opt']
nbl_WA_opt = general_cfg['nbl_WA_opt']
nbl_3x2pt_opt = general_cfg['nbl_3x2pt_opt']

if ell_max_WL == general_cfg['ell_max_WL_opt']:
    assert (nbl_WL_opt, nbl_GC_opt, nbl_WA_opt, nbl_3x2pt_opt) == (nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt), \
        'nbl_WL, nbl_GC, nbl_WA, nbl_3x2pt don\'t match with the expected values for the optimistic case'

# this is just to make the .format() more compact
variable_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins,
                  'ell_max_WL': ell_max_WL, 'ell_max_GC': ell_max_GC, 'ell_max_XC': ell_max_XC,
                  'nbl_WL': nbl_WL, 'nbl_GC': nbl_GC, 'nbl_WA': nbl_WA, 'nbl_3x2pt': nbl_3x2pt,
                  }

# ! import and reshape datavectors (cl) and response functions (rl)
cl_fld = general_cfg['cl_folder']
cl_filename = general_cfg['cl_filename']
cl_ll_1d = np.genfromtxt(
    f"{cl_fld.format(probe='WLO', which_pk=which_pk)}/{cl_filename.format(probe='WLO', nbl=nbl_WL, **variable_specs)}")
cl_gg_1d = np.genfromtxt(
    f"{cl_fld.format(probe='GCO', which_pk=which_pk)}/{cl_filename.format(probe='GCO', nbl=nbl_WL, **variable_specs)}")
cl_wa_1d = np.genfromtxt(
    f"{cl_fld.format(probe='WLA', which_pk=which_pk)}/{cl_filename.format(probe='WLA', nbl=nbl_WL, **variable_specs)}")
cl_3x2pt_1d = np.genfromtxt(
    f"{cl_fld.format(probe='3x2pt', which_pk=which_pk)}/{cl_filename.format(probe='3x2pt', nbl=nbl_WL, **variable_specs)}")


# reshape to 3 dimensions
cl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(cl_ll_1d, 'WL', nbl_WL_opt, zbins)
cl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(cl_gg_1d, 'GC', nbl_GC_opt, zbins)
cl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(cl_wa_1d, 'WA', nbl_WA_opt, zbins)
cl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(cl_3x2pt_1d, '3x2pt', nbl_3x2pt_opt, zbins)

# decrease font of the xticks
plt.rcParams.update({'xtick.labelsize': 19})
plt.rcParams.update({'ytick.labelsize': 19})
# decrease font of the x label
plt.rcParams.update({'axes.labelsize': 21})
# decrease font of the title
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
plt.savefig('/home/davide/Documenti/Lavoro/Programmi/phd_thesis_plots/plots/cls.pdf', dpi=500, bbox_inches='tight')


assert False, 'stop here and undo the latest changes with git, they were just to produce the cls plot'


ng_folder = covariance_cfg["ng_folder"]
ng_filename = f'{covariance_cfg["ng_filename"].format(**variable_specs)}'
ngtab = np.genfromtxt(f'{ng_folder}/'f'{ng_filename}')
z_center_values = ngtab[:, 0]
covariance_cfg['ng'] = ngtab[:, 1]
dzWL_fiducial = ngtab[:, 4]
dzGC_fiducial = ngtab[:, 4]

nofz_folder = covariance_cfg["nofz_folder"]
nofz_filename = f'{covariance_cfg["nofz_filename"].format(**variable_specs)}'
n_of_z = np.genfromtxt(f'{nofz_folder}/'f'{nofz_filename}')
zgrid_n_of_z = n_of_z[:, 0]
n_of_z = n_of_z[:, 1:]

# some check on the input nz files
assert np.all(covariance_cfg['ng'] < 5), 'ng values are likely < 5 *per bin*; this is just a rough check'
assert np.all(covariance_cfg['ng'] > 0), 'ng values must be positive'
assert np.all(z_center_values > 0), 'z_center values must be positive'
assert np.all(z_center_values < 3), 'z_center values are likely < 3; this is just a rough check'

# BNT_matrix_filename = general_cfg["BNT_matrix_filename"].format(**variable_specs)
# BNT_matrix = np.load(f'{general_cfg["BNT_matrix_path"]}/{BNT_matrix_filename}')

print('Computing BNT matrix...')
BNT_matrix = covmat_utils.compute_BNT_matrix(zbins, zgrid_n_of_z, n_of_z, plot_nz=False)

rl_fld = general_cfg['rl_folder']
rl_filename = general_cfg['rl_filename']
rl_ll_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLO', nbl=nbl_WL, **variable_specs)}")
rl_gg_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='GCO', nbl=nbl_WL, **variable_specs)}")
rl_wa_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLA', nbl=nbl_WL, **variable_specs)}")
rl_3x2pt_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='3x2pt', nbl=nbl_WL, **variable_specs)}")


rl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(rl_ll_1d, 'WL', nbl_WL_opt, zbins)
rl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(rl_gg_1d, 'GC', nbl_GC_opt, zbins)
rl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(rl_wa_1d, 'WA', nbl_WA_opt, zbins)
rl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(rl_3x2pt_1d, '3x2pt', nbl_3x2pt_opt, zbins)

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
    assert 1 > 2, 'you should check this'
    cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
    cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
    cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
    cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

    rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
    rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
    rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
    rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

# this is to pass the ll cuts to the covariance module
ell_cuts_dict = load_ell_cuts(kmax_h_over_Mpc)
ell_dict['ell_cuts_dict'] = ell_cuts_dict  # rename for better readability

# ! Vincenzo's method for cl_ell_cuts: get the idxs to delete for the flattened 1d cls
if general_cfg['center_or_min'] == 'center':
    prefix = 'ell'
elif general_cfg['center_or_min'] == 'min':
    prefix = 'ell_edges'
else:
    raise ValueError('general_cfg["center_or_min"] should be either "center" or "min"')

ell_dict['idxs_to_delete_dict'] = {
    'LL': get_idxs_to_delete(ell_dict[f'{prefix}_WL'], ell_cuts_dict['LL'], is_auto_spectrum=True),
    'GG': get_idxs_to_delete(ell_dict[f'{prefix}_GC'], ell_cuts_dict['GG'], is_auto_spectrum=True),
    'WA': get_idxs_to_delete(ell_dict[f'{prefix}_WA'], ell_cuts_dict['LL'], is_auto_spectrum=True),
    'GL': get_idxs_to_delete(ell_dict[f'{prefix}_XC'], ell_cuts_dict['GL'], is_auto_spectrum=False),
    'LG': get_idxs_to_delete(ell_dict[f'{prefix}_XC'], ell_cuts_dict['LG'], is_auto_spectrum=False),
    '3x2pt': get_idxs_to_delete_3x2pt(ell_dict[f'{prefix}_3x2pt'], ell_cuts_dict)
}

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

    # ! load kernels
    # TODO this should not be done if Sijkl is loaded; I have a problem with nz, which is part of the file name...
    wf_folder = Sijkl_cfg["wf_input_folder"].format(**variable_specs)
    wf_WL_filename = Sijkl_cfg["wf_WL_input_filename"]
    wf_GC_filename = Sijkl_cfg["wf_GC_input_filename"]
    wil = np.genfromtxt(f'{wf_folder}/{wf_WL_filename.format(**variable_specs)}')
    wig = np.genfromtxt(f'{wf_folder}/{wf_GC_filename.format(**variable_specs)}')

    # preprocess (remove redshift column)
    z_arr_wil, wil = Sijkl_utils.preprocess_wf(wil, zbins)
    z_arr_wig, wig = Sijkl_utils.preprocess_wf(wig, zbins)
    assert np.array_equal(z_arr_wil, z_arr_wig), \
        'the redshift arrays are different for the GC and WL kernels'
    z_arr = z_arr_wil

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
        Sijkl = Sijkl_utils.compute_Sijkl(csmlib.cosmo_par_dict_classy, z_arr, transp_stacked_wf,
                                          Sijkl_cfg['wf_normalization'])
        np.save(f'{Sijkl_folder}/{Sijkl_filename}', Sijkl)

else:
    warnings.warn('Sijkl is not computed, but set to identity')
    Sijkl = np.ones((n_probes * zbins, n_probes * zbins, n_probes * zbins, n_probes * zbins))

# ! compute covariance matrix
# the ng values are in the second column, for these input files 👇
# TODO: if already existing, don't compute the covmat, like done above for Sijkl
cov_dict = covmat_utils.compute_cov(general_cfg, covariance_cfg,
                                    ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, BNT_matrix)

# save covariance matrix and test against benchmarks
cov_folder = covariance_cfg['cov_folder'].format(cov_ell_cuts=str(covariance_cfg['cov_ell_cuts']),
                                                 **variable_specs)
covmat_utils.save_cov(cov_folder, covariance_cfg, cov_dict, **variable_specs)

if general_cfg['test_against_benchmarks']:
    cov_benchmark_folder = f'{cov_folder}/benchmarks'
    mm.test_folder_content(cov_folder, cov_benchmark_folder, covariance_cfg['cov_file_format'])

# ! compute Fisher matrix
if not FM_cfg['compute_FM']:
    # this guard is just to avoid indenting the whole code below
    raise KeyboardInterrupt('skipping FM computation, the script will exit now')

# set the fiducial values in a dictionary and a list
bias_fiducials = np.genfromtxt(f'{ng_folder}/gal_mag_fiducial_polynomial_fit.dat')
bias_fiducials_rows = np.where(bias_fiducials[:, 0] == general_cfg['magcut_source'] / 10)[
    0]  # take the correct magnitude limit
galaxy_bias_fit_fiducials = bias_fiducials[bias_fiducials_rows, 1]
magnification_bias_fit_fiducials = bias_fiducials[bias_fiducials_rows, 2]
fiducials_dict = {
    'cosmo': [ISTF_fid.primary['Om_m0'], ISTF_fid.primary['Om_b0'],
              ISTF_fid.primary['w_0'], ISTF_fid.primary['w_a'],
              ISTF_fid.primary['h_0'], ISTF_fid.primary['n_s'], ISTF_fid.primary['sigma_8'], 7.75],
    'IA': np.asarray([0.16, 1.66]),
    'shear_bias': np.zeros((zbins,)),
    'dzWL': dzWL_fiducial,  # for the time being, equal to the GC ones
    'galaxy_bias': galaxy_bias_fit_fiducials,
    'magnification_bias': magnification_bias_fit_fiducials,
}
fiducials_values_3x2pt = list(np.concatenate([fiducials_dict[key] for key in fiducials_dict.keys()]))

# set parameters' names, as a dict and as a list
param_names_dict = FM_cfg['param_names_dict']
param_names_3x2pt = FM_cfg['param_names_3x2pt']

assert param_names_dict.keys() == fiducials_dict.keys(), \
    'the parameter names and fiducial values dictionaries should have the same keys'

assert len(fiducials_values_3x2pt) == len(param_names_3x2pt), \
    'the fiducial values list and parameter names should have the same length'

# ! preprocess derivatives (or load the alreay preprocessed ones)
# import and store them in one big dictionary
start_time = time.perf_counter()
derivatives_folder = FM_cfg['derivatives_folder'].format(**variable_specs)

# check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
der_prefix = FM_cfg['derivatives_prefix']
vinc_filenames = mm.get_filenames_in_folder(derivatives_folder)
vinc_filenames = [vinc_filename for vinc_filename in vinc_filenames if vinc_filename.startswith(der_prefix)]

# perform some checks on the filenames before trimming them
for vinc_filename in vinc_filenames:
    assert f'{EP_or_ED}{zbins}' in vinc_filename, f'{EP_or_ED}{zbins} not in filename {vinc_filename}'
    assert f'ML{ML}' in vinc_filename, f'ML{ML} not in filename {vinc_filename}'
    assert f'MS{MS}' in vinc_filename, f'MS{MS} not in filename {vinc_filename}'

vinc_trimmed_filenames = [vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames]
vinc_trimmed_filenames = [vinc_trimmed_filename[len(der_prefix):] if vinc_trimmed_filename.startswith(der_prefix) else vinc_trimmed_filename
                          for vinc_trimmed_filename in vinc_trimmed_filenames]
vinc_param_names = list(set(vinc_trimmed_filenames))
vinc_param_names.sort()

my_sorted_param_names = param_names_3x2pt.copy()
my_sorted_param_names.sort()

# check whether the 2 lists match and print the elements that are in one list but not in the other
param_names_not_in_my_list = [vinc_param_name for vinc_param_name in vinc_param_names if
                              vinc_param_name not in my_sorted_param_names]
param_names_not_in_vinc_list = [my_sorted_param_name for my_sorted_param_name in my_sorted_param_names if
                                my_sorted_param_name not in vinc_param_names]
try:
    assert np.all(vinc_param_names == my_sorted_param_names), \
        f'\nparams present in input folder but not in the cfg file: {param_names_not_in_my_list}\n' \
        f'params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}'
except AssertionError as error:
    print(error)
    if param_names_not_in_vinc_list == ['logT_AGN']:
        print('the derivative w.r.t logT_AGN is missing in the input folder but '
              'the corresponding FM is still set to 0; moving on')
    else:
        raise AssertionError('there is something wrong with the parameter names in the derivatives folder')

if FM_cfg['load_preprocess_derivatives']:
    dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
    dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
    dC_WA_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_WA_4D.npy')
    dC_3x2pt_6D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy')

elif not FM_cfg['load_preprocess_derivatives']:
    der_prefix = FM_cfg['derivatives_prefix']
    dC_dict_1D = dict(mm.get_kv_pairs(derivatives_folder, "dat"))
    # check if dictionary is empty
    if not dC_dict_1D:
        raise ValueError(f'No derivatives found in folder {derivatives_folder}')

    # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
    dC_dict_LL_3D = {}
    dC_dict_GG_3D = {}
    dC_dict_WA_3D = {}
    dC_dict_3x2pt_5D = {}
    for key in dC_dict_1D.keys():
        if 'WLO' in key:
            dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WL', nbl_WL, zbins)
        elif 'GCO' in key:
            dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'GC', nbl_GC, zbins)
        elif 'WLA' in key:
            dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WA', nbl_WA, zbins)
        elif '3x2pt' in key:
            dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], '3x2pt', nbl_3x2pt, zbins)

    # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
    dC_LL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix)
    dC_GG_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix)
    dC_WA_4D = FM_utils.dC_dict_to_4D_array(dC_dict_WA_3D, param_names_3x2pt, nbl_WA, zbins, der_prefix)
    dC_3x2pt_6D = FM_utils.dC_dict_to_4D_array(dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins,
                                               der_prefix, is_3x2pt=True)

    # free up memory
    del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_WA_3D, dC_dict_3x2pt_5D
    gc.collect()

    print('derivatives reshaped in 4D arrays in {:.2f} seconds'.format(time.perf_counter() - start_time))

    # save these so they can simply be imported!
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_LL_4D.npy', dC_LL_4D)
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_GG_4D.npy', dC_GG_4D)
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_WA_4D.npy', dC_WA_4D)
    np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_3x2pt_6D.npy', dC_3x2pt_6D)

else:
    raise ValueError('"load_preprocess_derivatives" can only be True or False')

# store the derivatives arrays in a dictionary
deriv_dict = {'dC_LL_4D': dC_LL_4D,
              'dC_WA_4D': dC_WA_4D,
              'dC_GG_4D': dC_GG_4D,
              'dC_3x2pt_6D': dC_3x2pt_6D}

# ! compute and save fisher matrix
FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, FM_cfg, ell_dict, cov_dict, deriv_dict,
                              BNT_matrix)
FM_dict['param_names_dict'] = param_names_dict
FM_dict['fiducial_values_dict'] = fiducials_dict

fm_folder = FM_cfg['fm_folder'].format(ell_cuts=str(general_cfg['ell_cuts']),
                                       which_cuts=general_cfg['which_cuts'],
                                       center_or_min=general_cfg['center_or_min'])

FM_utils.save_FM(fm_folder, FM_dict, FM_cfg, cases_tosave, FM_cfg['save_FM_txt'], FM_cfg['save_FM_dict'],
                 **variable_specs)

if FM_cfg['test_against_benchmarks']:
    mm.test_folder_content(fm_folder, fm_folder + '/benchmarks', 'txt')

del cov_dict
gc.collect()

print('Script end')

"""
# ! save cls and responses:
# TODO this should go inside a function too
# this is just to set the correct probe names
probe_dav_dict = {'WL': 'LL_3D',
                  'GC': 'GG_3D',
                  'WA': 'WA_3D',
                  '3x2pt': '3x2pt_5D'}

# just a dict for the output file names
clrl_dict = {'cl_dict_3D': cl_dict_3D,
             'rl_dict_3D': rl_dict_3D,
             'cl_inputname': 'dv',
             'rl_inputname': 'rf',
             'cl_dict_key': 'C',
             'rl_dict_key': 'R'}
for cl_or_rl in ['cl', 'rl']:
    if general_cfg[f'save_{cl_or_rl}s_3d']:

        for probe_vinc, probe_dav in zip(['WLO', 'GCO', '3x2pt', 'WLA'], ['WL', 'GC', '3x2pt', 'WA']):
            # save cl and/or response; not very readable but it works, plus all the cases are in the for loop

            filepath = f'{general_cfg[f"{cl_or_rl}_folder"]}/3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}'
            filename = general_cfg[f'{cl_or_rl}_filename'].format(
                probe=probe_vinc, **variable_specs).replace(".dat", "_3D.npy")
            file = clrl_dict[f"{cl_or_rl}_dict_3D"][
                f'{clrl_dict[f"{cl_or_rl}_dict_key"]}_{probe_dav_dict[probe_dav]}']
            np.save(f'{filepath}/{filename}', file)

            # save ells and deltas
            if probe_dav != '3x2pt':  # no 3x2pt in ell_dict, it's the same as GC
                filepath = f'{general_cfg[f"{cl_or_rl}_folder"]}/' \
                           f'3D_reshaped_BNT_{general_cfg["cl_BNT_transform"]}/{probe_vinc}'
                ells_filename = f'ell_{probe_dav}_ellmaxWL{ell_max_WL}'
                np.savetxt(f'{filepath}/{ells_filename}.txt', ell_dict[f'ell_{probe_dav}'])
                np.savetxt(f'{filepath}/delta_{ells_filename}.txt', delta_dict[f'delta_l_{probe_dav}'])

"""
