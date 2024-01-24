import sys
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import matplotlib.lines as mlines
import gc
import matplotlib.gridspec as gridspec
import yaml
from scipy.ndimage import gaussian_filter1d
import pprint
from copy import deepcopy
import numpy.testing as npt
pp = pprint.PrettyPrinter(indent=4)


import os
ROOT = os.getenv('ROOT')
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.ell_values as ell_utils
import bin.cl_preprocessing as cl_utils
import bin.compute_Sijkl as Sijkl_utils
import bin.covariance as covmat_utils
import bin.fisher_matrix as FM_utils
import bin.my_module as mm
import bin.cosmo_lib as csmlib
import bin.wf_cl_lib as wf_cl_lib
import common_cfg.mpl_cfg as mpl_cfg

# job config
import jobs.SPV3_magcut_zcut_thesis.config.config_SPV3_magcut_zcut_thesis as cfg


plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
script_start_time = time.perf_counter()
os.environ['OMP_NUM_THREADS'] = '4'


# TODO check that the number of ell bins is the same as in the files
# TODO double check the delta values
# TODO update consistency_checks

# TODO reorder all these cutting functions...
# TODO recompute Sijkl to be safe
# TODO redefine the last delta value
# TODO check what happens for ell_cuts_LG (instead of GL) = ell_cuts_XC file
# TODO cut if ell > ell_edge_lower (!!)


###############################################################################
#################### PARAMETERS AND SETTINGS DEFINITION #######################
###############################################################################

def load_ell_cuts(kmax_h_over_Mpc, z_values_a, z_values_b):
    """loads ell_cut values, rescales them and load into a dictionary.
    z_values_a: redshifts at which to compute the ell_max for a fiven Limber wavenumber, for probe A
    z_values_b: redshifts at which to compute the ell_max for a fiven Limber wavenumber, for probe B
    """
    if kmax_h_over_Mpc is None:
        kmax_h_over_Mpc = general_cfg['kmax_h_over_Mpc_ref']

    if general_cfg['which_cuts'] == 'Francis':

        raise Exception('I want the output to be an array, see the Vincenzo case. probebly best to split these 2 funcs')
        assert general_cfg['EP_or_ED'] == 'ED', 'Francis cuts are only available for the ED case'

        ell_cuts_fldr = general_cfg['ell_cuts_folder']
        ell_cuts_filename = general_cfg['ell_cuts_filename']
        kmax_h_over_Mpc_ref = general_cfg['kmax_h_over_Mpc_ref']

        ell_cuts_LL = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="WL", **variable_specs)}')
        ell_cuts_GG = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="GC", **variable_specs)}')
        warnings.warn('I am not sure this ell_cut file is for GL, the filename is "XC"')
        ell_cuts_GL = np.genfromtxt(f'{ell_cuts_fldr}/{ell_cuts_filename.format(probe="XC", **variable_specs)}')
        ell_cuts_LG = ell_cuts_GL.T

        # ! linearly rescale ell cuts
        warnings.warn('is this the issue with the BNT? kmax_h_over_Mpc_ref is 1, I think...')
        ell_cuts_LL *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
        ell_cuts_GG *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
        ell_cuts_GL *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref
        ell_cuts_LG *= kmax_h_over_Mpc / kmax_h_over_Mpc_ref

        ell_cuts_dict = {
            'LL': ell_cuts_LL,
            'GG': ell_cuts_GG,
            'GL': ell_cuts_GL,
            'LG': ell_cuts_LG
        }

    elif general_cfg['which_cuts'] == 'Vincenzo':
        # the "Limber", or "standard" cuts

        kmax_1_over_Mpc = kmax_h_over_Mpc * h

        ell_cuts_array = np.zeros((zbins, zbins))
        for zi, zval_i in enumerate(z_values_a):
            for zj, zval_j in enumerate(z_values_b):
                r_of_zi = csmlib.ccl_comoving_distance(zval_i, use_h_units=False, cosmo_ccl=cosmo_ccl)
                r_of_zj = csmlib.ccl_comoving_distance(zval_j, use_h_units=False, cosmo_ccl=cosmo_ccl)
                ell_cut_i = kmax_1_over_Mpc * r_of_zi - 1 / 2
                ell_cut_j = kmax_1_over_Mpc * r_of_zj - 1 / 2
                ell_cuts_array[zi, zj] = np.min((ell_cut_i, ell_cut_j))

        return ell_cuts_array

    else:
        raise Exception('which_cuts must be either "Francis" or "Vincenzo"')

    return ell_cuts_dict


def cl_ell_cut_wrap(ell_dict, cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d, kmax_h_over_Mpc):
    """Wrapper for the ell cuts. Avoids the 'if general_cfg['cl_ell_cuts']' in the main loop
    (i.e., we use extraction)"""

    if not general_cfg['cl_ell_cuts']:
        return cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d

    warnings.warn('restore this?')
    # raise Exception('I decided to implement the cuts in 1dim, this function should not be used')

    print('Performing the cl ell cuts...')

    cl_ll_3d = cl_utils.cl_ell_cut(cl_ll_3d, ell_dict['ell_WL'], ell_cuts_dict['LL'])
    cl_wa_3d = cl_utils.cl_ell_cut(cl_wa_3d, ell_dict['ell_WA'], ell_cuts_dict['LL'])
    cl_gg_3d = cl_utils.cl_ell_cut(cl_gg_3d, ell_dict['ell_GC'], ell_cuts_dict['GG'])
    cl_3x2pt_5d = cl_utils.cl_ell_cut_3x2pt(cl_3x2pt_5d, ell_cuts_dict, ell_dict['ell_3x2pt'])

    return cl_ll_3d, cl_wa_3d, cl_gg_3d, cl_3x2pt_5d


def get_idxs_to_delete(ell_values, ell_cuts, is_auto_spectrum):
    """ ell_values can be the bin center or the bin lower edge; Francis suggests the second option is better"""

    if is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_val in ell_values:
            for zi in range(zbins):
                for zj in range(zi, zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1

    elif not is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_val in ell_values:
            for zi in range(zbins):
                for zj in range(zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1
    else:
        raise ValueError('is_auto_spectrum must be True or False')

    return idxs_to_delete


def get_idxs_to_delete_3x2pt(ell_values_3x2pt, ell_cuts_dict):
    """this function tries to implement the indexing for the flattening ell_probe_zpair"""

    if (covariance_cfg['triu_tril'], covariance_cfg['row_col_major']) != ('triu', 'row-major'):
        raise Exception('This function is only implemented for the triu, row-major case')

    idxs_to_delete_3x2pt = []
    count = 0
    for ell_val in ell_values_3x2pt:
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


def plot_nz_tocheck_func(zgrid_nz, n_of_z):
    if not covariance_cfg['plot_nz_tocheck']:
        return
    plt.figure()
    for zi in range(zbins):
        plt.plot(zgrid_nz, n_of_z[:, zi], label=f'zbin {zi}')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel('n(z)')


def plot_ell_cuts_for_thesis(ell_cuts_a, ell_cuts_b, ell_cuts_c, label_a, label_b, label_c, kmax_h_over_Mpc):
    # Get the global min and max values for the color scale
    vmin = min(ell_cuts_a.min(), ell_cuts_b.min(), ell_cuts_c.min())
    vmax = max(ell_cuts_a.max(), ell_cuts_b.max(), ell_cuts_c.min())

    # Create a gridspec layout
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.12])

    # Create axes based on the gridspec layout
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    cbar_ax = plt.subplot(gs[3])

    ticks = np.arange(1, zbins + 1)
    # Set x and y ticks for both subplots
    for ax in [ax0, ax1, ax2]:
        ax.set_xticks(np.arange(zbins))
        ax.set_yticks(np.arange(zbins))
        ax.set_xticklabels(ticks, fontsize=15)
        ax.set_yticklabels(ticks, fontsize=15)
        ax.set_xlabel('$z_{\\rm bin}$', fontsize=15)
        ax.set_ylabel('$z_{\\rm bin}$', fontsize=15)

    # Display the matrices with the shared color scale
    cax0 = ax0.matshow(ell_cuts_a, vmin=vmin, vmax=vmax)
    cax1 = ax1.matshow(ell_cuts_b, vmin=vmin, vmax=vmax)
    cax2 = ax2.matshow(ell_cuts_c, vmin=vmin, vmax=vmax)

    # Add titles to the plots
    ax0.set_title(label_a, fontsize=18)
    ax1.set_title(label_b, fontsize=18)
    ax2.set_title(label_c, fontsize=18)
    fig.suptitle(f'{mpl_cfg.kmax_tex} = {kmax_h_over_Mpc:.2f} {mpl_cfg.h_over_mpc_tex}', fontsize=18, y=0.85)

    # Add a shared colorbar on the right
    cbar = fig.colorbar(cax0, cax=cbar_ax)
    cbar.set_label('$\\ell^{\\rm max}_{ij}$', fontsize=15, loc='center', )
    cbar.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.show()


def plot_kernels_for_thesis():
    plt.figure()
    for zi in range(zbins):
        # if zi in [2, 10]:
        #     plt.axvline(z_means_ll[zi], ls='-', c=colors[zi], ymin=0, lw=2, zorder=1)
        #     plt.axvline(z_means_ll_bnt[zi], ls='--', c=colors[zi], ymin=0, lw=2, zorder=1)
        # plt.axvline(z_means[zi], ls='-', c=colors[zi], ymin=0, lw=2, zorder=1)

        plt.plot(zgrid_nz, wf_ll_ccl[:, zi], ls='-', c=colors[zi], alpha=0.6)
        plt.plot(zgrid_nz, wf_ll_ccl_bnt[:, zi], ls='-', c=colors[zi], alpha=0.6)

        plt.plot(zgrid_wf_vin, wf_ll_vin[:, zi], ls=':', label='$z_{%d}$' % (zi + 1), c=colors[zi], alpha=0.6)
        plt.plot(zgrid_wf_vin, wf_ll_vin_bnt[:, zi], ls=':', c=colors[zi], alpha=0.6)

    plt.title(
        f'interpolation_kind {shift_nz_interpolation_kind}, use_ia {include_ia_in_bnt_kernel_for_zcuts}, sigma_gauss {nz_gaussian_smoothing_sigma}\n'
        f'shift_dz {shift_nz}')
    plt.xlabel('$z$')
    plt.ylabel('${\cal K}_i^{\; \gamma}(z)^ \ \\rm{[Mpc^{-1}]}$')

    # Create the first legend
    ls_dict = {'--': 'standard',
               '-': 'BNT',
               ':': '$z_{\\rm mean}$'}

    handles = []
    for ls, label in ls_dict.items():
        handles.append(mlines.Line2D([], [], color='black', linestyle=ls, label=label))
    first_legend = plt.legend(handles=handles, loc='upper right')
    ax = plt.gca().add_artist(first_legend)
    plt.legend(loc='lower right')


def nz_gaussian_smmothing_func(zgrid_nz, n_of_z, plot=True):

    print(f'Applying a Gaussian filter of sigma = {nz_gaussian_smoothing_sigma} to the n(z)')
    n_of_z = gaussian_filter1d(n_of_z, nz_gaussian_smoothing_sigma, axis=0)

    if plot:
        plt.figure()
        for zi in range(zbins):
            plt.plot(zgrid_nz, n_of_z[:, zi], label=f'zbin {zi}', c=colors[zi], ls='-')
            plt.plot(zgrid_nz, n_of_z[:, zi], c=colors[zi], ls='--')
        plt.title(f'Gaussian filter w/ sigma = {nz_gaussian_smoothing_sigma}')

    return n_of_z

# * ====================================================================================================================
# * ====================================================================================================================
# * ====================================================================================================================


general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg
Sijkl_cfg = cfg.Sijkl_cfg
fm_cfg = cfg.FM_cfg
kmax_fom_400_ellcenter = 2.15443469


zbins = general_cfg['zbins']
EP_or_ED = general_cfg['EP_or_ED']

# for kmax_h_over_Mpc in general_cfg['kmax_h_over_Mpc_list']:
#     for general_cfg['which_pk'] in general_cfg['which_pk_list']:
for zbins in (13, ):

    # some convenence variables, just to make things more readable
    general_cfg['zbins'] = zbins
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    ell_max_3x2pt = general_cfg['ell_max_3x2pt']
    magcut_source = general_cfg['magcut_source']
    magcut_lens = general_cfg['magcut_lens']
    zcut_source = general_cfg['zcut_source']
    zcut_lens = general_cfg['zcut_lens']
    flat_or_nonflat = general_cfg['flat_or_nonflat']
    center_or_min = general_cfg['center_or_min']
    zmax = int(general_cfg['zmax'] * 10)
    triu_tril = covariance_cfg['triu_tril']
    row_col_major = covariance_cfg['row_col_major']
    GL_or_LG = covariance_cfg['GL_or_LG']
    n_probes = general_cfg['n_probes']
    which_pk = general_cfg['which_pk']
    idIA = general_cfg['idIA']
    idB = general_cfg['idB']
    idM = general_cfg['idM']
    idR = general_cfg['idR']
    idBM = general_cfg['idBM']
    which_ng_cov_suffix = 'G' + ''.join(covariance_cfg[covariance_cfg['SSC_code'] + '_cfg']['which_ng_cov'])
    bnt_transform = general_cfg['BNT_transform']
    shift_nz_interpolation_kind = covariance_cfg['shift_nz_interpolation_kind']
    nz_gaussian_smoothing = covariance_cfg['nz_gaussian_smoothing']  # does not seem to have a large effect...
    nz_gaussian_smoothing_sigma = covariance_cfg['nz_gaussian_smoothing_sigma']
    shift_nz = covariance_cfg['shift_nz']  # ! are vincenzo's kernels shifted?? it looks like they are not
    normalize_shifted_nz = covariance_cfg['normalize_shifted_nz']
    compute_bnt_with_shifted_nz_for_zcuts = covariance_cfg['compute_bnt_with_shifted_nz_for_zcuts']  # ! let's test this
    include_ia_in_bnt_kernel_for_zcuts = covariance_cfg['include_ia_in_bnt_kernel_for_zcuts']
    nbl_WL_opt = general_cfg['nbl_WL_opt']
    colors = cm.rainbow(np.linspace(0, 1, zbins))

    with open(general_cfg['fid_yaml_filename'].format(zbins=zbins)) as f:
        fid_pars_dict = yaml.safe_load(f)
    flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)
    general_cfg['flat_fid_pars_dict'] = flat_fid_pars_dict

    h = flat_fid_pars_dict['h']
    general_cfg['fid_pars_dict'] = fid_pars_dict

    cosmo_dict_ccl = csmlib.map_keys(mm.flatten_dict(fid_pars_dict), key_mapping=None)
    cosmo_ccl = csmlib.instantiate_cosmo_ccl_obj(cosmo_dict_ccl,
                                                 fid_pars_dict['other_params']['camb_extra_parameters'])

    # some checks
    assert general_cfg['flagship_version'] == 2, 'The input files used in this job for flagship version 2!'
    assert general_cfg['use_WA'] is False, 'We do not use Wadd for SPV3 at the moment'
    assert general_cfg[
        'flat_or_nonflat'] == 'Flat', 'We do not use non-flat cosmologies for SPV3 at the moment, if I recall correclty'
    assert general_cfg['which_cuts'] == 'Vincenzo', ('to begin with, use only Vincenzo/standard cuts. '
                                                     'For the thesis, probably use just these')
    if general_cfg['ell_cuts']:
        assert bnt_transform, 'you should BNT transform if you want to apply ell cuts'
    if bnt_transform:
        assert general_cfg['ell_cuts'] is False, 'you should not apply ell cuts if you want to BNT transform'

    if covariance_cfg['cov_BNT_transform']:
        assert general_cfg['cl_BNT_transform'] is False, \
            'the BNT transform should be applied either to the Cls or to the covariance'
        assert fm_cfg['derivatives_BNT_transform'], 'you should BNT transform the derivatives as well'

    assert (ell_max_WL, ell_max_GC) == (5000, 3000) or (1500, 750), \
        'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

    assert magcut_lens == 245, 'magcut_lens must be 245: the yaml file with the fiducial params is for magcut 245'
    assert magcut_source == 245, 'magcut_source must be 245: the yaml file with the fiducial params is for magcut 245'

    warnings.warn('find a better way to treat with the various ng covariances')
    # which cases to save: GO, GS or GO, GS and SS
    cases_tosave = ['GO', ]
    if covariance_cfg[f'compute_SSC']:
        # cases_tosave.append('G' + covariance_cfg[covariance_cfg['SSC_code'] + '_cfg']['which_ng_cov'])
        cases_tosave.append('GS')
    if covariance_cfg[f'save_cov_SSC']:
        cases_tosave.append('SS')

    # build the ind array and store it into the covariance dictionary
    ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
    covariance_cfg['ind'] = ind
    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    covariance_cfg['probe_ordering'] = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))

    if not general_cfg['ell_cuts']:
        general_cfg['ell_cuts_subfolder'] = ''
        kmax_h_over_Mpc = general_cfg['kmax_h_over_Mpc_ref']
    else:
        general_cfg['ell_cuts_subfolder'] = f'{general_cfg["which_cuts"]}/ell_{general_cfg["center_or_min"]}'

    # compute ell and delta ell values in the reference (optimistic) case
    assert general_cfg['nbl_WL_opt'] == 32, 'this is used as the reference binning, from which the cuts are made'
    assert general_cfg['ell_max_WL_opt'] == 5000, 'this is used as the reference binning, from which the cuts are made'
    ell_ref_nbl32, delta_l_ref_nbl32, ell_edges_ref_nbl32 = (
        ell_utils.compute_ells(general_cfg['nbl_WL_opt'], general_cfg['ell_min'], general_cfg['ell_max_WL_opt'],
                               recipe='ISTF', output_ell_bin_edges=True))

    # perform the cuts (not the redshift-dependent ones!) on the ell centers and edges
    ell_dict = {}
    ell_dict['ell_WL'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_WL])
    ell_dict['ell_GC'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_GC])
    ell_dict['ell_3x2pt'] = np.copy(ell_ref_nbl32[ell_ref_nbl32 < ell_max_3x2pt])
    ell_dict['ell_WA'] = np.copy(ell_ref_nbl32[(ell_ref_nbl32 > ell_max_GC) & (ell_ref_nbl32 < ell_max_WL)])
    ell_dict['ell_XC'] = np.copy(ell_dict['ell_3x2pt'])

    # store edges *except last one for dimensional consistency* in the ell_dict
    ell_dict['ell_edges_WL'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_WL])[:-1]
    ell_dict['ell_edges_GC'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_GC])[:-1]
    ell_dict['ell_edges_3x2pt'] = np.copy(ell_edges_ref_nbl32[ell_edges_ref_nbl32 < ell_max_3x2pt])[:-1]
    ell_dict['ell_edges_XC'] = np.copy(ell_dict['ell_edges_3x2pt'])
    ell_dict['ell_edges_WA'] = np.copy(
        ell_edges_ref_nbl32[(ell_edges_ref_nbl32 > ell_max_GC) & (ell_edges_ref_nbl32 < ell_max_WL)])[:-1]

    for key in ell_dict.keys():
        if ell_dict[key].size > 0:  # Check if the array is non-empty
            assert np.max(ell_dict[key]) > 15, f'ell values for key {key} must *not* be in log space'

    # set the corresponding number of ell bins
    nbl_WL = len(ell_dict['ell_WL'])
    nbl_GC = len(ell_dict['ell_GC'])
    nbl_WA = len(ell_dict['ell_WA'])
    nbl_3x2pt = nbl_GC
    # ! the main should not change the cfg...
    general_cfg['nbl_WL'] = nbl_WL
    general_cfg['nbl_GC'] = nbl_GC
    general_cfg['nbl_3x2pt'] = nbl_3x2pt

    delta_dict = {'delta_l_WL': np.copy(delta_l_ref_nbl32[:nbl_WL]),
                  'delta_l_GC': np.copy(delta_l_ref_nbl32[:nbl_GC]),
                  'delta_l_WA': np.copy(delta_l_ref_nbl32[nbl_GC:])}

    # this is just to make the .format() more compact
    variable_specs = {'EP_or_ED': EP_or_ED, 'zbins': zbins, 'magcut_lens': magcut_lens,
                      'zcut_lens': zcut_lens,
                      'magcut_source': magcut_source, 'zcut_source': zcut_source, 'zmax': zmax,
                      'ell_max_WL': ell_max_WL, 'ell_max_GC': ell_max_GC, 'ell_max_3x2pt': ell_max_3x2pt,
                      'nbl_WL': nbl_WL, 'nbl_GC': nbl_GC, 'nbl_WA': nbl_WA, 'nbl_3x2pt': nbl_3x2pt,
                      'kmax_h_over_Mpc': kmax_h_over_Mpc, 'center_or_min': center_or_min,
                      'idIA': idIA, 'idB': idB, 'idM': idM, 'idR': idR, 'idBM': idBM,
                      'flat_or_nonflat': flat_or_nonflat,
                      'which_pk': which_pk, 'BNT_transform': bnt_transform,
                      'which_ng_cov': which_ng_cov_suffix,
                      'ng_cov_code': covariance_cfg['SSC_code'],
                      'which_grids': covariance_cfg[covariance_cfg['SSC_code'] + '_cfg']['which_grids']
                      }
    pp.pprint(variable_specs)

    # ! import nuisance, to get fiducials and to shift the distribution
    nuisance_filename = covariance_cfg['nuisance_filename'].format(**variable_specs)
    nuisance_tab = np.genfromtxt(f'{covariance_cfg["nuisance_folder"]}/{nuisance_filename}')
    # this is not exactly equal to the result of wf_cl_lib.get_z_mean...
    z_means = nuisance_tab[:, 0]
    covariance_cfg['ng'] = nuisance_tab[:, 1]
    dzWL_fiducial = nuisance_tab[:, 4]
    dzGC_fiducial = nuisance_tab[:, 4]

    # get galaxy and magnification bias fiducials
    bias_fiducials = np.genfromtxt(f'{covariance_cfg["nuisance_folder"]}/gal_mag_fiducial_polynomial_fit.dat')
    # take the correct magnitude limit
    bias_fiducials_rows = np.where(bias_fiducials[:, 0] == general_cfg['magcut_source'] / 10)[0]
    galaxy_bias_fit_fiducials = bias_fiducials[bias_fiducials_rows, 1]
    magnification_bias_fit_fiducials = bias_fiducials[bias_fiducials_rows, 2]

    # check that the values in the yml file match the ones in the "gal_mag_fiducial_polynomial_fit" file
    gal_bias_fit_fiducials_names = [f'bG{zi:02d}' for zi in range(1, 5)]
    mag_bias_fit_fiducials_names = [f'bM{zi:02d}' for zi in range(1, 5)]
    galaxy_bias_fit_fiducials_yml = np.array([flat_fid_pars_dict[gal_bias_fit_fiducials_names[zi]] for zi in range(4)])
    mag_bias_fit_fiducials_yml = np.array([flat_fid_pars_dict[mag_bias_fit_fiducials_names[zi]] for zi in range(4)])
    np.testing.assert_array_equal(galaxy_bias_fit_fiducials, galaxy_bias_fit_fiducials_yml,
                                  err_msg='galaxy bias fiducials do not match')
    np.testing.assert_array_equal(magnification_bias_fit_fiducials, mag_bias_fit_fiducials_yml,
                                  err_msg='magnification bias fiducials do not match')

    # some check on the input nuisance values
    assert np.all(covariance_cfg['ng'] < 9), 'ng values are likely < 9 *per bin*; this is just a rough check'
    assert np.all(covariance_cfg['ng'] > 0), 'ng values must be positive'
    assert np.all(z_means > 0), 'z_center values must be positive'
    assert np.all(z_means < 3), 'z_center values are likely < 3; this is just a rough check'
    assert np.all(dzWL_fiducial == dzGC_fiducial), 'dzWL and dzGC shifts do not match'

    # just a check, to be sure that the nuisance file is the same one defined in the yaml file
    dz_shifts_names = [f'dzWL{zi:02d}' for zi in range(1, zbins + 1)]
    dz_shifts = np.array([flat_fid_pars_dict[dz_shifts_names[zi]] for zi in range(zbins)])
    np.testing.assert_array_equal(dz_shifts, dzWL_fiducial,
                                  err_msg='dzWL shifts do not match with the ones from the yml file')

    if EP_or_ED == 'ED':
        raise Exception('you should re-check the nz shifts in the yml fiducial for the ED case!!')
    # for zi in range(1, zbins + 1):
    #     fid_pars_dict['FM_ordered_params'][f'dzWL{zi:02d}'] = dzWL_fiducial[zi - 1].item()

    # warnings.warn('You should remove this, stop overwriting the yaml files
    # with open(general_cfg['fid_yaml_filename'].format(zbins=zbins), 'w') as f:
    #     yaml.dump(fid_pars_dict, f, sort_keys=False)

    # ! import n(z), for the BNT and the scale cuts
    nofz_folder = covariance_cfg["nofz_folder"]
    nofz_filename = covariance_cfg["nofz_filename"].format(**variable_specs)
    n_of_z = np.genfromtxt(f'{nofz_folder}/{nofz_filename}')
    zgrid_nz = n_of_z[:, 0]
    n_of_z = n_of_z[:, 1:]
    n_of_z_original = n_of_z

    # ! load vincenzo's kernels, including magnification bias, IA and n(z) shifts!
    wf_folder = Sijkl_cfg['wf_input_folder']
    wf_filename = Sijkl_cfg['wf_filename'].format(probe='{probe:s}', **variable_specs)
    wf_delta_vin = np.genfromtxt(f'{wf_folder}/{wf_filename.format(probe="delta")}')
    wf_gamma_vin = np.genfromtxt(f'{wf_folder}/{wf_filename.format(probe="gamma")}')
    wf_ia_vin = np.genfromtxt(f'{wf_folder}/{wf_filename.format(probe="ia")}')
    wf_mu_vin = np.genfromtxt(f'{wf_folder}/{wf_filename.format(probe="mu")}')

    zgrid_wf_vin = wf_delta_vin[:, 0]
    wf_delta_vin = wf_delta_vin[:, 1:]
    wf_gamma_vin = wf_gamma_vin[:, 1:]
    wf_ia_vin = wf_ia_vin[:, 1:]
    wf_mu_vin = wf_mu_vin[:, 1:]

    # IA bias array, to get wf lensing from Vincenzo's inputs
    ia_bias_vin = wf_cl_lib.build_ia_bias_1d_arr(zgrid_wf_vin, cosmo_ccl=cosmo_ccl,
                                                 flat_fid_pars_dict=flat_fid_pars_dict,
                                                 input_z_grid_lumin_ratio=None,
                                                 input_lumin_ratio=None,
                                                 output_F_IA_of_z=False)

    wf_lensing_vin = wf_gamma_vin + ia_bias_vin[:, None] * wf_ia_vin
    wf_galaxy_vin = wf_delta_vin + wf_mu_vin  # TODO in theory, I should BNT-tansform wf_mu...

    assert compute_bnt_with_shifted_nz_for_zcuts is False, 'The BNT used to compute the z_means and ell cuts is just for a simple case: no IA, no dz shift'
    assert shift_nz is True, 'The signal (and BNT used to transform it) is computed with a shifted n(z); You could use an un-shifted n(z) for the BNT, but' \
        'this would be slightly inconsistent (but also what I did so far).'
    assert include_ia_in_bnt_kernel_for_zcuts is False, 'We compute the BNT just for a simple case: no IA, no shift. This is because we want' \
                                                        ' to compute the z means'

    # ! apply a Gaussian filter
    if nz_gaussian_smoothing:
        n_of_z = nz_gaussian_smmothing_func(zgrid_nz, n_of_z, plot=True)

    # ! shift it (plus, re-normalize it after the shift)
    # * IMPORTANT NOTE: The BNT should be computed from the same n(z) (shifted or not) which is then used to compute
    # * the kernels which are then used to get the z_means, and finally the ell_cuts, for consistency. In other words,
    # * we cannot compute the kernels with a shifted n(z) and transforme them with a BNT computed from the unshifted n(z)
    # * and viceversa. If the n(z) are shifted, one of the BNT kernels will become negative, but this is just because
    # * two of the original kernels get very close after the shift: the transformation is correct.
    # * Having said that, I leave the possibility to continue with the rest of the code with a
    if compute_bnt_with_shifted_nz_for_zcuts:
        n_of_z = wf_cl_lib.shift_nz(zgrid_nz, n_of_z, dz_shifts, normalize=normalize_shifted_nz, plot_nz=False,
                                    interpolation_kind=shift_nz_interpolation_kind)
    nz_tuple = (zgrid_nz, n_of_z)

    bnt_matrix = covmat_utils.compute_BNT_matrix(zbins, zgrid_nz, n_of_z, cosmo_ccl=cosmo_ccl, plot_nz=False)

    # ! my kernels, which are needed because I want the flexibility to shift the n(z) or not
    gal_bias_1d_arr = wf_cl_lib.b_of_z_fs2_fit(zgrid_nz, general_cfg['magcut_source'] / 10,
                                               galaxy_bias_fit_fiducials)
    gal_bias_2d_arr = np.repeat(gal_bias_1d_arr.reshape(1, -1), zbins, axis=0).T
    gal_bias_tuple = (zgrid_nz, gal_bias_2d_arr)

    wf_lensing_ccl_arr = wf_cl_lib.wf_ccl(zgrid_nz, 'lensing', 'with_IA', flat_fid_pars_dict, cosmo_ccl,
                                          nz_tuple, ia_bias_tuple=None, gal_bias_tuple=gal_bias_tuple,
                                          return_ccl_obj=False, n_samples=len(zgrid_nz))
    wf_gamma_ccl_arr = wf_cl_lib.wf_ccl(zgrid_nz, 'lensing', 'without_IA', flat_fid_pars_dict, cosmo_ccl,
                                        nz_tuple, ia_bias_tuple=None, gal_bias_tuple=gal_bias_tuple,
                                        return_ccl_obj=False, n_samples=len(zgrid_nz))
    wf_ia_ccl_arr = wf_cl_lib.wf_ccl(zgrid_nz, 'lensing', 'IA_only', flat_fid_pars_dict, cosmo_ccl,
                                     nz_tuple, ia_bias_tuple=None, gal_bias_tuple=gal_bias_tuple,
                                     return_ccl_obj=False, n_samples=len(zgrid_nz))
    wf_galaxy_ccl_arr = wf_cl_lib.wf_ccl(zgrid_nz, 'galaxy', 'without_galaxy_bias', flat_fid_pars_dict, cosmo_ccl,
                                         nz_tuple, ia_bias_tuple=None, gal_bias_tuple=gal_bias_tuple,
                                         return_ccl_obj=False, n_samples=len(zgrid_nz))

    # this is to check against ccl in pyccl_cov
    general_cfg['wf_WL'] = wf_lensing_vin
    general_cfg['wf_GC'] = wf_galaxy_vin
    general_cfg['wf_delta'] = wf_delta_vin
    general_cfg['wf_mu'] = wf_mu_vin
    general_cfg['z_grid_wf'] = zgrid_wf_vin

    # BNT-transform the lensing kernels
    wf_gamma_ccl_bnt = (bnt_matrix @ wf_gamma_ccl_arr.T).T
    wf_lensing_ccl_bnt = (bnt_matrix @ wf_lensing_ccl_arr.T).T

    # compute z means
    if include_ia_in_bnt_kernel_for_zcuts:
        wf_ll_ccl = wf_lensing_ccl_arr
        wf_ll_ccl_bnt = wf_lensing_ccl_bnt
    else:
        wf_ll_ccl = wf_gamma_ccl_arr
        wf_ll_ccl_bnt = wf_gamma_ccl_bnt

    z_means_ll = wf_cl_lib.get_z_means(zgrid_nz, wf_ll_ccl)
    z_means_gg = wf_cl_lib.get_z_means(zgrid_nz, wf_galaxy_ccl_arr)
    z_means_ll_bnt = wf_cl_lib.get_z_means(zgrid_nz, wf_ll_ccl_bnt)

    # plot_kernels_for_thesis()

    plt.figure()
    for zi in range(zbins):
        plt.plot(zgrid_nz, wf_gamma_ccl_arr[:, zi], ls='-', c=colors[zi],
                 alpha=0.6, label='wf_gamma_ccl' if zi == 0 else None)
        plt.plot(zgrid_nz, wf_gamma_ccl_bnt[:, zi], ls='--', c=colors[zi],
                 alpha=0.6, label='wf_gamma_ccl_bnt' if zi == 0 else None)
        plt.axvline(z_means_ll_bnt[zi], ls=':', c=colors[zi])

    plt.legend()

    assert np.all(np.diff(z_means_ll) > 0), 'z_means_ll should be monotonically increasing'
    assert np.all(np.diff(z_means_gg) > 0), 'z_means_gg should be monotonically increasing'
    assert np.all(np.diff(z_means_ll_bnt) > 0), ('z_means_ll_bnt should be monotonically increasing '
                                                 '(not a strict condition, valid only if we do not shift the n(z) in this part)')

    ell_cuts_dict = {}
    ell_cuts_dict['LL'] = load_ell_cuts(kmax_h_over_Mpc, z_values_a=z_means_ll_bnt, z_values_b=z_means_ll_bnt)
    ell_cuts_dict['GG'] = load_ell_cuts(kmax_h_over_Mpc, z_values_a=z_means_gg, z_values_b=z_means_gg)
    ell_cuts_dict['GL'] = load_ell_cuts(kmax_h_over_Mpc, z_values_a=z_means_gg, z_values_b=z_means_ll_bnt)
    ell_cuts_dict['LG'] = load_ell_cuts(kmax_h_over_Mpc, z_values_a=z_means_ll_bnt, z_values_b=z_means_gg)
    ell_dict['ell_cuts_dict'] = ell_cuts_dict  # this is to pass the ll cuts to the covariance module

    # ! plot ell cuts matrix for thesis, for "reference" kmax (corresponding to FoM 400 for ell_cuts_center)
    # if kmax_h_over_Mpc == kmax_fom_400_ellcenter and center_or_min == 'center':
    #     plot_ell_cuts_for_thesis(ell_cuts_dict['LL'], ell_cuts_dict['GL'], ell_cuts_dict['GG'],
    #                              'LL', 'GL', 'GL', kmax_h_over_Mpc)
    #     plt.savefig(f'{ROOT}/phd_thesis_plots/plots/'
    #                 f'z_dependent_ell_cuts_kmax{kmax_h_over_Mpc:02f}.pdf', dpi=500, bbox_inches='tight')

    # mm.plot_bnt_matrix(BNT_matrix, zbins)
    # plt.savefig(f'{ROOT}/phd_thesis_plots/plots/bnt_matrix_fs2.pdf',
    #             dpi=500, bbox_inches='tight')

    if shift_nz:
        n_of_z = wf_cl_lib.shift_nz(zgrid_nz, n_of_z, dz_shifts, normalize=normalize_shifted_nz, plot_nz=False,
                                    interpolation_kind=shift_nz_interpolation_kind)
        nz_tuple = (zgrid_nz, n_of_z)
        # * this is important: the BNT matrix i use for the rest of the code (so not to compute the ell cuts) is instead
        # * consistent with the shifted n(z) used to compute the kernels
        bnt_matrix = covmat_utils.compute_BNT_matrix(zbins, zgrid_nz, n_of_z, cosmo_ccl=cosmo_ccl, plot_nz=False)

    general_cfg['nz_tuple'] = nz_tuple

    if general_cfg['use_CLOE_cls']:
        assert which_pk == 'HMCodeBar', 'I am using CLOE Cls, so I should use HMCodeBar'
        print(f'Using CLOE cls; pk is {which_pk}')

        cloe_bench_folder = general_cfg['cloe_bench_folder']
        warnings.warn('TODO fix number of bins here for correct WL handling in the lmax=3000 case')
        cl_ll_3d = np.load(f'{cloe_bench_folder}/Cls_zNLA3D_ShearShear_C00.npy')
        cl_gl_3d = np.load(f'{cloe_bench_folder}/Cls_zNLA3D_PosShear_C00.npy')[:nbl_3x2pt, :, :]
        cl_gg_3d = np.load(f'{cloe_bench_folder}/Cls_zNLA3D_PosPos_C00.npy')[:nbl_3x2pt, :, :]
        cl_wa_3d = cl_ll_3d[nbl_3x2pt:, :, :]
        cl_3x2pt_5d = cl_utils.build_3x2pt_datavector_5D(cl_ll_3d[:nbl_3x2pt, ...], cl_gl_3d,
                                                         cl_gg_3d, nbl_3x2pt, zbins, n_probes=2)
        # this repetition is not very nice, but no time right now
        if covariance_cfg[f'SSC_code'] == 'PySSC':
            assert covariance_cfg['compute_SSC'] is False, \
                'I am using mock responses; if you want to compute the SSC, you need to ' \
                'import the responses as well (see ssc_integrands_SPV3.py) for how to do it. moreover, ' \
                'pyssc w/ magbias is not yet ready'
        rl_ll_3d = np.ones_like(cl_ll_3d)
        rl_gl_3d = np.ones_like(cl_gl_3d)
        rl_gg_3d = np.ones_like(cl_gg_3d)
        rl_wa_3d = np.ones_like(cl_wa_3d)
        rl_3x2pt_5d = np.ones_like(cl_3x2pt_5d)

    else:

        if nbl_3x2pt == 29:
            warnings.warn('An error in cl_SPV3_1D_to_3D is probably indicating that the non-LL Vincenzos files are'
                          'for lmax = 3000.')

        # ! import and reshape datavectors (cl) and response functions (rl)
        cl_fld = general_cfg['cl_folder']
        cl_filename = general_cfg['cl_filename']
        cl_ll_1d = np.genfromtxt(
            f"{cl_fld.format(probe='WLO', which_pk=which_pk)}/{cl_filename.format(probe='WLO', **variable_specs)}")
        cl_gg_1d = np.genfromtxt(
            f"{cl_fld.format(probe='GCO', which_pk=which_pk)}/{cl_filename.format(probe='GCO', **variable_specs)}")
        cl_wa_1d = np.genfromtxt(
            f"{cl_fld.format(probe='WLA', which_pk=which_pk)}/{cl_filename.format(probe='WLA', **variable_specs)}")
        cl_3x2pt_1d = np.genfromtxt(
            f"{cl_fld.format(probe='3x2pt', which_pk=which_pk)}/{cl_filename.format(probe='3x2pt', **variable_specs)}")

        # ! reshape to 3d
        cl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(cl_ll_1d, 'WL', nbl_WL, zbins)
        cl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(cl_gg_1d, 'GC', nbl_GC, zbins)
        cl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(cl_wa_1d, 'WA', nbl_WA, zbins)
        cl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(cl_3x2pt_1d, '3x2pt', nbl_3x2pt, zbins)
        cl_gl_3d = deepcopy(cl_3x2pt_5d[1, 0, :, :, :])

        # ! import responses, not used at the moment (not using PySSC)
        # rl_fld = general_cfg['rl_folder'].format(which_pk=which_pk)
        # rl_filename = general_cfg['rl_filename'].format()
        # rl_ll_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLO', **variable_specs)}")
        # rl_gg_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='GCO', **variable_specs)}")
        # rl_wa_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='WLA', **variable_specs)}")
        # rl_3x2pt_1d = np.genfromtxt(f"{rl_fld}/{rl_filename.format(probe='3x2pt', **variable_specs)}")
        if covariance_cfg[f'SSC_code'] == 'PySSC':
            assert covariance_cfg['compute_SSC'] is False, \
                'I am using mock responses; if you want to compute the SSC, you need to ' \
                'import the responses as well (see ssc_integrands_SPV3.py) for how to do it. moreover, ' \
                'pyssc w/ magbias is not yet ready'
        rl_ll_1d = np.ones_like(cl_ll_1d)
        rl_gg_1d = np.ones_like(cl_gg_1d)
        rl_wa_1d = np.ones_like(cl_wa_1d)
        rl_3x2pt_1d = np.ones_like(cl_3x2pt_1d)

        rl_ll_3d = cl_utils.cl_SPV3_1D_to_3D(rl_ll_1d, 'WL', nbl_WL, zbins)
        rl_gg_3d = cl_utils.cl_SPV3_1D_to_3D(rl_gg_1d, 'GC', nbl_GC, zbins)
        rl_wa_3d = cl_utils.cl_SPV3_1D_to_3D(rl_wa_1d, 'WA', nbl_WA, zbins)
        rl_3x2pt_5d = cl_utils.cl_SPV3_1D_to_3D(rl_3x2pt_1d, '3x2pt', nbl_3x2pt, zbins)

    # check that cl_wa is equal to cl_ll in the last nbl_WA_opt bins
    if ell_max_WL == general_cfg['ell_max_WL_opt'] and general_cfg['use_WA']:
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
        cl_ll_3d = cl_utils.cl_BNT_transform(cl_ll_3d, bnt_matrix, 'L', 'L')
        cl_wa_3d = cl_utils.cl_BNT_transform(cl_wa_3d, bnt_matrix, 'L', 'L')
        cl_3x2pt_5d = cl_utils.cl_BNT_transform_3x2pt(cl_3x2pt_5d, bnt_matrix)
        warnings.warn('you should probably BNT-transform the responses too!')

    # ! cut datavectors and responses in the pessimistic case; be carful of WA, because it does not start from ell_min
    if ell_max_WL == 1500:
        warnings.warn(
            'you are cutting the datavectors and responses in the pessimistic case, but is this compatible '
            'with the redshift-dependent ell cuts? Yes, this is an old warning; nonetheless, check ')
        assert False, 'you should check this'
        cl_ll_3d = cl_ll_3d[:nbl_WL, :, :]
        cl_gg_3d = cl_gg_3d[:nbl_GC, :, :]
        cl_wa_3d = cl_ll_3d[nbl_GC:nbl_WL, :, :]
        cl_3x2pt_5d = cl_3x2pt_5d[:nbl_3x2pt, :, :]

        rl_ll_3d = rl_ll_3d[:nbl_WL, :, :]
        rl_gg_3d = rl_gg_3d[:nbl_GC, :, :]
        rl_wa_3d = rl_ll_3d[nbl_GC:nbl_WL, :, :]
        rl_3x2pt_5d = rl_3x2pt_5d[:nbl_3x2pt, :, :]

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

    # this is again to test against ccl cls
    general_cfg['cl_ll_3d'] = cl_ll_3d
    general_cfg['cl_gl_3d'] = cl_gl_3d
    general_cfg['cl_gg_3d'] = cl_gg_3d

    if covariance_cfg['compute_SSC'] and covariance_cfg['SSC_code'] == 'PySSC':

        transp_stacked_wf = np.vstack((wf_lensing_vin.T, wf_galaxy_vin.T))
        # ! compute or load Sijkl
        nz = z_arr.shape[0]  # get number of z points in nz to name the Sijkl file
        Sijkl_folder = Sijkl_cfg['Sijkl_folder']
        assert general_cfg[
            'cl_BNT_transform'] is False, 'for SSC, at the moment the BNT transform should not be ' \
            'applied to the cls, but to the covariance matrix (how ' \
            'should we deal with the responses in the former case?)'
        Sijkl_filename = Sijkl_cfg['Sijkl_filename'].format(
            flagship_version=general_cfg['flagship_version'],
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
                                        ell_dict, delta_dict, cl_dict_3D, rl_dict_3D, Sijkl, bnt_matrix)

    # save covariance matrix and test against benchmarks
    cov_folder = covariance_cfg['cov_folder'].format(cov_ell_cuts=str(covariance_cfg['cov_ell_cuts']),
                                                     **variable_specs)
    covmat_utils.save_cov(cov_folder, covariance_cfg, cov_dict, cases_tosave, **variable_specs)

    if EP_or_ED == 'EP' and covariance_cfg['SSC_code'] == 'exactSSC' and covariance_cfg['test_against_CLOE_benchmarks'] \
            and general_cfg['ell_cuts'] is False and which_pk == 'HMCodeBar':

        # load benchmark cov and check that it matches the one computed here; I am not actually using it
        if not general_cfg['BNT_transform']:

            # load CLOE benchmarks
            cov_cloe_bench_2d_G = np.load(f'{ROOT}/my_cloe_data/CovMat-3x2pt-Gauss-{nbl_WL_opt}Bins.npy')
            cov_cloe_bench_2dcloe_GSSC = np.load(f'{ROOT}/my_cloe_data/CovMat-3x2pt-GaussSSC-{nbl_WL_opt}Bins.npy')

            # reshape it in dav format
            cov_bench_2ddav_G = mm.cov_2d_cloe_to_dav(cov_cloe_bench_2d_G, nbl_WL_opt, zbins, 'ell', 'ell')
            cov_bench_2ddav_GSSC = mm.cov_2d_cloe_to_dav(cov_cloe_bench_2dcloe_GSSC, nbl_WL_opt, zbins, 'ell', 'ell')

            # ell cut, if needed
            assert cov_dict['cov_3x2pt_GO_2D'].shape == cov_dict['cov_3x2pt_GS_2D'].shape, \
                'cov_3x2pt_GO_2D and cov_3x2pt_GS_2D should have the same shape'
            n_cov_elements = cov_dict['cov_3x2pt_GO_2D'].shape[0]
            cov_bench_2ddav_G_lmax3000 = cov_bench_2ddav_G[:n_cov_elements, :n_cov_elements]
            cov_bench_2ddav_GSSC_lmax3000 = cov_bench_2ddav_GSSC[:n_cov_elements, :n_cov_elements]

            mm.compare_arrays(cov_dict['cov_3x2pt_GO_2D'], cov_bench_2ddav_G_lmax3000,
                              "cov_dict['cov_3x2pt_GO_2D']", "cov_bench_2ddav_G_lmax3000",
                              log_array=True, log_diff=False, abs_val=False, plot_diff_threshold=5)
            mm.compare_arrays(cov_dict['cov_3x2pt_GS_2D'], cov_bench_2ddav_GSSC_lmax3000,
                              "cov_dict['cov_3x2pt_GS_2D']", "cov_bench_2ddav_GSSC_lmax3000",
                              log_array=True, log_diff=False, abs_val=False, plot_diff_threshold=5)

        del cov_bench_2ddav_G_lmax3000, cov_bench_2ddav_GSSC_lmax3000
        gc.collect()

    if covariance_cfg['compute_GSSC_condition_number']:

        cond_number = np.linalg.cond(cov_dict['cov_3x2pt_GS_2D'])
        NUMPY_PRECISION = np.finfo(float).eps
        precision = cond_number * NUMPY_PRECISION
        print(f'kmax = {kmax_h_over_Mpc}, precision in the inversion of GS covariance = '
              f'{precision:.2e}, cond number = {cond_number:.2e}')

    if covariance_cfg['test_against_benchmarks']:
        cov_benchmark_folder = f'{cov_folder}/benchmarks'
        mm.test_folder_content(cov_folder, cov_benchmark_folder, covariance_cfg['cov_file_format'])

    if covariance_cfg['test_against_vincenzo'] and bnt_transform == False and not general_cfg['use_CLOE_cls']:
        cov_vinc_filename = covariance_cfg['cov_vinc_filename'].format(**variable_specs, probe='3x2pt')
        cov_vinc_g = np.load(f'{covariance_cfg["cov_vinc_folder"]}/{cov_vinc_filename}')['arr_0']
        num_elements_nbl29 = cov_dict['cov_3x2pt_GO_2D'].shape[0]
        npt.assert_allclose(cov_dict['cov_3x2pt_GO_2D'], cov_vinc_g[:num_elements_nbl29, :num_elements_nbl29],
                            rtol=1e-3, atol=0)
        print('covariance matrix matches with Vincenzo\'s ✅')

    # # ! remove from here
    # # these are the covmats used
    # cov_checkBNT_GSSC_noBNT = np.load(
    #     '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/OutputFiles/CheckBNT/cmfull-3x2pt-EP13-ML245-MS245-idIA2-idB3-idM3-idR1.npy')
    # cov_checkBNT_GSSC_BNT = np.load(
    #     '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/OutputFiles/CheckBNT/cmfull-3x2pt-EP13-ML245-MS245-idIA2-idB3-idM3-idR1-BNT.npy')

    # # mm.compare_arrays(cov_dict['cov_3x2pt_GS_2D'], cov_bench, 'cov_dict["cov_3x2pt_GS_2D"]',
    # #                   'cov_bench', plot_diff_threshold=5)

    # num_elements = cov_dict['cov_3x2pt_GS_2D'].shape[0]
    # diff = mm.percent_diff(cov_dict['cov_3x2pt_GS_2D'], cov_checkBNT_GSSC_BNT[:num_elements, :num_elements])
    # mm.matshow(diff, log=True, abs_val=True, threshold=1, title=f'per diff, BNT {general_cfg["BNT_transform"]}')

    # diff = mm.percent_diff(cov_bench_2ddav_GSSC, cov_checkBNT_GSSC_noBNT)
    # mm.matshow(diff, log=True, abs_val=True, threshold=1, title=f'per diff, BNT {general_cfg["BNT_transform"]}')

    # cov_vinc_no_bnt_4d = mm.cov_2D_to_4D(cov_checkBNT_GSSC_noBNT, nbl=32, block_index='vincenzo', optimize=True)

    # probe_ordering = (('L', 'L'), ('G', 'L'), ('G', 'G'))
    # cov_vinc_no_bnt_10d_dict = mm.cov_3x2pt_4d_to_10d_dict(cov_vinc_no_bnt_4d, zbins, probe_ordering, 32, ind.copy())

    # cov_vinc_no_bnt_4d_test = mm.cov_3x2pt_10D_to_4D(
    #     cov_vinc_no_bnt_10d_dict, probe_ordering, 32, zbins, ind.copy(), GL_or_LG)
    # np.testing.assert_allclose(cov_vinc_no_bnt_4d, cov_vinc_no_bnt_4d_test, rtol=1e-3, atol=0)

    # # turn to dict for the BNT function
    # X_dict = covmat_utils.build_X_matrix_BNT(bnt_matrix)
    # cov_vinc_bnt_10d_dict = covmat_utils.cov_3x2pt_BNT_transform(cov_vinc_no_bnt_10d_dict, X_dict)
    # cov_vinc_bnt_4d = mm.cov_3x2pt_10D_to_4D(cov_vinc_bnt_10d_dict, probe_ordering, 32, zbins, ind.copy(), GL_or_LG)
    # cov_vinc_bnt_2d = mm.cov_4D_to_2D(cov_vinc_bnt_4d)

    # if bnt_transform:
    #     mm.compare_arrays(cov_vinc_bnt_2d, cov_checkBNT_GSSC_BNT, 'cov_vinc_bnt_2d',
    #                       'cov_bench', log_diff=True, plot_diff_threshold=5)

    # # ! remove until here

    # TODO compute BNT for Vincenzo's covs, which are not exactly equal to the CLOE-datavector ones?
    # mm.matshow(cov_dict['cov_3x2pt_GS_2D'], log=True, title=f'BNT {BNT_transform}')
    # cov_bnt_filename = cov_vinc_filename.replace('cmfull-', 'cmfull-bnt-')
    # np.savetxt(f'covariance_cfg["cov_vinc_folder"]/{cov_bnt_filename}', cov_dict['cov_3x2pt_GS_2D'], fmt='%.7e')

    # ! compute Fisher matrix
    if not fm_cfg['compute_FM']:
        # this guard is just to avoid indenting the whole code below
        raise KeyboardInterrupt('skipping FM computation, the script will exit now')

    # import and store derivative in one big dictionary
    start_time = time.perf_counter()
    derivatives_folder = fm_cfg['derivatives_folder'].format(**variable_specs)

    # ! get vincenzo's derivatives' parameters, to check that they match with the yaml file
    # check the parameter names in the derivatives folder, to see whether I'm setting the correct ones in the config file
    der_prefix = fm_cfg['derivatives_prefix']
    vinc_filenames = mm.get_filenames_in_folder(derivatives_folder)
    vinc_filenames = [vinc_filename for vinc_filename in vinc_filenames if
                      vinc_filename.startswith(der_prefix)]

    # keep only the files corresponding to the correct magcut_lens, magcut_source and zbins
    vinc_filenames = [filename for filename in vinc_filenames if
                      all(x in filename for x in
                          [f'ML{magcut_lens}', f'MS{magcut_source}', f'{EP_or_ED}{zbins:02d}'])]
    vinc_filenames = [filename.replace('.dat', '') for filename in vinc_filenames]

    vinc_trimmed_filenames = [vinc_filename.split('-', 1)[0].strip() for vinc_filename in vinc_filenames]
    vinc_trimmed_filenames = [
        vinc_trimmed_filename[len(der_prefix):] if vinc_trimmed_filename.startswith(
            der_prefix) else vinc_trimmed_filename
        for vinc_trimmed_filename in vinc_trimmed_filenames]
    vinc_param_names = list(set(vinc_trimmed_filenames))
    vinc_param_names.sort()

    # ! get fiducials names and values from the yaml file
    # remove ODE if I'm studying only flat models
    if flat_or_nonflat == 'Flat' and 'ODE' in fid_pars_dict['FM_ordered_params']:
        fid_pars_dict['FM_ordered_params'].pop('ODE')
    fm_fid_dict = fid_pars_dict['FM_ordered_params']
    param_names_3x2pt = list(fm_fid_dict.keys())
    fm_cfg['param_names_3x2pt'] = param_names_3x2pt
    fm_cfg['nparams_tot'] = len(param_names_3x2pt)

    # sort them to compare with vincenzo's param names
    my_sorted_param_names = param_names_3x2pt.copy()
    my_sorted_param_names.sort()

    for dzgc_param_name in [f'dzGC{zi:02d}' for zi in range(1, zbins + 1)]:
        if dzgc_param_name in vinc_param_names:  # ! added this if statement, not very elegant
            vinc_param_names.remove(dzgc_param_name)

    # check whether the 2 lists match and print the elements that are in one list but not in the other
    param_names_not_in_my_list = [vinc_param_name for vinc_param_name in vinc_param_names if
                                  vinc_param_name not in my_sorted_param_names]
    param_names_not_in_vinc_list = [my_sorted_param_name for my_sorted_param_name in my_sorted_param_names
                                    if
                                    my_sorted_param_name not in vinc_param_names]

    # Check if the parameter names match
    if not np.all(vinc_param_names == my_sorted_param_names):
        # Print the mismatching parameters
        print(f'Params present in input folder but not in the cfg file: {param_names_not_in_my_list}')
        print(f'Params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}')

    # try:
    #     assert np.all(vinc_param_names == my_sorted_param_names), \
    #         f'Params present in input folder but not in the cfg file: {param_names_not_in_my_list}\n' \
    #         f'Params present in cfg file but not in the input folder: {param_names_not_in_vinc_list}'
    # except AssertionError as error:
    #     print(error)
    #     if param_names_not_in_vinc_list == ['logT']:
    #         print('The derivative w.r.t logT is missing in the input folder but '
    #             'the corresponding FM is still set to 0; moving on')
    #     else:
    #         raise AssertionError(
    #             'there is something wrong with the parameter names in the derivatives folder')

    # ! preprocess derivatives (or load the alreay preprocessed ones)
    if fm_cfg['load_preprocess_derivatives']:
        warnings.warn(
            'loading preprocessed derivatives is faster but a bit more dangerous, make sure all the specs are taken into account')
        dC_LL_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_LL_4D.npy')
        dC_GG_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_GG_4D.npy')
        dC_WA_4D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_WA_4D.npy')
        dC_3x2pt_6D = np.load(f'{derivatives_folder}/reshaped_into_4d_arrays/dC_3x2pt_6D.npy')

    elif not fm_cfg['load_preprocess_derivatives']:
        der_prefix = fm_cfg['derivatives_prefix']
        dC_dict_1D = dict(mm.get_kv_pairs(derivatives_folder, "dat"))
        # check if dictionary is empty
        if not dC_dict_1D:
            raise ValueError(f'No derivatives found in folder {derivatives_folder}')

        # separate in 4 different dictionaries and reshape them (no interpolation needed in this case)
        dC_dict_LL_3D = {}
        dC_dict_GG_3D = {}
        dC_dict_WA_3D = {}
        dC_dict_3x2pt_5D = {}

        for key in vinc_filenames:  # loop over these, I already selected ML, MS and so on
            if not key.startswith('dDVddzGC'):
                if 'WLO' in key:
                    dC_dict_LL_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WL', nbl_WL, zbins)
                elif 'GCO' in key:
                    dC_dict_GG_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'GC', nbl_GC, zbins)
                # elif 'WLA' in key:
                    # dC_dict_WA_3D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], 'WA', nbl_WA, zbins)
                elif '3x2pt' in key:
                    dC_dict_3x2pt_5D[key] = cl_utils.cl_SPV3_1D_to_3D(dC_dict_1D[key], '3x2pt', nbl_3x2pt,
                                                                      zbins)

        # turn the dictionaries of derivatives into npy array of shape (nbl, zbins, zbins, nparams)
        dC_LL_4D = FM_utils.dC_dict_to_4D_array(dC_dict_LL_3D, param_names_3x2pt, nbl_WL, zbins, der_prefix)
        dC_GG_4D = FM_utils.dC_dict_to_4D_array(dC_dict_GG_3D, param_names_3x2pt, nbl_GC, zbins, der_prefix)
        # dC_WA_4D = FM_utils.dC_dict_to_4D_array(dC_dict_WA_3D, param_names_3x2pt, nbl_WA, zbins, der_prefix)
        dC_WA_4D = np.ones((nbl_WA, zbins, zbins, dC_LL_4D.shape[-1]))
        dC_3x2pt_6D = FM_utils.dC_dict_to_4D_array(
            dC_dict_3x2pt_5D, param_names_3x2pt, nbl_3x2pt, zbins, der_prefix, is_3x2pt=True)

        # free up memory
        del dC_dict_1D, dC_dict_LL_3D, dC_dict_GG_3D, dC_dict_WA_3D, dC_dict_3x2pt_5D
        gc.collect()

        print(
            'derivatives reshaped in 4D arrays in {:.2f} seconds'.format(time.perf_counter() - start_time))

        # save these so they can simply be imported!
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_LL_4D.npy', dC_LL_4D)
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_GG_4D.npy', dC_GG_4D)
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_WA_4D.npy', dC_WA_4D)
        np.save(f'{derivatives_folder}/reshaped_into_np_arrays/dC_3x2pt_6D.npy', dC_3x2pt_6D)

    else:
        raise ValueError('"load_preprocess_derivatives" must be True or False')

    # store the derivatives arrays in a dictionary
    deriv_dict = {'dC_LL_4D': dC_LL_4D,
                  'dC_WA_4D': dC_WA_4D,
                  'dC_GG_4D': dC_GG_4D,
                  'dC_3x2pt_6D': dC_3x2pt_6D}

    # ! compute and save fisher matrix
    FM_dict = FM_utils.compute_FM(general_cfg, covariance_cfg, fm_cfg, ell_dict, cov_dict, deriv_dict,
                                  bnt_matrix)
    FM_dict['fiducial_values_dict'] = fm_fid_dict  # ordered fiducial parameters entering the FM

    fm_folder = fm_cfg['fm_folder'].format(ell_cuts=str(general_cfg['ell_cuts']),
                                           which_cuts=general_cfg['which_cuts'],
                                           BNT_transform=str(bnt_transform),
                                           center_or_min=general_cfg['center_or_min'])
    if not general_cfg['ell_cuts']:
        # not very nice, i defined the ell_cuts_subfolder above...
        fm_folder = fm_folder.replace(f'/{general_cfg["which_cuts"]}/ell_{center_or_min}', '')

    if fm_cfg['save_FM_dict']:
        FM_dict_filename = fm_cfg['FM_dict_filename'].format(**variable_specs)
        mm.save_pickle(f'{fm_folder}/{FM_dict_filename}.pickle', FM_dict)

    if fm_cfg['test_against_benchmarks']:
        mm.test_folder_content(fm_folder, fm_folder + '/benchmarks', 'txt')

    if fm_cfg['test_against_vincenzo'] and bnt_transform == False:
        fm_vinc_folder = fm_cfg["fm_vinc_folder"].format(**variable_specs, go_gs_vinc='GaussOnly')

        # for probe_vinc in ('WLO', 'GCO', '3x2pt'):
        for probe_vinc in ('3x2pt',):
            probe_dav = probe_vinc.replace('O', '')
            fm_vinc_filename = fm_cfg['fm_vinc_filename'].format(**variable_specs, probe=probe_vinc)
            fm_vinc_g = np.genfromtxt(f'{fm_vinc_folder}/{fm_vinc_filename}')
            print(probe_dav)

            diff = mm.percent_diff(FM_dict[f'FM_{probe_dav}_G'], fm_vinc_g)
            xticks = param_names_3x2pt
            plt.matshow(np.log10(np.abs(diff)))
            plt.colorbar()
            plt.xticks(np.arange(len(xticks)), xticks, rotation=90)

            mm.compare_arrays(FM_dict[f'FM_{probe_dav}_G'], fm_vinc_g, log_array=True, log_diff=False,
                              abs_val=False, plot_diff_threshold=5)

            npt.assert_allclose(FM_dict[f'FM_{probe_dav}_G'], fm_vinc_g, rtol=1e-3, atol=0)

    del cov_dict
    gc.collect()

print('Finished in {:.2f} minutes'.format((time.perf_counter() - script_start_time) / 60))
