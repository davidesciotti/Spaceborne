import time
import warnings
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from copy import deepcopy
from scipy.interpolate import UnivariateSpline, interp1d, RectBivariateSpline
import os

import spaceborne.my_module as mm
import spaceborne.bnt as bnt_utils

ROOT = os.getenv('ROOT')

probe_names_dict = {'LL': 'WL', 'GG': 'GC', '3x2pt': '3x2pt', }


def bin_2d_matrix(cov, ells_in, ells_out, ells_out_edges):

    assert cov.shape[0] == cov.shape[1] == len(ells_in), "ells_in must be the same length as the covariance matrix"
    assert len(ells_out) == len(ells_out_edges) - 1, "ells_out must be the same length as the number of edges - 1"

    binned_cov = np.zeros((len(ells_out), len(ells_out)))
    cov_interp_func = RectBivariateSpline(ells_in, ells_in, cov)

    ells_edges_low = ells_out_edges[:-1]
    ells_edges_high = ells_out_edges[1:]

    # Loop over the output bins
    for ell1_idx, _ in enumerate(ells_out):
        for ell2_idx, _ in enumerate(ells_out):

            # Get ell min/max for the current bins
            ell1_min = ells_edges_low[ell1_idx]
            ell1_max = ells_edges_high[ell1_idx]
            ell2_min = ells_edges_low[ell2_idx]
            ell2_max = ells_edges_high[ell2_idx]

            # isolate the relevant ranges of ell values from the original ells_in grid
            ell1_in = ells_in[(ell1_min <= ells_in) & (ells_in < ell1_max)]
            ell2_in = ells_in[(ell2_min <= ells_in) & (ells_in < ell2_max)]

            # mask the covariance to the relevant block
            cov_masked = cov[np.ix_(ell1_in, ell2_in)]

            # Calculate the bin widths
            delta_ell_1 = ell1_max - ell1_min
            delta_ell_2 = ell2_max - ell2_min

            # Option 1a: use the original grid for integration and the ell values as weights
            # ells1_in_xx, ells2_in_yy = np.meshgrid(ell1_in, ell2_in, indexing='ij')
            # partial_integral = simps(y=cov_masked * ells1_in_xx * ells2_in_yy, x=ell2_in, axis=1)
            # integral = simps(y=partial_integral, x=ell1_in)
            # binned_cov[ell1_idx, ell2_idx] = integral / (np.sum(ell1_in) * np.sum(ell2_in))

            # Option 1b: use the original grid for integration and no weights
            partial_integral = simps(y=cov_masked, x=ell2_in, axis=1)
            integral = simps(y=partial_integral, x=ell1_in)
            binned_cov[ell1_idx, ell2_idx] = integral / (delta_ell_1 * delta_ell_2)

            # # Option 2: create fine grids for integration over the ell ranges (GIVES GOOD RESULTS ONLY FOR nsteps=delta_ell!)
            # ell_fine_1 = np.linspace(ell1_min, ell1_max, 50)
            # ell_fine_2 = np.linspace(ell2_min, ell2_max, 50)

            # # Evaluate the spline on the fine grids
            # ell1_fine_xx, ell2_fine_yy = np.meshgrid(ell_fine_1, ell_fine_2, indexing='ij')
            # cov_interp_vals = cov_interp_func(ell_fine_1, ell_fine_2)

            # # Perform simps integration over the ell ranges
            # partial_integral = simps(y=cov_interp_vals * ell1_fine_xx * ell2_fine_yy, x=ell_fine_2, axis=1)
            # integral = simps(y=partial_integral, x=ell_fine_1)
            # # Normalize by the bin areas
            # binned_cov[ell1_idx, ell2_idx] = integral / (np.sum(ell_fine_1) * np.sum(ell_fine_2))

    return binned_cov


def ssc_integral_julia(d2CLL_dVddeltab, d2CGL_dVddeltab, d2CGG_dVddeltab,
                       ind_auto, ind_cross, cl_integral_prefactor, sigma2, z_grid, integration_type,
                       probe_ordering, num_threads=16):
    """Kernel to compute the 4D integral optimized using Simpson's rule using Julia."""

    suffix = 0
    folder_name = 'tmp'
    unique_folder_name = folder_name

    # Loop until we find a folder name that does not exist
    while os.path.exists(unique_folder_name):
        suffix += 1
        unique_folder_name = f'{folder_name}{suffix}'
    os.makedirs(unique_folder_name)
    folder_name = unique_folder_name

    np.save(f"{folder_name}/d2CLL_dVddeltab", d2CLL_dVddeltab)
    np.save(f"{folder_name}/d2CGL_dVddeltab", d2CGL_dVddeltab)
    np.save(f"{folder_name}/d2CGG_dVddeltab", d2CGG_dVddeltab)
    np.save(f"{folder_name}/ind_auto", ind_auto)
    np.save(f"{folder_name}/ind_cross", ind_cross)
    np.save(f"{folder_name}/cl_integral_prefactor", cl_integral_prefactor)
    np.save(f"{folder_name}/sigma2", sigma2)
    np.save(f"{folder_name}/z_grid", z_grid)
    os.system(
        f"julia --project=. --threads={num_threads} spaceborne/julia_integrator.jl {folder_name} {integration_type}")

    cov_filename = "cov_SSC_spaceborne_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D.npy"

    if integration_type == 'trapz-6D':
        cov_ssc_3x2pt_dict_8D = {}  # it's 10D, actually
        for probe_a, probe_b in probe_ordering:
            for probe_c, probe_d in probe_ordering:
                if str.join('', (probe_a, probe_b, probe_c, probe_d)) not in ['GLLL', 'GGLL', 'GGGL']:
                    print(f"Loading {probe_a}{probe_b}{probe_c}{probe_d}")
                    cov_ssc_3x2pt_dict_8D[(probe_a, probe_b, probe_c, probe_d)] = np.load(
                        f"{folder_name}/{cov_filename.format(probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d)}")

    else:
        cov_ssc_3x2pt_dict_8D = mm.load_cov_from_probe_blocks(
            path=f'{folder_name}',
            filename=cov_filename,
            probe_ordering=probe_ordering)

    os.system(f"rm -rf {folder_name}")
    return cov_ssc_3x2pt_dict_8D


def get_ellmax_nbl(probe, general_cfg):
    if probe == 'LL':
        ell_max = general_cfg['ell_max_WL']
        nbl = general_cfg['nbl_WL']
    elif probe == 'GG':
        ell_max = general_cfg['ell_max_GC']
        nbl = general_cfg['nbl_GC']
    elif probe == '3x2pt':
        ell_max = general_cfg['ell_max_3x2pt']
        nbl = general_cfg['nbl_3x2pt']
    else:
        raise ValueError('probe must be LL or GG or 3x2pt')
    return ell_max, nbl


def compute_cov(general_cfg, covariance_cfg, ell_dict, cl_dict, BNT_matrix, oc_obj, ):
    """
    This code computes the Gaussian-only, SSC-only and Gaussian+SSC
    covariance matrices, for different ordering options
    """

    # import settings:
    ell_max_WL = general_cfg['ell_max_WL']
    ell_max_GC = general_cfg['ell_max_GC']
    ell_max_3x2pt = general_cfg['ell_max_3x2pt']
    zbins = general_cfg['zbins']
    n_probes = general_cfg['n_probes']
    ng_cov_code = covariance_cfg['ng_cov_code']
    ng_cov_code_cfg = covariance_cfg[ng_cov_code + '_cfg']

    fsky = covariance_cfg['fsky']
    GL_or_LG = covariance_cfg['GL_or_LG']
    # ! must copy the array! Otherwise, it gets modified and changed at each call
    ind = covariance_cfg['ind'].copy()
    block_index = covariance_cfg['block_index']
    probe_ordering = covariance_cfg['probe_ordering']

    # (not the best) check to ensure that the (LL, XC, GG) ordering is respected
    assert probe_ordering[0] == ('L', 'L'), 'the XC probe should be in position 1 (not 0) of the datavector'
    assert probe_ordering[2] == ('G', 'G'), 'the XC probe should be in position 1 (not 0) of the datavector'

    # import ell values
    ell_WL, nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
    ell_GC, nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
    ell_WA, nbl_WA = ell_dict['ell_WA'], ell_dict['ell_WA'].shape[0]
    ell_3x2pt, nbl_3x2pt = ell_GC, nbl_GC

    cov_dict = {}

    # sanity checks
    if general_cfg['nbl_WL'] is None:
        assert nbl_WL == general_cfg['nbl'], 'nbl_WL != general_cfg["nbl"], there is a discrepancy'

    if general_cfg['nbl_WL'] is not None:
        assert nbl_WL == general_cfg['nbl_WL'], 'nbl_WL != general_cfg["nbl_WL"], there is a discrepancy'

    if nbl_WL == nbl_GC == nbl_3x2pt:
        print('all probes (but WAdd) have the same number of ell bins')

    # nbl for Wadd
    if ell_WA.size == 1:
        nbl_WA = 1  # in the case of just one bin it would give error
    else:
        nbl_WA = ell_WA.shape[0]

    # ell values in linear scale:
    if ell_WL.max() < 15:  # very rudimental check of whether they're in lin or log scale
        raise ValueError('looks like the ell values are in log scale. You should use linear scale instead.')

    # load deltas
    delta_l_WL = ell_dict['delta_l_WL']
    delta_l_GC = ell_dict['delta_l_GC']
    delta_l_WA = ell_dict['delta_l_WA']
    delta_l_3x2pt = delta_l_GC

    # load set correct output folder, get number of pairs
    # output_folder = mm.get_output_folder(ind_ordering, which_forecast)
    zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

    # if C_XC is C_LG, switch the ind.dat ordering for the correct rows
    if GL_or_LG == 'LG':
        print('\nAttention! switching columns in the ind array (for the XC part)')
        ind[zpairs_auto:(zpairs_auto + zpairs_cross), [2, 3]] = ind[zpairs_auto:(zpairs_auto + zpairs_cross), [3, 2]]

    # sanity check: the last 2 columns of ind_auto should be equal to the last two of ind_auto
    assert np.array_equiv(ind[:zpairs_auto, 2:], ind[-zpairs_auto:, 2:])

    # convenience vectors, used for the cov_4D_to_6D function
    ind_auto = ind[:zpairs_auto, :].copy()
    ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :].copy()
    ind_dict = {('L', 'L'): ind_auto,
                ('G', 'L'): ind_cross,
                ('G', 'G'): ind_auto}
    covariance_cfg['ind_dict'] = ind_dict

    # load Cls and responses
    cl_LL_3D = cl_dict['cl_LL_3D']
    cl_GG_3D = cl_dict['cl_GG_3D']
    cl_WA_3D = cl_dict['cl_WA_3D']
    cl_3x2pt_5D = cl_dict['cl_3x2pt_5D']

    # ! ======================================= COMPUTE GAUSS ONLY COVARIANCE =======================================
    start = time.perf_counter()
    # build noise vector
    sigma_eps2 = (covariance_cfg['sigma_eps_i'] * np.sqrt(2))**2
    ng_shear = np.array(covariance_cfg['ngal_lensing'])
    ng_clust = np.array(covariance_cfg['ngal_clustering'])
    noise_3x2pt_4D = mm.build_noise(zbins, n_probes, sigma_eps2=sigma_eps2,
                                    ng_shear=ng_shear, ng_clust=ng_clust,
                                    EP_or_ED=general_cfg['EP_or_ED'],
                                    which_shape_noise=covariance_cfg['which_shape_noise'])

    # create dummy ell axis, the array is just repeated along it
    nbl_max = np.max((nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA))
    noise_5D = np.zeros((n_probes, n_probes, nbl_max, zbins, zbins))
    for probe_A in (0, 1):
        for probe_B in (0, 1):
            for ell_idx in range(nbl_WL):
                noise_5D[probe_A, probe_B, ell_idx, :, :] = noise_3x2pt_4D[probe_A, probe_B, ...]

    # remember, the ell axis is a dummy one for the noise, is just needs to be of the
    # same length as the corresponding cl one
    noise_LL_5D = noise_5D[0, 0, :nbl_WL, :, :][np.newaxis, np.newaxis, ...]
    noise_GG_5D = noise_5D[1, 1, :nbl_GC, :, :][np.newaxis, np.newaxis, ...]
    noise_WA_5D = noise_5D[0, 0, :nbl_WA, :, :][np.newaxis, np.newaxis, ...]
    noise_3x2pt_5D = noise_5D[:, :, :nbl_3x2pt, :, :]

    if general_cfg['cl_BNT_transform']:
        print('BNT-transforming the noise spectra...')
        noise_LL_5D = bnt_utils.cl_BNT_transform(noise_LL_5D[0, 0, ...], BNT_matrix, 'L', 'L')[None, None, ...]
        noise_WA_5D = bnt_utils.cl_BNT_transform(noise_WA_5D[0, 0, ...], BNT_matrix, 'L', 'L')[None, None, ...]
        noise_3x2pt_5D = bnt_utils.cl_BNT_transform_3x2pt(noise_3x2pt_5D, BNT_matrix)

    # 5d versions of auto-probe spectra
    cl_LL_5D = cl_LL_3D[np.newaxis, np.newaxis, ...]
    cl_GG_5D = cl_GG_3D[np.newaxis, np.newaxis, ...]
    cl_WA_5D = cl_WA_3D[np.newaxis, np.newaxis, ...]
    
    if covariance_cfg['split_gaussian_cov']:
        cov_WL_GO_6D_sva, cov_WL_GO_6D_sn, cov_WL_GO_6D_mix = mm.covariance_einsum_split(cl_LL_5D, noise_LL_5D, fsky, ell_WL, delta_l_WL)[0, 0, 0, 0, ...]
        cov_GC_GO_6D_sva, cov_GC_GO_6D_sn, cov_GC_GO_6D_mix = mm.covariance_einsum_split(cl_GG_5D, noise_GG_5D, fsky, ell_GC, delta_l_GC)[0, 0, 0, 0, ...]
        cov_WA_GO_6D_sva, cov_WA_GO_6D_sn, cov_WA_GO_6D_mix = mm.covariance_einsum_split(cl_WA_5D, noise_WA_5D, fsky, ell_WA, delta_l_WA)[0, 0, 0, 0, ...]
        cov_3x2pt_GO_10D_sva, cov_3x2pt_GO_10D_sn, cov_3x2pt_GO_10D_mix = mm.covariance_einsum_split(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_3x2pt, delta_l_3x2pt)
        cov_WL_GO_6D = cov_WL_GO_6D_sva + cov_WL_GO_6D_sn + cov_WL_GO_6D_mix
        cov_GC_GO_6D = cov_GC_GO_6D_sva + cov_GC_GO_6D_sn + cov_GC_GO_6D_mix
        cov_WA_GO_6D = cov_WA_GO_6D_sva + cov_WA_GO_6D_sn + cov_WA_GO_6D_mix
        cov_3x2pt_GO_10D = cov_3x2pt_GO_10D_sva + cov_3x2pt_GO_10D_sn + cov_3x2pt_GO_10D_mix
    else:        
        cov_WL_GO_6D = mm.covariance_einsum(cl_LL_5D, noise_LL_5D, fsky, ell_WL, delta_l_WL)[0, 0, 0, 0, ...]
        cov_GC_GO_6D = mm.covariance_einsum(cl_GG_5D, noise_GG_5D, fsky, ell_GC, delta_l_GC)[0, 0, 0, 0, ...]
        cov_WA_GO_6D = mm.covariance_einsum(cl_WA_5D, noise_WA_5D, fsky, ell_WA, delta_l_WA)[0, 0, 0, 0, ...]
        cov_3x2pt_GO_10D = mm.covariance_einsum(cl_3x2pt_5D, noise_3x2pt_5D, fsky, ell_3x2pt, delta_l_3x2pt)
    
    print("Gauss. cov. matrices computed in %.2f seconds" % (time.perf_counter() - start))

    ######################## COMPUTE SSC COVARIANCE ###############################

    if ng_cov_code == 'Spaceborne':
        symmetrize_output_dict = {
            ('L', 'L'): False,
            ('G', 'L'): False,
            ('L', 'G'): False,
            ('G', 'G'): False,
        }
        cov_ssc_sb_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
            covariance_cfg['cov_ssc_3x2pt_dict_8D_sb'], nbl_3x2pt, zbins, ind_dict, probe_ordering, symmetrize_output_dict)
        cov_ssc_sb_3x2pt_10D = mm.cov_10D_dict_to_array(cov_ssc_sb_3x2pt_dict_10D, nbl_3x2pt, zbins, n_probes)
        cov_3x2pt_SS_10D = cov_ssc_sb_3x2pt_10D

        if covariance_cfg['OneCovariance_cfg']['use_OneCovariance_cNG']:
            print('Adding cNG covariance from OneCovariance...')

            # test that oc_obj.cov_cng_oc_3x2pt_10D is not identically zero
            assert not np.allclose(oc_obj.cov_cng_oc_3x2pt_10D, 0, atol=0, rtol=1e-10), \
                'OneCovariance covariance matrix is identically zero'

            cov_3x2pt_SS_10D += oc_obj.cov_cng_oc_3x2pt_10D

    elif ng_cov_code == 'OneCovariance':

        assert (
            covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == ['SSC', 'cNG'] or
            covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == ['SSC',] or
            covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == ['cNG',]
        ), "covariance_cfg['OneCovariance_cfg']['which_ng_cov'] not recognised"

        if covariance_cfg['OneCovariance_cfg']['use_OneCovariance_Gaussian']:

            print('Loading Gaussian covariance from OneCovariance...')
            # TODO do it with pyccl as well, after computing the G covariance
            cov_3x2pt_GO_10D = oc_obj.cov_g_oc_3x2pt_10D
            # Slice or reload to get the LL, GG and 3x2pt covariance
            cov_WL_GO_6D = deepcopy(cov_3x2pt_GO_10D[0, 0, 0, 0, :nbl_WL, :nbl_WL, :, :, :, :])
            cov_GC_GO_6D = deepcopy(cov_3x2pt_GO_10D[1, 1, 1, 1, :nbl_GC, :nbl_GC, :, :, :, :])
            cov_3x2pt_GO_10D = deepcopy(cov_3x2pt_GO_10D[:, :, :, :, :nbl_3x2pt, :nbl_3x2pt, :, :, :, :])

        if covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == ['SSC',]:
            cov_3x2pt_SS_10D = oc_obj.cov_ssc_oc_3x2pt_10D

        elif covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == ['cNG',]:
            cov_3x2pt_SS_10D = oc_obj.cov_cng_oc_3x2pt_10D
        
        elif covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == ['SSC', 'cNG']:
            cov_3x2pt_SS_10D = oc_obj.cov_ssc_oc_3x2pt_10D
            cov_3x2pt_SS_10D += oc_obj.cov_cng_oc_3x2pt_10D

        else:
            raise ValueError("covariance_cfg['OneCovariance_cfg']['which_ng_cov'] not recognised")

    elif ng_cov_code == 'PyCCL':

        print('Using PyCCL non-Gaussian covariance matrices...')

        assert (
            (covariance_cfg['PyCCL_cfg']['which_ng_cov'] == ['SSC', 'cNG']) or
            (covariance_cfg['PyCCL_cfg']['which_ng_cov'] == ['SSC',]) or
            (covariance_cfg['PyCCL_cfg']['which_ng_cov'] == ['cNG',])
        ), "covariance_cfg['PyCCL_cfg']['which_ng_cov'] not recognised"

        symmetrize_output_dict = {
            ('L', 'L'): False,
            ('G', 'L'): False,
            ('L', 'G'): False,
            ('G', 'G'): False,
        }

        if covariance_cfg['PyCCL_cfg']['which_ng_cov'] == ['SSC',]:

            cov_ssc_ccl_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
                covariance_cfg['cov_ssc_3x2pt_dict_8D_ccl'], nbl_3x2pt, zbins, ind_dict, probe_ordering, symmetrize_output_dict)
            cov_ssc_sb_3x2pt_10D = mm.cov_10D_dict_to_array(cov_ssc_ccl_3x2pt_dict_10D, nbl_3x2pt, zbins, n_probes)
            cov_3x2pt_SS_10D = cov_ssc_sb_3x2pt_10D

        elif covariance_cfg['PyCCL_cfg']['which_ng_cov'] == ['cNG', ]:

            cov_cng_ccl_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
                covariance_cfg['cov_cng_3x2pt_dict_8D_ccl'], nbl_3x2pt, zbins, ind_dict, probe_ordering, symmetrize_output_dict)
            cov_cng_sb_3x2pt_10D = mm.cov_10D_dict_to_array(cov_cng_ccl_3x2pt_dict_10D, nbl_3x2pt, zbins, n_probes)
            cov_3x2pt_SS_10D = cov_cng_sb_3x2pt_10D
        
        elif covariance_cfg['PyCCL_cfg']['which_ng_cov'] == ['SSC', 'cNG']:

            cov_ssc_ccl_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
                covariance_cfg['cov_ssc_3x2pt_dict_8D_ccl'], nbl_3x2pt, zbins, ind_dict, probe_ordering, symmetrize_output_dict)
            cov_cng_ccl_3x2pt_dict_10D = mm.cov_3x2pt_dict_8d_to_10d(
                covariance_cfg['cov_cng_3x2pt_dict_8D_ccl'], nbl_3x2pt, zbins, ind_dict, probe_ordering, symmetrize_output_dict)

            cov_ssc_sb_3x2pt_10D = mm.cov_10D_dict_to_array(cov_ssc_ccl_3x2pt_dict_10D, nbl_3x2pt, zbins, n_probes)
            cov_cng_sb_3x2pt_10D = mm.cov_10D_dict_to_array(cov_cng_ccl_3x2pt_dict_10D, nbl_3x2pt, zbins, n_probes)

            cov_3x2pt_SS_10D = cov_ssc_sb_3x2pt_10D
            cov_3x2pt_SS_10D += cov_cng_sb_3x2pt_10D


        else:
            raise ValueError("covariance_cfg['PyCCL_cfg']['which_ng_cov'] not recognised")

    else:
        raise NotImplementedError(f'ng_cov_code {ng_cov_code} not implemented')


    # In this case, you just need to slice get the LL, GG and 3x2pt covariance
    # WL slicing unnecessary, since I load with nbl_WL and max_WL but just in case
    cov_WA_SS_6D = deepcopy(cov_3x2pt_SS_10D[0, 0, 0, 0, nbl_3x2pt:nbl_WL, nbl_3x2pt:nbl_WL, :, :, :, :])
    cov_WL_SS_6D = deepcopy(cov_3x2pt_SS_10D[0, 0, 0, 0, :nbl_WL, :nbl_WL, :, :, :, :])
    cov_GC_SS_6D = deepcopy(cov_3x2pt_SS_10D[1, 1, 1, 1, :nbl_GC, :nbl_GC, :, :, :, :])
    cov_3x2pt_SS_10D = deepcopy(cov_3x2pt_SS_10D[:, :, :, :, :nbl_3x2pt, :nbl_3x2pt, :, :, :, :])

    # sum GO and SS in 6D (or 10D), not in 4D (it's the same)
    cov_WL_GS_6D = cov_WL_GO_6D + cov_WL_SS_6D
    cov_GC_GS_6D = cov_GC_GO_6D + cov_GC_SS_6D
    cov_WA_GS_6D = cov_WA_GO_6D + cov_WA_SS_6D
    cov_3x2pt_GS_10D = cov_3x2pt_GO_10D + cov_3x2pt_SS_10D

    # ! BNT transform
    if covariance_cfg['cov_BNT_transform']:

        print('BNT-transforming the covariance matrix...')
        start_time = time.perf_counter()

        # turn to dict for the BNT function
        cov_3x2pt_GO_10D_dict = mm.cov_10D_array_to_dict(cov_3x2pt_GO_10D, probe_ordering)
        cov_3x2pt_GS_10D_dict = mm.cov_10D_array_to_dict(cov_3x2pt_GS_10D, probe_ordering)

        X_dict = bnt_utils.build_X_matrix_BNT(BNT_matrix)
        cov_WL_GO_6D = bnt_utils.cov_BNT_transform(cov_WL_GO_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_WA_GO_6D = bnt_utils.cov_BNT_transform(cov_WA_GO_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_3x2pt_GO_10D_dict = bnt_utils.cov_3x2pt_BNT_transform(cov_3x2pt_GO_10D_dict, X_dict)

        cov_WL_GS_6D = bnt_utils.cov_BNT_transform(cov_WL_GS_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_WA_GS_6D = bnt_utils.cov_BNT_transform(cov_WA_GS_6D, X_dict, 'L', 'L', 'L', 'L')
        cov_3x2pt_GS_10D_dict = bnt_utils.cov_3x2pt_BNT_transform(cov_3x2pt_GS_10D_dict, X_dict)

        # revert to 10D arrays - this is not strictly necessary since cov_3x2pt_10D_to_4D accepts both a dictionary and
        # an array as input, but it's done to keep the variable names consistent
        cov_3x2pt_GO_10D = mm.cov_10D_dict_to_array(cov_3x2pt_GO_10D_dict, nbl_3x2pt, zbins, n_probes=2)
        cov_3x2pt_GS_10D = mm.cov_10D_dict_to_array(cov_3x2pt_GS_10D_dict, nbl_3x2pt, zbins, n_probes=2)

        print('Covariance matrices BNT-transformed in {:.2f} s'.format(time.perf_counter() - start_time))

    if GL_or_LG == 'GL':
        cov_XC_GO_6D = cov_3x2pt_GO_10D[1, 0, 1, 0, ...]
        cov_XC_SS_6D = cov_3x2pt_SS_10D[1, 0, 1, 0, ...]
        cov_XC_GS_6D = cov_3x2pt_GS_10D[1, 0, 1, 0, ...]
    elif GL_or_LG == 'LG':
        cov_XC_GO_6D = cov_3x2pt_GO_10D[0, 1, 0, 1, ...]
        cov_XC_SS_6D = cov_3x2pt_SS_10D[0, 1, 0, 1, ...]  # ! I'm doing this in a more exotic way above, for SS
        cov_XC_GS_6D = cov_3x2pt_GS_10D[0, 1, 0, 1, ...]
    else:
        raise ValueError('GL_or_LG must be "GL" or "LG"')

    # ! transform everything in 4D
    start = time.perf_counter()
    cov_WL_GO_4D = mm.cov_6D_to_4D(cov_WL_GO_6D, nbl_WL, zpairs_auto, ind_auto)
    cov_GC_GO_4D = mm.cov_6D_to_4D(cov_GC_GO_6D, nbl_GC, zpairs_auto, ind_auto)
    cov_WA_GO_4D = mm.cov_6D_to_4D(cov_WA_GO_6D, nbl_WA, zpairs_auto, ind_auto)
    cov_XC_GO_4D = mm.cov_6D_to_4D(cov_XC_GO_6D, nbl_3x2pt, zpairs_cross, ind_cross)
    cov_3x2pt_GO_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_GO_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(), GL_or_LG)

    cov_WL_GS_4D = mm.cov_6D_to_4D(cov_WL_GS_6D, nbl_WL, zpairs_auto, ind_auto)
    cov_GC_GS_4D = mm.cov_6D_to_4D(cov_GC_GS_6D, nbl_GC, zpairs_auto, ind_auto)
    cov_WA_GS_4D = mm.cov_6D_to_4D(cov_WA_GS_6D, nbl_WA, zpairs_auto, ind_auto)
    cov_XC_GS_4D = mm.cov_6D_to_4D(cov_XC_GS_6D, nbl_3x2pt, zpairs_cross, ind_cross)
    cov_3x2pt_GS_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_GS_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(), GL_or_LG)
    print('Covariance matrices reshaped (6D -> 4D) in {:.2f} s'.format(time.perf_counter() - start))

    cov_2x2pt_GO_4D = np.zeros((nbl_3x2pt, nbl_3x2pt, zpairs_cross + zpairs_auto, zpairs_cross + zpairs_auto))
    cov_2x2pt_GS_4D = np.zeros_like(cov_2x2pt_GO_4D)
    for ell1 in range(nbl_3x2pt):
        for ell2 in range(nbl_3x2pt):
            cov_2x2pt_GO_4D[ell1, ell2, :, :] = cov_3x2pt_GO_4D[ell1, ell2, zpairs_auto:, zpairs_auto:]
            cov_2x2pt_GS_4D[ell1, ell2, :, :] = cov_3x2pt_GS_4D[ell1, ell2, zpairs_auto:, zpairs_auto:]

    # ! transform everything in 2D
    start = time.perf_counter()
    cov_WL_GO_2D = mm.cov_4D_to_2D(cov_WL_GO_4D, block_index=block_index)
    cov_GC_GO_2D = mm.cov_4D_to_2D(cov_GC_GO_4D, block_index=block_index)
    cov_WA_GO_2D = mm.cov_4D_to_2D(cov_WA_GO_4D, block_index=block_index)
    cov_XC_GO_2D = mm.cov_4D_to_2D(cov_XC_GO_4D, block_index=block_index)
    cov_3x2pt_GO_2D = mm.cov_4D_to_2D(cov_3x2pt_GO_4D, block_index=block_index)
    cov_2x2pt_GO_2D = mm.cov_4D_to_2D(cov_2x2pt_GO_4D, block_index=block_index)

    cov_WL_GS_2D = mm.cov_4D_to_2D(cov_WL_GS_4D, block_index=block_index)
    cov_GC_GS_2D = mm.cov_4D_to_2D(cov_GC_GS_4D, block_index=block_index)
    cov_WA_GS_2D = mm.cov_4D_to_2D(cov_WA_GS_4D, block_index=block_index)
    cov_XC_GS_2D = mm.cov_4D_to_2D(cov_XC_GS_4D, block_index=block_index)
    cov_3x2pt_GS_2D = mm.cov_4D_to_2D(cov_3x2pt_GS_4D, block_index=block_index)
    cov_2x2pt_GS_2D = mm.cov_4D_to_2D(cov_2x2pt_GS_4D, block_index=block_index)

    cov_2x2pt_GO_2D = np.eye(cov_2x2pt_GO_2D.shape[0])
    cov_2x2pt_GS_2D = np.eye(cov_2x2pt_GS_2D.shape[0])
    print('Covariance matrices reshaped (4D -> 2D) in {:.2f} s'.format(time.perf_counter() - start))

    if covariance_cfg['cov_ell_cuts']:
        # perform the cuts on the 2D covs (way faster!)
        print('Performing ell cuts on the 2d covariance matrix...')
        cov_WL_GO_2D = mm.remove_rows_cols_array2D(cov_WL_GO_2D, ell_dict['idxs_to_delete_dict']['LL'])
        cov_GC_GO_2D = mm.remove_rows_cols_array2D(cov_GC_GO_2D, ell_dict['idxs_to_delete_dict']['GG'])
        cov_WA_GO_2D = mm.remove_rows_cols_array2D(cov_WA_GO_2D, ell_dict['idxs_to_delete_dict']['WA'])
        cov_XC_GO_2D = mm.remove_rows_cols_array2D(cov_XC_GO_2D, ell_dict['idxs_to_delete_dict'][GL_or_LG])
        cov_3x2pt_GO_2D = mm.remove_rows_cols_array2D(cov_3x2pt_GO_2D, ell_dict['idxs_to_delete_dict']['3x2pt'])
        # cov_2x2pt_GO_2D = mm.remove_rows_cols_array2D(cov_2x2pt_GO_2D, ell_dict['idxs_to_delete_dict']['2x2pt'])

        cov_WL_GS_2D = mm.remove_rows_cols_array2D(cov_WL_GS_2D, ell_dict['idxs_to_delete_dict']['LL'])
        cov_GC_GS_2D = mm.remove_rows_cols_array2D(cov_GC_GS_2D, ell_dict['idxs_to_delete_dict']['GG'])
        cov_WA_GS_2D = mm.remove_rows_cols_array2D(cov_WA_GS_2D, ell_dict['idxs_to_delete_dict']['WA'])
        cov_XC_GS_2D = mm.remove_rows_cols_array2D(cov_XC_GS_2D, ell_dict['idxs_to_delete_dict'][GL_or_LG])
        cov_3x2pt_GS_2D = mm.remove_rows_cols_array2D(cov_3x2pt_GS_2D, ell_dict['idxs_to_delete_dict']['3x2pt'])
        # cov_2x2pt_GS_2D = mm.remove_rows_cols_array2D(cov_2x2pt_GS_2D, ell_dict['idxs_to_delete_dict']['2x2pt'])

    ############################### save in dictionary ########################
    probe_names = ('WL', 'GC', '3x2pt', 'WA', 'XC', '2x2pt')

    covs_GO_4D = (cov_WL_GO_4D, cov_GC_GO_4D, cov_3x2pt_GO_4D, cov_WA_GO_4D, cov_XC_GO_4D, cov_2x2pt_GO_4D)
    covs_GS_4D = (cov_WL_GS_4D, cov_GC_GS_4D, cov_3x2pt_GS_4D, cov_WA_GS_4D, cov_XC_GS_4D, cov_2x2pt_GS_4D)

    covs_GO_2D = (cov_WL_GO_2D, cov_GC_GO_2D, cov_3x2pt_GO_2D, cov_WA_GO_2D, cov_XC_GO_2D, cov_2x2pt_GO_2D)
    covs_GS_2D = (cov_WL_GS_2D, cov_GC_GS_2D, cov_3x2pt_GS_2D, cov_WA_GS_2D, cov_XC_GS_2D, cov_2x2pt_GS_2D)

    if covariance_cfg['save_cov_SSC']:
        warnings.warn('2x2pt MISSING')
        cov_WL_SS_4D = mm.cov_6D_to_4D(cov_WL_SS_6D, nbl_WL, zpairs_auto, ind_auto)
        cov_GC_SS_4D = mm.cov_6D_to_4D(cov_GC_SS_6D, nbl_GC, zpairs_auto, ind_auto)
        cov_WA_SS_4D = mm.cov_6D_to_4D(cov_WA_SS_6D, nbl_WA, zpairs_auto, ind_auto)
        cov_XC_SS_4D = mm.cov_6D_to_4D(cov_XC_SS_6D, nbl_3x2pt, zpairs_cross, ind_cross)
        cov_3x2pt_SS_4D = mm.cov_3x2pt_10D_to_4D(cov_3x2pt_SS_10D, probe_ordering, nbl_3x2pt, zbins, ind.copy(),
                                                 GL_or_LG)

        cov_WL_SS_2D = mm.cov_4D_to_2D(cov_WL_SS_4D, block_index=block_index)
        cov_GC_SS_2D = mm.cov_4D_to_2D(cov_GC_SS_4D, block_index=block_index)
        cov_WA_SS_2D = mm.cov_4D_to_2D(cov_WA_SS_4D, block_index=block_index)
        cov_XC_SS_2D = mm.cov_4D_to_2D(cov_XC_SS_4D, block_index=block_index)
        cov_3x2pt_SS_2D = mm.cov_4D_to_2D(cov_3x2pt_SS_4D, block_index=block_index)

        # covs_SS_4D = (cov_WL_SS_4D, cov_GC_SS_4D, cov_3x2pt_SS_4D, cov_WA_SS_4D)
        covs_SS_2D = (cov_WL_SS_2D, cov_GC_SS_2D, cov_3x2pt_SS_2D, cov_WA_SS_2D, cov_XC_SS_2D)

        for probe_name, cov_SS_2D in zip(probe_names, covs_SS_2D):
            cov_dict[f'cov_{probe_name}_SS_2D'] = cov_SS_2D  # cov_dict[f'cov_{probe_name}_SS_4D'] = cov_SS_4D

    for probe_name, cov_GO_4D, cov_GO_2D, cov_GS_4D, cov_GS_2D in zip(probe_names, covs_GO_4D, covs_GO_2D, covs_GS_4D,
                                                                      covs_GS_2D):
        # save 4D
        # cov_dict[f'cov_{probe_name}_GO_4D'] = cov_GO_4D
        # cov_dict[f'cov_{probe_name}_GS_4D'] = cov_GS_4D
        # if covariance_cfg['save_cov_SSC']:

        # save 2D
        cov_dict[f'cov_{probe_name}_GO_2D'] = cov_GO_2D
        cov_dict[f'cov_{probe_name}_GS_2D'] = cov_GS_2D

    # '2DCLOE', i.e. the 'multi-diagonal', non-square blocks ordering, only for 3x2pt
    # note: we found out that this is not actually used in CLOE...
    if covariance_cfg['save_2DCLOE']:
        cov_dict[f'cov_3x2pt_GO_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_GO_4D, zbins, block_index='ell')
        cov_dict[f'cov_3x2pt_SS_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_SS_4D, zbins, block_index='ell')
        cov_dict[f'cov_3x2pt_GS_2DCLOE'] = mm.cov_4D_to_2DCLOE_3x2pt(cov_3x2pt_GS_4D, zbins, block_index='ell')

    print('Covariance matrices computed')

    return cov_dict


def save_cov(cov_folder, covariance_cfg, cov_dict, cases_tosave, **variable_specs):

    ell_max_WL = variable_specs['ell_max_WL']
    ell_max_GC = variable_specs['ell_max_GC']
    ell_max_3x2pt = variable_specs['ell_max_3x2pt']
    nbl_WL = variable_specs['nbl_WL']
    nbl_GC = variable_specs['nbl_GC']
    nbl_3x2pt = variable_specs['nbl_3x2pt']
    nbl_WA = variable_specs['nbl_WA']

    # which file format to use
    if covariance_cfg['cov_file_format'] == 'npy':
        save_funct = np.save
        extension = 'npy'
    elif covariance_cfg['cov_file_format'] == 'npz':
        save_funct = np.savez_compressed
        extension = 'npz'
    else:
        raise ValueError('cov_file_format not recognized: must be "npy" or "npz"')

    for ndim in (2, 4, 6):

        if covariance_cfg[f'save_cov_{ndim}D']:

            # set probes to save; the ndim == 6 case is different
            probe_list = ['WL', 'GC', '3x2pt', 'WA']
            ellmax_list = [ell_max_WL, ell_max_GC, ell_max_3x2pt, ell_max_WL]
            nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt, nbl_WA]
            # in this case, 3x2pt is saved in 10D as a dictionary
            if ndim == 6:
                probe_list = ['WL', 'GC', 'WA']
                ellmax_list = [ell_max_WL, ell_max_GC, ell_max_WL]
                nbl_list = [nbl_WL, nbl_GC, nbl_WA]

            for which_cov in cases_tosave:

                for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                    cov_filename = covariance_cfg['cov_filename'].format(which_cov=which_cov, probe=probe,
                                                                         ell_max=ell_max, nbl=nbl, ndim=ndim,
                                                                         **variable_specs)
                    save_funct(f'{cov_folder}/{cov_filename}.{extension}',
                               cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'])  # save in .npy or .npz

                # in this case, 3x2pt is saved in 10D as a dictionary
                # TODO these pickle files are too heavy, probably it's best to revert to npz
                if ndim == 6:
                    cov_3x2pt_filename = covariance_cfg['cov_filename'].format(which_cov=which_cov, probe='3x2pt',
                                                                               ell_max=ell_max_3x2pt, nbl=nbl_3x2pt,
                                                                               ndim=10, **variable_specs)
                    with open(f'{cov_folder}/{cov_3x2pt_filename}.pickle', 'wb') as handle:
                        pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)

            print(f'Covariance matrices saved in {covariance_cfg["cov_file_format"]}')

    # save in .dat for Vincenzo, only in the optimistic case and in 2D
    if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
        for probe, probe_vinc in zip(['WL', 'GC', '3x2pt', 'WA'], ['WLO', 'GCO', '3x2pt', 'WLA']):
            for GOGS_folder, GOGS_filename in zip(['GaussOnly', 'GaussSSC'], ['GO', 'GS']):
                cov_filename_vincenzo = covariance_cfg['cov_filename_vincenzo'].format(probe_vinc=probe_vinc,
                                                                                       GOGS_filename=GOGS_filename,
                                                                                       **variable_specs)
                np.savetxt(f'{cov_folder}/{GOGS_folder}/{cov_filename_vincenzo}',
                           cov_dict[f'cov_{probe}_{GOGS_filename}_2D'], fmt='%.8e')
        print('Covariance matrices saved')
