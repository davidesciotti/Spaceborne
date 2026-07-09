import contextlib
import itertools
import pickle
import subprocess
import time
import warnings
from collections.abc import Sequence
from copy import deepcopy
from functools import partial

import jax.numpy as jnp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
from jax import jit
from scipy.integrate import simpson as simps
from scipy.interpolate import CubicSpline, RectBivariateSpline, interp1d
from scipy.special import jv

import spaceborne.constants as const

"""
COVARIANCE DICTIONARY STRUCTURE AND FUNCTIONS USED TO RESHAPE IT
This is a small vademecum for the structure of the covariance dictionary 
used in Spaceborne and the functions used to reshape it.

The dictionary is structured as follows:
    cov_dict[term][probe_ab, probe_cd][dim] = np.ndarray (6d, 4d or 2d)

The possible keys are:
    term:              ['sva', 'sn', 'mix', 'g', 'ssc', 'cng', 'tot']
    probe_ab/probe_cd: - one of const.HS_DIAG_PROBES or const.RS_DIAG_PROBES, 
                         depending on the space ('harmonic' or 'real')
                       - '3x2pt' (special case, see below)
    dim:               ['2d', '4d', '6d']
    
An "exception" to this is the 3x2pt, for which the structure is:
    cov_dict['3x2pt'][dim] = np.ndarray (2d)
so beware that:
- The probe key is not a 2-tuple
- There are no 6d or 4d keys

The general philosophy is the following:
1. The covariance is computed for the individual probe blocks, trying to leverage 
   symmetries for computational efficiency (e.g., if LL, GG and their cross-cov 
   is required, only the LLLL, LLGG, GGGG blocks will acually be computed, and 
   GGLL will be filled by symmetry).
   The cov blocks will generally be computed in 4d (again, for efficiency reasons), 
   with the exception of:
   - the OneCovariance outputs (reshaped direcly in 6d) 
   - the harmonic-space Gaussian covariance (the einsum function outputs a 10d array but 
     computes everything at once, so it's quite convenient and we leave it as it is for 
     the moment)
2. The [sva, sn, mix, g, ssc, cng] terms are filled in this way (sva, sn mix) are 
   deleted if the split-Gaussian cov is not required
   



To reshape the covariance *blocks*, you can use:
6d -> 4d: 
    cov_dict = cov_dict_6d_probe_blocks_to_4d_and_2d()


[OLD]
To create the 4d 3x2pt covariance, you can use (you have to loop over terms here!):

cov_dict (4d) -> 3x2pt arr (4d): 
    cov_dict[term]['3x2pt']['4d'] = cov_dict_4d_blocks_4d_3x2pt (returns an array)
3x2pt arr (4d) -> 3x2pt arr (2d): 
    cov_dict[term]['3x2pt']['2d'] = cov_hs_obj.cov_4D_to_2D_3x2pt_func(
            cov_dict[term]['3x2pt']['4d'], **cov_hs_obj.cov_4D_to_2D_3x2pt_func_kw
            )

[NEW]
The 3x2pt covariance is assembled directly in 2d, with the function 
sl.build_cov_3x2pt_2d (returns an array)

Naming conventions (just to ease the notation):
- cov_term_dict = cov_dict[term]
- cov_probe_dict = cov_dict[term][probe_ab, probe_cd]
- cov_dim_dict = cov_dict[term][probe_ab, probe_cd][dim]
"""


def get_zsteps(z_min, z_max, delta_z):
    """
    Compute the number of grid points for linspace given a desired step size.

    Returns the count needed so that np.linspace(z_min, z_max, count) produces
    a grid with actual spacing <= delta_z (endpoint-inclusive).
    """
    if delta_z <= 0:
        raise ValueError(f'delta_z must be positive, got {delta_z}')
    if z_max <= z_min:
        raise ValueError(f'z_max must be greater than z_min, got {z_max=}, {z_min=}')
    return int(np.ceil((z_max - z_min) / delta_z)) + 1


def hartlap_factor(n_sim: int, n_data: int) -> float:
    """hartlap correction factor for the precision matrix:
    Cov^{-1}_{corrected} = hartlap_factor * Cov^{-1}_{measured}
    where hartlap_factor is the value returned by this function.

    Note: Requires n_sim > n_data + 2 for a positive correction factor.
    """
    if n_sim <= 1:
        raise ValueError('n_sim must be > 1 to avoid division by zero')

    if n_sim <= n_data + 2:
        import warnings

        warnings.warn(
            f'Hartlap factor is non-positive for n_sim={n_sim}, n_data={n_data}. '
            'Requires n_sim > n_data + 2 for a valid correction.',
            stacklevel=2,
        )

    return (n_sim - n_data - 2) / (n_sim - 1)


def percival_factor(n_sim, n_data, n_param):
    """
    Percival et al. 2014 correction factor for the inverse covariance matrix.
    Combined Hartlap + Percival factors.
    """
    A = 2 / (n_sim - n_data - 1) / (n_sim - n_data - 4)
    B = (n_sim - n_data - 2) / (n_sim - n_data - 1) / (n_sim - n_data - 4)
    m1 = 1 + B * (n_data - n_param)
    m2 = 1 + A + B * (n_data - n_param)
    beta = m1 / m2

    return beta


def get_probe_combs_wrapper(
    obs_space: str, probe_selection: dict, cross_cov: bool
) -> dict:
    """Wrapper function to produce all the different probe combinations lists,
    based on the configs and selected observable"""

    if obs_space == 'harmonic':
        DIAG_PROBES = const.HS_DIAG_PROBES
        ALL_PROBE_COMBS = const.HS_ALL_PROBE_COMBS
    elif obs_space == 'real':
        DIAG_PROBES = const.RS_DIAG_PROBES
        ALL_PROBE_COMBS = const.RS_ALL_PROBE_COMBS
    elif obs_space == 'cosebis':
        DIAG_PROBES = const.CS_DIAG_PROBES
        ALL_PROBE_COMBS = const.CS_ALL_PROBE_COMBS
    else:
        raise ValueError(f'Unknown observables space: {obs_space:s}')

    # start from the probe names
    unique_probe_names = [p for p in DIAG_PROBES if probe_selection[p]]

    # add cross terms if requested
    unique_probe_combs = build_probe_list(
        unique_probe_names, include_cross_terms=cross_cov
    )

    # probe combinations to be filled by symmetry or to exclude altogether
    symm_probe_combs, nonreq_probe_combs = get_probe_combs(
        unique_probe_combs, space=obs_space
    )

    # required probe combinations to include in the 2d arrays (must include the
    # cross-terms!)
    _req_probe_combs_2d = build_probe_list(unique_probe_names, include_cross_terms=True)

    # as req_probe_combs_2d still only contains the upper triangle,
    # add the symmetric blocks
    symm_probe_combs_2d, _ = get_probe_combs(_req_probe_combs_2d, space=obs_space)
    _req_probe_combs_2d += symm_probe_combs_2d

    # reorder!
    req_probe_combs_2d = []
    for probe in ALL_PROBE_COMBS:
        if probe in _req_probe_combs_2d:
            req_probe_combs_2d.append(probe)

    nonreq_probe_combs = [p for p in nonreq_probe_combs if p in req_probe_combs_2d]

    return {
        'unique_probe_combs': unique_probe_combs,
        'req_probe_combs_2d': req_probe_combs_2d,
        'nonreq_probe_combs': nonreq_probe_combs,
        'symm_probe_combs': symm_probe_combs,
    }


def copy_dict_leaf_level(original_dict, new_dict):

    for term in original_dict:
        for probe_2tpl in original_dict[term]:
            for dim in original_dict[term][probe_2tpl]:
                array = original_dict[term][probe_2tpl][dim]
                if array is not None:
                    new_dict[term][probe_2tpl][dim] = deepcopy(array)
                else:
                    new_dict[term][probe_2tpl][dim] = None

    return new_dict


def build_cl_3x2pt_5d(
    cl_ll_3d: np.ndarray, cl_gl_3d: np.ndarray, cl_gg_3d: np.ndarray
) -> np.ndarray:
    """Constructs the 5D Cl array for 3x2pt from the individual 3D Cl arrays."""

    assert cl_ll_3d.ndim == 3, 'cl_ll_3d must be a 3D array'
    assert cl_gl_3d.ndim == 3, 'cl_gl_3d must be a 3D array'
    assert cl_gg_3d.ndim == 3, 'cl_gg_3d must be a 3D array'

    assert cl_ll_3d.shape == cl_gl_3d.shape == cl_gg_3d.shape, (
        'cl_ll_3d, cl_gl_3d and cl_gg_3d must have the same shape'
    )
    assert cl_ll_3d.dtype == cl_gl_3d.dtype == cl_gg_3d.dtype, (
        'cl_ll_3d, cl_gl_3d and cl_gg_3d must have the same dtype'
    )

    cl_3x2pt_5d = np.zeros((2, 2, *cl_ll_3d.shape), dtype=cl_ll_3d.dtype)
    cl_3x2pt_5d[0, 0] = cl_ll_3d
    cl_3x2pt_5d[1, 0] = cl_gl_3d
    cl_3x2pt_5d[0, 1] = cl_gl_3d.transpose(0, 2, 1)
    cl_3x2pt_5d[1, 1] = cl_gg_3d

    return cl_3x2pt_5d


def sum_split_g_terms_allprobeblocks_alldims(cov_dict) -> None:
    # small sanity check probe combinations must match for terms (sva, sn, mix)
    if not (cov_dict['sva'].keys() == cov_dict['sn'].keys() == cov_dict['mix'].keys()):
        raise ValueError(
            'The probe combinations keys in the SVA, SN and MIX covariance '
            'dictionaries do not match!'
        )

    # sanity check: all the probes must match
    probes_sva = set(cov_dict['sva'].keys())
    probes_sn = set(cov_dict['sn'].keys())
    probes_mix = set(cov_dict['mix'].keys())
    if not (probes_sva == probes_sn == probes_mix):
        raise ValueError(
            'The probe combinations in the SVA, SN and MIX covariance '
            'dictionaries do not match!'
        )

    # now sum the terms to get the Gaussian, for all probe combinations and
    # dimensions
    for probe_2tpl in cov_dict['sva']:
        if probe_2tpl == '3x2pt':
            continue  # skip 3x2pt, built later

        # sanity check: all the dimensions must match
        dims_sva = set(cov_dict['sva'][probe_2tpl].keys())
        dims_sn = set(cov_dict['sn'][probe_2tpl].keys())
        dims_mix = set(cov_dict['mix'][probe_2tpl].keys())
        if not (dims_sva == dims_sn == dims_mix):
            raise ValueError(
                'The probe combinations in the SVA, SN and MIX covariance '
                'dictionaries do not match!'
            )

        # for each dim, perform the sum
        for dim in ['2d', '4d', '6d']:
            sva = cov_dict['sva'][probe_2tpl][dim]
            sn = cov_dict['sn'][probe_2tpl][dim]
            mix = cov_dict['mix'][probe_2tpl][dim]

            # Check consistency: either all None or all not None
            none_count = sum([sva is None, sn is None, mix is None])
            if none_count not in {0, 3}:
                raise ValueError(
                    f'For probe {probe_2tpl} and dim {dim}, '
                    f'SVA, SN, and MIX must all be None or all be non-None. '
                    f'Found: SVA={sva is not None}, SN={sn is not None}, '
                    f'MIX={mix is not None}'
                )

            if none_count == 3:
                continue

            cov_dict['g'][probe_2tpl][dim] = (
                cov_dict['sva'][probe_2tpl][dim]
                + cov_dict['sn'][probe_2tpl][dim]
                + cov_dict['mix'][probe_2tpl][dim]
            )


def fill_remaining_probe_blocks_6d(
    cov_dict, term, symm_probe_combs, nonreq_probe_combs, space, nbx, zbins
):
    """Fill the remaining probe combinations by symmetry or
    set them to 0 if not required."""

    # * fill the symmetric counterparts of the required blocks
    # * (excluding diagonal blocks)
    for probe_abcd in symm_probe_combs:
        probe_ab, probe_cd = split_probe_name(probe_abcd, space=space)
        cov_cdab = cov_dict[term][probe_cd, probe_ab]['6d']
        cov = (cov_cdab.transpose(1, 0, 4, 5, 2, 3)).copy()
        cov_dict[term][probe_ab, probe_cd]['6d'] = cov

    # * if block is not required, set it to 0
    for probe_abcd in nonreq_probe_combs:
        probe_ab, probe_cd = split_probe_name(probe_abcd, space=space)
        probe_2tpl = (probe_ab, probe_cd)

        cov_dict[term][probe_2tpl]['6d'] = np.zeros(
            (nbx, nbx, zbins, zbins, zbins, zbins)
        )


def postprocess_cov_dict(
    cov_dict,
    obs_space,
    nbx,
    ind_auto,
    ind_cross,
    zpairs_auto,
    zpairs_cross,
    block_index,
    cov_ordering_2d,
    req_probe_combs_2d,
):
    """
    Space-agnostic postprocessing:
    1. Reshape individual probe blocks from 6d to 4d and 2d, for each term
    2. Build 3x2pt based on the required probe combinations
    3. Sum g, ssc and cng to get tot term
    """

    # Step 1: Reshape probe-specific 6d → 4d → 2d
    cov_dict_6d_probe_blocks_to_4d_and_2d(
        cov_dict=cov_dict,
        obs_space=obs_space,
        nbx=nbx,
        ind_auto=ind_auto,
        ind_cross=ind_cross,
        zpairs_auto=zpairs_auto,
        zpairs_cross=zpairs_cross,
        block_index=block_index,
    )

    # Step 2: Build 3x2pt 2d covs
    for term in cov_dict:
        probe_list = cov_dict[term].keys()
        all_none = all(
            cov_dict[term][probe_2tpl]['4d'] is None for probe_2tpl in probe_list
        )

        if term != 'tot' and not all_none:
            cov_dict[term]['3x2pt']['2d'] = build_cov_3x2pt_2d(
                cov_term_dict=cov_dict[term],
                cov_ordering_2d=cov_ordering_2d,
                obs_space=obs_space,
            )

    # Step 3: Sum to get 'tot'
    set_cov_tot_2d_and_6d(
        cov_dict=cov_dict, req_probe_combs_2d=req_probe_combs_2d, space=obs_space
    )

    return cov_dict


def symmetrize_and_fill_probe_blocks(
    cov_term_dict: dict,
    dim: str,
    unique_probe_combs: list[str],
    nonreq_probe_combs: list[str],
    obs_space: str,
    nbx: int,
    zbins: int | None,
    ind_dict: dict,
    msg: str,
) -> dict:
    """Function to symmetrize and fill 4d covariance matrices.
    Say we want the LL, GG covariance blocks, plus their cross-covariance.

    Then the unique probe combinations are (I display the list in this way for clarity,
    it's actually a flat list [LLLL, LLGG, GGLL]):
    [LLLL, LLGG,
     -   , GGLL]
    The symmetric probe combinations are:
    [LLGG]
    The non-required probe combinations are:
    []
    """

    # Validate dim parameter
    if dim not in ('4d', '6d'):
        raise ValueError(f"dim must be '4d' or '6d', got: {dim}")

    # obtain the probe combinations to be filled by symmetry
    symm_probe_combs, _ = get_probe_combs(
        unique_probe_combs=unique_probe_combs, space=obs_space
    )

    # fill by symmetry
    for probe_abcd in symm_probe_combs:
        probe_ab, probe_cd = split_probe_name(probe_abcd, space=obs_space)
        probe_2tpl_orig = (probe_ab, probe_cd)
        probe_2tpl_symm = (probe_cd, probe_ab)
        print(f'{msg}filling probe combination {(probe_ab, probe_cd)} by symmetry')

        if dim == '4d':
            transpose_axes = (1, 0, 3, 2)
        elif dim == '6d':
            transpose_axes = (1, 0, 4, 5, 2, 3)

        cov_term_dict[probe_2tpl_orig][dim] = (
            cov_term_dict[probe_2tpl_symm][dim].transpose(*transpose_axes)
        ).copy()

    # # * if block is not required, set it to 0
    # set to 0 the non-required probe combinations (note that these are the blocks
    # which appear in the final nx2pt 2d covariance matrix! the blocks which are not
    # required at all, e.g. LLGL if we as for the LL, GG covariance)
    for probe_abcd in nonreq_probe_combs:
        probe_ab, probe_cd = split_probe_name(probe_abcd, space=obs_space)
        probe_2tpl = (probe_ab, probe_cd)
        zpairs_ab = ind_dict[probe_ab].shape[0]
        zpairs_cd = ind_dict[probe_cd].shape[0]
        print(f'{msg}skipping probe combination {(probe_ab, probe_cd)}')

        if dim == '4d':
            shape = (nbx, nbx, zpairs_ab, zpairs_cd)
        elif dim == '6d':
            shape = (nbx, nbx, zbins, zbins, zbins, zbins)

        cov_term_dict[probe_2tpl][dim] = np.zeros(shape)

    return cov_term_dict


def set_cov_tot_2d_and_6d(cov_dict: dict, req_probe_combs_2d: list, space: str) -> dict:
    """
    Sums G, SSC and cNG 2D covs to get the total covariance

    Note: simply looping over terms would sum sva + sn + mix + g, resulting in
    double counting of the Gaussian term.
    """

    # if neither ssc nor cng are present, no 'tot' term will be present either
    if 'tot' not in cov_dict:
        return cov_dict

    for dim in ('2d', '6d'):
        for probe_abcd in req_probe_combs_2d:
            probe_2tpl = split_probe_name(probe_abcd, space=space)

            # concise way to check that the key exists and the dict is not empty
            g = cov_dict['g'][probe_2tpl][dim] if 'g' in cov_dict else 0
            ssc = cov_dict['ssc'][probe_2tpl][dim] if 'ssc' in cov_dict else 0
            cng = cov_dict['cng'][probe_2tpl][dim] if 'cng' in cov_dict else 0

            cov_dict['tot'][probe_2tpl][dim] = g + ssc + cng

    # do the same for 3x2pt (for which only 2d exists)
    g = cov_dict['g']['3x2pt']['2d'] if 'g' in cov_dict else 0
    ssc = cov_dict['ssc']['3x2pt']['2d'] if 'ssc' in cov_dict else 0
    cng = cov_dict['cng']['3x2pt']['2d'] if 'cng' in cov_dict else 0

    cov_dict['tot']['3x2pt']['2d'] = g + ssc + cng

    return cov_dict


def symmetrize_probe_cov_dict_6d(cov_dict: dict):
    """Fills the symmetric 6D probe combinations (e.g., given gggt, fills gtgg)"""

    # iterate through the different terms and shorten the name of the dict
    for probe_cov_dict in cov_dict.values():
        # Create a list of keys to avoid modifying dict during iteration
        existing_probe_2tpl = list(probe_cov_dict.keys())

        # if present, remove "3x2pt" from the list
        existing_probe_2tpl = [
            probe for probe in existing_probe_2tpl if '3x2pt' not in probe
        ]

        # Validate that all keys are 2-tuples
        for _probe in existing_probe_2tpl:
            if not isinstance(_probe, tuple) or len(_probe) != 2:
                raise ValueError(
                    f'Expected 2-tuple key, got {type(_probe)} with value {_probe}'
                )

        for probe_ab, probe_cd in existing_probe_2tpl:
            # Only add symmetric if it doesn't already exist and it's
            # not auto-correlation
            if (probe_cd, probe_ab) not in probe_cov_dict and probe_ab != probe_cd:
                cov = (
                    probe_cov_dict[probe_ab, probe_cd]['6d']
                    .transpose(1, 0, 4, 5, 2, 3)
                    .copy()
                )
                probe_cov_dict[probe_cd, probe_ab] = {'6d': cov}

    return cov_dict


def validate_cov_dict_structure(cov_dict: dict, obs_space: str):
    """
    Validates that cov_dict follows the structure:
    cov_dict[term][probe_ab, probe_cd][dim] = np.ndarray

    Additionally, the function checks that the term, probe_ab, probe_cd, and dim
    keys have one of the expected values (among all the possible ones!)

    Args:
        cov_dict: Dictionary to validate
        obs_space: 'harmonic' or 'real'
        expected_probes: Optional list of expected probe tuples
        expected_dims: Optional list of expected dimensions (e.g., ['6d', '8d'])
    """
    expected_terms = ['sva', 'sn', 'mix', 'g', 'ssc', 'cng', 'tot']
    expected_dims = ['2d', '4d', '6d']
    if obs_space == 'harmonic':
        expected_probes_ab = const.HS_DIAG_PROBES
    elif obs_space == 'real':
        expected_probes_ab = const.RS_DIAG_PROBES
    else:
        raise ValueError('`obs_space` must be in ["harmonic", "real"]')

    if not isinstance(cov_dict, dict):
        raise ValueError('cov_dict must be a dictionary')

    for term, cov_probe_dict in cov_dict.items():
        if term not in expected_terms:
            raise ValueError(
                f'Unexpected term: {term}, expected one of {expected_terms}'
            )
        if not isinstance(cov_probe_dict, dict):
            raise ValueError(
                f"Term '{term}' must contain a dictionary, got {type(cov_probe_dict)}"
            )

        for probe_2tpl, cov_dim_dict in cov_probe_dict.items():
            # check the probe names
            if probe_2tpl[0] not in expected_probes_ab:
                raise ValueError(
                    f'Unexpected probe_ab: {probe_2tpl[0]}, expected one of '
                    f'{expected_probes_ab}'
                )
            if probe_2tpl[1] not in expected_probes_ab:
                raise ValueError(
                    f'Unexpected probe_ab: {probe_2tpl[1]}, expected one of '
                    f'{expected_probes_ab}'
                )
            # Validate probe key is a tuple of 2 elements
            if not isinstance(probe_2tpl, tuple) or len(probe_2tpl) != 2:
                raise ValueError(
                    f"Probe key {probe_2tpl} in term '{term}' must be "
                    'a tuple of 2 elements'
                )
            if not isinstance(cov_dim_dict, dict):
                raise ValueError(
                    f"Probe {probe_2tpl} in term '{term}' must contain "
                    f'a dictionary, got {type(cov_dim_dict)}'
                )

            for dim, value in cov_dim_dict.items():
                if dim not in expected_dims:
                    raise ValueError(
                        f"Unexpected dimension '{dim}' for probe {probe_2tpl} in term "
                        f"'{term}', expected one of {expected_dims}"
                    )
                if not isinstance(value, np.ndarray):
                    raise ValueError(
                        f"Value for dim '{dim}' of probe {probe_2tpl} in term "
                        f"'{term}' must be a numpy array, got {type(value)}"
                    )


def split_probe_name(
    full_probe_name: str, space: str, valid_probes: Sequence[str] | None = None
) -> tuple[str, str]:
    """Splits a full probe name (e.g., 'gtxim') into two component probes."""

    # this is the default: use hardcoded probe names
    if valid_probes is None:
        if space == 'harmonic':
            prefix = 'HS'
        elif space == 'real':
            prefix = 'RS'
        elif space == 'cosebis':
            prefix = 'CS'
        else:
            raise ValueError(
                f'`space` needs to be one of `harmonic` `real`, or `cosebis`, got {space}'
            )
        valid_probes = const.__getattribute__(f'{prefix}_DIAG_PROBES')
    else:
        assert isinstance(valid_probes, (list, tuple)), 'valid_probes must be a list'

    # Try splitting at each possible position
    for i in range(1, len(full_probe_name)):
        probe_ab, probe_cd = full_probe_name[:i], full_probe_name[i:]
        if probe_ab in valid_probes and probe_cd in valid_probes:
            return probe_ab, probe_cd

    raise ValueError(
        f'Invalid probe name: {full_probe_name}. '
        f'Expected two of {valid_probes} concatenated.'
    )


def compare_2d_covs(
    cov_a,
    cov_b,
    name_a,
    name_b,
    title,
    diff_threshold,
    compare_cov_2d=True,
    compare_corr_2d=True,
    compare_diag=True,
    compare_flat=True,
    compare_spectrum=True,
):
    # compare covariance
    if compare_cov_2d:
        compare_arrays(
            cov_a,
            cov_b,
            name_a,
            name_b,
            log_array=True,
            log_diff=False,
            abs_val=True,
            plot_diff_threshold=diff_threshold,
            title=title,
            early_return=False,
        )

    # compare correlation
    if compare_corr_2d:
        corr_a = cov2corr(cov_a)
        corr_b = cov2corr(cov_b)
        matshow_arr_kw = {'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1}
        compare_arrays(
            corr_a,
            corr_b,
            name_a,
            name_b,
            log_array=False,
            log_diff=False,
            matshow_arr_kw=matshow_arr_kw,
            plot_diff_hist=False,
            plot_diff_threshold=diff_threshold,
            title=title,
            early_return=False,
        )

    # compare cov diag
    if compare_diag:
        compare_funcs(
            x=None,
            y={
                f'{name_a}': np.diag(np.abs(cov_a)),
                f'{name_b}': np.diag(np.abs(cov_b)),
            },
            logscale_y=[True, False],
            ylim_diff=[-100, 100],
            title=title + ' abs diag',
        )

    # compare cov flat
    if compare_flat:
        compare_funcs(
            x=None,
            y={
                f'{name_a}': np.abs(cov_a).flatten(),
                f'{name_b}': np.abs(cov_b).flatten(),
            },
            logscale_y=[True, False],
            ylim_diff=[-100, 100],
            title=title + ' abs flat',
        )

    # compare SB against mat - cov spectrum
    if compare_spectrum:
        eig_a = np.linalg.eigvals(cov_a)
        eig_b = np.linalg.eigvals(cov_b)
        compare_funcs(
            x=None,
            y={f'eig {name_a}': eig_a, f'eig {name_b}': eig_b},
            logscale_y=[True, False],
            ylim_diff=[-100, 100],
            title=title + ' eig',
        )


def build_probe_list(probes, include_cross_terms=False):
    """Return the list of probe combinations to compute.

    Parameters
    ----------
    probes : list[str]
        List of individual probes to include, e.g. ['LL', 'GL', 'GG'].
    include_cross_terms : bool
        If True, include cross-combinations between different probes.

    Returns
    -------
    list[str]
        List of probe combinations, e.g. ['LLLL', 'LLGL', ...]

    """
    if not include_cross_terms:
        return [p + p for p in probes]

    # Sort to ensure consistent ordering
    # probes = sorted(probes)
    return [p1 + p2 for p1, p2 in itertools.combinations_with_replacement(probes, 2)]


def get_probe_combs(unique_probe_combs, space):
    """Given the desired probe combinations, builds a list the ones to be filled by
    symmetry and the ones fo be skipped"""

    assert space in ['harmonic', 'real', 'cosebis'], 'Invalid space specified'

    # get probe combinations' names
    if space == 'harmonic':
        prefix = 'HS'
    elif space == 'real':
        prefix = 'RS'
    elif space == 'cosebis':
        prefix = 'CS'

    ALL_PROBE_COMBS = const.__getattribute__(f'{prefix}_ALL_PROBE_COMBS')
    DIAG_PROBE_COMBS = const.__getattribute__(f'{prefix}_DIAG_PROBE_COMBS')

    # sanity checks
    for probe in unique_probe_combs:
        if probe not in ALL_PROBE_COMBS:
            raise ValueError(f'Probe {probe} not found in {ALL_PROBE_COMBS}')
        # real space probes have variable length
        if len(probe) != 4 and space == 'harmonic':
            raise ValueError(f'Probe {probe} must have length 4')

    # take the requested probes which are not diagonal
    _symm_probe_combs = set(unique_probe_combs) - set(DIAG_PROBE_COMBS)

    # invert probe_a, probe_b <-> probe_c, probe_d
    symm_probe_combs = []
    for probe in _symm_probe_combs:
        probe_ab, probe_cd = split_probe_name(probe, space)
        symm_probe_combs.append(probe_cd + probe_ab)

    # lastly, find the remaining (non required) probe combinations
    nonreq_probe_combs = (
        set(ALL_PROBE_COMBS) - set(unique_probe_combs) - set(symm_probe_combs)
    )

    return symm_probe_combs, nonreq_probe_combs


@contextlib.contextmanager
def timer(msg):
    print(msg, flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        stop = time.perf_counter()
        print(f'...done in {stop - start:.2f} s', flush=True)


def bin_2d_array(
    cov: np.ndarray,
    ells_in: np.ndarray,
    ells_out: np.ndarray,
    ells_out_edges: np.ndarray,
    weights_in: np.ndarray | None,
    which_binning: str = 'sum',
    interpolate: bool = True,
):
    assert cov.shape[0] == cov.shape[1] == len(ells_in), (
        'ells_in must be the same length as the covariance matrix'
    )
    assert len(ells_out) == len(ells_out_edges) - 1, (
        'ells_out must be the same length as the number of edges - 1'
    )
    assert which_binning in ['sum', 'integral'], (
        'which_binning must be either "sum" or "integral"'
    )

    binned_cov = np.zeros((len(ells_out), len(ells_out)))
    cov_interp_func = RectBivariateSpline(ells_in, ells_in, cov)

    ells_edges_low = ells_out_edges[:-1]
    ells_edges_high = ells_out_edges[1:]

    if weights_in is None:
        weights_in = np.ones_like(ells_in)

    assert len(weights_in) == len(ells_in), (
        'weights_in must be the same length as ells_in'
    )

    assert isinstance(interpolate, bool), 'interpolate must be a boolean'

    # Loop over the output bins
    for ell1_idx, _ in enumerate(ells_out):
        for ell2_idx, _ in enumerate(ells_out):
            # Get ell min/max for the current bins
            ell1_min = ells_edges_low[ell1_idx]
            ell1_max = ells_edges_high[ell1_idx]
            ell2_min = ells_edges_low[ell2_idx]
            ell2_max = ells_edges_high[ell2_idx]

            # isolate the relevant ranges of ell values from the original ells_in grid
            ell1_in_ix = np.where((ell1_min <= ells_in) & (ells_in < ell1_max))[0]
            ell2_in_ix = np.where((ell2_min <= ells_in) & (ells_in < ell2_max))[0]
            ell1_in = ells_in[ell1_in_ix]
            ell2_in = ells_in[ell2_in_ix]

            # mask the covariance to the relevant block
            cov_masked = cov[np.ix_(ell1_in_ix, ell2_in_ix)]

            # this equals the number of ell values within a bin in the unweighted case,
            # and delta_ell in the unweighted, unbinned case
            weights1_in = weights_in[ell1_in_ix]
            weights2_in = weights_in[ell2_in_ix]

            weights1_in_xx, weights2_in_yy = np.meshgrid(
                weights1_in, weights2_in, indexing='ij'
            )
            if which_binning == 'sum':
                partial_sum = np.sum(
                    cov_masked * weights1_in_xx * weights2_in_yy, axis=1
                )
                total = np.sum(partial_sum, axis=0)
                norm1 = np.sum(weights1_in)
                norm2 = np.sum(weights2_in)

            elif which_binning == 'integral' and not interpolate:
                partial_integral = simps(
                    y=cov_masked * weights1_in_xx * weights2_in_yy, x=ell2_in, axis=1
                )
                total = simps(y=partial_integral, x=ell1_in, axis=0)
                norm1 = simps(y=weights1_in, x=ell1_in)
                norm2 = simps(y=weights2_in, x=ell2_in)

            elif which_binning == 'integral' and interpolate:
                # Interpolate the covariance matrix to a finer grid if necessary
                ell1_fine = np.linspace(ell1_min, ell1_max, num=100)
                ell2_fine = np.linspace(ell2_min, ell2_max, num=100)
                cov_interp = cov_interp_func(ell1_fine, ell2_fine)

                # Create fine grids for weights if necessary
                weights1_fine = np.interp(ell1_fine, ell1_in, weights1_in)
                weights2_fine = np.interp(ell2_fine, ell2_in, weights2_in)

                # Perform the double integral
                partial_integral = simps(
                    y=cov_interp * weights1_fine[:, None] * weights2_fine[None, :],
                    x=ell2_fine,
                    axis=1,
                )
                total = simps(y=partial_integral, x=ell1_fine, axis=0)
                norm1 = simps(y=weights1_fine, x=ell1_fine)
                norm2 = simps(y=weights2_fine, x=ell2_fine)

            binned_cov[ell1_idx, ell2_idx] = total / (norm1 * norm2)

    return binned_cov


def bin_2d_array_vectorized(
    cov: np.ndarray,
    ells_in: np.ndarray,
    ells_out: np.ndarray,
    ells_out_edges: np.ndarray,
    weights_in: np.ndarray | None,
    which_binning: str = 'sum',
    interpolate: bool = True,
):
    """Vectorized version of bin_2d_array with pre-computed masks.

    Optimizations:
    - Pre-computes all bin masks once (for 'sum' binning mode)
    - Uses direct array slicing for better cache locality
    - Skips empty bins
    - Removes redundant meshgrid operations
    """
    assert cov.shape[0] == cov.shape[1] == len(ells_in), (
        'ells_in must be the same length as the covariance matrix'
    )
    assert len(ells_out) == len(ells_out_edges) - 1, (
        'ells_out must be the same length as the number of edges - 1'
    )
    assert which_binning in ['sum', 'integral'], (
        'which_binning must be either "sum" or "integral"'
    )

    if weights_in is None:
        weights_in = np.ones_like(ells_in)

    assert len(weights_in) == len(ells_in), (
        'weights_in must be the same length as ells_in'
    )
    assert type(interpolate) is bool, 'interpolate must be a boolean'

    n_bins = len(ells_out)
    binned_cov = np.zeros((n_bins, n_bins))

    if which_binning == 'sum':
        # Pre-compute bin masks and norms for all bins
        bin_masks = []
        bin_weights = []
        bin_norms = []

        for i in range(n_bins):
            mask = (ells_in >= ells_out_edges[i]) & (ells_in < ells_out_edges[i + 1])
            weights = weights_in[mask]
            bin_masks.append(mask)
            bin_weights.append(weights)
            bin_norms.append(np.sum(weights))

        # Vectorized binning
        for i in range(n_bins):
            mask_i = bin_masks[i]
            w_i = bin_weights[i]
            norm_i = bin_norms[i]

            if norm_i == 0:
                continue

            # Extract rows corresponding to bin i
            cov_rows = cov[mask_i, :]

            for j in range(n_bins):
                mask_j = bin_masks[j]
                w_j = bin_weights[j]
                norm_j = bin_norms[j]

                if norm_j == 0:
                    continue

                # Extract the block
                cov_block = cov_rows[:, mask_j]

                # Weighted sum using outer product
                total = np.sum(cov_block * w_i[:, None] * w_j[None, :])
                binned_cov[i, j] = total / (norm_i * norm_j)

    elif which_binning == 'integral' and not interpolate:
        # Vectorized integral version without interpolation
        ells_edges_low = ells_out_edges[:-1]
        ells_edges_high = ells_out_edges[1:]

        for ell1_idx in range(n_bins):
            ell1_min = ells_edges_low[ell1_idx]
            ell1_max = ells_edges_high[ell1_idx]

            mask1 = (ells_in >= ell1_min) & (ells_in < ell1_max)
            ell1_in = ells_in[mask1]
            weights1_in = weights_in[mask1]

            if len(ell1_in) == 0:
                continue

            norm1 = simps(y=weights1_in, x=ell1_in)
            cov_rows = cov[mask1, :]

            for ell2_idx in range(n_bins):
                ell2_min = ells_edges_low[ell2_idx]
                ell2_max = ells_edges_high[ell2_idx]

                mask2 = (ells_in >= ell2_min) & (ells_in < ell2_max)
                ell2_in = ells_in[mask2]
                weights2_in = weights_in[mask2]

                if len(ell2_in) == 0:
                    continue

                cov_block = cov_rows[:, mask2]
                norm2 = simps(y=weights2_in, x=ell2_in)

                # Double integral
                weights2_in_yy = weights2_in[None, :]
                weights1_in_xx = weights1_in[:, None]
                partial_integral = simps(
                    y=cov_block * weights1_in_xx * weights2_in_yy, x=ell2_in, axis=1
                )
                total = simps(y=partial_integral, x=ell1_in, axis=0)

                binned_cov[ell1_idx, ell2_idx] = total / (norm1 * norm2)

    elif which_binning == 'integral' and interpolate:
        # Use original implementation with spline for interpolated integral
        cov_interp_func = RectBivariateSpline(ells_in, ells_in, cov)
        ells_edges_low = ells_out_edges[:-1]
        ells_edges_high = ells_out_edges[1:]

        for ell1_idx in range(n_bins):
            ell1_min = ells_edges_low[ell1_idx]
            ell1_max = ells_edges_high[ell1_idx]

            mask1 = (ells_in >= ell1_min) & (ells_in < ell1_max)
            ell1_in = ells_in[mask1]
            weights1_in = weights_in[mask1]

            if len(ell1_in) == 0:
                continue

            for ell2_idx in range(n_bins):
                ell2_min = ells_edges_low[ell2_idx]
                ell2_max = ells_edges_high[ell2_idx]

                mask2 = (ells_in >= ell2_min) & (ells_in < ell2_max)
                ell2_in = ells_in[mask2]
                weights2_in = weights_in[mask2]

                if len(ell2_in) == 0:
                    continue

                # Interpolate to fine grid
                ell1_fine = np.linspace(ell1_min, ell1_max, num=100)
                ell2_fine = np.linspace(ell2_min, ell2_max, num=100)
                cov_interp = cov_interp_func(ell1_fine, ell2_fine)

                # Interpolate weights
                weights1_fine = np.interp(ell1_fine, ell1_in, weights1_in)
                weights2_fine = np.interp(ell2_fine, ell2_in, weights2_in)

                # Double integral
                partial_integral = simps(
                    y=cov_interp * weights1_fine[:, None] * weights2_fine[None, :],
                    x=ell2_fine,
                    axis=1,
                )
                total = simps(y=partial_integral, x=ell1_fine, axis=0)
                norm1 = simps(y=weights1_fine, x=ell1_fine)
                norm2 = simps(y=weights2_fine, x=ell2_fine)

                binned_cov[ell1_idx, ell2_idx] = total / (norm1 * norm2)

    return binned_cov


def bin_1d_array(
    ells_in, ells_out, ells_out_edges, cls_in, weights, which_binning, ells_eff=None
):
    """Bin the input power spectrum into the output bins.

    :param ells_in: array of input ells
    :param ells_out: array of output ells
    :param cls_in: array of input power spectrum
    :param weights: array of weights for the input power spectrum
    :return: array of binned power spectrum
    """
    weights_was_none = False
    if weights is None:
        weights = np.ones_like(ells_in)
        weights_was_none = True
    if len(ells_in) != len(cls_in):
        raise ValueError('ells_in and cls_in must have the same length')
    if len(ells_in) != len(weights):
        raise ValueError('ells_in and weights must have the same length')
    if np.any(ells_out < ells_in[0]) or np.any(ells_out > ells_in[-1]):
        raise ValueError('ells_out must be within the range of ells_in')
    if np.any(ells_out[1:] < ells_out[:-1]):
        raise ValueError('ells_out must be monotonically increasing')

    assert len(cls_in) == len(ells_in), (
        'ells_in must be the same length as the covariance matrix'
    )
    assert len(ells_out) == len(ells_out_edges) - 1, (
        'ells_out must be the same length as the number of edges - 1'
    )

    binned_cls = np.zeros(len(ells_out))
    _spline = CubicSpline(ells_in, cls_in)

    ells_edges_low = ells_out_edges[:-1]
    ells_edges_high = ells_out_edges[1:]

    # Loop over the output bins
    for ell_idx in range(len(ells_out)):
        # Get ell min/max for the current bins
        ell_min = ells_edges_low[ell_idx]
        ell_max = ells_edges_high[ell_idx]

        # this mask returns a bool array True at the ells_in indices satisfying the condition
        ell_bool_mask = (ell_min <= ells_in) & (ells_in < ell_max)
        ell_masked_idxs = np.nonzero(ell_bool_mask)[0]

        # isolate the relevant ranges of ell values from the original ells_in grid, weights and cov
        ells_in_masked = ells_in[ell_masked_idxs]
        cls_masked = cls_in[ell_masked_idxs]

        if ells_eff is not None and weights.shape == (len(ells_eff), len(ells_in)):
            ells_eff_idx = np.argmin(np.abs(ells_eff - ells_out[ell_idx]))
            weights_masked = weights[ells_eff_idx, ell_masked_idxs]
        elif weights.shape == (len(ells_in),):
            weights_masked = weights[ell_masked_idxs]
        else:
            raise ValueError(
                'weights must have shape (len(ells_in),) or, if ells_eff is '
                'passed, (len(ells_eff), len(ells_in))'
            )

        # Calculate the bin widths
        if weights_was_none:
            delta_ell = ell_max - ell_min
            assert delta_ell == np.sum(weights_masked), (
                'The weights must sum to the bin width'
            )

        # Option 1: use the original grid for integration and no weights
        if which_binning == 'integral':
            integral = simps(y=cls_masked * weights_masked, x=ells_in_masked)
            norm = simps(y=weights_masked, x=ells_in_masked)
            binned_cls[ell_idx] = integral / norm

        elif which_binning == 'sum':
            binned_cls[ell_idx] = np.sum(cls_masked * weights_masked) / np.sum(
                weights_masked
            )

        else:
            raise ValueError('which_binning should be "sum" or "integral"')

        # # Option 2: create fine grids for integration over the ell ranges (GIVES GOOD RESULTS ONLY FOR nsteps=delta_ell!)
        # ell_fine = np.linspace(ell_min, ell_max, 50)
        # cls_interp = spline(ell_fine)

        # # Perform simps integration over the ell ranges
        # integral = simps(y=cls_interp * ell_fine, x=ell_fine)
        # binned_cls[ell_idx] = integral / (np.sum(ell_fine))

    return binned_cls


def j0(x):
    return jv(0, x)


def j1(x):
    return jv(1, x)


def j2(x):
    return jv(2, x)


def savetxt_aligned(filename, array_2d, header_list, col_width=25, decimals=8):
    header = ''
    for i in range(len(header_list)):
        offset = 2 if i == 0 else 0
        string = f'{header_list[i]:<{col_width - offset}}'
        header += string

    # header = ''.join(
    # [f'{header_list[i]:<{col_width - 2}}' for i in range(len(header_list))]
    # )
    fmt = [f'%-{col_width}.{decimals}f'] * len(array_2d[0])
    np.savetxt(filename, array_2d, header=header, fmt=fmt, delimiter='')


def compare_funcs(
    x,
    y: dict,
    logscale_y=(False, False),
    logscale_x=False,
    ylabel=None,
    title=None,
    ylim_diff=None,
    plt_kw=None,
    ax=None,
):
    plt_kw = {} if plt_kw is None else plt_kw

    names = list(y.keys())
    y_tuple = list(y.values())
    colors = plt.get_cmap('tab10').colors  # Get tab colors

    if x is None:
        x = np.arange(len(y_tuple[0]))

    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[2, 1])
        fig.subplots_adjust(hspace=0)
    else:
        fig = ax[0].figure

    for i, _y in enumerate(y_tuple):
        ls = '--' if i > 0 else '-'
        # alpha = 0.8 if i > 0 else 1
        ax[0].plot(x, _y, label=names[i], c=colors[i], ls=ls, **plt_kw)
    ax[0].legend()

    for i in range(1, len(y_tuple)):
        ax[1].plot(
            x, percent_diff(y_tuple[i], y_tuple[0]), c=colors[i], ls='-', **plt_kw
        )
    ax[1].set_ylabel('A/B - 1 [%]')
    ax[1].axhspan(-10, 10, alpha=0.2, color='gray')

    for i in range(2):
        if logscale_y[i]:
            ax[i].set_yscale('log')

    if logscale_x:
        for i in range(2):
            ax[i].set_xscale('log')

    if ylim_diff is not None:
        ax[1].set_ylim(ylim_diff)

    if title is not None:
        fig.suptitle(title)

    if ylabel is not None:
        ax[0].set_ylabel(ylabel)


def get_git_info():
    try:
        branch = (
            subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode('utf-8')
        )

        commit = (
            subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode('utf-8')
        )

        return branch, commit
    except subprocess.CalledProcessError:
        return None, None


def check_interpolate_input_tab(
    input_tab: np.ndarray, z_grid_out: np.ndarray, zbins: int, kind: str = 'CubicSpline'
) -> tuple:
    """Interpolates the input table over the 0th dimension and returns the
    interpolated values on the specified grid.

    Parameters
    ----------
    - input_tab (numpy.ndarray): The input table with shape (z_points, zbins + 1).
    - z_grid_out (numpy.ndarray): The output grid for interpolation.
    - zbins (int): The number of redshift bins.
    - kind (str): Interpolation method. Options: 'linear' (default, recommended for
      top-hat functions), 'cubic' (cubic spline), 'nearest' (preserves exact top-hat).

    Returns
    -------
    - output_tab (numpy.ndarray): The interpolated table with shape
    (len(z_grid_out), zbins).

    """
    assert input_tab.shape[1] == zbins + 1, (
        'The input table should have shape (z_points, zbins + 1)'
        f', but has shape {input_tab.shape} and zbins = {zbins}'
    )

    z_in = input_tab[:, 0]
    vals_in = input_tab[:, 1:]

    if kind == 'CubicSpline':
        interp_func = CubicSpline(x=z_in, y=vals_in, axis=0)
        output_tab = interp_func(z_grid_out)
    elif kind in ['linear', 'nearest']:
        interp_func = interp1d(
            z_in,
            vals_in,
            axis=0,
            kind=kind,
            bounds_error=False,
            fill_value='extrapolate',
        )
        output_tab = interp_func(z_grid_out)
    else:
        raise ValueError(
            f"Unknown interpolation kind: {kind}. Use 'linear', 'CubicSpline', "
            f"or 'nearest'"
        )

    return output_tab, interp_func


def interp_2d_arr(x_in, y_in, z2d_in, x_out, y_out, output_masks):
    """Interpolate a 2D array onto a new grid using bicubic spline
    interpolation.

    Parameters
    ----------
    - x_in (numpy.ndarray): The x-coordinates of the input 2D array.
    - y_in (numpy.ndarray): The y-coordinates of the input 2D array.
    - z2d_in (numpy.ndarray): The 2D input array to be interpolated.
    - x_out (numpy.ndarray): The x-coordinates of the output grid.
    - y_out (numpy.ndarray): The y-coordinates of the output grid.
    - output_masks (bool): A boolean flag indicating whether to mask the output array.

    Returns
    -------
    - x_out_masked (numpy.ndarray): The x-coordinates of the output grid,
    clipped to avoid interpolation errors.
    - y_out_masked (numpy.ndarray): The y-coordinates of the output grid,
    clipped to avoid interpolation errors.
    - z2d_interp (numpy.ndarray): The interpolated 2D array.
    - x_mask (numpy.ndarray): A boolean mask indicating which elements of the
    original x_out array were used.
    - y_mask (numpy.ndarray): A boolean mask indicating which elements of the
    original y_out array were used.

    """
    z2d_func = RectBivariateSpline(x=x_in, y=y_in, z=z2d_in)

    # clip x and y grids to avoid interpolation errors
    x_mask = np.logical_and(x_in.min() <= x_out, x_out < x_in.max())
    y_mask = np.logical_and(y_in.min() <= y_out, y_out < y_in.max())
    x_out_masked = x_out[x_mask]
    y_out_masked = y_out[y_mask]

    if len(x_out_masked) < len(x_out):
        print(
            f'x array trimmed: old range [{x_out.min():.2e}, {x_out.max():.2e}], '
            f'new range [{x_out_masked.min():.2e}, {x_out_masked.max():.2e}]'
        )
    if len(y_out_masked) < len(y_out):
        print(
            f'y array trimmed: old range [{y_out.min():.2e}, {y_out.max():.2e}], '
            f'new range [{y_out_masked.min():.2e}, {y_out_masked.max():.2e}]'
        )

    # with RegularGridInterpolator:
    # TODO untested
    # z2d_func = RegularGridInterpolator((x_in, y_in), z2d_in, method='linear')
    # xx, yy = np.meshgrid(x_out_masked, y_out_masked)
    # z2d_interp = z2d_interp((xx, yy)).T

    z2d_interp = z2d_func(x_out_masked, y_out_masked)

    if output_masks:
        return x_out_masked, y_out_masked, z2d_interp, x_mask, y_mask
    else:
        return x_out_masked, y_out_masked, z2d_interp


def regularize_covariance(cov_matrix, lambda_reg=1e-5):
    """Regularizes the covariance matrix by adding lambda * I.

    Parameters
    ----------
    - cov_matrix: Original covariance matrix (numpy.ndarray)
    - lambda_reg: Regularization parameter

    Returns
    -------
    - Regularized covariance matrix

    """
    n = cov_matrix.shape[0]
    identity_matrix = np.eye(n)
    cov_matrix_reg = cov_matrix + lambda_reg * identity_matrix
    return cov_matrix_reg


def get_simpson_weights(n):
    """Function written by Marco Bonici."""
    number_intervals = (n - 1) // 2
    weight_array = np.zeros(n)
    if n == number_intervals * 2 + 1:
        for i in range(number_intervals):
            weight_array[2 * i] += 1 / 3
            weight_array[2 * i + 1] += 4 / 3
            weight_array[2 * i + 2] += 1 / 3
    else:
        weight_array[0] += 0.5
        weight_array[1] += 0.5
        for i in range(number_intervals):
            weight_array[2 * i + 1] += 1 / 3
            weight_array[2 * i + 2] += 4 / 3
            weight_array[2 * i + 3] += 1 / 3
        weight_array[-1] += 0.5
        weight_array[-2] += 0.5
        for i in range(number_intervals):
            weight_array[2 * i] += 1 / 3
            weight_array[2 * i + 1] += 4 / 3
            weight_array[2 * i + 2] += 1 / 3
        weight_array /= 2
    return weight_array


def zpair_from_zidx(zidx, ind):
    """Return the zpair corresponding to the zidx for a given ind array.

    To be thoroughly tested, but quite straightforward
    """
    assert ind.shape[1] == 2, (
        'ind array must have shape (n, 2), maybe you are passing the full '
        'ind file instead of ind_auto/ind_cross'
    )
    return np.where((ind == [zidx, zidx]).all(axis=1))[0][0]


def write_cl_ascii(ascii_folder, ascii_filename, cl_3d, ells, zbins):
    with open(f'{ascii_folder}/{ascii_filename}.ascii', 'w') as file:
        # Write header
        file.write(f'#ell\ttomo_i\ttomo_j\t{ascii_filename}\n')

        # Iterate over the array and write the data
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    # Format the line with appropriate spacing
                    file.write(f'{ell_val:.3f}\t{zi + 1}\t{zj + 1}\t{value:.10e}\n')


def write_cl_tab(folder, filename, cl_3d, ells, zbins):
    """Write the Cls in the SB txt format."""
    with open(f'{folder}/{filename}.txt', 'w') as file:
        file.write(f'#ell\t\tzi\tzj\t{filename}\n')
        for ell_idx, ell_val in enumerate(ells):
            for zi in range(zbins):
                for zj in range(zbins):
                    value = cl_3d[ell_idx, zi, zj]
                    file.write(f'{ell_val:.3f}\t\t{zi}\t{zj}\t{value:.10e}\n')


def read_cl_tab(
    folder: str, filename: str, ells: np.ndarray, zbins: int, rtol: float = 0.01
) -> np.ndarray:
    """Reads the Cls in the SB txt format."""
    cl_tab = np.genfromtxt(f'{folder}/{filename}')
    cl_3d = np.zeros((len(ells), zbins, zbins))

    for row in cl_tab:
        ell_ix = np.argmin(np.abs(ells - row[0]))
        if abs(ells[ell_ix] / row[0] - 1) > rtol:  # compare % difference
            raise ValueError(
                f'No matching ell found for {row[0]}: closest is {ells[ell_ix]}'
            )

        zi = int(row[1])
        zj = int(row[2])
        cl = row[3]
        cl_3d[ell_ix, zi, zj] = cl

    return cl_3d


def block_diag(array_3d):
    """Useful for visualizing nbl, zbins, zbins arrays at a glance."""
    nbl = array_3d.shape[0]
    return scipy.linalg.block_diag(*[array_3d[ell, :, :] for ell in range(nbl)])


def contour_FoM_calculator(sample, param1, param2, sigma_level=1):
    """This function has been written by Santiago Casas.

    Computes  the FoM from getDist samples. add()sample is a getDist
    sample object, you need as well the shapely package to compute
    polygons. The function returns the 1sigma FoM, but in principle you
    could compute 2-, or 3-sigma "FoMs"
    """
    from shapely.geometry import Polygon

    contour_coords = {}
    density = sample.get2DDensityGridData(j=param1, j2=param2, num_plot_contours=3)
    contour_levels = density.contours
    contours = plt.contour(density.x, density.y, density.P, sorted(contour_levels))
    for ii, contour in enumerate(contours.collections):
        paths = contour.get_paths()
        for path in paths:
            xy = path.vertices
            x = xy[:, 0]
            y = xy[:, 1]
            contour_coords[ii] = list(zip(x, y, strict=False))
    sigma_lvls = {3: 0, 2: 1, 1: 2}
    # 0: 3sigma, 1: 2sigma, 2: 1sigma
    poly = Polygon(contour_coords[sigma_lvls[sigma_level]])
    area = poly.area
    FoM_area = (2.3 * np.pi) / area
    return FoM_area, density


def figure_of_correlation(correl_matrix):
    """Compute the Figure of Correlation (FoC) from the correlation matrix
    correl_matrix.

    Parameters
    ----------
    - correl_matrix (2D numpy array): The correlation matrix.

    Returns
    -------
    - FoC (float): The Figure of Correlation.

    """
    # Invert the correlation matrix
    correl_matrix_inv = np.linalg.inv(correl_matrix)
    # Compute the FoC
    foc = np.sqrt(np.linalg.det(correl_matrix_inv))

    return foc


def find_inverse_from_array(input_x, input_y, desired_y, interpolation_kind='linear'):
    from pynverse import inversefunc

    input_y_func = interp1d(input_x, input_y, kind=interpolation_kind)
    desired_y = inversefunc(
        input_y_func, y_values=desired_y, domain=(input_x[0], input_x[-1])
    )
    return desired_y


def add_ls_legend(ls_dict):
    """Add a legend for line styles.

    Parameters
    ----------
    ls_dict : dict
        A dictionary mapping line styles to labels.
        E.g. {'-': 'delta', '--': 'gamma'}

    """
    handles = []
    for ls, label in ls_dict.items():
        handles.append(mlines.Line2D([], [], color='black', linestyle=ls, label=label))
    plt.legend(handles=handles, loc='best')


def find_nearest_idx(array, value):
    idx = np.abs(array - value).argmin()
    return idx


def flatten_dict(nested_dict):
    """Flatten a nested dictionary."""
    flattened = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def percent_diff(array_1, array_2, abs_value=False):
    array_1 = np.atleast_1d(array_1)  # Ensure array-like behavior
    array_2 = np.atleast_1d(array_2)

    diff = (array_1 / array_2 - 1) * 100

    # avoid nans
    both_zeros = np.logical_and(array_1 == 0, array_2 == 0)

    diff[both_zeros] = 0

    if abs_value:
        return np.abs(diff)
    else:
        # Convert back to scalar if necessary
        return diff.item() if diff.size == 1 else diff


def percent_diff_mean(array_1, array_2):
    """Result is in "percent" units."""
    mean = (array_1 + array_2) / 2.0
    diff = (array_1 / mean - 1) * 100
    return diff


def percent_diff_nan(array_1, array_2, eraseNaN=True, log=False, abs_val=False):
    """Calculate the percent difference between two arrays, handling NaN
    values.
    """
    # Handle NaN values
    if eraseNaN:
        # Mask where NaN values are present
        diff = np.ma.masked_where(
            np.isnan(array_1) | np.isnan(array_2), percent_diff(array_1, array_2)
        )
    else:
        diff = percent_diff(array_1, array_2)

    # Handle log transformation
    if log:
        # Mask zero differences before taking the log
        diff = np.ma.masked_where(diff == 0, diff)
        diff = np.log10(np.ma.abs(diff))  # Masked values will be ignored in the log

    # Handle absolute values
    if abs_val:
        diff = np.ma.abs(diff)

    return diff


def diff_threshold_check(diff, threshold):
    boolean = np.any(np.abs(diff) > threshold)
    print(f'has any element of the arrays a disagreement > {threshold}%? ', boolean)


def compute_smape(vec_true, vec_test, cov_mat=None):
    """Computes the SMAPE (Symmetric Mean Absolute Percentage Error) for a
    given 1D array with weighted elements.

    Args:
        vec_true (np.array): array of true values
        vec_test (np.array): array of predicted/approximated values
        cov_mat (np.array): covariance matrix for vec_true

    Returns:
        float: SMAPE value

    """
    if isinstance(vec_true, np.ndarray) and isinstance(vec_test, np.ndarray):
        assert len(vec_true) == len(vec_test), 'Arrays must have the same length'
        assert vec_true.ndim == 1 and vec_test.ndim == 1, 'arrays must be 1D'

    if cov_mat is not None:
        assert cov_mat.shape[0] == cov_mat.shape[1] == len(vec_true), (
            'cov_mat must be a square matrix with the same length as the input vectors'
        )
        weights = vec_true / np.sqrt(np.diag(cov_mat))
    else:
        weights = np.ones_like(vec_true)  # uniform weights

    numerator = weights * np.abs(vec_true - vec_test)
    denominator = np.abs(vec_true) + np.abs(vec_test)

    return 100 * np.mean(numerator / denominator)  # the output is already a precentage


def smape(y_true, y_pred):
    """Compute the point-by-point Symmetric Mean Absolute Percentage Error (SMAPE)
    between two arrays of the same shape.

    Parameters
    ----------
    y_true : array_like
        Reference or true values.
    y_pred : array_like
        Predicted or test values.

    Returns
    -------
    smape : ndarray
        Array of SMAPE values in percentage (same shape as inputs).

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = 0.5 * (np.abs(y_true) + np.abs(y_pred))

    with np.errstate(divide='ignore', invalid='ignore'):
        smape = 100 * numerator / denominator
        smape[denominator == 0] = 0.0  # Define SMAPE as 0 if denominator is 0

    return smape


def compare_arrays(
    A,
    B,
    name_A='A',
    name_B='B',
    plot_diff=True,
    plot_array=True,
    log_array=True,
    log_diff=False,
    abs_val=False,
    plot_diff_threshold=None,
    white_where_zero=True,
    plot_diff_hist=False,
    matshow_arr_kw=None,
    early_return=True,
    title='',
):
    fontsize = 25

    if matshow_arr_kw is None:
        matshow_arr_kw = {}

    rtols = [0, 1e-3, 1e-2, 5e-2]
    for rtol in rtols:
        if np.allclose(A, B, rtol=rtol, atol=0):
            print(
                f'{name_A} and {name_B} are close within '
                f'relative tolerance of {rtol * 100}% ✅'
            )
            if early_return:
                return
            break
    else:
        # runs only if the loop never 'break'-s (i.e., not close at any rtol)
        diff_AB = percent_diff_nan(A, B, eraseNaN=True, abs_val=abs_val)
        higher_rtol = rtols[-1] * 100  # in percent
        max_diff = np.max(diff_AB)
        no_outliers = np.sum(diff_AB > higher_rtol)
        additional_info = (
            f'\nMax discrepancy: {max_diff:.2f}%;'
            f'\nNumber of elements with discrepancy > {higher_rtol}%: {no_outliers}'
            f'\nFraction of elements with discrepancy > {higher_rtol}%: '
            f'{no_outliers / diff_AB.size:.5f}'
        )
        print(
            f'{name_A} and {name_B} differ by more than {higher_rtol}% ❌:'
            f'{additional_info}'
        )

    # Check that arrays are 2D if any plotting is requested.
    if plot_diff or plot_array:
        assert A.ndim == 2 and B.ndim == 2, 'Plotting is only implemented for 2D arrays'

    # Determine number of rows:
    nrows = (1 if plot_array else 0) + (1 if plot_diff else 0)
    ncols = 2  # Always show 2 panels per row

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(17, 7 * nrows), constrained_layout=True
    )

    # Ensure ax is always 2D
    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)  # Convert row array to 2D
    if ncols == 1:
        ax = np.expand_dims(ax, axis=1)  # Convert column array to 2D

    # If plotting arrays, prepare data and plot in first row.
    if plot_array:
        A_toplot, B_toplot = A.copy(), B.copy()
        if abs_val:
            A_toplot, B_toplot = np.abs(A_toplot), np.abs(B_toplot)
        if log_array:
            A_toplot, B_toplot = np.log10(A_toplot), np.log10(B_toplot)

        im = ax[0, 0].matshow(A_toplot, **matshow_arr_kw)
        ax[0, 0].set_title(f'{name_A}', fontsize=fontsize)
        fig.colorbar(im, ax=ax[0, 0])

        im = ax[0, 1].matshow(B_toplot, **matshow_arr_kw)
        ax[0, 1].set_title(f'{name_B}', fontsize=fontsize)
        fig.colorbar(im, ax=ax[0, 1])

    # If plotting differences, prepare diff data and plot in next row.
    if plot_diff:
        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=False, abs_val=abs_val)
        diff_BA = percent_diff_nan(B, A, eraseNaN=True, log=False, abs_val=abs_val)

        if plot_diff_threshold is not None:
            # Mask out small differences (set them to white via the colormap's
            # "bad" color)
            diff_AB = np.ma.masked_where(np.abs(diff_AB) < plot_diff_threshold, diff_AB)
            diff_BA = np.ma.masked_where(np.abs(diff_BA) < plot_diff_threshold, diff_BA)

        if log_diff:
            # Replace nonpositive with nan to avoid -inf
            diff_AB = np.log10(np.abs(diff_AB))
            diff_BA = np.log10(np.abs(diff_BA))

        im = ax[1, 0].matshow(diff_AB)
        ax[1, 0].set_title('(A/B - 1) * 100', fontsize=fontsize)
        fig.colorbar(im, ax=ax[1, 0])

        im = ax[1, 1].matshow(diff_BA)
        ax[1, 1].set_title('(B/A - 1) * 100', fontsize=fontsize)
        fig.colorbar(im, ax=ax[1, 1])

    fig.suptitle(
        f'log_array={log_array}, abs_val={abs_val}, log_diff={log_diff}\n'
        f'plot_diff_threshold={plot_diff_threshold}%\n{title}',
        fontsize=fontsize,
    )
    plt.show()

    if plot_diff_hist:
        diff_AB = percent_diff_nan(A, B, eraseNaN=True, log=False, abs_val=False)
        plt.figure()
        plt.hist(diff_AB.flatten(), bins=30, log=True, density=True)
        plt.xlabel('% difference')
        plt.ylabel('frequency')
        plt.show()


def matshow(
    array,
    title='title',
    log=True,
    abs_val=False,
    threshold=None,
    only_show_nans=False,
    matshow_kwargs: dict | None = None,
):
    """:param array:
    :param title:
    :param log:
    :param abs_val:
    :param threshold: if None, do not mask the values; otherwise,
    keep only the elements above the threshold (i.e., mask the ones below the threshold)
    :return:
    """
    if matshow_kwargs is None:
        matshow_kwargs = {}
    if only_show_nans:
        warnings.warn(
            'only_show_nans is True, better switch off log and abs_val for the moment',
            stacklevel=2,
        )
        # Set non-NaN elements to 0 and NaN elements to 1
        array = np.where(np.isnan(array), 1, 0)
        title += ' (only NaNs shown)'

    # the ordering of these is important: I want the log(abs), not abs(log)
    if abs_val:  # take the absolute value
        array = np.abs(array)
        title = 'abs ' + title
    if log:  # take the log
        with np.errstate(divide='ignore', invalid='ignore'):
            array = np.log10(array)
        title = 'log10 ' + title

    if threshold is not None:
        array = np.ma.masked_where(array < threshold, array)
        title += f' \n(masked below {threshold} %)'

    plt.matshow(array, **matshow_kwargs)
    plt.colorbar()
    plt.title(title)
    plt.show()


def generate_ind(triu_tril_square, row_col_major, size):
    """Generates a list of indices for the upper triangular part of a matrix
    :param triu_tril_square: str.

    if 'triu', returns the indices for the upper triangular part of the
    matrix. If 'tril', returns the indices for the lower triangular part
    of the matrix If 'full_square', returns the indices for the whole
    matrix
    :param row_col_major: str. if True, the indices are returned in row-
        major order; otherwise, in column-major order
    :param size: int. size of the matrix to take the indices of
    :return: list of indices
    """
    assert row_col_major in ['row-major', 'col-major'], (
        'row_col_major must be either "row-major" or "col-major"'
    )
    assert triu_tril_square in ['triu', 'tril', 'full_square'], (
        'triu_tril_square must be either "triu", "tril" or "full_square"'
    )

    if triu_tril_square == 'triu':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i, size)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i + 1)]
    elif triu_tril_square == 'tril':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i + 1)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i, size)]
    elif triu_tril_square == 'full_square':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(size)]
        elif row_col_major == 'col-major':
            ind = [(j, i) for i in range(size) for j in range(size)]

    return np.asarray(ind)


def build_full_ind(triu_tril, row_col_major, size):
    """Builds the good old ind file."""
    assert triu_tril in ['triu', 'tril'], 'triu_tril must be either "triu" or "tril"'
    assert row_col_major in ['row-major', 'col-major'], (
        'row_col_major must be either "row-major" or "col-major"'
    )

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    ll_columns = np.zeros((zpairs_auto, 2))
    gl_columns = np.hstack((np.ones((zpairs_cross, 1)), np.zeros((zpairs_cross, 1))))
    gg_columns = np.ones((zpairs_auto, 2))

    ll_columns = np.hstack(
        (ll_columns, generate_ind(triu_tril, row_col_major, size))
    ).astype(int)
    gl_columns = np.hstack(
        (gl_columns, generate_ind('full_square', row_col_major, size))
    ).astype(int)
    gg_columns = np.hstack(
        (gg_columns, generate_ind(triu_tril, row_col_major, size))
    ).astype(int)

    ind = np.vstack((ll_columns, gl_columns, gg_columns))

    assert ind.shape[0] == zpairs_3x2pt, 'ind has the wrong number of rows'

    return ind


def symmetrize_2d_array(array_2d):
    """Mirror the lower/upper triangle."""
    # if already symmetric, do nothing
    if check_symmetric(array_2d, exact=True):
        return array_2d

    # there is an implicit "else" here, since the function returns array_2d if
    # the array is symmetric
    assert array_2d.ndim == 2, 'array must be square'
    size = array_2d.shape[0]

    # check that either the upper or lower triangle (not including the diagonal) is null
    triu_elements = array_2d[np.triu_indices(size, k=+1)]
    tril_elements = array_2d[np.tril_indices(size, k=-1)]
    assert np.all(triu_elements == 0) or np.all(tril_elements == 0), (
        'neither the upper nor the lower triangle (excluding the diagonal) are null'
    )

    # if np.any(np.diag(array_2d)) != 0:
    # warnings.warn('the diagonal elements are all null', stacklevel=2)

    # symmetrize
    array_2d = np.where(array_2d, array_2d, array_2d.T)
    # check
    if not check_symmetric(array_2d, exact=False):
        warnings.warn('check failed: the array is not symmetric', stacklevel=2)

    return array_2d


def compute_FoM(FM, w0wa_idxs):
    cov_param = np.linalg.inv(FM)
    # cov_param_reduced = cov_param[start:stop, start:stop]
    cov_param_reduced = cov_param[np.ix_(w0wa_idxs, w0wa_idxs)]

    FM_reduced = np.linalg.inv(cov_param_reduced)
    FoM = np.sqrt(np.linalg.det(FM_reduced))
    return FoM


def get_zpairs(zbins):
    zpairs_auto = int((zbins * (zbins + 1)) / 2)
    zpairs_cross = zbins**2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


def _expand_diagonal_to_full(cov_diag):
    """Expand diagonal covariance to full (ell1, ell2) matrix.

    Parameters
    ----------
    cov_diag : np.ndarray
        Diagonal covariance with shape (A, B, C, D, nbl, i, j, k, l)

    Returns
    -------
    cov_full : np.ndarray
        Full covariance with shape (A, B, C, D, nbl, nbl, i, j, k, l)
        where only the diagonal (ell, ell) entries are non-zero
    """
    nbl = cov_diag.shape[4]
    cov_shape = (*cov_diag.shape[:4], nbl, nbl, *cov_diag.shape[5:])
    cov_full = np.zeros(cov_shape, dtype=cov_diag.dtype)

    for ell in range(nbl):
        cov_full[:, :, :, :, ell, ell] = cov_diag[:, :, :, :, ell]

    return cov_full


def _bin_cov_hs_g_diag(cov_ell_modes, ell_edges, ell_values):
    """Bin covariance computed at integer ell modes into ell bins.

    Parameters
    ----------
    cov_ell_modes : np.ndarray
        Covariance computed at integer ell modes, shape (A, B, C, D, n_ell_modes, i, j, k, l)
    ell_edges : np.ndarray
        Bin edges, shape (nbl + 1,)
    ell_values : np.ndarray
        Integer ell values, shape (n_ell_modes,)

    Returns
    -------
    cov_binned : np.ndarray
        Binned covariance, shape (A, B, C, D, nbl, i, j, k, l)

    Notes
    -----
    This function sums the covariance over all ell modes in each bin and divides
    by the bin width delta_ell = ell_upper - ell_lower. This matches OneCovariance's
    approach where the prefactor is 1/((2*ell+1)*fsky*delta_ell).
    """
    nbl = len(ell_edges) - 1

    # Initialize output array (the shape is the same as the input, but with fewer
    # ell bins)
    output_shape = (*cov_ell_modes.shape[:4], nbl, *cov_ell_modes.shape[5:])
    cov_binned = np.zeros(output_shape)

    # Sum over ell modes in each bin and divide by n_modes**2 (i.e., average over the
    # given ell bin)
    for bin_idx in range(nbl):
        ell_lower = int(ell_edges[bin_idx])
        ell_upper = int(ell_edges[bin_idx + 1])

        # Find which ell modes fall in this bin
        mask = (ell_values >= ell_lower) & (ell_values < ell_upper)
        n_modes = np.sum(mask)

        if n_modes == 0:
            raise ValueError(
                f'No ell modes found in bin {bin_idx} [{ell_lower}, {ell_upper}).'
                'The ell binning might be too fine for two integer values to fall '
                'within certain bins.'
            )

        # Sum over modes and divide by num_modes**2
        cov_binned[:, :, :, :, bin_idx, ...] = (
            np.sum(cov_ell_modes[:, :, :, :, mask, ...], axis=4) / n_modes**2
        )

    return cov_binned


@partial(jit, static_argnames=['mix'])
def cov_g_terms_helper_jax(a, b, prefactor, mix: bool):
    """Helper function to compute covariance terms (JAX version).
    Note: this function always returns only the ell-diagonal covariance.

    Parameters
    ----------
    a, b : jnp.ndarray
        Input arrays with shape (n_probes, n_probes, n_ell, zbins, zbins)
    prefactor : jnp.ndarray
        1D array of prefactors for each ell mode
    mix : bool
        If True, compute mixed terms (for cross-covariance of signal and noise)

    Returns
    -------
    cov : jnp.ndarray
        Diagonal covariance array with shape (A, B, C, D, nbl, zi, zj, zk, zl)
    """
    if mix:
        term_1 = jnp.einsum('ACLik, BDLjl -> ABCDLijkl', a, b)
        term_2 = jnp.einsum('ACLik, BDLjl -> ABCDLijkl', b, a)
        term_3 = jnp.einsum('ADLil, BCLjk -> ABCDLijkl', a, b)
        term_4 = jnp.einsum('ADLil, BCLjk -> ABCDLijkl', b, a)
    else:
        term_1 = jnp.einsum('ACLik, BDLjl -> ABCDLijkl', a, b)
        term_2 = jnp.einsum('ADLil, BCLjk -> ABCDLijkl', a, b)
        term_3 = 0
        term_4 = 0

    cov_diag = jnp.einsum(
        'ABCDLijkl, L -> ABCDLijkl', term_1 + term_2 + term_3 + term_4, prefactor
    )

    return cov_diag


def compute_g_cov(
    cl_5d: np.ndarray,
    nl_5d: np.ndarray,
    fsky: float,
    ell_values: np.ndarray,
    delta_ell: np.ndarray,
    split_terms: bool,
    return_only_ell_diagonal: bool = False,
    cov_hs_g_ell_bin_average: bool = True,
    ell_edges: np.ndarray = None,
):
    """Computes the Gaussian (1/fsky) covariance term, splitting into SVA, SN and MIX
    terms if required.

    Parameters
    ----------
    cl_5d : np.ndarray
        Power spectra with shape (n_probes, n_probes, nbl, zbins, zbins)
    nl_5d : np.ndarray
        Noise power spectra with shape (n_probes, n_probes, nbl, zbins, zbins)
    fsky : float
        Sky fraction
    ell_values : np.ndarray
        Ell values. If cov_hs_g_ell_bin_average=True, these are all integer ell values.
        If cov_hs_g_ell_bin_average=False, these are bin centers.
    delta_ell : np.ndarray
        Bin widths (only used if cov_hs_g_ell_bin_average=False)
    split_terms : bool
        If True, return (cov_sva, cov_sn, cov_mix). If False, return total cov.
    return_only_ell_diagonal : bool
        If True, return diagonal covariance shape (A,B,C,D,L,i,j,k,l).
        If False, return full covariance shape (A,B,C,D,L,M,i,j,k,l).
    cov_hs_g_ell_bin_average : bool
        If True, sum over integer ell modes and divide by number of modes**2
        If False, use traditional binned approach: Cov = prefactor / delta_ell
    ell_edges : np.ndarray, optional
        Bin edges (required if cov_hs_g_ell_bin_average=False). Shape (nbl+1,)

    Returns
    -------
    cov or (cov_sva, cov_sn, cov_mix) : np.ndarray or tuple
        Covariance array(s)
    """

    # sanity checks
    assert cl_5d.shape[0] in [1, 2], 'This function only works with 1 or 2 probes'
    assert cl_5d.shape[0] == cl_5d.shape[1], (
        'cl_5d must have shape (n_probes, n_probes, nbl, zbins, zbins)'
    )
    assert cl_5d.shape[-1] == cl_5d.shape[-2], (
        'cl_5d must have shape (n_probes, n_probes, nbl, zbins, zbins)'
    )
    assert nl_5d.shape == cl_5d.shape, (
        'nl_5d must have the same shape as cl_5d, '
        f'found {nl_5d.shape=}, and {cl_5d.shape=}'
    )

    if cov_hs_g_ell_bin_average and ell_edges is None:
        raise ValueError(
            'ell_edges must be provided when cov_hs_g_ell_bin_average=True'
        )

    # convenience variables
    clplusn_5d = cl_5d + nl_5d
    prefactor = 1 / ((2 * ell_values + 1) * fsky)

    if not cov_hs_g_ell_bin_average:
        prefactor /= delta_ell

    if not split_terms:
        cov = np.asarray(
            cov_g_terms_helper_jax(clplusn_5d, clplusn_5d, prefactor, mix=False)
        )

        # bin the integer ell modes
        if cov_hs_g_ell_bin_average:
            cov = _bin_cov_hs_g_diag(cov, ell_edges, ell_values)

        # if the user wants full (ell1, ell2) matrix, expand diagonal to full
        if not return_only_ell_diagonal:
            cov = _expand_diagonal_to_full(cov)

        return cov

    cov_sva = np.asarray(cov_g_terms_helper_jax(cl_5d, cl_5d, prefactor, mix=False))
    cov_sn = np.asarray(
        cov_g_terms_helper_jax(nl_5d, nl_5d, prefactor, mix=False)
    )
    cov_mix = np.asarray(cov_g_terms_helper_jax(cl_5d, nl_5d, prefactor, mix=True))

    # bin the integer ell modes
    if cov_hs_g_ell_bin_average:
        cov_sva = _bin_cov_hs_g_diag(cov_sva, ell_edges, ell_values)
        cov_sn = _bin_cov_hs_g_diag(cov_sn, ell_edges, ell_values)
        cov_mix = _bin_cov_hs_g_diag(cov_mix, ell_edges, ell_values)

    # if the user wants full (ell1, ell2) matrix, expand diagonal to full
    if not return_only_ell_diagonal:
        cov_sva = _expand_diagonal_to_full(cov_sva)
        cov_sn = _expand_diagonal_to_full(cov_sn)
        cov_mix = _expand_diagonal_to_full(cov_mix)

    return cov_sva, cov_sn, cov_mix


def cov_dict_6d_probe_blocks_to_4d_and_2d(
    cov_dict: dict,
    obs_space: str,
    nbx: int,
    ind_auto: np.ndarray,
    ind_cross: np.ndarray,
    zpairs_auto: int,
    zpairs_cross: int,
    block_index: str,
):
    """
    Takes the cov dictionary, validates its structure, and for each term reshapes each
    [ab, cd] probe block separately to 4d and 2d.

    Note: This is the updated version of cov_3x2pt_10D_to_4D.
    """

    if obs_space == 'harmonic':
        prefix = 'HS'
    elif obs_space == 'real':
        prefix = 'RS'
    elif obs_space == 'cosebis':
        prefix = 'CS'
    else:
        raise ValueError('`space` must be in ["harmonic", "real", "cosebis"]')

    auto_probes = const.__getattribute__(f'{prefix}_AUTO_PROBES')

    # reshape the probe-specific 6d arrays to 4d and 2d
    for term in cov_dict:  # noqa: PLC0206
        for probe_2tpl in cov_dict[term]:
            # skip the 3x2pt key, this function only reshapes probe blocks
            if probe_2tpl == '3x2pt':
                continue

            probe_ab, probe_cd = probe_2tpl

            # additional check: the input dictionary should only contain the '6d' dim
            for dim in ['4d', '2d']:
                assert cov_dict[term][probe_ab, probe_cd][dim] is None, (
                    f'In term {term}, probe combination {(probe_ab, probe_cd)}, '
                    f'dimension {dim} is already set to a non-None value. '
                    f'Please provide only the 6d array in the input dictionary.'
                )

            # extract array
            cov_6d = cov_dict[term][probe_2tpl]['6d']

            # reshape to 4d, then to 2d
            if cov_6d is None:
                cov_dict[term][probe_2tpl]['4d'] = None
                cov_dict[term][probe_2tpl]['2d'] = None
            else:
                # prepare ind and zpairs
                ind_ab = ind_auto if probe_ab in auto_probes else ind_cross
                ind_cd = ind_auto if probe_cd in auto_probes else ind_cross
                zpairs_ab = zpairs_auto if probe_ab in auto_probes else zpairs_cross
                zpairs_cd = zpairs_auto if probe_cd in auto_probes else zpairs_cross

                # reshape
                cov_dict[term][probe_ab, probe_cd]['4d'] = cov_6D_to_4D_blocks(
                    cov_6D=cov_6d,
                    nbl=nbx,
                    npairs_AB=zpairs_ab,
                    npairs_CD=zpairs_cd,
                    ind_AB=ind_ab,
                    ind_CD=ind_cd,
                )

                cov_dict[term][probe_ab, probe_cd]['2d'] = cov_4D_to_2D(
                    cov_4D=cov_dict[term][probe_ab, probe_cd]['4d'],
                    block_index=block_index,
                    optimize=True,
                )

    return cov_dict


def cov_6D_to_4D_blocks(cov_6D, nbl, npairs_AB, npairs_CD, ind_AB, ind_CD):
    """Reshapes the covariance even for the non-diagonal (hence, non-square)
    blocks needed to build the 3x2pt.

    Use npairs_AB = npairs_CD and ind_AB = ind_CD for the normal routine
    (valid for auto-covariance LL-LL, GG-GG, GL-GL and LG-LG). n_columns
    is used to determine whether the ind array has 2 or 4 columns (if
    it's given in the form of a dictionary or not)
    """
    assert ind_AB.shape[0] == npairs_AB, 'ind_AB.shape[0] != npairs_AB'
    assert ind_CD.shape[0] == npairs_CD, 'ind_CD.shape[0] != npairs_CD'
    assert (cov_6D.shape[0] == cov_6D.shape[1] == nbl) or (cov_6D.shape[0] == nbl), (
        'number of angular bins does not match first two cov axes or the first axis'
    )

    # this is to ensure compatibility with both 4-columns and 2-columns ind arrays
    # (dictionary)the penultimante element is the first index, the last one the
    # second index (see s - 1, s - 2 below)
    # number of columns: this is to understand the format of the file
    n_columns_AB = ind_AB.shape[1]
    n_columns_CD = ind_CD.shape[1]

    # check
    assert n_columns_AB == n_columns_CD, (
        'ind_AB and ind_CD must have the same number of columns'
    )
    ncol = n_columns_AB  # make the name shorter

    if cov_6D.shape[0] == cov_6D.shape[1] == nbl:
        cov_out = np.zeros((nbl, nbl, npairs_AB, npairs_CD))
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for ij in range(npairs_AB):
                    for kl in range(npairs_CD):
                        zi, zj, zk, zl = (
                            ind_AB[ij, ncol - 2],
                            ind_AB[ij, ncol - 1],
                            ind_CD[kl, ncol - 2],
                            ind_CD[kl, ncol - 1],
                        )
                        cov_out[ell1, ell2, ij, kl] = cov_6D[ell1, ell2, zi, zj, zk, zl]

    elif cov_6D.shape[0] != cov_6D.shape[1]:
        cov_out = np.zeros((nbl, npairs_AB, npairs_CD))
        for ij in range(npairs_AB):
            for kl in range(npairs_CD):
                zi, zj, zk, zl = (
                    ind_AB[ij, ncol - 2],
                    ind_AB[ij, ncol - 1],
                    ind_CD[kl, ncol - 2],
                    ind_CD[kl, ncol - 1],
                )
                cov_out[:, ij, kl] = cov_6D[:, zi, zj, zk, zl]

    return cov_out


def cov_4D_to_6D_blocks(
    cov_4D,
    nbl,
    zbins,
    ind_ab,
    ind_cd,
    symmetrize_output_ab: bool,
    symmetrize_output_cd: bool,
):
    """Reshapes the 4D covariance matrix to a 6D covariance matrix, even for
    the cross-probe (non-square) blocks needed to build the 3x2pt covariance.

    This function can be used for the normal routine (valid for auto-covariance,
    i.e., LL-LL, GG-GG, GL-GL and LG-LG)
    where `zpairs_ab = zpairs_cd` and `ind_ab = ind_cd`.

    Args:
        cov_4D (np.ndarray): The 4D covariance matrix.
        nbl (int): The number of ell bins.
        zbins (int): The number of redshift bins.
        ind_ab (np.ndarray): The indices for the first pair of redshift bins.
        ind_cd (np.ndarray): The indices for the second pair of redshift bins.
        symmetrize_output_ab (bool): Whether to symmetrize the output cov block
        for the first pair of probes.
        symmetrize_output_cd (bool): Whether to symmetrize the output cov block
        for the second pair of probes.

    Returns:
        np.ndarray: The 6D covariance matrix.

    """
    assert ind_ab.shape[1] == ind_cd.shape[1], (
        'ind_ab and ind_cd must have the same number of columns'
    )
    assert ind_ab.shape[1] == 2 or ind_ab.shape[1] == 4, (
        'ind_ab and ind_cd must have 2 or 4 columns'
    )
    assert cov_4D.ndim == 4, 'cov_4D must be a 4D array'
    assert cov_4D.shape[0] == nbl, 'cov_4D.shape[0] != nbl'
    assert cov_4D.shape[1] == nbl, 'cov_4D.shape[1] != nbl'

    ncols = ind_ab.shape[1]
    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    assert cov_4D.shape[2] == zpairs_ab, 'cov_4D.shape[2] != zpairs_ab'
    assert cov_4D.shape[3] == zpairs_cd, 'cov_4D.shape[3] != zpairs_cd'

    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    for ell2 in range(nbl):
        for ij in range(zpairs_ab):
            for kl in range(zpairs_cd):
                i, j, k, l = (
                    ind_ab[ij, ncols - 2],
                    ind_ab[ij, ncols - 1],
                    ind_cd[kl, ncols - 2],
                    ind_cd[kl, ncols - 1],
                )
                cov_6D[:, ell2, i, j, k, l] = cov_4D[:, ell2, ij, kl]

    # GL blocks are not symmetric
    # ! this part makes this function quite slow
    if symmetrize_output_ab:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(
                            cov_6D[ell1, ell2, :, :, i, j]
                        )

    if symmetrize_output_cd:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                for i in range(zbins):
                    for j in range(zbins):
                        cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(
                            cov_6D[ell1, ell2, i, j, :, :]
                        )

    return cov_6D


def cov_4D_to_6D_blocks_opt(
    cov_4D, nbl, zbins, ind_ab, ind_cd, symmetrize_output_ab, symmetrize_output_cd
):
    assert ind_ab.shape[1] == ind_cd.shape[1], (
        'ind_ab and ind_cd must have the same number of columns'
    )
    assert ind_ab.shape[1] in {2, 4}, 'ind_ab and ind_cd must have 2 or 4 columns'

    ncols = ind_ab.shape[1]
    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    cov_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))

    ell2_indices, ij_indices, kl_indices = np.ogrid[:nbl, :zpairs_ab, :zpairs_cd]
    i_indices = ind_ab[ij_indices, ncols - 2]
    j_indices = ind_ab[ij_indices, ncols - 1]
    k_indices = ind_cd[kl_indices, ncols - 2]
    l_indices = ind_cd[kl_indices, ncols - 1]

    cov_6D[:, ell2_indices, i_indices, j_indices, k_indices, l_indices] = cov_4D[
        :, ell2_indices, ij_indices, kl_indices
    ]

    if symmetrize_output_ab or symmetrize_output_cd:
        for ell1 in range(nbl):
            for ell2 in range(nbl):
                if symmetrize_output_ab:
                    for i in range(zbins):
                        for j in range(zbins):
                            cov_6D[ell1, ell2, :, :, i, j] = symmetrize_2d_array(
                                cov_6D[ell1, ell2, :, :, i, j]
                            )
                if symmetrize_output_cd:
                    for i in range(zbins):
                        for j in range(zbins):
                            cov_6D[ell1, ell2, i, j, :, :] = symmetrize_2d_array(
                                cov_6D[ell1, ell2, i, j, :, :]
                            )

    return cov_6D


def check_symmetric(array_2d, exact, rtol=1e-05):
    """:param a: 2d array
    :param exact: bool
    :param rtol: relative tolerance
    :return: bool, whether the array is symmetric or not
    """
    # """check if the matrix is symmetric, either exactly or within a tolerance
    # """
    assert isinstance(exact, bool), 'parameter "exact" must be either True or False'
    assert array_2d.ndim == 2, 'the array is not square'
    if exact:
        return np.array_equal(array_2d, array_2d.T)
    else:
        return np.allclose(array_2d, array_2d.T, rtol=rtol, atol=0)


def cov_4D_to_2D(
    cov_4D: np.ndarray, block_index: str, optimize: bool = True
) -> np.ndarray:
    """Reshapes the covariance from 4D to 2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of
    the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.

    this function can also convert to 2D non-square blocks; this is needed to build
    the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a
    zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function
     (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix
     (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from
    [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in"
    the missing elements in the
    higher-dimensional array are needed.
    """
    assert type(cov_4D) is np.ndarray, 'cov_4D must be numpy array'
    assert type(block_index) is str, 'block_index must be a string'
    assert type(optimize) is bool, 'optimize must be a boolean'

    assert block_index in ['ell', 'C-style'] + ['zpair', 'F-style'], (
        'block_index must be "ell", "C-style" or "zpair", "F-style"'
    )

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], (
        'the first two axes of the input covariance must have the same size'
    )
    # assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    nbl = int(cov_4D.shape[0])
    zpairs_AB = int(cov_4D.shape[2])
    zpairs_CD = int(cov_4D.shape[3])

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if optimize:
        if block_index in ['ell', 'C-style']:
            cov_2D.reshape(nbl, zpairs_AB, nbl, zpairs_CD)[:, :, :, :] = (
                cov_4D.transpose(0, 2, 1, 3)
            )

        elif block_index in ['zpair', 'F-style']:
            cov_2D.reshape(zpairs_AB, nbl, zpairs_CD, nbl)[:, :, :, :] = (
                cov_4D.transpose(2, 0, 3, 1)
            )
        return cov_2D

    # I tested that the 2 methods give the same results.
    # This code is kept to remember the
    # block_index * block_size + running_index unpacking
    if block_index in ['ell', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[
                            l1, l2, ipair, jpair
                        ]

    elif block_index in ['zpair', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[
                            l1, l2, ipair, jpair
                        ]

    return cov_2D


def build_cov_3x2pt_2d(
    cov_term_dict: dict, cov_ordering_2d: str, obs_space: str
) -> np.ndarray:
    """
    Constructs the 3x2pt covariance matrix in 2D, starting from the individual probe
    blocks stored in cov_term_dict (that is, the dictionary passed to this function)
    is cov_term_dict = cov_dict[term].

    This allows to skip entirely the cov 3x2pt 4d format, which is
    rather cumbersome and unnecessary.

    Note: the zpair_probe_scale ordering is not currently implemnted, but it should
    simply be a matter of replicating the second

    """
    assert cov_ordering_2d in [
        'scale_probe_zpair',
        'probe_scale_zpair',
        'probe_zpair_scale',
    ], (
        'cov_ordering_2d must be one of '
        '"scale_probe_zpair", "probe_scale_zpair", "probe_zpair_scale"'
    )

    # I loop like the diagonal probe blocks instead of taking directly the cov
    # dict keys to enforce probe ordering to be LL, GL, GG (or xip, xim, gt, w)
    if obs_space == 'harmonic':
        prefix = 'HS'
    elif obs_space == 'real':
        prefix = 'RS'
    elif obs_space == 'cosebis':
        prefix = 'CS'
    else:
        raise ValueError(
            f'obs_space must be "real", "harmonic" or "cosebis", not: {obs_space}'
        )

    diag_probes = const.__getattribute__(f'{prefix}_DIAG_PROBES')

    # get the number of ell bins
    probe_blocks = [k for k in cov_term_dict if cov_term_dict[k] is not None]
    first_block = probe_blocks[0]
    nbx = cov_term_dict[first_block]['4d'].shape[0]

    # make sure it's consistent across all probes and dimensions
    # (# ell bins = # ell^prime bins)
    for probe_2tpl in probe_blocks:
        if probe_2tpl == '3x2pt':
            continue
        _nbx = cov_term_dict[probe_2tpl]['4d'].shape[0]
        assert _nbx == cov_term_dict[probe_2tpl]['4d'].shape[1], (
            'axes lengths must match'
        )
        assert nbx == _nbx, 'axes lengths must match'

    if cov_ordering_2d in ['probe_scale_zpair', 'probe_zpair_scale']:
        rows = []
        for probe_ab in diag_probes:
            row_blocks = []
            for probe_cd in diag_probes:
                if (probe_ab, probe_cd) not in cov_term_dict:
                    continue

                row_blocks.append(cov_term_dict[probe_ab, probe_cd]['2d'])

            if row_blocks:
                rows.append(np.hstack(row_blocks))

        if rows:
            cov_3x2pt_2d = np.vstack(rows)
        else:
            raise ValueError('No valid probe combinations found!')

    # For scale_probe_zpair: outer loop is scale/ell, then probe, then zpair
    # Work directly with 4D arrays and extract zpair slices for each ell pair
    elif cov_ordering_2d == 'scale_probe_zpair':
        rows = []
        for ell1 in range(nbx):
            row_blocks = []
            for ell2 in range(nbx):
                # For this ell pair, stack all probe combinations
                probe_rows = []
                for probe_ab in diag_probes:
                    probe_cols = []
                    for probe_cd in diag_probes:
                        if (probe_ab, probe_cd) not in cov_term_dict:
                            continue

                        cov_4d = cov_term_dict[probe_ab, probe_cd]['4d']
                        # Extract the zpair × zpair slice for this ell pair
                        probe_cols.append(cov_4d[ell1, ell2, :, :])

                    if probe_cols:
                        probe_rows.append(np.hstack(probe_cols))

                if probe_rows:
                    row_blocks.append(np.vstack(probe_rows))

            if row_blocks:
                rows.append(np.hstack(row_blocks))

        if rows:
            cov_3x2pt_2d = np.vstack(rows)
        else:
            raise ValueError('No valid probe combinations found!')

    return cov_3x2pt_2d


def cov2corr(covariance):
    """Convert a covariance matrix to a correlation matrix."""
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)

    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = np.divide(covariance, outer_v)
        # Ensure zero covariance entries are explicitly zero
        correlation[covariance == 0] = 0
        # correlation[~np.isfinite(correlation)] = 0  # Set any NaN or inf values to 0

    return correlation


def build_noise(
    zbins: int,
    n_probes: int,
    sigma_eps2: list | tuple | np.ndarray,
    ng_shear: list | tuple | np.ndarray,
    ng_clust: list | tuple | np.ndarray,
    is_noiseless: bool = False,
) -> np.ndarray:
    """Builds the noise power spectra.

    Parameters
    ----------
    zbins : int
        Number of redshift bins.
    n_probes : int
        Number of probes.
    sigma_eps2 : list | tuple | np.ndarray
        Square of the *total* ellipticity dispersion.
        sigma_eps2 = sigma_eps ** 2, with
        sigma_eps = sigma_eps_i * sqrt(2),
        sigma_eps_i being the ellipticity dispersion *per component*
    ng_shear : list | tuple | np.ndarray
        Galaxy density of sources, relevant for cosmic shear
        If a scalar, cumulative galaxy density number density, per arcmin^2.
        This will assume equipopulated bins.
        If an array, galaxy number density, per arcmin^2, per redshift bin.
        Must have length zbins.
    ng_clust : list | tuple | np.ndarray
        Galaxy density of lenses, relevant for galaxy clustering
        If a scalar, cumulative galaxy density number density, per arcmin^2.
        This will assume equipopulated bins.
        If an array, galaxy number density, per arcmin^2, per redshift bin.
        Must have length zbins.
    is_noiseless : bool, optional
        If True, returns array of zeros of the right shape.

    Returns
    -------
    nl_4d : np.ndarray
        Noise power spectra matrices of shape (n_probes, n_probes, zbins, zbins)

    Notes
    -----
    The noise N is defined as:
        N_LL = sigma_eps^2 / (2 * n_bar)
        N_GG = 1 / n_bar
        N_GL = N_LG = 0

    """
    # assert appropriate inputs are list, tuple or np.ndarray
    for var, name in zip([ng_shear, ng_clust], ['ng_shear', 'ng_clust'], strict=False):
        #     [ng_shear, ng_clust, sigma_eps2],
        #     ['ng_shear', 'ng_clust', 'sigma_eps2'],
        # ):
        assert isinstance(var, (list, tuple, np.ndarray)), (
            f'{name} should be a list, tuple or np.ndarray'
        )

    # convert to np arrays if needed
    if isinstance(ng_shear, (list, tuple)):
        ng_shear = np.array(ng_shear)
    if isinstance(ng_clust, (list, tuple)):
        ng_clust = np.array(ng_clust)
    # if isinstance(ng_clust, (list, tuple)):
    # sigma_eps2 = np.array(sigma_eps2)

    conversion_factor = (180 / np.pi * 60) ** 2  # deg^2 to arcmin^2

    assert np.all(ng_shear > 0), 'ng_shear should be positive'
    assert np.all(ng_clust > 0), 'ng_clust should be positive'

    # if ng is an array, n_bar == ng (this is a slight misnomer, since ng is the
    # cumulative galaxy density, while
    # n_bar the galaxy density in each bin). In this case, if the bins are
    # quipopulated, the n_bar array should
    # have all entries almost identical.

    n_bar_shear = ng_shear * conversion_factor
    n_bar_clust = ng_clust * conversion_factor

    # create and fill N
    nl_4d = np.zeros((n_probes, n_probes, zbins, zbins))

    if is_noiseless:
        return nl_4d

    np.fill_diagonal(nl_4d[0, 0, :, :], sigma_eps2 / (2 * n_bar_shear))
    np.fill_diagonal(nl_4d[1, 1, :, :], 1 / n_bar_clust)
    nl_4d[0, 1, :, :] = 0
    nl_4d[1, 0, :, :] = 0

    return nl_4d
