import numpy as np

HS_PROBE_NAME_TO_IX_DICT = {'L': 0, 'G': 1}
PROBE_ORDERING = [('L', 'L'), ('G', 'L'), ('G', 'G')]
HS_DIAG_PROBES = ['LL', 'GL', 'GG']
RS_DIAG_PROBES = ['xip', 'xim', 'gt', 'gg']


def cov_10d_array_to_dict(cov_10d_array: np.ndarray) -> dict:
    """Transforms an array of shape
    (n_probes, n_probes, n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)
    to a dictionary of "shape"
    {(A, B, C, D): [nbl, nbl, zbins, zbins, zbins, zbins]}
    (where A, B, C, D is a tuple of strings, each one
    being either 'L' or 'G') to a numpy array of shape (n_probes, n_probes,
    n_probes, n_probes, nbl, nbl, zbins, zbins, zbins, zbins)
    """
    cov_10d_dict = {}
    for probe_a_str, probe_b_str in PROBE_ORDERING:
        for probe_c_str, probe_d_str in PROBE_ORDERING:
            probe_a_idx, probe_b_idx, probe_c_idx, probe_d_idx = (
                HS_PROBE_NAME_TO_IX_DICT[probe_a_str],
                HS_PROBE_NAME_TO_IX_DICT[probe_b_str],
                HS_PROBE_NAME_TO_IX_DICT[probe_c_str],
                HS_PROBE_NAME_TO_IX_DICT[probe_d_str],
            )
            cov_10d_dict[probe_a_str, probe_b_str, probe_c_str, probe_d_str] = (
                cov_10d_array[probe_a_idx, probe_b_idx, probe_c_idx, probe_d_idx, ...]
            )

    return cov_10d_dict


def cov_3x2pt_10d_to_4d(
    cov_3x2pt_10d: np.ndarray, zbins: int, nbl, req_probe_combs_2d: list[str]
) -> np.ndarray:
    """Takes the cov_3x2pt_10d dictionary, reshapes each A, B, C, d block
    separately in 4d, then stacks the blocks in the right order to output
    cov_3x2pt_4d (which is not a dictionary but a numpy array)

    probe_ordering: e.g. ['L', 'L'], ['G', 'L'], ['G', 'G']]
    """
    # if it's an array, convert to dictionary for the function to work
    assert isinstance(cov_3x2pt_10d, np.ndarray), 'cov_3x2pt_10d must be an array'

    cov_3x2pt_dict_10d = cov_10d_array_to_dict(cov_3x2pt_10d)

    ind_auto = np.array([(i, j) for i in range(zbins) for j in range(i, zbins)])
    ind_cross = np.array([(i, j) for i in range(zbins) for j in range(zbins)])
    ind_dict = {('L', 'L'): ind_auto, ('G', 'L'): ind_cross, ('G', 'G'): ind_auto}

    # these are only needed for a sanity check
    zpairs_auto = (zbins * (zbins + 1)) // 2
    zpairs_cross = zbins**2
    assert ind_auto.shape[0] == zpairs_auto, f'ind_auto should have {zpairs_auto} rows'
    assert ind_cross.shape[0] == zpairs_cross, (
        f'ind_cross should have {zpairs_cross} rows'
    )

    # initialize the 4d dictionary and list of probe combinations
    cov_3x2pt_dict_4d = {}

    # make each block 4d and store it with the right 'A', 'B', 'C, 'd' key
    for probe_a_str, probe_b_str, probe_c_str, probe_d_str in req_probe_combs_2d:
        cov_3x2pt_dict_4d[probe_a_str, probe_b_str, probe_c_str, probe_d_str] = (
            cov_6d_to_4d_blocks(
                cov_3x2pt_dict_10d[probe_a_str, probe_b_str, probe_c_str, probe_d_str],
                nbl,
                ind_dict[probe_a_str, probe_b_str],
                ind_dict[probe_c_str, probe_d_str],
            )
        )

    # concatenate the rows to construct the final matrix
    cov_3x2pt_4d = cov_3x2pt_8d_dict_to_4d(cov_3x2pt_dict_4d, req_probe_combs_2d)

    return cov_3x2pt_4d


def split_probe_name(
    full_probe_name: str, space: str, valid_probes=None
) -> tuple[str, str]:
    """Splits a full probe name (e.g., 'gtxim') into two component probes."""

    # this is the default: use hardcoded probe names
    if valid_probes is None:
        if space == 'harmonic':
            valid_probes = HS_DIAG_PROBES
        elif space == 'real':
            valid_probes = RS_DIAG_PROBES
        else:
            raise ValueError(
                f'`space` needs to be one of `harmonic` or `real`, got {space}'
            )
    else:
        assert isinstance(valid_probes, (list, tuple)), 'valid_probes must be a list'

    # Try splitting at each possible position
    for i in range(2, len(full_probe_name)):
        first, second = full_probe_name[:i], full_probe_name[i:]
        if first in valid_probes and second in valid_probes:
            return first, second

    raise ValueError(
        f'Invalid probe name: {full_probe_name}. '
        f'Expected two of {valid_probes} concatenated.'
    )


def cov_6d_to_4d_blocks(cov_6d, nbl, ind_ab, ind_cd):
    """Reshapes the covariance even for the non-diagonal (hence, non-square)
    blocks needed to build the 3x2pt.

    Use npairs_AB = npairs_CD and ind_ab = ind_cd for the normal routine
    (valid for auto-covariance LL-LL, GG-GG, GL-GL and LG-LG). n_columns
    is used to determine whether the ind array has 2 or 4 columns (if
    it's given in the form of a dictionary or not)
    """
    assert (cov_6d.shape[0] == cov_6d.shape[1] == nbl) or (cov_6d.shape[0] == nbl), (
        'number of angular bins does not match first two cov axes or the first axis'
    )
    zpairs_ab = ind_ab.shape[0]
    zpairs_cd = ind_cd.shape[0]

    cov_4d = np.zeros((nbl, nbl, zpairs_ab, zpairs_cd))
    for ell1 in range(nbl):
        for ell2 in range(nbl):
            for zij in range(zpairs_ab):
                for zkl in range(zpairs_cd):
                    zi, zj, zk, zl = (
                        ind_ab[zij, 0],
                        ind_ab[zij, 1],
                        ind_cd[zkl, 0],
                        ind_cd[zkl, 1],
                    )
                    cov_4d[ell1, ell2, zij, zkl] = cov_6d[ell1, ell2, zi, zj, zk, zl]

    return cov_4d


def cov_3x2pt_8d_dict_to_4d(cov_3x2pt_8d_dict, req_probe_combs_2d, space='harmonic'):
    """Convert a dictionary of 4D blocks into a single 4D array.

    This is the same code as
    in the last part of the function above.
    :param cov_3x2pt_8D_dict: Dictionary of 4D covariance blocks
    :param req_probe_combs_2d: List of probe combinations to use
    :return: 4D covariance array
    """

    # sanity check
    for key in cov_3x2pt_8d_dict:
        assert cov_3x2pt_8d_dict[key].ndim == 4, (
            f'covariance matrix {key} has ndim={cov_3x2pt_8d_dict[key].ndim} instead '
            f'of 4'
        )

    # check that the req_probe_combs_2d  are correct
    for probe in req_probe_combs_2d:
        if space == 'harmonic':
            probe_tpl = tuple(probe)
        elif space == 'real':
            probe_tpl = split_probe_name(probe, 'real')

        assert probe_tpl in cov_3x2pt_8d_dict, (
            f'Probe combination {probe_tpl} not found in the input dictionary'
        )

    final_rows = []

    if space == 'harmonic':
        row_ll_list, row_gl_list, row_gg_list = [], [], []
        for a, b, c, d in req_probe_combs_2d:
            if (a, b) == ('L', 'L'):
                row_ll_list.append(cov_3x2pt_8d_dict[a, b, c, d])
            elif (a, b) == ('G', 'L'):
                row_gl_list.append(cov_3x2pt_8d_dict[a, b, c, d])
            elif (a, b) == ('G', 'G'):
                row_gg_list.append(cov_3x2pt_8d_dict[a, b, c, d])
            else:
                raise ValueError(
                    f'Probe combination {a, b, c, d} does not start with '
                    '("L", "L") or ("G", "L") or ("G", "G") '
                )
        # concatenate the lists to make rows
        # (only concatenate and include rows that have content)
        if row_ll_list:
            row_ll = np.concatenate(row_ll_list, axis=3)
            final_rows.append(row_ll)
        if row_gl_list:
            row_gl = np.concatenate(row_gl_list, axis=3)
            final_rows.append(row_gl)
        if row_gg_list:
            row_gg = np.concatenate(row_gg_list, axis=3)
            final_rows.append(row_gg)

    elif space == 'real':
        row_xip_list, row_xim_list, row_gt_list, row_gg_list = [], [], [], []
        for probe in req_probe_combs_2d:
            probe_ab, probe_cd = split_probe_name(probe, 'real')
            if probe_ab == 'xip':
                row_xip_list.append(cov_3x2pt_8d_dict[probe_ab, probe_cd])
            elif probe_ab == 'xim':
                row_xim_list.append(cov_3x2pt_8d_dict[probe_ab, probe_cd])
            elif probe_ab == 'gt':
                row_gt_list.append(cov_3x2pt_8d_dict[probe_ab, probe_cd])
            elif probe_ab == 'gg':
                row_gg_list.append(cov_3x2pt_8d_dict[probe_ab, probe_cd])
            else:
                raise ValueError(
                    f'Probe combination {probe_ab, probe_cd} does not start with '
                    '("xip") or ("xim") or ("gt") or ("gg") '
                )
        # concatenate the lists to make rows
        # only concatenate and include rows that have content
        if row_xip_list:
            row_xip = np.concatenate(row_xip_list, axis=3)
            final_rows.append(row_xip)
        if row_xim_list:
            row_xim = np.concatenate(row_xim_list, axis=3)
            final_rows.append(row_xim)
        if row_gt_list:
            row_gt = np.concatenate(row_gt_list, axis=3)
            final_rows.append(row_gt)
        if row_gg_list:
            row_gg = np.concatenate(row_gg_list, axis=3)
            final_rows.append(row_gg)

    else:
        raise ValueError(f'space must be "harmonic" or "real", not: {space}')

    # concatenate the rows to construct the final matrix
    if final_rows:
        cov_3x2pt_4D = np.concatenate(final_rows, axis=2)
    else:
        # If no rows at all, return empty array with appropriate shape
        raise ValueError('No valid probe combinations found!')

    return cov_3x2pt_4D


# def test():
#     """Put this at the end of your main.py to run a quick consistency check of the
#     cov_transofrm module"""
#     cov_hc_dict = heracles.io.read('./stop_looking_at_your_laptops_eyes_on_me.fits')
#     cov_10d = io_handler.cov_heracles_dict_to_sb_10d(
#         cov_hc_dict, zbins, ell_obj.nbl_3x2pt, 2
#     )
#     cov_4d = sl.cov_3x2pt_10D_to_4D(
#         cov_3x2pt_10D=cov_10d,
#         probe_ordering=probe_ordering,
#         nbl=ell_obj.nbl_3x2pt,
#         zbins=zbins,
#         ind_copy=ind.copy(),
#         GL_OR_LG=GL_OR_LG,
#         req_probe_combs_2d=req_probe_combs_hs_2d,
#     )
#     cov_2d = sl.cov_4D_to_2DCLOE_3x2pt_hs(cov_4d, zbins, req_probe_combs_hs_2d, 'ell')

#     # ! SIMPLIFIED VERSION OF THE RESHAPING ROUTINES TO PASS TO GUADA

#     req_probe_combs_2d = req_probe_combs_hs_2d
#     from spaceborne import cov_transform as ct

#     cov_4d = ct.cov_3x2pt_10d_to_4d()
