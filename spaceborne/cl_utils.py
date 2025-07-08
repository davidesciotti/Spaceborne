import numpy as np

from spaceborne import sb_lib as sl


def build_3x2pt_datavector_5D(
    dv_LLfor3x2pt_3D, dv_GL_3D, dv_GG_3D, nbl, zbins, n_probes=2
):
    dv_3x2pt_5D = np.zeros((n_probes, n_probes, nbl, zbins, zbins))
    dv_3x2pt_5D[0, 0, :, :, :] = dv_LLfor3x2pt_3D
    dv_3x2pt_5D[1, 0, :, :, :] = dv_GL_3D
    dv_3x2pt_5D[0, 1, :, :, :] = np.transpose(dv_GL_3D, (0, 2, 1))
    dv_3x2pt_5D[1, 1, :, :, :] = dv_GG_3D
    return dv_3x2pt_5D


def cl_SPV3_1D_to_3D(cl_1d, probe: str, nbl: int, zbins: int):
    """This function reshapes the SPV3 cls, which have a different format wrt
    the usual input files, from 1 to 3
    dimensions (5 dimensions for the 3x2pt case)
    """
    zpairs_auto, zpairs_cross, zpairs_3x2pt = sl.get_zpairs(zbins)

    # the checks on zpairs in the if statements can only be done for the
    # optimistic case, since these are the only
    # datavectors I have (from which I can obtain the pessimistic ones simply by
    # removing some ell bins).

    # This case switch is not to repeat the assert below for each case
    if probe in ['WL', 'WA', 'GC']:
        zpairs = zpairs_auto
        is_symmetric = True
    elif probe == 'XC':
        zpairs = zpairs_cross
        is_symmetric = False
    elif probe == '3x2pt':
        zpairs = zpairs_3x2pt
    else:
        raise ValueError('probe must be WL, WA, XC, GC or 3x2pt')

    try:
        assert zpairs == int(cl_1d.shape[0] / nbl), (
            'the number of elements in the datavector is incompatible '
            'with the number of ell bins for this case/probe'
        )
    except ZeroDivisionError:
        if probe == 'WA':
            print('There are 0 bins for Wadd in this case, cl_wa will be empty')

    if probe != '3x2pt':
        cl_3d = sl.cl_1D_to_3D(cl_1d, nbl, zbins, is_symmetric=is_symmetric)

        # if cl is not a cross-spectrum, symmetrize
        if probe != 'XC':
            cl_3d = sl.fill_3D_symmetric_array(cl_3d, nbl, zbins)
        return cl_3d

    elif probe == '3x2pt':
        cl_2d = np.reshape(cl_1d, (nbl, zpairs_3x2pt))

        # split into 3 2d datavectors
        cl_ll_3x2pt_2d = cl_2d[:, :zpairs_auto]
        cl_gl_3x2pt_2d = cl_2d[:, zpairs_auto : zpairs_auto + zpairs_cross]
        cl_gg_3x2pt_2d = cl_2d[:, zpairs_auto + zpairs_cross :]

        # reshape them individually - the symmetrization is done within the function
        cl_ll_3x2pt_3d = sl.cl_2D_to_3D_symmetric(
            cl_ll_3x2pt_2d, nbl=nbl, zpairs=zpairs_auto, zbins=zbins
        )
        cl_gl_3x2pt_3d = sl.cl_2D_to_3D_asymmetric(
            cl_gl_3x2pt_2d, nbl=nbl, zbins=zbins, order='C'
        )
        cl_gg_3x2pt_3d = sl.cl_2D_to_3D_symmetric(
            cl_gg_3x2pt_2d, nbl=nbl, zpairs=zpairs_auto, zbins=zbins
        )

        # use them to populate the datavector
        cl_3x2pt = np.zeros((2, 2, nbl, zbins, zbins))
        cl_3x2pt[0, 0, :, :, :] = cl_ll_3x2pt_3d
        cl_3x2pt[1, 1, :, :, :] = cl_gg_3x2pt_3d
        cl_3x2pt[1, 0, :, :, :] = cl_gl_3x2pt_3d
        cl_3x2pt[0, 1, :, :, :] = np.transpose(cl_gl_3x2pt_3d, (0, 2, 1))

        # in this case, return the datavector (I could name it "cl_3d" and
        # avoid this return statement, but it's not 3d!)

        return cl_3x2pt


def cl_ell_cut(cl_3D, ell_values, ell_cuts_matrix):
    """Cut (sets to zero) the cl_3D array at the ell values
    specified in ell_cuts_matrix
    """
    # TODO call get_ell_cuts_indices function here to avoid repeating code

    nbl = cl_3D.shape[0]
    zbins = cl_3D.shape[1]

    assert cl_3D.ndim == 3, 'cl_3D must be 3D'
    assert ell_cuts_matrix.ndim == 2, 'ell_cuts_matrix must be 2D'
    assert cl_3D.shape[1] == cl_3D.shape[2], (
        "the last two axes' dimensions do not coincide"
    )
    assert nbl == ell_values.shape[0], (
        'the number of ell bins in cl_3D and ell_values must be the same'
    )
    assert zbins == ell_cuts_matrix.shape[0], (
        "the number of zbins in cl_3D and ell_cuts_matrix axes' length must be the same"
    )

    cl_3D_ell_cut = cl_3D.copy()
    for zi in range(zbins):
        for zj in range(zbins):
            ell_cut = ell_cuts_matrix[zi, zj]
            if np.any(ell_values > ell_cut):  # i.e., if you need to do a cut at all
                ell_idxs_tocut = np.where(ell_values > ell_cut)[0]
                cl_3D_ell_cut[ell_idxs_tocut, zi, zj] = 0

    return cl_3D_ell_cut


def cl_ell_cut_v2(cl_3D, ell_values, ell_cuts_matrix):
    """Cut (sets to zero) the cl_3D array at the ell values
    specified in ell_cuts_matrix.
    Smarter version, without for loops - only marginally faster
    """
    nbl = cl_3D.shape[0]
    zbins = cl_3D.shape[1]

    assert cl_3D.ndim == 3, 'cl_3D must be 3D'
    assert ell_cuts_matrix.ndim == 2, 'ell_cuts_matrix must be 2D'
    assert cl_3D.shape[1] == cl_3D.shape[2], (
        "the last two axes' dimensions do not coincide"
    )
    assert nbl == ell_values.shape[0], (
        'the number of ell bins in cl_3D and ell_values must be the same'
    )
    assert zbins == ell_cuts_matrix.shape[0], (
        "the number of zbins in cl_3D and ell_cuts_matrix axes' length must be the same"
    )

    # Create a 3D mask of the same shape as cl_3D where the
    # elements that should be cut are marked as True
    ell_cuts_matrix_3D = np.expand_dims(ell_cuts_matrix, 0)
    mask = ell_values[:, None, None] > ell_cuts_matrix_3D

    # Use the mask to set the corresponding elements of cl_3D to zero
    cl_3D_ell_cut = np.where(mask, 0, cl_3D)

    return cl_3D_ell_cut


def cl_ell_cut_3x2pt(cl_3x2pt_5D, ell_cuts_dict, ell_values_3x2pt):
    """Wrapper function to quickly implement the cl (or derivatives) ell cut
    for the 3x2pt datavector
    """
    cl_LLfor3x2pt_3D = cl_3x2pt_5D[0, 0, :, :, :]
    cl_LGfor3x2pt_3D = cl_3x2pt_5D[0, 1, :, :, :]
    cl_GLfor3x2pt_3D = cl_3x2pt_5D[1, 0, :, :, :]
    cl_GGfor3x2pt_3D = cl_3x2pt_5D[1, 1, :, :, :]

    cl_LLfor3x2pt_3D_ell_cut = cl_ell_cut(
        cl_LLfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['LL']
    )
    cl_LGfor3x2pt_3D_ell_cut = cl_ell_cut(
        cl_LGfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['LG']
    )
    cl_GLfor3x2pt_3D_ell_cut = cl_ell_cut(
        cl_GLfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['GL']
    )
    cl_GGfor3x2pt_3D_ell_cut = cl_ell_cut(
        cl_GGfor3x2pt_3D, ell_values_3x2pt, ell_cuts_dict['GG']
    )

    cl_3x2pt_5D_ell_cut = np.zeros(cl_3x2pt_5D.shape)
    cl_3x2pt_5D_ell_cut[0, 0, :, :, :] = cl_LLfor3x2pt_3D_ell_cut
    cl_3x2pt_5D_ell_cut[0, 1, :, :, :] = cl_LGfor3x2pt_3D_ell_cut
    cl_3x2pt_5D_ell_cut[1, 0, :, :, :] = cl_GLfor3x2pt_3D_ell_cut
    cl_3x2pt_5D_ell_cut[1, 1, :, :, :] = cl_GGfor3x2pt_3D_ell_cut

    return cl_3x2pt_5D_ell_cut
