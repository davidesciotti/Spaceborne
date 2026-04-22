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
