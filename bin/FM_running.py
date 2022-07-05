import sys
import time
from pathlib import Path
import numpy as np

project_path_here = Path.cwd().parent.parent.parent
sys.path.append(str(project_path_here.parent / 'common_lib'))
import my_module as mm

script_name = sys.argv[0]
start = time.perf_counter()


###############################################################################
################## CODE TO COMPUTE THE FISHER MATRIX ##########################
###############################################################################


###########################################

# XXX attention! the dC_LL matrix should have (nParams - 10) as dimension,
# since WL has no bias. This would complicate the structure of the datacector 
# and taking nParams instead seems to have ho impact on the final result.


def compute_FM(general_config, covariance_config, FM_config, ell_dict, cov_dict):
    # import settings:
    nbl = general_config['nbl']
    ell_max_GC = general_config['ell_max_GC']
    zbins = general_config['zbins']
    cl_folder = general_config['cl_folder']
    use_WA = general_config['use_WA']

    GL_or_LG = covariance_config['GL_or_LG']
    ind = covariance_config['ind']
    block_index = covariance_config['block_index']

    save_FM = FM_config['save_FM']
    nParams = FM_config['nParams']

    # import ell values
    ell_WL = ell_dict['ell_WL']
    ell_GC = ell_dict['ell_GC']
    ell_WA = ell_dict['ell_WA']
    ell_XC = ell_GC

    # set the flattening convention for the derivatives vector, based on the setting used to reduce the covariance
    # matrix' dimensions
    if block_index in ['ell', 'vincenzo', 'C-style']:
        which_flattening = 'C'
    elif block_index in ['ij', 'sylvain', 'F-style']:
        which_flattening = 'F'

    # # ! delete this
    # print('DELIBERATELY WRONG, DELETE THIS')
    # if block_index in ['ell', 'vincenzo', 'C-style']:
    #     which_flattening = 'F'
    # elif block_index in ['ij', 'sylvain', 'F-style']:
    #     which_flattening = 'C'
    # # ! end delete this

    # check to see if ell values are in linear or log scale
    if np.max(ell_WL) > 30:
        print('switching to log scale for the ell values')
        ell_WL = np.log10(ell_WL)
        ell_GC = np.log10(ell_GC)
        ell_WA = np.log10(ell_WA)
        ell_XC = ell_GC

    # nbl for Wadd: in the case of just one bin it would give error
    if ell_WA.size == 1:
        nbl_WA = 1
    else:
        nbl_WA = ell_WA.shape[0]

    nParams_bias = 10
    nParams_WL = nParams - nParams_bias

    # output_folder = mm.get_output_folder(ind_ordering, which_forecast)

    npairs, npairs_asimm, npairs_tot = mm.get_pairs(zbins)

    if GL_or_LG == 'LG':
        print('\nAttention! switching columns in the ind array (for the XC part)')
        ind[npairs:(npairs + npairs_asimm), [2, 3]] = ind[npairs:(npairs + npairs_asimm), [3, 2]]

    ############################################
    # initialize derivatives arrays
    dC_LL_WLonly = np.zeros((nbl, npairs, nParams))
    dC_LL = np.zeros((nbl, npairs, nParams))
    dC_XC = np.zeros((nbl, npairs_asimm, nParams))
    dC_GG = np.zeros((nbl, npairs, nParams))
    dC_WA = np.zeros((nbl_WA, npairs, nParams))

    # invert GO covmats
    start1 = time.perf_counter()
    cov_WL_GO_2D_inv = np.linalg.inv(cov_dict['cov_WL_GO_2D'])
    cov_GC_GO_2D_inv = np.linalg.inv(cov_dict['cov_GC_GO_2D'])
    cov_WA_GO_2D_inv = np.linalg.inv(cov_dict['cov_WA_GO_2D'])
    cov_3x2pt_GO_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_GO_2D'])
    print(f'GO covmats inverted in {(time.perf_counter() - start1):.2f} s')


    # invert GS covmats
    start2 = time.perf_counter()
    cov_WL_GS_2D_inv = np.linalg.inv(cov_dict['cov_WL_GS_2D'])
    cov_GC_GS_2D_inv = np.linalg.inv(cov_dict['cov_GC_GS_2D'])
    cov_WA_GS_2D_inv = np.linalg.inv(cov_dict['cov_WA_GS_2D'])
    cov_3x2pt_GS_2D_inv = np.linalg.inv(cov_dict['cov_3x2pt_GS_2D'])
    print(f'GO covmats inverted in {(time.perf_counter() - start2):.2f} s')

    # import derivatives: set correct naming conventions for each case
    if cl_folder in ["Cij_thesis", "Cij_15gen"]:
        print("attention! I'm taking the derivatives from the GCph folder in the 15gen case")
        if cl_folder == "Cij_thesis":
            derivatives_folder = project_path_here.parent / "common_data/vincenzo/thesis_data/Cij_derivatives_tesi"
        elif cl_folder == "Cij_15gen":
            derivatives_folder = 'C:/Users/dscio/Documents/Lavoro/Programmi/GCph/data/dCij'

        suffix = "N4TB-GR-eNLA"
        probe_code_LL = "GG"
        probe_code_GG = "DD"
        probe_code_XC = "DG"

    elif cl_folder == "Cij_14may":
        derivatives_folder = project_path_here.parent / f"common_data/vincenzo/14may/CijDers/EP{zbins}"
        suffix = "GR-Flat-eNLA-NA"
        probe_code_LL = "LL"
        probe_code_GG = "GG"
        probe_code_XC = "GL"

    # set parameters names for the different probes
    params_names_LL = ["Om", "Ob", "wz", "wa", "h", "ns", "s8", "Aia", "eIA", "bIA"]
    # this if-elif is just because the bias parametrers are called "b" in one case and "bL" in the other
    if cl_folder in ["Cij_thesis", "Cij_15gen"]:
        params_names_XC = params_names_LL + ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09", "b10"]
    elif cl_folder == "Cij_14may":
        params_names_XC = params_names_LL + ["bL01", "bL02", "bL03", "bL04", "bL05", "bL06", "bL07", "bL08",
                                             "bL09", "bL10"]
    params_names_GG = params_names_XC

    # import the derivatives in a dictionary
    dC_dict = dict(mm.get_kv_pairs(derivatives_folder, "dat"))

    ######### INTERPOLATION
    # XXXX dC_ALL_interpolated_dict may be initialized everytime, possible source of error
    # (I think everything's fine though)
    # XXXX todo comment the interpolator function

    # create dict to store interpolated Cij arrays
    dC_WLonly_interpolated_dict = {}
    dC_GConly_interpolated_dict = {}
    dC_3x2pt_interpolated_dict = {}
    dC_WA_interpolated_dict = {}


    # call the function to interpolate: PAY ATTENTION TO THE PARAMETERS PASSED!
    # WLonly
    dC_WLonly_interpolated_dict = mm.interpolator(probe_code=probe_code_LL,
                                                  dC_interpolated_dict=dC_WLonly_interpolated_dict,
                                                  dC_dict=dC_dict, params_names=params_names_LL, nbl=nbl,
                                                  npairs=npairs, ell_values=ell_WL, suffix=suffix)
    # GConly
    dC_GConly_interpolated_dict = mm.interpolator(probe_code=probe_code_GG,
                                                  dC_interpolated_dict=dC_GConly_interpolated_dict,
                                                  dC_dict=dC_dict, params_names=params_names_GG, nbl=nbl,
                                                  npairs=npairs, ell_values=ell_XC, suffix=suffix)
    # LL for 3x2pt
    dC_3x2pt_interpolated_dict = mm.interpolator(probe_code=probe_code_LL,
                                                 dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                                                 dC_dict=dC_dict, params_names=params_names_LL, nbl=nbl,
                                                 npairs=npairs, ell_values=ell_XC, suffix=suffix)
    # XC for 3x2pt
    dC_3x2pt_interpolated_dict = mm.interpolator(probe_code=probe_code_XC,
                                                 dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                                                 dC_dict=dC_dict, params_names=params_names_XC, nbl=nbl,
                                                 npairs=npairs_asimm, ell_values=ell_XC, suffix=suffix)
    # GG for 3x2pt
    dC_3x2pt_interpolated_dict = mm.interpolator(probe_code=probe_code_GG,
                                                 dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                                                 dC_dict=dC_dict, params_names=params_names_GG, nbl=nbl,
                                                 npairs=npairs, ell_values=ell_XC, suffix=suffix)
    # LL for WA
    dC_WA_interpolated_dict = mm.interpolator(probe_code=probe_code_LL, dC_interpolated_dict=dC_WA_interpolated_dict,
                                              dC_dict=dC_dict, params_names=params_names_LL, nbl=nbl_WA,
                                              npairs=npairs, ell_values=ell_WA, suffix=suffix)

    # fill the dC array using the interpolated dictionary
    # WLonly
    dC_LL_WLonly = mm.fill_dC_array(params_names=params_names_LL,
                                    dC_interpolated_dict=dC_WLonly_interpolated_dict,
                                    probe_code=probe_code_LL, dC=dC_LL_WLonly, suffix=suffix)
    # LL for 3x2pt
    dC_LL = mm.fill_dC_array(params_names=params_names_LL,
                             dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                             probe_code=probe_code_LL, dC=dC_LL, suffix=suffix)
    # XC for 3x2pt
    dC_XC = mm.fill_dC_array(params_names=params_names_XC,
                             dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                             probe_code=probe_code_XC, dC=dC_XC, suffix=suffix)
    # GG for 3x2pt and GConly
    dC_GG = mm.fill_dC_array(params_names=params_names_GG,
                             dC_interpolated_dict=dC_3x2pt_interpolated_dict,
                             probe_code=probe_code_GG, dC=dC_GG, suffix=suffix)
    # LL for WA
    dC_WA = mm.fill_dC_array(params_names=params_names_LL,
                             dC_interpolated_dict=dC_WA_interpolated_dict,
                             probe_code=probe_code_LL, dC=dC_WA, suffix=suffix)

    # ! reshape dC from (nbl, zpairs, nParams) to (nbl, zbins, zbins, nparams) - i.e., go from '2D' to '3D'
    # (+ 1 "excess" dimension). Note that Vincenzo uses np.triu to reduce the dimensions of the cl arrays,
    # but ind_vincenzo to organize the covariance matrix.

    dC_LL_4D = np.zeros((nbl, zbins, zbins, nParams))
    dC_GG_4D = np.zeros((nbl, zbins, zbins, nParams))
    dC_LL_WLonly_4D = np.zeros((nbl, zbins, zbins, nParams))
    dC_WA_4D = np.zeros((nbl_WA, zbins, zbins, nParams))

    # fill symmetric
    triu_idx = np.triu_indices(zbins)
    for ell in range(nbl):
        for alf in range(nParams):
            for i in range(npairs):
                dC_LL_4D[ell, triu_idx[0][i], triu_idx[1][i], alf] = dC_LL[ell, i, alf]
                dC_GG_4D[ell, triu_idx[0][i], triu_idx[1][i], alf] = dC_GG[ell, i, alf]
                dC_LL_WLonly_4D[ell, triu_idx[0][i], triu_idx[1][i], alf] = dC_LL_WLonly[ell, i, alf]
    # Wadd
    for ell in range(nbl_WA):
        for alf in range(nParams):
            for i in range(npairs):
                dC_WA_4D[ell, triu_idx[0][i], triu_idx[1][i]] = dC_WA[ell, i]

    # symmetrize
    for alf in range(nParams):
        dC_LL_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_LL_4D[:, :, :, alf], nbl, zbins)
        dC_GG_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_GG_4D[:, :, :, alf], nbl, zbins)
        dC_WA_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_WA_4D[:, :, :, alf], nbl_WA, zbins)
        dC_LL_WLonly_4D[:, :, :, alf] = mm.fill_3D_symmetric_array(dC_LL_WLonly_4D[:, :, :, alf], nbl, zbins)

    # fill asymmetric
    dC_XC_4D = np.reshape(dC_XC, (nbl, zbins, zbins, nParams))

    # ! flatten following 'ind' ordering
    dC_LL_3D = np.zeros((nbl, npairs, nParams))
    dC_GG_3D = np.zeros((nbl, npairs, nParams))
    dC_XC_3D = np.zeros((nbl, npairs_asimm, nParams))
    dC_WA_3D = np.zeros((nbl_WA, npairs, nParams))
    dC_LL_WLonly_3D = np.zeros((nbl, npairs, nParams))

    ind_LL = ind[:55, :]
    ind_GG = ind[:55, :]
    ind_XC = ind[55:155, :]  # ! watch out for the ind switch!!

    # collapse the 2 redshift dimensions: (i,j -> ij)
    for ell in range(nbl):
        for alf in range(nParams):
            dC_LL_3D[ell, :, alf] = mm.array_2D_to_1D_ind(dC_LL_4D[ell, :, :, alf], npairs, ind_LL)
            dC_GG_3D[ell, :, alf] = mm.array_2D_to_1D_ind(dC_GG_4D[ell, :, :, alf], npairs, ind_GG)
            dC_XC_3D[ell, :, alf] = mm.array_2D_to_1D_ind(dC_XC_4D[ell, :, :, alf], npairs_asimm, ind_XC)
            dC_LL_WLonly_3D[ell, :, alf] = mm.array_2D_to_1D_ind(dC_LL_WLonly_4D[ell, :, :, alf], npairs, ind_LL)

    for ell in range(nbl_WA):
        for alf in range(nParams):
            dC_WA_3D[ell, :, alf] = mm.array_2D_to_1D_ind(dC_WA_4D[ell, :, :, alf], npairs, ind_LL)

    ######################### FILL DATAVECTOR #####################################

    # fill 3D datavector
    D_WA_3D = dC_WA_3D
    D_WLonly_3D = dC_LL_WLonly_3D
    D_GConly_3D = dC_GG_3D
    D_3x2pt_3D = np.concatenate((dC_LL_3D, dC_XC_3D, dC_GG_3D), axis=1)

    # collapse ell and zpair - ATTENTION: np.reshape, like ndarray.flatten, accepts an 'ordering' parameter, which works
    # in the same way
    # not with the old datavector, which was ordered in a different way...
    D_WA_2D = np.reshape(D_WA_3D, (nbl_WA * npairs, nParams), order=which_flattening)
    D_WLonly_2D = np.reshape(D_WLonly_3D, (nbl * npairs, nParams), order=which_flattening)
    D_GConly_2D = np.reshape(D_GConly_3D, (nbl * npairs, nParams), order=which_flattening)
    D_3x2pt_2D = np.reshape(D_3x2pt_3D, (nbl * npairs_tot, nParams), order=which_flattening)

    ######################### COMPUTE FM #####################################

    # COMPUTE FM GO
    start3 = time.perf_counter()
    FM_WL_GO = mm.compute_FM_2D(nbl, npairs, nParams, cov_WL_GO_2D_inv, D_WLonly_2D)
    FM_GC_GO = mm.compute_FM_2D(nbl, npairs, nParams, cov_GC_GO_2D_inv, D_GConly_2D)
    FM_WA_GO = mm.compute_FM_2D(nbl_WA, npairs, nParams, cov_WA_GO_2D_inv, D_WA_2D)
    FM_3x2pt_GO = mm.compute_FM_2D(nbl, npairs_tot, nParams, cov_3x2pt_GO_2D_inv, D_3x2pt_2D)
    print(f'GO FM done in {(time.perf_counter() - start3):.2f} s')


    # COMPUTE FM GS
    start4 = time.perf_counter()
    FM_WL_GS = mm.compute_FM_2D(nbl, npairs, nParams, cov_WL_GS_2D_inv, D_WLonly_2D)
    FM_GC_GS = mm.compute_FM_2D(nbl, npairs, nParams, cov_GC_GS_2D_inv, D_GConly_2D)
    FM_WA_GS = mm.compute_FM_2D(nbl_WA, npairs, nParams, cov_WA_GS_2D_inv, D_WA_2D)
    FM_3x2pt_GS = mm.compute_FM_2D(nbl, npairs_tot, nParams, cov_3x2pt_GS_2D_inv, D_3x2pt_2D)
    print(f'GS FM done in {(time.perf_counter() - start4):.2f} s')

    # sum WA, this is the actual FM_3x2pt
    if use_WA:
        FM_3x2pt_GO += FM_WA_GO
        FM_3x2pt_GS += FM_WA_GS

    # delete null rows and columns to avoid having singular matrices
    # WL for 3x2pt has no bias
    FM_WL_GO_toSave = FM_WL_GO[:nParams_WL, :nParams_WL]
    FM_WL_GS_toSave = FM_WL_GS[:nParams_WL, :nParams_WL]
    # GConly has no IA parameters
    FM_GC_GO_toSave = np.delete(FM_GC_GO, (7, 8, 9), 0)
    FM_GC_GO_toSave = np.delete(FM_GC_GO_toSave, (7, 8, 9), 1)
    FM_GC_GS_toSave = np.delete(FM_GC_GS, (7, 8, 9), 0)
    FM_GC_GS_toSave = np.delete(FM_GC_GS_toSave, (7, 8, 9), 1)

    # save dictionary
    probe_names = ['WL', 'GC', '3x2pt']
    FMs_GO = [FM_WL_GO_toSave, FM_GC_GO_toSave, FM_3x2pt_GO]
    FMs_GS = [FM_WL_GS_toSave, FM_GC_GS_toSave, FM_3x2pt_GS]

    FM_dict = {}
    for probe_name, FM_GO, FM_GS in zip(probe_names, FMs_GO, FMs_GS):
        FM_dict[f'FM_{probe_name}_GO'] = FM_GO
        FM_dict[f'FM_{probe_name}_GS'] = FM_GS

    print("FMs computed in %.2f seconds" % (time.perf_counter() - start))

    return FM_dict

    # TODO: create pd dataframe
