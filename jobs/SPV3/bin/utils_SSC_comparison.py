import numpy as np
import sys
from pathlib import Path
# get project directory
project_path = Path.cwd().parent.parent.parent.parent
# import configuration and functions modules
sys.path.append(str(project_path / 'lib'))
import my_module as mm



"""
Some job-specific modules
"""

def consistency_checks(general_config, covariance_config):
    """
    perform some checks on the consistency of the inputs. 
    The most important are the first three lines
    """
    assert covariance_config['fsky'] == 0.375, 'For SSCcomp we used fsky = 0.375'
    assert covariance_config['ind_ordering'] == 'vincenzo', 'For SSCcomp we used Vincenzos ind ordering'
    assert covariance_config['GL_or_LG'] == 'GL', 'For SSCcomp we used GL'
    assert covariance_config['Rl'] == 4, 'For SSCcomp we used Rl=4'
    assert general_config['cl_folder'] == 'Cij_14may', 'For SSCcomp we used Cij_14may Cls'
    assert general_config['nbl'] == 30, 'For SSCcomp we used nbl = 30'
    assert general_config['ell_max_GC'] == 3000, 'For SSCcomp we used ell_max_GC = 3000'
    assert general_config['use_WA'] == True, 'For SSCcomp we used Wadd'


def import_and_interpolate_cls(general_config, ell_dict):
    """
    This code imports and interpolates and rearranges the Cls
    """
    
    # import settings:
    nbl = general_config['nbl']
    zbins = general_config['zbins']
    cl_folder = general_config['cl_folder']
            
    # import ell values:
    ell_WL = ell_dict['ell_WL']
    ell_GC = ell_dict['ell_GC']
    ell_WA = ell_dict['ell_WA']
    ell_XC = ell_GC
        
    # nbl for Wadd
    if np.asanyarray(ell_WA).size == 1: nbl_WA = 1 # in the case of just one bin it would give error
    else: nbl_WA = ell_WA.shape[0]
    
    npairs, npairs_asimm, npairs_tot = mm.get_zpairs(zbins)

    # import Vincenzo's (different versions of) Cls
    if cl_folder == "Cij_thesis":
        C_LL_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/thesis_data/Cij_tesi/CijGG-N4TB-GR-eNLA.dat')
        C_XC_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/thesis_data/Cij_tesi/CijDG-N4TB-GR-eNLA.dat')
        C_GG_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/thesis_data/Cij_tesi/CijDD-N4TB-GR-eNLA.dat')
    elif cl_folder == "Cij_15gen": # Cij-NonLin-eNLA_15gen
        C_LL_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/Cij-NonLin-eNLA_15gen/CijLL-LCDM-NonLin-eNLA.dat')
        C_XC_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/Cij-NonLin-eNLA_15gen/CijLG-LCDM-NonLin-eNLA.dat')
        C_GG_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/Cij-NonLin-eNLA_15gen/CijGG-LCDM-NonLin-eNLA.dat')
        C_LL_import[:,0] = np.log10(C_LL_import[:,0])
        C_XC_import[:,0] = np.log10(C_XC_import[:,0])
        C_GG_import[:,0] = np.log10(C_GG_import[:,0])
        print('ATTENTION! XC is LG, not GL!!')
    if cl_folder == "Cij_14may":
        C_LL_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/14may/EP10/CijLL-GR-Flat-eNLA-NA.dat')
        C_XC_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/14may/EP10/CijGL-GR-Flat-eNLA-NA.dat') # XXX GL, not LG!!
        C_GG_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/CijDers/14may/EP10/CijGG-GR-Flat-eNLA-NA.dat')

    # import responses
    R_LL_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/responses/rijllcorr-istf-alex.dat')
    R_GL_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/responses/rijglcorr-istf-alex.dat')
    R_GG_import = np.genfromtxt(project_path / 'config/common_data/vincenzo/responses/rijggcorr-istf-alex.dat')
    ell_Rl = R_LL_import[:, 0]

    ###########################################################################
    # interpolate Vincenzo's Cls in ell values
    # careful, this part is a bit tricky. Pay attention to the ell_WL,
    # ell_XC arguments in e.g. fLL(ell_XC) vs fLL(ell_WL)
    cl_dict_2D = {}
    cl_dict_2D['C_LL_2D'] = mm.cl_interpolator(npairs, C_LL_import, ell_XC, nbl)
    cl_dict_2D['C_GG_2D'] = mm.cl_interpolator(npairs, C_GG_import, ell_XC, nbl)
    cl_dict_2D['C_WA_2D'] = mm.cl_interpolator(npairs, C_LL_import, ell_WA, nbl_WA)
    cl_dict_2D['C_XC_2D'] = mm.cl_interpolator(npairs_asimm, C_XC_import, ell_XC, nbl)
    cl_dict_2D['C_LL_WLonly_2D'] = mm.cl_interpolator(npairs, C_LL_import, ell_WL, nbl)
        
    Rl_dict_2D = {}
    Rl_dict_2D['R_LL_2D'] = mm.cl_interpolator(npairs, R_LL_import, ell_XC, nbl)
    Rl_dict_2D['R_GG_2D'] = mm.cl_interpolator(npairs, R_GG_import, ell_XC, nbl)
    Rl_dict_2D['R_WA_2D'] = mm.cl_interpolator(npairs, R_LL_import, ell_WA, nbl_WA)
    Rl_dict_2D['R_XC_2D'] = mm.cl_interpolator(npairs_asimm, R_GL_import, ell_XC, nbl)
    Rl_dict_2D['R_LL_WLonly_2D'] = mm.cl_interpolator(npairs, R_LL_import, ell_WL, nbl)

    return cl_dict_2D, Rl_dict_2D






