import numpy as np

"""
perform consistency checks
"""

survey_area_ISTF = 15_000  # deg^2
survey_area_SPV3 = 14_000  # deg^2, new in 2022
deg2_in_sphere = 41252.96  # deg^2 in a spere

fsky_ISTF = survey_area_ISTF / deg2_in_sphere
fsky_sylvain = 0.375
fsky_SPV3 = survey_area_SPV3 / deg2_in_sphere



def get_specs(which_forecast):

    if which_forecast == 'IST':
        fsky = fsky_ISTF
        GL_or_LG = 'GL'
        ind_ordering = 'vincenzo'
        cl_folder = 'Cij_14may'

    elif which_forecast == 'sylvain':
        fsky = fsky_sylvain
        GL_or_LG = 'GL'
        ind_ordering = 'vincenzo'
        cl_folder = 'Cij_14may'

    elif which_forecast == 'CLOE':
        fsky = fsky_ISTF
        GL_or_LG = 'LG'
        ind_ordering = 'CLOE'
        cl_folder = 'Cl_CLOE'

    elif which_forecast == 'SPV3':
        fsky = fsky_SPV3
        GL_or_LG = 'GL'
        ind_ordering = 'triu'
        cl_folder = 'SPV3'

    elif which_forecast == 'SSCcomp_updt':
        fsky = fsky_ISTF
        GL_or_LG = 'GL'
        ind_ordering = 'triu'
        cl_folder = 'SPV3'

    else:
        raise ValueError('which_forecast must be IST, sylvain, CLOE, SPV3 or SSCcomp_updt')

    return fsky, GL_or_LG, ind_ordering, cl_folder


def consistency_checks(general_config, covariance_config):
    """
    perform some checks on the consistency of the inputs. 
    The most important are the first three lines
    """

    which_forecast = general_config['which_forecast']

    if which_forecast == 'IST':
        assert covariance_config['fsky'] == fsky_ISTF, f'{which_forecast} uses fsky = {fsky_ISTF}'
        assert covariance_config['ind_ordering'] == 'vincenzo', f'{which_forecast} used Vincenzos ind ordering'
        assert covariance_config['GL_or_LG'] == 'GL', f'{which_forecast} uses GL'
        assert covariance_config['Rl'] == 4, 'In the SSC comparison we used Rl=4'
        assert general_config['cl_folder'] == 'Cij_14may', 'Latest Cls are Cij_14may'
        assert general_config['nbl'] == 30, f'{which_forecast} uses nbl = 30'
        assert (general_config['ell_max_WL'], general_config['ell_max_GC']) == (5000, 3000) or (3000, 750),\
            'case is neither optimistic nor pessimistic'
        assert general_config['use_WA'] is True, f'{which_forecast} uses Wadd'

    elif which_forecast == 'sylvain':
        assert covariance_config['fsky'] == fsky_sylvain, f'For SSCcomp we used fsky = {fsky_sylvain}'
        assert covariance_config['ind_ordering'] == 'vincenzo', 'For SSCcomp we used Vincenzos ind ordering'
        assert covariance_config['GL_or_LG'] == 'GL', 'For SSCcomp we used GL'
        assert covariance_config['Rl'] == 4, 'For SSCcomp we used Rl=4'
        # assert general_config['cl_folder'] == 'common_ell_and_deltas', 'XXX check For SSCcomp we used Cij_14may Cls'
        assert general_config['nbl'] == 30, 'For SSCcomp we used nbl = 30'

        # this is just for the optimistic! in the pessimistic case we used general_config['ell_max_GC'] = 750
        # assert general_config['ell_max_GC'] == 3000, 'For SSCcomp we used ell_max_GC = 3000'
        assert general_config['use_WA'] is True, 'For SSCcomp we used Wadd'

    elif which_forecast == 'IST_NL':
        assert covariance_config['fsky'] == fsky_ISTF, f'{which_forecast} uses fsky = {fsky_ISTF}'
        assert covariance_config['ind_ordering'] == 'triu', f'{which_forecast} uses CLOEs ind ordering, which is triu row-major'
        assert covariance_config['GL_or_LG'] == 'GL', f'{which_forecast} uses GL'
        assert covariance_config['Rl'] == 4, f'{which_forecast} uses Rl = 4'
        assert general_config['ell_max_WL'] == 5000, f'{which_forecast} uses ell_max_WL = 5000'
        assert general_config['ell_max_GC'] == 5000, f'{which_forecast} uses ell_max_GC = 5000'
        assert general_config['cl_folder'] == 'Cl_CLOE', 'XXX check not quite sure about this cl_folder thing...'
        assert general_config['use_WA'] is False, f'{which_forecast} does not use Wadd'
        assert general_config['nbl'] == 20, f'{which_forecast} uses nbl = 20'

    elif which_forecast == 'SPV3':
        assert covariance_config['ind_ordering'] == 'triu', f'{which_forecast} used triu ind ordering'
        assert covariance_config['GL_or_LG'] == 'GL', f'{which_forecast} uses GL'
        assert covariance_config['fsky'] == fsky_SPV3, f'SPV3 uses fsky = {fsky_SPV3}'
        assert covariance_config['ng'] == 28.73, f'{which_forecast} uses ng = 28.73'
        assert covariance_config['sigma_eps2'] == (0.26 * np.sqrt(2)) ** 2
        assert covariance_config['block_index'] == 'ell'
        assert covariance_config['which_probe_response'] == 'variable'
        assert general_config['cl_folder'] == 'SPV3', f'{which_forecast} uses SPV3 cls'
        assert general_config['nbl_WL'] == 32 or 20, f'{which_forecast} uses nbl_WL = 32 or 20'
        assert general_config['ell_max_GC'] == 3000 or 750, f'{which_forecast} uses ell_max_GC = 3000 or 750'
        assert general_config['use_WA'] is True, f'{which_forecast} uses Wadd'

    elif which_forecast == 'SSCcomp_updt':
        assert covariance_config['ind_ordering'] == 'triu', f'{which_forecast} used triu ind ordering'
        assert covariance_config['GL_or_LG'] == 'GL', f'{which_forecast} uses GL'
        assert covariance_config['fsky'] == fsky_ISTF, f'{which_forecast} uses fsky = {fsky_ISTF}'
        assert covariance_config['ng'] == 28.73, f'{which_forecast} uses ng = 28.73'
        assert covariance_config['sigma_eps2'] == (0.3) ** 2
        assert covariance_config['block_index'] == 'ell'
        assert covariance_config['which_probe_response'] == 'variable'
        assert general_config['EP_or_ED'] == 'EP', f'{which_forecast} uses equipopulated bins'
        assert general_config['cl_folder'] == 'SPV3', f'{which_forecast} uses SPV3 cls'
        assert general_config['nbl'] == 30, f'{which_forecast} uses nbl = 20'
        assert general_config['ell_max_GC'] == 3000 or 750, f'{which_forecast} uses ell_max_GC = 3000 or 750'
        assert general_config['use_WA'] is True, f'{which_forecast} uses Wadd'

