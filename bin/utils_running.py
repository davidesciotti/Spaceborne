"""
perform consistency checks
"""

survey_area = 15_000  # deg^2
survey_area_SPV3 = 14_000  # deg^2, new in 2022
deg2_in_sphere = 41252.96  # deg^2 in a spere

fsky_IST = survey_area / deg2_in_sphere
fsky_syvain = 0.375
fsky_SPV3 = survey_area_SPV3 / deg2_in_sphere


def consistency_checks(general_config, covariance_config):
    """
    perform some checks on the consistency of the inputs. 
    The most important are the first three lines
    """

    if general_config['which_forecast'] == 'IST':
        assert covariance_config['fsky'] == fsky_IST, 'IST forecast uses fsky = 0.3636'
        assert covariance_config['ind_ordering'] == 'vincenzo', 'IST forecast used Vincenzos ind ordering'
        assert covariance_config['GL_or_LG'] == 'GL', 'IST forecast uses GL'
        assert covariance_config['Rl'] == 4, 'In the SSC comparison we used Rl=4'
        assert general_config['cl_folder'] == 'Cij_14may', 'Latest Cls are Cij_14may'
        assert general_config['nbl'] == 30, 'IST forecast uses nbl = 30'
        assert general_config['ell_max_GC'] == 3000, 'IST forecast uses ell_max_GC = 3000'
        assert general_config['use_WA'] is True, 'IST forecast uses Wadd'

    elif general_config['which_forecast'] == 'sylvain':
        assert covariance_config['fsky'] == fsky_syvain, 'For SSCcomp we used fsky = 0.375'
        assert covariance_config['ind_ordering'] == 'vincenzo', 'For SSCcomp we used Vincenzos ind ordering'
        assert covariance_config['GL_or_LG'] == 'GL', 'For SSCcomp we used GL'
        assert covariance_config['Rl'] == 4, 'For SSCcomp we used Rl=4'
        # assert general_config['cl_folder'] == 'common_ell_and_deltas', 'XXX check For SSCcomp we used Cij_14may Cls'
        assert general_config['nbl'] == 30, 'For SSCcomp we used nbl = 30'

        # this is just for the optimistic! in the pessimistic case we used general_config['ell_max_GC'] = 750
        # assert general_config['ell_max_GC'] == 3000, 'For SSCcomp we used ell_max_GC = 3000'
        assert general_config['use_WA'] is True, 'For SSCcomp we used Wadd'

    elif general_config['which_forecast'] == 'IST_NL':
        assert covariance_config['fsky'] == fsky_IST, 'IST_NL uses fsky = 0.3636'
        assert covariance_config['ind_ordering'] == 'triu', 'IST_NL uses CLOEs ind ordering, which is triu row-major'
        assert covariance_config['GL_or_LG'] == 'GL', 'IST_NL uses GL'
        assert covariance_config['Rl'] == 4, 'IST_NL uses Rl = 4'
        assert general_config['ell_max_WL'] == 5000, 'IST_NL uses ell_max_WL = 5000'
        assert general_config['ell_max_GC'] == 5000, 'IST_NL uses ell_max_GC = 5000'
        assert general_config['cl_folder'] == 'Cl_CLOE', 'XXX check not quite sure about this cl_folder thing...'
        assert general_config['use_WA'] is False, 'IST_NL does not use Wadd'
        assert general_config['nbl'] == 20, 'IST_NL uses nbl = 20'

    elif general_config['which_forecast'] == 'SPV3':
        assert covariance_config['fsky'] == fsky_SPV3, 'SPV3 uses fsky = 0.3636'
        assert covariance_config['ind_ordering'] == 'vincenzo', 'IST forecast used Vincenzos ind ordering'
        assert covariance_config['GL_or_LG'] == 'GL', 'IST forecast uses GL'
        assert covariance_config['Rl'] == 4, 'In the SSC comparison we used Rl=4'
        assert general_config['cl_folder'] == 'Cij_14may', 'Latest Cls are Cij_14may'
        assert general_config['nbl'] == 30, 'IST forecast uses nbl = 30'
        assert general_config['ell_max_GC'] == 3000, 'IST forecast uses ell_max_GC = 3000'
        assert general_config['use_WA'] is True, 'IST forecast uses Wadd'
