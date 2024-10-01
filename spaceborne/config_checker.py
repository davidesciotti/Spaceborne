import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.my_module as mm
import numpy as np


class SpaceborneConfigChecker:
    
    def __init__(self, cfg):
        self.general_cfg = cfg['general_cfg']
        self.covariance_cfg = cfg['covariance_cfg']
        self.fm_cfg = cfg['FM_cfg']

    def check_h_units(self):
        if self.general_cfg['use_h_units']:
            return "hoverMpc", "Mpcoverh3"
        else:
            return "1overMpc", "Mpc3"

    def check_ell_cuts(self):
        if self.general_cfg['ell_cuts']:
            assert self.general_cfg['BNT_transform'], 'You should BNT transform if you want to apply ell cuts.'

    def check_BNT_transform(self):
        if self.covariance_cfg['cov_BNT_transform']:
            assert not self.general_cfg['cl_BNT_transform'], 'The BNT transform should be applied either to the Cls or to the covariance.'
            assert self.fm_cfg['derivatives_BNT_transform'], 'You should BNT transform the derivatives as well.'

    def check_fsky(self):
        fsky_check = cosmo_lib.deg2_to_fsky(self.covariance_cfg['survey_area_deg2'])
        assert np.abs(mm.percent_diff(self.covariance_cfg['fsky'],
                      fsky_check)) < 1e-5, 'Fsky does not match the survey area.'

    def check_which_forecast(self):
        assert self.general_cfg['which_forecast'] == 'SPV3', 'ISTF forecasts not supported at the moment'

    def check_which_forecast(self):
        assert self.covariance_cfg['Spaceborne_cfg']['cl_integral_convention'] in [
            'PySSC', 'Euclid'], 'cl_integral_convention must be "PySSC" or "Euclid"'

    def check_z_grid_ssc_integrands(self):
        assert self.covariance_cfg['Spaceborne_cfg']['z_steps_ssc_integrands'] >= 250, \
            'z_steps_ssc_integrands is small, at the moment it used to compute various intermediate quantities'
            
    def check_nuisance(self):
        # ! some check on the input nuisance values
        # assert np.all(np.array(covariance_cfg['ngal_lensing']) <
        # 9), 'ngal_lensing values are likely < 9 *per bin*; this is just a rough check'
        # assert np.all(np.array(covariance_cfg['ngal_clustering']) <
        # 9), 'ngal_clustering values are likely < 9 *per bin*; this is just a rough check'
        assert np.all(np.array(self.covariance_cfg['ngal_lensing']) > 0), 'ngal_lensing values must be positive'
        assert np.all(np.array(self.covariance_cfg['ngal_clustering']) > 0), 'ngal_clustering values must be positive'
        assert np.all(np.array(self.covariance_cfg['zbin_centers']) > 0), 'z_center values must be positive'
        assert np.all(np.array(self.covariance_cfg['zbin_centers']) < 3), 'z_center values are likely < 3; this is just a rough check'
        assert self.general_cfg['magcut_source'] == 245, 'magcut_source should be 245, only magcut lens is varied'


    def test_ell_values(self):
        assert self.general_cfg['nbl_WL_opt'] == 32, 'this is used as the reference binning, from which the cuts are made'
        assert self.general_cfg['ell_max_WL_opt'] == 5000, 'this is used as the reference binning, from which the cuts are made'
        assert self.general_cfg['n_probes'] == 2, 'The code can only accept 2 probes at the moment'

    def run_all_checks(self):
        k_txt_label, pk_txt_label = self.check_h_units()
        self.check_ell_cuts()
        self.check_BNT_transform()
        self.check_fsky()
        self.check_which_forecast()
        self.check_z_grid_ssc_integrands()
        self.check_nuisance()
        self.test_ell_values()
        return k_txt_label, pk_txt_label
