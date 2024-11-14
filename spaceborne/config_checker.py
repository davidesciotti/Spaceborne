import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.my_module as mm
import numpy as np


class SpaceborneConfigChecker:
    def __init__(self, cfg):
        self.cfg = cfg

    def check_h_units(self):
        if self.cfg['misc']['use_h_units']:
            return "hoverMpc", "Mpcoverh3"
        else:
            return "1overMpc", "Mpc3"

    def check_ell_cuts(self):
        if self.cfg['ell_cuts']['apply_ell_cuts']:
            assert self.cfg['BNT']['BNT_transform'], 'You should BNT transform if you want to apply ell cuts.'

    def check_BNT_transform(self):
        if self.cfg['BNT']['cov_BNT_transform']:
            assert not self.cfg['BNT']['cl_BNT_transform'], 'The BNT transform should be applied either to the Cls or to the covariance.'

    def check_fsky(self):
        fsky_check = cosmo_lib.deg2_to_fsky(self.cfg['covariance']['survey_area_deg2'])
        assert np.abs(mm.percent_diff(self.cfg['covariance']['fsky'],
                      fsky_check)) < 1e-5, 'Fsky does not match the survey area.'


    def check_cl_integral_convention(self):
        assert self.cfg['covariance']['cl_integral_convention'] in ['PySSC', 'Euclid', 'Euclid_KE_approximation'], \
            'cl_integral_convention must be "PySSC" or "Euclid" or "Euclid_KE_approximation"'

    def check_types(self):
        assert isinstance(self.cfg['covariance']['include_b2g'], bool), 'include_b2 must be a boolean'
        assert isinstance(self.cfg['covariance']['b2g_from_halomodel'],
                          bool), 'b2g_from_halomodel must be a boolean'
        assert isinstance(self.cfg['covariance']
                          ['use_KE_approximation'], bool), 'b2g_from_halomodel must be a boolean'
        assert isinstance(self.cfg['covariance']
                          ['load_precomputed_sigma2'], bool), 'b2g_from_halomodel must be a boolean'
        assert isinstance(self.cfg['nz']['normalize_shifted_nz'], bool), 'b2g_from_halomodel must be a boolean'
        assert isinstance(self.cfg['misc']['save_outputs_as_test_benchmarks_path'], (str, bool)
                          ), 'save_outputs_as_test_benchmarks_path must be either a path (str) or a boolean'

        if isinstance(self.cfg['misc']['save_outputs_as_test_benchmarks_path'], bool):
            assert not self.cfg['misc']['save_outputs_as_test_benchmarks_path'], 'if boolean, '\
                'save_outputs_as_test_benchmarks_path must be False'

    def run_all_checks(self):
        k_txt_label, pk_txt_label = self.check_h_units()
        self.check_ell_cuts()
        self.check_BNT_transform()
        self.check_cl_integral_convention()
        self.check_fsky()
        self.check_types()
        return k_txt_label, pk_txt_label
