import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.sb_lib as sl
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
            assert self.cfg['ell_cuts']['which_cuts'] == 'standard', 'Other types of cuts not finished to implement'

    def check_BNT_transform(self):
        if self.cfg['BNT']['cov_BNT_transform']:
            assert not self.cfg['BNT']['cl_BNT_transform'], 'The BNT transform should be applied either to the Cls or to the covariance.'

    def check_fsky(self):
        fsky_check = cosmo_lib.deg2_to_fsky(self.cfg['mask']['survey_area_deg2'])
        assert np.abs(sl.percent_diff(self.cfg['mask']['fsky'],
                      fsky_check)) < 1e-5, 'Fsky does not match the survey area.'

    def check_KE_approximation(self):

        if self.cfg['covariance']['use_KE_approximation'] and self.cfg['covariance']['SSC_code'] == 'Spaceborne':
            assert self.cfg['covariance']['which_sigma2_b'] not in [None, 'full_curved_sky'], \
                'to use the flat-sky sigma2_b, set "flat_sky" in the cfg file. Also, bear in mind that the flat-sky '\
                'approximation for sigma2_b is likely inappropriate for the large Euclid survey area'

        elif not self.cfg['covariance']['use_KE_approximation'] and self.cfg['covariance']['SSC_code'] == 'Spaceborne':
            assert self.cfg['covariance']['which_sigma2_b'] not in [None, 'flat_sky'], \
                'If you\'re not using the KE approximation, you should set "full_curved_sky", "from_input_mask or "polar_cap_on_the_fly"'

    def check_types(self):
        assert isinstance(self.cfg['covariance']['include_b2g'], bool), \
            'include_b2 must be a boolean'
        assert isinstance(self.cfg['covariance']['use_KE_approximation'], bool), \
            'use_KE_approximation must be a boolean'
        assert isinstance(self.cfg['covariance']['load_cached_sigma2_b'], bool), \
            'load_cached_sigma2_b must be a boolean'
        assert isinstance(self.cfg['nz']['normalize_shifted_nz'], bool), \
            'b2g_from_halomodel must be a boolean'

    def check_ell_binning(self):
        assert self.cfg['ell_binning']['nbl_WL_opt'] == 32, 'this is used as the reference binning, from which the cuts are made'
        assert self.cfg['ell_binning']['ell_max_WL_opt'] == 5000, 'this is used as the reference binning, from which the cuts are made'
        assert (self.cfg['ell_binning']['ell_max_WL'], self.cfg['ell_binning']['ell_max_GC']) == (5000, 3000) or (1500, 750), \
            'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'

    def check_misc(self):
        assert self.cfg['covariance']['n_probes'] == 2, 'The code can only accept 2 probes at the moment'

    def check_nz(self):
        assert np.all(np.array(self.cfg['nz']['ngal_sources']) > 0), 'ngal_sources values must be positive'
        assert np.all(np.array(self.cfg['nz']['ngal_lenses']) > 0), 'ngal_lenses values must be positive'
        assert np.all(self.cfg['nz']['dzWL'] == self.cfg['nz']['dzGC']), 'dzWL and dzGC shifts do not match'
        assert len(self.cfg['nz']['ngal_sources']) == len(self.cfg['nz']['ngal_lenses']) == \
            len(self.cfg['nz']['dzWL']) == len(self.cfg['nz']['dzGC'])
        assert isinstance(self.cfg['nz']['ngal_sources'], list), 'n_gal_shear must be a list'
        assert isinstance(self.cfg['nz']['ngal_lenses'], list), 'n_gal_clust must be a list'

    def check_cosmo(self):
        if 'logT' in self.cfg['cosmology']:
            assert self.cfg['cosmology']['logT'] == self.cfg['extra_parameters']['camb']['HMCode_logT_AGN'], (
                'Value mismatch for logT_AGN in the parameters definition')

    def check_cov(self):
        assert self.cfg['covariance']['ep_or_ed'] in ('EP', 'ED'), 'EP_or_ED must be either EP or ED'
        assert self.cfg['covariance']['triu_tril'] in ('triu', 'tril'), 'triu_tril must be either "triu" or "tril"'
        assert self.cfg['covariance']['row_col_major'] in (
            'row-major', 'col-major'), 'row_col_major must be either "row-major" or "col-major"'

    def run_all_checks(self):
        k_txt_label, pk_txt_label = self.check_h_units()
        self.check_ell_cuts()
        self.check_BNT_transform()
        self.check_KE_approximation()
        self.check_fsky()
        self.check_types()
        self.check_ell_binning()
        self.check_misc()
        self.check_nz()
        self.check_cosmo()
        return k_txt_label, pk_txt_label
