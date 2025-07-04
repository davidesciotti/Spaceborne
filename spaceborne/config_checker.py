import numpy as np

from spaceborne import cosmo_lib
from spaceborne import sb_lib as sl


class SpaceborneConfigChecker:
    def __init__(self, cfg: dict, zbins: int):
        self.cfg = cfg
        self.zbins = zbins

    def check_h_units(self) -> tuple[str, str]:
        if self.cfg['misc']['use_h_units']:
            return 'hoverMpc', 'Mpcoverh3'
        else:
            return '1overMpc', 'Mpc3'

    def check_ell_cuts(self) -> None:
        if self.cfg['ell_cuts']['apply_ell_cuts']:
            assert self.cfg['ell_cuts']['which_cuts'] == 'standard', (
                'Other types of cuts not finished to implement'
            )

    def check_BNT_transform(self) -> None:
        if self.cfg['BNT']['cov_BNT_transform']:
            assert not self.cfg['BNT']['cl_BNT_transform'], (
                'The BNT transform should be applied either to the Cls '
                'or to the covariance.'
            )
        if self.cfg['BNT']['cl_BNT_transform']:
            assert not self.cfg['BNT']['cov_BNT_transform'], (
                'The BNT transform should be applied either to the Cls '
                'or to the covariance.'
            )

        # def check_fsky(self) -> None:
        fsky_check = cosmo_lib.deg2_to_fsky(self.cfg['mask']['survey_area_deg2'])
        # assert np.abs(sl.percent_diff(self.cfg['mask']['fsky'], fsky_check)) < 1e-5, (
        #     'fsky does not match the survey area.'
        # )

    def check_KE_approximation(self) -> None:
        if (
            self.cfg['covariance']['use_KE_approximation']
            and self.cfg['covariance']['SSC_code'] == 'Spaceborne'
        ):
            assert self.cfg['covariance']['which_sigma2_b'] not in [
                None,
                'full_curved_sky',
            ], (
                'to use the flat-sky sigma2_b, set "flat_sky" in the cfg file. '
                'Also, bear in mind that the flat-sky '
                'approximation for sigma2_b is likely inappropriate for the large '
                'Euclid survey area'
            )

        elif (
            not self.cfg['covariance']['use_KE_approximation']
            and self.cfg['covariance']['SSC_code'] == 'Spaceborne'
        ):
            assert self.cfg['covariance']['which_sigma2_b'] not in [None, 'flat_sky'], (
                "If you're not using the KE approximation, you should set "
                '"full_curved_sky", '
                '"from_input_mask or "polar_cap_on_the_fly"'
            )

    def check_types(self) -> None:
        assert isinstance(self.cfg['covariance']['include_b2g'], bool), (
            'include_b2 must be a boolean'
        )
        assert isinstance(self.cfg['covariance']['use_KE_approximation'], bool), (
            'use_KE_approximation must be a boolean'
        )
        assert isinstance(self.cfg['covariance']['load_cached_sigma2_b'], bool), (
            'load_cached_sigma2_b must be a boolean'
        )
        assert isinstance(self.cfg['nz']['normalize_shifted_nz'], bool), (
            'b2g_from_halomodel must be a boolean'
        )

    def check_ell_binning(self) -> None:
        # assert self.cfg['ell_binning']['nbl_WL_opt'] == 32, (
        #     'this is used as the reference binning, from which the cuts are made'
        # )
        # assert self.cfg['ell_binning']['ell_max_WL_opt'] == 5000, (
        #     'this is used as the reference binning, from which the cuts are made'
        # )
        # assert (
        #     self.cfg['ell_binning']['ell_max_WL'],
        #     self.cfg['ell_binning']['ell_max_GC'],
        # ) == (5000, 3000) or (1500, 750), (
        #     'ell_max_WL and ell_max_GC must be either (5000, 3000) or (1500, 750)'
        # )
        pass

    def check_misc(self) -> None:
        assert self.cfg['covariance']['n_probes'] == 2, (
            'The code can only accept 2 probes at the moment'
        )

    def check_nz(self) -> None:
        assert np.all(np.array(self.cfg['nz']['ngal_sources']) > 0), (
            'ngal_sources values must be positive'
        )
        assert np.all(np.array(self.cfg['nz']['ngal_lenses']) > 0), (
            'ngal_lenses values must be positive'
        )
        assert np.all(self.cfg['nz']['dzWL'] == self.cfg['nz']['dzGC']), (
            'dzWL and dzGC shifts do not match'
        )
        assert len(self.cfg['nz']['ngal_sources']) == len(self.cfg['nz']['ngal_lenses'])
        assert isinstance(self.cfg['nz']['ngal_sources'], list), (
            'n_gal_shear must be a list'
        )
        assert isinstance(self.cfg['nz']['ngal_lenses'], list), (
            'n_gal_clust must be a list'
        )
        if self.cfg['nz']['shift_nz']:
            assert len(self.cfg['nz']['dzWL']) == len(self.cfg['nz']['ngal_sources'])
            assert len(self.cfg['nz']['dzWL']) == len(self.cfg['nz']['ngal_lenses'])

        assert isinstance(self.cfg['nz']['ngal_sources'], list), (
            'n_gal_shear must be a list'
        )
        assert isinstance(self.cfg['nz']['ngal_lenses'], list), (
            'n_gal_clust must be a list'
        )

    def check_lists(self) -> None:
        assert len(self.cfg['C_ell']['galaxy_bias_fit_coeff']) == 4, (
            'galaxy_bias_fit_coeff must be of length 4'
        )
        assert len(self.cfg['C_ell']['magnification_bias_fit_coeff']) == 4, (
            'magnification_bias_fit_coeff must be of length 4'
        )

        assert len(self.cfg['C_ell']['mult_shear_bias']) == self.zbins, (
            f'mult_shear_bias must be of length zbins = {self.zbins}'
        )
        assert len(self.cfg['nz']['ngal_sources']) == self.zbins, (
            f'ngal_sources must be of length zbins = {self.zbins}'
        )
        assert len(self.cfg['nz']['ngal_lenses']) == self.zbins, (
            f'ngal_lenses must be of length zbins = {self.zbins}'
        )
        if self.cfg['nz']['shift_nz']:
            assert len(self.cfg['nz']['dzWL']) == self.zbins, (
                f'dzWL must be of length zbins = {self.zbins}'
            )
            assert len(self.cfg['nz']['dzGC']) == self.zbins, (
                f'dzGC must be of length zbins = {self.zbins}'
            )
        assert len(self.cfg['covariance']['sigma_eps_i']) == self.zbins, (
            f'sigma_eps_i must be of length zbins = {self.zbins}'
        )

    def check_cosmo(self) -> None:
        if 'logT' in self.cfg['cosmology']:
            assert (
                self.cfg['cosmology']['logT']
                == self.cfg['extra_parameters']['camb']['HMCode_logT_AGN']
            ), 'Value mismatch for logT_AGN in the parameters definition'

    def check_cov(self) -> None:
        assert self.cfg['covariance']['triu_tril'] in ('triu', 'tril'), (
            'triu_tril must be either "triu" or "tril"'
        )
        assert self.cfg['covariance']['probe_ordering'] == [
            ['L', 'L'],
            ['G', 'L'],
            ['G', 'G'],
        ], 'Other probe orderings not tested at the moment'
        assert self.cfg['covariance']['row_col_major'] in ('row-major', 'col-major'), (
            'row_col_major must be either "row-major" or "col-major"'
        )

    def check_nmt(self) -> None:
        if self.cfg['covariance']['coupled_cov'] and self.cfg['covariance']['G']:
            assert (
                self.cfg['namaster']['use_namaster']
                or self.cfg['sample_covariance']['compute_sample_cov']
            ), (
                'if the coupled Gaussian covariance is requested either '
                'cfg["namaster"]["use_namaster"] or '
                'cfg["sample_covariance"]["compute_sample_cov"] must be True'
            )
        if self.cfg['namaster']['use_namaster']:
            assert self.cfg['ell_binning']['binning_type'] != 'ref_cut', (
                'ref_cut case incompatible with nmt for the moment. '
                'Please use a different binning type.'
            )

    def run_all_checks(self) -> None:
        self.check_ell_cuts()
        self.check_nmt()
        self.check_BNT_transform()
        self.check_KE_approximation()
        self.check_lists()
        # self.check_fsky()
        self.check_types()
        self.check_ell_binning()
        self.check_misc()
        self.check_nz()
        self.check_cosmo()
        self.check_cov()
