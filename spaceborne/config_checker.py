import collections.abc

import numpy as np

from spaceborne import cosmo_lib


class SpaceborneConfigChecker:
    def __init__(self, cfg: dict):
        self.cfg = cfg

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
        # Cosmology
        for par, val in self.cfg['cosmology'].items():
            assert isinstance(val, float), f'Parameter {par} must be a float'

        # camb extra params
        assert isinstance(self.cfg.get('extra_parameters'), dict), (
            "Section 'extra_parameters' must be a dictionary"
        )
        assert isinstance(self.cfg['extra_parameters'].get('camb'), dict), (
            'extra_parameters: camb must be a dictionary'
        )
        camb_params = self.cfg['extra_parameters']['camb']
        assert isinstance(camb_params.get('halofit_version'), str), (
            'extra_parameters: camb: halofit_version must be a string'
        )
        assert isinstance(camb_params.get('kmax'), (float, int)), (
            'extra_parameters: camb: kmax must be a float or an int'
        )
        assert isinstance(camb_params.get('HMCode_logT_AGN'), float), (
            'extra_parameters: camb: HMCode_logT_AGN must be a float'
        )
        assert isinstance(camb_params.get('num_massive_neutrinos'), int), (
            'extra_parameters: camb: num_massive_neutrinos must be an int'
        )
        assert isinstance(camb_params.get('dark_energy_model'), str), (
            'extra_parameters: camb: dark_energy_model must be a string'
        )

        # IA
        for par, val in self.cfg['intrinsic_alignment'].items():
            if par != 'lumin_ratio_filename':
                assert isinstance(val, float), f'Parameter {par} must be a float'
            elif par == 'lumin_ratio_filename':
                assert val is None or isinstance(val, str), (
                    f'Parameter {par} must be a str or None'
                )

        # halo_model
        for par, val in self.cfg['halo_model'].items():
            assert isinstance(val, str), f'Parameter {par} must be a str'

        # probe selection
        for par, val in self.cfg['probe_selection'].items():
            assert isinstance(val, bool), f'Parameter {par} must be a bool'

        # C_ell
        assert isinstance(self.cfg.get('C_ell'), dict), (
            "Section 'C_ell' must be a dictionary"
        )
        c_ell_cfg = self.cfg['C_ell']
        assert isinstance(c_ell_cfg.get('use_input_cls'), bool), (
            'C_ell: use_input_cls must be a boolean'
        )
        assert isinstance(c_ell_cfg.get('cl_LL_path'), str), (
            'C_ell: cl_LL_path must be a string'
        )
        assert isinstance(c_ell_cfg.get('cl_GL_path'), str), (
            'C_ell: cl_GL_path must be a string'
        )
        assert isinstance(c_ell_cfg.get('cl_GG_path'), str), (
            'C_ell: cl_GG_path must be a string'
        )
        assert isinstance(c_ell_cfg.get('which_gal_bias'), str), (
            'C_ell: which_gal_bias must be a string'
        )
        assert isinstance(c_ell_cfg.get('which_mag_bias'), str), (
            'C_ell: which_mag_bias must be a string'
        )

        assert isinstance(c_ell_cfg.get('galaxy_bias_fit_coeff'), list), (
            'C_ell: galaxy_bias_fit_coeff must be a list'
        )
        assert all(isinstance(x, float) for x in c_ell_cfg['galaxy_bias_fit_coeff']), (
            'C_ell: All elements in galaxy_bias_fit_coeff must be floats'
        )

        assert isinstance(c_ell_cfg.get('magnification_bias_fit_coeff'), list), (
            'C_ell: magnification_bias_fit_coeff must be a list'
        )
        assert all(
            isinstance(x, float) for x in c_ell_cfg['magnification_bias_fit_coeff']
        ), 'C_ell: All elements in magnification_bias_fit_coeff must be floats'

        assert isinstance(c_ell_cfg.get('gal_bias_table_filename'), str), (
            'C_ell: gal_bias_table_filename must be a string'
        )
        assert isinstance(c_ell_cfg.get('mag_bias_table_filename'), str), (
            'C_ell: mag_bias_table_filename must be a string'
        )

        assert isinstance(c_ell_cfg.get('mult_shear_bias'), list), (
            'C_ell: mult_shear_bias must be a list'
        )
        assert all(isinstance(x, float) for x in c_ell_cfg['mult_shear_bias']), (
            'C_ell: All elements in mult_shear_bias must be floats'
        )

        assert isinstance(c_ell_cfg.get('has_rsd'), bool), (
            'C_ell: has_rsd must be a boolean'
        )
        assert isinstance(c_ell_cfg.get('has_IA'), bool), (
            'C_ell: has_IA must be a boolean'
        )
        assert isinstance(c_ell_cfg.get('has_magnification_bias'), bool), (
            'C_ell: has_magnification_bias must be a boolean'
        )

        assert isinstance(c_ell_cfg.get('cl_CCL_kwargs'), dict), (
            'C_ell: cl_CCL_kwargs must be a dictionary'
        )
        ccl_kwargs = c_ell_cfg['cl_CCL_kwargs']
        assert isinstance(ccl_kwargs.get('l_limber'), int), (
            'C_ell: cl_CCL_kwargs: l_limber must be an int'
        )
        assert isinstance(ccl_kwargs.get('limber_integration_method'), str), (
            'C_ell: cl_CCL_kwargs: limber_integration_method must be a string'
        )
        assert isinstance(ccl_kwargs.get('non_limber_integration_method'), str), (
            'C_ell: cl_CCL_kwargs: non_limber_integration_method must be a string'
        )

        # nz
        assert isinstance(self.cfg.get('nz'), dict), "Section 'nz' must be a dictionary"
        nz_cfg = self.cfg['nz']
        assert isinstance(nz_cfg.get('nz_sources_filename'), str), (
            'nz: nz_sources_filename must be a string'
        )
        assert isinstance(nz_cfg.get('nz_lenses_filename'), str), (
            'nz: nz_lenses_filename must be a string'
        )

        assert isinstance(nz_cfg.get('ngal_sources'), list), (
            'nz: ngal_sources must be a list'
        )
        assert all(isinstance(x, float) for x in nz_cfg['ngal_sources']), (
            'nz: All elements in ngal_sources must be floats'
        )

        assert isinstance(nz_cfg.get('ngal_lenses'), list), (
            'nz: ngal_lenses must be a list'
        )
        assert all(isinstance(x, float) for x in nz_cfg['ngal_lenses']), (
            'nz: All elements in ngal_lenses must be floats'
        )

        assert isinstance(nz_cfg.get('shift_nz'), bool), (
            'nz: shift_nz must be a boolean'
        )

        assert isinstance(nz_cfg.get('dzWL'), list), 'nz: dzWL must be a list'
        assert all(isinstance(x, float) for x in nz_cfg['dzWL']), (
            'nz: All elements in dzWL must be floats'
        )

        assert isinstance(nz_cfg.get('dzGC'), list), 'nz: dzGC must be a list'
        assert all(isinstance(x, float) for x in nz_cfg['dzGC']), (
            'nz: All elements in dzGC must be floats'
        )

        assert isinstance(nz_cfg.get('normalize_shifted_nz'), bool), (
            'nz: normalize_shifted_nz must be a boolean'
        )
        assert isinstance(nz_cfg.get('clip_zmin'), float), (
            'nz: clip_zmin must be a float'
        )
        assert isinstance(nz_cfg.get('clip_zmax'), float), (
            'nz: clip_zmax must be a float'
        )

        # Mask
        assert isinstance(self.cfg.get('mask'), dict), (
            "Section 'mask' must be a dictionary"
        )
        mask_cfg = self.cfg['mask']
        assert isinstance(mask_cfg.get('load_mask'), bool), (
            'mask: load_mask must be a boolean'
        )
        assert isinstance(mask_cfg.get('mask_path'), str), (
            'mask: mask_path must be a string'
        )
        assert isinstance(mask_cfg.get('generate_polar_cap'), bool), (
            'mask: generate_polar_cap must be a boolean'
        )
        assert isinstance(mask_cfg.get('nside'), (int, type(None))), (
            'mask: nside must be an int or None'
        )
        assert isinstance(mask_cfg.get('survey_area_deg2'), int), (
            'mask: survey_area_deg2 must be an int'
        )
        assert isinstance(mask_cfg.get('apodize'), bool), (
            'mask: apodize must be a boolean'
        )
        assert isinstance(mask_cfg.get('aposize'), float), (
            'mask: aposize must be a float'
        )

        # Namaster
        assert isinstance(self.cfg.get('namaster'), dict), (
            "Section 'namaster' must be a dictionary"
        )
        namaster_cfg = self.cfg['namaster']
        assert isinstance(namaster_cfg.get('use_namaster'), bool), (
            'namaster: use_namaster must be a boolean'
        )
        assert isinstance(namaster_cfg.get('spin0'), bool), (
            'namaster: spin0 must be a boolean'
        )
        assert isinstance(namaster_cfg.get('use_INKA'), bool), (
            'namaster: use_INKA must be a boolean'
        )
        assert isinstance(namaster_cfg.get('workspace_path'), str), (
            'namaster: workspace_path must be a string'
        )

        # Sample Covariance
        assert isinstance(self.cfg.get('sample_covariance'), dict), (
            "Section 'sample_covariance' must be a dictionary"
        )
        sample_cov_cfg = self.cfg['sample_covariance']
        assert isinstance(sample_cov_cfg.get('compute_sample_cov'), bool), (
            'sample_covariance: compute_sample_cov must be a boolean'
        )
        assert isinstance(sample_cov_cfg.get('which_cls'), str), (
            'sample_covariance: which_cls must be a string'
        )
        assert isinstance(sample_cov_cfg.get('nreal'), int), (
            'sample_covariance: nreal must be an int'
        )
        assert isinstance(sample_cov_cfg.get('fix_seed'), bool), (
            'sample_covariance: fix_seed must be a boolean'
        )

        # OneCovariance
        assert isinstance(self.cfg.get('OneCovariance'), dict), (
            "Section 'OneCovariance' must be a dictionary"
        )
        oc_cfg = self.cfg['OneCovariance']
        assert isinstance(oc_cfg.get('path_to_oc_executable'), str), (
            'OneCovariance: path_to_oc_executable must be a string'
        )
        assert isinstance(oc_cfg.get('consistency_checks'), bool), (
            'OneCovariance: consistency_checks must be a boolean'
        )
        assert isinstance(oc_cfg.get('oc_output_filename'), str), (
            'OneCovariance: oc_output_filename must be a string'
        )

        # Ell Binning
        assert isinstance(self.cfg.get('ell_binning'), dict), (
            "Section 'ell_binning' must be a dictionary"
        )
        ell_bin_cfg = self.cfg['ell_binning']
        assert isinstance(ell_bin_cfg.get('binning_type'), str), (
            'ell_binning: binning_type must be a string'
        )
        assert isinstance(ell_bin_cfg.get('ell_min_WL'), int), (
            'ell_binning: ell_min_WL must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_max_WL'), int), (
            'ell_binning: ell_max_WL must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_bins_WL'), int), (
            'ell_binning: ell_bins_WL must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_min_GC'), int), (
            'ell_binning: ell_min_GC must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_max_GC'), int), (
            'ell_binning: ell_max_GC must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_bins_GC'), int), (
            'ell_binning: ell_bins_GC must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_min_ref'), int), (
            'ell_binning: ell_min_ref must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_max_ref'), int), (
            'ell_binning: ell_max_ref must be an int'
        )
        assert isinstance(ell_bin_cfg.get('ell_bins_ref'), int), (
            'ell_binning: ell_bins_ref must be an int'
        )

        # BNT
        assert isinstance(self.cfg.get('BNT'), dict), (
            "Section 'BNT' must be a dictionary"
        )
        bnt_cfg = self.cfg['BNT']
        assert isinstance(bnt_cfg.get('cl_BNT_transform'), bool), (
            'BNT: cl_BNT_transform must be a boolean'
        )
        assert isinstance(bnt_cfg.get('cov_BNT_transform'), bool), (
            'BNT: cov_BNT_transform must be a boolean'
        )

        # Covariance
        assert isinstance(self.cfg.get('covariance'), dict), (
            "Section 'covariance' must be a dictionary"
        )
        cov_cfg = self.cfg['covariance']
        assert isinstance(cov_cfg.get('G'), bool), 'covariance: G must be a boolean'
        assert isinstance(cov_cfg.get('SSC'), bool), 'covariance: SSC must be a boolean'
        assert isinstance(cov_cfg.get('cNG'), bool), 'covariance: cNG must be a boolean'
        assert isinstance(cov_cfg.get('coupled_cov'), bool), (
            'covariance: coupled_cov must be a boolean'
        )
        assert isinstance(cov_cfg.get('triu_tril'), str), (
            'covariance: triu_tril must be a string'
        )
        assert isinstance(cov_cfg.get('row_col_major'), str), (
            'covariance: row_col_major must be a string'
        )
        assert isinstance(cov_cfg.get('covariance_ordering_2D'), str), (
            'covariance: covariance_ordering_2D must be a string'
        )
        assert isinstance(cov_cfg.get('save_full_cov'), bool), (
            'covariance: save_full_cov must be a boolean'
        )
        assert isinstance(cov_cfg.get('split_gaussian_cov'), bool), (
            'covariance: split_gaussian_cov must be a boolean'
        )

        assert isinstance(cov_cfg.get('sigma_eps_i'), list), (
            'covariance: sigma_eps_i must be a list'
        )
        assert all(isinstance(x, float) for x in cov_cfg['sigma_eps_i']), (
            'covariance: All elements in sigma_eps_i must be floats'
        )

        assert isinstance(cov_cfg.get('no_sampling_noise'), bool), (
            'covariance: no_sampling_noise must be a boolean'
        )
        assert isinstance(cov_cfg.get('which_pk_responses'), str), (
            'covariance: which_pk_responses must be a string'
        )
        assert isinstance(cov_cfg.get('which_b1g_in_resp'), str), (
            'covariance: which_b1g_in_resp must be a string'
        )
        assert isinstance(cov_cfg.get('include_b2g'), bool), (
            'covariance: include_b2g must be a boolean'
        )
        assert isinstance(cov_cfg.get('include_terasawa_terms'), bool), (
            'covariance: include_terasawa_terms must be a boolean'
        )
        assert isinstance(cov_cfg.get('sigma2_b_int_method'), str), (
            'covariance: sigma2_b_int_method must be a string'
        )
        assert isinstance(cov_cfg.get('load_cached_sigma2_b'), bool), (
            'covariance: load_cached_sigma2_b must be a boolean'
        )
        assert isinstance(cov_cfg.get('log10_k_min'), float), (
            'covariance: log10_k_min must be a float'
        )
        assert isinstance(cov_cfg.get('log10_k_max'), float), (
            'covariance: log10_k_max must be a float'
        )
        assert isinstance(cov_cfg.get('k_steps'), int), (
            'covariance: k_steps must be an int'
        )
        assert isinstance(cov_cfg.get('z_min'), float), (
            'covariance: z_min must be a float'
        )
        assert isinstance(cov_cfg.get('z_max'), float), (
            'covariance: z_max must be a float'
        )
        assert isinstance(cov_cfg.get('z_steps'), int), (
            'covariance: z_steps must be an int'
        )
        assert isinstance(cov_cfg.get('z_steps_trisp'), int), (
            'covariance: z_steps_trisp must be an int'
        )
        assert isinstance(cov_cfg.get('use_KE_approximation'), bool), (
            'covariance: use_KE_approximation must be a boolean'
        )
        assert isinstance(cov_cfg.get('cov_filename'), str), (
            'covariance: cov_filename must be a string'
        )

        # cov_real_space
        assert isinstance(self.cfg.get('cov_real_space'), dict), (
            "Section 'cov_real_space' must be a dictionary"
        )
        real_space_cfg = self.cfg['cov_real_space']
        assert isinstance(real_space_cfg.get('do_real_space'), bool), (
            'cov_real_space: do_real_space must be a boolean'
        )
        assert isinstance(real_space_cfg.get('theta_min_arcmin'), (float, int)), (
            'cov_real_space: theta_min_arcmin must be a float or an int'
        )
        assert isinstance(real_space_cfg.get('theta_max_arcmin'), (float, int)), (
            'cov_real_space: theta_max_arcmin must be a float or an int'
        )
        assert isinstance(real_space_cfg.get('theta_bins'), int), (
            'cov_real_space: theta_bins must be an int'
        )

        # PyCCL
        pyccl_cfg = self.cfg['PyCCL']
        assert isinstance(pyccl_cfg, dict), (
            "Section 'PyCCL' must be a dictionary"
        )
        assert isinstance(pyccl_cfg.get('cov_integration_method'), str), (
            'PyCCL: cov_integration_method must be a string'
        )
        assert isinstance(pyccl_cfg.get('load_cached_tkka'), bool), (
            'PyCCL: load_cached_tkka must be a boolean'
        )
        assert isinstance(pyccl_cfg.get('use_default_k_a_grids'), bool), (
            'PyCCL: use_default_k_a_grids must be a boolean'
        )
        assert isinstance(pyccl_cfg.get('n_samples_wf'), int), (
            'PyCCL: n_samples_wf must be an int'
        )

        assert isinstance(pyccl_cfg.get('spline_params'), (dict, type(None))), (
            'PyCCL: spline_params must be a dictionary or None'
        )
        if isinstance(pyccl_cfg.get('spline_params'), dict):
            spline_params = pyccl_cfg['spline_params']
            assert isinstance(spline_params.get('A_SPLINE_NA_PK'), int), (
                'PyCCL: spline_params: A_SPLINE_NA_PK must be an int'
            )
            assert isinstance(spline_params.get('K_MAX_SPLINE'), int), (
                'PyCCL: spline_params: K_MAX_SPLINE must be an int'
            )

        assert isinstance(pyccl_cfg.get('gsl_params'), (dict, type(None))), (
            'PyCCL: gsl_params must be a dictionary or None'
        )

        # precision
        precision_cfg = self.cfg['precision']
        assert isinstance(precision_cfg, dict), (
            "Section 'precision' must be a dictionary"
        )
        assert isinstance(precision_cfg.get('n_iter_nmt'), (int, type(None))), (
            'precision: n_iter_nmt must be an int or None'
        )
        assert isinstance(precision_cfg.get('n_sub'), int), (
            'precision: n_sub must be an int'
        )
        assert isinstance(precision_cfg.get('n_bisec_max'), int), (
            'precision: n_bisec_max must be an int'
        )
        assert isinstance(precision_cfg.get('rel_acc'), float), (
            'precision: rel_acc must be a float'
        )
        assert isinstance(precision_cfg.get('boost_bessel'), bool), (
            'precision: boost_bessel must be a boolean'
        )
        assert isinstance(precision_cfg.get('verbose'), bool), (
            'precision: verbose must be a boolean'
        )

        assert isinstance(precision_cfg.get('ell_min_rs'), int), (
            'precision: ell_min_rs must be an int'
        )
        assert isinstance(precision_cfg.get('ell_max_rs'), int), (
            'precision: ell_max_rs must be an int'
        )
        assert isinstance(precision_cfg.get('ell_bins_rs'), int), (
            'precision: ell_bins_rs must be an int'
        )
        assert isinstance(precision_cfg.get('theta_bins_fine'), int), (
            'precision: theta_bins_fine must be an int'
        )
        assert isinstance(precision_cfg.get('cov_rs_int_method'), str), (
            'precision: cov_rs_int_method must be a string'
        )

        # misc
        misc_cfg = self.cfg['misc']
        assert isinstance(misc_cfg, dict), "Section 'misc' must be a dictionary"
        assert isinstance(misc_cfg.get('num_threads'), int), (
            'misc: num_threads must be an int'
        )
        assert isinstance(misc_cfg.get('levin_batch_size'), int), (
            'misc: levin_batch_size must be an int'
        )
        assert isinstance(misc_cfg.get('test_numpy_inversion'), bool), (
            'misc: test_numpy_inversion must be a boolean'
        )
        assert isinstance(misc_cfg.get('test_condition_number'), bool), (
            'misc: test_condition_number must be a boolean'
        )
        assert isinstance(misc_cfg.get('test_cholesky_decomposition'), bool), (
            'misc: test_cholesky_decomposition must be a boolean'
        )
        assert isinstance(misc_cfg.get('test_symmetry'), bool), (
            'misc: test_symmetry must be a boolean'
        )
        assert isinstance(misc_cfg.get('output_path'), str), (
            'misc: output_path must be a string'
        )
        assert isinstance(misc_cfg.get('save_figs'), bool), (
            'misc: save_figs must be a boolean'
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

        if self.cfg['nz']['shift_nz']:
            assert len(self.cfg['nz']['dzWL']) == len(self.cfg['nz']['ngal_sources'])
            assert len(self.cfg['nz']['dzWL']) == len(self.cfg['nz']['ngal_lenses'])

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

    def run_all_checks(self) -> None:
        self.check_ell_cuts()
        self.check_nmt()
        self.check_BNT_transform()
        self.check_KE_approximation()
        # self.check_fsky()
        self.check_types()
        self.check_ell_binning()
        self.check_misc()
        self.check_nz()
        self.check_cosmo()
        self.check_cov()
