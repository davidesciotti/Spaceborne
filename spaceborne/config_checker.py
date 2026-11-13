import os
import warnings

import numpy as np


class SpaceborneConfigChecker:
    def __init__(self, cfg: dict, zbins: int):
        self.cfg = cfg
        self.zbins = zbins

    def check_BNT_transform(self) -> None:
        if self.cfg['covariance']['BNT_transform']:
            assert self.cfg['probe_selection']['space'] == 'harmonic', (
                'The BNT transform can only be applied in harmonic space.'
            )

        # def check_fsky(self) -> None:
        # _fsky_check = cosmo_lib.deg2_to_fsky(self.cfg['mask']['survey_area_deg2'])
        # assert np.abs(sl.percent_diff(self.cfg['mask']['fsky'], fsky_check)) < 1e-5, (
        #     'fsky does not match the survey area.'
        # )

    def check_types(self) -> None:
        # Cosmology
        cosmo_cfg = self.cfg['cosmology']
        required_cosmo_pars = [
            'Omega_m0', 'Omega_b0', 'Omega_k0', 'w0', 'wa', 'h', 'ns', 'sigma8', 'm_nu',
            'Neff',
        ]  # fmt: skip
        missing = [par for par in required_cosmo_pars if par not in cosmo_cfg]
        assert not missing, f'cosmology: missing parameters {missing}'
        assert 'ODE' not in cosmo_cfg, (
            'cosmology: ODE is not an input, it is set by closure'
        )
        for par, val in cosmo_cfg.items():
            assert isinstance(val, float), f'Parameter {par} must be a float'

        # camb extra params
        assert isinstance(self.cfg['extra_parameters'], dict), (
            "Section 'extra_parameters' must be a dictionary"
        )
        assert isinstance(self.cfg['extra_parameters']['camb'], dict), (
            'extra_parameters: camb must be a dictionary'
        )
        camb_params = self.cfg['extra_parameters']['camb']

        # Check required fields exist
        required_camb_fields = [
            'halofit_version',
            'kmax',
            'HMCode_logT_AGN',
            'num_massive_neutrinos',
            'dark_energy_model',
        ]
        for field in required_camb_fields:
            assert field in camb_params, f'extra_parameters: camb: {field} is required'

        assert isinstance(camb_params['halofit_version'], str), (
            'extra_parameters: camb: halofit_version must be a string'
        )
        assert isinstance(camb_params['kmax'], (float, int)), (
            'extra_parameters: camb: kmax must be a float or an int'
        )
        assert isinstance(camb_params['HMCode_logT_AGN'], float), (
            'extra_parameters: camb: HMCode_logT_AGN must be a float'
        )
        assert isinstance(camb_params['num_massive_neutrinos'], int), (
            'extra_parameters: camb: num_massive_neutrinos must be an int'
        )
        assert camb_params['num_massive_neutrinos'] in (1, 3), (
            'extra_parameters: camb: num_massive_neutrinos must be 1 (single) '
            'or 3 (equal mass split)'
        )
        assert isinstance(camb_params['dark_energy_model'], str), (
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
        _cfgps = self.cfg['probe_selection']
        assert isinstance(_cfgps['space'], str), (
            f'probe_selection.space must be a str, got {_cfgps["space"]}'
        )
        assert isinstance(_cfgps['cross_cov'], bool), (
            f'probe_selection.cross_cov must be a bool, got {_cfgps["cross_cov"]}'
        )

        if _cfgps['space'] == 'harmonic':
            assert isinstance(_cfgps['LL'], bool), (
                f'probe_selection.LL must be a bool, got {_cfgps["LL"]}'
            )
            assert isinstance(_cfgps['GL'], bool), (
                f'probe_selection.GL must be a bool, got {_cfgps["GL"]}'
            )
            assert isinstance(_cfgps['GG'], bool), (
                f'probe_selection.GG must be a bool, got {_cfgps["GG"]}'
            )

        if _cfgps['space'] == 'real':
            assert isinstance(_cfgps['xip'], bool), (
                f'probe_selection.xip must be a bool, got {_cfgps["xip"]}'
            )
            assert isinstance(_cfgps['xim'], bool), (
                f'probe_selection.xim must be a bool, got {_cfgps["xim"]}'
            )
            assert isinstance(_cfgps['gt'], bool), (
                f'probe_selection.gt must be a bool, got {_cfgps["gt"]}'
            )
            assert isinstance(_cfgps['w'], bool), (
                f'probe_selection.w must be a bool, got {_cfgps["w"]}'
            )

        if _cfgps['space'] == 'cosebis':
            assert isinstance(_cfgps['En'], bool), (
                f'probe_selection.En must be a bool, got {_cfgps["En"]}'
            )
            assert isinstance(_cfgps['Bn'], bool), (
                f'probe_selection.Bn must be a bool, got {_cfgps["Bn"]}'
            )
            assert isinstance(_cfgps['Psigl'], bool), (
                f'probe_selection.Psigl must be a bool, got {_cfgps["Psigl"]}'
            )
            assert isinstance(_cfgps['Psigg'], bool), (
                f'probe_selection.Psigg must be a bool, got {_cfgps["Psigg"]}'
            )

        # C_ell
        _cfgcl = self.cfg['C_ell']
        assert isinstance(_cfgcl, dict), "Section 'C_ell' must be a dictionary"
        assert isinstance(_cfgcl['use_input_cls'], bool), (
            'C_ell: use_input_cls must be a boolean'
        )
        assert isinstance(_cfgcl['cl_LL_filename'], str), (
            'C_ell: cl_LL_filename must be a string'
        )
        assert isinstance(_cfgcl['cl_GL_filename'], str), (
            'C_ell: cl_GL_filename must be a string'
        )
        assert isinstance(_cfgcl['cl_GG_filename'], str), (
            'C_ell: cl_GG_filename must be a string'
        )
        assert isinstance(_cfgcl['which_gal_bias'], str), (
            'C_ell: which_gal_bias must be a string'
        )
        assert isinstance(_cfgcl['which_mag_bias'], str), (
            'C_ell: which_mag_bias must be a string'
        )
        assert isinstance(_cfgcl['gal_bias_fit_coeff'], list), (
            'C_ell: gal_bias_fit_coeff must be a list'
        )
        assert all(isinstance(x, float) for x in _cfgcl['gal_bias_fit_coeff']), (
            'C_ell: All elements in gal_bias_fit_coeff must be floats'
        )
        assert isinstance(_cfgcl['mag_bias_fit_coeff'], list), (
            'C_ell: mag_bias_fit_coeff must be a list'
        )
        assert all(isinstance(x, float) for x in _cfgcl['mag_bias_fit_coeff']), (
            'C_ell: All elements in mag_bias_fit_coeff must be floats'
        )

        assert isinstance(_cfgcl['gal_bias_table_filename'], str), (
            'C_ell: gal_bias_table_filename must be a string'
        )
        assert isinstance(_cfgcl['mag_bias_table_filename'], str), (
            'C_ell: mag_bias_table_filename must be a string'
        )

        assert isinstance(_cfgcl['mult_shear_bias'], list), (
            'C_ell: mult_shear_bias must be a list'
        )
        assert all(isinstance(x, float) for x in _cfgcl['mult_shear_bias']), (
            'C_ell: All elements in mult_shear_bias must be floats'
        )

        assert isinstance(_cfgcl['has_rsd'], bool), 'C_ell: has_rsd must be a boolean'
        assert isinstance(_cfgcl['has_IA'], bool), 'C_ell: has_IA must be a boolean'
        assert isinstance(_cfgcl['has_mag_bias'], bool), (
            'C_ell: has_mag_bias must be a boolean'
        )

        assert isinstance(_cfgcl['cl_CCL_kwargs'], dict), (
            'C_ell: cl_CCL_kwargs must be a dictionary'
        )
        ccl_kwargs = _cfgcl['cl_CCL_kwargs']
        assert isinstance(ccl_kwargs['l_limber'], int), (
            'C_ell: cl_CCL_kwargs: l_limber must be an int'
        )
        assert isinstance(ccl_kwargs['limber_integration_method'], str), (
            'C_ell: cl_CCL_kwargs: limber_integration_method must be a string'
        )
        assert isinstance(ccl_kwargs['non_limber_integration_method'], str), (
            'C_ell: cl_CCL_kwargs: non_limber_integration_method must be a string'
        )

        # nz
        assert isinstance(self.cfg['nz'], dict), "Section 'nz' must be a dictionary"
        _cfgnz = self.cfg['nz']
        assert isinstance(_cfgnz['nz_sources_filename'], str), (
            'nz: nz_sources_filename must be a string'
        )
        assert isinstance(_cfgnz['nz_lenses_filename'], str), (
            'nz: nz_lenses_filename must be a string'
        )

        assert isinstance(_cfgnz['ngal_sources'], list), (
            'nz: ngal_sources must be a list'
        )
        assert isinstance(_cfgnz['ngal_lenses'], list), 'nz: ngal_lenses must be a list'
        assert all(isinstance(x, float) for x in _cfgnz['ngal_sources']), (
            'nz: All elements in ngal_sources must be floats'
        )
        assert all(isinstance(x, float) for x in _cfgnz['ngal_lenses']), (
            'nz: All elements in ngal_lenses must be floats'
        )
        assert isinstance(_cfgnz['shift_nz'], bool), 'nz: shift_nz must be a boolean'
        assert isinstance(_cfgnz['normalize_nz'], bool), (
            'nz: normalize_nz must be a boolean'
        )
        assert isinstance(_cfgnz['smooth_nz'], bool), 'nz: smooth_nz must be a boolean'
        assert isinstance(_cfgnz['sigma_smoothing'], (int, float)), (
            'nz: sigma_smoothing must be a float or an int'
        )

        if _cfgnz['shift_nz']:
            assert isinstance(_cfgnz['dz_sources'], list), 'nz: dz_sources must be a list'
            assert isinstance(_cfgnz['dz_lenses'], list), 'nz: dz_lenses must be a list'
            assert all(isinstance(x, float) for x in _cfgnz['dz_sources']), (
                'nz: All elements in dz_sources must be floats'
            )
            assert all(isinstance(x, float) for x in _cfgnz['dz_lenses']), (
                'nz: All elements in dz_lenses must be floats'
            )

        # Mask
        assert isinstance(self.cfg['mask'], dict), "Section 'mask' must be a dictionary"
        _cfgmask = self.cfg['mask']
        for probe in ['LL', 'GG']:
            assert isinstance(_cfgmask[probe]['geometry'], str), (
                f'mask[{probe}][geometry] must be a string'
            )
            assert isinstance(_cfgmask[probe]['footprint_filename'], str), (
                f'mask[{probe}][footprint_filename] must be a string'
            )
            assert isinstance(
                _cfgmask[probe]['weight_maps_filename'], (str, type(None))
            ), f'mask[{probe}][weight_maps_filename] must be a string or None'

        assert isinstance(_cfgmask['nside'], int), 'mask: nside must be an int'
        assert isinstance(_cfgmask['survey_area_deg2'], (int, float)), (
            'mask: survey_area_deg2 must be an int or float'
        )

        # Ensemble Covariance
        _cfgens = self.cfg['ensemble_covariance']
        assert isinstance(_cfgens, dict), (
            "Section 'ensemble_covariance' must be a dictionary"
        )
        assert isinstance(_cfgens['nreal'], int), (
            'ensemble_covariance: nreal must be an int'
        )
        assert isinstance(_cfgens['fix_seed'], bool), (
            'ensemble_covariance: fix_seed must be a boolean'
        )
        assert isinstance(_cfgens['save_sim_cls'], bool), (
            'ensemble_covariance: save_sim_cls must be a boolean'
        )
        assert _cfgens.get('noise_model', 'catalogue') in ('painted', 'catalogue'), (
            "ensemble_covariance: noise_model must be 'painted' or 'catalogue'"
        )

        # OneCovariance
        if 'OneCovariance' in self.cfg:
            _cfgoc = self.cfg['OneCovariance']
            assert isinstance(self.cfg['OneCovariance'], dict), (
                "Section 'OneCovariance' must be a dictionary"
            )
            assert isinstance(_cfgoc['path_to_oc_executable'], str), (
                'OneCovariance: path_to_oc_executable must be a string'
            )
            assert isinstance(_cfgoc['consistency_checks'], bool), (
                'OneCovariance: consistency_checks must be a boolean'
            )
            assert isinstance(_cfgoc['oc_output_filename'], str), (
                'OneCovariance: oc_output_filename must be a string, found '
                f'{_cfgoc["oc_output_filename"]} instead'
            )

        # Binning
        assert isinstance(self.cfg['binning'], dict), (
            "Section 'binning' must be a dictionary"
        )
        binning_cfg = self.cfg['binning']
        assert isinstance(binning_cfg['binning_type'], str), (
            'binning: binning_type must be a string'
        )
        assert isinstance(binning_cfg['ell_min'], (float, int)), (
            'binning: ell_min must be a float or an int'
        )
        assert isinstance(binning_cfg['ell_max'], (float, int)), (
            'binning: ell_max must be a float or an int'
        )
        assert isinstance(binning_cfg['ell_bins'], int), (
            'binning: ell_bins must be an int'
        )
        if self.cfg['probe_selection']['space'] == 'real':
            assert isinstance(binning_cfg['theta_min_arcmin'], (float, int)), (
                'binning: theta_min_arcmin must be a float or an int'
            )
            assert isinstance(binning_cfg['theta_max_arcmin'], (float, int)), (
                'binning: theta_max_arcmin must be a float or an int'
            )
            assert isinstance(binning_cfg['theta_bins'], int), (
                'binning: theta_bins must be an int'
            )
        if self.cfg['probe_selection']['space'] == 'cosebis':
            assert isinstance(binning_cfg['n_modes_cosebis'], int), (
                'binning: n_modes_cosebis must be an int'
            )

        # Covariance
        assert isinstance(self.cfg['covariance'], dict), (
            "Section 'covariance' must be a dictionary"
        )
        _covcfg = self.cfg['covariance']
        assert isinstance(_covcfg['BNT_transform'], bool), (
            'covariance: BNT_transform must be a boolean'
        )
        assert isinstance(_covcfg['G'], bool), 'covariance: G must be a boolean'
        assert isinstance(_covcfg['SSC'], bool), 'covariance: SSC must be a boolean'
        assert isinstance(_covcfg['cNG'], bool), 'covariance: cNG must be a boolean'
        assert isinstance(_covcfg['partial_sky_method'], str), (
            'covariance: partial_sky_method must be a string'
        )

        for _term in ['G', 'SSC', 'cNG']:
            if _covcfg.get(_term):  # Only validate code if term is enabled
                assert _covcfg[f'{_term}_code'] in [
                    'Spaceborne',
                    'PyCCL',
                    'OneCovariance',
                ], (
                    f'Invalid code for covariance term {_term}: {_covcfg[f"{_term}_code"]}. '
                    "Allowed values are: 'Spaceborne', 'PyCCL', 'OneCovariance'."
                )

        assert _covcfg['partial_sky_method'] in ['fsky', 'NaMaster', 'ensemble'], (
            'covariance: partial_sky_method must be either "fsky", "NaMaster", '
            'or "ensemble"'
        )
        assert isinstance(_covcfg['cov_type'], str), (
            'covariance: cov_type must be a string'
        )
        assert _covcfg['cov_type'] in ['coupled', 'decoupled'], (
            'covariance: cov_type must be either "coupled" or "decoupled"'
        )
        assert isinstance(_covcfg['triu_tril'], str), (
            'covariance: triu_tril must be a string'
        )
        assert isinstance(_covcfg['row_col_major'], str), (
            'covariance: row_col_major must be a string'
        )
        assert isinstance(_covcfg['covariance_ordering_2D'], str), (
            'covariance: covariance_ordering_2D must be a string'
        )
        assert isinstance(_covcfg['save_full_cov'], bool), (
            'covariance: save_full_cov must be a boolean'
        )
        assert isinstance(_covcfg['split_gaussian_cov'], bool), (
            'covariance: split_gaussian_cov must be a boolean'
        )

        assert isinstance(_covcfg['sigma_eps_i'], list), (
            'covariance: sigma_eps_i must be a list'
        )
        assert all(isinstance(x, float) for x in _covcfg['sigma_eps_i']), (
            'covariance: All elements in sigma_eps_i must be floats'
        )

        assert isinstance(_covcfg['include_sampling_noise'], bool), (
            'covariance: include_sampling_noise must be a boolean'
        )
        assert isinstance(_covcfg['which_pk_responses'], str), (
            'covariance: which_pk_responses must be a string'
        )
        assert isinstance(_covcfg['ng_cov_gal_bias_model'], str), (
            'covariance: ng_cov_gal_bias_model must be a string'
        )
        assert isinstance(_covcfg['include_b2g'], bool), (
            'covariance: include_b2g must be a boolean'
        )
        assert isinstance(_covcfg['include_terasawa_terms'], bool), (
            'covariance: include_terasawa_terms must be a boolean'
        )
        assert isinstance(_covcfg['cov_filename'], str), (
            'covariance: cov_filename must be a string'
        )
        assert isinstance(_covcfg['G_code'], (str, type(None))), (
            'covariance: G_code must be a string or None'
        )
        assert isinstance(_covcfg['SSC_code'], (str, type(None))), (
            'covariance: SSC_code must be a string or None'
        )
        assert isinstance(_covcfg['n_probes'], int), (
            'covariance: n_probes must be an int'
        )
        assert isinstance(_covcfg['save_mcms'], bool), (
            'covariance: save_mcms must be a boolean'
        )

        # PyCCL
        assert isinstance(self.cfg['PyCCL'], dict), (
            "Section 'PyCCL' must be a dictionary"
        )
        pyccl_cfg = self.cfg['PyCCL']
        assert isinstance(pyccl_cfg['cov_integration_method'], str), (
            'PyCCL: cov_integration_method must be a string'
        )
        assert isinstance(pyccl_cfg['use_default_k_a_grids'], bool), (
            'PyCCL: use_default_k_a_grids must be a boolean'
        )

        # precision
        assert isinstance(self.cfg['precision'], dict), (
            "Section 'precision' must be a dictionary"
        )
        precision_cfg = self.cfg['precision']
        assert isinstance(precision_cfg['ell_min_proj'], int), (
            'precision: ell_min_proj must be an int'
        )
        assert isinstance(precision_cfg['ell_max_proj'], int), (
            'precision: ell_max_proj must be an int'
        )
        assert isinstance(precision_cfg['ell_bins_proj_gauss'], int), (
            'precision: ell_bins_proj_gauss must be an int'
        )
        assert isinstance(precision_cfg['theta_max_arcmin_cosebis'], int), (
            'precision: theta_max_arcmin_cosebis must be an int'
        )
        assert isinstance(precision_cfg['theta_steps_cosebis'], int), (
            'precision: theta_steps_cosebis must be an int'
        )
        assert isinstance(precision_cfg['ell_bins_proj_nongauss'], int), (
            'precision: ell_bins_proj_nongauss must be an int'
        )
        assert isinstance(precision_cfg['ell_bins_proj_nongauss_cosebis'], int), (
            'precision: ell_bins_proj_nongauss_cosebis must be an int'
        )
        assert isinstance(precision_cfg['proj_gauss_integration_method'], str), (
            'precision: proj_gauss_integration_method must be a string'
        )
        assert isinstance(precision_cfg['proj_nongauss_integration_method'], str), (
            'precision: proj_nongauss_integration_method must be a string'
        )
        assert isinstance(precision_cfg['jax_enable_x64'], bool), (
            'precision: jax_enable_x64 must be a boolean'
        )
        assert isinstance(precision_cfg['log10_k_min'], float), (
            'precision: log10_k_min must be a float'
        )
        assert isinstance(precision_cfg['log10_k_max'], float), (
            'precision: log10_k_max must be a float'
        )
        assert isinstance(precision_cfg['k_steps'], int), (
            'precision: k_steps must be an int'
        )
        assert isinstance(precision_cfg['z_min'], float), (
            'precision: z_min must be a float'
        )
        assert isinstance(precision_cfg['z_max'], float), (
            'precision: z_max must be a float'
        )
        assert isinstance(precision_cfg['delta_z'], float), (
            'precision: delta_z must be a float'
        )
        assert isinstance(precision_cfg['delta_z_trisp_SSC'], float), (
            'precision: delta_z_trisp_SSC must be a float'
        )
        assert isinstance(precision_cfg['delta_z_trisp_cNG'], float), (
            'precision: delta_z_trisp_cNG must be a float'
        )
        assert isinstance(precision_cfg['use_KE_approximation'], bool), (
            'precision: use_KE_approximation must be a boolean'
        )
        assert isinstance(precision_cfg['iNKA'], bool), (
            'precision: iNKA must be a boolean'
        )
        assert isinstance(precision_cfg['spin0'], bool), (
            'precision: spin0 must be a boolean'
        )
        pixwin_nside = precision_cfg.get('pixwin_nside')
        assert pixwin_nside is None or (
            isinstance(pixwin_nside, int)
            and pixwin_nside > 0
            and pixwin_nside & (pixwin_nside - 1) == 0
        ), 'precision: pixwin_nside must be None or a power of 2'
        ell_max_nmt = precision_cfg.get('ell_max_nmt')
        assert ell_max_nmt is None or (
            isinstance(ell_max_nmt, int)
            and not isinstance(ell_max_nmt, bool)
            and ell_max_nmt > 0
        ), 'precision: ell_max_nmt must be None or a positive int'
        assert isinstance(precision_cfg.get('iNKA_eb_leakage', False), bool), (
            'precision: iNKA_eb_leakage must be a boolean'
        )
        assert isinstance(precision_cfg['n_samples_wf'], int), (
            'precision: n_samples_wf must be an int'
        )
        assert isinstance(precision_cfg['spline_params'], (dict, type(None))), (
            'precision: spline_params must be a dictionary or None'
        )

        assert isinstance(precision_cfg['gsl_params'], (dict, type(None))), (
            'precision: gsl_params must be a dictionary or None'
        )
        assert isinstance(precision_cfg['spline_params'], (dict, type(None))), (
            'precision: spline_params must be a dictionary or None'
        )

        # misc
        assert isinstance(self.cfg['misc'], dict), "Section 'misc' must be a dictionary"
        misc_cfg = self.cfg['misc']
        assert isinstance(misc_cfg['num_threads'], int), (
            'misc: num_threads must be an int'
        )
        assert isinstance(misc_cfg['test_numpy_inversion'], bool), (
            'misc: test_numpy_inversion must be a boolean'
        )
        assert isinstance(misc_cfg['test_condition_number'], bool), (
            'misc: test_condition_number must be a boolean'
        )
        assert isinstance(misc_cfg['test_cholesky_decomposition'], bool), (
            'misc: test_cholesky_decomposition must be a boolean'
        )
        assert isinstance(misc_cfg['test_symmetry'], bool), (
            'misc: test_symmetry must be a boolean'
        )
        assert isinstance(misc_cfg['output_path'], str), (
            'misc: output_path must be a string'
        )
        assert isinstance(misc_cfg['save_figs'], bool), (
            'misc: save_figs must be a boolean'
        )
        assert isinstance(misc_cfg['jax_platform'], str), (
            'misc: jax_platform must be a string'
        )
        assert isinstance(misc_cfg['cl_triangle_plot'], bool), (
            'misc: cl_triangle_plot must be a boolean'
        )
        assert isinstance(misc_cfg['plot_probe_names'], bool), (
            'misc: plot_probe_names must be a boolean'
        )
        assert isinstance(misc_cfg.get('check_for_updates', True), bool), (
            'misc: check_for_updates must be a boolean'
        )

    def check_misc(self) -> None:
        assert self.cfg['covariance']['n_probes'] == 2, (
            'The code can only accept 2 probes at the moment'
        )

    def check_nz(self) -> None:
        _cfnz = self.cfg['nz']
        assert np.all(np.array(_cfnz['ngal_sources']) > 0), (
            'ngal_sources values must be positive'
        )
        assert np.all(np.array(_cfnz['ngal_lenses']) > 0), (
            'ngal_lenses values must be positive'
        )
        assert len(_cfnz['ngal_sources']) == len(_cfnz['ngal_lenses'])

        if _cfnz['shift_nz']:
            assert len(_cfnz['dz_sources']) == len(_cfnz['ngal_sources'])
            assert len(_cfnz['dz_lenses']) == len(_cfnz['ngal_lenses'])

        assert isinstance(_cfnz['ngal_sources'], list), 'n_gal_shear must be a list'
        assert isinstance(_cfnz['ngal_lenses'], list), 'n_gal_clust must be a list'

    def check_lists(self) -> None:
        assert len(self.cfg['C_ell']['gal_bias_fit_coeff']) == 4, (
            'gal_bias_fit_coeff must be of length 4'
        )
        assert len(self.cfg['C_ell']['mag_bias_fit_coeff']) == 4, (
            'mag_bias_fit_coeff must be of length 4'
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
            assert len(self.cfg['nz']['dz_sources']) == self.zbins, (
                f'dz_sources must be of length zbins = {self.zbins}'
            )
            assert len(self.cfg['nz']['dz_lenses']) == self.zbins, (
                f'dz_lenses must be of length zbins = {self.zbins}'
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
        assert self.cfg['covariance']['ng_cov_gal_bias_model'] in (
            'linear_bias',
            'HOD',
        ), 'ng_cov_gal_bias_model must be either "linear_bias" or "HOD"'
        assert self.cfg['covariance']['triu_tril'] in ('triu', 'tril'), (
            'triu_tril must be either "triu" or "tril"'
        )
        assert self.cfg['covariance']['row_col_major'] in ('row-major', 'col-major'), (
            'row_col_major must be either "row-major" or "col-major"'
        )

        if self.cfg['covariance']['partial_sky_method'] in ['NaMaster', 'ensemble']:
            assert self.cfg['probe_selection']['space'] == 'harmonic', (
                'Namaster and ensemble covariances can only be computed for '
                'harmonic space for the moment'
            )
            assert not self.cfg['covariance']['split_gaussian_cov'], (
                'cfg["covariance"]["split_gaussian_cov"] cannot be '
                'set to True with either '
                'Namaster and ensemble covariances'
            )

        if self.cfg['covariance']['partial_sky_method'] == 'ensemble':
            assert self.cfg['covariance']['G'], (
                'The ensemble method for the partial-sky Gaussian covariance '
                'requires the Gaussian term to be computed'
            )
            assert self.cfg['probe_selection']['space'] == 'harmonic', (
                'The ensemble Gaussian covariance '
                'is only implemented for harmonic space'
            )
            assert (
                self.cfg['probe_selection']['LL']
                and self.cfg['probe_selection']['GL']
                and self.cfg['probe_selection']['GG']
            ), 'The ensemble Gaussian covariance is only implemented for the 3x2pt case'

    def check_probe_selection(self) -> None:
        allowed_keys = [
            'space', 'LL', 'GL', 'GG', 'xip', 'xim', 'gt', 'w', 
            'En', 'Bn', 'Psigl', 'Psigg', 'cross_cov',
        ]  # fmt: skip

        for key in self.cfg['probe_selection']:
            assert key in allowed_keys, f'Probe selection key {key} is not valid. '

        if self.cfg['probe_selection']['space'] == 'real':
            assert (
                self.cfg['probe_selection']['xip']
                + self.cfg['probe_selection']['xim']
                + self.cfg['probe_selection']['gt']
                + self.cfg['probe_selection']['w']
            ) > 0, (
                'At least one of xip or xim must be selected for real space covariance'
            )
            if self.cfg['covariance']['partial_sky_method'] == 'NaMaster':
                raise NotImplementedError(
                    'The projection of the partial-sky Gaussian covariance to real '
                    'space is not implemented yet'
                )
        elif self.cfg['probe_selection']['space'] == 'harmonic':
            assert (
                self.cfg['probe_selection']['LL']
                + self.cfg['probe_selection']['GL']
                + self.cfg['probe_selection']['GG']
            ) > 0, (
                'At least one of LL, GL, or GG must be selected for '
                'harmonic space covariance'
            )
        elif self.cfg['probe_selection']['space'] == 'cosebis':
            assert (
                self.cfg['probe_selection']['En']
                + self.cfg['probe_selection']['Bn']
                + self.cfg['probe_selection']['Psigl']
                + self.cfg['probe_selection']['Psigg']
            ) > 0, (
                'At least one of En, Bn, Psigl, or Psigg must be selected for '
                'cosebis space covariance'
            )

        else:
            raise ValueError(
                'probe_selection: "space" must be either "real", '
                f'"harmonic", or "cosebis" got {self.cfg["probe_selection"]["space"]}'
            )

    def check_nmt(self) -> None:
        if (
            self.cfg['covariance']['cov_type'] == 'coupled'
            and self.cfg['covariance']['G']
        ):
            assert self.cfg['covariance']['partial_sky_method'] in [
                'NaMaster',
                'ensemble',
            ], (
                'if the coupled Gaussian covariance is requested either '
                'cfg["covariance"]["partial_sky_method"] must be "NaMaster" '
                'or "ensemble"'
            )
        if self.cfg['covariance']['partial_sky_method'] == 'NaMaster':
            assert self.cfg['binning']['binning_type'] != 'ref_cut', (
                'ref_cut case incompatible with NaMaster for the moment. '
                'Please use a different binning type.'
            )
        if (
            self.cfg['covariance']['partial_sky_method'] == 'NaMaster'
            and self.cfg['covariance']['G_code'] != 'Spaceborne'
        ):
            raise ValueError(
                'If computing the partial-sky covariance with NaMaster, '
                '`covariance: G_code` should not be set (or set to "Spaceborne").'
            )
        # fsky has no pixel window
        if self.cfg['precision'].get('pixwin_nside') is not None:
            assert self.cfg['covariance']['partial_sky_method'] in (
                'NaMaster',
                'ensemble',
            ), (
                'precision: pixwin_nside requires partial_sky_method `NaMaster` '
                'or `ensemble`'
            )

    def check_onecov(self) -> None:
        # The 'OneCovariance' section is optional (see check_types); if it is
        # absent, OneCovariance simply cannot be used, so compare_against_oc
        # defaults to False.
        compare_against_oc = self.cfg.get('OneCovariance', {}).get(
            'compare_against_oc', False
        )

        uses_oc = (
            compare_against_oc
            or self.cfg['covariance']['G_code'] == 'OneCovariance'
            or self.cfg['covariance']['SSC_code'] == 'OneCovariance'
            or self.cfg['covariance']['cNG_code'] == 'OneCovariance'
        )

        if self.cfg['C_ell']['has_mag_bias'] and uses_oc:
            raise ValueError(
                'Magnification bias is enabled in the configuration, '
                'but OneCovariance does not include it. '
                'Results for GGL and GG might be inconsistent.'
            )

        if (
            self.cfg['covariance']['SSC']
            and not self.cfg['precision']['use_KE_approximation']
        ) and (
            compare_against_oc or self.cfg['covariance']['SSC_code'] == 'OneCovariance'
        ):
            raise ValueError(
                'The KE approximation for the SSC term is disabled in the '
                'configuration, but OneCovariance uses it. '
                'Results for GGL and GG might be inconsistent.'
            )

    def check_pyccl(self) -> None:
        cov_cfg = self.cfg['covariance']
        pyccl_ssc = cov_cfg['SSC'] and cov_cfg['SSC_code'] == 'PyCCL'
        pyccl_cng = cov_cfg['cNG'] and cov_cfg['cNG_code'] == 'PyCCL'
        has_mag = self.cfg['C_ell']['has_mag_bias']

        # the magnification component of the galaxy tracers would need the matter
        # response, while the PyCCL SSC pairs the galaxy legs with the galaxy one
        if pyccl_ssc and has_mag:
            raise ValueError(
                'The PyCCL SSC does not support magnification bias yet. '
                "Please set SSC_code: 'Spaceborne' or disable magnification bias."
            )

        # same for the HOD cNG, whose galaxy legs are paired with the HOD trispectrum.
        # The linear-bias cNG pairs the full galaxy kernel with the matter
        # trispectrum, and is therefore correct
        if pyccl_cng and cov_cfg['ng_cov_gal_bias_model'] == 'HOD' and has_mag:
            raise ValueError(
                'The PyCCL cNG with ng_cov_gal_bias_model: HOD does not support '
                'magnification bias yet. Please set '
                "ng_cov_gal_bias_model: 'linear_bias' or disable magnification bias."
            )

        # pyccl.halos.pk_4pt.halomod_trispectrum_2h_13 (checked up to v3.3.3) takes
        # shortcuts that are wrong for mixed matter/galaxy profiles
        if pyccl_cng and cov_cfg['ng_cov_gal_bias_model'] == 'HOD':
            warnings.warn(
                'The PyCCL cNG with ng_cov_gal_bias_model: HOD uses the HOD '
                'trispectrum, whose 2-halo (1+3) term could have a bug in CCL for mixed'
                ' matter/galaxy profiles (e.g. the LLGG and GLGG blocks). The HOD is '
                'also not consistent with the linear galaxy bias used in the C_ells.',
                stacklevel=2,
            )
        if cov_cfg['ng_cov_gal_bias_model'] == 'HOD':
            warnings.warn(
                'At the moment, the Cls use linear galaxy bias, so selecting '
                'ng_cov_gal_bias_model: HOD will create some inconsistency between '
                'the cNG and G terms.',
                stacklevel=2,
            )

        # CCL's angular_cl_cov_SSC/cNG ignore the RSD component of the tracers
        if (cov_cfg['SSC'] or cov_cfg['cNG']) and self.cfg['C_ell']['has_rsd']:
            warnings.warn(
                'has_rsd is True, but the non-Gaussian covariance terms do not include'
                ' this contribution.',
                stacklevel=2,
            )

    def check_mask(self) -> None:
        for probe in ['LL', 'GG']:
            assert self.cfg['mask'][probe]['geometry'] in [
                'footprint_file',
                'polar_cap',
            ], (
                f'cfg["mask"][{probe}]["geometry"] must be either'
                ' "footprint_file" or "polar_cap"'
            )

            if self.cfg['mask'][probe]['geometry'] == 'footprint_file':
                fp = self.cfg['mask'][probe]['footprint_filename']
                assert os.path.isfile(fp), f'Footprint file for {probe} not found: {fp}'

    def _check_bias_per_bin(self, key: str) -> None:
        """The per-bin bias list is only required when the 'per_bin' option is set."""
        vals = self.cfg['C_ell'].get(key)
        assert isinstance(vals, list), (
            f'C_ell: {key} must be a list of length zbins = {self.zbins}'
        )
        assert all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals
        ), f'C_ell: All elements in {key} must be numbers'
        assert len(vals) == self.zbins, (
            f'C_ell: {key} must be of length zbins = {self.zbins}, got {len(vals)}'
        )

    def check_options(self) -> None:
        bias_options = ['from_input', 'polynomial_fit', 'per_bin']

        assert self.cfg['C_ell']['which_gal_bias'] in bias_options, (
            f'which_gal_bias should be one of {bias_options}'
        )
        if self.cfg['C_ell']['which_gal_bias'] == 'per_bin':
            self._check_bias_per_bin('gal_bias_per_bin')

        if self.cfg['C_ell']['has_mag_bias']:
            assert self.cfg['C_ell']['which_mag_bias'] in bias_options, (
                f'which_mag_bias should be one of {bias_options}'
            )
            if self.cfg['C_ell']['which_mag_bias'] == 'per_bin':
                self._check_bias_per_bin('mag_bias_per_bin')

    def check_projection_methods(self) -> None:
        """Integration methods for the harmonic -> real-space projection."""
        space = self.cfg['probe_selection']['space']
        # no projection is performed in harmonic space
        if space == 'harmonic':
            return

        precision_cfg = self.cfg['precision']
        cov_cfg = self.cfg['covariance']
        allowed = {
            'proj_gauss_integration_method': ('simps', 'FFTLog'),
            'proj_nongauss_integration_method': ('simps', 'quad', 'FFTLog'),
        }

        # only check the methods of the terms which are actually projected
        integr_method_keys = []
        if cov_cfg['G']:
            integr_method_keys.append('proj_gauss_integration_method')
        if cov_cfg['SSC'] or cov_cfg['cNG']:
            integr_method_keys.append('proj_nongauss_integration_method')

        for key in integr_method_keys:
            integr_method = precision_cfg[key]
            if integr_method not in allowed[key]:
                raise ValueError(
                    f'precision: {key} must be one of {allowed[key]}, '
                    f'got {integr_method!r}'
                )

        if space == 'real':
            ell_bins_keys = {
                'proj_gauss_integration_method': 'ell_bins_proj_gauss',
                'proj_nongauss_integration_method': 'ell_bins_proj_nongauss',
            }
            for key in integr_method_keys:
                if precision_cfg[key] != 'FFTLog':
                    continue
                if self.cfg['binning']['binning_type'] != 'log':
                    raise ValueError(
                        f"precision: {key}='FFTLog' requires log-spaced theta bins "
                        "(binning_type: 'log')."
                    )
                nbl_key = ell_bins_keys[key]
                if precision_cfg[nbl_key] % 2 != 0:
                    raise ValueError(
                        f"precision: {key}='FFTLog' requires an even {nbl_key}, "
                        f'got {precision_cfg[nbl_key]}.'
                    )

        # warn that for COSEBIs only simps is used
        if space == 'cosebis':
            forced_keys = [k for k in integr_method_keys if precision_cfg[k] != 'simps']
            if forced_keys:
                warnings.warn(
                    f'For COSEBIs, the projection methods are forced to "simps"; '
                    f'the configured {forced_keys} will be ignored.',
                    stacklevel=2,
                )

    def run_all_checks(self) -> None:
        self.check_types()
        self.check_nmt()
        self.check_options()
        self.check_mask()
        self.check_BNT_transform()
        self.check_onecov()
        self.check_pyccl()
        self.check_lists()
        # self.check_fsky()
        self.check_probe_selection()
        self.check_misc()
        self.check_nz()
        self.check_cosmo()
        self.check_cov()
        self.check_projection_methods()
