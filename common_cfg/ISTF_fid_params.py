import numpy as np

primary = {
    'Om_m0': 0.32,
    'Om_b0': 0.05,
    'w_0': -1.,
    'w_a': 0.0,
    'h_0': 0.67,
    'n_s': 0.96,
    'sigma_8': 0.816,

    'Om_mh2': 0.143648,  # should be small omega
    'Om_bh2': 0.022445,  # should be small omega
}

extensions = {
    'm_nu': 0.06,  # eV
    'gamma': 0.55,
    'Om_Lambda0': 0.68,
    'Om_k0': 0.
}

other_cosmo_params = {
    'tau': 0.0925,
    'A_s': 2.12605e-9,
}

N_ur = 2.03351
N_eff = 1. + N_ur
neutrino_mass_fac = 94.07
g_factor = N_eff / 3
neutrino_params = {
    'N_ncdm': 1.,  # effective number of massive neutrino species in CLASS
    'N_ur': N_ur,  # effective number of massless neutrino species in CLASS
    'N_eff': N_eff,  # effective number neutrino species
    # 'Omega_nu': extensions['m_nu'] / (93.14 * primary['h_0'] ** 2)
    'Om_nu0': extensions['m_nu'] / (neutrino_mass_fac * g_factor ** 0.75 * primary['h_0'] ** 2)
    # ! slightly different from above
}

IA_free = {
    'A_IA': 1.72,
    'eta_IA': -0.41,
    'beta_IA': 2.17,
}

IA_fixed = {
    'C_IA': 0.0134,
}

other_survey_specs = {
    'survey_area': 15_000,  # deg**2
    'f_sky': 0.363610260832152,
    'n_gal': 30,  # galaxy number density [gal/arcmin**2]
    'sigma_eps': 0.3  # intrinsic ellipticity dispersion (the value here it's sigma_eps, not sigma_eps^2!!)
}

photoz_bins = {
    'zbins': 10,
    'z_median': 0.9,
    'z_minus': np.array([0.001, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576]),
    'z_plus': np.array([0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, 2.50]),
    'z_mean': np.array((0.2095, 0.489, 0.619, 0.7335, 0.8445, 0.9595, 1.087, 1.2395, 1.45, 2.038)),
    'all_zbin_edges': np.array([0.001, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, 2.50]),
}

photoz_pdf = {
    'f_out': 0.1,
    'sigma_o': 0.05,
    'sigma_b': 0.05,
    'c_o': 1.,
    'c_b': 1.,
    'z_o': 0.1,
    'z_b': 0.,
}

photoz_galaxy_bias = {
    'b01_photo': 1.0997727037892875,
    'b02_photo': 1.220245876862528,
    'b03_photo': 1.2723993083933989,
    'b04_photo': 1.316624471897739,
    'b05_photo': 1.35812370570578,
    'b06_photo': 1.3998214171814918,
    'b07_photo': 1.4446452851824907,
    'b08_photo': 1.4964959071110084,
    'b09_photo': 1.5652475842498528,
    'b10_photo': 1.7429859437184225,
}

spectro_bias = {
    'b1_spectro': 1.46,
    'b2_spectro': 1.61,
    'b3_spectro': 1.75,
    'b4_spectro': 1.90,
}

photoz_shear_bias = {
    'm1_photo': 0.0,
    'm2_photo': 0.0,
    'm3_photo': 0.0,
    'm4_photo': 0.0,
    'm5_photo': 0.0,
    'm6_photo': 0.0,
    'm7_photo': 0.0,
    'm8_photo': 0.0,
    'm9_photo': 0.0,
    'm10_photo': 0.0,
}

constants = {
    'c': 299792.458  # km/s
}

forecasts = {
    # these are altredy in percent!
    'LCDM_par_order': ('Om', 'Ob', 'h', 'ns', 's8'),
    'w0waCDM_par_order': ('Om', 'Ob', 'w0', 'wa', 'h', 'ns', 's8'),
    'WL_pes_LCDM_flat': np.array((0.018, 0.47, 0.21, 0.035, 0.0087)) * 100,
    'WL_opt_LCDM_flat': np.array((0.012, 0.42, 0.20, 0.030, 0.0061)) * 100,
    '3x2pt_pes_LCDM_flat': np.array((0.0081, 0.052, 0.027, 0.0085, 0.0038)) * 100,
    '3x2pt_opt_LCDM_flat': np.array((0.0028, 0.046, 0.020, 0.0036, 0.0013)) * 100,
    'WL_pes_w0waCDM_flat': np.array((0.044, 0.47, 0.16, 0.59, 0.21, 0.038, 0.019)) * 100,
    'WL_opt_w0waCDM_flat': np.array((0.034, 0.42, 0.14, 0.48, 0.20, 0.030, 0.013)) * 100,
    '3x2pt_pes_w0waCDM_flat': np.array((0.011, 0.054, 0.042, 0.17, 0.029, 0.010, 0.0048)) * 100,
    '3x2pt_opt_w0waCDM_flat': np.array((0.0059, 0.046, 0.027, 0.10, 0.020, 0.0039, 0.0022)) * 100,
    'GC_pes_w0waCDM_flat': np.array((0, 0, 0, 0, 0, 0, 0)) * 100,
    'GC_opt_w0waCDM_flat': np.array((0, 0, 0, 0, 0, 0, 0)) * 100,
}
