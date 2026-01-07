"""
Base class for projected covariance computations.

This module provides the shared infrastructure for computing covariances
of projected statistics (2PCF, COSEBIs, etc.) from harmonic-space C_ℓ.

The key insight is that different statistics share the same integrand building
(SVA, MIX terms) but differ in how they project from C_ℓ to the observable space.
"""

import numpy as np

from spaceborne import constants as const


def get_npair(theta_1_u, theta_1_l, survey_area_sr, n_eff_i, n_eff_j):
    """Compute total (ideal) number of pairs in a theta bin, i.e., N(theta).
    N(θ) = π (θ_u^2 - θ_l^2) × A × n_i × n_j
         = \int_{θ_l}^{θ_u} dθ (dN(θ)/dθ)
    """
    n_eff_i *= const.SR_TO_ARCMIN2
    n_eff_j *= const.SR_TO_ARCMIN2
    return np.pi * (theta_1_u**2 - theta_1_l**2) * survey_area_sr * n_eff_i * n_eff_j


def get_dnpair(theta, survey_area_sr, n_eff_i, n_eff_j):
    """Compute differential (ideal) number of pairs, i.e. dN(theta)/dtheta.
    dN(θ)/dθ = 2π θ × A × n_i × n_j
    """
    n_eff_i *= const.SR_TO_ARCMIN2
    n_eff_j *= const.SR_TO_ARCMIN2
    return 2 * np.pi * theta * survey_area_sr * n_eff_i * n_eff_j


def t_mix(probe_a_ix, zbins, sigma_eps_i):
    """
    Helper function for MIX term computation.

    Returns the appropriate variance term for the given probe.
    """
    t_munu = np.zeros(zbins)

    # xipxip or ximxim
    if probe_a_ix == 0:
        t_munu = sigma_eps_i**2

    # gggg
    elif probe_a_ix == 1:
        t_munu = np.ones(zbins)

    return t_munu


class CovarianceProjector:
    """
    Base class for all projected covariance computations.

    This class provides:
    - Shared setup (survey info, galaxy densities, etc.)
    - Integrand builders (SVA, MIX) that work from C_ℓ
    - Abstract projection interface for subclasses

    Subclasses (CovRealSpace, CovCosebis) implement:
    - Specific projection kernels (k_mu, W_n, etc.)
    - Statistic-specific infrastructure (theta bins, modes, etc.)
    """

    def __init__(self, cfg, pvt_cfg, mask_obj):
        """
        Initialize shared infrastructure.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary
        pvt_cfg : dict
            Private configuration with derived quantities
        mask_obj : object
            Mask object with survey geometry information
        """
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg
        self.mask_obj = mask_obj

        # Shared setup
        self.zbins = pvt_cfg['zbins']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.ind_dict = pvt_cfg['ind_dict']
        self.n_jobs = cfg['misc']['num_threads']

        self._set_survey_info()
        self._set_terms_toloop()
        # TODO add this
        # self._set_neff_and_sigma_eps()

    def _set_survey_info(self):
        """Set up survey geometry information."""
        self.survey_area_deg2 = self.mask_obj.survey_area_deg2
        self.survey_area_sr = self.mask_obj.survey_area_sr
        self.fsky = self.mask_obj.fsky
        self.srtoarcmin2 = const.SR_TO_ARCMIN2
        self.amax = max((self.survey_area_sr, self.survey_area_sr))

    def _set_terms_toloop(self):
        self.terms_toloop = []
        if self.cfg['covariance']['G']:
            self.terms_toloop.extend(('sva', 'sn', 'mix'))
        if self.cfg['covariance']['SSC']:
            self.terms_toloop.append('ssc')
        if self.cfg['covariance']['cNG']:
            self.terms_toloop.append('cng')
            
    def get_delta_tomo(self, probe_a_ix, probe_b_ix):
        if probe_a_ix == probe_b_ix:
            return np.eye(self.zbins)
        else:
            return np.zeros((self.zbins, self.zbins))
