"""
Base class for projected covariance computations.

This module provides the shared infrastructure for computing covariances
of projected statistics (2PCF, COSEBIs, etc.) from harmonic-space C_ℓ.

The key insight is that different statistics share the same integrand building
(SVA, MIX terms) but differ in how they project from C_ℓ to the observable space.
"""

import numpy as np

from spaceborne import constants as const


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

        # Tomographic setup
        self.zbins = pvt_cfg['zbins']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.ind_dict = pvt_cfg['ind_dict']

        # Shared setup
        self._set_survey_info()
        self._set_neff_and_sigma_eps()

        # Computational settings
        self.n_jobs = cfg['misc']['num_threads']

        # TODO these should be centralized somewhere
        self.n_probes_hs = 2
        self.n_probes_rs = 4

        # These will be set when computing covariances
        self.cl_3x2pt_5d = None
        self.ells = None
        self.nbl = None

    def _set_survey_info(self):
        """Set up survey geometry information."""
        self.survey_area_deg2 = self.mask_obj.survey_area_deg2
        self.survey_area_sr = self.mask_obj.survey_area_sr
        self.fsky = self.mask_obj.fsky
        self.srtoarcmin2 = const.SR_TO_ARCMIN2
        self.amax = max((self.survey_area_sr, self.survey_area_sr))

    def _set_neff_and_sigma_eps(self):
        """Set up galaxy number densities and shape noise."""
        self.n_eff_lens = self.cfg['nz']['ngal_lenses']
        self.n_eff_src = self.cfg['nz']['ngal_sources']
        # Shape: (n_probes_hs, zbins)
        self.n_eff_2d = np.row_stack((self.n_eff_lens, self.n_eff_lens, self.n_eff_src))

        self.sigma_eps_i = np.array(self.cfg['covariance']['sigma_eps_i'])
        self.sigma_eps_tot = self.sigma_eps_i * np.sqrt(2)

    def build_sva_integrand(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
        """
        Build the sample variance (SVA) integrand from C_ℓ.

        This implements the standard Gaussian covariance formula:
        Cov_G[C_ab, C_cd] = (C_ac * C_bd + C_ad * C_bc) / (prefactors)

        Parameters
        ----------
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix : int
            Probe indices (0=shear, 1=galaxy)

        Returns
        -------
        integrand_5d : np.ndarray
            Shape (nbl, zbins, zbins, zbins, zbins)
            Integrand to be projected with kernels
        """
        if self.cl_3x2pt_5d is None:
            raise ValueError('Must set cl_3x2pt_5d before building integrands')

        a = np.einsum(
            'Lik,Ljl->Lijkl',
            self.cl_3x2pt_5d[probe_a_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_d_ix],
        )
        b = np.einsum(
            'Lil,Ljk->Lijkl',
            self.cl_3x2pt_5d[probe_a_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_c_ix],
        )
        return a + b

    def build_mix_integrand(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
        """
        Build the mixed (MIX) term integrand.

        This is the cross-term between signal and noise in the Gaussian covariance.

        Parameters
        ----------
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix : int
            Probe indices (0=shear, 1=galaxy)

        Returns
        -------
        integrand_5d : np.ndarray
            Shape (nbl, zbins, zbins, zbins, zbins)
        """
        if self.cl_3x2pt_5d is None:
            raise ValueError('Must set cl_3x2pt_5d before building integrands')

        def _get_mix_prefac(probe_b_ix, probe_d_ix, zj, zl):
            prefac = (
                self.get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zl]
                * t_mix(probe_b_ix, self.zbins, self.sigma_eps_i)[zj]
                / (self.n_eff_2d[probe_b_ix, zj] * self.srtoarcmin2)
            )
            return prefac

        prefac = np.zeros((self.n_probes_hs, self.n_probes_hs, self.zbins, self.zbins))
        for pa in range(self.n_probes_hs):
            for pb in range(self.n_probes_hs):
                for zi in range(self.zbins):
                    for zj in range(self.zbins):
                        prefac[pa, pb, zi, zj] = _get_mix_prefac(pa, pb, zi, zj)

        a = np.einsum(
            'jl,Lik->Lijkl',
            prefac[probe_b_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_a_ix, probe_c_ix],
        )
        b = np.einsum(
            'ik,Ljl->Lijkl',
            prefac[probe_a_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_d_ix],
        )
        c = np.einsum(
            'jk,Lil->Lijkl',
            prefac[probe_b_ix, probe_c_ix],
            self.cl_3x2pt_5d[probe_a_ix, probe_d_ix],
        )
        d = np.einsum(
            'il,Ljk->Lijkl',
            prefac[probe_a_ix, probe_d_ix],
            self.cl_3x2pt_5d[probe_b_ix, probe_c_ix],
        )
        return a + b + c + d

    def get_delta_tomo(self, probe_a_ix, probe_b_ix):
        """
        Kronecker delta for tomographic bins.

        Returns identity matrix if same probe type, zeros otherwise.
        """
        if probe_a_ix == probe_b_ix:
            return np.eye(self.zbins)
        else:
            return np.zeros((self.zbins, self.zbins))

    def set_cl_and_ells(self, cl_3x2pt_5d, ells):
        """
        Set the C_ℓ data and ell grid.

        This must be called before computing covariances.
        """
        self.cl_3x2pt_5d = cl_3x2pt_5d
        self.ells = ells
        self.nbl = len(ells)

    # Abstract methods - subclasses must implement
    def project_integrand(self, integrand_5d, **kwargs):
        """
        Project the integrand with statistic-specific kernels.

        This is the key method that differs between statistics:
        - CovRealSpace: projects with k_mu(ell, theta) to theta bins
        - CovCosebis: projects with W_n(ell) to COSEBIs modes

        Must be implemented by subclasses.
        """
        raise NotImplementedError('Subclasses must implement projection')

    def compute_cov_term(self, probe_abcd, term, **kwargs):
        """
        Generic interface for computing a covariance term.

        This method:
        1. Builds the appropriate integrand (SVA, MIX, SN)
        2. Calls subclass-specific projection

        Must be implemented by subclasses with statistic-specific logic.
        """
        raise NotImplementedError('Subclasses must implement compute_cov_term')


# Note: Actual implementations of CovRealSpace and CovCosebis are in:
# - spaceborne/cov_real_space.py
# - spaceborne/cov_cosebis.py
#
# These files should import and inherit from CovarianceProjector.
