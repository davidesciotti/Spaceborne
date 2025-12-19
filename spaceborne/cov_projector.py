import numpy as np
from scipy.integrate import simps
from spaceborne import constants as const


def t_mix(probe_a_ix, zbins, sigma_eps_i):
    t_munu = np.zeros(zbins)

    # xipxip or ximxim
    if probe_a_ix == 0:
        t_munu = sigma_eps_i**2

    # gggg
    elif probe_a_ix == 1:
        t_munu = np.ones(zbins)

    return t_munu


class CovarianceProjector:
    """Base class with shared covariance computation machinery"""

    def __init__(self, cfg, pvt_cfg, mask_obj):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg
        self.mask_obj = mask_obj
        self.zbins = pvt_cfg['zbins']

        # Shared setup
        self._set_survey_info()
        self._set_neff_and_sigma_eps()
        self.n_jobs = cfg['misc']['num_threads']

    def _set_survey_info(self):
        """From CovRealSpace - unchanged"""
        self.survey_area_deg2 = self.mask_obj.survey_area_deg2
        self.survey_area_sr = self.mask_obj.survey_area_sr
        self.fsky = self.mask_obj.fsky
        self.srtoarcmin2 = const.SR_TO_ARCMIN2
        self.amax = max((self.survey_area_sr, self.survey_area_sr))

    def _set_neff_and_sigma_eps(self):
        """From CovRealSpace - unchanged"""
        self.n_eff_lens = self.cfg['nz']['ngal_lenses']
        self.n_eff_src = self.cfg['nz']['ngal_sources']
        self.n_eff_2d = np.vstack((self.n_eff_lens, self.n_eff_lens, self.n_eff_src))
        self.sigma_eps_i = np.array(self.cfg['covariance']['sigma_eps_i'])
        self.sigma_eps_tot = self.sigma_eps_i * np.sqrt(2)

    def build_sva_integrand(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
        """Shared logic: build the SVA integrand from C_ℓ
        Returns shape: (nbl, zbins, zbins, zbins, zbins)"""
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
        """Shared logic: build the MIX integrand"""

        def _get_mix_prefac(probe_b_ix, probe_d_ix, zj, zl):
            prefac = (
                self.get_delta_tomo(probe_b_ix, probe_d_ix)[zj, zl]
                * t_mix(probe_b_ix, self.zbins, self.sigma_eps_i)[zj]
                / (self.n_eff_2d[probe_b_ix, zj] * self.srtoarcmin2)
            )
            return prefac

        prefac = np.zeros((2, 2, self.zbins, self.zbins))  # n_probes_hs=2
        for pa in range(2):
            for pb in range(2):
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
        """Shared helper"""
        if probe_a_ix == probe_b_ix:
            return np.eye(self.zbins)
        else:
            return np.zeros((self.zbins, self.zbins))

    # Abstract method - subclasses implement specific projection
    def project_integrand(self, integrand_5d, **kwargs):
        raise NotImplementedError('Subclasses must implement projection')


class CovRealSpace(CovarianceProjector):
    """Real-space binned correlations - your existing class"""

    def __init__(self, cfg, pvt_cfg, mask_obj):
        super().__init__(cfg, pvt_cfg, mask_obj)

        # Real-space specific setup
        self._set_theta_binning()
        self._set_levin_bessel_precision()
        self.integration_method = cfg['precision']['cov_rs_int_method']
        self.levin_bin_avg = cfg['precision']['levin_bin_avg']
        # ... rest of your init

    def cov_sva_levin(
        self,
        probe_a_ix,
        probe_b_ix,
        probe_c_ix,
        probe_d_ix,
        zpairs_ab,
        zpairs_cd,
        ind_ab,
        ind_cd,
        mu,
        nu,
    ):
        """Use parent's integrand builder + child's projection"""
        integrand_5d = self.build_sva_integrand(
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix
        )

        # Real-space specific projection with k_mu kernels
        return self.cov_levin_wrapper(
            integrand_5d, zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu
        )


class CovCOSEBI(CovarianceProjector):
    """COSEBIs mode covariance"""

    def __init__(self, cfg, pvt_cfg, mask_obj, tmin, tmax, Nmax):
        super().__init__(cfg, pvt_cfg, mask_obj)

        # COSEBI-specific setup
        self.tmin, self.tmax, self.Nmax = tmin, tmax, Nmax
        from cloelib.auxiliary import cosebi_helpers as ch

        self.rn, self.nn, self.coeff_j = ch.get_roots_and_norms(tmax, tmin, Nmax)

    def cov_sva_cosebi(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
        """Use parent's integrand builder + COSEBI projection"""
        integrand_5d = self.build_sva_integrand(
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix
        )

        # COSEBI-specific projection with W_ell weights
        W_ells = self._get_W_ell_weights()
        return self._project_with_cosebi_weights(integrand_5d, W_ells)

    def _get_W_ell_weights(self):
        """COSEBI-specific"""
        from cloelib.auxiliary import cosebi_helpers as ch

        theta_grid = np.logspace(np.log10(self.tmin), np.log10(self.tmax), 1000)
        return ch.get_W_ell(theta_grid, self.Nmax, self.ells, self.n_jobs)

    def _project_with_cosebi_weights(self, integrand_5d, W_ells):
        """COSEBI-specific projection"""
        cov_6d = np.zeros(
            (self.Nmax, self.Nmax, self.zbins, self.zbins, self.zbins, self.zbins)
        )

        for n in range(1, self.Nmax + 1):
            for m in range(1, self.Nmax + 1):
                weight_nm = W_ells[n] * W_ells[m] * self.ells / (2 * np.pi * self.amax)
                integrand_weighted = integrand_5d * weight_nm[:, None, None, None, None]
                cov_6d[n - 1, m - 1] = simps(integrand_weighted, x=self.ells, axis=0)

        return cov_6d
