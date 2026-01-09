"""
Base class for projected covariance computations.

This module provides the shared infrastructure for computing covariances
of projected statistics (2PCF, COSEBIs, etc.) from harmonic-space C_ℓ.

The key insight is that different statistics share the same integrand building
(SVA, MIX terms) but differ in how they project from C_ℓ to the observable space.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import simpson as simps
from tqdm import tqdm

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


def get_delta_tomo(probe_a_ix: int, probe_b_ix: int, zbins: int) -> np.ndarray:
    return np.eye(zbins) if probe_a_ix == probe_b_ix else np.zeros((zbins, zbins))


def build_cov_sva_integrand_5d(cl_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
    """
    Build the SVA (sample variance) integrand in harmonic space.

    This integrand is UNIVERSAL - it's the same for all projection methods
    (real space, COSEBIs, band powers, etc.). The formula comes from the
    Gaussian covariance of C_ℓ:

    Cov[C_ab, C_cd] ∝ C_ac * C_bd + C_ad * C_bc

    Parameters
    ----------
    cl_5d : np.ndarray
        Power spectra array with shape (n_probes, n_probes, n_ell, zbins, zbins)
    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix : int
        Probe indices (0 for shear, 1 for galaxy clustering)

    Returns
    -------
    integrand_5d : np.ndarray
        Shape (n_ell, zbins, zbins, zbins, zbins)
        The harmonic-space integrand before projection
    """
    a = np.einsum(
        'Lik,Ljl->Lijkl', cl_5d[probe_a_ix, probe_c_ix], cl_5d[probe_b_ix, probe_d_ix]
    )
    b = np.einsum(
        'Lil,Ljk->Lijkl', cl_5d[probe_a_ix, probe_d_ix], cl_5d[probe_b_ix, probe_c_ix]
    )
    return a + b


def cov_sva_simps(
    ells,
    cl_5d,
    probe_a_ix,
    probe_b_ix,
    probe_c_ix,
    probe_d_ix,
    zi,
    zj,
    zk,
    zl,
    Amax,
    kernel_1_func_of_ell,
    kernel_2_func_of_ell,
):
    """
    Universal Simpson integrator for SVA covariance - projection kernel agnostic.

    This function computes a single matrix element of the SVA covariance by:
    1. Selecting the relevant C_ℓ spectra for the given tomographic bins
    2. Evaluating projection kernels (e.g., k_mu for real space, W_n for COSEBIs)
    3. Building the integrand: ℓ * kernel_1 * kernel_2 * (C_ik*C_jl + C_il*C_jk)
    4. Integrating with Simpson's rule

    Parameters
    ----------
    ells : np.ndarray
        Multipole values
    cl_5d : np.ndarray
        Power spectra with shape (n_probes, n_probes, n_ell, zbins, zbins)
    probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix : int
        Probe indices
    zi, zj, zk, zl : int
        Tomographic bin indices
    Amax : float
        Survey area in steradians
    kernel_1_func : callable
        First projection kernel function of ℓ (e.g., k_mu(ℓ, theta_1))
    kernel_2_func : callable
        Second projection kernel function of ℓ (e.g., k_nu(ℓ, theta_2))

    Returns
    -------
    cov_elem : float
        Single covariance matrix element
    """
    # Extract relevant C_ℓ for these tomographic bins
    c_ik = cl_5d[probe_a_ix, probe_c_ix, :, zi, zk]
    c_jl = cl_5d[probe_b_ix, probe_d_ix, :, zj, zl]
    c_il = cl_5d[probe_a_ix, probe_d_ix, :, zi, zl]
    c_jk = cl_5d[probe_b_ix, probe_c_ix, :, zj, zk]

    # Evaluate projection kernels
    kmu = kernel_1_func_of_ell(ells)
    knu = kernel_2_func_of_ell(ells)

    # Build integrand: ℓ * K_μ * K_ν * (C_ik*C_jl + C_il*C_jk)
    integrand = ells * kmu * knu * (c_ik * c_jl + c_il * c_jk)

    # Integrate with Simpson's rule
    integral = simps(y=integrand, x=ells)

    # Apply normalization factor
    return integral / (2.0 * np.pi * Amax)


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
        self.nbx = pvt_cfg['nbx']
        self.n_jobs = cfg['misc']['num_threads']

        self._set_survey_info()
        self._set_terms_toloop()
        # TODO add this
        # self._set_neff_and_sigma_eps()
        # TODO here (in the init) I should add the finely binned Cls, which are used in all projections!

        self.cov_shape_6d = (
            self.nbx,
            self.nbx,
            self.zbins,
            self.zbins,
            self.zbins,
            self.zbins,
        )

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

    def cov_parallel_helper(
        self,
        scale_ix_1,
        scale_ix_2,
        zij,
        zkl,
        ind_ab,
        ind_cd,
        cov_func,
        cov_func_kw,
        kernel_builder_func_kw,
    ):
        """
        Universal parallel helper for covariance computation.

        This method provides a unified interface for parallel covariance calculation
        across different projection methods (real space, COSEBIs, band powers, etc.).
        The projection-specific kernel construction is delegated to child classes
        via the kernel_builder callback.

        Parameters
        ----------
        scale_ix_1, scale_ix_2 : int
            First and second projection indices. These represent:
            - Theta bin indices for real space (theta_1_ix, theta_2_ix)
            - Mode indices for COSEBIs (mode_n, mode_m)
            - Ell bin indices for band powers, etc.
        zij, zkl : int
            Tomographic bin pair indices
        ind_ab, ind_cd : np.ndarray
            Arrays mapping pair indices to tomographic bin pairs
        func : callable
            Covariance function to compute (e.g., cov_sva_simps, cov_mix_simps)
        kernel_builder : callable
            Child-specific function that builds projection kernels.
            Signature: kernel_builder(scale_ix_1, scale_ix_2, **kwargs) -> (kernel_1, kernel_2)
            Examples:
            - Real space: builds k_mu(ell, theta) partial functions
            - COSEBIs: builds W_n(ell) lambda functions
        **kwargs : dict
            Additional arguments containing:
            - Data: probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, cl_5d, ells, Amax
            - Projection-specific: mu, nu, w_ells_arr, kernel_1_func, kernel_2_func, etc.

        Returns
        -------
        tuple
            (scale_ix_1, scale_ix_2, zi, zj, zk, zl, cov_value)
            Indices and computed covariance value for this combination
        """
        # Extract tomographic bin indices
        zi, zj = ind_ab[zij, :]
        zk, zl = ind_cd[zkl, :]

        # Build projection-specific kernels
        kernel_1 = self.build_projection_kernel(
            scale_ix=scale_ix_1,
            obs_space=self.obs_space,
            mu=kernel_builder_func_kw['mu'],
            kernel_func_kw=kernel_builder_func_kw,
        )
        kernel_2 = self.build_projection_kernel(
            scale_ix=scale_ix_2,
            obs_space=self.obs_space,
            mu=kernel_builder_func_kw['nu'],
            kernel_func_kw=kernel_builder_func_kw,
        )

        # Update kwargs with the constructed kernels
        cov_func_kw['kernel_1_func_of_ell'] = kernel_1
        cov_func_kw['kernel_2_func_of_ell'] = kernel_2

        # Compute covariance value
        cov_value = cov_func(zi=zi, zj=zj, zk=zk, zl=zl, **cov_func_kw)

        return (scale_ix_1, scale_ix_2, zi, zj, zk, zl, cov_value)

    def cov_simps_wrapper(
        self,
        zpairs_ab: np.ndarray,
        zpairs_cd: np.ndarray,
        ind_ab: np.ndarray,
        ind_cd: np.ndarray,
        cov_simps_func: Callable,
        cov_simps_func_kw: dict,
        kernel_builder_func_kw: dict,
    ) -> np.ndarray:
        """Helper to parallelize the cov_sva_simps and cov_mix_simps functions
        s1/s2 is the first scale index (e.g., theta or the COSEBIs mode)
        """
        cov_out_6d = np.zeros(self.cov_shape_6d)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.cov_parallel_helper)(
                scale_ix_1=s1,
                scale_ix_2=s2,
                zij=zij,
                zkl=zkl,
                ind_ab=ind_ab,
                ind_cd=ind_cd,
                cov_func=cov_simps_func,
                cov_func_kw=cov_simps_func_kw,
                kernel_builder_func_kw=kernel_builder_func_kw,
            )
            for s1 in tqdm(range(self.nbx))
            for s2 in range(self.nbx)
            for zij in range(zpairs_ab)
            for zkl in range(zpairs_cd)
        )

        for s1, s2, zi, zj, zk, zl, cov_value in results:
            cov_out_6d[s1, s2, zi, zj, zk, zl] = cov_value

        return cov_out_6d

    def build_projection_kernel(
        self,
        scale_ix: int,
        obs_space: str,
        kernel_func_kw: dict,
        mu: int | None = None,
        arb_kernel_func: Callable | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Based on the scale index (theta for 2PCF, n for COSEBIs) and the observables
        space, contrtuct the projection kernel as a function of ell.

        Parameters
        ----------
        scale_ix : int
            Scale index (theta bin for 2PCF, n for COSEBIs)
        obs_space : str
            Observable space ('real', 'cosebis' or 'arbitrary')
        mu : int | None
            Order of the bessel function for the real space case
        kernel_func_kw : dict
            Keyword arguments for the kernel function
        arb_kernel_func : callable, optional
            Arbitrary kernel function for 'arbitrary' observable space

        Returns
        -------
        kernel_func_of_ell : callable
            Partial kernel functions with signature: kernel(ell)
        """

        if obs_space == 'real':
            # in this case the kernel function is also probe-dependent (through mu)
            theta_l = self.theta_edges_fine[scale_ix]
            theta_u = self.theta_edges_fine[scale_ix + 1]
            kernel_func_of_ell = partial(
                self.k_mu, thetal=theta_l, thetau=theta_u, mu=mu
            )

        elif obs_space == 'cosebis':
            # in this case the kernel function neither probe nor ell-dependent, so I
            # define a simple function of ell that just returns the precomputed array
            w_ells_arr = kernel_func_kw['w_ells_arr']

            def kernel_func_of_ell(ell):
                return w_ells_arr[scale_ix]

        elif obs_space == 'arbitrary':
            # general case. the arbitrary kernel function must have signature
            # arb_kernel_func(ell, *, scale_ix, **kernel_func_kw)
            kernel_func_of_ell = partial(
                arb_kernel_func, scale_ix=scale_ix, **kernel_func_kw
            )

        else:
            raise ValueError(f'Observable space {obs_space} not recognized!')

        return kernel_func_of_ell
