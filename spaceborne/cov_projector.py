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
from scipy.integrate import quad_vec
from scipy.integrate import simpson as simps
from scipy.interpolate import RegularGridInterpolator, make_interp_spline
from tqdm import tqdm

from spaceborne import constants as const

_UNSET = object()


def get_npair(theta_1_u, theta_1_l, survey_area_sr, n_eff_i, n_eff_j):
    """Compute total (ideal) number of pairs in a theta bin, i.e., N(theta).
    N(θ) = π (θ_u^2 - θ_l^2) × A × n_i × n_j
         = \int_{θ_l}^{θ_u} dθ (dN(θ)/dθ)
    """
    n_eff_i_sr = n_eff_i * const.SR_TO_ARCMIN2
    n_eff_j_sr = n_eff_j * const.SR_TO_ARCMIN2
    return (
        np.pi * (theta_1_u**2 - theta_1_l**2) * survey_area_sr * n_eff_i_sr * n_eff_j_sr
    )


def get_dnpair(theta, survey_area_sr, n_eff_i, n_eff_j):
    """Compute differential (ideal) number of pairs, i.e. dN(theta)/dtheta.
    dN(θ)/dθ = 2π θ × A × n_i × n_j
    """
    n_eff_i_sr = n_eff_i * const.SR_TO_ARCMIN2
    n_eff_j_sr = n_eff_j * const.SR_TO_ARCMIN2
    return 2 * np.pi * theta * survey_area_sr * n_eff_i_sr * n_eff_j_sr


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

        # Shared setup
        self.zbins = pvt_cfg['zbins']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.ind_dict = pvt_cfg['ind_dict']
        self.nbx = pvt_cfg['nbx']
        self.n_jobs = cfg['misc']['num_threads']
        self.n_probes_hs = 2

        self._set_survey_info(mask_obj)
        self._set_terms_toloop()
        self._set_neff_and_sigma_eps()

        # TODO here (in the init) I should add the finely binned Cls, which are used in all projections!

        self.cov_shape_6d = (
            self.nbx,
            self.nbx,
            self.zbins,
            self.zbins,
            self.zbins,
            self.zbins,
        )

        self.obs_space = _UNSET

    def _set_survey_info(self, mask_obj):
        """Set up survey geometry information."""
        # TODO generalise to different survey areas (max(Aij, Akl))
        self.survey_area_deg2 = mask_obj.survey_area_deg2
        self.survey_area_sr = mask_obj.survey_area_sr
        self.fsky = mask_obj.fsky
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

    def _set_neff_and_sigma_eps(self):
        self.n_eff_lns = self.cfg['nz']['ngal_lenses']  # clustering
        self.n_eff_src = self.cfg['nz']['ngal_sources']  # lensing
        self.n_eff_2d = np.vstack((self.n_eff_src, self.n_eff_lns))
        self.sigma_eps_i = np.array(self.cfg['covariance']['sigma_eps_i'])

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

        # if not present in kernel_builder_func_kw (e.g. for the COSEBIs case),
        # set mu and nu to None
        mu = kernel_builder_func_kw.get('mu', None)
        nu = kernel_builder_func_kw.get('nu', None)

        # Build projection-specific kernels
        kernel_1 = self.build_projection_kernel(
            scale_ix=scale_ix_1,
            obs_space=self.obs_space,
            mu=mu,
            kernel_func_kw=kernel_builder_func_kw,
        )
        kernel_2 = self.build_projection_kernel(
            scale_ix=scale_ix_2,
            obs_space=self.obs_space,
            mu=nu,
            kernel_func_kw=kernel_builder_func_kw,
        )

        # Update kwargs with the constructed kernels. I instantiate a new dict to
        # avoid problems with parallelization
        local_kw = {
            **cov_func_kw,
            'kernel_1_func_of_ell': kernel_1,
            'kernel_2_func_of_ell': kernel_2,
        }

        # Compute covariance value
        cov_value = cov_func(zi=zi, zj=zj, zk=zk, zl=zl, **local_kw)

        return (scale_ix_1, scale_ix_2, zi, zj, zk, zl, cov_value)

    def cov_simps_wrapper(
        self,
        zpairs_ab: int,
        zpairs_cd: int,
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
        space, construct the projection kernel as a function of ell.

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

    def cov_mix_simps(
        self,
        probe_a_ix: int,
        probe_b_ix: int,
        probe_c_ix: int,
        probe_d_ix: int,
        zi: int,
        zj: int,
        zk: int,
        zl: int,
        kernel_1_func_of_ell: Callable[[np.ndarray], np.ndarray],
        kernel_2_func_of_ell: Callable[[np.ndarray], np.ndarray],
    ):  # fmt: skip
        def integrand_func(ells, inner_integrand):
            k1 = kernel_1_func_of_ell(ells)
            k2 = kernel_2_func_of_ell(ells)
            return (1 / (2 * np.pi * self.amax)) * ells * k1 * k2 * inner_integrand

        def get_prefac(probe_a_ix, probe_b_ix, zi, zj):
            prefac = (
                get_delta_tomo(probe_a_ix, probe_b_ix, self.zbins)[zi, zj]
                * t_mix(probe_a_ix, self.zbins, self.sigma_eps_i)[zi]
                / (self.n_eff_2d[probe_a_ix, zi] * self.srtoarcmin2)
            )
            return prefac

        # permutations should be performed as done in the SVA function
        integrand = integrand_func(
            self.ells_proj_g,
            self.cl_3x2pt_5d[probe_a_ix, probe_c_ix, :, zi, zk]
            * get_prefac(probe_b_ix, probe_d_ix, zj, zl)
            + self.cl_3x2pt_5d[probe_b_ix, probe_d_ix, :, zj, zl]
            * get_prefac(probe_a_ix, probe_c_ix, zi, zk)
            + self.cl_3x2pt_5d[probe_a_ix, probe_d_ix, :, zi, zl]
            * get_prefac(probe_b_ix, probe_c_ix, zj, zk)
            + self.cl_3x2pt_5d[probe_b_ix, probe_c_ix, :, zj, zk]
            * get_prefac(probe_a_ix, probe_d_ix, zi, zl),
        )

        integral = simps(y=integrand, x=self.ells_proj_g)

        # elif integration_method == 'quad':

        #     integral_1 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
        #                           args=(self.cl_3x2pt_5d[probe_a_ix, probe_c_ix, :, zi, zk],))[0]
        #     integral_2 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
        #                           args=(self.cl_3x2pt_5d[probe_b_ix, probe_d_ix, :, zj, zl],))[0]
        #     integral_3 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
        #                           args=(self.cl_3x2pt_5d[probe_a_ix, probe_d_ix, :, zi, zl],))[0]
        #     integral_4 = quad_vec(integrand_scalar, ell_values[0], ell_values[-1],
        #                           args=(self.cl_3x2pt_5d[probe_b_ix, probe_c_ix, :, zj, zk],))[0]

        # else:
        # raise ValueError(f'integration_method {integration_method} '
        # 'not recognized.')

        return integral

    def cov_sva_simps(
        self,
        probe_a_ix: int,
        probe_b_ix: int,
        probe_c_ix: int,
        probe_d_ix: int,
        zi: int,
        zj: int,
        zk: int,
        zl: int,
        kernel_1_func_of_ell: Callable[[np.ndarray], np.ndarray],
        kernel_2_func_of_ell: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """
        Universal Simpson integrator for SVA covariance - projection kernel agnostic.

        This function computes a single matrix element of the SVA covariance by:
        1. Selecting the relevant C_ℓ spectra for the given tomographic bins
        2. Evaluating projection kernels (e.g., k_mu for real space, W_n for COSEBIs)
        3. Building the integrand: ℓ * kernel_1 * kernel_2 * (C_ik*C_jl + C_il*C_jk)
        4. Integrating with Simpson's rule

        Parameters
        ----------
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix : int
            Probe indices
        zi, zj, zk, zl : int
            Tomographic bin indices
        kernel_1_func_of_ell : callable
            First projection kernel function of ℓ (e.g., k_mu(ℓ, theta_1))
        kernel_2_func_of_ell : callable
            Second projection kernel function of ℓ (e.g., k_nu(ℓ, theta_2))

        Returns
        -------
        cov_elem : float
            Single covariance matrix element
        """
        # Extract relevant C_ℓ for these tomographic bins
        c_ik = self.cl_3x2pt_5d[probe_a_ix, probe_c_ix, :, zi, zk]
        c_jl = self.cl_3x2pt_5d[probe_b_ix, probe_d_ix, :, zj, zl]
        c_il = self.cl_3x2pt_5d[probe_a_ix, probe_d_ix, :, zi, zl]
        c_jk = self.cl_3x2pt_5d[probe_b_ix, probe_c_ix, :, zj, zk]

        # Evaluate projection kernels
        # TODO this should probebly be done at init to save time (at the cost of flexibility?)
        kernel_1 = kernel_1_func_of_ell(self.ells_proj_g)
        kernel_2 = kernel_2_func_of_ell(self.ells_proj_g)

        # Build integrand: ℓ * K_μ * K_ν * (C_ik*C_jl + C_il*C_jk)
        integrand = self.ells_proj_g * kernel_1 * kernel_2 * (c_ik * c_jl + c_il * c_jk)

        # Integrate with Simpson's rule
        integral = simps(y=integrand, x=self.ells_proj_g)

        # Apply normalization factor
        return integral / (2.0 * np.pi * self.amax)

    def cov_ng_simps(
        self,
        ells_proj: np.ndarray,
        cov_ng_4d: np.ndarray,
        kernel_1_func_of_ell: Callable[[np.ndarray], np.ndarray],
        kernel_2_func_of_ell: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Simpson integrator for non-Gaussian covariance - projection kernel agnostic.

        This function projects a 4D harmonic-space non-Gaussian covariance to real (TBD)
        or COSEBIs space. It includes the evaluation of the projection kernels
        (e.g., k_mwu for real space, W_n for COSEBIs over the (possibly different)
        ell_1 and ell_2 grids.
        It then buils the integrand: ell_1 * ell_2 * kernel_1 * kernel_2 * cov_hs_ng_4d
        and integrates it with Simpson's rule.

        Parameters
        ----------
        ells_proj : np.ndarray
        cov_ng_4d : np.ndarray
        kernel_1_func_of_ell : Callable[[np.ndarray], np.ndarray]
        kernel_2_func_of_ell : Callable[[np.ndarray], np.ndarray]

        Returns
        -------
        cov_ijkl : np.ndarray
            for the input probe combination and term (ssc or cng), returns the
            zi, zj, zk, zl 4D matrix (no need for parallel wrappers in this case)
        """

        assert cov_ng_4d.ndim == 4, 'input array must be 4D'
        assert cov_ng_4d.shape[0] == cov_ng_4d.shape[1] == len(ells_proj), (
            'cov_ng_4d.shape[0] and cov_ng_4d.shape[1] must match len(ells_proj).'
            f'found cov_ng_4d.shape={cov_ng_4d.shape} and '
            f'len(ells_proj)={len(ells_proj)}'
        )

        # just to keep track, the two grids are the same
        ells_1 = ells_proj
        ells_2 = ells_proj

        # Evaluate projection kernels
        kernel_1 = kernel_1_func_of_ell(ells_1)
        kernel_2 = kernel_2_func_of_ell(ells_2)

        # reshape everything to match the NG cov shape
        integrand = (
            ells_1[:, None, None, None]
            * ells_2[None, :, None, None]
            * kernel_1[:, None, None, None]
            * kernel_2[None, :, None, None]
            * cov_ng_4d
        )

        # integrate along ells_1 and ells_2
        part_integral = simps(y=integrand, x=ells_1, axis=0)
        cov_ijkl = simps(y=part_integral, x=ells_2, axis=0)

        return cov_ijkl

    def cov_ng_quad_vec(
        self,
        ells_proj: np.ndarray,
        cov_ng_4d: np.ndarray,
        kernel_1_func_of_ell: Callable[[np.ndarray], np.ndarray],
        kernel_2_func_of_ell: Callable[[np.ndarray], np.ndarray],
        limit: int = 200,
        epsrel: float = 1.49e-4,
        num_threads: int = 1,
    ) -> np.ndarray:
        """
        Adaptive quad_vec integrator for non-Gaussian covariance projection.

        Same interface as cov_ng_simps. Uses scipy.integrate.quad_vec instead
        of fixed-grid Simpson, exploiting the fact that cov_ng_4d is smooth in
        ell while the projection kernels (Bessel functions, COSEBIS filters) are
        oscillatory. The two concerns are separated:

          - cov_ng_4d is interpolated with a 2D cubic spline (accurate on a
            coarse ells_proj grid because the covariance is smooth).
          - The adaptive integrator resolves the oscillatory kernels internally,
            calling the cheap spline at however many ell points it needs.

        Steps
        -----
        1. Build a 2D cubic interpolator for cov_ng_4d(ell1, ell2).
        2. For each ell1 in ells_proj compute the inner integral
               I(ell1) = ∫ dℓ₂ ℓ₂ K₂(ℓ₂) C(ell1, ℓ₂)
           using quad_vec, vectorised over all n_flat tomo-bin combinations.
        3. Fit a cubic spline to I(ell1) (smooth in ell1).
        4. Compute the outer integral
               result = ∫ dℓ₁ ℓ₁ K₁(ℓ₁) I(ℓ₁)
           with a second quad_vec call.

        Parameters
        ----------
        ells_proj : np.ndarray
        cov_ng_4d : np.ndarray
        kernel_1_func_of_ell : Callable[[np.ndarray], np.ndarray]
        kernel_2_func_of_ell : Callable[[np.ndarray], np.ndarray]
        limit : int
            Maximum number of adaptive subdivisions per quad_vec call.
        epsrel : float
            Relative tolerance for quad_vec.
        num_threads : int
            Number of worker threads passed to quad_vec. Use -1 for all
            available CPUs. Note that the inner quad_vec calls (one per ell1
            grid point) are independent, so threading is most effective there.

        Returns
        -------
        cov_ijkl : np.ndarray
            Shape (zpairs_ab, zpairs_cd), same as cov_ng_simps.
        """
        assert cov_ng_4d.ndim == 4, 'input array must be 4D'
        assert cov_ng_4d.shape[0] == cov_ng_4d.shape[1] == len(ells_proj), (
            'cov_ng_4d.shape[0] and cov_ng_4d.shape[1] must match len(ells_proj). '
            f'found cov_ng_4d.shape={cov_ng_4d.shape} and '
            f'len(ells_proj)={len(ells_proj)}'
        )

        ell_min, ell_max = ells_proj[0], ells_proj[-1]
        nbl = len(ells_proj)
        n_ij = cov_ng_4d.shape[2]
        n_kl = cov_ng_4d.shape[3]
        n_flat = n_ij * n_kl

        # Flatten tomo dims: (nbl, nbl, n_flat). Smooth in ell -> cubic is accurate
        # even on a coarse ells_proj grid.
        cov_flat = cov_ng_4d.reshape(nbl, nbl, n_flat)
        interp_2d = RegularGridInterpolator(
            (ells_proj, ells_proj), cov_flat,
            method='cubic', bounds_error=False, fill_value=0.0,
        )

        # Step 1: inner integral I(ell1) = ∫ dℓ₂ ℓ₂ K₂(ℓ₂) C(ell1, ℓ₂)
        # Evaluated at each ell1 in ells_proj; shape of inner_result: (nbl, n_flat)
        inner_result = np.zeros((nbl, n_flat))
        for i, ell1 in enumerate(ells_proj):
            def _inner(ell2, _ell1=ell1):
                k2 = kernel_2_func_of_ell(np.atleast_1d(ell2))[0]
                cov_val = interp_2d([[_ell1, ell2]])[0]  # shape: (n_flat,)
                return ell2 * k2 * cov_val

            inner_result[i], _ = quad_vec(
                _inner, ell_min, ell_max, limit=limit, epsrel=epsrel,
                workers=num_threads,
            )

        # Step 2: outer integral = ∫ dℓ₁ ℓ₁ K₁(ℓ₁) I(ℓ₁)
        # I(ell1) is smooth -> a cubic spline on ells_proj is sufficient.
        # BSpline with 2D coefficients: spl(scalar) -> shape (n_flat,)
        inner_spline = make_interp_spline(ells_proj, inner_result, k=3)

        def _outer(ell1):
            k1 = kernel_1_func_of_ell(np.atleast_1d(ell1))[0]
            return ell1 * k1 * inner_spline(ell1)  # shape: (n_flat,)

        result, _ = quad_vec(_outer, ell_min, ell_max, limit=limit, epsrel=epsrel,
                             workers=num_threads)
        return result.reshape(n_ij, n_kl)

