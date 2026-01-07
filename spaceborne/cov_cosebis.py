"""
This module contains functions to compute the covariance matrix in real space.
Nomenclature of the functions/variables:
hs = harmonic space
rs = real space
sva = sample variance
sn = sampling noise
mix = mixed term
"""

# TODO the NG cov has not been re-tested against OC
# TODO the NG cov needs a smaller number of ell bins for the simpson integration! It's
# TODO unpractical to compute it in 1000 ell values

import itertools
import warnings
from collections.abc import Callable
from functools import partial

import cloelib.auxiliary.cosebi_helpers as ch
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import simps
from tqdm import tqdm

from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import cov_real_space as crs
from spaceborne import sb_lib as sl

warnings.filterwarnings(
    'ignore', message=r'.*invalid escape sequence.*', category=SyntaxWarning
)

warnings.filterwarnings(
    'ignore',
    message=r'.*invalid value encountered in divide.*',
    category=RuntimeWarning,
)

_UNSET = object()


class CovCOSEBIs:
    def __init__(self, cfg, pvt_cfg, mask_obj):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg
        self.mask_obj = mask_obj

        self.zbins = pvt_cfg['zbins']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']
        self.zpairs_3x2pt = pvt_cfg['zpairs_3x2pt']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.ind_dict = pvt_cfg['ind_dict']
        self.cov_ordering_2d = pvt_cfg['cov_ordering_2d']
        self.n_modes = cfg['binning']['n_modes_cosebis']

        # instantiate cov dict with the required terms and probe combinations
        self.req_terms = pvt_cfg['req_terms']
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_cs_2d']
        dims = ['6d', '4d', '2d']

        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='cosebis')
            for probe in self.req_probe_combs_2d
        ]
        _req_probe_combs_2d.append('3x2pt')
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

        # setters
        self._set_survey_info()
        self._set_theta_binning()
        self._set_neff_and_sigma_eps()
        self._set_terms_toloop()

        # other miscellaneous settings
        self.n_jobs = self.cfg['misc']['num_threads']
        self.integration_method = self.cfg['precision']['cov_rs_int_method']

        assert self.integration_method in ['simps', 'levin'], (
            'integration method not implemented'
        )

        self.cov_cs_6d_shape = (
            self.n_modes, self.n_modes, self.zbins, self.zbins, self.zbins, self.zbins
            )  # fmt: skip

        # attributes set at runtime
        self.cl_3x2pt_5d = _UNSET
        self.ells = _UNSET
        self.nbl = _UNSET

    def _set_survey_info(self):
        self.survey_area_deg2 = self.mask_obj.survey_area_deg2
        self.survey_area_sr = self.mask_obj.survey_area_sr
        self.fsky = self.mask_obj.fsky
        self.srtoarcmin2 = const.SR_TO_ARCMIN2
        # maximum survey area in sr
        # TODO generalise to multiple survey areas
        self.amax = max((self.survey_area_sr, self.survey_area_sr))

    def _set_theta_binning(self):
        """Set the theta binning for the COSEBIs SN term integral."""
        self.theta_min_arcmin = self.cfg['precision']['theta_min_arcmin_cosebis']
        self.theta_max_arcmin = self.cfg['precision']['theta_max_arcmin_cosebis']
        self.nbt = self.cfg['precision']['theta_steps_cosebis']

        # ! new
        self.theta_min_arcmin = self.cfg['precision']['theta_min_arcmin_cosebis']
        self.theta_max_arcmin = self.cfg['precision']['theta_max_arcmin_cosebis']
        self.nbt = self.cfg['precision']['theta_steps_cosebis']

        # Convert to radians
        self.theta_min_rad = np.deg2rad(self.theta_min_arcmin / 60)
        self.theta_max_rad = np.deg2rad(self.theta_max_arcmin / 60)

        # No need for bin edges in this case, I can directly do this:
        self.theta_grid_rad = np.geomspace(
            self.theta_min_rad, self.theta_max_rad, self.nbt
        )

        # sanity checks
        assert len(self.theta_grid_rad) == self.nbt, 'theta_grid_rad length mismatch'
        assert np.min(np.diff(self.theta_grid_rad)) > 0, 'theta_grid_rad not sorted!'

    def _set_neff_and_sigma_eps(self):
        self.n_eff_lns = self.cfg['nz']['ngal_lenses']
        self.n_eff_src = self.cfg['nz']['ngal_sources']
        # ! old
        # self.n_eff_2d = np.row_stack((self.n_eff_lns, self.n_eff_lns, self.n_eff_src))
        # ! new
        self.n_eff_2d = np.row_stack((self.n_eff_src, self.n_eff_src, self.n_eff_lns))

        self.sigma_eps_i = np.array(self.cfg['covariance']['sigma_eps_i'])

    def _set_terms_toloop(self):
        self.terms_toloop = []
        if self.cfg['covariance']['G']:
            self.terms_toloop.extend(('sva', 'sn', 'mix'))
        if self.cfg['covariance']['SSC']:
            self.terms_toloop.append('ssc')
        if self.cfg['covariance']['cNG']:
            self.terms_toloop.append('cng')

    def set_w_ells(self):
        """
        Compute and set the COSEBIs W_n(ell) kernels for all modes and ells.

        Sets:
            self.w_ells_arr (np.ndarray): Array of shape (n_modes, n_ells) containing
                the computed W_n(ell) kernel values.
        """

        with sl.timer(
            f'Computing COSEBIs W_n(ell) kernels for {self.n_modes} modes...'
        ):
            w_ells_dict = ch.get_W_ell(
                thetagrid=self.theta_grid_rad,
                Nmax=self.n_modes,
                ells=self.ells,
                N_thread=self.n_jobs,
            )

        # turn to array of shape (n_modes, n_ells) and assign to self
        self.w_ells_arr = np.array(list(w_ells_dict.values()))

    def cov_sn_cs(self):
        """Compute the COSEBIs shape noise covariance term."""

        # firstly, construct the prefactor outside of the \theta integral
        first_term = np.einsum('i,j->ij', self.sigma_eps_i**2, self.sigma_eps_i**2) / 2
        kron = np.eye(self.zbins)
        second_term = np.einsum('ik,jl->ijkl', kron, kron) + np.einsum(
            'il,jk->ijkl', kron, kron
        )
        prefactor = first_term[:, :, None, None] * second_term

        # 1. Compute T_minus and T_plus
        t_minus = np.zeros((self.nbt, self.n_modes))
        t_plus = np.zeros((self.nbt, self.n_modes))

        rn, nn, coeff_j = ch.get_roots_and_norms(
            tmax=self.theta_max_rad, tmin=self.theta_min_rad, Nmax=self.n_modes
        )

        for n in range(self.n_modes):
            t_minus[:, n] = ch.tm(
                n=n + 1,
                t=self.theta_grid_rad,
                tmin=self.theta_min_rad,
                nn=nn,
                coeff_j=coeff_j,
            )
            t_plus[:, n] = ch.tp(
                n=n + 1, t=self.theta_grid_rad, tmin=self.theta_min_rad, nn=nn, rn=rn
            )
            # convert the mp.math object to normal floats
            t_minus[:, n] = np.array([float(x) for x in t_minus[:, n]])
            t_plus[:, n] = np.array([float(x) for x in t_plus[:, n]])

        # construct term in square brackets: [𝑇+𝑎(𝜃)𝑇+𝑏(𝜃) + 𝑇−𝑎(𝜃)𝑇−𝑏(𝜃)]
        # t is the theta index, a and b the mode indices
        t_term_1 = np.einsum('ta, tb -> tab', t_plus, t_plus)
        t_term_2 = np.einsum('ta, tb -> tab', t_minus, t_minus)
        t_term = t_term_1 + t_term_2  # shape (self.nbt, n_modes, n_modes)

        # 3. Compute dnpair (differential pairs per unit angle)
        # TODO IMPORTANT: nz src or lens below?
        npair_arr = np.zeros((self.nbt, self.zbins, self.zbins))
        for theta_ix, zi, zj in itertools.product(
            range(self.nbt), range(self.zbins), range(self.zbins)
        ):
            npair_arr[theta_ix, zi, zj] = crs.get_dnpair(
                theta=self.theta_grid_rad[theta_ix],
                survey_area_sr=self.survey_area_sr,
                n_eff_i=self.n_eff_src[zi],
                n_eff_j=self.n_eff_src[zj],
            )

        # * alternatively, you can do
        # for theta_ix, zi, zj in itertools.product(
        #     range(self.nbt), range(self.zbins), range(self.zbins)
        # ):
        #     npair_arr[theta_ix, zi, zj] = crs.get_npair(
        #         theta_1_u=self.theta_edges_rad[theta_ix],
        #         theta_1_l=self.theta_edges_rad[theta_ix + 1],
        #         survey_area_sr=self.survey_area_sr,
        #         n_eff_i=self.n_eff_src[zi],
        #         n_eff_j=self.n_eff_src[zj],
        #     )
        # dtheta = np.diff(self.theta_edges_rad)
        # npair_arr /= dtheta[:, None, None]

        # 4. Broadcast shapes, construct integrand and integrate
        integrand = (
            self.theta_grid_rad[:, None, None, None, None] ** 2
            * t_term[:, :, :, None, None]
            / npair_arr[:, None, None, :, :]
        )

        integral = simps(y=integrand, x=self.theta_grid_rad, axis=0)

        # overall shape is (n_modes, n_modes, zbins, zbins, zbins, zbins)
        return integral[:, :, :, :, None, None] * prefactor[None, None, :, :, :, :]

    def get_delta_tomo(self, probe_a_ix, probe_b_ix):
        if probe_a_ix == probe_b_ix:
            return np.eye(self.zbins)
        else:
            return np.zeros((self.zbins, self.zbins))

    def cov_simps_wrapper(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
        zpairs_ab, zpairs_cd, ind_ab, ind_cd, mu, nu, 
        cov_simps_func: Callable, 
        kernel_1_func: Callable, kernel_2_func: Callable
    ):  # fmt: skip
        """Helper to parallelize the cov_sva_simps and cov_mix_simps functions"""
        cov_rs_6d = np.zeros(self.cov_rs_6d_shape)

        kwargs = {
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
            'cl_5d': self.cl_3x2pt_5d,
            'ells': self.ells,
            'Amax': self.amax,
            'kernel_1_func': kernel_1_func,
            'kernel_2_func': kernel_2_func,
        }

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.cov_parallel_helper)(
                theta_1_ix=theta_1_ix, theta_2_ix=theta_2_ix, mu=mu, nu=nu,
                zij=zij, zkl=zkl, ind_ab=ind_ab, ind_cd=ind_cd,
                func=cov_simps_func,
                **kwargs,
            )
            for theta_1_ix in tqdm(range(self.nbt_fine))
            for theta_2_ix in range(self.nbt_fine)
            for zij in range(zpairs_ab)
            for zkl in range(zpairs_cd)
        )  # fmt: skip

        for theta_1, theta_2, zi, zj, zk, zl, cov_value in results:
            cov_rs_6d[theta_1, theta_2, zi, zj, zk, zl] = cov_value

        return cov_rs_6d

    def cov_parallel_helper(
        self, theta_1_ix, theta_2_ix, mu, nu, zij, zkl, ind_ab, ind_cd, func, **kwargs
    ):
        """This is the function actually called in parallel with joblib. It essentially
        extract the theta and z indices and calls the provided function to compute the
        covariance value for those indices."""
        theta_1_l = self.theta_edges_fine[theta_1_ix]
        theta_1_u = self.theta_edges_fine[theta_1_ix + 1]
        theta_2_l = self.theta_edges_fine[theta_2_ix]
        theta_2_u = self.theta_edges_fine[theta_2_ix + 1]

        zi, zj = ind_ab[zij, :]
        zk, zl = ind_cd[zkl, :]

        # TODO what happens if the kernel functions need more or less args
        # take the full kernel functions, pass the theta and mu/nu args and make them
        # partial functions of ell only
        kernel_1_full = kwargs['kernel_1_func']
        kernel_2_full = kwargs['kernel_2_func']

        kernel_1_partial = partial(
            kernel_1_full, thetal=theta_1_l, thetau=theta_1_u, mu=mu
        )
        kernel_2_partial = partial(
            kernel_2_full, thetal=theta_2_l, thetau=theta_2_u, mu=nu
        )

        kwargs['kernel_1_func'] = kernel_1_partial
        kwargs['kernel_2_func'] = kernel_2_partial

        return (
            theta_1_ix,
            theta_2_ix,
            zi,
            zj,
            zk,
            zl,
            func(zi=zi, zj=zj, zk=zk, zl=zl),
        )

    def cov_cosebis_parallel_helper(
        self, mode_n, mode_m, zij, zkl, ind_ab, ind_cd, func, w_ells_arr, **kwargs
    ):
        """Parallel helper for COSEBIs - no theta dependence, just mode indices.

        Parameters
        ----------
        mode_n : int
            First COSEBIs mode index
        mode_m : int
            Second COSEBIs mode index
        zij : int
            Index for first tomographic bin pair
        zkl : int
            Index for second tomographic bin pair
        ind_ab : np.ndarray
            Array of tomographic indices for first probe pair
        ind_cd : np.ndarray
            Array of tomographic indices for second probe pair
        func : callable
            Covariance function to call (e.g., cov_sva_simps)
        w_ells_arr : np.ndarray
            Array of W_n(ell) arrays, shape (n_modes, n_ells)
        **kwargs : dict
            Additional arguments to pass to func
        """
        zi, zj = ind_ab[zij, :]
        zk, zl = ind_cd[zkl, :]

        # Create mode-specific kernel callables (no theta dependence!)
        # w_ells_arr[mode_n] should already be evaluated on self.ells grid
        kernel_n = lambda ell: w_ells_arr[mode_n]
        kernel_m = lambda ell: w_ells_arr[mode_m]

        kwargs['kernel_1_func'] = kernel_n
        kwargs['kernel_2_func'] = kernel_m

        return (
            mode_n,
            mode_m,
            zi,
            zj,
            zk,
            zl,
            func(zi=zi, zj=zj, zk=zk, zl=zl, **kwargs),
        )

    def cov_cosebis_wrapper(
        self,
        probe_a_ix,
        probe_b_ix,
        probe_c_ix,
        probe_d_ix,
        zpairs_ab,
        zpairs_cd,
        ind_ab,
        ind_cd,
        w_ells_arr: np.ndarray,
        n_modes,
        cov_func,
    ):
        """Wrapper for COSEBIs covariance - loops over modes instead of theta bins.

        Parameters
        ----------
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix : int
            Probe indices (0 for shear, 1 for galaxy clustering)
        zpairs_ab : int
            Number of tomographic pairs for first probe combination
        zpairs_cd : int
            Number of tomographic pairs for second probe combination
        ind_ab : np.ndarray
            Array of tomographic indices for first probe pair
        ind_cd : np.ndarray
            Array of tomographic indices for second probe pair
        w_ells_arr : np.ndarray
            Array of W_n(ell) arrays, shape (n_modes, n_ells)
        n_modes : int
            Number of COSEBIs modes
        cov_func : callable
            Covariance function to compute (e.g., cov_sva_simps)

        Returns
        -------
        cov_cosebis_6d : np.ndarray
            Covariance array with shape (n_modes, n_modes, zbins, zbins, zbins, zbins)
        """
        cov_cosebis_6d = np.zeros(
            (n_modes, n_modes, self.zbins, self.zbins, self.zbins, self.zbins)
        )

        kwargs = {
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
            'cl_5d': self.cl_3x2pt_5d,
            'ells': self.ells,
            'Amax': self.amax,
            'w_ells_arr': w_ells_arr,
        }

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.cov_cosebis_parallel_helper)(
                mode_n=mode_n, mode_m=mode_m,
                zij=zij, zkl=zkl, ind_ab=ind_ab, ind_cd=ind_cd,
                func=cov_func,
                **kwargs,
            )
            for mode_n in tqdm(range(n_modes), desc='COSEBIs modes')
            for mode_m in range(n_modes)
            for zij in range(zpairs_ab)
            for zkl in range(zpairs_cd)
        )  # fmt: skip

        for mode_n, mode_m, zi, zj, zk, zl, cov_value in results:
            cov_cosebis_6d[mode_n, mode_m, zi, zj, zk, zl] = cov_value

        return cov_cosebis_6d

    def compute_cs_cov_term_probe_6d(self, cov_hs_obj, probe_abcd, term):
        """
        Computes the COSEBIs covariance matrix for the specified term and probe combination.

        Parameters
        ----------
        probe_abcd : str
            Probe combination string (e.g., 'xipxip')
        term : str
            Covariance term to compute ('sva', 'mix', 'sn')
        theta_min : float
            Minimum angular separation in arcmin
        theta_max : float
            Maximum angular separation in arcmin
        n_modes : int
            Number of COSEBIs modes to compute
        n_threads : int, optional
            Number of threads for pylevin integration (default: 1)

        Returns
        -------
        cov_cosebis_6d : np.ndarray
            COSEBIs covariance with shape (n_modes, n_modes, zbins, zbins, zbins, zbins)
        """
        probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'cosebis')
        probe_2tpl = (probe_ab, probe_cd)

        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = const.CS_PROBE_NAME_TO_IX_DICT[
            probe_abcd
        ]

        ind_ab = (
            self.ind_auto[:, 2:] if probe_a_ix == probe_b_ix else self.ind_cross[:, 2:]
        )
        ind_cd = (
            self.ind_auto[:, 2:] if probe_c_ix == probe_d_ix else self.ind_cross[:, 2:]
        )

        zpairs_ab = self.zpairs_auto if probe_a_ix == probe_b_ix else self.zpairs_cross
        zpairs_cd = self.zpairs_auto if probe_c_ix == probe_d_ix else self.zpairs_cross

        # Create a theta grid for computing the Hankel transform
        # You may want to use a finer grid than self.theta_centers_fine

        # Compute covariance based on term
        if term == 'sva':
            if 'Bn' in probe_2tpl:
                cov_out_6d = np.zeros(self.cov_cs_6d_shape)
            else:
                cov_out_6d = self.cov_cosebis_wrapper(
                    probe_a_ix,
                    probe_b_ix,
                    probe_c_ix,
                    probe_d_ix,
                    zpairs_ab,
                    zpairs_cd,
                    ind_ab,
                    ind_cd,
                    w_ells_arr=self.w_ells_arr,
                    n_modes=self.n_modes,
                    cov_func=crs.cov_sva_simps,
                )

        # TODO understand this
        # elif term == 'mix' and probe_abcd not in ['ggxim', 'ggxip']:
        elif term == 'mix':
            if 'Bn' in probe_2tpl:
                cov_out_6d = np.zeros(self.cov_cs_6d_shape)
            else:
                cov_out_6d = self.cov_cosebis_wrapper(
                    probe_a_ix,
                    probe_b_ix,
                    probe_c_ix,
                    probe_d_ix,
                    zpairs_ab,
                    zpairs_cd,
                    ind_ab,
                    ind_cd,
                    w_ells_arr=self.w_ells_arr,
                    n_modes=self.n_modes,
                    cov_func=partial(crs.cov_mix_simps, self=self),
                )

        elif term == 'mix' and probe_abcd in ['ggxim', 'ggxip']:
            cov_out_6d = np.zeros(self.cov_cs_6d_shape)

        elif term == 'sn' and probe_ab == probe_cd:
            cov_out_6d = self.cov_sn_cs()
        # TODO these ifs are not very nice...
        elif term == 'sn' and probe_ab != probe_cd:
            cov_out_6d = np.zeros(self.cov_cs_6d_shape)

        else:
            raise ValueError(
                f'Term {term} not recognized or not implemented for COSEBIs'
            )

        self.cov_dict[term][probe_2tpl]['6d'] = cov_out_6d
