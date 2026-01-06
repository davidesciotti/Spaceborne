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
        self.n_modes = cfg['precision']['n_modes_cosebis']
        # TODO decide where to put this: is it used elsewhere?
        self.theta_bins_sn = 1000

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

    def set_cov_2d_ordering(self):
        # settings for 2D covariance ordering
        if self.cov_ordering_2d == 'probe_scale_zpair':
            self.block_index = 'ell'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt_rs
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'zbins': self.zbins,
                'req_probe_combs_2d': self.req_probe_combs_2d,
            }
        elif self.cov_ordering_2d == 'probe_zpair_scale':
            self.block_index = 'zpair'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt_rs
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'zbins': self.zbins,
                'req_probe_combs_2d': self.req_probe_combs_2d,
            }
        elif self.cov_ordering_2d == 'scale_probe_zpair':
            self.block_index = 'ell'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2D
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'optimize': True,
            }
        elif self.cov_ordering_2d == 'zpair_probe_scale':
            self.block_index = 'zpair'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2D
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'optimize': True,
            }
        else:
            raise ValueError(f'Unknown 2D cov ordering: {self.cov_ordering_2d}')

    def _set_survey_info(self):
        self.survey_area_deg2 = self.mask_obj.survey_area_deg2
        self.survey_area_sr = self.mask_obj.survey_area_sr
        self.fsky = self.mask_obj.fsky
        self.srtoarcmin2 = const.SR_TO_ARCMIN2
        # maximum survey area in sr
        # TODO generalise to multiple survey areas
        self.amax = max((self.survey_area_sr, self.survey_area_sr))

    def _set_theta_binning(self):
        self.theta_min_arcmin = self.cfg['binning']['theta_min_arcmin']
        self.theta_max_arcmin = self.cfg['binning']['theta_max_arcmin']
        self.nbt_coarse = self.cfg['binning']['theta_bins']
        self.nbt_fine = self.nbt_coarse

        # TODO this should probably go in the ell_binning class (which should be
        # TODO renamed)
        if self.cfg['binning']['binning_type'] == 'log':
            _binning_func = np.geomspace
        elif self.cfg['binning']['binning_type'] == 'lin':
            _binning_func = np.linspace
        else:
            raise ValueError(
                f'Binning type: {self.cfg["binning"]["binning_type"]} '
                'not supported for real-space covariance'
            )

        # Use a loop to set up fine and coarse theta binning
        for bin_type in ['fine', 'coarse']:
            nbt = getattr(self, f'nbt_{bin_type}')
            theta_edges_deg = _binning_func(
                self.theta_min_arcmin / 60, self.theta_max_arcmin / 60, nbt + 1
            )
            theta_edges = np.deg2rad(theta_edges_deg)  # in radians
            theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
            setattr(self, f'theta_edges_{bin_type}', theta_edges)
            setattr(self, f'theta_centers_{bin_type}', theta_centers)
            assert len(theta_centers) == nbt, 'theta_centers length mismatch'

    def _set_neff_and_sigma_eps(self):
        self.n_eff_lens = self.cfg['nz']['ngal_lenses']
        self.n_eff_src = self.cfg['nz']['ngal_sources']
        # in this way the indices correspond to xip, xim, g
        self.n_eff_2d = np.row_stack((self.n_eff_lens, self.n_eff_lens, self.n_eff_src))

        self.sigma_eps_i = np.array(self.cfg['covariance']['sigma_eps_i'])
        self.sigma_eps_tot = self.sigma_eps_i * np.sqrt(2)

    def _set_terms_toloop(self):
        self.terms_toloop = []
        if self.cfg['covariance']['G']:
            self.terms_toloop.extend(('sn',))  # TODO restore
            # self.terms_toloop.extend(('sva', 'sn', 'mix'))
        if self.cfg['covariance']['SSC']:
            self.terms_toloop.append('ssc')
        if self.cfg['covariance']['cNG']:
            self.terms_toloop.append('cng')

    def cov_sn_cs(self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix):
        # import ipdb

        # ipdb.set_trace()

        # firstly, construct the prefactor outside of the \theta integral
        first_term = np.einsum('i,j->ij', self.sigma_eps_i**2, self.sigma_eps_i**2) / 2
        kron = np.eye(self.zbins)
        second_term = np.einsum('ik,jl->ijkl', kron, kron) + np.einsum(
            'il,jk->ijkl', kron, kron
        )
        prefactor = first_term[:, :, None, None] * second_term

        # Compute T_minus and T_plus
        # 1. create integration grid in theta (use fine grid)
        theta_edges_deg = np.geomspace(
            self.theta_min_arcmin / 60,
            self.theta_max_arcmin / 60,
            self.theta_bins_sn + 1,
        )
        theta_edges_rad = np.deg2rad(theta_edges_deg)  # in radians
        theta_centers_rad = (theta_edges_rad[:-1] + theta_edges_rad[1:]) / 2.0

        assert np.diff(theta_centers_rad).min() > 0, 'theta grid not sorted!'
        theta_min_rad = theta_edges_rad[0]
        theta_max_rad = theta_edges_rad[-1]
        thetas = theta_centers_rad

        # 2. compute the T terms
        rn, nn, coeff_j = ch.get_roots_and_norms(
            theta_max_rad, theta_min_rad, self.n_modes
        )

        t_minus = np.zeros((len(thetas), self.n_modes))
        t_plus = np.zeros((len(thetas), self.n_modes))
        for n in range(self.n_modes):
            t_minus[:, n] = ch.tm(
                n=n + 1, t=thetas, tmin=theta_min_rad, nn=nn, coeff_j=coeff_j
            )
            # convert the mp.math object to normal floats
            t_minus[:, n] = np.array([float(x) for x in t_minus[:, n]])

            # Evaluate T_n^+ on integration grid
            t_plus[:, n] = ch.tp(n=n + 1, t=thetas, tmin=theta_min_rad, nn=nn, rn=rn)
            t_plus[:, n] = np.array([float(x) for x in t_plus[:, n]])

        # construct term in square brackets: [𝑇+𝑎(𝜃)𝑇+𝑏(𝜃) + 𝑇−𝑎(𝜃)𝑇−𝑏(𝜃)]
        # t is the theta index, a and b the mode indices
        t_term_1 = np.einsum('ta, tb -> tab', t_plus, t_plus)
        t_term_2 = np.einsum('ta, tb -> tab', t_minus, t_minus)
        t_term = t_term_1 + t_term_2  # shape (len(thetas), n_modes, n_modes)

        # 3. Compute npair
        # TODO IMPORTANT: nz src or lens below?
        npair_arr = np.zeros((self.theta_bins_sn, self.zbins, self.zbins))
        for theta_ix in range(self.theta_bins_sn):
            for zi in range(self.zbins):
                for zj in range(self.zbins):
                    theta_1_l = theta_edges_rad[theta_ix]
                    theta_1_u = theta_edges_rad[theta_ix + 1]
                    npair_arr[theta_ix, zi, zj] = crs.get_npair(
                        theta_1_u,
                        theta_1_l,
                        self.survey_area_sr,
                        self.n_eff_src[zi],
                        self.n_eff_src[zj],
                    )

        # 1. Calculate bin widths (dtheta) in RADIANS
        # Assuming theta_edges_rad defines the bins for your npair counts
        dtheta = np.diff(theta_edges_rad)

        # 3. Convert Counts (N) to Density (n)
        # n_pair(theta) = N_count / dtheta
        npair_arr /= dtheta[:, None, None]

        # import ipdb; ipdb.set_trace()

        # 4. Broadcast shapes, construct integrand and integrate
        integrand = (
            thetas[:, None, None, None, None] ** 2
            * t_term[:, :, :, None, None]
            / npair_arr[:, None, None, :, :]
        )

        integral = simps(y=integrand, x=thetas, axis=0)

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

    def _sum_split_g_terms_allprobeblocks_alldims(self) -> None:
        # small sanity check probe combinations must match for terms (sva, sn, mix)
        if not (
            self.cov_dict['sva'].keys()
            == self.cov_dict['sn'].keys()
            == self.cov_dict['mix'].keys()
        ):
            raise ValueError(
                'The probe combinations keys in the SVA, SN and MIX covariance '
                'dictionaries do not match!'
            )

        # sanity check: all the probes must match
        probes_sva = set(self.cov_dict['sva'].keys())
        probes_sn = set(self.cov_dict['sn'].keys())
        probes_mix = set(self.cov_dict['mix'].keys())
        if not (probes_sva == probes_sn == probes_mix):
            raise ValueError(
                'The probe combinations in the SVA, SN and MIX covariance '
                'dictionaries do not match!'
            )

        # now sum the terms to get the Gaussian, for all probe combinations and
        # dimensions
        for probe_2tpl in self.cov_dict['sva']:
            if probe_2tpl == '3x2pt':
                continue  # skip 3x2pt, built later

            # sanity check: all the dimensions must match
            dims_sva = set(self.cov_dict['sva'][probe_2tpl].keys())
            dims_sn = set(self.cov_dict['sn'][probe_2tpl].keys())
            dims_mix = set(self.cov_dict['mix'][probe_2tpl].keys())
            if not (dims_sva == dims_sn == dims_mix):
                raise ValueError(
                    'The probe combinations in the SVA, SN and MIX covariance '
                    'dictionaries do not match!'
                )

            # for each dim, perform the sum
            for dim in ['2d', '4d', '6d']:
                self.cov_dict['g'][probe_2tpl][dim] = (
                    self.cov_dict['sva'][probe_2tpl][dim]
                    + self.cov_dict['sn'][probe_2tpl][dim]
                    + self.cov_dict['mix'][probe_2tpl][dim]
                )

    def _build_cov_3x2pt_4d_and_2d(self) -> None:
        """For each covariance term, constructs the 4d and 2d 3x2pt covs from
        the 6d probe-specific ones.

        Note: remember that there is no 6d 3x2pt 6d or 10d cov!

        Note: This exact same function is also defined in cov_harmonic_space.py
        """

        # TODO deprecate this func

        for term in self.cov_dict:
            if term == 'tot':
                continue  # tot is built at the end, skip it

            self.cov_dict[term]['3x2pt']['4d'] = (
                sl.cov_dict_4d_probeblocks_to_3x2pt_4d_array(
                    self.cov_dict[term], obs_space='real'
                )
            )
            self.cov_dict[term]['3x2pt']['2d'] = self.cov_4D_to_2D_3x2pt_func(
                self.cov_dict[term]['3x2pt']['4d'], **self.cov_4D_to_2D_3x2pt_func_kw
            )

        # this function modifies the cov_dict in place, no need to reassign the result
        # to self.cov_dict
        sl.set_cov_tot_2d_and_6d(
            cov_dict=self.cov_dict,
            req_probe_combs_2d=self.req_probe_combs_2d,
            space='real',
        )

    def combine_terms_and_probes(self, unique_probe_combs):
        """For all the required terms, constructs the 3x2pt
        (or nx2pt, depending on the n required probes) 2D cov,
        taking into account the required probe combinations
        (this is taken care of by cov_4D_to_2DCLOE_3x2pt_rs).
        sack (join) probes into a single 2D cov (for each term) and store it in the
        object"""

        # ! construct 3x2pt 2D cov for each term and store them in the object
        for term in self.terms_toloop:
            # first construct the dict
            cov_term_3x2pt_4d_dict = self.build_cov_3x2pt_8d_dict(
                self.req_probe_combs_2d, term
            )
            # then turn to 4D array
            cov_term_3x2pt_4d_arr = sl.cov_3x2pt_8D_dict_to_4D(
                cov_term_3x2pt_4d_dict, self.req_probe_combs_2d, space='real'
            )
            # then to 2D array
            cov_term_3x2pt_2d_arr = self.cov_4D_to_2D_3x2pt_func(
                cov_term_3x2pt_4d_arr, **self.cov_4D_to_2D_3x2pt_func_kw
            )
            # set attribute
            setattr(self, f'cov_3x2pt_{term}_2d', cov_term_3x2pt_2d_arr)

        # ! sum terms to get G and TOT 2D 3x2pt covs and store them in the object
        self.cov_3x2pt_g_2d = sum(
            getattr(self, f'cov_3x2pt_{term}_2d') for term in ['sva', 'sn', 'mix']
        )
        self.cov_3x2pt_tot_2d = sum(
            getattr(self, f'cov_3x2pt_{term}_2d') for term in self.terms_toloop
        )

        for probe in unique_probe_combs:
            # ! sum to get G and TOT 2D probe-specific covs and store them in the object
            # ! (not needed in this new "approach" to the files I wish to save)
            # cov_probe_g_2d = sum(
            #     getattr(self, f'cov_{probe}_{term}_2d') for term in ['sva', 'sn', 'mix']
            # )
            # cov_probe_tot_2d = sum(
            #     getattr(self, f'cov_{probe}_{term}_2d') for term in self.terms_toloop
            # )
            # setattr(self, f'cov_{probe}_g_2d', cov_probe_g_2d)
            # setattr(self, f'cov_{probe}_tot_2d', cov_probe_tot_2d)

            # ! sum terms to get, G, TOT 6D probe-specific covs
            # ! and store them in the object (required if save_full_cov is True).
            # ! note that the 6D covs are already computed and stored in the object
            # ! in the compute_realspace_cov function
            cov_probe_g_6d = sum(
                getattr(self, f'cov_{probe}_{term}_6d') for term in ['sva', 'sn', 'mix']
            )
            cov_probe_tot_6d = sum(
                getattr(self, f'cov_{probe}_{term}_6d') for term in self.terms_toloop
            )
            setattr(self, f'cov_{probe}_g_6d', cov_probe_g_6d)
            setattr(self, f'cov_{probe}_tot_6d', cov_probe_tot_6d)

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
            cov_out_6d = self.cov_sn_cs(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix)
        # TODO these ifs are not very nice...
        elif term == 'sn' and probe_ab != probe_cd:
            cov_out_6d = np.zeros(self.cov_cs_6d_shape)

        else:
            raise ValueError(
                f'Term {term} not recognized or not implemented for COSEBIs'
            )

        self.cov_dict[term][probe_2tpl]['6d'] = cov_out_6d

    def _cov_probeblocks_6d_to_4d_and_2d(self, term):
        """
        For the input term, transforms all 6d probe-blocks into 4d and 2d.
        Note: this does not apply to 3x2pt!
        """

        for probe_2tpl in self.cov_dict[term]:
            if probe_2tpl == '3x2pt':
                continue  # skip 3x2pt, handled elsewhere

            probe_abcd = probe_2tpl[0] + probe_2tpl[1]

            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = (
                const.CS_PROBE_NAME_TO_IX_DICT[probe_abcd]
            )

            ind_ab = (
                self.ind_auto[:, 2:] if probe_a_ix == probe_b_ix
                else self.ind_cross[:, 2:]
            )  # fmt: skip
            ind_cd = (
                self.ind_auto[:, 2:] if probe_c_ix == probe_d_ix
                else self.ind_cross[:, 2:]
            )  # fmt: skip

            zpairs_ab = (
                self.zpairs_auto if probe_a_ix == probe_b_ix else self.zpairs_cross
            )
            zpairs_cd = (
                self.zpairs_auto if probe_c_ix == probe_d_ix else self.zpairs_cross
            )

            # just a sanity check
            assert zpairs_ab == ind_ab.shape[0], 'zpairs-ind inconsistency'
            assert zpairs_cd == ind_cd.shape[0], 'zpairs-ind inconsistency'

            cov_6d = self.cov_dict[term][probe_2tpl]['6d']
            cov_4d = sl.cov_6D_to_4D_blocks(
                cov_6d, self.nbt_coarse, zpairs_ab, zpairs_cd, ind_ab, ind_cd
            )
            cov_2d = sl.cov_4D_to_2D(
                cov_4d, block_index=self.block_index, optimize=True
            )
            self.cov_dict[term][probe_2tpl]['4d'] = cov_4d
            self.cov_dict[term][probe_2tpl]['2d'] = cov_2d
