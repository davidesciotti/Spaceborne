import itertools
import warnings

import numpy as np
from scipy.integrate import simpson as simps
from scipy.interpolate import make_interp_spline

from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import cov_projector as cp
from spaceborne import sb_lib as sl
from spaceborne.cov_projector import CovarianceProjector

warnings.filterwarnings(
    'ignore', message=r'.*invalid escape sequence.*', category=SyntaxWarning
)

warnings.filterwarnings(
    'ignore',
    message=r'.*invalid value encountered in divide.*',
    category=RuntimeWarning,
)


class CovCOSEBIs(CovarianceProjector):
    obs_space = 'cosebis'

    def __init__(
        self,
        cfg: dict,
        pvt_cfg: dict,
        cl_3x2pt_5d: np.ndarray,
        nl_3x2pt_4d: np.ndarray,
        ells_proj_g: np.ndarray,
        ells_proj_ng: np.ndarray,
    ):
        super().__init__(
            cfg, pvt_cfg, cl_3x2pt_5d, nl_3x2pt_4d, ells_proj_g, ells_proj_ng
        )

        self._ch = None
        # (n_modes, nbl_proj_ng) NG projection matrix, see _proj_ng_cs_4d
        self._proj_mat_ng = None

        self.n_modes = cfg['binning']['n_modes_cosebis']
        assert self.n_modes == self.nbs, 'n_modes_cosebis must equal nbs!'
        self.symmetrize_output_dict = pvt_cfg['symmetrize_output_dict']

        # ! instantiate cov_dict
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_cs_2d']
        dims = ['6d', '4d', '2d']
        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='cosebis')
            for probe in self.req_probe_combs_2d
        ]
        _req_probe_combs_2d.append('3x2pt')
        # note: self.req_terms is instantiated in the parent class
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

        # setters
        self._set_theta_binning()
        self._set_w_ell_arrays()

    @property
    def ch(self):
        """Lazy import of cloelib.auxiliary.cosebi_helpers."""
        if self._ch is None:
            try:
                import cloelib.auxiliary.cosebi_helpers as ch

                self._ch = ch
            except ImportError as e:
                raise ImportError(
                    f'Could not import {e.name!r}, which is required to compute '
                    'the COSEBIs covariance. Install cloelib with its pylevin '
                    'extra (pip install "cloelib[pylevin,mpmath]"; see '
                    'environment.yaml) and rerun the code.'
                ) from e
        return self._ch

    def _set_theta_binning(self):
        """Set the theta binning for the COSEBIs SN term integral."""
        self.theta_min_arcmin = self.cfg['precision']['theta_min_arcmin_cosebis']
        self.theta_max_arcmin = self.cfg['precision']['theta_max_arcmin_cosebis']
        self.nbt = self.cfg['precision']['theta_steps_cosebis']

        # Convert to radians
        self.theta_min_rad = np.deg2rad(self.theta_min_arcmin / 60)
        self.theta_max_rad = np.deg2rad(self.theta_max_arcmin / 60)

        # No need for bin edges in the case of COSEBIs, I can directly do this:
        self.theta_grid_rad = np.geomspace(
            self.theta_min_rad, self.theta_max_rad, self.nbt
        )

        # sanity checks
        assert len(self.theta_grid_rad) == self.nbt, 'theta_grid_rad length mismatch'
        assert np.min(np.diff(self.theta_grid_rad)) > 0, 'theta_grid_rad not sorted!'

    def _compute_w_ells(self, ells: np.ndarray) -> np.ndarray:
        """Compute the COSEBIs W_n(ell) kernels for all modes, on the given ells.

        Returns
        -------
        np.ndarray, shape (n_modes, len(ells))
        """

        w_ells_dict = self.ch.get_W_ell(
            thetagrid=self.theta_grid_rad,
            Nmax=self.nbs,
            ells=ells,
            N_thread=self.n_jobs,
        )

        # add a guard against non-int keys
        mode_keys = sorted(k for k in w_ells_dict if isinstance(k, (int, np.integer)))
        w_ells_arr = np.array([w_ells_dict[k] for k in mode_keys])
        return w_ells_arr

    def _set_w_ell_arrays(self):
        """Compute the W_n(ell) kernels on a fine ell grid, and spline them onto the
        ell grid of the Gaussian projection.

        W_n(ell) oscillates with period ~2pi/theta_min in ell, so the fine grid has to
        resolve these oscillations: the NG projection is performed directly on it
        (see ``_proj_ng_cs_4d``), while the (smooth) harmonic-space NG covariance can
        be sampled on the much coarser ells_proj_ng.
        """
        assert np.isclose(self.ells_proj_g[0], self.ells_proj_ng[0]) and np.isclose(
            self.ells_proj_g[-1], self.ells_proj_ng[-1]
        ), 'ells_proj_g and ells_proj_ng must span the same ell range'

        self.ells_w_fine = np.geomspace(
            self.ells_proj_g[0],
            self.ells_proj_g[-1],
            self.cfg['precision']['ell_bins_proj_nongauss_cosebis'],
        )
        # shape (n_modes, len(self.ells_w_fine))
        with sl.timer(f'Computing COSEBIs W_n(ell) kernels for {self.nbs} modes...'):
            self.w_ells_arr_g = self._compute_w_ells(self.ells_proj_g)

            self.w_ells_arr_fine = (
                self._compute_w_ells(self.ells_w_fine)
                if self.cfg['covariance']['SSC'] or self.cfg['covariance']['cNG']
                else None
            )

    def _proj_ng_cs_4d(self, cov_hs_ng_4d: np.ndarray) -> np.ndarray:
        r"""Project the harmonic-space NG covariance to COSEBIs space:

            cov[n, m] = \int d\ell_1 d\ell_2 \ell_1 \ell_2 W_n(\ell_1) W_m(\ell_2)
                        cov_hs(\ell_1, \ell_2)

        (without the 1/(4 pi^2) prefactor). Since W_n(ell) is too oscillatory to be
        integrated on ells_proj_ng, cov_hs is cubic-splined (in log ell) onto the fine
        grid on which W_n(ell) is computed, and the integral is performed there.
        Both the spline and the quadrature are linear in cov_hs, so they are
        combined into a single (n_modes, nbl_proj_ng) projection matrix:

            cov[n, m] = sum_ij P[n, i] P[m, j] cov_hs[i, j]

        Parameters
        ----------
        cov_hs_ng_4d : np.ndarray, shape (nbl_proj_ng, nbl_proj_ng, zpairs_ab, zpairs_cd)

        Returns
        -------
        np.ndarray, shape (n_modes, n_modes, zpairs_ab, zpairs_cd)
        """
        nbl_ng = len(self.ells_proj_ng)
        if cov_hs_ng_4d.shape[:2] != (nbl_ng, nbl_ng):
            raise ValueError(
                f'cov_hs_ng_4d.shape={cov_hs_ng_4d.shape} inconsistent with '
                f'len(ells_proj_ng)={nbl_ng}'
            )

        # the projection matrix only depends on the ell grids: compute it once
        if self._proj_mat_ng is None:
            ells_fine = self.ells_w_fine

            # Same trick as real-space, mainly to save memory
            # due to the size of the ell_fine grid (10^4 x 10^4 x zbins^4)
            # (works both for simps and trapz)
            # simps(y, x) == simps_weights @ y. Shape: (len(ells_fine),)
            intgr_weights = np.trapezoid(y=np.eye(len(ells_fine)), x=ells_fine, axis=0)

            # memory-efficient way to get trapz integration weights:
            # dx = np.diff(ells_fine)
            # intgr_weights = np.zeros_like(ells_fine)
            # intgr_weights[:-1] = dx / 2
            # intgr_weights[1:] += dx / 2

            # Same goes for the spline:
            # spline_of_y(x_fine) = S @ y with S of shape (N_fine, N_coarse)
            # evaluate on log grid, where the covariance is smooth
            interp_op = make_interp_spline(
                np.log(self.ells_proj_ng), np.eye(nbl_ng), k=3, axis=0
            )(np.log(ells_fine))  # Shape (len(ells_fine), nbl_ng)

            # batch together simpson weights, ells, and W_n(ell) into a single
            # projection matrix, shape (n_modes, nbl_proj_ng)
            self._proj_mat_ng = (
                self.w_ells_arr_fine * ells_fine * intgr_weights
            ) @ interp_op

        proj_mat = self._proj_mat_ng

        # L = ell1, M = ell2, p = zpairs_ab, q = zpairs_cd, n = mode_n, m = mode_m
        return np.einsum('nL,mM,LMpq->nmpq', proj_mat, proj_mat, cov_hs_ng_4d)

    def cov_sn_cs(self, amax_abcd: float) -> np.ndarray:
        """Compute the COSEBIs shape noise covariance term."""

        # firstly, construct the prefactor outside of the \theta integral
        first_term = np.einsum('i,j->ij', self.sigma_eps_i**2, self.sigma_eps_i**2) / 2
        kron = np.eye(self.zbins)
        second_term = np.einsum('ik,jl->ijkl', kron, kron) + np.einsum(
            'il,jk->ijkl', kron, kron
        )
        prefactor = first_term[:, :, None, None] * second_term

        # 1. Compute T_minus and T_plus
        t_minus = np.zeros((self.nbt, self.nbs))
        t_plus = np.zeros((self.nbt, self.nbs))

        rn, nn, coeff_j = self.ch.get_roots_and_norms(
            tmax=self.theta_max_rad, tmin=self.theta_min_rad, Nmax=self.nbs
        )

        for n in range(self.nbs):
            t_minus[:, n] = self.ch.tm(
                n=n + 1,
                t=self.theta_grid_rad,
                tmin=self.theta_min_rad,
                nn=nn,
                coeff_j=coeff_j,
            )
            t_plus[:, n] = self.ch.tp(
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
        npair_arr = np.zeros((self.nbt, self.zbins, self.zbins))
        for theta_ix, zi, zj in itertools.product(
            range(self.nbt), range(self.zbins), range(self.zbins)
        ):
            npair_arr[theta_ix, zi, zj] = cp.get_dnpair(
                theta=self.theta_grid_rad[theta_ix],
                survey_area_sr=amax_abcd,
                n_eff_i=self.n_eff_src[zi],
                n_eff_j=self.n_eff_src[zj],
            )

        # * alternatively, you can do
        # for theta_ix, zi, zj in itertools.product(
        #     range(self.nbt), range(self.zbins), range(self.zbins)
        # ):
        #     npair_arr[theta_ix, zi, zj] = cp.get_npair(
        #         theta_1_u=self.theta_edges_rad[theta_ix],
        #         theta_1_l=self.theta_edges_rad[theta_ix + 1],
        #         survey_area_sr=amax_abcd,
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

    def compute_cs_cov_term_probe_6d(
        self, cov_hs_ng_dict: dict | None, probe_abcd: str, term: str, amax_abcd: float
    ) -> None:
        """
        Computes the COSEBIs covariance matrix for the specified term and probe combination.

        Parameters
        ----------
        probe_abcd : str
            Probe combination string (e.g., 'xipxip')
        term : str
            Covariance term to compute ('sva', 'mix', 'sn')

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

        # Compute term-specific covariance
        if term in ['sva', 'mix'] and 'Bn' in probe_2tpl:
            # the Gaussian SVA and MIX terms vanish for B-modes
            cov_out_6d = np.zeros(self.cov_shape_6d)

        elif term in ['sva', 'mix']:
            if term == 'sva':
                cl_integrand_5d = cp.build_cl_integrand_5d_sva(
                    self.cl_3x2pt_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix
                )
            else:
                cl_integrand_5d = cp.build_cl_integrand_5d_mix(
                    self.cl_3x2pt_5d,
                    self.nl_3x2pt_4d,
                    probe_a_ix,
                    probe_b_ix,
                    probe_c_ix,
                    probe_d_ix,
                )

            # the COSEBIs kernel is the precomputed W_n(ell), one row per mode
            cov_out_6d = self.proj_mix_sva_simps_vectorized(
                cl_integrand_5d=cl_integrand_5d,
                amax_abcd=amax_abcd,
                kernel_func_kw={'w_ells_arr': self.w_ells_arr_g},
            )

        elif term == 'sn':
            if probe_ab == probe_cd:
                cov_out_6d = self.cov_sn_cs(amax_abcd=amax_abcd)
            else:
                cov_out_6d = np.zeros(self.cov_shape_6d)

        elif term in ['ssc', 'cng'] and (probe_ab, probe_cd) == ('En', 'En'):
            if cov_hs_ng_dict is None:
                raise ValueError(
                    f'Non-Gaussian covariance term {term} requested, '
                    'but no harmonic-space non-Gaussian covariance dictionary provided.'
                )

            # recover corresponding harmonic-space probe names
            probe_abcd_hs = (
                const.HS_PROBE_IX_TO_NAME_DICT[probe_a_ix]
                + const.HS_PROBE_IX_TO_NAME_DICT[probe_b_ix]
                + const.HS_PROBE_IX_TO_NAME_DICT[probe_c_ix]
                + const.HS_PROBE_IX_TO_NAME_DICT[probe_d_ix]
            )
            probe_ab_hs, probe_cd_hs = sl.split_probe_name(probe_abcd_hs, 'harmonic')

            assert probe_ab_hs == 'LL' and probe_cd_hs == 'LL', (
                'Since no Psi-statistics covariance is implemented, '
                'the input non-Gaussian harmonic-space covariance matrix to project '
                'for COSEBIs should only be the (LL, LL) one.'
                f'found ({probe_ab_hs}, {probe_cd_hs}) instead.'
            )

            # project hs non-gaussian cov to COSEBIs space
            cov_hs_ng_4d = cov_hs_ng_dict[term][probe_ab_hs, probe_cd_hs]['4d']

            cov_cs_ng_4d = self._proj_ng_cs_4d(cov_hs_ng_4d)
            assert cov_cs_ng_4d.shape == (self.nbs, self.nbs, zpairs_ab, zpairs_cd)

            # reshape to 6d and symmetrize if needed
            cov_ng_cs_6d = sl.cov_4D_to_6D_blocks(
                cov_4D=cov_cs_ng_4d,
                nbl=self.nbs,
                zbins=self.zbins,
                ind_ab=ind_ab,
                ind_cd=ind_cd,
                symmetrize_output_ab=self.symmetrize_output_dict[probe_ab_hs],
                symmetrize_output_cd=self.symmetrize_output_dict[probe_cd_hs],
            )

            # normalize
            norm = 4 * np.pi**2
            cov_ng_cs_6d /= norm

            cov_out_6d = cov_ng_cs_6d

        elif term in ['ssc', 'cng'] and (probe_ab, probe_cd) != ('En', 'En'):
            cov_out_6d = np.zeros(self.cov_shape_6d)

        else:
            raise ValueError(
                f'Term {term} not recognized or not implemented for COSEBIs'
            )

        self.cov_dict[term][probe_2tpl]['6d'] = cov_out_6d
