# TODO the NG cov has not been re-tested against OC
# TODO the NG cov needs a smaller number of ell bins for the simpson integration! It's
# TODO unpractical to compute it in 1000 ell values

import itertools
import warnings

import cloelib.auxiliary.cosebi_helpers as ch
import numpy as np
from scipy.integrate import simpson as simps

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

_UNSET = object()


class CovCOSEBIs(CovarianceProjector):
    def __init__(self, cfg, pvt_cfg, mask_obj):
        super().__init__(cfg, pvt_cfg, mask_obj)

        self.obs_space = 'cosebis'

        self.n_modes = cfg['binning']['n_modes_cosebis']
        assert self.n_modes == self.nbx, 'n_modes_cosebis must equal nbx!'

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
        self._set_theta_binning()

        # attributes set at runtime
        self.cl_3x2pt_5d = _UNSET
        self.ells = _UNSET
        self.nbl = _UNSET

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

    def set_w_ells(self):
        """
        Compute and set the COSEBIs W_n(ell) kernels for all modes and ells.

        Sets:
            self.w_ells_arr (np.ndarray): Array of shape (n_modes, n_ells) containing
                the computed W_n(ell) kernel values.
        """

        with sl.timer(f'Computing COSEBIs W_n(ell) kernels for {self.nbx} modes...'):
            w_ells_dict = ch.get_W_ell(
                thetagrid=self.theta_grid_rad,
                Nmax=self.nbx,
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
        t_minus = np.zeros((self.nbt, self.nbx))
        t_plus = np.zeros((self.nbt, self.nbx))

        rn, nn, coeff_j = ch.get_roots_and_norms(
            tmax=self.theta_max_rad, tmin=self.theta_min_rad, Nmax=self.nbx
        )

        for n in range(self.nbx):
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
        npair_arr = np.zeros((self.nbt, self.zbins, self.zbins))
        for theta_ix, zi, zj in itertools.product(
            range(self.nbt), range(self.zbins), range(self.zbins)
        ):
            npair_arr[theta_ix, zi, zj] = cp.get_dnpair(
                theta=self.theta_grid_rad[theta_ix],
                survey_area_sr=self.survey_area_sr,
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

    def compute_cs_cov_term_probe_6d(
        self, cov_hs_dict: dict | None, probe_abcd: str, term: str
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

        # Create a theta grid for computing the Hankel transform
        # You may want to use a finer grid than self.theta_centers_fine

        # Arguments for the covariance function (same for SVA and MIX)
        cov_simps_func_kw = {
            'probe_a_ix': probe_a_ix,
            'probe_b_ix': probe_b_ix,
            'probe_c_ix': probe_c_ix,
            'probe_d_ix': probe_d_ix,
        }

        # Arguments for the kernel builder
        # For COSEBIs it's only w_ells_arr (constant across all mode pairs)
        # The mode indices (mode_n, mode_m) are passed as scale_ix_1, scale_ix_2 by
        # the wrapper
        kernel_builder_func_kw = {'w_ells_arr': self.w_ells_arr}

        # Compute term-specific covariance
        if term == 'sva':
            if 'Bn' in probe_2tpl:
                cov_out_6d = np.zeros(self.cov_shape_6d)
            else:
                cov_out_6d = self.cov_simps_wrapper(
                    zpairs_ab=zpairs_ab,
                    zpairs_cd=zpairs_cd,
                    ind_ab=ind_ab,
                    ind_cd=ind_cd,
                    cov_simps_func=self.cov_sva_simps,
                    cov_simps_func_kw=cov_simps_func_kw,
                    kernel_builder_func_kw=kernel_builder_func_kw,
                )

        elif term == 'mix':
            if 'Bn' in probe_2tpl:
                cov_out_6d = np.zeros(self.cov_shape_6d)
            else:
                cov_out_6d = self.cov_simps_wrapper(
                    zpairs_ab=zpairs_ab,
                    zpairs_cd=zpairs_cd,
                    ind_ab=ind_ab,
                    ind_cd=ind_cd,
                    cov_simps_func=self.cov_mix_simps,
                    cov_simps_func_kw=cov_simps_func_kw,
                    kernel_builder_func_kw=kernel_builder_func_kw,
                )

        elif term == 'sn':
            if probe_ab == probe_cd:
                cov_out_6d = self.cov_sn_cs()
            else:
                cov_out_6d = np.zeros(self.cov_shape_6d)

        else:
            raise ValueError(
                f'Term {term} not recognized or not implemented for COSEBIs'
            )

        self.cov_dict[term][probe_2tpl]['6d'] = cov_out_6d
