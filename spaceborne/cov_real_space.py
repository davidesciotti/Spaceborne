"""
This module contains functions to compute the covariance matrix in real space.
Nomenclature of the functions/variables:
hs = harmonic space
rs = real space
sva = sample variance
sn = sampling noise
mix = mixed term
"""

import warnings
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline

from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import cov_projector as cp
from spaceborne import sb_lib as sl
from spaceborne.cov_projector import CovarianceProjector
from spaceborne.twobessel_fang import TwoBessel

warnings.filterwarnings(
    'ignore', message=r'.*invalid escape sequence.*', category=SyntaxWarning
)

warnings.filterwarnings(
    'ignore',
    message=r'.*invalid value encountered in divide.*',
    category=RuntimeWarning,
)


# ====================== Kernels
def b_mu(x, mu):
    r"""Implements the piecewise definition of the bracketed term b_mu(x)
    from Eq. (E.2) in Joachimi et al. (2008).
    These are just the results of
    \int_{\theta_l}^{\theta_u} d\theta \theta J_\mu(\ell \theta)
    """
    if mu == 0:
        return x * sl.j1(x)
    elif mu == 2:
        return -x * sl.j1(x) - 2.0 * sl.j0(x)
    elif mu == 4:
        # be careful with x=0!
        return (x - 8.0 / x) * sl.j1(x) - 8.0 * sl.j2(x)
    else:
        raise ValueError('mu must be one of {0, 2, 4}.')


def k_mu(ell, *, thetal, thetau, mu):
    r"""Computes the kernel K_mu(ell * theta_i) in Eq. (E.2):

    K_mu(l * theta_i) = 2 / [ (theta_u^2 - theta_l^2) * l^2 ]
                        * [ b_mu(l * theta_u) - b_mu(l * theta_l) ].
    """
    prefactor = 2.0 / ((thetau**2 - thetal**2) * (ell**2))
    return prefactor * (b_mu(ell * thetau, mu) - b_mu(ell * thetal, mu))


# ===================== Sampling noise term
def t_sn(probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, zbins, sigma_eps_i):
    """
    Returns t^{sn}_{(ij)(mn)} as a (zbins, zbins) array
    over (i,j) of the FIRST pair (ij)
    Assumes sigma_eps_i is sigma_{epsilon1, i} (standard deviation of
    the ellipticity per component)
    """

    # shorten the name for clarity
    sig2 = sigma_eps_i**2

    # auto-source case (e.g. xip/xip or xim/xim)
    if probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 0:
        # T(i,j) = 2 * sig2[i] * sig2[j]
        return 2.0 * sig2[:, None] * sig2[None, :]

    # auto-lens case (e.g. gg/gg)
    if probe_a_ix == probe_b_ix == probe_c_ix == probe_d_ix == 1:
        return np.ones((zbins, zbins))

    # mixed case: each pair contains one lens and one source (e.g. gt/gt)
    if {probe_a_ix, probe_b_ix} == {0, 1} and {probe_c_ix, probe_d_ix} == {0, 1}:
        if probe_a_ix == 0:
            return sig2[:, None] * np.ones((1, zbins))
        else:
            return np.ones((zbins, 1)) * sig2[None, :]

    return np.zeros((zbins, zbins))


def proj_cov_2d_fftlog(
    cov_hs_ell1ell2_in,
    ells_proj,
    theta_edges,
    theta_centers,
    mu: int,
    nu: int,
    nu1=1.01,  # for accuracy issues, play witin ~ [0.5, 1.5]
    nu2=1.01,  # for accuracy issues, play witin ~ [0.5, 1.5]
    N_extrap_low=0,  # number of extrapolation points at low ell (default 0, no extrapolation)
    N_extrap_high=0,  # number of extrapolation points at high ell (default 0, no extrapolation)
    c_window_width=0.25,
    N_pad=0,  # pads the input with 0s (less precise than extrapolation, but faster)
):
    r"""
    Project the NG covariance from harmonic to real space using the FFTLog-based
    double Hankel transform (TwoBessel, Fang et al. 2020).

    Computes the bin-averaged result:

        C_RS(θ_p, θ_q) = ∫ dell1 ell1 K_mu(ell1, θ_p)
        * ∫ dell2 ell2 K_nu(ell2, θ_q) C_HS(ell1, ell2)

    where K_mu(ell, θ_p) = 2/(θ_u² - θ_l²) ∫_{θ_l}^{θ_u} θ' J_mu(ellθ') dθ' is the
    bin-averaging kernel (Joachimi et al. 2008). The analytic bin-averaging is handled
    by ``TwoBessel.two_Bessel_binave`` via the ``g_l_smooth`` kernel.

    Parameters
    ----------
    cov_hs_ell1ell2_in : np.ndarray, shape (nbl, nbl, tomo_shape)
    ells : np.ndarray, shape (nbl,), log-spaced ell grid
    theta_edges : np.ndarray, shape (nbt+1,), log-spaced bin edges in radians
    theta_centers : np.ndarray, shape (nbt,), bin centres in radians
    mu, nu : int, cylindrical Bessel orders
    nu1, nu2 : float, FFTLog power-law bias parameters (default 1.01)
    c_window_width : float, smoothing fraction for FFT coefficients (default 0.25)
    N_pad : int, zero-padding length (default 0)

    Returns
    -------
    np.ndarray, shape (nbt, nbt, zpairs_ab, zpairs_cd)
    """

    # preparations and some sanity checks
    nbl = cov_hs_ell1ell2_in.shape[0]
    assert cov_hs_ell1ell2_in.shape[1] == nbl, (
        'cov_hs_ell1ell2_in must have shape (nbl, nbl, ...)'
    )
    assert len(ells_proj) == nbl, (
        'ells_proj length must match cov_hs_ell1ell2_in shape, '
        f'got {len(ells_proj)} vs {nbl}.'
    )

    nbt = len(theta_centers)

    if nbl % 2 != 0:
        raise ValueError(
            f'ells must have even length for FFTLog (got {nbl}). '
            'Set ell_bins_proj_gauss/ell_bins_proj_nongauss to an even number.'
        )

    dlnells = np.diff(np.log(ells_proj))
    if not np.allclose(dlnells, dlnells[0], rtol=1e-8, atol=0.0):
        raise ValueError(
            'FFTLog requires a log-spaced ell grid. '
            'Ensure ells_proj_ng is generated with np.geomspace.'
        )

    dln_theta_edges = np.diff(np.log(theta_edges))
    if not np.allclose(dln_theta_edges, dln_theta_edges[0], rtol=1e-8, atol=0.0):
        raise ValueError("integration_method='FFTLog' requires log-spaced theta bins.")

    # Constant log bin width (requires log-spaced theta bins)
    dlntheta = np.log(theta_edges[1] / theta_edges[0])

    # Warn if theta_centers fall outside the natural FFTLog output range
    theta_out_min = 1.0 / ells_proj[-1]
    theta_out_max = 1.0 / ells_proj[0]

    if theta_centers[0] < theta_out_min or theta_centers[-1] > theta_out_max:
        warnings.warn(
            f'theta_centers [{theta_centers[0]}, {theta_centers[-1]}] rad '
            f'extends outside FFTLog output range '
            f'[{theta_out_min}, {theta_out_max}] rad. '
            'Consider widening ell_min_proj / ell_max_proj.',
            stacklevel=2,
        )

    tomo_shape = cov_hs_ell1ell2_in.shape[2:]
    tomo_elements = int(np.prod(tomo_shape)) if tomo_shape else 1

    cov_hs_3d = cov_hs_ell1ell2_in.reshape(nbl, nbl, tomo_elements)

    result_3d = np.zeros((nbt, nbt, tomo_elements))

    for tomo_ix in range(tomo_elements):
        # Build integrand f(ell1,ell2) = ell1² ell2² C_HS so that
        # TwoBessel gives ∫dell1 ell1 J_mu ∫dell2 ell2 J_nu · C_HS
        fx1x2 = (
            cov_hs_3d[:, :, tomo_ix] * ells_proj[:, None] ** 2 * ells_proj[None, :] ** 2
        )
        tb = TwoBessel(
            x1=ells_proj,
            x2=ells_proj,
            fx1x2=fx1x2,
            nu1=nu1,
            nu2=nu2,
            N_extrap_low=N_extrap_low,
            N_extrap_high=N_extrap_high,
            c_window_width=c_window_width,
            N_pad=N_pad,
        )

        theta1_out, theta2_out, integral = tb.two_Bessel_binave(
            mu, nu, dlntheta, dlntheta
        )

        # Interpolate onto the desired theta grid (log-log 2D spline).
        # NOTE: two_Bessel_binave's bin-averaging kernel (g_l_smooth) averages F over
        # [y, y*exp(dlntheta)], so the output at grid point y is the bin-average for a
        # bin whose *lower edge* is y. The bin with lower edge theta_edges[i] is the
        # bin centred on theta_centers[i], so we must sample at the lower bin edges,
        # NOT at the geometric centres (doing the latter shifts every value up by
        # half a bin, producing a per-bin sawtooth bias).
        interp = RectBivariateSpline(
            np.log(theta1_out), np.log(theta2_out), integral, kx=3, ky=3
        )
        theta_lower_edges = theta_edges[:-1]
        result_3d[:, :, tomo_ix] = interp(
            np.log(theta_lower_edges), np.log(theta_lower_edges)
        )

    return result_3d.reshape(nbt, nbt, *tomo_shape)


def proj_cov_2d_parallel_helper(
    s1: int,
    s2: int,
    theta_edges: np.ndarray,
    mu: int,
    nu: int,
    integration_method: str,
    ells_proj_ng: np.ndarray,
    cov_hs_ng_4d: np.ndarray,
):
    # TODO make kernel agnostic using kernel builder
    # TODO move to covariance_projector.py
    kernel_1 = partial(k_mu, thetal=theta_edges[s1], thetau=theta_edges[s1 + 1], mu=mu)
    kernel_2 = partial(k_mu, thetal=theta_edges[s2], thetau=theta_edges[s2 + 1], mu=nu)

    block = cp.proj_cov_2d(
        ells_proj=ells_proj_ng,
        cov_hs_ng_4d=cov_hs_ng_4d,
        kernel_1_func_of_ell=kernel_1,
        kernel_2_func_of_ell=kernel_2,
        integration_method=integration_method,
    )

    return s1, s2, block


# ! ====================================================================================
# ! ====================================================================================
# ! ====================================================================================


class CovRealSpace(CovarianceProjector):
    obs_space = 'real'

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

        # ! instantiate cov_dict
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_rs_2d']
        dims = ['6d', '4d', '2d']
        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='real')
            for probe in self.req_probe_combs_2d
        ]
        _req_probe_combs_2d.append('3x2pt')
        # note: self.req_terms is instantiated in the parent class
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )
        self.symmetrize_output_dict = pvt_cfg['symmetrize_output_dict']

        # theta binning
        self._set_theta_binning()

        # integration methods, validated in config_checker.check_projection_methods
        self.proj_g_int_method = self.cfg['precision']['proj_gauss_integration_method']
        self.proj_ng_int_method = self.cfg['precision'][
            'proj_nongauss_integration_method'
        ]

    def _set_theta_binning(self):
        self.theta_min_arcmin = self.cfg['binning']['theta_min_arcmin']
        self.theta_max_arcmin = self.cfg['binning']['theta_max_arcmin']
        self.nbt = self.cfg['binning']['theta_bins']

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

        # Use a loop to set up theta binning
        theta_edges_deg = _binning_func(
            self.theta_min_arcmin / 60, self.theta_max_arcmin / 60, self.nbt + 1
        )
        theta_edges = np.deg2rad(theta_edges_deg)  # in radians

        if self.cfg['binning']['binning_type'] == 'log':
            theta_centers = np.sqrt(theta_edges[:-1] * theta_edges[1:])
        elif self.cfg['binning']['binning_type'] == 'lin':
            theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0

        # ! the theta values used throughout the code are in radians!
        self.theta_edges = theta_edges
        self.theta_centers = theta_centers

        assert len(theta_centers) == self.nbt, 'theta_centers length mismatch'

    def cov_sn_rs(
        self, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu, amax_abcd
    ):
        npair_arr = np.zeros((self.nbt, self.zbins, self.zbins))
        for theta_ix in range(self.nbt):
            theta_l = self.theta_edges[theta_ix]
            theta_u = self.theta_edges[theta_ix + 1]
            for zi in range(self.zbins):
                for zj in range(self.zbins):
                    npair_arr[theta_ix, zi, zj] = cp.get_npair(
                        theta_u,
                        theta_l,
                        amax_abcd,
                        self.n_eff_2d[probe_a_ix, zi],
                        self.n_eff_2d[probe_b_ix, zj],
                    )

        delta_mu_nu = 1.0 if (mu == nu) else 0.0
        delta_theta = np.eye(self.nbt)

        t_arr = t_sn(
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, self.zbins, self.sigma_eps_i
        )

        term = (
            cp.get_delta_tomo(probe_a_ix, probe_c_ix, self.zbins)[
                None, None, :, None, :, None
            ]
            * cp.get_delta_tomo(probe_b_ix, probe_d_ix, self.zbins)[
                None, None, None, :, None, :
            ]
            + cp.get_delta_tomo(probe_a_ix, probe_d_ix, self.zbins)[
                None, None, :, None, None, :
            ]
            * cp.get_delta_tomo(probe_b_ix, probe_c_ix, self.zbins)[
                None, None, None, :, :, None
            ]
        )

        cov_sn_rs_6d = (
            delta_mu_nu
            * delta_theta[:, :, None, None, None, None]
            * term
            * t_arr[None, None, :, :, None, None]
            / npair_arr[:, None, :, :, None, None]
        )
        return cov_sn_rs_6d

    def _proj_gauss_integrand(self, cl_integrand_5d, mu, nu, amax_abcd):
        """Project a Gaussian (SVA or MIX) integrand with the configured method."""
        if self.proj_g_int_method == 'simps':
            return self.proj_mix_sva_simps_vectorized(
                cl_integrand_5d=cl_integrand_5d, amax_abcd=amax_abcd, mu=mu, nu=nu
            )
        if self.proj_g_int_method == 'FFTLog':
            return self.proj_sva_mix_fftlog_wrapper(
                cl_integrand_5d=cl_integrand_5d, mu=mu, nu=nu, amax_abcd=amax_abcd
            )
        raise ValueError(
            f'Unknown proj_gauss_integration_method {self.proj_g_int_method}'
        )

    def proj_sva_mix_fftlog_wrapper(self, cl_integrand_5d, mu, nu, amax_abcd):
        # Gaussian (SVA/MIX) covariance via the 2D-FFTLog diagonal trick.

        nbl = cl_integrand_5d.shape[0]
        dlnell = np.log(self.ells_proj_g[1] / self.ells_proj_g[0])
        integrand_6d = np.zeros((nbl, nbl) + cl_integrand_5d.shape[1:])
        for i in range(nbl):
            integrand_6d[i, i, ...] = cl_integrand_5d[i, ...] / (
                self.ells_proj_g[i] ** 2 * dlnell
            )

        # prefactors
        integrand_6d /= 2.0 * np.pi * amax_abcd

        # integrate
        # N_pad is essential for high-order Bessels (mu/nu >= 2, i.e. gt/xim): the
        # integrand ell^2 C(ell) does not decay at the ell grid boundary, so without
        # zero-padding the FFTLog rings and overestimates the small-theta result.
        # Padding converges by ~nbl points (extrapolation cannot be
        # used here because the diagonal-only input has off-diagonal zeros).
        integral_6d = proj_cov_2d_fftlog(
            cov_hs_ell1ell2_in=integrand_6d,
            ells_proj=self.ells_proj_g,
            theta_edges=self.theta_edges,
            theta_centers=self.theta_centers,
            mu=mu,
            nu=nu,
            N_pad=nbl,
        )

        return integral_6d

    def compute_rs_cov_term_probe_6d(
        self, cov_hs_ng_dict: dict | None, probe_abcd: str, term: str, amax_abcd: float
    ) -> None:
        """
        Computes the real space covariance matrix for the specified term
        and probe combination, in 6d
        """

        if term not in const.ALL_COV_TERMS:
            raise ValueError(f'Covariance term {term} not recognized!')

        probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'real')
        probe_2tpl = (probe_ab, probe_cd)

        mu, nu = const.MU_DICT[probe_ab], const.MU_DICT[probe_cd]
        probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix = const.RS_PROBE_NAME_TO_IX_DICT[
            probe_abcd
        ]

        ind_ab = (
            self.ind_auto[:, 2:] if probe_a_ix == probe_b_ix
            else self.ind_cross[:, 2:]
        )  # fmt: skip
        ind_cd = (
            self.ind_auto[:, 2:] if probe_c_ix == probe_d_ix
            else self.ind_cross[:, 2:]
        )  # fmt: skip

        zpairs_ab = self.zpairs_auto if probe_a_ix == probe_b_ix else self.zpairs_cross
        zpairs_cd = self.zpairs_auto if probe_c_ix == probe_d_ix else self.zpairs_cross

        # just a sanity check
        assert zpairs_ab == ind_ab.shape[0], 'zpairs-ind inconsistency'
        assert zpairs_cd == ind_cd.shape[0], 'zpairs-ind inconsistency'

        # Compute covariance:
        if term == 'sva':
            cl_integrand_5d = cp.build_cl_integrand_5d_sva(
                self.cl_3x2pt_5d, probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix
            )
            cov_out_6d = self._proj_gauss_integrand(cl_integrand_5d, mu, nu, amax_abcd)

        elif term == 'mix' and probe_abcd not in ['wxim', 'wxip']:
            cl_integrand_5d = cp.build_cl_integrand_5d_mix(
                self.cl_3x2pt_5d,
                self.nl_3x2pt_4d,
                probe_a_ix,
                probe_b_ix,
                probe_c_ix,
                probe_d_ix,
            )
            cov_out_6d = self._proj_gauss_integrand(cl_integrand_5d, mu, nu, amax_abcd)

        elif term == 'mix' and probe_abcd in ['wxim', 'wxip']:
            cov_out_6d = np.zeros(self.cov_shape_6d)

        elif term == 'sn':
            # this is 0 for
            # ['xipxim', 'gtxim', 'gtxip', 'wxim', 'wgt', 'wxip']
            # but is very fast to compute so I don't skip these terms
            cov_out_6d = self.cov_sn_rs(
                probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, mu, nu, amax_abcd
            )

        elif term in ['ssc', 'cng']:
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

            # project hs non-gaussian cov to real space
            cov_hs_ng_4d = cov_hs_ng_dict[term][probe_ab_hs, probe_cd_hs]['4d']

            if self.proj_ng_int_method == 'quad':
                cov_rs_ng_4d = self.proj_ng_quad(cov_hs_ng_4d, mu=mu, nu=nu)

            elif self.proj_ng_int_method == 'simps':
                cov_rs_ng_4d = np.zeros((self.nbs, self.nbs, zpairs_ab, zpairs_cd))

                # to parallelize over the scale (theta, in this case) indices s1 and s2,
                # rely on proj_cov_2d_parallel_helper
                # TODO I could only loop over the upper triangle of s1, s2 and then
                # symmetrize...
                results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(proj_cov_2d_parallel_helper)(
                        s1=s1,
                        s2=s2,
                        theta_edges=self.theta_edges,
                        mu=mu,
                        nu=nu,
                        integration_method=self.proj_ng_int_method,
                        ells_proj_ng=self.ells_proj_ng,
                        cov_hs_ng_4d=cov_hs_ng_4d,
                    )
                    for s1 in range(self.nbs)
                    for s2 in range(self.nbs)
                )

                for s1, s2, block in results:
                    cov_rs_ng_4d[s1, s2] = block

            elif self.proj_ng_int_method == 'FFTLog':
                cov_rs_ng_4d = proj_cov_2d_fftlog(
                    cov_hs_ell1ell2_in=cov_hs_ng_4d,
                    ells_proj=self.ells_proj_ng,
                    theta_edges=self.theta_edges,
                    theta_centers=self.theta_centers,
                    mu=mu,
                    nu=nu,
                    N_pad=2 * len(self.ells_proj_ng),
                )

            # reshape to 6d and symmetrize if needed
            cov_rs_ng_6d = sl.cov_4D_to_6D_blocks(
                cov_4D=cov_rs_ng_4d,
                nbl=self.nbs,
                zbins=self.zbins,
                ind_ab=ind_ab,
                ind_cd=ind_cd,
                symmetrize_output_ab=self.symmetrize_output_dict[probe_ab_hs],
                symmetrize_output_cd=self.symmetrize_output_dict[probe_cd_hs],
            )

            # normalize
            norm = 4 * np.pi**2
            cov_rs_ng_6d /= norm

            cov_out_6d = cov_rs_ng_6d

        else:
            raise ValueError(f'Covariance term not recognised: {term}')

        # finally, assign the newly computed 6D cov to the appropriate key in cov_dict
        self.cov_dict[term][probe_2tpl]['6d'] = cov_out_6d

    def k_mu(self, ell, *, thetal, thetau, mu):
        """Thin wrapper around k_mu, just to make it a class method"""
        return k_mu(ell, thetal=thetal, thetau=thetau, mu=mu)
