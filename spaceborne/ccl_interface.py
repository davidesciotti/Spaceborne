"""This module should be run with pyccl >= v3.2.1"""

from functools import partial

import numpy as np
import pyccl as ccl
from tqdm import tqdm

from spaceborne import constants as const
from spaceborne import cosmo_lib, wf_cl_lib
from spaceborne import cov_dict as cd
from spaceborne import sb_lib as sl

_UNSET = object()


def apply_mult_shear_bias(cl_ll_3d, cl_gl_3d, mult_shear_bias, zbins):
    assert len(mult_shear_bias) == zbins, (
        'mult_shear_bias should be a vector of length zbins'
    )

    if np.all(mult_shear_bias == 0):
        return cl_ll_3d, cl_gl_3d

    print('Applying multiplicative shear bias')
    for ell_idx in range(cl_ll_3d.shape[0]):
        for zi in range(zbins):
            for zj in range(zbins):
                cl_ll_3d[ell_idx, zi, zj] *= (1 + mult_shear_bias[zi]) * (
                    1 + mult_shear_bias[zj]
                )

    for ell_idx in range(cl_gl_3d.shape[0]):
        for zi in range(zbins):
            for zj in range(zbins):
                cl_gl_3d[ell_idx, zi, zj] *= 1 + mult_shear_bias[zj]

    return cl_ll_3d, cl_gl_3d


def compute_cl_3x2pt_5d(
    ccl_obj,
    ells: np.ndarray,
    zbins: int,
    mult_shear_bias: np.ndarray,
    cl_ccl_kwargs: dict,
    n_probes_hs: int = 2,
) -> np.ndarray:
    """Just a wrapper to quickly compute the CCL cls, including the multiplicative
    shear bias"""

    nbl = len(ells)

    cl_ll_3d = ccl_obj.compute_cls(
        ells,
        ccl_obj.p_of_k_a,
        ccl_obj.wf_lensing_obj,
        ccl_obj.wf_lensing_obj,
        cl_ccl_kwargs,
    )
    cl_gl_3d = ccl_obj.compute_cls(
        ells,
        ccl_obj.p_of_k_a,
        ccl_obj.wf_galaxy_obj,
        ccl_obj.wf_lensing_obj,
        cl_ccl_kwargs,
    )
    cl_gg_3d = ccl_obj.compute_cls(
        ells,
        ccl_obj.p_of_k_a,
        ccl_obj.wf_galaxy_obj,
        ccl_obj.wf_galaxy_obj,
        cl_ccl_kwargs,
    )

    # don't forget to apply mult shear bias
    cl_ll_3d, cl_gl_3d = apply_mult_shear_bias(
        cl_ll_3d, cl_gl_3d, mult_shear_bias, zbins
    )

    cl_3x2pt_5d = np.zeros((n_probes_hs, n_probes_hs, nbl, zbins, zbins))
    cl_3x2pt_5d[0, 0] = cl_ll_3d
    cl_3x2pt_5d[1, 0] = cl_gl_3d
    cl_3x2pt_5d[0, 1] = cl_gl_3d.transpose(0, 2, 1)
    cl_3x2pt_5d[1, 1] = cl_gg_3d

    return cl_3x2pt_5d


class CCLInterface:
    def __init__(
        self,
        cosmology_dict: dict,
        extra_parameters_dict: dict,
        ia_dict: dict,
        halo_model_dict: dict,
        spline_params: dict | None,
        gsl_params: dict | None,
    ):
        self.cosmology_dict = cosmology_dict
        self.extra_parameters_dict = extra_parameters_dict
        self.ia_dict = ia_dict

        if spline_params is not None:
            for key in spline_params:
                ccl.spline_params[key] = spline_params[key]

        if gsl_params is not None:
            for key in gsl_params:
                ccl.gsl_params[key] = gsl_params[key]

        self.flat_fid_pars_dict = sl.flatten_dict(self.cosmology_dict)
        cosmo_dict_ccl = cosmo_lib.map_keys(self.cosmology_dict, key_mapping=None)
        self.cosmo_ccl = cosmo_lib.instantiate_cosmo_ccl_obj(
            cosmo_dict_ccl, self.extra_parameters_dict
        )

        self.gal_bias_func_dict = {
            'analytical': wf_cl_lib.b_of_z_analytical,
            'leporifit': wf_cl_lib.b_of_z_fs1_leporifit,
            'pocinofit': wf_cl_lib.b_of_z_fs1_pocinofit,
            'fs2_fit': wf_cl_lib.b_of_z_fs2_fit,
        }

        # initialize halo model
        self.mass_def = getattr(ccl.halos, halo_model_dict['mass_def'])
        self.c_m_relation = getattr(ccl.halos, halo_model_dict['concentration'])(
            mass_def=self.mass_def
        )
        self.hmf = getattr(ccl.halos, halo_model_dict['mass_function'])(
            mass_def=self.mass_def
        )
        self.hbf = getattr(ccl.halos, halo_model_dict['halo_bias'])(
            mass_def=self.mass_def
        )
        self.hmc = ccl.halos.HMCalculator(
            mass_function=self.hmf, halo_bias=self.hbf, mass_def=self.mass_def
        )
        self.halo_profile_dm = getattr(ccl.halos, halo_model_dict['halo_profile_dm'])(
            mass_def=self.mass_def, concentration=self.c_m_relation
        )
        self.halo_profile_hod = getattr(ccl.halos, halo_model_dict['halo_profile_hod'])(
            mass_def=self.mass_def, concentration=self.c_m_relation
        )

        # declare attributes set at runtime
        self.p_of_k_a = _UNSET
        self.zbins: int = _UNSET
        self.output_path: str = _UNSET
        self.which_b1g_in_resp: str = _UNSET
        self.lumin_ratio_2d_arr: np.ndarray | None = _UNSET
        self.a_grid_trisp_SSC: np.ndarray = _UNSET
        self.a_grid_trisp_cNG: np.ndarray = _UNSET
        self.logn_k_grid_trisp_SSC: np.ndarray = _UNSET
        self.logn_k_grid_trisp_cNG: np.ndarray = _UNSET
        self.wf_galaxy_arr = _UNSET
        self.cl_ll_3d: np.ndarray = _UNSET
        self.cl_gl_3d: np.ndarray = _UNSET
        self.cl_gg_3d: np.ndarray = _UNSET
        self.cl_3x2pt_5d: np.ndarray = _UNSET
        self.separable_growth: bool = _UNSET

    def set_nz(self, nz_full_src, nz_full_lns):
        # unpack the array
        self.zgrid_nz_src = nz_full_src[:, 0]
        self.zgrid_nz_lns = nz_full_lns[:, 0]
        self.nz_src = nz_full_src[:, 1:]
        self.nz_lns = nz_full_lns[:, 1:]

        # set tuple
        self.nz_src_tuple = (self.zgrid_nz_src, self.nz_src)
        self.nz_lns_tuple = (self.zgrid_nz_lns, self.nz_lns)

    def check_nz_tuple(self, zbins):
        assert isinstance(self.nz_src_tuple, tuple), 'nz_src_tuple must be a tuple'
        assert isinstance(self.nz_lns_tuple, tuple), 'nz_lns_tuple must be a tuple'

        assert self.nz_src_tuple[1].shape == (len(self.zgrid_nz_src), zbins), (
            'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'
        )
        assert self.nz_lns_tuple[1].shape == (len(self.zgrid_nz_lns), zbins), (
            'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'
        )

    def set_ia_bias_tuple(self, z_grid_src, has_ia):
        self.has_ia = has_ia

        if self.has_ia:
            ia_bias_1d = wf_cl_lib.build_ia_bias_1d_arr(
                z_grid_src,
                cosmo_ccl=self.cosmo_ccl,
                ia_dict=self.ia_dict,
                lumin_ratio_2d_arr=self.lumin_ratio_2d_arr,
                output_F_IA_of_z=False,
            )

            self.ia_bias_tuple = (z_grid_src, ia_bias_1d)

        else:
            self.ia_bias_tuple = None

    def set_gal_bias_tuple_spv3(self, z_grid_lns, magcut_lens, poly_fit_values):
        # 1. set galaxy bias function (i.e., the callable)
        _gal_bias_func = self.gal_bias_func_dict['fs2_fit']
        self.gal_bias_func = partial(
            _gal_bias_func, magcut_lens=magcut_lens, poly_fit_values=poly_fit_values
        )

        # construct the 2d array & tuple; this is mainly to ensure compatibility with
        # # the wf_ccl function. In this case, the same array is given for each bin
        # (each column)
        gal_bias_1d = self.gal_bias_func(z_grid_lns)
        self.gal_bias_2d = np.repeat(gal_bias_1d.reshape(1, -1), self.zbins, axis=0).T
        self.gal_bias_tuple = (z_grid_lns, self.gal_bias_2d)

    def save_gal_bias_table_ascii(self, z_grid_lns, filename):
        assert filename.endswith('.ascii'), 'filename must end with.ascii'
        gal_bias_table = np.hstack((z_grid_lns.reshape(-1, 1), self.gal_bias_2d))
        np.savetxt(filename, gal_bias_table)

    def set_mag_bias_tuple(
        self, z_grid_lns, has_magnification_bias, magcut_lens, poly_fit_values
    ):
        """
        Set the magnification bias values and store in a tuple. In this function,
        we call "mag_bias" the usual s(z).

        Note: In the cases handled by this function (no magnification bias,
        polinomial fit), the magnification bias is the same for all redshift bins,
        thus the use of np.repeat to construct the 2d array.
        """

        if has_magnification_bias:
            # this is only to ensure compatibility with wf_ccl function. In reality,
            # the same array is given for each bin
            mag_bias_1d = wf_cl_lib.s_of_z_fs2_fit(
                z_grid_lns, magcut_lens=magcut_lens, poly_fit_values=poly_fit_values
            )
            self.mag_bias_2d = np.repeat(
                mag_bias_1d.reshape(1, -1), self.zbins, axis=0
            ).T
            self.mag_bias_tuple = (z_grid_lns, self.mag_bias_2d)
        else:
            # this is the correct way to set the magnification bias values so that the
            # actual bias is 1, ant the corresponding
            # wf_mu is zero (which is, in theory, the case mag_bias_tuple=None, which
            # however causes pyccl to crash!)
            # mag_bias_2d = (np.ones_like(gal_bias_2d) * + 2) / 5
            # mag_bias_tuple = (zgrid_nz, mag_bias_2d)
            self.mag_bias_tuple = None

    def set_kernel_obj(self, has_rsd, n_samples_wf):

        unit_bias_tuple = (self.zgrid_nz_lns, np.ones_like(self.zgrid_nz_lns))

        self.wf_lensing_obj = []
        self.wf_galaxy_obj = []
        self.wf_density_obj = []  # density-only (no gal bias, no mag, no RSD)
        self.wf_mag_obj = []  # magnification-only (no matter, no RSD)

        for zi in range(self.zbins):
            # ! Lensing
            self.wf_lensing_obj.append(
                ccl.tracers.WeakLensingTracer(
                    cosmo=self.cosmo_ccl,
                    dndz=(self.nz_src_tuple[0], self.nz_src_tuple[1][:, zi]),
                    ia_bias=self.ia_bias_tuple,
                    use_A_ia=False,
                    n_samples=n_samples_wf,
                )
            )

            # ! Galaxy
            # this is needed to be able to pass mag_bias = None for each zbin
            if self.mag_bias_tuple is None:
                mag_bias_arg = self.mag_bias_tuple
            else:
                mag_bias_arg = (self.mag_bias_tuple[0], self.mag_bias_tuple[1][:, zi])
                self.wf_mag_obj.append(
                    ccl.tracers.NumberCountsTracer(
                        cosmo=self.cosmo_ccl,
                        has_rsd=False,
                        dndz=(self.nz_lns_tuple[0], self.nz_lns_tuple[1][:, zi]),
                        bias=None,
                        mag_bias=mag_bias_arg,
                        n_samples=n_samples_wf,
                    )
                )

            self.wf_galaxy_obj.append(
                ccl.tracers.NumberCountsTracer(
                    cosmo=self.cosmo_ccl,
                    has_rsd=has_rsd,
                    dndz=(self.nz_lns_tuple[0], self.nz_lns_tuple[1][:, zi]),
                    bias=(self.gal_bias_tuple[0], self.gal_bias_tuple[1][:, zi]),
                    mag_bias=mag_bias_arg,
                    n_samples=n_samples_wf,
                )
            )
            self.wf_density_obj.append(
                ccl.tracers.NumberCountsTracer(
                    cosmo=self.cosmo_ccl,
                    has_rsd=False,
                    dndz=(self.nz_lns_tuple[0], self.nz_lns_tuple[1][:, zi]),
                    bias=unit_bias_tuple,
                    mag_bias=None,
                    n_samples=n_samples_wf,
                )
            )

    def compute_cls(self, ell_grid, p_of_k_a, kernel_a, kernel_b, cl_ccl_kwargs: dict):
        cl_ab_3d = wf_cl_lib.cl_ccl(
            wf_a=kernel_a,
            wf_b=kernel_b,
            ells=ell_grid,
            zbins=self.zbins,
            p_of_k_a=p_of_k_a,
            cosmo=self.cosmo_ccl,
            cl_ccl_kwargs=cl_ccl_kwargs,
        )

        return cl_ab_3d

    def set_kernel_arr(self, z_grid_wf, has_magnification_bias):
        self.z_grid_wf = z_grid_wf
        a_arr = cosmo_lib.z_to_a(z_grid_wf)
        comoving_distance = ccl.comoving_radial_distance(self.cosmo_ccl, a_arr)

        wf_lensing_tot_arr = np.asarray(
            [
                self.wf_lensing_obj[zbin_idx].get_kernel(comoving_distance)
                for zbin_idx in range(self.zbins)
            ]
        )

        wf_galaxy_tot_arr = np.asarray(
            [
                self.wf_galaxy_obj[zbin_idx].get_kernel(comoving_distance)
                for zbin_idx in range(self.zbins)
            ]
        )

        # lensing
        self.wf_gamma_arr = wf_lensing_tot_arr[:, 0, :].T
        if self.has_ia:
            self.wf_ia_arr = wf_lensing_tot_arr[:, 1, :].T
            self.wf_ia_contribution_arr = (
                self.ia_bias_tuple[1][:, None] * self.wf_ia_arr
            )
            self.wf_lensing_arr = self.wf_gamma_arr + self.wf_ia_contribution_arr
        else:
            self.wf_ia_arr = np.zeros_like(self.wf_gamma_arr)
            self.wf_ia_contribution_arr = np.zeros_like(self.wf_gamma_arr)
            self.wf_lensing_arr = self.wf_gamma_arr

        # galaxy
        self.wf_delta_arr = wf_galaxy_tot_arr[:, 0, :].T
        self.wf_mu_arr = (
            wf_galaxy_tot_arr[:, -1, :].T
            if has_magnification_bias
            else np.zeros_like(self.wf_delta_arr)
        )

        # in the case of ISTF, the galaxt bias is bin-per-bin and is therefore included
        # in the kernels. Add it here
        # for a fair comparison with vincenzo's kernels, in the plot.
        # * Note that the galaxy bias is included in the wf_ccl_obj in any way, both in
        # * ISTF and SPV3 cases! It must
        # * in fact be passed to the angular_cov_SSC function
        self.wf_galaxy_wo_gal_bias_arr = self.wf_delta_arr + self.wf_mu_arr
        self.wf_galaxy_w_gal_bias_arr = (
            self.wf_delta_arr * self.gal_bias_2d + self.wf_mu_arr
        )

    # ! ================================================================================

    def set_cov_dict(self, pvt_cfg, ccl_ng_cov_terms_list):
        """Instantiate the covariance dictionary with the required terms.
        This is not done at initialization since this class does not exclusively handle
        covariance calculations.

        Note: this class only computes
          - non-gaussian terms required, also depending on SSC/cNG_code settings
          - 4d dim
        """

        _req_terms = [term.lower() for term in ccl_ng_cov_terms_list]
        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='harmonic')
            for probe in pvt_cfg['req_probe_combs_hs_2d']
        ]
        _dims = ['4d']

        self.cov_dict = cd.create_cov_dict(_req_terms, _req_probe_combs_2d, dims=_dims)

    def sigma2_b_func(
        self, z_grid: np.ndarray, cl_footp_norm_abcd: np.ndarray
    ) -> tuple:
        self.a_grid_sigma2_b = cosmo_lib.z_to_a(z_grid)[::-1]

        # normalize the mask and pass it to sigma2_B_from_mask
        sigma2_b = ccl.covariances.sigma2_B_from_mask(
            cosmo=self.cosmo_ccl, a_arr=self.a_grid_sigma2_b, mask_wl=cl_footp_norm_abcd
        )
        return self.a_grid_sigma2_b, sigma2_b

    def build_trisp_dict(
        self, which_ng_cov: str, unique_probe_combs: list, gal_bias_1d: np.ndarray
    ):

        # the default pk must be passed to the Tk3D functions as None, not as
        # 'delta_matter:delta_matter'
        p_of_k_a = (
            None if self.p_of_k_a == 'delta_matter:delta_matter' else self.p_of_k_a
        )

        # TODO get default grids info when passing a, k = None
        # or, to set to the default:
        # a_grid_trisp = None
        # logn_k_grid_trisp = None

        # set relevant dictionaries with the different probe combinations as keys
        self.set_dicts_for_trisp(gal_bias_1d=gal_bias_1d)
        self.trisp_dict = {}

        print('')

        with sl.timer(f'Computing {which_ng_cov} trispectrum, '):
            # in this case, only the LLLL trispectrum is needed. The compute_trisp_abcd
            # already only uses the 'L' entries, but I keep the 'LLLL' arg for clarity
            if which_ng_cov == 'cNG' and self.which_b1g_in_resp == 'from_input':
                trisp_mmmm = self.compute_trisp_abcd(
                    which_ng_cov, 'LLLL', p_of_k_a=p_of_k_a
                )

            for probe_abcd in unique_probe_combs:
                probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')

                if which_ng_cov == 'cNG' and self.which_b1g_in_resp == 'from_input':
                    # set all keys to the same trispectrum
                    trisp_abcd = trisp_mmmm
                else:
                    trisp_abcd = self.compute_trisp_abcd(
                        which_ng_cov, probe_abcd, p_of_k_a=p_of_k_a
                    )

                self.trisp_dict[probe_ab, probe_cd] = trisp_abcd

    def compute_trisp_abcd(self, which_ng_cov, probe_abcd, p_of_k_a):
        probe_a, probe_b, probe_c, probe_d = probe_abcd

        trisp_func, additional_args = self.get_trisp_func(
            probe_a, probe_b, probe_c, probe_d, which_ng_cov
        )
        trisp_abcd = trisp_func(
            cosmo=self.cosmo_ccl,
            hmc=self.hmc,
            extrap_order_lok=1,
            extrap_order_hik=1,
            use_log=False,
            p_of_k_a=p_of_k_a,
            **additional_args,
        )

        return trisp_abcd

    def get_trisp_func(self, probe_a, probe_b, probe_c, probe_d, which_ng_cov):
        if which_ng_cov == 'SSC' and self.which_b1g_in_resp == 'from_HOD':
            trisp_func = ccl.halos.pk_4pt.halomod_Tk3D_SSC
            additional_args = {
                'prof': self.halo_profile_dict[probe_a],
                'prof2': self.halo_profile_dict[probe_b],
                'prof3': self.halo_profile_dict[probe_c],
                'prof4': self.halo_profile_dict[probe_d],
                'prof12_2pt': self.prof_2pt_dict[probe_a, probe_b],
                'prof34_2pt': self.prof_2pt_dict[probe_c, probe_d],
                'lk_arr': self.logn_k_grid_trisp_SSC,
                'a_arr': self.a_grid_trisp_SSC,
                'extrap_pk': True,
            }

        elif which_ng_cov == 'SSC' and self.which_b1g_in_resp == 'from_input':
            # prof should be the matter profile, since the bias is passed as an argument
            trisp_func = ccl.halos.pk_4pt.halomod_Tk3D_SSC_linear_bias
            additional_args = {
                'prof': self.halo_profile_dict['L'],
                'bias1': self.gal_bias_dict[probe_a],
                'bias2': self.gal_bias_dict[probe_b],
                'bias3': self.gal_bias_dict[probe_c],
                'bias4': self.gal_bias_dict[probe_d],
                'is_number_counts1': self.is_number_counts_dict[probe_a],
                'is_number_counts2': self.is_number_counts_dict[probe_b],
                'is_number_counts3': self.is_number_counts_dict[probe_c],
                'is_number_counts4': self.is_number_counts_dict[probe_d],
                'lk_arr': self.logn_k_grid_trisp_SSC,
                'a_arr': self.a_grid_trisp_SSC,
                'extrap_pk': True,
            }

        elif which_ng_cov == 'cNG' and self.which_b1g_in_resp == 'from_HOD':
            trisp_func = ccl.halos.pk_4pt.halomod_Tk3D_cNG
            additional_args = {
                'prof': self.halo_profile_dict[probe_a],
                'prof2': self.halo_profile_dict[probe_b],
                'prof3': self.halo_profile_dict[probe_c],
                'prof4': self.halo_profile_dict[probe_d],
                'prof12_2pt': self.prof_2pt_dict[probe_a, probe_b],
                'prof13_2pt': self.prof_2pt_dict[probe_a, probe_c],
                'prof14_2pt': self.prof_2pt_dict[probe_a, probe_d],
                'prof24_2pt': self.prof_2pt_dict[probe_b, probe_d],
                'prof32_2pt': self.prof_2pt_dict[probe_c, probe_b],
                'prof34_2pt': self.prof_2pt_dict[probe_c, probe_d],
                'lk_arr': self.logn_k_grid_trisp_cNG,
                'a_arr': self.a_grid_trisp_cNG,
                'separable_growth': self.separable_growth,
            }

        elif which_ng_cov == 'cNG' and self.which_b1g_in_resp == 'from_input':
            # In this case, I only need T_mmmm, as this is multiplied by the galaxy bias
            # as Cov_gggg = \int w_g^4 T_mmmm
            # where w_g = w_delta * b + w_mu (both of which need to be paired with
            # the matter profile)
            trisp_func = ccl.halos.pk_4pt.halomod_Tk3D_cNG
            additional_args = {
                'prof': self.halo_profile_dict['L'],
                'prof2': self.halo_profile_dict['L'],
                'prof3': self.halo_profile_dict['L'],
                'prof4': self.halo_profile_dict['L'],
                'prof12_2pt': self.prof_2pt_dict['L', 'L'],
                'prof13_2pt': self.prof_2pt_dict['L', 'L'],
                'prof14_2pt': self.prof_2pt_dict['L', 'L'],
                'prof24_2pt': self.prof_2pt_dict['L', 'L'],
                'prof32_2pt': self.prof_2pt_dict['L', 'L'],
                'prof34_2pt': self.prof_2pt_dict['L', 'L'],
                'lk_arr': self.logn_k_grid_trisp_cNG,
                'a_arr': self.a_grid_trisp_cNG,
                'separable_growth': self.separable_growth,
            }

        else:
            raise ValueError(
                f'Invalid combination: which_ng_cov = {which_ng_cov!r} '
                "(must be 'SSC' or 'cNG'), which_b1g_in_resp = "
                f"{self.which_b1g_in_resp!r} (must be 'from_input' or 'from_HOD')."
            )

        return trisp_func, additional_args

    def set_dicts_for_trisp(self, gal_bias_1d):

        self.halo_profile_dict = {'L': self.halo_profile_dm, 'G': self.halo_profile_hod}

        self.prof_2pt_dict = {
            # see https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
            ('L', 'L'): ccl.halos.Profile2pt(),
            ('G', 'L'): ccl.halos.Profile2pt(),
            ('L', 'G'): ccl.halos.Profile2pt(),
            ('G', 'G'): ccl.halos.Profile2ptHOD(),
        }

        self.is_number_counts_dict = {'L': False, 'G': True}

        self.gal_bias_dict = {'L': np.ones_like(gal_bias_1d), 'G': gal_bias_1d}

    def compute_ng_cov_probe_block(
        self,
        which_ng_cov: str,
        kernel_A: list,
        kernel_B: list,
        kernel_C: list,
        kernel_D: list,
        ell: np.ndarray,
        trisp_abcd: ccl.tk3d.Tk3D,
        fsky: float,
        sigma2_b_tpl: tuple | None,
        ind_AB: np.ndarray,
        ind_CD: np.ndarray,
        integration_method: str,
        symmetrize_zpairs: bool,
    ):
        zpairs_AB = ind_AB.shape[0]
        zpairs_CD = ind_CD.shape[0]
        nbl = len(ell)

        # switch between the two functions, which are identical except for the
        # sigma2_b argument
        if which_ng_cov == 'SSC':
            ccl_ng_cov_func = ccl.covariances.angular_cl_cov_SSC
            sigma2_b_arg = {'sigma2_B': sigma2_b_tpl}
        elif which_ng_cov == 'cNG':
            ccl_ng_cov_func = ccl.covariances.angular_cl_cov_cNG
            sigma2_b_arg = {}
        else:
            raise ValueError("Invalid value for which_ng_cov. Must be 'SSC' or 'cNG'.")

        cov_ng_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))

        # Diagonal probe blocks case e.g. LLLL, GGGG, GLGL where (A,B) == (C,D)
        if symmetrize_zpairs:
            for ij in range(zpairs_AB):
                for kl in range(ij, zpairs_CD):  # Note: loop starts from ij
                    res = ccl_ng_cov_func(
                        self.cosmo_ccl,
                        tracer1=kernel_A[ind_AB[ij, -2]],
                        tracer2=kernel_B[ind_AB[ij, -1]],
                        ell=ell,
                        t_of_kk_a=trisp_abcd,
                        fsky=fsky,
                        tracer3=kernel_C[ind_CD[kl, -2]],
                        tracer4=kernel_D[ind_CD[kl, -1]],
                        ell2=None,
                        integration_method=integration_method,
                        **sigma2_b_arg,
                    )
                    cov_ng_4D[:, :, ij, kl] = res
                    if kl != ij:
                        cov_ng_4D[:, :, kl, ij] = res.T

        # Off-diagonal probe blocks case e.g. LLGL, LLGG, etc.
        else:
            for ij in range(zpairs_AB):
                for kl in range(zpairs_CD):
                    cov_ng_4D[:, :, ij, kl] = ccl_ng_cov_func(
                        self.cosmo_ccl,
                        tracer1=kernel_A[ind_AB[ij, -2]],
                        tracer2=kernel_B[ind_AB[ij, -1]],
                        ell=ell,
                        t_of_kk_a=trisp_abcd,
                        fsky=fsky,
                        tracer3=kernel_C[ind_CD[kl, -2]],
                        tracer4=kernel_D[ind_CD[kl, -1]],
                        ell2=None,
                        integration_method=integration_method,
                        **sigma2_b_arg,
                    ).T

        return cov_ng_4D

    def compute_ng_cov_3x2pt(
        self,
        which_ng_cov,
        ells,
        integration_method,
        unique_probe_combs,
        nonreq_probe_combs,
        ind_dict,
    ):
        """Compute the specified non-Gaussian covariance term (ssc or cng) for each
        probe block.
        The probe blocks are organised in several categories:
         - unique_probe_combs: These are actually computed
         - symm_probe_combs: The symmetric counterparts of the unique blocks,
           filled by symmetry (can be an empty list if cross_cov=False)
         - nonreq_probe_combs: Probe blocks that are not required and will be set to
           zero. In the new cov_dict structure, these only include the blocks that
           explicitly need to be set to 0, not the ones to be skipped entirely.
           For example, if
           {LL: True, GL: True, cross_cov: False}
           then nonreq_probe_combs = [LLGL, GLLL]
           as opposed to the above plus all the other probe combinations that are not
           requested at all.
        """
        # key of cov_dict
        ng_term = which_ng_cov.lower()

        if which_ng_cov == 'SSC':
            kernel_dict = {'L': self.wf_lensing_obj, 'G': self.wf_density_obj}
        elif which_ng_cov == 'cNG' and self.which_b1g_in_resp == 'from_input':
            kernel_dict = {'L': self.wf_lensing_obj, 'G': self.wf_galaxy_obj}
        elif which_ng_cov == 'cNG' and self.which_b1g_in_resp == 'from_HOD':
            kernel_dict = {'L': self.wf_lensing_obj, 'G': self.wf_density_obj}
        else:
            raise ValueError(
                f'Invalid combination: which_ng_cov = {which_ng_cov!r} '
                "(must be 'SSC' or 'cNG'), which_b1g_in_resp = "
                f"{self.which_b1g_in_resp!r} (must be 'from_input' or 'from_HOD')."
            )

        print('')
        # * compute required blocks
        for probe_abcd in tqdm(unique_probe_combs):
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            probe_2tpl = (probe_ab, probe_cd)
            probe_a, probe_b, probe_c, probe_d = probe_abcd
            symmetrize_zpairs = (probe_a, probe_b) == (probe_c, probe_d)

            tqdm.write(
                f'{which_ng_cov} cov: computing probe combination {(probe_ab, probe_cd)}'
            )

            _sigma2_b_tpl = (
                self.sigma2_b_tpl_dict[probe_ab, probe_cd]
                if which_ng_cov == 'SSC'
                else None
            )

            self.cov_dict[ng_term][probe_2tpl]['4d'] = self.compute_ng_cov_probe_block(
                which_ng_cov=which_ng_cov,
                kernel_A=kernel_dict[probe_a],
                kernel_B=kernel_dict[probe_b],
                kernel_C=kernel_dict[probe_c],
                kernel_D=kernel_dict[probe_d],
                ell=ells,
                trisp_abcd=self.trisp_dict[probe_ab, probe_cd],
                fsky=self.fsky_max_abcd_dict[probe_ab, probe_cd],
                sigma2_b_tpl=_sigma2_b_tpl,
                ind_AB=ind_dict[probe_ab],
                ind_CD=ind_dict[probe_cd],
                integration_method=integration_method,
                symmetrize_zpairs=symmetrize_zpairs,
            )

        # * symmetrize and set to 0 the remaning probe blocks
        sl.symmetrize_and_fill_probe_blocks(
            cov_term_dict=self.cov_dict[ng_term],
            dim='4d',
            unique_probe_combs=unique_probe_combs,
            nonreq_probe_combs=nonreq_probe_combs,
            obs_space='harmonic',
            nbx=len(ells),
            zbins=None,
            ind_dict=ind_dict,
            msg=f'{which_ng_cov} cov: ',
        )

    def check_cov_blocks_symmetry(self):
        # Test if cov is symmetric in ell1, ell2 (only for the diagonal covariance
        # blocks: the off-diagonal need *not* to be symmetric in ell1, ell2)
        for term in self.cov_dict:
            for probe_2tpl in self.cov_dict[term]:
                probe_abcd = probe_2tpl[0] + probe_2tpl[1]

                if probe_abcd in const.HS_DIAG_PROBE_COMBS:
                    try:
                        cov_2d = sl.cov_4D_to_2D(
                            self.cov_dict[term][probe_2tpl]['4d'], block_index='ell'
                        )
                        atol, rtol = 0, 1e-1
                        np.testing.assert_allclose(
                            cov_2d,
                            cov_2d.T,
                            atol=atol,
                            rtol=rtol,
                            err_msg=f'cov {term} {probe_abcd} 2d is '
                            'not symmetric in ell1, ell2',
                        )
                        np.testing.assert_allclose(
                            self.cov_dict[term][probe_2tpl]['4d'],
                            np.transpose(
                                self.cov_dict[term][probe_2tpl]['4d'], (1, 0, 3, 2)
                            ),
                            atol=atol,
                            rtol=rtol,
                            err_msg=f'cov {term} {probe_abcd} 4d is '
                            'is not symmetric in ell1, ell2',
                        )
                    except AssertionError as error:
                        print(f'Probe combination: {term} {probe_abcd}')
                        print(error)
