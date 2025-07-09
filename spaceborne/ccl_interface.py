"""This module should be run with pyccl >= v3.0.0"""

import time
from functools import partial

import healpy as hp
import numpy as np
import pyccl as ccl
from tqdm import tqdm

from spaceborne import cosmo_lib, mask_utils, wf_cl_lib
from spaceborne import sb_lib as sl


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


class PycclClass:
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
        # self.check_specs()   # prolly I don't need these ingredients at all!

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

    def check_specs(self):
        assert self.probe in ['LL', 'GG', '3x2pt'], (
            'probe must be either LL, GG, or 3x2pt'
        )
        assert self.which_ng_cov in ['SSC', 'cNG'], (
            'which_ng_cov must be either SSC or cNG'
        )
        assert self.GL_or_LG == 'GL', (
            'you should update ind_cross (used in ind_dict) '
            'for GL, but we work with GL...'
        )
        assert self.has_rsd is False, 'RSD not validated yet...'

    def pk_obj_from_file(self, pk_filename, plot_pk_z0):
        k_grid_Pk, z_grid_Pk, pk_mm_2d = sl.pk_vinc_file_to_2d_npy(
            pk_filename, plot_pk_z0=plot_pk_z0
        )
        pk_flipped_in_z = np.flip(pk_mm_2d, axis=1)
        scale_factor_grid_pk = cosmo_lib.z_to_a(z_grid_Pk)[::-1]  # flip it
        p_of_k_a = ccl.pk2d.Pk2D(
            a_arr=scale_factor_grid_pk,
            lk_arr=np.log(k_grid_Pk),
            pk_arr=pk_flipped_in_z.T,
            is_logp=False,
        )
        return p_of_k_a

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

    def set_gal_bias_tuple_istf(self, z_grid_lns, bias_function_str, bias_model):
        self.gal_bias_func = self.gal_bias_func_dict[bias_function_str]
        # TODO it's probably better to pass directly the zbin(_lns) centers and edges,
        # TODO rather than nesting them in a cfg file...
        z_means_lns = np.array(
            [
                self.flat_fid_pars_dict[f'zmean{zbin:02d}_photo']
                for zbin in range(1, self.zbins + 1)
            ]
        )
        gal_bias_1d = self.gal_bias_func(z_means_lns)

        z_edges_lns = np.array(
            [
                self.flat_fid_pars_dict[f'zedge{zbin:02d}_photo']
                for zbin in range(1, self.zbins + 2)
            ]
        )
        self.gal_bias_2d = wf_cl_lib.build_galaxy_bias_2d_arr(
            gal_bias_1d,
            z_means_lns,
            z_edges_lns,
            self.zbins,
            z_grid_lns,
            bias_model=bias_model,
            plot_bias=True,
        )
        self.gal_bias_tuple = (z_grid_lns, self.gal_bias_2d)

    def save_gal_bias_table_ascii(self, z_grid_lns, filename):
        assert filename.endswith('.ascii'), 'filename must end with.ascii'
        gal_bias_table = np.hstack((z_grid_lns.reshape(-1, 1), self.gal_bias_2d))
        np.savetxt(filename, gal_bias_table)

    def set_mag_bias_tuple(
        self, z_grid_lns, has_magnification_bias, magcut_lens, poly_fit_values
    ):
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
        self.wf_lensing_obj = [
            ccl.tracers.WeakLensingTracer(
                cosmo=self.cosmo_ccl,
                dndz=(self.nz_src_tuple[0], self.nz_src_tuple[1][:, zbin_idx]),
                ia_bias=self.ia_bias_tuple,
                use_A_ia=False,
                n_samples=n_samples_wf,
            )
            for zbin_idx in range(self.zbins)
        ]

        self.wf_galaxy_obj = []
        for zbin_idx in range(self.zbins):
            # this is needed to be eble to pass mag_bias = None for each zbin
            if self.mag_bias_tuple is None:
                mag_bias_arg = self.mag_bias_tuple
            else:
                mag_bias_arg = (
                    self.mag_bias_tuple[0],
                    self.mag_bias_tuple[1][:, zbin_idx],
                )

            self.wf_galaxy_obj.append(
                ccl.tracers.NumberCountsTracer(
                    cosmo=self.cosmo_ccl,
                    has_rsd=has_rsd,
                    dndz=(self.nz_lns_tuple[0], self.nz_lns_tuple[1][:, zbin_idx]),
                    bias=(self.gal_bias_tuple[0], self.gal_bias_tuple[1][:, zbin_idx]),
                    mag_bias=mag_bias_arg,
                    n_samples=n_samples_wf,
                )
            )

    def set_ell_grid(self, ell_grid):
        self.ell_grid = ell_grid

    def compute_cls(self, ell_grid, p_of_k_a, kernel_a, kernel_b, cl_ccl_kwargs: dict):
        cl_ab_3d = wf_cl_lib.cl_ccl(
            kernel_a,
            kernel_b,
            ell_grid,
            self.zbins,
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
            self.wf_lensing_arr = (
                self.wf_gamma_arr + self.ia_bias_tuple[1][:, None] * self.wf_ia_arr
            )
        else:
            self.wf_ia_arr = np.zeros_like(self.wf_gamma_arr)
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

    # TODO deprecate this func
    def set_sigma2_b_old(
        self, z_grid, fsky, which_sigma2_b, nside_mask, mask_path=None
    ):
        self.a_grid_sigma2_b = cosmo_lib.z_to_a(z_grid)[::-1]
        area_deg2 = fsky * 4 * np.pi * (180 / np.pi) ** 2

        if which_sigma2_b == 'polar_cap_on_the_fly':
            mask = mask_utils.generate_polar_cap_func(area_deg2, nside_mask)

        elif which_sigma2_b == 'from_input_mask':
            mask = hp.read_map(mask_path)

        # normalize the mask and pass it to sigma2_B_from_mask
        if which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
            hp.mollview(
                mask,
                coord=['C', 'E'],
                title='polar cap generated on-the fly',
                cmap='inferno_r',
            )
            cl_mask = hp.anafast(mask)
            ell_mask = np.arange(len(cl_mask))
            cl_mask_norm = cl_mask * (2 * ell_mask + 1) / (4 * np.pi * fsky) ** 2

            # quick check
            fsky_mask = np.sqrt(cl_mask[0] / (4 * np.pi))
            assert np.fabs(fsky_mask / fsky) < 1.01, (
                'fsky_in is not the same as the fsky of the mask'
            )

            # normalization has been checked from
            # https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/scripts/compute_SSC_mask_power.py
            # and is the same as CSST paper https://zenodo.org/records/7813033
            sigma2_b = ccl.covariances.sigma2_B_from_mask(
                cosmo=self.cosmo_ccl, a_arr=self.a_grid_sigma2_b, mask_wl=cl_mask_norm
            )
            self.sigma2_b_tuple = (self.a_grid_sigma2_b, sigma2_b)

        elif which_sigma2_b == 'flat_sky':
            sigma2_b = ccl.covariances.sigma2_B_disc(
                cosmo=self.cosmo_ccl, a_arr=self.a_grid_sigma2_b, fsky=fsky
            )
            self.sigma2_b_tuple = (self.a_grid_sigma2_b, sigma2_b)

        elif which_sigma2_b is None:
            self.sigma2_b_tuple = None

        else:
            raise ValueError(
                'which_sigma2_b must be either "from_input_mask", '
                '"polar_cap_on_the_fly" or None'
            )

    def set_sigma2_b(self, z_grid, which_sigma2_b, mask_obj):
        self.a_grid_sigma2_b = cosmo_lib.z_to_a(z_grid)[::-1]

        # normalize the mask and pass it to sigma2_B_from_mask
        if which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
            sigma2_b = ccl.covariances.sigma2_B_from_mask(
                cosmo=self.cosmo_ccl,
                a_arr=self.a_grid_sigma2_b,
                mask_wl=mask_obj.cl_mask_norm,
            )
            self.sigma2_b_tuple = (self.a_grid_sigma2_b, sigma2_b)

        elif which_sigma2_b == 'flat_sky':
            sigma2_b = ccl.covariances.sigma2_B_disc(
                cosmo=self.cosmo_ccl, a_arr=self.a_grid_sigma2_b, fsky=mask_obj.fsky
            )
            self.sigma2_b_tuple = (self.a_grid_sigma2_b, sigma2_b)

        elif which_sigma2_b is None:
            self.sigma2_b_tuple = None

        else:
            raise ValueError(
                'which_sigma2_b must be either "from_input_mask", '
                '"polar_cap_on_the_fly" or None'
            )

    def initialize_trispectrum(self, which_ng_cov, probe_ordering, pyccl_cfg):
        # some setup
        comp_load_str = 'Loading' if pyccl_cfg['load_cached_tkka'] else 'Computing'
        tkka_path = f'{self.output_path}/cache/trispectrum/{which_ng_cov}'
        k_a_str = self._print_grid_info(which_ng_cov)

        # the default pk must be passed to the Tk3D functions as None, not as
        # 'delta_matter:delta_matter'
        _p_of_k_a = (
            None if self.p_of_k_a == 'delta_matter:delta_matter' else self.p_of_k_a
        )

        # TODO get default grids info when passing a, k = None
        # or, to set to the default:
        # a_grid_tkka = None
        # logn_k_grid_tkka = None

        # set relevant dictionaries with the different probe combinations as keys
        self.set_dicts_for_trisp()

        self.tkka_dict = {}
        for row, (A, B) in tqdm(enumerate(probe_ordering)):
            for col, (C, D) in enumerate(probe_ordering):
                # skip the lower triangle of the matrix
                if col < row:
                    continue

                probe_block = A + B + C + D

                start = time.perf_counter()

                print(
                    f'{comp_load_str} {which_ng_cov}, trispectrum, '
                    f'probe combination {probe_block}'
                )

                # Attempt to load from cache, fall back to computing if necessary
                tkka_abcd = None
                if pyccl_cfg['load_cached_tkka']:
                    try:
                        tkka_abcd = self._load_and_set_tkka(
                            which_ng_cov, tkka_path, k_a_str, probe_block
                        )
                    except FileNotFoundError as e:
                        print(
                            'No trispectra files found in the cache. '
                            'Proceeding to compute them. Error message:\n'
                        )
                        print(e)

                if tkka_abcd is None:
                    tkka_abcd = self._compute_and_save_tkka(
                        which_ng_cov, tkka_path, k_a_str, probe_block, p_of_k_a=None
                    )

                self.tkka_dict[A, B, C, D] = tkka_abcd

                print(f'done in {(time.perf_counter() - start) / 60:.2f} m')

    def _compute_and_save_tkka(
        self, which_ng_cov, tkka_path, k_a_str, probe_block, p_of_k_a
    ):
        A, B, C, D = probe_block
        tkka_func, additional_args = self.get_tkka_func(A, B, C, D, which_ng_cov)
        tkka_abcd = tkka_func(
            cosmo=self.cosmo_ccl,
            hmc=self.hmc,
            extrap_order_lok=1,
            extrap_order_hik=1,
            use_log=False,
            p_of_k_a=p_of_k_a,
            **additional_args,
        )

        a_arr, lk1_arr, lk2_arr, tk_arrays = tkka_abcd.get_spline_arrays()
        np.save(f'{tkka_path}/a_arr_{k_a_str}.npy', a_arr)
        np.save(f'{tkka_path}/lnk1_arr_{k_a_str}.npy', lk1_arr)
        np.save(f'{tkka_path}/lnk2_arr_{k_a_str}.npy', lk2_arr)

        if which_ng_cov == 'SSC':
            np.save(f'{tkka_path}/pk1_arr_{probe_block}_{k_a_str}.npy', tk_arrays[0])
            np.save(f'{tkka_path}/pk2_arr_{probe_block}_{k_a_str}.npy', tk_arrays[1])
        elif which_ng_cov == 'cNG':
            np.save(f'{tkka_path}/trisp_{probe_block}_{k_a_str}.npy', tk_arrays[0])

        return tkka_abcd

    def _load_and_set_tkka(self, which_ng_cov, tkka_path, k_a_str, probe_block):
        a_arr = np.load(f'{tkka_path}/a_arr_{k_a_str}.npy')
        lk1_arr = np.load(f'{tkka_path}/lnk1_arr_{k_a_str}.npy')
        lk2_arr = np.load(f'{tkka_path}/lnk2_arr_{k_a_str}.npy')
        (
            np.testing.assert_allclose(lk1_arr, lk2_arr, atol=0, rtol=1e-9),
            ('k1_arr and lk2_arr different'),
        )

        if which_ng_cov == 'SSC':
            pk1_arr = np.load(f'{tkka_path}/pk1_arr_{probe_block}_{k_a_str}.npy')
            pk2_arr = np.load(f'{tkka_path}/pk2_arr_{probe_block}_{k_a_str}.npy')
            tk3d_kwargs = {'tkk_arr': None, 'pk1_arr': pk1_arr, 'pk2_arr': pk2_arr}
        elif which_ng_cov == 'cNG':
            tkk_arr = np.load(f'{tkka_path}/trisp_{probe_block}_{k_a_str}.npy')
            tk3d_kwargs = {'tkk_arr': tkk_arr, 'pk1_arr': None, 'pk2_arr': None}

        tkka_abcd = ccl.tk3d.Tk3D(
            a_arr=a_arr,
            lk_arr=lk1_arr,
            is_logt=False,
            extrap_order_lok=1,
            extrap_order_hik=1,
            **tk3d_kwargs,
        )

        return tkka_abcd

    def _print_grid_info(self, which_ng_cov):
        # get grids
        a_grid = getattr(self, f'a_grid_tkka_{which_ng_cov}', None)
        logn_k_grid = getattr(self, f'logn_k_grid_tkka_{which_ng_cov}', None)

        # print info
        if a_grid is not None and logn_k_grid is not None:
            print(
                f'{which_ng_cov} trispectrum: z points = {a_grid.size}, '
                f'k points = {logn_k_grid.size}'
            )

        # set string
        k_a_str = (
            f'amin{a_grid.min():.2f}_amax{a_grid.max():.2f}'
            f'_asteps{a_grid.size}_lnkmin{logn_k_grid.min():.2f}'
            f'_lnkmax{logn_k_grid.max():.2f}_ksteps{logn_k_grid.size}'
        )
        return k_a_str

    def get_tkka_func(self, A, B, C, D, which_ng_cov):
        if which_ng_cov == 'SSC':
            if self.which_b1g_in_resp == 'from_HOD':
                tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_SSC
                additional_args = {
                    'prof': self.halo_profile_dict[A],
                    'prof2': self.halo_profile_dict[B],
                    'prof3': self.halo_profile_dict[C],
                    'prof4': self.halo_profile_dict[D],
                    'prof12_2pt': self.prof_2pt_dict[A, B],
                    'prof34_2pt': self.prof_2pt_dict[C, D],
                    'lk_arr': self.logn_k_grid_tkka_SSC,
                    'a_arr': self.a_grid_tkka_SSC,
                    'extrap_pk': True,
                }

            elif self.which_b1g_in_resp == 'from_input':
                tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_SSC_linear_bias
                additional_args = {
                    # prof should be HaloProfileNFW, in this case
                    'prof': self.halo_profile_dict['L'],
                    'bias1': self.gal_bias_dict[A],
                    'bias2': self.gal_bias_dict[B],
                    'bias3': self.gal_bias_dict[C],
                    'bias4': self.gal_bias_dict[D],
                    'is_number_counts1': self.is_number_counts_dict[A],
                    'is_number_counts2': self.is_number_counts_dict[B],
                    'is_number_counts3': self.is_number_counts_dict[C],
                    'is_number_counts4': self.is_number_counts_dict[D],
                    'lk_arr': self.logn_k_grid_tkka_SSC,
                    'a_arr': self.a_grid_tkka_SSC,
                    'extrap_pk': True,
                }

        elif which_ng_cov == 'cNG':
            tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_cNG
            additional_args = {
                'prof': self.halo_profile_dict[A],
                'prof2': self.halo_profile_dict[B],
                'prof3': self.halo_profile_dict[C],
                'prof4': self.halo_profile_dict[D],
                'prof12_2pt': self.prof_2pt_dict[A, B],
                'prof13_2pt': self.prof_2pt_dict[A, C],
                'prof14_2pt': self.prof_2pt_dict[A, D],
                'prof24_2pt': self.prof_2pt_dict[B, D],
                'prof32_2pt': self.prof_2pt_dict[C, B],
                'prof34_2pt': self.prof_2pt_dict[C, D],
                'lk_arr': self.logn_k_grid_tkka_cNG,
                'a_arr': self.a_grid_tkka_cNG,
                'separable_growth': False,
            }

        else:
            raise ValueError(
                f'Invalid value for which_ng_cov. It is {which_ng_cov}, '
                "must be 'SSC' or 'cNG'."
            )

        return tkka_func, additional_args

    def set_dicts_for_trisp(self):
        # tODO pass this? make sure to be consistent
        gal_bias_1d = self.gal_bias_func(cosmo_lib.a_to_z(self.a_grid_tkka_SSC))

        self.halo_profile_dict = {
            'L': self.halo_profile_dm,
            'G': self.halo_profile_hod,
        }

        self.prof_2pt_dict = {
            # see https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
            ('L', 'L'): ccl.halos.Profile2pt(),
            ('G', 'L'): ccl.halos.Profile2pt(),
            ('L', 'G'): ccl.halos.Profile2pt(),
            ('G', 'G'): ccl.halos.Profile2ptHOD(),
        }

        self.is_number_counts_dict = {
            'L': False,
            'G': True,
        }

        self.gal_bias_dict = {
            'L': np.ones_like(gal_bias_1d),
            'G': gal_bias_1d,
        }

    def compute_ng_cov_ccl(
        self,
        which_ng_cov,
        kernel_A,
        kernel_B,
        kernel_C,
        kernel_D,
        ell,
        tkka,
        f_sky,
        ind_AB,
        ind_CD,
        integration_method,
    ):
        zpairs_AB = ind_AB.shape[0]
        zpairs_CD = ind_CD.shape[0]
        nbl = len(ell)

        start_time = time.perf_counter()
        # switch between the two functions, which are identical except for the
        # sigma2_b argument
        if which_ng_cov == 'SSC':
            ccl_ng_cov_func = ccl.covariances.angular_cl_cov_SSC
            sigma2_b_arg = {
                'sigma2_B': self.sigma2_b_tuple,
            }
        elif which_ng_cov == 'cNG':
            ccl_ng_cov_func = ccl.covariances.angular_cl_cov_cNG
            sigma2_b_arg = {}
        else:
            raise ValueError("Invalid value for which_ng_cov. Must be 'SSC' or 'cNG'.")

        cov_ng_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))
        for ij in tqdm(range(zpairs_AB)):
            for kl in range(zpairs_CD):
                cov_ng_4D[:, :, ij, kl] = ccl_ng_cov_func(
                    self.cosmo_ccl,
                    tracer1=kernel_A[ind_AB[ij, -2]],
                    tracer2=kernel_B[ind_AB[ij, -1]],
                    ell=ell,
                    t_of_kk_a=tkka,
                    fsky=f_sky,
                    tracer3=kernel_C[ind_CD[kl, -2]],
                    tracer4=kernel_D[ind_CD[kl, -1]],
                    ell2=None,
                    integration_method=integration_method,
                    **sigma2_b_arg,
                )

        print(
            f'{which_ng_cov} computed with pyccl in '
            f'{(time.perf_counter() - start_time) / 60:.0f} min'
        )

        return cov_ng_4D

    def compute_ng_cov_3x2pt(
        self, which_ng_cov, ell, f_sky, integration_method, probe_ordering, ind_dict
    ):
        cov_ng_3x2pt_dict_8D = {}

        kernel_dict = {'L': self.wf_lensing_obj, 'G': self.wf_galaxy_obj}

        for row, (probe_a, probe_b) in enumerate(probe_ordering):
            for col, (probe_c, probe_d) in enumerate(probe_ordering):
                if col >= row:
                    print(
                        'CCL 3x2pt cov: working on probe combination ',
                        probe_a,
                        probe_b,
                        probe_c,
                        probe_d,
                    )
                    cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                        self.compute_ng_cov_ccl(
                            which_ng_cov=which_ng_cov,
                            kernel_A=kernel_dict[probe_a],
                            kernel_B=kernel_dict[probe_b],
                            kernel_C=kernel_dict[probe_c],
                            kernel_D=kernel_dict[probe_d],
                            ell=ell,
                            tkka=self.tkka_dict[probe_a, probe_b, probe_c, probe_d],
                            f_sky=f_sky,
                            ind_AB=ind_dict[probe_a, probe_b],
                            ind_CD=ind_dict[probe_c, probe_d],
                            integration_method=integration_method,
                        )
                    )

                else:
                    print(  # fmt: skip
                        'CCL 3x2pt cov: skipping probe combination ',
                        probe_a, probe_b, probe_c, probe_d
                    )  # fmt: skip
                    cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = np.copy(
                        cov_ng_3x2pt_dict_8D[
                            probe_c, probe_d, probe_a, probe_b
                        ].transpose(1, 0, 3, 2)
                    )

        self.cov_ng_3x2pt_dict_8D = cov_ng_3x2pt_dict_8D

        if which_ng_cov == 'SSC':
            self.cov_ssc_ccl_3x2pt_dict_8D = self.cov_ng_3x2pt_dict_8D
        if which_ng_cov == 'cNG':
            self.cov_cng_ccl_3x2pt_dict_8D = self.cov_ng_3x2pt_dict_8D

        self.check_cov_blocks_simmetry()

    def check_cov_blocks_simmetry(self):
        # Test if cov is symmetric in ell1, ell2 (only for the diagonal covariance
        # blocks: the off-diagonal need *not* to be symmetric in ell1, ell2)
        for key in self.cov_ng_3x2pt_dict_8D:
            if key in (
                ('L', 'L', 'L', 'L'),
                ('G', 'L', 'G', 'L'),
                ('G', 'G', 'G', 'G'),
            ):
                try:
                    cov_2d = sl.cov_4D_to_2D(
                        self.cov_ng_3x2pt_dict_8D[key], block_index='ell'
                    )
                    assert np.allclose(cov_2d, cov_2d.T, atol=0, rtol=1e-5)
                    np.testing.assert_allclose(
                        self.cov_ng_3x2pt_dict_8D[key],
                        #    np.transpose(self.cov_ng_3x2pt_dict_8D[key], (1, 0, 2, 3)),
                        np.transpose(self.cov_ng_3x2pt_dict_8D[key], (1, 0, 3, 2)),
                        rtol=1e-5,
                        atol=0,
                        err_msg=f'cov_ng_4D {key} is not symmetric in ell1, ell2',
                    )
                except AssertionError as error:
                    print(error)

    def save_cov_blocks(self, cov_path, cov_filename):
        for probe_a, probe_b, probe_c, probe_d in self.cov_ng_3x2pt_dict_8D:
            cov_filename_fmt = cov_filename.format(
                probe_a=probe_a, probe_b=probe_b, probe_c=probe_c, probe_d=probe_d
            )

            np.savez_compressed(
                f'{cov_path}/{cov_filename_fmt}',
                self.cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d],
            )

    def load_cov_blocks(self, cov_path, cov_filename, probe_ordering):
        self.cov_ng_3x2pt_dict_8D = {}

        for row, (probe_a, probe_b) in enumerate(probe_ordering):
            for col, (probe_c, probe_d) in enumerate(probe_ordering):
                if col >= row:
                    print(probe_a, probe_b, probe_c, probe_d)

                    cov_filename_fmt = cov_filename.format(
                        probe_a=probe_a,
                        probe_b=probe_b,
                        probe_c=probe_c,
                        probe_d=probe_d,
                    )
                    self.cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                        np.load(f'{cov_path}/{cov_filename_fmt}')['arr_0']
                    )

                else:
                    self.cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                        self.cov_ng_3x2pt_dict_8D[
                            probe_c, probe_d, probe_a, probe_b
                        ].transpose(1, 0, 3, 2)
                    )
