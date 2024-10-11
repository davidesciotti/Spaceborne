""" This module should be run with pyccl >= v3.0.0
"""

from functools import partial
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
from pyccl.errors import CCLError
import numpy as np
import pyccl as ccl
import os
import sys
from matplotlib import cm
from tqdm import tqdm
import healpy as hp
from scipy.interpolate import interp1d

import spaceborne.my_module as mm
import spaceborne.cosmo_lib as cosmo_lib
import spaceborne.wf_cl_lib as wf_cl_lib
import spaceborne.sigma2_SSC as sigma2_SSC
import common_cfg.mpl_cfg as mpl_cfg
import spaceborne.mask_fits_to_cl as mask_utils

plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)

ccl.spline_params['A_SPLINE_NA_PK'] = 240  # gives CAMB error if too high


class PycclClass():

    def __init__(self, fid_pars_dict):

        # self.zbins = general_cfg['zbins']
        # self.nz_tuple = general_cfg['nz_tuple']
        # self.ind = covariance_cfg['ind']
        # self.GL_or_LG = covariance_cfg['GL_or_LG']
        # self.nbl = len(ell_grid)

        # self.pyccl_cfg = covariance_cfg['PyCCL_cfg']
        # self.n_samples_wf = self.pyccl_cfg['n_samples_wf']
        # this is needed only for a visual check of the cls, which are not used for SSC anyways
        # self.has_rsd = general_cfg['has_rsd']
        # self.has_magnification_bias = general_cfg['has_magnification_bias']
        # ! settings

        # get number of redshift pairs
        # self.zpairs_auto, self.zpairs_cross, self.zpairs_3x2pt = mm.get_zpairs(self.zbins)
        # self.ind_auto = self.ind[:self.zpairs_auto, :]
        # self.ind_cross = self.ind[self.zpairs_auto:self.zpairs_auto + self.zpairs_cross, :]

        # Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
        # functions
        self.flat_fid_pars_dict = mm.flatten_dict(fid_pars_dict)
        cosmo_dict_ccl = cosmo_lib.map_keys(self.flat_fid_pars_dict, key_mapping=None)
        self.cosmo_ccl = cosmo_lib.instantiate_cosmo_ccl_obj(cosmo_dict_ccl,
                                                             fid_pars_dict['other_params']['camb_extra_parameters'])

        self.gal_bias_func_dict = {
            'analytical': wf_cl_lib.b_of_z_analytical,
            'leporifit': wf_cl_lib.b_of_z_fs1_leporifit,
            'pocinofit': wf_cl_lib.b_of_z_fs1_pocinofit,
            'fs2_fit': wf_cl_lib.b_of_z_fs2_fit,
        }
        # self.check_specs()   # prolly I don't need these ingredients at all!

        # initialize halo model
        # from https://github.com/LSSTDESC/CCL/blob/4df2a29eca58d7cd171bc1986e059fd35f425d45/benchmarks/test_covariances.py
        # see also https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/tools/covariance_NG.py#L282

        self.mass_def = ccl.halos.MassDef200m
        self.c_m_relation = ccl.halos.ConcentrationDuffy08(mass_def=self.mass_def)
        self.hmf = ccl.halos.MassFuncTinker10(mass_def=self.mass_def)
        self.hbf = ccl.halos.HaloBiasTinker10(mass_def=self.mass_def)
        self.hmc = ccl.halos.HMCalculator(mass_function=self.hmf, halo_bias=self.hbf, mass_def=self.mass_def)
        self.halo_profile_nfw = ccl.halos.HaloProfileNFW(mass_def=self.mass_def, concentration=self.c_m_relation)
        self.halo_profile_hod = ccl.halos.HaloProfileHOD(mass_def=self.mass_def, concentration=self.c_m_relation)

    def check_specs(self):
        assert self.probe in ['LL', 'GG', '3x2pt'], 'probe must be either LL, GG, or 3x2pt'
        assert self.which_ng_cov in ['SSC', 'cNG'], 'which_ng_cov must be either SSC or cNG'
        assert self.GL_or_LG == 'GL', 'you should update ind_cross (used in ind_dict) for GL, but we work with GL...'
        assert self.has_rsd == False, 'RSD not validated yet...'

    # fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX
    # notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
    # Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb
    # HALO MODEL PRESCRIPTIONS:
    # KiDS1000 Methodology: https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)
    # Krause2017: https://arxiv.org/pdf/1601.05779.pdf

    def pk_obj_from_file(self, pk_filename, plot_pk_z0):

        k_grid_Pk, z_grid_Pk, pk_mm_2d = mm.pk_vinc_file_to_2d_npy(pk_filename, plot_pk_z0=plot_pk_z0)
        pk_flipped_in_z = np.flip(pk_mm_2d, axis=1)
        scale_factor_grid_pk = cosmo_lib.z_to_a(z_grid_Pk)[::-1]  # flip it
        p_of_k_a = ccl.pk2d.Pk2D(a_arr=scale_factor_grid_pk, lk_arr=np.log(k_grid_Pk),
                                 pk_arr=pk_flipped_in_z.T, is_logp=False)
        return p_of_k_a

    def set_nz(self, n_of_z_load):
        self.zgrid_nz = n_of_z_load[:, 0]
        self.n_of_z = n_of_z_load[:, 1:]
        self.nz_tuple = (self.zgrid_nz, self.n_of_z)

    def check_nz_tuple(self, zbins):
        assert isinstance(self.nz_tuple, tuple), 'nz_tuple must be a tuple'
        assert self.nz_tuple[1].shape == (len(self.zgrid_nz), zbins), \
            'nz_tuple must be a 2D array with shape (len(z_grid_nofz), zbins)'

    def set_ia_bias_tuple(self, z_grid, has_ia):

        self.has_ia = has_ia

        if self.has_ia:
            ia_bias_1d = wf_cl_lib.build_ia_bias_1d_arr(z_grid, cosmo_ccl=self.cosmo_ccl,
                                                        flat_fid_pars_dict=self.flat_fid_pars_dict,
                                                        input_z_grid_lumin_ratio=None,
                                                        input_lumin_ratio=None, output_F_IA_of_z=False)
            self.ia_bias_tuple = (z_grid, ia_bias_1d)

        else:
            self.ia_bias_tuple = None

    def set_gal_bias_tuple_spv3(self, z_grid, magcut_lens, poly_fit_values):

        gal_bias_func = self.gal_bias_func_dict['fs2_fit']
        self.gal_bias_func_ofz = partial(gal_bias_func, magcut_lens=magcut_lens / 10, poly_fit_values=poly_fit_values)
        gal_bias_1d = self.gal_bias_func_ofz(z_grid)

        # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
        self.gal_bias_2d = np.repeat(gal_bias_1d.reshape(1, -1), self.zbins, axis=0).T
        self.gal_bias_tuple = (z_grid, self.gal_bias_2d)

    def set_gal_bias_tuple_istf(self, z_grid, bias_function_str, bias_model):
        gal_bias_func = self.gal_bias_func_dict[bias_function_str]
        z_means = np.array([self.flat_fid_pars_dict[f'zmean{zbin:02d}_photo'] for zbin in range(1, self.zbins + 1)])
        gal_bias_1d = gal_bias_func(z_means)

        z_edges = np.array([self.flat_fid_pars_dict[f'zedge{zbin:02d}_photo'] for zbin in range(1, self.zbins + 2)])
        self.gal_bias_2d = wf_cl_lib.build_galaxy_bias_2d_arr(
            gal_bias_1d, z_means, z_edges, self.zbins, z_grid, bias_model=bias_model, plot_bias=True)
        self.gal_bias_tuple = (z_grid, self.gal_bias_2d)

    def save_gal_bias_table_ascii(self, z_grid, filename):
        assert filename.endswith('.ascii'), 'filename must end with.ascii'
        gal_bias_table = np.hstack((z_grid.reshape(-1, 1), self.gal_bias_2d))
        np.savetxt(filename, gal_bias_table)

    def set_mag_bias_tuple(self, z_grid, has_magnification_bias, magcut_lens, poly_fit_values):
        if has_magnification_bias:
            # this is only to ensure compatibility with wf_ccl function. In reality, the same array is given for each bin
            mag_bias_1d = wf_cl_lib.s_of_z_fs2_fit(z_grid, magcut_lens=magcut_lens, poly_fit_values=poly_fit_values)
            mag_bias_2d = np.repeat(mag_bias_1d.reshape(1, -1), self.zbins, axis=0).T
            self.mag_bias_tuple = (z_grid, mag_bias_2d)
        else:
            # this is the correct way to set the magnification bias values so that the actual bias is 1, ant the corresponding
            # wf_mu is zero (which is, in theory, the case mag_bias_tuple=None, which however causes pyccl to crash!)
            # mag_bias_2d = (np.ones_like(gal_bias_2d) * + 2) / 5
            # mag_bias_tuple = (zgrid_nz, mag_bias_2d)
            self.mag_bias_tuple = None

    def set_kernel_obj(self, has_rsd, n_samples_wf):

        self.wf_lensing_obj = [ccl.tracers.WeakLensingTracer(cosmo=self.cosmo_ccl,
                                                             dndz=(self.nz_tuple[0],
                                                                   self.nz_tuple[1][:, zbin_idx]),
                                                             ia_bias=self.ia_bias_tuple,
                                                             use_A_ia=False,
                                                             n_samples=n_samples_wf) for zbin_idx in range(self.zbins)]
        self.wf_galaxy_obj = []
        for zbin_idx in range(self.zbins):

            # this is needed to be eble to pass mag_bias = None for each zbin
            if self.mag_bias_tuple is None:
                mag_bias_arg = self.mag_bias_tuple
            else:
                mag_bias_arg = (self.mag_bias_tuple[0], self.mag_bias_tuple[1][:, zbin_idx])

            self.wf_galaxy_obj.append(ccl.tracers.NumberCountsTracer(cosmo=self.cosmo_ccl,
                                                                     has_rsd=has_rsd,
                                                                     dndz=(self.nz_tuple[0],
                                                                           self.nz_tuple[1][:, zbin_idx]),
                                                                     bias=(self.gal_bias_tuple[0],
                                                                           self.gal_bias_tuple[1][:, zbin_idx]),
                                                                     mag_bias=mag_bias_arg,
                                                                     n_samples=n_samples_wf))

    def set_ell_grid(self, ell_grid):
        self.ell_grid = ell_grid

    def compute_cls(self, ell_grid, p_of_k_a, kernel_a, kernel_b, limber_integration_method):

        cl_ab_3d = wf_cl_lib.cl_PyCCL(kernel_a, kernel_b, ell_grid, self.zbins,
                                      p_of_k_a=p_of_k_a, cosmo=self.cosmo_ccl,
                                      limber_integration_method=limber_integration_method)

        return cl_ab_3d

    def set_kernel_arr(self, z_grid_wf, has_magnification_bias):

        self.z_grid_wf = z_grid_wf
        a_arr = cosmo_lib.z_to_a(z_grid_wf)
        comoving_distance = ccl.comoving_radial_distance(self.cosmo_ccl, a_arr)

        wf_lensing_tot_arr = np.asarray([self.wf_lensing_obj[zbin_idx].get_kernel(comoving_distance)
                                        for zbin_idx in range(self.zbins)])

        wf_galaxy_tot_arr = np.asarray([self.wf_galaxy_obj[zbin_idx].get_kernel(comoving_distance)
                                       for zbin_idx in range(self.zbins)])

        # lensing
        self.wf_gamma_arr = wf_lensing_tot_arr[:, 0, :].T
        if self.has_ia:
            self.wf_ia_arr = wf_lensing_tot_arr[:, 1, :].T
            self.wf_lensing_arr = self.wf_gamma_arr + self.ia_bias_tuple[1][:, None] * self.wf_ia_arr
        else:
            self.wf_ia_arr = np.zeros_like(self.wf_gamma_arr)
            self.wf_lensing_arr = self.wf_gamma_arr

        # galaxy
        self.wf_delta_arr = wf_galaxy_tot_arr[:, 0, :].T
        self.wf_mu_arr = wf_galaxy_tot_arr[:, -1, :].T if has_magnification_bias else np.zeros_like(self.wf_delta_arr)

        # in the case of ISTF, the galaxt bias is bin-per-bin and is therefore included in the kernels. Add it here
        # for a fair comparison with vincenzo's kernels, in the plot.
        # * Note that the galaxy bias is included in the wf_ccl_obj in any way, both in ISTF and SPV3 cases! It must
        # * in fact be passed to the angular_cov_SSC function
        self.wf_galaxy_wo_gal_bias_arr = self.wf_delta_arr + self.wf_mu_arr
        self.wf_galaxy_w_gal_bias_arr = self.wf_delta_arr * self.gal_bias_2d + self.wf_mu_arr

    # ! ==========================================================================================================================================================================

    def set_sigma2_b(self, z_grid, fsky, which_sigma2_b, 
                     nside_mask, mask_path=None):

        self.a_grid_sigma2_b = cosmo_lib.z_to_a(z_grid)[::-1]
        area_deg2 = fsky * 4 * np.pi * (180 / np.pi)**2

        if which_sigma2_b == 'polar_cap_on_the_fly':
            mask = mask_utils.generate_polar_cap(area_deg2, nside_mask)

        elif which_sigma2_b == 'from_input_mask':
            mask = hp.read_map(mask_path)

        if which_sigma2_b in ['polar_cap_on_the_fly', 'from_input_mask']:
            hp.mollview(mask, coord=['C', 'E'], title='polar cap generated on-the fly', cmap='inferno_r')
            cl_mask = hp.anafast(mask)
            ell_mask = np.arange(len(cl_mask))
            cl_mask_norm = cl_mask * (2 * ell_mask + 1) / (4 * np.pi * fsky)**2
            # quick check
            fsky_mask = np.sqrt(cl_mask[0] / (4 * np.pi))
            print(f'fsky from mask: {fsky_mask:.4f}')
            assert np.fabs(fsky_mask / fsky) < 1.01, 'fsky_in is not the same as the fsky of the mask'

        if which_sigma2_b == 'from_input_mask':
            # normalization has been checked from https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/scripts/compute_SSC_mask_power.py
            # and is the same as CSST paper https://zenodo.org/records/7813033
            sigma2_b = ccl.covariances.sigma2_B_from_mask(
                cosmo=self.cosmo_ccl, a_arr=self.a_grid_sigma2_b, mask_wl=cl_mask_norm, p_of_k_a='delta_matter:delta_matter')
            self.sigma2_b_tuple = (self.a_grid_sigma2_b, sigma2_b)

        elif which_sigma2_b == 'polar_cap_on_the_fly':
            sigma2_b = ccl.covariances.sigma2_B_from_mask(
                cosmo=self.cosmo_ccl, a_arr=self.a_grid_sigma2_b, mask_wl=cl_mask_norm)
            self.sigma2_b_tuple = (self.a_grid_sigma2_b, sigma2_b)

        elif which_sigma2_b == 'flat_sky':
            sigma2_b = ccl.covariances.sigma2_B_disc(
                cosmo=self.cosmo_ccl, a_arr=self.a_grid_sigma2_b, fsky=fsky)
            self.sigma2_b_tuple = (self.a_grid_sigma2_b, sigma2_b)

        elif which_sigma2_b == None:
            self.sigma2_b_tuple = None
            
        # breakpoint()

        else:
            raise ValueError('which_sigma2_b must be either "from_input_mask", "polar_cap_on_the_fly" or None')

    def initialize_trispectrum(self, which_ng_cov, probe_ordering, pyccl_cfg):

        # save_tkka = pyccl_cfg['save_tkka']
        comp_load_str = 'Loading' if pyccl_cfg['load_precomputed_tkka'] else 'Computing'

        # tkka_folder = f'Tk3D_{which_ng_cov}'
        # tkka_path = f'{pyccl_cfg["cov_path"]}/{tkka_folder}'

        # k_z_str = f'zmin{pyccl_cfg["z_grid_tkka_min"]:.1f}_zmax{pyccl_cfg["z_grid_tkka_max"]:.1f}_zsteps{pyccl_cfg[f"z_grid_tkka_steps_{which_ng_cov}"]:d}_' \
        # f'kmin{pyccl_cfg["k_grid_tkka_min"]:.1e}_kmax{pyccl_cfg["k_grid_tkka_max"]:.1e}_ksteps{pyccl_cfg[f"k_grid_tkka_steps_{which_ng_cov}"]:d}'

        self.a_grid_tkka_SSC = np.linspace(
            cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_max']),
            cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_min']),
            pyccl_cfg['z_grid_tkka_steps_SSC'])

        self.z_grid_tkka_SSC = cosmo_lib.a_to_z(self.a_grid_tkka_SSC)[::-1]

        logn_k_grid_tkka_SSC = np.log(np.geomspace(pyccl_cfg['k_grid_tkka_min'],
                                                   pyccl_cfg['k_grid_tkka_max'],
                                                   pyccl_cfg['k_grid_tkka_steps_SSC']))

        a_grid_tkka_cNG = np.linspace(
            cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_max']),
            cosmo_lib.z_to_a(pyccl_cfg['z_grid_tkka_min']),
            pyccl_cfg['z_grid_tkka_steps_cNG'])

        logn_k_grid_tkka_cNG = np.log(np.geomspace(pyccl_cfg['k_grid_tkka_min'],
                                                   pyccl_cfg['k_grid_tkka_max'],
                                                   pyccl_cfg['k_grid_tkka_steps_cNG']))

        # or, to set to the default:
        # a_grid_tkka = None
        # logn_k_grid_tkka = None

        tkka_start_time = time.perf_counter()
        # TODO pk from input files
        # This is the correct way to initialize the trispectrum (I Asked David Alonso about this.)
        halo_profile_dict = {
            'L': self.halo_profile_nfw,
            'G': self.halo_profile_hod,
        }
        prof_2pt_dict = {
            # see again https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
            ('L', 'L'): ccl.halos.Profile2pt(),
            ('G', 'L'): ccl.halos.Profile2pt(),
            ('L', 'G'): ccl.halos.Profile2pt(),
            ('G', 'G'): ccl.halos.Profile2ptHOD(),
        }

        # store the trispectrum for the various probes in a dictionary

        # the default pk must be passed to yhe Tk3D functions as None, not as 'delta_matter:delta_matter'
        p_of_k_a = None if self.p_of_k_a == 'delta_matter:delta_matter' else self.p_of_k_a

        if self.a_grid_tkka_SSC is not None and logn_k_grid_tkka_SSC is not None and which_ng_cov == 'SSC':
            print(f'SSC tkka: z points = {self.a_grid_tkka_SSC.size}, k points = {logn_k_grid_tkka_SSC.size}')
        if a_grid_tkka_cNG is not None and logn_k_grid_tkka_cNG is not None and which_ng_cov == 'cNG':
            print(f'cNG tkka: z points = {a_grid_tkka_cNG.size}, k points = {logn_k_grid_tkka_cNG.size}')

        self.tkka_dict = {}
        self.responses_dict = {}
        for row, (A, B) in tqdm(enumerate(probe_ordering)):
            for col, (C, D) in enumerate(probe_ordering):
                probe_block = A + B + C + D

                if col >= row:
                    print(f'{comp_load_str} trispectrum for {which_ng_cov}, probe combination {probe_block}')

                if col >= row and pyccl_cfg['load_precomputed_tkka']:

                    assert False, 'Probably this section must be deleted'

                    save_tkka = False

                    a_arr = np.load(f'{tkka_path}/a_arr_tkka_{probe_block}_{k_z_str}.npy')
                    k1_arr = np.load(f'{tkka_path}/k1_arr_tkka_{probe_block}_{k_z_str}.npy')
                    k2_arr = np.load(f'{tkka_path}/k2_arr_tkka_{probe_block}_{k_z_str}.npy')
                    if which_ng_cov == 'SSC':
                        pk1_arr_tkka = np.load(f'{tkka_path}/pk1_arr_tkka_{probe_block}_{k_z_str}.npy')
                        pk2_arr_tkka = np.load(f'{tkka_path}/pk2_arr_tkka_{probe_block}_{k_z_str}.npy')
                        tk3d_kwargs = {
                            'tkk_arr': None,
                            'pk1_arr': pk1_arr_tkka,
                            'pk2_arr': pk2_arr_tkka,
                        }
                    elif which_ng_cov == 'cNG':
                        tkk_arr = np.load(f'{tkka_path}/tkk_arr_{probe_block}_{k_z_str}.npy')
                        tk3d_kwargs = {
                            'tkk_arr': tkk_arr,
                            'pk1_arr': None,
                            'pk2_arr': None,
                        }

                    assert np.array_equal(k1_arr, k2_arr), 'k1_arr and k2_arr must be equal'

                    self.tkka_dict[A, B, C, D] = ccl.tk3d.Tk3D(a_arr=a_arr,
                                                               lk_arr=k1_arr,
                                                               is_logt=False,
                                                               extrap_order_lok=1,
                                                               extrap_order_hik=1,
                                                               **tk3d_kwargs,
                                                               )

                elif col >= row and not pyccl_cfg['load_precomputed_tkka']:

                    # not very nice to put this if-else in the for loop, but A, B, C, D are referenced only here
                    if which_ng_cov == 'SSC':
                        tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_SSC
                        additional_args = {
                            'prof12_2pt': prof_2pt_dict[A, B],
                            'prof34_2pt': prof_2pt_dict[C, D],
                            'lk_arr': logn_k_grid_tkka_SSC,
                            'a_arr': self.a_grid_tkka_SSC,
                            'extrap_pk': True,
                        }
                    elif which_ng_cov == 'cNG':
                        tkka_func = ccl.halos.pk_4pt.halomod_Tk3D_cNG
                        additional_args = {
                            'prof12_2pt': prof_2pt_dict[A, B],
                            'prof13_2pt': prof_2pt_dict[A, C],
                            'prof14_2pt': prof_2pt_dict[A, D],
                            'prof24_2pt': prof_2pt_dict[B, D],
                            'prof32_2pt': prof_2pt_dict[C, B],
                            'prof34_2pt': prof_2pt_dict[C, D],
                            'lk_arr': logn_k_grid_tkka_cNG,
                            'a_arr': a_grid_tkka_cNG,
                        }
                    else:
                        raise ValueError(
                            f"Invalid value for which_ng_cov. It is {which_ng_cov}, must be 'SSC' or 'cNG'.")

                    self.tkka_dict[A, B, C, D], self.responses_dict[A, B, C, D] = tkka_func(cosmo=self.cosmo_ccl,
                                                                                            hmc=self.hmc,
                                                                                            prof=halo_profile_dict[A],
                                                                                            prof2=halo_profile_dict[B],
                                                                                            prof3=halo_profile_dict[C],
                                                                                            prof4=halo_profile_dict[D],
                                                                                            extrap_order_lok=1, extrap_order_hik=1,
                                                                                            use_log=False,
                                                                                            p_of_k_a=p_of_k_a,
                                                                                            **additional_args)

        print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - tkka_start_time))

        return

    def compute_ng_cov_ccl(self, which_ng_cov, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                           ind_AB, ind_CD, integration_method):
        zpairs_AB = ind_AB.shape[0]
        zpairs_CD = ind_CD.shape[0]
        nbl = len(ell)

        start_time = time.perf_counter()
        # switch between the two functions, which are identical except for the sigma2_b argument
        if which_ng_cov == 'SSC':
            ccl_ng_cov_func = ccl.covariances.angular_cl_cov_SSC
            sigma2_b_arg = {'sigma2_B': self.sigma2_b_tuple,
                            }
        elif which_ng_cov == 'cNG':
            ccl_ng_cov_func = ccl.covariances.angular_cl_cov_cNG
            sigma2_b_arg = {}
        else:
            raise ValueError("Invalid value for which_ng_cov. Must be 'SSC' or 'cNG'.")

        cov_ng_4D = np.zeros((nbl, nbl, zpairs_AB, zpairs_CD))
        for ij in tqdm(range(zpairs_AB)):
            for kl in range(zpairs_CD):
                cov_ng_4D[:, :, ij, kl] = ccl_ng_cov_func(self.cosmo_ccl,
                                                          tracer1=kernel_A[ind_AB[ij, -2]],
                                                          tracer2=kernel_B[ind_AB[ij, -1]],
                                                          ell=ell,
                                                          t_of_kk_a=tkka,
                                                          fsky=f_sky,
                                                          tracer3=kernel_C[ind_CD[kl, -2]],
                                                          tracer4=kernel_D[ind_CD[kl, -1]],
                                                          ell2=None,
                                                          integration_method=integration_method,
                                                          **sigma2_b_arg)

        print(f'{which_ng_cov} computed with pyccl in {(time.perf_counter() - start_time) / 60:.0f} min')

        return cov_ng_4D

    def compute_ng_cov_3x2pt(self, which_ng_cov, ell, f_sky, integration_method,
                             probe_ordering, ind_dict):

        cov_ng_3x2pt_dict_8D = {}

        kernel_dict = {
            'L': self.wf_lensing_obj,
            'G': self.wf_galaxy_obj
        }

        for row, (probe_a, probe_b) in enumerate(probe_ordering):
            for col, (probe_c, probe_d) in enumerate(probe_ordering):
                if col >= row:

                    print('3x2pt: working on probe combination ', probe_a, probe_b, probe_c, probe_d)
                    cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                        self.compute_ng_cov_ccl(which_ng_cov=which_ng_cov,
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
                                                ))

                    # TODO delete this
                    # save only the upper triangle blocks
                    # if pyccl_cfg['save_cov']:
                    #     cov_path = pyccl_cfg['cov_path']
                    #     cov_filename_fmt = cov_filename.format(probe_a=probe_a, probe_b=probe_b,
                    #                                            probe_c=probe_c, probe_d=probe_d)

                    #     nbl_grid_here = len(ell)
                    #     assert f'nbl{nbl_grid_here}' in cov_filename, \
                    #         f'cov_filename could be inconsistent with the actual grid used'
                    #     np.savez_compressed(
                    #         f'{cov_path}/{cov_filename_fmt}', cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d])

                else:
                    print('3x2pt: skipping probe combination ', probe_a, probe_b, probe_c, probe_d)
                    cov_ng_3x2pt_dict_8D[probe_a, probe_b, probe_c, probe_d] = (
                        cov_ng_3x2pt_dict_8D[probe_c, probe_d, probe_a, probe_b].transpose(1, 0, 3, 2))

        self.cov_ng_3x2pt_dict_8D = cov_ng_3x2pt_dict_8D

        self.check_cov_blocks_simmetry()

        return

    def check_cov_blocks_simmetry(self):
        # Test if cov is symmetric in ell1, ell2 (only for the diagonal covariance blocks:
        # the off-diagonal need *not* to be symmetric in ell1, ell2)
        for key in self.cov_ng_3x2pt_dict_8D.keys():
            if (key == ('L', 'L', 'L', 'L')) or (key == ('G', 'L', 'G', 'L')) or (key == ('G', 'G', 'G', 'G')):
                try:
                    cov_2d = mm.cov_4D_to_2D(self.cov_ng_3x2pt_dict_8D[key])
                    assert np.allclose(cov_2d, cov_2d.T, atol=0, rtol=1e-5)
                    np.testing.assert_allclose(self.cov_ng_3x2pt_dict_8D[key], 
                                            #    np.transpose(self.cov_ng_3x2pt_dict_8D[key], (1, 0, 2, 3)), 
                                               np.transpose(self.cov_ng_3x2pt_dict_8D[key], (1, 0, 3, 2)), 
                                               rtol=1e-5, atol=0,
                                               err_msg=f'cov_ng_4D {key} is not symmetric in ell1, ell2')
                except AssertionError as error:
                    print(error)

        return
