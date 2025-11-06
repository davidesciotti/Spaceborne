import itertools
import time
import warnings
from collections import defaultdict

import numpy as np

from spaceborne import bnt as bnt_utils
from spaceborne import constants as const
from spaceborne import sb_lib as sl
from spaceborne.ccl_interface import CCLInterface
from spaceborne.cov_partial_sky import NmtCov
from spaceborne.ell_utils import EllBinning
from spaceborne.oc_interface import OneCovarianceInterface


class SpaceborneCovariance:
    def __init__(
        self,
        cfg: dict,
        pvt_cfg: dict,
        ell_obj: EllBinning,
        nmt_cov_obj: NmtCov | None,
        bnt_matrix: np.ndarray | None,
    ):
        self.cfg = cfg
        self.cov_cfg = cfg['covariance']
        self.ell_dict = {}
        self.ell_obj = ell_obj
        self.bnt_matrix = bnt_matrix
        self.probe_names_dict = {'LL': 'WL', 'GG': 'GC', '3x2pt': '3x2pt'}
        self.all_terms = ['sva', 'sn', 'mix', 'g', 'ssc', 'cng', 'tot']
        # TODO these should probably be defined on a higher level
        self.llll_ixs = (0, 0, 0, 0)
        self.glgl_ixs = (1, 0, 1, 0)
        self.gggg_ixs = (1, 1, 1, 1)

        self.zbins = pvt_cfg['zbins']
        self.GL_OR_LG = pvt_cfg['GL_OR_LG']
        if self.GL_OR_LG == 'LG':  # on-the-fly check
            raise ValueError('the cross-correlation between G and L must be GL, not LG')

        self.fsky = pvt_cfg['fsky']
        self.symmetrize_output_dict = pvt_cfg['symmetrize_output_dict']
        self.unique_probe_combs = pvt_cfg['unique_probe_combs']
        self.probe_ordering = pvt_cfg['probe_ordering']  # TODO delete this??

        # get_probe_combs is called to get all_req_probe_combs
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_2d']

        self.n_probes = self.cov_cfg['n_probes']
        # 'include' instead of 'compute' because it might be loaded from file
        self.include_ssc = self.cov_cfg['SSC']
        self.include_cng = self.cov_cfg['cNG']
        self.g_code = self.cov_cfg['G_code']
        self.ssc_code = self.cov_cfg['SSC_code']
        self.cng_code = self.cov_cfg['cNG_code']
        self.cov_ordering_2d = self.cov_cfg['covariance_ordering_2D']
        self.use_nmt = self.cfg['namaster']['use_namaster']
        self.do_sample_cov = self.cfg['sample_covariance']['compute_sample_cov']
        # other useful objects
        self.nmt_cov_obj = nmt_cov_obj

        # Instantiate the nested cov dict of structure
        # [TERM][PROBE_AB, PROBE_CD][DIM]: np.ndarray
        self.cov_dict = defaultdict(lambda: defaultdict(dict))

        if self.cov_ordering_2d == 'probe_scale_zpair':
            self.block_index = 'ell'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt_hs
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'zbins': self.zbins,
                'req_probe_combs_2d': self.req_probe_combs_2d,
            }
        elif self.cov_ordering_2d == 'probe_zpair_scale':
            self.block_index = 'zpair'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt_hs
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

    def set_ind_and_zpairs(self, ind):
        # set indices array
        self.ind = ind
        self.zpairs_auto, self.zpairs_cross, self.zpairs_3x2pt = sl.get_zpairs(
            self.zbins
        )
        self.ind_auto = ind[: self.zpairs_auto, :].copy()
        self.ind_cross = ind[
            self.zpairs_auto : self.zpairs_cross + self.zpairs_auto, :
        ].copy()

        self.ind_dict = {
            ('L', 'L'): self.ind_auto,
            ('G', 'L'): self.ind_cross,
            ('G', 'G'): self.ind_auto,
        }

    def consistency_checks(self):
        # sanity checks

        assert not (self.use_nmt and self.do_sample_cov), (
            'either cfg["namaster"]["use_namaster"] or '
            'cfg["sample_covariance"]["compute_sample_cov"] should be True, '
            'not both (but they can both be False)'
        )

        assert tuple(self.probe_ordering[0]) == ('L', 'L'), (
            'the XC probe should be in position 1 (not 0) of the datavector'
        )
        assert tuple(self.probe_ordering[2]) == ('G', 'G'), (
            'the XC probe should be in position 1 (not 0) of the datavector'
        )

        if (
            self.ell_obj.ells_WL.max() < 15
        ):  # very rudimental check of whether they're in lin or log scale
            raise ValueError(
                'looks like the ell values are in log scale. '
                'You should use linear scale instead.'
            )

        # if C_XC is C_LG, switch the ind ordering for the correct rows
        if self.GL_OR_LG == 'LG':
            print('\nAttention! switching columns in the ind array (for the XC part)')
            self.ind[
                self.zpairs_auto : (self.zpairs_auto + self.zpairs_cross), [2, 3]
            ] = self.ind[
                self.zpairs_auto : (self.zpairs_auto + self.zpairs_cross), [3, 2]
            ]

        # sanity check: the last 2 columns of ind_auto should be equal to the
        # last two of ind_auto
        assert np.array_equiv(
            self.ind[: self.zpairs_auto, 2:], self.ind[-self.zpairs_auto :, 2:]
        )

        assert self.ssc_code in ['Spaceborne', 'PyCCL', 'OneCovariance'], (
            "covariance_cfg['SSC_code'] not recognised"
        )
        assert self.cng_code in ['PyCCL', 'OneCovariance'], (
            "covariance_cfg['cNG_code'] not recognised"
        )

    def reshape_cov(
        self,
        cov_in,
        ndim_in,
        ndim_out,
        nbl,
        zpairs=None,
        ind_probe=None,
        is_3x2pt=False,
    ):
        """Reshape a covariance matrix between dimensions (6/2D -> 4/2D).

        Parameters
        ----------
        cov_in : np.ndarray
            Input covariance matrix.
        ndim_in : int
            Input dimension of the covariance matrix (e.g., 6 or 10).
        ndim_out : int
            Desired output dimension of the covariance matrix (e.g., 4 or 2).
        nbl : int
            Number of multipole bins.
        zpairs : int, optional
            Number of redshift pairs. Required for 6D -> 4D reshaping.
        ind_probe : np.ndarray, optional
            Probe index array for 6D -> 4D reshaping.
        is_3x2pt : bool, optional
            If True, indicates that the covariance is a 3x2pt covariance.

        Returns
        -------
        np.ndarray
            Reshaped covariance matrix.

        Raises
        ------
        ValueError
            If the combination of ndim_in, ndim_out, and is_3x2pt is not supported.
        """

        # Validate inputs
        if ndim_in not in [6, 10, 4]:
            raise ValueError(
                f'Unsupported ndim_in={ndim_in}. Only 6D or 10D supported.'
            )
        if ndim_out not in [2, 4]:
            raise ValueError(
                f'Unsupported ndim_out={ndim_out}. Only 2D or 4D supported.'
            )

        # Reshape logic
        if ndim_in == 6:
            assert cov_in.ndim == 6, 'Input covariance must be 6D for this operation.'
            assert not is_3x2pt, 'input 3x2pt cov should be 10d.'
            cov_out = sl.cov_6D_to_4D(cov_in, nbl, zpairs, ind_probe)

        elif ndim_in == 10:
            assert cov_in.ndim == 10, 'Input covariance must be 10D for this operation.'
            assert is_3x2pt, 'input 3x2pt cov should be 10d.'
            cov_out = sl.cov_3x2pt_10D_to_4D(
                cov_3x2pt_10D=cov_in,
                probe_ordering=self.probe_ordering,
                nbl=nbl,
                zbins=self.zbins,
                ind_copy=self.ind.copy(),
                GL_OR_LG=self.GL_OR_LG,
                req_probe_combs_2d=self.req_probe_combs_2d,
            )

        elif ndim_in == 4:
            cov_out = cov_in.copy()

        if ndim_out == 2:
            if is_3x2pt:
                # the 3x2pt has an additional layer of complexity for the ordering,
                # as it includes multiple probes
                cov_out = self.cov_4D_to_2D_3x2pt_func(
                    cov_out, **self.cov_4D_to_2D_3x2pt_func_kw
                )
            else:
                cov_out = sl.cov_4D_to_2D(cov_out, block_index=self.block_index)

        return cov_out

    def set_gauss_cov(self, ccl_obj, split_gaussian_cov):
        start = time.perf_counter()

        # signal
        cl_3x2pt_5d = ccl_obj.cl_3x2pt_5d

        # ! noise
        sigma_eps2 = (np.array(self.cov_cfg['sigma_eps_i']) * np.sqrt(2)) ** 2
        ng_shear = np.array(self.cfg['nz']['ngal_sources'])
        ng_clust = np.array(self.cfg['nz']['ngal_lenses'])
        noise_3x2pt_4d = sl.build_noise(
            self.zbins,
            self.n_probes,
            sigma_eps2=sigma_eps2,
            ng_shear=ng_shear,
            ng_clust=ng_clust,
            is_noiseless=self.cov_cfg['no_sampling_noise'],
        )

        # create dummy ell axis, the array is just repeated along it
        noise_3x2pt_5d = np.repeat(
            noise_3x2pt_4d[:, :, np.newaxis, :, :], self.ell_obj.nbl_3x2pt, axis=2
        )

        # bnt-transform the noise spectra if needed
        if self.cfg['BNT']['cl_BNT_transform']:
            print('BNT-transforming the noise spectra...')
            noise_3x2pt_5d = bnt_utils.cl_bnt_transform_3x2pt(
                noise_3x2pt_5d, self.bnt_matrix
            )

        # ! compute 3x2pt fsky Gaussian covariance: by default, split SVA, SN and MIX
        (cov_3x2pt_sva_10d, cov_3x2pt_sn_10d, cov_3x2pt_mix_10d) = sl.covariance_einsum(
            cl_5d=cl_3x2pt_5d,
            noise_5d=noise_3x2pt_5d,
            fsky=self.fsky,
            ell_values=self.ell_obj.ells_3x2pt,
            delta_ell=self.ell_obj.delta_l_3x2pt,
            split_terms=True,
            return_only_diagonal_ells=False,
        )

        # the Gaussian HS cov is computed for all probes at once, still
        for probe_abcd in self.req_probe_combs_2d:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            probe_2tpl = (probe_ab, probe_cd)
            probe_ixs = tuple(const.HS_PROBE_NAME_TO_IX_DICT[p] for p in probe_abcd)
            self.cov_dict['sva'][probe_2tpl]['6d'] = cov_3x2pt_sva_10d[*probe_ixs]
            self.cov_dict['sn'][probe_2tpl]['6d'] = cov_3x2pt_sn_10d[*probe_ixs]
            self.cov_dict['mix'][probe_2tpl]['6d'] = cov_3x2pt_mix_10d[*probe_ixs]
            # sum to get G
            self.cov_dict['g'][probe_2tpl]['6d'] = (
                self.cov_dict['sva'][probe_2tpl]['6d']
                + self.cov_dict['sn'][probe_2tpl]['6d']
                + self.cov_dict['mix'][probe_2tpl]['6d']
            )

        # ! Partial sky with nmt
        # ! this case overwrites self.cov_3x2pt_g_10d only, but the cfg checker will
        # ! raise an error if you require to split the G cov and use_nmt or
        # ! do_sample_cov are True
        if self.use_nmt or self.do_sample_cov:
            # noise vector doesn't have to be recomputed, but repeated a larger number
            # of times (ell by ell)
            noise_3x2pt_unb_5d = np.repeat(
                noise_3x2pt_4d[:, :, np.newaxis, :, :],
                repeats=self.nmt_cov_obj.nbl_3x2pt_unb,
                axis=2,
            )
            self.nmt_cov_obj.noise_3x2pt_unb_5d = noise_3x2pt_unb_5d
            cov_3x2pt_gnmt_10d = self.nmt_cov_obj.build_psky_cov()

            self.cov_dict['g'].update(
                sl.cov_10d_arr_to_dict(
                    cov_3x2pt_gnmt_10d,
                    dim='6d',
                    req_probe_combs_2d=self.req_probe_combs_2d,
                    space='harmonic',
                )
            )

            # delete the SVA, SN and MIX terms to avoid confusion, only the g one
            # remains in the partial sky case
            for term in ['sva', 'sn', 'mix']:
                self.cov_dict[term].update(
                    sl.cov_10d_arr_to_dict(
                        None,
                        dim='6d',
                        req_probe_combs_2d=self.req_probe_combs_2d,
                        space='harmonic',
                        empty=True,
                    )
                )

        print(f'Gauss. cov. matrices computed in {(time.perf_counter() - start):.2f} s')

        # reshape to 2D
        self._cov_6d_and_3x2pt_to_4d_and_2d()

    def _cov_6d_and_3x2pt_to_4d_and_2d(self):
        """Reshapes the 3x2pt 10d cov into 2D.

        Parameters
        ----------
        split_gaussian_cov : bool
            Whether to split (hence to reshape) the SVA/SN/MIX parts of the G cov
        """

        # populate the dict with 4d and 2d probe-specific arrays (from the 6d ones)
        self.cov_dict = sl.add_4d_and_2d_to_cov_dict_6d(
            cov_dict=self.cov_dict,
            space='harmonic',
            nbx=self.ell_obj.nbl_3x2pt,
            ind_auto=self.ind_auto,
            ind_cross=self.ind_cross,
            zpairs_auto=self.zpairs_auto,
            zpairs_cross=self.zpairs_cross,
            block_index=self.block_index,
        )
        

        # now create 3x2pt 4d and 2d (there is no 3x2pt 6d or 10d!!!)
        for term in self.all_terms:
            if not self.cov_dict[term]:
                continue
            self.cov_dict[term]['3x2pt']['4d'] = sl.cov_dict_4d_to_3x2pt_4d_arr(
                self.cov_dict[term], self.req_probe_combs_2d, space='harmonic'
            )
            self.cov_dict[term]['3x2pt']['2d'] = self.cov_4D_to_2D_3x2pt_func(
                self.cov_dict[term]['3x2pt']['4d'], **self.cov_4D_to_2D_3x2pt_func_kw
            )

    def _cov_8d_dict_to_10d_arr(self, cov_dict_8d):
        """Helper function to process a single covariance component"""
        cov_dict_10d = sl.cov_3x2pt_dict_8d_to_10d(
            cov_3x2pt_dict_8D=cov_dict_8d,
            nbl=self.ell_obj.nbl_3x2pt,
            zbins=self.zbins,
            ind_dict=self.ind_dict,
            unique_probe_combs=self.unique_probe_combs,
            space='harmonic',
            symmetrize_output_dict=self.symmetrize_output_dict,
        )

        return sl.cov_10d_dict_to_array(
            cov_dict_10d, self.ell_obj.nbl_3x2pt, self.zbins, self.n_probes
        )

    def _add_ssc(self, ccl_obj: CCLInterface, oc_obj: OneCovarianceInterface):
        """Helper function to get the SSC from the required code and uniform its
        shape"""
        if not self.include_ssc:
            print('Skipping SSC computation')
            return

        if self.ssc_code == 'Spaceborne':
            cov_3x2pt_ssc_10d = self._cov_8d_dict_to_10d_arr(
                self.cov_ssc_sb_3x2pt_dict_8D
            )
        elif self.ssc_code == 'PyCCL':
            cov_3x2pt_ssc_10d = self._cov_8d_dict_to_10d_arr(
                ccl_obj.cov_ssc_ccl_3x2pt_dict_8D
            )
        elif self.ssc_code == 'OneCovariance':
            cov_3x2pt_ssc_10d = oc_obj.cov_3x2pt_ssc_10d

        assert not np.allclose(cov_3x2pt_ssc_10d, 0, atol=0, rtol=1e-10), (
            f'{self.ssc_code} SSC covariance matrix is identically zero'
        )

        # assign to dictionary
        self.cov_dict['ssc'].update(
            sl.cov_10d_arr_to_dict(
                cov_3x2pt_ssc_10d,
                dim='6d',
                req_probe_combs_2d=self.req_probe_combs_2d,
                space='harmonic',
            )
        )

    def _add_cng(self, ccl_obj: CCLInterface, oc_obj: OneCovarianceInterface):
        """Helper function to get the cNG from the required code and uniform its
        shape"""
        if not self.include_cng:
            print('Skipping cNG computation')
            return

        if self.cng_code == 'PyCCL':
            cov_3x2pt_cng_10d = self._cov_8d_dict_to_10d_arr(
                ccl_obj.cov_cng_ccl_3x2pt_dict_8D
            )
        elif self.cng_code == 'OneCovariance':
            cov_3x2pt_cng_10d = oc_obj.cov_3x2pt_cng_10d

        assert not np.allclose(cov_3x2pt_cng_10d, 0, atol=0, rtol=1e-10), (
            f'{self.cng_code} cNG covariance matrix is identically zero'
        )

        # assign to dictionary
        self.cov_dict['cng'].update(
            sl.cov_10d_arr_to_dict(
                cov_3x2pt_cng_10d,
                dim='6d',
                req_probe_combs_2d=self.req_probe_combs_2d,
                space='harmonic',
            )
        )

    def _slice_3x2pt_cov(self, split_gaussian_cov: bool) -> None:
        """Helper function to slice the 3x2pt covariance into WL, GC and XC.
        Note that I am not touching the ell bins here, not even just to exclude
        a subset of them
        (e.g. cov_WL_g_6d = cov_3x2pt_g_10d[llll_ixs, :nbl_WL, :nbl_WL, ...])"""

        self.cov_WL_g_6d = self.cov_3x2pt_g_10d[*self.llll_ixs].copy()
        self.cov_WL_ssc_6d = self.cov_3x2pt_ssc_10d[*self.llll_ixs].copy()
        self.cov_WL_cng_6d = self.cov_3x2pt_cng_10d[*self.llll_ixs].copy()

        self.cov_GC_g_6d = self.cov_3x2pt_g_10d[*self.gggg_ixs].copy()
        self.cov_GC_ssc_6d = self.cov_3x2pt_ssc_10d[*self.gggg_ixs].copy()
        self.cov_GC_cng_6d = self.cov_3x2pt_cng_10d[*self.gggg_ixs].copy()

        self.cov_XC_g_6d = self.cov_3x2pt_g_10d[*self.glgl_ixs].copy()
        self.cov_XC_ssc_6d = self.cov_3x2pt_ssc_10d[*self.glgl_ixs].copy()
        self.cov_XC_cng_6d = self.cov_3x2pt_cng_10d[*self.glgl_ixs].copy()

        if split_gaussian_cov:
            self.cov_WL_sva_6d = self.cov_3x2pt_sva_10d[*self.llll_ixs].copy()
            self.cov_WL_sn_6d = self.cov_3x2pt_sn_10d[*self.llll_ixs].copy()
            self.cov_WL_mix_6d = self.cov_3x2pt_mix_10d[*self.llll_ixs].copy()

            self.cov_GC_sva_6d = self.cov_3x2pt_sva_10d[*self.gggg_ixs].copy()
            self.cov_GC_sn_6d = self.cov_3x2pt_sn_10d[*self.gggg_ixs].copy()
            self.cov_GC_mix_6d = self.cov_3x2pt_mix_10d[*self.gggg_ixs].copy()

            self.cov_XC_sva_6d = self.cov_3x2pt_sva_10d[*self.glgl_ixs].copy()
            self.cov_XC_sn_6d = self.cov_3x2pt_sn_10d[*self.glgl_ixs].copy()
            self.cov_XC_mix_6d = self.cov_3x2pt_mix_10d[*self.glgl_ixs].copy()

    def _all_covs_10d_or_6d_to_2d(self, split_gaussian_cov):
        """reshapes all covs (g, sva, sn, mix, ssc, cng) for all probes to 2D"""
        reshape_args = [ 
            # WL
            ('cov_WL_g_2d', self.cov_WL_g_6d, 6, self.ell_obj.nbl_WL, self.zpairs_auto, self.ind_auto, False),
            ('cov_WL_ssc_2d', self.cov_WL_ssc_6d, 6, self.ell_obj.nbl_WL, self.zpairs_auto, self.ind_auto, False),
            ('cov_WL_cng_2d', self.cov_WL_cng_6d, 6, self.ell_obj.nbl_WL, self.zpairs_auto, self.ind_auto, False),
            
            # GC
            ('cov_GC_g_2d', self.cov_GC_g_6d, 6, self.ell_obj.nbl_GC, self.zpairs_auto, self.ind_auto, False),
            ('cov_GC_ssc_2d', self.cov_GC_ssc_6d, 6, self.ell_obj.nbl_GC, self.zpairs_auto, self.ind_auto, False),
            ('cov_GC_cng_2d', self.cov_GC_cng_6d, 6, self.ell_obj.nbl_GC, self.zpairs_auto, self.ind_auto, False),
            
            # XC
            ('cov_XC_g_2d', self.cov_XC_g_6d, 6, self.ell_obj.nbl_XC, self.zpairs_cross, self.ind_cross, False),
            ('cov_XC_ssc_2d', self.cov_XC_ssc_6d, 6, self.ell_obj.nbl_XC, self.zpairs_cross, self.ind_cross, False),
            ('cov_XC_cng_2d', self.cov_XC_cng_6d, 6, self.ell_obj.nbl_XC, self.zpairs_cross, self.ind_cross, False),
            
            # 3x2pt
            ('cov_3x2pt_g_2d', self.cov_3x2pt_g_10d, 10, self.ell_obj.nbl_3x2pt, self.zpairs_auto, self.ind, True),
            ('cov_3x2pt_ssc_2d', self.cov_3x2pt_ssc_10d,10, self.ell_obj.nbl_3x2pt, self.zpairs_auto, self.ind, True),
            ('cov_3x2pt_cng_2d', self.cov_3x2pt_cng_10d,10, self.ell_obj.nbl_3x2pt, self.zpairs_auto, self.ind, True)
        ]  # fmt: skip

        if split_gaussian_cov:
            reshape_args.extend([
                ('cov_WL_sva_2d', self.cov_WL_sva_6d, 6, self.ell_obj.nbl_WL, self.zpairs_auto, self.ind_auto, False),
                ('cov_WL_sn_2d', self.cov_WL_sn_6d, 6, self.ell_obj.nbl_WL, self.zpairs_auto, self.ind_auto, False),
                ('cov_WL_mix_2d', self.cov_WL_mix_6d, 6, self.ell_obj.nbl_WL, self.zpairs_auto, self.ind_auto, False),
                
                ('cov_GC_sva_2d', self.cov_GC_sva_6d, 6, self.ell_obj.nbl_GC, self.zpairs_auto, self.ind_auto, False),
                ('cov_GC_sn_2d', self.cov_GC_sn_6d, 6, self.ell_obj.nbl_GC, self.zpairs_auto, self.ind_auto, False),
                ('cov_GC_mix_2d', self.cov_GC_mix_6d, 6, self.ell_obj.nbl_GC, self.zpairs_auto, self.ind_auto, False),
                
                ('cov_XC_sva_2d', self.cov_XC_sva_6d, 6, self.ell_obj.nbl_XC, self.zpairs_cross, self.ind_cross, False),
                ('cov_XC_sn_2d', self.cov_XC_sn_6d, 6, self.ell_obj.nbl_XC, self.zpairs_cross, self.ind_cross, False),
                ('cov_XC_mix_2d', self.cov_XC_mix_6d, 6, self.ell_obj.nbl_XC, self.zpairs_cross, self.ind_cross, False),
            ]
            )  # fmt: skip

        # Loop over and set attributes
        for name, cov, ndim_in, _nbl, _zpairs, ind_probe, is_3x2pt in reshape_args:
            setattr(
                self,
                name,
                self.reshape_cov(
                    cov_in=cov,
                    ndim_in=ndim_in,
                    ndim_out=2,
                    nbl=_nbl,
                    zpairs=_zpairs,
                    ind_probe=ind_probe,
                    is_3x2pt=is_3x2pt,
                ),
            )

    def _cov_2d_ell_cuts(self, split_gaussian_cov):
        # TODO reimplement this (I think it still works, but needs to be tested)
        # TODO use split_gaussian_cov
        if not self.cfg['ell_cuts']['cov_ell_cuts']:
            return
        else:
            raise NotImplementedError('Ell cuts not implemented for the moment')

            print('Performing ell cuts on the 2d covariance matrix...')
            self.cov_WL_g_2d = sl.remove_rows_cols_array2D(
                self.cov_WL_g_2d, self.ell_dict['idxs_to_delete_dict']['LL']
            )
            self.cov_GC_g_2d = sl.remove_rows_cols_array2D(
                self.cov_GC_g_2d, self.ell_dict['idxs_to_delete_dict']['GG']
            )
            self.cov_XC_g_2d = sl.remove_rows_cols_array2D(
                self.cov_XC_g_2d, self.ell_dict['idxs_to_delete_dict'][self.GL_OR_LG]
            )
            self.cov_3x2pt_g_2d = sl.remove_rows_cols_array2D(
                self.cov_3x2pt_g_2d, self.ell_dict['idxs_to_delete_dict']['3x2pt']
            )

            self.cov_WL_ssc_2d = sl.remove_rows_cols_array2D(
                self.cov_WL_ssc_2d, self.ell_dict['idxs_to_delete_dict']['LL']
            )
            self.cov_GC_ssc_2d = sl.remove_rows_cols_array2D(
                self.cov_GC_ssc_2d, self.ell_dict['idxs_to_delete_dict']['GG']
            )
            self.cov_XC_ssc_2d = sl.remove_rows_cols_array2D(
                self.cov_XC_ssc_2d, self.ell_dict['idxs_to_delete_dict'][self.GL_OR_LG]
            )
            self.cov_3x2pt_ssc_2d = sl.remove_rows_cols_array2D(
                self.cov_3x2pt_ssc_2d, self.ell_dict['idxs_to_delete_dict']['3x2pt']
            )

            self.cov_WL_cng_2d = sl.remove_rows_cols_array2D(
                self.cov_WL_cng_2d, self.ell_dict['idxs_to_delete_dict']['LL']
            )
            self.cov_GC_cng_2d = sl.remove_rows_cols_array2D(
                self.cov_GC_cng_2d, self.ell_dict['idxs_to_delete_dict']['GG']
            )
            self.cov_XC_cng_2d = sl.remove_rows_cols_array2D(
                self.cov_XC_cng_2d, self.ell_dict['idxs_to_delete_dict'][self.GL_OR_LG]
            )
            self.cov_3x2pt_cng_2d = sl.remove_rows_cols_array2D(
                self.cov_3x2pt_cng_2d, self.ell_dict['idxs_to_delete_dict']['3x2pt']
            )

    def build_covs(
        self,
        ccl_obj: CCLInterface,
        oc_obj: OneCovarianceInterface,
        split_gaussian_cov: bool,
    ):
        """
        Combines, reshaped and returns the Gaussian (g), non-Gaussian (ng) and
        Gaussian+non-Gaussian (tot) covariance matrices
        for different probe combinations.

        Parameters
        ----------
        ccl_obj : object
            PyCCL interface object containing PyCCL covariance terms, as well as cls
        oc_obj : object
            OneCovariance interface object containing OneCovariance covariance terms
        split_gaussian_cov: bool
            Whether to split (hence to reshape) the SVA/SN/MIX parts of the G cov

        Returns
        -------
        dict
            Dictionary containing the computed covariance matrices with keys:
            - cov_{probe}_g_2d: Gaussian-only covariance
            - cov_{probe}_ng_2d: ng-only covariance (SSC, cNG or the sum of the two)
            - cov_{probe}_tot_2d: g + ng covariance
            where {probe} can be: WL (weak lensing), GC (galaxy clustering),
            3x2pt (WL + XC + GC), XC (cross-correlation)
        """

        if self.g_code == 'OneCovariance':
            self.cov_3x2pt_g_10d = oc_obj.cov_3x2pt_g_10d
            if split_gaussian_cov:
                self.cov_3x2pt_sva_10d = oc_obj.cov_3x2pt_sva_10d
                self.cov_3x2pt_sn_10d = oc_obj.cov_3x2pt_sn_10d
                self.cov_3x2pt_mix_10d = oc_obj.cov_3x2pt_mix_10d

        # ! reshape and set SSC and cNG - the "if include SSC/cNG"
        # ! are inside the function
        self._add_ssc(ccl_obj, oc_obj)
        self._add_cng(ccl_obj, oc_obj)

        # ! BNT transform (6/10D covs needed for this implementation)
        if self.cfg['BNT']['cov_BNT_transform']:
            print('BNT-transforming the covariance matrix...')
            start = time.perf_counter()
            self._bnt_transform_3x2pt_wrapper()
            print(f'...done in {time.perf_counter() - start:.2f} s')

        # ! compute coupled NG cov - the "if coupled" is inside the function
        self._couple_cov_ng_3x2pt()

        # ! slice the 3x2pt cov to get the probe-specific ones
        # TODO implemet probe-specific binning instead!
        self._slice_3x2pt_cov(split_gaussian_cov)

        # ! reshape everything to 2D
        self._all_covs_10d_or_6d_to_2d(split_gaussian_cov)

        # ! perform ell cuts on the 2D covs
        self._cov_2d_ell_cuts(split_gaussian_cov)

        # ! sum different terms to get total cov
        self.cov_WL_tot_2d = self.cov_WL_g_2d + self.cov_WL_ssc_2d + self.cov_WL_cng_2d
        self.cov_GC_tot_2d = self.cov_GC_g_2d + self.cov_GC_ssc_2d + self.cov_GC_cng_2d
        self.cov_XC_tot_2d = self.cov_XC_g_2d + self.cov_XC_ssc_2d + self.cov_XC_cng_2d
        self.cov_3x2pt_tot_2d = (
            self.cov_3x2pt_g_2d + self.cov_3x2pt_ssc_2d + self.cov_3x2pt_cng_2d
        )
        self.cov_3x2pt_tot_10d = (
            self.cov_3x2pt_g_10d + self.cov_3x2pt_ssc_10d + self.cov_3x2pt_cng_10d
        )

        print('Covariance matrices computed')

    def _bnt_transform_3x2pt_wrapper(self):
        # turn 3x2pt 10d array to dict for the BNT function
        cov_3x2pt_g_10d_dict = sl.cov_10d_array_to_dict(
            self.cov_3x2pt_g_10d, self.probe_ordering
        )
        cov_3x2pt_ssc_10d_dict = sl.cov_10d_array_to_dict(
            self.cov_3x2pt_ssc_10d, self.probe_ordering
        )
        cov_3x2pt_cng_10d_dict = sl.cov_10d_array_to_dict(
            self.cov_3x2pt_cng_10d, self.probe_ordering
        )

        # BNT-transform WL and 3x2pt g, ng and tot covariances
        x_dict = bnt_utils.build_x_matrix_bnt(self.bnt_matrix)
        # TODO BNT and scale cuts of G term should go in the gauss cov function!
        cov_3x2pt_g_10d_dict = bnt_utils.cov_3x2pt_bnt_transform(
            cov_3x2pt_g_10d_dict, x_dict
        )
        cov_3x2pt_ssc_10d_dict = bnt_utils.cov_3x2pt_bnt_transform(
            cov_3x2pt_ssc_10d_dict, x_dict
        )
        cov_3x2pt_cng_10d_dict = bnt_utils.cov_3x2pt_bnt_transform(
            cov_3x2pt_cng_10d_dict, x_dict
        )

        # revert to 10D arrays - this is not strictly necessary since
        # cov_3x2pt_10d_to_4D accepts both a dictionary and
        # an array as input, but it's done to keep the variable names consistent
        # ! BNT IS LINEAR, SO BNT(COV_TOT) = \SUM_i BNT(COV_i), but should check
        self.cov_3x2pt_g_10d = sl.cov_10d_dict_to_array(
            cov_3x2pt_g_10d_dict, self.ell_obj.nbl_3x2pt, self.zbins, n_probes=2
        )
        self.cov_3x2pt_ssc_10d = sl.cov_10d_dict_to_array(
            cov_3x2pt_ssc_10d_dict, self.ell_obj.nbl_3x2pt, self.zbins, n_probes=2
        )
        self.cov_3x2pt_cng_10d = sl.cov_10d_dict_to_array(
            cov_3x2pt_cng_10d_dict, self.ell_obj.nbl_3x2pt, self.zbins, n_probes=2
        )

    def _couple_cov_ng_3x2pt(self):
        if not self.cov_cfg['coupled_cov']:
            return

        if self.cfg['BNT']['cov_BNT_transform']:
            warnings.warn(
                'BNT transformation has not been tested for coupled covariance '
                'matrices.',
                stacklevel=2,
            )

        print('Coupling the non-Gaussian covariance...')
        from spaceborne import cov_partial_sky

        # construct mcm array for better probe handling (especially for 3x2pt)
        mcm_3x2pt_arr = np.zeros(
            (
                self.n_probes,
                self.n_probes,
                self.ell_obj.nbl_3x2pt,
                self.ell_obj.nbl_3x2pt,
            )
        )
        mcm_3x2pt_arr[0, 0] = self.nmt_cov_obj.mcm_ee_binned
        mcm_3x2pt_arr[1, 0] = self.nmt_cov_obj.mcm_te_binned
        mcm_3x2pt_arr[0, 1] = self.nmt_cov_obj.mcm_et_binned
        mcm_3x2pt_arr[1, 1] = self.nmt_cov_obj.mcm_tt_binned

        # cov_WL_ssc_6d = cov_partial_sky.couple_cov_6d(
        #     mcm_3x2pt_arr[0, 0], cov_WL_ssc_6d, mcm_3x2pt_arr[0, 0].T
        # )
        # cov_WL_cng_6d = cov_partial_sky.couple_cov_6d(
        #     mcm_3x2pt_arr[0, 0], cov_WL_cng_6d, mcm_3x2pt_arr[0, 0].T
        # )
        # cov_GC_ssc_6d = cov_partial_sky.couple_cov_6d(
        #     mcm_3x2pt_arr[1, 1], cov_GC_ssc_6d, mcm_3x2pt_arr[1, 1].T
        # )
        # cov_GC_cng_6d = cov_partial_sky.couple_cov_6d(
        #     mcm_3x2pt_arr[1, 1], cov_GC_cng_6d, mcm_3x2pt_arr[1, 1].T
        # )
        # cov_XC_ssc_6d = cov_partial_sky.couple_cov_6d(
        #     mcm_3x2pt_arr[1, 0], cov_XC_ssc_6d, mcm_3x2pt_arr[1, 0].T
        # )
        # cov_XC_cng_6d = cov_partial_sky.couple_cov_6d(
        #     mcm_3x2pt_arr[1, 0], cov_XC_cng_6d, mcm_3x2pt_arr[1, 0].T
        # )

        for a, b, c, d in itertools.product(range(2), repeat=4):
            self.cov_3x2pt_ssc_10d[a, b, c, d] = cov_partial_sky.couple_cov_6d(
                mcm_3x2pt_arr[a, b],
                self.cov_3x2pt_ssc_10d[a, b, c, d],
                mcm_3x2pt_arr[c, d].T,
            )
            self.cov_3x2pt_cng_10d[a, b, c, d] = cov_partial_sky.couple_cov_6d(
                mcm_3x2pt_arr[a, b],
                self.cov_3x2pt_cng_10d[a, b, c, d],
                mcm_3x2pt_arr[c, d].T,
            )
        print('...done')

    def get_ellmax_nbl(self, probe, covariance_cfg):
        if probe == 'LL':
            ell_max = covariance_cfg['ell_max_WL']
            nbl = covariance_cfg['nbl_WL']
        elif probe == 'GG':
            ell_max = covariance_cfg['ell_max_GC']
            nbl = covariance_cfg['nbl_GC']
        elif probe == '3x2pt':
            ell_max = covariance_cfg['ell_max_3x2pt']
            nbl = covariance_cfg['nbl_3x2pt']
        else:
            raise ValueError('probe must be LL or GG or 3x2pt')
        return ell_max, nbl
