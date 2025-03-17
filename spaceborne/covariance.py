import os
import pickle
import time
from copy import deepcopy

import numpy as np
from scipy.integrate import simpson as simps
from scipy.interpolate import RectBivariateSpline

from spaceborne import bnt as bnt_utils
from spaceborne import sb_lib as sl


class SpaceborneCovariance:
    def __init__(self, cfg, pvt_cfg, ell_dict, bnt_matrix):
        self.cfg = cfg
        self.cov_cfg = cfg['covariance']
        self.ell_dict = ell_dict
        self.bnt_matrix = bnt_matrix
        self.probe_names_dict = {
            'LL': 'WL',
            'GG': 'GC',
            '3x2pt': '3x2pt',
        }

        self.zbins = pvt_cfg['zbins']
        self.cov_terms_list = pvt_cfg['cov_terms_list']
        self.GL_OR_LG = pvt_cfg['GL_OR_LG']

        self.n_probes = self.cov_cfg['n_probes']
        # 'include' instead of 'compute' because it might be loaded from file
        self.include_ssc = self.cov_cfg['SSC']
        self.include_cng = self.cov_cfg['cNG']
        self.g_code = self.cov_cfg['G_code']
        self.ssc_code = self.cov_cfg['SSC_code']
        self.cng_code = self.cov_cfg['cNG_code']
        self.fsky = self.cfg['mask']['fsky']
        # must copy the array! Otherwise, it gets modified and changed at each call
        self.cov_ordering_2d = self.cov_cfg['covariance_ordering_2D']
        self.probe_ordering = self.cov_cfg['probe_ordering']

        if self.cov_ordering_2d == 'probe_ell_zpair':
            self.block_index = 'ell'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'zbins': self.zbins,
            }
        elif self.cov_ordering_2d == 'probe_zpair_ell':
            self.block_index = 'zpair'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2DCLOE_3x2pt
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'zbins': self.zbins,
            }
        elif self.cov_ordering_2d == 'ell_probe_zpair':
            self.block_index = 'ell'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2D
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'optimize': True,
            }
        elif self.cov_ordering_2d == 'zpair_probe_ell':
            self.block_index = 'zpair'
            self.cov_4D_to_2D_3x2pt_func = sl.cov_4D_to_2D
            self.cov_4D_to_2D_3x2pt_func_kw = {
                'block_index': self.block_index,
                'optimize': True,
            }
        else:
            raise ValueError(f'Unknown 2D cov ordering: {self.cov_ordering_2d}')

        # set ell values
        self.ell_WL, self.nbl_WL = ell_dict['ell_WL'], ell_dict['ell_WL'].shape[0]
        self.ell_GC, self.nbl_GC = ell_dict['ell_GC'], ell_dict['ell_GC'].shape[0]
        self.ell_3x2pt, self.nbl_3x2pt = self.ell_GC, self.nbl_GC

        self.cov_dict = {}

    def set_ind_and_zpairs(self, ind, zbins):
        # set indices array
        self.ind = ind
        self.zbins = zbins
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
        # TODO? (this like below)
        # self.ind_dict = build_ind_dict(triu_tril, row_col_major, size, GL_OR_LG)

    def consistency_checks(self):
        # sanity checks

        assert tuple(self.probe_ordering[0]) == ('L', 'L'), (
            'the XC probe should be in position 1 (not 0) of the datavector'
        )
        assert tuple(self.probe_ordering[2]) == ('G', 'G'), (
            'the XC probe should be in position 1 (not 0) of the datavector'
        )

        if (
            self.ell_WL.max() < 15
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

        assert (
            (self.cov_terms_list == ['G', 'SSC', 'cNG'])
            or (
                self.cov_terms_list
                == [
                    'G',
                    'SSC',
                ]
            )
            or (self.cov_terms_list == ['G', 'cNG'])
            or (
                self.cov_terms_list
                == [
                    'G',
                ]
            )  # TODO finish testing this?
        ), 'cov_terms_list not recognised'

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
        """
        Reshape a covariance matrix between dimensions (6/2D -> 4/2D).

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

        # raise NotImplementedError('Is this function really useful?')

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
                cov_in,
                self.probe_ordering,
                nbl,
                self.zbins,
                self.ind.copy(),
                self.GL_OR_LG,
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

        cl_LL_3D = ccl_obj.cl_ll_3d
        cl_GG_3D = ccl_obj.cl_gg_3d
        cl_3x2pt_5D = ccl_obj.cl_3x2pt_5d

        delta_l_WL = self.ell_dict['delta_l_WL']
        delta_l_GC = self.ell_dict['delta_l_GC']
        delta_l_3x2pt = delta_l_GC

        # build noise vector
        sigma_eps2 = (self.cov_cfg['sigma_eps_i'] * np.sqrt(2)) ** 2
        ng_shear = np.array(self.cfg['nz']['ngal_sources'])
        ng_clust = np.array(self.cfg['nz']['ngal_lenses'])
        noise_3x2pt_4D = sl.build_noise(
            self.zbins,
            self.n_probes,
            sigma_eps2=sigma_eps2,
            ng_shear=ng_shear,
            ng_clust=ng_clust,
            is_noiseless=self.cov_cfg['noiseless_spectra'],
        )

        # create dummy ell axis, the array is just repeated along it
        nbl_max = max(self.nbl_WL, self.nbl_GC, self.nbl_3x2pt)
        noise_5D = np.repeat(noise_3x2pt_4D[:, :, np.newaxis, :, :], nbl_max, axis=2)

        # the ell axis is a dummy one for the noise, is just needs to be of the
        # same length as the corresponding cl one
        noise_LL_5D = noise_5D[0, 0, : self.nbl_WL, :, :][np.newaxis, np.newaxis, ...]
        noise_GG_5D = noise_5D[1, 1, : self.nbl_GC, :, :][np.newaxis, np.newaxis, ...]
        noise_3x2pt_5D = noise_5D[:, :, : self.nbl_3x2pt, :, :]

        # bnt-transform the noise spectra if needed
        if self.cfg['BNT']['cl_BNT_transform']:
            print('BNT-transforming the noise spectra...')
            noise_LL_5D = bnt_utils.cl_bnt_transform(
                noise_LL_5D[0, 0, ...], self.bnt_matrix, 'L', 'L'
            )[None, None, ...]
            noise_3x2pt_5D = bnt_utils.cl_bnt_transform_3x2pt(
                noise_3x2pt_5D, self.bnt_matrix
            )

        # reshape auto-probe spectra to 5D
        cl_LL_5D = cl_LL_3D[np.newaxis, np.newaxis, ...]
        cl_GG_5D = cl_GG_3D[np.newaxis, np.newaxis, ...]

        if split_gaussian_cov:
            self.cov_WL_g_6D_sva, self.cov_WL_g_6D_sn, self.cov_WL_g_6D_mix = (
                sl.covariance_einsum_split(
                    cl_LL_5D, noise_LL_5D, self.fsky, self.ell_WL, delta_l_WL
                )
            )
            self.cov_GC_g_6D_sva, self.cov_GC_g_6D_sn, self.cov_GC_g_6D_mix = (
                sl.covariance_einsum_split(
                    cl_GG_5D, noise_GG_5D, self.fsky, self.ell_GC, delta_l_GC
                )
            )
            (
                self.cov_3x2pt_g_10D_sva,
                self.cov_3x2pt_g_10D_sn,
                self.cov_3x2pt_g_10D_mix,
            ) = sl.covariance_einsum_split(
                cl_3x2pt_5D, noise_3x2pt_5D, self.fsky, self.ell_3x2pt, delta_l_3x2pt
            )
            self.cov_WL_g_6D_sva = self.cov_WL_g_6D_sva[0, 0, 0, 0, ...]
            self.cov_WL_g_6D_sn = self.cov_WL_g_6D_sn[0, 0, 0, 0, ...]
            self.cov_WL_g_6D_mix = self.cov_WL_g_6D_mix[0, 0, 0, 0, ...]
            self.cov_GC_g_6D_sva = self.cov_GC_g_6D_sva[0, 0, 0, 0, ...]
            self.cov_GC_g_6D_sn = self.cov_GC_g_6D_sn[0, 0, 0, 0, ...]
            self.cov_GC_g_6D_mix = self.cov_GC_g_6D_mix[0, 0, 0, 0, ...]

            self.cov_WL_g_6D = (
                self.cov_WL_g_6D_sva + self.cov_WL_g_6D_sn + self.cov_WL_g_6D_mix
            )
            self.cov_GC_g_6D = (
                self.cov_GC_g_6D_sva + self.cov_GC_g_6D_sn + self.cov_GC_g_6D_mix
            )
            self.cov_3x2pt_g_10D = (
                self.cov_3x2pt_g_10D_sva
                + self.cov_3x2pt_g_10D_sn
                + self.cov_3x2pt_g_10D_mix
            )

            if self.GL_OR_LG == 'GL':
                self.cov_XC_g_6D_sva = self.cov_3x2pt_g_10D_sva[1, 0, 1, 0, ...]
                self.cov_XC_g_6D_sn = self.cov_3x2pt_g_10D_sn[1, 0, 1, 0, ...]
                self.cov_XC_g_6D_mix = self.cov_3x2pt_g_10D_mix[1, 0, 1, 0, ...]
            elif self.GL_OR_LG == 'LG':
                self.cov_XC_g_6D_sva = self.cov_3x2pt_g_10D_sva[0, 1, 0, 1, ...]
                self.cov_XC_g_6D_sn = self.cov_3x2pt_g_10D_sn[0, 1, 0, 1, ...]
                self.cov_XC_g_6D_mix = self.cov_3x2pt_g_10D_mix[0, 1, 0, 1, ...]
            else:
                raise ValueError('GL_OR_LG must be "GL" or "LG"')

            self.cov_WL_g_2D_sva = self.reshape_cov(
                self.cov_WL_g_6D_sva,
                6,
                2,
                self.nbl_WL,
                self.zpairs_auto,
                self.ind_auto,
                is_3x2pt=False,
            )
            self.cov_WL_g_2D_sn = self.reshape_cov(
                self.cov_WL_g_6D_sn,
                6,
                2,
                self.nbl_WL,
                self.zpairs_auto,
                self.ind_auto,
                is_3x2pt=False,
            )
            self.cov_WL_g_2D_mix = self.reshape_cov(
                self.cov_WL_g_6D_mix,
                6,
                2,
                self.nbl_WL,
                self.zpairs_auto,
                self.ind_auto,
                is_3x2pt=False,
            )

            self.cov_GC_g_2D_sva = self.reshape_cov(
                self.cov_GC_g_6D_sva,
                6,
                2,
                self.nbl_GC,
                self.zpairs_auto,
                self.ind_auto,
                is_3x2pt=False,
            )
            self.cov_GC_g_2D_sn = self.reshape_cov(
                self.cov_GC_g_6D_sn,
                6,
                2,
                self.nbl_GC,
                self.zpairs_auto,
                self.ind_auto,
                is_3x2pt=False,
            )
            self.cov_GC_g_2D_mix = self.reshape_cov(
                self.cov_GC_g_6D_mix,
                6,
                2,
                self.nbl_GC,
                self.zpairs_auto,
                self.ind_auto,
                is_3x2pt=False,
            )

            self.cov_XC_g_2D_sva = self.reshape_cov(
                self.cov_XC_g_6D_sva,
                6,
                2,
                self.nbl_3x2pt,
                self.zpairs_cross,
                self.ind_cross,
                is_3x2pt=False,
            )
            self.cov_XC_g_2D_sn = self.reshape_cov(
                self.cov_XC_g_6D_sn,
                6,
                2,
                self.nbl_3x2pt,
                self.zpairs_cross,
                self.ind_cross,
                is_3x2pt=False,
            )
            self.cov_XC_g_2D_mix = self.reshape_cov(
                self.cov_XC_g_6D_mix,
                6,
                2,
                self.nbl_3x2pt,
                self.zpairs_cross,
                self.ind_cross,
                is_3x2pt=False,
            )

            self.cov_3x2pt_g_2D_sva = self.reshape_cov(
                self.cov_3x2pt_g_10D_sva,
                10,
                2,
                self.nbl_3x2pt,
                self.zpairs_auto,
                self.ind,
                is_3x2pt=True,
            )
            self.cov_3x2pt_g_2D_mix = self.reshape_cov(
                self.cov_3x2pt_g_10D_mix,
                10,
                2,
                self.nbl_3x2pt,
                self.zpairs_auto,
                self.ind,
                is_3x2pt=True,
            )
            self.cov_3x2pt_g_2D_sn = self.reshape_cov(
                self.cov_3x2pt_g_10D_sn,
                10,
                2,
                self.nbl_3x2pt,
                self.zpairs_auto,
                self.ind,
                is_3x2pt=True,
            )

        else:
            self.cov_WL_g_6D = sl.covariance_einsum(
                cl_LL_5D, noise_LL_5D, self.fsky, self.ell_WL, delta_l_WL
            )[0, 0, 0, 0, ...]
            self.cov_GC_g_6D = sl.covariance_einsum(
                cl_GG_5D, noise_GG_5D, self.fsky, self.ell_GC, delta_l_GC
            )[0, 0, 0, 0, ...]
            self.cov_3x2pt_g_10D = sl.covariance_einsum(
                cl_3x2pt_5D, noise_3x2pt_5D, self.fsky, self.ell_3x2pt, delta_l_3x2pt
            )

        # this part is in common, the split case also sets the total cov
        if self.GL_OR_LG == 'GL':
            self.cov_XC_g_6D = self.cov_3x2pt_g_10D[1, 0, 1, 0, ...]
        elif self.GL_OR_LG == 'LG':
            self.cov_XC_g_6D = self.cov_3x2pt_g_10D[0, 1, 0, 1, ...]
        else:
            raise ValueError('GL_OR_LG must be "GL" or "LG"')

        self.cov_WL_g_2D = self.reshape_cov(
            self.cov_WL_g_6D,
            6,
            2,
            self.nbl_WL,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )
        self.cov_GC_g_2D = self.reshape_cov(
            self.cov_GC_g_6D,
            6,
            2,
            self.nbl_GC,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )
        self.cov_XC_g_2D = self.reshape_cov(
            self.cov_XC_g_6D,
            6,
            2,
            self.nbl_GC,
            self.zpairs_cross,
            self.ind_cross,
            is_3x2pt=False,
        )
        self.cov_3x2pt_g_2D = self.reshape_cov(
            self.cov_3x2pt_g_10D,
            10,
            2,
            self.nbl_3x2pt,
            self.zpairs_auto,
            self.ind,
            is_3x2pt=True,
        )

        print(
            'Gauss. cov. matrices computed in %.2f seconds'
            % (time.perf_counter() - start)
        )

    def _cov_8d_dict_to_10d_arr(self, cov_dict_8D):
        """Helper function to process a single covariance component"""

        cov_dict_10D = sl.cov_3x2pt_dict_8d_to_10d(
            cov_dict_8D,
            self.nbl_3x2pt,
            self.zbins,
            self.ind_dict,
            self.probe_ordering,
            self.symmetrize_output_dict,
        )

        return sl.cov_10D_dict_to_array(
            cov_dict_10D, self.nbl_3x2pt, self.zbins, self.n_probes
        )

    def build_covs(self, ccl_obj, oc_obj):
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

        Returns
        -------
        dict
            Dictionary containing the computed covariance matrices with keys:
            - cov_{probe}_g_2D: Gaussian-only covariance
            - cov_{probe}_ng_2D: ng-only covariance (SSC, cNG or the sum of the two)
            - cov_{probe}_tot_2D: g + ng covariance
            where {probe} can be: WL (weak lensing), GC (galaxy clustering),
            3x2pt (WL + XC + GC), XC (cross-correlation), 2x2pt (XC + GC)
        """

        self.cov_dict = {}

        if self.g_code == 'OneCovariance':
            raise NotImplementedError(
                'OneCovariance g term not yet implemented: split terms '
                'and probe-specific ell binning missing'
            )
            self.cov_WL_g_6D = oc_obj.cov_g_oc_3x2pt_10D[0, 0, 0, 0]
            self.cov_GC_g_6D = oc_obj.cov_g_oc_3x2pt_10D[1, 1, 1, 1]
            self.cov_3x2pt_g_10D = oc_obj.cov_g_oc_3x2pt_10D

        # ! construct 10D total 3x2pt NG (SSC + NG) covariance matrix depending
        # ! on chosen cov and terms
        if self.include_ssc:
            print(f'Including SSC from {self.ssc_code} in total covariance')
            if self.ssc_code == 'Spaceborne':
                self.cov_3x2pt_ssc_10D = self._cov_8d_dict_to_10d_arr(
                    self.cov_ssc_sb_3x2pt_dict_8D
                )
            elif self.ssc_code == 'PyCCL':
                self.cov_3x2pt_ssc_10D = self._cov_8d_dict_to_10d_arr(
                    ccl_obj.cov_ssc_ccl_3x2pt_dict_8D
                )
            elif self.ssc_code == 'OneCovariance':
                self.cov_3x2pt_ssc_10D = oc_obj.cov_ssc_oc_3x2pt_10D
            assert not np.allclose(self.cov_3x2pt_ssc_10D, 0, atol=0, rtol=1e-10), (
                f'{self.ssc_code} SSC covariance matrix is identically zero'
            )
        else:
            print('SSC not requested, setting it to zero')
            self.cov_3x2pt_ssc_10D = np.zeros_like(self.cov_3x2pt_g_10D)

        if self.include_cng:
            print(f'Including SSC from {self.ssc_code} in total covariance')
            if self.cng_code == 'PyCCL':
                self.cov_3x2pt_cng_10D = self._cov_8d_dict_to_10d_arr(
                    ccl_obj.cov_cng_ccl_3x2pt_dict_8D
                )
            elif self.cng_code == 'OneCovariance':
                self.cov_3x2pt_cng_10D = oc_obj.cov_cng_oc_3x2pt_10D
            assert not np.allclose(self.cov_3x2pt_cng_10D, 0, atol=0, rtol=1e-10), (
                f'{self.cng_code} cNG covariance matrix is identically zero'
            )
        else:
            print('cNG term not requested, setting it to zero')
            self.cov_3x2pt_cng_10D = np.zeros_like(self.cov_3x2pt_g_10D)

        # sum SSC and cNG
        self.cov_3x2pt_ng_10D = self.cov_3x2pt_ssc_10D + self.cov_3x2pt_cng_10D

        # # ! Select appropriate non-Gaussian covariance terms to include and from
        # ! which code
        # if self.ssc_code == 'Spaceborne':
        #     cov_3x2pt_ng_10D = self._cov_8d_dict_to_10d_arr(
        #         self.cov_ssc_sb_3x2pt_dict_8D
        #     )

        #     if self.covariance_cfg['which_cNG'] == 'OneCovariance':
        #         print('Adding cNG covariance from OneCovariance...')

        #         # test that oc_obj.cov_cng_oc_3x2pt_10D is not identically zero
        #         assert not np.allclose(
        #             oc_obj.cov_cng_oc_3x2pt_10D, 0, atol=0, rtol=1e-10
        #         ), 'OneCovariance covariance matrix is identically zero'

        #         cov_3x2pt_ng_10D += oc_obj.cov_cng_oc_3x2pt_10D

        #     elif self.covariance_cfg['which_cNG'] == 'PyCCL':
        #         print('Adding cNG covariance from PyCCL...')

        #         cov_cng_ccl_3x2pt_10D = self._cov_8d_dict_to_10d_arr(
        #             ccl_obj.cov_cng_ccl_3x2pt_dict_8D
        #         )

        #         # test that oc_obj.cov_cng_oc_3x2pt_10D is not identically zero
        #         assert not np.allclose(cov_cng_ccl_3x2pt_10D, 0, atol=0, rtol=1e-10), (
        #             'OneCovariance covariance matrix is identically zero'
        #         )

        #         cov_3x2pt_ng_10D += cov_cng_ccl_3x2pt_10D

        # elif self.ng_cov_code == 'OneCovariance':
        #     if self.covariance_cfg['OneCovariance_cfg']['use_OneCovariance_Gaussian']:
        #         print('Loading Gaussian covariance from OneCovariance...')
        #         # TODO do it with pyccl as well, after computing the G covariance
        #         cov_3x2pt_g_10D = oc_obj.cov_g_oc_3x2pt_10D

        #         # Slice or reload to get the LL, GG and 3x2pt covariance
        #         cov_WL_g_6D = deepcopy(
        #             cov_3x2pt_g_10D[
        #                 0, 0, 0, 0, : self.nbl_WL, : self.nbl_WL, :, :, :, :
        #             ]
        #         )
        #         cov_GC_g_6D = deepcopy(
        #             cov_3x2pt_g_10D[
        #                 1, 1, 1, 1, : self.nbl_GC, : self.nbl_GC, :, :, :, :
        #             ]
        #         )
        #         cov_3x2pt_g_10D = deepcopy(
        #             cov_3x2pt_g_10D[
        #                 :, :, :, :, : self.nbl_3x2pt, : self.nbl_3x2pt, :, :, :, :
        #             ]
        #         )

        #     if self.covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == [
        #         'SSC',
        #     ]:
        #         cov_3x2pt_ng_10D = oc_obj.cov_ssc_oc_3x2pt_10D

        #     elif self.covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == [
        #         'cNG',
        #     ]:
        #         cov_3x2pt_ng_10D = oc_obj.cov_cng_oc_3x2pt_10D

        #     elif self.covariance_cfg['OneCovariance_cfg']['which_ng_cov'] == [
        #         'SSC',
        #         'cNG',
        #     ]:
        #         cov_3x2pt_ng_10D = oc_obj.cov_ssc_oc_3x2pt_10D
        #         cov_3x2pt_ng_10D += oc_obj.cov_cng_oc_3x2pt_10D

        #     else:
        #         raise ValueError(
        #             "covariance_cfg['OneCovariance_cfg']['which_ng_cov'] not recognised"
        #         )

        # elif self.ng_cov_code == 'PyCCL':
        #     print('Using PyCCL non-Gaussian covariance matrices...')

        #     if self.covariance_cfg['PyCCL_cfg']['which_ng_cov'] == [
        #         'SSC',
        #     ]:
        #         cov_3x2pt_ng_10D = self._cov_8d_dict_to_10d_arr(
        #             ccl_obj.cov_ssc_ccl_3x2pt_dict_8D
        #         )

        #     elif self.covariance_cfg['PyCCL_cfg']['which_ng_cov'] == [
        #         'cNG',
        #     ]:
        #         cov_3x2pt_ng_10D = self._cov_8d_dict_to_10d_arr(
        #             ccl_obj.cov_cng_ccl_3x2pt_dict_8D
        #         )

        #     elif self.covariance_cfg['PyCCL_cfg']['which_ng_cov'] == ['SSC', 'cNG']:
        #         cov_3x2pt_ng_10D = self._cov_8d_dict_to_10d_arr(
        #             ccl_obj.cov_ssc_ccl_3x2pt_dict_8D
        #         )
        #         cov_3x2pt_ng_10D += self._cov_8d_dict_to_10d_arr(
        #             ccl_obj.cov_cng_ccl_3x2pt_dict_8D
        #         )

        #     else:
        #         raise ValueError(
        #             "covariance_cfg['PyCCL_cfg']['which_ng_cov'] not recognised"
        #         )

        # else:
        #     raise NotImplementedError(f'ng_cov_code {self.ng_cov_code} not implemented')

        # In this case, you just need to slice get the LL, GG and 3x2pt covariance
        # WL slicing unnecessary, since I load with nbl_WL and max_WL but just in case
        cov_WL_ssc_6D = deepcopy(
            self.cov_3x2pt_ssc_10D[0, 0, 0, 0, : self.nbl_WL, : self.nbl_WL, :, :, :, :]
        )
        cov_WL_cng_6D = deepcopy(
            self.cov_3x2pt_cng_10D[0, 0, 0, 0, : self.nbl_WL, : self.nbl_WL, :, :, :, :]
        )
        cov_GC_ssc_6D = deepcopy(
            self.cov_3x2pt_ssc_10D[1, 1, 1, 1, : self.nbl_GC, : self.nbl_GC, :, :, :, :]
        )
        cov_GC_cng_6D = deepcopy(
            self.cov_3x2pt_cng_10D[1, 1, 1, 1, : self.nbl_GC, : self.nbl_GC, :, :, :, :]
        )
        # TODO I think this is unnecessary
        self.cov_3x2pt_ssc_10D = deepcopy(
            self.cov_3x2pt_ssc_10D[
                :, :, :, :, : self.nbl_3x2pt, : self.nbl_3x2pt, :, :, :, :
            ]
        )
        self.cov_3x2pt_cng_10D = deepcopy(
            self.cov_3x2pt_cng_10D[
                :, :, :, :, : self.nbl_3x2pt, : self.nbl_3x2pt, :, :, :, :
            ]
        )

        # ! BNT transform (6/10D covs needed for this implementation)
        if self.cfg['BNT']['cov_BNT_transform']:
            print('BNT-transforming the covariance matrix...')
            start = time.perf_counter()

            # turn 3x2pt 10d array to dict for the BNT function
            cov_3x2pt_g_10D_dict = sl.cov_10D_array_to_dict(
                self.cov_3x2pt_g_10D, self.probe_ordering
            )
            cov_3x2pt_ssc_10D_dict = sl.cov_10D_array_to_dict(
                self.cov_3x2pt_ssc_10D, self.probe_ordering
            )
            cov_3x2pt_cng_10D_dict = sl.cov_10D_array_to_dict(
                self.cov_3x2pt_cng_10D, self.probe_ordering
            )

            # BNT-transform WL and 3x2pt g, ng and tot covariances
            X_dict = bnt_utils.build_x_matrix_bnt(self.bnt_matrix)
            # TODO BNT and scale cuts of G term should go in the gauss cov function!
            self.cov_WL_g_6D = bnt_utils.cov_bnt_transform(
                self.cov_WL_g_6D, X_dict, 'L', 'L', 'L', 'L'
            )
            cov_WL_ssc_6D = bnt_utils.cov_bnt_transform(
                cov_WL_ssc_6D, X_dict, 'L', 'L', 'L', 'L'
            )
            cov_WL_cng_6D = bnt_utils.cov_bnt_transform(
                cov_WL_cng_6D, X_dict, 'L', 'L', 'L', 'L'
            )
            cov_3x2pt_g_10D_dict = bnt_utils.cov_3x2pt_bnt_transform(
                cov_3x2pt_g_10D_dict, X_dict
            )
            cov_3x2pt_ssc_10D_dict = bnt_utils.cov_3x2pt_bnt_transform(
                cov_3x2pt_ssc_10D_dict, X_dict
            )
            cov_3x2pt_cng_10D_dict = bnt_utils.cov_3x2pt_bnt_transform(
                cov_3x2pt_cng_10D_dict, X_dict
            )

            # revert to 10D arrays - this is not strictly necessary since
            # cov_3x2pt_10D_to_4D accepts both a dictionary and
            # an array as input, but it's done to keep the variable names consistent
            # ! BNT IS LINEAR, SO BNT(COV_TOT) = \SUM_i BNT(COV_i), but should check
            self.cov_3x2pt_g_10D = sl.cov_10D_dict_to_array(
                cov_3x2pt_g_10D_dict, self.nbl_3x2pt, self.zbins, n_probes=2
            )
            self.cov_3x2pt_ssc_10D = sl.cov_10D_dict_to_array(
                cov_3x2pt_ssc_10D_dict, self.nbl_3x2pt, self.zbins, n_probes=2
            )
            self.cov_3x2pt_cng_10D = sl.cov_10D_dict_to_array(
                cov_3x2pt_cng_10D_dict, self.nbl_3x2pt, self.zbins, n_probes=2
            )

            print(
                f'Covariance matrices BNT-transformed in '
                f'{time.perf_counter() - start:.2f} s'
            )

        if self.GL_OR_LG == 'GL':
            cov_XC_g_6D = self.cov_3x2pt_g_10D[1, 0, 1, 0, ...]
            cov_XC_ssc_6D = self.cov_3x2pt_ssc_10D[1, 0, 1, 0, ...]
            cov_XC_cng_6D = self.cov_3x2pt_cng_10D[1, 0, 1, 0, ...]
        elif self.GL_OR_LG == 'LG':
            cov_XC_g_6D = self.cov_3x2pt_g_10D[0, 1, 0, 1, ...]
            # ! I'm doing this in a more exotic way above, for ng
            cov_XC_ssc_6D = self.cov_3x2pt_ssc_10D[0, 1, 0, 1, ...]
            cov_XC_cng_6D = self.cov_3x2pt_cng_10D[0, 1, 0, 1, ...]
        else:
            raise ValueError('GL_OR_LG must be "GL" or "LG"')

        # # ! reshape everything to 2D
        self.cov_WL_g_2D = self.reshape_cov(
            self.cov_WL_g_6D,
            6,
            2,
            self.nbl_WL,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )
        self.cov_WL_ssc_2D = self.reshape_cov(
            cov_WL_ssc_6D,
            6,
            2,
            self.nbl_WL,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )
        self.cov_WL_cng_2D = self.reshape_cov(
            cov_WL_cng_6D,
            6,
            2,
            self.nbl_WL,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )

        self.cov_GC_g_2D = self.reshape_cov(
            self.cov_GC_g_6D,
            6,
            2,
            self.nbl_GC,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )
        self.cov_GC_ssc_2D = self.reshape_cov(
            cov_GC_ssc_6D,
            6,
            2,
            self.nbl_GC,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )
        self.cov_GC_cng_2D = self.reshape_cov(
            cov_GC_cng_6D,
            6,
            2,
            self.nbl_GC,
            self.zpairs_auto,
            self.ind_auto,
            is_3x2pt=False,
        )

        self.cov_XC_g_2D = self.reshape_cov(
            cov_XC_g_6D,
            6,
            2,
            self.nbl_3x2pt,
            self.zpairs_cross,
            self.ind_cross,
            is_3x2pt=False,
        )
        self.cov_XC_ssc_2D = self.reshape_cov(
            cov_XC_ssc_6D,
            6,
            2,
            self.nbl_3x2pt,
            self.zpairs_cross,
            self.ind_cross,
            is_3x2pt=False,
        )
        self.cov_XC_cng_2D = self.reshape_cov(
            cov_XC_cng_6D,
            6,
            2,
            self.nbl_3x2pt,
            self.zpairs_cross,
            self.ind_cross,
            is_3x2pt=False,
        )

        self.cov_3x2pt_g_2D = self.reshape_cov(
            self.cov_3x2pt_g_10D,
            10,
            2,
            self.nbl_3x2pt,
            self.zpairs_auto,
            self.ind,
            is_3x2pt=True,
        )
        self.cov_3x2pt_ssc_2D = self.reshape_cov(
            self.cov_3x2pt_ssc_10D,
            10,
            2,
            self.nbl_3x2pt,
            self.zpairs_auto,
            self.ind,
            is_3x2pt=True,
        )
        self.cov_3x2pt_cng_2D = self.reshape_cov(
            self.cov_3x2pt_cng_10D,
            10,
            2,
            self.nbl_3x2pt,
            self.zpairs_auto,
            self.ind,
            is_3x2pt=True,
        )

        # ! perform ell cuts on the 2D covs
        if self.cfg['ell_cuts']['cov_ell_cuts']:
            print('Performing ell cuts on the 2d covariance matrix...')
            self.cov_WL_g_2D = sl.remove_rows_cols_array2D(
                self.cov_WL_g_2D, self.ell_dict['idxs_to_delete_dict']['LL']
            )
            self.cov_GC_g_2D = sl.remove_rows_cols_array2D(
                self.cov_GC_g_2D, self.ell_dict['idxs_to_delete_dict']['GG']
            )
            self.cov_XC_g_2D = sl.remove_rows_cols_array2D(
                self.cov_XC_g_2D, self.ell_dict['idxs_to_delete_dict'][self.GL_OR_LG]
            )
            self.cov_3x2pt_g_2D = sl.remove_rows_cols_array2D(
                self.cov_3x2pt_g_2D, self.ell_dict['idxs_to_delete_dict']['3x2pt']
            )
            self.cov_2x2pt_g_2D = sl.remove_rows_cols_array2D(
                self.cov_2x2pt_g_2D, self.ell_dict['idxs_to_delete_dict']['2x2pt']
            )

            self.cov_WL_ssc_2D = sl.remove_rows_cols_array2D(
                self.cov_WL_ssc_2D, self.ell_dict['idxs_to_delete_dict']['LL']
            )
            self.cov_GC_ssc_2D = sl.remove_rows_cols_array2D(
                self.cov_GC_ssc_2D, self.ell_dict['idxs_to_delete_dict']['GG']
            )
            self.cov_XC_ssc_2D = sl.remove_rows_cols_array2D(
                self.cov_XC_ssc_2D, self.ell_dict['idxs_to_delete_dict'][self.GL_OR_LG]
            )
            self.cov_3x2pt_ssc_2D = sl.remove_rows_cols_array2D(
                self.cov_3x2pt_ssc_2D, self.ell_dict['idxs_to_delete_dict']['3x2pt']
            )
            self.cov_2x2pt_ssc_2D = sl.remove_rows_cols_array2D(
                self.cov_2x2pt_ssc_2D, self.ell_dict['idxs_to_delete_dict']['2x2pt']
            )

            self.cov_WL_cng_2D = sl.remove_rows_cols_array2D(
                self.cov_WL_cng_2D, self.ell_dict['idxs_to_delete_dict']['LL']
            )
            self.cov_GC_cng_2D = sl.remove_rows_cols_array2D(
                self.cov_GC_cng_2D, self.ell_dict['idxs_to_delete_dict']['GG']
            )
            self.cov_XC_cng_2D = sl.remove_rows_cols_array2D(
                self.cov_XC_cng_2D, self.ell_dict['idxs_to_delete_dict'][self.GL_OR_LG]
            )
            self.cov_3x2pt_cng_2D = sl.remove_rows_cols_array2D(
                self.cov_3x2pt_cng_2D, self.ell_dict['idxs_to_delete_dict']['3x2pt']
            )
            self.cov_2x2pt_cng_2D = sl.remove_rows_cols_array2D(
                self.cov_2x2pt_cng_2D, self.ell_dict['idxs_to_delete_dict']['2x2pt']
            )

        # store in dictionary
        # TODO is this necessaty? I can probably delete!
        probe_names = ('WL', 'GC', '3x2pt', 'XC', '2x2pt')
        covs_g_2D = (
            self.cov_WL_g_2D,
            self.cov_GC_g_2D,
            self.cov_3x2pt_g_2D,
            self.cov_XC_g_2D,
        )
        covs_ssc_2D = (
            self.cov_WL_ssc_2D,
            self.cov_GC_ssc_2D,
            self.cov_3x2pt_ssc_2D,
            self.cov_XC_ssc_2D,
        )
        covs_cng_2D = (
            self.cov_WL_cng_2D,
            self.cov_GC_cng_2D,
            self.cov_3x2pt_cng_2D,
            self.cov_XC_cng_2D,
        )
        covs_tot_2D = (
            self.cov_WL_g_2D + self.cov_WL_ssc_2D + self.cov_WL_cng_2D,
            self.cov_GC_g_2D + self.cov_GC_ssc_2D + self.cov_GC_cng_2D,
            self.cov_3x2pt_g_2D + self.cov_3x2pt_ssc_2D + self.cov_3x2pt_cng_2D,
            self.cov_XC_g_2D + self.cov_XC_ssc_2D + self.cov_XC_cng_2D,
        )

        for probe_name, cov_g_2D, cov_ssc_2D, cov_cng_2D, cov_tot_2D in zip(
            probe_names, covs_g_2D, covs_ssc_2D, covs_cng_2D, covs_tot_2D
        ):
            self.cov_dict[f'cov_{probe_name}_g_2D'] = cov_g_2D
            self.cov_dict[f'cov_{probe_name}_ssc_2D'] = cov_ssc_2D
            self.cov_dict[f'cov_{probe_name}_cng_2D'] = cov_cng_2D
            self.cov_dict[f'cov_{probe_name}_tot_2D'] = cov_tot_2D

        print('Covariance matrices computed')

        return self.cov_dict

    def bin_2d_matrix(self, cov, ells_in, ells_out, ells_out_edges):
        assert cov.shape[0] == cov.shape[1] == len(ells_in), (
            'ells_in must be the same length as the covariance matrix'
        )
        assert len(ells_out) == len(ells_out_edges) - 1, (
            'ells_out must be the same length as the number of edges - 1'
        )

        binned_cov = np.zeros((len(ells_out), len(ells_out)))
        cov_interp_func = RectBivariateSpline(ells_in, ells_in, cov)

        ells_edges_low = ells_out_edges[:-1]
        ells_edges_high = ells_out_edges[1:]

        # Loop over the output bins
        for ell1_idx, _ in enumerate(ells_out):
            for ell2_idx, _ in enumerate(ells_out):
                # Get ell min/max for the current bins
                ell1_min = ells_edges_low[ell1_idx]
                ell1_max = ells_edges_high[ell1_idx]
                ell2_min = ells_edges_low[ell2_idx]
                ell2_max = ells_edges_high[ell2_idx]

                # isolate the relevant ranges of ell values from the original ells_in grid
                ell1_in = ells_in[(ell1_min <= ells_in) & (ells_in < ell1_max)]
                ell2_in = ells_in[(ell2_min <= ells_in) & (ells_in < ell2_max)]

                # mask the covariance to the relevant block
                cov_masked = cov[np.ix_(ell1_in, ell2_in)]

                # Calculate the bin widths
                delta_ell_1 = ell1_max - ell1_min
                delta_ell_2 = ell2_max - ell2_min

                # Option 1a: use the original grid for integration and the ell values
                # as weights
                # ells1_in_xx, ells2_in_yy = np.meshgrid(ell1_in, ell2_in, indexing='ij')
                # partial_integral = simps(y=cov_masked * ells1_in_xx * ells2_in_yy,
                # x=ell2_in, axis=1)
                # integral = simps(y=partial_integral, x=ell1_in)
                # binned_cov[ell1_idx, ell2_idx] = integral / (
                # np.sum(ell1_in) * np.sum(ell2_in)
                # )

                # Option 1b: use the original grid for integration and no weights
                partial_integral = simps(y=cov_masked, x=ell2_in, axis=1)
                integral = simps(y=partial_integral, x=ell1_in)
                binned_cov[ell1_idx, ell2_idx] = integral / (delta_ell_1 * delta_ell_2)

                # # Option 2: create fine grids for integration over the ell ranges
                # (GIVES GOOD RESULTS ONLY FOR nsteps=delta_ell!)
                # ell_fine_1 = np.linspace(ell1_min, ell1_max, 50)
                # ell_fine_2 = np.linspace(ell2_min, ell2_max, 50)

                # # Evaluate the spline on the fine grids
                # ell1_fine_xx, ell2_fine_yy = np.meshgrid(
                #     ell_fine_1, ell_fine_2, indexing='ij'
                # )
                # cov_interp_vals = cov_interp_func(ell_fine_1, ell_fine_2)

                # # Perform simps integration over the ell ranges
                # partial_integral = simps(
                #     y=cov_interp_vals * ell1_fine_xx * ell2_fine_yy,
                #     x=ell_fine_2,
                #     axis=1,
                # )
                # integral = simps(y=partial_integral, x=ell_fine_1)
                # # Normalize by the bin areas
                # binned_cov[ell1_idx, ell2_idx] = integral / (
                #     np.sum(ell_fine_1) * np.sum(ell_fine_2)
                # )

        return binned_cov

    def ssc_integral_julia(
        self,
        d2CLL_dVddeltab,
        d2CGL_dVddeltab,
        d2CGG_dVddeltab,
        cl_integral_prefactor,
        sigma2,
        z_grid,
        integration_type,
        probe_ordering,
        num_threads=16,
    ):
        """Kernel to compute the 4D integral optimized using Simpson's rule using
        Julia."""

        suffix = 0
        folder_name = 'tmp'
        unique_folder_name = folder_name

        # Loop until we find a folder name that does not exist
        while os.path.exists(unique_folder_name):
            suffix += 1
            unique_folder_name = f'{folder_name}{suffix}'
        os.makedirs(unique_folder_name)
        folder_name = unique_folder_name

        np.save(f'{folder_name}/d2CLL_dVddeltab', d2CLL_dVddeltab)
        np.save(f'{folder_name}/d2CGL_dVddeltab', d2CGL_dVddeltab)
        np.save(f'{folder_name}/d2CGG_dVddeltab', d2CGG_dVddeltab)
        np.save(f'{folder_name}/ind_auto', self.ind_auto)
        np.save(f'{folder_name}/ind_cross', self.ind_cross)
        np.save(f'{folder_name}/cl_integral_prefactor', cl_integral_prefactor)
        np.save(f'{folder_name}/sigma2', sigma2)
        np.save(f'{folder_name}/z_grid', z_grid)
        os.system(
            f'julia --project=. --threads={num_threads} {self.jl_integrator_path} {folder_name} {integration_type}'
        )

        cov_filename = (
            'cov_SSC_spaceborne_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D.npy'
        )

        if integration_type == 'trapz-6D':
            cov_ssc_sb_3x2pt_dict_8D = {}  # it's 10D, actually
            for probe_a, probe_b in probe_ordering:
                for probe_c, probe_d in probe_ordering:
                    if str.join('', (probe_a, probe_b, probe_c, probe_d)) not in [
                        'GLLL',
                        'GGLL',
                        'GGGL',
                    ]:
                        print(f'Loading {probe_a}{probe_b}{probe_c}{probe_d}')
                        _cov_filename = cov_filename.format(
                            probe_a=probe_a,
                            probe_b=probe_b,
                            probe_c=probe_c,
                            probe_d=probe_d,
                        )
                        cov_ssc_sb_3x2pt_dict_8D[
                            (probe_a, probe_b, probe_c, probe_d)
                        ] = np.load(f'{folder_name}/{_cov_filename}')

        else:
            cov_ssc_sb_3x2pt_dict_8D = sl.load_cov_from_probe_blocks(
                path=f'{folder_name}',
                filename=cov_filename,
                probe_ordering=probe_ordering,
            )

        os.system(f'rm -rf {folder_name}')

        self.cov_ssc_sb_3x2pt_dict_8D = cov_ssc_sb_3x2pt_dict_8D

        return self.cov_ssc_sb_3x2pt_dict_8D

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

    def save_cov(cov_folder, covariance_cfg, cov_dict, cases_tosave, **variable_specs):
        """
        This function is deprecated.
        """

        ell_max_WL = variable_specs['ell_max_WL']
        ell_max_GC = variable_specs['ell_max_GC']
        ell_max_3x2pt = variable_specs['ell_max_3x2pt']
        nbl_WL = variable_specs['nbl_WL']
        nbl_GC = variable_specs['nbl_GC']
        nbl_3x2pt = variable_specs['nbl_3x2pt']

        # which file format to use
        if covariance_cfg['cov_file_format'] == 'npy':
            save_funct = np.save
            extension = 'npy'
        elif covariance_cfg['cov_file_format'] == 'npz':
            save_funct = np.savez_compressed
            extension = 'npz'
        else:
            raise ValueError('cov_file_format not recognized: must be "npy" or "npz"')

        for ndim in (2, 4, 6):
            if covariance_cfg[f'save_cov_{ndim}D']:
                # set probes to save; the ndim == 6 case is different
                probe_list = ['WL', 'GC', '3x2pt']
                ellmax_list = [ell_max_WL, ell_max_GC, ell_max_3x2pt]
                nbl_list = [nbl_WL, nbl_GC, nbl_3x2pt]
                # in this case, 3x2pt is saved in 10D as a dictionary
                if ndim == 6:
                    probe_list = ['WL', 'GC']
                    ellmax_list = [ell_max_WL, ell_max_GC]
                    nbl_list = [nbl_WL, nbl_GC]

                for which_cov in cases_tosave:
                    for probe, ell_max, nbl in zip(probe_list, ellmax_list, nbl_list):
                        cov_filename = covariance_cfg['cov_filename'].format(
                            which_cov=which_cov,
                            probe=probe,
                            ell_max=ell_max,
                            nbl=nbl,
                            ndim=ndim,
                            **variable_specs,
                        )
                        save_funct(
                            f'{cov_folder}/{cov_filename}.{extension}',
                            cov_dict[f'cov_{probe}_{which_cov}_{ndim}D'],
                        )  # save in .npy or .npz

                    # in this case, 3x2pt is saved in 10D as a dictionary
                    # TODO these pickle files are too heavy, probably it's best to
                    # revert to npz
                    if ndim == 6:
                        cov_3x2pt_filename = covariance_cfg['cov_filename'].format(
                            which_cov=which_cov,
                            probe='3x2pt',
                            ell_max=ell_max_3x2pt,
                            nbl=nbl_3x2pt,
                            ndim=10,
                            **variable_specs,
                        )
                        with open(
                            f'{cov_folder}/{cov_3x2pt_filename}.pickle', 'wb'
                        ) as handle:
                            pickle.dump(cov_dict[f'cov_3x2pt_{which_cov}_10D'], handle)

                print(
                    f'Covariance matrices saved in {covariance_cfg["cov_file_format"]}'
                )

        # save in .dat for Vincenzo, only in the optimistic case and in 2D
        if covariance_cfg['save_cov_dat'] and ell_max_WL == 5000:
            for probe, probe_vinc in zip(
                ['WL', 'GC', '3x2pt'], ['WLO', 'GCO', '3x2pt']
            ):
                for GOGS_folder, GOGS_filename in zip(
                    ['GaussOnly', 'GaussSSC'], ['GO', 'GS']
                ):
                    cov_filename_vincenzo = covariance_cfg[
                        'cov_filename_vincenzo'
                    ].format(
                        probe_vinc=probe_vinc,
                        GOGS_filename=GOGS_filename,
                        **variable_specs,
                    )
                    np.savetxt(
                        f'{cov_folder}/{GOGS_folder}/{cov_filename_vincenzo}',
                        cov_dict[f'cov_{probe}_{GOGS_filename}_2D'],
                        fmt='%.8e',
                    )
            print('Covariance matrices saved')
