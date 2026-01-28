import time
import warnings
from copy import deepcopy

import numpy as np

from spaceborne import bnt as bnt_utils
from spaceborne import constants as const
from spaceborne import cov_dict as cd
from spaceborne import sb_lib as sl
from spaceborne.ccl_interface import CCLInterface
from spaceborne.cov_partial_sky import NmtCov
from spaceborne.cov_ssc import SpaceborneSSC
from spaceborne.ell_utils import EllBinning
from spaceborne.oc_interface import OneCovarianceInterface


class SpaceborneCovariance:
    def __init__(
        self,
        cfg: dict,
        pvt_cfg: dict,
        ell_obj: EllBinning,
        cov_nmt_obj: NmtCov | None,
        bnt_matrix: np.ndarray | None,
    ):
        self.cfg = cfg
        self.cov_cfg = cfg['covariance']
        self.ell_dict = {}
        self.ell_obj = ell_obj
        self.bnt_matrix = bnt_matrix
        self.probe_names_dict = {'LL': 'WL', 'GG': 'GC', '3x2pt': '3x2pt'}
        # TODO these should probably be defined on a higher level
        self.llll_ixs = (0, 0, 0, 0)
        self.glgl_ixs = (1, 0, 1, 0)
        self.gggg_ixs = (1, 1, 1, 1)

        self.zbins = pvt_cfg['zbins']

        self.fsky = pvt_cfg['fsky']
        self.symmetrize_output_dict = pvt_cfg['symmetrize_output_dict']
        self.unique_probe_combs = pvt_cfg['unique_probe_combs']

        # ordering-related stuff
        self.probe_ordering = pvt_cfg['probe_ordering']  # TODO delete this??
        self.ind = pvt_cfg['ind']
        self.ind_auto = pvt_cfg['ind_auto']
        self.ind_cross = pvt_cfg['ind_cross']
        self.ind_dict = pvt_cfg['ind_dict']
        self.zpairs_auto = pvt_cfg['zpairs_auto']
        self.zpairs_cross = pvt_cfg['zpairs_cross']
        self.zpairs_3x2pt = pvt_cfg['zpairs_3x2pt']
        self.block_index = pvt_cfg['block_index']

        # instantiate cov dict with the required terms and probe combinations
        self.req_terms = pvt_cfg['req_terms']
        self.req_probe_combs_2d = pvt_cfg['req_probe_combs_hs_2d']
        self.nonreq_probe_combs = pvt_cfg['nonreq_probe_combs_hs']
        dims = ['6d', '4d', '2d']

        _req_probe_combs_2d = [
            sl.split_probe_name(probe, space='harmonic')
            for probe in self.req_probe_combs_2d
        ]
        _req_probe_combs_2d.append('3x2pt')
        self.cov_dict = cd.create_cov_dict(
            self.req_terms, _req_probe_combs_2d, dims=dims
        )

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
        self.cov_nmt_obj = cov_nmt_obj

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

    def set_gauss_cov(self, ccl_obj: CCLInterface):
        start = time.perf_counter()
        
        print('\nComputing Gaussian harmonic-space covariance matrix...')

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
        noise_3x2pt_unb_5d = np.repeat(
            noise_3x2pt_4d[:, :, np.newaxis, :, :], self.ell_obj.nbl_3x2pt_unb, axis=2
        )

        # bnt-transform the noise spectra if needed
        if self.cfg['BNT']['cl_BNT_transform']:
            print('BNT-transforming the noise spectra...')
            noise_3x2pt_5d = bnt_utils.cl_bnt_transform_3x2pt(
                noise_3x2pt_5d, self.bnt_matrix
            )

        if self.cfg['precision']['cov_hs_g_ell_bin_average']:
            # unbinned cls and noise; need the edges to compute the number of modes 
            # (after casting them to int. n_modes is equivalent to delta_ell modulo the 
            # fact that for delta_ell we consider non-integer ell values)
            _cl_5d = self.cl_3x2pt_unb_5d
            _noise_5d = noise_3x2pt_unb_5d
            _ell_values = self.ell_obj.ells_3x2pt_unb
            _ell_edges = self.ell_obj.ell_edges_3x2pt
        else:
            # evaluate the covariance at the center of the ell bin and normalise by 
            # delta_ell
            _cl_5d = cl_3x2pt_5d
            _noise_5d = noise_3x2pt_5d
            _ell_values = self.ell_obj.ells_3x2pt
            _ell_edges = None

        # ! compute 3x2pt fsky Gaussian covariance: by default, split SVA, SN and MIX
        # the Gaussian HS cov is computed for all probes at once, still
        (cov_3x2pt_sva_10d, cov_3x2pt_sn_10d, cov_3x2pt_mix_10d) = sl.compute_g_cov(
            cl_5d=_cl_5d,
            noise_5d=_noise_5d,
            fsky=self.fsky,
            ell_values=_ell_values,
            delta_ell=self.ell_obj.delta_l_3x2pt,
            split_terms=True,
            return_only_ell_diagonal=False,
            cov_hs_g_ell_bin_average=self.cfg['precision']['cov_hs_g_ell_bin_average'],
            ell_edges=_ell_edges,
        )

        # assign the different probes in the 10d array to the appropriate dict keys
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

        # zero-out the blocks not requested
        for probe_abcd in self.nonreq_probe_combs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
            probe_2tpl = (probe_ab, probe_cd)
            probe_ixs = tuple(const.HS_PROBE_NAME_TO_IX_DICT[p] for p in probe_abcd)
            for term in ('sva', 'sn', 'mix', 'g'):
                self.cov_dict[term][probe_2tpl]['6d'] = np.zeros_like(
                    self.cov_dict[term][probe_2tpl]['6d']
                )

        # ! Partial sky with nmt
        # ! this case overwrites self.cov_3x2pt_g_10d only, but the cfg checker will
        # ! raise an error if you require to split the G cov and use_nmt or
        # ! do_sample_cov are True
        if self.use_nmt or self.do_sample_cov:
            if self.cov_nmt_obj is None:
                raise ValueError(
                    'cov_nmt_obj is required when use_namaster or compute_sample_cov '
                    'is True'
                )

            # noise vector doesn't have to be recomputed, but repeated a larger number
            # of times (ell by ell)
            noise_3x2pt_unb_5d = np.repeat(
                noise_3x2pt_4d[:, :, np.newaxis, :, :],
                repeats=self.cov_nmt_obj.nbl_3x2pt_unb,
                axis=2,
            )
            self.cov_nmt_obj.noise_3x2pt_unb_5d = noise_3x2pt_unb_5d
            cov_nmt_dict = self.cov_nmt_obj.build_psky_cov()

            # assign the G term from namaster
            for probe_abcd in self.req_probe_combs_2d:
                probe_ab, probe_cd = sl.split_probe_name(probe_abcd, space='harmonic')
                probe_2tpl = (probe_ab, probe_cd)
                self.cov_dict['g'][probe_2tpl]['6d'] = deepcopy(
                    cov_nmt_dict['g'][probe_2tpl]['6d']
                )

            # delete the SVA, SN and MIX terms to avoid confusion, only the g one
            # remains in the partial sky case
            for term in ('sva', 'sn', 'mix'):
                if term in self.cov_dict:
                    del self.cov_dict[term]

        print(f'...done in {(time.perf_counter() - start):.2f} s')

    def _remove_split_terms_from_dict(self, split_gaussian_cov: bool):
        """Helper function to remove the SVA/SN/MIX parts of the G cov if
        split_gaussian_cov is False (i.e., when we don't need the split terms).

        Note: I already remove the sva, sn, mix terms when saving the covs at the end
        of main.py, so strictly speaking this is redundant."""
        if split_gaussian_cov:
            return

        for term in ('sva', 'sn', 'mix'):
            if term in self.cov_dict:
                del self.cov_dict[term]

    def _add_cov_ng(
        self,
        ccl_obj: CCLInterface,
        cov_ssc_obj: SpaceborneSSC,
        cov_oc_obj: OneCovarianceInterface,
    ):
        """Helper function to retrieve the non-Gaussian covariance from the required
        code-specific object.

        Note:  this function needs to assign the cov_dict['ssc'][<probe_2tpl>]['6d']
               6d covariances only, since the reshaping is handled downstream.
               For Spaceborne and PyCCL, the covariance is computed in 4D for
               efficiency, which means that the array has to be reshaped to 6d here.
               OneCovariance, on the other hand, already provides the 6d covariances.
        """

        for ng_term in ['ssc', 'cng']:
            # guards
            if ng_term == 'ssc' and not self.include_ssc:
                print('\nSkipping SSC computation')
                continue
            if ng_term == 'cng' and not self.include_cng:
                print('\nSkipping cNG computation')
                continue

            # set convenience variables
            _cov_ng_code = getattr(self, f'{ng_term}_code')

            # get the relevant dictionary. Note that the structure is still
            # slightly different here
            # TODO homogenize this?
            if _cov_ng_code == 'Spaceborne':
                _cov_ng_dict = cov_ssc_obj.cov_dict[ng_term]
            elif _cov_ng_code == 'PyCCL':
                _cov_ng_dict = ccl_obj.cov_dict[ng_term]

            # in these 2 cases, assign only the 6d covs to self.cov_dict, since the
            # reshaping to 4d and 2d is handled downstream
            if _cov_ng_code in ['Spaceborne', 'PyCCL']:
                for probe_2tpl in self.cov_dict[ng_term]:
                    if probe_2tpl == '3x2pt':
                        continue

                    probe_ab, probe_cd = probe_2tpl

                    # sanity check: no 6d covs should be assigned yet
                    assert self.cov_dict[ng_term][probe_2tpl]['6d'] is None, (
                        f'self.cov_dict[{ng_term}][{probe_2tpl}][6d] is not None '
                        'before assignment!'
                    )

                    self.cov_dict[ng_term][probe_2tpl]['6d'] = sl.cov_4D_to_6D_blocks(
                        cov_4D=_cov_ng_dict[probe_2tpl]['4d'],
                        nbl=self.ell_obj.nbl_3x2pt,
                        zbins=self.zbins,
                        ind_ab=self.ind_dict[probe_ab],
                        ind_cd=self.ind_dict[probe_cd],
                        symmetrize_output_ab=self.symmetrize_output_dict[probe_ab],
                        symmetrize_output_cd=self.symmetrize_output_dict[probe_cd],
                    )

            # in the OneCovariance case, assign the 6d covs directly
            elif _cov_ng_code == 'OneCovariance':
                self.cov_dict[ng_term] = deepcopy(cov_oc_obj.cov_dict[ng_term])
            else:
                raise ValueError(f'Unknown code: {_cov_ng_code}')

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

    def combine_and_reshape_covs(
        self,
        ccl_obj: CCLInterface,
        cov_ssc_obj: SpaceborneSSC | None,
        cov_oc_obj: OneCovarianceInterface | None,
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
        cov_oc_obj : object
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
            # first of all, check that the keys  coincide
            _terms_tocheck = ['g']
            if split_gaussian_cov:
                _terms_tocheck.extend(['sva', 'sn', 'mix'])

            for _term in _terms_tocheck:
                # check terms
                assert _term in self.cov_dict, '_term not in self.cov_dict'
                assert _term in cov_oc_obj.cov_dict, '_term not in cov_oc_obj.cov_dict'

                # check probes
                probe_list_sb = set(self.cov_dict[_term].keys())
                probe_list_oc = set(cov_oc_obj.cov_dict[_term].keys())
                assert probe_list_sb == probe_list_oc, (
                    f'probe_list_sb: {probe_list_sb}, probe_list_oc: {probe_list_oc}'
                )

                # check dims
                for _probe_2tpl in probe_list_sb:
                    dim_list_sb = set(self.cov_dict[_term][_probe_2tpl].keys())
                    dim_list_oc = set(cov_oc_obj.cov_dict[_term][_probe_2tpl].keys())
                    assert dim_list_sb == dim_list_oc, (
                        f'dim_list_sb: {dim_list_sb}, dim_list_oc: {dim_list_oc}'
                    )

            # TODO delete this
            # having checked the covs, overwrite the relevand dict items
            # self.cov_dict['g'] = deepcopy(cov_oc_obj.cov_dict['g'])
            # if split_gaussian_cov:
            #     self.cov_dict['sva'] = deepcopy(cov_oc_obj.cov_dict['sva'])
            #     self.cov_dict['sn'] = deepcopy(cov_oc_obj.cov_dict['sn'])
            #     self.cov_dict['mix'] = deepcopy(cov_oc_obj.cov_dict['mix'])

            for term in self.cov_dict:
                for probe_2tpl in self.cov_dict[term]:
                    for dim in self.cov_dict[term][probe_2tpl]:
                        if self.cov_dict[term][probe_2tpl][dim] is not None:
                            self.cov_dict[term][probe_2tpl][dim] = deepcopy(
                                cov_oc_obj.cov_dict[term][probe_2tpl][dim]
                            )

        # ! reshape and set SSC and cNG - the "if include SSC/cNG"
        # ! are inside the function
        self._add_cov_ng(ccl_obj, cov_ssc_obj, cov_oc_obj)

        for term in self.cov_dict:
            for probe_2tpl in self.cov_dict[term]:
                assert self.cov_dict[term][probe_2tpl]['4d'] is None, (
                    '4d arrays should be empty at this point'
                )
                assert self.cov_dict[term][probe_2tpl]['2d'] is None, (
                    '2d arrays should be empty at this point'
                )

        # ! BNT transform (6/10D covs needed for this implementation)
        if self.cfg['BNT']['cov_BNT_transform']:
            print('BNT-transforming the covariance matrix...')
            start = time.perf_counter()
            self.cov_dict = bnt_utils.bnt_transform_cov_dict(
                self.cov_dict, self.bnt_matrix, self.req_probe_combs_2d
            )
            print(f'...done in {time.perf_counter() - start:.2f} s')

        # ! compute coupled NG cov - the "if coupled" is inside the function
        self._couple_cov_ng()

        # ! reshape probe-specific 6d covs to 4d and 2d
        sl.cov_dict_6d_probe_blocks_to_4d_and_2d(
            cov_dict=self.cov_dict,
            obs_space='harmonic',
            nbx=self.ell_obj.nbl_3x2pt,
            ind_auto=self.ind_auto,
            ind_cross=self.ind_cross,
            zpairs_auto=self.zpairs_auto,
            zpairs_cross=self.zpairs_cross,
            block_index=self.block_index,
        )

        # ! construct 3x2pt 4d and 2d covs (there is no 6d 3x2pt!)
        for term in self.cov_dict:
            if term == 'tot':
                continue  # tot is built at the end, skip it
            self.cov_dict[term]['3x2pt']['2d'] = sl.build_cov_3x2pt_2d(
                self.cov_dict[term], self.cov_ordering_2d, obs_space='harmonic'
            )

        # ! sum g + ssc + cng to get tot (only 2D)
        # this function modifies the cov_dict in place, no need to reassign the result
        # to self.cov_dict
        sl.set_cov_tot_2d_and_6d(
            cov_dict=self.cov_dict,
            req_probe_combs_2d=self.req_probe_combs_2d,
            space='harmonic',
        )

        # ! clean up dictionaries:
        self._remove_split_terms_from_dict(split_gaussian_cov)

        # ! perform ell cuts on the 2D covs
        self._cov_2d_ell_cuts(split_gaussian_cov)

    def _couple_cov_ng(self):
        if not self.cov_cfg['coupled_cov']:
            return

        if self.cfg['BNT']['cov_BNT_transform']:
            warnings.warn(
                'BNT transformation has not been tested for coupled covariance '
                'matrices.',
                stacklevel=2,
            )

        if self.cov_nmt_obj is None:
            raise ValueError(
                'cov_nmt_obj is required when coupled_cov is True. Found None.'
            )

        from spaceborne import cov_partial_sky

        with sl.timer('\nCoupling non-Gaussian covariance matrices...'):
            # construct mcm array for better probe handling (especially for 3x2pt)
            mcm_dict = {}
            mcm_dict['LL'] = self.cov_nmt_obj.mcm_ee_binned
            mcm_dict['GL'] = self.cov_nmt_obj.mcm_te_binned
            # mcm_3x2pt_dict['LG'] = self.cov_nmt_obj.mcm_et_binned
            mcm_dict['GG'] = self.cov_nmt_obj.mcm_tt_binned

            for k, v in mcm_dict.items():
                assert v.shape == (self.ell_obj.nbl_3x2pt, self.ell_obj.nbl_3x2pt), (
                    f'mcm {k} has wrong shape {v.shape}'
                )

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

            for ng_term in ['ssc', 'cng']:
                if ng_term not in self.cov_dict:
                    continue

                for probe_abcd in self.req_probe_combs_2d:
                    probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'harmonic')
                    self.cov_dict[ng_term][probe_ab, probe_cd]['6d'] = (
                        cov_partial_sky.couple_cov_6d(
                            mcm_dict[probe_ab],
                            self.cov_dict[ng_term][probe_ab, probe_cd]['6d'],
                            mcm_dict[probe_cd].T,
                        )
                    )

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
