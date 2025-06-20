import numpy as np

from spaceborne import cosmo_lib


def nmt_linear_binning(lmin, lmax, bw, w=None):
    import pymaster as nmt

    nbl = (lmax - lmin) // bw + 1
    bins = np.linspace(lmin, lmax + 1, nbl + 1)
    ell = np.arange(lmin, lmax + 1)
    i = np.digitize(ell, bins) - 1
    b = nmt.NmtBin(bpws=i, ells=ell, weights=w, lmax=lmax)

    return b


def nmt_log_binning(lmin, lmax, nbl, w=None):
    import pymaster as nmt

    op = np.log10

    def inv(x):
        return 10**x

    bins = inv(np.linspace(op(lmin), op(lmax + 1), nbl + 1))
    ell = np.arange(lmin, lmax + 1)
    i = np.digitize(ell, bins) - 1
    b = nmt.NmtBin(bpws=i, ells=ell, weights=w, lmax=lmax)

    return b


def get_lmid(ells, k):
    """returns the effective ell values for the k-th diagonal"""
    return 0.5 * (ells[k:] + ells[:-k])


def load_ell_cuts(
    kmax_h_over_Mpc, z_values_a, z_values_b, cosmo_ccl, zbins, h, kmax_h_over_Mpc_ref
):
    """loads ell_cut values, rescales them and load into a dictionary.
    z_values_a: redshifts at which to compute the ell_max for a given Limber
    wavenumber, for probe A
    z_values_b: redshifts at which to compute the ell_max for a given Limber
    wavenumber, for probe B
    """

    if kmax_h_over_Mpc is None:
        kmax_h_over_Mpc = kmax_h_over_Mpc_ref

    kmax_1_over_Mpc = kmax_h_over_Mpc * h

    ell_cuts_array = np.zeros((zbins, zbins))
    for zi, zval_i in enumerate(z_values_a):
        for zj, zval_j in enumerate(z_values_b):
            r_of_zi = cosmo_lib.ccl_comoving_distance(
                zval_i, use_h_units=False, cosmo_ccl=cosmo_ccl
            )
            r_of_zj = cosmo_lib.ccl_comoving_distance(
                zval_j, use_h_units=False, cosmo_ccl=cosmo_ccl
            )
            ell_cut_i = kmax_1_over_Mpc * r_of_zi - 1 / 2
            ell_cut_j = kmax_1_over_Mpc * r_of_zj - 1 / 2
            ell_cuts_array[zi, zj] = np.min((ell_cut_i, ell_cut_j))

    return ell_cuts_array


def get_idxs_to_delete(ell_values, ell_cuts, is_auto_spectrum, zbins):
    """ell_values can be the bin center or the bin lower edge; Francis
    suggests the second option is better"""

    if is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_val in ell_values:
            for zi in range(zbins):
                for zj in range(zi, zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1

    elif not is_auto_spectrum:
        idxs_to_delete = []
        count = 0
        for ell_val in ell_values:
            for zi in range(zbins):
                for zj in range(zbins):
                    if ell_val > ell_cuts[zi, zj]:
                        idxs_to_delete.append(count)
                    count += 1
    else:
        raise ValueError('is_auto_spectrum must be True or False')

    return idxs_to_delete


def get_idxs_to_delete_3x2pt(ell_values_3x2pt, ell_cuts_dict, zbins, covariance_cfg):
    """this function tries to implement the indexing for the
    flattening ell_probe_zpair"""

    if (covariance_cfg['triu_tril'], covariance_cfg['row_col_major']) != (
        'triu',
        'row-major',
    ):
        raise Exception(
            'This function is only implemented for the triu, row-major case'
        )

    idxs_to_delete_3x2pt = []
    count = 0
    for ell_val in ell_values_3x2pt:
        for zi in range(zbins):
            for zj in range(zi, zbins):
                if ell_val > ell_cuts_dict['LL'][zi, zj]:
                    idxs_to_delete_3x2pt.append(count)
                count += 1
        for zi in range(zbins):
            for zj in range(zbins):
                if ell_val > ell_cuts_dict['GL'][zi, zj]:
                    idxs_to_delete_3x2pt.append(count)
                count += 1
        for zi in range(zbins):
            for zj in range(zi, zbins):
                if ell_val > ell_cuts_dict['GG'][zi, zj]:
                    idxs_to_delete_3x2pt.append(count)
                count += 1

    # check if the array is monotonically increasing
    assert np.all(np.diff(idxs_to_delete_3x2pt) > 0)

    return list(idxs_to_delete_3x2pt)


def get_idxs_to_delete_3x2pt_v0(
    ell_values_3x2pt, ell_cuts_dict, nbl_3x2pt, zpairs_auto, zpairs_cross
):
    """this implements the indexing for the flattening probe_ell_zpair"""
    raise Exception(
        'Concatenation must be done *before* flattening, this function '
        'is not compatible with the '
        '"ell-block ordering of the covariance matrix"'
    )
    idxs_to_delete_LL = get_idxs_to_delete(
        ell_values_3x2pt, ell_cuts_dict['LL'], is_auto_spectrum=True
    )
    idxs_to_delete_GL = get_idxs_to_delete(
        ell_values_3x2pt, ell_cuts_dict['GL'], is_auto_spectrum=False
    )
    idxs_to_delete_GG = get_idxs_to_delete(
        ell_values_3x2pt, ell_cuts_dict['GG'], is_auto_spectrum=True
    )

    # when concatenating, we need to add the offset from the
    # stacking of the 3 datavectors
    idxs_to_delete_3x2pt = np.concatenate(
        (
            np.array(idxs_to_delete_LL),
            nbl_3x2pt * zpairs_auto + np.array(idxs_to_delete_GL),
            nbl_3x2pt * (zpairs_auto + zpairs_cross) + np.array(idxs_to_delete_GG),
        )
    )

    # check if the array is monotonically increasing
    assert np.all(np.diff(idxs_to_delete_3x2pt) > 0)

    return list(idxs_to_delete_3x2pt)


def generate_ell_and_deltas(general_config):
    """old function, but useful to compute ell and delta_ell for Wadd!"""
    nbl_WL = general_config['nbl_WL']
    nbl_GC = general_config['nbl_GC']
    assert nbl_WL == nbl_GC, 'nbl_WL and nbl_GC must be the same'
    nbl = nbl_WL

    ell_min = general_config['ell_min']
    ell_max_WL = general_config['ell_max_WL']
    ell_max_GC = general_config['ell_max_GC']
    ell_max_3x2pt = general_config['ell_max_3x2pt']
    use_WA = general_config['use_WA']

    ell_dict = {}
    delta_dict = {}

    # XC has the same ell values as GC
    # ell_max_XC = ell_max_GC
    # ell_max_WA = ell_max_XC

    # creating nbl ell values logarithmically equi-spaced between 10 and ell_max
    ell_WL = np.logspace(np.log10(ell_min), np.log10(ell_max_WL), nbl + 1)  # WL
    ell_GC = np.logspace(np.log10(ell_min), np.log10(ell_max_GC), nbl + 1)  # GC
    ell_3x2pt = np.logspace(
        np.log10(ell_min), np.log10(ell_max_3x2pt), nbl + 1
    )  # 3x2pt

    # central values of each bin
    l_centr_WL = (ell_WL[1:] + ell_WL[:-1]) / 2
    l_centr_GC = (ell_GC[1:] + ell_GC[:-1]) / 2
    l_centr_3x2pt = (ell_3x2pt[1:] + ell_3x2pt[:-1]) / 2

    # automatically compute ell_WA
    if use_WA:
        ell_WA = np.log10(np.asarray(l_centr_WL[np.where(l_centr_WL > ell_max_3x2pt)]))
    # FIXME: this is a very bad way to implement use_WA = False. I'm computing it anyway
    # for some random values.
    else:
        ell_WA = np.log10(
            np.asarray(l_centr_WL[np.where(l_centr_WL > ell_max_3x2pt / 2)])
        )
    nbl_WA = ell_WA.shape[0]

    # generate the deltas
    delta_l_WL = np.diff(ell_WL)
    delta_l_GC = np.diff(ell_GC)
    delta_l_3x2pt = np.diff(ell_3x2pt)
    delta_l_WA = np.diff(ell_WL)[-nbl_WA:]  # take only the last nbl_WA (e.g. 4) values

    # take the log10 of the values
    logarithm_WL = np.log10(l_centr_WL)
    logarithm_GC = np.log10(l_centr_GC)
    logarithm_3x2pt = np.log10(l_centr_3x2pt)

    # update the ell_WL, ell_GC arrays with the right values
    ell_WL = logarithm_WL
    ell_GC = logarithm_GC
    ell_3x2pt = logarithm_3x2pt

    if use_WA and np.any(l_centr_WL == ell_max_GC):
        # check in the unlikely case that one element of l_centr_WL is == ell_max_GC.
        # Anyway, the recipe
        # says (l_centr_WL > ell_max_GC, NOT >=).
        print(
            'warning: one element of l_centr_WL is == ell_max_GC; the recipe says '
            'to take only the elements >, but you may want to double check what '
            'to do in this case'
        )

    # save the values
    ell_dict['ell_WL'] = 10**ell_WL
    ell_dict['ell_GC'] = 10**ell_GC
    ell_dict['ell_WA'] = 10**ell_WA
    ell_dict['ell_3x2pt'] = 10**ell_3x2pt

    delta_dict['delta_l_WL'] = delta_l_WL
    delta_dict['delta_l_GC'] = delta_l_GC
    delta_dict['delta_l_WA'] = delta_l_WA
    delta_dict['delta_l_3x2pt'] = delta_l_3x2pt

    return ell_dict, delta_dict


def compute_ells(
    nbl: int, ell_min: int, ell_max: int, recipe, output_ell_bin_edges: bool = False
):
    """Compute the ell values and the bin widths for a given recipe.

    Parameters
    ----------
    nbl : int
        Number of ell bins.
    ell_min : int
        Minimum ell value.
    ell_max : int
        Maximum ell value.
    recipe : str
        Recipe to use. Must be either "ISTF" or "ISTNL".
    output_ell_bin_edges : bool, optional
        If True, also return the ell bin edges, by default False

    Returns
    -------
    ells : np.ndarray
        Central ell values.
    deltas : np.ndarray
        Bin widths
    ell_bin_edges : np.ndarray, optional
        ell bin edges. Returned only if output_ell_bin_edges is True.
    """
    if recipe == 'ISTF':
        ell_bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
        ells = (ell_bin_edges[1:] + ell_bin_edges[:-1]) / 2.0
        deltas = np.diff(ell_bin_edges)

    elif recipe == 'ISTNL':
        ell_bin_edges = np.linspace(np.log(ell_min), np.log(ell_max), nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.0
        ells = np.exp(ells)
        deltas = np.diff(np.exp(ell_bin_edges))

    elif recipe == 'lin':
        ell_bin_edges = np.linspace(ell_min, ell_max, nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.0
        deltas = np.diff(ell_bin_edges)

    else:
        raise ValueError('recipe must be either "ISTF" or "ISTNL" or "lin"')

    if output_ell_bin_edges:
        return ells, deltas, ell_bin_edges

    return ells, deltas


class EllBinning:
    """
    Handles the setup of ell bins based on configuration.

    Calculates and stores ell bin centers, edges, and widths for different
    probe combinations (WL, GC, XC, 3x2pt) based on the specified
    binning type and cuts.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the EllBinning object.

        Args:
            config: The 'ell_binning' section of the main configuration dictionary.
        """
        self.binning_type = cfg['ell_binning']['binning_type']

        self.ell_min_WL = cfg['ell_binning']['ell_min_WL']
        self.ell_max_WL = cfg['ell_binning']['ell_max_WL']
        self.nbl_WL = cfg['ell_binning']['ell_bins_WL']

        self.ell_min_GC = cfg['ell_binning']['ell_min_GC']
        self.ell_max_GC = cfg['ell_binning']['ell_max_GC']
        self.nbl_GC = cfg['ell_binning']['ell_bins_GC']

        self.ell_min_ref = cfg['ell_binning']['ell_min_ref']
        self.ell_max_ref = cfg['ell_binning']['ell_max_ref']
        self.nbl_ref = cfg['ell_binning']['ell_bins_ref']

        self.use_namaster = cfg['namaster']['use_namaster']
        self.do_sample_cov = cfg['sample_covariance']['compute_sample_cov']

    def build_ell_bins(self):
        """
        Builds ell bins based on the specified configuration.
        """

        # if self.use_namaster:
        #     # 1. instantiate nmt bin object
        #     if self.binning_type == 'lin':
        #         self.nmt_bin_obj = nmt_linear_binning(
        #             lmin=self.ell_min_GC, lmax=self.ell_max_GC, bw=self.ells_per_band
        #         )
        #     elif self.binning_type == 'log':
        #         self.nmt_bin_obj = nmt_log_binning(
        #             lmin=self.ell_min_GC, lmax=self.ell_max_GC, nbl=self.nbl_GC
        #         )
        #     else:
        #         raise ValueError('binning_type must be either "lin" or "log"')

        #     # 2. get el binning details: ells, deltas, edges
        #     self.ells_eff = (
        #         self.nmt_bin_obj.get_effective_ells()
        #     )  # effective ells per bandpower
        #     self.nbl_eff = len(self.ells_eff)

        #     # notice that bin_obj.get_ell_list(nbl_eff) is out of bounds
        #     self.ells_eff_edges = np.array(
        #         [self.nmt_bin_obj.get_ell_list(i)[0] for i in range(self.nbl_eff)]
        #     )
        #     self.ells_eff_edges = np.append(
        #         self.ells_eff_edges,
        #         self.nmt_bin_obj.get_ell_list(self.nbl_eff - 1)[-1] + 1,
        #     )  # careful f the +1!
        #     self.lmin_eff = self.ells_eff_edges[0]
        #     self.lmax_eff = self.nmt_bin_obj.lmax

        #     self.delta_ells_eff = np.diff(self.ells_eff_edges)

        #     # TODO test this again?
        #     # if self.binning_type == 'lin':
        #     #     assert np.all(self.delta_ells_eff == self.ells_per_band), (
        #     #         'delta_ell from bpw does not match ells_per_band'
        #     #     )

        #     # ells_bpw = ells_unb[lmin_eff : lmax_eff + 1]
        #     # delta_ells_bpw = np.diff(
        #     # np.array(
        #     # [self.nmt_bin_obj.get_ell_list(i)[0] for i in range(self.nbl_eff)]
        #     # )
        #     # )

        if self.binning_type == 'unbinned':
            self.ells_WL = np.arange(self.ell_min_WL, self.ell_max_WL + 1)
            self.ells_GC = np.arange(self.ell_min_GC, self.ell_max_GC + 1)

            self.delta_l_WL = np.ones_like(self.ells_WL)
            self.delta_l_GC = np.ones_like(self.ells_GC)

            # TODO this is a bit sloppy, but it's never used
            self.ell_edges_WL = np.arange(self.ell_min_WL, self.ell_max_WL + 2)
            self.ell_edges_GC = np.arange(self.ell_min_GC, self.ell_max_GC + 2)

        elif self.binning_type == 'log':
            self.ells_WL, self.delta_l_WL, self.ell_edges_WL = compute_ells(
                nbl=self.nbl_WL,
                ell_min=self.ell_min_WL,
                ell_max=self.ell_max_WL,
                recipe='ISTF',
                output_ell_bin_edges=True,
            )

            self.ells_GC, self.delta_l_GC, self.ell_edges_GC = compute_ells(
                nbl=self.nbl_GC,
                ell_min=self.ell_min_GC,
                ell_max=self.ell_max_GC,
                recipe='ISTF',
                output_ell_bin_edges=True,
            )

        elif self.binning_type == 'lin':
            self.ells_WL, self.delta_l_WL, self.ell_edges_WL = compute_ells(
                nbl=self.nbl_WL,
                ell_min=self.ell_min_WL,
                ell_max=self.ell_max_WL,
                recipe='lin',
                output_ell_bin_edges=True,
            )

            self.ells_GC, self.delta_l_GC, self.ell_edges_GC = compute_ells(
                nbl=self.nbl_GC,
                ell_min=self.ell_min_GC,
                ell_max=self.ell_max_GC,
                recipe='lin',
                output_ell_bin_edges=True,
            )

        elif self.binning_type == 'ref_cut':
            # TODO this is only done for backwards-compatibility reasons
            self.ells_ref, self.delta_l_ref, self.ell_edges_ref = compute_ells(
                nbl=self.nbl_ref,
                ell_min=self.ell_min_ref,
                ell_max=self.ell_max_ref,
                recipe='ISTF',
                output_ell_bin_edges=True,
            )

            self.ells_WL = np.copy(self.ells_ref[self.ells_ref < self.ell_max_WL])
            self.ells_GC = np.copy(self.ells_ref[self.ells_ref < self.ell_max_GC])

            # TODO why not save all edges??
            # store edges *except last one for dimensional consistency* in the ell_dict
            edge_mask_wl = (self.ell_edges_ref < self.ell_max_WL) | np.isclose(
                self.ell_edges_ref, self.ell_max_WL, atol=0, rtol=1e-5
            )
            edge_mask_gc = (self.ell_edges_ref < self.ell_max_GC) | np.isclose(
                self.ell_edges_ref, self.ell_max_GC, atol=0, rtol=1e-5
            )

            self.ell_edges_WL = np.copy(self.ell_edges_ref[edge_mask_wl])
            self.ell_edges_GC = np.copy(self.ell_edges_ref[edge_mask_gc])

            self.delta_l_WL = np.copy(self.delta_l_ref[: len(self.ells_WL)])
            self.delta_l_GC = np.copy(self.delta_l_ref[: len(self.ells_GC)])

        else:
            raise ValueError(f'binning_type {self.binning_type} not recognized.')

        if self.use_namaster or self.do_sample_cov:
            # TODO what about WL?
            import pymaster as nmt

            # this function requires int edges!
            self.ell_edges_WL = self.ell_edges_WL.astype(int)
            self.ell_edges_GC = self.ell_edges_GC.astype(int)

            self.nmt_bin_obj_WL = nmt.NmtBin.from_edges(
                self.ell_edges_WL[:-1], self.ell_edges_WL[1:]
            )
            self.nmt_bin_obj_GC = nmt.NmtBin.from_edges(
                self.ell_edges_GC[:-1], self.ell_edges_GC[1:]
            )

            self.ells_WL = self.nmt_bin_obj_WL.get_effective_ells()
            self.ells_GC = self.nmt_bin_obj_GC.get_effective_ells()

            self.delta_l_WL = np.diff(self.ell_edges_WL)
            self.delta_l_GC = np.diff(self.ell_edges_GC)

            self.ell_min_WL = self.nmt_bin_obj_WL.get_ell_min(0)
            self.ell_max_WL = self.nmt_bin_obj_WL.lmax
            self.ell_min_GC = self.nmt_bin_obj_GC.get_ell_min(0)
            self.ell_max_GC = self.nmt_bin_obj_GC.lmax

            # test that ell_max retrieved with the two methods coincide
            assert self.nmt_bin_obj_WL.lmax == self.nmt_bin_obj_WL.get_ell_max(
                self.nbl_WL - 1
            ), 'ell_max from nmt_bin_obj_WL does not match ell_max from get_ell_max'
            assert self.nmt_bin_obj_GC.lmax == self.nmt_bin_obj_GC.get_ell_max(
                self.nbl_GC - 1
            ), 'ell_max from nmt_bin_obj_GC does not match ell_max from get_ell_max'

        # XC follows GC
        self.ells_XC = np.copy(self.ells_GC)
        self.ell_edges_XC = np.copy(self.ell_edges_GC)
        self.delta_l_XC = np.copy(self.delta_l_GC)
        self.ell_min_XC = np.copy(self.ell_min_GC)
        self.ell_max_XC = np.copy(self.ell_max_GC)

        # 3x2pt as well
        # TODO change this to be more general
        self.ells_3x2pt = np.copy(self.ells_GC)
        self.ell_edges_3x2pt = np.copy(self.ell_edges_GC)
        self.delta_l_3x2pt = np.copy(self.delta_l_GC)
        self.ell_min_3x2pt = np.copy(self.ell_min_GC)
        self.ell_max_3x2pt = np.copy(self.ell_max_GC)

        # set nbl
        self.nbl_WL = len(self.ells_WL)
        self.nbl_GC = len(self.ells_GC)
        self.nbl_XC = len(self.ells_XC)
        self.nbl_3x2pt = len(self.ells_3x2pt)

    def compute_ells_3x2pt_unbinned(self):
        # recompute Cls ell by ell
        self.ells_3x2pt_unb = np.arange(self.ell_max_3x2pt + 1)
        self.nbl_3x2pt_unb = len(self.ells_3x2pt_unb)
        self.ell_max_3x2pt_unb = self.ells_3x2pt_unb[-1]
        assert self.nbl_3x2pt_unb == self.ell_max_3x2pt + 1, (
            'nbl_tot does not match ell_max_3x2pt + 1'
        )

    def _validate_bins(self):
        for probe in ['WL', 'GC', 'XC', '3x2pt']:
            ells = getattr(self, f'ells_{probe}')

            if ells is None or ells.size == 0:
                raise ValueError(f'ell values for probe {probe} are empty.')

            if not isinstance(ells, np.ndarray):
                raise TypeError(
                    f'ell values for probe {probe} must be a numpy array, got {type(ells)} instead.'
                )

            if ells.ndim != 1:
                raise ValueError(
                    f'ell values for probe {probe} must be a 1D array, got {ells.ndim}D array.'
                )

            if not np.all(np.isfinite(ells)):
                raise ValueError(f'ell values for probe {probe} contain NaN or Inf.')

            if not np.all(ells >= 0):
                raise ValueError(f'ell values for probe {probe} must be non-negative.')

            if not np.all(np.diff(ells) >= 0):
                raise ValueError(
                    f'ell values for probe {probe} must be sorted in non-decreasing order.'
                )

            if not np.issubdtype(ells.dtype, np.number):
                raise TypeError(
                    f'ell values for probe {probe} must be of numeric type.'
                )
