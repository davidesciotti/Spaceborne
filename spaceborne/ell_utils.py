import warnings
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
    """
    Define a logarithmic ell binning scheme with optional weights.
    Function written by Sylvain Gouyou Beauchamps.

    Parameters
    ----------
    lmin : int
        Minimum ell value for the binning.
    lmax : int
        Maximum ell value for the binning.
    nbl : int
        Number of bins.
    w : array-like, optional
        Weights for the ell values.

    Returns
    -------
    b : nmt.NmtBin
        NaMaster binning object with logarithmic bins.
    """

    import pymaster as nmt

    op = np.log10

    def inv(x):
        return 10**x

    bins = inv(np.linspace(op(lmin), op(lmax + 1), nbl + 1))
    ell = np.arange(lmin, lmax + 1)
    i = np.digitize(ell, bins) - 1
    if w is None:
        w = np.ones(ell.size)
    b = nmt.NmtBin(bpws=i, ells=ell, weights=w, lmax=lmax)
    return b


def get_lmid(ells, k):
    """Returns the effective ell values for the k-th diagonal"""
    return 0.5 * (ells[k:] + ells[:-k])


def load_ell_cuts(
    kmax_h_over_Mpc, z_values_a, z_values_b, cosmo_ccl, zbins, h, kmax_h_over_Mpc_ref
):
    """Loads ell_cut values, rescales them and load into a dictionary.
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


def get_idxs_to_delete_3x2pt(ell_values_3x2pt, ell_cuts_dict, zbins, covariance_cfg):
    """This function tries to implement the indexing for the
    flattening scale_probe_zpair
    """
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


def compute_ells(
    nbl: int,
    ell_min: int,
    ell_max: int,
    binning_type: str,
    output_ell_bin_edges: bool = False,
):
    """Compute the ell values and the bin width\s for a given binning_type.

    Parameters
    ----------
    nbl : int
        Number of ell bins.
    ell_min : int
        Minimum ell value.
    ell_max : int
        Maximum ell value.
    binning_type : str
        Binning type to use. Must be either "ISTF" or "ISTNL".
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
    if binning_type == 'ISTF':
        ell_bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.0
        deltas = np.diff(ell_bin_edges)

    elif binning_type == 'ISTNL':
        ell_bin_edges = np.linspace(np.log(ell_min), np.log(ell_max), nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.0
        ells = np.exp(ells)
        deltas = np.diff(np.exp(ell_bin_edges))

    elif binning_type == 'lin':
        ell_bin_edges = np.linspace(ell_min, ell_max, nbl + 1)
        ells = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.0  # arithmetic mean
        deltas = np.diff(ell_bin_edges)

    elif binning_type == 'log':
        ell_bin_edges = np.geomspace(ell_min, ell_max, nbl + 1)
        ells = np.sqrt(ell_bin_edges[:-1] * ell_bin_edges[1:])  # geometric mean
        deltas = np.diff(ell_bin_edges)

    else:
        raise ValueError('recipe must be either "ISTF", "ISTNL", "lin" or "log"')

    if output_ell_bin_edges:
        return ells, deltas, ell_bin_edges

    return ells, deltas


def compute_ells_oc(
    nbl: int,
    ell_min: int,
    ell_max: int,
    binning_type: str,
    output_ell_bin_edges: bool = False,
):
    """Computes ell grid as done in OneCovariance"""

    if binning_type == 'lin':
        ell_bin_edges = np.linspace(ell_min, ell_max, nbl + 1).astype(int)
        ells = 0.5 * (ell_bin_edges[1:] + ell_bin_edges[:-1])

    elif binning_type == 'log':
        # log-spaced bin edges and geometric mean for the bin centers
        # OC casts the ell bin edges to int
        ell_bin_edges = np.unique(np.geomspace(ell_min, ell_max, nbl + 1).astype(int))
        # this is the geometric mean
        ells = np.exp(0.5 * (np.log(ell_bin_edges[1:]) + np.log(ell_bin_edges[:-1])))

    else:
        raise ValueError('binning_type must be either "lin" or "log"')

    # TODO I think OneCovariance computes this differently
    deltas = np.diff(ell_bin_edges)

    if output_ell_bin_edges:
        return ells, deltas, ell_bin_edges

    return ells, deltas


class EllBinning:
    """Handles the setup of ell bins based on configuration.

    Calculates and stores ell bin centers, edges, and widths for different
    probe combinations (WL, GC, XC, 3x2pt) based on the specified
    binning type and cuts.
    """

    def __init__(self, cfg: dict):
        """Initializes the EllBinning object.

        Args:
            config: The 'ell_binning' section of the main configuration dictionary.

        """
        self.cfg = cfg

        self.binning_type = cfg['binning']['binning_type']
        self.partial_sky_method = cfg['covariance']['partial_sky_method']
        self.do_sample_cov = cfg['sample_covariance']['compute_sample_cov']

        # Only load filenames if using 'from_input' binning type
        if self.binning_type == 'from_input':
            self.wl_bins_filename = cfg['binning']['ell_bins_filename']
            self.gc_bins_filename = cfg['binning']['ell_bins_filename']
        else:
            self.wl_bins_filename = None
            self.gc_bins_filename = None
            # in this case, take ell_min, ell_max, nbl from config
            self.set_ell_min_max_from_cfg(self.cfg)

    def set_ell_min_max_from_cfg(self, cfg):
        self.ell_min_WL = cfg['binning']['ell_min']
        self.ell_max_WL = cfg['binning']['ell_max']
        self.nbl_WL = cfg['binning']['ell_bins']

        self.ell_min_GC = cfg['binning']['ell_min']
        self.ell_max_GC = cfg['binning']['ell_max']
        self.nbl_GC = cfg['binning']['ell_bins']

    def build_ell_bins(self):
        """Builds ell bins based on the specified configuration."""

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
                binning_type='ISTF',
                output_ell_bin_edges=True,
            )

            self.ells_GC, self.delta_l_GC, self.ell_edges_GC = compute_ells(
                nbl=self.nbl_GC,
                ell_min=self.ell_min_GC,
                ell_max=self.ell_max_GC,
                binning_type='ISTF',
                output_ell_bin_edges=True,
            )

        elif self.binning_type == 'lin':
            self.ells_WL, self.delta_l_WL, self.ell_edges_WL = compute_ells(
                nbl=self.nbl_WL,
                ell_min=self.ell_min_WL,
                ell_max=self.ell_max_WL,
                binning_type='lin',
                output_ell_bin_edges=True,
            )

            self.ells_GC, self.delta_l_GC, self.ell_edges_GC = compute_ells(
                nbl=self.nbl_GC,
                ell_min=self.ell_min_GC,
                ell_max=self.ell_max_GC,
                binning_type='lin',
                output_ell_bin_edges=True,
            )

        elif self.binning_type == 'from_input':
            # TODO unify with theta bins!!
            wl_bins_in = np.genfromtxt(self.wl_bins_filename)
            gc_bins_in = np.genfromtxt(self.gc_bins_filename)

            # import ells and edges
            self.ells_WL = wl_bins_in[:, 0]
            self.ells_GC = gc_bins_in[:, 0]
            self.delta_l_WL = wl_bins_in[:, 1]
            self.delta_l_GC = gc_bins_in[:, 1]
            ell_edges_lo_WL = wl_bins_in[:, 2]
            ell_edges_lo_GC = gc_bins_in[:, 2]
            ell_edges_hi_WL = wl_bins_in[:, 3]
            ell_edges_hi_GC = gc_bins_in[:, 3]

            # assign nbl, ell_min and ell_max
            self.nbl_WL = len(self.ells_WL)
            self.nbl_GC = len(self.ells_GC)
            self.ell_min_WL = ell_edges_lo_WL[0]
            self.ell_min_GC = ell_edges_lo_GC[0]
            self.ell_max_WL = ell_edges_hi_WL[-1]
            self.ell_max_GC = ell_edges_hi_GC[-1]

            # sanity check
            if not np.all(ell_edges_lo_WL < ell_edges_hi_WL):
                raise ValueError('All WL bin lower edges must be less than upper edges')
            if not np.all(ell_edges_lo_GC < ell_edges_hi_GC):
                raise ValueError('All GC bin lower edges must be less than upper edges')

            # combine upper and lower bin edges
            self.ell_edges_WL = np.unique(np.append(ell_edges_lo_WL, ell_edges_hi_WL))
            self.ell_edges_GC = np.unique(np.append(ell_edges_lo_GC, ell_edges_hi_GC))

            # sanity check
            if not np.all(np.diff(self.ell_edges_WL) > 0):
                raise ValueError(
                    'WL bin edges must be strictly increasing after combining'
                )
            if not np.all(np.diff(self.ell_edges_GC) > 0):
                raise ValueError(
                    'GC bin edges must be strictly increasing after combining'
                )

        elif self.binning_type == 'ref_cut':
            # TODO this is only done for backwards-compatibility reasons

            ell_min_ref = self.cfg['binning']['ell_min_ref']
            ell_max_ref = self.cfg['binning']['ell_max_ref']
            nbl_ref = self.cfg['binning']['ell_bins_ref']

            self.ells_ref, self.delta_l_ref, self.ell_edges_ref = compute_ells(
                nbl=nbl_ref,
                ell_min=ell_min_ref,
                ell_max=ell_max_ref,
                binning_type='ISTF',
                output_ell_bin_edges=True,
            )

            self.ells_WL = self.ells_ref[self.ells_ref < self.ell_max_WL].copy()
            self.ells_GC = self.ells_ref[self.ells_ref < self.ell_max_GC].copy()

            # TODO why not save all edges??
            # store edges *except last one for dimensional consistency* in the ell_dict
            edge_mask_wl = (self.ell_edges_ref < self.ell_max_WL) | np.isclose(
                self.ell_edges_ref, self.ell_max_WL, atol=0, rtol=1e-5
            )
            edge_mask_gc = (self.ell_edges_ref < self.ell_max_GC) | np.isclose(
                self.ell_edges_ref, self.ell_max_GC, atol=0, rtol=1e-5
            )

            self.ell_edges_WL = self.ell_edges_ref[edge_mask_wl].copy()
            self.ell_edges_GC = self.ell_edges_ref[edge_mask_gc].copy()

            self.delta_l_WL = self.delta_l_ref[: len(self.ells_WL)].copy()
            self.delta_l_GC = self.delta_l_ref[: len(self.ells_GC)].copy()

        else:
            raise ValueError(f'binning_type {self.binning_type} not recognized.')

        if (
            self.cfg['covariance']['partial_sky_method'] == 'NaMaster'
            or self.cfg['sample_covariance']['compute_sample_cov']
        ):
            # TODO what about WL?
            import pymaster as nmt

            warnings.warn(
                'Instantiating namaster bin object from the provided bin edges. '
                'Please make note that in order to do this, these are cast to int.',
                stacklevel=2,
            )

            # this function requires int edges!
            self.ell_edges_WL = self.ell_edges_WL.astype(int)
            self.ell_edges_GC = self.ell_edges_GC.astype(int)

            self.nmt_bin_obj_WL = nmt.NmtBin.from_edges(
                self.ell_edges_WL[:-1], self.ell_edges_WL[1:] + 1
            )
            self.nmt_bin_obj_GC = nmt.NmtBin.from_edges(
                self.ell_edges_GC[:-1], self.ell_edges_GC[1:] + 1
            )

            self.ells_WL = self.nmt_bin_obj_WL.get_effective_ells()
            self.ells_GC = self.nmt_bin_obj_GC.get_effective_ells()

            self.delta_l_WL = np.diff(self.ell_edges_WL)
            self.delta_l_GC = np.diff(self.ell_edges_GC)

            self.ell_min_WL = self.nmt_bin_obj_WL.get_ell_min(0)
            self.ell_min_GC = self.nmt_bin_obj_GC.get_ell_min(0)
            self.ell_max_WL = self.nmt_bin_obj_WL.lmax
            self.ell_max_GC = self.nmt_bin_obj_GC.lmax

            # test that ell_max retrieved with the two methods coincide
            assert self.nmt_bin_obj_WL.lmax == self.nmt_bin_obj_WL.get_ell_max(
                self.nbl_WL - 1
            ), 'ell_max from nmt_bin_obj_WL does not match ell_max from get_ell_max'
            assert self.nmt_bin_obj_GC.lmax == self.nmt_bin_obj_GC.get_ell_max(
                self.nbl_GC - 1
            ), 'ell_max from nmt_bin_obj_GC does not match ell_max from get_ell_max'

        # XC follows GC
        self.ells_XC = self.ells_GC.copy()
        self.ell_edges_XC = self.ell_edges_GC.copy()
        self.delta_l_XC = self.delta_l_GC.copy()
        self.ell_min_XC = self.ell_min_GC
        self.ell_max_XC = self.ell_max_GC

        # 3x2pt as well
        # TODO change this to be more general
        self.ells_3x2pt = self.ells_GC.copy()
        self.ell_edges_3x2pt = self.ell_edges_GC.copy()
        self.delta_l_3x2pt = self.delta_l_GC.copy()
        self.ell_min_3x2pt = self.ell_min_GC
        self.ell_max_3x2pt = self.ell_max_GC

        # set nbl
        self.nbl_WL = len(self.ells_WL)
        self.nbl_GC = len(self.ells_GC)
        self.nbl_XC = len(self.ells_XC)
        self.nbl_3x2pt = len(self.ells_3x2pt)

    def compute_ells_3x2pt_unbinned(self):
        """Needed for the partial-sky covariance"""
        self.ells_3x2pt_unb = np.arange(self.ell_max_3x2pt + 1)
        self.nbl_3x2pt_unb = len(self.ells_3x2pt_unb)
        self.ell_max_3x2pt_unb = self.ells_3x2pt_unb[-1]
        assert self.nbl_3x2pt_unb == self.ell_max_3x2pt + 1, (
            'nbl_tot does not match ell_max_3x2pt + 1'
        )

    def compute_ells_3x2pt_proj(self):
        """Needed for the projection of the harmonic-space covariance to 
        theta/COSEBIs space"""
        self.ells_3x2pt_proj = np.geomspace(
            self.cfg['precision']['ell_min_proj'],
            self.cfg['precision']['ell_max_proj'],
            self.cfg['precision']['ell_bins_proj'],
        )
        self.ells_3x2pt_proj_ng = np.geomspace(
            self.cfg['precision']['ell_min_proj'],
            self.cfg['precision']['ell_max_proj'],
            self.cfg['precision']['ell_bins_proj_nongauss'],
        )
        # these are probably useless, but just to keep consistency
        self.nbl_3x2pt_proj = len(self.ells_3x2pt_proj)
        self.nbl_3x2pt_proj_ng = len(self.ells_3x2pt_proj_ng)

    def _validate_bins(self):
        for probe in ['GC', 'XC', '3x2pt']:
            ells_probe = getattr(self, f'ells_{probe}')
            np.testing.assert_allclose(
                ells_probe,
                self.ells_WL,
                err_msg='for the moment, ell binning should be the same for all probes',
                atol=0,
                rtol=1e-5,
            )

        for probe in ['WL', 'GC', 'XC', '3x2pt']:
            ells = getattr(self, f'ells_{probe}')

            if ells is None or ells.size == 0:
                raise ValueError(f'ell values for probe {probe} are empty.')

            if not isinstance(ells, np.ndarray):
                raise TypeError(
                    f'ell values for probe {probe} must be a numpy array, '
                    f'got {type(ells)} instead.'
                )

            if ells.ndim != 1:
                raise ValueError(
                    f'ell values for probe {probe} must be a 1D array, '
                    f'got {ells.ndim}D array.'
                )

            if not np.all(np.isfinite(ells)):
                raise ValueError(f'ell values for probe {probe} contain NaN or Inf.')

            if not np.all(ells >= 0):
                raise ValueError(f'ell values for probe {probe} must be non-negative.')

            if not np.all(np.diff(ells) >= 0):
                raise ValueError(
                    f'ell values for probe {probe} must be sorted in '
                    'non-decreasing order.'
                )

            if not np.issubdtype(ells.dtype, np.number):
                raise TypeError(
                    f'ell values for probe {probe} must be of numeric type.'
                )
