"""Class for reading in data in various formats"""

import itertools

import numpy as np

from spaceborne import constants as const


def load_nz_euclidlib(nz_filename):
    """basically, this function turns the nz dict into a np array"""
    import euclidlib as el

    z, nz = el.photo.redshift_distributions(nz_filename)
    nztab = np.zeros((len(z), len(nz)))
    for zi in nz:
        nztab[:, zi - 1] = nz[zi]  # array is 0-based, dict is 1-based
    return z, nztab


def import_cl_tab(cl_tab_in: np.ndarray):
    assert cl_tab_in.shape[1] == 4, 'input cls should have 4 columns'
    assert np.min(cl_tab_in[:, 1]) == 0, (
        'tomographic redshift indices should start from 0'
    )
    assert np.min(cl_tab_in[:, 2]) == 0, (
        'tomographic redshift indices should start from 0'
    )
    assert np.max(cl_tab_in[:, 1]) == np.max(cl_tab_in[:, 2]), (
        'tomographic redshift indices should be \
        the same for both z_i and z_j'
    )

    zbins = int(np.max(cl_tab_in[:, 1]) + 1)
    ell_values = np.unique(cl_tab_in[:, 0])

    cl_3d = np.zeros((len(ell_values), zbins, zbins))

    for row in range(cl_tab_in.shape[0]):
        ell_val, zi, zj = (
            cl_tab_in[row, 0],
            int(cl_tab_in[row, 1]),
            int(cl_tab_in[row, 2]),
        )

        ell_ix = np.where(np.isclose(ell_values, ell_val, atol=0, rtol=1e-4))[0][0]

        cl_3d[ell_ix, zi, zj] = cl_tab_in[row, 3]

    return ell_values, cl_3d


def check_cl_symm(cl_3d):
    """To check that the input auto-cls are symmetric"""
    for ell in range(cl_3d.shape[0]):
        np.testing.assert_allclose(
            cl_3d[ell],
            cl_3d[ell].T,
            atol=0,
            rtol=1e-4,
            err_msg='cl_3d is not symmetric',
        )


def load_cl_euclidlib(filename, key_a, key_b):
    import euclidlib as el

    # checks
    assert filename.endswith('.fits'), 'filename must end with .fits'
    assert key_a in ['SHE', 'POS'], 'key_a must be "SHE" or "POS"'
    assert key_b in ['SHE', 'POS'], 'key_b must be "SHE" or "POS"'

    is_auto_spectrum = key_a == key_b

    # import .fits using euclidlib
    # TODO the bare `except` is horrible but I'm not yet sure what's the definitive
    #      structure of the euclidlib import
    try:
        cl_dict = el.photo.harmonic_space.angular_power_spectra(filename)
    except:
        cl_dict = el.photo.angular_power_spectra(filename)

    # extract ells
    ells = cl_dict[key_a, key_b, 1, 1].ell

    nbl = ells.size

    # extract zbins (check consistency of columns first)
    zbins_i = max(i[2] for i in cl_dict)
    zbins_j = max(i[3] for i in cl_dict)
    assert zbins_i == zbins_j, 'zbins are not the same for all columns'
    zbins = zbins_i

    # check that they match no matter the redshift bin combination
    triu_ix = np.triu_indices(zbins)

    idxs = (
        zip(*triu_ix, strict=True)
        if is_auto_spectrum
        else itertools.product(range(zbins), range(zbins))
    )

    for zi, zj in idxs:
        assert np.all(ells == cl_dict[key_a, key_b, zi + 1, zj + 1].ell), (
            'ells are not the same for (zi, zj) combinations'
        )

    # populate 3D array
    cl_3d = np.zeros((nbl, zbins, zbins))
    for zi, zj in itertools.product(range(zbins), range(zbins)):
        args = (cl_dict, key_a, key_b, zi + 1, zj + 1)
        if zj >= zi:
            cl_3d[:, zi, zj] = _select_spin_component(*args)
        elif is_auto_spectrum:
            cl_3d[:, zi, zj] = cl_3d[:, zj, zi]
        else:
            cl_3d[:, zi, zj] = _select_spin_component(*args)

    return ells, cl_3d


def _select_spin_component(cl_dict, key_a, key_b, ziplus1, zjplus1):
    """Selects the spin components, aka homogenises the dimensions to assign data to
    cl_3d.
    Important note: E-modes are hardcoded at the moment;
    index 1 is for B modes, but you would have to change the structure of the cl_5d
    array (at the moment it's:
    cl_5d[0, 0, ...] = SHE_E x SHE_E
    cl_5d[1, 0, ...] = POS   x SHE_E
    cl_5d[0, 1, ...] = SHE_E x POS
    cl_5d[1, 1, ...] = POS   x POS
    BUT: Theory B modes should always be 0...
    """
    cl_array = cl_dict[(key_a, key_b, ziplus1, zjplus1)].array

    # in case there are no B modes, e.g. in the input spectra passed by Guada
    if cl_array.ndim == 1:
        return cl_array

    if key_a == 'POS' and key_b == 'POS':
        return cl_array  # POS x POS
    elif (key_a == 'POS' and key_b == 'SHE') or (key_a == 'SHE' and key_b == 'POS'):
        return cl_array[0]  # POS × E
    elif key_a == 'SHE' and key_b == 'SHE':
        return cl_array[0][0]  # E × E
    else:
        raise ValueError(f'Unexpected probe combination: {key_a}, {key_b}')


def cov_sb_10d_to_heracles_dict(cov_10d, squeeze):
    """SB = 'Spaceborne'
    HC = 'Heracles'

    this dictionary specifies, within the 2 axes assigned to SHE, which ones
    correspond to the E and B modes. This is not used since the analytical covariance
    has no B modes.
    This is also the reason why, regardless of probe and spin, the values are
    stored in the 0-th index, i.e. arr_out[0, 0, 0, 0, :, :]
    she_spin_dict = {
        'E': 0,
        'B': 1,
    }
    """

    from heracles.result import Result

    # sanity checks
    assert cov_10d.ndim == 10, 'input covariance is not 10-dimensional'
    assert (
        cov_10d.shape[0] == cov_10d.shape[1] == cov_10d.shape[2] == cov_10d.shape[3]
    ), "The dimensions of the first 4 axes don't match"
    assert cov_10d.shape[4] == cov_10d.shape[5], (
        "The dimensions of the first 5th and 6th axes don't match"
    )
    assert (
        cov_10d.shape[6] == cov_10d.shape[7] == cov_10d.shape[8] == cov_10d.shape[9]
    ), "The dimensions of the last 4 axes don't match"

    # prints
    n_probes = cov_10d.shape[0]
    zbins = cov_10d.shape[-1]
    nbl = cov_10d.shape[4]

    cov_dict = {}

    for probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix in itertools.product(
        range(n_probes), repeat=4
    ):
        for zi, zj, zk, zl in itertools.product(range(zbins), repeat=4):
            # get probe name
            probe_a_str = const.HS_PROBE_IX_TO_NAME_DICT_HERACLES[probe_a_ix]
            probe_b_str = const.HS_PROBE_IX_TO_NAME_DICT_HERACLES[probe_b_ix]
            probe_c_str = const.HS_PROBE_IX_TO_NAME_DICT_HERACLES[probe_c_ix]
            probe_d_str = const.HS_PROBE_IX_TO_NAME_DICT_HERACLES[probe_d_ix]

            # get probe dimensions
            probe_a_dims = const.HS_PROBE_DIMS_DICT_HERACLES[probe_a_str]
            probe_b_dims = const.HS_PROBE_DIMS_DICT_HERACLES[probe_b_str]
            probe_c_dims = const.HS_PROBE_DIMS_DICT_HERACLES[probe_c_str]
            probe_d_dims = const.HS_PROBE_DIMS_DICT_HERACLES[probe_d_str]

            probe_str_list = [probe_a_str, probe_b_str, probe_c_str, probe_d_str]

            # instantiate array with the 4 additional axes for the spins
            arr_out = np.zeros(
                shape=(probe_a_dims, probe_b_dims, probe_c_dims, probe_d_dims, nbl, nbl)
            )

            # since only SHE_B goes in the 1 index, all ell1, ell2 arrays are stored
            # in the 0 index
            arr_out[0, 0, 0, 0, :, :] = cov_10d[
                probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                :, :, zi, zj, zk, zl,
            ]  # fmt: skip

            if squeeze:
                # Remove singleton dimensions if required
                arr_out = np.squeeze(arr_out)

                # Now pass the axes of the ell1, ell2 indices to the heracles Result
                # class. Since we are removing the singleton dimensions, we need to
                # find the index of the first "surviving" axis
                # (i.e., the first axis with dimension > 1, i.e., SHE).
                # Then, the second one will just be the next index (we never deal
                # with more complicated cases here)

                # Jaime's example:
                # cov_dict = \
                # {('POS', 'POS', 'POS', 'POS', 1, 1, 1, 1): Result(axis=(0, 1)),
                # ('POS', 'POS', 'POS', 'SHE', 1, 1, 1, 1): Result(axis=(1, 2)),
                # ('POS', 'POS', 'SHE', 'SHE', 1, 1, 1, 1): Result(axis=(2, 3)),
                # ('POS', 'SHE', 'SHE', 'SHE', 1, 1, 1, 1): Result(axis=(3, 4)),
                # ('SHE', 'SHE', 'SHE', 'SHE', 1, 1, 1, 1): Result(axis=(4, 5))}

                ax1 = probe_str_list.count('SHE')

            else:
                # If the singleton dimensions are not removed, the ell1, ell2
                # axes are always the last two, after the 4 spin axes
                ax1 = len(probe_str_list)

            ax2 = ax1 + 1

            # old
            # cov_dict[
            #     (probe_a_str, probe_b_str,
            #     probe_c_str, probe_d_str,
            #     zi, zj, zk, zl)
            # ] = arr_out

            # new
            cov_dict[
                (probe_a_str, probe_b_str, 
                probe_c_str, probe_d_str, 
                zi, zj, zk, zl)
            ] = Result(arr_out, axis=(ax1, ax2))  # fmt: skip

    return cov_dict


def first_element_of_leading_axes(arr: np.ndarray) -> np.ndarray:
    """Helper function to return arr[0, 0, ..., :, :] for any
    array with ≥2 dimensions."""
    if arr.ndim < 2:
        raise ValueError('Array must have at least two dimensions')
    return arr[(0,) * (arr.ndim - 2) + (slice(None), slice(None))]


def cov_heracles_dict_to_sb_10d(
    cov_hc_dict: dict, zbins: int, ell_bins: int, n_probes: int = 2
) -> np.ndarray:
    """The inverse of cov_sb_10d_to_heracles_dict. Loads a heracles-format
    dictionary of 2D arrays and constructs the good old 3x2pt 10D array"""

    # Validate input
    if not cov_hc_dict:
        raise ValueError('Input dictionary cannot be empty')

    cov_10d = np.zeros((
            n_probes, n_probes, n_probes, n_probes, ell_bins, ell_bins,
            zbins, zbins, zbins, zbins
        ))  # fmt: skip

    for k, v in cov_hc_dict.items():
        # Extract the relevant indices from the key
        probe_a_str, probe_b_str, probe_c_str, probe_d_str, zi, zj, zk, zl = k

        # get probe name
        probe_a_ix = const.HS_PROBE_NAME_TO_IX_DICT_HERACLES[probe_a_str]
        probe_b_ix = const.HS_PROBE_NAME_TO_IX_DICT_HERACLES[probe_b_str]
        probe_c_ix = const.HS_PROBE_NAME_TO_IX_DICT_HERACLES[probe_c_str]
        probe_d_ix = const.HS_PROBE_NAME_TO_IX_DICT_HERACLES[probe_d_str]

        # fill with the last 2 axes of v
        cov_10d[
            probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix, :, :, zi, zj, zk, zl
        ] = first_element_of_leading_axes(v)

    return cov_10d


def check_ells_for_spline(ells):
    """Make sure ells are sorted and unique for spline interpolation"""
    assert np.all(np.diff(ells) > 0), 'ells are not sorted'
    assert len(np.unique(ells)) == len(ells), 'ells are not unique'


class IOHandler:
    """Handles loading of input data (n(z) and Cls) from various file formats.

    Supports both Spaceborne (.txt/.dat) and Euclidlib (.fits) formats,
    automatically detecting the format based on file extensions.

    Args:
        cfg: Configuration dictionary
        pvt_cfg: Private configuration dictionary

    """

    def __init__(self, cfg, pvt_cfg):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg
        self.cl_cfg = cfg['C_ell']
        self.output_path = cfg['misc']['output_path']
        self.cov_filename = cfg['covariance']['cov_filename']

    def print_cl_path(self):
        """Print the path of the input Cl files"""
        if self.cfg['C_ell']['use_input_cls']:
            print(f'Using input Cls for LL from file\n{self.cl_cfg["cl_LL_path"]}')
            print(f'Using input Cls for GGL from file\n{self.cl_cfg["cl_GL_path"]}')
            print(f'Using input Cls for GG from file\n{self.cl_cfg["cl_GG_path"]}')
        else:
            return

    def get_nz_fmt(self):
        """Get the format of the input nz files"""
        nz_cfg = self.cfg['nz']  # shorten name

        if (
            nz_cfg['nz_sources_filename'].endswith('.txt')
            and nz_cfg['nz_lenses_filename'].endswith('.txt')
            or nz_cfg['nz_sources_filename'].endswith('.dat')
            and nz_cfg['nz_lenses_filename'].endswith('.dat')
        ):
            self.nz_fmt = 'spaceborne'

        elif nz_cfg['nz_sources_filename'].endswith('.fits') and nz_cfg[
            'nz_lenses_filename'
        ].endswith('.fits'):
            self.nz_fmt = 'euclidlib'

        else:
            raise ValueError(
                'Unsupported or inconsistent format for input nz: all input files '
                'should use the .txt, .dat, or .fits extensions (and all extensions '
                'must be the same)'
            )

    def get_cl_fmt(self):
        """Get the format of the input cl files"""
        if self.cl_cfg['use_input_cls']:
            if (
                self.cl_cfg['cl_LL_path'].endswith('.txt')
                and self.cl_cfg['cl_GL_path'].endswith('.txt')
                and self.cl_cfg['cl_GG_path'].endswith('.txt')
            ) or (
                self.cl_cfg['cl_LL_path'].endswith('.dat')
                and self.cl_cfg['cl_GL_path'].endswith('.dat')
                and self.cl_cfg['cl_GG_path'].endswith('.dat')
            ):
                self.cl_fmt = 'spaceborne'

            elif (
                self.cl_cfg['cl_LL_path'].endswith('.fits')
                and self.cl_cfg['cl_GL_path'].endswith('.fits')
                and self.cl_cfg['cl_GG_path'].endswith('.fits')
            ):
                self.cl_fmt = 'euclidlib'

            else:
                raise ValueError(
                    'Unsupported or inconsistent format for input cls: all input files '
                    'should use the .txt, .dat, or .fits extensions (and all extensions'
                    ' must be the same)'
                )
        else:
            self.cl_fmt = None

    def load_nz(self):
        """Wrapper for loading nz files"""
        if self.nz_fmt == 'spaceborne':
            self._load_nz_sb()
        elif self.nz_fmt == 'euclidlib':
            self._load_nz_el()

    def _load_nz_sb(self):
        # The shape of these input files should be `(zpoints, zbins + 1)`, with `zpoints` the
        # number of points over which the distribution is measured and zbins the number of
        # redshift bins. The first column should contain the redshifts values.
        # We also define:
        # - `nz_full`: nz table including a column for the z values
        # - `nz`:      nz table excluding a column for the z values
        # - `nz_original`: nz table as imported (it may be subjected to shifts later on)
        nz_src_tab_full = np.genfromtxt(self.cfg['nz']['nz_sources_filename'])
        nz_lns_tab_full = np.genfromtxt(self.cfg['nz']['nz_lenses_filename'])
        self.zgrid_nz_src = nz_src_tab_full[:, 0]
        self.zgrid_nz_lns = nz_lns_tab_full[:, 0]
        self.nz_src = nz_src_tab_full[:, 1:]
        self.nz_lns = nz_lns_tab_full[:, 1:]

    def _load_nz_el(self):
        """This is just to assign src and lns data to self"""
        self.zgrid_nz_src, self.nz_src = load_nz_euclidlib(
            self.cfg['nz']['nz_sources_filename']
        )
        self.zgrid_nz_lns, self.nz_lns = load_nz_euclidlib(
            self.cfg['nz']['nz_lenses_filename']
        )

    def load_cls(self):
        """Wrapper for loading cl files, which calls either the sb or el reading
        routines
        """
        if self.cl_fmt == 'spaceborne':
            self._load_cls_sb()
        elif self.cl_fmt == 'euclidlib':
            self._load_cls_el()

        # check symmetry
        check_cl_symm(self.cl_ll_3d_in)
        check_cl_symm(self.cl_gg_3d_in)

    def _load_cls_sb(self):
        cl_ll_tab = np.genfromtxt(self.cl_cfg['cl_LL_path'])
        cl_gl_tab = np.genfromtxt(self.cl_cfg['cl_GL_path'])
        cl_gg_tab = np.genfromtxt(self.cl_cfg['cl_GG_path'])

        self.ells_WL_in, self.cl_ll_3d_in = import_cl_tab(cl_ll_tab)
        self.ells_XC_in, self.cl_gl_3d_in = import_cl_tab(cl_gl_tab)
        self.ells_GC_in, self.cl_gg_3d_in = import_cl_tab(cl_gg_tab)

    def _load_cls_el(self):
        self.ells_WL_in, self.cl_ll_3d_in = load_cl_euclidlib(
            self.cl_cfg['cl_LL_path'], 'SHE', 'SHE'
        )
        self.ells_XC_in, self.cl_gl_3d_in = load_cl_euclidlib(
            self.cl_cfg['cl_GL_path'], 'POS', 'SHE'
        )
        self.ells_GC_in, self.cl_gg_3d_in = load_cl_euclidlib(
            self.cl_cfg['cl_GG_path'], 'POS', 'POS'
        )

    def check_ells_in(self, ell_obj):
        """Make sure ells are sorted and unique for spline interpolation"""
        for _ells in [
            self.ells_WL_in, ell_obj.ells_WL,
            self.ells_XC_in, ell_obj.ells_XC,
            self.ells_GC_in, ell_obj.ells_GC,
        ]:  # fmt: skip
            check_ells_for_spline(_ells)

    def save_cov_euclidlib(self, cov_hs_obj):
        """Helper function to save the covariance in the heracles/cloelikeeuclidlib
        .fits format. Works only for the harmonic-space covariance for the moment.
        """

        def save_term(cov_10d, term_name):
            """Helper to make the code more readable"""
            cov_hc_dict = cov_sb_10d_to_heracles_dict(cov_10d, squeeze=True)
            heracles.io.write(
                f'{self.output_path}/{self.cov_filename}_{term_name}.fits', cov_hc_dict
            )

        # sanity checks
        assert self.cfg['covariance']['save_cov_fits'], (
            'cfg["covariance"]["save_cov_fits"] should be True to '
            'save covariance in .fits format'
        )
        assert self.cfg['probe_selection']['space'] == 'harmonic', (
            'cfg["probe_selection"]["space"] should be "harmonic" to '
            'save covariance in .fits format'
        )

        import heracles

        if self.cfg['covariance']['G']:
            save_term(cov_hs_obj.cov_3x2pt_g_10d, 'Gauss')

        if self.cfg['covariance']['SSC']:
            save_term(cov_hs_obj.cov_3x2pt_ssc_10d, 'SSC')

        if self.cfg['covariance']['cNG']:
            save_term(cov_hs_obj.cov_3x2pt_cng_10d, 'cNG')

        if self.cfg['covariance']['split_gaussian_cov']:
            save_term(cov_hs_obj.cov_3x2pt_sva_10d, 'SVA')
            save_term(cov_hs_obj.cov_3x2pt_sn_10d, 'SN')
            save_term(cov_hs_obj.cov_3x2pt_mix_10d, 'MIX')

        if self.cfg['covariance']['cNG'] or self.cfg['covariance']['SSC']:
            save_term(cov_hs_obj.cov_3x2pt_tot_10d, 'TOT')
