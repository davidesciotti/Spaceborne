"""Class for reading in data in various formats"""

import numpy as np
import itertools


def load_cl_euclidlib(filename, key_a, key_b):
    import euclidlib as el

    # checks
    assert filename.endswith('.fits'), 'filename must end with .fits'
    assert key_a in ['SHE', 'POS'], 'key_a must be "SHE" or "POS"'
    assert key_b in ['SHE', 'POS'], 'key_b must be "SHE" or "POS"'

    is_auto_spectrum = key_a == key_b

    # import .fits using el
    cl_dict = el.photo.angular_power_spectra(filename)

    # extract ells
    ells = cl_dict[key_a, key_b, 1, 1].ell
    nbl = ells.size

    # extract zbins (check consistency of columns first)
    zbins_i = max(i[2] for i in cl_dict)
    zbins_j = max(i[3] for i in cl_dict)
    assert zbins_i == zbins_j, 'zbins are not the same for all columns'
    zbins = zbins_i

    cl_3d = np.zeros((nbl, zbins, zbins))

    for zi, zj in itertools.product(range(zbins), range(zbins)):
        if zj >= zi:
            cl_3d[:, zi, zj] = cl_dict[(key_a, key_b, zi + 1, zj + 1)][0]
        else:
            if is_auto_spectrum:
                cl_3d[:, zi, zj] = cl_3d[:, zj, zi]
            else:
                cl_3d[:, zi, zj] = cl_dict[key_a, key_b, zi + 1, zj + 1][0]

    return ells, cl_3d


def cov_sb_10d_to_heracles_dict(cov_10d, squeeze):
    """
    SB = 'Spaceborne'
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

    # this dictionary maps the SB probe indices to the HC probe names (keys)
    probe_name_dict = {
        0: 'POS',
        1: 'SHE',
    }

    # this dictionary specifies the dimension of the corresponding axes in the output
    # arrays. The dimensions correspond to the spin, except POS (spin-0) still needs 1
    # dimension (not 0!)
    probe_dims_dict = {
        'POS': 1,
        'SHE': 2,
    }

    # just a check
    print('Translating covariance from Spaceborne to Heracles format...')

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

    n_probes = cov_10d.shape[0]
    zbins = cov_10d.shape[-1]
    nbl = cov_10d.shape[4]

    print(f'cov_10d shape = {cov_10d.shape}')
    print(f'{n_probes = }')
    print(f'{nbl = }')
    print(f'{zbins = }')

    cov_dict = {}

    for probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix in itertools.product(
        range(n_probes), repeat=4
    ):
        for zi, zj, zk, zl in itertools.product(range(zbins), repeat=4):
            # get probe names and spins
            probe_a_str = probe_name_dict[probe_a_ix]
            probe_b_str = probe_name_dict[probe_b_ix]
            probe_c_str = probe_name_dict[probe_c_ix]
            probe_d_str = probe_name_dict[probe_d_ix]

            probe_a_dims = probe_dims_dict[probe_a_str]
            probe_b_dims = probe_dims_dict[probe_b_str]
            probe_c_dims = probe_dims_dict[probe_c_str]
            probe_d_dims = probe_dims_dict[probe_d_str]

            arr_out = np.zeros(
                shape=(
                    probe_a_dims,
                    probe_b_dims,
                    probe_c_dims,
                    probe_d_dims,
                    nbl,
                    nbl,
                )
            )

            arr_out[0, 0, 0, 0, :, :] = cov_10d[
                probe_a_ix, probe_b_ix, probe_c_ix, probe_d_ix,
                :, :, zi, zj, zk, zl,
            ]  # fmt: skip

            if squeeze:
                arr_out = np.squeeze(arr_out)

            # fmt: off
            cov_dict[
                (probe_a_str, probe_b_str,
                probe_c_str, probe_d_str,
                zi, zj, zk, zl)
            ] = arr_out

    print('done')

    return cov_dict


class IOHandler:
    def __init__(self, cfg, pvt_cfg):
        self.cfg = cfg
        self.pvt_cfg = pvt_cfg
        self.cl_cfg = cfg['C_ell']

    def print_cl_path(self):
        """
        Print the path of the input Cl files
        """

        if self.cfg['C_ell']['use_input_cls']:
            print(f'Using input Cls for LL from file\n{self.cl_cfg["cl_LL_path"]}')
            print(f'Using input Cls for GGL from file\n{self.cl_cfg["cl_GL_path"]}')
            print(f'Using input Cls for GG from file\n{self.cl_cfg["cl_GG_path"]}')
        else:
            return

    def get_nz_fmt(self):
        """
        Get the format of the input nz files
        """
        nz_cfg = self.cfg['nz']  # shorten name

        if nz_cfg['nz_sources_filename'].endswith('.txt') and nz_cfg[
            'nz_lenses_filename'
        ].endswith('.txt'):
            self.nz_fmt = 'spaceborne'

        elif nz_cfg['nz_sources_filename'].endswith('.dat') and nz_cfg[
            'nz_lenses_filename'
        ].endswith('.dat'):
            self.nz_fmt = 'spaceborne'

        elif nz_cfg['nz_sources_filename'].endswith('.fits') and nz_cfg[
            'nz_lenses_filename'
        ].endswith('.fits'):
            self.nz_fmt = 'euclidlib'

        else:
            raise ValueError(
                'Unsupported or inconsistent format for input nz: all input files should'
                'use the .txt, .dat, or .fits extensions (and all extensions must be '
                'the same)'
            )

    def get_cl_fmt(self):
        """
        Get the format of the input cl files
        """

        if self.cl_cfg['use_input_cls']:
            if (
                self.cl_cfg['cl_LL_path'].endswith('.txt')
                and self.cl_cfg['cl_GL_path'].endswith('.txt')
                and self.cl_cfg['cl_GG_path'].endswith('.txt')
            ):
                self.cl_fmt = 'spaceborne'

            elif (
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
                    'Unsupported or inconsistent format for input cls: all input files should'
                    'use the .txt, .dat, or .fits extensions (and all extensions must be '
                    'the same)'
                )

    def load_nz(self):
        """Wrapper for loading nz files"""
        if self.nz_fmt == 'spaceborne':
            self._load_nz_sb()
        elif self.nz_fmt == 'euclidlib':
            self._load_nz_el()

    def _load_nz_sb(self):
        pass

    def _load_nz_el(self):
        pass

    def load_cls(self):
        """Wrapper for loading cl files"""
        if self.cl_fmt == 'spaceborne':
            self._load_cls_sb()
        elif self.cl_fmt == 'euclidlib':
            self._load_cls_el()

    def _load_cls_sb(self):
        pass

    def _load_cls_el(self):
        self.ells_WL_in, self.cl_ll_3d_in = self.load_cl_euclidlib(
            self.cl_cfg['cl_LL_path'], 'SHE', 'SHE'
        )
        self.ells_XC_in, self.cl_gl_3d_in = self.load_cl_euclidlib(
            self.cl_cfg['cl_GL_path'], 'POS', 'SHE'
        )
        self.ells_GG_in, self.cl_gg_3d_in = self.load_cl_euclidlib(
            self.cl_cfg['cl_GG_path'], 'POS', 'POS'
        )
