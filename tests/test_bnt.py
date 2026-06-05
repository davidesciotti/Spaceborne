"""Unit tests for the spaceborne.bnt module (BNT transform).

The BNT (Bernardeau-Nishimichi-Taruya) transform is a linear reshuffling of the
tomographic redshift bins. These tests pin down its mathematical properties:
linearity, the identity transform being a no-op, the banded structure of the BNT
matrix, and the consistency between the C(l)-level and covariance-level transforms.
"""

import numpy as np
import pyccl
import pytest

from spaceborne import bnt


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=1234)


@pytest.fixture
def bnt_matrix(rng):
    """A representative lower-triangular BNT-like matrix (unit diagonal)."""
    zbins = 4
    m = np.eye(zbins)
    m[1, 0] = -1.0
    m[2, 0], m[2, 1] = 0.3, -1.2
    m[3, 1], m[3, 2] = 0.5, -1.4
    return m


# ----------------------------------------------------------------------------- #
# cl_bnt_transform
# ----------------------------------------------------------------------------- #
class TestClBntTransform:
    """Tests for the C(l)-level BNT transform: cl -> M @ cl @ M.T per ell."""

    def test_identity_matrix_is_noop(self, rng):
        """With the identity BNT matrix, the LL transform leaves the Cls unchanged."""
        zbins, nbl = 3, 5
        cl_3d = rng.standard_normal((nbl, zbins, zbins))
        identity = np.eye(zbins)

        out = bnt.cl_bnt_transform(cl_3d, identity, 'L', 'L')

        np.testing.assert_allclose(out, cl_3d)

    def test_shape_preserved(self, bnt_matrix, rng):
        """The transform preserves the (nbl, zbins, zbins) shape."""
        zbins = bnt_matrix.shape[0]
        nbl = 7
        cl_3d = rng.standard_normal((nbl, zbins, zbins))

        out = bnt.cl_bnt_transform(cl_3d, bnt_matrix, 'L', 'L')

        assert out.shape == (nbl, zbins, zbins)

    def test_matches_manual_sandwich(self, bnt_matrix, rng):
        """The result equals M @ cl @ M.T computed ell-by-ell by hand."""
        zbins = bnt_matrix.shape[0]
        nbl = 4
        cl_3d = rng.standard_normal((nbl, zbins, zbins))

        out = bnt.cl_bnt_transform(cl_3d, bnt_matrix, 'L', 'L')

        for ell_idx in range(nbl):
            expected = bnt_matrix @ cl_3d[ell_idx] @ bnt_matrix.T
            np.testing.assert_allclose(out[ell_idx], expected)

    def test_gg_probe_ignores_bnt_matrix(self, bnt_matrix, rng):
        """The 'G' probe uses the identity, so GG transform is a no-op."""
        zbins = bnt_matrix.shape[0]
        nbl = 3
        cl_3d = rng.standard_normal((nbl, zbins, zbins))

        out = bnt.cl_bnt_transform(cl_3d, bnt_matrix, 'G', 'G')

        np.testing.assert_allclose(out, cl_3d)

    def test_gl_transforms_only_first_index(self, bnt_matrix, rng):
        """GL transform applies M on the left and identity on the right."""
        zbins = bnt_matrix.shape[0]
        nbl = 3
        cl_3d = rng.standard_normal((nbl, zbins, zbins))

        out = bnt.cl_bnt_transform(cl_3d, bnt_matrix, 'G', 'L')

        for ell_idx in range(nbl):
            expected = np.eye(zbins) @ cl_3d[ell_idx] @ bnt_matrix.T
            np.testing.assert_allclose(out[ell_idx], expected)

    def test_linearity(self, bnt_matrix, rng):
        """T(a*x + b*y) == a*T(x) + b*T(y): the transform is linear."""
        zbins = bnt_matrix.shape[0]
        nbl = 4
        x = rng.standard_normal((nbl, zbins, zbins))
        y = rng.standard_normal((nbl, zbins, zbins))
        a, b = 2.5, -1.3

        lhs = bnt.cl_bnt_transform(a * x + b * y, bnt_matrix, 'L', 'L')
        rhs = a * bnt.cl_bnt_transform(x, bnt_matrix, 'L', 'L') + b * (
            bnt.cl_bnt_transform(y, bnt_matrix, 'L', 'L')
        )

        np.testing.assert_allclose(lhs, rhs)

    def test_raises_on_non_3d_input(self, bnt_matrix):
        """A 2D cl array is rejected."""
        with pytest.raises(AssertionError, match='cl_3d must be 3D'):
            bnt.cl_bnt_transform(np.zeros((3, 3)), bnt_matrix, 'L', 'L')

    def test_raises_on_shape_mismatch(self, bnt_matrix, rng):
        """A zbins mismatch between cl and BNT matrix is rejected."""
        bad = rng.standard_normal((4, 2, 2))  # zbins=2 != bnt zbins
        with pytest.raises(AssertionError, match='number of ell bins'):
            bnt.cl_bnt_transform(bad, bnt_matrix, 'L', 'L')


# ----------------------------------------------------------------------------- #
# cl_bnt_transform_3x2pt
# ----------------------------------------------------------------------------- #
class TestClBntTransform3x2pt:
    """Tests for the 3x2pt wrapper of the C(l) BNT transform."""

    def test_shape_preserved(self, bnt_matrix, rng):
        zbins = bnt_matrix.shape[0]
        nbl = 5
        cl_5d = rng.standard_normal((2, 2, nbl, zbins, zbins))

        out = bnt.cl_bnt_transform_3x2pt(cl_5d, bnt_matrix)

        assert out.shape == cl_5d.shape

    def test_gg_block_untouched(self, bnt_matrix, rng):
        """The [1, 1] (GG) block is copied through unchanged."""
        zbins = bnt_matrix.shape[0]
        nbl = 5
        cl_5d = rng.standard_normal((2, 2, nbl, zbins, zbins))

        out = bnt.cl_bnt_transform_3x2pt(cl_5d, bnt_matrix)

        np.testing.assert_allclose(out[1, 1], cl_5d[1, 1])

    def test_ll_block_matches_single_probe(self, bnt_matrix, rng):
        """The [0, 0] (LL) block matches the single-probe LL transform."""
        zbins = bnt_matrix.shape[0]
        nbl = 5
        cl_5d = rng.standard_normal((2, 2, nbl, zbins, zbins))

        out = bnt.cl_bnt_transform_3x2pt(cl_5d, bnt_matrix)
        expected_ll = bnt.cl_bnt_transform(cl_5d[0, 0], bnt_matrix, 'L', 'L')

        np.testing.assert_allclose(out[0, 0], expected_ll)


# ----------------------------------------------------------------------------- #
# build_x_matrix_bnt
# ----------------------------------------------------------------------------- #
class TestBuildXMatrix:
    """Tests for the rank-4 X matrices used by the covariance BNT transform."""

    def test_keys_and_shapes(self, bnt_matrix):
        zbins = bnt_matrix.shape[0]
        x = bnt.build_x_matrix_bnt(bnt_matrix)

        assert set(x.keys()) == {'LL', 'GG', 'GL', 'LG'}
        for key in x:
            assert x[key].shape == (zbins, zbins, zbins, zbins)

    def test_gg_is_double_kronecker(self, bnt_matrix):
        """X['GG'][a, e, b, f] == delta_ae * delta_bf."""
        zbins = bnt_matrix.shape[0]
        x = bnt.build_x_matrix_bnt(bnt_matrix)
        delta = np.eye(zbins)
        expected = np.einsum('ae, bf -> aebf', delta, delta)

        np.testing.assert_allclose(x['GG'], expected)

    def test_ll_is_bnt_outer_bnt(self, bnt_matrix):
        """X['LL'][a, e, b, f] == M[a, e] * M[b, f]."""
        x = bnt.build_x_matrix_bnt(bnt_matrix)
        expected = np.einsum('ae, bf -> aebf', bnt_matrix, bnt_matrix)

        np.testing.assert_allclose(x['LL'], expected)

    def test_mixed_blocks(self, bnt_matrix):
        """GL transforms the first index only, LG the second only."""
        zbins = bnt_matrix.shape[0]
        delta = np.eye(zbins)
        x = bnt.build_x_matrix_bnt(bnt_matrix)

        np.testing.assert_allclose(
            x['GL'], np.einsum('ae, bf -> aebf', delta, bnt_matrix)
        )
        np.testing.assert_allclose(
            x['LG'], np.einsum('ae, bf -> aebf', bnt_matrix, delta)
        )


# ----------------------------------------------------------------------------- #
# cov_bnt_transform
# ----------------------------------------------------------------------------- #
class TestCovBntTransform:
    """Tests for the covariance-level (6D) BNT transform."""

    def test_identity_is_noop(self, rng):
        """With an identity BNT matrix, the covariance is unchanged."""
        zbins, nbl = 3, 2
        x = bnt.build_x_matrix_bnt(np.eye(zbins))
        cov_6d = rng.standard_normal((nbl, nbl, zbins, zbins, zbins, zbins))

        out = bnt.cov_bnt_transform(cov_6d, x, 'LL', 'LL')

        np.testing.assert_allclose(out, cov_6d)

    def test_shape_preserved(self, bnt_matrix, rng):
        zbins = bnt_matrix.shape[0]
        nbl = 2
        x = bnt.build_x_matrix_bnt(bnt_matrix)
        cov_6d = rng.standard_normal((nbl, nbl, zbins, zbins, zbins, zbins))

        out = bnt.cov_bnt_transform(cov_6d, x, 'LL', 'LL')

        assert out.shape == cov_6d.shape

    def test_raises_on_non_6d_input(self, bnt_matrix):
        x = bnt.build_x_matrix_bnt(bnt_matrix)
        with pytest.raises(AssertionError, match='6 dimensions'):
            bnt.cov_bnt_transform(np.zeros((2, 2, 3, 3)), x, 'LL', 'LL')

    def test_outer_product_consistency(self, bnt_matrix, rng):
        """Key cross-check linking the C(l) and covariance transforms.

        If cov[L, M, a, b, c, d] = clA[L, a, b] * clB[M, c, d], then the BNT
        transform of the covariance must equal the outer product of the BNT
        transformed Cls. This ties cov_bnt_transform to cl_bnt_transform.
        """
        zbins = bnt_matrix.shape[0]
        nbl = 3
        x = bnt.build_x_matrix_bnt(bnt_matrix)

        cl_a = rng.standard_normal((nbl, zbins, zbins))
        cl_b = rng.standard_normal((nbl, zbins, zbins))

        # cov as an outer product of the two Cl arrays
        cov_6d = cl_a[:, None, :, :, None, None] * cl_b[None, :, None, None, :, :]

        out = bnt.cov_bnt_transform(cov_6d, x, 'LL', 'LL')

        cl_a_bnt = bnt.cl_bnt_transform(cl_a, bnt_matrix, 'L', 'L')
        cl_b_bnt = bnt.cl_bnt_transform(cl_b, bnt_matrix, 'L', 'L')
        expected = (
            cl_a_bnt[:, None, :, :, None, None] * cl_b_bnt[None, :, None, None, :, :]
        )

        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


# ----------------------------------------------------------------------------- #
# compute_bnt_matrix
# ----------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def cosmo_ccl():
    """A cheap vanilla CCL cosmology (background only is needed)."""
    return pyccl.CosmologyVanillaLCDM()


def _gaussian_nz(z_grid, zbins, z_means, sigma=0.15):
    """Build a simple set of Gaussian n(z), one bump per tomographic bin."""
    n_of_z = np.empty((len(z_grid), zbins))
    for zi, z_mean in enumerate(z_means):
        n_of_z[:, zi] = np.exp(-0.5 * ((z_grid - z_mean) / sigma) ** 2)
    return n_of_z


class TestComputeBntMatrix:
    """Tests for the construction of the BNT matrix from n(z) and cosmology."""

    @pytest.fixture
    def setup(self):
        zbins = 4
        z_grid = np.linspace(0.01, 3.0, 200)
        z_means = np.linspace(0.4, 1.6, zbins)
        n_of_z = _gaussian_nz(z_grid, zbins, z_means)
        return zbins, z_grid, n_of_z

    def test_shape(self, setup, cosmo_ccl):
        zbins, z_grid, n_of_z = setup
        m = bnt.compute_bnt_matrix(zbins, z_grid, n_of_z, cosmo_ccl, plot_nz=False)
        assert m.shape == (zbins, zbins)

    def test_unit_diagonal(self, setup, cosmo_ccl):
        zbins, z_grid, n_of_z = setup
        m = bnt.compute_bnt_matrix(zbins, z_grid, n_of_z, cosmo_ccl, plot_nz=False)
        np.testing.assert_allclose(np.diag(m), np.ones(zbins))

    def test_fixed_first_entries(self, setup, cosmo_ccl):
        """The first column entries are fixed by construction: [0,0]=1, [1,0]=-1."""
        zbins, z_grid, n_of_z = setup
        m = bnt.compute_bnt_matrix(zbins, z_grid, n_of_z, cosmo_ccl, plot_nz=False)
        assert m[0, 0] == 1.0
        assert m[1, 0] == -1.0

    def test_lower_triangular_banded(self, setup, cosmo_ccl):
        """Only the diagonal and the two sub-diagonals (i-1, i-2) are populated.

        Everything strictly above the diagonal, and below the (i-2) band, is zero.
        """
        zbins, z_grid, n_of_z = setup
        m = bnt.compute_bnt_matrix(zbins, z_grid, n_of_z, cosmo_ccl, plot_nz=False)

        for i in range(zbins):
            for j in range(zbins):
                if j > i or j < i - 2:
                    assert m[i, j] == 0.0, f'entry [{i}, {j}] should be zero'

    def test_raises_on_wrong_nz_rows(self, cosmo_ccl):
        zbins = 3
        z_grid = np.linspace(0.01, 3.0, 50)
        bad_nz = np.ones((49, zbins))  # rows != len(z_grid)
        with pytest.raises(AssertionError, match='zgrid_n_of_z rows'):
            bnt.compute_bnt_matrix(zbins, z_grid, bad_nz, cosmo_ccl, plot_nz=False)

    def test_raises_on_wrong_nz_cols(self, cosmo_ccl):
        zbins = 3
        z_grid = np.linspace(0.01, 3.0, 50)
        bad_nz = np.ones((50, zbins + 1))  # cols != zbins
        with pytest.raises(AssertionError, match='zbins columns'):
            bnt.compute_bnt_matrix(zbins, z_grid, bad_nz, cosmo_ccl, plot_nz=False)

    def test_raises_on_non_monotonic_zgrid(self, cosmo_ccl):
        zbins = 3
        z_grid = np.linspace(0.01, 3.0, 50)
        z_grid[10] = z_grid[9]  # break strict monotonicity
        n_of_z = np.ones((50, zbins))
        with pytest.raises(AssertionError, match='monotonically increasing'):
            bnt.compute_bnt_matrix(zbins, z_grid, n_of_z, cosmo_ccl, plot_nz=False)

    def test_warns_when_zgrid_starts_at_zero(self, cosmo_ccl):
        """A z_grid starting at 0 triggers a warning (null comoving distance)."""
        zbins = 3
        z_grid = np.linspace(0.0, 3.0, 100)
        z_means = np.linspace(0.4, 1.6, zbins)
        n_of_z = _gaussian_nz(z_grid, zbins, z_means)
        with pytest.warns(UserWarning, match='z_grid starts at 0'):
            bnt.compute_bnt_matrix(zbins, z_grid, n_of_z, cosmo_ccl, plot_nz=False)


# ----------------------------------------------------------------------------- #
# bnt_transform_cov_dict
# ----------------------------------------------------------------------------- #
class TestBntTransformCovDict:
    """Tests for the in-place BNT transform of a full covariance dictionary."""

    def _make_cov_dict(self, terms, probe_combs, nbl, zbins, rng):
        cov_dict = {}
        for term in terms:
            cov_dict[term] = {}
            for probe_abcd in probe_combs:
                from spaceborne import sb_lib as sl

                probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'harmonic')
                cov_dict[term][probe_ab, probe_cd] = {
                    '6d': rng.standard_normal((nbl, nbl, zbins, zbins, zbins, zbins))
                }
        return cov_dict

    def test_transforms_each_probe_block(self, bnt_matrix, rng):
        """Each probe block is replaced by its BNT-transformed counterpart."""
        zbins = bnt_matrix.shape[0]
        nbl = 2
        probe_combs = ['LLLL', 'GLLL']
        cov_dict = self._make_cov_dict(['g'], probe_combs, nbl, zbins, rng)

        # keep a copy of the inputs to compute the expected transform
        x = bnt.build_x_matrix_bnt(bnt_matrix)
        from spaceborne import sb_lib as sl

        expected = {}
        for probe_abcd in probe_combs:
            probe_ab, probe_cd = sl.split_probe_name(probe_abcd, 'harmonic')
            expected[probe_ab, probe_cd] = bnt.cov_bnt_transform(
                cov_dict['g'][probe_ab, probe_cd]['6d'].copy(), x, probe_ab, probe_cd
            )

        bnt.bnt_transform_cov_dict(cov_dict, bnt_matrix, probe_combs)

        for key, expected_arr in expected.items():
            np.testing.assert_allclose(cov_dict['g'][key]['6d'], expected_arr)

    def test_tot_term_is_skipped(self, bnt_matrix, rng):
        """The 'tot' term is left untouched (it is assembled later)."""
        zbins = bnt_matrix.shape[0]
        nbl = 2
        probe_combs = ['LLLL']
        cov_dict = self._make_cov_dict(['tot'], probe_combs, nbl, zbins, rng)
        original = cov_dict['tot']['LL', 'LL']['6d'].copy()

        bnt.bnt_transform_cov_dict(cov_dict, bnt_matrix, probe_combs)

        np.testing.assert_allclose(cov_dict['tot']['LL', 'LL']['6d'], original)
