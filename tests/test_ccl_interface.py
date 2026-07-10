"""Unit tests for spaceborne.ccl_interface.apply_mult_shear_bias.

This is the only pure, standalone function in ccl_interface.py that doesn't
need a CCLInterface instance or a CCL cosmology: it just rescales small
in-memory Cl arrays by (1+m) factors per tomographic bin. We build random
3D Cl arrays and check the scaling directly by index arithmetic.

Note the function mutates and returns its inputs in place (``cl_ll_3d[...]
*=  ...``); we assert both the returned values and the in-place aliasing.

The CCLInterface class and compute_cl_3x2pt_5d are intentionally not
covered here (they need a full CCL cosmology + tracer setup).
"""

import numpy as np
import pytest

from spaceborne import ccl_interface as ci


@pytest.fixture
def rng():
    """Deterministic random generator so tests are reproducible."""
    return np.random.default_rng(seed=99)


@pytest.fixture
def zbins():
    return 3


@pytest.fixture
def mult_shear_bias():
    return np.array([0.01, -0.02, 0.03])


class TestApplyMultShearBias:
    """Tests for the multiplicative shear bias rescaling."""

    def test_cl_ll_scales_as_one_plus_mi_one_plus_mj(self, rng, zbins, mult_shear_bias):
        nbl = 4
        cl_ll_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_gl_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_ll_orig = cl_ll_3d.copy()

        out_ll, _ = ci.apply_mult_shear_bias(cl_ll_3d, cl_gl_3d, mult_shear_bias, zbins)

        for zi in range(zbins):
            for zj in range(zbins):
                expected = (
                    cl_ll_orig[:, zi, zj]
                    * (1 + mult_shear_bias[zi])
                    * (1 + mult_shear_bias[zj])
                )
                np.testing.assert_allclose(out_ll[:, zi, zj], expected)

    def test_cl_gl_scales_as_one_plus_m_second_index(self, rng, zbins, mult_shear_bias):
        """cl_gl_3d[ell, zi, zj] is only rescaled by (1 + m[zj]) (see source:
        the loop only multiplies by ``1 + mult_shear_bias[zj]``, i.e. the
        second/shear tomographic index)."""
        nbl = 4
        cl_ll_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_gl_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_gl_orig = cl_gl_3d.copy()

        _, out_gl = ci.apply_mult_shear_bias(cl_ll_3d, cl_gl_3d, mult_shear_bias, zbins)

        for zi in range(zbins):
            for zj in range(zbins):
                expected = cl_gl_orig[:, zi, zj] * (1 + mult_shear_bias[zj])
                np.testing.assert_allclose(out_gl[:, zi, zj], expected)

    def test_zero_bias_is_identity(self, rng, zbins):
        nbl = 4
        cl_ll_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_gl_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_ll_orig = cl_ll_3d.copy()
        cl_gl_orig = cl_gl_3d.copy()

        out_ll, out_gl = ci.apply_mult_shear_bias(
            cl_ll_3d, cl_gl_3d, np.zeros(zbins), zbins
        )

        np.testing.assert_array_equal(out_ll, cl_ll_orig)
        np.testing.assert_array_equal(out_gl, cl_gl_orig)

    def test_mutates_and_returns_inputs_in_place(self, rng, zbins, mult_shear_bias):
        nbl = 2
        cl_ll_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_gl_3d = rng.standard_normal((nbl, zbins, zbins))

        out_ll, out_gl = ci.apply_mult_shear_bias(
            cl_ll_3d, cl_gl_3d, mult_shear_bias, zbins
        )

        assert out_ll is cl_ll_3d
        assert out_gl is cl_gl_3d

    def test_wrong_length_mult_shear_bias_raises(self, rng, zbins):
        nbl = 2
        cl_ll_3d = rng.standard_normal((nbl, zbins, zbins))
        cl_gl_3d = rng.standard_normal((nbl, zbins, zbins))

        with pytest.raises(AssertionError):
            ci.apply_mult_shear_bias(cl_ll_3d, cl_gl_3d, np.zeros(zbins + 1), zbins)
