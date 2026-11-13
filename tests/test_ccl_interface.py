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
import pyccl as ccl
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


# ---------------------------------------------------------------------------
# Regression tests for the ell1 <-> ell2 axis orientation of
# CCLInterface.compute_ng_cov_probe_block.
#
# Convention under test: the returned array has shape
# (nbl, nbl, zpairs_AB, zpairs_CD), with axis 0 = ell of the (A, B) pair and
# axis 1 = ell of the (C, D) pair.
#
# CCL's ccl.covariances.angular_cl_cov_cNG returns a raw array whose ell1 and
# ell2 axes are swapped relative to that convention, so the source method
# must transpose. To make a missing/misplaced transpose actually break a
# test, we use a custom, deliberately non-symmetric trispectrum
# T(k1, k2) = k1**2 * k2 (built via ccl.tk3d.Tk3D). Since density tracers use
# Limber's k = (ell + 0.5) / chi(a), and all four legs share the same
# redshift kernel(s), the ell-dependence of the resulting cNG covariance
# factorizes *exactly* into a power law of (ell_AB + 0.5) and
# (ell_CD + 0.5) with different exponents (2 and 1) -- giving an
# analytically known, asymmetric reference to check against.
#
# The helper functions below (leading underscore, not collected by pytest)
# hold the actual assertions so they can be reused, unmodified, by a
# standalone sensitivity script that monkeypatches
# ci.CCLInterface.compute_ng_cov_probe_block to reintroduce the historical
# "missing transpose" bug and confirm these tests catch it.
# ---------------------------------------------------------------------------


def _gauss_nz(z, z0, sigma=0.05):
    return np.exp(-0.5 * ((z - z0) / sigma) ** 2)


def _build_ng_cov_env():
    """Cheap CCL setup (fiducial cosmology, 2 Gaussian n(z) bins, a coarse
    but accurate-enough Tk3D grid) shared by all ell-orientation tests."""
    cosmo = ccl.CosmologyVanillaLCDM()

    lk_arr = np.linspace(np.log(1e-7), np.log(1e4), 200)
    a_arr = np.linspace(0.02, 1.0, 80)
    k_arr = np.exp(lk_arr)
    # Tk3D reads tkk_arr[ia, i, j] as T(k1=k[j], k2=k[i]) (see Tk3D.__call__), so
    # this is T(k1, k2) = k1**2 * k2, independent of a
    tkk_2d = np.outer(k_arr, k_arr**2)
    tkk_arr = np.repeat(tkk_2d[None, :, :], len(a_arr), axis=0)
    trisp = ccl.tk3d.Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk_arr, is_logt=False)

    z = np.linspace(0.001, 2.0, 500)
    z0_list = [0.3, 0.7]
    dens_tracers = [
        ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(z, _gauss_nz(z, z0)),
            bias=(z, np.ones_like(z)),
            mag_bias=None,
        )
        for z0 in z0_list
    ]
    lens_tracers = [
        ccl.WeakLensingTracer(cosmo, dndz=(z, _gauss_nz(z, z0))) for z0 in z0_list
    ]

    obj = object.__new__(ci.CCLInterface)
    obj.cosmo_ccl = cosmo

    # last two columns are (zi, zj); auto pairs for 2 bins: (0,0), (0,1), (1,1)
    ind = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]])

    return {
        'obj': obj,
        'cosmo': cosmo,
        'trisp': trisp,
        'dens_tracers': dens_tracers,
        'lens_tracers': lens_tracers,
        'ell': np.array([10.0, 20.0]),
        'ind': ind,
    }


def _ccl_cov_cng(env, tracer1, tracer2, tracer3, tracer4):
    """Direct (reference) CCL call, kept independent of the code under test."""
    return ccl.covariances.angular_cl_cov_cNG(
        env['cosmo'],
        tracer1=tracer1,
        tracer2=tracer2,
        ell=env['ell'],
        t_of_kk_a=env['trisp'],
        tracer3=tracer3,
        tracer4=tracer4,
        ell2=env['ell'],
        fsky=1.0,
        integration_method='qag_quad',
    )


def _assert_asymmetric(ref, label):
    """Guard against a vacuous test: fail loudly if the reference itself
    happens to be symmetric under ell1 <-> ell2, since then a missing
    transpose would go undetected."""
    rel_asymmetry = np.max(np.abs(ref - ref.T)) / np.max(np.abs(ref))
    assert rel_asymmetry > 1e-3, (
        f'{label}: reference is symmetric under ell1<->ell2 '
        f'(rel_asymmetry={rel_asymmetry}); this test would not catch a '
        'missing transpose'
    )


def _check_diagonal_same_pair_ratio(env):
    """symmetrize_zpairs=True, ij == kl == (bin0, bin0): exercises the
    "kl == ij" path and checks the analytic power-law ratio implied by
    k = (ell + 0.5) / chi and T(k1, k2) = k1**2 * k2, with k1 <-> ell_AB (axis 0):
    cov[1, 0] / cov[0, 1] = (20.5**2 * 10.5) / (10.5**2 * 20.5) = 20.5 / 10.5."""
    ind = env['ind']
    cov = env['obj'].compute_ng_cov_probe_block(
        which_ng_cov='cNG',
        kernel_A=env['dens_tracers'],
        kernel_B=env['dens_tracers'],
        kernel_C=env['dens_tracers'],
        kernel_D=env['dens_tracers'],
        ell=env['ell'],
        trisp_abcd=env['trisp'],
        fsky=1.0,
        sigma2_b_tpl=None,
        ind_AB=ind,
        ind_CD=ind,
        integration_method='qag_quad',
        symmetrize_zpairs=True,
    )
    ratio = cov[1, 0, 0, 0] / cov[0, 1, 0, 0]
    expected = 20.5 / 10.5
    assert abs(ratio - expected) / expected < 1e-5, (
        f'ratio={ratio}, expected={expected}'
    )
    return cov


def _check_diagonal_cross_pair(env):
    """symmetrize_zpairs=True, ij != kl, with different tomographic bins on
    the two sides so the entry is genuinely asymmetric in ell1 <-> ell2.
    Compares both cov[:, :, ij, kl] and cov[:, :, kl, ij] against
    independent, direct CCL calls."""
    ind = env['ind']
    dens = env['dens_tracers']
    ij, kl = 0, 1  # pairs (0, 0) and (0, 1)
    zi_ij, zj_ij = ind[ij, -2], ind[ij, -1]
    zi_kl, zj_kl = ind[kl, -2], ind[kl, -1]

    cov = env['obj'].compute_ng_cov_probe_block(
        which_ng_cov='cNG',
        kernel_A=dens,
        kernel_B=dens,
        kernel_C=dens,
        kernel_D=dens,
        ell=env['ell'],
        trisp_abcd=env['trisp'],
        fsky=1.0,
        sigma2_b_tpl=None,
        ind_AB=ind,
        ind_CD=ind,
        integration_method='qag_quad',
        symmetrize_zpairs=True,
    )

    ref_ij_kl = _ccl_cov_cng(env, dens[zi_ij], dens[zj_ij], dens[zi_kl], dens[zj_kl])
    _assert_asymmetric(ref_ij_kl, 'diagonal branch, ij != kl')
    np.testing.assert_allclose(cov[:, :, ij, kl], ref_ij_kl.T, rtol=1e-8)

    # cov[:, :, kl, ij] is filled by the "kl != ij" symmetry-fill path; check
    # it against an independent direct CCL call with AB/CD swapped (no
    # transpose needed here, since the same raw ccl_out already has ell_AB
    # and ell_CD on the axes cov[:, :, kl, ij] expects -- see the source
    # method's symmetrize_zpairs=True branch).
    ref_kl_ij = _ccl_cov_cng(env, dens[zi_kl], dens[zj_kl], dens[zi_ij], dens[zj_ij])
    np.testing.assert_allclose(cov[:, :, kl, ij], ref_kl_ij, rtol=1e-8)


def _check_offdiagonal(env):
    """symmetrize_zpairs=False (e.g. an LLGG-like block): every [ij, kl]
    entry is computed independently, so every entry must match a direct CCL
    call, transposed."""
    ind = env['ind']
    lens = env['lens_tracers']
    dens = env['dens_tracers']

    cov = env['obj'].compute_ng_cov_probe_block(
        which_ng_cov='cNG',
        kernel_A=lens,
        kernel_B=lens,
        kernel_C=dens,
        kernel_D=dens,
        ell=env['ell'],
        trisp_abcd=env['trisp'],
        fsky=1.0,
        sigma2_b_tpl=None,
        ind_AB=ind,
        ind_CD=ind,
        integration_method='qag_quad',
        symmetrize_zpairs=False,
    )

    found_asymmetric_entry = False
    for ij in range(ind.shape[0]):
        for kl in range(ind.shape[0]):
            zi1, zj1 = ind[ij, -2], ind[ij, -1]
            zi2, zj2 = ind[kl, -2], ind[kl, -1]
            ref = _ccl_cov_cng(env, lens[zi1], lens[zj1], dens[zi2], dens[zj2])
            np.testing.assert_allclose(cov[:, :, ij, kl], ref.T, rtol=1e-8)
            if np.max(np.abs(ref - ref.T)) / np.max(np.abs(ref)) > 1e-3:
                found_asymmetric_entry = True

    assert found_asymmetric_entry, (
        'no [ij, kl] entry was asymmetric under ell1<->ell2; this test '
        'would not catch a missing transpose'
    )


@pytest.fixture(scope='module')
def ng_cov_env():
    return _build_ng_cov_env()


class TestNgCovProbeBlockEllOrientation:
    """Regression tests for the ell1 <-> ell2 axis convention of
    ``CCLInterface.compute_ng_cov_probe_block`` (see module-level comment
    above for the rationale and the trispectrum construction)."""

    def test_diagonal_branch_same_pair_ratio(self, ng_cov_env):
        _check_diagonal_same_pair_ratio(ng_cov_env)

    def test_diagonal_branch_cross_pair_orientation(self, ng_cov_env):
        _check_diagonal_cross_pair(ng_cov_env)

    def test_offdiagonal_branch_orientation(self, ng_cov_env):
        _check_offdiagonal(ng_cov_env)


# ---------------------------------------------------------------------------
# Per-bin (redshift-independent) galaxy and magnification bias
# ---------------------------------------------------------------------------
@pytest.fixture
def per_bin_env():
    """Minimal CCLInterface (2 Gaussian lens/source bins) able to run
    set_kernel_obj, without instantiating the full class."""
    zbins = 2
    z = np.linspace(0.001, 2.0, 400)
    nz_2d = np.column_stack([_gauss_nz(z, z0) for z0 in (0.4, 0.9)])

    obj = object.__new__(ci.CCLInterface)
    obj.cosmo_ccl = ccl.CosmologyVanillaLCDM()
    obj.zbins = zbins
    obj.nz_src_tuple = (z, nz_2d)
    obj.nz_lns_tuple = (z, nz_2d)
    obj.zgrid_nz_lns = z
    obj.ia_bias_tuple = None
    obj.mag_bias_tuple = None
    return {'obj': obj, 'z': z, 'zbins': zbins, 'ell': np.geomspace(20, 2000, 6)}


class TestPerBinBias:
    gal_bias = np.array([1.2, 1.7])
    mag_bias = np.array([-1.0, 0.5])  # b_mag = 5 s - 2

    def test_gal_bias_arrays(self, per_bin_env):
        obj, z = per_bin_env['obj'], per_bin_env['z']
        obj.set_gal_bias_tuple_per_bin(z, list(self.gal_bias))

        assert obj.gal_bias_2d.shape == (z.size, per_bin_env['zbins'])
        np.testing.assert_array_equal(
            obj.gal_bias_2d, np.tile(self.gal_bias, (z.size, 1))
        )
        assert obj.gal_bias_tuple[0] is z
        # the callable works on any grid (used e.g. for the SSC responses)
        z_other = np.linspace(0.1, 1.5, 7)
        np.testing.assert_array_equal(
            obj.gal_bias_func(z_other), np.tile(self.gal_bias, (z_other.size, 1))
        )

    def test_mag_bias_converted_to_s(self, per_bin_env):
        obj, z = per_bin_env['obj'], per_bin_env['z']
        obj.set_mag_bias_tuple_per_bin(z, list(self.mag_bias))

        s_expected = (self.mag_bias + 2) / 5
        np.testing.assert_allclose(obj.mag_bias_2d, np.tile(s_expected, (z.size, 1)))
        assert obj.mag_bias_tuple[1] is obj.mag_bias_2d

    @pytest.mark.parametrize('setter', ['gal', 'mag'])
    def test_wrong_length_raises(self, per_bin_env, setter):
        obj, z = per_bin_env['obj'], per_bin_env['z']
        with pytest.raises(AssertionError, match='zbins'):
            getattr(obj, f'set_{setter}_bias_tuple_per_bin')(z, [1.0, 1.0, 1.0])

    def test_galaxy_cls_scale_as_bi_bj(self, per_bin_env):
        """Without RSD and magnification, the galaxy tracer of bin i is b_i times
        the unit-bias density tracer: C^gg_ij = b_i b_j C^dd_ij."""
        obj, z, ell = per_bin_env['obj'], per_bin_env['z'], per_bin_env['ell']
        obj.mag_bias_tuple = None
        obj.set_gal_bias_tuple_per_bin(z, self.gal_bias)
        obj.set_kernel_obj(has_rsd=False, n_samples_wf=256)

        cosmo = obj.cosmo_ccl
        for zi in range(2):
            for zj in range(zi, 2):
                cl_gg = ccl.angular_cl(
                    cosmo, obj.wf_galaxy_obj[zi], obj.wf_galaxy_obj[zj], ell
                )
                cl_dd = ccl.angular_cl(
                    cosmo, obj.wf_density_obj[zi], obj.wf_density_obj[zj], ell
                )
                np.testing.assert_allclose(
                    cl_gg, self.gal_bias[zi] * self.gal_bias[zj] * cl_dd, rtol=1e-10
                )

    def test_mag_tracer_matches_direct_ccl(self, per_bin_env):
        """The per-bin magnification tracer equals a CCL NumberCountsTracer built
        directly with the constant s_i = (b_mag_i + 2) / 5."""
        obj, z, ell = per_bin_env['obj'], per_bin_env['z'], per_bin_env['ell']
        obj.set_gal_bias_tuple_per_bin(z, self.gal_bias)
        obj.set_mag_bias_tuple_per_bin(z, self.mag_bias)
        obj.set_kernel_obj(has_rsd=False, n_samples_wf=256)

        cosmo = obj.cosmo_ccl
        for zi in range(2):
            s_i = (self.mag_bias[zi] + 2) / 5
            ref = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=False,
                dndz=(z, obj.nz_lns_tuple[1][:, zi]),
                bias=None,
                mag_bias=(z, np.full_like(z, s_i)),
                n_samples=256,
            )
            np.testing.assert_allclose(
                ccl.angular_cl(cosmo, obj.wf_mag_obj[zi], obj.wf_mag_obj[zi], ell),
                ccl.angular_cl(cosmo, ref, ref, ell),
                rtol=1e-10,
            )
