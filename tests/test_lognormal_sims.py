"""Unit tests for the lognormal (GLASS) signal fields of the ensemble covariance."""

import healpy as hp
import numpy as np
import pyccl as ccl
import pytest

from spaceborne import cov_partial_sky, lognormal_sims, responses


@pytest.fixture(scope='module')
def cosmo():
    return ccl.Cosmology(Omega_c=0.27, Omega_b=0.049, h=0.67, sigma8=0.81, n_s=0.96)


@pytest.fixture(scope='module')
def kernels(cosmo):
    """One source and one lens bin, CCL kernels on a z grid (galaxy bias included)."""
    z = np.linspace(0.01, 2.0, 800)
    chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + z))
    bias = 1.4
    tr_l = ccl.WeakLensingTracer(cosmo, dndz=(z, z**2 * np.exp(-((z / 0.5) ** 1.5))))
    tr_g = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)),
        bias=(z, bias * np.ones_like(z)),
    )
    wf_l = tr_l.get_kernel(chi)[0][:, None]
    wf_g = tr_g.get_kernel(chi)[0][:, None] * bias
    return z, chi, wf_l, wf_g, tr_l, tr_g


class TestShells:
    def test_kernel_weights_sum_to_integral(self, kernels):
        _, chi, wf_l, _, _, _ = kernels
        edges = lognormal_sims.shell_edges(chi[0], chi[-1], 200.0)
        kern = lognormal_sims.shell_kernel_weights(edges, chi, wf_l)
        assert kern.shape == (len(edges) - 1, 1)
        np.testing.assert_allclose(
            kern.sum(), np.trapezoid(wf_l[:, 0], chi), rtol=1e-10
        )

    def test_limber_projection_matches_ccl(self, cosmo, kernels):
        # independent Limber shells sum up to the Limber C_ell of the kernels
        _, chi, wf_l, wf_g, tr_l, tr_g = kernels
        lmax = 300
        edges = lognormal_sims.shell_edges(chi[0], chi[-1], 100.0)
        spectra = lognormal_sims.build_shell_spectra(
            cosmo, edges, lmax, ncorr=1, ell_max_nonlimber=-1
        )
        k_l = lognormal_sims.shell_kernel_weights(edges, chi, wf_l)
        k_g = lognormal_sims.shell_kernel_weights(edges, chi, wf_g)
        ell = np.array([20, 50, 100, 300])
        f_ell = lognormal_sims.shear_ell_factor(lmax)
        cl_ll = lognormal_sims.projected_cls(spectra, k_l, k_l, lmax)[:, 0, 0]
        cl_gg = lognormal_sims.projected_cls(spectra, k_g, k_g, lmax)[:, 0, 0]
        cl_gl = lognormal_sims.projected_cls(spectra, k_g, k_l, lmax)[:, 0, 0]
        np.testing.assert_allclose(
            (cl_ll * f_ell**2)[ell], ccl.angular_cl(cosmo, tr_l, tr_l, ell), rtol=0.01
        )
        np.testing.assert_allclose(
            (cl_gl * f_ell)[ell], ccl.angular_cl(cosmo, tr_g, tr_l, ell), rtol=0.02
        )
        np.testing.assert_allclose(
            cl_gg[ell], ccl.angular_cl(cosmo, tr_g, tr_g, ell), rtol=0.03
        )

    def test_nonlimber_sum_rule(self, cosmo):
        # the exact auto- plus cross-spectra of a shell with its neighbours add up to
        # its Limber spectrum: sum_s' C_ss' = C_s^Limber (here at linear scales)
        edges = np.arange(1500.0, 1500.0 + 9 * 150.0 + 1, 150.0)
        ells = np.array([30, 60])
        n_shells = len(edges) - 1
        cl_nl = lognormal_sims.shell_cls_nonlimber(
            cosmo, edges, ells, np.full(n_shells, 100), ncorr=4
        )
        cl_lim = lognormal_sims.shell_cls_limber(cosmo, edges, 60)
        mid = 4
        total = sum(cl_nl[min(mid, sp), max(mid, sp)] for sp in range(mid - 4, mid + 5))
        np.testing.assert_allclose(total, cl_lim[mid][ells], rtol=0.03)
        # neighbouring shells are positively correlated on these scales
        assert np.all(cl_nl[mid, mid + 1] > 0)
        assert np.all(cl_nl[mid, mid] < cl_lim[mid][ells])


@pytest.fixture(scope='module')
def setup(cosmo, kernels):
    """Lognormal and Gaussian-shell generators for the same kernels."""
    pytest.importorskip('glass')
    z, _, wf_l, wf_g, _, _ = kernels
    nside, lmax = 32, 64
    gens = {
        tr: lognormal_sims.build_generator(
            cosmo,
            z,
            wf_l,
            wf_g,
            nside,
            lmax,
            shell_thickness=400.0,
            ncorr=2,
            shift=1.0,
            ell_max_nonlimber=40,
            transform=tr,
        )
        for tr in ('lognormal', 'gaussian')
    }
    return nside, lmax, gens


class TestGenerator:
    def test_maps_shape_and_seed(self, setup):
        nside, _, gens = setup
        maps_gg, maps_ll = gens['lognormal'].make_maps(seed=3)
        assert len(maps_gg) == len(maps_ll) == 1
        assert maps_gg[0].shape == (hp.nside2npix(nside),)
        assert len(maps_ll[0]) == 2
        again_gg, _ = gens['lognormal'].make_maps(seed=3)
        np.testing.assert_array_equal(maps_gg[0], again_gg[0])
        other_gg, _ = gens['lognormal'].make_maps(seed=4)
        assert not np.allclose(maps_gg[0], other_gg[0])

    @pytest.mark.parametrize('transform', ['lognormal', 'gaussian'])
    def test_spectra_match_theory(self, setup, transform):
        _, lmax, gens = setup
        gen = gens[transform]
        nreal = 40
        cl_tt, cl_ee, cl_te, cl_bb = (np.zeros(lmax + 1) for _ in range(4))
        for seed in range(nreal):
            maps_gg, maps_ll = gen.make_maps(seed)
            alm_t = hp.map2alm(maps_gg[0], lmax=lmax, iter=3)
            alm_e, alm_b = hp.map2alm_spin(maps_ll[0], 2, lmax=lmax)
            cl_tt += hp.alm2cl(alm_t) / nreal
            cl_ee += hp.alm2cl(alm_e) / nreal
            cl_te += hp.alm2cl(alm_t, alm_e) / nreal
            cl_bb += hp.alm2cl(alm_b) / nreal
        sel = slice(10, 50)
        # 40 modes x 40 realisations: ~1% scatter on the band average
        for sim, theory in (
            (cl_tt, gen.cl_gg[:, 0, 0]),
            (cl_ee, gen.cl_ll[:, 0, 0]),
            (cl_te, gen.cl_gl[:, 0, 0]),
        ):
            np.testing.assert_allclose(sim[sel].mean(), theory[sel].mean(), rtol=0.05)
        assert cl_bb[sel].mean() < 1e-3 * cl_ee[sel].mean()

    def test_lognormal_is_skewed_gaussian_is_not(self, setup):
        _, _, gens = setup
        skew = {}
        for tr, gen in gens.items():
            m = np.concatenate([gen.make_maps(seed)[0][0] for seed in range(5)])
            skew[tr] = np.mean((m - m.mean()) ** 3) / np.var(m) ** 1.5
        assert skew['lognormal'] > 0.1
        assert abs(skew['gaussian']) < 0.03

    def test_worker_uses_generator(self, setup):
        nside, lmax, gens = setup
        w = np.ones(hp.nside2npix(nside))
        out = cov_partial_sky._compute_one_realization(
            seed=1,
            cl_ring_big_list=None,
            lmax=49,
            lmax_sim=lmax,
            noise_amp_gg=None,
            noise_amp_ll=None,
            nside=nside,
            zbins=1,
            weight_maps_gg=w,
            weight_maps_ll=w,
            nl_gg_diag=np.array([0.0]),
            nl_ll_diag=np.array([0.0]),
            coupled_cls=True,
            spin0=False,
            wsp_path_template='',
            nbl=1,
            ell_min_edges=np.array([10]),
            ell_max_edges=np.array([50]),
            lognormal_gen=gens['lognormal'],
        )
        cl_gg, cl_gl, cl_ll = out
        assert cl_gg.shape == cl_gl.shape == cl_ll.shape == (1, 1, 1)
        assert cl_gg[0, 0, 0] > 0 and cl_ll[0, 0, 0] > 0
        # one full-sky realisation of a 40-multipole band: within ~20% of theory
        gen = gens['lognormal']
        np.testing.assert_allclose(
            cl_gg[0, 0, 0], gen.cl_gg[10:50, 0, 0].mean(), rtol=0.3
        )


class TestLognormalResponse:
    def test_shapes_and_values(self, cosmo):
        resp = object.__new__(responses.SpaceborneResponses)
        resp.k_grid = np.geomspace(1e-3, 1.0, 12)
        resp.z_grid = np.array([0.2, 0.8, 1.5])
        resp.cosmo_ccl = cosmo
        resp.use_h_units = False
        bias = np.array([[1.0, 1.5], [1.2, 1.8], [1.5, 2.2]])
        resp.set_lognormal_resp(shift=0.5, gal_bias_2d=bias)
        assert resp.dPmm_ddeltab.shape == (12, 3)
        assert resp.dPgm_ddeltab.shape == (12, 3, 2)
        assert resp.dPgg_ddeltab.shape == (12, 3, 2, 2)
        np.testing.assert_allclose(resp.dPmm_ddeltab / resp.pk_mm, 4.0)
        np.testing.assert_allclose(
            resp.dPgm_ddeltab[5, 1], 4.0 * resp.pk_mm[5, 1] * bias[1]
        )
        np.testing.assert_allclose(
            resp.dPgg_ddeltab[5, 2], 4.0 * resp.pk_mm[5, 2] * np.outer(bias[2], bias[2])
        )
