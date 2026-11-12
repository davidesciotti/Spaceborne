"""Lognormal realisations of the 3x2pt fields for the ensemble covariance.

The matter density contrast is discretised in nested shells of constant comoving
thickness. Each shell is a (lognormal or Gaussian) random field on the sphere, sampled
with GLASS. The tomographic fields are *linear projections* of the shells with the
same radial kernels used for the theory C_ells:

    f_i^A(n) = sum_s K_i^A(s) delta_s(n),     K_i^A(s) = int_shell dchi W_i^A(chi)

with delta_s the shell-averaged density contrast and W the CCL kernels (lensing +
intrinsic alignment for the shear; density * galaxy bias + magnification for the
galaxies). The simulated spectra are then

    C_ij^AB(ell) = sum_{s, s'} K_i^A(s) K_j^B(s') C_ss'(ell)

i.e. the discretised version of the theory ones, with every ingredient of the config
(n(z), biases, IA, magnification) included consistently.

Shell spectra: at low ell they are computed without the Limber approximation
(linear theory, rescaled by the non-linear boost) and include the correlations between
neighbouring shells; above, Limber auto-spectra only. The low-ell part matters for the
variance of the density averaged over the footprint (i.e. the SSC), which is badly
wrong for thin, uncorrelated Limber shells.

Non-Gaussianity: a lognormal shell delta = shift * (exp(g - var/2) - 1) responds to a
long mode delta_L as P -> P * (1 + delta_L / shift)^2, i.e. the sims carry a
super-sample covariance with a constant response dlnP/ddelta_b = 2 / shift (plus a
smaller in-survey trispectrum). This is *not* the halo-model response: to compare
analytic SSC and sims like-for-like, use covariance.which_pk_responses: lognormal.
"""

import hashlib
import os
import time
import warnings

import healpy as hp
import numpy as np
import pyccl as ccl
from scipy.integrate import cumulative_trapezoid
from scipy.special import spherical_jn

K_MAX_LIMBER = 30.0  # 1/Mpc, cap for the Limber wavenumber (lowest-z shells)


def import_glass():
    try:
        import glass
    except ImportError as err:
        raise ImportError(
            "ensemble_covariance: field_type != 'gaussian' requires the GLASS "
            'package (pip install glass)'
        ) from err
    return glass


def shear_ell_factor(lmax: int) -> np.ndarray:
    """ell-dependent prefactor of the shear (and IA) C_ells, as applied by CCL."""
    ell = np.arange(lmax + 1, dtype=float)
    fac = np.zeros(lmax + 1)
    fac[2:] = (
        np.sqrt((ell[2:] + 2) * (ell[2:] + 1) * ell[2:] * (ell[2:] - 1))
        / (ell[2:] + 0.5) ** 2
    )
    return fac


def shell_edges(chi_min: float, chi_max: float, dchi: float) -> np.ndarray:
    n_shells = max(int(np.ceil((chi_max - chi_min) / dchi)), 1)
    return np.linspace(chi_min, chi_max, n_shells + 1)


def shell_kernel_weights(
    chi_edges: np.ndarray, chi_grid: np.ndarray, wf_arr: np.ndarray
) -> np.ndarray:
    """K[s, i] = integral of the kernel W_i(chi) over shell s.

    wf_arr has shape (len(chi_grid), zbins), chi_grid is increasing.
    """
    cum = cumulative_trapezoid(wf_arr, chi_grid, axis=0, initial=0.0)
    cum_edges = np.array(
        [np.interp(chi_edges, chi_grid, cum[:, zi]) for zi in range(wf_arr.shape[1])]
    ).T
    return np.diff(cum_edges, axis=0)


def _gauss_legendre_nodes(chi_edges: np.ndarray, n_sub: int):
    """Nodes (n_shells, n_sub) and weights (n_sub,) for integrals over each shell;
    the weights are normalised to sum to 1 (shell average)."""
    x, w = np.polynomial.legendre.leggauss(n_sub)
    lo, hi = chi_edges[:-1, None], chi_edges[1:, None]
    return 0.5 * (hi + lo) + 0.5 * (hi - lo) * x[None, :], 0.5 * w


def shell_cls_limber(
    cosmo_ccl, chi_edges: np.ndarray, lmax: int, n_sub: int = 12
) -> np.ndarray:
    """Limber auto-spectra of the shell-averaged matter density, shape
    (n_shells, lmax + 1), with the non-linear matter power spectrum:
    C_s(ell) = 1/dchi^2 * int_shell dchi P((ell + 1/2) / chi, z(chi)) / chi^2
    """
    ell = np.arange(lmax + 1)
    chi_q, w_q = _gauss_legendre_nodes(chi_edges, n_sub)
    a_q = ccl.scale_factor_of_chi(cosmo_ccl, chi_q.ravel()).reshape(chi_q.shape)
    dchi = np.diff(chi_edges)
    cl = np.zeros((len(dchi), lmax + 1))
    for s in range(len(dchi)):
        for q in range(n_sub):
            k = np.minimum((ell + 0.5) / chi_q[s, q], K_MAX_LIMBER)
            pk = ccl.nonlin_matter_power(cosmo_ccl, k, a_q[s, q])
            cl[s] += w_q[q] * pk / chi_q[s, q] ** 2
        cl[s] /= dchi[s]
    return cl


def shell_cls_nonlimber(
    cosmo_ccl,
    chi_edges: np.ndarray,
    ells: np.ndarray,
    ell_max_shell: np.ndarray,
    ncorr: int,
    k_max: float = 0.4,
    n_sub: int = 32,
) -> dict:
    """Exact (non-Limber) spectra of the shell-averaged matter density, including the
    cross-correlations between shells up to `ncorr` neighbours:

    C_ss'(ell) = 2/pi int dk k^2 P_lin(k, z=0) I_s(ell, k) I_s'(ell, k)
    I_s(ell, k) = 1/dchi int_shell dchi D(chi) j_ell(k chi)

    rescaled by the non-linear boost sqrt(B_s B_s'), B_s = P_nl / P_lin at
    k = (ell + 1/2) / chi_s. Shell s is included only for ell <= ell_max_shell[s].

    Returns {(s, s'): array(len(ells))} for s <= s' <= s + ncorr.
    """
    n_shells = len(chi_edges) - 1
    chi_q, w_q = _gauss_legendre_nodes(chi_edges, n_sub)
    a_q = ccl.scale_factor_of_chi(cosmo_ccl, chi_q.ravel())
    growth_q = ccl.growth_factor(cosmo_ccl, a_q).reshape(chi_q.shape)
    chi_c = 0.5 * (chi_edges[1:] + chi_edges[:-1])
    a_c = ccl.scale_factor_of_chi(cosmo_ccl, chi_c)
    growth_c = ccl.growth_factor(cosmo_ccl, a_c)

    # j_ell(k chi) oscillates in k with period 2 pi / chi: 8 samples per period
    dk = 2 * np.pi / (8 * chi_edges[-1])
    k = np.arange(dk, k_max, dk)
    k2pk = k**2 * ccl.linear_matter_power(cosmo_ccl, k, 1.0)

    out = {
        (s, sp): np.zeros(len(ells))
        for s in range(n_shells)
        for sp in range(s, min(s + ncorr, n_shells - 1) + 1)
    }
    for ix, ell in enumerate(ells):
        act = np.where(ell_max_shell >= ell)[0]
        if len(act) == 0:
            continue
        x = k[:, None, None] * chi_q[act][None, :, :]
        radial = np.einsum(
            'ksq,sq->sk', spherical_jn(int(ell), x), growth_q[act] * w_q[None, :]
        )
        boost = np.array(
            [
                ccl.nonlin_matter_power(
                    cosmo_ccl, min((ell + 0.5) / chi_c[s], K_MAX_LIMBER), a_c[s]
                )
                / (
                    growth_c[s] ** 2
                    * ccl.linear_matter_power(
                        cosmo_ccl, min((ell + 0.5) / chi_c[s], K_MAX_LIMBER), 1.0
                    )
                )
                for s in act
            ]
        )
        pos = {s: i for i, s in enumerate(act)}
        for s in act:
            for sp in range(s, min(s + ncorr, n_shells - 1) + 1):
                if sp not in pos:
                    continue
                i, j = pos[s], pos[sp]
                out[s, sp][ix] = (
                    2 / np.pi * np.sum(k2pk * radial[i] * radial[j]) * dk
                ) * np.sqrt(boost[i] * boost[j])
    return out


def build_shell_spectra(
    cosmo_ccl,
    chi_edges: np.ndarray,
    lmax: int,
    ncorr: int,
    ell_max_nonlimber: int,
    k_nonlimber: float = 0.1,
    cache_dir: str | None = None,
) -> dict:
    """Spectra of all shell pairs up to ncorr neighbours, {(s, s'): array(lmax + 1)}.

    For each shell, ell <= min(ell_max_nonlimber, k_nonlimber * chi_s) is non-Limber
    (and correlated with the neighbours); above, Limber auto-spectrum, no correlation.
    k_nonlimber [1/Mpc] keeps the non-Limber part in the (quasi-)linear regime.
    """
    n_shells = len(chi_edges) - 1
    cl_limber = shell_cls_limber(cosmo_ccl, chi_edges, lmax)
    spectra = {
        (s, sp): (cl_limber[s].copy() if s == sp else np.zeros(lmax + 1))
        for s in range(n_shells)
        for sp in range(s, min(s + ncorr, n_shells - 1) + 1)
    }
    ell_max_nonlimber = min(ell_max_nonlimber, lmax)
    if ell_max_nonlimber < 0:
        return spectra

    # non-Limber up to ell_max_shell; shells are *evaluated* up to twice that (the
    # transition zone) to get the cross-spectra of the pairs straddling the transition
    ell_max_shell = np.minimum(
        ell_max_nonlimber, np.floor(k_nonlimber * chi_edges[:-1])
    ).astype(int)
    ell_max_eval = np.minimum(
        ell_max_nonlimber, np.floor(2 * k_nonlimber * chi_edges[:-1])
    ).astype(int)
    ells = np.unique(
        np.concatenate(
            (
                np.arange(min(12, ell_max_nonlimber) + 1),
                np.geomspace(12, max(ell_max_nonlimber, 12), 36).astype(int),
            )
        )
    )
    ells = ells[ells <= ell_max_nonlimber]

    cache_file = None
    if cache_dir is not None:
        key = hashlib.sha1(
            np.concatenate(
                (
                    chi_edges,
                    ells,
                    ell_max_shell,
                    ell_max_eval,
                    [ncorr],
                    [ccl.sigma8(cosmo_ccl), cosmo_ccl['Omega_m'], cosmo_ccl['h']],
                )
            ).tobytes()
        ).hexdigest()[:16]
        cache_file = f'{cache_dir}/lognormal_shell_cls_{key}.npz'
    if cache_file is not None and os.path.exists(cache_file):
        loaded = np.load(cache_file)
        cl_nl = {
            tuple(int(x) for x in name.split('_')): loaded[name] for name in loaded
        }
    else:
        start = time.perf_counter()
        cl_nl = shell_cls_nonlimber(cosmo_ccl, chi_edges, ells, ell_max_eval, ncorr)
        print(
            f'Non-Limber shell spectra ({n_shells} shells, {len(ells)} multipoles) '
            f'computed in {time.perf_counter() - start:.1f} s'
        )
        if cache_file is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_file, **{f'{s}_{sp}': v for (s, sp), v in cl_nl.items()})

    # - both shells non-Limber at ell: exact auto- and cross-spectra
    # - pair straddling the transition (one shell non-Limber, the other Limber): half
    #   the exact cross-spectrum. The Limber auto-spectrum of a shell stands for the
    #   sum of its exact auto- and cross-spectra, so the full cross-term would be
    #   counted twice and none would leave the non-Limber shell short of it; half
    #   keeps the projected spectra continuous across the transition
    # - both Limber: auto-spectra only
    ell_all = np.arange(lmax + 1)
    for (s, sp), vals in cl_nl.items():
        lo, hi = sorted((ell_max_shell[s], ell_max_shell[sp]))
        exact = np.interp(ell_all, ells, vals)
        both = ell_all <= lo
        spectra[s, sp][both] = exact[both]
        if s != sp:
            straddle = (ell_all > lo) & (ell_all <= min(hi, ell_max_eval[s]))
            spectra[s, sp][straddle] = 0.5 * exact[straddle]
    return spectra


def projected_cls(
    spectra: dict, kern_a: np.ndarray, kern_b: np.ndarray, lmax: int
) -> np.ndarray:
    """C_ij^AB(ell) = sum_{s, s'} K_i^A(s) K_j^B(s') C_ss'(ell); shape (lmax+1, zi, zj).
    kern_a, kern_b have shape (n_shells, zbins)."""
    cl = np.zeros((lmax + 1, kern_a.shape[1], kern_b.shape[1]))
    for (s, sp), c in spectra.items():
        w = np.outer(kern_a[s], kern_b[sp])
        if sp != s:
            w = w + np.outer(kern_a[sp], kern_b[s])
        cl += c[:, None, None] * w[None, :, :]
    return cl


class LognormalGenerator:
    """Samples the tomographic delta_g and shear maps from lognormal matter shells.

    Parameters
    ----------
    spectra : {(s, s'): C_ss'(ell)} from `build_shell_spectra`, arrays of len lmax + 1
    kern_ll, kern_gg : (n_shells, zbins) shell weights of the lensing and galaxy kernels
    nside, lmax : resolution and band limit of the simulated signal
    ncorr : number of correlated neighbouring shells
    shift : lognormal shift (1 = standard lognormal matter field)
    transform : 'lognormal' or 'gaussian' (Gaussian shells: same two-point statistics,
        no non-Gaussianity. The null test of the lognormal ensemble)
    cache_dir : where to cache the Gaussian spectra of the shells (None: no cache)
    """

    def __init__(
        self,
        spectra: dict,
        kern_ll: np.ndarray,
        kern_gg: np.ndarray,
        nside: int,
        lmax: int,
        ncorr: int,
        shift: float = 1.0,
        transform: str = 'lognormal',
        cache_dir: str | None = None,
    ):
        assert transform in ('lognormal', 'gaussian'), transform
        glass = import_glass()
        self.n_shells, self.zbins = kern_ll.shape
        self.kern_ll = kern_ll
        self.kern_gg = kern_gg
        self.nside = nside
        self.lmax = lmax
        self.ncorr = ncorr
        self.shift = shift
        self.transform = transform
        self.f_ell = shear_ell_factor(lmax)

        # Gaussian spectra that give the target ones after the transformation, in
        # GLASS order: (0,0), (1,1), (1,0), (2,2), (2,1), (2,0), ...
        # [Note] glass.solve_gaussian_spectra warns about non-convergence for the
        # cross-spectra, because its *relative* tolerance is ill-defined where they
        # vanish (above the non-Limber range); check the absolute accuracy instead.
        # The solver starts from the closed-form solution, already accurate to
        # machine precision for the cross-spectra: a few iterations are enough.
        fields = self._fields(glass)
        pairs = [(j, i) for i in range(self.n_shells) for j in range(i, -1, -1)]
        cache_file = None
        if cache_dir is not None:
            digest = hashlib.sha1()
            for j, i in pairs:
                if i - j <= ncorr:
                    digest.update(np.ascontiguousarray(spectra[j, i]).tobytes())
            digest.update(f'{transform}_{shift}_{ncorr}_{lmax}'.encode())
            cache_file = f'{cache_dir}/lognormal_gls_{digest.hexdigest()[:16]}.npz'
        if cache_file is not None and os.path.exists(cache_file):
            loaded = np.load(cache_file)
            self.gls = [loaded[f'gl_{n}'] for n in range(len(pairs))]
            worst = float(loaded['worst'])
        else:
            self.gls = []
            worst = 0.0
            for j, i in pairs:
                if i - j > ncorr or not np.any(spectra[j, i]):
                    self.gls.append(np.zeros(0))
                    continue
                cl = spectra[j, i]
                with warnings.catch_warnings(), np.errstate(all='ignore'):
                    warnings.simplefilter('ignore')
                    gl, cl_out, _ = glass.grf.solve(
                        cl,
                        fields[i],
                        fields[j],
                        pad=2 * len(cl),
                        maxiter=5,
                        monopole=0.0 if cl[0] == 0 else None,
                    )
                ref = np.sqrt(spectra[i, i] * spectra[j, j])[2:]
                err = np.abs(np.asarray(cl_out)[: len(cl)] - cl)[2:] / ref
                worst = max(worst, float(np.max(err)))
                self.gls.append(np.asarray(gl))
            if cache_file is not None:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(
                    cache_file,
                    worst=worst,
                    **{f'gl_{n}': gl for n, gl in enumerate(self.gls)},
                )
        if worst > 1e-3:
            warnings.warn(
                'lognormal_sims: the Gaussian spectra of the shells reproduce the '
                f'target ones only to {worst:.1e} (relative to the auto-spectra)',
                stacklevel=2,
            )

        # expected spectra of the simulated fields (discretised theory)
        self.cl_ll = projected_cls(spectra, kern_ll, kern_ll, lmax) * (
            self.f_ell[:, None, None] ** 2
        )
        self.cl_gl = (
            projected_cls(spectra, kern_gg, kern_ll, lmax) * self.f_ell[:, None, None]
        )
        self.cl_gg = projected_cls(spectra, kern_gg, kern_gg, lmax)

    def _fields(self, glass):
        if self.transform == 'lognormal':
            return [glass.grf.Lognormal(self.shift) for _ in range(self.n_shells)]
        return [glass.grf.Normal() for _ in range(self.n_shells)]

    def make_maps(self, seed, pw_t=None, pw_e=None, n_iter: int = 1):
        """One realisation. Returns (list of zbins T maps, list of zbins [Q, U] maps),
        as expected by cov_partial_sky._compute_one_realization.

        pw_t, pw_e: pixel windows to apply to the fields (see pixwin_nside), or None.
        """
        glass = import_glass()
        rng = np.random.default_rng(seed)
        npix = hp.nside2npix(self.nside)
        kappa = np.zeros((self.zbins, npix))
        delta_g = np.zeros((self.zbins, npix))

        shells = glass.generate(
            self._fields(glass), self.gls, self.nside, ncorr=self.ncorr, rng=rng
        )
        for s, shell in enumerate(shells):
            delta = np.asarray(shell)
            for zi in range(self.zbins):
                if self.kern_ll[s, zi] != 0.0:
                    kappa[zi] += self.kern_ll[s, zi] * delta
                if self.kern_gg[s, zi] != 0.0:
                    delta_g[zi] += self.kern_gg[s, zi] * delta

        maps_gg, maps_ll = [], []
        fl_e = self.f_ell if pw_e is None else self.f_ell * pw_e[: self.lmax + 1]
        for zi in range(self.zbins):
            alm_e = hp.almxfl(hp.map2alm(kappa[zi], lmax=self.lmax, iter=n_iter), fl_e)
            maps_ll.append(
                hp.alm2map_spin(
                    [alm_e, np.zeros_like(alm_e)],
                    nside=self.nside,
                    spin=2,
                    lmax=self.lmax,
                )
            )
            if pw_t is None:
                maps_gg.append(delta_g[zi])
            else:
                alm_t = hp.map2alm(delta_g[zi], lmax=self.lmax, iter=n_iter)
                maps_gg.append(
                    hp.alm2map(
                        hp.almxfl(alm_t, pw_t[: self.lmax + 1]),
                        self.nside,
                        lmax=self.lmax,
                    )
                )
        return maps_gg, maps_ll


def build_generator(
    cosmo_ccl,
    z_grid: np.ndarray,
    wf_lensing: np.ndarray,
    wf_galaxy: np.ndarray,
    nside: int,
    lmax: int,
    shell_thickness: float,
    ncorr: int,
    shift: float,
    ell_max_nonlimber: int,
    transform: str = 'lognormal',
    cache_dir: str | None = None,
) -> LognormalGenerator:
    """Builds the generator from the radial kernels on z_grid (shape (z, zbins), CCL
    convention, including galaxy bias / IA / magnification where requested)."""
    chi_grid = ccl.comoving_radial_distance(cosmo_ccl, 1 / (1 + z_grid))
    chi_edges = shell_edges(chi_grid[0], chi_grid[-1], shell_thickness)
    kern_ll = shell_kernel_weights(chi_edges, chi_grid, wf_lensing)
    kern_gg = shell_kernel_weights(chi_edges, chi_grid, wf_galaxy)
    spectra = build_shell_spectra(
        cosmo_ccl, chi_edges, lmax, ncorr, ell_max_nonlimber, cache_dir=cache_dir
    )
    gen = LognormalGenerator(
        spectra,
        kern_ll,
        kern_gg,
        nside,
        lmax,
        ncorr,
        shift=shift,
        transform=transform,
        cache_dir=cache_dir,
    )
    gen.chi_edges = chi_edges
    return gen
