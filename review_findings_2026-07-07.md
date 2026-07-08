# Spaceborne scientific-correctness review — 2026-07-07

Scope: scientific correctness (equations, numerical assembly, external-library calls),
with a deep dive on the partial-sky Gaussian covariance (`cov_partial_sky.py`,
`cov_harmonic_space.py`, `mask_utils.py`, Knox/iNKA/ensemble consistency), plus a
Sonnet-subagent sweep of the rest of the science pipeline. References: arXiv:2410.06962
(OneCovariance paper), TJPCov (`covariance_fourier_gaussian_nmt.py`), OneCovariance
(`onecov/cov_ell_space.py`), pymaster 2.6 (numerical spot checks run in this env).

Coverage: every module in the science pipeline was read in full (the real-space
reviewer initially hit a session limit and was re-run to completion).

## Top-priority summary

1. ~~**[critical]** `cov_partial_sky.py:1172-1180` — MCM extraction slices
   `get_coupling_matrix()[:nbl_unb, :nbl_unb]`, but NaMaster interleaves spectra
   components (`index = ℓ·n_cls + i_cl`). The GL (n_cls=2) and LL (n_cls=4) MCMs are
   scrambled → every **coupled SSC/cNG** term and the saved
   `mode_coupling_matrices.npz` (gl/ll) are wrong. Verified numerically.~~
2. **[critical]** `ell_utils.py:279-289` — `ell_edges.astype(int)` truncates log-spaced
   edges before `NmtBin.from_edges`; fine/low-ℓ binning yields duplicate edges → opaque
   NaMaster crash, and silently shifted bin edges even when it doesn't crash.
3. ~~**[warning]** `cov_harmonic_space.py:453-459` — `_couple_cov_ng` early-returns unless
   **both** SSC and cNG are requested (`or` instead of `and`): a `cov_type: coupled` run
   with only SSC (or only cNG) silently leaves the NG term uncoupled.~~
4. ~~**[warning]** `cov_partial_sky.py:1289-1311` — shape noise is fed **only to EE**
   (`cl_bb = 0`) in both the analytic-NaMaster and ensemble paths. TJPCov adds shape
   noise to EE **and BB** (`SN[i][0] = SN[i][-1]`); physically, white ellipticity noise
   has equal E/B power, and mask-induced B→E mixing feeds BB noise into the EE
   covariance. SB's two paths are mutually consistent but both underestimate the noise
   part of the masked shear covariance.~~
5. ~~**[warning]** `cov_ssc.py:42-49` — `sigma2_z1z2_fft`'s cosine transform omits the
   `exp(-i·r·k_min)` phase for a grid starting at `k_min ≠ 0`. Verified numerically:
   0.01–1% error on C(r), growing with r (r·k_min ≈ 0.13 at r ~~ 13 Gpc for
   k_min = 1e-5). Biases σ²_b mostly for widely separated z-pairs; one-line fix.~~
6. ~~**[warning]** `cov_partial_sky.py:167-169` — with `spin0: True`, **all** probe blocks
   are decoupled with `w00_dict`, which is built from the **clustering** masks; if the
   LL and GG masks differ, shear blocks are decoupled with the wrong mask (while the
   iNKA rescale correctly uses the LL masks — internally inconsistent).~~
7. **[warning]** `responses.py:645-647` vs `main.py:1481-1483` —
   `which_pk_responses: 'separate_universe'` produces 2D `dPgm/dPgg` arrays that fail
   `dPxx_ddeltab_klimber`'s 3D/4D shape asserts: the documented option crashes for
   tomographic runs.
8. **[warning]** `ccl_interface.py:333-339` — when `has_rsd: True`, the RSD kernel
   component is dropped from `wf_delta_arr` (only density + magnification extracted),
   so the SSC response kernels would be inconsistent with the Gaussian Cls (latent:
   default is `has_rsd: False`).
9. ~~**[resolved question]** `cov_real_space.py:128-155` — `t_sn` (live) is **correct**
   for the mixed γt×γt case; `_t_sn` (diagonal-only) is wrong. See §Real space below.~~
10. ~~**[warning]** `sb_lib.py:1932` — `symmetrize_2d_array`'s triangle check
    `np.all(x) == 0` is a non-functional validator (means "at least one zero"), so a
    partially-filled-triangle indexing bug elsewhere would pass silently.~~

---

## Partial-sky Gaussian covariance (deep dive)

### Confirmed correct (checked against paper / TJPCov / NaMaster conventions)

- **iNKA input spectra** (`build_psky_cov:1263-1303`): signal passed as
  `couple_cell(Cl) / mean(w_a·w_b)` with the correct per-bin-pair mask moments and the
  correct field pairing (w00 = gg(zi)×gg(zj), w02 = gg(zi)×ll(zj), w22 = ll(zi)×ll(zj));
  matches TJPCov `get_cl_for_cov` exactly.
- **Flat noise addition** (`:1305-1308`): adding the unmasked white `N_ℓ` (not divided
  by fsky) is algebraically identical to TJPCov's `coupled_noise/mean(w_a w_b)` for
  homogeneous noise, since `couple_cell(N)_ℓ ≈ N·mean(w²)`. Consistent with CLAUDE.md.
  (The unresolved inhomogeneous-noise question is a physics-input issue, not a bug.)
- **`nmt.gaussian_covariance` wiring** (`nmt_gaussian_cov:216-229`): the four Cl
  arguments correctly map to pairs (a1b1)=(zi,zk), (a1b2)=(zi,zl), (a2b1)=(zj,zk),
  (a2b2)=(zj,zl) with the right probe orientation (`cl_et = cl_te.T` for the '20' key);
  spin-block reshape `(nell, n_ab, nell, n_cd)` and `[:, 0, :, 0]` slice extract the
  EE/TE/TT-type element correctly.
- **Ensemble path**: `build_cl_tomo_TEB_ring_ord` produces exactly healpy `synalm
  new=True` diagonal-major ordering (offset-major loop, T/E/B interleaved per bin;
  transposes for the lower-triangle labels are correct); pseudo-Cl estimator (alm2cl of
  masked maps / compute_coupled_cell) matches the `coupled` analytic covariance;
  `np.cov(..., bias=False)` is the unbiased N−1 estimator; decoupled branch uses the
  same per-bin-pair workspaces as the analytic covariance.
- **Binning consistency across the three methods**: `bin_2d_array_vectorized`'s 'sum'
  mode computes the weighted band **mean** normalized by `(Σw_i)(Σw_j)` — identical to
  `NmtBin.bin_cell` uniform-weight averaging used to bin the ensemble pseudo-Cls, and to
  `_bin_cov_hs_g_diag`'s `Σ/n_modes²` used by the Knox path. Bin edges `[lo, hi)`
  conventions agree between `NmtBin.from_edges(edges[:-1], edges[1:])`, ensemble worker
  edge reconstruction (`get_ell_max(i)+1`), and the masks in both binning helpers.
- **Knox path** (`compute_g_cov`, `cov_g_terms_helper_jax`): prefactor
  `1/((2ℓ+1)·fsky·Δℓ)` and term structure `C^AC_ik C^BD_jl + C^AD_il C^BC_jk` (with the
  4-term signal×noise MIX) match arXiv:2410.06962 Eq. (46)-type expression and
  OneCovariance `__calc_prefac_covELL`; per-probe-pair `max(fsky_ab, fsky_cd)` matches
  OC's `max(survey_area)` convention for cross blocks.
- **Noise amplitudes** (`build_noise`): `N_LL = σ²_ε,tot/(2n̄) = σ²_ε,1/n̄`,
  `N_GG = 1/n̄`, `N_GL = 0`, z-diagonal; `arcmin⁻² → sr⁻¹` conversion correct
  (the inline comment "deg^2 to arcmin^2" is wrong, the number is right — use
  `constants.SR_TO_ARCMIN2`). Matches the paper's Eq. (44) definitions.
- **`couple_cov_6d_tomo`** einsum applies `M_ab · cov · M_cd^T` with per-bin-pair MCMs
  correctly; `bin_mcm` follows the standard NaMaster binned-MCM recipe.
- **Probe-block symmetrization**: 4d transpose `(1,0,3,2)` / 6d `(1,0,4,5,2,3)`
  correctly swap ell and z axes together (verified independently by two reviewers).
- **fsky bookkeeping** (`mask_utils.py`): binary-footprint `mean(w_a·w_b)` for
  Knox/SSC vs per-bin-pair weight-map moments for iNKA are deliberately distinct and
  correctly separated; σ²_b mask-Cl normalization `(2ℓ+1)/((4π)² fsky_ab fsky_cd)`
  matches the TJPCov convention.

### Findings

- **[critical] `spaceborne/cov_partial_sky.py:1172-1180`** — `compute_and_save_mcms`
  extracts the unbinned MCM as `get_coupling_matrix()[:nbl_unb, :nbl_unb]`. NaMaster
  stores the coupling matrix with **interleaved** components, `index = ℓ·n_cls + i_cl`
  (verified numerically with pymaster 2.6: the interleaved reading reproduces
  `couple_cell` to 1e-15; the `[:n, :n]` slice and the block reading are 100% off).
  For w00 (n_cls=1) the slice is the full matrix and correct; for w02 (n_cls=2) and w22
  (n_cls=4) the extracted "TE" and "EE" MCMs are scrambled mixtures of components over
  a truncated ℓ range. Downstream this corrupts `mcm_te_binned` / `mcm_ee_binned`,
  hence `_couple_cov_ng` (`cov_harmonic_space.py:511-523`): every **coupled SSC/cNG**
  block involving GL or LL is wrong, as is the saved `mode_coupling_matrices.npz`
  (gl/ll entries). Fix: extract `M[0::n_cls, 0::n_cls]` (the EE→EE / TE→TE part; note
  this deliberately drops the EE←BB coupling block, consistent with the code's
  EE-only NKA treatment — worth a comment). Spot-check script:
  scratchpad `check_mcm_ordering.py`.

- **[warning] `spaceborne/cov_partial_sky.py:1289-1311` (and `:1397-1399`)** — shape
  noise enters only the EE spectrum; `cl_bb_4covnmt = 0` (analytic) and `cl_BB = 0`
  (ensemble synalm input). White ellipticity noise has equal power in E and B; TJPCov
  sets the spin-2 noise in components 0 **and** 3 (EE and BB,
  `covariance_fourier_gaussian_nmt.py:279`). Under mask coupling, the BB noise
  contributes to the EE pseudo-Cl covariance, so both SB paths underestimate the noise
  contribution to the masked shear covariance (they agree with each other, so this does
  **not** explain an analytic-vs-ensemble discrepancy — but both differ from a real
  survey and from TJPCov). Suggested direction: add `N_ℓ` to `cl_bb_4covnmt` and to the
  ensemble `cl_BB` input (and optionally zero the ℓ=0,1 entries of the spin-2 noise as
  TJPCov does).

- **[warning] `spaceborne/cov_harmonic_space.py:453-459`** — `_couple_cov_ng` returns
  early if `'ssc' not in req_terms` **or** `'cng' not in req_terms`. `req_terms`
  (main.py:496-508) contains each NG term independently, so a coupled-covariance run
  with only SSC (or only cNG) requested silently skips the NG coupling — the G term is
  coupled, the NG term is not. The per-term `if ng_term not in self.cov_dict: continue`
  inside the loop shows the intended behavior. Fix: return only when *neither* term is
  requested.

- **[warning] `spaceborne/cov_partial_sky.py:167-169`** — in `spin0` mode,
  `wsp_spin0_dict` maps the '02' and '22' keys to `w00_dict`, which is built from the
  **GG (clustering) masks** (`build_wsp:895-901`, fields from `weight_maps_gg`). If the
  LL and GG masks/weight maps differ, LL and GL blocks are decoupled (and, for
  `coupled=True`, normalized) with the wrong mask's mode-coupling, while the iNKA
  rescale (`build_psky_cov:1295-1303`) uses the correct per-probe masks. Suggested
  direction: build spin-0 workspaces per probe-mask pair for spin0 mode (or assert
  masks are equal when `spin0=True`).

- **[suggestion] `spaceborne/cov_partial_sky.py:1376-1378`** — the printed datavector
  length `(zbins**2 + (zbins + 1) // 2 * 2) * nbl` is wrong (missing `zbins *`:
  should be `zbins**2 + zbins*(zbins+1)//2 * 2` for GL + LL + GG). Print-only.

- **[suggestion] `spaceborne/cov_partial_sky.py:1206-1220`** — the whole NaMaster path
  uses `nmt_bin_obj_GC` and the GC/3x2pt ℓ-range for all probes (author-flagged TODO).
  If WL and GC binning ever differ, LL blocks silently inherit the GC bands. Worth a
  hard assert until properly generalized.

- **[suggestion] `spaceborne/cov_partial_sky.py:848-852`** — the cache-reuse warning is
  the only guard on `load_cached_nmt_workspaces`; nothing checks that the cached
  workspaces match the current mask/nside/binning (a stale cache silently changes
  results). Suggested direction: store a small metadata sidecar (mask hash, nside,
  edges) and validate on load.

## SSC / cNG / responses

- **[warning] `spaceborne/cov_ssc.py:42-49`** — the FFT-based cosine transform treats
  `Re[rfft(P)]·dk` as `∫P(k)cos(kr)dk` on a grid starting at `k_min ≠ 0`; the phase
  factor `exp(-i·r·k_min)` is missing. Verified numerically (toy P(k), same nk/k-range):
  relative error on C(r) grows from ~1e-4 at r≈500 Mpc to ~1% at r≈13 Gpc
  (`r·k_min ≈ 0.13`). Affects the default path (`which_sigma2_b='from_input_mask'`,
  no KE approximation), mostly σ²_b(z1,z2) for widely separated z-pairs (via r_plus and
  large r_minus). Fix: `c_r = (np.exp(-1j*r_grid*k_min) * fft_coeffs).real`.
  Spot-check: scratchpad `check_fft_phase.py`.

- **[warning] `spaceborne/responses.py:336-341` + `main.py:1481-1483` vs
  `responses.py:645-647`** — `set_su_resp` produces 2D `(k, z)` `dPgm/dPgg` arrays
  (bias is `b1_arr[None, :]`), but `dPxx_ddeltab_klimber` asserts 3D/4D per-bin shapes:
  `which_pk_responses: 'separate_universe'` (documented in config.yaml) crashes for any
  tomographic run. Loud failure, but a broken documented option.

- **[warning] `spaceborne/responses.py:507-524` vs `:581-609`** — with
  `include_terasawa_terms: True` and the default `which_b1g_in_resp: 'from_input'`,
  `trsw_gm`/`trsw_gg` are computed but never used (only `trsw_mm` enters via
  `dPmm_ddeltab`); the flag silently applies only part of the correction
  (author-flagged TODO at :580).

- **[warning] `main.py:1550` vs `spaceborne/cov_ssc.py:77-81`** —
  `which_sigma2_b='flat_sky'` is dispatched as valid by main.py for the Spaceborne SSC
  path but `sigma2_z1z2_fft` raises on it (and has no flat-sky branch). Loud, but
  inconsistent dispatch.

- **[warning] `main.py:365`** — `cfg['covariance']['which_sigma2_b']` is hardcoded to
  `'from_input_mask'` after config load, so the other documented values (and the
  config_checker validation for them) are unreachable from the YAML.

- **[suggestion] `spaceborne/responses.py:155-201`** — `I_2_1_dav` docstring shows the
  first-order-bias `I¹₁` integral, but the implementation swaps in the **second-order**
  halo bias before integrating. Code appears intentional; docstring misleads.

- **[suggestion] `spaceborne/cov_ssc.py:52-60`** — `c_0 = simps(Pk0, k_grid)` as lower
  `fill_value` is dead code (r=0 is inside the interpolation domain); harmless but
  misleading.

## Cl / kernel inputs (`wf_cl_lib.py`, `ccl_interface.py`)

Verified consistent: NLA IA sign/normalization vs CCL internals (`use_A_ia=False`
bypass correct); magnification kernel extraction (CCL bakes in `5s-2`); galaxy-bias
re-multiplication onto the bare density kernel; GL 5D index conventions and the
`(1+m)` multiplicative-bias application per leg.

- **[warning] `spaceborne/ccl_interface.py:333-339`** — `wf_delta_arr` takes
  `get_kernel()` component 0 and (if magnification) component −1; the RSD component
  (index 1 when `has_rsd=True`) is never included, so the SSC response kernels would
  silently omit RSD while the Gaussian Cls include it. Latent (`has_rsd: False`
  default); add a guard/warning.
- **[suggestion] `spaceborne/ccl_interface.py:244-251`** — comment claims
  `mag_bias_tuple=None` crashes pyccl, then the code does exactly that; reconcile
  against the pinned pyccl version.
- **[suggestion] `spaceborne/ccl_interface.py:772-806`** — covariance ℓ1↔ℓ2 symmetry
  self-check uses `rtol=0.1` and swallows failures (print-only); a real indexing
  asymmetry would pass.

## Real space

- **[resolved] `spaceborne/cov_real_space.py:128-155` vs `:158-184`** — the open
  `t_sn` vs `_t_sn` question: **`t_sn` (live) is correct; `_t_sn` is wrong.** In
  `cov_sn_rs` (:802-823) the tomographic Kronecker deltas are applied *separately* via
  the `get_delta_tomo` product (`δ_ik δ_jl + δ_il δ_jk`), so `t_sn` must supply only
  the noise amplitude for the (i,j) pair. For γt(ij) with lens i, source j, the SN term
  is `δ_ik δ_jl · σ²_ε1(source j) / N_pair(ij)` — nonzero for **all** (i,j), including
  i≠j, because the identical lens–source galaxy pairs enter both measurements. This is
  exactly the harmonic-space analog `Cov_SN(C^GL_ij, C^GL_kl) = δ_ik δ_jl · N^gg_i ·
  N^εε_j` (OneCovariance `__covELL_split_gaussian`, gmgm sn term: `noise_g^ik ·
  noise_kappa^jl`, nonzero for i≠j; the `1/N_pair ∝ 1/(n_i n_j)` factor carries the
  shot-noise leg). `_t_sn`'s diagonal-only fill would zero the shape noise for every
  cross lens–source pair — the dominant γt configurations. Suggested direction: delete
  `_t_sn`, flip the strict-xfail test into a regression test pinning `t_sn`.
- Verified: `t_mix/(n_eff·SR_TO_ARCMIN2)` reproduces `N_ℓ = σ²_ε1/n̄_sr` (matches
  `build_noise` conventions); `cov_sn_rs` pair-count normalization uses the correct
  per-leg densities.

Full-file sweep (verified consistent): `b_mu`/`k_mu` bin-averaging kernels
(Joachimi+08 Eq. E.2; antiderivative constant cancels in the edge difference, checked
numerically against direct quadrature); `MU_DICT` Bessel orders used consistently
everywhere; SN Wick structure and area/unit conventions match OneCovariance's
`__covTHETA_split_gaussian` including the all-shear factor of 2; MIX einsum matches
`proj_cov_mix_simps`; SSC/cNG real-space normalization `1/(4π²)` with **no** extra
`1/A` is correct because SB's harmonic NG covs are already fsky-normalized upstream
(CCL `fsky=` kwarg) — a different pipeline split than OC, not a bug; FFTLog `mu/nu`
axis assignment into `TwoBessel.two_Bessel_binave` is correct; the
upper-triangle-only fills are safe given how `cov_6D_to_4D_blocks` reads `ind` entries.

- **[warning] `spaceborne/cov_real_space.py:1305-1339`** — the θ re-binning branch
  reads `getattr(self, f'cov_{term}_rs_6d')`, but the per-term 6D covariances are only
  ever *local* variables (`cov_out_6d`), never stored under those attribute names.
  Currently unreachable only because `_set_theta_binning` (:731) hardcodes
  `nbt_fine = nbt_coarse`; decoupling them would raise `AttributeError` immediately.
  Point the rebinner at `cov_out_6d`.
- **[warning] `spaceborne/cov_real_space.py:1181, 1207`** — the MIX-term skip uses
  OneCovariance-style probe names (`'ggxim'`, `'ggxip'`) that can never match SB's
  probe-combo strings (`'wxim'`, `'wxip'`): the shortcut is permanently dead and the
  w×ξ± MIX blocks are always computed. The result still comes out zero (the
  `get_delta_tomo(1,0)` factors vanish), so numbers are unaffected — but the guard is
  silently defeated and relies on that cancellation. Comment at :1211-1212 has the
  same stale names.
- **[warning] `spaceborne/cov_real_space.py:59-127` + `:1052-1100`** — numerical
  fragility in the μ=4 (ξ₋) Levin bin-averaged path: `b_mu(x, 4) = (x−8/x)J₁(x) −
  8J₂(x)` is stable as a single expression, but `b_mu_nobessel`/`kmuknu_nobessel`
  split it into separate single-Bessel terms integrated by independent pylevin calls
  and summed afterwards. At small `x = ℓθ_l` the `8/x` pieces are large and must
  cancel between terms, so per-term `rel_acc` does not bound the error of the
  cancelled sum — the ξ₋ covariance at the smallest θ / lowest ℓ can be much less
  accurate than requested when `levin_bin_avg: true`. This is a plausible mechanism
  for the previously observed "Levin disagrees for μ≥2" precision issue. Suggested
  direction: pass the recombined kernel to Levin, or validate against
  simps at the smallest θ bin.

## Projection / COSEBIs / FFTLog

Verified by the subagent: the FFTLog Gaussian diagonal fix (`integrand / (ℓ²·dlnℓ)`
feeding `proj_cov_2d_fftlog`'s `ℓ₁²ℓ₂²`) is in place and dimensionally consistent;
COSEBIs shape-noise structure matches OneCovariance.

- **[warning] `spaceborne/cov_cosebis.py:121-197`** — `cov_sn_cs` hardcodes the shear
  shape-noise model but is dispatched for any `probe_ab == probe_cd`, including the
  (currently disabled) `Psigl`/`Psigg` clustering statistics — wrong noise model if
  ever enabled.
- **[warning] `spaceborne/cov_projector.py:503-509`** — MIX prefactor divides by
  `n_eff` with no zero/negative guard; a zero-density bin propagates inf/NaN silently.
- **[suggestion] `spaceborne/cov_cosebis.py:99-119`** — no check that the external
  W_ell mode keys are complete/contiguous; silent misalignment risk.
- **[suggestion]** docstring/formula mismatches: `cov_projector.py:109-127`
  (`proj_cov_2d`), `twobessel_fang.py:326-332` (`g_l_smooth`) — code correct, docs not.

## Assembly / reshaping / cross-cutting

Verified consistent (two independent reviewers + numerical checks): triu/tril ×
row/col-major `ind` generation; `cov_4D_to_2D` optimized vs loop equivalence; 2D
ordering options genuinely distinct and internally consistent; `FrozenDict` block
allocation honors `probe_selection`; probe-index conventions uniform across modules;
unbinned ℓ-grids match between `ell_utils` and `cov_partial_sky`.

- **[critical] `spaceborne/ell_utils.py:279-289`** — `astype(int)` truncation of
  log-spaced edges before `NmtBin.from_edges`: duplicate edges for fine/low-ℓ binning
  (reproduced live: opaque `ZeroDivisionError` inside NaMaster), and silent sub-bin
  edge shifts even when it doesn't crash. `compute_ells_oc` guards with `np.unique`;
  this path doesn't. Round (or unique+assert) instead of truncating.
- **[warning] `spaceborne/sb_lib.py:1932`** — `assert np.all(triu) == 0 or
  np.all(tril) == 0` does not test "all zero" (`np.all(x) == 0` ⇒ "at least one
  falsy"); the symmetrization safety net on the SSC/cNG 6D reshape path is
  non-functional. Use `np.all(x == 0)`.
- **[suggestion] `spaceborne/sb_lib.py:2736`** — duplicate unit constant with wrong
  comment ("deg^2 to arcmin^2"); use `constants.SR_TO_ARCMIN2`.
- **[suggestion]** dead/conflicting convention artifacts: `constants.py:161-166`
  (`RS_PROBE_NAME_TO_IX_DICT_SHORT`, unused, conflicts with live convention),
  `bnt.py:126` + `constants.py:79` (unreachable `'LG'` keys),
  `ccl_interface.py:499-502` (assert message in dead tuple),
  `ell_utils.py:90` (`compute_ells_oc` silent nbl reduction — currently dead code).

## Verdict on Knox / iNKA / ensemble consistency

The three partial-sky Gaussian methods are built on mutually consistent conventions:
same noise definition and flat-noise treatment, same band definitions and band-average
normalization, matched coupled/decoupled states, correct per-bin-pair iNKA
normalization matching TJPCov. No convention mismatch was found that would explain an
analytic-vs-ensemble discrepancy at the *Gaussian signal* level. The two genuine
physics gaps found in this area are (a) the missing BB shape noise — common to both
paths, so it cancels in an analytic-vs-ensemble comparison but biases both vs. reality
— and (b) the scrambled spin-2 MCMs, which affect only the coupled SSC/cNG terms, not
the Gaussian NaMaster covariance itself.
