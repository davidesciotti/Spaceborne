# Spaceborne optimization opportunities

Performance/memory audit, 2026-07-10. Four parallel audits: (a) main.py + harmonic/SSC,
(b) NaMaster partial-sky + ensemble, (c) real-space/COSEBIs projection, (d) reshaping/IO/Cls.

Recurring theme: the physics kernels (CCL, NaMaster, Bessel integrals) are fine — the time
and memory go to pure-Python reshape loops, per-scalar re-instantiation of expensive
objects, and large arrays kept alive or allocated dense when they're diagonal.

## Tier 1 — quick wins, high impact (touch every run)

### 1. Vectorize the 6D↔4D reshapes — the fast version already exists, unused
- `cov_6D_to_4D_blocks` (`sb_lib.py:2261`): 4 nested Python loops (`ell1, ell2, ij, kl`),
  up to ~9M interpreted iterations *per term × probe block*; runs for every block in
  postprocessing (called from `sb_lib.py:2243` via `cov_dict_6d_probe_blocks_to_4d_and_2d`,
  and from `cov_real_space.py:971`).
- Its inverse `cov_4D_to_6D_blocks` (`sb_lib.py:2318`) has the same pattern, called from 5
  production sites: `cov_harmonic_space.py:300`, `cov_real_space.py:1005`,
  `cov_real_space.py:1254`, `cov_partial_sky.py:1382`, `cov_cosebis.py:337`.
- The vectorized `cov_4D_to_6D_blocks_opt` (`sb_lib.py:2401`, `np.ogrid` fancy indexing)
  has **zero callers** — dead code.
- **Fix**: swap the 5 call sites to `_opt`; write the fancy-indexing mirror for 6D→4D
  (`cov_6D[:, :, zi[:,None], zj[:,None], zk[None,:], zl[None,:]]` with index arrays
  precomputed from `ind_AB`/`ind_CD` — no ell loop needed, leading axes are sliced).
- Likely the single largest wall-clock win in the assemble stage.

### 2. Symmetrization: ~900K Python calls → one transpose
- `sb_lib.py:2380-2396` (duplicated in `_opt` at `:2425-2439`): calls
  `symmetrize_2d_array` once per (ell1, ell2, zi, zj) ≈ nbl²·zbins² = 900K times; each call
  re-runs `np.array_equal` and rebuilds triangle indices.
- **Fix**: one vectorized statement, e.g. `cov_6D = 0.5*(cov_6D + cov_6D.transpose(0,1,3,2,4,5))`
  for ab (axes 4,5 for cd); given the populated-triangle invariant it's effectively a copy.

### 3. Ensemble mode: NaMaster workspaces re-read from disk every realization
- `_compute_one_realization` (`cov_partial_sky.py:519-529`): when `coupled_cls=False`,
  every call does zbins²×3 `NmtWorkspace().read_from()` FITS reads → 300K reads for
  nreal=1000, zbins=10, instead of once per worker.
- **Fix**: module-level memo dict keyed by path; loky reuses worker processes, so a simple
  `if path not in _CACHE` persists across dispatches.

### 4. Ensemble mode: peak memory = 3·zbins full-resolution maps per worker
- `_compute_one_realization` (`cov_partial_sky.py:540-544`): list comprehensions
  materialize ALL zbins T maps + ALL (Q,U) pairs (float64) before masking; freed only at
  `:574`, after `mask_maps_and_compute_alms`. Peak ≈ 3·zbins·npix·8 B (~48 GB at zbins=10,
  nside=4096) × n_jobs.
- **Fix**: stream one z-bin through alm2map → noise → mask → map2alm, freeing each raw map
  immediately (pattern already exists inside `mask_maps_and_compute_alms`, `:292`; move it
  one level up).

### 5. Gaussian cov: dense ℓ×ℓ' 10D array that is 96% zeros + view-aliasing leak
- `_expand_diagonal_to_full` (`sb_lib.py:1964-1985`), consumed at
  `cov_harmonic_space.py:122-160`: allocates
  `(2,2,2,2,nbl,nbl,zbins⁴)` ≈ 1.15 GB per Gaussian term (SVA/SN/MIX) with only the
  ℓ-diagonal nonzero (Knox has no ℓ coupling).
- Worse: probe blocks extracted at `cov_harmonic_space.py:142-150` by basic indexing are
  **views** — each stored 72 MB 6D block pins the full 1.15 GB parent for the entire run
  (~3.5 GB dead weight for sva+sn+mix when BNT is off and the chain is never broken).
- **Fix**: slice requested probe blocks (with `.copy()`) from the *diagonal* array (30×
  smaller) and expand to nbl×nbl only per requested block.

### 6. Free 4D blocks after building 2D
- Only 6D and 2D are ever saved (`main.py:1920`, `:1946`); the 4D block per (term, probe)
  stays alive for the whole run. Only `cov_partial_sky.py:1394` nulls it.
- **Fix**: `cov_dict[term][probe]['4d'] = None` right after `cov_4D_to_2D` in
  `cov_dict_6d_probe_blocks_to_4d_and_2d` (`sb_lib.py:2252-2256`). Tens of MB × up to 9
  blocks × terms.

## Tier 2 — high impact in specific run modes

### 7. Real space: bin-averaged Levin wrappers re-instantiate pylevin per scalar
- `dl1dl2_binavg_bessel_wrapper` (`cov_real_space.py:262-294`): one `pylevin` construction
  per (ℓ₁, θ, Bessel-term) triple — construction cost is independent of how many θ points
  are requested. Non-binavg sibling `dl1dl2_nobinavg_bessel_wrapper` (`:340-375`) already
  batches the whole θ array. NG-term path with `levin_bin_avg=True`.
- `levin_binavg_helper` (`cov_real_space.py:1024-1063`): same bug for the Gaussian SVA/MIX
  term — fresh `pylevin` (`:442`) per (θ₁, θ₂) pair, up to nbt²×4 constructions.
  `levin_integrate_bessel_double_wrapper` (`:378-426`) already shows the batched meshgrid
  pattern.
- **Fix**: mirror the batched siblings — one call per (ℓ₁ or term, full θ grid).

### 8. FFTLog projector recomputes the Bessel kernel per z-block
- `proj_cov_2d_fftlog` (`cov_real_space.py:572-611`): fresh `TwoBessel`/`TwoSphBessel` per
  zpair block (up to ~10⁴). Grid/kernel quantities (`m, n, eta, z1, z2, g_l_smooth` — the
  complex-gamma kernel, `twobessel_fang.py:95-134, 281-332`) depend only on (ℓ-grid, μ, ν,
  dlnθ) — identical across blocks; only the data FFT (`c_mn`) is block-dependent. Also a
  fresh `RectBivariateSpline` per block (`:601-603`), and per-iteration padded-array churn
  (`twobessel_fang.py:82-93`).
- **Fix**: factor kernel construction out of the loop; batch the FFT across a stacked
  `(n_blocks, N1, N2)` axis via `rfft2(axes=...)`; one vectorized interpolation call.

### 9. Halo-model responses: zbins² redundant expensive calls
- `main.py:1508-1521`: `set_hm_resp` (≈8 CCL halo-mass-function integrals per a-grid point,
  `responses.py:350-628`) called zbins² = 100 times.
  - `from_HOD` branch: b1g unused → all 100 calls identical.
  - `from_input` branch: `dPmm_ddeltab` is bias-independent; `dPgm`/`dPgg` are closed-form
    in the bias vectors → one call + broadcasting.
- Code's own TODO at `main.py:1482-1483` acknowledges it.

### 10. Simps projector: one joblib task per scalar output element
- `proj_cov_simps_parallel_helper_wrapper` / `proj_cov_parallel_helper`
  (`cov_projector.py:301-423`, `proj_g_int_method=='simps'`): dispatches
  nbt² × zpairs_ab × zpairs_cd tasks; each rebuilds the Bessel kernel
  (`build_projection_kernel`, `:361-372`) that depends only on (θ, μ), not the z indices.
  TODO at `cov_projector.py:590` acknowledges it.
- **Fix**: build the integrand once via the existing `build_cov_sva_integrand_5d` (already
  used by the levin/FFTLog path), evaluate the kernel once per (θ₁,θ₂), vectorize Simpson
  over the zpair tensor (like the `proj_cov_2d` simps branch, `cov_projector.py:201-223`).

### 11. Unexploited symmetries in the NaMaster path
- `nmt_gaussian_cov` (`cov_partial_sky.py:204-206`): full zij×zkl grid of
  `nmt.gaussian_covariance` calls even for auto-blocks; `build_cw` (`:1015-1017`) already
  implements the upper-triangle+transpose reduction — mirror it (~2× fewer calls).
- `build_wsp`/cache (`cov_partial_sky.py:868-903`, `:1095-1101`): wgg/wll workspaces
  computed for both (zi,zj) and (zj,zi) though the MCM is symmetric; only wgl needs the
  full cross product → `combinations_with_replacement` + mirror.
- `sim_cls_to_ensemble_cov` (`cov_partial_sky.py:482-493`): loops full zbins⁴; own TODO at
  `:411` ("use only independent z pairs") → iterate `ind_dict`-style unique zpairs.

## Tier 3 — smaller / policy decisions

- **`map2alm iter=3`** in ensemble (`cov_partial_sky.py:272`, `:310`): ~4× base SHT cost
  per bin per realization; refinement error likely ≪ MC noise — benchmark iter=0/1. Note
  `map2alm_spin` (`:311`) ignores the knob entirely (inconsistent).
- **Cls recomputed ~5-6× on different ℓ-grids** (`main.py:1074-1211` call sites of
  `compute_cl_3x2pt_5d` via `wf_cl_lib.py:465-502`): compute once on a fine grid and
  spline-interpolate (docstring at `wf_cl_lib.py:36-38` already flags it). More invasive.
- **`deepcopy` on bare ndarrays** → `.copy()` (`cov_harmonic_space.py:187-189, 245-247,
  322-324`), 72 MB arrays on the per-term hot path.
- **`apply_mult_shear_bias`** triple Python loop → broadcast (`ccl_interface.py:18-39`).
- **Per-ℓ spline evaluation** in `responses.py:686-702`: ~3000 `RectBivariateSpline` calls
  → one flat `grid=False` call per (zi,zj), ~30× fewer calls.
- **`_cl_3x2pt_5d_sb` computed unconditionally** for a diagnostic plot (`main.py:1071-1081`)
  even when `use_input_cls=False` (redundant with `main.py:1091`) and plots disabled —
  gate it.
- **Mask plots always rendered** regardless of `save_figs` (`main.py:848-863`); only the
  save is gated (`main.py:2335-2346`).
- **cov_ssc.py:266-272**: ~55/100-iteration reindex loop, trivially fancy-indexable.
- **COSEBIs/real-space shape-noise caching**: `cov_sn_cs` (`cov_cosebis.py:121-197`)
  depends only on `amax_abcd` (includes mpmath-precision root finding) but is recomputed
  per probe combination; `cov_sn_rs` npair triple loop (`cov_real_space.py:743-789`,
  depends only on the 2×2 probe indices) — cache both.
- **`prefac` rebuilt per call** in `proj_mix_levin_or_fftlog` (`cov_real_space.py:894-909`),
  depends only on n_eff/sigma_eps/zbins — compute once in `__init__`.
- **float32 (policy)**: ensemble maps/alms could safely be float32 (MC noise dominates);
  float32 *storage* of covariance outputs would halve the multi-GB working set and file
  sizes but changes the precision of a public science product. Also note: with
  `jax_enable_x64: false` (`main.py:50-59`), JAX ops silently run float32 next to float64
  NumPy — a precision mismatch, not a deliberate trade-off.
- **θ₁↔θ₂ / (s1,s2) upper-triangle symmetrization** in real-space NG path — own TODO at
  `cov_real_space.py:1207-1208`; ~2× fewer Bessel/FFT evaluations.

## Dead code noticed along the way (delete when convenient)

- `sb_lib.py:2401` `cov_4D_to_6D_blocks_opt` — unused *because* the slow twin is called
  instead (fix #1 makes it the live path).
- `sb_lib.py:2521-2539` `cov_4D_to_2D` unoptimized fallback — unreachable
  (`optimize=False` never passed).
- `sb_lib.py:776` `bin_2d_array` — never called (only `bin_2d_array_vectorized` is used).
- `sb_lib.py:1404` `zpair_from_zidx` — no callers.

## Suggested order of attack

Tier 1 (items 1-6) is roughly a day of work combined and touches every run. Tier 2 depends
on run mode: ensemble users → 3, 4, 11; real-space/COSEBIs users → 7, 8, 10; halo-model
SSC → 9. Validate each numerical change by diffing `covmats_2D.npz` against a pre-change
reference run.
