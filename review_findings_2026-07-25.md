# Spaceborne partial-sky validation review — 2026-07-25

Scope: closing out the partial-sky (NaMaster) validation. Specifically — (a) the status of
the `upd_nmt_noise` branch and whether its two pending features (the NaMaster ℓ_max buffer
and `coupled_noise`) are relevant, and (b) any leftover gap that would produce incorrect
results or a spurious discrepancy against TJPCov.

References: TJPCov (`~/Documents/Work/Code/TJPCov`, `tjpcov/covariance_fourier_gaussian_nmt.py`,
`tjpcov/covariance_builder.py`, its test suite and configs) used as benchmark;
pymaster 2.6 numerical spot checks run in this environment; Nicola et al. 2020
(arXiv:2010.09717) noise convention; García-García et al. 2019 (arXiv:1906.11765) for the
spin-0 approximation. Follows on from `review_findings_2026-07-07.md`.

---

## Top-priority summary

1. **[blocker for the comparison]** There is nothing to merge — `upd_nmt_noise` is already
   an ancestor of `develop`, and commit `a358b7ff` reverted both of its features 13
   minutes after the merge. Restore with `git revert a358b7ff`, **not** a merge.
2. **[substantive]** The ℓ_max buffer is real and needed: the top science bandpower is
   biased **low by ~13%**, measured. But as implemented it is guarded to
   `cov_type: coupled`, and TJPCov only ever produces **decoupled** — so buffer-on and
   TJPCov-comparable are currently mutually exclusive. A cleaner fix removes the
   restriction.
3. **[substantive]** `coupled_noise`'s implementation is correct and reproduces TJPCov's
   default exactly, but its docstring derives a different (wrong) expression, and turning
   it on **breaks agreement with Spaceborne's own ensemble covariance**.
4. **[gotcha]** `spin0: True` is the shipped default; TJPCov has no spin-0 approximation.
   Any TJPCov comparison must run `spin0: False`.
5. **[gotcha]** Restoring the two config keys verbatim will `KeyError` on every external
   config — `load_config` has no defaults merge.

---

## 0. Branch state — nothing to merge

```
$ git rev-list --left-right --count develop...origin/upd_nmt_noise
46      0
```

`upd_nmt_noise` (tip `d8328645`, 07-09 15:34) is an **ancestor of `develop`**. It was
merged at `1df27292` (07-09 17:48). Then `a358b7ff` *"fir merge leftovers"* (07-09 18:01)
**reverted both features**:

- `precision.ell_max_buffer_nmt` and all its plumbing (`main.py`,
  `mask_utils.get_ell_buffer_nmt`, `mask_utils.estimate_ell_cutoff`,
  `cov_nmt_obj.ell_max_nmt` / `.ell_max_buffer`)
- `precision.coupled_noise` and the `coupled_noise_factor` multiplication in
  `cov_partial_sky.py`

This appears accidental rather than deliberate. `develop` still carries the comment in
`wf_cl_lib.compute_cls_or_interpolate_input_cls:48-54` reading *"The partial-sky
covariance deliberately requests ell=0 … **and an ell_max buffer**"* — documentation for a
feature that no longer exists in the file it documents.

**Do not merge the branch.** `develop` is newer in places the branch would drag backwards:
the `ensemble_decoupled` forced workspace save (`978921e0`), and the
`config_checker` / `test_config_checker` "OneCovariance section is optional" work. A
targeted `git revert a358b7ff` is the right operation.

---

## 1. ℓ_max buffer — relevant, quantified

### The defect

`develop` builds the NaMaster fields, workspaces, covariance workspace **and** the
theory + noise Cl grid only up to the *science* `ell_max_3x2pt`
(`cov_partial_sky.py:1242-1256`, `main.py:1065-1076`). The mode-coupling sum
`Σ_ℓ' M_ℓℓ' C_ℓ'` is therefore truncated exactly where it still has support, and the top
bandpowers come out biased **low**.

### Measurement

pymaster 2.6, sharp binary mask, `nside=64`, `fsky≈0.3`, `Δℓ=10`, science `ℓ_max=90`;
ratio of the decoupled covariance diagonal, *no-buffer / buffered-to-3·nside−1*:

| band centre | 15 | 25 | 35 | 45 | 55 | 65 | 75 | 85 |
|---|---|---|---|---|---|---|---|---|
| ratio | 0.993 | 0.992 | 0.991 | 0.989 | 0.987 | 0.983 | 0.972 | **0.867** |

~13% low in the top bandpower, ~3% in the next, ~1% floor everywhere. The effect is worse
for shape noise (flat to the pixel scale) than for the falling signal, and its extent
scales with the mask's power-spectrum bandwidth and with Δℓ.

### It also explains a known analytic-vs-ensemble inconsistency

`_compute_one_realization` injects noise at **full pixel resolution**
(`cov_partial_sky.py:548-562`) — deliberately, per its own comment — so the ensemble's
noise pseudo-Cl is untruncated. The analytic path can only couple noise from
`ℓ' ≤ ell_max_3x2pt`. The two therefore disagree at the top bins *by construction*,
independently of any other issue.

### TJPCov corroborates the direction

TJPCov has no buffer knob because it never needs one — it always runs at the healpix band
limit. `get_nell` resolves `nell = bins.lmax + 1`
(`covariance_fourier_gaussian_nmt.py:607-608`) and its test bandpower edges reach
`3*nside` (`tests/test_covariance_fourier_gaussian_nmt.py:79-88`), with
`NaMaster.f.lmax: null` leaving the field at pymaster's own `3*nside-1` default.

### Why the current implementation still won't close the comparison

Two blockers:

- **The buffer is guarded to `cov_type: coupled`.** `upd_nmt_noise`'s
  `cov_partial_sky.py:1288-1292` raises `NotImplementedError` for decoupled. But TJPCov
  **only ever returns decoupled** — `coupled=True` raises `NotImplementedError` there too
  (`covariance_fourier_gaussian_nmt.py:186-192`). So the two configurations don't overlap.
- The workaround it uses (a per-ℓ `wsp_bin_obj`) exists because of a real pymaster
  constraint, verified in this environment:

  ```
  ValueError: Maximum multipoles in bins (61) and fields (95) are not the same.
  ```

### Recommended fix (verified working)

Instead of a scalar `ell_max_buffer_nmt`, **extend the science bandpower edges with
trailing bands up to `3*nside-1`** so that `bins.lmax == field.lmax`, then slice
`cov[:n_sci, :n_sci]`. Verified end-to-end in pymaster 2.6 for `coupled=False`:

```python
lo = np.concatenate([lo_sci, np.arange(hi_sci[-1], lmax_ext + 1, dl)])
hi = np.concatenate([hi_sci, np.minimum(lo_x + dl, lmax_ext + 1)])
b  = nmt.NmtBin.from_edges(lo, hi)          # b.lmax == field lmax  -> accepted
...
cov = nmt.gaussian_covariance(cw, ..., wa, wb, coupled=False)[:n_sci, :n_sci]
```

This is what TJPCov does implicitly. It works for **both** coupled and decoupled, it
replaces an arbitrary tunable with the physically correct answer (always run to the band
limit), and it lets you delete the `wsp_bin_obj` special case, the
`bin_cell(cl[:nbl_nmt])` slicing in `_compute_one_realization`, and the
`mcm[:nbl_ext, :nbl_ext]` slicing in `compute_and_save_mcms`.

The auto-estimator `get_ell_buffer_nmt` / `estimate_ell_cutoff` is moot either way — it is
commented out in `main.py` on the branch, behind `# TODO decide which one to use`.

---

## 2. `coupled_noise` — implementation right, docstring wrong, ensemble breaks

### The factor

```python
factor[zi] = w_obs.mean() ** 2 / (w_obs**2).mean()      # over observed pixels, w > 0
```

**The docstring is wrong.** It derives `N_ℓ^cov = Ω_pix⟨w²σ²⟩/⟨w²⟩` and then notes that
σ_ε is a per-bin scalar — in which case σ² pulls out, the ratio is exactly σ², and the
factor would be **1**, not `⟨w⟩²/⟨w²⟩`.

### The code is nevertheless correct — under one assumption

If the weight map is ∝ galaxy number density (i.e. inverse-variance weighting, `n_p ∝ w_p`,
so `σ²_p Ω_pix = σ_ε²/(n̄₀ w_p)`):

```
N^coupled = Ω_pix⟨w²σ²⟩ = (σ_ε²/n̄₀)·⟨w⟩
```

TJPCov's default (`use_coupled_noise=True`) forms
`(couple_cell(C) + N^coupled) / ⟨w_a w_b⟩` (`get_cl_for_cov`,
`covariance_fourier_gaussian_nmt.py:121-146`), whose noise part is
`(σ_ε²/n̄₀)⟨w⟩/⟨w²⟩`. With `N_flat = σ_ε²/(n̄₀⟨w⟩_obs)`, this equals

```
N_flat · ⟨w⟩²_obs / ⟨w²⟩_obs
```

— **exactly the implemented factor.** It reproduces TJPCov's default behaviour. It is a
no-op (`= 1`) for a binary footprint, so it only matters when weight maps are in use.

### The problem: it desynchronises the ensemble

`_compute_one_realization` injects **homogeneous** per-pixel noise —
`sigma_pix = sqrt(nl_gg_diag[zi] / omega_pix)`, identical for every pixel
(`cov_partial_sky.py:557,560`). That corresponds to `σ²_p = const`, hence
`N^coupled = Ω_pix σ²⟨w²⟩` and a factor of **1**.

So enabling `coupled_noise` trades agreement with TJPCov for disagreement with your own
Monte Carlo, by exactly `⟨w⟩²/⟨w²⟩`, whenever the weight maps are non-binary.

To have both, the ensemble must draw depth-varying noise:

```
σ_pix,p = sqrt( N_ℓ · ⟨w⟩_obs / (w_p · Ω_pix) )      for w_p > 0
```

which closes the loop algebraically (verified by hand: the resulting coupled pseudo-Cl is
`N_flat·⟨w⟩_obs·⟨w⟩_full`, and NaMaster's flat-input coupling gives `⟨w²⟩_full·N_input`,
equating to `N_input = N_flat·⟨w⟩²_obs/⟨w²⟩_obs`).

### Open decision

**Do the weight maps represent galaxy number density (inverse-variance weighting), or pure
coverage/completeness at uniform depth?**

- density → enable `coupled_noise`, *and* fix the ensemble noise draw as above
- coverage → leave it off; `develop` is already correct

This is a modelling choice about the input maps, not something derivable from the code.

---

## 3. Remaining gaps

Ranked by likelihood of producing wrong numbers or a spurious TJPCov discrepancy.

### a) `spin0: True` is the shipped default

TJPCov has **no** spin-0 approximation — spin comes from the sacc tracer quantity
(`covariance_builder.py:541-566`) and it always uses exact spin-2 with the true `ncell`.
Comparing Spaceborne's default against TJPCov measures the García-García approximation
error, not a code discrepancy. Run `precision.spin0: False` for the benchmark.

Note also (carried over from the 07-08 rework): **no end-to-end `spin0: True` run has ever
been watched**, and `cov_partial_sky` has no direct unit tests.

### b) Workspace cache keys don't encode ℓ_max

`self.wsp_fname = f'wsp_spin0{self.spin0}_' + '{:s}{:s}_zi{zi:d}zj{zj:d}.fits'` and the
`cw_fname` equivalent (`cov_partial_sky.py:736-740`) encode `spin0` but **not** the lmax.
Enabling the buffer changes the field lmax, so a stale cache is at best a crash and at
worst a silently mixed run. This is a one-line f-string change and it becomes mandatory
the moment the buffer lands.

### c) `load_config` has no defaults merge

`main.py:38-39` is a bare `yaml.safe_load`. Both reverted keys are read as unconditional
`cfg['precision'][...]` lookups, so restoring them verbatim will `KeyError` on every
external config — which per `CLAUDE.md` means *all* production runs and the entire
`batch_run_utils` sweep. Use `.get(key, default)`.

### d) Ensemble E/B alms skip the Jacobi iteration

`mask_maps_and_compute_alms` uses `hp.map2alm(..., iter=n_iter)` with `n_iter=3` for T but
`hp.map2alm_spin(...)` with no iteration for E/B (`cov_partial_sky.py:310-311`). Small in
absolute terms, but it is an **asymmetry between GG and LL inside the very object being
used to validate**. Make them consistent; `iter=0` for both is the pseudo-Cl-native choice.

### e) `hp.remove_monopole` applied to Q and U separately

`cov_partial_sky.py:306-308`. Harmless for T — a full-sky constant is pure monopole, so
only ℓ=0 is touched. For Q/U it is not: subtracting constants from Q and U does project
onto E/B at ℓ≥2. The amplitude is second-order-small (the monopole of a masked shear map
is tiny), so this is tidiness, not a bug.

---

## 4. Verified as already correct — matches TJPCov

These were checked explicitly and need no further attention:

- **iNKA leg structure.** Spaceborne pre-computes `couple_cell(Cl_zi_zj)/mean(w_a·w_b)`
  per (probe, zi, zj) pair using that pair's own workspace
  (`cov_partial_sky.py:1276-1319`), then indexes it inside the probe loop. This is
  algebraically identical to TJPCov's per-leg `w13/w14/w23/w24` construction, and the
  `wa`/`wb` passed to `nmt.gaussian_covariance` correspond to TJPCov's `w12`/`w34`.
- **Per-pair effective fsky.** Both divide by the pair-specific `mean(w_a * w_b)`, not a
  global fsky — correct for fractional weight maps where `⟨w²⟩ ≠ ⟨w⟩`.
- **Noise placement.** EE **and** BB both receive the shear noise with ℓ=0,1 nulled
  (`cl_bb_4covnmt = nl_ll_4covnmt`, `nl_ll_4covnmt[:2] = 0`), matching TJPCov's
  `SN[i][0] = SN[i][-1]` with `SN[i][0, :2] = SN[i][-1, :2] = 0`.
- **Zero noise on GL and on cross-z pairs**, matching TJPCov's `auto = tr[i1] == tr[i2]`
  guard. EB/BE = 0 in both.
- The two `7d34dd32` bugfixes (`np.diagonal(nl_ll_4covnmt[2])`, `cl_bb_4covens = 0`) are
  present on `develop` — they were *not* caught by the `a358b7ff` revert.

---

## 5. Recommended action

1. `git revert a358b7ff` to restore both features on `develop`.
2. Convert the two restored config lookups to `.get(..., default)`, with `coupled_noise`
   defaulting to **False**.
3. Replace the `ell_max_buffer_nmt` scalar and the coupled-only guard with the
   extended-bandpower-edges approach up to `3*nside-1`; add the lmax to the workspace
   cache filename.
4. Run the validation triplet with `spin0: False` and `cov_type: decoupled`:
   analytic NaMaster vs ensemble vs TJPCov.
5. Settle the weight-map semantics question in §2 before enabling `coupled_noise`.

Steps 1–3 are small and mechanical. Step 5 is a modelling decision.
