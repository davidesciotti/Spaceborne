# Spaceborne architecture review

_Date: 2026-07-08. Scope: `main.py` + the `spaceborne/` package, reviewed for
architecture and programming-principle improvements (not a line-level bug hunt).
Line references are snapshots and may drift as the code changes._

## Bottom line

The code is *scientifically* well-factored: there is a real data-model
convention (`cov_dict[term][probe][dim]`), a genuine projector base class
(`CovarianceProjector` → `CovRealSpace`/`CovCOSEBIs`), and consistent reuse of
the reshape helpers in `sb_lib.py`. What holds it back is **software plumbing**,
and the same three anti-patterns recur in almost every module.

None of the fixes below require a rewrite or new frameworks. The highest-leverage
ones are mechanical (de-duplication, function extraction). The guiding constraint
for this review was *improve the architecture without over-complicating the
code* — see [What not to do](#what-not-to-do) for the explicit guardrails.

---

## The three systemic issues

Fix the pattern, not just the instance — each of these recurs across many files.

### 1. Objects are constructed half-built, then finished by `main.py`

This "two-phase lifecycle" is the most pervasive smell and the biggest
correctness risk. Constructors run, then `main.py` bolts required attributes on
from outside:

- `ccl_obj` receives **13+** external `obj.attr = ...` assignments scattered over
  ~1000 lines (`main.py:594-598, 708-711, 846, 1001, 1039, 1572`).
- Same pattern for `cov_nmt_obj` (`main.py:1111-1114`), `cov_rs_obj`
  (`main.py:1124-1132`), `cov_oc_obj` (`main.py:1296-1299`).
- Classes even write into *each other*: `CovHarmonicSpace` sets
  `self.cov_nmt_obj.noise_3x2pt_unb_5d = ...` (`cov_harmonic_space.py:212`) and
  later reads `cov_nmt_obj.mcm_ee_binned`, an attribute that only exists after
  `compute_and_save_mcms` has run (`cov_partial_sky.py:1166`).

Every cov class uses an `_UNSET = object()` sentinel
(`cov_harmonic_space.py:17`, `cov_partial_sky.py:17`, `cov_real_space.py:39`,
`cov_cosebis.py:27`, `ccl_interface.py:15`) to paper over this — which is itself
the evidence that the construction contract is not real.

**Consequence:** no class can be instantiated or tested in isolation, and a
reordering bug in `main.py` silently leaves a field unset until something deep in
the pipeline reads it — a real risk in a codebase where correctness depends on
grids being mutually consistent.

**Principle:** a constructor should leave the object usable. Pass these values
through `__init__` (or one `configure(...)` method with a named signature) so the
object declares what it actually needs.

### 2. `main.py` is a 2280-line flat script

It contains 3 `def`s; everything else is module-level statements sharing ~40
mutable globals. The `# ! ===` banners (`Mask` `main.py:815`, `n(z)` `849`,
`Radial kernels` `995`, `Cls` `1027`, `OneCovariance` `1177`, `Combine
covariance terms` `1610`, …) already mark clean phase boundaries — they are
effectively docstrings for functions that do not exist yet. State flows via
globals read hundreds of lines from where they are set (e.g. `single_b_of_z` set
at `main.py:960`, reused at `1415`). There is no way to run or test a single
phase.

**Principle:** named phases with explicit inputs/outputs. Extract each banner
into a plain function `phase(cfg, ...) -> values`, orchestrated by a thin
`main()` at the bottom. This is the enabler for everything else — including
issue #1, whose fix becomes obvious once each phase declares its I/O.

### 3. Config is two untyped dicts threaded everywhere

`cfg` (234 references in `main.py` alone) plus a second runtime dict `pvt_cfg`
(built at `main.py:752`) are passed together into nearly every constructor, then
unpacked by ~25 blind `self.x = cfg[...][...]` assignments per class.
`config_checker.py` is ~490 lines of hand-rolled `assert isinstance(...)`
(`config_checker.py:48-540`) — a JSON schema written by hand.

Worse, `main.py` **silently mutates user config**: it injects implementation-only
keys (`main.py:287-388`), overwrites `probe_selection` based on derived logic
(`556-573`), and auto-bumps `log10_k_max` with only a `warnings.warn`
(`681`). So the `run_config.yaml` provenance file (dumped at `main.py:1972`)
mixes user intent with derived state — good for reproducibility, bad for
auditability.

**Principle:** one typed, validated config object; keep *requested* config
distinct from *effective* config. A pydantic/dataclass model with `Literal`
types for the enum-like strings (`partial_sky_method`, `G_code`, `SSC_code`,
`which_sigma2_b`, …) deletes ~60% of `config_checker.py` and gives typo-safety at
every `cfg[...]` site. Keep the ~15 genuine cross-field checks
(`config_checker.py:605-784`, e.g. "NaMaster + ref_cut binning is unsupported")
as explicit validators — do not force those into the type system.

`CCLInterface.__init__` (`ccl_interface.py:92-100`) is the positive
counterexample: it takes narrow named dict params instead of the whole `cfg`.
Other modules (especially `oc_interface.py`, with 80+ inline `cfg[...]` lookups)
should follow it.

---

## Secondary themes

### `main.py` mutates other objects' internal `cov_dict` directly

It rescales `cov_ssc_obj.cov_dict['ssc'][...][dim] /= fsky` in place
(`main.py:1546`) and hand-merges OneCovariance results with three near-identical
nested loops (`main.py:1774-1804`). These are behaviors that belong on the owning
class: `cov_ssc_obj.rescale_by_fsky(...)` and a `merge_term(...)` helper.

### `sb_lib.py` is a 2761-line junk drawer

68 functions, 0 classes. The cov-reshape engine, JAX Gaussian math, ell-binning,
plotting/debug helpers, generic numeric utils, and text I/O all coexist. Natural
split by responsibility: `cov_reshape.py` (the cov-dict engine), `binning.py`
(ell binning), `plotting_debug.py` (`compare_*`, `matshow`), leaving slim generic
utils behind.

> **Correction to an earlier draft:** an initial pass flagged ~28% of the file as
> dead. That is overstated. `hartlap_factor`, `percival_factor`,
> `regularize_covariance`, and `bin_2d_array` are all exercised by
> `tests/test_sb_lib.py` and are intentional utilities. Only
> `cov_4D_to_6D_blocks_opt` (`sb_lib.py:2404`), `figure_of_correlation`, and
> `contour_FoM_calculator` are genuinely unreferenced anywhere. **Split by
> moving, not deleting**, and verify against `tests/` before removing anything.

### Clear-cut duplication (each a double-maintenance bug waiting to happen)

- `t_mix` defined byte-for-byte twice: `cov_real_space.py:157` and
  `cov_projector.py:47` (the shared base module). `CovRealSpace` should call
  `cp.t_mix`.
- Real-space vs COSEBIs assembly blocks: `main.py:1620-1683` and `1687-1750`,
  ~60 lines each, differing only in object names and lookup dicts. The author
  already flagged this at `main.py:1685`.
- ~300 lines of pylevin setup repeated 3× in `cov_real_space.py:171-476` —
  collapsible to one `_build_levin(...)` factory.
- The `'3x2pt'` skip-guard (`if key == '3x2pt': continue`) copy-pasted 9× across
  4 files (see the data-model note below).
- Three near-identical OneCovariance term-copy loops (`main.py:1774-1804`),
  collapsible to `for term in ('g', 'ssc', 'cng'): ...`.

### External boundaries leak implementation details

- `OneCovarianceInterface` (`oc_interface.py`, 1200 lines) mixes ~5-6
  responsibilities: `.ini` building (`511-879`), a bash-subprocess path
  (`881-901`) *and* an in-process path that reaches into OneCovariance's private
  attributes (`903-967`, `ellspace.aux_response_mm` at `1189`), two output-format
  parsers (`.mat` at `1081-1144`, `.list` at `246-407`), and diagnostics
  (`compare_sb_and_oc`, `35-136`). Split into ini-builder / runner / parser /
  comparator.
- `responses.py:77,185-199` monkeypatches CCL's *private* API (`hmc._bf`,
  `hmc._get_ingredients`, `hmc._integrate_over_mbf`) — silently breakable on a
  CCL upgrade and not mockable in tests. It also loads a hardcoded
  `./input/Resp_G1_fromsims.dat` relative path (`responses.py:81`) that breaks
  when run from another cwd.

### A verified circular import

`ccl_interface.py:11` imports `wf_cl_lib` (for bias functions), and
`wf_cl_lib.py:39` imports `ccl_interface.compute_cl_3x2pt_5d`. It works only
because use is deferred past import time — fragile to reordering. Moving
`compute_cls_or_interpolate_input_cls` (`wf_cl_lib.py:28-96`) out of `wf_cl_lib`
makes it a clean leaf module.

### The cov-dict `'3x2pt'` key is a structural wart

It is a plain string among tuple keys and carries no `4d`/`6d` entry, unlike
every other key, forcing an `if key == '3x2pt': continue` guard in ~9 places
(documented as a trap at `sb_lib.py:39-43`). Splitting it out — either a separate
`assembled_3x2pt` field or a small typed leaf container — removes all the special
cases. This is the *one* place a dataclass clearly earns its keep; the
term/probe keys themselves are fine as open-ended strings/tuples.

---

## Recommended roadmap (ranked by impact ÷ effort)

| # | Change | Effort | Why |
|---|--------|--------|-----|
| 1 | **De-dupe the obvious copies**: unify real-space/COSEBIs block, point `CovRealSpace` at `cp.t_mix`, extract a `_build_levin()` factory, collapse the 3 OneCovariance term-copy loops | Low | Zero behavior risk, ~400 lines removed, kills double-maintenance bugs |
| 2 | **Extract `main.py` banner sections into functions** with explicit args/returns; thin `main()` at the bottom | Med | Unlocks testability; makes #4 mechanical; no algorithm changes |
| 3 | **Split `sb_lib.py` by responsibility** (move, don't delete); drop only the 3 genuinely-dead functions | Low-Med | Pure import fixups; makes the reshape API discoverable |
| 4 | **Kill the two-phase lifecycle**: pass bolted-on attributes through `__init__`/`configure()`, starting with `ccl_obj` | Med | Biggest correctness win; do after #2 |
| 5 | **Typed config schema** (pydantic/dataclass) replacing `check_types`; keep the semantic validators | Med | Deletes ~490 lines, typo-safety everywhere; separates requested vs effective config |
| 6 | **Move in-place `cov_dict` mutations onto owning classes** (`rescale_by_fsky`, `merge_term`) | Low | Removes the worst `main.py`↔internal coupling |
| 7 | **Isolate boundaries**: break the circular import, split `OneCovarianceInterface`, wrap the CCL monkeypatch behind an adapter + version check, config-ify the hardcoded path | Med-High | Fragility/portability; do incrementally |

Items **1, 3, and 6** are safe to start immediately with essentially no
regression risk.

---

## What not to do

Guardrails so these changes don't trade one problem for over-engineering:

- **No DI framework, no `Pipeline`/`Stage` class hierarchy.** Plain functions
  with explicit args (item 2) capture ~80% of the benefit at a fraction of the
  risk, and match the "script with named phases" style the codebase already wants
  (per CLAUDE.md's own framing of `main.py` as the orchestrator).
- **No forced ABC across the cov classes.** Harmonic / real-space / SSC math is
  genuinely different; the shared *data-model convention* is the right unifier.
  At most, document a minimal informal contract (`.cov_dict`, one `compute()`
  entrypoint).
- **Keep string/tuple keys for `term` and `probe`.** The physics vocabulary is
  open-ended (`g`/`ssc`/`cng`/`tot`; 3-16 probe names by space); only the
  `4d`/`6d` *leaf* benefits from a typed container.
- **Do not mass-delete "unused" functions.** Verify against `tests/` first — most
  are intentional, test-covered utilities.
